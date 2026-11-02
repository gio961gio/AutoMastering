"""Reference Protocol & Audio Alignment Engine for Sprint 2.

Defines:
  1. ReferenceMode: "same_track" | "stylistic"
  2. AlignmentResult: strict separation of file delay and DDSP internal chain latency (159 samples).
  3. Time alignment estimation via normalized cross-correlation with sub-sample parabolic refinement.
  4. Guardrails preventing waveform point-to-point comparisons in stylistic reference mode.
"""

from dataclasses import dataclass, asdict
from enum import Enum
import math
from typing import Dict, Optional, Tuple, Union
import numpy as np
from scipy import signal


class ReferenceMode(str, Enum):
    """Reference mastering comparison mode."""
    SAME_TRACK = "same_track"
    STYLISTIC = "stylistic"


@dataclass
class AlignmentResult:
    """Detailed temporal alignment metadata."""
    lag_samples: int
    lag_seconds: float
    subsample_offset: float
    total_lag_samples: float
    correlation_coefficient: float
    is_aligned: bool
    reliability_score: float
    chain_latency_samples: int = 159
    sample_rate: float = 48000.0

    def to_dict(self) -> Dict[str, Union[int, float, bool]]:
        return asdict(self)


def verify_audio_pair(
    mix: np.ndarray,
    ref: np.ndarray,
    sample_rate: float = 48000.0,
    expected_sample_rate: float = 48000.0,
    mode: ReferenceMode = ReferenceMode.SAME_TRACK,
) -> None:
    """Validate sample rate and stereo channel dimensionality for an audio pair."""
    if sample_rate != expected_sample_rate:
        raise ValueError(f"Expected sample rate {expected_sample_rate} Hz, got {sample_rate} Hz")
    if mix.ndim != 2 or mix.shape[0] != 2:
        raise ValueError(f"Unmastered mix must have shape [2, T], got {mix.shape}")
    if ref.ndim != 2 or ref.shape[0] != 2:
        raise ValueError(f"Reference audio must have shape [2, T], got {ref.shape}")

    if mode == ReferenceMode.SAME_TRACK:
        # Check non-empty signals
        if mix.shape[1] < 4800 or ref.shape[1] < 4800:
            raise ValueError("Audio tracks must be at least 100 ms (4800 samples) for same-track alignment")


def estimate_time_alignment(
    mix: np.ndarray,
    ref: np.ndarray,
    sample_rate: float = 48000.0,
    max_search_sec: float = 3.0,
    min_correlation_threshold: float = 0.35,
    chain_latency_samples: int = 159,
) -> AlignmentResult:
    """Estimate time misalignment between unmastered mix and reference master via cross-correlation.

    Args:
        mix: Stereo mix [2, T_mix]
        ref: Stereo reference master [2, T_ref]
        sample_rate: Audio sampling frequency in Hz
        max_search_sec: Maximum expected lead/lag window to search in seconds
        min_correlation_threshold: Threshold below which alignment is deemed unreliable
        chain_latency_samples: Internal DDSP chain latency (159 samples at 48kHz)

    Returns:
        AlignmentResult with integer lag, sub-sample parabolic interpolation, and reliability metrics.
    """
    verify_audio_pair(mix, ref, sample_rate, expected_sample_rate=sample_rate, mode=ReferenceMode.SAME_TRACK)

    # Use first N seconds or available length for robust correlation estimation
    analysis_len = min(mix.shape[1], ref.shape[1], int(15.0 * sample_rate))
    mix_sub = mix[:, :analysis_len]
    ref_sub = ref[:, :analysis_len]

    # Convert to mono
    mix_mono = np.mean(mix_sub, axis=0).astype(np.float64)
    ref_mono = np.mean(ref_sub, axis=0).astype(np.float64)

    # Remove DC bias
    mix_mono -= np.mean(mix_mono)
    ref_mono -= np.mean(ref_mono)

    norm_mix = np.linalg.norm(mix_mono)
    norm_ref = np.linalg.norm(ref_mono)

    if norm_mix < 1e-6 or norm_ref < 1e-6:
        # One of the signals is virtually silent
        return AlignmentResult(
            lag_samples=0,
            lag_seconds=0.0,
            subsample_offset=0.0,
            total_lag_samples=0.0,
            correlation_coefficient=0.0,
            is_aligned=False,
            reliability_score=0.0,
            chain_latency_samples=chain_latency_samples,
            sample_rate=sample_rate,
        )

    # Compute cross-correlation: ref relative to mix
    # Positive lag means ref is delayed relative to mix (ref appears later)
    max_lag = int(max_search_sec * sample_rate)
    corr = signal.correlate(ref_mono, mix_mono, mode='full')
    lags = signal.correlation_lags(len(ref_mono), len(mix_mono), mode='full')

    # Restrict to search window
    mask = (lags >= -max_lag) & (lags <= max_lag)
    search_corr = corr[mask]
    search_lags = lags[mask]

    peak_idx = int(np.argmax(search_corr))
    best_lag = int(search_lags[peak_idx])

    # F-07: Normalized Pearson correlation computed specifically on the overlapping support
    # Positive lag: ref is delayed relative to mix -> ref[best_lag:] corresponds to mix[:T - best_lag]
    # Negative lag: ref is advanced relative to mix -> mix[-best_lag:] corresponds to ref[:T + best_lag]
    T_ref = len(ref_mono)
    T_mix = len(mix_mono)
    if best_lag >= 0:
        ref_overlap = ref_mono[best_lag : min(T_ref, T_mix + best_lag)]
        mix_overlap = mix_mono[0 : len(ref_overlap)]
    else:
        D = -best_lag
        mix_overlap = mix_mono[D : min(T_mix, T_ref + D)]
        ref_overlap = ref_mono[0 : len(mix_overlap)]

    overlap_len = len(ref_overlap)
    min_overlap = int(0.1 * sample_rate)  # At least 100ms overlap

    if overlap_len >= min_overlap:
        ref_ov_zero = ref_overlap - np.mean(ref_overlap)
        mix_ov_zero = mix_overlap - np.mean(mix_overlap)
        norm_r = np.linalg.norm(ref_ov_zero)
        norm_m = np.linalg.norm(mix_ov_zero)
        if norm_r > 1e-6 and norm_m > 1e-6:
            norm_corr = float(np.dot(ref_ov_zero, mix_ov_zero) / (norm_r * norm_m))
        else:
            norm_corr = 0.0
    else:
        norm_corr = 0.0

    norm_corr = max(-1.0, min(1.0, norm_corr))

    # Sub-sample parabolic interpolation around peak
    subsample_delta = 0.0
    if 0 < peak_idx < len(search_corr) - 1:
        y_m1 = search_corr[peak_idx - 1]
        y_0 = search_corr[peak_idx]
        y_p1 = search_corr[peak_idx + 1]
        denom = 2.0 * (y_m1 - 2.0 * y_0 + y_p1)
        if abs(denom) > 1e-12:
            delta = (y_m1 - y_p1) / denom
            if -0.5 <= delta <= 0.5:
                subsample_delta = float(delta)

    is_aligned = norm_corr >= min_correlation_threshold
    reliability = float(max(0.0, norm_corr))

    return AlignmentResult(
        lag_samples=best_lag,
        lag_seconds=float(best_lag / sample_rate),
        subsample_offset=subsample_delta,
        total_lag_samples=float(best_lag + subsample_delta),
        correlation_coefficient=norm_corr,
        is_aligned=is_aligned,
        reliability_score=reliability,
        chain_latency_samples=chain_latency_samples,
        sample_rate=sample_rate,
    )


def apply_fractional_delay(
    audio: np.ndarray,
    subsample_offset: float,
    filter_half_length: int = 32,
) -> np.ndarray:
    """Apply sub-sample fractional delay compensation via windowed sinc FIR filter.

    Convention:
        subsample_offset (delta): fractional sample delay of audio relative to target.
        When delta > 0, audio is delayed by delta samples.
        To align with target, audio must be advanced by delta samples (shifted by +delta).
        When delta < 0, audio is advanced by -delta samples.
        To align with target, audio must be delayed by -delta samples (shifted by delta).

    Filter formulation:
        h[k] = sinc(k + delta) * blackman(2K + 1) for k in [-K, K], normalized so sum(h) = 1.
        Convolution with h[k] computes y[n] = sum_k h[k] x[n - k], which evaluates
        x at t + delta, compensating the fractional delay with > 65 dB attenuation
        of sinc truncation ripples and precision < 1e-4.

    Args:
        audio: Input audio array with shape [channels, T] or [T].
        subsample_offset: Fractional offset delta in [-0.5, 0.5] samples.
        filter_half_length: K = half-length of Blackman-windowed sinc kernel (default: 32).

    Returns:
        Fractional-delay compensated audio with identical shape.
    """
    if abs(subsample_offset) < 1e-4:
        return np.ascontiguousarray(audio)

    K = filter_half_length
    k = np.arange(-K, K + 1, dtype=np.float64)
    w = np.blackman(2 * K + 1)
    # sinc(k + delta) shifts signal by +delta (evaluating at future t + delta)
    h = np.sinc(k + subsample_offset) * w
    h_sum = np.sum(h)
    if abs(h_sum) > 1e-12:
        h /= h_sum

    was_1d = audio.ndim == 1
    if was_1d:
        audio = audio[np.newaxis, :]

    out = np.zeros_like(audio)
    for c in range(audio.shape[0]):
        out[c] = signal.convolve(audio[c], h, mode="same")

    if was_1d:
        out = out[0]
    return np.ascontiguousarray(out)


def get_same_track_slice_indices(
    alignment: AlignmentResult,
    mix_len: int,
    ref_len: int,
    enable_subsample: bool = True,
    filter_half_length: int = 32,
) -> Tuple[int, int, int]:
    """Compute (mix_start, ref_start, valid_length) for same-track alignment.

    Args:
        alignment: AlignmentResult with estimated lag and sub-sample offset.
        mix_len: Number of samples in unmastered mix [T_mix].
        ref_len: Number of samples in reference master [T_ref].
        enable_subsample: Whether sub-sample fractional compensation is enabled.
        filter_half_length: Half-length K of windowed sinc FIR filter.

    Returns:
        mix_start: Starting sample index in raw mix audio.
        ref_start: Starting sample index in reference audio.
        valid_length: Common overlapping length in samples.
    """
    lag = alignment.lag_samples
    if lag >= 0:
        valid_len = min(mix_len, ref_len - lag)
        if valid_len <= 0:
            raise ValueError(
                f"No valid overlapping temporal region between mix (len={mix_len}) "
                f"and ref (len={ref_len}) with lag={lag} samples"
            )
        mix_start = 0
        ref_start = lag
    else:
        pos_lag = -lag
        valid_len = min(mix_len - pos_lag, ref_len)
        if valid_len <= 0:
            raise ValueError(
                f"No valid overlapping temporal region between mix (len={mix_len}) "
                f"and ref (len={ref_len}) with lag={lag} samples"
            )
        mix_start = pos_lag
        ref_start = 0

    if enable_subsample and abs(alignment.subsample_offset) >= 1e-4:
        K = filter_half_length
        if valid_len > 2 * K + 64:
            mix_start += K
            ref_start += K
            valid_len -= 2 * K

    return mix_start, ref_start, valid_len


def apply_same_track_alignment(
    mix: np.ndarray,
    ref: np.ndarray,
    alignment: AlignmentResult,
    enable_subsample: bool = True,
    filter_half_length: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align ref audio onto mix audio timeline using estimated lag.

    Convention:
        lag_samples > 0: ref is delayed relative to mix (ref appears later in time).
            mix aligned: mix[:, 0 : valid_len]
            ref aligned: ref[:, lag : lag + valid_len]
        lag_samples < 0: ref is advanced relative to mix (ref appears earlier in time).
            pos_lag = -lag
            mix aligned: mix[:, pos_lag : pos_lag + valid_len]
            ref aligned: ref[:, 0 : valid_len]
        where valid_len is the common overlapping duration.

    Sub-sample refinement:
        If enable_subsample=True and alignment.subsample_offset != 0,
        apply_fractional_delay is applied to ref_aligned to compensate the sub-sample delay.
        To maintain an exact common valid temporal support without FIR boundary transients,
        both mix and ref are trimmed by filter_half_length samples on both ends when
        valid_len > 2 * filter_half_length + 64.

    Crucial latency separation:
        This aligns file-level delay between raw mix and master files.
        It does NOT apply chain latency compensation (159 samples at 48kHz), which
        is the internal DDSP latency compensated separately when evaluating DDSP output.

    Raises:
        ValueError: If signals have no valid overlapping temporal region.
    """
    T_mix = mix.shape[1]
    T_ref = ref.shape[1]

    mix_start, ref_start, valid_len = get_same_track_slice_indices(
        alignment,
        mix_len=T_mix,
        ref_len=T_ref,
        enable_subsample=False,  # slice integer bounds first
        filter_half_length=filter_half_length,
    )

    aligned_mix = mix[:, mix_start : mix_start + valid_len]
    aligned_ref = ref[:, ref_start : ref_start + valid_len]

    # Sub-sample fractional delay compensation
    if enable_subsample and abs(alignment.subsample_offset) >= 1e-4:
        aligned_ref = apply_fractional_delay(
            aligned_ref,
            subsample_offset=alignment.subsample_offset,
            filter_half_length=filter_half_length,
        )
        # Symmetrically trim boundary transient to maintain exact common valid region
        K = filter_half_length
        if aligned_mix.shape[1] > 2 * K + 64:
            aligned_mix = aligned_mix[:, K : -K]
            aligned_ref = aligned_ref[:, K : -K]

    return np.ascontiguousarray(aligned_mix), np.ascontiguousarray(aligned_ref)


def assert_stylistic_guards(mode: ReferenceMode) -> None:
    """Enforce strict guardrails against waveform matching in stylistic mode."""
    if mode == ReferenceMode.STYLISTIC:
        raise ValueError(
            "Guardrail Error: Point-to-point waveform alignment and frame-by-frame STFT loss "
            "are strictly prohibited in 'stylistic' reference mode. "
            "Use AggregatePerceptualLoss for distributional/feature matching."
        )
