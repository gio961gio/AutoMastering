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

    # Normalized Pearson-like correlation coefficient at peak
    peak_val = search_corr[peak_idx]
    norm_corr = float(peak_val / (norm_mix * norm_ref))
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


def apply_same_track_alignment(
    mix: np.ndarray,
    ref: np.ndarray,
    alignment: AlignmentResult,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align ref audio onto mix audio timeline using estimated lag.

    Crucial: this aligns file-level delay between raw mix and master.
    It does NOT apply chain latency compensation (159 samples), which is only
    compensated when comparing the DDSP output with the reference.
    """
    lag = alignment.lag_samples
    T_mix = mix.shape[1]
    T_ref = ref.shape[1]

    if lag >= 0:
        # Ref starts 'lag' samples later than mix:
        # Aligned region: mix[:, lag:], ref[:, :T_mix - lag]
        valid_len = min(T_mix - lag, T_ref)
        if valid_len <= 0:
            return mix, ref
        aligned_mix = mix[:, lag : lag + valid_len]
        aligned_ref = ref[:, :valid_len]
    else:
        # Ref starts '-lag' samples earlier than mix:
        pos_lag = -lag
        valid_len = min(T_mix, T_ref - pos_lag)
        if valid_len <= 0:
            return mix, ref
        aligned_mix = mix[:, :valid_len]
        aligned_ref = ref[:, pos_lag : pos_lag + valid_len]

    return np.ascontiguousarray(aligned_mix), np.ascontiguousarray(aligned_ref)


def assert_stylistic_guards(mode: ReferenceMode) -> None:
    """Enforce strict guardrails against waveform matching in stylistic mode."""
    if mode == ReferenceMode.STYLISTIC:
        raise ValueError(
            "Guardrail Error: Point-to-point waveform alignment and frame-by-frame STFT loss "
            "are strictly prohibited in 'stylistic' reference mode. "
            "Use AggregatePerceptualLoss for distributional/feature matching."
        )
