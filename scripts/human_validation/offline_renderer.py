"""Offline Renderer with 159-Sample Latency Compensation and Tail Flush.

Processes audio through a freshly initialized instance of the frozen C++20
`py_core_dsp.MasteringChain`, appends zero-padding to flush the 159-sample
internal pipeline delay, and slices `[latency : latency + T]` so that:
1. Output duration and sample count match the input (`T` samples) bit-for-bit;
2. The 159-sample chain latency is compensated exactly once;
3. The final transient/tail of the track is never truncated;
4. Consecutive renders are 100% state-independent.
"""

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Ensure python/ is importable
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import py_core_dsp
from ddsp_mastering.optimize.metrics import calculate_sample_peak_dbfs, calculate_true_peak_dbtp
from scripts.human_validation.validate_inputs import validate_audio_array


DECLARED_LATENCY_SAMPLES: int = int(py_core_dsp.TOTAL_CHAIN_LATENCY_SAMPLES)


def compute_diagnostic_headroom_trim(
    audio_2xT: np.ndarray,
    sample_rate: float = 48000.0,
    target_dbtp: float = -6.0,
    trim_only_if_above: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute and apply a uniform static headroom trim to bring input to a safe diagnostic level.

    Never applies dynamic normalization or compression. The exact same static linear gain
    is meant to be applied before rendering conditions A, B, and C in Experiment A.

    Args:
        audio_2xT: Input audio array of shape (2, T).
        sample_rate: Sample rate in Hz (default: 48000.0).
        target_dbtp: Diagnostic True Peak target in dBTP (default: -6.0 dBTP).
        trim_only_if_above: If True, only attenuates when input True Peak > target_dbtp;
                            if False, normalizes linearly to target_dbtp whenever non-silent.

    Returns:
        Tuple of (trimmed_audio_2xT, trim_metadata_dict).
    """
    stereo, _ = validate_audio_array(audio_2xT, allow_mono=True)
    in_sp_dbfs = calculate_sample_peak_dbfs(stereo)
    in_tp_dbtp = calculate_true_peak_dbtp(stereo, sample_rate=sample_rate)

    if not math.isfinite(in_tp_dbtp):
        trim_db = 0.0
    elif trim_only_if_above and in_tp_dbtp <= target_dbtp:
        trim_db = 0.0
    else:
        trim_db = float(target_dbtp - in_tp_dbtp)

    linear_gain = float(10.0 ** (trim_db / 20.0))
    trimmed = np.ascontiguousarray(stereo * linear_gain, dtype=np.float64)

    out_sp_dbfs = calculate_sample_peak_dbfs(trimmed)
    out_tp_dbtp = calculate_true_peak_dbtp(trimmed, sample_rate=sample_rate)

    meta: Dict[str, Any] = {
        "target_diagnostic_dbtp": float(target_dbtp),
        "trim_only_if_above": bool(trim_only_if_above),
        "input_sample_peak_dbfs": float(in_sp_dbfs),
        "input_true_peak_dbtp": float(in_tp_dbtp),
        "applied_static_trim_db": float(trim_db),
        "applied_linear_gain": float(linear_gain),
        "post_trim_sample_peak_dbfs": float(out_sp_dbfs),
        "post_trim_true_peak_dbtp": float(out_tp_dbtp),
    }
    return trimmed, meta


def render_offline_with_tail_flush(
    audio: np.ndarray,
    mode: str = "theta",
    theta: Optional[Sequence[float]] = None,
    z: Optional[Sequence[float]] = None,
    sample_rate: float = 48000.0,
    flush_samples: Optional[int] = None,
    allow_mono: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Render audio offline through a fresh C++20 MasteringChain with tail flush and PDC slice.

    Args:
        audio: Input audio array (2, T), (T, 2), (1, T), or (T,).
        mode: One of "bypass", "theta", or "z".
        theta: Physical 41-parameter vector (required when mode == "theta").
        z: Normalized 41-parameter vector in [0, 1]^41 (required when mode == "z").
        sample_rate: Sampling rate in Hz (must be 48000.0 per engine contract).
        flush_samples: Zero-padding tail length in samples (default: 159, must be >= 159).
        allow_mono: Whether to duplicate 1-channel input to stereo.

    Returns:
        Tuple of (aligned_output_2xT_float64, render_metadata_dict).
    """
    if abs(sample_rate - 48000.0) > 1e-2:
        raise ValueError(f"Offline renderer requires 48000.0 Hz sample rate, got {sample_rate} Hz.")

    stereo_in, mono_dup = validate_audio_array(audio, allow_mono=allow_mono)
    num_samples = int(stereo_in.shape[1])

    latency = DECLARED_LATENCY_SAMPLES
    if flush_samples is None:
        flush = latency
    else:
        flush = int(flush_samples)
        if flush < latency:
            raise ValueError(
                f"flush_samples ({flush}) must be at least DECLARED_LATENCY_SAMPLES ({latency})."
            )

    # Create a brand-new C++20 MasteringChain instance and reset state
    chain = py_core_dsp.MasteringChain(float(sample_rate))
    chain.reset()

    if mode == "bypass":
        chain.set_bypass(True)
    elif mode == "theta":
        if theta is None or len(theta) != int(py_core_dsp.NUM_PARAMS):
            raise ValueError(f"mode='theta' requires a 41-element theta vector.")
        theta_list = [float(v) for v in theta]
        if not all(math.isfinite(v) for v in theta_list):
            raise ValueError("theta contains NaN or Inf values.")
        chain.set_bypass(False)
        chain.set_parameters_theta(theta_list)
    elif mode == "z":
        if z is None or len(z) != int(py_core_dsp.NUM_PARAMS):
            raise ValueError(f"mode='z' requires a 41-element normalized z vector.")
        z_list = [float(v) for v in z]
        if not all(math.isfinite(v) and -1e-6 <= v <= 1.0 + 1e-6 for v in z_list):
            raise ValueError("z contains NaN, Inf, or values outside [0, 1].")
        chain.set_bypass(False)
        chain.set_parameters_z(z_list)
    else:
        raise ValueError(f"Unrecognized render mode '{mode}'. Must be 'bypass', 'theta', or 'z'.")

    # Pad input with `flush` zero samples at the tail so the internal 159-sample delay line flushes completely
    padded_in = np.pad(
        stereo_in,
        pad_width=((0, 0), (0, flush)),
        mode="constant",
        constant_values=0.0,
    )
    padded_in = np.ascontiguousarray(padded_in, dtype=np.float64)

    raw_out = chain.process(padded_in)
    # Compensate the 159-sample chain latency once and extract exact T samples
    aligned_out = np.ascontiguousarray(raw_out[:, latency : latency + num_samples], dtype=np.float64)

    assert aligned_out.shape == (2, num_samples), (
        f"Expected output shape (2, {num_samples}), got {aligned_out.shape}"
    )

    render_meta: Dict[str, Any] = {
        "mode": mode,
        "bypass_active": bool(mode == "bypass"),
        "sample_rate": float(sample_rate),
        "input_samples": num_samples,
        "output_samples": int(aligned_out.shape[1]),
        "duration_sec": float(num_samples / sample_rate),
        "mono_duplicated_to_stereo": bool(mono_dup),
        "declared_latency_samples": latency,
        "tail_flush_zero_samples": flush,
        "total_processed_samples": int(padded_in.shape[1]),
        "extracted_start_sample": latency,
        "extracted_end_sample": latency + num_samples,
        "latency_compensated_once": True,
    }
    return aligned_out, render_meta
