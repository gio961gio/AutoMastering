"""Objective Audio Measurement & Null-Test Analysis for Human Validation Exports."""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from ddsp_mastering.loss.loudness import (
    DifferentiableLoudnessLoss,
    calculate_integrated_loudness_bs1770,
)
from ddsp_mastering.optimize.metrics import (
    MasteringBenchmarkSuite,
    calculate_sample_peak_dbfs,
    calculate_true_peak_dbtp,
)
from scripts.human_validation.validate_inputs import (
    compute_file_sha256,
    load_and_validate_wav,
    validate_audio_array,
)


def measure_audio_signal(
    audio: np.ndarray,
    sample_rate: float = 48000.0,
    file_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute standardized audio engineering metrics for an audio array or exported WAV."""
    stereo_2xT, mono_dup = validate_audio_array(audio, allow_mono=True)
    num_samples = int(stereo_2xT.shape[1])
    duration_sec = float(num_samples / float(sample_rate))

    sp_dbfs = calculate_sample_peak_dbfs(stereo_2xT)
    tp_dbtp = calculate_true_peak_dbtp(stereo_2xT, sample_rate=sample_rate)

    # RMS in dBFS across both channels
    mean_sq = float(np.mean(stereo_2xT ** 2))
    rms_dbfs = float(10.0 * math.log10(mean_sq)) if mean_sq > 1e-24 else -float("inf")

    # Integrated Loudness BS.1770-5 (with explicit fallback tracking if < 400 ms)
    try:
        lufs = float(calculate_integrated_loudness_bs1770(stereo_2xT, sample_rate=int(round(sample_rate))))
        loudness_method = "itu_bs1770_5_integrated"
    except ValueError:
        proxy_calc = DifferentiableLoudnessLoss(sample_rate=sample_rate)
        t_in = torch.from_numpy(stereo_2xT.astype(np.float32)).unsqueeze(0)
        lufs = float(proxy_calc.calculate_ungated_k_weighted_loudness(t_in).mean().item())
        loudness_method = "ungated_k_weighted_proxy_short_audio"

    # PLR (Peak-to-Loudness Ratio)
    if math.isfinite(sp_dbfs) and math.isfinite(lufs):
        plr_db: Optional[float] = float(sp_dbfs - lufs)
    else:
        plr_db = None

    if math.isfinite(tp_dbtp) and math.isfinite(lufs):
        tp_plr_db: Optional[float] = float(tp_dbtp - lufs)
    else:
        tp_plr_db = None

    # Stereo correlation in [-1, +1]
    left = stereo_2xT[0]
    right = stereo_2xT[1]
    denom = float(np.sqrt(np.sum(left ** 2) * np.sum(right ** 2)))
    if denom > 1e-20:
        stereo_corr: Optional[float] = float(np.clip(np.sum(left * right) / denom, -1.0, 1.0))
    else:
        stereo_corr = None

    # Mid-to-Side energy ratio in dB
    t_stereo = torch.from_numpy(stereo_2xT.astype(np.float32)).unsqueeze(0)
    ms_ratio_db = float(MasteringBenchmarkSuite.calculate_mid_side_energy_ratio(t_stereo))

    res: Dict[str, Any] = {
        "sample_rate": int(round(sample_rate)),
        "channels": 2,
        "mono_duplicated_to_stereo": bool(mono_dup),
        "num_samples": num_samples,
        "duration_sec": duration_sec,
        "sample_peak_dbfs": float(sp_dbfs),
        "true_peak_dbtp": float(tp_dbtp),
        "integrated_lufs": float(lufs),
        "loudness_method": loudness_method,
        "rms_dbfs": float(rms_dbfs),
        "plr_db": plr_db,
        "plr_definition": "sample_peak_dbfs - integrated_lufs",
        "true_peak_plr_db": tp_plr_db,
        "stereo_correlation": stereo_corr,
        "mid_side_ratio_db": ms_ratio_db,
    }
    if file_path is not None and os.path.isfile(file_path):
        res["file_name"] = os.path.basename(file_path)
        res["sha256"] = compute_file_sha256(file_path)
    return res


def compute_null_test_metrics(sig_a: np.ndarray, sig_b: np.ndarray) -> Dict[str, float]:
    """Compute sample-by-sample null difference metrics between two aligned signals."""
    a, _ = validate_audio_array(sig_a, allow_mono=True)
    b, _ = validate_audio_array(sig_b, allow_mono=True)
    if a.shape != b.shape:
        raise ValueError(f"Null test requires identical shapes, got {a.shape} vs {b.shape}")

    diff = a - b
    max_abs = float(np.max(np.abs(diff)))
    rms_diff = float(np.sqrt(np.mean(diff ** 2)))
    max_err_db = float(20.0 * math.log10(max_abs + 1e-12))
    rms_null_db = float(20.0 * math.log10(rms_diff + 1e-12))
    return {
        "max_abs_diff": max_abs,
        "rms_diff": rms_diff,
        "max_err_dbfs": max_err_db,
        "rms_null_dbfs": rms_null_db,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze exported WAV files for Human Validation.")
    parser.add_argument("files", nargs="+", help="WAV file path(s) to analyze.")
    args = parser.parse_args()

    out = []
    for p in args.files:
        audio, meta = load_and_validate_wav(p, required_sr=48000, allow_mono=True)
        metrics = measure_audio_signal(audio, sample_rate=48000.0, file_path=p)
        metrics["subtype"] = meta["subtype"]
        out.append(metrics)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
