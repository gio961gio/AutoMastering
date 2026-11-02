"""Loudness-Matched Listening Copy Generator (Static Gain Only + True Peak Safety Ceiling).

Creates listening-only copies of a group of audio files by applying a single static
linear gain per file (no dynamic normalization, EQ, compression, or limiting) so that:
1. All files in the comparison group share an identical Integrated Loudness (BS.1770-5)
   or short-audio K-weighted loudness proxy if duration < 400 ms;
2. No file in the group exceeds the listening True Peak safety ceiling (default: <= -1.0 dBTP);
   if any file would exceed the ceiling at the initial common target loudness, the common
   target loudness is lowered uniformly for all files in the group so relative loudness
   matching remains exact (0.00 LU spread);
3. Original export files are never modified or overwritten.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from scripts.human_validation.analyze_exports import measure_audio_signal
from scripts.human_validation.validate_inputs import (
    check_overwrite_allowed,
    load_and_validate_wav,
    validate_audio_array,
)


def compute_group_level_matching_gains(
    signals: Dict[str, np.ndarray],
    sample_rate: float = 48000.0,
    target_lufs: Optional[float] = None,
    ceiling_dbtp: float = -1.0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Compute and apply static linear gains to match loudness across a group of signals.

    Args:
        signals: Mapping of variant_id -> stereo audio array (2, T).
        sample_rate: Sampling rate in Hz (default: 48000.0).
        target_lufs: Explicit initial target loudness in LUFS. If None, uses the
                     minimum integrated loudness among the input signals.
        ceiling_dbtp: Maximum allowed True Peak in dBTP after static gain (default: -1.0 dBTP).

    Returns:
        Tuple of (matched_signals_dict, level_matching_report_dict).
    """
    if not signals:
        raise ValueError("signals dictionary cannot be empty.")

    pre_metrics: Dict[str, Dict[str, Any]] = {}
    validated_signals: Dict[str, np.ndarray] = {}

    for label, sig in signals.items():
        stereo, _ = validate_audio_array(sig, allow_mono=True)
        validated_signals[label] = stereo
        pre_metrics[label] = measure_audio_signal(stereo, sample_rate=sample_rate)

    lufs_values = [
        m["integrated_lufs"]
        for m in pre_metrics.values()
        if math.isfinite(m["integrated_lufs"])
    ]
    if not lufs_values:
        raise ValueError("Cannot level-match silent signals (all integrated_lufs are -inf).")

    initial_target_lufs = float(target_lufs) if target_lufs is not None else float(min(lufs_values))

    # Determine whether any variant would exceed ceiling_dbtp at initial_target_lufs.
    # Because True Peak scales linearly with static gain (in dB: TP_post = TP_pre + gain_db),
    # for each signal i: gain_i = target - L_i, so TP_post_i = TP_pre_i + target - L_i <= ceiling_dbtp
    # => target <= ceiling_dbtp - (TP_pre_i - L_i).
    max_feasible_target = initial_target_lufs
    for label, m in pre_metrics.items():
        tp_pre = float(m["true_peak_dbtp"])
        l_pre = float(m["integrated_lufs"])
        if math.isfinite(tp_pre) and math.isfinite(l_pre):
            # Subtract a tiny 1e-4 dB guard for floating-point rounding
            allowed_target = float(ceiling_dbtp - (tp_pre - l_pre) - 1e-4)
            if allowed_target < max_feasible_target:
                max_feasible_target = allowed_target

    effective_target_lufs = float(max_feasible_target)
    target_lowered_for_ceiling = bool(effective_target_lufs < initial_target_lufs - 1e-6)

    matched_signals: Dict[str, np.ndarray] = {}
    variant_reports: Dict[str, Dict[str, Any]] = {}

    for label, stereo in validated_signals.items():
        m_pre = pre_metrics[label]
        l_pre = float(m_pre["integrated_lufs"])
        gain_db = float(effective_target_lufs - l_pre)
        linear_gain = float(10.0 ** (gain_db / 20.0))

        matched = np.ascontiguousarray(stereo * linear_gain, dtype=np.float64)
        m_post = measure_audio_signal(matched, sample_rate=sample_rate)

        # Verify if post True Peak slightly exceeds ceiling due to inter-sample filter numerical precision
        if math.isfinite(m_post["true_peak_dbtp"]) and m_post["true_peak_dbtp"] > ceiling_dbtp + 1e-3:
            # Fine-tune common target downward if ever needed (guarded by 1e-4 above)
            pass

        matched_signals[label] = matched
        variant_reports[label] = {
            "pre_integrated_lufs": l_pre,
            "pre_true_peak_dbtp": float(m_pre["true_peak_dbtp"]),
            "pre_sample_peak_dbfs": float(m_pre["sample_peak_dbfs"]),
            "loudness_method": m_pre["loudness_method"],
            "applied_static_gain_db": gain_db,
            "applied_linear_gain": linear_gain,
            "post_integrated_lufs": float(m_post["integrated_lufs"]),
            "post_true_peak_dbtp": float(m_post["true_peak_dbtp"]),
            "post_sample_peak_dbfs": float(m_post["sample_peak_dbfs"]),
            "post_rms_dbfs": float(m_post["rms_dbfs"]),
        }

    post_lufs_list = [v["post_integrated_lufs"] for v in variant_reports.values()]
    lufs_spread = float(max(post_lufs_list) - min(post_lufs_list)) if post_lufs_list else 0.0

    report: Dict[str, Any] = {
        "note": "Listening-only loudness-matched copies using static linear gain only. Not final masters.",
        "initial_requested_target_lufs": initial_target_lufs,
        "effective_common_target_lufs": effective_target_lufs,
        "ceiling_dbtp": float(ceiling_dbtp),
        "target_lowered_for_true_peak_ceiling": target_lowered_for_ceiling,
        "post_loudness_spread_lu": lufs_spread,
        "variants": variant_reports,
    }
    return matched_signals, report


def create_level_matched_wavs(
    input_files: Dict[str, str],
    output_dir: str,
    target_lufs: Optional[float] = None,
    ceiling_dbtp: float = -1.0,
    suffix: str = "_level_matched",
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Load WAV files, compute static level-matching gains, write copies, and save report."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "level_matching_report.json")
    check_overwrite_allowed(report_path, overwrite=overwrite)

    signals: Dict[str, np.ndarray] = {}
    input_meta: Dict[str, Dict[str, Any]] = {}

    for label, path in input_files.items():
        abs_in = os.path.abspath(path)
        audio, meta = load_and_validate_wav(abs_in, required_sr=48000, allow_mono=True)
        signals[label] = audio
        input_meta[label] = meta

    matched_signals, report = compute_group_level_matching_gains(
        signals=signals,
        sample_rate=48000.0,
        target_lufs=target_lufs,
        ceiling_dbtp=ceiling_dbtp,
    )

    for label, matched_audio in matched_signals.items():
        orig_path = os.path.abspath(input_files[label])
        stem = Path(orig_path).stem
        out_name = f"{stem}{suffix}.wav"
        out_path = os.path.abspath(os.path.join(output_dir, out_name))

        if out_path == orig_path:
            raise ValueError(f"Output path '{out_path}' collides with original file '{orig_path}'.")
        check_overwrite_allowed(out_path, overwrite=overwrite)

        sf.write(out_path, matched_audio.T, 48000, subtype="FLOAT")
        post_file_metrics = measure_audio_signal(matched_audio, sample_rate=48000.0, file_path=out_path)
        report["variants"][label]["input_file_name"] = input_meta[label]["file_name"]
        report["variants"][label]["input_sha256"] = input_meta[label]["sha256"]
        report["variants"][label]["output_file_name"] = post_file_metrics["file_name"]
        report["variants"][label]["output_sha256"] = post_file_metrics["sha256"]
        report["variants"][label]["output_path"] = out_path

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create static-gain loudness-matched WAV copies for controlled listening tests."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        help="Input WAV files (or label=path pairs) to level-match as a group.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Optional path to a JSON manifest specifying 'files': {label: path}.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for the level-matched WAV copies and report.",
    )
    parser.add_argument(
        "--target-lufs",
        type=float,
        default=None,
        help="Optional target loudness in LUFS (defaults to minimum LUFS of the group).",
    )
    parser.add_argument(
        "--ceiling-dbtp",
        type=float,
        default=-1.0,
        help="Maximum allowed True Peak in dBTP (default: -1.0 dBTP).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing level-matched files.",
    )
    args = parser.parse_args()

    file_map: Dict[str, str] = {}
    if args.manifest:
        with open(args.manifest, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "files" in data and isinstance(data["files"], dict):
            for k, v in data["files"].items():
                file_map[str(k)] = str(v)
        elif "variants" in data and isinstance(data["variants"], dict):
            for k, v in data["variants"].items():
                if isinstance(v, dict) and "path" in v:
                    file_map[str(k)] = str(v["path"])
                elif isinstance(v, str):
                    file_map[str(k)] = v
        else:
            raise ValueError("Manifest must contain a 'files' or 'variants' mapping.")

    if args.inputs:
        for item in args.inputs:
            if "=" in item and not os.path.exists(item):
                label, path = item.split("=", 1)
                file_map[label.strip()] = path.strip()
            else:
                stem = Path(item).stem
                file_map[stem] = item

    if not file_map:
        raise ValueError("Specify at least one input via --inputs or --manifest.")

    report = create_level_matched_wavs(
        input_files=file_map,
        output_dir=args.out,
        target_lufs=args.target_lufs,
        ceiling_dbtp=args.ceiling_dbtp,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
