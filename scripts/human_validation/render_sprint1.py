"""Experiment A (Sprint 1): Neutrality, True Bypass & Legacy Initialization Baseline Renderer.

Renders the 3 conditions of Sprint 1 on the exact same input audio (at 48 kHz) after
applying a uniform static diagnostic headroom trim (default target: -6.0 dBTP):
  - Condition A: True Bypass (`chain.set_bypass(True)`)
  - Condition B: Legacy Initialization Baseline (`z=0.5` from pre-Gate-A commit `86ae20f08dc7fbebc7f67af064a15b648a492fc7`)
  - Condition C: Corrected Neutral (`THETA_NEUTRAL` / `Z_NEUTRAL` imported directly from `ddsp_mastering.param_space`)

Strict Compatibility Contract for Condition B (Section 3):
  At historical commit `86ae20f08dc7fbebc7f67af064a15b648a492fc7`, `u = zeros(41)` produced `z = [0.5]^41`.
  Under the historical `PARAM_SPECS` of `86ae20f08dc7fbebc7f67af064a15b648a492fc7`, 5 of the 41 physical
  parameters fall OUTSIDE the current frozen engine's `PARAM_SPECS` bounds:
    - [21] comp_low_ratio = 4.472136 (current bounds: [1.0, 4.0])
    - [22] comp_low_attack_ms = 7.071068 ms (current bounds: [10.0, 120.0] ms)
    - [26] comp_mid_ratio = 4.472136 (current bounds: [1.0, 4.0])
    - [31] comp_high_ratio = 4.472136 (current bounds: [1.0, 4.0])
    - [38] saturator_drive_db = 9.0 dB (current bounds: [0.0, 6.0] dB)
  Furthermore, the historical saturator normalized `tanh(x * gd)` by `1 / sqrt(gd)` whereas the
  current frozen engine normalizes by `1 / gd`.

  Therefore:
  - By default (`--legacy-mode strict`), this script does NOT force out-of-range parameters or
    fabricate a fake baseline. It writes a structured finding `LEGACY_BASELINE_NOT_REPRODUCIBLE`
    (`finding_legacy_baseline.json`) explaining the exact 5 out-of-range parameters and exits
    after exporting Conditions A and C (unless an explicit replay policy is chosen).
  - If the user/supervisor explicitly passes `--legacy-mode clamped_to_current_bounds`,
    `--legacy-mode raw_theta_on_current_engine`, or `--legacy-mode current_z_half`, Condition B
    is rendered and labeled as:
      "Legacy initialization baseline replayed on frozen current engine"
    with the chosen policy documented in `trackXX_B_manifest.json`.
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from ddsp_mastering.param_space import NUM_PARAMS, PARAM_SPECS, THETA_NEUTRAL, Z_NEUTRAL
from scripts.human_validation import (
    ENGINE_REFERENCE_CODE_COMMIT,
    ENGINE_REFERENCE_DOC_COMMIT,
    LEGACY_BASELINE_SOURCE_COMMIT,
    PACKAGE_VERSION,
)
from scripts.human_validation.analyze_exports import (
    compute_null_test_metrics,
    measure_audio_signal,
)
from scripts.human_validation.make_blind_set import create_blind_listening_set
from scripts.human_validation.make_level_matched import create_level_matched_wavs
from scripts.human_validation.offline_renderer import (
    compute_diagnostic_headroom_trim,
    render_offline_with_tail_flush,
)
from scripts.human_validation.validate_inputs import (
    check_overwrite_allowed,
    load_and_validate_wav,
)


# Exact historical parameter specifications from commit 86ae20f08dc7fbebc7f67af064a15b648a492fc7
# (`git show 86ae20f08dc7fbebc7f67af064a15b648a492fc7:python/ddsp_mastering/param_space.py`)
LEGACY_PARAM_SPECS_86AE20F0: List[Tuple[str, float, float, bool]] = [
    # 0..17: 6-Band Parametric EQ
    ("eq_band1_f0", 20.0, 250.0, True),
    ("eq_band1_gain_db", -12.0, 12.0, False),
    ("eq_band1_q", 0.5, 5.0, True),
    ("eq_band2_f0", 100.0, 500.0, True),
    ("eq_band2_gain_db", -12.0, 12.0, False),
    ("eq_band2_q", 0.5, 5.0, True),
    ("eq_band3_f0", 400.0, 2000.0, True),
    ("eq_band3_gain_db", -12.0, 12.0, False),
    ("eq_band3_q", 0.5, 5.0, True),
    ("eq_band4_f0", 1500.0, 6000.0, True),
    ("eq_band4_gain_db", -12.0, 12.0, False),
    ("eq_band4_q", 0.5, 5.0, True),
    ("eq_band5_f0", 4000.0, 12000.0, True),
    ("eq_band5_gain_db", -12.0, 12.0, False),
    ("eq_band5_q", 0.5, 5.0, True),
    ("eq_band6_f0", 6000.0, 15000.0, True),
    ("eq_band6_gain_db", -12.0, 12.0, False),
    ("eq_band6_q", 0.5, 5.0, True),
    # 18..19: Crossover
    ("xover_f_low", 80.0, 350.0, True),
    ("xover_f_high", 1500.0, 6000.0, True),
    # 20..34: 3-Band Compressor
    ("comp_low_thresh_db", -40.0, 0.0, False),
    ("comp_low_ratio", 1.0, 20.0, True),
    ("comp_low_attack_ms", 0.5, 100.0, True),
    ("comp_low_release_ms", 10.0, 1000.0, True),
    ("comp_low_makeup_db", -6.0, 12.0, False),
    ("comp_mid_thresh_db", -40.0, 0.0, False),
    ("comp_mid_ratio", 1.0, 20.0, True),
    ("comp_mid_attack_ms", 0.5, 100.0, True),
    ("comp_mid_release_ms", 10.0, 1000.0, True),
    ("comp_mid_makeup_db", -6.0, 12.0, False),
    ("comp_high_thresh_db", -40.0, 0.0, False),
    ("comp_high_ratio", 1.0, 20.0, True),
    ("comp_high_attack_ms", 0.5, 100.0, True),
    ("comp_high_release_ms", 10.0, 1000.0, True),
    ("comp_high_makeup_db", -6.0, 12.0, False),
    # 35..37: Multiband Stereo Imager
    ("imager_width_low", 0.0, 1.2, False),
    ("imager_width_mid", 0.5, 2.0, False),
    ("imager_width_high", 0.5, 2.0, False),
    # 38..39: Harmonic Saturator
    ("saturator_drive_db", 0.0, 18.0, False),
    ("saturator_mix", 0.0, 1.0, False),
    # 40: Lookahead Limiter
    ("limiter_push_gain_db", 0.0, 12.0, False),
]


def reconstruct_legacy_baseline_86ae20f0() -> Tuple[List[float], List[Dict[str, Any]]]:
    """Reconstruct the 41-parameter physical vector theta_legacy at z=0.5 from commit 86ae20f0."""
    assert len(LEGACY_PARAM_SPECS_86AE20F0) == NUM_PARAMS
    theta_legacy: List[float] = []
    explanations: List[Dict[str, Any]] = []

    for idx, (legacy_name, l_min, l_max, l_is_log) in enumerate(LEGACY_PARAM_SPECS_86AE20F0):
        z_val = 0.5
        if l_is_log:
            val = float(l_min * ((l_max / l_min) ** z_val))
            formula = f"{l_min} * ({l_max}/{l_min})^0.5"
        else:
            val = float(l_min + z_val * (l_max - l_min))
            formula = f"{l_min} + 0.5 * ({l_max} - {l_min})"

        curr_spec = PARAM_SPECS[idx]
        in_curr_bounds = bool(curr_spec.min_val - 1e-9 <= val <= curr_spec.max_val + 1e-9)
        clamped_val = float(min(max(val, curr_spec.min_val), curr_spec.max_val))

        theta_legacy.append(val)
        explanations.append({
            "index": idx,
            "name": curr_spec.name,
            "legacy_name_at_86ae20f0": legacy_name,
            "legacy_z": 0.5,
            "legacy_bounds_86ae20f0": [l_min, l_max],
            "legacy_is_log_86ae20f0": l_is_log,
            "formula_at_z_0_5": formula,
            "theta_legacy_value": val,
            "current_engine_bounds": [curr_spec.min_val, curr_spec.max_val],
            "current_engine_is_log": curr_spec.is_log,
            "theta_neutral_current": float(THETA_NEUTRAL[idx]),
            "compatible_with_current_bounds": in_curr_bounds,
            "clamped_to_current_bounds_value": clamped_val,
        })

    return theta_legacy, explanations


def check_legacy_baseline_compatibility() -> Dict[str, Any]:
    """Verify whether the historical z=0.5 baseline from 86ae20f0 is compatible with the frozen current engine."""
    theta_legacy, table = reconstruct_legacy_baseline_86ae20f0()
    out_of_range = [row for row in table if not row["compatible_with_current_bounds"]]

    return {
        "finding_code": "LEGACY_BASELINE_NOT_REPRODUCIBLE" if out_of_range else "LEGACY_BASELINE_COMPATIBLE",
        "historical_source_commit": LEGACY_BASELINE_SOURCE_COMMIT,
        "frozen_engine_reference_commit": ENGINE_REFERENCE_CODE_COMMIT,
        "historical_initialization_mechanism": (
            "In commit 86ae20f08dc7fbebc7f67af064a15b648a492fc7, AutoMasteringOptimizer initialized "
            "u = torch.zeros(1, 41), which mapped via sigmoid(0) to z = [0.5]^41."
        ),
        "is_strictly_compatible_with_current_frozen_bounds": len(out_of_range) == 0,
        "num_total_params": NUM_PARAMS,
        "num_out_of_range_params": len(out_of_range),
        "out_of_range_parameters": out_of_range,
        "dsp_topology_differences_since_86ae20f0": [
            (
                "HarmonicSaturator gain normalization changed in Sprint 1 (Gate A): at 86ae20f0, "
                "tanh(x * gd) was divided by sqrt(gd); in the frozen current engine, it is divided "
                "by gd for unity small-signal gain."
            ),
            (
                "ParamSpace physical bounds were tightened in Sprint 1 (Gate A) so compressor ratios "
                "are capped at 4.0:1 (vs 20.0:1 at 86ae20f0), low-band attack starts at 10.0 ms "
                "(vs 0.5 ms at 86ae20f0), and saturator drive is capped at 6.0 dB (vs 18.0 dB at 86ae20f0)."
            ),
        ],
        "theta_legacy_86ae20f0": theta_legacy,
        "parameter_provenance_table": table,
        "available_decision_options": {
            "strict": (
                "Default per Section 3: stop Condition B without forcing out-of-range values and emit "
                "this LEGACY_BASELINE_NOT_REPRODUCIBLE finding so the user/supervisor can choose a replay policy."
            ),
            "clamped_to_current_bounds": (
                "Replay theta_legacy on the frozen current engine after clamping the 5 out-of-range "
                "values to the current PARAM_SPECS bounds (comp ratios -> 4.0, comp_low_attack -> 10.0 ms, "
                "saturator_drive -> 6.0 dB). Keeps all 36 in-range legacy values untouched."
            ),
            "raw_theta_on_current_engine": (
                "Pass the exact 41-element theta_legacy vector directly to py_core_dsp.MasteringChain.set_parameters_theta "
                "(which accepts physical floats in C++ without Python bound clamping). Labeled explicitly as "
                "'Legacy initialization baseline replayed on frozen current engine (unclamped physical theta)'."
            ),
            "current_z_half": (
                "Map z = [0.5]^41 through the current frozen PARAM_SPECS. Not historical 86ae20f0 theta, "
                "available only if explicitly requested for comparison."
            ),
        },
    }


def _get_git_head_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT), text=True
        ).strip()
        return out
    except Exception:
        return "unknown"


def _get_dependency_versions() -> Dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "soundfile": sf.__version__,
    }


def run_experiment_a_sprint1(
    input_wav_path: str,
    output_dir: str,
    track_id: str = "track01",
    diagnostic_target_dbtp: float = -6.0,
    legacy_mode: str = "strict",
    seed: int = 2026,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Execute Experiment A (Sprint 1) on a 48 kHz input WAV."""
    valid_modes = ("strict", "clamped_to_current_bounds", "raw_theta_on_current_engine", "current_z_half")
    if legacy_mode not in valid_modes:
        raise ValueError(f"Invalid legacy_mode '{legacy_mode}'. Must be one of {valid_modes}.")

    os.makedirs(output_dir, exist_ok=True)
    exports_dir = os.path.join(output_dir, "exports")
    manifests_dir = os.path.join(output_dir, "manifests")
    blind_dir = os.path.join(output_dir, "blind")
    os.makedirs(exports_dir, exist_ok=True)
    os.makedirs(manifests_dir, exist_ok=True)

    # 1. Validate and load source WAV (48 kHz strictly required)
    raw_audio, input_meta = load_and_validate_wav(input_wav_path, required_sr=48000, allow_mono=True)

    # Save source SHA-256 and metadata
    sha_file = os.path.join(output_dir, f"{track_id}_source.sha256")
    meta_file = os.path.join(output_dir, f"{track_id}_metadata.json")
    check_overwrite_allowed(sha_file, overwrite=overwrite)
    check_overwrite_allowed(meta_file, overwrite=overwrite)

    # 2. Apply uniform static diagnostic headroom trim (-6.0 dBTP default)
    trimmed_audio, trim_meta = compute_diagnostic_headroom_trim(
        raw_audio,
        sample_rate=48000.0,
        target_dbtp=diagnostic_target_dbtp,
        trim_only_if_above=True,
    )

    # 3. Check Legacy Baseline compatibility (Section 3)
    legacy_finding = check_legacy_baseline_compatibility()
    finding_path = os.path.join(output_dir, "finding_legacy_baseline.json")
    check_overwrite_allowed(finding_path, overwrite=overwrite)
    with open(finding_path, "w", encoding="utf-8") as f:
        json.dump(legacy_finding, f, indent=2)

    with open(sha_file, "w", encoding="utf-8") as f:
        f.write(f"{input_meta['sha256']}  {input_meta['file_name']}\n")

    git_head = _get_git_head_sha()
    deps = _get_dependency_versions()

    # 4. Render Condition A: True Bypass
    path_a = os.path.join(exports_dir, f"{track_id}_A_true_bypass.wav")
    manifest_a_path = os.path.join(manifests_dir, f"{track_id}_A_manifest.json")
    check_overwrite_allowed(path_a, overwrite=overwrite)
    check_overwrite_allowed(manifest_a_path, overwrite=overwrite)

    audio_a, render_meta_a = render_offline_with_tail_flush(
        trimmed_audio, mode="bypass", sample_rate=48000.0
    )
    sf.write(path_a, audio_a.T, 48000, subtype="FLOAT")
    metrics_a = measure_audio_signal(audio_a, sample_rate=48000.0, file_path=path_a)
    null_a_vs_trimmed = compute_null_test_metrics(audio_a, trimmed_audio)

    manifest_a: Dict[str, Any] = {
        "track_id": track_id,
        "condition": "A_true_bypass",
        "condition_description": "True Bypass (unprocessed signal path with 159-sample delay line compensated)",
        "source_metadata": input_meta,
        "diagnostic_headroom_trim": trim_meta,
        "engine_provenance": {
            "engine_reference_code_commit": ENGINE_REFERENCE_CODE_COMMIT,
            "engine_reference_doc_commit": ENGINE_REFERENCE_DOC_COMMIT,
            "package_head_commit": git_head,
            "package_version": PACKAGE_VERSION,
            "dependencies": deps,
        },
        "preset": {
            "preset_name": "True Bypass",
            "bypass_active": True,
            "theta_41": None,
            "z_41": None,
        },
        "render_details": render_meta_a,
        "null_test_vs_trimmed_input": null_a_vs_trimmed,
        "audio_metrics": metrics_a,
    }
    with open(manifest_a_path, "w", encoding="utf-8") as f:
        json.dump(manifest_a, f, indent=2)

    # 5. Render Condition C: Corrected Neutral (loaded directly from ddsp_mastering.param_space)
    path_c = os.path.join(exports_dir, f"{track_id}_C_corrected_neutral.wav")
    manifest_c_path = os.path.join(manifests_dir, f"{track_id}_C_manifest.json")
    check_overwrite_allowed(path_c, overwrite=overwrite)
    check_overwrite_allowed(manifest_c_path, overwrite=overwrite)

    theta_c = [float(v) for v in THETA_NEUTRAL]
    z_c = [float(v) for v in Z_NEUTRAL]
    audio_c, render_meta_c = render_offline_with_tail_flush(
        trimmed_audio, mode="theta", theta=theta_c, sample_rate=48000.0
    )
    sf.write(path_c, audio_c.T, 48000, subtype="FLOAT")
    metrics_c = measure_audio_signal(audio_c, sample_rate=48000.0, file_path=path_c)
    null_c_vs_a = compute_null_test_metrics(audio_c, audio_a)

    manifest_c: Dict[str, Any] = {
        "track_id": track_id,
        "condition": "C_corrected_neutral",
        "condition_description": "Corrected Neutral preset (THETA_NEUTRAL / Z_NEUTRAL from ddsp_mastering.param_space)",
        "source_metadata": input_meta,
        "diagnostic_headroom_trim": trim_meta,
        "engine_provenance": {
            "engine_reference_code_commit": ENGINE_REFERENCE_CODE_COMMIT,
            "engine_reference_doc_commit": ENGINE_REFERENCE_DOC_COMMIT,
            "package_head_commit": git_head,
            "package_version": PACKAGE_VERSION,
            "dependencies": deps,
        },
        "preset": {
            "preset_name": "Corrected Neutral (THETA_NEUTRAL)",
            "bypass_active": False,
            "theta_41": theta_c,
            "z_41": z_c,
            "param_names": [s.name for s in PARAM_SPECS],
        },
        "render_details": render_meta_c,
        "null_test_vs_true_bypass": null_c_vs_a,
        "audio_metrics": metrics_c,
    }
    with open(manifest_c_path, "w", encoding="utf-8") as f:
        json.dump(manifest_c, f, indent=2)

    # 6. Handle Condition B (Legacy Initialization Baseline) according to legacy_mode
    exported_files_for_matching: Dict[str, str] = {
        "A_true_bypass": path_a,
        "C_corrected_neutral": path_c,
    }

    manifest_b: Optional[Dict[str, Any]] = None
    if legacy_mode != "strict":
        path_b = os.path.join(exports_dir, f"{track_id}_B_legacy_baseline.wav")
        manifest_b_path = os.path.join(manifests_dir, f"{track_id}_B_manifest.json")
        check_overwrite_allowed(path_b, overwrite=overwrite)
        check_overwrite_allowed(manifest_b_path, overwrite=overwrite)

        if legacy_mode == "clamped_to_current_bounds":
            theta_b = [
                float(row["clamped_to_current_bounds_value"])
                for row in legacy_finding["parameter_provenance_table"]
            ]
            preset_label = (
                "Legacy initialization baseline replayed on frozen current engine "
                "(86ae20f0 z=0.5 clamped to current PARAM_SPECS bounds)"
            )
            audio_b, render_meta_b = render_offline_with_tail_flush(
                trimmed_audio, mode="theta", theta=theta_b, sample_rate=48000.0
            )
        elif legacy_mode == "raw_theta_on_current_engine":
            theta_b = [float(v) for v in legacy_finding["theta_legacy_86ae20f0"]]
            preset_label = (
                "Legacy initialization baseline replayed on frozen current engine "
                "(86ae20f0 z=0.5 raw physical theta)"
            )
            audio_b, render_meta_b = render_offline_with_tail_flush(
                trimmed_audio, mode="theta", theta=theta_b, sample_rate=48000.0
            )
        elif legacy_mode == "current_z_half":
            z_half = [0.5] * NUM_PARAMS
            from ddsp_mastering.param_space import z_to_theta
            theta_b_tensor = z_to_theta(torch.tensor(z_half, dtype=torch.float64))
            theta_b = [float(v) for v in theta_b_tensor.tolist()]
            preset_label = (
                "Legacy initialization baseline replayed on frozen current engine "
                "(z=0.5 mapped via current PARAM_SPECS)"
            )
            audio_b, render_meta_b = render_offline_with_tail_flush(
                trimmed_audio, mode="theta", theta=theta_b, sample_rate=48000.0
            )
        else:
            raise ValueError(f"Unhandled legacy_mode: {legacy_mode}")

        sf.write(path_b, audio_b.T, 48000, subtype="FLOAT")
        metrics_b = measure_audio_signal(audio_b, sample_rate=48000.0, file_path=path_b)
        null_b_vs_a = compute_null_test_metrics(audio_b, audio_a)

        manifest_b = {
            "track_id": track_id,
            "condition": "B_legacy_baseline",
            "condition_description": "Legacy initialization baseline replayed on frozen current engine",
            "legacy_replay_policy": legacy_mode,
            "legacy_historical_source_commit": LEGACY_BASELINE_SOURCE_COMMIT,
            "source_metadata": input_meta,
            "diagnostic_headroom_trim": trim_meta,
            "engine_provenance": {
                "engine_reference_code_commit": ENGINE_REFERENCE_CODE_COMMIT,
                "engine_reference_doc_commit": ENGINE_REFERENCE_DOC_COMMIT,
                "package_head_commit": git_head,
                "package_version": PACKAGE_VERSION,
                "dependencies": deps,
            },
            "preset": {
                "preset_name": preset_label,
                "bypass_active": False,
                "theta_41": theta_b,
                "param_names": [s.name for s in PARAM_SPECS],
                "compatibility_finding": legacy_finding["finding_code"],
                "out_of_range_parameters_at_86ae20f0": legacy_finding["out_of_range_parameters"],
            },
            "render_details": render_meta_b,
            "null_test_vs_true_bypass": null_b_vs_a,
            "audio_metrics": metrics_b,
        }
        with open(manifest_b_path, "w", encoding="utf-8") as f:
            json.dump(manifest_b, f, indent=2)

        exported_files_for_matching = {
            "A_true_bypass": path_a,
            "B_legacy_baseline": path_b,
            "C_corrected_neutral": path_c,
        }

    # 7. Create loudness-matched copies (static gain only, ceiling <= -1.0 dBTP)
    level_match_report = create_level_matched_wavs(
        input_files=exported_files_for_matching,
        output_dir=exports_dir,
        target_lufs=None,
        ceiling_dbtp=-1.0,
        suffix="_level_matched",
        overwrite=overwrite,
    )

    # 8. Create blind listening set (for the level-matched copies so loudness bias is removed,
    #    plus program-level blind set)
    lm_files_for_blind: Dict[str, str] = {
        label: info["output_path"]
        for label, info in level_match_report["variants"].items()
    }
    blind_res = create_blind_listening_set(
        variant_files=lm_files_for_blind,
        output_dir=blind_dir,
        seed=seed,
        session_id=f"{track_id}_sprint1",
        key_output_path=os.path.join(blind_dir, f"{track_id}_blind_key.json"),
        overwrite=overwrite,
    )

    summary_payload: Dict[str, Any] = {
        "track_id": track_id,
        "status": (
            "FINDING_LEGACY_BASELINE_NOT_REPRODUCIBLE_STRICT"
            if legacy_mode == "strict"
            else "ALL_THREE_CONDITIONS_EXPORTED"
        ),
        "legacy_mode": legacy_mode,
        "legacy_finding_code": legacy_finding["finding_code"],
        "legacy_finding_file": os.path.abspath(finding_path),
        "source_metadata": input_meta,
        "diagnostic_headroom_trim": trim_meta,
        "null_test_A_vs_trimmed_input": null_a_vs_trimmed,
        "null_test_C_vs_A": null_c_vs_a,
        "exports": exported_files_for_matching,
        "level_matching_report": os.path.abspath(os.path.join(exports_dir, "level_matching_report.json")),
        "blind_session": {
            k: v for k, v in blind_res.items() if k != "_internal_key"
        },
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render Sprint 1 Human Validation Experiment A (True Bypass / Legacy Baseline / Corrected Neutral)."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the 48 kHz unmastered mix WAV file.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for Experiment A artifacts (e.g. human_validation/private/track01).",
    )
    parser.add_argument(
        "--track-id",
        type=str,
        default="track01",
        help="Anonymous identifier for the track (default: track01).",
    )
    parser.add_argument(
        "--diagnostic-target-dbtp",
        type=float,
        default=-6.0,
        help="Uniform diagnostic headroom target in dBTP (default: -6.0 dBTP).",
    )
    parser.add_argument(
        "--legacy-mode",
        type=str,
        default="strict",
        choices=["strict", "clamped_to_current_bounds", "raw_theta_on_current_engine", "current_z_half"],
        help=(
            "Policy for Condition B (Legacy Baseline from 86ae20f0). "
            "'strict' (default) emits finding LEGACY_BASELINE_NOT_REPRODUCIBLE because 5/41 legacy params "
            "exceed current PARAM_SPECS bounds, and exports A and C. "
            "Choose 'clamped_to_current_bounds', 'raw_theta_on_current_engine', or 'current_z_half' "
            "to also export Condition B under an explicit replay policy."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Deterministic seed for blind set ordering (default: 2026).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing files in the output directory.",
    )
    args = parser.parse_args()

    res = run_experiment_a_sprint1(
        input_wav_path=args.input,
        output_dir=args.out,
        track_id=args.track_id,
        diagnostic_target_dbtp=args.diagnostic_target_dbtp,
        legacy_mode=args.legacy_mode,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
