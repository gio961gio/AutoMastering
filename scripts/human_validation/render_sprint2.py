"""Experiment B (Sprint 2): Multi-Chunk Optimization & Multi-Section A/B Human Validation Renderer.

Executes the Sprint 2 multi-chunk optimization protocol (`same_track` or `stylistic`)
to find a **single preset per track** (`z_opt`), renders the full-track master and
True Bypass reference via the offline tail-flush renderer (159-sample latency compensated
once, zero tail truncation), extracts at least 3 distinct listening sections (quiet/intro-verse,
dense/chorus-drop, transient/bridge-outro) without artificial fades, and produces:
  - `<prefix>_original_or_bypass.wav`
  - `<prefix>_mastered_full.wav`
  - `<prefix>_preset.json`
  - `<prefix>_manifest.json`
  - `<prefix>_metrics_pre_post.json`
  - `<prefix>_sections.json` (recording both optimization training chunks and listening sections)
  - `<prefix>_sections/` (program-level section pairs)
  - `<prefix>_sections_level_matched/` (static-gain loudness-matched section pairs)
  - `<prefix>_sections_blind/` (anonymized blind pairs + secret key per section)
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

from ddsp_mastering.data import (
    MultiChunkSelector,
    ReferenceMode,
    apply_same_track_alignment,
    estimate_time_alignment,
    get_same_track_slice_indices,
)
from ddsp_mastering.optimize import AutoMasteringOptimizer, MasteringBenchmarkSuite
from ddsp_mastering.param_space import PARAM_SPECS, map_z_to_theta
from scripts.human_validation import (
    ENGINE_REFERENCE_CODE_COMMIT,
    ENGINE_REFERENCE_DOC_COMMIT,
    PACKAGE_VERSION,
)
from scripts.human_validation.analyze_exports import measure_audio_signal
from scripts.human_validation.make_blind_set import create_blind_listening_set
from scripts.human_validation.make_level_matched import create_level_matched_wavs
from scripts.human_validation.offline_renderer import render_offline_with_tail_flush
from scripts.human_validation.validate_inputs import (
    check_overwrite_allowed,
    load_and_validate_wav,
    validate_audio_array,
)


def select_listening_sections(
    audio_2xT: np.ndarray,
    sample_rate: float = 48000.0,
    section_duration_sec: float = 12.0,
    explicit_sections: Optional[List[Tuple[str, float, float]]] = None,
) -> List[Dict[str, Any]]:
    """Select at least 3 distinct listening sections (quiet, dense, transient) without overlap when possible.

    Args:
        audio_2xT: Stereo audio array of shape (2, T).
        sample_rate: Sample rate in Hz (default: 48000.0).
        section_duration_sec: Target duration per listening section in seconds (10-20s recommended).
        explicit_sections: Optional list of `(role_label, start_sec, end_sec)` tuples.

    Returns:
        List of section descriptor dicts with `section_id`, `role`, `start_sec`, `end_sec`,
        `start_sample`, `end_sample`, `duration_sec`, `rms_dbfs`, `crest_db`.
    """
    stereo, _ = validate_audio_array(audio_2xT, allow_mono=True)
    total_samples = int(stereo.shape[1])
    sr = int(round(sample_rate))
    total_sec = float(total_samples / float(sr))

    if explicit_sections:
        sections: List[Dict[str, Any]] = []
        for idx, (role, start_s, end_s) in enumerate(explicit_sections):
            s_smp = max(0, int(round(start_s * sr)))
            e_smp = min(total_samples, int(round(end_s * sr)))
            if e_smp <= s_smp:
                raise ValueError(f"Invalid explicit section ({role}, {start_s}, {end_s}) for track of {total_sec:.2f}s.")
            seg = stereo[:, s_smp:e_smp]
            rms_val = float(np.sqrt(np.mean(seg ** 2)) + 1e-12)
            peak_val = float(np.max(np.abs(seg)) + 1e-12)
            sections.append({
                "section_id": f"section{idx + 1:02d}",
                "role": str(role),
                "start_sec": float(s_smp / float(sr)),
                "end_sec": float(e_smp / float(sr)),
                "duration_sec": float((e_smp - s_smp) / float(sr)),
                "start_sample": s_smp,
                "end_sample": e_smp,
                "rms_dbfs": float(20.0 * math.log10(rms_val)),
                "crest_factor_db": float(20.0 * math.log10(peak_val / rms_val)),
            })
        return sections

    # Automatic selection of 3 sections: quiet (intro/verse), dense (chorus/drop), transient (bridge/outro)
    win_samples = int(round(section_duration_sec * sr))
    if total_samples < win_samples * 3:
        # Adapt window length if the test track is shorter than 3 * section_duration_sec
        win_samples = max(1, total_samples // 3)

    hop_samples = max(1, min(sr // 2, win_samples // 4))
    candidates: List[Dict[str, Any]] = []

    for s_smp in range(0, max(1, total_samples - win_samples + 1), hop_samples):
        e_smp = min(total_samples, s_smp + win_samples)
        seg = stereo[:, s_smp:e_smp]
        mean_sq = float(np.mean(seg ** 2))
        rms_lin = math.sqrt(mean_sq) if mean_sq > 1e-24 else 1e-12
        peak_lin = float(np.max(np.abs(seg)))
        rms_db = float(20.0 * math.log10(max(rms_lin, 1e-12)))
        crest_db = float(20.0 * math.log10(max(peak_lin, 1e-12) / max(rms_lin, 1e-12)))
        candidates.append({
            "start_sample": s_smp,
            "end_sample": e_smp,
            "rms_dbfs": rms_db,
            "crest_factor_db": crest_db,
            "is_active": bool(rms_db > -55.0),
        })

    active_cands = [c for c in candidates if c["is_active"]] or candidates

    def _overlaps(c1: Dict[str, Any], chosen: List[Dict[str, Any]]) -> bool:
        for ch in chosen:
            if not (c1["end_sample"] <= ch["start_sample"] or c1["start_sample"] >= ch["end_sample"]):
                return True
        return False

    chosen_sections: List[Tuple[str, Dict[str, Any]]] = []

    # 1. Dense section (highest RMS energy)
    by_rms_desc = sorted(active_cands, key=lambda c: c["rms_dbfs"], reverse=True)
    dense_cand = by_rms_desc[0]
    chosen_sections.append(("dense_chorus_or_drop", dense_cand))

    # 2. Quiet section (lowest active RMS that does not overlap dense_cand)
    by_rms_asc = sorted(active_cands, key=lambda c: c["rms_dbfs"])
    quiet_cand = next((c for c in by_rms_asc if not _overlaps(c, [dense_cand])), by_rms_asc[0])
    chosen_sections.append(("quiet_intro_or_verse", quiet_cand))

    # 3. Transient section (highest crest factor that does not overlap already chosen)
    by_crest_desc = sorted(active_cands, key=lambda c: c["crest_factor_db"], reverse=True)
    trans_cand = next(
        (c for c in by_crest_desc if not _overlaps(c, [dense_cand, quiet_cand])),
        by_crest_desc[0],
    )
    chosen_sections.append(("transient_bridge_or_outro", trans_cand))

    # Sort chosen sections chronologically for natural listening order
    chosen_sections.sort(key=lambda pair: pair[1]["start_sample"])

    result: List[Dict[str, Any]] = []
    for idx, (role, c) in enumerate(chosen_sections):
        s_smp = int(c["start_sample"])
        e_smp = int(c["end_sample"])
        result.append({
            "section_id": f"section{idx + 1:02d}",
            "role": role,
            "start_sec": float(s_smp / float(sr)),
            "end_sec": float(e_smp / float(sr)),
            "duration_sec": float((e_smp - s_smp) / float(sr)),
            "start_sample": s_smp,
            "end_sample": e_smp,
            "rms_dbfs": float(c["rms_dbfs"]),
            "crest_factor_db": float(c["crest_factor_db"]),
        })
    return result


def parse_explicit_sections_arg(sections_arg: Optional[List[str]]) -> Optional[List[Tuple[str, float, float]]]:
    """Parse CLI `--sections role:start:end` specifications."""
    if not sections_arg:
        return None
    parsed: List[Tuple[str, float, float]] = []
    for idx, item in enumerate(sections_arg):
        parts = item.split(":")
        if len(parts) == 2:
            role = f"custom_section_{idx + 1:02d}"
            start_s, end_s = float(parts[0]), float(parts[1])
        elif len(parts) == 3:
            role = parts[0].strip()
            start_s, end_s = float(parts[1]), float(parts[2])
        else:
            raise ValueError(
                f"Invalid --sections item '{item}'. Expected format 'role:start_sec:end_sec' or 'start_sec:end_sec'."
            )
        parsed.append((role, start_s, end_s))
    return parsed


def _get_git_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def run_experiment_b_sprint2(
    raw_wav_path: str,
    ref_wav_path: str,
    output_dir: str,
    prefix: str = "track01",
    mode: str = "same_track",
    num_chunks: int = 4,
    chunk_len_sec: float = 3.0,
    num_steps: int = 30,
    lr: float = 0.04,
    seed: int = 2026,
    section_duration_sec: float = 12.0,
    explicit_sections: Optional[List[Tuple[str, float, float]]] = None,
    desired_sound_notes: str = "",
    keys_dir: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Execute Experiment B (Sprint 2) for a single track and produce all required exports and sections."""
    if not raw_wav_path or not os.path.isfile(raw_wav_path):
        raise FileNotFoundError(
            f"Missing raw mix WAV file: '{raw_wav_path}'. "
            f"Provide a valid 48 kHz unmastered mix WAV."
        )
    if not ref_wav_path or not os.path.isfile(ref_wav_path):
        raise FileNotFoundError(
            f"Missing reference WAV file: '{ref_wav_path}'. "
            f"Per Section 9, Experiment B requires an explicit reference WAV (either same_track or stylistic) "
            f"and refuses to fabricate a synthetic reference."
        )

    ref_mode = ReferenceMode(mode)
    os.makedirs(output_dir, exist_ok=True)
    actual_keys_dir = keys_dir if keys_dir is not None else os.path.join(output_dir, "keys")
    os.makedirs(actual_keys_dir, exist_ok=True)

    # Check output paths before running optimization
    orig_wav_path = os.path.join(output_dir, f"{prefix}_original_or_bypass.wav")
    master_wav_path = os.path.join(output_dir, f"{prefix}_mastered_full.wav")
    preset_json_path = os.path.join(output_dir, f"{prefix}_preset.json")
    manifest_json_path = os.path.join(output_dir, f"{prefix}_manifest.json")
    metrics_json_path = os.path.join(output_dir, f"{prefix}_metrics_pre_post.json")
    sections_json_path = os.path.join(output_dir, f"{prefix}_sections.json")

    for p in (
        orig_wav_path,
        master_wav_path,
        preset_json_path,
        manifest_json_path,
        metrics_json_path,
        sections_json_path,
    ):
        check_overwrite_allowed(p, overwrite=overwrite)

    # 1. Load & validate 48 kHz inputs (never resample silently)
    raw_full_64, raw_meta = load_and_validate_wav(raw_wav_path, required_sr=48000, allow_mono=True)
    ref_full_64, ref_meta = load_and_validate_wav(ref_wav_path, required_sr=48000, allow_mono=True)

    raw_full_32 = np.ascontiguousarray(raw_full_64, dtype=np.float32)
    ref_full_32 = np.ascontiguousarray(ref_full_64, dtype=np.float32)

    # 2. Time alignment (same_track only)
    alignment_meta: Dict[str, Any] = {}
    warnings_list: List[str] = []

    if ref_mode == ReferenceMode.SAME_TRACK:
        alignment = estimate_time_alignment(raw_full_32, ref_full_32, sample_rate=48000.0)
        alignment_meta = alignment.to_dict()
        if not alignment.is_aligned:
            warnings_list.append(
                f"Low cross-correlation ({alignment.correlation_coefficient:.4f}) in same_track mode; "
                f"verify that mix and reference are truly the same musical performance."
            )
        raw_aligned, ref_aligned = apply_same_track_alignment(raw_full_32, ref_full_32, alignment)
    else:
        alignment = None
        raw_aligned = raw_full_32
        ref_aligned = ref_full_32

    # 3. Multi-chunk selection for optimization
    selector = MultiChunkSelector(
        sample_rate=48000.0,
        num_chunks=num_chunks,
        chunk_duration_sec=chunk_len_sec,
        seed=seed,
    )
    mix_chunks_spec = selector.select_chunks(raw_aligned)
    mix_chunks_np = selector.extract_chunk_tensors(raw_aligned, mix_chunks_spec)
    mix_chunk_tensors = [torch.tensor(c, dtype=torch.float32).unsqueeze(0) for c in mix_chunks_np]

    if ref_mode == ReferenceMode.SAME_TRACK:
        ref_chunk_tensors = [
            torch.tensor(ref_aligned[:, c.start_sample : c.end_sample], dtype=torch.float32).unsqueeze(0)
            for c in mix_chunks_spec
        ]
        ref_chunks_spec_dicts = [c.to_dict() for c in mix_chunks_spec]
    else:
        ref_chunks_spec = selector.select_chunks(ref_full_32)
        ref_chunks_np = selector.extract_chunk_tensors(ref_full_32, ref_chunks_spec)
        ref_chunk_tensors = [torch.tensor(c, dtype=torch.float32).unsqueeze(0) for c in ref_chunks_np]
        ref_chunks_spec_dicts = [c.to_dict() for c in ref_chunks_spec]

    # 4. Optimize a single preset per track using the frozen optimizer
    optimizer = AutoMasteringOptimizer(
        sample_rate=48000.0,
        lr=lr,
        num_steps=num_steps,
        target_lufs=None,
        weight_stft=1.0,
        weight_loudness=0.8,
        weight_param_reg=0.01,
        use_hybrid_eq=True,
    )
    z_opt, _, history = optimizer.optimize_multi_chunk(
        mix_chunks=mix_chunk_tensors,
        ref_chunks=ref_chunk_tensors,
        reference_mode=mode,
        verbose=False,
        align_latency=True,
    )

    # Save canonical preset JSON
    optimizer.export_preset_json(z_opt, output_path=preset_json_path)

    # 5. Offline full-track render with 159-sample latency compensation & tail flush
    bypass_full_64, bypass_render_meta = render_offline_with_tail_flush(
        raw_full_64, mode="bypass", sample_rate=48000.0
    )
    mastered_full_64, master_render_meta = render_offline_with_tail_flush(
        raw_full_64, mode="z", z=z_opt.tolist(), sample_rate=48000.0
    )

    sf.write(orig_wav_path, bypass_full_64.T, 48000, subtype="FLOAT")
    sf.write(master_wav_path, mastered_full_64.T, 48000, subtype="FLOAT")

    orig_signal_metrics = measure_audio_signal(bypass_full_64, sample_rate=48000.0, file_path=orig_wav_path)
    master_signal_metrics = measure_audio_signal(mastered_full_64, sample_rate=48000.0, file_path=master_wav_path)
    ref_signal_metrics = measure_audio_signal(ref_full_64, sample_rate=48000.0, file_path=ref_wav_path)

    # 6. Pre/Post Benchmark Suite Evaluation
    suite = MasteringBenchmarkSuite(sample_rate=48000.0)
    if ref_mode == ReferenceMode.SAME_TRACK:
        mix_start, ref_start, eval_len = get_same_track_slice_indices(
            alignment,
            mix_len=raw_full_64.shape[1],
            ref_len=ref_full_64.shape[1],
            enable_subsample=True,
        )
        raw_eval_tensor = torch.tensor(raw_aligned, dtype=torch.float32).unsqueeze(0)
        ref_eval_tensor = torch.tensor(ref_aligned, dtype=torch.float32).unsqueeze(0)
        # Note: mastered_full_64 is already latency-compensated by render_offline_with_tail_flush!
        # Therefore we slice [mix_start : mix_start + eval_len] and pass align_latency=False so
        # the 159-sample latency is NOT compensated a second time.
        mastered_eval_seg = mastered_full_64[:, mix_start : mix_start + eval_len].astype(np.float32)
        mastered_eval_tensor = torch.tensor(mastered_eval_seg, dtype=torch.float32).unsqueeze(0)

        pre_bench = suite.evaluate(
            raw_eval_tensor, ref_eval_tensor, align_latency=False, reference_mode="same_track"
        )
        post_bench = suite.evaluate(
            mastered_eval_tensor, ref_eval_tensor, align_latency=False, reference_mode="same_track"
        )
    else:
        raw_eval_tensor = torch.tensor(raw_full_32, dtype=torch.float32).unsqueeze(0)
        ref_eval_tensor = torch.tensor(ref_full_32, dtype=torch.float32).unsqueeze(0)
        mastered_eval_tensor = torch.tensor(
            mastered_full_64.astype(np.float32), dtype=torch.float32
        ).unsqueeze(0)

        pre_bench = suite.evaluate(
            raw_eval_tensor, ref_eval_tensor, align_latency=False, reference_mode="stylistic"
        )
        post_bench = suite.evaluate(
            mastered_eval_tensor, ref_eval_tensor, align_latency=False, reference_mode="stylistic"
        )

    # 7. Extract >= 3 listening sections (without artificial fades)
    sections_dir = os.path.join(output_dir, f"{prefix}_sections")
    sections_lm_dir = os.path.join(output_dir, f"{prefix}_sections_level_matched")
    sections_blind_dir = os.path.join(output_dir, f"{prefix}_sections_blind_delivery")
    os.makedirs(sections_dir, exist_ok=True)
    os.makedirs(sections_lm_dir, exist_ok=True)
    os.makedirs(sections_blind_dir, exist_ok=True)

    listening_sections = select_listening_sections(
        bypass_full_64,
        sample_rate=48000.0,
        section_duration_sec=section_duration_sec,
        explicit_sections=explicit_sections,
    )

    section_artifacts: List[Dict[str, Any]] = []
    for idx, sec in enumerate(listening_sections):
        sec_id = sec["section_id"]
        s_smp = int(sec["start_sample"])
        e_smp = int(sec["end_sample"])

        orig_sec_audio = bypass_full_64[:, s_smp:e_smp]
        mast_sec_audio = mastered_full_64[:, s_smp:e_smp]

        orig_sec_path = os.path.join(sections_dir, f"{sec_id}_original.wav")
        mast_sec_path = os.path.join(sections_dir, f"{sec_id}_mastered.wav")
        check_overwrite_allowed(orig_sec_path, overwrite=overwrite)
        check_overwrite_allowed(mast_sec_path, overwrite=overwrite)

        sf.write(orig_sec_path, orig_sec_audio.T, 48000, subtype="FLOAT")
        sf.write(mast_sec_path, mast_sec_audio.T, 48000, subtype="FLOAT")

        orig_sec_metrics = measure_audio_signal(orig_sec_audio, sample_rate=48000.0, file_path=orig_sec_path)
        mast_sec_metrics = measure_audio_signal(mast_sec_audio, sample_rate=48000.0, file_path=mast_sec_path)

        # Level-matched pair for this section (static gain only, <= -1.0 dBTP ceiling)
        sec_lm_subdir = os.path.join(sections_lm_dir, sec_id)
        lm_report = create_level_matched_wavs(
            input_files={
                f"{sec_id}_original": orig_sec_path,
                f"{sec_id}_mastered": mast_sec_path,
            },
            output_dir=sec_lm_subdir,
            target_lufs=None,
            ceiling_dbtp=-1.0,
            suffix="_level_matched",
            overwrite=overwrite,
        )

        # Blind set for this section (both program-level and level-matched)
        # Secret keys are saved strictly in actual_keys_dir, separate from delivery audio
        prog_seed = int(seed + (idx + 1) * 1009)
        prog_key_file = os.path.join(actual_keys_dir, f"{prefix}_{sec_id}_program_blind_key.json")
        sec_blind_prog_dir = os.path.join(sections_blind_dir, f"{sec_id}_program_level")
        blind_prog = create_blind_listening_set(
            variant_files={
                "original_or_bypass": orig_sec_path,
                "mastered": mast_sec_path,
            },
            output_dir=sec_blind_prog_dir,
            seed=prog_seed,
            session_id=f"{prefix}_{sec_id}_program",
            key_output_path=prog_key_file,
            overwrite=overwrite,
        )

        lm_seed = int(seed + (idx + 1) * 2017)
        lm_key_file = os.path.join(actual_keys_dir, f"{prefix}_{sec_id}_level_matched_blind_key.json")
        sec_blind_lm_dir = os.path.join(sections_blind_dir, f"{sec_id}_level_matched")
        blind_lm = create_blind_listening_set(
            variant_files={
                "original_or_bypass_lm": lm_report["variants"][f"{sec_id}_original"]["output_path"],
                "mastered_lm": lm_report["variants"][f"{sec_id}_mastered"]["output_path"],
            },
            output_dir=sec_blind_lm_dir,
            seed=lm_seed,
            session_id=f"{prefix}_{sec_id}_level_matched",
            key_output_path=lm_key_file,
            overwrite=overwrite,
        )

        sec_entry = dict(sec)
        sec_entry["program_level_files"] = {
            "original_wav": os.path.abspath(orig_sec_path),
            "original_metrics": orig_sec_metrics,
            "mastered_wav": os.path.abspath(mast_sec_path),
            "mastered_metrics": mast_sec_metrics,
        }
        sec_entry["level_matched_report"] = lm_report
        sec_entry["blind_program_level"] = {
            k: v for k, v in blind_prog.items() if k != "_internal_key"
        }
        sec_entry["blind_level_matched"] = {
            k: v for k, v in blind_lm.items() if k != "_internal_key"
        }
        sec_entry["secret_keys"] = {
            "program_level_key": os.path.abspath(prog_key_file),
            "level_matched_key": os.path.abspath(lm_key_file),
        }
        section_artifacts.append(sec_entry)

    # 8. Write sections JSON (containing both optimization training chunks and listening sections)
    sections_payload: Dict[str, Any] = {
        "track_prefix": prefix,
        "reference_mode": mode,
        "optimization_training_chunks": [c.to_dict() for c in mix_chunks_spec],
        "reference_chunks": ref_chunks_spec_dicts,
        "listening_sections": section_artifacts,
    }
    with open(sections_json_path, "w", encoding="utf-8") as f:
        json.dump(sections_payload, f, indent=2)

    # 9. Write quantitative pre/post metrics JSON
    theta_vec = map_z_to_theta(z_opt)
    param_dict = {spec.name: float(theta_vec[i].item()) for i, spec in enumerate(PARAM_SPECS)}

    metrics_payload: Dict[str, Any] = {
        "disclaimer": (
            "Objective metrics and loss reductions do not by themselves prove perceptual improvement. "
            "Human listening across all sections (program-level and loudness-matched) is mandatory."
        ),
        "track_prefix": prefix,
        "reference_mode": mode,
        "full_track_signal_metrics": {
            "original_or_bypass": orig_signal_metrics,
            "mastered_full": master_signal_metrics,
            "reference": ref_signal_metrics,
        },
        "benchmark_suite_pre": pre_bench,
        "benchmark_suite_post": post_bench,
        "Applicable_spectral_metrics_note": (
            "spectral_convergence and log_spectral_distance_db are valid ONLY in 'same_track' mode. "
            "In 'stylistic' mode only aggregate_spectral_distance is applicable."
        ),
        "warnings": warnings_list,
    }
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    # 10. Write complete track manifest JSON
    manifest_payload: Dict[str, Any] = {
        "track_prefix": prefix,
        "reference_mode": mode,
        "desired_sound_notes": desired_sound_notes,
        "engine_provenance": {
            "engine_reference_code_commit": ENGINE_REFERENCE_CODE_COMMIT,
            "engine_reference_doc_commit": ENGINE_REFERENCE_DOC_COMMIT,
            "package_head_commit": _get_git_head_sha(),
            "package_version": PACKAGE_VERSION,
        },
        "input_mix_metadata": raw_meta,
        "reference_metadata": ref_meta,
        "alignment": alignment_meta,
        "optimization_config": {
            "single_preset_per_track": True,
            "num_chunks": int(num_chunks),
            "chunk_duration_sec": float(chunk_len_sec),
            "seed": int(seed),
            "num_steps": int(num_steps),
            "learning_rate": float(lr),
            "use_hybrid_eq": True,
            "initial_loss": float(history["loss"][0]) if ("loss" in history and history["loss"]) else None,
            "final_loss": float(history["loss"][-1]) if ("loss" in history and history["loss"]) else None,
        },
        "optimized_preset": {
            "preset_json_file": os.path.basename(preset_json_path),
            "normalized_z_41": [float(v) for v in z_opt.tolist()],
            "physical_theta_41": [float(v) for v in theta_vec.tolist()],
            "parameters": param_dict,
        },
        "full_track_renders": {
            "original_or_bypass": {
                "file_name": os.path.basename(orig_wav_path),
                "render_details": bypass_render_meta,
                "metrics": orig_signal_metrics,
            },
            "mastered_full": {
                "file_name": os.path.basename(master_wav_path),
                "render_details": master_render_meta,
                "metrics": master_signal_metrics,
            },
        },
        "sections_file": os.path.basename(sections_json_path),
        "metrics_pre_post_file": os.path.basename(metrics_json_path),
        "secret_keys_dir": os.path.abspath(actual_keys_dir),
        "sections_blind_delivery_dir": os.path.abspath(sections_blind_dir),
        "warnings": warnings_list,
    }
    with open(manifest_json_path, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)

    return {
        "track_prefix": prefix,
        "reference_mode": mode,
        "original_or_bypass_wav": os.path.abspath(orig_wav_path),
        "mastered_full_wav": os.path.abspath(master_wav_path),
        "preset_json": os.path.abspath(preset_json_path),
        "manifest_json": os.path.abspath(manifest_json_path),
        "metrics_pre_post_json": os.path.abspath(metrics_json_path),
        "sections_json": os.path.abspath(sections_json_path),
        "secret_keys_dir": os.path.abspath(actual_keys_dir),
        "sections_blind_delivery_dir": os.path.abspath(sections_blind_dir),
        "num_listening_sections": len(section_artifacts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render Sprint 2 Human Validation Experiment B (Multi-Chunk A/B across Sections)."
    )
    parser.add_argument("--raw", type=str, required=True, help="Path to 48 kHz unmastered mix WAV.")
    parser.add_argument("--ref", type=str, required=True, help="Path to 48 kHz reference WAV.")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["same_track", "stylistic"],
        help="Reference mode: 'same_track' (identical musical program) or 'stylistic' (different song).",
    )
    parser.add_argument("--prefix", type=str, default="track01", help="Track identifier prefix (e.g. track01).")
    parser.add_argument("--out", type=str, required=True, help="Output directory for Experiment B artifacts.")
    parser.add_argument(
        "--keys-dir",
        type=str,
        default=None,
        help="Dedicated private directory for secret blind keys (default: <out>/keys).",
    )
    parser.add_argument("--chunks", type=int, default=4, help="Number of optimization chunks (default: 4).")
    parser.add_argument("--chunk-len", type=float, default=3.0, help="Optimization chunk length in seconds (default: 3.0).")
    parser.add_argument("--steps", type=int, default=30, help="Optimization steps (default: 30).")
    parser.add_argument("--lr", type=float, default=0.04, help="Learning rate (default: 0.04).")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed (default: 2026).")
    parser.add_argument(
        "--section-len",
        type=float,
        default=12.0,
        help="Listening section duration in seconds when auto-selected (default: 12.0s).",
    )
    parser.add_argument(
        "--sections",
        nargs="*",
        default=None,
        help="Optional explicit listening sections formatted as 'role:start_sec:end_sec' (at least 3 recommended).",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Brief description of the desired sound for this track.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing files.")
    args = parser.parse_args()

    explicit_secs = parse_explicit_sections_arg(args.sections)
    res = run_experiment_b_sprint2(
        raw_wav_path=args.raw,
        ref_wav_path=args.ref,
        output_dir=args.out,
        prefix=args.prefix,
        mode=args.mode,
        num_chunks=args.chunks,
        chunk_len_sec=args.chunk_len,
        num_steps=args.steps,
        lr=args.lr,
        seed=args.seed,
        section_duration_sec=args.section_len,
        explicit_sections=explicit_secs,
        desired_sound_notes=args.notes,
        keys_dir=args.keys_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
