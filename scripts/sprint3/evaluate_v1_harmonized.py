"""Sprint 3 Step V1 / V1b Harmonized Ablation Runner.

Evaluates V1 (ERB-48) and V1b (ERB-48 + De-Harshness) under the 100% harmonized protocol:
  - Bit-identical chunk selection and coordinates (4 chunks of 144,000 samples / 3.0s at 48 kHz);
  - Bit-identical pre-roll extraction using Percorso A (from mix_aligned in same_track);
  - Bit-identical full-track target LUFS (reference ITU-R BS.1770-4/5 integrated);
  - Exact 30 steps Adam, lr=0.04, hybrid schedule (steps 1-20 FFT, 21-30 IIR);
  - Full unweighted individual loss decomposition across all 10 terms;
  - Blocking assertion against docs/sprint3/SPRINT3_V2_ABLATION_RESULTS.json.

Exports results to docs/sprint3/SPRINT3_V1_HARMONIZED_RESULTS.json without touching historical files.
"""

import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ddsp_mastering.dsp import enable_fast_cpu_biquad
enable_fast_cpu_biquad()

from ddsp_mastering.dsp.chain import DifferentiableMasteringChain
from ddsp_mastering.optimize import AutoMasteringOptimizer, MasteringBenchmarkSuite
from ddsp_mastering.param_space import PARAM_SPECS, Z_NEUTRAL, map_theta_to_z, map_z_to_theta
from scripts.human_validation.offline_renderer import render_offline_with_tail_flush
from scripts.sprint3.harmonized_session_prep import (
    SESSIONS_SPEC,
    compute_individual_loss_breakdown,
    prepare_harmonized_session,
)


def load_reproducibility_manifest() -> Dict[str, Any]:
    p = _REPO_ROOT / "docs" / "human_validation" / "SPRINT2_V0_REPRODUCIBILITY_MANIFEST.json"
    return json.loads(p.read_text(encoding="utf-8"))


def load_v2_ablation_results() -> Dict[str, Any]:
    p = _REPO_ROOT / "docs" / "sprint3" / "SPRINT3_V2_ABLATION_RESULTS.json"
    if not p.exists():
        raise FileNotFoundError(
            f"Required V2 results file not found: {p}. Run evaluate_v2_ablation.py first!"
        )
    return json.loads(p.read_text(encoding="utf-8"))


def evaluate_single_session_v1_harmonized(
    spec: Dict[str, str],
    seed_int: int,
    v2_session_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate V1 and V1b under the harmonized protocol for a single session."""
    prefix = spec["track_prefix"]
    mode = spec["mode"]
    cache_dir = _REPO_ROOT / "docs" / "sprint3" / ".cache_v1_harmonized_final"
    cache_dir.mkdir(parents=True, exist_ok=True)
    session_cache_file = cache_dir / f"{prefix}_v1_harmonized.json"

    if session_cache_file.exists():
        print(f"[{prefix}] Loaded harmonized V1/V1b session from cache: {session_cache_file}", flush=True)
        return json.loads(session_cache_file.read_text(encoding="utf-8"))

    t0 = time.time()
    prep = prepare_harmonized_session(spec, seed_int=seed_int, repo_root=_REPO_ROOT)
    raw_full_64 = prep["raw_full_64"]
    ref_full_64 = prep["ref_full_64"]
    mix_chunks = prep["mix_chunks"]
    ref_chunks = prep["ref_chunks"]
    preroll_list = prep["preroll_list"]
    ref_ext_chunks = prep["ref_ext_chunks"]
    ref_preroll_list = prep["ref_preroll_list"]
    chunk_metadata = prep["chunk_metadata"]
    target_lufs = prep["target_lufs"]
    ref_eval_tensor = prep["ref_eval_tensor"]
    mix_start = prep["mix_start"]
    eval_len = prep["eval_len"]
    suite = MasteringBenchmarkSuite(sample_rate=48000.0)

    # -------------------------------------------------------------------------
    # Strict blocking cross-validation against V2 ablation results
    # -------------------------------------------------------------------------
    assert str(seed_int) == str(v2_session_data["seed"]), (
        f"[{prefix}] Seed mismatch: {seed_int} != {v2_session_data['seed']}"
    )
    assert abs(target_lufs - float(v2_session_data["target_lufs"])) < 1e-5, (
        f"[{prefix}] Target LUFS mismatch: {target_lufs} != {v2_session_data['target_lufs']}"
    )
    assert len(chunk_metadata) == len(v2_session_data["chunk_metadata"]), (
        f"[{prefix}] Chunk metadata length mismatch"
    )
    for k in range(len(chunk_metadata)):
        cm_prep = chunk_metadata[k]
        cm_v2 = v2_session_data["chunk_metadata"][k]
        for key in ["mix_start_sample", "mix_end_sample", "mix_preroll_samples", "ref_start_sample", "ref_end_sample"]:
            assert cm_prep[key] == cm_v2[key], (
                f"[{prefix}] Chunk {k} key {key} mismatch: {cm_prep[key]} != {cm_v2[key]}"
            )

    K = len(mix_chunks)
    weights = [1.0 / K] * K

    def _evaluate_objective_iir(opt_obj: AutoMasteringOptimizer, z_in: torch.Tensor, eval_tlufs: float) -> Tuple[float, Dict[str, float]]:
        with torch.no_grad():
            tot_l = 0.0
            agg_m: Dict[str, float] = {}
            for k in range(K):
                n_pre = preroll_list[k]
                mk = mix_chunks[k]
                y_ext = opt_obj.chain(mk, z=z_in, eq_mode="iir")
                y_k = y_ext[..., n_pre:]
                ref_k = ref_chunks[k]
                ref_ext_k = ref_ext_chunks[k]
                ref_pre_k = ref_preroll_list[k]
                if mode == "same_track":
                    lk, mk_dict = opt_obj.loss_fn(
                        pred_audio=y_k,
                        target_audio=ref_k,
                        target_lufs=eval_tlufs,
                        normalized_params=z_in,
                        align_latency=True,
                        pred_ext_audio=y_ext,
                        pred_preroll_samples=n_pre,
                        ref_ext_audio=ref_ext_k,
                        ref_preroll_samples=ref_pre_k,
                    )
                else:
                    anchor_k = mk[..., n_pre:]
                    lk, mk_dict = opt_obj.stylistic_loss_fn(
                        pred_audio=y_k,
                        ref_audio=ref_k,
                        normalized_params=z_in,
                        target_lufs=eval_tlufs,
                        mix_anchor_audio=anchor_k,
                        pred_ext_audio=y_ext,
                        pred_preroll_samples=n_pre,
                        ref_ext_audio=ref_ext_k,
                        ref_preroll_samples=ref_pre_k,
                    )
                tot_l += weights[k] * float(lk.item())
                for mk_key, mk_val in mk_dict.items():
                    agg_m[mk_key] = agg_m.get(mk_key, 0.0) + weights[k] * float(mk_val)
            return tot_l, agg_m

    name_to_idx = {param_spec.name: i for i, param_spec in enumerate(PARAM_SPECS)}

    # -------------------------------------------------------------------------
    # 1. Variant V1: ERB-48 spectral loss (weight_erb=0.25 in same_track, spectral_mode="erb48" in stylistic)
    # -------------------------------------------------------------------------
    opt_v1 = AutoMasteringOptimizer(
        sample_rate=48000.0,
        lr=0.04,
        num_steps=30,
        target_lufs=target_lufs,
        weight_stft=1.0,
        weight_loudness=0.8,
        weight_param_reg=0.01,
        use_hybrid_eq=True,
        weight_erb=0.25,
        stylistic_spectral_mode="erb48",
        weight_de_harsh=0.0,
        weight_trans=0.0,
        plr_mode="box",
    )

    z_opt_v1, _, _ = opt_v1.optimize_multi_chunk(
        mix_chunks=mix_chunks,
        ref_chunks=ref_chunks,
        reference_mode=mode,
        verbose=False,
        align_latency=True,
        mix_preroll_samples=preroll_list,
        target_lufs=target_lufs,
    )

    loss_z0_v1, metrics_init_v1 = _evaluate_objective_iir(opt_v1, Z_NEUTRAL, target_lufs)
    loss_zopt_v1, metrics_final_v1 = _evaluate_objective_iir(opt_v1, z_opt_v1, target_lufs)
    red_v1 = ((loss_zopt_v1 - loss_z0_v1) / loss_z0_v1) * 100.0 if loss_z0_v1 > 0.0 else 0.0

    breakdown_init_v1 = compute_individual_loss_breakdown(opt_v1.chain, Z_NEUTRAL, prep)
    breakdown_final_v1 = compute_individual_loss_breakdown(opt_v1.chain, z_opt_v1, prep)

    mastered_v1_64, _ = render_offline_with_tail_flush(
        raw_full_64, mode="z", z=z_opt_v1.tolist(), sample_rate=48000.0
    )
    if mode == "same_track":
        mastered_eval_seg_v1 = mastered_v1_64[:, mix_start : mix_start + eval_len].astype(np.float32)
        bm_full_v1 = suite.evaluate(
            torch.tensor(mastered_eval_seg_v1).unsqueeze(0), ref_eval_tensor, align_latency=False, reference_mode="same_track"
        )
    else:
        bm_full_v1 = suite.evaluate(
            torch.tensor(mastered_v1_64.astype(np.float32)).unsqueeze(0), ref_eval_tensor, align_latency=False, reference_mode="stylistic"
        )

    theta_v1 = map_z_to_theta(z_opt_v1)
    theta_v1_dict = {param_spec.name: float(theta_v1[name_to_idx[param_spec.name]].item()) for param_spec in PARAM_SPECS}

    v1_warnings = []
    if bm_full_v1.get("pred_true_peak_dbtp", -99.0) > -0.30:
        v1_warnings.append(f"True peak ceiling violation: {bm_full_v1['pred_true_peak_dbtp']:.2f} dBTP > -0.30 dBTP")

    v1_config = {
        "target_lufs": float(target_lufs),
        "preroll_samples": 120000,
        "weight_erb": 0.25 if mode == "same_track" else 1.0,
        "spectral_mode": "erb48" if mode == "stylistic" else "mel40",
        "weight_de_harsh": 0.0,
        "weight_trans": 0.0,
        "plr_mode": "box",
    }
    v1_data = {
        "config": v1_config,
        "preroll_sec": 2.5,
        "preroll_samples_per_chunk": preroll_list,
        "target_lufs": float(target_lufs),
        "initial_loss_z0_iir": float(loss_z0_v1),
        "final_loss_z_opt_iir": float(loss_zopt_v1),
        "loss_reduction_iir_pct": float(red_v1),
        "chunk_metrics_init_iir": metrics_init_v1,
        "chunk_metrics_final_iir": metrics_final_v1,
        "loss_breakdown_init": breakdown_init_v1,
        "loss_breakdown_final": breakdown_final_v1,
        "full_track_benchmark": bm_full_v1,
        "warnings": v1_warnings,
        "normalized_z_opt": [float(v) for v in z_opt_v1.detach().cpu().numpy().tolist()],
        "physical_parameters_theta": theta_v1_dict,
    }

    # -------------------------------------------------------------------------
    # 2. Variant V1b: ERB-48 + Asymmetric De-Harshness (weight_de_harsh=0.15)
    # -------------------------------------------------------------------------
    opt_v1b = AutoMasteringOptimizer(
        sample_rate=48000.0,
        lr=0.04,
        num_steps=30,
        target_lufs=target_lufs,
        weight_stft=1.0,
        weight_loudness=0.8,
        weight_param_reg=0.01,
        use_hybrid_eq=True,
        weight_erb=0.25,
        stylistic_spectral_mode="erb48",
        weight_de_harsh=0.15,
        weight_trans=0.0,
        plr_mode="box",
    )

    z_opt_v1b, _, _ = opt_v1b.optimize_multi_chunk(
        mix_chunks=mix_chunks,
        ref_chunks=ref_chunks,
        reference_mode=mode,
        verbose=False,
        align_latency=True,
        mix_preroll_samples=preroll_list,
        target_lufs=target_lufs,
    )

    loss_z0_v1b, metrics_init_v1b = _evaluate_objective_iir(opt_v1b, Z_NEUTRAL, target_lufs)
    loss_zopt_v1b, metrics_final_v1b = _evaluate_objective_iir(opt_v1b, z_opt_v1b, target_lufs)
    red_v1b = ((loss_zopt_v1b - loss_z0_v1b) / loss_z0_v1b) * 100.0 if loss_z0_v1b > 0.0 else 0.0

    breakdown_init_v1b = compute_individual_loss_breakdown(opt_v1b.chain, Z_NEUTRAL, prep)
    breakdown_final_v1b = compute_individual_loss_breakdown(opt_v1b.chain, z_opt_v1b, prep)

    mastered_v1b_64, _ = render_offline_with_tail_flush(
        raw_full_64, mode="z", z=z_opt_v1b.tolist(), sample_rate=48000.0
    )
    if mode == "same_track":
        mastered_eval_seg_v1b = mastered_v1b_64[:, mix_start : mix_start + eval_len].astype(np.float32)
        bm_full_v1b = suite.evaluate(
            torch.tensor(mastered_eval_seg_v1b).unsqueeze(0), ref_eval_tensor, align_latency=False, reference_mode="same_track"
        )
    else:
        bm_full_v1b = suite.evaluate(
            torch.tensor(mastered_v1b_64.astype(np.float32)).unsqueeze(0), ref_eval_tensor, align_latency=False, reference_mode="stylistic"
        )

    theta_v1b = map_z_to_theta(z_opt_v1b)
    theta_v1b_dict = {param_spec.name: float(theta_v1b[name_to_idx[param_spec.name]].item()) for param_spec in PARAM_SPECS}

    v1b_warnings = []
    if bm_full_v1b.get("pred_true_peak_dbtp", -99.0) > -0.30:
        v1b_warnings.append(f"True peak ceiling violation: {bm_full_v1b['pred_true_peak_dbtp']:.2f} dBTP > -0.30 dBTP")

    v1b_config = {
        "target_lufs": float(target_lufs),
        "preroll_samples": 120000,
        "weight_erb": 0.25 if mode == "same_track" else 1.0,
        "spectral_mode": "erb48" if mode == "stylistic" else "mel40",
        "weight_de_harsh": 0.15,
        "weight_trans": 0.0,
        "plr_mode": "box",
    }
    v1b_data = {
        "config": v1b_config,
        "preroll_sec": 2.5,
        "preroll_samples_per_chunk": preroll_list,
        "target_lufs": float(target_lufs),
        "initial_loss_z0_iir": float(loss_z0_v1b),
        "final_loss_z_opt_iir": float(loss_zopt_v1b),
        "loss_reduction_iir_pct": float(red_v1b),
        "chunk_metrics_init_iir": metrics_init_v1b,
        "chunk_metrics_final_iir": metrics_final_v1b,
        "loss_breakdown_init": breakdown_init_v1b,
        "loss_breakdown_final": breakdown_final_v1b,
        "full_track_benchmark": bm_full_v1b,
        "warnings": v1b_warnings,
        "normalized_z_opt": [float(v) for v in z_opt_v1b.detach().cpu().numpy().tolist()],
        "physical_parameters_theta": theta_v1b_dict,
    }

    elapsed = time.time() - t0
    session_payload = {
        "track_prefix": prefix,
        "reference_mode": mode,
        "seed": str(seed_int),
        "target_lufs": float(target_lufs),
        "elapsed_sec": float(elapsed),
        "chunk_metadata": chunk_metadata,
        "variants": {
            "V0p": v2_session_data["variants"]["V0p"],
            "V1": v1_data,
            "V1b": v1b_data,
        },
    }

    session_cache_file.write_text(json.dumps(session_payload, indent=2), encoding="utf-8")
    print(
        f"[{prefix}] ({elapsed:.1f}s) "
        f"V1: init={loss_z0_v1:.4f}->final={loss_zopt_v1:.4f} ({red_v1:+.2f}%) | "
        f"V1b: init={loss_z0_v1b:.4f}->final={loss_zopt_v1b:.4f} ({red_v1b:+.2f}%) | "
        f"LUFS={bm_full_v1b['pred_lufs']:.2f} | PLR={bm_full_v1b.get('pred_plr_db', 0.0):.2f}",
        flush=True,
    )
    gc.collect()
    return session_payload


def main():
    print("================================================================================")
    print("Sprint 3 Step V1 / V1b Harmonized Ablation Runner")
    print("================================================================================")

    v2_results = load_v2_ablation_results()
    v2_by_prefix = {s["track_prefix"]: s for s in v2_results["sessions"]}

    manifest = load_reproducibility_manifest()
    seed_by_prefix: Dict[str, int] = {}
    for s in manifest["sessions"]:
        seed_by_prefix[s["track_prefix"]] = int(s["seed"])

    session_results = []
    for spec in SESSIONS_SPEC:
        prefix = spec["track_prefix"]
        seed_int = seed_by_prefix[prefix]
        v2_sess = v2_by_prefix[prefix]
        res = evaluate_single_session_v1_harmonized(spec, seed_int, v2_sess)
        session_results.append(res)

    out_file = _REPO_ROOT / "docs" / "sprint3" / "SPRINT3_V1_HARMONIZED_RESULTS.json"
    results_payload = {
        "artifact_type": "AutoMastering_Sprint3_V1_Harmonized_Results",
        "version": "1.0.0",
        "seed_serialization_note": "All 64-bit unsigned integer seeds (uint64) in 'seed' are serialized as decimal strings.",
        "v1_hyperparameters": {
            "weight_erb_same_track": 0.25,
            "weight_erb_stylistic": 1.0,
            "weight_de_harsh_v1": 0.0,
            "weight_de_harsh_v1b": 0.15,
            "weight_trans": 0.0,
            "plr_mode": "box",
        },
        "sessions": session_results,
    }
    out_file.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
    print(f"\n[DONE] Saved Harmonized V1/V1b results to: {out_file}")


if __name__ == "__main__":
    main()
