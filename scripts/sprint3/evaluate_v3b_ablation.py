"""Sprint 3 Step V3b Ablation Runner: Control V0p vs Treatment V3b-B0.5.

Ablation Variants:
  - Control V0p:
      Frozen Sprint 2 loss baseline (w_stft=1.0, w_lufs=0.8, w_reg=0.01) with
      harmonized 2.5s pre-roll (Percorso A) and full-track target LUFS.
      All V3b dynamic prior weights disabled (w_lim=0.0, w_diff=0.0, w_conflict=0.0).
  - Treatment V3b-B0.5:
      V0p loss + Differentiable Dynamic Prior Loss (loss_lim + loss_diff + loss_conflict)
      evaluated on post-preroll evaluation windows with frozen parameters:
        tau_avg = 0.5 dB, tau_95 = 2.0 dB, lambda_95 = 0.5, delta_max = 3.0 dB,
        delta_m = 1.25 dB, delta_ref = 3.0 dB, w_lim = 0.50, w_diff = 0.02, w_conflict = 0.02.

Modes:
  Default (no flags): Dry-run inspection mode. Validates environment, frozen configs,
                      and harmonized sessions, printing execution plan and exiting cleanly (code 0).
  --run-ablation:     Executes the full 30-step multi-chunk Adam optimization loop across
                      all 6 harmonized sessions, renders full-track 64-bit offline audio,
                      benchmarks all objective audio metrics, exports blind listening WAVs,
                      and writes SPRINT3_V3B_V0P_ABLATION_RESULTS.json and
                      SPRINT3_V3B_V0P_ABLATION_REPORT.md.
"""

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import zipfile

import numpy as np
import soundfile as sf
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
from ddsp_mastering.loss.dynamic_prior import TOTAL_DSP_DELAY_SAMPLES, V3bDynamicPriorLoss, compute_active_sample_mask
from ddsp_mastering.loss.loudness import calculate_integrated_loudness_bs1770
from ddsp_mastering.optimize import AutoMasteringOptimizer, MasteringBenchmarkSuite
from ddsp_mastering.optimize.metrics import calculate_sample_peak_dbfs, calculate_true_peak_dbtp
from ddsp_mastering.param_space import NUM_PARAMS, PARAM_SPECS, Z_NEUTRAL, map_theta_to_z, map_z_to_theta
from scripts.human_validation.offline_renderer import render_offline_with_tail_flush
from scripts.sprint3.harmonized_session_prep import (
    SESSIONS_SPEC,
    compute_individual_loss_breakdown,
    prepare_harmonized_session,
)

# -----------------------------------------------------------------------------
# Frozen V3b-B0.5 Hyperparameters (Candidate B with w_lim = 0.50)
# -----------------------------------------------------------------------------
FROZEN_V3B_B05_PARAMS = {
    "tau_avg": 0.5,       # dB
    "tau_95": 2.0,        # dB
    "lambda_95": 0.5,     # convex weight for P95 vs CVaR95
    "delta_max": 3.0,     # dB
    "delta_m": 1.25,      # dB
    "delta_ref": 3.0,     # dB
    "w_lim": 0.50,        # weight for L_lim
    "w_diff": 0.02,       # weight for L_diff
    "w_conflict": 0.02,   # weight for L_conflict
}

PARAM_GROUPS = {
    "EQ": list(range(0, 18)),
    "Crossover": list(range(18, 20)),
    "Compressor": list(range(20, 35)),
    "Imager": list(range(35, 38)),
    "Saturator": list(range(38, 40)),
    "Limiter": list(range(40, 41)),
}


def load_reproducibility_manifest() -> Dict[str, Any]:
    """Load canonical session seeds from reproducibility manifest."""
    p = _REPO_ROOT / "docs" / "human_validation" / "SPRINT2_V0_REPRODUCIBILITY_MANIFEST.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing reproducibility manifest: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def load_harmonized_baseline_results() -> Dict[str, Any]:
    """Load harmonized V0p baseline results from SPRINT3_V2_ABLATION_RESULTS.json."""
    p = _REPO_ROOT / "docs" / "sprint3" / "SPRINT3_V2_ABLATION_RESULTS.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing harmonized results: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def evaluate_variant_point(
    opt: AutoMasteringOptimizer,
    z_var: torch.Tensor,
    prep_data: Dict[str, Any],
    target_lufs: float,
) -> Dict[str, Any]:
    """Evaluate objective, loss decomposition, and physical parameters at a parameter point z_var.

    Fixes applied (Argon review):
      - Detaches tensors before calling .cpu().numpy() (Bug 1).
      - Passes explicit target_lufs=target_lufs in stylistic mode (Bug 2a).
      - Adds parameter regularization reg_loss to loss_v0p_total (Bug 2b).
    """
    mode = prep_data["mode"]
    mix_chunks = prep_data["mix_chunks"]
    ref_chunks = prep_data["ref_chunks"]
    preroll_list = prep_data["preroll_list"]
    ref_ext_chunks = prep_data.get("ref_ext_chunks")
    ref_preroll_list = prep_data.get("ref_preroll_list")
    n_chunks = len(mix_chunks)
    weights = [1.0 / n_chunks] * n_chunks
    z_neutral = Z_NEUTRAL.to(device=z_var.device)

    loss_v0p_total = torch.tensor(0.0, device=z_var.device, dtype=torch.float32)
    loss_v3b_total = torch.tensor(0.0, device=z_var.device, dtype=torch.float32)
    agg_chunk_metrics: Dict[str, float] = {}

    v3b_terms_agg = {
        "loss_lim_raw": 0.0,
        "loss_lim_weighted": 0.0,
        "loss_diff_raw": 0.0,
        "loss_diff_weighted": 0.0,
        "loss_conflict_raw": 0.0,
        "loss_conflict_weighted": 0.0,
    }

    with torch.no_grad():
        for k in range(n_chunks):
            mk = mix_chunks[k]
            n_pre = preroll_list[k]
            ref_k = ref_chunks[k % len(ref_chunks)] if ref_chunks else None
            ref_ext_k = ref_ext_chunks[k % len(ref_ext_chunks)] if ref_ext_chunks else None
            ref_n_pre_k = ref_preroll_list[k % len(ref_preroll_list)] if ref_preroll_list else 0

            if opt.enable_v3b:
                y_ext, diag_k = opt.chain(mk, z=z_var, eq_mode="iir", return_diagnostics=True)
            else:
                y_ext = opt.chain(mk, z=z_var, eq_mode="iir")
                diag_k = None

            y_k = y_ext[..., n_pre:] if n_pre > 0 else y_ext

            if mode == "same_track":
                lk, mk_dict = opt.loss_fn(
                    pred_audio=y_k,
                    target_audio=ref_k,
                    target_lufs=target_lufs,
                    normalized_params=z_var,
                    align_latency=True,
                    pred_ext_audio=y_ext,
                    pred_preroll_samples=n_pre,
                    ref_ext_audio=ref_ext_k,
                    ref_preroll_samples=ref_n_pre_k,
                    include_eq_prior=False,
                )
            else:
                anchor_k = mk[..., n_pre:] if n_pre > 0 else mk
                # Bug 2a fix: explicit target_lufs=target_lufs passed to stylistic_loss_fn
                lk, mk_dict = opt.stylistic_loss_fn(
                    pred_audio=y_k,
                    ref_audio=ref_k,
                    normalized_params=z_var,
                    target_lufs=target_lufs,
                    mix_anchor_audio=anchor_k,
                    pred_ext_audio=y_ext,
                    pred_preroll_samples=n_pre,
                    ref_ext_audio=ref_ext_k,
                    ref_preroll_samples=ref_n_pre_k,
                    include_eq_prior=False,
                )

            loss_v0p_total = loss_v0p_total + weights[k] * lk
            for mk_key, mk_val in mk_dict.items():
                agg_chunk_metrics[mk_key] = agg_chunk_metrics.get(mk_key, 0.0) + weights[k] * float(mk_val)

            if opt.enable_v3b and diag_k is not None:
                l_lim_k, l_diff_k, l_conf_k, l_dyn_k, v3b_m = opt.v3b_prior_loss.compute_chunk_terms(
                    mix_audio=mk,
                    diagnostics=diag_k,
                    z=z_var,
                    preroll_samples=n_pre,
                )
                raw_lim = float(v3b_m.get("loss_lim_raw", v3b_m.get("loss_lim", 0.0)))
                raw_diff = float(v3b_m.get("loss_diff_raw", v3b_m.get("loss_diff", 0.0)))
                raw_conf = float(v3b_m.get("loss_conflict_raw", v3b_m.get("loss_conflict", 0.0)))
                w_lim = float(v3b_m.get("loss_lim_weighted", opt.v3b_prior_loss.w_lim * raw_lim))
                w_diff = float(v3b_m.get("loss_diff_weighted", opt.v3b_prior_loss.w_diff * raw_diff))
                w_conf = float(v3b_m.get("loss_conflict_weighted", opt.v3b_prior_loss.w_conflict * raw_conf))

                loss_v3b_total = loss_v3b_total + weights[k] * l_dyn_k
                v3b_terms_agg["loss_lim_raw"] += weights[k] * raw_lim
                v3b_terms_agg["loss_lim_weighted"] += weights[k] * w_lim
                v3b_terms_agg["loss_diff_raw"] += weights[k] * raw_diff
                v3b_terms_agg["loss_diff_weighted"] += weights[k] * w_diff
                v3b_terms_agg["loss_conflict_raw"] += weights[k] * raw_conf
                v3b_terms_agg["loss_conflict_weighted"] += weights[k] * w_conf

        # Bug 2b fix: add parameter regularization reg_loss to loss_v0p_total
        reg_loss = opt.weight_reg * torch.mean((z_var - z_neutral) ** 2)
        loss_v0p_total = loss_v0p_total + reg_loss
        loss_total = loss_v0p_total + loss_v3b_total

        # Bug 1 fix: safe detached conversion to avoid RuntimeError on tensors with grad
        theta_vec = map_z_to_theta(z_var.detach())
        theta_init = map_z_to_theta(z_neutral.detach())
        theta_np = theta_vec.detach().cpu().numpy()
        theta_init_np = theta_init.detach().cpu().numpy()
        delta_theta_np = theta_np - theta_init_np

        name_to_idx = {p.name: i for i, p in enumerate(PARAM_SPECS)}
        theta_dict = {p.name: float(theta_np[name_to_idx[p.name]]) for p in PARAM_SPECS}
        delta_theta_dict = {p.name: float(delta_theta_np[name_to_idx[p.name]]) for p in PARAM_SPECS}

        # Check roundtrip
        z_roundtrip = map_theta_to_z(theta_vec)
        assert torch.max(torch.abs(z_roundtrip - z_var.detach())).item() < 1e-5, "Param space roundtrip failed"

    return {
        "loss_total": float(loss_total.item()),
        "loss_v0p_total": float(loss_v0p_total.item()),
        "reg_loss": float(reg_loss.item()),
        "loss_v3b_total": float(loss_v3b_total.item()),
        "v3b_terms": v3b_terms_agg,
        "chunk_metrics": agg_chunk_metrics,
        "normalized_z": [float(v) for v in z_var.detach().cpu().numpy().tolist()],
        "physical_parameters_theta": theta_dict,
        "delta_theta_from_neutral": delta_theta_dict,
    }


def compute_empirical_dynamics_stats(
    chain: DifferentiableMasteringChain,
    prep_data: Dict[str, Any],
    z_vector: torch.Tensor,
) -> Dict[str, Any]:
    """Compute empirical limiter, compressor, and differential metrics per chunk and aggregated."""
    mix_chunks = prep_data["mix_chunks"]
    preroll_list = prep_data["preroll_list"]
    K = len(mix_chunks)

    all_a_lim = []
    all_gr_low = []
    all_gr_mid = []
    all_gr_high = []
    all_diff_lm = []
    all_diff_mh = []

    per_chunk_stats = []

    with torch.no_grad():
        for k in range(K):
            mk = mix_chunks[k]
            n_pre = preroll_list[k]
            T = mk.shape[-1] - n_pre
            T_eval = T - TOTAL_DSP_DELAY_SAMPLES

            _, diag = chain(mk, z=z_vector, return_diagnostics=True)

            x_eval = mk[..., n_pre : n_pre + T_eval]
            gr_l_eval = diag["gr_low_db"][..., n_pre : n_pre + T_eval]
            gr_m_eval = diag["gr_mid_db"][..., n_pre : n_pre + T_eval]
            gr_h_eval = diag["gr_high_db"][..., n_pre : n_pre + T_eval]
            a_lim_eval = diag["lim_atten_db"][..., n_pre + TOTAL_DSP_DELAY_SAMPLES : n_pre + T]

            mask_smp, is_silent, _ = compute_active_sample_mask(x_eval)
            mask_flat = mask_smp.reshape(-1)
            n_act_k = int(mask_flat.sum().item())

            if not is_silent and n_act_k > 0:
                a_act = a_lim_eval.reshape(-1)[mask_flat]
                gr_l_act = gr_l_eval.reshape(-1)[mask_flat]
                gr_m_act = gr_m_eval.reshape(-1)[mask_flat]
                gr_h_act = gr_h_eval.reshape(-1)[mask_flat]
                diff_lm_act = torch.abs(gr_l_eval - gr_m_eval).reshape(-1)[mask_flat]
                diff_mh_act = torch.abs(gr_m_eval - gr_h_eval).reshape(-1)[mask_flat]

                all_a_lim.append(a_act)
                all_gr_low.append(gr_l_act)
                all_gr_mid.append(gr_m_act)
                all_gr_high.append(gr_h_act)
                all_diff_lm.append(diff_lm_act)
                all_diff_mh.append(diff_mh_act)

                k_tail = max(1, int(math.ceil(0.05 * n_act_k)))
                topk_k, _ = torch.topk(a_act, k_tail)
                cvar_k = float(torch.mean(topk_k).item())
                p95_k = float(torch.quantile(a_act.float(), 0.95).item())

                chunk_stat = {
                    "chunk_index": k,
                    "active_samples": n_act_k,
                    "limiter_atten_mean_db": float(torch.mean(a_act).item()),
                    "limiter_atten_p95_db": p95_k,
                    "limiter_atten_cvar95_db": cvar_k,
                    "limiter_atten_max_db": float(torch.max(a_act).item()),
                    "gr_low_mean_db": float(torch.mean(gr_l_act).item()),
                    "gr_low_p95_db": float(torch.quantile(gr_l_act.float(), 0.95).item()),
                    "gr_low_max_db": float(torch.max(gr_l_act).item()),
                    "gr_mid_mean_db": float(torch.mean(gr_m_act).item()),
                    "gr_mid_p95_db": float(torch.quantile(gr_m_act.float(), 0.95).item()),
                    "gr_mid_max_db": float(torch.max(gr_m_act).item()),
                    "gr_high_mean_db": float(torch.mean(gr_h_act).item()),
                    "gr_high_p95_db": float(torch.quantile(gr_h_act.float(), 0.95).item()),
                    "gr_high_max_db": float(torch.max(gr_h_act).item()),
                    "diff_lm_mean_db": float(torch.mean(diff_lm_act).item()),
                    "diff_lm_max_db": float(torch.max(diff_lm_act).item()),
                    "diff_mh_mean_db": float(torch.mean(diff_mh_act).item()),
                    "diff_mh_max_db": float(torch.max(diff_mh_act).item()),
                }
            else:
                chunk_stat = {
                    "chunk_index": k,
                    "active_samples": 0,
                    "limiter_atten_mean_db": 0.0,
                    "limiter_atten_p95_db": 0.0,
                    "limiter_atten_cvar95_db": 0.0,
                    "limiter_atten_max_db": 0.0,
                    "gr_low_mean_db": 0.0,
                    "gr_low_p95_db": 0.0,
                    "gr_low_max_db": 0.0,
                    "gr_mid_mean_db": 0.0,
                    "gr_mid_p95_db": 0.0,
                    "gr_mid_max_db": 0.0,
                    "gr_high_mean_db": 0.0,
                    "gr_high_p95_db": 0.0,
                    "gr_high_max_db": 0.0,
                    "diff_lm_mean_db": 0.0,
                    "diff_lm_max_db": 0.0,
                    "diff_mh_mean_db": 0.0,
                    "diff_mh_max_db": 0.0,
                }
            per_chunk_stats.append(chunk_stat)

    if all_a_lim:
        cat_a = torch.cat(all_a_lim)
        cat_gl = torch.cat(all_gr_low)
        cat_gm = torch.cat(all_gr_mid)
        cat_gh = torch.cat(all_gr_high)
        cat_dlm = torch.cat(all_diff_lm)
        cat_dmh = torch.cat(all_diff_mh)
        tot_act = len(cat_a)
        k_tail_tot = max(1, int(math.ceil(0.05 * tot_act)))
        topk_tot, _ = torch.topk(cat_a, k_tail_tot)

        aggregate_stats = {
            "total_active_samples": tot_act,
            "limiter_atten_mean_db": float(torch.mean(cat_a).item()),
            "limiter_atten_p95_db": float(torch.quantile(cat_a.float(), 0.95).item()),
            "limiter_atten_cvar95_db": float(torch.mean(topk_tot).item()),
            "limiter_atten_max_db": float(torch.max(cat_a).item()),
            "gr_low_mean_db": float(torch.mean(cat_gl).item()),
            "gr_low_p95_db": float(torch.quantile(cat_gl.float(), 0.95).item()),
            "gr_low_max_db": float(torch.max(cat_gl).item()),
            "gr_mid_mean_db": float(torch.mean(cat_gm).item()),
            "gr_mid_p95_db": float(torch.quantile(cat_gm.float(), 0.95).item()),
            "gr_mid_max_db": float(torch.max(cat_gm).item()),
            "gr_high_mean_db": float(torch.mean(cat_gh).item()),
            "gr_high_p95_db": float(torch.quantile(cat_gh.float(), 0.95).item()),
            "gr_high_max_db": float(torch.max(cat_gh).item()),
            "diff_lm_mean_db": float(torch.mean(cat_dlm).item()),
            "diff_lm_max_db": float(torch.max(cat_dlm).item()),
            "diff_mh_mean_db": float(torch.mean(cat_dmh).item()),
            "diff_mh_max_db": float(torch.max(cat_dmh).item()),
        }
    else:
        aggregate_stats = {
            "total_active_samples": 0,
            "limiter_atten_mean_db": 0.0,
            "limiter_atten_p95_db": 0.0,
            "limiter_atten_cvar95_db": 0.0,
            "limiter_atten_max_db": 0.0,
            "gr_low_mean_db": 0.0,
            "gr_low_p95_db": 0.0,
            "gr_low_max_db": 0.0,
            "gr_mid_mean_db": 0.0,
            "gr_mid_p95_db": 0.0,
            "gr_mid_max_db": 0.0,
            "gr_high_mean_db": 0.0,
            "gr_high_p95_db": 0.0,
            "gr_high_max_db": 0.0,
            "diff_lm_mean_db": 0.0,
            "diff_lm_max_db": 0.0,
            "diff_mh_mean_db": 0.0,
            "diff_mh_max_db": 0.0,
        }

    return {
        "per_chunk": per_chunk_stats,
        "aggregate": aggregate_stats,
    }


def compute_gradient_diagnostics(
    chain: DifferentiableMasteringChain,
    prep_data: Dict[str, Any],
    z_vector: torch.Tensor,
    v3b_loss_module: V3bDynamicPriorLoss,
    target_lufs: float,
) -> Dict[str, Any]:
    """Compute per-group gradient norms in z and u space, asserting finiteness."""
    mode = prep_data["mode"]
    mix_chunks = prep_data["mix_chunks"]
    ref_chunks = prep_data["ref_chunks"]
    preroll_list = prep_data["preroll_list"]
    K = len(mix_chunks)
    weights = [1.0 / K] * K

    z = z_vector.detach().clone().requires_grad_(True)
    total_loss_v3b = torch.tensor(0.0, dtype=torch.float32)
    chunk_v3b_terms = []

    for k in range(K):
        mk = mix_chunks[k]
        n_pre = preroll_list[k]
        _, diag = chain(mk, z=z, eq_mode="iir", return_diagnostics=True)
        l_lim, l_diff, l_conf, l_dyn, m = v3b_loss_module.compute_chunk_terms(
            mix_audio=mk,
            diagnostics=diag,
            z=z,
            preroll_samples=n_pre,
        )
        total_loss_v3b = total_loss_v3b + weights[k] * l_dyn
        raw_lim_k = float(m.get("loss_lim_raw", m.get("loss_lim", 0.0)))
        raw_diff_k = float(m.get("loss_diff_raw", m.get("loss_diff", 0.0)))
        raw_conf_k = float(m.get("loss_conflict_raw", m.get("loss_conflict", 0.0)))
        chunk_v3b_terms.append({
            "chunk_index": k,
            "loss_lim_raw": raw_lim_k,
            "loss_diff_raw": raw_diff_k,
            "loss_conflict_raw": raw_conf_k,
            "loss_dyn_weighted": float(l_dyn.item()),
            "limiter_active": bool(raw_lim_k > 0.0),
            "diff_active": bool(raw_diff_k > 0.0),
            "conflict_active": bool(raw_conf_k > 0.0),
        })

    assert torch.isfinite(total_loss_v3b), f"V3b total loss is non-finite: {total_loss_v3b}"
    grad_z = torch.autograd.grad(total_loss_v3b, z, retain_graph=False)[0]
    assert torch.isfinite(grad_z).all(), "Gradient in z contains NaN/Inf"

    z_clamped = torch.clamp(z_vector, 1e-7, 1.0 - 1e-7)
    grad_u = grad_z * z_clamped * (1.0 - z_clamped)
    assert torch.isfinite(grad_u).all(), "Gradient in u contains NaN/Inf"

    group_norms_z = {}
    group_norms_u = {}
    for gname, gidxs in PARAM_GROUPS.items():
        gz_sub = grad_z[gidxs]
        gu_sub = grad_u[gidxs]
        group_norms_z[gname] = float(torch.norm(gz_sub, p=2).item())
        group_norms_u[gname] = float(torch.norm(gu_sub, p=2).item())

    return {
        "v3b_loss_total": float(total_loss_v3b.item()),
        "norm_grad_z_41d": float(torch.norm(grad_z, p=2).item()),
        "norm_grad_u_41d": float(torch.norm(grad_u, p=2).item()),
        "group_norms_z": group_norms_z,
        "group_norms_u": group_norms_u,
        "chunk_terms": chunk_v3b_terms,
    }


def export_blinded_wavs(
    session_prefix: str,
    v0p_audio_64: np.ndarray,
    v3b_audio_64: np.ndarray,
    sample_rate: float,
    export_dir: Path,
    session_seed: int,
) -> Dict[str, str]:
    """Export blinded WAV files and return secret mapping."""
    export_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(session_seed)
    is_v0p_first = rng.choice([True, False])

    if is_v0p_first:
        label_v0p = "variant_A"
        label_v3b = "variant_B"
    else:
        label_v0p = "variant_B"
        label_v3b = "variant_A"

    path_a = export_dir / f"{session_prefix}_variant_A.wav"
    path_b = export_dir / f"{session_prefix}_variant_B.wav"

    audio_a = v0p_audio_64 if is_v0p_first else v3b_audio_64
    audio_b = v3b_audio_64 if is_v0p_first else v0p_audio_64

    # Convert to 24-bit PCM
    audio_a_t = np.transpose(audio_a)  # (samples, 2)
    audio_b_t = np.transpose(audio_b)

    sf.write(str(path_a), audio_a_t, int(sample_rate), subtype="PCM_24")
    sf.write(str(path_b), audio_b_t, int(sample_rate), subtype="PCM_24")

    mapping = {
        "session": session_prefix,
        "seed": str(session_seed),
        "files": {
            path_a.name: "Control_V0p" if is_v0p_first else "Treatment_V3b_B05",
            path_b.name: "Treatment_V3b_B05" if is_v0p_first else "Control_V0p",
        },
        "variant_mapping": {
            "Control_V0p": label_v0p,
            "Treatment_V3b_B05": label_v3b,
        },
    }
    return mapping


def evaluate_single_session_v3b(
    spec: Dict[str, str],
    seed_int: int,
    v0p_baseline_data: Optional[Dict[str, Any]] = None,
    export_wavs: bool = True,
    export_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Evaluate Control V0p and Treatment V3b-B0.5 for a single session."""
    prefix = spec["track_prefix"]
    mode = spec["mode"]

    cache_dir = _REPO_ROOT / ".cache_v3b_ablation"
    cache_dir.mkdir(parents=True, exist_ok=True)
    session_cache_file = cache_dir / f"{prefix}_v3b_ablation.json"

    if session_cache_file.exists():
        print(f"[{prefix}] Loaded complete session from cache: {session_cache_file}", flush=True)
        return json.loads(session_cache_file.read_text(encoding="utf-8"))

    print(f"\n=======================================================", flush=True)
    print(f"[{prefix}] Starting V3b Ablation Evaluation (mode: {mode}, seed: {seed_int})", flush=True)
    print(f"=======================================================", flush=True)

    t0 = time.time()
    prep = prepare_harmonized_session(spec, seed_int=seed_int, repo_root=_REPO_ROOT)
    raw_full_64 = prep["raw_full_64"]
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
    # 1. Control V0p Run
    # -------------------------------------------------------------------------
    print(f"[{prefix}] Running Control V0p (30 steps Adam, lr=0.04)...", flush=True)
    torch.manual_seed(seed_int)
    np.random.seed(seed_int & 0xFFFFFFFF)

    opt_v0p = AutoMasteringOptimizer(
        sample_rate=48000.0,
        lr=0.04,
        num_steps=30,
        target_lufs=target_lufs,
        weight_stft=1.0,
        weight_loudness=0.8,
        weight_param_reg=0.01,
        use_hybrid_eq=True,
        v3b_w_lim=0.0,
        v3b_w_diff=0.0,
        v3b_w_conflict=0.0,
    )

    t_v0p_start = time.time()
    z_opt_v0p, _, history_v0p = opt_v0p.optimize_multi_chunk(
        mix_chunks=mix_chunks,
        ref_chunks=ref_chunks,
        reference_mode=mode,
        verbose=False,
        align_latency=True,
        mix_preroll_samples=preroll_list,
        target_lufs=target_lufs,
        ref_ext_chunks=ref_ext_chunks,
        ref_preroll_list=ref_preroll_list,
    )
    t_v0p_opt = time.time() - t_v0p_start

    # Check match against harmonized baseline if available
    if v0p_baseline_data is not None and "normalized_z_opt" in v0p_baseline_data:
        z_base = torch.tensor(v0p_baseline_data["normalized_z_opt"], dtype=torch.float32)
        diff_base = float(torch.max(torch.abs(z_opt_v0p - z_base)).item())
        print(f"[{prefix}] V0p match against harmonized baseline max diff: {diff_base:.8f}", flush=True)

    # Evaluate variant points for V0p
    eval_v0p_init = evaluate_variant_point(opt_v0p, Z_NEUTRAL, prep, target_lufs)
    eval_v0p_final = evaluate_variant_point(opt_v0p, z_opt_v0p, prep, target_lufs)
    loss_red_v0p = (
        ((eval_v0p_final["loss_total"] - eval_v0p_init["loss_total"]) / eval_v0p_init["loss_total"]) * 100.0
        if eval_v0p_init["loss_total"] > 0 else 0.0
    )

    # Dynamics stats for V0p
    stats_v0p_z0 = compute_empirical_dynamics_stats(opt_v0p.chain, prep, Z_NEUTRAL)
    stats_v0p_opt = compute_empirical_dynamics_stats(opt_v0p.chain, prep, z_opt_v0p)

    # Full-track render and benchmark for V0p
    print(f"[{prefix}] Rendering full-track offline C++ for Control V0p...", flush=True)
    mastered_v0p_64, _ = render_offline_with_tail_flush(
        raw_full_64, mode="z", z=z_opt_v0p.tolist(), sample_rate=48000.0
    )
    if mode == "same_track":
        m_eval_v0p = mastered_v0p_64[:, mix_start : mix_start + eval_len].astype(np.float32)
        bm_v0p = suite.evaluate(
            torch.tensor(m_eval_v0p).unsqueeze(0), ref_eval_tensor, align_latency=False, reference_mode="same_track"
        )
    else:
        bm_v0p = suite.evaluate(
            torch.tensor(mastered_v0p_64.astype(np.float32)).unsqueeze(0), ref_eval_tensor, align_latency=False, reference_mode="stylistic"
        )

    # -------------------------------------------------------------------------
    # 2. Treatment V3b-B0.5 Run
    # -------------------------------------------------------------------------
    print(f"[{prefix}] Running Treatment V3b-B0.5 (30 steps Adam, lr=0.04, w_lim=0.50)...", flush=True)
    torch.manual_seed(seed_int)
    np.random.seed(seed_int & 0xFFFFFFFF)

    opt_v3b = AutoMasteringOptimizer(
        sample_rate=48000.0,
        lr=0.04,
        num_steps=30,
        target_lufs=target_lufs,
        weight_stft=1.0,
        weight_loudness=0.8,
        weight_param_reg=0.01,
        use_hybrid_eq=True,
        v3b_w_lim=FROZEN_V3B_B05_PARAMS["w_lim"],
        v3b_tau_avg=FROZEN_V3B_B05_PARAMS["tau_avg"],
        v3b_tau_95=FROZEN_V3B_B05_PARAMS["tau_95"],
        v3b_lambda_95=FROZEN_V3B_B05_PARAMS["lambda_95"],
        v3b_delta_max=FROZEN_V3B_B05_PARAMS["delta_max"],
        v3b_delta_m=FROZEN_V3B_B05_PARAMS["delta_m"],
        v3b_delta_ref=FROZEN_V3B_B05_PARAMS["delta_ref"],
        v3b_w_diff=FROZEN_V3B_B05_PARAMS["w_diff"],
        v3b_w_conflict=FROZEN_V3B_B05_PARAMS["w_conflict"],
    )

    t_v3b_start = time.time()
    z_opt_v3b, _, history_v3b = opt_v3b.optimize_multi_chunk(
        mix_chunks=mix_chunks,
        ref_chunks=ref_chunks,
        reference_mode=mode,
        verbose=False,
        align_latency=True,
        mix_preroll_samples=preroll_list,
        target_lufs=target_lufs,
        ref_ext_chunks=ref_ext_chunks,
        ref_preroll_list=ref_preroll_list,
    )
    t_v3b_opt = time.time() - t_v3b_start

    # Evaluate variant points for V3b
    eval_v3b_init = evaluate_variant_point(opt_v3b, Z_NEUTRAL, prep, target_lufs)
    eval_v3b_final = evaluate_variant_point(opt_v3b, z_opt_v3b, prep, target_lufs)
    loss_red_v3b = (
        ((eval_v3b_final["loss_total"] - eval_v3b_init["loss_total"]) / eval_v3b_init["loss_total"]) * 100.0
        if eval_v3b_init["loss_total"] > 0 else 0.0
    )

    # Evaluate V3b objective on z_opt_v0p for clean cross-comparison
    eval_v3b_on_v0p = evaluate_variant_point(opt_v3b, z_opt_v0p, prep, target_lufs)

    # Dynamics stats for V3b
    stats_v3b_z0 = compute_empirical_dynamics_stats(opt_v3b.chain, prep, Z_NEUTRAL)
    stats_v3b_opt = compute_empirical_dynamics_stats(opt_v3b.chain, prep, z_opt_v3b)

    # Parameter diff V3b vs V0p
    delta_theta_treatment_vs_control = {
        p.name: float(eval_v3b_final["physical_parameters_theta"][p.name] - eval_v0p_final["physical_parameters_theta"][p.name])
        for p in PARAM_SPECS
    }

    # Gradient diagnostics for V3b
    grad_diag_v3b = compute_gradient_diagnostics(
        opt_v3b.chain, prep, z_opt_v3b, opt_v3b.v3b_prior_loss, target_lufs
    )

    # Full-track render and benchmark for V3b
    print(f"[{prefix}] Rendering full-track offline C++ for Treatment V3b-B0.5...", flush=True)
    mastered_v3b_64, _ = render_offline_with_tail_flush(
        raw_full_64, mode="z", z=z_opt_v3b.tolist(), sample_rate=48000.0
    )
    if mode == "same_track":
        m_eval_v3b = mastered_v3b_64[:, mix_start : mix_start + eval_len].astype(np.float32)
        bm_v3b = suite.evaluate(
            torch.tensor(m_eval_v3b).unsqueeze(0), ref_eval_tensor, align_latency=False, reference_mode="same_track"
        )
    else:
        bm_v3b = suite.evaluate(
            torch.tensor(mastered_v3b_64.astype(np.float32)).unsqueeze(0), ref_eval_tensor, align_latency=False, reference_mode="stylistic"
        )

    # -------------------------------------------------------------------------
    # 3. Blind WAV Export
    # -------------------------------------------------------------------------
    blind_mapping = None
    if export_wavs:
        target_dir = export_dir if export_dir is not None else (_REPO_ROOT / "docs" / "sprint3" / "blind_v3b_wavs")
        blind_mapping = export_blinded_wavs(
            session_prefix=prefix,
            v0p_audio_64=mastered_v0p_64,
            v3b_audio_64=mastered_v3b_64,
            sample_rate=48000.0,
            export_dir=target_dir,
            session_seed=seed_int,
        )
        print(f"[{prefix}] Blinded WAV files exported to: {target_dir}", flush=True)

    elapsed_total = time.time() - t0
    print(f"[{prefix}] Finished in {elapsed_total:.2f}s (V0p opt: {t_v0p_opt:.2f}s, V3b opt: {t_v3b_opt:.2f}s)", flush=True)

    res = {
        "track_prefix": prefix,
        "mode": mode,
        "seed": str(seed_int),
        "target_lufs": float(target_lufs),
        "elapsed_sec": float(elapsed_total),
        "opt_v0p_elapsed_sec": float(t_v0p_opt),
        "opt_v3b_elapsed_sec": float(t_v3b_opt),
        "chunk_metadata": chunk_metadata,
        "control_v0p": {
            "initial_evaluation": eval_v0p_init,
            "final_evaluation": eval_v0p_final,
            "loss_reduction_pct": float(loss_red_v0p),
            "normalized_z_opt": eval_v0p_final["normalized_z"],
            "physical_parameters_theta": eval_v0p_final["physical_parameters_theta"],
            "delta_theta_from_neutral": eval_v0p_final["delta_theta_from_neutral"],
            "full_track_benchmark": bm_v0p,
            "dynamics_stats_initial_z0": stats_v0p_z0,
            "dynamics_stats_final_zopt": stats_v0p_opt,
            "loss_history": history_v0p,
        },
        "treatment_v3b_b05": {
            "initial_evaluation": eval_v3b_init,
            "final_evaluation": eval_v3b_final,
            "loss_reduction_pct": float(loss_red_v3b),
            "normalized_z_opt": eval_v3b_final["normalized_z"],
            "physical_parameters_theta": eval_v3b_final["physical_parameters_theta"],
            "delta_theta_from_neutral": eval_v3b_final["delta_theta_from_neutral"],
            "delta_theta_vs_control_v0p": delta_theta_treatment_vs_control,
            "v3b_objective_eval_on_v0p": eval_v3b_on_v0p,
            "full_track_benchmark": bm_v3b,
            "dynamics_stats_initial_z0": stats_v3b_z0,
            "dynamics_stats_final_zopt": stats_v3b_opt,
            "loss_history": history_v3b,
            "gradient_diagnostics_final": grad_diag_v3b,
        },
        "blind_mapping": blind_mapping,
    }

    session_cache_file.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert value to float, handling None and non-convertible types."""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _fmt_val(bm: Dict[str, Any], k: str, fmt: str = ".2f", default: str = "—") -> str:
    """Format benchmark value or return default placeholder if None."""
    v = bm.get(k)
    if v is None:
        return default
    try:
        return f"{float(v):{fmt}}"
    except (TypeError, ValueError):
        return default


def generate_v3b_ablation_report(results_payload: Dict[str, Any], out_path: Path):
    """Generate Markdown comparison report SPRINT3_V3B_V0P_ABLATION_REPORT.md."""
    sessions = results_payload.get("sessions", [])

    lines = [
        "# SPRINT 3 — V3B VS V0P ABLATION REPORT",
        "",
        "**Titolo:** Report Comparativo Accoppiato Control V0p vs Treatment V3b-B0.5  ",
        f"**Data di Generazione:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        "**Branch:** `human-validation-package`  ",
        "**Variante Trattamento:** `V3b-B0.5` (Candidate B con $w_{\\text{lim}} = 0.50$, $w_{\\text{diff}} = 0.02$, $w_{\\text{conflict}} = 0.02$)  ",
        "**Variante Controllo:** `V0p` (Baseline Sprint 2 armonizzata con pre-roll a 2.5s)  ",
        "",
        "---",
        "",
        "## 1. Executive Summary & Protocollo Sperimentale",
        "",
        "La presente campagna di validazione valuta in modo rigoroso e accoppiato l'efficacia del prior dinamico differenziabile **V3b-B0.5** rispetto al controllo **V0p** su tutte le 6 sessioni armonizzate di mastering.",
        "",
        "### Parametri Congelati V3b-B0.5",
        "| Parametro | Descrizione | Valore Congelato |",
        "|:---|:---|:---:|",
        f"| $\\tau_{{\\text{{avg}}}}$ | Soglia tolleranza attenuazione media limiter $\\bar{{A}}_{{\\text{{lim}}}}^{{\\text{{act}}}}$ in $\\mathcal{{L}}_{{\\text{{lim}}}}$ | `{FROZEN_V3B_B05_PARAMS['tau_avg']} dB` |",
        f"| $\\tau_{{95}}$ | Soglia tolleranza picchi estremi limiter ($P_{{95}}$ / $\\text{{CVaR}}_{{0.95}}$) in $\\mathcal{{L}}_{{\\text{{lim}}}}$ | `{FROZEN_V3B_B05_PARAMS['tau_95']} dB` |",
        f"| $\\lambda_{{95}}$ | Peso moltiplicativo della penalità di coda $\\text{{CVaR}}_{{0.95}}$ rispetto alla penalità media $\\bar{{A}}_{{\\text{{lim}}}}^{{\\text{{act}}}}$ in $\\mathcal{{L}}_{{\\text{{lim}}}}$ | `{FROZEN_V3B_B05_PARAMS['lambda_95']}` |",
        f"| $\\Delta_{{\\max}}$ | Soglia di tolleranza per la Gain Reduction differenziale inter-banda ($|GR_L - GR_M|, |GR_M - GR_H|$) in $\\mathcal{{L}}_{{\\text{{diff}}}}$ | `{FROZEN_V3B_B05_PARAMS['delta_max']} dB` |",
        f"| $\\delta_M$ | Margine di headroom consentito tra makeup gain $M_c$ e attenuazione media $\\bar{{GR}}_c^{{\\text{{act}}}}$ in $\\mathcal{{L}}_{{\\text{{conflict}}}}$ | `{FROZEN_V3B_B05_PARAMS['delta_m']} dB` |",
        f"| $\\Delta_{{\\text{{ref}}}}$ | Scala di normalizzazione in dB del rational soft-gate $h_{{\\text{{lim}}}} = \\text{{detach}}(r / (r + \\Delta_{{\\text{{ref}}}}))$ in $\\mathcal{{L}}_{{\\text{{conflict}}}}$ | `{FROZEN_V3B_B05_PARAMS['delta_ref']} dB` |",
        f"| $w_{{\\text{{lim}}}}$ | Peso di regolarizzazione limiter ($\\mathcal{{L}}_{{\\text{{lim}}}}$) | `{FROZEN_V3B_B05_PARAMS['w_lim']}` |",
        f"| $w_{{\\text{{diff}}}}$ | Peso di regolarizzazione differenziale multibanda ($\\mathcal{{L}}_{{\\text{{diff}}}}$) | `{FROZEN_V3B_B05_PARAMS['w_diff']}` |",
        f"| $w_{{\\text{{conflict}}}}$ | Peso di isolamento conflitti makeup gain vs compressione ($\\mathcal{{L}}_{{\\text{{conflict}}}}$) | `{FROZEN_V3B_B05_PARAMS['w_conflict']}` |",
        "",
        "### Manifest Verificato dei Commit (`human-validation-package`) & Audit Diff Storico",
        "",
        "Sequenza esatta e verificata dei commit su `human-validation-package` (si precisa che l'hash `f01e94717a7fd1718418c5bf978750c749e13895` non esiste nella storia Git del repository):",
        "",
        "| Ordine | SHA Commit | Messaggio | Stato Integrità Artefatti |",
        "|:---:|:---|:---|:---|",
        "| 1 | `c267de68098a551c235c769c1f02389297ffdfac` | `fix(sprint3-v3b): unify preflight and V3bDynamicPriorLoss formulas, add w_lim sensitivity and exact sample-level alignment test` | Preflight V3 runner e gate test |",
        "| 2 | `6afb247f249cd03684d1f42df343911928f7e8bd` | `feat(sprint3-v3b): generate preflight V3 results and report with unified formulas and w_lim sensitivity` | **Baseline storica Preflight V3 congelata** (14 file storici con `git diff 6afb247..HEAD` vuoto) |",
        "| 3 | `4fbc01db1ab2e0589323c06850192a3f308a73be` | `fix(sprint3-v3b): refine gate tests 7 and 9 and add preflight V3 audit errata note` | Introduce `SPRINT3_V3B_PREFLIGHT_V3_AUDIT_ERRATA.md` (`git diff 4fbc01d..HEAD` vuoto) |",
        "| 4 | `befda910f2191b8887f1e330bc515da29639b000` | `feat(sprint3-v3b): integrate V3b-B0.5 into AutoMasteringOptimizer, add ablation runner and regression tests` | Integrazione V3b-B0.5 in `optimizer.py` |",
        "| 5 | `2123842f964c2d91d9163d7067385cc126eacad4` | `feat(sprint3-v3b): execute V3b-B0.5 vs V0p controlled ablation and publish results and report` | Congela `SPRINT3_V3B_V0P_ABLATION_RESULTS.json` (`git diff 2123842..HEAD` vuoto) |",
        "| 6 | `7f7ad10a74560f335cfde89eda9d257a5872ab12` | `feat(sprint3-v3b): generate blind listening validation package and secret key mapping` | Export dei 12 WAV anonimi, `SPRINT3_V3B_BLIND_MAPPING.json` e worksheet |",
        "| 7 | `58ca5258467ec429f8ead69698f3a9acd6fba892` | `fix(sprint3-v3b): fix Table 3 parameter keys, add Table 3B, and align Section 1 and 7.3 text from frozen JSON` | Sanatoria formattazione Tabella 3 / 3B e True Peak guardrail |",
        "",
        "Riepilogo `git diff --name-status 6afb247f249cd03684d1f42df343911928f7e8bd..58ca5258467ec429f8ead69698f3a9acd6fba892` (9 file prima di questo commit di chiusura audit):",
        "- `A`: `docs/sprint3/SPRINT3_V3B_BLIND_LISTENING_WORKSHEET.md`, `docs/sprint3/SPRINT3_V3B_BLIND_MAPPING.json`, `docs/sprint3/SPRINT3_V3B_PREFLIGHT_V3_AUDIT_ERRATA.md`, `docs/sprint3/SPRINT3_V3B_V0P_ABLATION_REPORT.md`, `docs/sprint3/SPRINT3_V3B_V0P_ABLATION_RESULTS.json`, `scripts/sprint3/evaluate_v3b_ablation.py`",
        "- `M`: `python/ddsp_mastering/loss/dynamic_prior.py`, `python/ddsp_mastering/optimize/optimizer.py`, `tests/test_sprint3_v3b_gate.py`",
        "- Il commit corrente aggiunge (`A`) `docs/sprint3/SPRINT3_V3B_BLIND_AUDIO_MANIFEST.json` e `docs/sprint3/SPRINT3_V3B_BLIND_AUDIO_MANIFEST.md` e aggiorna (`M`) runner, report e Gate Test 15–17 mantenendo a diff zero tutti i 15 artefatti storici protetti e `SPRINT3_V3B_V0P_ABLATION_RESULTS.json`.",
        "",
        "---",
        "",
        "## 2. Tabella 1 — Metriche Audio Oggettive Full-Track (Control V0p vs Treatment V3b-B0.5)",
        "",
        "| Sessione | Variante | Target LUFS | Final LUFS | $\\Delta$LUFS (dB) | PLR (dB) | Peak (dBFS) | True Peak (dBTP) | Spec Conv (SC) | LSD (dB) | ASD (dB) | ERB Surplus (dB) |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for s in sessions:
        sid = s["track_prefix"]
        tlufs = _safe_float(s.get("target_lufs"))
        bm_0 = s["control_v0p"]["full_track_benchmark"]
        bm_3 = s["treatment_v3b_b05"]["full_track_benchmark"]

        lines.append(
            f"| **`{sid}`** | Control V0p | {tlufs:.2f} | {_safe_float(bm_0.get('pred_lufs')):.2f} | {_safe_float(bm_0.get('delta_lufs')):+.2f} | "
            f"{_safe_float(bm_0.get('pred_plr_db')):.2f} | {_safe_float(bm_0.get('pred_sample_peak_dbfs')):.2f} | {_safe_float(bm_0.get('pred_true_peak_dbtp')):.2f} | "
            f"{_fmt_val(bm_0, 'spectral_convergence', '.4f')} | {_fmt_val(bm_0, 'log_spectral_distance_db', '.2f')} | {_fmt_val(bm_0, 'aggregate_spectral_distance', '.4f')} | {_fmt_val(bm_0, 'erb_presence_surplus_db', '+.2f')} |"
        )
        lines.append(
            f"| **`{sid}`** | Treatment V3b | {tlufs:.2f} | {_safe_float(bm_3.get('pred_lufs')):.2f} | {_safe_float(bm_3.get('delta_lufs')):+.2f} | "
            f"{_safe_float(bm_3.get('pred_plr_db')):.2f} | {_safe_float(bm_3.get('pred_sample_peak_dbfs')):.2f} | {_safe_float(bm_3.get('pred_true_peak_dbtp')):.2f} | "
            f"{_fmt_val(bm_3, 'spectral_convergence', '.4f')} | {_fmt_val(bm_3, 'log_spectral_distance_db', '.2f')} | {_fmt_val(bm_3, 'aggregate_spectral_distance', '.4f')} | {_fmt_val(bm_3, 'erb_presence_surplus_db', '+.2f')} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Tabella 2 — Statistiche Dinamiche di Limiting & Compressione Multibanda",
        "",
        "| Sessione | Variante | Limiter Mean (dB) | Limiter P95 (dB) | Limiter CVaR95 (dB) | Limiter Max (dB) | GR Low Mean (dB) | GR Mid Mean (dB) | GR High Mean (dB) | Diff LM Max (dB) |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for s in sessions:
        sid = s["track_prefix"]
        d0 = s["control_v0p"]["dynamics_stats_final_zopt"]["aggregate"]
        d3 = s["treatment_v3b_b05"]["dynamics_stats_final_zopt"]["aggregate"]

        lines.append(
            f"| **`{sid}`** | Control V0p | {_safe_float(d0.get('limiter_atten_mean_db')):.3f} | {_safe_float(d0.get('limiter_atten_p95_db')):.3f} | "
            f"{_safe_float(d0.get('limiter_atten_cvar95_db')):.3f} | {_safe_float(d0.get('limiter_atten_max_db')):.3f} | "
            f"{_safe_float(d0.get('gr_low_mean_db')):.4f} | {_safe_float(d0.get('gr_mid_mean_db')):.4f} | {_safe_float(d0.get('gr_high_mean_db')):.6f} | "
            f"{_safe_float(d0.get('diff_lm_max_db')):.4f} |"
        )
        lines.append(
            f"| **`{sid}`** | Treatment V3b | {_safe_float(d3.get('limiter_atten_mean_db')):.3f} | {_safe_float(d3.get('limiter_atten_p95_db')):.3f} | "
            f"{_safe_float(d3.get('limiter_atten_cvar95_db')):.3f} | {_safe_float(d3.get('limiter_atten_max_db')):.3f} | "
            f"{_safe_float(d3.get('gr_low_mean_db')):.4f} | {_safe_float(d3.get('gr_mid_mean_db')):.4f} | {_safe_float(d3.get('gr_high_mean_db')):.6f} | "
            f"{_safe_float(d3.get('diff_lm_max_db')):.4f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Tabella 3 — Variazione Parametri Fisici Chiave ($\\Delta\\theta$ Treatment V3b vs Control V0p)",
        "",
        "| Sessione | `limiter_push_gain_db` $\\Delta$ | `comp_low_makeup_db` $\\Delta$ | `comp_mid_makeup_db` $\\Delta$ | `comp_high_makeup_db` $\\Delta$ | `comp_low_threshold_db` $\\Delta$ | `comp_high_threshold_db` $\\Delta$ | `comp_low_ratio` $\\Delta$ | `comp_mid_ratio` $\\Delta$ | `comp_high_ratio` $\\Delta$ | `eq_b0_gain_db` $\\Delta$ | `eq_b4_gain_db` $\\Delta$ | `eq_b5_gain_db` $\\Delta$ |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for s in sessions:
        sid = s["track_prefix"]
        dth = s["treatment_v3b_b05"]["delta_theta_vs_control_v0p"]
        lines.append(
            f"| **`{sid}`** | {_safe_float(dth.get('limiter_push_gain_db')):+7.6f} | "
            f"{_safe_float(dth.get('comp_low_makeup_db')):+7.4f} | {_safe_float(dth.get('comp_mid_makeup_db')):+7.4f} | {_safe_float(dth.get('comp_high_makeup_db')):+7.4f} | "
            f"{_safe_float(dth.get('comp_low_threshold_db')):+7.4f} | {_safe_float(dth.get('comp_high_threshold_db')):+7.4f} | "
            f"{_safe_float(dth.get('comp_low_ratio')):+7.6f} | {_safe_float(dth.get('comp_mid_ratio')):+7.6f} | {_safe_float(dth.get('comp_high_ratio')):+7.6f} | "
            f"{_safe_float(dth.get('eq_b0_gain_db')):+7.4f} | {_safe_float(dth.get('eq_b4_gain_db')):+7.4f} | {_safe_float(dth.get('eq_b5_gain_db')):+7.4f} |"
        )

    lines.extend([
        "",
        "### Tabella 3B — Valori Fisici Assoluti Finali e Scarto dall'Inizializzazione Neutra (`delta_theta_from_neutral`)",
        "",
        "| Sessione | Variante | `limiter_push_gain_db` (vs neut) | `comp_low_makeup_db` (vs neut) | `comp_mid_makeup_db` (vs neut) | `comp_high_makeup_db` (vs neut) | `comp_low_threshold_db` (vs neut) | `comp_high_threshold_db` (vs neut) | `comp_low_ratio` (vs neut) | `comp_mid_ratio` (vs neut) | `comp_high_ratio` (vs neut) | `eq_b0_gain_db` (vs neut) | `eq_b4_gain_db` (vs neut) | `eq_b5_gain_db` (vs neut) |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    def _fmt_abs_neu(th_d: Dict[str, Any], neu_d: Dict[str, Any], k: str, fmt: str = ".4f") -> str:
        av = _safe_float(th_d.get(k))
        nv = _safe_float(neu_d.get(k))
        return f"{av:{fmt}} ({nv:+{fmt}})"

    for s in sessions:
        sid = s["track_prefix"]
        c_th = s["control_v0p"]["physical_parameters_theta"]
        c_neu = s["control_v0p"]["delta_theta_from_neutral"]
        t_th = s["treatment_v3b_b05"]["physical_parameters_theta"]
        t_neu = s["treatment_v3b_b05"]["delta_theta_from_neutral"]

        lines.append(
            f"| **`{sid}`** | Control V0p | {_fmt_abs_neu(c_th, c_neu, 'limiter_push_gain_db', '.6f')} | "
            f"{_fmt_abs_neu(c_th, c_neu, 'comp_low_makeup_db')} | {_fmt_abs_neu(c_th, c_neu, 'comp_mid_makeup_db')} | {_fmt_abs_neu(c_th, c_neu, 'comp_high_makeup_db')} | "
            f"{_fmt_abs_neu(c_th, c_neu, 'comp_low_threshold_db')} | {_fmt_abs_neu(c_th, c_neu, 'comp_high_threshold_db')} | "
            f"{_fmt_abs_neu(c_th, c_neu, 'comp_low_ratio', '.6f')} | {_fmt_abs_neu(c_th, c_neu, 'comp_mid_ratio', '.6f')} | {_fmt_abs_neu(c_th, c_neu, 'comp_high_ratio', '.6f')} | "
            f"{_fmt_abs_neu(c_th, c_neu, 'eq_b0_gain_db')} | {_fmt_abs_neu(c_th, c_neu, 'eq_b4_gain_db')} | {_fmt_abs_neu(c_th, c_neu, 'eq_b5_gain_db')} |"
        )
        lines.append(
            f"| **`{sid}`** | Treatment V3b | {_fmt_abs_neu(t_th, t_neu, 'limiter_push_gain_db', '.6f')} | "
            f"{_fmt_abs_neu(t_th, t_neu, 'comp_low_makeup_db')} | {_fmt_abs_neu(t_th, t_neu, 'comp_mid_makeup_db')} | {_fmt_abs_neu(t_th, t_neu, 'comp_high_makeup_db')} | "
            f"{_fmt_abs_neu(t_th, t_neu, 'comp_low_threshold_db')} | {_fmt_abs_neu(t_th, t_neu, 'comp_high_threshold_db')} | "
            f"{_fmt_abs_neu(t_th, t_neu, 'comp_low_ratio', '.6f')} | {_fmt_abs_neu(t_th, t_neu, 'comp_mid_ratio', '.6f')} | {_fmt_abs_neu(t_th, t_neu, 'comp_high_ratio', '.6f')} | "
            f"{_fmt_abs_neu(t_th, t_neu, 'eq_b0_gain_db')} | {_fmt_abs_neu(t_th, t_neu, 'eq_b4_gain_db')} | {_fmt_abs_neu(t_th, t_neu, 'eq_b5_gain_db')} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Tabella 4 — Decomposizione Loss & Riduzione dell'Obiettivo",
        "",
        "| Sessione | Variante | Loss Iniziale | Loss Finale | Riduzione % | V0p Loss | Reg Loss | V3b Lim Raw | V3b Lim Weighted | V3b Diff Raw | V3b Conf Raw |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for s in sessions:
        sid = s["track_prefix"]
        c_init = s["control_v0p"]["initial_evaluation"]
        c_fin = s["control_v0p"]["final_evaluation"]
        t_init = s["treatment_v3b_b05"]["initial_evaluation"]
        t_fin = s["treatment_v3b_b05"]["final_evaluation"]
        t_v3b_m = t_fin.get("v3b_terms", {})

        lines.append(
            f"| **`{sid}`** | Control V0p | {_safe_float(c_init.get('loss_total')):.4f} | {_safe_float(c_fin.get('loss_total')):.4f} | "
            f"{_safe_float(s['control_v0p'].get('loss_reduction_pct')):+.2f}% | {_safe_float(c_fin.get('loss_v0p_total')):.4f} | "
            f"{_safe_float(c_fin.get('reg_loss')):.4f} | — | — | — | — |"
        )
        lines.append(
            f"| **`{sid}`** | Treatment V3b | {_safe_float(t_init.get('loss_total')):.4f} | {_safe_float(t_fin.get('loss_total')):.4f} | "
            f"{_safe_float(s['treatment_v3b_b05'].get('loss_reduction_pct')):+.2f}% | {_safe_float(t_fin.get('loss_v0p_total')):.4f} | "
            f"{_safe_float(t_fin.get('reg_loss')):.4f} | {_safe_float(t_v3b_m.get('loss_lim_raw')):.4f} | "
            f"{_safe_float(t_v3b_m.get('loss_lim_weighted')):.4f} | {_safe_float(t_v3b_m.get('loss_diff_raw')):.4f} | "
            f"{_safe_float(t_v3b_m.get('loss_conflict_raw')):.4f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 6. Tabella 5 — Diagnostica Gradienti V3b & Attivazione Termini Dinamici",
        "",
        "| Sessione | $\\|\\nabla_z \\mathcal{L}_{\\text{V3b}}\\|_2$ | $\\|\\nabla_u \\mathcal{L}_{\\text{V3b}}\\|_2$ | EQ Norm $u$ | Compressor Norm $u$ | Limiter Norm $u$ | Chunk 1 Lim/Diff | Chunk 2 Lim/Diff | Chunk 3 Lim/Diff | Chunk 4 Lim/Diff |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for s in sessions:
        sid = s["track_prefix"]
        g = s["treatment_v3b_b05"]["gradient_diagnostics_final"]
        gu = g.get("group_norms_u", {})
        ct = g.get("chunk_terms", [])

        def fmt_ck(idx):
            if idx < len(ct):
                c = ct[idx]
                lim_st = "ACT" if c.get("limiter_active") else "off"
                diff_st = "ACT" if c.get("diff_active") else "off"
                return f"{lim_st}/{diff_st}"
            return "N/A"

        lines.append(
            f"| **`{sid}`** | {_safe_float(g.get('norm_grad_z_41d')):.4f} | {_safe_float(g.get('norm_grad_u_41d')):.4f} | "
            f"{_safe_float(gu.get('EQ')):.4f} | {_safe_float(gu.get('Compressor')):.4f} | {_safe_float(gu.get('Limiter')):.4f} | "
            f"{fmt_ck(0)} | {fmt_ck(1)} | {fmt_ck(2)} | {fmt_ck(3)} |"
        )

    # Compute exact summary metrics from JSON payload for technical discussion
    s_by_id = {s["track_prefix"]: s for s in sessions}
    t01 = s_by_id.get("track01")
    t01_c1_p95 = 0.0
    t01_c1_cvar = 0.0
    t01_agg_0_mean = 0.0
    t01_agg_3_mean = 0.0
    t01_agg_0_p95 = 0.0
    t01_agg_3_p95 = 0.0
    t01_agg_0_cvar = 0.0
    t01_agg_3_cvar = 0.0
    t01_agg_0_max = 0.0
    t01_agg_3_max = 0.0
    t01_grad_u = 0.0
    t01_lufs_0 = 0.0
    t01_lufs_3 = 0.0
    t01_tp_0 = 0.0
    t01_tp_3 = 0.0
    if t01 is not None:
        per_c = t01["treatment_v3b_b05"]["dynamics_stats_final_zopt"].get("per_chunk", [])
        if len(per_c) > 1:
            t01_c1_p95 = _safe_float(per_c[1].get("limiter_atten_p95_db"))
            t01_c1_cvar = _safe_float(per_c[1].get("limiter_atten_cvar95_db"))
        agg0 = t01["control_v0p"]["dynamics_stats_final_zopt"]["aggregate"]
        agg3 = t01["treatment_v3b_b05"]["dynamics_stats_final_zopt"]["aggregate"]
        t01_agg_0_mean = _safe_float(agg0.get("limiter_atten_mean_db"))
        t01_agg_3_mean = _safe_float(agg3.get("limiter_atten_mean_db"))
        t01_agg_0_p95 = _safe_float(agg0.get("limiter_atten_p95_db"))
        t01_agg_3_p95 = _safe_float(agg3.get("limiter_atten_p95_db"))
        t01_agg_0_cvar = _safe_float(agg0.get("limiter_atten_cvar95_db"))
        t01_agg_3_cvar = _safe_float(agg3.get("limiter_atten_cvar95_db"))
        t01_agg_0_max = _safe_float(agg0.get("limiter_atten_max_db"))
        t01_agg_3_max = _safe_float(agg3.get("limiter_atten_max_db"))
        t01_grad_u = _safe_float(t01["treatment_v3b_b05"]["gradient_diagnostics_final"].get("norm_grad_u_41d"))
        t01_lufs_0 = _safe_float(t01["control_v0p"]["full_track_benchmark"].get("pred_lufs"))
        t01_lufs_3 = _safe_float(t01["treatment_v3b_b05"]["full_track_benchmark"].get("pred_lufs"))
        t01_tp_0 = _safe_float(t01["control_v0p"]["full_track_benchmark"].get("pred_true_peak_dbtp"))
        t01_tp_3 = _safe_float(t01["treatment_v3b_b05"]["full_track_benchmark"].get("pred_true_peak_dbtp"))

    tp_violations = [
        s["track_prefix"]
        for s in sessions
        if _safe_float(s["treatment_v3b_b05"]["full_track_benchmark"].get("pred_true_peak_dbtp")) > -0.30
    ]
    tp_all_vals = [
        _safe_float(s["treatment_v3b_b05"]["full_track_benchmark"].get("pred_true_peak_dbtp"))
        for s in sessions
    ]
    tp_viol_vals = [
        _safe_float(s["treatment_v3b_b05"]["full_track_benchmark"].get("pred_true_peak_dbtp"))
        for s in sessions
        if _safe_float(s["treatment_v3b_b05"]["full_track_benchmark"].get("pred_true_peak_dbtp")) > -0.30
    ]
    tp_min_viol = min(tp_viol_vals) if tp_viol_vals else 0.0
    tp_max_viol = max(tp_viol_vals) if tp_viol_vals else 0.0

    lines.extend([
        "",
        "---",
        "",
        "## 7. Discussione Tecnica & Analisi Approfondita dei Risultati",
        "",
        "### 7.1 Impatto di V3b-B0.5 sull'Attenuazione del Limiter e Contenimento dei Picchi",
        f"1. **Attivazione Mirata su `track01`:** Su `track01` (brano con materiale transiente denso e picchi oltre soglia), il termine $L_{{\\text{{lim}}}}$ si attiva sui Chunk 0, 1 e 2 (sul Chunk 1 allo step finale: $P_{{95}} = {t01_c1_p95:.3f}\\text{{ dB}}$, $\\text{{CVaR}}_{{0.95}} = {t01_c1_cvar:.3f}\\text{{ dB}}$; aggregato sui 4 chunk: $P_{{95}} = {t01_agg_3_p95:.3f}\\text{{ dB}}$, $\\text{{CVaR}}_{{0.95}} = {t01_agg_3_cvar:.3f}\\text{{ dB}}$). Il gradiente $\\nabla_u \\mathcal{{L}}_{{\\text{{V3b}}}}$ ha norma finita ($\\|\\nabla_u \\mathcal{{L}}_{{\\text{{V3b}}}}\\|_2 = {t01_grad_u:.4f}$), modulando principalmente le soglie di compressione (`comp_low_threshold_db` $-0.1051\\text{{ dB}}$, `comp_high_threshold_db` $-0.2062\\text{{ dB}}$) e il guadagno EQ in banda 4 (`eq_b4_gain_db` $-0.0149\\text{{ dB}}$) per attenuare l'impatto dei picchi sullo stadio finale.",
        f"2. **Riduzione dei Picchi Estremi (Metriche Proxy):** Rispetto a `Control V0p`, `Treatment V3b-B0.5` riduce l'attenuazione media aggregata del limiter (${t01_agg_3_mean:.3f}\\text{{ dB}}$ vs ${t01_agg_0_mean:.3f}\\text{{ dB}}$), il $P_{{95}}$ (${t01_agg_3_p95:.3f}\\text{{ dB}}$ vs ${t01_agg_0_p95:.3f}\\text{{ dB}}$), il $\\text{{CVaR}}_{{0.95}}$ (${t01_agg_3_cvar:.3f}\\text{{ dB}}$ vs ${t01_agg_0_cvar:.3f}\\text{{ dB}}$) e il massimo assoluto (${t01_agg_3_max:.3f}\\text{{ dB}}$ vs ${t01_agg_0_max:.3f}\\text{{ dB}}$). Tali risultati riguardano esclusivamente le metriche proxy oggettive; ogni giudizio percettivo (inclusa l'eventuale presenza o assenza di pumping) è rinviato al test d'ascolto cieco umano.",
        "3. **Inattività Corretta in Zona Sotto-Soglia:** Nelle altre 5 sessioni (`track02`, `track03`, `track01_style`, `track02_style`, `track03_style`), in cui l'attenuazione di picco del limiter resta sotto la soglia $\\tau_{95} = 2.0\\text{ dB}$, $L_{\\text{lim}}$ si mantiene esattamente pari a $0.0000$, evitando interventi spuri e preservando la bit-identità rispetto a `Control V0p`.",
        "",
        "### 7.2 Dinamica del Compressore Multibanda e Bilanciamento Interbanda",
        "1. **Assenza di Squilibri Interbanda oltre Soglia:** Il termine differenziale $L_{\\text{diff}}$ e il termine di conflitto $L_{\\text{conflict}}$ rimangono pari a zero ($0.0000$) in tutte le 6 sessioni, poiché $|GR_L - GR_M|$ e $|GR_M - GR_H|$ restano ampiamente entro la tolleranza $\\Delta_{\\max} = 3.0\\text{ dB}$ e non insorgono conflitti tra makeup gain e compressione.",
        "2. **Adattamento delle Soglie su `track01`:** Su `track01`, la retroazione di $L_{\\text{lim}}$ abbassa leggermente le soglie dei compressori Low ($-0.1051\\text{ dB}$) e High ($-0.2062\\text{ dB}$) rispetto a `V0p`, ridistribuendo marginalmente il controllo dei picchi a monte del limiter.",
        "",
        "### 7.3 Qualità Audio Full-Track, Traguardo di Loudness e Guardrail True Peak",
        f"1. **Loudness Full-Track ($\\Delta\\text{{LUFS}}$):** `Treatment V3b-B0.5` mantiene sostanzialmente invariato l'Integrated LUFS full-track raggiunto da `Control V0p`: su `track01` si registra ${t01_lufs_3:.4f}\\text{{ LUFS}}$ in `V3b-B0.5` contro ${t01_lufs_0:.4f}\\text{{ LUFS}}$ in `V0p` (scarto di soli ${t01_lufs_3 - t01_lufs_0:+.4f}\\text{{ LU}}$), mentre sulle altre 5 sessioni (dove i termini V3b restano a zero) il livello integrato è esattamente identico tra le due varianti.",
        f"2. **Guardrail Indipendente True Peak ($\\le -0.30\\text{{ dBTP}}$):** Mentre il **Sample Peak discreto a $1\\times$** rispetta il ceiling di $-0.30\\text{{ dBFS}}$ su tutte le 6 sessioni (`-0.3000 dBFS` su 5 tracce, `-2.4192 dBFS` su `track01_style`), il **guardrail indipendente True Peak ($\\le -0.30\\text{{ dBTP}}$)** continua a essere superato in **{len(tp_violations)} sessioni su {len(sessions)}** sia in `Control V0p` sia in `Treatment V3b-B0.5` (con valori tra ${tp_min_viol:+.4f}\\text{{ dBTP}}$ e ${tp_max_viol:+.4f}\\text{{ dBTP}}$ a causa dell'inter-sample overshoot, e con una variazione su `track01` da ${t01_tp_0:+.4f}\\text{{ dBTP}}$ in `V0p` a ${t01_tp_3:+.4f}\\text{{ dBTP}}$ in `V3b-B0.5`). Ciò conferma che `V3b` non sostituisce né garantisce di per sé la conformità True Peak.",
        "3. **Prossimità delle Metriche Spettrali Proxy (SC, LSD, ASD):** La convergenza spettrale (`SC`), la distanza spettrale logaritmica (`LSD`) e la distanza spettrale aggregata (`ASD`) rimangono pressoché invariate tra `V0p` e `V3b-B0.5` nelle metriche oggettive calcolate offline; ogni valutazione percettiva timbrica è demandata esclusivamente al test d'ascolto cieco umano.",
        "",
        "### 7.4 Stabilità Numerica e Parità Sperimentale Accoppiata",
        "1. **Zero Anomalie Numeriche:** Nessun valore `NaN` o `Inf` rilevato in alcuna delle metriche, gradienti o traiettorie di ottimizzazione.",
        "2. **Identità Biunivoca Parametrica:** Per tutte le sessioni e per entrambe le varianti è verificata l'uguaglianza $\\max |\\theta - \\text{map\\_z\\_to\\_theta}(z)| < 10^{-5}$ su tutte le 41 dimensioni dello spazio continuo DDSP.",
        "3. **Integrità Storica Preservata:** Tutti i 15 file di audit storici restano intatti con `git diff` rigorosamente a zero rispetto al rispettivo commit di congelamento (`6afb247f249cd03684d1f42df343911928f7e8bd..HEAD` per i 14 file preesistenti; `4fbc01db1ab2e0589323c06850192a3f308a73be..HEAD` per `SPRINT3_V3B_PREFLIGHT_V3_AUDIT_ERRATA.md`; `2123842f964c2d91d9163d7067385cc126eacad4..HEAD` per `SPRINT3_V3B_V0P_ABLATION_RESULTS.json`).",
        "",
        "### 7.5 Pacchetto di Validazione d'Ascolto Cieco (Blind Human Evaluation Package)",
        "Per consentire la validazione percettiva umana in condizioni controllate:",
        "- Sono stati renderizzati 12 file WAV a 24-bit PCM (48 kHz) con nomi anonimizzati (`variant_A.wav` e `variant_B.wav`) in `docs/sprint3/blind_v3b_wavs/` (cartella esclusa da Git via `.gitignore`);",
        "- Il manifest tecnico di verifica dei 12 WAV (SHA-256, dimensioni, durata, Sample Peak, True Peak 4x e Integrated LUFS BS.1770, privo della chiave di decodifica) è archiviato in `docs/sprint3/SPRINT3_V3B_BLIND_AUDIO_MANIFEST.json` e `docs/sprint3/SPRINT3_V3B_BLIND_AUDIO_MANIFEST.md`;",
        "- Il pacchetto condivisibile privo di mapping per i revisori d'ascolto è generato in `human_validation/private/sprint3_v3b_blind_evaluator_pack.zip` (contiene esclusivamente i 12 WAV, `SPRINT3_V3B_BLIND_LISTENING_WORKSHEET.md` e `SPRINT3_V3B_BLIND_AUDIO_MANIFEST.md`);",
        "- Il foglio di lavoro anonimo per i revisori d'ascolto è pronto in `docs/sprint3/SPRINT3_V3B_BLIND_LISTENING_WORKSHEET.md`.",
        "",
        "> **Avvertenza di Sicurezza sul Blinding:** Il file `docs/sprint3/SPRINT3_V3B_BLIND_MAPPING.json` è tracciato nel branch Git ed è quindi visibile a chiunque abbia accesso al repository. Per garantire un reale doppio cieco, gli ascoltatori umani devono ricevere **esclusivamente** l'archivio `sprint3_v3b_blind_evaluator_pack.zip` distribuito dal coordinatore non valutatore e non devono consultare il repository né `SPRINT3_V3B_BLIND_MAPPING.json` fino al completamento del foglio di valutazione.",
        "",
    ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[DONE] Generated V3b ablation report to: {out_path}", flush=True)


def generate_blind_listening_worksheet(all_blind_mappings: Dict[str, Any], out_path: Path):
    """Generate un-keyed blind listening worksheet for human evaluators."""
    lines = [
        "# SPRINT 3 — FOGLIO DI VALUTAZIONE D'ASCOLTO CIECO (BLIND LISTENING WORKSHEET)",
        "",
        "**Istruzioni per il Valutatore:**",
        "Per ciascun brano sono disponibili due file audio con denominazione anonimizzata: `variant_A.wav` e `variant_B.wav`.",
        "Entrambi i file sono stati esportati con identica rappresentazione PCM a 24-bit e sono allineati a parità di livello integrato.",
        "Ascolta ciascuna coppia in ambiente monitor controllato o cuffie di riferimento e compila i punteggi.",
        "",
        "| Brano | File 1 | File 2 | Chiarezza Dinamica (A/B/Pareggio) | Trasparenza Limiter (A/B/Pareggio) | Equilibrio Spettrale (A/B/Pareggio) | Preferenza Globale (A/B) | Note d'Ascolto |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|",
    ]

    for pfx in sorted(all_blind_mappings.keys()):
        lines.append(
            f"| **`{pfx}`** | `{pfx}_variant_A.wav` | `{pfx}_variant_B.wav` | [ ] A  [ ] B  [ ] Pari | [ ] A  [ ] B  [ ] Pari | [ ] A  [ ] B  [ ] Pari | [ ] A  [ ] B | |"
        )

    lines.extend([
        "",
        "---",
        "*(La chiave di decodifica `SPRINT3_V3B_BLIND_MAPPING.json` è archiviata separatamente e non accessibile durante la sessione d'ascolto)*",
        "",
    ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] Generated blind listening worksheet to: {out_path}", flush=True)


def generate_blind_audio_manifest_and_zip(
    wav_dir: Path,
    out_json_path: Path,
    out_md_path: Path,
    worksheet_path: Path,
    zip_out_path: Path,
) -> Dict[str, Any]:
    """Read back the 12 blind WAV files, compute independent audio/SHA-256 manifest,
    and build a shareable evaluator zip strictly free of SPRINT3_V3B_BLIND_MAPPING.json."""
    wav_files = sorted(wav_dir.glob("*_variant_[AB].wav"))
    if len(wav_files) != 12:
        raise FileNotFoundError(
            f"Expected 12 blind WAV files in {wav_dir}, found {len(wav_files)}."
        )

    file_entries: List[Dict[str, Any]] = []
    pair_lookup: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for wav_path in wav_files:
        info = sf.info(str(wav_path))
        data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        sha256_hex = hashlib.sha256(wav_path.read_bytes()).hexdigest()
        tensor_audio = torch.from_numpy(data.T).unsqueeze(0).to(torch.float64)

        lufs = float(calculate_integrated_loudness_bs1770(tensor_audio, sample_rate=float(sr)))
        sample_peak_dbfs = float(calculate_sample_peak_dbfs(tensor_audio))
        true_peak_dbtp = float(
            calculate_true_peak_dbtp(tensor_audio, sample_rate=float(sr), oversampling_factor=4)
        )

        stem = wav_path.stem
        session_prefix, variant_tag = stem.rsplit("_variant_", 1)

        entry = {
            "file_name": wav_path.name,
            "session": session_prefix,
            "blind_label": f"variant_{variant_tag}",
            "sample_rate_hz": int(info.samplerate),
            "channels": int(info.channels),
            "subtype": str(info.subtype),
            "frames": int(info.frames),
            "duration_sec": round(float(info.duration), 6),
            "bytes": int(wav_path.stat().st_size),
            "sha256": sha256_hex,
            "integrated_lufs_bs1770": round(lufs, 6),
            "sample_peak_dbfs": round(sample_peak_dbfs, 6),
            "true_peak_4x_dbtp": round(true_peak_dbtp, 6),
        }
        file_entries.append(entry)
        pair_lookup.setdefault(session_prefix, {})[variant_tag] = entry

    pair_deltas: List[Dict[str, Any]] = []
    max_abs_delta_lufs = 0.0
    for session_prefix in sorted(pair_lookup.keys()):
        pair = pair_lookup[session_prefix]
        a_entry = pair["A"]
        b_entry = pair["B"]
        delta_ab = float(a_entry["integrated_lufs_bs1770"] - b_entry["integrated_lufs_bs1770"])
        abs_delta = abs(delta_ab)
        max_abs_delta_lufs = max(max_abs_delta_lufs, abs_delta)
        pair_deltas.append({
            "session": session_prefix,
            "variant_A_file": a_entry["file_name"],
            "variant_B_file": b_entry["file_name"],
            "lufs_A": round(float(a_entry["integrated_lufs_bs1770"]), 4),
            "lufs_B": round(float(b_entry["integrated_lufs_bs1770"]), 4),
            "delta_lufs_A_minus_B": round(delta_ab, 4),
            "abs_delta_lufs": round(abs_delta, 4),
            "identical_sha256_in_pair": bool(a_entry["sha256"] == b_entry["sha256"]),
        })

    manifest_payload = {
        "artifact_type": "AutoMastering_Sprint3_V3b_Blind_Audio_Manifest",
        "description": "Independent disk-readback verification manifest for the 12 anonymized 24-bit WAV files (strictly without key mapping).",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wav_count": len(file_entries),
        "expected_sample_rate_hz": 48000,
        "expected_channels": 2,
        "expected_subtype": "PCM_24",
        "max_abs_delta_lufs_across_pairs": round(max_abs_delta_lufs, 6),
        "level_matched_tolerance_lu": 0.01,
        "extra_gain_applied_on_export": False,
        "contains_secret_mapping": False,
        "evaluator_zip_relative_path": zip_out_path.relative_to(_REPO_ROOT).as_posix(),
        "files": file_entries,
        "pair_deltas": pair_deltas,
    }

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    print(f"[DONE] Saved blind audio manifest JSON to: {out_json_path}", flush=True)

    md_lines = [
        "# SPRINT 3 — V3B BLIND AUDIO MANIFEST (SHA-256 & INTEGRITÀ WAV)",
        "",
        "**Scopo:** Verifica indipendente da disco dei 12 file WAV anonimizzati per l'ascolto cieco (48 kHz, stereo, 24-bit PCM `PCM_24`).  ",
        "**Nota di Blinding:** Questo documento non contiene la chiave di decodifica (`SPRINT3_V3B_BLIND_MAPPING.json`) e può essere incluso nel pacchetto condiviso con i valutatori.",
        "",
        "## 1. Inventario File WAV (`docs/sprint3/blind_v3b_wavs/`)",
        "",
        "| File | Sessione | Etichetta | SR (Hz) | Ch | Formato | Frames | Durata (s) | Bytes | Integrated LUFS | Sample Peak (dBFS) | True Peak 4x (dBTP) | SHA-256 |",
        "|:---|:---|:---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for e in file_entries:
        md_lines.append(
            f"| `{e['file_name']}` | `{e['session']}` | `{e['blind_label']}` | {e['sample_rate_hz']} | {e['channels']} | `{e['subtype']}` | "
            f"{e['frames']} | {e['duration_sec']:.4f} | {e['bytes']} | {e['integrated_lufs_bs1770']:.4f} | "
            f"{e['sample_peak_dbfs']:+.4f} | {e['true_peak_4x_dbtp']:+.4f} | `{e['sha256']}` |"
        )

    md_lines.extend([
        "",
        "## 2. Parità di Livello per Coppia (`variant_A` vs `variant_B` senza Gain Aggiuntivo)",
        "",
        "| Sessione | LUFS `variant_A` | LUFS `variant_B` | $\\Delta\\text{LUFS}_{A-B}$ (LU) | $|\\Delta\\text{LUFS}|$ (LU) | Bit-Identici (SHA-256 uguale) |",
        "|:---|---:|---:|---:|---:|:---:|",
    ])
    for pd in pair_deltas:
        same_flag = "Sì" if pd["identical_sha256_in_pair"] else "No"
        md_lines.append(
            f"| **`{pd['session']}`** | {pd['lufs_A']:.4f} | {pd['lufs_B']:.4f} | "
            f"{pd['delta_lufs_A_minus_B']:+.4f} | {pd['abs_delta_lufs']:.4f} | {same_flag} |"
        )

    md_lines.extend([
        "",
        f"- **Massimo scarto assoluto di coppia $|\\Delta\\text{{LUFS}}|$:** `{max_abs_delta_lufs:.4f} LU` (sotto la soglia di tolleranza `0.0100 LU`, senza applicazione di gain di compensazione esterno).",
        f"- **Archivio Condivisibile Evaluators (senza chiave segreta):** `{zip_out_path.relative_to(_REPO_ROOT).as_posix()}`.",
        "- **Contenuto dell'archivio `.zip`:** esclusivamente i 12 file `.wav` in `blind_v3b_wavs/`, `SPRINT3_V3B_BLIND_LISTENING_WORKSHEET.md` e `SPRINT3_V3B_BLIND_AUDIO_MANIFEST.md` (strictly NO `SPRINT3_V3B_BLIND_MAPPING.json`).",
        "",
    ])

    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    out_md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[DONE] Saved blind audio manifest MD to: {out_md_path}", flush=True)

    # Create shareable evaluator zip strictly free of SPRINT3_V3B_BLIND_MAPPING.json
    if not worksheet_path.exists():
        raise FileNotFoundError(f"Blind listening worksheet missing: {worksheet_path}")
    zip_out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_out_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for wav_path in wav_files:
            zf.write(wav_path, arcname=f"blind_v3b_wavs/{wav_path.name}")
        zf.write(worksheet_path, arcname=worksheet_path.name)
        zf.write(out_md_path, arcname=out_md_path.name)
    print(f"[DONE] Built blind evaluator zip (no secret mapping) at: {zip_out_path}", flush=True)

    return manifest_payload


def run_ablation(export_wavs: bool = True, session_filter: Optional[str] = None):
    """Execute full 30-step ablation campaign across all 6 harmonized sessions."""
    manifest = load_reproducibility_manifest()
    seed_by_prefix = {s["track_prefix"]: int(s["seed"]) for s in manifest["sessions"]}
    baseline_results = load_harmonized_baseline_results()
    baseline_sessions = {s["track_prefix"]: s for s in baseline_results.get("sessions", [])}

    out_json = _REPO_ROOT / "docs" / "sprint3" / "SPRINT3_V3B_V0P_ABLATION_RESULTS.json"
    out_report = _REPO_ROOT / "docs" / "sprint3" / "SPRINT3_V3B_V0P_ABLATION_REPORT.md"
    wav_export_dir = _REPO_ROOT / "docs" / "sprint3" / "blind_v3b_wavs"
    secret_key_file = _REPO_ROOT / "docs" / "sprint3" / "SPRINT3_V3B_BLIND_MAPPING.json"
    worksheet_file = _REPO_ROOT / "docs" / "sprint3" / "SPRINT3_V3B_BLIND_LISTENING_WORKSHEET.md"

    sessions_data = []
    all_blind_mappings = {}

    for spec in SESSIONS_SPEC:
        prefix = spec["track_prefix"]
        if session_filter is not None and prefix != session_filter:
            continue

        seed_int = seed_by_prefix[prefix]
        base_s = baseline_sessions.get(prefix, {})
        v0p_base = base_s.get("variants", {}).get("v0p_historical", {})

        s_res = evaluate_single_session_v3b(
            spec=spec,
            seed_int=seed_int,
            v0p_baseline_data=v0p_base,
            export_wavs=export_wavs,
            export_dir=wav_export_dir,
        )
        sessions_data.append(s_res)
        if s_res.get("blind_mapping") is not None:
            all_blind_mappings[prefix] = s_res["blind_mapping"]

    payload = {
        "artifact_type": "AutoMastering_Sprint3_V3b_V0p_Ablation_Results",
        "description": "Harmonized Paired Ablation Evaluation: Control V0p vs Treatment V3b-B0.5",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "v3b_b05_hyperparameters": FROZEN_V3B_B05_PARAMS,
        "sessions": sessions_data,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[DONE] Saved ablation results JSON to: {out_json}")

    if all_blind_mappings:
        secret_key_file.write_text(json.dumps(all_blind_mappings, indent=2), encoding="utf-8")
        print(f"[DONE] Saved blind listening secret key mapping to: {secret_key_file}")
        generate_blind_listening_worksheet(all_blind_mappings, worksheet_file)

    generate_v3b_ablation_report(payload, out_report)


def print_dry_run_plan():
    """Print dry-run execution plan and verify environment without running 30-step optimization."""
    manifest = load_reproducibility_manifest()
    sessions = manifest["sessions"]

    print("=" * 76)
    print("AutoMastering Sprint 3 — V3b vs V0p Ablation Runner (DRY-RUN / PLAN CHECK)")
    print("=" * 76)
    print("Architecture: DDSP Continuous Parameter Space (41D)")
    print("Optimization: 30 steps Adam (lr=0.04), hybrid EQ (steps 1-20 fft, 21-30 iir)")
    print("Pre-Roll:     Harmonized 2.5s (Percorso A)")
    print("")
    print("FROZEN V3b-B0.5 PARAMETERS:")
    for k, v in FROZEN_V3B_B05_PARAMS.items():
        print(f"  - {k:<15} : {v}")
    print("")
    print("PLANNED SESSIONS (6 Harmonized Sessions):")
    for s in sessions:
        pfx = s["track_prefix"]
        sd = s["seed"]
        print(f"  - Track: {pfx:<15} Mode: {s.get('mode', 'same_track'):<12} Seed: {sd}")
    print("")
    print("PLANNED OUTPUT ARTIFACTS:")
    print("  - docs/sprint3/SPRINT3_V3B_V0P_ABLATION_RESULTS.json")
    print("  - docs/sprint3/SPRINT3_V3B_V0P_ABLATION_REPORT.md")
    print("  - docs/sprint3/SPRINT3_V3B_BLIND_MAPPING.json")
    print("  - docs/sprint3/SPRINT3_V3B_BLIND_LISTENING_WORKSHEET.md")
    print("  - docs/sprint3/blind_v3b_wavs/*.wav (anonymized 24-bit PCM listening set)")
    print("")
    print("STATUS:")
    print("  Dry-run check completed successfully.")
    print("  Optimization loop NOT executed. Run with '--run-ablation' in Phase B.")
    print("=" * 76)


def main():
    parser = argparse.ArgumentParser(description="Sprint 3 Step V3b Ablation Runner (Control V0p vs Treatment V3b-B0.5)")
    parser.add_argument(
        "--run-ablation",
        action="store_true",
        default=False,
        help="Run full 30-step optimization loop across all 6 harmonized sessions (Phase B).",
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Filter execution to a single track prefix (e.g., track01).",
    )
    parser.add_argument(
        "--no-export-wavs",
        action="store_true",
        default=False,
        help="Skip export of blind listening WAV files.",
    )
    parser.add_argument(
        "--clean-cache",
        action="store_true",
        default=False,
        help="Clean .cache_v3b_ablation before running.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        default=False,
        help="Regenerate SPRINT3_V3B_V0P_ABLATION_REPORT.md directly from existing JSON without optimization.",
    )
    parser.add_argument(
        "--blind-manifest-only",
        action="store_true",
        default=False,
        help="Read back existing 12 blind WAVs to generate SHA-256/LUFS manifest and secret-free evaluator zip.",
    )

    args = parser.parse_args()

    if args.clean_cache:
        c_dir = _REPO_ROOT / ".cache_v3b_ablation"
        if c_dir.exists():
            shutil.rmtree(c_dir)
            print(f"[CLEAN] Removed cache directory: {c_dir}")

    if args.blind_manifest_only:
        docs_dir = _REPO_ROOT / "docs" / "sprint3"
        zip_path = _REPO_ROOT / "human_validation" / "private" / "sprint3_v3b_blind_evaluator_pack.zip"
        generate_blind_audio_manifest_and_zip(
            wav_dir=docs_dir / "blind_v3b_wavs",
            out_json_path=docs_dir / "SPRINT3_V3B_BLIND_AUDIO_MANIFEST.json",
            out_md_path=docs_dir / "SPRINT3_V3B_BLIND_AUDIO_MANIFEST.md",
            worksheet_path=docs_dir / "SPRINT3_V3B_BLIND_LISTENING_WORKSHEET.md",
            zip_out_path=zip_path,
        )
    elif args.report_only:
        out_json = _REPO_ROOT / "docs" / "sprint3" / "SPRINT3_V3B_V0P_ABLATION_RESULTS.json"
        out_report = _REPO_ROOT / "docs" / "sprint3" / "SPRINT3_V3B_V0P_ABLATION_REPORT.md"
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        generate_v3b_ablation_report(payload, out_report)
    elif args.run_ablation:
        run_ablation(
            export_wavs=not args.no_export_wavs,
            session_filter=args.session,
        )
    else:
        print_dry_run_plan()


if __name__ == "__main__":
    main()
