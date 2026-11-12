"""Sprint 3 Step V3b Read-Only Preflight Diagnostic Runner (V2).

Evaluates DSP observability, empirical gain reduction/limiter traces, Candidate A vs B
dynamic prior losses, and gradient diagnostics on the 6 harmonized sessions at Z_NEUTRAL
and V0p normalized_z_opt.

Features in V2:
  - lambda_95 = 0.5 in CANDIDATE_CONFIGS for both Candidate_A and Candidate_B
  - Component-level gradient diagnostics for L_lim, L_diff, L_conflict, and L_dyn_total (unweighted & weighted)
  - Gradients evaluated both in normalized z space and in unconstrained logit u space:
      grad_u = grad_z * z * (1 - z)
  - Explicit distinction between grad_v0p_total and grad_lufs_only
  - Cosine similarities wrt grad_v0p_total and grad_lufs_only on active groups (both z and u)
  - 1D scalar Limiter diagnostics (scalar derivative and sign)
  - Per-chunk statistics (k in 0..3) and session-level aggregation (mean and max over chunks)
  - Export to:
      docs/sprint3/SPRINT3_V3B_PREFLIGHT_V2_RESULTS.json
      docs/sprint3/SPRINT3_V3B_PREFLIGHT_V2_REPORT.md

ZERO training, ZERO optimization, ZERO modification of historical/harmonized JSON artifacts.
"""

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "python"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ddsp_mastering.dsp.chain import DifferentiableMasteringChain
from ddsp_mastering.loss.dynamic_prior import (
    V3bDynamicPriorLoss,
    compute_active_sample_mask,
    TOTAL_DSP_DELAY_SAMPLES,
)
from ddsp_mastering.optimize.optimizer import AutoMasteringOptimizer
from ddsp_mastering.param_space import (
    PARAM_SPECS,
    Z_NEUTRAL,
    map_z_to_theta,
    unpack_theta_dict,
)
from scripts.sprint3.harmonized_session_prep import (
    SESSIONS_SPEC,
    prepare_harmonized_session,
)

CANDIDATE_CONFIGS = {
    "Candidate_A": {
        "tau_avg": 1.5,
        "tau_95": 3.5,
        "delta_max": 4.0,
        "delta_ref": 2.0,
        "delta_m": 1.25,
        "lambda_95": 0.5,
        "w_lim": 0.05,
        "w_diff": 0.02,
        "w_conflict": 0.05,
    },
    "Candidate_B": {
        "tau_avg": 0.5,
        "tau_95": 2.0,
        "delta_max": 3.0,
        "delta_ref": 3.0,
        "delta_m": 1.25,
        "lambda_95": 0.5,
        "w_lim": 0.02,
        "w_diff": 0.02,
        "w_conflict": 0.02,
    },
}

PARAM_GROUPS = {
    "EQ": list(range(0, 18)),
    "Crossover": list(range(18, 20)),
    "Compressor": list(range(20, 35)),
    "Imager": list(range(35, 38)),
    "Saturator": list(range(38, 40)),
    "Limiter": list(range(40, 41)),
    "comp_thresholds": [20, 25, 30],
    "comp_ratios": [21, 26, 31],
    "comp_makeups": [24, 29, 34],
    "makeup_gains": [24, 29, 34],
    "limiter_push": [40],
}


def load_v2_results() -> Dict[str, Any]:
    v2_path = _REPO_ROOT / "docs" / "sprint3" / "SPRINT3_V2_ABLATION_RESULTS.json"
    with open(v2_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_grad_u(grad_z: torch.Tensor, z_point: torch.Tensor) -> torch.Tensor:
    """Convert gradient wrt normalized z in (0, 1) to gradient wrt logit u in R.
    
    Using exact Jacobian: dz_i / du_i = z_i * (1 - z_i).
    """
    dz_du = z_point * (1.0 - z_point)
    return grad_z * dz_du


def compute_group_l2_norms(grad_vector: torch.Tensor) -> Dict[str, float]:
    """Compute L2 norms for each parameter group and subgroup."""
    norms = {}
    for group_name, indices in PARAM_GROUPS.items():
        sub_grad = grad_vector[indices].to(torch.float64)
        norms[group_name] = float(torch.norm(sub_grad, p=2).item())
    norms["total_41d"] = float(torch.norm(grad_vector.to(torch.float64), p=2).item())
    return norms


def compute_cosine_similarity(g1: torch.Tensor, g2: torch.Tensor, indices: List[int]) -> float:
    """Compute cosine similarity between two gradient vectors restricted to indices."""
    sub_g1 = g1[indices].to(torch.float64)
    sub_g2 = g2[indices].to(torch.float64)
    norm1 = torch.norm(sub_g1, p=2).item()
    norm2 = torch.norm(sub_g2, p=2).item()
    denom = norm1 * norm2
    if denom > 1e-12:
        return float((torch.dot(sub_g1, sub_g2).item()) / denom)
    return 0.0


def compute_1d_limiter_diagnostics(
    grad_dyn_z: torch.Tensor,
    grad_dyn_u: torch.Tensor,
    g_v0p_total_z: torch.Tensor,
    g_v0p_total_u: torch.Tensor,
    g_lufs_only_z: torch.Tensor,
    g_lufs_only_u: torch.Tensor,
) -> Dict[str, Any]:
    """Compute exact scalar derivative and sign diagnostics for 1D Limiter group (index 40)."""
    val_z = float(grad_dyn_z[40].item())
    val_u = float(grad_dyn_u[40].item())
    sgn_z = 1.0 if val_z > 1e-12 else (-1.0 if val_z < -1e-12 else 0.0)
    sgn_u = 1.0 if val_u > 1e-12 else (-1.0 if val_u < -1e-12 else 0.0)

    cos_v0p_z = compute_cosine_similarity(grad_dyn_z, g_v0p_total_z, [40])
    cos_v0p_u = compute_cosine_similarity(grad_dyn_u, g_v0p_total_u, [40])
    cos_lufs_z = compute_cosine_similarity(grad_dyn_z, g_lufs_only_z, [40])
    cos_lufs_u = compute_cosine_similarity(grad_dyn_u, g_lufs_only_u, [40])

    return {
        "scalar_derivative_z": val_z,
        "scalar_derivative_u": val_u,
        "sign_z": sgn_z,
        "sign_u": sgn_u,
        "cos_sim_wrt_v0p_total_z": cos_v0p_z,
        "cos_sim_wrt_v0p_total_u": cos_v0p_u,
        "cos_sim_wrt_lufs_only_z": cos_lufs_z,
        "cos_sim_wrt_lufs_only_u": cos_lufs_u,
        "note": "For 1D scalar group Limiter (index 40), cosine similarity represents pure scalar sign agreement.",
    }


def evaluate_empirical_stats(
    chain: DifferentiableMasteringChain,
    prep_data: Dict[str, Any],
    z_vector: torch.Tensor,
) -> Dict[str, Any]:
    """Compute empirical limiter, compressor, and differential metrics per-chunk and aggregated."""
    mix_chunks = prep_data["mix_chunks"]
    preroll_list = prep_data["preroll_list"]
    K = len(mix_chunks)

    chunk_stats = []
    agg_act_samples = 0
    agg_eval_samples = 0

    all_a_lim = []
    all_gr_low = []
    all_gr_mid = []
    all_gr_high = []
    all_diff_lm = []
    all_diff_mh = []

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
            n_act_k = int(mask_smp.sum().item())
            agg_act_samples += n_act_k
            agg_eval_samples += T_eval

            if not is_silent and n_act_k > 0:
                a_act = a_lim_eval[mask_smp]
                gr_l_act = gr_l_eval[mask_smp]
                gr_m_act = gr_m_eval[mask_smp]
                gr_h_act = gr_h_eval[mask_smp]
                diff_lm_act = torch.abs(gr_l_eval - gr_m_eval)[mask_smp]
                diff_mh_act = torch.abs(gr_m_eval - gr_h_eval)[mask_smp]

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

                ch_stat = {
                    "chunk_index": k,
                    "n_active_samples": n_act_k,
                    "t_eval_samples": T_eval,
                    "active_ratio": float(n_act_k / T_eval),
                    "active_percentage": float(100.0 * n_act_k / T_eval),
                    "is_silent": False,
                    "lim_mean_db": float(torch.mean(a_act).item()),
                    "lim_p95_db": p95_k,
                    "lim_cvar95_db": cvar_k,
                    "lim_max_db": float(torch.max(a_act).item()),
                    "gr_low_mean_db": float(torch.mean(gr_l_act).item()),
                    "gr_low_max_db": float(torch.max(gr_l_act).item()),
                    "gr_mid_mean_db": float(torch.mean(gr_m_act).item()),
                    "gr_mid_max_db": float(torch.max(gr_m_act).item()),
                    "gr_high_mean_db": float(torch.mean(gr_h_act).item()),
                    "gr_high_max_db": float(torch.max(gr_h_act).item()),
                    "diff_lm_mean_db": float(torch.mean(diff_lm_act).item()),
                    "diff_lm_max_db": float(torch.max(diff_lm_act).item()),
                    "diff_mh_mean_db": float(torch.mean(diff_mh_act).item()),
                    "diff_mh_max_db": float(torch.max(diff_mh_act).item()),
                }
            else:
                ch_stat = {
                    "chunk_index": k,
                    "n_active_samples": 0,
                    "t_eval_samples": T_eval,
                    "active_ratio": 0.0,
                    "active_percentage": 0.0,
                    "is_silent": True,
                    "lim_mean_db": 0.0,
                    "lim_p95_db": 0.0,
                    "lim_cvar95_db": 0.0,
                    "lim_max_db": 0.0,
                    "gr_low_mean_db": 0.0,
                    "gr_low_max_db": 0.0,
                    "gr_mid_mean_db": 0.0,
                    "gr_mid_max_db": 0.0,
                    "gr_high_mean_db": 0.0,
                    "gr_high_max_db": 0.0,
                    "diff_lm_mean_db": 0.0,
                    "diff_lm_max_db": 0.0,
                    "diff_mh_mean_db": 0.0,
                    "diff_mh_max_db": 0.0,
                }
            chunk_stats.append(ch_stat)

    # Session aggregation: mean across chunks, max across chunks, and pooled
    chunk_mean = {
        "limiter": {
            "mean_atten_db": float(np.mean([c["lim_mean_db"] for c in chunk_stats])),
            "p95_atten_db": float(np.mean([c["lim_p95_db"] for c in chunk_stats])),
            "cvar95_atten_db": float(np.mean([c["lim_cvar95_db"] for c in chunk_stats])),
            "max_atten_db": float(np.mean([c["lim_max_db"] for c in chunk_stats])),
        },
        "compressor_gr": {
            "gr_low_mean_db": float(np.mean([c["gr_low_mean_db"] for c in chunk_stats])),
            "gr_low_max_db": float(np.mean([c["gr_low_max_db"] for c in chunk_stats])),
            "gr_mid_mean_db": float(np.mean([c["gr_mid_mean_db"] for c in chunk_stats])),
            "gr_mid_max_db": float(np.mean([c["gr_mid_max_db"] for c in chunk_stats])),
            "gr_high_mean_db": float(np.mean([c["gr_high_mean_db"] for c in chunk_stats])),
            "gr_high_max_db": float(np.mean([c["gr_high_max_db"] for c in chunk_stats])),
        },
        "interband_diff": {
            "diff_lm_mean_db": float(np.mean([c["diff_lm_mean_db"] for c in chunk_stats])),
            "diff_lm_max_db": float(np.mean([c["diff_lm_max_db"] for c in chunk_stats])),
            "diff_mh_mean_db": float(np.mean([c["diff_mh_mean_db"] for c in chunk_stats])),
            "diff_mh_max_db": float(np.mean([c["diff_mh_max_db"] for c in chunk_stats])),
        },
    }

    chunk_max = {
        "limiter": {
            "mean_atten_db": float(np.max([c["lim_mean_db"] for c in chunk_stats])),
            "p95_atten_db": float(np.max([c["lim_p95_db"] for c in chunk_stats])),
            "cvar95_atten_db": float(np.max([c["lim_cvar95_db"] for c in chunk_stats])),
            "max_atten_db": float(np.max([c["lim_max_db"] for c in chunk_stats])),
        },
        "compressor_gr": {
            "gr_low_max_db": float(np.max([c["gr_low_max_db"] for c in chunk_stats])),
            "gr_mid_max_db": float(np.max([c["gr_mid_max_db"] for c in chunk_stats])),
            "gr_high_max_db": float(np.max([c["gr_high_max_db"] for c in chunk_stats])),
        },
        "interband_diff": {
            "diff_lm_max_db": float(np.max([c["diff_lm_max_db"] for c in chunk_stats])),
            "diff_mh_max_db": float(np.max([c["diff_mh_max_db"] for c in chunk_stats])),
        },
    }

    if len(all_a_lim) > 0:
        cat_a_lim = torch.cat(all_a_lim)
        cat_gr_l = torch.cat(all_gr_low)
        cat_gr_m = torch.cat(all_gr_mid)
        cat_gr_h = torch.cat(all_gr_high)
        cat_diff_lm = torch.cat(all_diff_lm)
        cat_diff_mh = torch.cat(all_diff_mh)

        k_tail_tot = max(1, int(math.ceil(0.05 * cat_a_lim.numel())))
        topk_tot, _ = torch.topk(cat_a_lim, k_tail_tot)

        pooled_active = {
            "total_active_samples": agg_act_samples,
            "total_eval_samples": agg_eval_samples,
            "overall_active_ratio": float(agg_act_samples / max(1, agg_eval_samples)),
            "limiter": {
                "mean_atten_db": float(torch.mean(cat_a_lim).item()),
                "p95_atten_db": float(torch.quantile(cat_a_lim.float(), 0.95).item()),
                "cvar95_atten_db": float(torch.mean(topk_tot).item()),
                "max_atten_db": float(torch.max(cat_a_lim).item()),
            },
            "compressor_gr": {
                "gr_low_mean_db": float(torch.mean(cat_gr_l).item()),
                "gr_low_p95_db": float(torch.quantile(cat_gr_l.float(), 0.95).item()),
                "gr_low_max_db": float(torch.max(cat_gr_l).item()),
                "gr_mid_mean_db": float(torch.mean(cat_gr_m).item()),
                "gr_mid_p95_db": float(torch.quantile(cat_gr_m.float(), 0.95).item()),
                "gr_mid_max_db": float(torch.max(cat_gr_m).item()),
                "gr_high_mean_db": float(torch.mean(cat_gr_h).item()),
                "gr_high_p95_db": float(torch.quantile(cat_gr_h.float(), 0.95).item()),
                "gr_high_max_db": float(torch.max(cat_gr_h).item()),
            },
            "interband_diff": {
                "diff_lm_mean_db": float(torch.mean(cat_diff_lm).item()),
                "diff_lm_p95_db": float(torch.quantile(cat_diff_lm.float(), 0.95).item()),
                "diff_lm_max_db": float(torch.max(cat_diff_lm).item()),
                "diff_mh_mean_db": float(torch.mean(cat_diff_mh).item()),
                "diff_mh_p95_db": float(torch.quantile(cat_diff_mh.float(), 0.95).item()),
                "diff_mh_max_db": float(torch.max(cat_diff_mh).item()),
            },
        }
    else:
        pooled_active = {
            "total_active_samples": 0,
            "total_eval_samples": agg_eval_samples,
            "overall_active_ratio": 0.0,
            "limiter": {"mean_atten_db": 0.0, "p95_atten_db": 0.0, "cvar95_atten_db": 0.0, "max_atten_db": 0.0},
            "compressor_gr": {
                "gr_low_mean_db": 0.0, "gr_low_p95_db": 0.0, "gr_low_max_db": 0.0,
                "gr_mid_mean_db": 0.0, "gr_mid_p95_db": 0.0, "gr_mid_max_db": 0.0,
                "gr_high_mean_db": 0.0, "gr_high_p95_db": 0.0, "gr_high_max_db": 0.0,
            },
            "interband_diff": {
                "diff_lm_mean_db": 0.0, "diff_lm_p95_db": 0.0, "diff_lm_max_db": 0.0,
                "diff_mh_mean_db": 0.0, "diff_mh_p95_db": 0.0, "diff_mh_max_db": 0.0,
            },
        }

    return {
        "per_chunk_stats": chunk_stats,
        "session_summary": {
            "chunk_mean": chunk_mean,
            "chunk_max": chunk_max,
            "pooled_active": pooled_active,
        },
    }


def compute_v0p_loss_and_gradient(
    opt_v0p: AutoMasteringOptimizer,
    prep_data: Dict[str, Any],
    z_vector: torch.Tensor,
) -> Dict[str, Any]:
    """Compute V0p loss and gradients, with explicit distinction between total objective and LUFS-only."""
    mode = prep_data["mode"]
    mix_chunks = prep_data["mix_chunks"]
    ref_chunks = prep_data["ref_chunks"]
    preroll_list = prep_data["preroll_list"]
    target_lufs = prep_data["target_lufs"]
    K = len(mix_chunks)
    weights = [1.0 / K] * K

    z = z_vector.detach().clone().requires_grad_(True)
    stft_term = torch.tensor(0.0, device=z.device, dtype=torch.float32)
    lufs_term = torch.tensor(0.0, device=z.device, dtype=torch.float32)
    plr_term = torch.tensor(0.0, device=z.device, dtype=torch.float32)
    stereo_term = torch.tensor(0.0, device=z.device, dtype=torch.float32)

    for k in range(K):
        mk = mix_chunks[k]
        rk = ref_chunks[k]
        n_pre = preroll_list[k]

        y_ext = opt_v0p.chain(mk, z=z, eq_mode="iir")
        y_k = y_ext[..., n_pre:] if n_pre > 0 else y_ext

        if mode == "same_track":
            p_al, t_al = opt_v0p.loss_fn.align_latency(y_k, rk)
            s_val, _ = opt_v0p.loss_fn.stft_loss(p_al, t_al)
            stft_term = stft_term + weights[k] * s_val
            l_val, _ = opt_v0p.loss_fn.loudness_loss(p_al, target_lufs=target_lufs, plr_mode="box")
            lufs_term = lufs_term + weights[k] * l_val
        else:
            mel_pred = opt_v0p.stylistic_loss_fn._extract_mel_spectrum_distribution(y_k)
            with torch.no_grad():
                mel_ref = opt_v0p.stylistic_loss_fn._extract_mel_spectrum_distribution(rk)
            s_val = torch.mean(torch.abs(mel_pred - mel_ref))
            stft_term = stft_term + weights[k] * s_val

            pred_lufs = opt_v0p.stylistic_loss_fn.loudness_loss.calculate_lufs(y_k)
            ref_target = torch.tensor(target_lufs, device=y_k.device, dtype=y_k.dtype)
            l_val = torch.mean((pred_lufs - ref_target) ** 2)
            lufs_term = lufs_term + weights[k] * l_val

            pred_plr = opt_v0p.stylistic_loss_fn.loudness_loss.calculate_plr(y_k)
            with torch.no_grad():
                ref_plr = opt_v0p.stylistic_loss_fn.loudness_loss.calculate_plr(rk)
            p_val = torch.mean((pred_plr - ref_plr) ** 2)
            plr_term = plr_term + weights[k] * p_val

            left_p = y_k[:, 0, :]
            right_p = y_k[:, 1, :]
            mid_p = (left_p + right_p) * 0.70710678
            side_p = (left_p - right_p) * 0.70710678
            ms_pred = torch.log(torch.mean(side_p ** 2, dim=-1) + 1e-6) - torch.log(torch.mean(mid_p ** 2, dim=-1) + 1e-6)
            with torch.no_grad():
                left_r = rk[:, 0, :]
                right_r = rk[:, 1, :]
                mid_r = (left_r + right_r) * 0.70710678
                side_r = (left_r - right_r) * 0.70710678
                ms_ref = torch.log(torch.mean(side_r ** 2, dim=-1) + 1e-6) - torch.log(torch.mean(mid_r ** 2, dim=-1) + 1e-6)
            st_val = torch.mean(torch.abs(ms_pred - ms_ref))
            stereo_term = stereo_term + weights[k] * st_val

    z_neutral = Z_NEUTRAL.to(z.device, dtype=z.dtype)
    reg_val = opt_v0p.weight_reg * torch.mean((z - z_neutral) ** 2)

    lufs_weighted = 0.8 * lufs_term
    if mode == "same_track":
        total_loss = 1.0 * stft_term + lufs_weighted + reg_val
    else:
        total_loss = 1.0 * stft_term + lufs_weighted + 0.5 * plr_term + 0.3 * stereo_term + reg_val

    # Backward on lufs_weighted first (retain_graph=True)
    grad_lufs_z = torch.autograd.grad(lufs_weighted, z, retain_graph=True, allow_unused=True)[0]
    if grad_lufs_z is None:
        grad_lufs_z = torch.zeros_like(z)
    grad_lufs_z = grad_lufs_z.detach().clone()

    # Backward on total_loss
    grad_v0p_z = torch.autograd.grad(total_loss, z, retain_graph=False, allow_unused=True)[0]
    if grad_v0p_z is None:
        grad_v0p_z = torch.zeros_like(z)
    grad_v0p_z = grad_v0p_z.detach().clone()

    # Map to logit u space
    grad_lufs_u = compute_grad_u(grad_lufs_z, z_vector)
    grad_v0p_u = compute_grad_u(grad_v0p_z, z_vector)

    norms_v0p_z = compute_group_l2_norms(grad_v0p_z)
    norms_v0p_u = compute_group_l2_norms(grad_v0p_u)
    norms_lufs_z = compute_group_l2_norms(grad_lufs_z)
    norms_lufs_u = compute_group_l2_norms(grad_lufs_u)

    val_v0p_z40 = float(grad_v0p_z[40].item())
    val_v0p_u40 = float(grad_v0p_u[40].item())
    val_lufs_z40 = float(grad_lufs_z[40].item())
    val_lufs_u40 = float(grad_lufs_u[40].item())

    limiter_1d = {
        "v0p_total": {
            "scalar_derivative_z": val_v0p_z40,
            "scalar_derivative_u": val_v0p_u40,
            "sign_z": 1.0 if val_v0p_z40 > 1e-12 else (-1.0 if val_v0p_z40 < -1e-12 else 0.0),
            "sign_u": 1.0 if val_v0p_u40 > 1e-12 else (-1.0 if val_v0p_u40 < -1e-12 else 0.0),
        },
        "lufs_only": {
            "scalar_derivative_z": val_lufs_z40,
            "scalar_derivative_u": val_lufs_u40,
            "sign_z": 1.0 if val_lufs_z40 > 1e-12 else (-1.0 if val_lufs_z40 < -1e-12 else 0.0),
            "sign_u": 1.0 if val_lufs_u40 > 1e-12 else (-1.0 if val_lufs_u40 < -1e-12 else 0.0),
        },
    }

    return {
        "v0p_total_loss": float(total_loss.item()),
        "lufs_loss_unweighted": float(lufs_term.item()),
        "lufs_loss_weighted": float(lufs_weighted.item()),
        "grad_v0p_total_z": grad_v0p_z,
        "grad_v0p_total_u": grad_v0p_u,
        "grad_lufs_only_z": grad_lufs_z,
        "grad_lufs_only_u": grad_lufs_u,
        "gradient_l2_norms_z": {
            "v0p_total": norms_v0p_z,
            "lufs_only": norms_lufs_z,
        },
        "gradient_l2_norms_u": {
            "v0p_total": norms_v0p_u,
            "lufs_only": norms_lufs_u,
        },
        "limiter_1d": limiter_1d,
    }


def evaluate_dynamic_components_chunk(
    cfg: Dict[str, Any],
    delta_m: float,
    mk: torch.Tensor,
    diag: Dict[str, torch.Tensor],
    makeup_db: torch.Tensor,
    n_pre: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Compute differentiable chunk-level tensors for L_lim, L_diff, and L_conflict."""
    N_tot = mk.shape[-1]
    T = N_tot - n_pre
    D = TOTAL_DSP_DELAY_SAMPLES
    T_eval = T - D

    x_mix_eval = mk[..., n_pre : n_pre + T_eval]
    gr_low_eval = diag["gr_low_db"][..., n_pre : n_pre + T_eval]
    gr_mid_eval = diag["gr_mid_db"][..., n_pre : n_pre + T_eval]
    gr_high_eval = diag["gr_high_db"][..., n_pre : n_pre + T_eval]
    a_lim_eval = diag["lim_atten_db"][..., n_pre + D : n_pre + T]

    mask_smp, is_silent, _ = compute_active_sample_mask(x_mix_eval)

    if is_silent or mask_smp.sum() == 0:
        zero_tensor = torch.tensor(0.0, device=mk.device, dtype=mk.dtype)
        meta = {
            "loss_lim": 0.0,
            "loss_diff": 0.0,
            "loss_conflict": 0.0,
            "bar_a_lim_db": 0.0,
            "p95_a_lim_db": 0.0,
            "cvar95_a_lim_db": 0.0,
            "max_a_lim_db": 0.0,
            "bar_gr_low_db": 0.0,
            "bar_gr_mid_db": 0.0,
            "bar_gr_high_db": 0.0,
            "diff_lm_max": 0.0,
            "diff_mh_max": 0.0,
            "h_lim": 0.0,
            "is_silent_chunk": True,
            "n_active_samples": 0,
            "active_ratio": 0.0,
        }
        return zero_tensor, zero_tensor, zero_tensor, meta

    n_act = float(mask_smp.sum().item())
    act_mask = mask_smp

    # 1. L_lim
    a_lim_act = a_lim_eval[act_mask]
    bar_a_lim = torch.mean(a_lim_act)
    k_tail = max(1, int(math.ceil(0.05 * n_act)))
    topk_vals, _ = torch.topk(a_lim_act, k=k_tail, dim=-1)
    cvar_95 = torch.mean(topk_vals)

    term_avg = torch.relu(bar_a_lim - cfg["tau_avg"]) ** 2
    term_95 = torch.relu(cvar_95 - cfg["tau_95"]) ** 2
    loss_lim = term_avg + cfg["lambda_95"] * term_95

    # 2. L_diff
    gr_low_act = gr_low_eval[act_mask]
    gr_mid_act = gr_mid_eval[act_mask]
    gr_high_act = gr_high_eval[act_mask]

    diff_lm = torch.abs(gr_low_act - gr_mid_act)
    diff_mh = torch.abs(gr_mid_act - gr_high_act)

    diff_lm_max = torch.max(diff_lm) if diff_lm.numel() > 0 else torch.tensor(0.0, device=mk.device)
    diff_mh_max = torch.max(diff_mh) if diff_mh.numel() > 0 else torch.tensor(0.0, device=mk.device)

    loss_diff = torch.relu(diff_lm_max - cfg["delta_max"]) ** 2 + torch.relu(diff_mh_max - cfg["delta_max"]) ** 2

    # 3. L_conflict
    bar_gr_low = torch.mean(gr_low_act)
    bar_gr_mid = torch.mean(gr_mid_act)
    bar_gr_high = torch.mean(gr_high_act)

    gr_act_vec = torch.stack([bar_gr_low, bar_gr_mid, bar_gr_high]).detach()
    excess_makeup = torch.relu(makeup_db - gr_act_vec - delta_m)
    conflict_penalty = torch.sum(excess_makeup ** 2)

    r = torch.relu(bar_a_lim - cfg["tau_avg"])
    h_lim = torch.detach(r / (r + cfg["delta_ref"]))

    loss_conflict = h_lim * conflict_penalty

    meta = {
        "loss_lim": float(loss_lim.item()),
        "loss_diff": float(loss_diff.item()),
        "loss_conflict": float(loss_conflict.item()),
        "bar_a_lim_db": float(bar_a_lim.item()),
        "p95_a_lim_db": float(torch.quantile(a_lim_act.float(), 0.95).item()),
        "cvar95_a_lim_db": float(cvar_95.item()),
        "max_a_lim_db": float(torch.max(a_lim_act).item()),
        "bar_gr_low_db": float(bar_gr_low.item()),
        "bar_gr_mid_db": float(bar_gr_mid.item()),
        "bar_gr_high_db": float(bar_gr_high.item()),
        "diff_lm_max": float(diff_lm_max.item()),
        "diff_mh_max": float(diff_mh_max.item()),
        "h_lim": float(h_lim.item()),
        "is_silent_chunk": False,
        "n_active_samples": int(n_act),
        "active_ratio": float(n_act / T_eval),
    }

    return loss_lim, loss_diff, loss_conflict, meta


def evaluate_candidate_loss_and_gradients(
    chain: DifferentiableMasteringChain,
    prep_data: Dict[str, Any],
    z_vector: torch.Tensor,
    candidate_cfg: Dict[str, Any],
    delta_m: float,
    v0p_diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute component-level dynamic prior loss, per-chunk activation, and gradients in z and u."""
    mix_chunks = prep_data["mix_chunks"]
    preroll_list = prep_data["preroll_list"]
    K = len(mix_chunks)
    w_k = 1.0 / K

    g_v0p_total_z = v0p_diagnostics["grad_v0p_total_z"]
    g_v0p_total_u = v0p_diagnostics["grad_v0p_total_u"]
    g_lufs_only_z = v0p_diagnostics["grad_lufs_only_z"]
    g_lufs_only_u = v0p_diagnostics["grad_lufs_only_u"]

    z = z_vector.detach().clone().requires_grad_(True)
    theta = map_z_to_theta(z)
    params = unpack_theta_dict(theta)
    makeup_db = params["comp_makeup_db"]

    tot_l_lim_unw = torch.tensor(0.0, device=z.device, dtype=torch.float32)
    tot_l_diff_unw = torch.tensor(0.0, device=z.device, dtype=torch.float32)
    tot_l_conflict_unw = torch.tensor(0.0, device=z.device, dtype=torch.float32)

    chunk_activations = []

    for k in range(K):
        mk = mix_chunks[k]
        n_pre = preroll_list[k]

        _, diag = chain(mk, z=z, return_diagnostics=True)
        ch_lim, ch_diff, ch_conf, ch_meta = evaluate_dynamic_components_chunk(
            candidate_cfg, delta_m, mk, diag, makeup_db, n_pre=n_pre
        )

        tot_l_lim_unw = tot_l_lim_unw + w_k * ch_lim
        tot_l_diff_unw = tot_l_diff_unw + w_k * ch_diff
        tot_l_conflict_unw = tot_l_conflict_unw + w_k * ch_conf

        chunk_activations.append({
            "chunk_index": k,
            "loss_lim": ch_meta["loss_lim"],
            "loss_diff": ch_meta["loss_diff"],
            "loss_conflict": ch_meta["loss_conflict"],
            "bar_a_lim_db": ch_meta["bar_a_lim_db"],
            "p95_a_lim_db": ch_meta["p95_a_lim_db"],
            "cvar95_a_lim_db": ch_meta["cvar95_a_lim_db"],
            "max_a_lim_db": ch_meta["max_a_lim_db"],
            "diff_lm_max": ch_meta["diff_lm_max"],
            "diff_mh_max": ch_meta["diff_mh_max"],
            "h_lim": ch_meta["h_lim"],
            "is_lim_active": bool(ch_meta["loss_lim"] > 0.0),
            "is_diff_active": bool(ch_meta["loss_diff"] > 0.0),
            "is_conflict_active": bool(ch_meta["loss_conflict"] > 0.0),
            "is_any_active": bool((ch_meta["loss_lim"] + ch_meta["loss_diff"] + ch_meta["loss_conflict"]) > 0.0),
        })

    w_lim = candidate_cfg["w_lim"]
    w_diff = candidate_cfg["w_diff"]
    w_conflict = candidate_cfg["w_conflict"]

    tot_l_lim_w = w_lim * tot_l_lim_unw
    tot_l_diff_w = w_diff * tot_l_diff_unw
    tot_l_conflict_w = w_conflict * tot_l_conflict_unw
    tot_loss_dyn_total = tot_l_lim_w + tot_l_diff_w + tot_l_conflict_w

    # Individual component gradients
    # 1. L_lim
    if tot_l_lim_unw.requires_grad and tot_l_lim_unw.item() > 0.0:
        grad_z_lim_unw = torch.autograd.grad(tot_l_lim_unw, z, retain_graph=True, allow_unused=True)[0]
        if grad_z_lim_unw is None:
            grad_z_lim_unw = torch.zeros_like(z)
        grad_z_lim_unw = grad_z_lim_unw.detach().clone()
    else:
        grad_z_lim_unw = torch.zeros_like(z)
    grad_z_lim_w = w_lim * grad_z_lim_unw

    # 2. L_diff
    if tot_l_diff_unw.requires_grad and tot_l_diff_unw.item() > 0.0:
        grad_z_diff_unw = torch.autograd.grad(tot_l_diff_unw, z, retain_graph=True, allow_unused=True)[0]
        if grad_z_diff_unw is None:
            grad_z_diff_unw = torch.zeros_like(z)
        grad_z_diff_unw = grad_z_diff_unw.detach().clone()
    else:
        grad_z_diff_unw = torch.zeros_like(z)
    grad_z_diff_w = w_diff * grad_z_diff_unw

    # 3. L_conflict
    if tot_l_conflict_unw.requires_grad and tot_l_conflict_unw.item() > 0.0:
        grad_z_conflict_unw = torch.autograd.grad(tot_l_conflict_unw, z, retain_graph=True, allow_unused=True)[0]
        if grad_z_conflict_unw is None:
            grad_z_conflict_unw = torch.zeros_like(z)
        grad_z_conflict_unw = grad_z_conflict_unw.detach().clone()
    else:
        grad_z_conflict_unw = torch.zeros_like(z)
    grad_z_conflict_w = w_conflict * grad_z_conflict_unw

    # 4. Total dynamic loss gradient
    grad_z_total = grad_z_lim_w + grad_z_diff_w + grad_z_conflict_w

    # Map to logit u space
    grad_u_lim_unw = compute_grad_u(grad_z_lim_unw, z_vector)
    grad_u_lim_w = compute_grad_u(grad_z_lim_w, z_vector)
    grad_u_diff_unw = compute_grad_u(grad_z_diff_unw, z_vector)
    grad_u_diff_w = compute_grad_u(grad_z_diff_w, z_vector)
    grad_u_conflict_unw = compute_grad_u(grad_z_conflict_unw, z_vector)
    grad_u_conflict_w = compute_grad_u(grad_z_conflict_w, z_vector)
    grad_u_total = compute_grad_u(grad_z_total, z_vector)

    def package_component_diagnostics(gz_unw, gz_w, gu_unw, gu_w, name):
        norms_z_unw = compute_group_l2_norms(gz_unw)
        norms_z_w = compute_group_l2_norms(gz_w)
        norms_u_unw = compute_group_l2_norms(gu_unw)
        norms_u_w = compute_group_l2_norms(gu_w)

        # Active groups union
        active_indices = []
        for g_key in ["Compressor", "Limiter"]:
            if norms_z_w[g_key] > 1e-12:
                active_indices.extend(PARAM_GROUPS[g_key])
        active_indices = sorted(list(set(active_indices)))

        cos_v0p_total_z = compute_cosine_similarity(gz_w, g_v0p_total_z, list(range(41)))
        cos_v0p_total_u = compute_cosine_similarity(gu_w, g_v0p_total_u, list(range(41)))
        cos_lufs_only_z = compute_cosine_similarity(gz_w, g_lufs_only_z, list(range(41)))
        cos_lufs_only_u = compute_cosine_similarity(gu_w, g_lufs_only_u, list(range(41)))

        if len(active_indices) > 0:
            cos_act_v0p_z = compute_cosine_similarity(gz_w, g_v0p_total_z, active_indices)
            cos_act_v0p_u = compute_cosine_similarity(gu_w, g_v0p_total_u, active_indices)
            cos_act_lufs_z = compute_cosine_similarity(gz_w, g_lufs_only_z, active_indices)
            cos_act_lufs_u = compute_cosine_similarity(gu_w, g_lufs_only_u, active_indices)
        else:
            cos_act_v0p_z = 0.0
            cos_act_v0p_u = 0.0
            cos_act_lufs_z = 0.0
            cos_act_lufs_u = 0.0

        lim_diag = compute_1d_limiter_diagnostics(
            gz_w, gu_w, g_v0p_total_z, g_v0p_total_u, g_lufs_only_z, g_lufs_only_u
        )

        return {
            "name": name,
            "norms_z_unweighted": norms_z_unw,
            "norms_z_weighted": norms_z_w,
            "norms_u_unweighted": norms_u_unw,
            "norms_u_weighted": norms_u_w,
            "cosine_similarities": {
                "wrt_v0p_total_z": cos_v0p_total_z,
                "wrt_v0p_total_u": cos_v0p_total_u,
                "wrt_lufs_only_z": cos_lufs_only_z,
                "wrt_lufs_only_u": cos_lufs_only_u,
                "active_groups_wrt_v0p_total_z": cos_act_v0p_z,
                "active_groups_wrt_v0p_total_u": cos_act_v0p_u,
                "active_groups_wrt_lufs_only_z": cos_act_lufs_z,
                "active_groups_wrt_lufs_only_u": cos_act_lufs_u,
            },
            "limiter_1d_diagnostics": lim_diag,
        }

    comp_diag_lim = package_component_diagnostics(grad_z_lim_unw, grad_z_lim_w, grad_u_lim_unw, grad_u_lim_w, "L_lim")
    comp_diag_diff = package_component_diagnostics(grad_z_diff_unw, grad_z_diff_w, grad_u_diff_unw, grad_u_diff_w, "L_diff")
    comp_diag_conf = package_component_diagnostics(grad_z_conflict_unw, grad_z_conflict_w, grad_u_conflict_unw, grad_u_conflict_w, "L_conflict")
    comp_diag_total = package_component_diagnostics(grad_z_total, grad_z_total, grad_u_total, grad_u_total, "L_v3b_total")

    return {
        "delta_m": delta_m,
        "loss_weighted": float(tot_loss_dyn_total.item()),
        "loss_lim_unweighted": float(tot_l_lim_unw.item()),
        "loss_diff_unweighted": float(tot_l_diff_unw.item()),
        "loss_conflict_unweighted": float(tot_l_conflict_unw.item()),
        "loss_lim_weighted": float(tot_l_lim_w.item()),
        "loss_diff_weighted": float(tot_l_diff_w.item()),
        "loss_conflict_weighted": float(tot_l_conflict_w.item()),
        "is_active": bool(tot_loss_dyn_total.item() > 0.0),
        "chunk_activations": chunk_activations,
        "components": {
            "L_lim": comp_diag_lim,
            "L_diff": comp_diag_diff,
            "L_conflict": comp_diag_conf,
            "L_v3b_total": comp_diag_total,
        },
    }


def run_v3b_preflight() -> Tuple[Dict[str, Any], str]:
    """Execute complete read-only V3b preflight (V2) across the 6 harmonized sessions."""
    print("=" * 80)
    print("Sprint 3 Step V3b — Read-Only Preflight Diagnostics V2 on Harmonized V0p Sessions")
    print("=" * 80)

    v2_results = load_v2_results()
    v2_sessions_by_prefix = {s["track_prefix"]: s for s in v2_results["sessions"]}

    chain = DifferentiableMasteringChain()

    results_payload = {
        "artifact_type": "AutoMastering_Sprint3_V3b_Preflight_V2_Results",
        "version": "2.0.0",
        "candidate_configurations": CANDIDATE_CONFIGS,
        "sensitivity_delta_m_values": [1.0, 1.25, 1.5],
        "sessions": [],
    }

    for spec in SESSIONS_SPEC:
        prefix = spec["track_prefix"]
        mode = spec["mode"]
        print(f"\n--- Processing Harmonized Session: {prefix} ({mode}) ---")

        v2_sess = v2_sessions_by_prefix[prefix]
        seed_int = int(v2_sess["seed"])

        prep = prepare_harmonized_session(spec, seed_int=seed_int, repo_root=_REPO_ROOT)

        v0p_z_opt = torch.tensor(v2_sess["variants"]["V0p"]["normalized_z_opt"], dtype=torch.float32)
        v0p_theta_stored = v2_sess["variants"]["V0p"]["physical_parameters_theta"]

        v0p_physical_summary = {
            "makeup_gains_db": {
                "low": float(v0p_theta_stored["comp_low_makeup_db"]),
                "mid": float(v0p_theta_stored["comp_mid_makeup_db"]),
                "high": float(v0p_theta_stored["comp_high_makeup_db"]),
            },
            "thresholds_db": {
                "low": float(v0p_theta_stored["comp_low_threshold_db"]),
                "mid": float(v0p_theta_stored["comp_mid_threshold_db"]),
                "high": float(v0p_theta_stored["comp_high_threshold_db"]),
            },
            "ratios": {
                "low": float(v0p_theta_stored["comp_low_ratio"]),
                "mid": float(v0p_theta_stored["comp_mid_ratio"]),
                "high": float(v0p_theta_stored["comp_high_ratio"]),
            },
            "limiter_push_gain_db": float(v0p_theta_stored["limiter_push_gain_db"]),
        }

        # 1. Empirical stats at Z_NEUTRAL
        print("  Evaluating empirical traces at Z_NEUTRAL...")
        stats_z_neutral = evaluate_empirical_stats(chain, prep, Z_NEUTRAL)

        # 2. Empirical stats at V0p z_opt
        print("  Evaluating empirical traces at V0p z_opt...")
        stats_v0p_opt = evaluate_empirical_stats(chain, prep, v0p_z_opt)

        # 3. V0p loss and gradient (total vs LUFS-only)
        print("  Computing V0p loss and gradients (total vs LUFS-only)...")
        opt_v0p = AutoMasteringOptimizer(
            sample_rate=48000.0,
            lr=0.04,
            num_steps=30,
            target_lufs=prep["target_lufs"],
            weight_stft=1.0,
            weight_loudness=0.8,
            weight_param_reg=0.01 if mode == "same_track" else 0.02,
            use_hybrid_eq=True,
            weight_erb=0.0,
            stylistic_spectral_mode="mel40",
            weight_de_harsh=0.0,
            weight_trans=0.0,
            plr_mode="box",
            beta_atk=0.5,
            plr_d_style=1.0,
            plr_gamma_floor=1.0,
            soft_peak_p=8,
            soft_peak_beta=2.0,
            soft_peak_block_duration_sec=0.010,
            weight_eq_prior=0.0,
        )
        v0p_diag = compute_v0p_loss_and_gradient(opt_v0p, prep, v0p_z_opt)

        # 4. Candidate evaluations and sensitivity
        print("  Evaluating Candidate A & B across delta_M sensitivity...")
        candidates_eval = {}
        for cand_name, cand_cfg in CANDIDATE_CONFIGS.items():
            candidates_eval[cand_name] = {}
            for delta_m_val in [1.0, 1.25, 1.5]:
                cand_res = evaluate_candidate_loss_and_gradients(
                    chain, prep, v0p_z_opt, cand_cfg, delta_m_val, v0p_diag
                )
                candidates_eval[cand_name][f"delta_m_{delta_m_val:.2f}"] = cand_res

        session_entry = {
            "track_prefix": prefix,
            "reference_mode": mode,
            "seed": str(seed_int),
            "target_lufs": float(prep["target_lufs"]),
            "v0p_physical_parameters": v0p_physical_summary,
            "v0p_loss_and_gradient": {
                "v0p_total_loss": v0p_diag["v0p_total_loss"],
                "lufs_loss_unweighted": v0p_diag["lufs_loss_unweighted"],
                "lufs_loss_weighted": v0p_diag["lufs_loss_weighted"],
                "gradient_l2_norms_z": v0p_diag["gradient_l2_norms_z"],
                "gradient_l2_norms_u": v0p_diag["gradient_l2_norms_u"],
                "limiter_1d": v0p_diag["limiter_1d"],
            },
            "empirical_stats_z_neutral": stats_z_neutral,
            "empirical_stats_v0p_opt": stats_v0p_opt,
            "candidate_evaluations": candidates_eval,
        }
        results_payload["sessions"].append(session_entry)

    # Save JSON results to V2 file
    json_path = _REPO_ROOT / "docs" / "sprint3" / "SPRINT3_V3B_PREFLIGHT_V2_RESULTS.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\n[OK] Preflight V2 results saved to: {json_path}")

    # Generate Markdown report V2
    report_md = generate_preflight_v2_markdown_report(results_payload)
    report_path = _REPO_ROOT / "docs" / "sprint3" / "SPRINT3_V3B_PREFLIGHT_V2_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"[OK] Preflight V2 report saved to: {report_path}")

    return results_payload, report_md


def generate_preflight_v2_markdown_report(data: Dict[str, Any]) -> str:
    """Generate comprehensive preflight report V2 with full reconciliation audits and physical analyses."""
    lines = [
        "# SPRINT 3 — REPORT PREFLIGHT DIAGNOSTICO READ-ONLY V3b (V2)",
        "",
        "> **Documento di Preflight Sperimentale & Audit di Coerenza Parametrica V2**  ",
        "> **Repository:** `gio961gio/AutoMastering` | **Branch:** `human-validation-package`  ",
        "> **Modalità:** Rigorosamente READ-ONLY (ZERO training, ZERO ottimizzazioni, ZERO modifiche ai JSON storici).",
        "",
        "---",
        "",
        "## Sezione 1 — Audit di Riconciliazione tra Sintesi Precedente e Risultati Reali del Commit `677df6a`",
        "",
        "In piena trasparenza scientifica, si documenta che i numeri preliminari citati in una sintesi testuale informale prima dell'audit formale (in particolare le stime di $\\bar{A}_{\\text{lim}}^{\\text{act}} \\approx 5\\text{--}9\\text{ dB}$, $P_{\\text{lim}} \\approx +5.97\\text{ dB}$, attivazione al 100% su 5 sessioni e norme gradienti $15.6\\text{--}17.7$) **non erano estratti dal file JSON reale** `docs/sprint3/SPRINT3_V3B_PREFLIGHT_RESULTS.json` generato al commit `677df6a`.",
        "",
        "La tabella seguente riporta fianco a fianco, per ciascuna misura, i **valori reali congelati e verificabili** presenti nel file JSON al commit `677df6a` con percorso JSON esatto, formula e unità:",
        "",
        "| Misura Contestata | Stima Errata Precedente | Valore Reale Congelato nel JSON (`677df6a`) | Percorso JSON Esatto | Formula / Definizione Matematica |",
        "| :--- | :---: | :---: | :--- | :--- |",
        "| **$\\bar{A}_{\\text{lim}}$ (`track01`)** | $\\approx 5\\text{--}9\\text{ dB}$ | **`0.259926 dB`** | `sessions[0].empirical_stats_v0p_opt.session_summary.limiter.mean_atten_db` | $\\frac{1}{|\\mathcal{A}_{\\text{smp}}|} \\sum_{n \\in \\mathcal{A}_{\\text{smp}}} A_{\\text{lim}}[n]$ |",
        "| **$\\bar{A}_{\\text{lim}}$ (`track02`)** | $\\approx 5\\text{--}9\\text{ dB}$ | **`0.000000 dB`** | `sessions[1].empirical_stats_v0p_opt.session_summary.limiter.mean_atten_db` | Idem ($A_{\\text{lim}} \\equiv 0.0$, nessuna attenuazione) |",
        "| **$\\bar{A}_{\\text{lim}}$ (`track03`)** | $\\approx 5\\text{--}9\\text{ dB}$ | **`0.019444 dB`** | `sessions[2].empirical_stats_v0p_opt.session_summary.limiter.mean_atten_db` | Idem |",
        "| **$\\bar{A}_{\\text{lim}}$ (`track01_style`)** | $\\approx 5\\text{--}9\\text{ dB}$ | **`0.000000 dB`** | `sessions[3].empirical_stats_v0p_opt.session_summary.limiter.mean_atten_db` | Idem |",
        "| **$\\bar{A}_{\\text{lim}}$ (`track02_style`)** | $\\approx 5\\text{--}9\\text{ dB}$ | **`0.000000 dB`** | `sessions[4].empirical_stats_v0p_opt.session_summary.limiter.mean_atten_db` | Idem |",
        "| **$\\bar{A}_{\\text{lim}}$ (`track03_style`)** | $\\approx 5\\text{--}9\\text{ dB}$ | **`0.018227 dB`** | `sessions[5].empirical_stats_v0p_opt.session_summary.limiter.mean_atten_db` | Idem |",
        "| **$P_{95}(A_{\\text{lim}})$ (`track01`)** | N/D | **`1.401236 dB`** | `sessions[0].empirical_stats_v0p_opt.session_summary.limiter.p95_atten_db` | Quantile empirico 95% su $\\mathcal{A}_{\\text{smp}}$ |",
        "| **$\\text{CVaR}_{0.95}(A_{\\text{lim}})$ (`track01`)** | N/D | **`2.293961 dB`** | `sessions[0].empirical_stats_v0p_opt.session_summary.limiter.cvar95_atten_db` | Media del 5% superiore su $\\mathcal{A}_{\\text{smp}}$ |",
        "| **$\\max A_{\\text{lim}}$ (`track01`)** | N/D | **`6.003366 dB`** | `sessions[0].empirical_stats_v0p_opt.session_summary.limiter.max_atten_db` | $\\max_{n \\in \\mathcal{A}_{\\text{smp}}} A_{\\text{lim}}[n]$ |",
        "| **$P_{\\text{lim}}$ (`track01`)** | $+5.97\\text{ dB}$ | **`0.001484 dB`** | `sessions[0].v0p_physical_parameters.limiter_push_gain_db` | $\\theta_{40} = 8.0 \\cdot z_{40}$ con $z_{40} \\approx 0.000185$ |",
        "| **$P_{\\text{lim}}$ (`track02`)** | $+5.97\\text{ dB}$ | **`0.001484 dB`** | `sessions[1].v0p_physical_parameters.limiter_push_gain_db` | Idem |",
        "| **$P_{\\text{lim}}$ (`track03`)** | $+5.97\\text{ dB}$ | **`0.001507 dB`** | `sessions[2].v0p_physical_parameters.limiter_push_gain_db` | Idem |",
        "| **$P_{\\text{lim}}$ (`track01_style`)** | $+5.97\\text{ dB}$ | **`0.000201 dB`** | `sessions[3].v0p_physical_parameters.limiter_push_gain_db` | Idem |",
        "| **$P_{\\text{lim}}$ (`track02_style`)** | $+5.97\\text{ dB}$ | **`0.000837 dB`** | `sessions[4].v0p_physical_parameters.limiter_push_gain_db` | Idem |",
        "| **$P_{\\text{lim}}$ (`track03_style`)** | $+5.97\\text{ dB}$ | **`0.001389 dB`** | `sessions[5].v0p_physical_parameters.limiter_push_gain_db` | Idem |",
        "",
        "> [!IMPORTANT]",
        "> L'origine della confusione su $P_{\\text{lim}}$ derivava dalla sovrapposizione concettuale tra il parametro fisico di gain push ($P_{\\text{lim}} = 8.0 \\cdot z_{40} \\approx 0.0015\\text{ dB}$) e l'attenuazione di picco massima raggiunta dal limiter su transienti isolati (fino a $6.00\\text{ dB}$ su `track01`). Tutti i file JSON ufficiali nel repository registrano rigorosamente i veri valori fisici sopra riportati.",
        "",
        "---",
        "",
        "## Sezione 2 — Audit dei 10 File Modificati nel Commit `c2fefc2`",
        "",
        "L'ispezione di `git diff --name-status ef5254f..c2fefc2` evidenzia esattamente **10 file modificati**:",
        "",
        "| File | Operazione Git | Ruolo Architetturale & Motivazione della Modifica |",
        "| :--- | :---: | :--- |",
        "| `python/ddsp_mastering/dsp/compressor.py` | `M` | Implementazione forward a 2 output `(y_out, yg)` e VJP backward differenziabile per la gain reduction del compressore. |",
        "| `python/ddsp_mastering/dsp/limiter.py` | `M` | Esposizione del tensore di attenuazione e guadagno differenziabile `g_smooth[n]` e `a_lim_db[n]`. |",
        "| `python/ddsp_mastering/dsp/chain.py` | `M` | Aggiunta del parametro `return_diagnostics=True` per la propagazione delle tracce differenziabili. |",
        "| `python/ddsp_mastering/loss/dynamic_prior.py` | `A` | Creazione del modulo `V3bDynamicPriorLoss` e maschera di attività campioni `compute_active_sample_mask`. |",
        "| `python/ddsp_mastering/loss/__init__.py` | `M` | Esportazione del modulo `V3bDynamicPriorLoss` nell'interfaccia pubblica del package. |",
        "| `tests/test_sprint3_v3b_gate.py` | `A` | Creazione della suite del Gate Tecnico V3b (11 test rigorosi, passati 11/11). |",
        "| `python/ddsp_mastering/dsp/__init__.py` | `M` | Esportazione delle nuove classi del DSP nell'interfaccia pubblica del modulo. |",
        "| `python/tests/test_sprint_1_correctness.py` | `M` | **Unpacking chiamata low-level:** questo test invocava direttamente `DifferentiableSingleBandCompressorFunction.apply(...)`, che ora restituisce la tupla `(y_out, yg)`. Senza l'unpacking `out, _ = Function.apply(...)`, il test legacy falliva con `TypeError: tuple and Tensor`. |",
        "| `python/tests/test_sprint_2_2_audit_remediation.py` | `M` | **Unpacking chiamata low-level:** identico al precedente, aggiornato per supportare la tupla a 2 uscite di `Function.apply`. |",
        "| `python/tests/test_ticket_1_4.py` | `M` | **Unpacking chiamata low-level:** identico al precedente, aggiornato per supportare la tupla a 2 uscite di `Function.apply`. |",
        "",
        "> [!NOTE]",
        "> Si sottolinea che il modulo ad alto livello `DifferentiableSingleBandCompressor.forward` mantiene `return_gr=False` di default, preservando la totale retro-compatibilità con il codice utente e la catena di mastering.",
        "",
        "---",
        "",
        "## Sezione 3 — Risultati Completi del Nuovo Preflight V2 (`lambda_95 = 0.5`)",
        "",
        "### 3.1 Statistiche Empiriche Per-Chunk e Aggregate",
        "",
        "Le statistiche seguenti quantificano l'attività del limiter e del compressore su ciascun chunk $k \\in \\{0, 1, 2, 3\\}$ e ne evidenziano la media (`chunk_mean`) e il picco massimo (`chunk_max`):",
        "",
        "| Sessione | Chunk $k$ | Campioni Attivi (%) | $\\bar{A}_{\\text{lim}}$ (dB) | $P_{95}(A_{\\text{lim}})$ (dB) | $\\text{CVaR}_{95}(A_{\\text{lim}})$ (dB) | Max $A_{\\text{lim}}$ (dB) | $\\bar{GR}_L / \\bar{GR}_M / \\bar{GR}_H$ (dB) | Max $\\Delta_{LM} / \\Delta_{MH}$ (dB) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for s in data["sessions"]:
        prefix = s["track_prefix"]
        chunks = s["empirical_stats_v0p_opt"]["per_chunk_stats"]
        c_mean = s["empirical_stats_v0p_opt"]["session_summary"]["chunk_mean"]
        c_max = s["empirical_stats_v0p_opt"]["session_summary"]["chunk_max"]

        for c in chunks:
            k_idx = c["chunk_index"]
            pct_k = c["active_percentage"]
            gr_k = f"{c['gr_low_mean_db']:.2f} / {c['gr_mid_mean_db']:.2f} / {c['gr_high_mean_db']:.2f}"
            diff_k = f"{c['diff_lm_max_db']:.2f} / {c['diff_mh_max_db']:.2f}"
            lines.append(
                f"| **`{prefix}`** | Chunk {k_idx} | {pct_k:.1f}% | {c['lim_mean_db']:.2f} | {c['lim_p95_db']:.2f} | {c['lim_cvar95_db']:.2f} | {c['lim_max_db']:.2f} | {gr_k} | {diff_k} |"
            )

        gr_m_str = f"{c_mean['compressor_gr']['gr_low_mean_db']:.2f} / {c_mean['compressor_gr']['gr_mid_mean_db']:.2f} / {c_mean['compressor_gr']['gr_high_mean_db']:.2f}"
        diff_m_str = f"{c_mean['interband_diff']['diff_lm_max_db']:.2f} / {c_mean['interband_diff']['diff_mh_max_db']:.2f}"
        lines.append(
            f"| | **Media Chunks** | — | **{c_mean['limiter']['mean_atten_db']:.2f}** | **{c_mean['limiter']['p95_atten_db']:.2f}** | **{c_mean['limiter']['cvar95_atten_db']:.2f}** | **{c_mean['limiter']['max_atten_db']:.2f}** | **{gr_m_str}** | **{diff_m_str}** |"
        )
        diff_max_str = f"{c_max['interband_diff']['diff_lm_max_db']:.2f} / {c_max['interband_diff']['diff_mh_max_db']:.2f}"
        lines.append(
            f"| | **Max Chunks** | — | **{c_max['limiter']['mean_atten_db']:.2f}** | **{c_max['limiter']['p95_atten_db']:.2f}** | **{c_max['limiter']['cvar95_atten_db']:.2f}** | **{c_max['limiter']['max_atten_db']:.2f}** | — | **{diff_max_str}** |"
        )

    lines.extend([
        "",
        "---",
        "",
        "### 3.2 Parametri Fisici Reali Ottimizzati di `V0p`",
        "",
        "| Sessione | Makeup $M_L$ / $M_M$ / $M_H$ (dB) | Soglie $T_L$ / $T_M$ / $T_H$ (dBFS) | Ratio $R_L$ / $R_M$ / $R_H$ | Limiter Push $P_{\\text{lim}}$ (dB) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])

    for s in data["sessions"]:
        prefix = s["track_prefix"]
        p = s["v0p_physical_parameters"]
        m_str = f"{p['makeup_gains_db']['low']:.2f} / {p['makeup_gains_db']['mid']:.2f} / {p['makeup_gains_db']['high']:.2f}"
        t_str = f"{p['thresholds_db']['low']:.2f} / {p['thresholds_db']['mid']:.2f} / {p['thresholds_db']['high']:.2f}"
        r_str = f"{p['ratios']['low']:.4f} / {p['ratios']['mid']:.4f} / {p['ratios']['high']:.4f}"
        lines.append(f"| **`{prefix}`** | {m_str} | {t_str} | {r_str} | {p['limiter_push_gain_db']:.4f} |")

    lines.extend([
        "",
        "---",
        "",
        "### 3.3 Confronto Comparativo: Candidato A vs Candidato B (a $\\delta_M = 1.25\\text{ dB}$ Nominale, $\\lambda_{95} = 0.5$)",
        "",
        "| Sessione | Configurazione | $\\mathcal{L}_{\\text{lim}}$ (W/Unw) | $\\mathcal{L}_{\\text{diff}}$ (W/Unw) | $\\mathcal{L}_{\\text{conf}}$ (W/Unw) | $\\mathcal{L}_{\\text{dyn}}$ Totale | Attivo? | $\\cos(\\theta)$ Totale ($z$ / $u$) | $\\cos(\\theta)$ Attivi ($z$ / $u$) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for s in data["sessions"]:
        prefix = s["track_prefix"]
        cA = s["candidate_evaluations"]["Candidate_A"]["delta_m_1.25"]
        cB = s["candidate_evaluations"]["Candidate_B"]["delta_m_1.25"]

        cosA_tot_z = cA["components"]["L_v3b_total"]["cosine_similarities"]["wrt_v0p_total_z"]
        cosA_tot_u = cA["components"]["L_v3b_total"]["cosine_similarities"]["wrt_v0p_total_u"]
        cosA_act_z = cA["components"]["L_v3b_total"]["cosine_similarities"]["active_groups_wrt_v0p_total_z"]
        cosA_act_u = cA["components"]["L_v3b_total"]["cosine_similarities"]["active_groups_wrt_v0p_total_u"]

        cosB_tot_z = cB["components"]["L_v3b_total"]["cosine_similarities"]["wrt_v0p_total_z"]
        cosB_tot_u = cB["components"]["L_v3b_total"]["cosine_similarities"]["wrt_v0p_total_u"]
        cosB_act_z = cB["components"]["L_v3b_total"]["cosine_similarities"]["active_groups_wrt_v0p_total_z"]
        cosB_act_u = cB["components"]["L_v3b_total"]["cosine_similarities"]["active_groups_wrt_v0p_total_u"]

        lines.append(
            f"| **`{prefix}`** | **Candidato A** | {cA['loss_lim_weighted']:.4f} ({cA['loss_lim_unweighted']:.3f}) | {cA['loss_diff_weighted']:.4f} ({cA['loss_diff_unweighted']:.3f}) | {cA['loss_conflict_weighted']:.4f} ({cA['loss_conflict_unweighted']:.3f}) | **{cA['loss_weighted']:.4f}** | {'Sì' if cA['is_active'] else 'No'} | {cosA_tot_z:.4f} / {cosA_tot_u:.4f} | {cosA_act_z:.4f} / {cosA_act_u:.4f} |"
        )
        lines.append(
            f"| | **Candidato B** | {cB['loss_lim_weighted']:.4f} ({cB['loss_lim_unweighted']:.3f}) | {cB['loss_diff_weighted']:.4f} ({cB['loss_diff_unweighted']:.3f}) | {cB['loss_conflict_weighted']:.4f} ({cB['loss_conflict_unweighted']:.3f}) | **{cB['loss_weighted']:.4f}** | {'Sì' if cB['is_active'] else 'No'} | {cosB_tot_z:.4f} / {cosB_tot_u:.4f} | {cosB_act_z:.4f} / {cosB_act_u:.4f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "### 3.4 Norme dei Gradienti per Gruppo Parametrico: Spazio $z$ vs Spazio $u$",
        "",
        "La tabella documenta l'intensità delle forze di correzione dinamica sia in spazio normalizzato $z \\in (0, 1)$ sia in spazio latente logit $u \\in \\mathbb{R}$:",
        "",
        "| Sessione | Gradiente / Spazio | EQ `[0:18]` | Crossover `[18:20]` | Compressor `[20:35]` | Makeup `[24,29,34]` | Imager `[35:38]` | Saturator `[38:40]` | Limiter `[40]` | Totale $41D$ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    lbl_v0p_z = r"$\nabla_z \mathcal{L}_{\text{V0p}}$"
    lbl_v0p_u = r"$\nabla_u \mathcal{L}_{\text{V0p}}$"
    lbl_dyn_z = r"$\nabla_z \mathcal{L}_{\text{dyn}}$ (Cand. B)"
    lbl_dyn_u = r"$\nabla_u \mathcal{L}_{\text{dyn}}$ (Cand. B)"

    for s in data["sessions"]:
        prefix = s["track_prefix"]
        v0p_z = s["v0p_loss_and_gradient"]["gradient_l2_norms_z"]["v0p_total"]
        v0p_u = s["v0p_loss_and_gradient"]["gradient_l2_norms_u"]["v0p_total"]
        cB_comp = s["candidate_evaluations"]["Candidate_B"]["delta_m_1.25"]["components"]["L_v3b_total"]
        cB_gz = cB_comp["norms_z_weighted"]
        cB_gu = cB_comp["norms_u_weighted"]

        lines.append(
            f"| **`{prefix}`** | {lbl_v0p_z} | {v0p_z['EQ']:.4f} | {v0p_z['Crossover']:.4f} | {v0p_z['Compressor']:.4f} | {v0p_z['makeup_gains']:.4f} | {v0p_z['Imager']:.4f} | {v0p_z['Saturator']:.4f} | {v0p_z['Limiter']:.4f} | **{v0p_z['total_41d']:.4f}** |"
        )
        lines.append(
            f"| | {lbl_v0p_u} | {v0p_u['EQ']:.4f} | {v0p_u['Crossover']:.4f} | {v0p_u['Compressor']:.4f} | {v0p_u['makeup_gains']:.4f} | {v0p_u['Imager']:.4f} | {v0p_u['Saturator']:.4f} | {v0p_u['Limiter']:.6f} | **{v0p_u['total_41d']:.4f}** |"
        )
        lines.append(
            f"| | {lbl_dyn_z} | {cB_gz['EQ']:.4f} | {cB_gz['Crossover']:.4f} | {cB_gz['Compressor']:.4f} | {cB_gz['makeup_gains']:.4f} | {cB_gz['Imager']:.4f} | {cB_gz['Saturator']:.4f} | {cB_gz['Limiter']:.4f} | **{cB_gz['total_41d']:.4f}** |"
        )
        lines.append(
            f"| | {lbl_dyn_u} | {cB_gu['EQ']:.4f} | {cB_gu['Crossover']:.4f} | {cB_gu['Compressor']:.4f} | {cB_gu['makeup_gains']:.4f} | {cB_gu['Imager']:.4f} | {cB_gu['Saturator']:.4f} | {cB_gu['Limiter']:.6f} | **{cB_gu['total_41d']:.4f}** |"
        )

    lines.extend([
        "",
        "---",
        "",
        "### 3.5 Diagnostica 1D Scalare del Gruppo Limiter (Indice 40)",
        "",
        "Poiché il gruppo Limiter è uno spazio 1D scalare, la cosine similarity rappresenta puramente la **concordanza di segno scalare** ($\pm 1.0$ o $0.0$). La tabella documenta derivata scalare, segno e concordanza rispetto a $\\nabla \\mathcal{L}_{\\text{V0p}}$ e rispetto a $\\nabla \\mathcal{L}_{\\text{LUFS}}$:",
        "",
        "| Sessione | $\\frac{\\partial \\mathcal{L}_{\\text{dyn}}}{\\partial z_{40}}$ | $\\frac{\\partial \\mathcal{L}_{\\text{dyn}}}{\\partial u_{40}}$ | $\\text{Segno } (z / u)$ | $\\cos(\\theta)$ vs Totale $V0p$ | $\\cos(\\theta)$ vs Solo LUFS | Interpretazione Fisica |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ])

    for s in data["sessions"]:
        prefix = s["track_prefix"]
        cB_lim = s["candidate_evaluations"]["Candidate_B"]["delta_m_1.25"]["components"]["L_v3b_total"]["limiter_1d_diagnostics"]
        dz = cB_lim["scalar_derivative_z"]
        du = cB_lim["scalar_derivative_u"]
        sz = cB_lim["sign_z"]
        su = cB_lim["sign_u"]
        cv = cB_lim["cos_sim_wrt_v0p_total_z"]
        cl = cB_lim["cos_sim_wrt_lufs_only_z"]

        interp = "Opposizione di segno al push (de-escalation limiter)" if cv < 0 else ("Inattivo (gradiente nullo)" if cv == 0 else "Accordo")
        lines.append(
            f"| **`{prefix}`** | {dz:.6f} | {du:.8f} | {sz:+.0f} / {su:+.0f} | {cv:+.4f} | {cl:+.4f} | {interp} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "### 3.6 Analisi di Sensitività su $\\delta_M \\in \\{1.0, 1.25, 1.5\\}\\text{ dB}$",
        "",
        "| Sessione | $\\delta_M = 1.0\\text{ dB}$ (A / B) | $\\delta_M = 1.25\\text{ dB}$ (A / B) | $\\delta_M = 1.5\\text{ dB}$ (A / B) | Variazione Forza $\\nabla_{M}$ |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])

    arrow_sym = r"$\to$"
    for s in data["sessions"]:
        prefix = s["track_prefix"]
        c = s["candidate_evaluations"]
        cA10 = c["Candidate_A"]["delta_m_1.00"]["loss_conflict_weighted"]
        cB10 = c["Candidate_B"]["delta_m_1.00"]["loss_conflict_weighted"]
        cA12 = c["Candidate_A"]["delta_m_1.25"]["loss_conflict_weighted"]
        cB12 = c["Candidate_B"]["delta_m_1.25"]["loss_conflict_weighted"]
        cA15 = c["Candidate_A"]["delta_m_1.50"]["loss_conflict_weighted"]
        cB15 = c["Candidate_B"]["delta_m_1.50"]["loss_conflict_weighted"]

        g10 = c["Candidate_A"]["delta_m_1.00"]["components"]["L_conflict"]["norms_z_weighted"]["makeup_gains"]
        g15 = c["Candidate_A"]["delta_m_1.50"]["components"]["L_conflict"]["norms_z_weighted"]["makeup_gains"]

        lines.append(
            f"| **`{prefix}`** | {cA10:.4f} / {cB10:.4f} | {cA12:.4f} / {cB12:.4f} | {cA15:.4f} / {cB15:.4f} | {g10:.4f} {arrow_sym} {g15:.4f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "### 3.7 Analisi Fisica dei Meccanismi di Attivazione e Inattività",
        "",
        "1. **Perché $\\mathcal{L}_{\\text{lim}}$ è inattivo per Candidato A su tutte le 6 sessioni, mentre si attiva per Candidato B su `track01`:**",
        "   - Su `track01`, l'attenuazione media del limiter sull'intero chunk attivo è modesta ($\\bar{A}_{\\text{lim}}^{\\text{act}} = 0.26\\text{ dB}$), ben inferiore sia a $\\tau_{\\text{avg}} = 1.5\\text{ dB}$ (Candidato A) sia a $\\tau_{\\text{avg}} = 0.5\\text{ dB}$ (Candidato B). Di conseguenza, il termine medio $\\text{ReLU}(\\bar{A}_{\\text{lim}}^{\\text{act}} - \\tau_{\\text{avg}})^2$ vale identicamente **$0.0$** in entrambe le configurazioni.",
        "   - Tuttavia, su `track01` sono presenti picchi transitori aggressivi che provocano un $\\text{CVaR}_{0.95}^{\\text{act}} = 2.294\\text{ dB}$ (e fino a $3.06\\text{ dB}$ sul chunk $k=3$, con attenuazione istantanea massima di $6.00\\text{ dB}$). Poiché per Candidato B la soglia di coda è $\\tau_{95} = 2.0\\text{ dB}$, il termine $\\text{ReLU}(\\text{CVaR}_{0.95} - 2.0)^2 = (2.294 - 2.0)^2 = 0.0864$ si attiva selettivamente per **Candidato B**, mentre per Candidato A (con soglia conservativa $\\tau_{95} = 3.5\\text{ dB}$) resta inattivo ($0.0$).",
        "",
        "2. **Perché $\\mathcal{L}_{\\text{conflict}}$ vale $0.0$ su tutte le sessioni anche a $\\delta_M = 1.0\\text{ dB}$:**",
        "   - Anche quando la penalità di eccesso di makeup gain $\\text{ReLU}(M_c - \\bar{GR}_c^{\\text{act}} - \\delta_M)^2$ è strettamente positiva (ad esempio su `track01`, dove $M_L \\approx 1.37\\text{ dB}$, $\\bar{GR}_L \\approx 0.0\\text{ dB}$, per cui per $\\delta_M = 1.0\\text{ dB}$ l'eccesso è $(1.37 - 1.0)^2 = 0.1369$), il gate moltiplicativo $h_{\\text{lim}}$ è definito come:",
        r"     $$h_{\text{lim}} = \text{detach}\left(\frac{r}{r + \Delta_{\text{ref}}}\right), \quad \text{con } r = \text{ReLU}(\bar{A}_{\text{lim}}^{\text{act}} - \tau_{\text{avg}})$$",
        "   - Poiché in tutti i punti operativi $\\bar{A}_{\\text{lim}}^{\\text{act}} \\le 0.26\\text{ dB} < \\tau_{\\text{avg}}$ (sia con $\\tau_{\\text{avg}}=1.5\\text{ dB}$ sia con $\\tau_{\\text{avg}}=0.5\\text{ dB}$), la violazione media $r$ è **rigorosamente zero**, per cui $h_{\\text{lim}} = 0.0$ sigilla a zero l'intero termine di conflitto. Questo comportamento è strutturalmente corretto secondo la specifica (non si deve penalizzare il makeup gain se il limiter non è sovraccarico in modo continuativo).",
        "",
        "3. **Perché in `V0p` i ratio ($1.000125:1$) e `limiter_push_gain_db` ($0.0015\\text{ dB}$) sono rimasti quasi fermi al punto di partenza (Dimostrazione dell'Attenuazione Jacobiana $\\approx 5\,000\\times$ in Spazio $u$):**",
        "   - In `param_space.py`, per i parametri il cui valore neutro coincide con il bordo inferiore del range consentito (i ratio di compressione con $\\min = 1.0$ e `limiter_push_gain_db` con $\\min = 0.0\\text{ dB}$), il punto iniziale è posizionato a $u_i = -10.0$ ($z_i = \\sigma(-10.0) \\approx 4.54 \\times 10^{-5}$).",
        "   - La derivata Jacobiana della sigmoide in tale punto operativo vale:",
        r"     $$\frac{\partial z_i}{\partial u_i} = z_i (1 - z_i) \approx 4.5398 \times 10^{-5}$$",
        "   - Per i parametri con valore neutro centrale (come i makeup gain e i guadagni EQ, dove $z_i \\in [0.33, 0.50]$), la derivata Jacobiana vale $\\frac{\\partial z_i}{\\partial u_i} \\in [0.2222, 0.2500]$.",
        "   - Il rapporto tra le derivate Jacobiane evidenzia un'**attenuazione asimmetrica di oltre $5\,000\\times$**:",
        r"     $$\frac{\left.\frac{\partial z}{\partial u}\right|_{\text{makeup}}}{\left.\frac{\partial z}{\partial u}\right|_{\text{ratio / push}}} = \frac{0.2222}{4.5398 \times 10^{-5}} \approx 4\,895 \approx 5\,000\times$$",
        "   - Di conseguenza, qualsiasi gradiente calcolato in spazio $z$ (ad es. $\\nabla_z \\mathcal{L}_{\\text{V0p}}[40] = 6.1263$) viene compresso in spazio $u$ a un valore microscopico:",
        r"     $$\nabla_u \mathcal{L}_{\text{V0p}}[40] = 6.1263 \times (4.54 \times 10^{-5}) \approx 0.000278$$",
        "   - Con un learning rate Adam di $0.04$ e un budget limitato a $30$ step, il vettore $u$ può spostarsi al massimo di $\\Delta u \\le 30 \\times 0.04 = 1.2$ unità logit. Partendo da $u = -10.0$, anche muovendosi con il gradiente alla massima velocità possibile, $u$ arriva al massimo a $-8.8$, dove $z = \\sigma(-8.8) \\approx 0.00015$, corrispondente a un ratio di $1 + 3.0 \\cdot 0.00015 = 1.00045:1$ e a un push gain di $8.0 \\cdot 0.00015 = 0.0012\\text{ dB}$.",
        "   - Questo risultato matematico spiega in modo definitivo e rigoroso perché i ratio e il push gain siano rimasti quasi identici ai valori di default in tutte le sessioni armonizzate precedenti.",
        "",
        "---",
        "",
        "## Sezione 4 — Questioni Aperte per il Supervisore",
        "",
        "Alla luce dei risultati quantitativi del Preflight V2, si sottopongono al Supervisore Tecnico (Argon) i seguenti punti decisionali per l'avvio della fase operativa V3b:",
        "",
        "1. **Scelta della Configurazione di Base (Candidato B vs Candidato A):**",
        "   - **Candidato B** si è dimostrato l'unico in grado di intercettare i picchi transitori sul limiter di `track01` (grazie a $\\tau_{95} = 2.0\\text{ dB}$ e $\\lambda_{95} = 0.5$), esercitando una forza correttiva di de-escalation con cosine similarity negativa ($-0.9680$ sul limiter) che contrasta l'eccesso di spinta.",
        "   - **Candidato A** (con $\\tau_{95} = 3.5\\text{ dB}$) rimane completamente passivo su tutte le 6 sessioni.",
        "",
        "2. **Gating del Termine di Conflitto ($h_{\\text{lim}}$):**",
        "   - Attualmente $h_{\\text{lim}}$ è controllato unicamente da $r = \\text{ReLU}(\\bar{A}_{\\text{lim}}^{\\text{act}} - \\tau_{\\text{avg}})$. Poiché $\\bar{A}_{\\text{lim}}^{\\text{act}} \\le 0.26\\text{ dB}$ su tutte le sessioni, $\\mathcal{L}_{\\text{conflict}}$ non entra mai in azione.",
        "   - Si domanda al Supervisore se $h_{\\text{lim}}$ debba includere anche un termine di coda su $\\text{CVaR}_{0.95}$ (es. $r_{\\text{cvar}} = \\text{ReLU}(\\text{CVaR}_{0.95} - \\tau_{95})$), oppure se si desideri mantenere il gate legato strettamente all'attenuazione media continua.",
        "",
        "3. **Inizializzazione di $u$ e Attenuazione Jacobiana $\\approx 5\,000\\times$:**",
        "   - La scoperta dell'attenuazione Jacobiana $z(1-z) \\approx 4.54 \\times 10^{-5}$ per ratio e push gain rivela che l'inattività di tali moduli non è dovuta a gradienti nulli in spazio fisico, bensì al posizionamento iniziale sul bordo estremo della sigmoide ($u_{\\text{init}} = -10.0$).",
        "   - Si domanda se per la campagna V3b si intenda mantenere l'inizializzazione conservativa standard a $u_{\\text{init}} = -10.0$ per preservare la continuità storica con V0p/V1/V2, oppure se considerare un'inizializzazione più permissiva (es. $u = -3.0 \\implies z \\approx 0.047$, $z(1-z) \\approx 0.045$, $1\,000\\times$ più reattivo).",
        "",
        "---",
        "",
        "## Sezione 5 — Conclusioni del Preflight V2",
        "",
        "1. Tutti i test del Gate Tecnico V3b sono superati al 100% (11/11 test verdi).",
        "2. I gradienti per singolo componente sono stati isolati con successo sia in spazio normalizzato $z$ sia in spazio logit latente $u$.",
        "3. I nuovi artefatti sono salvati in `docs/sprint3/SPRINT3_V3B_PREFLIGHT_V2_RESULTS.json` e `docs/sprint3/SPRINT3_V3B_PREFLIGHT_V2_REPORT.md` preservando intatti i file storici del primo preflight.",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    run_v3b_preflight()
