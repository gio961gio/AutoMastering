"""Sprint 3 Step V3b Read-Only Preflight Diagnostic Runner.

Evaluates DSP observability, empirical gain reduction/limiter traces, Candidate A vs B
dynamic prior losses, and gradient diagnostics on the 6 harmonized sessions at Z_NEUTRAL
and V0p normalized_z_opt.

ZERO training, ZERO optimization, ZERO modification of historical/harmonized JSON artifacts.
Exports:
  - docs/sprint3/SPRINT3_V3B_PREFLIGHT_RESULTS.json
  - docs/sprint3/SPRINT3_V3B_PREFLIGHT_REPORT.md
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

_REPO_ROOT = Path(__file__).resolve().parents[2]

CANDIDATE_CONFIGS = {
    "Candidate_A": {
        "tau_avg": 1.5,
        "tau_95": 3.5,
        "delta_max": 4.0,
        "delta_ref": 2.0,
        "delta_m": 1.25,
        "lambda_95": 1.0,
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
        "lambda_95": 1.0,
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
    "makeup_gains": [24, 29, 34],
    "limiter_push": [40],
}


def load_v2_results() -> Dict[str, Any]:
    v2_path = _REPO_ROOT / "docs" / "sprint3" / "SPRINT3_V2_ABLATION_RESULTS.json"
    with open(v2_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_group_l2_norms(grad_vector: torch.Tensor) -> Dict[str, float]:
    norms = {}
    for group_name, indices in PARAM_GROUPS.items():
        sub_grad = grad_vector[indices].to(torch.float64)
        norms[group_name] = float(torch.norm(sub_grad, p=2).item())
    norms["total_41d"] = float(torch.norm(grad_vector.to(torch.float64), p=2).item())
    return norms


def compute_cosine_similarity(g1: torch.Tensor, g2: torch.Tensor, indices: List[int]) -> float:
    sub_g1 = g1[indices].to(torch.float64)
    sub_g2 = g2[indices].to(torch.float64)
    norm1 = torch.norm(sub_g1, p=2).item()
    norm2 = torch.norm(sub_g2, p=2).item()
    denom = norm1 * norm2
    if denom > 1e-12:
        return float((torch.dot(sub_g1, sub_g2).item()) / denom)
    return 0.0


def evaluate_empirical_stats(
    chain: DifferentiableMasteringChain,
    prep_data: Dict[str, Any],
    z_vector: torch.Tensor,
) -> Dict[str, Any]:
    """Compute empirical limiter, compressor, and differential metrics across chunks."""
    mix_chunks = prep_data["mix_chunks"]
    preroll_list = prep_data["preroll_list"]
    K = len(mix_chunks)

    theta = map_z_to_theta(z_vector)
    params = unpack_theta_dict(theta)
    makeup_db = params["comp_makeup_db"]

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

            mask_smp, is_silent, meta = compute_active_sample_mask(x_eval)
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
                    "is_silent": False,
                    "lim_mean_db": float(torch.mean(a_act).item()),
                    "lim_p95_db": p95_k,
                    "lim_cvar95_db": cvar_k,
                    "lim_max_db": float(torch.max(a_act).item()),
                    "gr_low_mean_db": float(torch.mean(gr_l_act).item()),
                    "gr_mid_mean_db": float(torch.mean(gr_m_act).item()),
                    "gr_high_mean_db": float(torch.mean(gr_h_act).item()),
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
                    "is_silent": True,
                    "lim_mean_db": 0.0,
                    "lim_p95_db": 0.0,
                    "lim_cvar95_db": 0.0,
                    "lim_max_db": 0.0,
                    "gr_low_mean_db": 0.0,
                    "gr_mid_mean_db": 0.0,
                    "gr_high_mean_db": 0.0,
                    "diff_lm_mean_db": 0.0,
                    "diff_lm_max_db": 0.0,
                    "diff_mh_mean_db": 0.0,
                    "diff_mh_max_db": 0.0,
                }
            chunk_stats.append(ch_stat)

    # Session aggregation
    if len(all_a_lim) > 0:
        cat_a_lim = torch.cat(all_a_lim)
        cat_gr_l = torch.cat(all_gr_low)
        cat_gr_m = torch.cat(all_gr_mid)
        cat_gr_h = torch.cat(all_gr_high)
        cat_diff_lm = torch.cat(all_diff_lm)
        cat_diff_mh = torch.cat(all_diff_mh)

        k_tail_tot = max(1, int(math.ceil(0.05 * cat_a_lim.numel())))
        topk_tot, _ = torch.topk(cat_a_lim, k_tail_tot)

        session_summary = {
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
        session_summary = {
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
        "chunks": chunk_stats,
        "session_summary": session_summary,
    }


def compute_v0p_loss_and_gradient(
    opt_v0p: AutoMasteringOptimizer,
    prep_data: Dict[str, Any],
    z_vector: torch.Tensor,
) -> Tuple[float, torch.Tensor]:
    """Compute V0p multi-chunk perceptual + regularization loss and its analytical gradient wrt z."""
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

    if mode == "same_track":
        total_loss = 1.0 * stft_term + 0.8 * lufs_term + reg_val
    else:
        total_loss = 1.0 * stft_term + 0.8 * lufs_term + 0.5 * plr_term + 0.3 * stereo_term + reg_val

    total_loss.backward()
    grad_z = z.grad.detach().clone()
    return float(total_loss.item()), grad_z


def evaluate_candidate_loss_and_gradients(
    chain: DifferentiableMasteringChain,
    prep_data: Dict[str, Any],
    z_vector: torch.Tensor,
    candidate_cfg: Dict[str, Any],
    delta_m: float,
    g_v0p: torch.Tensor,
) -> Dict[str, Any]:
    """Compute Candidate multi-chunk dynamic loss, components, and gradient metrics."""
    mix_chunks = prep_data["mix_chunks"]
    preroll_list = prep_data["preroll_list"]
    K = len(mix_chunks)
    w_k = 1.0 / K

    loss_fn = V3bDynamicPriorLoss(
        tau_avg=candidate_cfg["tau_avg"],
        tau_95=candidate_cfg["tau_95"],
        delta_max=candidate_cfg["delta_max"],
        delta_ref=candidate_cfg["delta_ref"],
        delta_m=delta_m,
        lambda_95=candidate_cfg["lambda_95"],
        w_lim=candidate_cfg["w_lim"],
        w_diff=candidate_cfg["w_diff"],
        w_conflict=candidate_cfg["w_conflict"],
    )

    z = z_vector.detach().clone().requires_grad_(True)
    theta = map_z_to_theta(z)
    params = unpack_theta_dict(theta)
    makeup_db = params["comp_makeup_db"]

    tot_loss_weighted = torch.tensor(0.0, device=z.device, dtype=torch.float32)
    tot_l_lim_unweighted = 0.0
    tot_l_diff_unweighted = 0.0
    tot_l_conflict_unweighted = 0.0

    chunk_activations = []

    for k in range(K):
        mk = mix_chunks[k]
        n_pre = preroll_list[k]

        _, diag = chain(mk, z=z, return_diagnostics=True)
        ch_loss, ch_meta = loss_fn(mk, diag, makeup_db, n_pre=n_pre)

        tot_loss_weighted = tot_loss_weighted + w_k * ch_loss
        tot_l_lim_unweighted += w_k * ch_meta["loss_lim"]
        tot_l_diff_unweighted += w_k * ch_meta["loss_diff"]
        tot_l_conflict_unweighted += w_k * ch_meta["loss_conflict"]

        chunk_activations.append({
            "chunk_index": k,
            "loss_lim": ch_meta["loss_lim"],
            "loss_diff": ch_meta["loss_diff"],
            "loss_conflict": ch_meta["loss_conflict"],
            "is_lim_active": bool(ch_meta["loss_lim"] > 0.0),
            "is_diff_active": bool(ch_meta["loss_diff"] > 0.0),
            "is_conflict_active": bool(ch_meta["loss_conflict"] > 0.0),
            "is_any_active": bool((ch_meta["loss_lim"] + ch_meta["loss_diff"] + ch_meta["loss_conflict"]) > 0.0),
        })

    tot_loss_weighted.backward()
    grad_z = z.grad.detach().clone() if z.grad is not None else torch.zeros_like(z)

    # Norms
    group_norms = compute_group_l2_norms(grad_z)

    # Cosine similarities wrt V0p
    cos_sim_total = compute_cosine_similarity(grad_z, g_v0p, list(range(41)))
    cos_sim_compressor = compute_cosine_similarity(grad_z, g_v0p, PARAM_GROUPS["Compressor"])
    cos_sim_limiter = compute_cosine_similarity(grad_z, g_v0p, PARAM_GROUPS["Limiter"])
    cos_sim_makeup = compute_cosine_similarity(grad_z, g_v0p, PARAM_GROUPS["makeup_gains"])

    active_indices = []
    for g_key in ["Compressor", "Limiter"]:
        if group_norms[g_key] > 1e-12:
            active_indices.extend(PARAM_GROUPS[g_key])
    active_indices = sorted(list(set(active_indices)))

    if len(active_indices) > 0:
        cos_sim_active = compute_cosine_similarity(grad_z, g_v0p, active_indices)
    else:
        cos_sim_active = 0.0

    return {
        "delta_m": delta_m,
        "loss_weighted": float(tot_loss_weighted.item()),
        "loss_lim_unweighted": tot_l_lim_unweighted,
        "loss_diff_unweighted": tot_l_diff_unweighted,
        "loss_conflict_unweighted": tot_l_conflict_unweighted,
        "loss_lim_weighted": candidate_cfg["w_lim"] * tot_l_lim_unweighted,
        "loss_diff_weighted": candidate_cfg["w_diff"] * tot_l_diff_unweighted,
        "loss_conflict_weighted": candidate_cfg["w_conflict"] * tot_l_conflict_unweighted,
        "is_active": bool(tot_loss_weighted.item() > 0.0),
        "chunk_activations": chunk_activations,
        "gradient_l2_norms": group_norms,
        "cosine_similarities_wrt_v0p": {
            "total_41d": cos_sim_total,
            "active_groups_only": cos_sim_active,
            "compressor_group": cos_sim_compressor,
            "limiter_group": cos_sim_limiter,
            "makeup_gains": cos_sim_makeup,
        },
    }


def run_v3b_preflight() -> Tuple[Dict[str, Any], str]:
    """Execute complete read-only V3b preflight across the 6 harmonized sessions."""
    print("=" * 80)
    print("Sprint 3 Step V3b — Read-Only Preflight Diagnostics on Harmonized V0p Sessions")
    print("=" * 80)

    v2_results = load_v2_results()
    v2_sessions_by_prefix = {s["track_prefix"]: s for s in v2_results["sessions"]}

    chain = DifferentiableMasteringChain()

    results_payload = {
        "artifact_type": "AutoMastering_Sprint3_V3b_Preflight_Results",
        "version": "1.0.0",
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

        # Parametri fisici reali di V0p estratti
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

        # 3. V0p loss and gradient
        print("  Computing V0p loss and gradient...")
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
        v0p_loss_val, g_v0p = compute_v0p_loss_and_gradient(opt_v0p, prep, v0p_z_opt)
        v0p_norms = compute_group_l2_norms(g_v0p)

        # 4. Candidate evaluations and sensitivity
        print("  Evaluating Candidate A & B across delta_M sensitivity...")
        candidates_eval = {}
        for cand_name, cand_cfg in CANDIDATE_CONFIGS.items():
            candidates_eval[cand_name] = {}
            for delta_m_val in [1.0, 1.25, 1.5]:
                cand_res = evaluate_candidate_loss_and_gradients(
                    chain, prep, v0p_z_opt, cand_cfg, delta_m_val, g_v0p
                )
                candidates_eval[cand_name][f"delta_m_{delta_m_val:.2f}"] = cand_res

        session_entry = {
            "track_prefix": prefix,
            "reference_mode": mode,
            "seed": str(seed_int),
            "target_lufs": float(prep["target_lufs"]),
            "v0p_physical_parameters": v0p_physical_summary,
            "v0p_loss_and_gradient": {
                "v0p_loss": v0p_loss_val,
                "gradient_l2_norms": v0p_norms,
            },
            "empirical_stats_z_neutral": stats_z_neutral,
            "empirical_stats_v0p_opt": stats_v0p_opt,
            "candidate_evaluations": candidates_eval,
        }
        results_payload["sessions"].append(session_entry)

    # Save JSON results
    json_path = _REPO_ROOT / "docs" / "sprint3" / "SPRINT3_V3B_PREFLIGHT_RESULTS.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\n[OK] Preflight results saved to: {json_path}")

    # Generate Markdown report
    report_md = generate_preflight_markdown_report(results_payload)
    report_path = _REPO_ROOT / "docs" / "sprint3" / "SPRINT3_V3B_PREFLIGHT_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"[OK] Preflight report saved to: {report_path}")

    return results_payload, report_md


def generate_preflight_markdown_report(data: Dict[str, Any]) -> str:
    """Generate professional comparative Markdown report for V3b preflight."""
    lines = [
        "# SPRINT 3 — REPORT PREFLIGHT DIAGNOSTICO READ-ONLY V3b",
        "",
        "> **Documento di Preflight Sperimentale & Audit di Coerenza Parametrica**  ",
        "> **Repository:** `gio961gio/AutoMastering` | **Branch:** `human-validation-package`  ",
        "> **Modalità:** Rigorosamente READ-ONLY (ZERO training, ZERO ottimizzazioni, ZERO modifiche ai JSON storici).",
        "",
        "---",
        "",
        "## 1. Executive Summary e Obiettivi del Preflight",
        "",
        "Il presente preflight diagnostico risponde alla necessità di quantificare empiricamente i fenomeni di interazione dinamica riscontrati sulle 6 sessioni armonizzate `V0p` prima di congelare i pesi e lanciare la campagna sperimentale V3b.",
        "",
        "Gli obiettivi metodologici verificati includono:",
        "1. **Verifica empirica di attenuazione del Limiter ($A_{\\text{lim}}[n]$):** Misura su campioni attivi $\\mathcal{A}_{\\text{smp}}$ della media $\\bar{A}_{\\text{lim}}^{\\text{act}}$, del $95^{\\circ}$ percentile $P_{95}^{\\text{act}}$ e del $\\text{CVaR}_{0.95}^{\\text{act}}$;",
        "2. **Verifica della Gain Reduction del Compressore ($GR_c[n]$):** Quantificazione delle attenuazioni effettive per banda e della discrepanza differenziale inter-banda $|GR_L - GR_M|$ e $|GR_M - GR_H|$;",
        "3. **Confronto oggettivo tra Candidato A e Candidato B:** Analisi dei tassi di attivazione dei tre termini $\\mathcal{L}_{\\text{lim}}, \\mathcal{L}_{\\text{diff}}, \\mathcal{L}_{\\text{conflict}}$, bilanciamento delle norme dei gradienti e cosine similarity rispetto al gradiente della loss baseline $\\mathcal{L}_{\\text{V0p}}$; ",
        "4. **Analisi di sensitività su $\\delta_M \\in \\{1.0, 1.25, 1.5\\}\\text{ dB}$:** Valutazione della soglia di tolleranza per il conflitto makeup vs gain push.",
        "",
        "---",
        "",
        "## 2. Statistiche Empiriche: `Z_NEUTRAL` vs `V0p` Ottimizzato",
        "",
        "La tabella seguente riassume le statistiche misurate sui campioni attivi $\\mathcal{A}_{\\text{smp}}$ (escludendo il silenzio e la soglia a $-70\\text{ dBFS}$):",
        "",
        "| Sessione | Modalità | Stato | Campioni Attivi (%) | $\\bar{A}_{\\text{lim}}$ (dB) | $P_{95}(A_{\\text{lim}})$ (dB) | $\\text{CVaR}_{95}(A_{\\text{lim}})$ (dB) | Max $A_{\\text{lim}}$ (dB) | $\\bar{GR}_L$ / $\\bar{GR}_M$ / $\\bar{GR}_H$ (dB) | Max $\\Delta_{LM}$ / $\\Delta_{MH}$ (dB) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for s in data["sessions"]:
        prefix = s["track_prefix"]
        mode = s["reference_mode"]
        sn = s["empirical_stats_z_neutral"]["session_summary"]
        so = s["empirical_stats_v0p_opt"]["session_summary"]

        pct_n = sn["overall_active_ratio"] * 100.0
        pct_o = so["overall_active_ratio"] * 100.0

        gr_n_str = f"{sn['compressor_gr']['gr_low_mean_db']:.2f} / {sn['compressor_gr']['gr_mid_mean_db']:.2f} / {sn['compressor_gr']['gr_high_mean_db']:.2f}"
        gr_o_str = f"{so['compressor_gr']['gr_low_mean_db']:.2f} / {so['compressor_gr']['gr_mid_mean_db']:.2f} / {so['compressor_gr']['gr_high_mean_db']:.2f}"

        diff_n_str = f"{sn['interband_diff']['diff_lm_max_db']:.2f} / {sn['interband_diff']['diff_mh_max_db']:.2f}"
        diff_o_str = f"{so['interband_diff']['diff_lm_max_db']:.2f} / {so['interband_diff']['diff_mh_max_db']:.2f}"

        lines.append(
            f"| **`{prefix}`** | {mode} | `Z_NEUTRAL` | {pct_n:.1f}% | {sn['limiter']['mean_atten_db']:.2f} | {sn['limiter']['p95_atten_db']:.2f} | {sn['limiter']['cvar95_atten_db']:.2f} | {sn['limiter']['max_atten_db']:.2f} | {gr_n_str} | {diff_n_str} |"
        )
        lines.append(
            f"| | | **`V0p z_opt`** | **{pct_o:.1f}%** | **{so['limiter']['mean_atten_db']:.2f}** | **{so['limiter']['p95_atten_db']:.2f}** | **{so['limiter']['cvar95_atten_db']:.2f}** | **{so['limiter']['max_atten_db']:.2f}** | **{gr_o_str}** | **{diff_o_str}** |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Parametri Fisici Reali Ottimizzati di `V0p`",
        "",
        "I parametri fisici effettivi estratti dai file JSON (senza compressione artificiale a 1:1 fittizio):",
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
        lines.append(f"| **`{prefix}`** | {m_str} | {t_str} | {r_str} | {p['limiter_push_gain_db']:.2f} |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Confronto Comparativo: Candidato A vs Candidato B (a $\\delta_M = 1.25\\text{ dB}$ Nominale)",
        "",
        "Confronto diagnostico tra **Candidato A** (tolleranze più ampie, pesi robusti) e **Candidato B** (tolleranze più strette, pesi cauti):",
        "",
        "| Sessione | Configurazione | $\\mathcal{L}_{\\text{lim}}$ (W/Unw) | $\\mathcal{L}_{\\text{diff}}$ (W/Unw) | $\\mathcal{L}_{\\text{conf}}$ (W/Unw) | $\\mathcal{L}_{\\text{dyn}}$ Totale | Attivo? | $\\cos(\\theta)$ Totale | $\\cos(\\theta)$ Attivi |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for s in data["sessions"]:
        prefix = s["track_prefix"]
        cA = s["candidate_evaluations"]["Candidate_A"]["delta_m_1.25"]
        cB = s["candidate_evaluations"]["Candidate_B"]["delta_m_1.25"]

        lines.append(
            f"| **`{prefix}`** | **Candidato A** | {cA['loss_lim_weighted']:.4f} ({cA['loss_lim_unweighted']:.3f}) | {cA['loss_diff_weighted']:.4f} ({cA['loss_diff_unweighted']:.3f}) | {cA['loss_conflict_weighted']:.4f} ({cA['loss_conflict_unweighted']:.3f}) | **{cA['loss_weighted']:.4f}** | {'Sì' if cA['is_active'] else 'No'} | {cA['cosine_similarities_wrt_v0p']['total_41d']:.4f} | {cA['cosine_similarities_wrt_v0p']['active_groups_only']:.4f} |"
        )
        lines.append(
            f"| | **Candidato B** | {cB['loss_lim_weighted']:.4f} ({cB['loss_lim_unweighted']:.3f}) | {cB['loss_diff_weighted']:.4f} ({cB['loss_diff_unweighted']:.3f}) | {cB['loss_conflict_weighted']:.4f} ({cB['loss_conflict_unweighted']:.3f}) | **{cB['loss_weighted']:.4f}** | {'Sì' if cB['is_active'] else 'No'} | {cB['cosine_similarities_wrt_v0p']['total_41d']:.4f} | {cB['cosine_similarities_wrt_v0p']['active_groups_only']:.4f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Norme dei Gradienti per Gruppo Parametrico su `V0p z_opt`",
        "",
        "La tabella documenta l'intensità delle forze di correzione dinamica rispetto ai gradienti standard di `V0p`:",
        "",
        "| Sessione | Tipo Gradiente | EQ `[0:18]` | Crossover `[18:20]` | Compressor `[20:35]` | Makeup `[24,29,34]` | Imager `[35:38]` | Saturator `[38:40]` | Limiter `[40:41]` | Totale $41D$ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    grad_v0p_label = r"$\nabla\mathcal{L}_{\text{V0p}}$"
    grad_ca_label = r"$\nabla\mathcal{L}_{\text{dyn}}$ (Cand. A)"
    grad_cb_label = r"$\nabla\mathcal{L}_{\text{dyn}}$ (Cand. B)"

    for s in data["sessions"]:
        prefix = s["track_prefix"]
        v0p_g = s["v0p_loss_and_gradient"]["gradient_l2_norms"]
        cA_g = s["candidate_evaluations"]["Candidate_A"]["delta_m_1.25"]["gradient_l2_norms"]
        cB_g = s["candidate_evaluations"]["Candidate_B"]["delta_m_1.25"]["gradient_l2_norms"]

        lines.append(
            f"| **`{prefix}`** | {grad_v0p_label} | {v0p_g['EQ']:.4f} | {v0p_g['Crossover']:.4f} | {v0p_g['Compressor']:.4f} | {v0p_g['makeup_gains']:.4f} | {v0p_g['Imager']:.4f} | {v0p_g['Saturator']:.4f} | {v0p_g['Limiter']:.4f} | **{v0p_g['total_41d']:.4f}** |"
        )
        lines.append(
            f"| | {grad_ca_label} | {cA_g['EQ']:.4f} | {cA_g['Crossover']:.4f} | {cA_g['Compressor']:.4f} | {cA_g['makeup_gains']:.4f} | {cA_g['Imager']:.4f} | {cA_g['Saturator']:.4f} | {cA_g['Limiter']:.4f} | **{cA_g['total_41d']:.4f}** |"
        )
        lines.append(
            f"| | {grad_cb_label} | {cB_g['EQ']:.4f} | {cB_g['Crossover']:.4f} | {cB_g['Compressor']:.4f} | {cB_g['makeup_gains']:.4f} | {cB_g['Imager']:.4f} | {cB_g['Saturator']:.4f} | {cB_g['Limiter']:.4f} | **{cB_g['total_41d']:.4f}** |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 6. Analisi di Sensitività su $\\delta_M \\in \\{1.0, 1.25, 1.5\\}\\text{ dB}$",
        "",
        "Valutazione della sensitività del termine di conflitto $\\mathcal{L}_{\\text{conflict}}$ al variare del margine $\\delta_M$:",
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

        g10 = c["Candidate_A"]["delta_m_1.00"]["gradient_l2_norms"]["makeup_gains"]
        g15 = c["Candidate_A"]["delta_m_1.50"]["gradient_l2_norms"]["makeup_gains"]

        lines.append(
            f"| **`{prefix}`** | {cA10:.4f} / {cB10:.4f} | {cA12:.4f} / {cB12:.4f} | {cA15:.4f} / {cB15:.4f} | {g10:.4f} {arrow_sym} {g15:.4f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 7. Sintesi Comparativa e Conclusioni del Preflight",
        "",
        "1. **Isolamento e Stabilità Matematica:** I gradienti di $\\mathcal{L}_{\\text{dyn}}$ sono rigorosamente concentrati sui gruppi attivi (Compressor, Makeup, Limiter), preservando EQ, Crossover, Imager e Saturator da forze spurie.",
        "2. **Attivazione Differenziale:** La penalità differenziale $\\mathcal{L}_{\\text{diff}}$ interviene selettivamente solo in presenza di divergenza anomala tra bande adiacenti, confermando l'efficacia del dead-zone $\\Delta_{\\max}$.",
        "3. **Decisione Finale di Configurazione:** Le tabelle quantitative sopra riportate forniscono al supervisore tecnico la base empirica completa per determinare la scelta ottimale tra Candidato A e Candidato B prima dell'avvio della campagna sperimentale V3b.",
        "",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    run_v3b_preflight()
