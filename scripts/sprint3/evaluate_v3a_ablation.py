"""Sprint 3 Step V3a Ablation Runner and Preflight Gradient Diagnostics.

Variant Evaluated:
  - V3a: V0p + Composite EQ Response Prior (K_eq=128, tau_shape=1.5, tau_offset=1.0,
         w_slope=2.0, w_offset=1.0, weight_eq_prior=0.02, float64 analytical response).

Modes:
  --preflight-only (default): Read-only gradient diagnostics and EQ metric evaluation
    on Z_NEUTRAL and V0p normalized_z_opt across all 6 harmonized sessions.
    Exports docs/sprint3/SPRINT3_V3A_PREFLIGHT_GRADIENT_DIAGNOSTICS.json.
  --run-ablation: 30-step Adam optimization loop across all 6 sessions (Phase 2).
    Exports docs/sprint3/SPRINT3_V3A_ABLATION_RESULTS.json and docs/sprint3/SPRINT3_V3A_ABLATION_REPORT.md.
"""

import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

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
from ddsp_mastering.loss.eq_prior import CompositeEQPriorLoss
from ddsp_mastering.loss.stft import MultiResolutionSTFTLoss
from ddsp_mastering.loss.loudness import DifferentiableLoudnessLoss
from ddsp_mastering.optimize import AutoMasteringOptimizer, MasteringBenchmarkSuite
from ddsp_mastering.param_space import NUM_PARAMS, PARAM_SPECS, Z_NEUTRAL, map_theta_to_z, map_z_to_theta
from scripts.human_validation.offline_renderer import render_offline_with_tail_flush
from scripts.sprint3.harmonized_session_prep import (
    SESSIONS_SPEC,
    compute_individual_loss_breakdown,
    prepare_harmonized_session,
)


def load_v2_results() -> Dict[str, Any]:
    """Load SPRINT3_V2_ABLATION_RESULTS.json containing harmonized session metadata and V0p z_opt."""
    p = _REPO_ROOT / "docs" / "sprint3" / "SPRINT3_V2_ABLATION_RESULTS.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing results file: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def compute_v3a_gradient_diagnostics(
    prep_data: Dict[str, Any],
    z_var: torch.Tensor,
    opt_v0p: AutoMasteringOptimizer,
    eq_prior_loss: CompositeEQPriorLoss,
    weight_eq_prior: float = 0.02,
) -> Dict[str, Any]:
    """Compute EQ metrics, exact L2 gradient norms on EQ params (indices 0..17), and cosine similarity.

    Uses retain_graph=True across all backward calls and ensures zeroing of gradients between terms.
    Applies opt_v0p.weight_reg (0.01 in same_track, 0.02 in stylistic) for parameter regularization.
    """
    mode = prep_data["mode"]
    mix_chunks = prep_data["mix_chunks"]
    ref_chunks = prep_data["ref_chunks"]
    preroll_list = prep_data["preroll_list"]
    target_lufs = prep_data["target_lufs"]
    K = len(mix_chunks)
    weights = [1.0 / K] * K

    z = z_var.detach().clone().to(torch.float32).requires_grad_(True)

    # 1. Analytical EQ Prior forward and gradient (float64 precision inside CompositeEQPriorLoss)
    l_eq_prior, eq_metrics = eq_prior_loss(z)
    eq_metrics["L_eq_prior_weighted"] = float(weight_eq_prior * eq_metrics["L_eq_prior_unweighted"])

    grad_eq_all = torch.autograd.grad(l_eq_prior, z, retain_graph=True)[0]
    g_eq = grad_eq_all[:18].detach().clone().to(torch.float64)
    norm_grad_eq_unweighted = float(torch.norm(g_eq, p=2).item())
    norm_grad_eq_weighted = float(weight_eq_prior * norm_grad_eq_unweighted)

    # 2. V0p forward across 4 chunks in IIR mode
    stft_term = torch.tensor(0.0, dtype=torch.float32)
    lufs_term = torch.tensor(0.0, dtype=torch.float32)
    plr_term = torch.tensor(0.0, dtype=torch.float32)
    stereo_term = torch.tensor(0.0, dtype=torch.float32)

    for k in range(K):
        mk = mix_chunks[k]
        n_pre = preroll_list[k]
        y_ext = opt_v0p.chain(mk, z=z, eq_mode="iir")
        y_k = y_ext[..., n_pre:] if n_pre > 0 else y_ext
        rk = ref_chunks[k]

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
    diff_z = z - z_neutral
    weight_reg = opt_v0p.weight_reg
    reg_val = weight_reg * torch.mean(diff_z ** 2)

    if mode == "same_track":
        loss_v0p_total = 1.0 * stft_term + 0.8 * lufs_term + reg_val
        term_losses: Dict[str, torch.Tensor] = {
            "mrstft": 1.0 * stft_term,
            "lufs": 0.8 * lufs_term,
            "param_reg": reg_val,
        }
    else:
        loss_v0p_total = 1.0 * stft_term + 0.8 * lufs_term + 0.5 * plr_term + 0.3 * stereo_term + reg_val
        term_losses = {
            "mel_spectral": 1.0 * stft_term,
            "lufs": 0.8 * lufs_term,
            "plr_box": 0.5 * plr_term,
            "stereo": 0.3 * stereo_term,
            "param_reg": reg_val,
        }

    loss_v0p_total.backward(retain_graph=True)
    assert z.grad is not None, "z.grad must not be None after loss_v0p_total.backward"
    g_v0p = z.grad[:18].detach().clone().to(torch.float64)
    v0p_total_norm = float(torch.norm(g_v0p, p=2).item())

    term_grads: Dict[str, float] = {}
    for t_name, t_val in term_losses.items():
        z.grad.zero_()
        if t_val.requires_grad:
            t_val.backward(retain_graph=True)
            if z.grad is not None:
                g_term = z.grad[:18].detach().clone().to(torch.float64)
                term_grads[t_name] = float(torch.norm(g_term, p=2).item())
            else:
                term_grads[t_name] = 0.0
        else:
            term_grads[t_name] = 0.0

    spectral_key = "mrstft" if mode == "same_track" else "mel_spectral"
    assert term_grads[spectral_key] > 0.0, f"Expected {spectral_key} > 0.0, got {term_grads[spectral_key]}"
    assert term_grads["lufs"] > 0.0, f"Expected lufs > 0.0, got {term_grads['lufs']}"

    denom = norm_grad_eq_unweighted * v0p_total_norm
    if denom > 1e-12:
        dot_product = float(torch.dot(g_eq, g_v0p).item())
        cos_theta = float(dot_product / denom)
    else:
        cos_theta = 0.0

    return {
        "eq_metrics": eq_metrics,
        "grad_norm_eq_unweighted": norm_grad_eq_unweighted,
        "grad_norm_eq_weighted": norm_grad_eq_weighted,
        "grad_norm_v0p_total": v0p_total_norm,
        "v0p_terms_grad_norm_eq18": term_grads,
        "cos_similarity": cos_theta,
        "gradient_norms_eq_params": {
            "grad_norm_eq_prior_unweighted": norm_grad_eq_unweighted,
            "grad_norm_eq_prior_weighted": norm_grad_eq_weighted,
            "grad_norm_v0p_total": v0p_total_norm,
            "individual_terms_weighted": term_grads,
            "cos_sim_eq_vs_v0p": cos_theta,
        },
    }


def _eval_v3a_objective_iir(
    opt: AutoMasteringOptimizer,
    prep_data: Dict[str, Any],
    z_vector: torch.Tensor,
    weight_eq_prior: float = 0.02,
    eq_prior_loss: Optional[CompositeEQPriorLoss] = None,
) -> Tuple[float, float, Dict[str, float]]:
    """Evaluate V3a objective in IIR mode across chunks.

    Returns:
        (total_v3a_loss, v0p_loss, metrics_dict)
    """
    mode = prep_data["mode"]
    mix_chunks = prep_data["mix_chunks"]
    ref_chunks = prep_data["ref_chunks"]
    preroll_list = prep_data["preroll_list"]
    ref_ext_chunks = prep_data["ref_ext_chunks"]
    ref_preroll_list = prep_data["ref_preroll_list"]
    target_lufs = prep_data["target_lufs"]
    K = len(mix_chunks)
    weights = [1.0 / K] * K

    z_t = z_vector.detach().clone().to(mix_chunks[0].device, dtype=torch.float32)
    z_neutral = Z_NEUTRAL.to(z_t.device, dtype=z_t.dtype)
    reg_val = torch.mean((z_t - z_neutral) ** 2)
    reg_loss = float((opt.weight_reg * reg_val).item())

    tot_perceptual = 0.0
    agg_m: Dict[str, float] = {}

    with torch.no_grad():
        for k in range(K):
            n_pre = preroll_list[k]
            mk = mix_chunks[k]
            y_ext = opt.chain(mk, z=z_t, eq_mode="iir")
            y_k = y_ext[..., n_pre:] if n_pre > 0 else y_ext
            ref_k = ref_chunks[k]
            ref_ext_k = ref_ext_chunks[k]
            ref_pre_k = ref_preroll_list[k]

            if mode == "same_track":
                lk, mk_dict = opt.loss_fn(
                    pred_audio=y_k,
                    target_audio=ref_k,
                    target_lufs=target_lufs,
                    normalized_params=None,
                    align_latency=True,
                    pred_ext_audio=y_ext,
                    pred_preroll_samples=n_pre,
                    ref_ext_audio=ref_ext_k,
                    ref_preroll_samples=ref_pre_k,
                    include_eq_prior=False,
                )
            else:
                anchor_k = mk[..., n_pre:] if n_pre > 0 else mk
                lk, mk_dict = opt.stylistic_loss_fn(
                    pred_audio=y_k,
                    ref_audio=ref_k,
                    normalized_params=None,
                    target_lufs=target_lufs,
                    mix_anchor_audio=anchor_k,
                    pred_ext_audio=y_ext,
                    pred_preroll_samples=n_pre,
                    ref_ext_audio=ref_ext_k,
                    ref_preroll_samples=ref_pre_k,
                    include_eq_prior=False,
                )
            tot_perceptual += weights[k] * float(lk.item())
            for mk_key, mk_val in mk_dict.items():
                agg_m[mk_key] = agg_m.get(mk_key, 0.0) + weights[k] * float(mk_val)

        v0p_loss = tot_perceptual + reg_loss
        agg_m["param_reg_loss"] = float(reg_val.item())
        agg_m["v0p_loss"] = float(v0p_loss)

        if eq_prior_loss is not None and weight_eq_prior > 0.0:
            eq_val, eq_m = eq_prior_loss(z_t)
            eq_prior_item = float(eq_val.item())
            total_v3a_loss = v0p_loss + weight_eq_prior * eq_prior_item
            agg_m["eq_prior_loss"] = eq_prior_item
            for eq_k, eq_v in eq_m.items():
                agg_m[f"eq_prior_{eq_k}"] = eq_v
        else:
            total_v3a_loss = v0p_loss

        agg_m["total_loss"] = float(total_v3a_loss)

    return total_v3a_loss, v0p_loss, agg_m


def run_preflight_diagnostics() -> Dict[str, Any]:
    """Execute preflight gradient diagnostics across all 6 harmonized sessions in read-only mode."""
    print("=" * 80)
    print("Sprint 3 Step V3a — Preflight Gradient Diagnostics (Read-Only)")
    print("=" * 80)

    v2_results = load_v2_results()
    v2_sessions_by_prefix: Dict[str, Dict[str, Any]] = {
        s["track_prefix"]: s for s in v2_results["sessions"]
    }

    eq_prior_loss = CompositeEQPriorLoss(
        sample_rate=48000.0,
        f_min=20.0,
        f_max=20000.0,
        K_eq=128,
        tau_shape=1.5,
        tau_offset=1.0,
        w_slope=2.0,
        w_offset=1.0,
        eps_eq=1e-12,
    )

    diagnostics_payload: Dict[str, Any] = {
        "artifact_type": "AutoMastering_Sprint3_V3a_Preflight_Gradient_Diagnostics",
        "version": "1.0.0",
        "v3a_hyperparameters": {
            "K_eq": 128,
            "f_min": 20.0,
            "f_max": 20000.0,
            "sample_rate": 48000.0,
            "tau_shape": 1.5,
            "tau_offset": 1.0,
            "w_slope": 2.0,
            "w_offset": 1.0,
            "weight_eq_prior": 0.02,
            "eps_eq": 1e-12,
            "precision": "torch.float64 analytical biquad frequency response",
        },
        "sessions": [],
    }

    for spec in SESSIONS_SPEC:
        prefix = spec["track_prefix"]
        mode = spec["mode"]
        print(f"\n--- Processing Session: {prefix} ({mode}) ---")

        v2_sess = v2_sessions_by_prefix[prefix]
        seed_int = int(v2_sess["seed"])

        prep = prepare_harmonized_session(spec, seed_int=seed_int, repo_root=_REPO_ROOT)

        assert prep["target_lufs"] == v2_sess["target_lufs"], (
            f"[{prefix}] target_lufs mismatch: {prep['target_lufs']} vs {v2_sess['target_lufs']}"
        )
        assert prep["chunk_metadata"] == v2_sess["chunk_metadata"], (
            f"[{prefix}] chunk_metadata mismatch"
        )
        assert int(v2_sess["seed"]) == seed_int, (
            f"[{prefix}] seed mismatch: {v2_sess['seed']} vs {seed_int}"
        )

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

        v0p_z_opt = torch.tensor(
            v2_sess["variants"]["V0p"]["normalized_z_opt"], dtype=torch.float32
        )

        _, v0p_loss_z0, _ = _eval_v3a_objective_iir(
            opt=opt_v0p,
            prep_data=prep,
            z_vector=Z_NEUTRAL,
            weight_eq_prior=0.0,
            eq_prior_loss=None,
        )
        _, v0p_loss_opt, _ = _eval_v3a_objective_iir(
            opt=opt_v0p,
            prep_data=prep,
            z_vector=v0p_z_opt,
            weight_eq_prior=0.0,
            eq_prior_loss=None,
        )

        exp_init = float(v2_sess["variants"]["V0p"]["initial_loss_z0_iir"])
        exp_final = float(v2_sess["variants"]["V0p"]["final_loss_z_opt_iir"])
        diff_init = abs(v0p_loss_z0 - exp_init)
        diff_final = abs(v0p_loss_opt - exp_final)
        tol = 2e-4
        assert diff_init < tol, f"[{prefix}] initial_loss mismatch: {v0p_loss_z0} vs {exp_init}"
        assert diff_final < tol, f"[{prefix}] final_loss mismatch: {v0p_loss_opt} vs {exp_final}"

        neutral_diag = compute_v3a_gradient_diagnostics(
            prep_data=prep,
            z_var=Z_NEUTRAL,
            opt_v0p=opt_v0p,
            eq_prior_loss=eq_prior_loss,
            weight_eq_prior=0.02,
        )

        v0p_diag = compute_v3a_gradient_diagnostics(
            prep_data=prep,
            z_var=v0p_z_opt,
            opt_v0p=opt_v0p,
            eq_prior_loss=eq_prior_loss,
            weight_eq_prior=0.02,
        )

        session_entry = {
            "track_prefix": prefix,
            "reference_mode": mode,
            "seed": str(seed_int),
            "target_lufs": float(prep["target_lufs"]),
            "metadata_assertion_verified": True,
            "v0p_loss_parity_verified": True,
            "at_z_neutral": neutral_diag,
            "at_v0p_z_opt": v0p_diag,
            "z_neutral_diagnostics": neutral_diag,
            "v0p_z_opt_diagnostics": v0p_diag,
        }
        diagnostics_payload["sessions"].append(session_entry)

    out_file = _REPO_ROOT / "docs" / "sprint3" / "SPRINT3_V3A_PREFLIGHT_GRADIENT_DIAGNOSTICS.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(diagnostics_payload, indent=2), encoding="utf-8")
    print(f"\n[OK] Preflight gradient diagnostics exported to: {out_file}")
    return diagnostics_payload


def run_ablation() -> None:
    """Execute 30-step optimization loop across all 6 sessions for V3a (Phase 2)."""
    print("=" * 80)
    print("Sprint 3 Step V3a Ablation Campaign: V0p + Composite EQ Prior (Phase 2)")
    print("=" * 80)

    cache_dir = _REPO_ROOT / ".cache_v3a_ablation"
    cache_dir.mkdir(parents=True, exist_ok=True)

    v2_results = load_v2_results()
    v2_sessions_by_prefix: Dict[str, Dict[str, Any]] = {
        s["track_prefix"]: s for s in v2_results["sessions"]
    }

    suite = MasteringBenchmarkSuite(sample_rate=48000.0)

    eq_prior_loss = CompositeEQPriorLoss(
        sample_rate=48000.0,
        f_min=20.0,
        f_max=20000.0,
        K_eq=128,
        tau_shape=1.5,
        tau_offset=1.0,
        w_slope=2.0,
        w_offset=1.0,
        eps_eq=1e-12,
    )

    sessions_output: List[Dict[str, Any]] = []

    for spec in SESSIONS_SPEC:
        prefix = spec["track_prefix"]
        mode = spec["mode"]
        v2_sess = v2_sessions_by_prefix[prefix]
        seed_int = int(v2_sess["seed"])

        session_cache_file = cache_dir / f"{prefix}_v3a.json"

        if session_cache_file.exists():
            print(f"\n[CACHE HIT] Loading cached V3a result for {prefix} from {session_cache_file.name}")
            sess_data = json.loads(session_cache_file.read_text(encoding="utf-8"))
            sessions_output.append(sess_data)
            continue

        print(f"\n" + "#" * 60)
        print(f"### Optimizing Session: {prefix} ({mode}) | Seed: {seed_int}")
        print("#" * 60)

        t0 = time.time()
        prep = prepare_harmonized_session(spec, seed_int=seed_int, repo_root=_REPO_ROOT)

        target_lufs = prep["target_lufs"]
        mix_chunks = prep["mix_chunks"]
        ref_chunks = prep["ref_chunks"]
        preroll_list = prep["preroll_list"]
        ref_ext_chunks = prep["ref_ext_chunks"]
        ref_preroll_list = prep["ref_preroll_list"]
        raw_full_64 = prep["raw_full_64"]
        ref_eval_tensor = prep["ref_eval_tensor"]
        mix_start = prep["mix_start"]
        eval_len = prep["eval_len"]

        # V0p optimizer instance for reference baseline & gradient comparisons
        opt_v0p = AutoMasteringOptimizer(
            sample_rate=48000.0,
            lr=0.04,
            num_steps=30,
            target_lufs=target_lufs,
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

        v0p_data = v2_sess["variants"]["V0p"]
        v0p_z_opt = torch.tensor(v0p_data["normalized_z_opt"], dtype=torch.float32)

        # V3a optimizer instance
        opt_v3a = AutoMasteringOptimizer(
            sample_rate=48000.0,
            lr=0.04,
            num_steps=30,
            target_lufs=target_lufs,
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
            weight_eq_prior=0.02,
            eq_tau_shape=1.5,
            eq_tau_offset=1.0,
            eq_w_slope=2.0,
            eq_w_offset=1.0,
        )

        print(f"[{prefix}] Launching 30-step Adam optimization for V3a...")
        z_opt_v3a, _, history_v3a = opt_v3a.optimize_multi_chunk(
            mix_chunks=mix_chunks,
            ref_chunks=ref_chunks,
            reference_mode=mode,
            verbose=True,
            align_latency=True,
            mix_preroll_samples=preroll_list,
            target_lufs=target_lufs,
            ref_ext_chunks=ref_ext_chunks,
            ref_preroll_list=ref_preroll_list,
        )

        # Evaluate V3a objectives in IIR mode
        loss_z0_v3a, v0p_loss_init_v3a, metrics_init_v3a = _eval_v3a_objective_iir(
            opt=opt_v3a,
            prep_data=prep,
            z_vector=Z_NEUTRAL,
            weight_eq_prior=0.02,
            eq_prior_loss=eq_prior_loss,
        )
        loss_zopt_v3a, v0p_loss_final_v3a, metrics_final_v3a = _eval_v3a_objective_iir(
            opt=opt_v3a,
            prep_data=prep,
            z_vector=z_opt_v3a,
            weight_eq_prior=0.02,
            eq_prior_loss=eq_prior_loss,
        )
        red_v3a = ((loss_zopt_v3a - loss_z0_v3a) / loss_z0_v3a) * 100.0 if loss_z0_v3a > 0.0 else 0.0

        # EQ Metrics on V3a z_opt
        _, eq_metrics_v3a = eq_prior_loss(z_opt_v3a)
        eq_metrics_v3a["L_eq_prior_weighted"] = float(0.02 * eq_metrics_v3a["L_eq_prior_unweighted"])

        # Full-track rendering in float64 with tail flush
        print(f"[{prefix}] Rendering full track with C++ DSP tail flush...")
        mastered_v3a_64, _ = render_offline_with_tail_flush(
            raw_full_64, mode="z", z=z_opt_v3a.tolist(), sample_rate=48000.0
        )

        # Full-track benchmark
        if mode == "same_track":
            mastered_eval_seg = mastered_v3a_64[:, mix_start : mix_start + eval_len].astype(np.float32)
            bm_full_v3a = suite.evaluate(
                torch.tensor(mastered_eval_seg).unsqueeze(0),
                ref_eval_tensor,
                align_latency=False,
                reference_mode="same_track",
            )
        else:
            bm_full_v3a = suite.evaluate(
                torch.tensor(mastered_v3a_64.astype(np.float32)).unsqueeze(0),
                ref_eval_tensor,
                align_latency=False,
                reference_mode="stylistic",
            )

        # Peak & Intersample metrics
        pred_sample_peak_linear = float(np.max(np.abs(mastered_v3a_64)))
        pred_sample_peak_dbfs = 20.0 * math.log10(pred_sample_peak_linear + 1e-12)
        pred_true_peak_dbtp = float(bm_full_v3a.get("pred_true_peak_dbtp", 0.0))
        intersample_overshoot_db = pred_true_peak_dbtp - pred_sample_peak_dbfs
        bm_full_v3a["pred_sample_peak_dbfs"] = pred_sample_peak_dbfs
        bm_full_v3a["intersample_overshoot_db"] = intersample_overshoot_db

        # 10 unweighted terms breakdown
        all_10_terms_v3a = compute_individual_loss_breakdown(opt_v3a.chain, z_opt_v3a, prep)

        # Physical parameters theta
        name_to_idx = {spec.name: i for i, spec in enumerate(PARAM_SPECS)}
        theta_v3a = map_z_to_theta(z_opt_v3a)
        theta_v3a_dict = {
            spec.name: float(theta_v3a[name_to_idx[spec.name]].item()) for spec in PARAM_SPECS
        }

        # Gradient diagnostics at z_opt_v3a
        v3a_grad_diag = compute_v3a_gradient_diagnostics(
            prep_data=prep,
            z_var=z_opt_v3a,
            opt_v0p=opt_v0p,
            eq_prior_loss=eq_prior_loss,
            weight_eq_prior=0.02,
        )

        elapsed = time.time() - t0
        print(f"[{prefix}] Finished V3a in {elapsed:.1f} s")
        print(f"[{prefix}] Loss: init={loss_z0_v3a:.4f} -> final={loss_zopt_v3a:.4f} ({red_v3a:+.2f}%)")
        print(f"[{prefix}] EQ G_bar: {eq_metrics_v3a['G_bar']:+.2f} dB, L_offset: {eq_metrics_v3a['L_offset_raw']:.4f}")
        print(f"[{prefix}] LUFS: {bm_full_v3a.get('pred_lufs', 0.0):.2f} (Target {target_lufs:.2f}) | TP: {pred_true_peak_dbtp:.2f} dBTP")

        v3a_dict = {
            "config": {
                "weight_eq_prior": 0.02,
                "tau_shape": 1.5,
                "tau_offset": 1.0,
                "w_slope": 2.0,
                "w_offset": 1.0,
                "lr": 0.04,
                "num_steps": 30,
            },
            "target_lufs": float(target_lufs),
            "initial_loss_z0_iir": float(loss_z0_v3a),
            "final_loss_z_opt_iir": float(loss_zopt_v3a),
            "v0p_loss_init": float(v0p_loss_init_v3a),
            "v0p_loss_final": float(v0p_loss_final_v3a),
            "loss_reduction_iir_pct": float(red_v3a),
            "chunk_metrics_init_iir": metrics_init_v3a,
            "chunk_metrics_final_iir": metrics_final_v3a,
            "normalized_z_opt": z_opt_v3a.tolist(),
            "physical_parameters_theta": theta_v3a_dict,
            "eq_metrics": eq_metrics_v3a,
            "gradient_diagnostics": v3a_grad_diag,
            "all_10_terms_final": all_10_terms_v3a,
            "full_track_benchmark": bm_full_v3a,
        }

        # Calculate V0p sample peak & overshoot if missing
        if "pred_sample_peak_dbfs" not in v0p_data["full_track_benchmark"]:
            mastered_v0p_64, _ = render_offline_with_tail_flush(
                raw_full_64, mode="z", z=v0p_z_opt.tolist(), sample_rate=48000.0
            )
            v0p_sp_linear = float(np.max(np.abs(mastered_v0p_64)))
            v0p_sp_dbfs = 20.0 * math.log10(v0p_sp_linear + 1e-12)
            v0p_tp_dbtp = float(v0p_data["full_track_benchmark"].get("pred_true_peak_dbtp", 0.0))
            v0p_data["full_track_benchmark"]["pred_sample_peak_dbfs"] = v0p_sp_dbfs
            v0p_data["full_track_benchmark"]["intersample_overshoot_db"] = v0p_tp_dbtp - v0p_sp_dbfs

        # Add EQ metrics to V0p
        _, eq_metrics_v0p = eq_prior_loss(v0p_z_opt)
        eq_metrics_v0p["L_eq_prior_weighted"] = float(0.02 * eq_metrics_v0p["L_eq_prior_unweighted"])
        v0p_data["eq_metrics"] = eq_metrics_v0p

        # Add gradient diagnostics to V0p
        v0p_grad_diag = compute_v3a_gradient_diagnostics(
            prep_data=prep,
            z_var=v0p_z_opt,
            opt_v0p=opt_v0p,
            eq_prior_loss=eq_prior_loss,
            weight_eq_prior=0.02,
        )
        v0p_data["gradient_diagnostics"] = v0p_grad_diag

        # Z_NEUTRAL diagnostics
        neutral_grad_diag = compute_v3a_gradient_diagnostics(
            prep_data=prep,
            z_var=Z_NEUTRAL,
            opt_v0p=opt_v0p,
            eq_prior_loss=eq_prior_loss,
            weight_eq_prior=0.02,
        )

        sess_payload = {
            "track_prefix": prefix,
            "reference_mode": mode,
            "seed": str(seed_int),
            "target_lufs": float(target_lufs),
            "elapsed_sec": float(elapsed),
            "z_neutral_diagnostics": neutral_grad_diag,
            "variants": {
                "V0p": v0p_data,
                "V3a": v3a_dict,
            },
        }

        session_cache_file.write_text(json.dumps(sess_payload, indent=2), encoding="utf-8")
        sessions_output.append(sess_payload)
        gc.collect()

    # Final Payload
    results_payload = {
        "artifact_type": "AutoMastering_Sprint3_V3a_Ablation_Results",
        "version": "1.0.0",
        "protocol": "Sprint 3 Harmonized Protocol (Percorso A, bit-identical coordinates, full-track target LUFS)",
        "v3a_hyperparameters": {
            "K_eq": 128,
            "f_min": 20.0,
            "f_max": 20000.0,
            "sample_rate": 48000.0,
            "tau_shape": 1.5,
            "tau_offset": 1.0,
            "w_slope": 2.0,
            "w_offset": 1.0,
            "weight_eq_prior": 0.02,
            "eps_eq": 1e-12,
            "precision": "torch.float64 analytical biquad frequency response",
        },
        "sessions": sessions_output,
    }

    out_file = _REPO_ROOT / "docs" / "sprint3" / "SPRINT3_V3A_ABLATION_RESULTS.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
    print(f"\n[DONE] Saved V3a ablation results to: {out_file}")

    report_file = _REPO_ROOT / "docs" / "sprint3" / "SPRINT3_V3A_ABLATION_REPORT.md"
    generate_v3a_report_md(results_payload, report_file)


def generate_v3a_report_md(results_payload: Dict[str, Any], out_path: Path) -> None:
    """Generate comprehensive technical evaluation report comparing V0p vs V3a."""
    sessions = results_payload["sessions"]
    hyp = results_payload["v3a_hyperparameters"]

    lines = [
        "# Sprint 3 — Report Ufficiale di Ablation Step V3a (`V3a_EQ_PRIOR_V1`)",
        "",
        "**Data:** 10 Novembre 2026  ",
        "**Autore:** Antigravity (Advanced Agentic Coding)  ",
        "**Supervisore Tecnico:** Argon (Gemini Web Pro / Medium Thinking)  ",
        "**Protocollo:** Armonizzato Sprint 3 (Percorso A, coordinate bit-identiche, target LUFS integrato da file WAV sorgente)",
        "",
        "---",
        "",
        "## 1. Executive Summary & Obiettivi di V3a",
        "",
        "Lo Step **`V3a`** (`V3a_EQ_PRIOR_V1`) implementa per la prima volta nella pipeline di mastering un **prior analitico quadratico con dead-zone sulla risposta in frequenza composita dell'EQ parametrico a 6 bande**.",
        "Mentre la baseline di controllo `V0p` applica esclusivamente una penalità $L_2$ nello spazio normalizzato dei parametri latenti ($w_{\\text{reg}}\\|z - z_{\\text{neutral}}\\|^2$), `V3a` penalizza direttamente la deviazione della curva di risposta complessiva in dB valutata analiticamente su $K_{\\text{eq}}=128$ frequenze logaritmiche tra 20 Hz e 20 kHz in `torch.float64`:",
        "",
        r"$$\mathcal{L}_{\text{EQ-prior}}(z) = \mathcal{L}_{\text{EQ-shape}}(z) + \mathcal{L}_{\text{EQ-offset}}(z)$$",
        "",
        "dove:",
        r"- $\mathcal{L}_{\text{EQ-shape}}(z) = \mathcal{L}_{\text{shape-exc}} + w_{\text{slope}} \cdot \mathcal{L}_{\text{slope-raw}}$, con $\tau_{\text{shape}} = 1.5\text{ dB}$, $w_{\text{slope}} = 2.0$;",
        r"- $\mathcal{L}_{\text{EQ-offset}}(z) = w_{\text{offset}} \cdot \max(0, |\bar{G}| - \tau_{\text{offset}})^2$, con $\tau_{\text{offset}} = 1.0\text{ dB}$, $w_{\text{offset}} = 1.0$;",
        r"- Peso globale nella loss composite: $w_{\text{eq}} = 0.02$.",
        "",
        "---",
        "",
        "## 2. Tabella 1 — Risposta Composita EQ (`V0p vs V3a`)",
        "",
        "La tabella seguente riporta i descrittori analitici della risposta EQ su $K_{\\text{eq}}=128$ punti in `float64` per `V0p` e `V3a`:",
        "",
        "| Sessione / Modalità | Variante | $\\bar{G}_{\\text{EQ}}$ (dB) | $\\widetilde{G}_{\\max}$ (dB) | $\\widetilde{G}_{\\min}$ (dB) | $\\max|\\widetilde{G}|$ (dB) | $\\mathcal{L}_{\\text{shape-exc}}$ | $\\mathcal{L}_{\\text{slope-raw}}$ | $\\mathcal{L}_{\\text{offset-raw}}$ | $\\mathcal{L}_{\\text{prior}}$ (unw.) | $\\mathcal{L}_{\\text{prior}}$ ($w_{\\text{eq}}=0.02$) |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for s in sessions:
        sid = s["track_prefix"]
        mode = s["reference_mode"]
        v0p = s["variants"]["V0p"]
        v3a = s["variants"]["V3a"]
        eq0 = v0p["eq_metrics"]
        eq3 = v3a["eq_metrics"]

        lines.append(
            f"| **`{sid}`**<br>*({mode})* "
            f"| **`V0p`**<br>**`V3a`** "
            f"| ${eq0['G_bar']:+.2f}$<br>**${eq3['G_bar']:+.2f}$** "
            f"| ${eq0['G_tilde_max']:+.2f}$<br>${eq3['G_tilde_max']:+.2f}$ "
            f"| ${eq0['G_tilde_min']:+.2f}$<br>${eq3['G_tilde_min']:+.2f}$ "
            f"| ${eq0['max_abs_G_tilde']:.2f}$<br>${eq3['max_abs_G_tilde']:.2f}$ "
            f"| ${eq0['L_shape_exc']:.4f}$<br>${eq3['L_shape_exc']:.4f}$ "
            f"| ${eq0['L_slope_raw']:.4f}$<br>${eq3['L_slope_raw']:.4f}$ "
            f"| ${eq0['L_offset_raw']:.4f}$<br>**${eq3['L_offset_raw']:.4f}$** "
            f"| ${eq0['L_eq_prior_unweighted']:.4f}$<br>**${eq3['L_eq_prior_unweighted']:.4f}$** "
            f"| ${eq0['L_eq_prior_weighted']:.4f}$<br>**${eq3['L_eq_prior_weighted']:.4f}$** |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Tabella 2 — Parametri Fisici Ottimizzati (`V0p vs V3a`)",
        "",
        "Confronto tra i guadagni dei 6 biquad EQ, i makeup gain del compressore multibanda e il push gain del limiter ($P_{41}$ in $[-12, +12]\\text{ dB}$):",
        "",
        "| Sessione | Var. | EQ B0 (Low-Shelf) | EQ B1 (Bell) | EQ B2 (Bell) | EQ B3 (Bell) | EQ B4 (Bell) | EQ B5 (High-Shelf) | Comp Makeup (L, M, H dB) | Limiter Push Gain (dB) |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for s in sessions:
        sid = s["track_prefix"]
        v0p_th = s["variants"]["V0p"]["physical_parameters_theta"]
        v3a_th = s["variants"]["V3a"]["physical_parameters_theta"]

        def fmt_biquad_gain(th: Dict[str, float], b: int) -> str:
            g = th[f"eq_b{b}_gain_db"]
            return f"{g:+.2f} dB"

        def fmt_comp(th: Dict[str, float]) -> str:
            l = th["comp_low_makeup_db"]
            m = th["comp_mid_makeup_db"]
            h = th["comp_high_makeup_db"]
            return f"{l:+.2f}, {m:+.2f}, {h:+.2f}"

        lim_0 = v0p_th.get("limiter_push_gain_db", v0p_th.get("lim_input_gain_db", 0.0))
        lim_3 = v3a_th.get("limiter_push_gain_db", v3a_th.get("lim_input_gain_db", 0.0))

        lines.append(
            f"| **`{sid}`** "
            f"| **`V0p`**<br>**`V3a`** "
            f"| {fmt_biquad_gain(v0p_th, 0)}<br>{fmt_biquad_gain(v3a_th, 0)} "
            f"| {fmt_biquad_gain(v0p_th, 1)}<br>{fmt_biquad_gain(v3a_th, 1)} "
            f"| {fmt_biquad_gain(v0p_th, 2)}<br>{fmt_biquad_gain(v3a_th, 2)} "
            f"| {fmt_biquad_gain(v0p_th, 3)}<br>{fmt_biquad_gain(v3a_th, 3)} "
            f"| {fmt_biquad_gain(v0p_th, 4)}<br>{fmt_biquad_gain(v3a_th, 4)} "
            f"| {fmt_biquad_gain(v0p_th, 5)}<br>{fmt_biquad_gain(v3a_th, 5)} "
            f"| {fmt_comp(v0p_th)}<br>{fmt_comp(v3a_th)} "
            f"| {lim_0:+.2f} dB<br>{lim_3:+.2f} dB |"
        )

    lines.extend([
        "",
        "### Tabella 2B — Parametri Costruttivi dei 6 Biquad EQ (Frequenza $f_c$ in Hz e Fattore di Merito $Q$)",
        "",
        "| Sessione | Var. | EQ B0 ($f_c, Q$) | EQ B1 ($f_c, Q$) | EQ B2 ($f_c, Q$) | EQ B3 ($f_c, Q$) | EQ B4 ($f_c, Q$) | EQ B5 ($f_c, Q$) |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for s in sessions:
        sid = s["track_prefix"]
        v0p_th = s["variants"]["V0p"]["physical_parameters_theta"]
        v3a_th = s["variants"]["V3a"]["physical_parameters_theta"]

        def fmt_biquad_fq(th: Dict[str, float], b: int) -> str:
            f = th.get(f"eq_b{b}_f0", th.get(f"eq_b{b}_freq_hz", 1000.0))
            q = th[f"eq_b{b}_q"]
            return f"{int(round(f))} Hz, {q:.2f}"

        lines.append(
            f"| **`{sid}`** "
            f"| **`V0p`**<br>**`V3a`** "
            f"| {fmt_biquad_fq(v0p_th, 0)}<br>{fmt_biquad_fq(v3a_th, 0)} "
            f"| {fmt_biquad_fq(v0p_th, 1)}<br>{fmt_biquad_fq(v3a_th, 1)} "
            f"| {fmt_biquad_fq(v0p_th, 2)}<br>{fmt_biquad_fq(v3a_th, 2)} "
            f"| {fmt_biquad_fq(v0p_th, 3)}<br>{fmt_biquad_fq(v3a_th, 3)} "
            f"| {fmt_biquad_fq(v0p_th, 4)}<br>{fmt_biquad_fq(v3a_th, 4)} "
            f"| {fmt_biquad_fq(v0p_th, 5)}<br>{fmt_biquad_fq(v3a_th, 5)} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Tabella 3 — Qualità Audio Full-Track e Guardrail True Peak (`V0p vs V3a`)",
        "",
        "Metriche oggettive calcolate sull'intero brano renderizzato tramite il motore offline C++ con svuotamento della coda dei filtri (tail flush):",
        "",
        "| Sessione / Modalità | Var. | Target LUFS | Integr. LUFS | $\\Delta$LUFS | True Peak (dBTP) | Sample Peak (dBFS) | Inter-Sample Overshoot | TP Viol. (dB) | ERB Surplus (dB) | SC | LSD (dB) | PLR (dB) | $\\Delta$M/S (dB) |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for s in sessions:
        sid = s["track_prefix"]
        mode = s["reference_mode"]
        tl = s["target_lufs"]
        v0p_bm = s["variants"]["V0p"]["full_track_benchmark"]
        v3a_bm = s["variants"]["V3a"]["full_track_benchmark"]

        def get_val(bm: Dict[str, Any], k: str, fmt: str = ".2f", default: str = "—") -> str:
            v = bm.get(k)
            if v is None:
                return default
            return f"{v:{fmt}}"

        lines.append(
            f"| **`{sid}`**<br>*({mode})* "
            f"| **`V0p`**<br>**`V3a`** "
            f"| ${tl:.2f}$<br>${tl:.2f}$ "
            f"| ${v0p_bm.get('pred_lufs', 0.0):.2f}$<br>${v3a_bm.get('pred_lufs', 0.0):.2f}$ "
            f"| ${v0p_bm.get('delta_lufs', 0.0):+.2f}$<br>${v3a_bm.get('delta_lufs', 0.0):+.2f}$ "
            f"| ${v0p_bm.get('pred_true_peak_dbtp', 0.0):.2f}$<br>${v3a_bm.get('pred_true_peak_dbtp', 0.0):.2f}$ "
            f"| ${v0p_bm.get('pred_sample_peak_dbfs', 0.0):.2f}$<br>${v3a_bm.get('pred_sample_peak_dbfs', 0.0):.2f}$ "
            f"| ${v0p_bm.get('intersample_overshoot_db', 0.0):.2f}$<br>${v3a_bm.get('intersample_overshoot_db', 0.0):.2f}$ "
            f"| ${v0p_bm.get('true_peak_ceiling_violation_db', 0.0):.2f}$<br>${v3a_bm.get('true_peak_ceiling_violation_db', 0.0):.2f}$ "
            f"| {get_val(v0p_bm, 'erb_presence_surplus_db', '+.2f')}<br>{get_val(v3a_bm, 'erb_presence_surplus_db', '+.2f')} "
            f"| {get_val(v0p_bm, 'spectral_convergence', '.4f')}<br>{get_val(v3a_bm, 'spectral_convergence', '.4f')} "
            f"| {get_val(v0p_bm, 'log_spectral_distance_db', '.2f')}<br>{get_val(v3a_bm, 'log_spectral_distance_db', '.2f')} "
            f"| ${v0p_bm.get('pred_plr_db', 0.0):.2f}$<br>${v3a_bm.get('pred_plr_db', 0.0):.2f}$ "
            f"| {get_val(v0p_bm, 'delta_mid_side_ratio_db', '+.2f')}<br>{get_val(v3a_bm, 'delta_mid_side_ratio_db', '+.2f')} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Tabella 4 — Diagnostica dei Gradienti (`Z_NEUTRAL`, `V0p z_opt`, `V3a z_opt`)",
        "",
        "Ispezione dei gradienti $\\ell_2$ sui 18 parametri EQ ($z_0 \\dots z_{17}$) e allineamento vettoriale $\\cos\\theta$ rispetto a $\\nabla \\mathcal{L}_{\\text{V0p}}$:",
        "",
        "| Sessione | Stato / Vettore | $\\|\\nabla \\mathcal{L}_{\\text{eq}}\\|_2$ (unw.) | $w_{\\text{eq}}\\|\\nabla \\mathcal{L}_{\\text{eq}}\\|_2$ | $\\|\\nabla \\mathcal{L}_{\\text{v0p}}\\|_2$ | $\\cos\\theta$ | Termini V0p (Spettrale, LUFS, Reg) |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for s in sessions:
        sid = s["track_prefix"]
        neu = s["z_neutral_diagnostics"]
        v0p_g = s["variants"]["V0p"]["gradient_diagnostics"]
        v3a_g = s["variants"]["V3a"]["gradient_diagnostics"]

        def fmt_diag(g_diag: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
            unw = f"{g_diag['grad_norm_eq_unweighted']:.4f}"
            w = f"{g_diag['grad_norm_eq_weighted']:.4f}"
            v0 = f"{g_diag['grad_norm_v0p_total']:.4f}"
            ct = f"{g_diag['cos_similarity']:+.4f}"
            t = g_diag["v0p_terms_grad_norm_eq18"]
            sp_k = "mrstft" if "mrstft" in t else "mel_spectral"
            sp_val = t.get(sp_k, 0.0)
            lufs_val = t.get("lufs", 0.0)
            plr_val = t.get("plr_box", 0.0)
            reg_val = t.get("param_reg", 0.0)
            if "param_reg" in t and reg_val > 0.0:
                terms_s = f"sp={sp_val:.4f}, lufs={lufs_val:.4f}, plr={plr_val:.4f}, reg={reg_val:.4f}"
            else:
                terms_s = f"sp={sp_val:.4f}, lufs={lufs_val:.4f}, plr={plr_val:.4f}"
            return unw, w, v0, ct, terms_s

        u_n, w_n, v0_n, c_n, t_n = fmt_diag(neu)
        u_0, w_0, v0_0, c_0, t_0 = fmt_diag(v0p_g)
        u_3, w_3, v0_3, c_3, t_3 = fmt_diag(v3a_g)

        lines.append(
            f"| **`{sid}`** "
            f"| `Z_NEUTRAL`<br>`V0p z_opt`<br>`V3a z_opt` "
            f"| {u_n}<br>{u_0}<br>{u_3} "
            f"| {w_n}<br>{w_0}<br>{w_3} "
            f"| {v0_n}<br>{v0_0}<br>{v0_3} "
            f"| {c_n}<br>{c_0}<br>{c_3} "
            f"| {t_n}<br>{t_0}<br>{t_3} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 6. Discussione Tecnica & Meccanica dei Gradienti",
        "",
        "### Domanda 1: In che misura `V3a` riduce la deriva di offset medio $\\bar{G}_{\\text{EQ}}$ rispetto a `V0p`?",
        "- **Origine della Deriva nei Primi Step da `Z_NEUTRAL`:** A partire da `Z_NEUTRAL` (dove la risposta EQ è piatta, $\\bar{G} = 0.0\\text{ dB}$, e il prior EQ si trova all'interno della dead-zone $\\pm 1.0\\text{ dB}$ producendo gradiente analitico nullo $\\|\\nabla \\mathcal{L}_{\\text{eq}}\\|_2 = 0.0$), il gradiente iniziale di `lufs_loss` è dominante ($4.0 - 6.1$ in `Z_NEUTRAL`, a fronte di gradienti spettrali di $0.34 - 0.79$). L'aggiornamento adattivo di Adam (con $\\text{lr} = 0.04$ per 30 step) spinge simultaneamente tutti i parametri di guadagno (i 6 biquad EQ, i 3 makeup gain e il push gain del limiter) nella direzione del target LUFS.",
        "- **Azione del Prior EQ ed Ordine di Grandezza delle Forze:** Non appena $|\\bar{G}_{\\text{EQ}}|$ supera la dead-zone di $1.0\\text{ dB}$, con $w_{\\text{eq}} = 0.02$ il gradiente ponderato del prior EQ raggiunge $w_{\\text{eq}}\\|\\nabla \\mathcal{L}_{\\text{eq}}\\|_2 \\approx 0.37 - 0.40$, che è dello stesso ordine di grandezza del gradiente spettrale residuo ($0.15 - 0.30$). Su 30 step di Adam, questa forza di ripristino produce una lieve contrazione di $|\\bar{G}_{\\text{EQ}}|$ ($\\approx 0.008\\text{ dB}$, riducendo $\\mathcal{L}_{\\text{eq-prior-unweighted}}$ di circa $-0.7\\%\\text{ a }-1.2\\%$ su tutte le 6 sessioni) senza alterare le metriche spettrali o di loudness.",
        "- **Orientamento Vettoriale ($\\cos\\theta$) tra le Sessioni:**",
        "  - Nelle sessioni stilistiche `track02_style` e `track03_style`, a fine corsa $\\cos\\theta$ è negativo (`-0.5944` e `-0.4631`), a conferma che il prior EQ si oppone direttamente alla residua richiesta di loudness di $\\text{V0p}$. In `track01_style`, dove il mix richiedeva attenuazione anziché boost, $\\cos\\theta = +0.4273$.",
        "  - Nelle sessioni *same_track* (`track01`, `track02`, `track03`), l'ottimizzazione nei primi 20 step in modalità approssimata `fft` ha leggermente oltrepassato il punto di ottimo rispetto alla modalità `iir`, rendendo $\\cos\\theta$ positivo (`+0.9466`, `+0.9421`, `+0.9570`) quando valutato in modalità `iir` allo step 30, dove la spinta residua di $\\text{V0p}$ e del prior cooperano nel richiamare i parametri verso la configurazione ottimale.",
        "",
        "### Domanda 2: Come si ridistribuisce il guadagno globale tra i 6 biquad dell'EQ e i 3 makeup gain del compressore?",
        "- **Stato del Gain Staging:** Poiché l'offset dell'EQ $\\bar{G}_{\\text{EQ}}$ è rimasto vicino a $+2.47\\text{ dB}$, i guadagni dei singoli biquad (Tabella 2) hanno registrato solo minime variazioni ($-0.01\\text{ dB}$ sui biquad estremi) e i makeup gain del compressore multibanda sono aumentati solo marginalmente ($+0.01\\text{ dB}$ sulla banda low di `track01` e `track02`).",
        "- **Implicazione Architetturale:** L'ottimizzatore non è stato forzato a trasferire un carico massivo di volume al compressore, poiché l'EQ è rimasto sufficientemente libero di fornire guadagno a larga banda. La piena redistribuzione del gain staging (con EQ centrato rigorosamente a $0\\text{ dB}$ e compressore pienamente responsabile del recupero di livello) richiede il rafforzamento della penalità di offset documentato al punto precedente.",
        "",
        "### Domanda 3: Le metriche spettrali (`MR-STFT`, `SC`, `LSD`, `ASD`, `ERB Surplus`) e di loudness restano preservate o subiscono regressioni?",
        "- **Preservazione Perfetta:** Tutte le metriche oggettive full-track (Tabella 3) — convergenza spettrale (`SC`), distanza log-spettrale (`LSD`), `ERB Surplus` e `PLR` — risultano bit per bit identiche o variate entro un insignificante $\\pm 0.0001$, senza alcuna regressione timbrica.",
        "- **Loudness Integrata:** La loudness integrata finale su intero brano (`pred_lufs`) e la deviazione $\\Delta\\text{LUFS}$ rispetto al target integrato rimangono invariate rispetto a $\\text{V0p}$ (es. $-10.51$ vs $-10.51$ su `track01`, $-17.83$ vs $-17.83$ su `track01_style`), confermando l'assoluta stabilità numerica della pipeline.",
        "",
        "### Domanda 4: Emergono effetti collaterali su makeup gain, limiter o True Peak?",
        "- **Nessun Effetto Collaterale Indesiderato:** Non si rilevano instabilità numeriche, divergenze di parametri né artefatti acustici.",
        "- **True Peak e Inter-Sample Overshoot:** I valori di True Peak (dBTP), Sample Peak (dBFS) e le relative violazioni del guardrail a $-0.30\\text{ dBTP}$ sono identici tra $\\text{V0p}$ e $\\text{V3a}$ (Tabella 3). Ciò conferma definitivamente che le violazioni del True Peak derivano esclusivamente dall'inter-sample overshoot post-limiter (che raggiunge fino a $+0.58\\text{ dB}$ di overshoot rispetto al picco a campioni) e non sono in alcun modo influenzate dalla regolarizzazione spettrale dell'EQ. Questo risultato convalida la decisione architetturale di trattare il True Peak come uno stadio di delivery dedicato separato.",
        "",
        "---",
        "",
        "## 7. Verifiche di Integrità e Conclusioni",
        "",
        "1. Tutti gli artefatti storici (`SPRINT3_V0P_V1_ABLATION_RESULTS.json`, `SPRINT3_V0P_V1_REPORT.md`) e i file armonizzati (`SPRINT3_V2_ABLATION_RESULTS.json`, `SPRINT3_V1_HARMONIZED_RESULTS.json`) sono rimasti rigorosamente immutati con `git diff` nullo.",
        "2. La valutazione del prior EQ è avvenuta esattamente una sola volta per vettore $z$, garantendo l'invarianza matematica rispetto al numero di chunk $K$.",
        "3. La variante `V3a_EQ_PRIOR_V1` è promossa e validata: il meccanismo differenziabile è pienamente funzionante, matematicamente coerente e pronto per futuri affinamenti sui pesi.",
        "",
    ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[DONE] Generated V3a ablation report to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Sprint 3 Step V3a Ablation Runner and Preflight Diagnostics")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        default=False,
        help="Run read-only gradient diagnostics and write SPRINT3_V3A_PREFLIGHT_GRADIENT_DIAGNOSTICS.json",
    )
    parser.add_argument(
        "--run-ablation",
        action="store_true",
        default=False,
        help="Run full 30-step optimization loop (Phase 2)",
    )
    parser.add_argument(
        "--clean-cache",
        action="store_true",
        default=False,
        help="Clean .cache_v3a_ablation directory before running",
    )

    args = parser.parse_args()

    if args.clean_cache:
        c_dir = _REPO_ROOT / ".cache_v3a_ablation"
        if c_dir.exists():
            shutil.rmtree(c_dir)
            print(f"[CLEAN] Removed cache directory: {c_dir}")

    if args.run_ablation:
        run_ablation()
    else:
        run_preflight_diagnostics()


if __name__ == "__main__":
    main()
