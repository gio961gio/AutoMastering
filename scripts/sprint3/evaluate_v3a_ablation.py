"""Sprint 3 Step V3a Ablation Runner and Preflight Gradient Diagnostics.

Variant Evaluated:
  - V3a: V0p + Composite EQ Response Prior (K_eq=128, tau_shape=1.5, tau_offset=1.0,
         w_slope=2.0, w_offset=1.0, weight_eq_prior=0.02, float64 analytical response).

Modes:
  --preflight-only (default): Read-only gradient diagnostics and EQ metric evaluation
    on Z_NEUTRAL and V0p normalized_z_opt across all 6 harmonized sessions.
    Exports docs/sprint3/SPRINT3_V3A_PREFLIGHT_GRADIENT_DIAGNOSTICS.json.
  --run-ablation: 30-step Adam optimization loop across all 6 sessions (Phase 2, pending review).
"""

import argparse
import gc
import json
import math
import os
from pathlib import Path
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
from scripts.sprint3.harmonized_session_prep import SESSIONS_SPEC, prepare_harmonized_session


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
            # Stylistic comparison
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

    # Parameter Regularization
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

    # Gradients of total V0p objective (indices 0..17 EQ parameters)
    loss_v0p_total.backward(retain_graph=True)
    assert z.grad is not None, "z.grad must not be None after loss_v0p_total.backward"
    g_v0p = z.grad[:18].detach().clone().to(torch.float64)
    v0p_total_norm = float(torch.norm(g_v0p, p=2).item())

    # Individual term gradient norms on EQ parameters
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

    # Assert non-zero gradients for primary spectral and LUFS terms
    spectral_key = "mrstft" if mode == "same_track" else "mel_spectral"
    assert term_grads[spectral_key] > 0.0, f"Expected {spectral_key} > 0.0, got {term_grads[spectral_key]}"
    assert term_grads["lufs"] > 0.0, f"Expected lufs > 0.0, got {term_grads['lufs']}"

    # Cosine Similarity between EQ prior gradient and V0p gradient
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
        # Backward-compatibility aliases for diagnostic reports
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
    where:
        v0p_loss is the full objective of V0p (including reg_loss = opt.weight_reg * mean((z - z_neutral)**2))
        total_v3a_loss = v0p_loss + weight_eq_prior * L_eq_prior
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

        # 1. Assert bit-level identity of coordinates with SPRINT3_V2_ABLATION_RESULTS.json
        assert prep["target_lufs"] == v2_sess["target_lufs"], (
            f"[{prefix}] target_lufs mismatch: {prep['target_lufs']} vs {v2_sess['target_lufs']}"
        )
        assert prep["chunk_metadata"] == v2_sess["chunk_metadata"], (
            f"[{prefix}] chunk_metadata mismatch"
        )
        assert int(v2_sess["seed"]) == seed_int, (
            f"[{prefix}] seed mismatch: {v2_sess['seed']} vs {seed_int}"
        )
        print(f"[{prefix}] Chunk coordinates, seed, and target LUFS verified identical.")

        # V0p optimizer instance
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

        # 2. Objective Parity Assertion against SPRINT3_V2_ABLATION_RESULTS.json
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
        print(
            f"[{prefix}] V0p Parity: init_loss={v0p_loss_z0:.4f} (exp={exp_init:.4f}, diff={diff_init:.2e}), "
            f"final_loss={v0p_loss_opt:.4f} (exp={exp_final:.4f}, diff={diff_final:.2e})"
        )
        tol = 2e-4
        assert diff_init < tol, f"[{prefix}] initial_loss mismatch: {v0p_loss_z0} vs {exp_init}"
        assert diff_final < tol, f"[{prefix}] final_loss mismatch: {v0p_loss_opt} vs {exp_final}"

        # 3. Gradient Diagnostics at Z_NEUTRAL
        neutral_diag = compute_v3a_gradient_diagnostics(
            prep_data=prep,
            z_var=Z_NEUTRAL,
            opt_v0p=opt_v0p,
            eq_prior_loss=eq_prior_loss,
            weight_eq_prior=0.02,
        )

        # 4. Gradient Diagnostics at V0p normalized_z_opt
        v0p_diag = compute_v3a_gradient_diagnostics(
            prep_data=prep,
            z_var=v0p_z_opt,
            opt_v0p=opt_v0p,
            eq_prior_loss=eq_prior_loss,
            weight_eq_prior=0.02,
        )

        print(
            f"[{prefix}] Z_NEUTRAL: grad_norm_eq={neutral_diag['grad_norm_eq_unweighted']:.4e}, "
            f"grad_norm_v0p={neutral_diag['grad_norm_v0p_total']:.4f}, "
            f"terms={neutral_diag['v0p_terms_grad_norm_eq18']}"
        )
        print(
            f"[{prefix}] V0p z_opt: G_bar={v0p_diag['eq_metrics']['G_bar']:+.2f} dB, "
            f"G_tilde_max={v0p_diag['eq_metrics']['G_tilde_max']:+.2f} dB, "
            f"G_tilde_min={v0p_diag['eq_metrics']['G_tilde_min']:+.2f} dB, "
            f"L_eq_prior_unweighted={v0p_diag['eq_metrics']['L_eq_prior_unweighted']:.4f}, "
            f"L_eq_prior_weighted={v0p_diag['eq_metrics']['L_eq_prior_weighted']:.4f}"
        )
        print(
            f"[{prefix}] V0p z_opt Gradients: ||grad_eq||_2={v0p_diag['grad_norm_eq_unweighted']:.4f}, "
            f"w_eq*||grad_eq||_2={v0p_diag['grad_norm_eq_weighted']:.4f}, "
            f"||grad_v0p||_2={v0p_diag['grad_norm_v0p_total']:.4f}, "
            f"cos_theta={v0p_diag['cos_similarity']:+.4f}, "
            f"terms={v0p_diag['v0p_terms_grad_norm_eq18']}"
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
            # Aliases for backward compatibility:
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
    """30-step optimization loop across all 6 sessions for V3a (Phase 2, pending Argon review)."""
    print("\n" + "!" * 80)
    print("⚠️ STOP CONSTRAINT: Ablation optimization (30 steps) is blocked until Argon technical review approves Phase 1 Pre-Run diagnostics.")
    print("Do NOT run --run-ablation until authorization is received.")
    print("!" * 80 + "\n")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Sprint 3 Step V3a Ablation Runner and Preflight Diagnostics")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        default=True,
        help="Run read-only gradient diagnostics and write SPRINT3_V3A_PREFLIGHT_GRADIENT_DIAGNOSTICS.json",
    )
    parser.add_argument(
        "--run-ablation",
        action="store_true",
        default=False,
        help="Run full 30-step optimization loop (Phase 2, pending review)",
    )

    args = parser.parse_args()

    if args.run_ablation:
        run_ablation()
    else:
        run_preflight_diagnostics()


if __name__ == "__main__":
    main()
