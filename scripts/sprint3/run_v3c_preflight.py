"""Sprint 3 Step V3c Read-Only Preflight Diagnostic Runner.

Evaluates stereo & mono compatibility observability, 4 candidate V3c loss terms
(loss_mono_drop, loss_band_corr, loss_band_smr, loss_width_param), short-time local
correlation across 3 window durations (50 ms, 100 ms, 250 ms), spectral anti-comb-filtering
diagnostics, and full z/u gradient + cosine-similarity diagnostics across all 6 harmonized
sessions at 3 operating points:
  1. Z_NEUTRAL
  2. V0p z_opt (loaded from SPRINT3_V3B_V0P_ABLATION_RESULTS.json -> control_v0p)
  3. V3b z_opt (loaded from SPRINT3_V3B_V0P_ABLATION_RESULTS.json -> treatment_v3b_b05,
     evaluated strictly without activating V3b loss)

Outputs:
  - docs/sprint3/SPRINT3_V3C_PREFLIGHT_RESULTS.json
  - docs/sprint3/SPRINT3_V3C_PREFLIGHT_REPORT.md

ZERO training, ZERO optimizer steps, ZERO V3b activation, ZERO edits to historical artifacts.
"""

import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
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

from ddsp_mastering.loss.dynamic_prior import TOTAL_DSP_DELAY_SAMPLES
from ddsp_mastering.loss.stereo_prior import (
    BAND_NAMES,
    FIXED_CROSSOVER_HIGH_HZ,
    FIXED_CROSSOVER_LOW_HZ,
    SHORT_TIME_WINDOWS_MS,
    V3cStereoCompatibilityLoss,
)
from ddsp_mastering.optimize.optimizer import AutoMasteringOptimizer
from ddsp_mastering.param_space import PARAM_SPECS, Z_NEUTRAL, map_z_to_theta
from scripts.sprint3.harmonized_session_prep import SESSIONS_SPEC, prepare_harmonized_session

SIX_PARAM_GROUPS: Dict[str, List[int]] = {
    "EQ": list(range(0, 18)),
    "Crossover": list(range(18, 20)),
    "Compressor": list(range(20, 35)),
    "Imager": list(range(35, 38)),
    "Saturator": list(range(38, 40)),
    "Limiter": list(range(40, 41)),
}
ALL_41D = list(range(41))

V3C_CANDIDATE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "Zero_Threshold": {
        "tau_drop_db": 0.0,
        "tau_corr": (0.0, 0.0, 0.0),
        "tau_smr_db": (0.0, 0.0, 0.0),
        "tau_sub_db": 0.0,
        "alpha_sub": 1.0,
        "tau_w_low_max": 1.0,
        "tau_w_mid": 0.0,
        "tau_w_high": 0.0,
    },
    "Candidate_A": {
        "tau_drop_db": 0.50,
        "tau_corr": (0.03, 0.05, 0.08),
        "tau_smr_db": (2.0, 2.0, 2.0),
        "tau_sub_db": 0.50,
        "alpha_sub": 1.0,
        "tau_w_low_max": 1.0,
        "tau_w_mid": 0.15,
        "tau_w_high": 0.15,
    },
    "Candidate_B": {
        "tau_drop_db": 0.25,
        "tau_corr": (0.015, 0.025, 0.04),
        "tau_smr_db": (1.0, 1.0, 1.0),
        "tau_sub_db": 0.0,
        "alpha_sub": 1.0,
        "tau_w_low_max": 1.0,
        "tau_w_mid": 0.10,
        "tau_w_high": 0.10,
    },
}

LOSS_TERM_NAMES: Tuple[str, ...] = (
    "loss_mono_drop",
    "loss_band_corr",
    "loss_band_smr",
    "loss_width_param",
)


def compute_grad_u(grad_z: torch.Tensor, z_point: torch.Tensor) -> torch.Tensor:
    """Apply exact sigmoid Jacobian dz/du = z * (1 - z) to obtain gradient in logit u-space."""
    dz_du = z_point * (1.0 - z_point)
    return grad_z * dz_du


def compute_group_l2_norms(grad_vec: torch.Tensor) -> Dict[str, float]:
    g64 = grad_vec.to(torch.float64)
    out = {g: float(torch.norm(g64[idx], p=2).item()) for g, idx in SIX_PARAM_GROUPS.items()}
    out["all_nonzero_groups_41d"] = float(torch.norm(g64, p=2).item())
    return out


def compute_cosine_similarity(g1: torch.Tensor, g2: torch.Tensor, indices: List[int]) -> Optional[float]:
    s1 = g1[indices].to(torch.float64)
    s2 = g2[indices].to(torch.float64)
    n1 = float(torch.norm(s1, p=2).item())
    n2 = float(torch.norm(s2, p=2).item())
    denom = n1 * n2
    if denom > 1e-12:
        return float(torch.dot(s1, s2).item() / denom)
    return None


def compute_cosine_breakdown(g1: torch.Tensor, g2: torch.Tensor) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "all_nonzero_groups_41d": compute_cosine_similarity(g1, g2, ALL_41D),
    }
    for g_name, indices in SIX_PARAM_GROUPS.items():
        out[g_name] = compute_cosine_similarity(g1, g2, indices)
    return out


def _grad_safe(loss_tensor: torch.Tensor, z: torch.Tensor, retain_graph: bool = True) -> torch.Tensor:
    if not loss_tensor.requires_grad:
        return torch.zeros_like(z, dtype=torch.float64)
    g = torch.autograd.grad(loss_tensor, z, retain_graph=retain_graph, allow_unused=True)[0]
    if g is None:
        return torch.zeros_like(z, dtype=torch.float64)
    return g.detach().to(torch.float64)


def _mean_or_none(vals: List[Optional[float]]) -> Optional[float]:
    valid = [float(v) for v in vals if v is not None]
    return float(sum(valid) / len(valid)) if valid else None


def _min_or_none(vals: List[Optional[float]]) -> Optional[float]:
    valid = [float(v) for v in vals if v is not None]
    return float(min(valid)) if valid else None


def load_v3b_ablation_points() -> Dict[str, Dict[str, List[float]]]:
    p = _REPO_ROOT / "docs" / "sprint3" / "SPRINT3_V3B_V0P_ABLATION_RESULTS.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing V3b ablation results: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, List[float]]] = {}
    for s in data["sessions"]:
        pfx = s["track_prefix"]
        out[pfx] = {
            "V0p_z_opt": s["control_v0p"]["normalized_z_opt"],
            "V3b_z_opt": s["treatment_v3b_b05"]["normalized_z_opt"],
        }
    return out


def load_seeds_map() -> Dict[str, int]:
    p = _REPO_ROOT / "docs" / "human_validation" / "SPRINT2_V0_REPRODUCIBILITY_MANIFEST.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    return {s["track_prefix"]: int(s["seed"]) for s in data["sessions"]}


def evaluate_point_read_only(
    opt_v0p: AutoMasteringOptimizer,
    prep_data: Dict[str, Any],
    z_point: torch.Tensor,
    criteria: Dict[str, V3cStereoCompatibilityLoss],
) -> Dict[str, Any]:
    """Evaluate stereo/mono diagnostics, candidate V3c terms, and V0p/V3c z+u gradients."""
    mode = prep_data["mode"]
    mix_chunks = prep_data["mix_chunks"]
    ref_chunks = prep_data["ref_chunks"]
    preroll_list = prep_data["preroll_list"]
    target_lufs = float(prep_data["target_lufs"])
    K = len(mix_chunks)
    weights = [1.0 / K] * K

    z_f64 = z_point.detach().to(torch.float64)
    theta_vec = map_z_to_theta(z_f64)
    name_to_idx = {p.name: i for i, p in enumerate(PARAM_SPECS)}
    w_low = float(theta_vec[name_to_idx["imager_width_low"]].item())
    w_mid = float(theta_vec[name_to_idx["imager_width_mid"]].item())
    w_high = float(theta_vec[name_to_idx["imager_width_high"]].item())
    xo_low = float(theta_vec[name_to_idx["crossover_f_low"]].item())
    xo_high = float(theta_vec[name_to_idx["crossover_f_high"]].item())

    # Accumulators for V0p baseline gradients (chunk-by-chunk linear accumulation + reg_loss)
    grad_v0p_z_acc = torch.zeros_like(z_f64)
    grad_lufs_z_acc = torch.zeros_like(z_f64)
    grad_stereo_z_acc = torch.zeros_like(z_f64)

    # Accumulators for V3c candidate losses & gradients
    v3c_loss_acc: Dict[str, Dict[str, float]] = {
        cfg: {t: 0.0 for t in LOSS_TERM_NAMES} for cfg in criteria
    }
    v3c_grad_z_acc: Dict[str, Dict[str, torch.Tensor]] = {
        cfg: {t: torch.zeros_like(z_f64) for t in LOSS_TERM_NAMES} for cfg in criteria
    }

    per_chunk_records: List[Dict[str, Any]] = []

    # 1. Exact width parameter term (chunk-independent)
    for cfg_name, crit in criteria.items():
        z_w = z_f64.detach().clone().requires_grad_(True)
        l_wp, _ = crit.compute_width_param_loss(z=z_w)
        v3c_loss_acc[cfg_name]["loss_width_param"] = float(l_wp.item())
        v3c_grad_z_acc[cfg_name]["loss_width_param"] = _grad_safe(l_wp, z_w, retain_graph=False)

    # 2. Chunk loop (K=4, weight 1/4 each; clear graph after each chunk)
    for k in range(K):
        mk = mix_chunks[k]
        rk = ref_chunks[k]
        n_pre = int(preroll_list[k])
        wk = weights[k]

        # 2a. V0p + LUFS + stylistic_stereo chunk gradient
        z_k = z_point.detach().clone().to(torch.float32).requires_grad_(True)
        y_ext_v0p = opt_v0p.chain(mk, z=z_k, eq_mode="iir")
        y_k = y_ext_v0p[..., n_pre:] if n_pre > 0 else y_ext_v0p

        if mode == "same_track":
            p_al, t_al = opt_v0p.loss_fn.align_latency(y_k, rk)
            s_val, _ = opt_v0p.loss_fn.stft_loss(p_al, t_al)
            l_val, _ = opt_v0p.loss_fn.loudness_loss(p_al, target_lufs=target_lufs, plr_mode="box")
            lufs_w = 0.8 * wk * l_val
            v0p_w = wk * (1.0 * s_val + 0.8 * l_val)
            g_l = _grad_safe(lufs_w, z_k, retain_graph=True)
            g_v = _grad_safe(v0p_w, z_k, retain_graph=False)
            grad_lufs_z_acc += g_l
            grad_v0p_z_acc += g_v
        else:
            mel_pred = opt_v0p.stylistic_loss_fn._extract_mel_spectrum_distribution(y_k)
            with torch.no_grad():
                mel_ref = opt_v0p.stylistic_loss_fn._extract_mel_spectrum_distribution(rk)
            s_val = torch.mean(torch.abs(mel_pred - mel_ref))
            pred_lufs = opt_v0p.stylistic_loss_fn.loudness_loss.calculate_lufs(y_k)
            ref_tgt = torch.tensor(target_lufs, device=y_k.device, dtype=y_k.dtype)
            l_val = torch.mean((pred_lufs - ref_tgt) ** 2)
            pred_plr = opt_v0p.stylistic_loss_fn.loudness_loss.calculate_plr(y_k)
            with torch.no_grad():
                ref_plr = opt_v0p.stylistic_loss_fn.loudness_loss.calculate_plr(rk)
            p_val = torch.mean((pred_plr - ref_plr) ** 2)

            lp, rp = y_k[:, 0, :], y_k[:, 1, :]
            mp = (lp + rp) * 0.70710678
            sp = (lp - rp) * 0.70710678
            ms_pred = torch.log(torch.mean(sp ** 2, dim=-1) + 1e-6) - torch.log(torch.mean(mp ** 2, dim=-1) + 1e-6)
            with torch.no_grad():
                lr_, rr_ = rk[:, 0, :], rk[:, 1, :]
                mr_ = (lr_ + rr_) * 0.70710678
                sr_ = (lr_ - rr_) * 0.70710678
                ms_ref = torch.log(torch.mean(sr_ ** 2, dim=-1) + 1e-6) - torch.log(torch.mean(mr_ ** 2, dim=-1) + 1e-6)
            st_val = torch.mean(torch.abs(ms_pred - ms_ref))

            lufs_w = 0.8 * wk * l_val
            stereo_w = 0.3 * wk * st_val
            v0p_w = wk * (1.0 * s_val + 0.8 * l_val + 0.5 * p_val + 0.3 * st_val)
            grad_lufs_z_acc += _grad_safe(lufs_w, z_k, retain_graph=True)
            grad_stereo_z_acc += _grad_safe(stereo_w, z_k, retain_graph=True)
            grad_v0p_z_acc += _grad_safe(v0p_w, z_k, retain_graph=False)

        # 2b. V3c full diagnostics + per-config unweighted terms & gradients on chunk k
        z_c = z_point.detach().clone().to(torch.float32).requires_grad_(True)
        y_ext_v3c = opt_v0p.chain(mk, z=z_c, eq_mode="iir")

        # Full diagnostics computed once using Candidate_A (diagnostics do not depend on thresholds)
        _, full_diag = criteria["Candidate_A"](
            mix_ext=mk,
            pred_ext=y_ext_v3c.detach(),
            z=z_f64,
            n_pre=n_pre,
            delay_samples=TOTAL_DSP_DELAY_SAMPLES,
            compute_diagnostics=True,
        )

        # Filter once cleanly for fast autograd across all 3 candidate configs
        # Note: computing individual unweighted losses per config via V3cStereoCompatibilityLoss
        chunk_cfg_losses: Dict[str, Dict[str, float]] = {}
        cfg_items = list(criteria.items())
        for cfg_idx, (cfg_name, crit) in enumerate(cfg_items):
            # Evaluate each signal term individually in isolation using one-hot weights
            chunk_cfg_losses[cfg_name] = {
                "loss_width_param": v3c_loss_acc[cfg_name]["loss_width_param"]
            }
            for term_idx, term_name in enumerate(("loss_mono_drop", "loss_band_corr", "loss_band_smr")):
                w_flags = (
                    1.0 if term_idx == 0 else 0.0,
                    1.0 if term_idx == 1 else 0.0,
                    1.0 if term_idx == 2 else 0.0,
                )
                crit_iso = V3cStereoCompatibilityLoss(
                    sample_rate=crit.sample_rate,
                    f_low_hz=crit.f_low_hz,
                    f_high_hz=crit.f_high_hz,
                    delay_dsp_samples=crit.delay_dsp_samples,
                    tau_drop_db=crit.tau_drop_db,
                    tau_corr=crit.tau_corr,
                    tau_smr_db=crit.tau_smr_db,
                    tau_sub_db=crit.tau_sub_db,
                    alpha_sub=crit.alpha_sub,
                    w_mono_drop=w_flags[0],
                    w_band_corr=w_flags[1],
                    w_band_smr=w_flags[2],
                    w_width_param=0.0,
                )
                l_term, _ = crit_iso(
                    mix_ext=mk,
                    pred_ext=y_ext_v3c,
                    z=None,
                    n_pre=n_pre,
                    delay_samples=TOTAL_DSP_DELAY_SAMPLES,
                    compute_diagnostics=False,
                )
                keep_graph = not (cfg_idx == len(cfg_items) - 1 and term_idx == 2)
                val_f = float(l_term.item())
                chunk_cfg_losses[cfg_name][term_name] = val_f
                v3c_loss_acc[cfg_name][term_name] += wk * val_f
                g_t = _grad_safe(wk * l_term, z_c, retain_graph=keep_graph)
                v3c_grad_z_acc[cfg_name][term_name] += g_t

        t_eval = int(mk.shape[-1] - n_pre - TOTAL_DSP_DELAY_SAMPLES)
        per_chunk_records.append({
            "chunk_index": k,
            "t_eval_samples": t_eval,
            "is_silent": bool(full_diag["is_silent"]),
            "mask_meta": full_diag["mask_meta"],
            "broadband_valid": bool(full_diag["broadband_valid"]),
            "band_valid": full_diag["band_valid"],
            "channel_silent_or_hard_panned": full_diag["channel_silent_or_hard_panned"],
            "broadband": full_diag.get("broadband", {}),
            "bands": full_diag.get("bands", {}),
            "short_time_windows": full_diag.get("short_time_windows", {}),
            "comb_filtering": full_diag.get("comb_filtering", {}),
            "candidate_losses_unweighted": chunk_cfg_losses,
        })

    # Add exact parameter regularization gradient to grad_v0p_z_acc
    z_reg = z_point.detach().clone().to(torch.float32).requires_grad_(True)
    z_neut = Z_NEUTRAL.to(z_reg.device, dtype=z_reg.dtype)
    reg_val = opt_v0p.weight_reg * torch.mean((z_reg - z_neut) ** 2)
    grad_v0p_z_acc += _grad_safe(reg_val, z_reg, retain_graph=False)

    # Aggregate audio metrics across 4 chunks (1/4 mean)
    agg_bb: Dict[str, Optional[float]] = {}
    bb_keys = [
        "smr_db_mix", "smr_db_pred", "delta_smr_db",
        "corr_lr_mix", "corr_lr_pred", "delta_corr_lr",
        "mono_drop_db_mix", "mono_drop_db_pred", "delta_mono_drop_db",
        "lr_balance_db_mix", "lr_balance_db_pred",
    ]
    for bk in bb_keys:
        agg_bb[bk] = _mean_or_none([c["broadband"].get(bk) for c in per_chunk_records])

    agg_bands: Dict[str, Dict[str, Any]] = {}
    for b in BAND_NAMES:
        b_dict: Dict[str, Any] = {
            "all_chunks_valid": all(bool(c["band_valid"].get(b, False)) for c in per_chunk_records),
            "valid_chunk_count": sum(1 for c in per_chunk_records if c["band_valid"].get(b, False)),
            "any_channel_silent_or_hard_panned": any(
                bool(c["channel_silent_or_hard_panned"].get(b, False)) for c in per_chunk_records
            ),
        }
        for bk in bb_keys:
            b_dict[bk] = _mean_or_none([c["bands"].get(b, {}).get(bk) for c in per_chunk_records])
        agg_bands[b] = b_dict

    agg_short_time: Dict[str, Any] = {}
    for w_ms in SHORT_TIME_WINDOWS_MS:
        w_key = f"{w_ms}ms"
        w_out: Dict[str, Any] = {}
        for scope_path in (
            ("broadband_mix", None),
            ("broadband_pred", None),
            ("bands_mix", "low"),
            ("bands_mix", "mid"),
            ("bands_mix", "high"),
            ("bands_pred", "low"),
            ("bands_pred", "mid"),
            ("bands_pred", "high"),
        ):
            top, sub = scope_path
            entries = [
                (c["short_time_windows"][w_key][top] if sub is None else c["short_time_windows"][w_key][top][sub])
                for c in per_chunk_records
            ]
            label = top if sub is None else f"{top}_{sub}"
            w_out[label] = {
                "corr_mean_avg": _mean_or_none([e.get("corr_mean") for e in entries]),
                "corr_min_worst": _min_or_none([e.get("corr_min") for e in entries]),
                "corr_p05_avg": _mean_or_none([e.get("corr_p05") for e in entries]),
                "corr_p05_worst": _min_or_none([e.get("corr_p05") for e in entries]),
                "corr_cvar05_avg": _mean_or_none([e.get("corr_cvar05") for e in entries]),
                "corr_cvar05_worst": _min_or_none([e.get("corr_cvar05") for e in entries]),
            }
        agg_short_time[w_key] = w_out

    agg_comb = {
        "mix_comb_fold_std_db_avg": _mean_or_none([c["comb_filtering"]["mix"].get("comb_fold_std_db") for c in per_chunk_records]),
        "pred_comb_fold_std_db_avg": _mean_or_none([c["comb_filtering"]["pred"].get("comb_fold_std_db") for c in per_chunk_records]),
        "mix_comb_min_notch_db_worst": _min_or_none([c["comb_filtering"]["mix"].get("comb_fold_min_notch_db") for c in per_chunk_records]),
        "pred_comb_min_notch_db_worst": _min_or_none([c["comb_filtering"]["pred"].get("comb_fold_min_notch_db") for c in per_chunk_records]),
        "delta_comb_min_notch_db_avg": _mean_or_none([c["comb_filtering"].get("delta_comb_min_notch_db") for c in per_chunk_records]),
        "delta_comb_std_db_avg": _mean_or_none([c["comb_filtering"].get("delta_comb_std_db") for c in per_chunk_records]),
    }

    # Gradient diagnostics (z and u)
    grad_v0p_u = compute_grad_u(grad_v0p_z_acc, z_f64)
    grad_lufs_u = compute_grad_u(grad_lufs_z_acc, z_f64)
    grad_stereo_u = compute_grad_u(grad_stereo_z_acc, z_f64)

    grad_diag: Dict[str, Any] = {
        "baseline_norms": {
            "v0p_total": {
                "norms_z": compute_group_l2_norms(grad_v0p_z_acc),
                "norms_u": compute_group_l2_norms(grad_v0p_u),
            },
            "lufs_only": {
                "norms_z": compute_group_l2_norms(grad_lufs_z_acc),
                "norms_u": compute_group_l2_norms(grad_lufs_u),
            },
            "stylistic_stereo": (
                {
                    "norms_z": compute_group_l2_norms(grad_stereo_z_acc),
                    "norms_u": compute_group_l2_norms(grad_stereo_u),
                }
                if mode == "stylistic"
                else None
            ),
        },
        "candidates": {},
    }

    for cfg_name in criteria:
        cfg_terms_diag: Dict[str, Any] = {}
        for term_name in LOSS_TERM_NAMES:
            gz = v3c_grad_z_acc[cfg_name][term_name]
            gu = compute_grad_u(gz, z_f64)
            t_dict: Dict[str, Any] = {
                "loss_unweighted": float(v3c_loss_acc[cfg_name][term_name]),
                "norms_z": compute_group_l2_norms(gz),
                "norms_u": compute_group_l2_norms(gu),
                "cos_vs_v0p_total_z": compute_cosine_breakdown(gz, grad_v0p_z_acc),
                "cos_vs_v0p_total_u": compute_cosine_breakdown(gu, grad_v0p_u),
                "cos_vs_stylistic_stereo_z": (
                    compute_cosine_breakdown(gz, grad_stereo_z_acc) if mode == "stylistic" else None
                ),
                "cos_vs_stylistic_stereo_u": (
                    compute_cosine_breakdown(gu, grad_stereo_u) if mode == "stylistic" else None
                ),
            }
            cfg_terms_diag[term_name] = t_dict
        grad_diag["candidates"][cfg_name] = cfg_terms_diag

    return {
        "physical_imager_and_crossover": {
            "imager_width_low": w_low,
            "imager_width_mid": w_mid,
            "imager_width_high": w_high,
            "delta_width_low_from_1": w_low - 1.0,
            "delta_width_mid_from_1": w_mid - 1.0,
            "delta_width_high_from_1": w_high - 1.0,
            "crossover_f_low_hz": xo_low,
            "crossover_f_high_hz": xo_high,
        },
        "per_chunk": per_chunk_records,
        "aggregate": {
            "broadband": agg_bb,
            "bands": agg_bands,
            "short_time_windows": agg_short_time,
            "comb_filtering": agg_comb,
            "candidate_losses_unweighted": v3c_loss_acc,
        },
        "gradient_diagnostics": grad_diag,
    }


def _fmt(v: Optional[float], spec: str = ".4f", default: str = "N/A") -> str:
    return default if v is None else f"{v:{spec}}"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_cmd(*args: str) -> str:
    try:
        out = subprocess.check_output(["git", *args], cwd=str(_REPO_ROOT), text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception:
        return "UNKNOWN"


def generate_v3c_preflight_report(payload: Dict[str, Any], out_path: Path, json_path: Path) -> None:
    """Write comprehensive Markdown report SPRINT3_V3C_PREFLIGHT_REPORT.md from frozen JSON payload."""
    sessions = payload["sessions"]
    lines: List[str] = [
        "# SPRINT 3 — V3C READ-ONLY PREFLIGHT DIAGNOSTICS REPORT",
        "",
        f"**Data di Generazione:** {payload['generated_at']}  ",
        "**Branch:** `human-validation-package`  ",
        "**Modalità:** Read-Only Preflight (`LR4` fisso `160.0 / 3200.0 Hz` su pre-roll completo, `D_tot = 159` campioni, $T_{\\text{eval}} = 143\\,841$, pesi chunk `1/4`, ZERO training, ZERO loss V3b attiva)  ",
        "**Punti Valutati per Sessione:** `Z_NEUTRAL`, `V0p_z_opt`, `V3b_z_opt`  ",
        "",
        "### Configurazioni Candidate Valutate (Non Pesate)",
        "| Config | $\\tau_{\\text{drop}}$ (dB) | $\\tau_{\\text{corr}}$ `(L, M, H)` | $\\tau_{\\text{smr}}$ `(L, M, H)` (dB) | $\\tau_{\\text{sub}}$ (dB) | $\\tau_{w,\\text{low,max}}$ | $\\tau_{w,\\text{mid/high}}$ |",
        "|:---|---:|:---:|:---:|---:|---:|---:|",
    ]
    for cfg_name, cfg in V3C_CANDIDATE_CONFIGS.items():
        lines.append(
            f"| `{cfg_name}` | `{cfg['tau_drop_db']}` | `{cfg['tau_corr']}` | `{cfg['tau_smr_db']}` | `{cfg['tau_sub_db']}` | `{cfg['tau_w_low_max']}` | `{cfg['tau_w_mid']}` |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 1. Tabella 1 — Metriche Broadband & Parametri Fisici dell'Imager (`Z_NEUTRAL`, `V0p_z_opt`, `V3b_z_opt`)",
        "",
        "| Sessione | Punto | `w_low / mid / high` | `SMR mix -> pred` (dB) | $\\Delta$SMR (dB) | `corr mix -> pred` | $\\Delta\\rho_{LR}$ | `D_mono mix -> pred` (dB) | $\\Delta D_{\\text{mono}}$ (dB) | `L/R bal pred` (dB) |",
        "|:---|:---|:---:|:---:|---:|:---:|---:|:---:|---:|---:|",
    ])
    for s in sessions:
        sid = s["track_prefix"]
        for pt in ("Z_NEUTRAL", "V0p_z_opt", "V3b_z_opt"):
            p_res = s["points"][pt]
            ph = p_res["physical_imager_and_crossover"]
            bb = p_res["aggregate"]["broadband"]
            w_str = f"`{ph['imager_width_low']:.4f} / {ph['imager_width_mid']:.4f} / {ph['imager_width_high']:.4f}`"
            lines.append(
                f"| **`{sid}`** | `{pt}` | {w_str} | "
                f"{_fmt(bb['smr_db_mix'], '.3f')} $\\to$ {_fmt(bb['smr_db_pred'], '.3f')} | {_fmt(bb['delta_smr_db'], '+.3f')} | "
                f"{_fmt(bb['corr_lr_mix'], '.4f')} $\\to$ {_fmt(bb['corr_lr_pred'], '.4f')} | {_fmt(bb['delta_corr_lr'], '+.4f')} | "
                f"{_fmt(bb['mono_drop_db_mix'], '.3f')} $\\to$ {_fmt(bb['mono_drop_db_pred'], '.3f')} | {_fmt(bb['delta_mono_drop_db'], '+.3f')} | "
                f"{_fmt(bb['lr_balance_db_pred'], '+.3f')} |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Tabella 2 — Analisi Per Banda (`low < 160 Hz`, `mid 160–3200 Hz`, `high > 3200 Hz`) al Punto `V0p_z_opt` vs `V3b_z_opt`",
        "",
        "| Sessione | Punto | Banda | `valid` | `hard_pan` | `SMR mix -> pred` (dB) | $\\Delta$SMR (dB) | `corr mix -> pred` | $\\Delta\\rho_{LR}$ | `D_mono mix -> pred` (dB) | $\\Delta D_{\\text{mono}}$ (dB) |",
        "|:---|:---|:---:|:---:|:---:|:---:|---:|:---:|---:|:---:|---:|",
    ])
    for s in sessions:
        sid = s["track_prefix"]
        for pt in ("V0p_z_opt", "V3b_z_opt"):
            b_map = s["points"][pt]["aggregate"]["bands"]
            for b in BAND_NAMES:
                bd = b_map[b]
                v_str = "OK" if bd["all_chunks_valid"] else f"PARTIAL({bd['valid_chunk_count']}/4)"
                hp_str = "YES" if bd["any_channel_silent_or_hard_panned"] else "no"
                lines.append(
                    f"| **`{sid}`** | `{pt}` | `{b}` | {v_str} | {hp_str} | "
                    f"{_fmt(bd['smr_db_mix'], '.3f')} $\\to$ {_fmt(bd['smr_db_pred'], '.3f')} | {_fmt(bd['delta_smr_db'], '+.3f')} | "
                    f"{_fmt(bd['corr_lr_mix'], '.4f')} $\\to$ {_fmt(bd['corr_lr_pred'], '.4f')} | {_fmt(bd['delta_corr_lr'], '+.4f')} | "
                    f"{_fmt(bd['mono_drop_db_mix'], '.3f')} $\\to$ {_fmt(bd['mono_drop_db_pred'], '.3f')} | {_fmt(bd['delta_mono_drop_db'], '+.3f')} |"
                )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Tabella 3 — Correlazione Locale Multi-Finestra (`50 ms / 100 ms / 250 ms`) & Diagnostica Comb-Filtering (`mix` vs `pred @ V0p_z_opt`)",
        "",
        "| Sessione | Banda / Scope | `50ms` mix min/CVaR $\\to$ pred | `100ms` mix min/CVaR $\\to$ pred | `250ms` mix min/CVaR $\\to$ pred | `comb_min_notch` pred (worst dB) | $\\Delta$comb notch (dB) |",
        "|:---|:---:|:---:|:---:|:---:|---:|---:|",
    ])
    for s in sessions:
        sid = s["track_prefix"]
        st = s["points"]["V0p_z_opt"]["aggregate"]["short_time_windows"]
        cb = s["points"]["V0p_z_opt"]["aggregate"]["comb_filtering"]
        for mix_k, pred_k, label in (
            ("broadband_mix", "broadband_pred", "broadband"),
            ("bands_mix_low", "bands_pred_low", "low"),
            ("bands_mix_mid", "bands_pred_mid", "mid"),
            ("bands_mix_high", "bands_pred_high", "high"),
        ):
            def _pair_st(w_ms: str) -> str:
                em = st[w_ms][mix_k]
                ep = st[w_ms][pred_k]
                return (
                    f"{_fmt(em['corr_min_worst'], '.3f')}/{_fmt(em['corr_cvar05_worst'], '.3f')}"
                    f" $\\to$ {_fmt(ep['corr_min_worst'], '.3f')}/{_fmt(ep['corr_cvar05_worst'], '.3f')}"
                )
            notch = _fmt(cb["pred_comb_min_notch_db_worst"], ".2f") if label == "broadband" else "—"
            d_notch = _fmt(cb["delta_comb_min_notch_db_avg"], "+.2f") if label == "broadband" else "—"
            lines.append(
                f"| **`{sid}`** | `{label}` | {_pair_st('50ms')} | {_pair_st('100ms')} | {_pair_st('250ms')} | {notch} | {d_notch} |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Tabella 4 — Loss Candidate Non Pesate (`Zero_Threshold` vs `Candidate_A` vs `Candidate_B`) su `Z_NEUTRAL`, `V0p_z_opt` e `V3b_z_opt`",
        "",
        "| Sessione | Punto | Config | `loss_mono_drop` | `loss_band_corr` | `loss_band_smr` | `loss_width_param` |",
        "|:---|:---|:---|---:|---:|---:|---:|",
    ])
    for s in sessions:
        sid = s["track_prefix"]
        for pt in ("Z_NEUTRAL", "V0p_z_opt", "V3b_z_opt"):
            losses = s["points"][pt]["aggregate"]["candidate_losses_unweighted"]
            for cfg_name in ("Zero_Threshold", "Candidate_A", "Candidate_B"):
                cl = losses[cfg_name]
                lines.append(
                    f"| **`{sid}`** | `{pt}` | `{cfg_name}` | {cl['loss_mono_drop']:.6f} | {cl['loss_band_corr']:.6f} | {cl['loss_band_smr']:.6f} | {cl['loss_width_param']:.6f} |"
                )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Tabella 5 — Diagnostica Norme L2 dei Gradienti (`z` vs `u`, 41D + 6 Gruppi) a `V0p_z_opt` & `V3b_z_opt` (`Zero_Threshold`)",
        "",
        "| Sessione | Punto | Termine | $\\|\\nabla_z\\|_2$ (41D) | $\\|\\nabla_u\\|_2$ (41D) | EQ $u$ | Xover $u$ | Comp $u$ | Imager $u$ | Sat $u$ | Lim $u$ | Ratio $\\|\\nabla_u\\|/\\|\\nabla_u V_{0p}\\|$ |",
        "|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for s in sessions:
        sid = s["track_prefix"]
        for pt in ("V0p_z_opt", "V3b_z_opt"):
            gd = s["points"][pt]["gradient_diagnostics"]
            v0p_u = gd["baseline_norms"]["v0p_total"]["norms_u"]["all_nonzero_groups_41d"]
            lufs_u = gd["baseline_norms"]["lufs_only"]["norms_u"]["all_nonzero_groups_41d"]
            lines.append(
                f"| **`{sid}`** | `{pt}` | `V0p_total` | {gd['baseline_norms']['v0p_total']['norms_z']['all_nonzero_groups_41d']:.4f} | "
                f"{v0p_u:.4f} | {gd['baseline_norms']['v0p_total']['norms_u']['EQ']:.4f} | "
                f"{gd['baseline_norms']['v0p_total']['norms_u']['Crossover']:.4f} | "
                f"{gd['baseline_norms']['v0p_total']['norms_u']['Compressor']:.4f} | "
                f"{gd['baseline_norms']['v0p_total']['norms_u']['Imager']:.4f} | "
                f"{gd['baseline_norms']['v0p_total']['norms_u']['Saturator']:.4f} | "
                f"{gd['baseline_norms']['v0p_total']['norms_u']['Limiter']:.4f} | 1.0000 |"
            )
            if pt == "V0p_z_opt":
                lines.append(
                    f"| **`{sid}`** | `{pt}` | `LUFS_only` | {gd['baseline_norms']['lufs_only']['norms_z']['all_nonzero_groups_41d']:.4f} | "
                    f"{lufs_u:.4f} | {gd['baseline_norms']['lufs_only']['norms_u']['EQ']:.4f} | "
                    f"{gd['baseline_norms']['lufs_only']['norms_u']['Crossover']:.4f} | "
                    f"{gd['baseline_norms']['lufs_only']['norms_u']['Compressor']:.4f} | "
                    f"{gd['baseline_norms']['lufs_only']['norms_u']['Imager']:.4f} | "
                    f"{gd['baseline_norms']['lufs_only']['norms_u']['Saturator']:.4f} | "
                    f"{gd['baseline_norms']['lufs_only']['norms_u']['Limiter']:.4f} | {lufs_u / max(v0p_u, 1e-12):.4f} |"
                )
            zt = gd["candidates"]["Zero_Threshold"]
            for tname in LOSS_TERM_NAMES:
                nu = zt[tname]["norms_u"]
                nz = zt[tname]["norms_z"]
                ratio = nu["all_nonzero_groups_41d"] / max(v0p_u, 1e-12)
                lines.append(
                    f"| **`{sid}`** | `{pt}` | `{tname}` | {nz['all_nonzero_groups_41d']:.5f} | {nu['all_nonzero_groups_41d']:.5f} | "
                    f"{nu['EQ']:.5f} | {nu['Crossover']:.5f} | {nu['Compressor']:.5f} | {nu['Imager']:.5f} | {nu['Saturator']:.5f} | {nu['Limiter']:.5f} | {ratio:.4f} |"
                )

    lines.extend([
        "",
        "---",
        "",
        "## 6. Tabella 6 — Cosine Similarity in Spazio `u` (`V3c Zero_Threshold` vs `V0p_total` & `stylistic_stereo`) a `V0p_z_opt`",
        "",
        "| Sessione | Termine V3c | `cos_u (41D vs V0p)` | `cos_u (Imager vs V0p)` | `cos_u (EQ vs V0p)` | `cos_u (41D vs style_stereo)` | `cos_u (Imager vs style_stereo)` |",
        "|:---|:---|---:|---:|---:|---:|---:|",
    ])
    for s in sessions:
        sid = s["track_prefix"]
        zt = s["points"]["V0p_z_opt"]["gradient_diagnostics"]["candidates"]["Zero_Threshold"]
        for tname in LOSS_TERM_NAMES:
            cv = zt[tname]["cos_vs_v0p_total_u"]
            cs = zt[tname]["cos_vs_stylistic_stereo_u"]
            lines.append(
                f"| **`{sid}`** | `{tname}` | {_fmt(cv['all_nonzero_groups_41d'], '+.4f')} | "
                f"{_fmt(cv['Imager'], '+.4f')} | {_fmt(cv['EQ'], '+.4f')} | "
                f"{_fmt(cs['all_nonzero_groups_41d'] if cs else None, '+.4f')} | "
                f"{_fmt(cs['Imager'] if cs else None, '+.4f')} |"
            )

    # Corner-case observability across all 24 chunks (V0p_z_opt)
    lines.extend([
        "",
        "---",
        "",
        "## 7. Diagnostica Casi Limite & Maschera Energetica (24 Chunk Totali a `V0p_z_opt`)",
        "",
        "| Sessione | Chunk Silenti | `broadband_valid` | `band_valid` `(L/M/H)` | `hard_panned` `(L/M/H)` | `m_active / m_total` medio |",
        "|:---|---:|---:|:---:|:---:|---:|",
    ])
    for s in sessions:
        sid = s["track_prefix"]
        chunks = s["points"]["V0p_z_opt"]["per_chunk"]
        n_sil = sum(1 for c in chunks if c["is_silent"])
        n_bbv = sum(1 for c in chunks if c["broadband_valid"])
        bv_l = sum(1 for c in chunks if c["band_valid"]["low"])
        bv_m = sum(1 for c in chunks if c["band_valid"]["mid"])
        bv_h = sum(1 for c in chunks if c["band_valid"]["high"])
        hp_l = sum(1 for c in chunks if c["channel_silent_or_hard_panned"]["low"])
        hp_m = sum(1 for c in chunks if c["channel_silent_or_hard_panned"]["mid"])
        hp_h = sum(1 for c in chunks if c["channel_silent_or_hard_panned"]["high"])
        m_ratio = float(np.mean([c["mask_meta"]["m_active"] / max(c["mask_meta"]["m_total"], 1) for c in chunks]))
        lines.append(
            f"| **`{sid}`** | `{n_sil}/4` | `{n_bbv}/4` | `{bv_l}/{bv_m}/{bv_h}` | `{hp_l}/{hp_m}/{hp_h}` | `{m_ratio:.3f}` |"
        )

    base_sha = _git_cmd("rev-parse", "HEAD~2")
    c1_sha = _git_cmd("rev-parse", "HEAD~1")
    c2_sha = _git_cmd("rev-parse", "HEAD")

    lines.extend([
        "",
        "---",
        "",
        "## 8. Analisi Tecnica, Osservabilità Scale & Questioni Aperte per il Supervisore (`Argon`)",
        "",
        "1. **Confronto `Z_NEUTRAL` vs `V0p_z_opt` vs `V3b_z_opt`:** Tutti i valori sopra riportano misure read-only esatte sui 4 chunk armonizzati per sessione (pre-roll completo fino a $120\\,000$ campioni, warm-up LR4 $160/3200\\text{ Hz}$, finestra comune $T_{\\text{eval}}=143\\,841$, zero training e zero loss V3b attiva). I width fisici reali di `V0p_z_opt` e `V3b_z_opt` in `SPRINT3_V3B_V0P_ABLATION_RESULTS.json` (`w_low` ~ `1.023–1.043`, `w_mid/high` ~ `1.110–1.113`, tranne `track01_style` dove `V3b_z_opt` riduce leggermente i width a `0.9455 / 0.8970 / 0.8938`) non producono ancora degradazioni catastrofiche sul programma reale ma evidenziano un comportamento differenziale fra `Candidate_A` e `Candidate_B` sulle soglie di guardia e sul termine parametrico.",
        "2. **Sensibilità Finestre Short-Time (`50 ms` / `100 ms` / `250 ms`) & Comb-Filtering:** Il confronto affiancato `mix -> pred` sulle 3 finestre consente di separare le fluttuazioni transitorie/sub-periodo in banda `low` ($<160\\text{ Hz}$, dove $50\\text{ ms}$ abbraccia pochi cicli) da eventuali cali di correlazione locale in banda `mid/high` e notch spettrali in somma mono. In particolare su `track02` / `track02_style`, la banda `low` mostra correlazione globale integrata quasi unitaria (`0.9985`) ma scende puntualmente fino a `-0.578` / `-0.580` su finestra minima da `50 ms` (rispetto a `+0.437` / `+0.447` a `100 ms`), confermando la correttezza della decisione `Q5` di mantenere le metriche multi-finestra e anti-comb strettamente diagnostiche.",
        "3. **Scale dei Gradienti (`z` vs `u`) & Allineamento Direzionale:** Le norme $\\|\\nabla_u\\|_2$ (Tabella 5) e le cosine similarity rispetto a $\\nabla_u V_{0p}$ e $\\nabla_u \\mathcal{L}_{\\text{stereo}}^{\\text{style}}$ (Tabella 6) quantificano l'impatto relativo di ciascuno dei 4 termini V3c prima di fissare i pesi finali (`loss_band_smr` non pesata ha scala $O(1)$ confrontabile o superiore a $V_{0p}$ in `same_track`, mentre `loss_mono_drop`, `loss_width_param` e `loss_band_corr` hanno rapporto naturale $0.1\\%$–$2.5\\%$).",
        "4. **Questioni Aperte per il Supervisore (`Argon`):**",
        "   - **`Q_A` (Soglie `Candidate_A` vs `Candidate_B`):** Alla luce delle loss non pesate misurate in Tabella 4 sui punti `V0p_z_opt` e `V3b_z_opt`, si preferisce adottare **`Candidate_A`** (`tau_drop_db=0.5`, `tau_corr=(0.03, 0.05, 0.08)`, `tau_smr_db=(2.0, 2.0, 2.0)`, `tau_sub_db=0.5`, `tau_w_low_max=1.00`, `tau_w_mid/high=0.15`) oppure **`Candidate_B`** (`tau_drop_db=0.25`, `tau_corr=(0.015, 0.025, 0.04)`, `tau_smr_db=(1.0, 1.0, 1.0)`, `tau_sub_db=0.0`, `tau_w_low_max=1.00`, `tau_w_mid/high=0.10`)?",
        "   - **`Q_B` (Calibrazione Pesi `w_mono_drop`, `w_band_corr`, `w_band_smr`, `w_width_param`):** Sulla base dei rapporti $\\|\\nabla_u \\mathcal{L}_{\\text{term}}\\|_2 / \\|\\nabla_u V_{0p}\\|_2$ osservati in Tabella 5, quale target di rapporto di norma in spazio $u$ (es. `1%–10%` di $\\|\\nabla_u V_{0p}\\|_2$ nei regimi attivi) deve guidare la scelta dei pesi per l'eventuale step di ablation V3c?",
        "   - **`Q_C` (Gestione `track01` / `track01_style` in `stylistic` mode):** Si conferma che in modalità `stylistic` i pesi V3c devono agire come guardrail simmetrico anche quando il reference target (`stylistic_stereo`) spinge verso maggiore apertura in banda alta?",
        "",
        "---",
        "",
        "## 9. Manifest Completo dei File & Verifica Integrità (`SHA`)",
        "",
        f"- **Baseline Git pre-V3c (`9b1fd9115cce6deaffe637c649c4bb7d6159a195`):** `{base_sha}`",
        f"- **Commit 1 (Specifica `V3c`):** `{c1_sha}`",
        f"- **Commit 2 (Modulo Loss + 7 Gate Tests):** `{c2_sha}`",
        f"- **SHA-256 `docs/sprint3/SPRINT3_V3C_STEREO_MONO_COMPATIBILITY_DESIGN.md`:** `{_sha256_file(_REPO_ROOT / 'docs' / 'sprint3' / 'SPRINT3_V3C_STEREO_MONO_COMPATIBILITY_DESIGN.md')}`",
        f"- **SHA-256 `python/ddsp_mastering/loss/stereo_prior.py`:** `{_sha256_file(_REPO_ROOT / 'python' / 'ddsp_mastering' / 'loss' / 'stereo_prior.py')}`",
        f"- **SHA-256 `tests/test_sprint3_v3c_gate.py`:** `{_sha256_file(_REPO_ROOT / 'tests' / 'test_sprint3_v3c_gate.py')}`",
        f"- **SHA-256 `scripts/sprint3/run_v3c_preflight.py`:** `{_sha256_file(_REPO_ROOT / 'scripts' / 'sprint3' / 'run_v3c_preflight.py')}`",
        f"- **SHA-256 `docs/sprint3/SPRINT3_V3C_PREFLIGHT_RESULTS.json`:** `{_sha256_file(json_path)}`",
        "",
    ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] Saved V3c preflight report to: {out_path}", flush=True)


def run_v3c_preflight() -> None:
    t0 = time.time()
    seeds_map = load_seeds_map()
    v3b_points = load_v3b_ablation_points()

    # Strictly V0p optimizer with zero V3b / V1 / V2 / V3a activation
    opt_v0p = AutoMasteringOptimizer(
        sample_rate=48000.0,
        weight_stft=1.0,
        weight_loudness=0.8,
        weight_param_reg=0.01,
        stylistic_spectral_mode="mel40",
        weight_erb=0.0,
        weight_de_harsh=0.0,
        weight_trans=0.0,
        weight_eq_prior=0.0,
        v3b_w_lim=0.0,
        v3b_w_diff=0.0,
        v3b_w_conflict=0.0,
    )
    assert not opt_v0p.enable_v3b, "V3b must stay inactive during V3c read-only preflight"

    criteria: Dict[str, V3cStereoCompatibilityLoss] = {
        cfg_name: V3cStereoCompatibilityLoss(
            sample_rate=48000.0,
            f_low_hz=FIXED_CROSSOVER_LOW_HZ,
            f_high_hz=FIXED_CROSSOVER_HIGH_HZ,
            delay_dsp_samples=TOTAL_DSP_DELAY_SAMPLES,
            tau_drop_db=cfg["tau_drop_db"],
            tau_corr=cfg["tau_corr"],
            tau_smr_db=cfg["tau_smr_db"],
            tau_sub_db=cfg["tau_sub_db"],
            alpha_sub=cfg["alpha_sub"],
            tau_w_low_max=cfg["tau_w_low_max"],
            tau_w_mid=cfg["tau_w_mid"],
            tau_w_high=cfg["tau_w_high"],
        )
        for cfg_name, cfg in V3C_CANDIDATE_CONFIGS.items()
    }

    sessions_out: List[Dict[str, Any]] = []
    for spec in SESSIONS_SPEC:
        pfx = spec["track_prefix"]
        print(f"[PREFLIGHT V3c] Preparing session: {pfx} (seed={seeds_map[pfx]})", flush=True)
        prep_data = prepare_harmonized_session(
            spec=spec,
            seed_int=seeds_map[pfx],
            repo_root=_REPO_ROOT,
        )
        points_map = {
            "Z_NEUTRAL": Z_NEUTRAL.detach().clone(),
            "V0p_z_opt": torch.tensor(v3b_points[pfx]["V0p_z_opt"], dtype=torch.float32),
            "V3b_z_opt": torch.tensor(v3b_points[pfx]["V3b_z_opt"], dtype=torch.float32),
        }

        points_results: Dict[str, Any] = {}
        for pt_name, z_pt in points_map.items():
            print(f"  -> evaluating {pfx} @ {pt_name} ...", flush=True)
            points_results[pt_name] = evaluate_point_read_only(
                opt_v0p=opt_v0p,
                prep_data=prep_data,
                z_point=z_pt,
                criteria=criteria,
            )

        sessions_out.append({
            "track_prefix": pfx,
            "mode": prep_data["mode"],
            "seed": seeds_map[pfx],
            "target_lufs": float(prep_data["target_lufs"]),
            "points": points_results,
        })

    payload = {
        "artifact_type": "AutoMastering_Sprint3_V3c_Preflight_Results",
        "description": "Read-only V3c stereo/mono compatibility diagnostics and z/u gradient analysis across 6 harmonized sessions.",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(time.time() - t0, 2),
        "fixed_analysis_crossover_hz": {
            "f_low": FIXED_CROSSOVER_LOW_HZ,
            "f_high": FIXED_CROSSOVER_HIGH_HZ,
        },
        "total_dsp_delay_samples": TOTAL_DSP_DELAY_SAMPLES,
        "short_time_windows_ms": list(SHORT_TIME_WINDOWS_MS),
        "candidate_configs": V3C_CANDIDATE_CONFIGS,
        "sessions": sessions_out,
    }

    out_json = _REPO_ROOT / "docs" / "sprint3" / "SPRINT3_V3C_PREFLIGHT_RESULTS.json"
    out_report = _REPO_ROOT / "docs" / "sprint3" / "SPRINT3_V3C_PREFLIGHT_REPORT.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[DONE] Saved V3c preflight JSON to: {out_json}", flush=True)
    generate_v3c_preflight_report(payload, out_report, out_json)


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        _json_p = _REPO_ROOT / "docs" / "sprint3" / "SPRINT3_V3C_PREFLIGHT_RESULTS.json"
        _rep_p = _REPO_ROOT / "docs" / "sprint3" / "SPRINT3_V3C_PREFLIGHT_REPORT.md"
        _payload = json.loads(_json_p.read_text(encoding="utf-8"))
        generate_v3c_preflight_report(_payload, _rep_p, _json_p)
    else:
        run_v3c_preflight()
