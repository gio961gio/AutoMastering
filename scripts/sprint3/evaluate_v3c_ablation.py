"""Sprint 3 Step V3c Pilot Ablation Runner: Control V0p vs Treatment V3c (same_track only).

Authorized scope (Phase 2 — Pilot Ablation V3c vs V0p):
  - Strictly ONLY the 3 same_track sessions: track01, track02, track03.
  - Stylistic sessions (track01_style, track02_style, track03_style) remain
    strictly excluded due to structural conflict with stylistic_stereo_loss.
  - Control Arm (V0p): enable_v3c=False, sprint2 baseline weights
    (w_stft=1.0, w_lufs=0.8, w_reg=0.01), V1/V1b/V2/V3a/V3b disabled.
  - Treatment Arm (V3c): enable_v3c=True with frozen Candidate A thresholds and
    frozen pilot same_track weights (w_mono_drop=1.00, w_band_corr=10.00,
    w_band_smr=0.05, w_width_param=0.25), identical V0p params, Z_NEUTRAL init,
    30 steps Adam (lr=0.04, 1-20 fft -> 21-30 iir), 120,000-sample pre-roll,
    159-sample DSP delay, and fixed 160/3200 Hz LR4 analysis crossover.

Modes:
  Default (no flags): Dry-run verification of plan and frozen configs (exit 0).
  --run-ablation:     Run full paired ablation on track01..03, offline C++ renders,
                      blind WAV export, SHA-256 manifest, evaluator zip, and report.
  --report-only:      Regenerate SPRINT3_V3C_ABLATION_PILOT_REPORT.md from JSON.
"""

import argparse
import gc
import hashlib
import json
import math
import random
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

from ddsp_mastering.loss.dynamic_prior import TOTAL_DSP_DELAY_SAMPLES
from ddsp_mastering.loss.loudness import calculate_integrated_loudness_bs1770
from ddsp_mastering.loss.stereo_prior import (
    BAND_NAMES,
    CANDIDATE_A_THRESHOLDS,
    PILOT_SAME_TRACK_WEIGHTS,
    V3cStereoCompatibilityLoss,
)
from ddsp_mastering.optimize import AutoMasteringOptimizer, MasteringBenchmarkSuite
from ddsp_mastering.optimize.metrics import calculate_sample_peak_dbfs, calculate_true_peak_dbtp
from ddsp_mastering.param_space import PARAM_SPECS, Z_NEUTRAL, map_theta_to_z, map_z_to_theta
from scripts.human_validation.offline_renderer import render_offline_with_tail_flush
from scripts.sprint3.harmonized_session_prep import (
    SESSIONS_SPEC,
    compute_individual_loss_breakdown,
    prepare_harmonized_session,
)

PILOT_SAME_TRACK_PREFIXES: Tuple[str, ...] = ("track01", "track02", "track03")
V3C_TERM_KEYS: Tuple[str, ...] = (
    "loss_mono_drop",
    "loss_band_corr",
    "loss_band_smr",
    "loss_width_param",
)

PARAM_GROUPS: Dict[str, List[int]] = {
    "EQ": list(range(0, 18)),
    "Crossover": list(range(18, 20)),
    "Compressor": list(range(20, 35)),
    "Imager": list(range(35, 38)),
    "Saturator": list(range(38, 40)),
    "Limiter": list(range(40, 41)),
}

FROZEN_V0P_CONFIG: Dict[str, Any] = {
    "lr": 0.04,
    "num_steps": 30,
    "weight_stft": 1.0,
    "weight_loudness": 0.8,
    "weight_param_reg": 0.01,
    "use_hybrid_eq": True,
    "plr_mode": "box",
    "weight_erb": 0.0,
    "weight_de_harsh": 0.0,
    "weight_trans": 0.0,
    "weight_eq_prior": 0.0,
    "v3b_w_lim": 0.0,
    "v3b_w_diff": 0.0,
    "v3b_w_conflict": 0.0,
}

FROZEN_V3C_CONFIG: Dict[str, Any] = {
    **CANDIDATE_A_THRESHOLDS,
    **PILOT_SAME_TRACK_WEIGHTS,
}


def load_reproducibility_manifest() -> Dict[str, Any]:
    """Load canonical session seeds from Sprint 2 reproducibility manifest."""
    p = _REPO_ROOT / "docs" / "human_validation" / "SPRINT2_V0_REPRODUCIBILITY_MANIFEST.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing reproducibility manifest: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def load_harmonized_v0p_reference() -> Dict[str, Dict[str, Any]]:
    """Load harmonized V0p control parameters from SPRINT3_V3B_V0P_ABLATION_RESULTS.json."""
    p = _REPO_ROOT / "docs" / "sprint3" / "SPRINT3_V3B_V0P_ABLATION_RESULTS.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing harmonized V0p reference: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    return {s["track_prefix"]: s["control_v0p"] for s in data.get("sessions", [])}


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        v = float(val)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _mean_or_none(vals: List[Optional[float]]) -> Optional[float]:
    valid = [float(v) for v in vals if v is not None]
    return float(sum(valid) / len(valid)) if valid else None


def _min_or_none(vals: List[Optional[float]]) -> Optional[float]:
    valid = [float(v) for v in vals if v is not None]
    return float(min(valid)) if valid else None


def _assert_finite_nested(obj: Any, path: str = "root") -> None:
    if isinstance(obj, float):
        assert math.isfinite(obj), f"Non-finite float at {path}: {obj}"
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _assert_finite_nested(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_finite_nested(v, f"{path}[{i}]")


def _extract_theta_and_deltas(z_vec: torch.Tensor) -> Tuple[List[float], Dict[str, float], Dict[str, float]]:
    """Convert 41D z vector to physical theta dict and neutral deltas, verifying roundtrip < 1e-5."""
    z_det = z_vec.detach().to(torch.float32)
    z_neutral = Z_NEUTRAL.to(device=z_det.device, dtype=torch.float32)
    theta_vec = map_z_to_theta(z_det)
    theta_init = map_z_to_theta(z_neutral)

    z_roundtrip = map_theta_to_z(theta_vec)
    rt_diff = float(torch.max(torch.abs(z_roundtrip - z_det)).item())
    assert rt_diff < 1e-5, f"Param space roundtrip failed: max_diff={rt_diff}"

    theta_np = theta_vec.cpu().numpy()
    theta_init_np = theta_init.cpu().numpy()
    theta_dict = {p.name: float(theta_np[i]) for i, p in enumerate(PARAM_SPECS)}
    delta_dict = {p.name: float(theta_np[i] - theta_init_np[i]) for i, p in enumerate(PARAM_SPECS)}
    return [float(v) for v in z_det.cpu().numpy().tolist()], theta_dict, delta_dict


def _aggregate_chunk_v3c_diagnostics(
    chunk_diags: List[Dict[str, Any]],
    v3c_crit: V3cStereoCompatibilityLoss,
) -> Dict[str, Any]:
    """Aggregate V3cStereoCompatibilityLoss diagnostics across K weighted chunks."""
    n_chunks = len(chunk_diags)
    weights = [1.0 / n_chunks] * n_chunks

    raw_terms = {k: 0.0 for k in V3C_TERM_KEYS}
    for d, w in zip(chunk_diags, weights):
        for k in V3C_TERM_KEYS:
            raw_terms[k] += w * float(d[k])

    weighted_terms = {
        "loss_mono_drop": v3c_crit.w_mono_drop * raw_terms["loss_mono_drop"],
        "loss_band_corr": v3c_crit.w_band_corr * raw_terms["loss_band_corr"],
        "loss_band_smr": v3c_crit.w_band_smr * raw_terms["loss_band_smr"],
        "loss_width_param": v3c_crit.w_width_param * raw_terms["loss_width_param"],
    }
    total_weighted = sum(weighted_terms.values())

    bb_keys = (
        "e_tot_mix", "e_tot_pred",
        "corr_lr_mix", "corr_lr_pred", "delta_corr_lr",
        "smr_db_mix", "smr_db_pred", "delta_smr_db",
        "mono_drop_db_mix", "mono_drop_db_pred", "delta_mono_drop_db",
        "lr_balance_db_mix", "lr_balance_db_pred",
    )
    bb_valid_chunks = [d["broadband"] for d in chunk_diags if d.get("broadband", {}).get("valid")]
    broadband_agg = {
        "valid_chunk_count": len(bb_valid_chunks),
        **{bk: _mean_or_none([c.get(bk) for c in bb_valid_chunks]) for bk in bb_keys},
    }

    band_keys = (
        "e_tot_mix", "e_tot_pred",
        "corr_lr_mix", "corr_lr_pred", "delta_corr_lr",
        "smr_db_mix", "smr_db_pred", "delta_smr_db",
        "mono_drop_db_mix", "mono_drop_db_pred", "delta_mono_drop_db",
        "lr_balance_db_mix", "lr_balance_db_pred",
    )
    bands_agg: Dict[str, Any] = {}
    for b in BAND_NAMES:
        b_chunks = [d["bands"][b] for d in chunk_diags if d.get("bands", {}).get(b, {}).get("band_valid")]
        bands_agg[b] = {
            "valid_chunk_count": len(b_chunks),
            "pred_energy_below_emin_count": sum(
                1 for d in chunk_diags if d.get("bands", {}).get(b, {}).get("pred_energy_below_emin")
            ),
            **{bk: _mean_or_none([c.get(bk) for c in b_chunks]) for bk in band_keys},
        }

    st_agg: Dict[str, Any] = {}
    for w_ms in v3c_crit.window_durations_ms:
        w_key = f"{w_ms}ms"
        st_entries = [d["short_time_windows"][w_key] for d in chunk_diags if "short_time_windows" in d]
        st_agg[w_key] = {}
        for scope in ("broadband_mix", "broadband_pred"):
            subs = [e[scope] for e in st_entries if e.get(scope, {}).get("valid")]
            st_agg[w_key][scope] = {
                "valid_chunk_count": len(subs),
                "corr_mean": _mean_or_none([s.get("corr_mean") for s in subs]),
                "corr_min": _min_or_none([s.get("corr_min") for s in subs]),
                "corr_p05": _mean_or_none([s.get("corr_p05") for s in subs]),
                "corr_cvar05": _mean_or_none([s.get("corr_cvar05") for s in subs]),
            }
        for group_key in ("bands_mix", "bands_pred"):
            st_agg[w_key][group_key] = {}
            for b in BAND_NAMES:
                subs = [e[group_key][b] for e in st_entries if e.get(group_key, {}).get(b, {}).get("valid")]
                st_agg[w_key][group_key][b] = {
                    "valid_chunk_count": len(subs),
                    "corr_mean": _mean_or_none([s.get("corr_mean") for s in subs]),
                    "corr_min": _min_or_none([s.get("corr_min") for s in subs]),
                    "corr_p05": _mean_or_none([s.get("corr_p05") for s in subs]),
                    "corr_cvar05": _mean_or_none([s.get("corr_cvar05") for s in subs]),
                }

    comb_entries = [d["comb_filtering"] for d in chunk_diags if "comb_filtering" in d]
    comb_mix_valid = [c["mix"] for c in comb_entries if c.get("mix", {}).get("valid")]
    comb_pred_valid = [c["pred"] for c in comb_entries if c.get("pred", {}).get("valid")]
    comb_agg = {
        "mix": {
            "comb_fold_mean_db": _mean_or_none([c.get("comb_fold_mean_db") for c in comb_mix_valid]),
            "comb_fold_std_db": _mean_or_none([c.get("comb_fold_std_db") for c in comb_mix_valid]),
            "comb_fold_min_notch_db": _min_or_none([c.get("comb_fold_min_notch_db") for c in comb_mix_valid]),
            "comb_ripple_p95_p05_db": _mean_or_none([c.get("comb_ripple_p95_p05_db") for c in comb_mix_valid]),
        },
        "pred": {
            "comb_fold_mean_db": _mean_or_none([c.get("comb_fold_mean_db") for c in comb_pred_valid]),
            "comb_fold_std_db": _mean_or_none([c.get("comb_fold_std_db") for c in comb_pred_valid]),
            "comb_fold_min_notch_db": _min_or_none([c.get("comb_fold_min_notch_db") for c in comb_pred_valid]),
            "comb_ripple_p95_p05_db": _mean_or_none([c.get("comb_ripple_p95_p05_db") for c in comb_pred_valid]),
        },
        "delta_comb_min_notch_db": _mean_or_none([c.get("delta_comb_min_notch_db") for c in comb_entries]),
        "delta_comb_std_db": _mean_or_none([c.get("delta_comb_std_db") for c in comb_entries]),
    }

    last_diag = chunk_diags[-1]
    widths_physical = {
        "imager_width_low": float(last_diag.get("imager_width_low", 1.0)),
        "imager_width_mid": float(last_diag.get("imager_width_mid", 1.0)),
        "imager_width_high": float(last_diag.get("imager_width_high", 1.0)),
    }

    return {
        "loss_v3c_unweighted_sum": float(sum(raw_terms.values())),
        "loss_v3c_total_weighted": float(total_weighted),
        "terms_raw": raw_terms,
        "terms_weighted": weighted_terms,
        "imager_widths": widths_physical,
        "broadband": broadband_agg,
        "bands": bands_agg,
        "short_time_windows": st_agg,
        "comb_filtering": comb_agg,
    }


def evaluate_point_full_iir(
    opt_v0p_ref: AutoMasteringOptimizer,
    v3c_crit: V3cStereoCompatibilityLoss,
    z_point: torch.Tensor,
    prep: Dict[str, Any],
    target_lufs: float,
) -> Dict[str, Any]:
    """Read-only IIR evaluation of separate V0p objective, V3c objective, and diagnostics at z_point."""
    mix_chunks = prep["mix_chunks"]
    ref_chunks = prep["ref_chunks"]
    preroll_list = prep["preroll_list"]
    ref_ext_chunks = prep.get("ref_ext_chunks")
    ref_preroll_list = prep.get("ref_preroll_list")

    z_f32 = z_point.detach().to(torch.float32)

    with torch.no_grad():
        v0p_tot_t, v0p_metrics = opt_v0p_ref.evaluate_loss_iir(
            z=z_f32,
            mix_chunks=mix_chunks,
            ref_chunks=ref_chunks,
            preroll_samples=preroll_list,
            target_lufs=target_lufs,
            reference_mode="same_track",
            ref_ext_chunks=ref_ext_chunks,
            ref_preroll_list=ref_preroll_list,
            compute_v3c_diagnostics=False,
        )
        unweighted_breakdown = compute_individual_loss_breakdown(opt_v0p_ref.chain, z_f32, prep)

        chunk_v3c_diags: List[Dict[str, Any]] = []
        for mk, n_pre in zip(mix_chunks, preroll_list):
            y_ext = opt_v0p_ref.chain(mk, z=z_f32, eq_mode="iir")
            _, d_v3c = v3c_crit(
                mix_ext=mk,
                pred_ext=y_ext,
                z=z_f32,
                n_pre=n_pre,
                delay_samples=TOTAL_DSP_DELAY_SAMPLES,
                compute_diagnostics=True,
            )
            chunk_v3c_diags.append(d_v3c)

    v3c_agg = _aggregate_chunk_v3c_diagnostics(chunk_v3c_diags, v3c_crit)
    z_list, theta_dict, delta_neutral = _extract_theta_and_deltas(z_f32)

    loss_v0p_val = float(v0p_tot_t.item())
    loss_v3c_w = float(v3c_agg["loss_v3c_total_weighted"])

    return {
        "v0p_objective_iir": {
            "loss_v0p_total": loss_v0p_val,
            "weighted_components": {
                "stft_weighted": FROZEN_V0P_CONFIG["weight_stft"] * unweighted_breakdown["mrstft"],
                "lufs_weighted": FROZEN_V0P_CONFIG["weight_loudness"] * unweighted_breakdown["lufs"],
                "plr_box_weighted": 1.0 * unweighted_breakdown["plr_box"],
                "param_reg_weighted": FROZEN_V0P_CONFIG["weight_param_reg"] * unweighted_breakdown["param_reg"],
            },
            "unweighted_components": {
                "mrstft": unweighted_breakdown["mrstft"],
                "lufs": unweighted_breakdown["lufs"],
                "plr_box": unweighted_breakdown["plr_box"],
                "param_reg": unweighted_breakdown["param_reg"],
                "erb": unweighted_breakdown["erb"],
                "de_harsh": unweighted_breakdown["de_harsh"],
                "transient": unweighted_breakdown["transient"],
                "plr_ref_aware": unweighted_breakdown["plr_ref_aware"],
            },
            "eval_metrics_raw": {
                k: float(v) for k, v in v0p_metrics.items() if isinstance(v, (int, float))
            },
        },
        "v3c_objective_and_stereo_iir": v3c_agg,
        "combined_composite_loss_v0p_plus_v3c": float(loss_v0p_val + loss_v3c_w),
        "normalized_z": z_list,
        "physical_parameters_theta": theta_dict,
        "delta_theta_from_neutral": delta_neutral,
    }


def evaluate_full_track_stereo_diagnostics(
    raw_full_64: np.ndarray,
    mastered_64: np.ndarray,
    mix_start: int,
    eval_len: int,
    z_vec: torch.Tensor,
) -> Dict[str, Any]:
    """Evaluate V3c stereo/mono, short-time, and comb diagnostics on time-aligned full-track C++ renders."""
    crit_full = V3cStereoCompatibilityLoss(
        sample_rate=48000.0,
        delay_dsp_samples=0,
        **CANDIDATE_A_THRESHOLDS,
        **PILOT_SAME_TRACK_WEIGHTS,
    )
    mix_slice = torch.from_numpy(
        np.ascontiguousarray(raw_full_64[:, mix_start : mix_start + eval_len])
    ).unsqueeze(0)
    pred_slice = torch.from_numpy(
        np.ascontiguousarray(mastered_64[:, mix_start : mix_start + eval_len])
    ).unsqueeze(0)

    with torch.no_grad():
        _, diag = crit_full(
            mix_ext=mix_slice,
            pred_ext=pred_slice,
            z=z_vec.detach().to(torch.float64),
            n_pre=0,
            delay_samples=0,
            compute_diagnostics=True,
        )
    return diag


def compute_gradient_norms_at_point(
    opt: AutoMasteringOptimizer,
    prep: Dict[str, Any],
    z_point: torch.Tensor,
    target_lufs: float,
) -> Dict[str, Any]:
    """Compute finite z-space and u-space gradient norms for V0p, V3c (if enabled), and active total."""
    mix_chunks = prep["mix_chunks"]
    ref_chunks = prep["ref_chunks"]
    preroll_list = prep["preroll_list"]
    ref_ext_chunks = prep.get("ref_ext_chunks")
    ref_preroll_list = prep.get("ref_preroll_list")
    K = len(mix_chunks)
    weights = [1.0 / K] * K

    def _compute_for_objective(kind: str) -> Dict[str, Any]:
        z_var = torch.clamp(z_point.detach().to(torch.float32), 1e-4, 1.0 - 1e-4).requires_grad_(True)
        tot = torch.tensor(0.0, dtype=torch.float32)

        if kind == "v0p":
            tot, _ = opt.loss_fn(
                opt.chain(mix_chunks[0], z=z_var, eq_mode="iir")[..., preroll_list[0] :],
                target_audio=ref_chunks[0],
                target_lufs=target_lufs,
                normalized_params=z_var,
                align_latency=True,
                pred_ext_audio=None,
                pred_preroll_samples=preroll_list[0],
                include_eq_prior=False,
            )
            tot = weights[0] * tot
            for k in range(1, K):
                y_ext = opt.chain(mix_chunks[k], z=z_var, eq_mode="iir")
                y_k = y_ext[..., preroll_list[k] :]
                lk, _ = opt.loss_fn(
                    y_k,
                    target_audio=ref_chunks[k],
                    target_lufs=target_lufs,
                    normalized_params=z_var,
                    align_latency=True,
                    pred_ext_audio=y_ext,
                    pred_preroll_samples=preroll_list[k],
                    ref_ext_audio=ref_ext_chunks[k] if ref_ext_chunks else None,
                    ref_preroll_samples=ref_preroll_list[k] if ref_preroll_list else 0,
                    include_eq_prior=False,
                )
                tot = tot + weights[k] * lk
        elif kind == "v3c":
            assert opt.v3c_prior_loss is not None
            for k in range(K):
                mk = mix_chunks[k]
                n_pre = preroll_list[k]
                y_ext = opt.chain(mk, z=z_var, eq_mode="iir")
                l_v3c_k, _ = opt.v3c_prior_loss(
                    mix_ext=mk,
                    pred_ext=y_ext,
                    z=z_var,
                    n_pre=n_pre,
                    delay_samples=TOTAL_DSP_DELAY_SAMPLES,
                    compute_diagnostics=False,
                )
                tot = tot + weights[k] * l_v3c_k.to(torch.float32)
        else:
            tot, _ = opt.evaluate_loss_iir(
                z=z_var,
                mix_chunks=mix_chunks,
                ref_chunks=ref_chunks,
                preroll_samples=preroll_list,
                target_lufs=target_lufs,
                reference_mode="same_track",
                ref_ext_chunks=ref_ext_chunks,
                ref_preroll_list=ref_preroll_list,
                compute_v3c_diagnostics=False,
            )

        assert torch.isfinite(tot), f"Non-finite objective ({kind}): {tot}"
        gz = torch.autograd.grad(tot, z_var, retain_graph=False)[0].detach()
        assert torch.isfinite(gz).all(), f"Non-finite grad_z ({kind})"
        zc = torch.clamp(z_point.detach().to(torch.float32), 1e-4, 1.0 - 1e-4)
        gu = gz * zc * (1.0 - zc)
        assert torch.isfinite(gu).all(), f"Non-finite grad_u ({kind})"

        return {
            "loss": float(tot.item()),
            "norm_grad_z_41d": float(torch.norm(gz, p=2).item()),
            "norm_grad_u_41d": float(torch.norm(gu, p=2).item()),
            "group_norms_z": {g: float(torch.norm(gz[idx], p=2).item()) for g, idx in PARAM_GROUPS.items()},
            "group_norms_u": {g: float(torch.norm(gu[idx], p=2).item()) for g, idx in PARAM_GROUPS.items()},
        }

    out: Dict[str, Any] = {
        "total_active": _compute_for_objective("total"),
    }
    if opt.enable_v3c:
        out["v3c_only_weighted"] = _compute_for_objective("v3c")
    return out


def compute_param_shift_metrics(
    z_v3c: List[float],
    z_v0p: List[float],
    theta_v3c: Dict[str, float],
    theta_v0p: Dict[str, float],
) -> Dict[str, Any]:
    """Compute per-parameter theta shifts and per-group L2 shift norms in z and theta."""
    delta_theta = {
        p.name: float(theta_v3c[p.name] - theta_v0p[p.name]) for p in PARAM_SPECS
    }
    dz_arr = np.array(z_v3c, dtype=np.float64) - np.array(z_v0p, dtype=np.float64)
    dth_arr = np.array([delta_theta[p.name] for p in PARAM_SPECS], dtype=np.float64)

    group_l2_z = {g: float(np.linalg.norm(dz_arr[idx])) for g, idx in PARAM_GROUPS.items()}
    group_l2_theta = {g: float(np.linalg.norm(dth_arr[idx])) for g, idx in PARAM_GROUPS.items()}
    return {
        "delta_theta": delta_theta,
        "l2_norm_delta_z_41d": float(np.linalg.norm(dz_arr)),
        "l2_norm_delta_theta_41d": float(np.linalg.norm(dth_arr)),
        "group_l2_delta_z": group_l2_z,
        "group_l2_delta_theta": group_l2_theta,
    }


def export_blinded_wavs(
    session_prefix: str,
    v0p_audio_64: np.ndarray,
    v3c_audio_64: np.ndarray,
    sample_rate: float,
    export_dir: Path,
    session_seed: int,
) -> Dict[str, Any]:
    """Export anonymized 24-bit stereo WAV pair (variant_A / variant_B) and return secret mapping."""
    export_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(session_seed)
    is_v0p_first = rng.choice([True, False])

    label_v0p = "variant_A" if is_v0p_first else "variant_B"
    label_v3c = "variant_B" if is_v0p_first else "variant_A"

    path_a = export_dir / f"{session_prefix}_variant_A.wav"
    path_b = export_dir / f"{session_prefix}_variant_B.wav"

    audio_a = v0p_audio_64 if is_v0p_first else v3c_audio_64
    audio_b = v3c_audio_64 if is_v0p_first else v0p_audio_64

    sf.write(str(path_a), np.transpose(audio_a), int(sample_rate), subtype="PCM_24")
    sf.write(str(path_b), np.transpose(audio_b), int(sample_rate), subtype="PCM_24")

    return {
        "session": session_prefix,
        "seed": str(session_seed),
        "files": {
            path_a.name: "Control_V0p" if is_v0p_first else "Treatment_V3c",
            path_b.name: "Treatment_V3c" if is_v0p_first else "Control_V0p",
        },
        "variant_mapping": {
            "Control_V0p": label_v0p,
            "Treatment_V3c": label_v3c,
        },
    }


def generate_blind_listening_worksheet(all_blind_mappings: Dict[str, Any], out_path: Path) -> None:
    """Generate un-keyed blind listening worksheet for human evaluators."""
    lines = [
        "# SPRINT 3 — V3C PILOT BLIND LISTENING WORKSHEET",
        "",
        "**Istruzioni per il Valutatore:**",
        "Per ciascuno dei 3 brani `same_track` (`track01`, `track02`, `track03`) sono forniti due master anonimizzati: `variant_A.wav` e `variant_B.wav` (`48 kHz`, stereo, `PCM_24`).",
        "Valuta l'immagine stereo, la compatibilità in mono (fold-down check se disponibile), la solidità del basso e la naturalezza timbrica/dinamica complessiva prima di compilare la preferenza.",
        "",
        "| Brano | File 1 | File 2 | Immagine Stereo & Larghezza (A/B/Pari) | Solidità Mono Fold-Down (A/B/Pari) | Controllo Basso & Sub (A/B/Pari) | Trasparenza & Timbro (A/B/Pari) | Preferenza Globale (A/B) | Note d'Ascolto |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|",
    ]
    for pfx in sorted(all_blind_mappings.keys()):
        lines.append(
            f"| **`{pfx}`** | `{pfx}_variant_A.wav` | `{pfx}_variant_B.wav` | [ ] A  [ ] B  [ ] Pari | [ ] A  [ ] B  [ ] Pari | [ ] A  [ ] B  [ ] Pari | [ ] A  [ ] B  [ ] Pari | [ ] A  [ ] B | |"
        )
    lines.extend([
        "",
        "---",
        "*(La chiave di decodifica `SPRINT3_V3C_BLIND_MAPPING.json` è archiviata separatamente ed esclusa dall'archivio valutatori)*",
        "",
    ])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] Saved V3c blind listening worksheet to: {out_path}", flush=True)


def generate_blind_audio_manifest_and_zip(
    wav_dir: Path,
    out_json_path: Path,
    out_md_path: Path,
    worksheet_path: Path,
    zip_out_path: Path,
) -> Dict[str, Any]:
    """Verify the 6 blind WAV files from disk, emit SHA-256/LUFS/TP manifest, and build secret-free evaluator zip."""
    wav_files = sorted(wav_dir.glob("*_variant_[AB].wav"))
    if len(wav_files) != 6:
        raise FileNotFoundError(f"Expected 6 blind WAV files in {wav_dir}, found {len(wav_files)}.")

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
        "artifact_type": "AutoMastering_Sprint3_V3c_Pilot_Blind_Audio_Manifest",
        "description": "Independent disk-readback verification manifest for the 6 anonymized same_track 24-bit WAV files (strictly without key mapping).",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wav_count": len(file_entries),
        "scope_sessions": list(PILOT_SAME_TRACK_PREFIXES),
        "expected_sample_rate_hz": 48000,
        "expected_channels": 2,
        "expected_subtype": "PCM_24",
        "max_abs_delta_lufs_across_pairs": round(max_abs_delta_lufs, 6),
        "extra_gain_applied_on_export": False,
        "contains_secret_mapping": False,
        "evaluator_zip_relative_path": zip_out_path.relative_to(_REPO_ROOT).as_posix(),
        "files": file_entries,
        "pair_deltas": pair_deltas,
    }

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    print(f"[DONE] Saved V3c blind audio manifest JSON to: {out_json_path}", flush=True)

    md_lines = [
        "# SPRINT 3 — V3C PILOT BLIND AUDIO MANIFEST (SHA-256 & INTEGRITÀ WAV)",
        "",
        "**Scopo:** Verifica indipendente da disco dei 6 file WAV anonimizzati per l'ascolto cieco pilota `same_track` (`track01`, `track02`, `track03` — `48 kHz`, stereo, `PCM_24`).  ",
        "**Nota di Blinding:** Questo documento non contiene la chiave di decodifica (`SPRINT3_V3C_BLIND_MAPPING.json`) ed è incluso nel pacchetto valutatori.",
        "",
        "## 1. Inventario File WAV (`docs/sprint3/blind_v3c_pilot_wavs/`)",
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
        "## 2. Parità di Livello per Coppia (`variant_A` vs `variant_B` — Nessun Gain Aggiuntivo)",
        "",
        "| Sessione | LUFS `variant_A` | LUFS `variant_B` | $\\Delta\\text{LUFS}_{A-B}$ (LU) | $|\\Delta\\text{LUFS}|$ (LU) | SHA-256 Uguale nella Coppia |",
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
        f"- **Massimo scarto assoluto di coppia $|\\Delta\\text{{LUFS}}|$:** `{max_abs_delta_lufs:.4f} LU` (render naturali senza compensazione esterna).",
        f"- **Archivio Condivisibile Evaluators (privo di mapping):** `{zip_out_path.relative_to(_REPO_ROOT).as_posix()}` (generato su disco locale, escluso da Git via `.gitignore` per limite dimensione file > 100 MB).",
        "- **Contenuto dell'archivio `.zip`:** esclusivamente i 6 `.wav` in `blind_v3c_pilot_wavs/`, `SPRINT3_V3C_BLIND_LISTENING_WORKSHEET.md` e `SPRINT3_V3C_BLIND_AUDIO_MANIFEST.md`.",
        "",
    ])
    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    out_md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[DONE] Saved V3c blind audio manifest MD to: {out_md_path}", flush=True)

    if not worksheet_path.exists():
        raise FileNotFoundError(f"Blind listening worksheet missing: {worksheet_path}")
    zip_out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_out_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for wav_path in wav_files:
            zf.write(wav_path, arcname=f"blind_v3c_pilot_wavs/{wav_path.name}")
        zf.write(worksheet_path, arcname=worksheet_path.name)
        zf.write(out_md_path, arcname=out_md_path.name)
    print(f"[DONE] Built V3c blind evaluator zip at: {zip_out_path}", flush=True)
    return manifest_payload


def evaluate_single_session_v3c(
    spec: Dict[str, str],
    seed_int: int,
    ref_v0p_ctrl: Dict[str, Any],
    export_wavs: bool,
    export_dir: Path,
) -> Dict[str, Any]:
    """Run paired Control V0p vs Treatment V3c ablation on a single same_track session."""
    prefix = spec["track_prefix"]
    mode = spec["mode"]
    assert mode == "same_track" and prefix in PILOT_SAME_TRACK_PREFIXES

    cache_dir = _REPO_ROOT / ".cache_v3c_ablation_pilot"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{prefix}_v3c_ablation_pilot.json"

    if cache_file.exists():
        print(f"[{prefix}] Loaded session result from cache: {cache_file}", flush=True)
        return json.loads(cache_file.read_text(encoding="utf-8"))

    print(f"\n=======================================================", flush=True)
    print(f"[{prefix}] Running V3c Pilot Ablation (mode={mode}, seed={seed_int})", flush=True)
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
    target_lufs = float(prep["target_lufs"])
    ref_eval_tensor = prep["ref_eval_tensor"]
    mix_start = int(prep["mix_start"])
    eval_len = int(prep["eval_len"])
    suite = MasteringBenchmarkSuite(sample_rate=48000.0)

    v3c_eval_crit = V3cStereoCompatibilityLoss(
        sample_rate=48000.0,
        delay_dsp_samples=TOTAL_DSP_DELAY_SAMPLES,
        **FROZEN_V3C_CONFIG,
    )

    # -------------------------------------------------------------------------
    # Arm A: Control V0p (enable_v3c=False)
    # -------------------------------------------------------------------------
    print(f"[{prefix}] Arm A — Control V0p (30 steps Adam, lr=0.04, enable_v3c=False)...", flush=True)
    torch.manual_seed(seed_int)
    np.random.seed(seed_int & 0xFFFFFFFF)
    opt_v0p = AutoMasteringOptimizer(
        sample_rate=48000.0,
        target_lufs=target_lufs,
        enable_v3c=False,
        **FROZEN_V0P_CONFIG,
    )
    t_a_start = time.time()
    z_opt_v0p, _, hist_v0p = opt_v0p.optimize_multi_chunk(
        mix_chunks=mix_chunks,
        ref_chunks=ref_chunks,
        reference_mode="same_track",
        verbose=False,
        align_latency=True,
        mix_preroll_samples=preroll_list,
        target_lufs=target_lufs,
        ref_ext_chunks=ref_ext_chunks,
        ref_preroll_list=ref_preroll_list,
    )
    t_a_sec = time.time() - t_a_start

    # Exact parity check against harmonized Sprint 3 V0p reference
    z_ref_base = torch.tensor(ref_v0p_ctrl["normalized_z_opt"], dtype=torch.float32)
    v0p_ref_diff = float(torch.max(torch.abs(z_opt_v0p - z_ref_base)).item())
    assert v0p_ref_diff < 1e-5, (
        f"[{prefix}] Control V0p diverged from harmonized baseline: max_diff={v0p_ref_diff}"
    )
    print(f"[{prefix}] Control V0p verified against harmonized baseline (max_diff={v0p_ref_diff:.3e})", flush=True)

    eval_v0p_init = evaluate_point_full_iir(opt_v0p, v3c_eval_crit, Z_NEUTRAL, prep, target_lufs)
    eval_v0p_final = evaluate_point_full_iir(opt_v0p, v3c_eval_crit, z_opt_v0p, prep, target_lufs)
    grad_v0p_final = compute_gradient_norms_at_point(opt_v0p, prep, z_opt_v0p, target_lufs)

    mastered_v0p_64, _ = render_offline_with_tail_flush(
        raw_full_64, mode="z", z=z_opt_v0p.tolist(), sample_rate=48000.0
    )
    m_eval_v0p = mastered_v0p_64[:, mix_start : mix_start + eval_len].astype(np.float32)
    bm_v0p = suite.evaluate(
        torch.tensor(m_eval_v0p).unsqueeze(0), ref_eval_tensor, align_latency=False, reference_mode="same_track"
    )
    ft_stereo_v0p = evaluate_full_track_stereo_diagnostics(
        raw_full_64, mastered_v0p_64, mix_start, eval_len, z_opt_v0p
    )

    # -------------------------------------------------------------------------
    # Arm B: Treatment V3c (enable_v3c=True, Candidate A + frozen pilot weights)
    # -------------------------------------------------------------------------
    print(f"[{prefix}] Arm B — Treatment V3c (30 steps Adam, lr=0.04, enable_v3c=True)...", flush=True)
    torch.manual_seed(seed_int)
    np.random.seed(seed_int & 0xFFFFFFFF)
    opt_v3c = AutoMasteringOptimizer(
        sample_rate=48000.0,
        target_lufs=target_lufs,
        enable_v3c=True,
        v3c_tau_drop_db=CANDIDATE_A_THRESHOLDS["tau_drop_db"],
        v3c_tau_corr=CANDIDATE_A_THRESHOLDS["tau_corr"],
        v3c_tau_smr_db=CANDIDATE_A_THRESHOLDS["tau_smr_db"],
        v3c_tau_sub_db=CANDIDATE_A_THRESHOLDS["tau_sub_db"],
        v3c_alpha_sub=CANDIDATE_A_THRESHOLDS["alpha_sub"],
        v3c_tau_w_low_max=CANDIDATE_A_THRESHOLDS["tau_w_low_max"],
        v3c_tau_w_mid=CANDIDATE_A_THRESHOLDS["tau_w_mid"],
        v3c_tau_w_high=CANDIDATE_A_THRESHOLDS["tau_w_high"],
        v3c_w_mono_drop=PILOT_SAME_TRACK_WEIGHTS["w_mono_drop"],
        v3c_w_band_corr=PILOT_SAME_TRACK_WEIGHTS["w_band_corr"],
        v3c_w_band_smr=PILOT_SAME_TRACK_WEIGHTS["w_band_smr"],
        v3c_w_width_param=PILOT_SAME_TRACK_WEIGHTS["w_width_param"],
        **FROZEN_V0P_CONFIG,
    )
    t_b_start = time.time()
    z_opt_v3c, _, hist_v3c = opt_v3c.optimize_multi_chunk(
        mix_chunks=mix_chunks,
        ref_chunks=ref_chunks,
        reference_mode="same_track",
        verbose=False,
        align_latency=True,
        mix_preroll_samples=preroll_list,
        target_lufs=target_lufs,
        ref_ext_chunks=ref_ext_chunks,
        ref_preroll_list=ref_preroll_list,
    )
    t_b_sec = time.time() - t_b_start

    eval_v3c_init = evaluate_point_full_iir(opt_v0p, v3c_eval_crit, Z_NEUTRAL, prep, target_lufs)
    eval_v3c_final = evaluate_point_full_iir(opt_v0p, v3c_eval_crit, z_opt_v3c, prep, target_lufs)
    grad_v3c_final = compute_gradient_norms_at_point(opt_v3c, prep, z_opt_v3c, target_lufs)

    mastered_v3c_64, _ = render_offline_with_tail_flush(
        raw_full_64, mode="z", z=z_opt_v3c.tolist(), sample_rate=48000.0
    )
    m_eval_v3c = mastered_v3c_64[:, mix_start : mix_start + eval_len].astype(np.float32)
    bm_v3c = suite.evaluate(
        torch.tensor(m_eval_v3c).unsqueeze(0), ref_eval_tensor, align_latency=False, reference_mode="same_track"
    )
    ft_stereo_v3c = evaluate_full_track_stereo_diagnostics(
        raw_full_64, mastered_v3c_64, mix_start, eval_len, z_opt_v3c
    )

    param_shifts = compute_param_shift_metrics(
        z_v3c=eval_v3c_final["normalized_z"],
        z_v0p=eval_v0p_final["normalized_z"],
        theta_v3c=eval_v3c_final["physical_parameters_theta"],
        theta_v0p=eval_v0p_final["physical_parameters_theta"],
    )

    blind_mapping = None
    if export_wavs:
        blind_mapping = export_blinded_wavs(
            session_prefix=prefix,
            v0p_audio_64=mastered_v0p_64,
            v3c_audio_64=mastered_v3c_64,
            sample_rate=48000.0,
            export_dir=export_dir,
            session_seed=seed_int,
        )
        print(f"[{prefix}] Exported blind WAV pair to {export_dir}", flush=True)

    elapsed_total = time.time() - t0
    res = {
        "track_prefix": prefix,
        "mode": mode,
        "seed": str(seed_int),
        "target_lufs": target_lufs,
        "elapsed_sec": float(elapsed_total),
        "opt_v0p_elapsed_sec": float(t_a_sec),
        "opt_v3c_elapsed_sec": float(t_b_sec),
        "v0p_harmonized_baseline_max_diff_z": float(v0p_ref_diff),
        "num_steps": int(FROZEN_V0P_CONFIG["num_steps"]),
        "chunk_metadata": chunk_metadata,
        "control_v0p": {
            "initial_evaluation": eval_v0p_init,
            "final_evaluation": eval_v0p_final,
            "v3c_eval_on_v0p": eval_v0p_final["v3c_objective_and_stereo_iir"],
            "normalized_z_opt": eval_v0p_final["normalized_z"],
            "physical_parameters_theta": eval_v0p_final["physical_parameters_theta"],
            "delta_theta_from_neutral": eval_v0p_final["delta_theta_from_neutral"],
            "gradient_diagnostics_final": grad_v0p_final,
            "full_track_benchmark": bm_v0p,
            "full_track_stereo_diagnostics": ft_stereo_v0p,
            "loss_history": hist_v0p,
        },
        "treatment_v3c": {
            "initial_evaluation": eval_v3c_init,
            "final_evaluation": eval_v3c_final,
            "normalized_z_opt": eval_v3c_final["normalized_z"],
            "physical_parameters_theta": eval_v3c_final["physical_parameters_theta"],
            "delta_theta_from_neutral": eval_v3c_final["delta_theta_from_neutral"],
            "param_shifts_vs_control_v0p": param_shifts,
            "gradient_diagnostics_final": grad_v3c_final,
            "full_track_benchmark": bm_v3c,
            "full_track_stereo_diagnostics": ft_stereo_v3c,
            "loss_history": hist_v3c,
        },
        "blind_mapping": blind_mapping,
    }
    _assert_finite_nested(res, f"session[{prefix}]")
    cache_file.write_text(json.dumps(res, indent=2), encoding="utf-8")

    del raw_full_64, mastered_v0p_64, mastered_v3c_64, prep
    gc.collect()
    return res


def _fmt_opt(val: Optional[float], fmt: str = ".4f", default: str = "N/A") -> str:
    if val is None:
        return default
    return f"{float(val):{fmt}}"


def generate_v3c_ablation_report(
    payload: Dict[str, Any],
    manifest_payload: Optional[Dict[str, Any]],
    out_path: Path,
) -> None:
    """Generate comprehensive SPRINT3_V3C_ABLATION_PILOT_REPORT.md directly from JSON payload."""
    sessions = payload["sessions"]
    lines: List[str] = [
        "# SPRINT 3 — V3C PILOT ABLATION REPORT (`V3c vs V0p`, `same_track` only)",
        "",
        "**Campagna:** Ablation Pilota Accoppiata `Control V0p` vs `Treatment V3c` (`Candidate A` + pesi pilota congelati)  ",
        f"**Data di Generazione:** {payload['timestamp']}  ",
        f"**Durata Complessiva:** {payload['elapsed_sec']:.2f} s  ",
        "**Branch:** `human-validation-package` | **Baseline Commit Fase 1:** `a98c2d67b87d798d449a1018157d50d2a2e7be20`  ",
        "**Scope Autorizzato:** esclusivamente le 3 sessioni `same_track` (`track01`, `track02`, `track03`); `track01_style..03_style` escluse dal pilot.",
        "",
        "> **Nota Metodologica Fondamentale:** Tutte le tabelle seguenti riportano **separatamente** le componenti dell'obiettivo `V0p` (comuni a entrambi i bracci) e i termini `V3c` (calcolati anche in sola lettura su `Control V0p` come `v3c_eval_on_v0p`), senza mai confondere `V0p + V3c` con il solo totale `V0p`. Tutti i risultati qui presentati sono **metriche quantitative oggettive/proxy**; nessun giudizio percettivo d'ascolto umano è stato ancora svolto.",
        "",
        "---",
        "",
        "## 1. Riepilogo Integrazione Optimizer (Fase 1) & Protocollo Sperimentale (Fase 2)",
        "",
        "- **Integrazione Opt-In (`a98c2d67b87d798d449a1018157d50d2a2e7be20`):** `AutoMasteringOptimizer` espone `enable_v3c: bool = False` (con `self.v3c_prior_loss = None` di default, preservando bit-identità V0p/V3b), blocca esplicitamente con `ValueError` qualsiasi modalità diversa da `same_track`, invoca `v3c_prior_loss(mix_ext=mix_k, pred_ext=pred_ext_k, z=z, n_pre=n_pre_k, delay_samples=159, compute_diagnostics=False)` prima della media pesata `1/K` dei chunk (conservando `1x` su `loss_width_param`), ed espone `evaluate_loss_iir` / `_eval_loss_iir`. Validato dai Gate 1–9 (`223 passed`).",
        "- **Soglie Frozen `Candidate A`:** `tau_drop_db=0.50`, `tau_corr=(0.03, 0.05, 0.08)`, `tau_smr_db=(2.0, 2.0, 2.0)`, `tau_sub_db=0.50`, `alpha_sub=1.0`, `tau_w_low_max=1.00`, `tau_w_mid=0.15`, `tau_w_high=0.15`.",
        "- **Pesi Frozen `same_track`:** `w_mono_drop=1.00`, `w_band_corr=10.00`, `w_band_smr=0.05`, `w_width_param=0.25`.",
        "- **Parità Sperimentale:** stesso `prepare_harmonized_session`, 4 chunk da `144,000` campioni con `120,000` campioni di pre-roll, delay DSP `159`, LR4 fisso `160/3200 Hz`, init `Z_NEUTRAL`, `30` step Adam (`lr=0.04`), schedule `fft (1–20) -> iir (21–30)`.",
        "",
        "---",
        "",
        "## 2. Tabella 1 — Obiettivi Separati (`V0p` IIR vs `V3c` Weighted & Raw + `v3c_eval_on_v0p`) e Larghezze Imager",
        "",
        "| Sessione | Braccio | Max Diff vs V0p Base | `loss_v0p_total` (IIR) | `mrstft` (u) | `lufs` (u) | `plr_box` (u) | `loss_v3c_w` | `mono_drop` (u/w) | `band_corr` (u/w) | `band_smr` (u/w) | `width_param` (u/w) | Imager `(L, M, H)` |",
        "|:---|:---|:---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for s in sessions:
        pfx = s["track_prefix"]
        for arm_label, arm_key in (("Control V0p", "control_v0p"), ("Treatment V3c", "treatment_v3c")):
            arm = s[arm_key]
            fe = arm["final_evaluation"]
            v0p_o = fe["v0p_objective_iir"]
            u_c = v0p_o["unweighted_components"]
            v3c_o = fe["v3c_objective_and_stereo_iir"]
            tr = v3c_o["terms_raw"]
            tw = v3c_o["terms_weighted"]
            iw = v3c_o["imager_widths"]
            diff_str = f"{s['v0p_harmonized_baseline_max_diff_z']:.2e}" if arm_key == "control_v0p" else "—"
            lines.append(
                f"| **`{pfx}`** | {arm_label} | {diff_str} | {v0p_o['loss_v0p_total']:.6f} | "
                f"{u_c['mrstft']:.4f} | {u_c['lufs']:.4f} | {u_c['plr_box']:.4f} | "
                f"**{v3c_o['loss_v3c_total_weighted']:.6f}** | "
                f"{tr['loss_mono_drop']:.5f}/{tw['loss_mono_drop']:.5f} | "
                f"{tr['loss_band_corr']:.6f}/{tw['loss_band_corr']:.6f} | "
                f"{tr['loss_band_smr']:.5f}/{tw['loss_band_smr']:.5f} | "
                f"{tr['loss_width_param']:.5f}/{tw['loss_width_param']:.5f} | "
                f"({iw['imager_width_low']:.4f}, {iw['imager_width_mid']:.4f}, {iw['imager_width_high']:.4f}) |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Tabella 2 — Compatibilità Stereo/Mono Chunk-Level IIR (Broadband & Bande: Mix vs Master)",
        "",
        "| Sessione | Braccio | BB $\\rho_{LR}$ (Mix $\\to$ Master, $\\Delta$) | Low $\\rho_{LR}$ / $\\Delta\\text{SMR}$ dB | Mid $\\rho_{LR}$ / $\\Delta\\text{SMR}$ dB | High $\\rho_{LR}$ / $\\Delta\\text{SMR}$ dB | BB $D_{\\text{mono}}$ (Mix $\\to$ Master) |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|",
    ])
    for s in sessions:
        pfx = s["track_prefix"]
        for arm_label, arm_key in (("Control V0p", "control_v0p"), ("Treatment V3c", "treatment_v3c")):
            st = s[arm_key]["final_evaluation"]["v3c_objective_and_stereo_iir"]
            bb = st["broadband"]
            bl = st["bands"]["low"]
            bm = st["bands"]["mid"]
            bh = st["bands"]["high"]
            lines.append(
                f"| **`{pfx}`** | {arm_label} | "
                f"{_fmt_opt(bb['corr_lr_mix'], '.4f')} -> {_fmt_opt(bb['corr_lr_pred'], '.4f')} ({_fmt_opt(bb['delta_corr_lr'], '+.4f')}) | "
                f"{_fmt_opt(bl['corr_lr_pred'], '.4f')} ({_fmt_opt(bl['delta_corr_lr'], '+.4f')}) / {_fmt_opt(bl['delta_smr_db'], '+.2f')} | "
                f"{_fmt_opt(bm['corr_lr_pred'], '.4f')} ({_fmt_opt(bm['delta_corr_lr'], '+.4f')}) / {_fmt_opt(bm['delta_smr_db'], '+.2f')} | "
                f"{_fmt_opt(bh['corr_lr_pred'], '.4f')} ({_fmt_opt(bh['delta_corr_lr'], '+.4f')}) / {_fmt_opt(bh['delta_smr_db'], '+.2f')} | "
                f"{_fmt_opt(bb['mono_drop_db_mix'], '.3f')} -> {_fmt_opt(bb['mono_drop_db_pred'], '.3f')} |"
            )

    lines.extend([
        "",
        "### 3B. Diagnostiche Short-Time (`50 ms`, `100 ms`, `250 ms`) e Anti-Comb-Filtering (Chunk-Level & Full-Track C++)",
        "",
        "| Sessione | Braccio | BB Corr Mean (50/100/250 ms) | BB Corr P05 (50/100/250 ms) | BB Corr Min / CVaR05 (100 ms) | Comb Min Notch (Mix $\\to$ Pred, $\\Delta$ dB) | Full-Track Comb $\\Delta$ Notch / $\\Delta$ Std dB |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|",
    ])
    for s in sessions:
        pfx = s["track_prefix"]
        for arm_label, arm_key in (("Control V0p", "control_v0p"), ("Treatment V3c", "treatment_v3c")):
            st = s[arm_key]["final_evaluation"]["v3c_objective_and_stereo_iir"]
            w50 = st["short_time_windows"]["50ms"]["broadband_pred"]
            w100 = st["short_time_windows"]["100ms"]["broadband_pred"]
            w250 = st["short_time_windows"]["250ms"]["broadband_pred"]
            cb = st["comb_filtering"]
            ft_cb = s[arm_key]["full_track_stereo_diagnostics"]["comb_filtering"]
            lines.append(
                f"| **`{pfx}`** | {arm_label} | "
                f"{_fmt_opt(w50['corr_mean'], '.4f')}/{_fmt_opt(w100['corr_mean'], '.4f')}/{_fmt_opt(w250['corr_mean'], '.4f')} | "
                f"{_fmt_opt(w50['corr_p05'], '.4f')}/{_fmt_opt(w100['corr_p05'], '.4f')}/{_fmt_opt(w250['corr_p05'], '.4f')} | "
                f"{_fmt_opt(w100['corr_min'], '.4f')} / {_fmt_opt(w100['corr_cvar05'], '.4f')} | "
                f"{_fmt_opt(cb['mix']['comb_fold_min_notch_db'], '.2f')} -> {_fmt_opt(cb['pred']['comb_fold_min_notch_db'], '.2f')} ({_fmt_opt(cb['delta_comb_min_notch_db'], '+.2f')}) | "
                f"{_fmt_opt(ft_cb['delta_comb_min_notch_db'], '+.2f')} / {_fmt_opt(ft_cb['delta_comb_std_db'], '+.2f')} |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Tabella 3 — Metriche Audio Full-Track Post-Render C++ & Guardrail True Peak (`<= -0.30 dBTP`)",
        "",
        "| Sessione | Braccio | Target LUFS | Final LUFS | $\\Delta$LUFS | PLR (dB) | Sample Peak (dBFS) | **True Peak (dBTP)** | Guardrail $\\le -0.30$ | SC | LSD (dB) | ERB Surplus (dB) | $\\Delta$ M/S Ratio (dB) |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])
    for s in sessions:
        pfx = s["track_prefix"]
        tlufs = s["target_lufs"]
        for arm_label, arm_key in (("Control V0p", "control_v0p"), ("Treatment V3c", "treatment_v3c")):
            bm = s[arm_key]["full_track_benchmark"]
            tp = _safe_float(bm.get("pred_true_peak_dbtp"))
            tp_status = "PASS" if tp <= -0.30 else f"EXCEED (+{tp + 0.30:.2f})"
            lines.append(
                f"| **`{pfx}`** | {arm_label} | {tlufs:.2f} | {_safe_float(bm.get('pred_lufs')):.2f} | "
                f"{_safe_float(bm.get('delta_lufs')):+.2f} | {_safe_float(bm.get('pred_plr_db')):.2f} | "
                f"{_safe_float(bm.get('pred_sample_peak_dbfs')):+.4f} | **{tp:+.4f}** | {tp_status} | "
                f"{_safe_float(bm.get('spectral_convergence')):.4f} | {_safe_float(bm.get('log_spectral_distance_db')):.3f} | "
                f"{_safe_float(bm.get('erb_presence_surplus_db')):+.3f} | {_safe_float(bm.get('delta_ms_ratio_db')):+.4f} |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Tabella 4 — Spostamenti Parametrici $\\Delta\\theta = \\theta_{\\text{V3c}} - \\theta_{\\text{V0p}}$, Norme dei 6 Gruppi e Gradienti Finali",
        "",
        "| Sessione | $\\Delta w_{\\text{low}}$ | $\\Delta w_{\\text{mid}}$ | $\\Delta w_{\\text{high}}$ | $\\|\\Delta z\\|_2$ (41D) | EQ $\\|\\Delta z\\|_2$ | Xover $\\|\\Delta z\\|_2$ | Comp $\\|\\Delta z\\|_2$ | Imager $\\|\\Delta z\\|_2$ | Sat $\\|\\Delta z\\|_2$ | Lim $\\|\\Delta z\\|_2$ | Final $\\|\\nabla_u V_{3c}\\|_2$ / $\\|\\nabla_u \\text{Tot}\\|_2$ |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ])
    for s in sessions:
        pfx = s["track_prefix"]
        ps = s["treatment_v3c"]["param_shifts_vs_control_v0p"]
        dt = ps["delta_theta"]
        gz = ps["group_l2_delta_z"]
        gf = s["treatment_v3c"]["gradient_diagnostics_final"]
        lines.append(
            f"| **`{pfx}`** | {dt['imager_width_low']:+.4f} | {dt['imager_width_mid']:+.4f} | {dt['imager_width_high']:+.4f} | "
            f"{ps['l2_norm_delta_z_41d']:.5f} | {gz['EQ']:.5f} | {gz['Crossover']:.5f} | {gz['Compressor']:.5f} | "
            f"{gz['Imager']:.5f} | {gz['Saturator']:.5f} | {gz['Limiter']:.5f} | "
            f"{gf['v3c_only_weighted']['norm_grad_u_41d']:.5f} / {gf['total_active']['norm_grad_u_41d']:.5f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 6. Analisi delle Regressioni Proxy, Limiti Osservati e Pacchetto Blind",
        "",
    ])
    for s in sessions:
        pfx = s["track_prefix"]
        v0p_fe = s["control_v0p"]["final_evaluation"]
        v3c_fe = s["treatment_v3c"]["final_evaluation"]
        v3c_on_0 = s["control_v0p"]["v3c_eval_on_v0p"]
        v3c_on_c = v3c_fe["v3c_objective_and_stereo_iir"]
        d_mrstft = v3c_fe["v0p_objective_iir"]["unweighted_components"]["mrstft"] - v0p_fe["v0p_objective_iir"]["unweighted_components"]["mrstft"]
        d_lsd = s["treatment_v3c"]["full_track_benchmark"]["log_spectral_distance_db"] - s["control_v0p"]["full_track_benchmark"]["log_spectral_distance_db"]
        d_v3c = v3c_on_c["loss_v3c_total_weighted"] - v3c_on_0["loss_v3c_total_weighted"]
        lines.append(
            f"- **`{pfx}`:** `loss_v3c_total_weighted` varia da `{v3c_on_0['loss_v3c_total_weighted']:.6f}` (`V0p`) a `{v3c_on_c['loss_v3c_total_weighted']:.6f}` (`V3c`, delta `{d_v3c:+.6f}`), mentre le metriche spettrali comuni registrano $\\Delta\\text{{MRSTFT}} = {d_mrstft:+.6f}$ e $\\Delta\\text{{LSD}} = {d_lsd:+.4f}\\text{{ dB}}$ (nessun valore NaN/Inf in 30 step)."
        )

    lines.extend([
        "",
        "- **Distinzione Proxy vs Ascolto Umano:** Gli scostamenti su $\\rho_{LR}$, SMR, $D_{\\text{mono}}$, MRSTFT e LSD rappresentano esclusivamente proxy differenziabili/oggettivi; la validazione percettiva finale dipende dall'ascolto cieco umano sui 6 file WAV in `docs/sprint3/blind_v3c_pilot_wavs/`.",
        "- **Guardrail True Peak:** Come atteso (poiché `V3c` opera sulla compatibilità stereo/mono e non sostituisce un ceiling True Peak 4x), l'eventuale superamento di `-0.30 dBTP` già presente in `Control V0p` viene monitorato esplicitamente come guardrail indipendente nella Tabella 3.",
    ])

    if manifest_payload is not None:
        lines.append(
            f"- **Pacchetto Blind Audio:** `{manifest_payload['wav_count']}` WAV (`48 kHz`, stereo, `PCM_24`), massimo scarto di coppia $|\\Delta\\text{{LUFS}}| = {manifest_payload['max_abs_delta_lufs_across_pairs']:.4f} LU`, manifest pubblico in `docs/sprint3/SPRINT3_V3C_BLIND_AUDIO_MANIFEST.md`, archivio condivisibile privo di mapping in `{manifest_payload['evaluator_zip_relative_path']}`."
        )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] Generated V3c ablation pilot report to: {out_path}", flush=True)


def run_ablation(export_wavs: bool = True) -> None:
    """Execute full paired Phase 2 V3c vs V0p pilot ablation on track01..03."""
    t_all = time.time()
    manifest = load_reproducibility_manifest()
    seed_by_prefix = {s["track_prefix"]: int(s["seed"]) for s in manifest["sessions"]}
    ref_v0p_map = load_harmonized_v0p_reference()

    pilot_specs = [
        s for s in SESSIONS_SPEC
        if s["track_prefix"] in PILOT_SAME_TRACK_PREFIXES and s["mode"] == "same_track"
    ]
    assert len(pilot_specs) == 3, f"Expected 3 same_track specs, got {len(pilot_specs)}"

    docs_dir = _REPO_ROOT / "docs" / "sprint3"
    out_json = docs_dir / "SPRINT3_V3C_ABLATION_PILOT_RESULTS.json"
    out_report = docs_dir / "SPRINT3_V3C_ABLATION_PILOT_REPORT.md"
    wav_dir = docs_dir / "blind_v3c_pilot_wavs"
    mapping_path = docs_dir / "SPRINT3_V3C_BLIND_MAPPING.json"
    worksheet_path = docs_dir / "SPRINT3_V3C_BLIND_LISTENING_WORKSHEET.md"
    manifest_json_path = docs_dir / "SPRINT3_V3C_BLIND_AUDIO_MANIFEST.json"
    manifest_md_path = docs_dir / "SPRINT3_V3C_BLIND_AUDIO_MANIFEST.md"
    zip_out_path = docs_dir / "blind_v3c_pilot_evaluator_pack.zip"

    sessions_data: List[Dict[str, Any]] = []
    all_blind_mappings: Dict[str, Any] = {}

    for spec in pilot_specs:
        pfx = spec["track_prefix"]
        s_res = evaluate_single_session_v3c(
            spec=spec,
            seed_int=seed_by_prefix[pfx],
            ref_v0p_ctrl=ref_v0p_map[pfx],
            export_wavs=export_wavs,
            export_dir=wav_dir,
        )
        sessions_data.append(s_res)
        if s_res.get("blind_mapping") is not None:
            all_blind_mappings[pfx] = s_res["blind_mapping"]

    elapsed_all = time.time() - t_all
    payload = {
        "artifact_type": "AutoMastering_Sprint3_V3c_Pilot_Ablation_Results",
        "description": "Harmonized Paired Pilot Ablation Evaluation: Control V0p vs Treatment V3c (same_track only)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": float(elapsed_all),
        "optimizer_integration_commit": "a98c2d67b87d798d449a1018157d50d2a2e7be20",
        "authorized_scope_sessions": list(PILOT_SAME_TRACK_PREFIXES),
        "excluded_stylistic_sessions": ["track01_style", "track02_style", "track03_style"],
        "frozen_v0p_config": FROZEN_V0P_CONFIG,
        "frozen_v3c_config": FROZEN_V3C_CONFIG,
        "sessions": sessions_data,
    }
    _assert_finite_nested(payload, "ablation_payload")
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[DONE] Saved V3c pilot ablation results JSON to: {out_json}", flush=True)

    manifest_payload = None
    if export_wavs and all_blind_mappings:
        mapping_path.write_text(json.dumps(all_blind_mappings, indent=2), encoding="utf-8")
        print(f"[DONE] Saved V3c blind key mapping to: {mapping_path}", flush=True)
        generate_blind_listening_worksheet(all_blind_mappings, worksheet_path)
        manifest_payload = generate_blind_audio_manifest_and_zip(
            wav_dir=wav_dir,
            out_json_path=manifest_json_path,
            out_md_path=manifest_md_path,
            worksheet_path=worksheet_path,
            zip_out_path=zip_out_path,
        )

    generate_v3c_ablation_report(payload, manifest_payload, out_report)


def print_dry_run_plan() -> None:
    """Print dry-run execution plan without launching optimization."""
    manifest = load_reproducibility_manifest()
    seeds = {s["track_prefix"]: s["seed"] for s in manifest["sessions"]}
    print("=" * 72)
    print("AutoMastering Sprint 3 — V3c Pilot Ablation Runner (DRY-RUN)")
    print("=" * 72)
    print("Scope (same_track ONLY):", ", ".join(PILOT_SAME_TRACK_PREFIXES))
    for pfx in PILOT_SAME_TRACK_PREFIXES:
        print(f"  - {pfx:<10} seed={seeds.get(pfx)}")
    print("Excluded stylistic sessions: track01_style, track02_style, track03_style")
    print("Frozen Candidate A + Pilot Weights:", FROZEN_V3C_CONFIG)
    print("Run with '--run-ablation' to execute.")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sprint 3 V3c Pilot Ablation Runner (same_track only)")
    parser.add_argument("--run-ablation", action="store_true", default=False)
    parser.add_argument("--no-export-wavs", action="store_true", default=False)
    parser.add_argument("--clean-cache", action="store_true", default=False)
    parser.add_argument("--report-only", action="store_true", default=False)
    args = parser.parse_args()

    docs_dir = _REPO_ROOT / "docs" / "sprint3"
    if args.clean_cache:
        c_dir = _REPO_ROOT / ".cache_v3c_ablation_pilot"
        if c_dir.exists():
            shutil.rmtree(c_dir)
            print(f"[CLEAN] Removed cache dir: {c_dir}", flush=True)

    if args.report_only:
        payload = json.loads((docs_dir / "SPRINT3_V3C_ABLATION_PILOT_RESULTS.json").read_text(encoding="utf-8"))
        m_path = docs_dir / "SPRINT3_V3C_BLIND_AUDIO_MANIFEST.json"
        m_payload = json.loads(m_path.read_text(encoding="utf-8")) if m_path.exists() else None
        generate_v3c_ablation_report(payload, m_payload, docs_dir / "SPRINT3_V3C_ABLATION_PILOT_REPORT.md")
    elif args.run_ablation:
        run_ablation(export_wavs=not args.no_export_wavs)
    else:
        print_dry_run_plan()


if __name__ == "__main__":
    main()
