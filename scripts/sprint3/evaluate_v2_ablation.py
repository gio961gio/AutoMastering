"""Sprint 3 Step V2 Ablation Runner: V0p vs V2 (Transient/Envelope & Reference-Aware PLR Loss).

Evaluates:
  - V0p: Context control baseline (2.5s pre-roll, historical Sprint 2 loss).
  - V2:  2.5s pre-roll + Transient/Envelope Loss (weight_trans=0.2, beta_atk=0.5)
         + Reference-Aware PLR Loss with Boltzmann soft-peak mean (plr_mode="reference_aware").

Runs multi-chunk optimization (num_steps=30, lr=0.04, exact Sprint 2 seeds and chunk coordinates)
across all 6 sessions (track01..03 same_track and track01..03_style stylistic),
renders full-track offline C++ output with tail flush, and benchmarks objective metrics.
Exports results to docs/sprint3/SPRINT3_V2_ABLATION_RESULTS.json.
"""

import gc
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.signal import lfilter as scipy_lfilter
import torch
import torchaudio.functional as taf

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class _FastCPUAnalyticalBiquad(torch.autograd.Function):
    """Exact float64 analytical forward & adjoint IIR biquad filter using C/SciPy lfilter on CPU.

    Matches `torchaudio.functional.lfilter(..., clamp=False)` to machine epsilon (~1e-14 in float64)
    while running >50x faster on CPU for long buffers (5.5s pre-roll chunks and full tracks).
    """

    @staticmethod
    def forward(
        ctx,
        waveform: torch.Tensor,
        a_coeffs: torch.Tensor,
        b_coeffs: torch.Tensor,
    ) -> torch.Tensor:
        orig_shape = waveform.shape
        orig_dtype = waveform.dtype
        device = waveform.device
        N = orig_shape[-1]

        x_2d = waveform.detach().to(torch.float64).reshape(-1, N)
        M = x_2d.shape[0]
        x_np = np.ascontiguousarray(x_2d.cpu().numpy(), dtype=np.float64)

        a_2d = a_coeffs.detach().to(torch.float64).reshape(-1, 3).cpu().numpy()
        b_2d = b_coeffs.detach().to(torch.float64).reshape(-1, 3).cpu().numpy()

        # Normalize by a0 if a0 != 1.0
        a0 = a_2d[:, 0:1]
        b_norm = b_2d / a0
        a_norm = a_2d / a0

        if b_norm.shape[0] == 1 or np.allclose(b_norm, b_norm[0:1]) and np.allclose(a_norm, a_norm[0:1]):
            y_np = scipy_lfilter(b_norm[0], a_norm[0], x_np, axis=-1)
        else:
            y_np = np.empty_like(x_np)
            for m in range(M):
                idx = min(m, b_norm.shape[0] - 1)
                y_np[m] = scipy_lfilter(b_norm[idx], a_norm[idx], x_np[m], axis=-1)

        y_t = torch.from_numpy(y_np).to(device=device, dtype=orig_dtype).reshape(orig_shape)
        ctx.save_for_backward(
            waveform.detach().to(torch.float64),
            torch.from_numpy(y_np).to(device=device, dtype=torch.float64),
            a_coeffs.detach().to(torch.float64),
            b_coeffs.detach().to(torch.float64),
        )
        return y_t

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_saved, y_2d, a_coeffs, b_coeffs = ctx.saved_tensors
        orig_shape = x_saved.shape
        device = grad_output.device
        N = orig_shape[-1]

        x_np = np.ascontiguousarray(x_saved.reshape(-1, N).cpu().numpy(), dtype=np.float64)
        y_np = np.ascontiguousarray(y_2d.reshape(-1, N).cpu().numpy(), dtype=np.float64)
        gy_np = np.ascontiguousarray(
            grad_output.detach().to(torch.float64).reshape(-1, N).cpu().numpy(), dtype=np.float64
        )
        M = x_np.shape[0]

        a_2d = a_coeffs.reshape(-1, 3).cpu().numpy()
        b_2d = b_coeffs.reshape(-1, 3).cpu().numpy()
        a0 = a_2d[:, 0:1]
        b_norm = b_2d / a0
        a_norm = a_2d / a0

        shared_coeffs = (
            b_norm.shape[0] == 1
            or (np.allclose(b_norm, b_norm[0:1]) and np.allclose(a_norm, a_norm[0:1]))
        )
        if shared_coeffs:
            psi_rev = scipy_lfilter(np.array([1.0], dtype=np.float64), a_norm[0], gy_np[:, ::-1], axis=-1)
            psi = np.ascontiguousarray(psi_rev[:, ::-1])
        else:
            psi = np.empty_like(gy_np)
            for m in range(M):
                idx = min(m, a_norm.shape[0] - 1)
                psi_rev = scipy_lfilter(np.array([1.0], dtype=np.float64), a_norm[idx], gy_np[m, ::-1], axis=-1)
                psi[m] = psi_rev[::-1]

        grad_x = None
        if ctx.needs_input_grad[0]:
            if shared_coeffs:
                gx_rev = scipy_lfilter(b_norm[0], np.array([1.0], dtype=np.float64), psi[:, ::-1], axis=-1)
                gx_np = np.ascontiguousarray(gx_rev[:, ::-1])
            else:
                gx_np = np.empty_like(psi)
                for m in range(M):
                    idx = min(m, b_norm.shape[0] - 1)
                    gx_rev = scipy_lfilter(b_norm[idx], np.array([1.0], dtype=np.float64), psi[m, ::-1], axis=-1)
                    gx_np[m] = gx_rev[::-1]
            grad_x = torch.from_numpy(gx_np).to(device=device, dtype=grad_output.dtype).reshape(orig_shape)

        grad_a = None
        if ctx.needs_input_grad[1]:
            da0 = np.zeros(M, dtype=np.float64)
            da1 = -(psi[:, 1:] * y_np[:, :-1]).sum(axis=-1)
            da2 = -(psi[:, 2:] * y_np[:, :-2]).sum(axis=-1)
            da_mat = np.stack([da0, da1, da2], axis=-1) / a0
            if a_coeffs.dim() == 1:
                da_mat = da_mat.sum(axis=0)
            elif a_coeffs.shape[0] == 1 and M > 1:
                da_mat = da_mat.sum(axis=0, keepdims=True)
            grad_a = torch.from_numpy(da_mat).to(device=device, dtype=a_coeffs.dtype).reshape(a_coeffs.shape)

        grad_b = None
        if ctx.needs_input_grad[2]:
            db0 = (psi * x_np).sum(axis=-1)
            db1 = (psi[:, 1:] * x_np[:, :-1]).sum(axis=-1)
            db2 = (psi[:, 2:] * x_np[:, :-2]).sum(axis=-1)
            db_mat = np.stack([db0, db1, db2], axis=-1) / a0
            if b_coeffs.dim() == 1:
                db_mat = db_mat.sum(axis=0)
            elif b_coeffs.shape[0] == 1 and M > 1:
                db_mat = db_mat.sum(axis=0, keepdims=True)
            grad_b = torch.from_numpy(db_mat).to(device=device, dtype=b_coeffs.dtype).reshape(b_coeffs.shape)

        return grad_x, grad_a, grad_b


_ORIG_TAF_LFILTER = taf.lfilter


def _fast_cpu_lfilter(
    waveform: torch.Tensor,
    a_coeffs: torch.Tensor,
    b_coeffs: torch.Tensor,
    clamp: bool = True,
    batching: bool = True,
) -> torch.Tensor:
    if not clamp and not waveform.is_cuda and a_coeffs.shape[-1] == 3 and b_coeffs.shape[-1] == 3:
        return _FastCPUAnalyticalBiquad.apply(waveform, a_coeffs, b_coeffs)
    return _ORIG_TAF_LFILTER(waveform, a_coeffs, b_coeffs, clamp=clamp, batching=batching)


taf.lfilter = _fast_cpu_lfilter

from ddsp_mastering.data import (
    MultiChunkSelector,
    apply_same_track_alignment,
    estimate_time_alignment,
    get_same_track_slice_indices,
)
from ddsp_mastering.data.chunking import ChunkSpec
from ddsp_mastering.dsp.chain import DifferentiableMasteringChain
from ddsp_mastering.loss.combined import MasteringPerceptualLoss
from ddsp_mastering.loss.stylistic import AggregatePerceptualLoss
from ddsp_mastering.optimize import AutoMasteringOptimizer, MasteringBenchmarkSuite
from ddsp_mastering.param_space import PARAM_SPECS, Z_NEUTRAL, map_theta_to_z, map_z_to_theta
from scripts.human_validation.offline_renderer import render_offline_with_tail_flush
from scripts.human_validation.validate_inputs import load_and_validate_wav

SESSIONS_SPEC = [
    {
        "track_prefix": "track01",
        "mode": "same_track",
        "mix_path": "human_validation/private/input/track01_mix.wav",
        "ref_path": "human_validation/private/input/track01_ref.wav",
    },
    {
        "track_prefix": "track02",
        "mode": "same_track",
        "mix_path": "human_validation/private/input/track02_mix.wav",
        "ref_path": "human_validation/private/input/track02_ref.wav",
    },
    {
        "track_prefix": "track03",
        "mode": "same_track",
        "mix_path": "human_validation/private/input/track03_mix.wav",
        "ref_path": "human_validation/private/input/track03_ref.wav",
    },
    {
        "track_prefix": "track01_style",
        "mode": "stylistic",
        "mix_path": "human_validation/private/input/track01_mix.wav",
        "ref_path": "human_validation/private/input/track01_ref_altro_brano.wav",
    },
    {
        "track_prefix": "track02_style",
        "mode": "stylistic",
        "mix_path": "human_validation/private/input/track02_mix.wav",
        "ref_path": "human_validation/private/input/track02_ref_altro_brano.wav",
    },
    {
        "track_prefix": "track03_style",
        "mode": "stylistic",
        "mix_path": "human_validation/private/input/track03_mix.wav",
        "ref_path": "human_validation/private/input/track03_ref_altro_brano.wav",
    },
]


def load_reproducibility_manifest() -> Dict[str, Any]:
    p = _REPO_ROOT / "docs" / "human_validation" / "SPRINT2_V0_REPRODUCIBILITY_MANIFEST.json"
    return json.loads(p.read_text(encoding="utf-8"))


def load_v0p_v1_results() -> Dict[str, Any]:
    p = _REPO_ROOT / "docs" / "sprint3" / "SPRINT3_V0P_V1_ABLATION_RESULTS.json"
    return json.loads(p.read_text(encoding="utf-8"))


def evaluate_single_session_v2(
    spec: Dict[str, str],
    seed_int: int,
    v0p_variant_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate V2 for a single session and pair it with V0p control baseline."""
    prefix = spec["track_prefix"]
    mode = spec["mode"]
    cache_dir = _REPO_ROOT / "docs" / "sprint3" / ".cache_v2"
    cache_dir.mkdir(parents=True, exist_ok=True)
    var_cache_file = cache_dir / f"{prefix}_V2.json"

    if var_cache_file.exists():
        print(f"[{prefix}:V2] Loaded from cache", flush=True)
        v2_data = json.loads(var_cache_file.read_text(encoding="utf-8"))
    else:
        t0 = time.time()
        raw_full_64, _ = load_and_validate_wav(str(_REPO_ROOT / spec["mix_path"]), required_sr=48000, allow_mono=True)
        ref_full_64, _ = load_and_validate_wav(str(_REPO_ROOT / spec["ref_path"]), required_sr=48000, allow_mono=True)
        raw_full_32 = np.ascontiguousarray(raw_full_64, dtype=np.float32)
        ref_full_32 = np.ascontiguousarray(ref_full_64, dtype=np.float32)

        if mode == "same_track":
            alignment = estimate_time_alignment(raw_full_32, ref_full_32, sample_rate=48000.0)
            raw_aligned, ref_aligned = apply_same_track_alignment(raw_full_32, ref_full_32, alignment)
        else:
            alignment = None
            raw_aligned = raw_full_32
            ref_aligned = ref_full_32

        selector = MultiChunkSelector(
            sample_rate=48000.0,
            num_chunks=4,
            chunk_duration_sec=3.0,
            seed=seed_int,
        )
        mix_chunks_spec = selector.select_chunks(raw_aligned)
        if mode == "same_track":
            ref_chunk_tensors = [
                torch.tensor(ref_aligned[:, c.start_sample : c.end_sample], dtype=torch.float32).unsqueeze(0)
                for c in mix_chunks_spec
            ]
        else:
            ref_chunks_spec = selector.select_chunks(ref_full_32)
            ref_chunks_np = selector.extract_chunk_tensors(ref_full_32, ref_chunks_spec)
            ref_chunk_tensors = [torch.tensor(c, dtype=torch.float32).unsqueeze(0) for c in ref_chunks_np]

        suite = MasteringBenchmarkSuite(sample_rate=48000.0)

        # Extract with 2.5s pre-roll
        ext_np, preroll_counts = selector.extract_chunk_tensors_with_preroll(
            raw_aligned, mix_chunks_spec, preroll_sec=2.5
        )
        mix_chunk_tensors = [torch.tensor(c, dtype=torch.float32).unsqueeze(0) for c in ext_np]

        # V2 Configuration: isolated transient & reference-aware PLR
        opt = AutoMasteringOptimizer(
            sample_rate=48000.0,
            lr=0.04,
            num_steps=30,
            target_lufs=None,
            weight_stft=1.0,
            weight_loudness=0.8,
            weight_param_reg=0.01,
            use_hybrid_eq=True,
            weight_erb=0.0,  # isolated from ERB
            stylistic_spectral_mode="mel40",
            weight_de_harsh=0.0,
            weight_trans=0.2,  # V2 transient loss weight
            plr_mode="reference_aware",  # V2 reference-aware PLR
            beta_atk=0.5,
            plr_d_style=2.0,
            plr_gamma_floor=0.05,
            soft_peak_p=8,
            soft_peak_beta=12.0,
            soft_peak_block_duration_sec=0.400,
        )

        z_opt, _, _ = opt.optimize_multi_chunk(
            mix_chunks=mix_chunk_tensors,
            ref_chunks=ref_chunk_tensors,
            reference_mode=mode,
            verbose=False,
            align_latency=True,
            mix_preroll_samples=preroll_counts,
        )

        K = len(mix_chunk_tensors)
        weights = [1.0 / K] * K
        with torch.no_grad():
            lufs_vals = [opt.loss_fn.loudness_loss.calculate_lufs(rc).item() for rc in ref_chunk_tensors]
            eff_target_lufs = float(sum(lufs_vals) / len(lufs_vals))

        def _eval_loss_iir(z_in: torch.Tensor) -> Tuple[float, Dict[str, float]]:
            with torch.no_grad():
                tot_l = 0.0
                agg_m: Dict[str, float] = {}
                for k in range(K):
                    n_pre = preroll_counts[k]
                    mk = mix_chunk_tensors[k]
                    y_ext = opt.chain(mk, z=z_in, eq_mode="iir")
                    y_k = y_ext[..., n_pre:]
                    ref_k = ref_chunk_tensors[k]
                    if mode == "same_track":
                        lk, mk_dict = opt.loss_fn(
                            pred_audio=y_k,
                            target_audio=ref_k,
                            target_lufs=eff_target_lufs,
                            normalized_params=z_in,
                            align_latency=True,
                            pred_ext_audio=y_ext,
                            pred_preroll_samples=n_pre,
                        )
                    else:
                        anchor_k = mk[..., n_pre:]
                        lk, mk_dict = opt.stylistic_loss_fn(
                            pred_audio=y_k,
                            ref_audio=ref_k,
                            normalized_params=z_in,
                            target_lufs=eff_target_lufs,
                            mix_anchor_audio=anchor_k,
                            pred_ext_audio=y_ext,
                            pred_preroll_samples=n_pre,
                        )
                    tot_l += weights[k] * float(lk.item())
                    for mk_key, mk_val in mk_dict.items():
                        agg_m[mk_key] = agg_m.get(mk_key, 0.0) + weights[k] * float(mk_val)
                return tot_l, agg_m

        loss_z0_iir, _ = _eval_loss_iir(Z_NEUTRAL)
        loss_zopt_iir, metrics_final = _eval_loss_iir(z_opt)
        reduction_pct = (
            ((loss_zopt_iir - loss_z0_iir) / loss_z0_iir) * 100.0 if loss_z0_iir > 0.0 else 0.0
        )

        # Full-track offline C++ render
        mastered_full_64, _ = render_offline_with_tail_flush(
            raw_full_64, mode="z", z=z_opt.tolist(), sample_rate=48000.0
        )

        if mode == "same_track":
            mix_start, _, eval_len = get_same_track_slice_indices(
                alignment, mix_len=raw_full_64.shape[1], ref_len=ref_full_64.shape[1], enable_subsample=True
            )
            ref_eval_tensor = torch.tensor(ref_aligned, dtype=torch.float32).unsqueeze(0)
            mastered_eval_seg = mastered_full_64[:, mix_start : mix_start + eval_len].astype(np.float32)
            mastered_eval_tensor = torch.tensor(mastered_eval_seg, dtype=torch.float32).unsqueeze(0)
            bm_full = suite.evaluate(
                mastered_eval_tensor, ref_eval_tensor, align_latency=False, reference_mode="same_track"
            )
        else:
            ref_eval_tensor = torch.tensor(ref_full_32, dtype=torch.float32).unsqueeze(0)
            mastered_eval_tensor = torch.tensor(
                mastered_full_64.astype(np.float32), dtype=torch.float32
            ).unsqueeze(0)
            bm_full = suite.evaluate(
                mastered_eval_tensor, ref_eval_tensor, align_latency=False, reference_mode="stylistic"
            )

        name_to_idx = {spec.name: i for i, spec in enumerate(PARAM_SPECS)}
        theta_opt = map_z_to_theta(z_opt)
        theta_dict = {spec.name: float(theta_opt[name_to_idx[spec.name]].item()) for spec in PARAM_SPECS}

        v2_data = {
            "preroll_sec": 2.5,
            "preroll_samples_per_chunk": preroll_counts,
            "initial_loss_z0_iir": float(loss_z0_iir),
            "final_loss_z_opt_iir": float(loss_zopt_iir),
            "loss_reduction_iir_pct": float(reduction_pct),
            "chunk_metrics_final_iir": metrics_final,
            "full_track_benchmark": bm_full,
            "normalized_z_opt": [float(v) for v in z_opt.detach().cpu().numpy().tolist()],
            "physical_parameters_theta": theta_dict,
        }
        var_cache_file.write_text(json.dumps(v2_data, indent=2), encoding="utf-8")
        elapsed = time.time() - t0
        print(
            f"[{prefix}:V2] ({elapsed:.1f}s) init={loss_z0_iir:.4f} -> final={loss_zopt_iir:.4f} "
            f"({reduction_pct:+.2f}%) | LUFS={bm_full['pred_lufs']:.2f} | PLR={bm_full.get('pred_plr_db', 0.0):.2f} "
            f"| trans_loss={metrics_final.get('trans_same_total_loss', metrics_final.get('trans_style_total_loss', 0.0)):.4f}",
            flush=True,
        )
        gc.collect()

    return {
        "track_prefix": prefix,
        "reference_mode": mode,
        "seed": str(seed_int),
        "variants": {
            "V0p": v0p_variant_data,
            "V2": v2_data,
        },
    }


def main():
    print("================================================================================")
    print("Sprint 3 Step V2 Ablation: Transient/Envelope & Reference-Aware PLR Loss")
    print("================================================================================")

    manifest = load_reproducibility_manifest()
    v0p_v1_results = load_v0p_v1_results()

    seed_by_prefix: Dict[str, int] = {}
    for s in manifest["sessions"]:
        seed_by_prefix[s["track_prefix"]] = int(s["seed"])

    v0p_by_prefix: Dict[str, Any] = {}
    for s in v0p_v1_results["sessions"]:
        v0p_by_prefix[s["track_prefix"]] = s["variants"]["V0p"]

    session_results = []
    for spec in SESSIONS_SPEC:
        prefix = spec["track_prefix"]
        seed_int = seed_by_prefix[prefix]
        v0p_data = v0p_by_prefix[prefix]
        res = evaluate_single_session_v2(spec, seed_int, v0p_data)
        session_results.append(res)

    out_file = _REPO_ROOT / "docs" / "sprint3" / "SPRINT3_V2_ABLATION_RESULTS.json"
    results_payload = {
        "artifact_type": "AutoMastering_Sprint3_V2_Ablation_Results",
        "version": "1.0.0",
        "seed_serialization_note": "All 64-bit unsigned integer seeds (uint64) in 'seed' are serialized as decimal strings.",
        "v2_hyperparameters": {
            "weight_trans": 0.2,
            "plr_mode": "reference_aware",
            "beta_atk": 0.5,
            "plr_d_style": 2.0,
            "plr_gamma_floor": 0.05,
            "soft_peak_p": 8,
            "soft_peak_beta": 12.0,
            "soft_peak_block_duration_sec": 0.400,
        },
        "sessions": session_results,
    }
    out_file.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
    print(f"\n[DONE] Saved V2 ablation results to: {out_file}")


if __name__ == "__main__":
    main()
