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
    """Evaluate V0p (updated control) and V2 for a single session."""
    prefix = spec["track_prefix"]
    mode = spec["mode"]
    cache_dir = _REPO_ROOT / "docs" / "sprint3" / ".cache_v2"
    cache_dir.mkdir(parents=True, exist_ok=True)
    session_cache_file = cache_dir / f"{prefix}_v2spec3_tlufs.json"

    if session_cache_file.exists():
        print(f"[{prefix}] Loaded complete session from cache", flush=True)
        return json.loads(session_cache_file.read_text(encoding="utf-8"))

    t0 = time.time()
    raw_full_64, _ = load_and_validate_wav(str(_REPO_ROOT / spec["mix_path"]), required_sr=48000, allow_mono=True)
    ref_full_64, _ = load_and_validate_wav(str(_REPO_ROOT / spec["ref_path"]), required_sr=48000, allow_mono=True)
    raw_full_32 = np.ascontiguousarray(raw_full_64, dtype=np.float32)
    ref_full_32 = np.ascontiguousarray(ref_full_64, dtype=np.float32)
    suite = MasteringBenchmarkSuite(sample_rate=48000.0)

    selector = MultiChunkSelector(
        sample_rate=48000.0,
        num_chunks=4,
        chunk_duration_sec=3.0,
        seed=seed_int,
    )

    n_preroll_samples = 120000
    mix_chunks, ref_chunks, preroll_list = [], [], []
    ref_ext_chunks, ref_preroll_list = [], []
    chunk_metadata = []

    if mode == "same_track":
        alignment = estimate_time_alignment(raw_full_64, ref_full_64, sample_rate=48000.0)
        mix_aligned, ref_aligned = apply_same_track_alignment(
            raw_full_64, ref_full_64, alignment, enable_subsample=True
        )
        mix_start, _, eval_len = get_same_track_slice_indices(
            alignment,
            mix_len=raw_full_64.shape[1],
            ref_len=ref_full_64.shape[1],
            enable_subsample=True,
        )
        ref_eval_tensor = torch.tensor(ref_aligned, dtype=torch.float32).unsqueeze(0)
        ref_bm = suite.evaluate(ref_eval_tensor, ref_eval_tensor, align_latency=False, reference_mode="same_track")
        target_lufs = float(ref_bm["pred_lufs"])
        ref_bm["integrated_lufs"] = target_lufs

        chunks = selector.select_chunks(
            mix_aligned.astype(np.float32),
            ref_aligned.astype(np.float32),
            mode="same_track",
        )
        ref_src_32 = ref_aligned.astype(np.float32)

        for k, ch in enumerate(chunks):
            s_idx = int(ch["start_sample"])
            e_idx = int(ch["end_sample"])
            abs_s = mix_start + s_idx
            abs_e = mix_start + e_idx
            pre_start = max(0, abs_s - n_preroll_samples)
            actual_pre = abs_s - pre_start
            mix_seg = torch.tensor(raw_full_32[:, pre_start:abs_e], dtype=torch.float32).unsqueeze(0)

            rs, re = s_idx, e_idx
            ref_seg = torch.tensor(ref_src_32[:, rs:re], dtype=torch.float32).unsqueeze(0)

            ref_pre_start = max(0, rs - n_preroll_samples)
            actual_ref_pre = rs - ref_pre_start
            ref_ext_seg = torch.tensor(ref_src_32[:, ref_pre_start:re], dtype=torch.float32).unsqueeze(0)

            mix_chunks.append(mix_seg)
            ref_chunks.append(ref_seg)
            preroll_list.append(int(actual_pre))
            ref_ext_chunks.append(ref_ext_seg)
            ref_preroll_list.append(int(actual_ref_pre))

            chunk_metadata.append({
                "chunk_index": k,
                "mix_start_sample": int(s_idx),
                "mix_end_sample": int(e_idx),
                "mix_abs_start_sample": int(abs_s),
                "mix_abs_end_sample": int(abs_e),
                "mix_preroll_samples": int(actual_pre),
                "ref_start_sample": int(rs),
                "ref_end_sample": int(re),
                "ref_preroll_samples": int(actual_ref_pre),
            })
    else:
        alignment = None
        mix_start = 0
        eval_len = raw_full_64.shape[1]
        ref_eval_tensor = torch.tensor(ref_full_32, dtype=torch.float32).unsqueeze(0)
        ref_bm = suite.evaluate(ref_eval_tensor, ref_eval_tensor, align_latency=False, reference_mode="stylistic")
        target_lufs = float(ref_bm["pred_lufs"])
        ref_bm["integrated_lufs"] = target_lufs

        mix_sel = selector.select_chunks(raw_full_32, None, mode="stylistic")
        ref_sel = selector.select_chunks(ref_full_32, None, mode="stylistic")

        for k, (m_ch, r_ch) in enumerate(zip(mix_sel, ref_sel)):
            ms = int(m_ch["start_sample"])
            me = int(m_ch["end_sample"])
            pre_start = max(0, ms - n_preroll_samples)
            actual_pre = ms - pre_start
            mix_seg = torch.tensor(raw_full_32[:, pre_start:me], dtype=torch.float32).unsqueeze(0)

            rs = int(r_ch["start_sample"])
            re = int(r_ch["end_sample"])
            ref_seg = torch.tensor(ref_full_32[:, rs:re], dtype=torch.float32).unsqueeze(0)

            ref_pre_start = max(0, rs - n_preroll_samples)
            actual_ref_pre = rs - ref_pre_start
            ref_ext_seg = torch.tensor(ref_full_32[:, ref_pre_start:re], dtype=torch.float32).unsqueeze(0)

            mix_chunks.append(mix_seg)
            ref_chunks.append(ref_seg)
            preroll_list.append(int(actual_pre))
            ref_ext_chunks.append(ref_ext_seg)
            ref_preroll_list.append(int(actual_ref_pre))

            chunk_metadata.append({
                "chunk_index": k,
                "mix_start_sample": int(ms),
                "mix_end_sample": int(me),
                "mix_abs_start_sample": int(ms),
                "mix_abs_end_sample": int(me),
                "mix_preroll_samples": int(actual_pre),
                "ref_start_sample": int(rs),
                "ref_end_sample": int(re),
                "ref_preroll_samples": int(actual_ref_pre),
            })

    # Punto 5: Strict verification of 4 chunks of 3.0s (144000 samples)
    assert len(mix_chunks) == 4 and len(ref_chunks) == 4, (
        f"Expected 4 chunks for session {prefix}, got mix={len(mix_chunks)}, ref={len(ref_chunks)}"
    )
    for k_idx in range(4):
        eval_mix_len = mix_chunks[k_idx].shape[-1] - preroll_list[k_idx]
        eval_ref_len = ref_chunks[k_idx].shape[-1]
        eval_ref_ext_len = ref_ext_chunks[k_idx].shape[-1] - ref_preroll_list[k_idx]
        assert eval_mix_len == 144000, f"Mix chunk {k_idx} eval len != 144000: {eval_mix_len}"
        assert eval_ref_len == 144000, f"Ref chunk {k_idx} len != 144000: {eval_ref_len}"
        assert eval_ref_ext_len == 144000, f"Ref ext chunk {k_idx} eval len != 144000: {eval_ref_ext_len}"

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

    # -------------------------------------------------------------------------
    # 1. V0p Updated Control: Same full-track target_lufs, plr_mode="box", weight_trans=0.0
    # -------------------------------------------------------------------------
    opt_v0p = AutoMasteringOptimizer(
        sample_rate=48000.0,
        lr=0.04,
        num_steps=30,
        target_lufs=target_lufs,
        weight_stft=1.0,
        weight_loudness=0.8,
        weight_param_reg=0.01,
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
    )

    z_opt_v0p, _, _ = opt_v0p.optimize_multi_chunk(
        mix_chunks=mix_chunks,
        ref_chunks=ref_chunks,
        reference_mode=mode,
        verbose=False,
        align_latency=True,
        mix_preroll_samples=preroll_list,
        target_lufs=target_lufs,
    )

    loss_z0_v0p, metrics_init_v0p = _evaluate_objective_iir(opt_v0p, Z_NEUTRAL, target_lufs)
    loss_zopt_v0p, metrics_final_v0p = _evaluate_objective_iir(opt_v0p, z_opt_v0p, target_lufs)
    red_v0p = ((loss_zopt_v0p - loss_z0_v0p) / loss_z0_v0p) * 100.0 if loss_z0_v0p > 0.0 else 0.0

    mastered_v0p_64, _ = render_offline_with_tail_flush(
        raw_full_64, mode="z", z=z_opt_v0p.tolist(), sample_rate=48000.0
    )
    if mode == "same_track":
        mastered_eval_seg_v0p = mastered_v0p_64[:, mix_start : mix_start + eval_len].astype(np.float32)
        bm_full_v0p = suite.evaluate(
            torch.tensor(mastered_eval_seg_v0p).unsqueeze(0), ref_eval_tensor, align_latency=False, reference_mode="same_track"
        )
    else:
        bm_full_v0p = suite.evaluate(
            torch.tensor(mastered_v0p_64.astype(np.float32)).unsqueeze(0), ref_eval_tensor, align_latency=False, reference_mode="stylistic"
        )

    name_to_idx = {param_spec.name: i for i, param_spec in enumerate(PARAM_SPECS)}
    theta_v0p = map_z_to_theta(z_opt_v0p)
    theta_v0p_dict = {param_spec.name: float(theta_v0p[name_to_idx[param_spec.name]].item()) for param_spec in PARAM_SPECS}

    # -------------------------------------------------------------------------
    # 2. V2 Configuration: Transient/envelope & reference-aware PLR with frozen v3 hyperparameters
    # -------------------------------------------------------------------------
    opt_v2 = AutoMasteringOptimizer(
        sample_rate=48000.0,
        lr=0.04,
        num_steps=30,
        target_lufs=target_lufs,
        weight_stft=1.0,
        weight_loudness=0.8,
        weight_param_reg=0.01,
        use_hybrid_eq=True,
        weight_erb=0.0,
        stylistic_spectral_mode="mel40",
        weight_de_harsh=0.0,
        weight_trans=0.2,
        plr_mode="reference_aware_same" if mode == "same_track" else "reference_aware",
        beta_atk=0.5,
        plr_d_style=1.0,
        plr_gamma_floor=1.0,
        soft_peak_p=8,
        soft_peak_beta=2.0,
        soft_peak_block_duration_sec=0.010,
    )

    z_opt_v2, _, _ = opt_v2.optimize_multi_chunk(
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

    loss_z0_v2, metrics_init_v2 = _evaluate_objective_iir(opt_v2, Z_NEUTRAL, target_lufs)
    loss_zopt_v2, metrics_final_v2 = _evaluate_objective_iir(opt_v2, z_opt_v2, target_lufs)
    red_v2 = ((loss_zopt_v2 - loss_z0_v2) / loss_z0_v2) * 100.0 if loss_z0_v2 > 0.0 else 0.0

    mastered_v2_64, _ = render_offline_with_tail_flush(
        raw_full_64, mode="z", z=z_opt_v2.tolist(), sample_rate=48000.0
    )
    if mode == "same_track":
        mastered_eval_seg_v2 = mastered_v2_64[:, mix_start : mix_start + eval_len].astype(np.float32)
        bm_full_v2 = suite.evaluate(
            torch.tensor(mastered_eval_seg_v2).unsqueeze(0), ref_eval_tensor, align_latency=False, reference_mode="same_track"
        )
    else:
        bm_full_v2 = suite.evaluate(
            torch.tensor(mastered_v2_64.astype(np.float32)).unsqueeze(0), ref_eval_tensor, align_latency=False, reference_mode="stylistic"
        )

    theta_v2 = map_z_to_theta(z_opt_v2)
    theta_v2_dict = {param_spec.name: float(theta_v2[name_to_idx[param_spec.name]].item()) for param_spec in PARAM_SPECS}

    # Evaluate V2 objective on z_opt_v0p for clean cross-comparison
    loss_z0_v2_on_v0p, m_init_v2_on_v0p = _evaluate_objective_iir(opt_v2, Z_NEUTRAL, target_lufs)
    loss_zopt_v2_on_v0p, m_final_v2_on_v0p = _evaluate_objective_iir(opt_v2, z_opt_v0p, target_lufs)
    v2_on_v0p_dict = {
        "initial_loss_z0_iir": float(loss_z0_v2_on_v0p),
        "final_loss_z_opt_iir": float(loss_zopt_v2_on_v0p),
        "loss_reduction_iir_pct": float(
            ((loss_zopt_v2_on_v0p - loss_z0_v2_on_v0p) / loss_z0_v2_on_v0p) * 100.0 if loss_z0_v2_on_v0p > 0.0 else 0.0
        ),
        "chunk_metrics_init_iir": m_init_v2_on_v0p,
        "chunk_metrics_final_iir": m_final_v2_on_v0p,
    }

    v0p_config = {
        "target_lufs": float(target_lufs),
        "preroll_samples": 120000,
        "erb_weight": 0.0,
        "de_harsh_weight": 0.0,
        "weight_trans": 0.0,
        "beta_atk": 0.5,
        "plr_mode": "box",
        "plr_d_style": 1.0,
        "plr_gamma_floor": 1.0,
        "soft_peak_p": 8,
        "soft_peak_beta": 2.0,
        "soft_peak_block_duration_sec": 0.010,
    }
    v0p_data = {
        "config": v0p_config,
        "preroll_sec": 2.5,
        "preroll_samples_per_chunk": preroll_list,
        "target_lufs": float(target_lufs),
        "initial_loss_z0_iir": float(loss_z0_v0p),
        "final_loss_z_opt_iir": float(loss_zopt_v0p),
        "loss_reduction_iir_pct": float(red_v0p),
        "chunk_metrics_init_iir": metrics_init_v0p,
        "chunk_metrics_final_iir": metrics_final_v0p,
        "v2_objective_eval_on_v0p": v2_on_v0p_dict,
        "full_track_benchmark": bm_full_v0p,
        "normalized_z_opt": [float(v) for v in z_opt_v0p.detach().cpu().numpy().tolist()],
        "physical_parameters_theta": theta_v0p_dict,
    }

    v2_config = {
        "target_lufs": float(target_lufs),
        "preroll_samples": 120000,
        "erb_weight": 0.0,
        "de_harsh_weight": 0.0,
        "weight_trans": 0.2,
        "beta_atk": 0.5,
        "plr_mode": "reference_aware_same" if mode == "same_track" else "reference_aware",
        "plr_d_style": 1.0,
        "plr_gamma_floor": 1.0,
        "soft_peak_p": 8,
        "soft_peak_beta": 2.0,
        "soft_peak_block_duration_sec": 0.010,
    }
    v2_data = {
        "config": v2_config,
        "preroll_sec": 2.5,
        "preroll_samples_per_chunk": preroll_list,
        "ref_preroll_samples_per_chunk": ref_preroll_list,
        "target_lufs": float(target_lufs),
        "initial_loss_z0_iir": float(loss_z0_v2),
        "final_loss_z_opt_iir": float(loss_zopt_v2),
        "loss_reduction_iir_pct": float(red_v2),
        "chunk_metrics_init_iir": metrics_init_v2,
        "chunk_metrics_final_iir": metrics_final_v2,
        "full_track_benchmark": bm_full_v2,
        "normalized_z_opt": [float(v) for v in z_opt_v2.detach().cpu().numpy().tolist()],
        "physical_parameters_theta": theta_v2_dict,
    }

    session_payload = {
        "track_prefix": prefix,
        "reference_mode": mode,
        "seed": str(seed_int),
        "target_lufs": float(target_lufs),
        "chunk_metadata": chunk_metadata,
        "variants": {
            "v0p_historical": v0p_variant_data,
            "V0p": v0p_data,
            "V2": v2_data,
        },
    }

    session_cache_file.write_text(json.dumps(session_payload, indent=2), encoding="utf-8")
    elapsed = time.time() - t0
    print(
        f"[{prefix}] ({elapsed:.1f}s) "
        f"V0p: init={loss_z0_v0p:.4f}->final={loss_zopt_v0p:.4f} ({red_v0p:+.2f}%) | "
        f"V2: init={loss_z0_v2:.4f}->final={loss_zopt_v2:.4f} ({red_v2:+.2f}%) | "
        f"LUFS={bm_full_v2['pred_lufs']:.2f} | PLR={bm_full_v2.get('pred_plr_db', 0.0):.2f}",
        flush=True,
    )
    gc.collect()
    return session_payload


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
            "plr_d_style": 1.0,
            "plr_gamma_floor": 1.0,
            "soft_peak_p": 8,
            "soft_peak_beta": 2.0,
            "soft_peak_block_duration_sec": 0.010,
        },
        "sessions": session_results,
    }
    out_file.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
    print(f"\n[DONE] Saved V2 ablation results to: {out_file}")


if __name__ == "__main__":
    main()
