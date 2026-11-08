"""Sprint 3 Incremental Ablation Runner: V0 vs V0p vs V1 (and V1b).

Evaluates:
  1. Pre-roll convergence table (`T_pre` in `{0.0, 0.25, 1.0, 2.0, 2.5, 3.0}` seconds
     vs full-track streaming state) on `track01` chunk `[10.0, 13.0] s` under
     maximum compressor time constants (`release = 600 ms`, `attack = 120 ms`).
  2. Multi-chunk optimization (`num_steps = 30`, `lr = 0.04`, exact Sprint 2 seeds
     and chunk coordinates) across all 6 sessions (`track01..03` `same_track` and
     `track01..03_style` `stylistic`) for:
       - `V0`:  0.0 s pre-roll, historical Sprint 2 loss
       - `V0p`: 2.5 s pre-roll, historical Sprint 2 loss (context control baseline)
       - `V1`:  2.5 s pre-roll + ERB-48 spectral statistics (`weight_erb=0.25` in
                `same_track`, `spectral_mode='erb48'` in `stylistic`)
       - `V1b`: `V1` + asymmetric relative de-harshness penalty (`weight_de_harsh=0.15`)
  3. Full-track C++ tail-flush rendering (`render_offline_with_tail_flush`) and
     objective evaluation via `MasteringBenchmarkSuite`, exporting sanitized results
     to `docs/sprint3/SPRINT3_V0P_V1_ABLATION_RESULTS.json`.
"""

import gc
import json
import math
import os
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

VARIANTS_CFG = {
    "V0": {"preroll_sec": 0.0, "weight_erb": 0.0, "spectral_mode": "mel40", "weight_de_harsh": 0.0},
    "V0p": {"preroll_sec": 2.5, "weight_erb": 0.0, "spectral_mode": "mel40", "weight_de_harsh": 0.0},
    "V1": {"preroll_sec": 2.5, "weight_erb": 0.25, "spectral_mode": "erb48", "weight_de_harsh": 0.0},
    "V1b": {"preroll_sec": 2.5, "weight_erb": 0.25, "spectral_mode": "erb48", "weight_de_harsh": 0.15},
}


def compute_preroll_convergence_diagnostics() -> Dict[str, Any]:
    """Evaluate pre-roll convergence on track01 [10.0s, 13.0s] under max time constants (600 ms release)."""
    sr = 48000
    raw_full_64, _ = load_and_validate_wav(
        str(_REPO_ROOT / "human_validation/private/input/track01_mix.wav"), required_sr=48000, allow_mono=True
    )
    ref_full_64, _ = load_and_validate_wav(
        str(_REPO_ROOT / "human_validation/private/input/track01_ref.wav"), required_sr=48000, allow_mono=True
    )
    raw_32 = np.ascontiguousarray(raw_full_64, dtype=np.float32)
    ref_32 = np.ascontiguousarray(ref_full_64, dtype=np.float32)
    alignment = estimate_time_alignment(raw_32, ref_32, sample_rate=48000.0)
    raw_audio_32, ref_audio_32 = apply_same_track_alignment(raw_32, ref_32, alignment)

    name_to_idx = {spec.name: i for i, spec in enumerate(PARAM_SPECS)}
    theta = map_z_to_theta(Z_NEUTRAL).clone()
    theta[name_to_idx["comp_low_threshold_db"]] = -24.0
    theta[name_to_idx["comp_low_ratio"]] = 4.0
    theta[name_to_idx["comp_low_attack_ms"]] = 120.0
    theta[name_to_idx["comp_low_release_ms"]] = 600.0
    theta[name_to_idx["comp_low_makeup_db"]] = 3.0
    theta[name_to_idx["comp_mid_threshold_db"]] = -24.0
    theta[name_to_idx["comp_mid_ratio"]] = 4.0
    theta[name_to_idx["comp_mid_attack_ms"]] = 80.0
    theta[name_to_idx["comp_mid_release_ms"]] = 600.0
    theta[name_to_idx["comp_mid_makeup_db"]] = 2.5
    z_max_tc = map_theta_to_z(theta)

    s_k = int(10.0 * sr)
    e_k = int(13.0 * sr)
    chunk_spec = ChunkSpec(
        chunk_id=0,
        start_sample=s_k,
        end_sample=e_k,
        start_sec=10.0,
        end_sec=13.0,
        duration_sec=3.0,
        rms_db=-10.0,
        crest_factor_db=10.0,
        spectral_centroid_hz=1500.0,
    )
    selector = MultiChunkSelector(sample_rate=float(sr), num_chunks=1, chunk_duration_sec=3.0)
    chain = DifferentiableMasteringChain(sample_rate=float(sr))
    same_loss_fn = MasteringPerceptualLoss(
        sample_rate=float(sr), weight_stft=1.0, weight_loudness=0.8, weight_param_reg=0.01, weight_erb=0.25
    )
    style_loss_fn = AggregatePerceptualLoss(
        sample_rate=float(sr), weight_spectral=1.0, weight_loudness=0.8, weight_param_reg=0.01, spectral_mode="erb48"
    )

    with torch.no_grad():
        full_prefix = torch.tensor(raw_audio_32[:, :e_k], dtype=torch.float32).unsqueeze(0)
        y_stream = chain(full_prefix, z=z_max_tc, eq_mode="iir")[..., s_k:e_k]
        y_ref_chunk = torch.tensor(ref_audio_32[:, s_k:e_k], dtype=torch.float32).unsqueeze(0)
        x_mix_chunk = torch.tensor(raw_audio_32[:, s_k:e_k], dtype=torch.float32).unsqueeze(0)
        target_lufs = same_loss_fn.loudness_loss.calculate_lufs(y_ref_chunk).item()

        gt_same, _ = same_loss_fn(y_stream, y_ref_chunk, target_lufs=target_lufs, normalized_params=z_max_tc)
        gt_style, _ = style_loss_fn(
            y_stream, y_ref_chunk, normalized_params=z_max_tc, target_lufs=target_lufs, mix_anchor_audio=x_mix_chunk
        )

    rows = []
    for t_pre in [0.0, 0.25, 1.0, 2.0, 2.5, 3.0]:
        ext_list, pre_counts = selector.extract_chunk_tensors_with_preroll(
            raw_audio_32, [chunk_spec], preroll_sec=t_pre
        )
        n_pre = pre_counts[0]
        x_ext = torch.tensor(ext_list[0], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            y_ext = chain(x_ext, z=z_max_tc, eq_mode="iir")
            y_eval = y_ext[..., n_pre:]
            diff = y_eval - y_stream
            rms_err = float(torch.sqrt(torch.mean(diff ** 2)).item())
            max_err = float(torch.max(torch.abs(diff)).item())
            rms_err_db = float(20.0 * math.log10(rms_err + 1e-12))
            max_err_db = float(20.0 * math.log10(max_err + 1e-12))

            l_same, _ = same_loss_fn(y_eval, y_ref_chunk, target_lufs=target_lufs, normalized_params=z_max_tc)
            l_style, _ = style_loss_fn(
                y_eval, y_ref_chunk, normalized_params=z_max_tc, target_lufs=target_lufs, mix_anchor_audio=x_mix_chunk
            )
            rel_same = abs(l_same.item() - gt_same.item()) / max(abs(gt_same.item()), 1e-8)
            rel_style = abs(l_style.item() - gt_style.item()) / max(abs(gt_style.item()), 1e-8)

            rows.append({
                "preroll_sec": float(t_pre),
                "preroll_samples": int(n_pre),
                "rms_diff_vs_streaming_dbfs": rms_err_db,
                "max_diff_vs_streaming_dbfs": max_err_db,
                "same_track_loss_iir": float(l_same.item()),
                "same_track_rel_diff_vs_streaming": float(rel_same),
                "stylistic_loss_iir": float(l_style.item()),
                "stylistic_rel_diff_vs_streaming": float(rel_style),
            })

    return {
        "track_prefix": "track01",
        "chunk_start_sec": 10.0,
        "chunk_end_sec": 13.0,
        "streaming_ground_truth_same_track_loss_iir": float(gt_same.item()),
        "streaming_ground_truth_stylistic_loss_iir": float(gt_style.item()),
        "convergence_rows": rows,
    }


def evaluate_single_session(
    spec: Dict[str, str],
    seed_int: int,
    v0_z_list: List[float],
) -> Dict[str, Any]:
    """Evaluate V0, V0p, V1, V1b for a single session with per-variant disk caching."""
    prefix = spec["track_prefix"]
    mode = spec["mode"]
    cache_dir = _REPO_ROOT / "docs" / "sprint3" / ".cache_v0p_v1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    session_cache_file = cache_dir / f"{prefix}.json"
    if session_cache_file.exists():
        print(f"=== Loaded cached session {prefix} ===", flush=True)
        return json.loads(session_cache_file.read_text(encoding="utf-8"))

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
    session_variants: Dict[str, Any] = {}

    for var_name, vcfg in VARIANTS_CFG.items():
        var_cache_file = cache_dir / f"{prefix}_{var_name}.json"
        if var_cache_file.exists():
            session_variants[var_name] = json.loads(var_cache_file.read_text(encoding="utf-8"))
            b = session_variants[var_name]["full_track_benchmark"]
            print(
                f"[{prefix}:{var_name}] (cached) init={session_variants[var_name]['initial_loss_z0_iir']:.4f} "
                f"-> final={session_variants[var_name]['final_loss_z_opt_iir']:.4f} "
                f"| LUFS={b['pred_lufs']:.2f} | PLR={b['pred_plr_db']:.2f} "
                f"| ERB_surplus={b['erb_presence_surplus_db']:+.2f} dB",
                flush=True,
            )
            continue

        v_t0 = time.time()
        ext_np, preroll_counts = selector.extract_chunk_tensors_with_preroll(
            raw_aligned, mix_chunks_spec, preroll_sec=vcfg["preroll_sec"]
        )
        mix_chunk_tensors = [torch.tensor(c, dtype=torch.float32).unsqueeze(0) for c in ext_np]

        opt = AutoMasteringOptimizer(
            sample_rate=48000.0,
            lr=0.04,
            num_steps=30,
            target_lufs=None,
            weight_stft=1.0,
            weight_loudness=0.8,
            weight_param_reg=0.01,
            use_hybrid_eq=True,
            weight_erb=vcfg["weight_erb"],
            stylistic_spectral_mode=vcfg["spectral_mode"],
            weight_de_harsh=vcfg["weight_de_harsh"],
        )

        if var_name == "V0":
            z_opt = torch.tensor(v0_z_list, dtype=torch.float32)
        else:
            z_opt, _, _ = opt.optimize_multi_chunk(
                mix_chunks=mix_chunk_tensors,
                ref_chunks=ref_chunk_tensors,
                reference_mode=mode,
                verbose=False,
                align_latency=True,
                mix_preroll_samples=preroll_counts if vcfg["preroll_sec"] > 0.0 else None,
            )

        K = len(mix_chunk_tensors)
        weights = [1.0 / K] * K
        with torch.no_grad():
            lufs_vals = [opt.loss_fn.loudness_loss.calculate_lufs(rc).item() for rc in ref_chunk_tensors]
            eff_target_lufs = float(sum(lufs_vals) / len(lufs_vals))

            def _eval_iir(z_vec: torch.Tensor) -> Tuple[float, Dict[str, float]]:
                tot = 0.0
                agg_m: Dict[str, float] = {}
                for k_idx in range(K):
                    n_pre_k = preroll_counts[k_idx]
                    mk = mix_chunk_tensors[k_idx]
                    pk_ext = opt.chain(mk, z=z_vec, eq_mode="iir")
                    pk = pk_ext[..., n_pre_k:] if n_pre_k > 0 else pk_ext
                    mk_eval = mk[..., n_pre_k:] if n_pre_k > 0 else mk
                    rk = ref_chunk_tensors[k_idx]
                    if mode == "stylistic":
                        cl, cm = opt.stylistic_loss_fn(
                            pred_audio=pk,
                            ref_audio=rk,
                            normalized_params=z_vec,
                            target_lufs=eff_target_lufs,
                            mix_anchor_audio=mk_eval,
                        )
                    else:
                        cl, cm = opt.loss_fn(
                            pred_audio=pk,
                            target_audio=rk,
                            target_lufs=eff_target_lufs,
                            normalized_params=z_vec,
                            align_latency=True,
                        )
                    tot += weights[k_idx] * float(cl.item())
                    for mk_key, mk_val in cm.items():
                        agg_m[mk_key] = agg_m.get(mk_key, 0.0) + weights[k_idx] * float(mk_val)
                return tot, agg_m

            init_iir, _ = _eval_iir(Z_NEUTRAL)
            final_iir, final_metrics = _eval_iir(z_opt)

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
            bench = suite.evaluate(
                mastered_eval_tensor, ref_eval_tensor, align_latency=False, reference_mode="same_track"
            )
        else:
            ref_eval_tensor = torch.tensor(ref_full_32, dtype=torch.float32).unsqueeze(0)
            mastered_eval_tensor = torch.tensor(
                mastered_full_64.astype(np.float32), dtype=torch.float32
            ).unsqueeze(0)
            bench = suite.evaluate(
                mastered_eval_tensor, ref_eval_tensor, align_latency=False, reference_mode="stylistic"
            )

        theta_opt = map_z_to_theta(z_opt)
        param_dict = {
            PARAM_SPECS[i].name: float(theta_opt[i].item()) for i in range(len(PARAM_SPECS))
        }

        var_payload = {
            "preroll_sec": vcfg["preroll_sec"],
            "preroll_samples_per_chunk": preroll_counts,
            "initial_loss_z0_iir": init_iir,
            "final_loss_z_opt_iir": final_iir,
            "loss_reduction_iir_pct": float(100.0 * (final_iir - init_iir) / max(abs(init_iir), 1e-9)),
            "chunk_metrics_final_iir": final_metrics,
            "full_track_benchmark": bench,
            "normalized_z_opt": [float(v) for v in z_opt.tolist()],
            "physical_parameters_theta": param_dict,
        }
        var_cache_file.write_text(json.dumps(var_payload, indent=2) + "\n", encoding="utf-8")
        session_variants[var_name] = var_payload

        print(
            f"[{prefix}:{var_name}] ({time.time() - v_t0:.1f}s) init={init_iir:.4f} -> final={final_iir:.4f} "
            f"| LUFS={bench['pred_lufs']:.2f} (ref={bench['target_lufs']:.2f}) "
            f"| PLR={bench['pred_plr_db']:.2f} | ERB_surplus={bench['erb_presence_surplus_db']:+.2f} dB",
            flush=True,
        )
        del opt, mix_chunk_tensors, mastered_full_64, mastered_eval_tensor, ref_eval_tensor
        gc.collect()

    print(f"=== Completed session {prefix} in {time.time() - t0:.1f}s ===", flush=True)
    res_obj = {
        "track_prefix": prefix,
        "reference_mode": mode,
        "seed": str(seed_int),
        "variants": session_variants,
    }
    session_cache_file.write_text(json.dumps(res_obj, indent=2) + "\n", encoding="utf-8")
    return res_obj


def main() -> None:
    t_start = time.time()
    v0_manifest_path = _REPO_ROOT / "docs" / "human_validation" / "SPRINT2_V0_REPRODUCIBILITY_MANIFEST.json"
    v0_manifest = json.loads(v0_manifest_path.read_text(encoding="utf-8"))
    seed_by_prefix = {s["track_prefix"]: int(s["seed"]) for s in v0_manifest["sessions"]}
    v0_z_by_prefix = {
        s["track_prefix"]: s["optimized_preset"]["normalized_z_41"]
        for s in v0_manifest["sessions"]
    }

    cache_dir = _REPO_ROOT / "docs" / "sprint3" / ".cache_v0p_v1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    preroll_cache = cache_dir / "preroll_conv.json"
    if preroll_cache.exists():
        preroll_conv_report = json.loads(preroll_cache.read_text(encoding="utf-8"))
        print("1. Loaded cached pre-roll convergence diagnostics.", flush=True)
    else:
        print("1. Computing pre-roll convergence diagnostics on track01...", flush=True)
        preroll_conv_report = compute_preroll_convergence_diagnostics()
        preroll_cache.write_text(json.dumps(preroll_conv_report, indent=2) + "\n", encoding="utf-8")
        print("   Pre-roll convergence diagnostics completed.", flush=True)
    gc.collect()

    ordered_sessions = []
    for spec in SESSIONS_SPEC:
        pref = spec["track_prefix"]
        res = evaluate_single_session(spec, seed_by_prefix[pref], v0_z_by_prefix[pref])
        ordered_sessions.append(res)

    output_payload = {
        "artifact_type": "AutoMastering_Sprint3_V0p_V1_Ablation_Results",
        "version": "1.0.0",
        "seed_serialization_note": (
            "All 64-bit unsigned integer seeds (uint64) in 'seed' are serialized as decimal strings "
            "to prevent IEEE-754 double-precision rounding in JavaScript parsers. Convert to integer "
            "via int(seed_str) in Python before seeding the RNG."
        ),
        "preroll_convergence_diagnostics": preroll_conv_report,
        "sessions": ordered_sessions,
    }

    out_path = _REPO_ROOT / "docs" / "sprint3" / "SPRINT3_V0P_V1_ABLATION_RESULTS.json"
    out_path.write_text(json.dumps(output_payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved ablation results to {out_path} in {time.time() - t_start:.1f}s total.", flush=True)


if __name__ == "__main__":
    main()
