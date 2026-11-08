"""Sprint 3 Step V2 Ablation Runner: V0p vs V2 (Transient/Envelope & Reference-Aware PLR Loss).

Evaluates:
  - V0p: Context control baseline (2.5s pre-roll, historical Sprint 2 loss).
  - V2:  2.5s pre-roll + Transient/Envelope Loss (weight_trans=0.2) + Reference-Aware
         PLR Loss with Boltzmann soft-peak mean (plr_mode="reference_aware").

Runs multi-chunk optimization (num_steps=30, lr=0.04, exact Sprint 2 seeds and chunk coordinates)
across all 6 sessions (track01..03 same_track and track01..03_style stylistic),
renders full-track offline C++ output, and benchmarks objective metrics.
Exports results to docs/sprint3/SPRINT3_V2_ABLATION_RESULTS.json.
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
    """Fast float64 analytical IIR biquad using SciPy lfilter on CPU."""

    @staticmethod
    def forward(ctx, waveform: torch.Tensor, a_coeffs: torch.Tensor, b_coeffs: torch.Tensor) -> torch.Tensor:
        orig_shape = waveform.shape
        orig_dtype = waveform.dtype
        device = waveform.device
        N = orig_shape[-1]

        x_2d = waveform.detach().to(torch.float64).reshape(-1, N)
        M = x_2d.shape[0]
        x_np = np.ascontiguousarray(x_2d.cpu().numpy(), dtype=np.float64)

        a_2d = a_coeffs.detach().to(torch.float64).reshape(-1, 3).cpu().numpy()
        b_2d = b_coeffs.detach().to(torch.float64).reshape(-1, 3).cpu().numpy()

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
        gy_np = np.ascontiguousarray(grad_output.detach().to(torch.float64).reshape(-1, N).cpu().numpy(), dtype=np.float64)

        a_2d = a_coeffs.detach().reshape(-1, 3).cpu().numpy()
        b_2d = b_coeffs.detach().reshape(-1, 3).cpu().numpy()
        a0 = a_2d[:, 0:1]
        b_norm = b_2d / a0
        a_norm = a_2d / a0

        gy_rev = np.ascontiguousarray(gy_np[:, ::-1])
        M = x_np.shape[0]

        if b_norm.shape[0] == 1 or np.allclose(b_norm, b_norm[0:1]) and np.allclose(a_norm, a_norm[0:1]):
            gx_rev = scipy_lfilter(b_norm[0], a_norm[0], gy_rev, axis=-1)
            u_rev = scipy_lfilter([1.0], a_norm[0], gy_rev, axis=-1)
        else:
            gx_rev = np.empty_like(gy_rev)
            u_rev = np.empty_like(gy_rev)
            for m in range(M):
                idx = min(m, b_norm.shape[0] - 1)
                gx_rev[m] = scipy_lfilter(b_norm[idx], a_norm[idx], gy_rev[m], axis=-1)
                u_rev[m] = scipy_lfilter([1.0], a_norm[idx], gy_rev[m], axis=-1)

        gx_np = np.ascontiguousarray(gx_rev[:, ::-1])
        u_np = np.ascontiguousarray(u_rev[:, ::-1])

        gb0 = np.sum(gy_np * x_np, axis=-1, keepdims=True)
        gb1 = np.sum(gy_np[:, 1:] * x_np[:, :-1], axis=-1, keepdims=True)
        gb2 = np.sum(gy_np[:, 2:] * x_np[:, :-2], axis=-1, keepdims=True)
        gb_norm = np.concatenate([gb0, gb1, gb2], axis=-1)

        ga1 = -np.sum(u_np[:, 1:] * y_np[:, :-1], axis=-1, keepdims=True)
        ga2 = -np.sum(u_np[:, 2:] * y_np[:, :-2], axis=-1, keepdims=True)
        ga_norm = np.concatenate([np.zeros_like(ga1), ga1, ga2], axis=-1)

        grad_x = torch.from_numpy(gx_np).to(device=device, dtype=grad_output.dtype).reshape(orig_shape)
        grad_b = torch.from_numpy(gb_norm / a0).to(device=device, dtype=b_coeffs.dtype).reshape(b_coeffs.shape)
        grad_a = torch.from_numpy(ga_norm / a0).to(device=device, dtype=a_coeffs.dtype).reshape(a_coeffs.shape)

        return grad_x, grad_a, grad_b


def _patched_taf_lfilter(waveform, a_coeffs, b_coeffs, clamp=False, batching=True):
    if waveform.device.type == "cpu":
        return _FastCPUAnalyticalBiquad.apply(waveform, a_coeffs, b_coeffs)
    return _ORIG_TAF_LFILTER(waveform, a_coeffs, b_coeffs, clamp=clamp, batching=batching)


_ORIG_TAF_LFILTER = taf.lfilter
taf.lfilter = _patched_taf_lfilter

from ddsp_mastering.data.chunking import MultiChunkSelector
from scripts.human_validation.validate_inputs import load_and_validate_wav
from ddsp_mastering.dsp.chain import DifferentiableMasteringChain
from ddsp_mastering.dsp.alignment import apply_same_track_alignment, estimate_time_alignment
from ddsp_mastering.optimize.benchmark import MasteringBenchmarkSuite
from ddsp_mastering.optimize.optimizer import AutoMasteringOptimizer
from ddsp_mastering.param_space import PARAM_SPECS, Z_NEUTRAL, map_z_to_theta

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
        "ref_path": "human_validation/private/input/track01_style_ref.wav",
    },
    {
        "track_prefix": "track02_style",
        "mode": "stylistic",
        "mix_path": "human_validation/private/input/track02_mix.wav",
        "ref_path": "human_validation/private/input/track02_style_ref.wav",
    },
    {
        "track_prefix": "track03_style",
        "mode": "stylistic",
        "mix_path": "human_validation/private/input/track03_mix.wav",
        "ref_path": "human_validation/private/input/track03_style_ref.wav",
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
                    y_ext = opt.chain(mix_chunk_tensors[k], z=z_in, eq_mode="iir")
                    y_k = y_ext[..., n_pre:]
                    ref_k = ref_chunk_tensors[k]
                    if mode == "same_track":
                        lk, mk = opt.loss_fn(
                            y_k,
                            ref_k,
                            target_lufs=eff_target_lufs,
                            normalized_params=z_in,
                            align_latency=True,
                        )
                    else:
                        anchor_k = mix_chunk_tensors[k][..., n_pre:]
                        lk, mk = opt.stylistic_loss_fn(
                            y_k,
                            ref_k,
                            normalized_params=z_in,
                            target_lufs=eff_target_lufs,
                            mix_anchor_audio=anchor_k,
                        )
                    tot_l += weights[k] * float(lk.item())
                    for mk_key, mk_val in mk.items():
                        agg_m[mk_key] = agg_m.get(mk_key, 0.0) + weights[k] * float(mk_val)
                return tot_l, agg_m

        loss_z0_iir, _ = _eval_loss_iir(Z_NEUTRAL)
        loss_zopt_iir, metrics_final = _eval_loss_iir(z_opt)
        reduction_pct = (
            ((loss_zopt_iir - loss_z0_iir) / loss_z0_iir) * 100.0 if loss_z0_iir > 0.0 else 0.0
        )

        # Full-track offline C++ render
        chain_eval = DifferentiableMasteringChain(sample_rate=48000.0)
        mastered_wav_32 = chain_eval.render_offline_with_tail_flush(
            raw_aligned,
            z=z_opt,
            eq_mode="iir",
            block_size=512,
            tail_duration_sec=3.0,
        )

        bm_full = suite.evaluate_master(
            pred_audio=mastered_wav_32,
            target_audio=ref_aligned if mode == "same_track" else ref_full_32,
            reference_mode=mode,
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
            f"({reduction_pct:+.2f}%) | LUFS={bm_full['pred_lufs']:.2f} | PLR={bm_full['pred_plr_db']:.2f} "
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
        "sessions": session_results,
    }
    out_file.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
    print(f"\n[DONE] Saved V2 ablation results to: {out_file}")


if __name__ == "__main__":
    main()
