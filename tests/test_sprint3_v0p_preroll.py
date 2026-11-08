"""Sprint 3 — V0p Pre-Roll / Warm-Up Convergence & Context Control Tests.

Verifies:
  1. `test_preroll_convergence`: Comparing T_pre in {0.25s, 1.0s, 2.0s, 2.5s, 3.0s}
     against full-track streaming state on a chunk starting at s_k = 10.0s under
     maximum compressor release (600 ms) and attack (120 ms) time constants.
     Confirms stabilization between 2.5s and 3.0s (signal RMS diff <= -60 dBFS,
     relative loss change < 1e-3).
  2. `test_preroll_early_chunk_uses_full_available_context`: Confirms that when a
     chunk starts at s_k < 2.5s (e.g. s_0 = 1.0s), all available context from the
     true start of the track (0 .. s_k - 1) is used without artificial mid-track
     state initialization, reproducing the exact full-track streaming state at s_k.
  3. `test_v0p_zero_preroll_reproduces_v0_identically`: Confirms 100% backward
     compatibility with V0 when mix_preroll_samples is None or 0.
"""

import math
import numpy as np
import pytest
import torch

from ddsp_mastering.data.chunking import ChunkSpec, MultiChunkSelector
from ddsp_mastering.dsp.chain import DifferentiableMasteringChain
from ddsp_mastering.loss.combined import MasteringPerceptualLoss
from ddsp_mastering.loss.stylistic import AggregatePerceptualLoss
from ddsp_mastering.optimize.optimizer import AutoMasteringOptimizer
from ddsp_mastering.param_space import PARAM_SPECS, Z_NEUTRAL, map_theta_to_z, map_z_to_theta


def _build_max_time_constant_z() -> torch.Tensor:
    """Construct a parameter vector z with maximum release (600 ms) and attack (120 ms)
    plus active compression so compressor envelope memory is maximally exercised.
    """
    name_to_idx = {spec.name: i for i, spec in enumerate(PARAM_SPECS)}
    theta = map_z_to_theta(Z_NEUTRAL).clone()

    # Set Low & Mid compressors to maximum release (600 ms) and attack (120 ms / 80 ms) with active GR
    theta[name_to_idx["comp_low_threshold_db"]] = -26.0
    theta[name_to_idx["comp_low_ratio"]] = 4.0
    theta[name_to_idx["comp_low_attack_ms"]] = 120.0
    theta[name_to_idx["comp_low_release_ms"]] = 600.0
    theta[name_to_idx["comp_low_makeup_db"]] = 3.0

    theta[name_to_idx["comp_mid_threshold_db"]] = -26.0
    theta[name_to_idx["comp_mid_ratio"]] = 4.0
    theta[name_to_idx["comp_mid_attack_ms"]] = 80.0
    theta[name_to_idx["comp_mid_release_ms"]] = 600.0
    theta[name_to_idx["comp_mid_makeup_db"]] = 2.5

    theta[name_to_idx["comp_high_threshold_db"]] = -24.0
    theta[name_to_idx["comp_high_ratio"]] = 4.0
    theta[name_to_idx["comp_high_attack_ms"]] = 50.0
    theta[name_to_idx["comp_high_release_ms"]] = 600.0
    theta[name_to_idx["comp_high_makeup_db"]] = 2.0

    return map_theta_to_z(theta)


def _generate_dynamic_test_track(duration_sec: float = 14.0, sr: int = 48000) -> np.ndarray:
    """Generate a deterministic stereo signal with strong burst dynamics before s_k = 10.0s."""
    rng = np.random.default_rng(12345)
    n_samples = int(round(duration_sec * sr))
    t = np.arange(n_samples, dtype=np.float64) / float(sr)

    # Multi-band harmonic carrier (Low + Mid + High)
    left = (
        0.35 * np.sin(2.0 * np.pi * 65.0 * t)
        + 0.25 * np.sin(2.0 * np.pi * 440.0 * t)
        + 0.15 * np.sin(2.0 * np.pi * 2200.0 * t)
        + 0.05 * rng.standard_normal(n_samples)
    )
    right = (
        0.35 * np.sin(2.0 * np.pi * 65.0 * t)
        + 0.25 * np.cos(2.0 * np.pi * 445.0 * t)
        + 0.15 * np.sin(2.0 * np.pi * 2350.0 * t)
        + 0.05 * rng.standard_normal(n_samples)
    )

    # Strong dynamic modulation envelope with a loud burst right before t = 10.0s
    env = 0.25 + 0.20 * np.sin(2.0 * np.pi * 0.4 * t)
    burst_mask = (t >= 7.0) & (t < 10.0)
    env[burst_mask] = 0.95
    # Evaluation region [10.0s, 13.0s] drops to moderate level so release tail from 600ms is active
    eval_mask = (t >= 10.0) & (t <= 13.0)
    env[eval_mask] = 0.30 + 0.10 * np.sin(2.0 * np.pi * 1.5 * (t[eval_mask] - 10.0))

    stereo = np.stack([left * env, right * env], axis=0).astype(np.float32)
    return np.ascontiguousarray(stereo)


def test_preroll_convergence():
    """Verify convergence across T_pre in {0.25s, 1.0s, 2.0s, 2.5s, 3.0s} vs full streaming."""
    sr = 48000
    audio_np = _generate_dynamic_test_track(duration_sec=14.0, sr=sr)
    z_max_tc = _build_max_time_constant_z()

    chain = DifferentiableMasteringChain(sample_rate= float(sr))
    same_loss_fn = MasteringPerceptualLoss(sample_rate=float(sr), target_lufs=-14.0)
    style_loss_fn = AggregatePerceptualLoss(sample_rate=float(sr))

    s_k = int(10.0 * sr)
    e_k = int(13.0 * sr)
    chunk_spec = ChunkSpec(
        chunk_id=0,
        start_sample=s_k,
        end_sample=e_k,
        start_sec=10.0,
        end_sec=13.0,
        duration_sec=3.0,
        rms_db=-12.0,
        crest_factor_db=10.0,
        spectral_centroid_hz=1200.0,
    )
    selector = MultiChunkSelector(sample_rate=float(sr), num_chunks=1, chunk_duration_sec=3.0)

    # 1. Exact full-track streaming ground truth from t = 0 to e_k
    full_prefix_t = torch.tensor(audio_np[:, :e_k], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        y_full_stream = chain(full_prefix_t, z=z_max_tc, eq_mode="iir")
        y_gt = y_full_stream[..., s_k:e_k]
        # Use y_gt shifted slightly as target reference so losses are non-zero and well-scaled
        y_ref = y_gt * 0.90

        gt_same_loss, _ = same_loss_fn(y_gt, y_ref, normalized_params=z_max_tc, align_latency=True)
        gt_style_loss, _ = style_loss_fn(y_gt, y_ref, normalized_params=z_max_tc, target_lufs=-14.0)

    preroll_durations = [0.25, 1.0, 2.0, 2.5, 3.0]
    rms_diffs_db = {}
    same_losses = {}
    style_losses = {}

    for t_pre in preroll_durations:
        ext_list, pre_counts = selector.extract_chunk_tensors_with_preroll(
            audio_np, [chunk_spec], preroll_sec=t_pre
        )
        n_pre = pre_counts[0]
        assert n_pre == int(round(t_pre * sr))
        x_ext = torch.tensor(ext_list[0], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            y_ext = chain(x_ext, z=z_max_tc, eq_mode="iir")
            y_eval = y_ext[..., n_pre:]

            diff_rms = torch.sqrt(torch.mean((y_eval - y_gt) ** 2)).item()
            rms_diffs_db[t_pre] = 20.0 * math.log10(diff_rms + 1e-12)

            l_same, _ = same_loss_fn(y_eval, y_ref, normalized_params=z_max_tc, align_latency=True)
            l_style, _ = style_loss_fn(y_eval, y_ref, normalized_params=z_max_tc, target_lufs=-14.0)
            same_losses[t_pre] = l_same.item()
            style_losses[t_pre] = l_style.item()

    # Verify monotonic improvement of signal state convergence
    assert rms_diffs_db[2.5] < rms_diffs_db[0.25] - 25.0
    assert rms_diffs_db[3.0] <= rms_diffs_db[2.5] + 1e-6

    # Between 2.5s and 3.0s (and vs ground truth), RMS difference must be <= -60 dBFS
    assert rms_diffs_db[2.5] <= -60.0, f"Expected <= -60 dBFS at 2.5s, got {rms_diffs_db[2.5]:.2f} dBFS"
    assert rms_diffs_db[3.0] <= -60.0, f"Expected <= -60 dBFS at 3.0s, got {rms_diffs_db[3.0]:.2f} dBFS"

    # Relative loss variation between 2.5s and 3.0s must be < 1e-3
    rel_diff_same = abs(same_losses[2.5] - same_losses[3.0]) / max(abs(gt_same_loss.item()), 1e-8)
    rel_diff_style = abs(style_losses[2.5] - style_losses[3.0]) / max(abs(gt_style_loss.item()), 1e-8)
    assert rel_diff_same < 1e-3, f"Relative same_track loss diff 2.5s vs 3.0s = {rel_diff_same:.6e}"
    assert rel_diff_style < 1e-3, f"Relative stylistic loss diff 2.5s vs 3.0s = {rel_diff_style:.6e}"


def test_preroll_early_chunk_uses_full_available_context():
    """Verify that when s_k < 2.5s (e.g. s_0 = 1.0s), all available context from t = 0 is used."""
    sr = 48000
    audio_np = _generate_dynamic_test_track(duration_sec=6.0, sr=sr)
    z_max_tc = _build_max_time_constant_z()
    chain = DifferentiableMasteringChain(sample_rate=float(sr))

    s_0 = int(1.0 * sr)  # 48000 samples (< 2.5s = 120000 samples)
    e_0 = int(4.0 * sr)
    chunk_spec = ChunkSpec(
        chunk_id=0,
        start_sample=s_0,
        end_sample=e_0,
        start_sec=1.0,
        end_sec=4.0,
        duration_sec=3.0,
        rms_db=-12.0,
        crest_factor_db=10.0,
        spectral_centroid_hz=1200.0,
    )
    selector = MultiChunkSelector(sample_rate=float(sr), num_chunks=1, chunk_duration_sec=3.0)
    ext_list, pre_counts = selector.extract_chunk_tensors_with_preroll(
        audio_np, [chunk_spec], preroll_sec=2.5
    )

    # Effective pre-roll must be exactly s_0 = 48000 samples (starting at sample 0)
    assert pre_counts[0] == s_0
    assert ext_list[0].shape == (2, e_0)

    with torch.no_grad():
        y_full = chain(torch.tensor(audio_np[:, :e_0]).unsqueeze(0), z=z_max_tc, eq_mode="iir")
        y_gt = y_full[..., s_0:e_0]

        y_ext = chain(torch.tensor(ext_list[0]).unsqueeze(0), z=z_max_tc, eq_mode="iir")
        y_eval = y_ext[..., pre_counts[0]:]

    max_err = torch.max(torch.abs(y_eval - y_gt)).item()
    assert max_err == 0.0, f"Expected exact zero difference from t=0 context, got {max_err}"


def test_v0p_zero_preroll_reproduces_v0_identically():
    """Confirm that optimize_multi_chunk with mix_preroll_samples=[0, ...] matches None (V0)."""
    sr = 48000
    audio_np = _generate_dynamic_test_track(duration_sec=6.0, sr=sr)
    c1 = torch.tensor(audio_np[:, 0 : sr * 2], dtype=torch.float32).unsqueeze(0)
    r1 = (c1 * 1.15).clamp(-0.98, 0.98)

    opt_v0 = AutoMasteringOptimizer(sample_rate=float(sr), lr=0.04, num_steps=3)
    z_v0, chunks_v0, hist_v0 = opt_v0.optimize_multi_chunk(
        mix_chunks=[c1],
        ref_chunks=[r1],
        reference_mode="same_track",
    )

    opt_v0p_zero = AutoMasteringOptimizer(sample_rate=float(sr), lr=0.04, num_steps=3)
    z_v0p, chunks_v0p, hist_v0p = opt_v0p_zero.optimize_multi_chunk(
        mix_chunks=[c1],
        ref_chunks=[r1],
        reference_mode="same_track",
        mix_preroll_samples=[0],
    )

    assert torch.allclose(z_v0, z_v0p, atol=1e-7)
    assert torch.allclose(chunks_v0[0], chunks_v0p[0], atol=1e-7)
    assert np.allclose(hist_v0["loss"], hist_v0p["loss"], atol=1e-7)
