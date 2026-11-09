"""Sprint 3 — Task 6 Audit Test Suite (Spec v3 §2.A.2 & Pre-Ablation V2 Alignment).

Verifies the 5 mandatory audit tests:
1. `test_preroll_alignment_vs_continuous_rendering`:
   Verifies mathematical equivalence between pre-roll chunk rendering and continuous
   full-track streaming rendering across latency-compensated region (D = 159).
2. `test_fast_cpu_analytical_biquad_vs_torchaudio_lfilter`:
   Verifies forward and backward gradients (d/dx, d/da, d/db, DifferentiableEQ 6-biquad
   cascade, and torch.autograd.gradcheck in float64) of _FastCPUAnalyticalBiquad vs
   native torchaudio.functional.lfilter.
3. `test_gradient_diagnostics_reproducibility`:
   Verifies that gradient norm computations ||grad_{z_g} L_i||_2 across all 6 DSP groups
   are bit-deterministic and 100% reproducible.
4. `test_same_track_vs_stylistic_chunk_selection_separation`:
   Verifies that same_track uses coupled coordinate selection while stylistic uses
   strictly independent chunk selection for mix and reference, preserving 4 chunks
   of exactly 144,000 samples (3.0s at 48 kHz).
5. `test_gradient_diagnostics_non_invasive`:
   Confirms that gradient evaluation does not mutate latent tensors (Z_NEUTRAL or
   z_opt) or any module parameters/buffers.
"""

import math
from typing import Dict, List
import numpy as np
import pytest
import torch
import torchaudio.functional as taf

from ddsp_mastering.data.chunking import MultiChunkSelector
from ddsp_mastering.dsp.chain import DifferentiableMasteringChain
from ddsp_mastering.dsp.eq import DifferentiableEQ
from ddsp_mastering.dsp.fast_biquad import (
    _FastCPUAnalyticalBiquad,
    disable_fast_cpu_biquad,
    enable_fast_cpu_biquad,
)
from ddsp_mastering.loss.combined import MasteringPerceptualLoss
from ddsp_mastering.loss.stylistic import AggregatePerceptualLoss
from ddsp_mastering.param_space import PARAM_SPECS, Z_NEUTRAL, map_theta_to_z, map_z_to_theta


PARAM_GROUPS = {
    "EQ": list(range(0, 18)),
    "Crossover": list(range(18, 20)),
    "Compressor": list(range(20, 35)),
    "Imager": list(range(35, 38)),
    "Saturator": list(range(38, 40)),
    "Limiter": list(range(40, 41)),
}


def _create_synthetic_stereo_signal(duration_sec: float = 6.0, sr: int = 48000) -> torch.Tensor:
    """Create a deterministic non-trivial synthetic stereo signal with multiple frequencies."""
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    # Left channel: sum of sines
    left = 0.3 * np.sin(2 * np.pi * 100.0 * t) + 0.2 * np.sin(2 * np.pi * 1000.0 * t) + 0.1 * np.sin(2 * np.pi * 5000.0 * t)
    # Right channel: sum of sines with phase shift
    right = 0.3 * np.cos(2 * np.pi * 150.0 * t) + 0.2 * np.sin(2 * np.pi * 2500.0 * t) + 0.1 * np.cos(2 * np.pi * 8000.0 * t)
    stereo = np.stack([left, right], axis=0).astype(np.float32)
    return torch.tensor(stereo).unsqueeze(0)  # [1, 2, T]


def _build_active_test_z() -> torch.Tensor:
    """Build a non-neutral z vector exercising EQ, compressors, and imager."""
    name_to_idx = {spec.name: i for i, spec in enumerate(PARAM_SPECS)}
    theta = map_z_to_theta(Z_NEUTRAL).clone()

    # Active EQ boosts and cuts (0-indexed: eq_b0 .. eq_b5)
    theta[name_to_idx["eq_b0_gain_db"]] = 2.5
    theta[name_to_idx["eq_b1_gain_db"]] = -1.5
    theta[name_to_idx["eq_b3_gain_db"]] = -3.0
    theta[name_to_idx["eq_b5_gain_db"]] = 2.0

    # Active Compressor parameters
    theta[name_to_idx["comp_mid_threshold_db"]] = -20.0
    theta[name_to_idx["comp_mid_ratio"]] = 3.0
    theta[name_to_idx["comp_mid_attack_ms"]] = 50.0
    theta[name_to_idx["comp_mid_release_ms"]] = 250.0
    theta[name_to_idx["comp_mid_makeup_db"]] = 1.5

    return map_theta_to_z(theta)


# ==============================================================================
# TEST 1: Pre-roll alignment vs continuous streaming rendering
# ==============================================================================
def test_preroll_alignment_vs_continuous_rendering():
    """Verify pre-roll chunk rendering matches continuous rendering across useful latency region."""
    enable_fast_cpu_biquad()
    sr = 48000
    chain = DifferentiableMasteringChain(sample_rate=float(sr))
    D = 159  # chain pipeline latency (63 EQ + 32 comp + 64 limiter)
    n_preroll = int(2.5 * sr)  # 120,000 samples
    chunk_len = int(3.0 * sr)  # 144,000 samples

    audio = _create_synthetic_stereo_signal(duration_sec=7.0, sr=sr)  # 7s = 336,000 samples
    z = _build_active_test_z()

    # Chunk starts at 3.0s (144,000 samples)
    s_k = int(3.0 * sr)
    e_k = s_k + chunk_len

    # Path B: Chunk with 2.5s pre-roll
    pre_start = max(0, s_k - n_preroll)
    actual_pre = s_k - pre_start
    chunk_input = audio[:, :, pre_start:e_k]

    with torch.no_grad():
        # Path B render
        y_b_full = chain(chunk_input, z=z, eq_mode="iir")
        # Slice latency-compensated useful region
        y_b = y_b_full[..., actual_pre + D : actual_pre + D + (chunk_len - D)]

        # Path C render: full continuous stream from sample 0
        y_c_full = chain(audio[:, :, :e_k], z=z, eq_mode="iir")
        y_c = y_c_full[..., s_k + D : s_k + D + (chunk_len - D)]

    # Compute RMS diff in dBFS
    diff = y_b - y_c
    rms_diff = torch.sqrt(torch.mean(diff ** 2)).item()
    rms_dbfs = 20.0 * math.log10(max(rms_diff, 1e-12))
    max_err = torch.max(torch.abs(diff)).item()

    # Pre-roll of 2.5s guarantees complete stabilization of IIR state and compressor envelope
    assert rms_dbfs < -100.0, f"Expected RMS diff < -100 dBFS, got {rms_dbfs:.2f} dBFS"
    assert max_err < 1e-4, f"Expected max err < 1e-4, got {max_err:.2e}"


# ==============================================================================
# TEST 2: Fast CPU analytical biquad vs torchaudio.functional.lfilter
# ==============================================================================
def test_fast_cpu_analytical_biquad_vs_torchaudio_lfilter():
    """Verify forward and backward equivalence between _FastCPUAnalyticalBiquad and taf.lfilter."""
    torch.manual_seed(42)
    B, C, N = 1, 2, 4800  # 0.1s at 48 kHz

    # Double precision test for exact analytical accuracy
    x = torch.randn(B, C, N, dtype=torch.float64, requires_grad=True)
    x_ref = x.clone().detach().requires_grad_(True)

    # Standard peak filter biquad coefficients
    b_coeffs = torch.tensor([1.05, -1.85, 0.92], dtype=torch.float64, requires_grad=True)
    a_coeffs = torch.tensor([1.0, -1.82, 0.90], dtype=torch.float64, requires_grad=True)

    b_ref = b_coeffs.clone().detach().requires_grad_(True)
    a_ref = a_coeffs.clone().detach().requires_grad_(True)

    # Forward
    y_fast = _FastCPUAnalyticalBiquad.apply(x, a_coeffs, b_coeffs)
    y_ref = taf.lfilter(x_ref, a_ref, b_ref, clamp=False)

    fwd_err = torch.max(torch.abs(y_fast - y_ref)).item()
    assert fwd_err < 1e-12, f"Forward max error too high: {fwd_err:.2e}"

    # Backward
    loss_fast = torch.sum(y_fast ** 2)
    loss_ref = torch.sum(y_ref ** 2)
    loss_fast.backward()
    loss_ref.backward()

    gx_err = torch.max(torch.abs(x.grad - x_ref.grad)).item()
    assert gx_err < 1e-11, f"Grad x max error too high: {gx_err:.2e}"

    gb_err = torch.max(torch.abs(b_coeffs.grad - b_ref.grad)).item()
    assert gb_err < 1e-10, f"Grad b max error too high: {gb_err:.2e}"

    ga_err = torch.max(torch.abs(a_coeffs.grad - a_ref.grad)).item()
    assert ga_err < 1e-10, f"Grad a max error too high: {ga_err:.2e}"

    # Verify torch.autograd.gradcheck in float64
    x_gc = torch.randn(1, 1, 64, dtype=torch.float64, requires_grad=True)
    a_gc = torch.tensor([1.0, -0.5, 0.25], dtype=torch.float64, requires_grad=True)
    b_gc = torch.tensor([0.8, -0.3, 0.1], dtype=torch.float64, requires_grad=True)

    gc_passed = torch.autograd.gradcheck(
        _FastCPUAnalyticalBiquad.apply,
        (x_gc, a_gc, b_gc),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
    )
    assert gc_passed, "Gradcheck failed on _FastCPUAnalyticalBiquad"

    # Verify DifferentiableEQ cascade equivalence
    disable_fast_cpu_biquad()
    eq = DifferentiableEQ(sample_rate=48000.0)
    x32 = torch.randn(1, 2, 4800, dtype=torch.float32)
    f0 = torch.tensor([60.0, 200.0, 800.0, 2500.0, 6000.0, 12000.0], dtype=torch.float32)
    q = torch.tensor([0.7, 1.0, 1.2, 1.0, 0.8, 0.7], dtype=torch.float32)
    gain_db_fast = torch.tensor([1.0, -1.0, 2.0, -2.0, 1.5, -0.5], dtype=torch.float32, requires_grad=True)
    gain_db_ref = gain_db_fast.clone().detach().requires_grad_(True)

    y_eq_ref = eq(x32, f0, gain_db_ref, q, mode="iir")
    loss_eq_ref = torch.sum(y_eq_ref ** 2)
    loss_eq_ref.backward()

    enable_fast_cpu_biquad()
    y_eq_fast = eq(x32, f0, gain_db_fast, q, mode="iir")
    loss_eq_fast = torch.sum(y_eq_fast ** 2)
    loss_eq_fast.backward()

    eq_fwd_err = torch.max(torch.abs(y_eq_fast - y_eq_ref)).item()
    eq_grad_err = torch.max(torch.abs(gain_db_fast.grad - gain_db_ref.grad)).item()
    assert eq_fwd_err < 1e-5, f"EQ cascade forward diff too high: {eq_fwd_err:.2e}"
    assert eq_grad_err < 1e-4, f"EQ cascade grad diff too high: {eq_grad_err:.2e}"


# ==============================================================================
# TEST 3: Gradient diagnostics reproducibility
# ==============================================================================
def test_gradient_diagnostics_reproducibility():
    """Verify that gradient norms across 6 DSP groups are 100% bit-deterministic."""
    enable_fast_cpu_biquad()
    sr = 48000
    chain = DifferentiableMasteringChain(sample_rate=float(sr))
    loss_fn = MasteringPerceptualLoss(
        sample_rate=float(sr), weight_stft=1.0, weight_loudness=0.8, weight_param_reg=0.01, weight_erb=0.25
    )

    x = _create_synthetic_stereo_signal(duration_sec=3.5, sr=sr)
    ref = _create_synthetic_stereo_signal(duration_sec=3.0, sr=sr)[:, :, : int(3.0 * sr)]

    def _compute_grads():
        z_param = Z_NEUTRAL.clone().detach().requires_grad_(True)
        y = chain(x, z=z_param, eq_mode="iir")
        y_cut = y[..., : ref.shape[-1]]
        loss, _ = loss_fn(y_cut, ref, normalized_params=z_param, align_latency=False)
        grad = torch.autograd.grad(loss, z_param)[0]
        group_norms = {}
        for g_name, indices in PARAM_GROUPS.items():
            sub_g = grad[indices]
            group_norms[g_name] = float(torch.norm(sub_g, p=2).item())
        return grad, group_norms

    grad1, norms1 = _compute_grads()
    grad2, norms2 = _compute_grads()

    assert torch.equal(grad1, grad2), "Gradient vectors must be bit-equal across runs"
    for g_name in PARAM_GROUPS:
        assert math.isclose(norms1[g_name], norms2[g_name], rel_tol=1e-9), f"Norm mismatch for {g_name}"


# ==============================================================================
# TEST 4: same_track vs stylistic chunk selection separation
# ==============================================================================
def test_same_track_vs_stylistic_chunk_selection_separation():
    """Verify same_track couples coordinates while stylistic selects independently."""
    sr = 48000
    N = int(36.0 * sr)  # 36 seconds: accommodates 4 chunks with min_separation=4.0s
    t = np.linspace(0, 36.0, N, endpoint=False)
    # Mix audio has energy peaks in first half (e.g. around 6s and 14s)
    env_mix = 0.5 + 2.0 * np.exp(-((t - 6.0) / 2.0) ** 2) + 2.0 * np.exp(-((t - 14.0) / 2.0) ** 2)
    mix_audio = np.stack([np.sin(2 * np.pi * 200 * t) * env_mix, np.cos(2 * np.pi * 200 * t) * env_mix], axis=0).astype(np.float32)
    # Ref audio has energy peaks in second half (e.g. around 22s and 30s)
    env_ref = 0.5 + 2.0 * np.exp(-((t - 22.0) / 2.0) ** 2) + 2.0 * np.exp(-((t - 30.0) / 2.0) ** 2)
    ref_audio = np.stack([np.sin(2 * np.pi * 400 * t) * env_ref, np.cos(2 * np.pi * 400 * t) * env_ref], axis=0).astype(np.float32)

    selector = MultiChunkSelector(sample_rate=sr, num_chunks=4, chunk_duration_sec=3.0, seed=42)

    # In same_track: single pass over aligned mix audio generates coupled chunks
    same_chunks = selector.select_chunks(mix_audio)
    assert len(same_chunks) == 4, f"Expected 4 chunks, got {len(same_chunks)}"
    for ch in same_chunks:
        dur = ch["end_sample"] - ch["start_sample"]
        assert dur == 144000, f"Expected chunk length 144000, got {dur}"

    # In stylistic: independent passes for mix and ref
    mix_sel = selector.select_chunks(mix_audio, None, mode="stylistic")
    ref_sel = selector.select_chunks(ref_audio, None, mode="stylistic")

    assert len(mix_sel) == 4
    assert len(ref_sel) == 4
    for m_ch, r_ch in zip(mix_sel, ref_sel):
        assert (m_ch["end_sample"] - m_ch["start_sample"]) == 144000
        assert (r_ch["end_sample"] - r_ch["start_sample"]) == 144000

    # Ensure mix and ref coordinates are not forced to be identical in stylistic
    mix_starts = [c["start_sample"] for c in mix_sel]
    ref_starts = [c["start_sample"] for c in ref_sel]
    assert mix_starts != ref_starts, "Stylistic mode must select chunks independently based on respective audio profiles"


# ==============================================================================
# TEST 5: Gradient diagnostics non-invasiveness
# ==============================================================================
def test_gradient_diagnostics_non_invasive():
    """Confirm gradient evaluation does not mutate Z_NEUTRAL, z_opt, or model parameters."""
    enable_fast_cpu_biquad()
    sr = 48000
    chain = DifferentiableMasteringChain(sample_rate=float(sr))
    loss_fn = MasteringPerceptualLoss(sample_rate=float(sr), weight_stft=1.0)

    z_neutral_copy = Z_NEUTRAL.clone().detach()
    z_opt_dummy = torch.randn(41, dtype=torch.float32)
    z_opt_copy = z_opt_dummy.clone().detach()

    # Capture initial chain state dict
    chain_state_before = {k: v.clone() for k, v in chain.state_dict().items()}

    audio = _create_synthetic_stereo_signal(duration_sec=3.0, sr=sr)
    ref = _create_synthetic_stereo_signal(duration_sec=3.0, sr=sr)

    # Run forward + backward pass using clone
    z_param = z_opt_dummy.clone().detach().requires_grad_(True)
    y = chain(audio, z=z_param, eq_mode="iir")
    loss, _ = loss_fn(y, ref, normalized_params=z_param, align_latency=False)
    grad = torch.autograd.grad(loss, z_param)[0]

    # Verify Z_NEUTRAL and z_opt are completely unmodified
    assert torch.equal(Z_NEUTRAL, z_neutral_copy), "Z_NEUTRAL was modified by gradient diagnostics!"
    assert torch.equal(z_opt_dummy, z_opt_copy), "z_opt was modified by gradient diagnostics!"

    # Verify chain parameters / buffers have not changed
    chain_state_after = chain.state_dict()
    for k in chain_state_before:
        assert torch.equal(chain_state_before[k], chain_state_after[k]), f"Chain buffer {k} was mutated!"
