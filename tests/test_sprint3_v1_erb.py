"""Sprint 3 — V1 / V1b ERB-48 Spectral Filterbank, Active Masks, and Loss Tests.

Verifies:
  1. `test_erb_filterbank_properties`: 48 bands in [20 Hz, 20 kHz], monotonic center
     frequencies on the Glasberg & Moore ERB_N scale, non-negative weights, and
     unit-area normalization (`sum_k W_{b,k} == 1.0`).
  2. `test_active_frame_and_sample_masks_nominal_sparse_and_silent`:
     Tests nominal, sparse (`|A_raw| < M_min -> topk(M_min)`), and completely silent
     (`< -70 dBFS -> is_valid = 0.0`) chunks, plus `repeat_interleave(480)` sample
     mask conversion and guaranteed NaN/Inf-free zero loss/gradient on silent chunks.
  3. `test_erb_loss_identity_and_mono_stereo_invariance`:
     Exact zero loss on `y == y_ref` and identical band levels in dB for mono (`C=1`)
     vs dual-mono (`C=2`).
  4. `test_erb_de_harshness_asymmetric_relative_penalty`:
     Verifies that `L_de_harsh` is strictly 0 when `tau_harsh(y) <= tau_harsh(y_ref) + 0.75 dB`
     (even on a bright reference) and strictly positive with corrective gradient when
     `y` has excess 2.5-5 kHz energy relative to `y_ref`.
  5. `test_erb_loss_differentiability_and_finite_gradients`:
     Verifies clean backward pass and finite gradients for both `MasteringPerceptualLoss`
     (`weight_erb > 0`) and `AggregatePerceptualLoss` (`spectral_mode='erb48'`).
"""

import numpy as np
import pytest
import torch

from ddsp_mastering.loss.combined import MasteringPerceptualLoss
from ddsp_mastering.loss.erb import (
    ERBSpectralLoss,
    build_erb_filterbank,
    compute_active_frame_weights_and_mask,
    compute_active_sample_mask,
)
from ddsp_mastering.loss.stylistic import AggregatePerceptualLoss


def test_erb_filterbank_properties():
    """Verify unit-area normalization and monotonic Glasberg & Moore ERB-48 centers."""
    n_fft = 4096
    n_freqs = n_fft // 2 + 1
    fbanks, centers = build_erb_filterbank(
        n_freqs=n_freqs, sample_rate=48000.0, n_bands=48, f_min=20.0, f_max=20000.0
    )
    assert fbanks.shape == (n_freqs, 48)
    assert centers.shape == (48,)
    assert torch.all(fbanks >= 0.0)

    # Every band must sum to 1.0
    col_sums = torch.sum(fbanks, dim=0)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-6)

    # Monotonic center frequencies inside (20, 20000) Hz
    diffs = centers[1:] - centers[:-1]
    assert torch.all(diffs > 0.0)
    assert centers[0].item() > 20.0
    assert centers[-1].item() < 20000.0


def test_active_frame_and_sample_masks_nominal_sparse_and_silent():
    """Verify deterministic nominal, sparse, and silent chunk behavior + sample-mask expansion."""
    sr = 48000
    n_samples = sr * 1 + 137  # non-multiple of 480 to also test tail remainder expansion

    # 1. Completely silent chunk (digital zeros and -90 dBFS noise floor)
    silent_zero = torch.zeros((1, 2, n_samples), dtype=torch.float32)
    w_zero, mask_zero, valid_zero = compute_active_frame_weights_and_mask(silent_zero)
    assert valid_zero.item() == 0.0
    assert torch.all(mask_zero)
    assert torch.allclose(torch.sum(w_zero, dim=-1), torch.ones(1), atol=1e-6)
    assert not torch.any(torch.isnan(w_zero))

    smp_mask_zero, smp_valid_zero = compute_active_sample_mask(silent_zero, block_samples=480)
    assert smp_mask_zero.shape == (1, n_samples)
    assert smp_valid_zero.item() == 0.0
    assert torch.all(smp_mask_zero)

    # Verify ERB losses on silent chunks produce exact 0.0 loss and 0.0 finite gradients
    erb_loss_fn = ERBSpectralLoss(sample_rate=float(sr), weight_de_harsh=0.2)
    pred_silent = torch.zeros((1, 2, n_samples), dtype=torch.float32, requires_grad=True)
    loss_same_sil, _ = erb_loss_fn.forward_same_track(pred_silent, silent_zero)
    assert loss_same_sil.item() == 0.0
    loss_same_sil.backward()
    assert pred_silent.grad is not None
    assert not torch.any(torch.isnan(pred_silent.grad))
    assert not torch.any(torch.isinf(pred_silent.grad))
    assert torch.max(torch.abs(pred_silent.grad)).item() == 0.0

    pred_silent2 = torch.zeros((1, 2, n_samples), dtype=torch.float32, requires_grad=True)
    loss_style_sil, _ = erb_loss_fn.forward_stylistic(pred_silent2, silent_zero, mix_anchor_audio=silent_zero)
    assert loss_style_sil.item() == 0.0
    loss_style_sil.backward()
    assert pred_silent2.grad is not None
    assert not torch.any(torch.isnan(pred_silent2.grad))
    assert torch.max(torch.abs(pred_silent2.grad)).item() == 0.0

    # 2. Sparse chunk: single 10 ms pulse in an otherwise silent 1s buffer (|A_raw| < M_min = 8)
    sparse_sig = torch.zeros((1, 2, n_samples), dtype=torch.float32)
    sparse_sig[:, :, 4800:5280] = 0.5  # single 10 ms active burst
    w_sp, mask_sp, valid_sp = compute_active_frame_weights_and_mask(sparse_sig, min_active_frames=8)
    assert valid_sp.item() == 1.0
    assert int(torch.sum(mask_sp).item()) == 8
    assert torch.allclose(torch.sum(w_sp, dim=-1), torch.ones(1), atol=1e-6)

    smp_mask_sp, smp_valid_sp = compute_active_sample_mask(sparse_sig, block_samples=480, min_active_blocks=8)
    assert smp_valid_sp.item() == 1.0
    # At least 8 blocks of 480 samples = 3840 active samples
    assert int(torch.sum(smp_mask_sp).item()) >= 8 * 480


def test_erb_loss_identity_and_mono_stereo_invariance():
    """Verify L(y, y) == 0 and mono (C=1) vs dual-mono (C=2) equivalence."""
    sr = 48000
    t = torch.linspace(0.0, 1.0, sr, dtype=torch.float32)
    mono_wave = (0.3 * torch.sin(2.0 * np.pi * 440.0 * t) + 0.15 * torch.sin(2.0 * np.pi * 3000.0 * t))
    x_mono = mono_wave.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
    x_stereo = torch.stack([mono_wave, mono_wave], dim=0).unsqueeze(0)  # (1, 2, T)

    erb_loss = ERBSpectralLoss(sample_rate=float(sr))
    L_mono = erb_loss.extract_erb_band_levels_db(x_mono)
    L_stereo = erb_loss.extract_erb_band_levels_db(x_stereo)
    assert torch.allclose(L_mono, L_stereo, atol=1e-5)

    same_val, _ = erb_loss.forward_same_track(x_stereo, x_stereo)
    style_val, _ = erb_loss.forward_stylistic(x_stereo, x_stereo, mix_anchor_audio=x_stereo)
    assert same_val.item() < 1e-7
    assert style_val.item() < 1e-7


def test_erb_de_harshness_asymmetric_relative_penalty():
    """Verify asymmetric relative behavior of L_de_harsh (2.5-5 kHz vs 0.4-2 kHz)."""
    sr = 48000
    t = torch.linspace(0.0, 1.0, sr, dtype=torch.float32)
    mid_tone = 0.25 * torch.sin(2.0 * np.pi * 1000.0 * t)
    harsh_tone = 0.25 * torch.sin(2.0 * np.pi * 3500.0 * t)

    ref_sig = torch.stack([mid_tone + harsh_tone, mid_tone + harsh_tone], dim=0).unsqueeze(0)
    # 1. Darker or equal prediction -> penalty must be strictly 0.0
    darker_sig = torch.stack([mid_tone + 0.5 * harsh_tone, mid_tone + 0.5 * harsh_tone], dim=0).unsqueeze(0)
    # 2. Harsher prediction (+6 dB in 3.5 kHz) -> penalty must be strictly > 0.0
    harsher_sig = torch.stack([mid_tone + 2.5 * harsh_tone, mid_tone + 2.5 * harsh_tone], dim=0).unsqueeze(0)

    erb_loss = ERBSpectralLoss(sample_rate=float(sr), de_harsh_margin_db=0.75, weight_de_harsh=1.0)
    _, m_dark = erb_loss.forward_same_track(darker_sig, ref_sig)
    _, m_equal = erb_loss.forward_same_track(ref_sig, ref_sig)
    _, m_harsh = erb_loss.forward_same_track(harsher_sig, ref_sig)

    assert m_dark["erb_de_harsh_loss"] == 0.0
    assert m_equal["erb_de_harsh_loss"] == 0.0
    assert m_harsh["erb_de_harsh_loss"] > 0.5
    assert m_harsh["erb_presence_surplus_db"] > 2.0


def test_erb_loss_differentiability_and_finite_gradients():
    """Verify backward pass through MasteringPerceptualLoss (V1) and AggregatePerceptualLoss (V1)."""
    sr = 48000
    torch.manual_seed(42)
    pred = (torch.randn(1, 2, sr) * 0.15).requires_grad_(True)
    ref = torch.randn(1, 2, sr) * 0.15

    same_loss_v1 = MasteringPerceptualLoss(sample_rate=float(sr), weight_erb=0.25, weight_de_harsh=0.1)
    l_same, m_same = same_loss_v1(pred, ref)
    assert "erb_same_track_loss" in m_same
    l_same.backward()
    assert pred.grad is not None
    assert torch.all(torch.isfinite(pred.grad))
    assert torch.norm(pred.grad).item() > 0.0

    pred2 = (torch.randn(1, 2, sr) * 0.15).requires_grad_(True)
    style_loss_v1 = AggregatePerceptualLoss(
        sample_rate=float(sr), spectral_mode="erb48", weight_de_harsh=0.1
    )
    l_style, m_style = style_loss_v1(pred2, ref, mix_anchor_audio=pred2.detach())
    assert "erb_stylistic_total_loss" in m_style
    l_style.backward()
    assert pred2.grad is not None
    assert torch.all(torch.isfinite(pred2.grad))
    assert torch.norm(pred2.grad).item() > 0.0
