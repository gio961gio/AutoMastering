"""Tests for Sprint 3 Step V2: Multi-band Transient/Envelope Loss and Reference-Aware PLR Loss.

Covers:
  1. Boltzmann-weighted soft-peak mean mono vs stereo identity (delta < 1e-12).
  2. Boltzmann-weighted soft-peak duration invariance (zero duration bias).
  3. Boltzmann-weighted soft-peak differentiability and analytical gradient flow.
  4. Transient/envelope loss identity on identical signals (same_track and stylistic).
  5. Transient/envelope loss differentiability and gradient norm.
  6. Transient/envelope loss deterministic silence safety (finite 0.0, no NaN/Inf).
  7. Reference-aware PLR loss in same_track (Huber_1.0) and stylistic (dead-zone + floor).
  8. Backwards compatibility of default optimizer settings.
"""

import math
import pytest
import torch
import torch.nn.functional as F

from ddsp_mastering.loss.loudness import (
    DifferentiableLoudnessLoss,
    boltzmann_soft_peak_dbfs,
)
from ddsp_mastering.loss.transient import (
    TransientEnvelopeLoss,
    compute_transient_active_mask,
)
from ddsp_mastering.optimize.optimizer import AutoMasteringOptimizer


def test_boltzmann_soft_peak_mono_stereo_identity():
    """Verify that boltzmann_soft_peak_dbfs yields identical values for mono and dual-mono signals (§2.C.2)."""
    torch.manual_seed(42)
    sample_rate = 48000.0
    num_samples = int(sample_rate * 2.0)

    # Synthetic signal with pulses and dynamics
    t = torch.linspace(0, 2.0, num_samples, dtype=torch.float32)
    carrier = 0.5 * torch.sin(2.0 * math.pi * 440.0 * t)
    pulse = torch.zeros_like(t)
    pulse[1000] = 0.95
    pulse[20000] = 0.80
    mono_audio = (carrier + pulse).unsqueeze(0).unsqueeze(0)  # (1, 1, T)
    stereo_audio = mono_audio.repeat(1, 2, 1)  # (1, 2, T) identical channels

    peak_mono = boltzmann_soft_peak_dbfs(mono_audio, sample_rate=sample_rate)
    peak_stereo = boltzmann_soft_peak_dbfs(stereo_audio, sample_rate=sample_rate)

    diff = abs(peak_mono.item() - peak_stereo.item())
    assert diff < 1e-12, f"Expected exact mono-stereo identity, got diff = {diff}"


def test_boltzmann_soft_peak_duration_invariance():
    """Verify that repeating a periodic block in time produces zero duration bias (§2.C.2)."""
    torch.manual_seed(123)
    sample_rate = 48000.0
    block_samples = 480  # 10 ms at 48 kHz
    single_block = torch.randn(1, 2, block_samples, dtype=torch.float32)
    repeated_blocks = single_block.repeat(1, 1, 10)  # 100 ms

    peak_single = boltzmann_soft_peak_dbfs(single_block, sample_rate=sample_rate)
    peak_repeated = boltzmann_soft_peak_dbfs(repeated_blocks, sample_rate=sample_rate)

    diff = abs(peak_single.item() - peak_repeated.item())
    assert diff < 1e-12, f"Expected duration invariance, got diff = {diff}"


def test_boltzmann_soft_peak_differentiability():
    """Verify smooth gradient propagation through Boltzmann-weighted soft-peak mean."""
    torch.manual_seed(456)
    x = torch.randn(2, 2, 4800, dtype=torch.float64, requires_grad=True)
    peak_soft = boltzmann_soft_peak_dbfs(x, sample_rate=48000.0)

    loss = torch.sum(peak_soft)
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.norm().item() > 0.0


def test_transient_envelope_identity_same_track():
    """Verify that same_track transient loss evaluates to exactly 0.0 on identical audio."""
    torch.manual_seed(789)
    tl = TransientEnvelopeLoss(sample_rate=48000.0)
    audio = torch.randn(1, 2, 48000, dtype=torch.float32)

    loss, metrics = tl.forward_same_track(audio, audio)
    assert abs(loss.item()) < 1e-6
    assert abs(metrics["trans_same_crest_loss"]) < 1e-6
    assert abs(metrics["trans_same_onset_loss"]) < 1e-6


def test_transient_envelope_identity_stylistic():
    """Verify that stylistic multi-quantile transient loss evaluates to exactly 0.0 on identical audio."""
    torch.manual_seed(999)
    tl = TransientEnvelopeLoss(sample_rate=48000.0)
    audio = torch.randn(1, 2, 48000, dtype=torch.float32)

    loss, metrics = tl.forward_stylistic(audio, audio)
    assert abs(loss.item()) < 1e-6
    assert abs(metrics["trans_style_crest_loss"]) < 1e-6
    assert abs(metrics["trans_style_onset_loss"]) < 1e-6


def test_transient_envelope_gradient_flow():
    """Verify finite non-zero gradients through multi-band crossover, envelopes, and quantiles."""
    torch.manual_seed(321)
    tl = TransientEnvelopeLoss(sample_rate=48000.0)
    pred = torch.randn(1, 2, 24000, dtype=torch.float32, requires_grad=True)
    target = torch.randn(1, 2, 24000, dtype=torch.float32)

    # 1. same_track gradient
    loss_same, _ = tl.forward_same_track(pred, target)
    loss_same.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.norm().item() > 0.0

    # 2. stylistic gradient
    pred.grad.zero_()
    loss_style, _ = tl.forward_stylistic(pred, target)
    loss_style.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.norm().item() > 0.0


def test_transient_envelope_silence_safety():
    """Verify that silent chunks evaluate to finite 0.0 loss without NaN or zero division (§2.A.2)."""
    tl = TransientEnvelopeLoss(sample_rate=48000.0)
    pred = torch.randn(1, 2, 24000, dtype=torch.float32, requires_grad=True)
    silence = torch.zeros(1, 2, 24000, dtype=torch.float32)

    loss_same, _ = tl.forward_same_track(pred, silence)
    assert loss_same.item() == 0.0
    loss_same.backward()
    assert torch.isfinite(pred.grad).all()

    pred.grad.zero_()
    loss_style, _ = tl.forward_stylistic(pred, silence)
    assert loss_style.item() == 0.0
    loss_style.backward()
    assert torch.isfinite(pred.grad).all()


def test_reference_aware_plr_same_track():
    """Verify reference-aware PLR loss in same_track mode."""
    loudness = DifferentiableLoudnessLoss(sample_rate=48000.0, plr_mode="reference_aware_same")
    audio = torch.randn(1, 2, 48000, dtype=torch.float32)

    # Identity check
    loss_id, metrics_id = loudness.compute_reference_aware_plr_loss(audio, audio, mode="same_track")
    assert abs(loss_id.item()) < 1e-6
    assert abs(metrics_id["plr_proxy_diff_db"]) < 1e-6

    # Forward pass integration
    tot_loss, m = loudness(audio, ref_audio=audio)
    assert m["plr_loss"] == 0.0


def test_reference_aware_plr_stylistic_deadzone_and_floor():
    """Verify dead-zone and dynamic floor penalty in stylistic PLR mode (§2.C.2)."""
    loudness = DifferentiableLoudnessLoss(sample_rate=48000.0, plr_mode="reference_aware_style")
    torch.manual_seed(777)
    audio = torch.randn(1, 2, 48000, dtype=torch.float32)

    # Identical reference -> 0.0 PLR loss (well within +/- 1.0 dB dead-zone)
    loss_id, m_id = loudness.compute_reference_aware_plr_loss(audio, audio, mode="stylistic")
    assert abs(loss_id.item()) < 1e-6
    assert abs(m_id["plr_deadzone_loss"]) < 1e-6
    assert abs(m_id["plr_floor_loss"]) < 1e-6

    # When pred is heavily compressed (scaled down high-amplitude peaks), floor penalty triggers
    compressed_pred = torch.clamp(audio, -0.05, 0.05)
    loss_comp, m_comp = loudness.compute_reference_aware_plr_loss(compressed_pred, audio, mode="stylistic")
    assert loss_comp.item() > 0.0


def test_optimizer_v2_settings_and_backward_compatibility():
    """Verify that optimizer initializes correctly with V2 and default V0 settings."""
    # V0 default
    opt_v0 = AutoMasteringOptimizer()
    assert opt_v0.plr_mode == "box"
    assert opt_v0.weight_trans == 0.0

    # V2 configuration
    opt_v2 = AutoMasteringOptimizer(
        plr_mode="reference_aware",
        weight_trans=0.2,
        weight_erb=0.25,
        weight_de_harsh=0.15,
        stylistic_spectral_mode="erb48",
    )
    assert opt_v2.plr_mode == "reference_aware"
    assert opt_v2.weight_trans == 0.2
    assert opt_v2.loss_fn.weight_trans == 0.2
    assert opt_v2.stylistic_loss_fn.weight_trans == 0.2
