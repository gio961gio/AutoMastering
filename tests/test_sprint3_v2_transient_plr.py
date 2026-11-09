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
import torchaudio.functional as taf

from ddsp_mastering.loss.combined import MasteringPerceptualLoss
from ddsp_mastering.loss.loudness import (
    DifferentiableLoudnessLoss,
    boltzmann_soft_peak_dbfs,
)
from ddsp_mastering.loss.transient import (
    TransientEnvelopeLoss,
    compute_transient_active_mask,
)
from ddsp_mastering.optimize.metrics import calculate_true_peak_dbtp
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


def test_boltzmann_soft_peak_gain_homogeneity():
    """Verify that boltzmann_soft_peak_dbfs preserves gain homogeneity down to -40 dBFS (§2.C.2, Fix A)."""
    torch.manual_seed(101)
    sample_rate = 48000.0
    x = torch.randn(1, 2, int(sample_rate * 2.0), dtype=torch.float64) * 0.5  # peak ~ -6 dBFS
    base_peak = boltzmann_soft_peak_dbfs(x, sample_rate=sample_rate)

    for atten_db in [-6.0, -12.0, -20.0, -30.0, -40.0]:
        gain = 10.0 ** (atten_db / 20.0)
        scaled_x = x * gain
        scaled_peak = boltzmann_soft_peak_dbfs(scaled_x, sample_rate=sample_rate)
        expected_peak = base_peak + atten_db
        diff = abs(scaled_peak.item() - expected_peak.item())
        assert diff < 0.05, f"At {atten_db} dB, expected ~{expected_peak.item():.3f}, got {scaled_peak.item():.3f} (diff={diff:.4f})"


def test_boltzmann_soft_peak_partial_last_block():
    """Verify that boltzmann_soft_peak_dbfs processes the last partial block without dropping samples (§2.C.2, Fix B)."""
    sample_rate = 48000.0
    # 143841 samples = 7 full blocks of 19200 (134400) + 9441 remainder samples
    T = 143841
    x_silence = torch.zeros(1, 2, T, dtype=torch.float64)
    # Put a distinct peak only in the remainder region (sample 140000)
    x_silence[:, :, 140000] = 0.8  # ~ -1.94 dBFS

    peak_with_pulse = boltzmann_soft_peak_dbfs(x_silence, sample_rate=sample_rate)
    # The peak must reflect the pulse, not digital silence
    assert peak_with_pulse.item() > -20.0, f"Pulse in remainder block was ignored: got {peak_with_pulse.item():.2f} dBFS"

    # Also test T <= N_b (short chunk < 400ms, e.g. 9600 samples = 200ms)
    x_short = torch.randn(1, 2, 9600, dtype=torch.float64) * 0.1
    peak_short = boltzmann_soft_peak_dbfs(x_short, sample_rate=sample_rate)
    assert torch.isfinite(peak_short)


def test_same_track_plr_alignment_and_block_count():
    """Verify that MasteringPerceptualLoss aligns pred and ref before reference-aware PLR loss (§2.C.2, Fix C)."""
    sample_rate = 48000.0
    loss_fn = MasteringPerceptualLoss(
        sample_rate=sample_rate,
        weight_stft=1.0,
        weight_loudness=0.8,
        plr_mode="reference_aware",
        latency_samples=159,
    )
    # Both pred and ref have length T = 48000
    pred = torch.randn(1, 2, 48000, dtype=torch.float32)
    ref = torch.randn(1, 2, 48000, dtype=torch.float32)

    total_loss, metrics = loss_fn(pred, ref, align_latency=True)
    assert torch.isfinite(total_loss)
    assert "plr_loss" in metrics

    # Also verify that compute_reference_aware_plr_loss raises ValueError on mismatched lengths
    loudness_calc = DifferentiableLoudnessLoss(sample_rate=sample_rate)
    pred_short = pred[:, :, :40000]
    with pytest.raises(ValueError, match="must be aligned and have identical length"):
        loudness_calc.compute_reference_aware_plr_loss(pred_short, ref, mode="same_track")


def test_transient_loss_preroll_context_warmup():
    """Verify that pre-roll context warms up the crossover and cumsum envelopes in TransientEnvelopeLoss (§2.C.1, Fix D)."""
    sample_rate = 48000.0
    tl = TransientEnvelopeLoss(sample_rate=sample_rate)
    n_pre = int(sample_rate * 2.5)  # 120000 samples
    T_eval = int(sample_rate * 1.0)  # 48000 samples

    torch.manual_seed(555)
    full_audio = torch.randn(1, 2, n_pre + T_eval, dtype=torch.float32)
    eval_only = full_audio[:, :, n_pre:]

    # Envelopes extracted with pre-roll context
    crest_with_pre, onset_with_pre = tl.extract_band_envelopes(full_audio, preroll_samples=n_pre)
    # Envelopes extracted without pre-roll context (replicate-padded at boundary)
    crest_no_pre, onset_no_pre = tl.extract_band_envelopes(eval_only, preroll_samples=0)

    # At the start of the evaluation region (frame 0), envelopes differ due to warmed-up filter state and left context
    diff_crest_start = torch.abs(crest_with_pre[:, :, 0] - crest_no_pre[:, :, 0]).max().item()
    assert diff_crest_start > 0.05, f"Expected pre-roll warmup difference at frame 0, got diff={diff_crest_start}"

    # Verify forward_same_track and forward_stylistic with pred_preroll_samples
    target_eval = torch.randn(1, 2, T_eval, dtype=torch.float32)
    l_same, m_same = tl.forward_same_track(full_audio, target_eval, pred_preroll_samples=n_pre, ref_preroll_samples=0)
    assert torch.isfinite(l_same)
    assert m_same["trans_same_total_loss"] >= 0.0

    l_style, m_style = tl.forward_stylistic(full_audio, target_eval, pred_preroll_samples=n_pre, ref_preroll_samples=0)
    assert torch.isfinite(l_style)
    assert m_style["trans_style_total_loss"] >= 0.0


def test_true_peak_streaming_vs_monolithic_boundary_equivalence():
    """Verify that streaming chunking with 128-sample halo in calculate_true_peak_dbtp matches monolithic calculation (§2.E, Fix F)."""
    sample_rate = 48000.0
    torch.manual_seed(42)
    # 520,000 samples crosses the 480,000 chunk boundary by 40,000 samples
    T = 520000
    x = torch.randn(2, T, dtype=torch.float64) * 0.1

    # Place a sharp inter-sample peak right at the chunk boundary (sample 480000)
    x[:, 479998:480002] = torch.tensor([[0.5, -0.9, 0.8, -0.4], [0.4, -0.85, 0.75, -0.35]], dtype=torch.float64)

    # 1. Monolithic calculation
    filter_width = 64
    x_padded = F.pad(x, (filter_width, filter_width), mode="constant", value=0.0)
    mono_oversampled = taf.resample(
        x_padded,
        orig_freq=1,
        new_freq=4,
        lowpass_filter_width=filter_width,
        rolloff=1.0,
        resampling_method="sinc_interp_kaiser",
        beta=14.769656459379492,
    )
    mono_peak_linear = max(torch.max(torch.abs(x)).item(), torch.max(torch.abs(mono_oversampled)).item())
    mono_true_peak_dbtp = 20.0 * math.log10(mono_peak_linear)

    # 2. Streaming calculate_true_peak_dbtp (which uses chunk_size=480000)
    stream_true_peak_dbtp = calculate_true_peak_dbtp(x, sample_rate=sample_rate)

    diff = abs(stream_true_peak_dbtp - mono_true_peak_dbtp)
    assert diff < 1e-4, f"Streaming true peak differs from monolithic by {diff:.6f} dBTP (stream={stream_true_peak_dbtp:.4f}, mono={mono_true_peak_dbtp:.4f})"

