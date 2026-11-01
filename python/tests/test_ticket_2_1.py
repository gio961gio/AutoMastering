"""Tests for Ticket 2.1: Multi-Resolution STFT Loss & Perceptual Loudness Loss."""

import math
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy.signal import lfilter

from ddsp_mastering.loss import (
    SingleResolutionSTFTLoss,
    MultiResolutionSTFTLoss,
    DifferentiableKWeighting,
    DifferentiableLoudnessLoss,
    MasteringPerceptualLoss,
    get_k_weighting_coefficients,
)
from ddsp_mastering.dsp.chain import DifferentiableMasteringChain


@pytest.fixture
def stereo_audio():
    """Generates synthetic stereo audio (48 kHz, 1.0s) with musical harmonic content."""
    torch.manual_seed(42)
    fs = 48000
    t = torch.linspace(0, 1.0, fs)
    # Fundamental + harmonics
    left = (
        0.3 * torch.sin(2 * math.pi * 100.0 * t)
        + 0.2 * torch.sin(2 * math.pi * 440.0 * t)
        + 0.1 * torch.sin(2 * math.pi * 2500.0 * t)
        + 0.05 * torch.randn_like(t)
    )
    right = (
        0.3 * torch.sin(2 * math.pi * 100.0 * t)
        + 0.15 * torch.sin(2 * math.pi * 440.0 * t + 0.3)
        + 0.12 * torch.sin(2 * math.pi * 2500.0 * t - 0.2)
        + 0.05 * torch.randn_like(t)
    )
    return torch.stack([left, right], dim=0).unsqueeze(0)  # (1, 2, 48000)


def test_stft_loss_identity(stereo_audio):
    """Test that STFT loss on identical signals is practically zero."""
    loss_module = MultiResolutionSTFTLoss()
    loss, metrics = loss_module(stereo_audio, stereo_audio)

    assert loss.item() < 1e-4, f"Identity loss should be ~0, got {loss.item()}"
    assert metrics["stft_sc_lr"] < 1e-4
    assert metrics["stft_mag_lr"] < 1e-4


def test_stft_loss_m_s_sensitivity(stereo_audio):
    """Test that Mid/Side loss specifically penalizes stereo changes."""
    loss_module = MultiResolutionSTFTLoss(use_mid_side=True)

    # Signal with widened side channel
    widened = stereo_audio.clone()
    mid = (widened[:, 0:1, :] + widened[:, 1:2, :]) / 1.41421356
    side = (widened[:, 0:1, :] - widened[:, 1:2, :]) / 1.41421356
    widened_audio = torch.cat(
        [(mid + 2.0 * side) / 1.41421356, (mid - 2.0 * side) / 1.41421356],
        dim=1,
    )

    loss_ms, metrics_ms = loss_module(widened_audio, stereo_audio)
    assert metrics_ms["stft_loss_ms"] > 0.05, "Mid/Side loss should detect stereo alteration"


def test_stft_loss_gradient_flow(stereo_audio):
    """Test gradient flow through MultiResolutionSTFTLoss."""
    x = stereo_audio.clone().requires_grad_(True)
    target = stereo_audio * 0.8

    loss_module = MultiResolutionSTFTLoss()
    loss, _ = loss_module(x, target)
    loss.backward()

    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))
    assert torch.norm(x.grad) > 0.0, "Gradients should be non-zero"


def test_k_weighting_coefficients():
    """Verify ITU-R BS.1770-4 coefficients match standard specification at 48 kHz."""
    b1, a1, b2, a2 = get_k_weighting_coefficients(48000.0)

    # Standard reference values
    assert abs(b1[0].item() - 1.53512485958697) < 1e-8
    assert abs(a1[1].item() - (-1.69065929318241)) < 1e-8
    assert abs(b2[0].item() - 1.0) < 1e-8
    assert abs(a2[1].item() - (-1.99004745483398)) < 1e-8


def test_k_weighting_filter_parity():
    """Verify differentiable K-weighting against Scipy lfilter reference."""
    k_filter = DifferentiableKWeighting(sample_rate=48000.0)

    np.random.seed(123)
    T = 48000
    x_np = (np.random.randn(T) * 0.1).astype(np.float32)
    x_torch = torch.from_numpy(x_np).unsqueeze(0).unsqueeze(0)  # (1, 1, T)

    # Scipy reference
    b1, a1, b2, a2 = get_k_weighting_coefficients(48000.0)
    y1 = lfilter(b1.numpy(), a1.numpy(), x_np)
    y_ref = lfilter(b2.numpy(), a2.numpy(), y1)

    # Differentiable PyTorch filter
    y_torch = k_filter(x_torch).squeeze().detach().numpy()

    # Compare excluding boundary filter warm-up
    diff = np.abs(y_torch[500:-500] - y_ref[500:-500])
    max_err = np.max(diff)
    assert max_err < 1e-3, f"Max error between PyTorch and Scipy K-filter: {max_err}"


def test_lufs_calculation_calibration():
    """Verify LUFS calculation on standard 1 kHz stereo sine wave."""
    loudness_module = DifferentiableLoudnessLoss(sample_rate=48000.0)

    # 1 kHz sine at 0 dBFS peak
    fs = 48000
    t = torch.linspace(0, 1.0, fs)
    sine_1k = torch.sin(2 * math.pi * 1000.0 * t)
    stereo_sine = torch.stack([sine_1k, sine_1k], dim=0).unsqueeze(0)

    lufs = loudness_module.calculate_lufs(stereo_sine).item()
    # At 1 kHz, K-filter gain is approximately +0.69 dB, power of sine is -3 dB per channel,
    # sum of both channels is 0 dB, minus 0.691 calibration -> LUFS should be close to -0.691 dB
    # Standard BS.1770 test tone check:
    assert -1.5 < lufs < 0.5, f"Expected LUFS near -0.69 for 0 dBFS 1kHz stereo sine, got {lufs:.2f}"


def test_loudness_loss_gradient_flow(stereo_audio):
    """Verify gradients propagate smoothly through LUFS and PLR losses."""
    x = stereo_audio.clone().requires_grad_(True)
    loudness_module = DifferentiableLoudnessLoss(target_lufs=-14.0)

    loss, metrics = loudness_module(x)
    loss.backward()

    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))
    assert torch.norm(x.grad) > 0.0, "Gradients should flow back to audio"


def test_mastering_loss_latency_alignment(stereo_audio):
    """Verify that latency alignment compensation correctly accounts for 159 samples."""
    loss_module = MasteringPerceptualLoss(latency_samples=159)

    # Simulate an audio signal delayed by 159 samples
    delayed_audio = torch.zeros_like(stereo_audio)
    delayed_audio[:, :, 159:] = stereo_audio[:, :, :-159]

    # Without alignment, loss should be higher
    loss_unaligned, _ = loss_module(
        pred_audio=delayed_audio, target_audio=stereo_audio, align_latency=False
    )

    # With alignment, the reference matching region is aligned
    loss_aligned, _ = loss_module(
        pred_audio=delayed_audio, target_audio=stereo_audio, align_latency=True
    )

    assert loss_aligned < loss_unaligned, (
        f"Latency alignment should lower STFT mismatch: {loss_aligned.item()} vs {loss_unaligned.item()}"
    )


def test_end_to_end_optimization_with_mastering_chain(stereo_audio):
    """Verify that MasteringPerceptualLoss drives optimization of DifferentiableMasteringChain."""
    torch.manual_seed(42)
    chain = DifferentiableMasteringChain(sample_rate=48000.0)

    # Target: louder, balanced master
    loss_fn = MasteringPerceptualLoss(
        sample_rate=48000.0,
        target_lufs=-12.0,
        weight_stft=1.0,
        weight_loudness=1.0,
        weight_param_reg=0.01,
    )

    # Parameter vector to optimize
    z = torch.nn.Parameter(torch.full((41,), 0.5, dtype=torch.float32))
    optimizer = torch.optim.Adam([z], lr=0.05)

    initial_loss = None
    final_loss = None

    # Run 10 optimization steps on 0.25s audio chunk for speed
    audio_chunk = stereo_audio[:, :, :12000]

    for step in range(10):
        optimizer.zero_grad()
        # Constrain z to [0, 1] via sigmoid
        z_bounded = torch.sigmoid(z)
        pred = chain(audio_chunk, z_bounded)

        loss, metrics = loss_fn(
            pred_audio=pred,
            target_audio=audio_chunk,
            normalized_params=z_bounded,
            align_latency=True,
        )

        if step == 0:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    assert final_loss < initial_loss, (
        f"Mastering loss should decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
    )
