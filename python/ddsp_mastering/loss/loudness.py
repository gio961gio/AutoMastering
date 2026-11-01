"""Differentiable ITU-R BS.1770-4 / EBU R128 K-Weighting & Loudness Loss.

Provides differentiable K-weighting filtering, Integrated and Short-Term LUFS
calculation, Target Loudness Loss, and Crest Factor / Dynamic Range (PLR) penalties.
"""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_k_weighting_coefficients(
    sample_rate: float = 48000.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute exact ITU-R BS.1770-4 K-weighting biquad filter coefficients.

    Stage 1: Pre-filter (high-shelf acoustic head model, f0 ~ 1.5 kHz, +4 dB)
    Stage 2: RLB filter (2nd-order high-pass, f0 ~ 38 Hz)

    Returns:
        (b_stage1, a_stage1, b_stage2, a_stage2) as 1D tensors of shape (3,).
    """
    if abs(sample_rate - 48000.0) < 1.0:
        # Standard ITU-R BS.1770-4 calibrated values at 48 kHz
        b1 = torch.tensor([1.53512485958697, -2.69169618940638, 1.19839281085285], dtype=torch.float64)
        a1 = torch.tensor([1.0, -1.69065929318241, 0.73248077421585], dtype=torch.float64)
        b2 = torch.tensor([1.0, -2.0, 1.0], dtype=torch.float64)
        a2 = torch.tensor([1.0, -1.99004745483398, 0.99007225035662], dtype=torch.float64)
        return b1, a1, b2, a2

    # Bilinear transform design for general sample rate
    # Stage 1: High shelf filter (f0 = 1681.9744509555319 Hz, gain = 3.99984385397 dB, Q = 0.7071)
    f0 = 1681.9744509555319
    gain_db = 3.999843853973523
    v0 = 10.0 ** (gain_db / 20.0)
    k = math.tan(math.pi * f0 / sample_rate)
    denom1 = 1.0 + math.sqrt(2.0) * k + k * k
    b1_0 = (v0 + math.sqrt(2.0 * v0) * k + k * k) / denom1
    b1_1 = (2.0 * (k * k - v0)) / denom1
    b1_2 = (v0 - math.sqrt(2.0 * v0) * k + k * k) / denom1
    a1_1 = (2.0 * (k * k - 1.0)) / denom1
    a1_2 = (1.0 - math.sqrt(2.0) * k + k * k) / denom1

    b1 = torch.tensor([b1_0, b1_1, b1_2], dtype=torch.float64)
    a1 = torch.tensor([1.0, a1_1, a1_2], dtype=torch.float64)

    # Stage 2: High-pass RLB weighting filter (f0 = 38.13547087602444 Hz, Q = 0.5003270373253937)
    f2 = 38.13547087602444
    q2 = 0.5003270373253937
    k2 = math.tan(math.pi * f2 / sample_rate)
    denom2 = 1.0 + k2 / q2 + k2 * k2
    b2_0 = 1.0 / denom2
    b2_1 = -2.0 / denom2
    b2_2 = 1.0 / denom2
    a2_1 = (2.0 * (k2 * k2 - 1.0)) / denom2
    a2_2 = (1.0 - k2 / q2 + k2 * k2) / denom2

    b2 = torch.tensor([b2_0, b2_1, b2_2], dtype=torch.float64)
    a2 = torch.tensor([1.0, a2_1, a2_2], dtype=torch.float64)

    return b1, a1, b2, a2


class DifferentiableKWeighting(nn.Module):
    """Frequency-domain differentiable K-weighting filter for audio tensors.

    Evaluates analytical complex frequency response H(e^{j omega}) of the cascaded
    ITU-R BS.1770-4 biquad stages, performing zero-phase or linear frequency-domain
    filtering via torch.fft.
    """

    def __init__(self, sample_rate: float = 48000.0):
        super().__init__()
        self.sample_rate = sample_rate
        b1, a1, b2, a2 = get_k_weighting_coefficients(sample_rate)
        self.register_buffer("b1", b1)
        self.register_buffer("a1", a1)
        self.register_buffer("b2", b2)
        self.register_buffer("a2", a2)

    def compute_frequency_response(self, n_fft: int, device: torch.device) -> torch.Tensor:
        """Evaluate cascaded analytical complex frequency response H(e^{j omega})."""
        omega = torch.linspace(0, math.pi, n_fft // 2 + 1, device=device, dtype=torch.float64)
        z1 = torch.exp(-1j * omega)
        z2 = torch.exp(-2j * omega)

        b1 = self.b1.to(device=device)
        a1 = self.a1.to(device=device)
        b2 = self.b2.to(device=device)
        a2 = self.a2.to(device=device)

        H1 = (b1[0] + b1[1] * z1 + b1[2] * z2) / (a1[0] + a1[1] * z1 + a1[2] * z2)
        H2 = (b2[0] + b2[1] * z1 + b2[2] * z2) / (a2[0] + a2[1] * z1 + a2[2] * z2)
        H = H1 * H2
        return H

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply K-weighting filter to audio tensor.

        Args:
            x: Audio tensor of shape (B, C, T) or (B, T).

        Returns:
            Filtered audio tensor of same shape.
        """
        orig_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(1)

        B, C, T = x.shape
        device = x.device
        dtype = x.dtype

        # Compute analytical H(omega)
        H = self.compute_frequency_response(T, device=device).to(dtype=torch.complex64)

        # Real FFT along time dimension
        X = torch.fft.rfft(x, n=T, dim=-1)
        Y = X * H
        y = torch.fft.irfft(Y, n=T, dim=-1)

        return y.to(dtype=dtype).reshape(orig_shape)


class DifferentiableLoudnessLoss(nn.Module):
    """Computes differentiable ITU-R BS.1770-4 LUFS and Perceptual Loudness Loss."""

    def __init__(
        self,
        sample_rate: float = 48000.0,
        target_lufs: float = -14.0,
        lufs_loss_weight: float = 1.0,
        plr_loss_weight: float = 0.2,
        target_plr: Optional[float] = 10.0,
        plr_min: float = 8.0,
        plr_max: float = 14.0,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.target_lufs = target_lufs
        self.lufs_loss_weight = lufs_loss_weight
        self.plr_loss_weight = plr_loss_weight
        self.target_plr = target_plr
        self.plr_min = plr_min
        self.plr_max = plr_max
        self.eps = eps

        self.k_filter = DifferentiableKWeighting(sample_rate=sample_rate)

    def calculate_lufs(self, x: torch.Tensor) -> torch.Tensor:
        """Compute differentiable Integrated LUFS for stereo audio tensor.

        Args:
            x: Stereo audio tensor of shape (B, 2, T).

        Returns:
            Integrated LUFS tensor of shape (B,).
        """
        assert x.dim() == 3 and x.shape[1] == 2, "Expected stereo input (B, 2, T)"

        # Apply K-weighting
        y_k = self.k_filter(x)  # (B, 2, T)

        # Mean square energy per channel
        # BS.1770 weights: Left = 1.0, Right = 1.0 (Surround channels have 1.41, not present here)
        z_L = torch.mean(y_k[:, 0, :] ** 2, dim=-1)
        z_R = torch.mean(y_k[:, 1, :] ** 2, dim=-1)
        z_total = z_L + z_R

        # ITU-R BS.1770 loudness formula: -0.691 + 10 * log10(z_total)
        lufs = -0.691 + 10.0 * torch.log10(z_total + self.eps)
        return lufs

    def calculate_peak_dbfs(self, x: torch.Tensor) -> torch.Tensor:
        """Compute smooth differentiable peak amplitude in dBFS."""
        abs_x = torch.abs(x)
        peak_val = torch.max(abs_x.reshape(x.shape[0], -1), dim=-1)[0]
        peak_dbfs = 20.0 * torch.log10(peak_val + 1e-6)
        return peak_dbfs

    def calculate_plr(self, x: torch.Tensor) -> torch.Tensor:
        """Compute differentiable Peak-to-Loudness Ratio (PLR / Crest Factor) in dB."""
        peak_dbfs = self.calculate_peak_dbfs(x)
        lufs = self.calculate_lufs(x)
        return peak_dbfs - lufs

    def forward(
        self,
        pred: torch.Tensor,
        target_lufs: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loudness and dynamic range loss.

        Args:
            pred: Processed audio tensor of shape (B, 2, T).
            target_lufs: Optional override of target LUFS.

        Returns:
            Tuple of (total_loudness_loss, metrics_dict).
        """
        tgt = target_lufs if target_lufs is not None else self.target_lufs

        # 1. Integrated LUFS calculation
        pred_lufs = self.calculate_lufs(pred)  # (B,)

        # Smooth L1 / Huber loss around target LUFS
        lufs_diff = pred_lufs - tgt
        lufs_loss = F.smooth_l1_loss(pred_lufs, torch.full_like(pred_lufs, tgt), beta=1.0)

        # 2. Dynamic Range / Crest Factor / PLR (Peak to Loudness Ratio)
        peak_dbfs = self.calculate_peak_dbfs(pred)  # (B,)
        plr = peak_dbfs - pred_lufs  # (B,)

        # Penalize over-compression (PLR < plr_min) and under-compression (PLR > plr_max)
        plr_under_penalty = F.relu(self.plr_min - plr) ** 2
        plr_over_penalty = F.relu(plr - self.plr_max) ** 2
        plr_loss = torch.mean(plr_under_penalty + plr_over_penalty)

        total_loss = (
            self.lufs_loss_weight * lufs_loss
            + self.plr_loss_weight * plr_loss
        )

        metrics = {
            "lufs_pred": pred_lufs.mean().item(),
            "lufs_target": float(tgt),
            "lufs_error_db": lufs_diff.abs().mean().item(),
            "lufs_loss": lufs_loss.item(),
            "peak_dbfs": peak_dbfs.mean().item(),
            "plr_db": plr.mean().item(),
            "plr_loss": plr_loss.item(),
            "loudness_loss_total": total_loss.item(),
        }

        return total_loss, metrics
