"""Differentiable ITU-R BS.1770-5 / EBU R128 K-Weighting & Loudness Loss.

Provides differentiable causal K-weighting filtering, standard-compliant
ITU-R BS.1770-5 / EBU Tech 3341 Integrated Loudness calculation with dual gating
(-70 LKFS absolute, -10 LU relative), differentiable loudness proxy loss, and
Crest Factor / Dynamic Range (PLR) penalties.
"""

import math
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as taf


def get_k_weighting_coefficients(
    sample_rate: float = 48000.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute exact ITU-R BS.1770-5 K-weighting biquad filter coefficients.

    Stage 1: Pre-filter (high-shelf acoustic head model, f0 ~ 1.5 kHz, +4 dB)
    Stage 2: RLB filter (2nd-order high-pass, f0 ~ 38 Hz)

    Returns:
        (b_stage1, a_stage1, b_stage2, a_stage2) as 1D tensors of shape (3,).
    """
    if abs(sample_rate - 48000.0) < 1.0:
        # Standard ITU-R BS.1770-5 Table 1 calibrated values at 48 kHz
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
    """Causal differentiable K-weighting filter for audio tensors.

    Evaluates exact cascaded ITU-R BS.1770-5 biquad filtering (Stage 1 high shelf,
    Stage 2 RLB high pass). Uses causal biquad filtering via torchaudio.functional.lfilter
    in float64 precision to ensure strict time-domain causality, zero circular wrap-around,
    and robust gradient propagation across odd and even sequence lengths.
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
        """Evaluate cascaded analytical complex frequency response H(e^{j omega}) at exact rfft bins."""
        omega = torch.fft.rfftfreq(n_fft, d=1.0, device=device) * (2.0 * math.pi)
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
        """Apply causal K-weighting filter to audio tensor.

        Args:
            x: Audio tensor of shape (B, C, T), (C, T), or (B, T).

        Returns:
            Filtered audio tensor of same shape and dtype.
        """
        orig_shape = x.shape
        orig_dtype = x.dtype
        device = x.device

        # Reshape to (..., T) for taf.lfilter
        if x.dim() == 1:
            x_in = x.unsqueeze(0)
        else:
            x_in = x

        # Ensure float64 for biquad numerical precision
        x64 = x_in.to(device=device, dtype=torch.float64)
        b1 = self.b1.to(device=device)
        a1 = self.a1.to(device=device)
        b2 = self.b2.to(device=device)
        a2 = self.a2.to(device=device)

        # Stage 1: Pre-filter high shelf
        y1 = taf.lfilter(x64, a1, b1, clamp=False)
        # Stage 2: RLB high pass
        y2 = taf.lfilter(y1, a2, b2, clamp=False)

        y = y2.to(dtype=orig_dtype)
        return y.reshape(orig_shape)


def calculate_integrated_loudness_bs1770(
    audio: Union[torch.Tensor, np.ndarray],
    sample_rate: float = 48000.0,
    block_size_sec: float = 0.400,
    overlap: float = 0.75,
    gamma_a: float = -70.0,
    relative_threshold_lu: float = 10.0,
) -> float:
    """Compute integrated gated loudness conforming to ITU-R BS.1770-5 and EBU Tech 3341.

    Features:
      - 400 ms sliding rectangular integration blocks with 75% overlap (100 ms step).
      - Absolute gate at gamma_a = -70.0 LKFS.
      - Relative gate at gamma_r = (preliminary loudness) - 10.0 LU.
      - Mean-square energy averaging of surviving gated blocks.
      - Exact ITU-R BS.1770-5 Table 1 coefficients at 48 kHz.

    Args:
        audio: Audio signal as torch.Tensor or numpy.ndarray.
               Expected shapes: (2, T), (1, 2, T), (T, 2) or (T,) mono.
        sample_rate: Sampling frequency in Hz (default: 48000.0).
        block_size_sec: Gating block size in seconds (default: 0.400 s / 400 ms).
        overlap: Overlap fraction between successive blocks (default: 0.75 / 75%).
        gamma_a: Absolute loudness gating threshold in LKFS (default: -70.0 LKFS).
        relative_threshold_lu: Relative threshold below ungated level in LU (default: 10.0 LU).

    Returns:
        Integrated loudness in LUFS / LKFS as a float.
        Returns -float('inf') if all blocks fall below the absolute or relative threshold (e.g. silence).

    Raises:
        ValueError: If audio duration is shorter than a single gating block (block_size_sec).
    """
    if isinstance(audio, np.ndarray):
        x = torch.from_numpy(audio)
    elif isinstance(audio, torch.Tensor):
        x = audio.detach().cpu()
    else:
        raise TypeError(f"audio must be torch.Tensor or np.ndarray, got {type(audio)}")

    # Standardize shape to (C, T)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    elif x.dim() == 2:
        if x.shape[0] > x.shape[1] and x.shape[1] in (1, 2):
            x = x.transpose(0, 1)
    elif x.dim() == 3:
        if x.shape[0] == 1:
            x = x.squeeze(0)
        else:
            raise ValueError(f"Batch dimension > 1 not supported in calculate_integrated_loudness_bs1770: {x.shape}")
    else:
        raise ValueError(f"Unsupported audio tensor shape: {x.shape}")

    num_channels, num_samples = x.shape
    min_block_samples = int(round(block_size_sec * sample_rate))
    if num_samples < min_block_samples:
        duration_sec = num_samples / sample_rate
        raise ValueError(
            f"Audio duration ({duration_sec:.3f}s / {num_samples} samples) is shorter than "
            f"minimum gating block ({block_size_sec:.3f}s / {min_block_samples} samples) "
            "per ITU-R BS.1770-5."
        )

    if num_channels == 1:
        G = [1.0]
    elif num_channels == 2:
        G = [1.0, 1.0]
    else:
        raise ValueError(f"Only mono (1) and stereo (2) channels are currently supported, got {num_channels}")

    b1, a1, b2, a2 = get_k_weighting_coefficients(sample_rate)
    x64 = x.to(torch.float64)
    y1 = taf.lfilter(x64, a1, b1, clamp=False)
    y_k = taf.lfilter(y1, a2, b2, clamp=False)

    hop_samples = int(round(block_size_sec * (1.0 - overlap) * sample_rate))
    num_blocks = (num_samples - min_block_samples) // hop_samples + 1

    if num_blocks <= 0:
        raise ValueError("No complete gating blocks available.")

    # Vectorized mean-square energy per channel and block via unfold
    y_sq = y_k ** 2
    blocks = y_sq.unfold(-1, min_block_samples, hop_samples) # (C, num_blocks, min_block_samples)
    z = blocks.mean(dim=-1) # (C, num_blocks)

    G_t = torch.tensor(G, dtype=torch.float64).unsqueeze(1) # (C, 1)
    z_weighted = torch.sum(G_t * z, dim=0) # (num_blocks,)

    # Block loudness l_j
    l_j = torch.full_like(z_weighted, -float('inf'))
    pos_mask = z_weighted > 0.0
    l_j[pos_mask] = -0.691 + 10.0 * torch.log10(z_weighted[pos_mask])

    # 1. Absolute threshold: l_j >= gamma_a (BS.1770-5 specifies >= -70 LKFS)
    abs_indices = torch.nonzero(l_j >= gamma_a, as_tuple=True)[0]
    if len(abs_indices) == 0:
        return -float('inf')

    # 2. Relative threshold calculation
    z_abs_avg = torch.mean(z_weighted[abs_indices])
    gamma_r = -0.691 + 10.0 * torch.log10(z_abs_avg) - relative_threshold_lu

    # 3. Relative threshold gating: l_j > gamma_r AND l_j >= gamma_a
    rel_indices = torch.nonzero((l_j > gamma_r) & (l_j >= gamma_a), as_tuple=True)[0]
    if len(rel_indices) == 0:
        return -float('inf')

    # Final integrated loudness from average mean-square of blocks surviving both gates
    z_final_avg = torch.mean(z_weighted[rel_indices])
    if z_final_avg <= 0.0:
        return -float('inf')

    integrated_lufs = -0.691 + 10.0 * torch.log10(z_final_avg).item()
    return float(integrated_lufs)


class DifferentiableLoudnessLoss(nn.Module):
    """Computes differentiable K-weighted Loudness proxy and Perceptual Loudness Loss."""

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

    def calculate_ungated_k_weighted_loudness(self, x: torch.Tensor) -> torch.Tensor:
        """Compute differentiable ungated K-weighted loudness proxy for stereo audio tensor.

        NOTE: This is a smooth, differentiable proxy suitable for autograd optimization.
        For standard-compliant integrated loudness (ITU-R BS.1770-5 with absolute/relative gating),
        use calculate_integrated_loudness_bs1770().

        Args:
            x: Stereo audio tensor of shape (B, 2, T).

        Returns:
            Ungated K-weighted loudness tensor of shape (B,).
        """
        assert x.dim() == 3 and x.shape[1] == 2, "Expected stereo input (B, 2, T)"

        # Apply causal K-weighting
        y_k = self.k_filter(x)  # (B, 2, T)

        # Mean square energy per channel
        # BS.1770 weights: Left = 1.0, Right = 1.0
        z_L = torch.mean(y_k[:, 0, :] ** 2, dim=-1)
        z_R = torch.mean(y_k[:, 1, :] ** 2, dim=-1)
        z_total = z_L + z_R

        # ITU-R BS.1770 loudness formula: -0.691 + 10 * log10(z_total)
        loudness = -0.691 + 10.0 * torch.log10(z_total + self.eps)
        return loudness

    def calculate_lufs(self, x: torch.Tensor) -> torch.Tensor:
        """Backward-compatible alias for calculate_ungated_k_weighted_loudness."""
        return self.calculate_ungated_k_weighted_loudness(x)

    def calculate_peak_dbfs(self, x: torch.Tensor) -> torch.Tensor:
        """Compute smooth differentiable peak amplitude in dBFS."""
        abs_x = torch.abs(x)
        peak_val = torch.max(abs_x.reshape(x.shape[0], -1), dim=-1)[0]
        peak_dbfs = 20.0 * torch.log10(peak_val + 1e-6)
        return peak_dbfs

    def calculate_plr(self, x: torch.Tensor) -> torch.Tensor:
        """Compute differentiable Peak-to-Loudness Ratio (PLR / Crest Factor) in dB."""
        peak_dbfs = self.calculate_peak_dbfs(x)
        lufs = self.calculate_ungated_k_weighted_loudness(x)
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

        # 1. Ungated K-weighted loudness proxy calculation
        pred_lufs = self.calculate_ungated_k_weighted_loudness(pred)  # (B,)

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
            "loudness_proxy_pred": pred_lufs.mean().item(),
            "loudness_proxy_target": float(tgt),
            "loudness_proxy_error_db": lufs_diff.abs().mean().item(),
            "lufs_loss": lufs_loss.item(),
            "peak_dbfs": peak_dbfs.mean().item(),
            "plr_db": plr.mean().item(),
            "plr_loss": plr_loss.item(),
            "loudness_loss_total": total_loss.item(),
            # Backwards compatibility keys
            "lufs_pred": pred_lufs.mean().item(),
            "lufs_target": float(tgt),
            "lufs_error_db": lufs_diff.abs().mean().item(),
        }

        return total_loss, metrics
