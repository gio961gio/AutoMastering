"""Combined Perceptual Mastering Loss with Latency Alignment Compensation.

Unifies Multi-Resolution M/S STFT Loss, Differentiable LUFS/PLR Loudness Loss,
and Parameter Regularization for self-supervised mastering optimization.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from ddsp_mastering.loss.stft import MultiResolutionSTFTLoss
from ddsp_mastering.loss.loudness import DifferentiableLoudnessLoss
from ddsp_mastering.param_space import Z_NEUTRAL


class MasteringPerceptualLoss(nn.Module):
    """Unified loss module for AutoMastering model optimization.

    Handles:
      1. Fixed latency alignment compensation (159 samples at 48 kHz).
      2. Multi-Resolution Spectral Convergence and Log-Magnitude Loss (L/R and M/S).
      3. Differentiable ITU-R BS.1770-4 Integrated LUFS and Crest Factor (PLR) Loss.
      4. Parameter Regularization prior (preventing extreme parameter values).
    """

    def __init__(
        self,
        sample_rate: float = 48000.0,
        latency_samples: int = 159,
        target_lufs: float = -14.0,
        weight_stft: float = 1.0,
        weight_loudness: float = 0.5,
        weight_param_reg: float = 0.01,
        fft_sizes: Tuple[int, ...] = (512, 2048, 8192),
        hop_sizes: Tuple[int, ...] = (128, 512, 2048),
        win_lengths: Tuple[int, ...] = (512, 2048, 8192),
        use_mid_side: bool = True,
        mid_side_weight: float = 0.5,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.latency_samples = latency_samples
        self.weight_stft = weight_stft
        self.weight_loudness = weight_loudness
        self.weight_param_reg = weight_param_reg

        self.stft_loss = MultiResolutionSTFTLoss(
            fft_sizes=fft_sizes,
            hop_sizes=hop_sizes,
            win_lengths=win_lengths,
            use_mid_side=use_mid_side,
            mid_side_weight=mid_side_weight,
        )

        self.loudness_loss = DifferentiableLoudnessLoss(
            sample_rate=sample_rate,
            target_lufs=target_lufs,
        )

    def align_latency(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Align predicted output with target by compensating for chain latency.

        Because the mastering chain introduces a deterministic latency D (159 samples),
        the valid aligned region corresponds to:
          pred[:, :, D:] and target[:, :, :T - D]
        """
        D = self.latency_samples
        T = pred.shape[-1]
        if D <= 0 or T <= D:
            return pred, target

        aligned_pred = pred[..., D:]
        aligned_target = target[..., : T - D]
        return aligned_pred, aligned_target

    def forward(
        self,
        pred_audio: torch.Tensor,
        target_audio: Optional[torch.Tensor] = None,
        target_lufs: Optional[float] = None,
        normalized_params: Optional[torch.Tensor] = None,
        align_latency: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the composite perceptual loss.

        Args:
            pred_audio: Output from mastering chain (B, 2, T).
            target_audio: Reference target audio (B, 2, T), if available.
            target_lufs: Desired target LUFS (defaults to self.target_lufs).
            normalized_params: Latent controls z in [0, 1]^41 for regularization.
            align_latency: Whether to slice off chain latency before reference comparison.

        Returns:
            Tuple of (total_loss, metrics_dict).
        """
        metrics: Dict[str, float] = {}
        total_loss = torch.tensor(0.0, device=pred_audio.device, dtype=pred_audio.dtype)

        # 1. Reference-matching STFT loss (if target_audio is provided)
        if target_audio is not None:
            if align_latency:
                p_aligned, t_aligned = self.align_latency(pred_audio, target_audio)
            else:
                p_aligned, t_aligned = pred_audio, target_audio

            stft_val, stft_metrics = self.stft_loss(p_aligned, t_aligned)
            metrics.update(stft_metrics)
            total_loss = total_loss + self.weight_stft * stft_val

        # 2. Differentiable Loudness and Dynamic Range Loss
        loudness_val, loudness_metrics = self.loudness_loss(
            pred_audio, target_lufs=target_lufs
        )
        metrics.update(loudness_metrics)
        total_loss = total_loss + self.weight_loudness * loudness_val

        # 3. Parameter Regularization Loss (pulls z towards physical neutral preset Z_NEUTRAL)
        if normalized_params is not None:
            z_target = Z_NEUTRAL.to(device=normalized_params.device, dtype=normalized_params.dtype)
            param_reg = torch.mean((normalized_params - z_target) ** 2)
            metrics["param_reg_loss"] = param_reg.item()
            total_loss = total_loss + self.weight_param_reg * param_reg

        metrics["mastering_loss_total"] = total_loss.item()
        return total_loss, metrics
