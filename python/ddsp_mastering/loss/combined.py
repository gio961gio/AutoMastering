"""Combined Perceptual Mastering Loss with Latency Alignment Compensation.

Unifies Multi-Resolution M/S STFT Loss, Differentiable LUFS/PLR Loudness Loss,
and Parameter Regularization for self-supervised mastering optimization.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from ddsp_mastering.loss.stft import MultiResolutionSTFTLoss
from ddsp_mastering.loss.loudness import DifferentiableLoudnessLoss
from ddsp_mastering.loss.erb import ERBSpectralLoss
from ddsp_mastering.loss.transient import TransientEnvelopeLoss
from ddsp_mastering.param_space import Z_NEUTRAL


class MasteringPerceptualLoss(nn.Module):
    """Unified loss module for AutoMastering model optimization.

    Handles:
      1. Fixed latency alignment compensation (159 samples at 48 kHz).
      2. Multi-Resolution Spectral Convergence and Log-Magnitude Loss (L/R and M/S).
      3. Optional ERB-48 perceptual spectral loss alongside MR-STFT (Sprint 3 V1)
         and optional asymmetric relative de-harshness penalty (Sprint 3 V1b).
      4. Multi-band Transient/Envelope Loss (Sprint 3 V2).
      5. Differentiable ITU-R BS.1770-4 Integrated LUFS and Crest Factor (PLR) Loss
         (box mode for V0/V0p/V1, reference-aware Boltzmann soft-peak for V2).
      6. Parameter Regularization prior (preventing extreme parameter values).
    """

    def __init__(
        self,
        sample_rate: float = 48000.0,
        latency_samples: int = 159,
        target_lufs: float = -14.0,
        weight_stft: float = 1.0,
        weight_loudness: float = 0.5,
        weight_param_reg: float = 0.01,
        weight_erb: float = 0.0,
        weight_de_harsh: float = 0.0,
        weight_trans: float = 0.0,
        beta_atk: float = 0.5,
        plr_mode: str = "box",
        plr_d_style: float = 1.0,
        plr_gamma_floor: float = 1.0,
        soft_peak_p: int = 8,
        soft_peak_beta: float = 2.0,
        soft_peak_block_duration_sec: float = 0.010,
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
        self.weight_erb = weight_erb
        self.weight_de_harsh = weight_de_harsh
        self.weight_trans = weight_trans
        self.beta_atk = beta_atk
        self.plr_mode = plr_mode
        self.plr_d_style = plr_d_style
        self.plr_gamma_floor = plr_gamma_floor
        self.soft_peak_p = soft_peak_p
        self.soft_peak_beta = soft_peak_beta
        self.soft_peak_block_duration_sec = soft_peak_block_duration_sec

        self.stft_loss = MultiResolutionSTFTLoss(
            fft_sizes=fft_sizes,
            hop_sizes=hop_sizes,
            win_lengths=win_lengths,
            use_mid_side=use_mid_side,
            mid_side_weight=mid_side_weight,
        )

        eff_dh_ratio = (weight_de_harsh / weight_erb) if weight_erb > 0.0 else weight_de_harsh
        self.erb_loss = ERBSpectralLoss(
            sample_rate=sample_rate,
            weight_de_harsh=eff_dh_ratio,
        )

        self.transient_loss = TransientEnvelopeLoss(
            sample_rate=sample_rate,
            beta_atk=beta_atk,
        )

        self.loudness_loss = DifferentiableLoudnessLoss(
            sample_rate=sample_rate,
            target_lufs=target_lufs,
            plr_mode=plr_mode,
            soft_peak_p=soft_peak_p,
            soft_peak_beta=soft_peak_beta,
            soft_peak_block_duration_sec=soft_peak_block_duration_sec,
            plr_d_style=plr_d_style,
            plr_gamma_floor=plr_gamma_floor,
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
        pred_ext_audio: Optional[torch.Tensor] = None,
        pred_preroll_samples: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the composite perceptual loss.

        Args:
            pred_audio: Output from mastering chain (B, 2, T).
            target_audio: Reference target audio (B, 2, T), if available.
            target_lufs: Desired target LUFS (defaults to self.target_lufs).
            normalized_params: Latent controls z in [0, 1]^41 for regularization.
            align_latency: Whether to slice off chain latency before reference comparison.
            pred_ext_audio: Optional full audio tensor with pre-roll (B, 2, N_pre + T) for transient loss.
            pred_preroll_samples: Number of pre-roll samples in `pred_ext_audio`.

        Returns:
            Tuple of (total_loss, metrics_dict).
        """
        metrics: Dict[str, float] = {}
        total_loss = torch.tensor(0.0, device=pred_audio.device, dtype=pred_audio.dtype)

        # 1. Reference-matching STFT loss (if target_audio is provided)
        p_aligned = pred_audio
        t_aligned = target_audio
        if target_audio is not None:
            if align_latency:
                p_aligned, t_aligned = self.align_latency(pred_audio, target_audio)
            else:
                p_aligned, t_aligned = pred_audio, target_audio

            stft_val, stft_metrics = self.stft_loss(p_aligned, t_aligned)
            metrics.update(stft_metrics)
            total_loss = total_loss + self.weight_stft * stft_val

            if self.weight_erb > 0.0 or self.weight_de_harsh > 0.0:
                erb_val, erb_metrics = self.erb_loss.forward_same_track(p_aligned, t_aligned)
                metrics.update(erb_metrics)
                scale = self.weight_erb if self.weight_erb > 0.0 else 1.0
                total_loss = total_loss + scale * erb_val

            if self.weight_trans > 0.0:
                if pred_ext_audio is not None and pred_preroll_samples > 0:
                    eff_pre = pred_preroll_samples + (self.latency_samples if align_latency else 0)
                    trans_val, trans_metrics = self.transient_loss.forward_same_track(
                        pred_ext_audio, t_aligned, pred_preroll_samples=eff_pre, ref_preroll_samples=0
                    )
                else:
                    trans_val, trans_metrics = self.transient_loss.forward_same_track(
                        p_aligned, t_aligned, pred_preroll_samples=0, ref_preroll_samples=0
                    )
                metrics.update(trans_metrics)
                total_loss = total_loss + self.weight_trans * trans_val

        # 2. Differentiable Loudness and Dynamic Range Loss
        pred_for_loudness = p_aligned if target_audio is not None else pred_audio
        ref_for_plr = t_aligned if (target_audio is not None and self.plr_mode != "box") else None
        loudness_val, loudness_metrics = self.loudness_loss(
            pred_for_loudness,
            target_lufs=target_lufs,
            ref_audio=ref_for_plr,
            plr_mode=self.plr_mode,
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
