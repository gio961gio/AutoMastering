"""Audio Engineering Quantitative Benchmark Suite.

Computes psychoacoustic and engineering metrics:
1. Spectral Convergence (SC)
2. Log-Spectral Distance (LSD in dB)
3. Integrated Loudness Error (|Delta LUFS| in LU)
4. True Peak & Ceiling Compliance (<= -0.3 dBFS)
5. Dynamic Range / Crest Factor (PLR in dB)
6. Stereo Mid-to-Side Energy Ratio Error (Delta E_M/S in dB)
7. Parameter Recovery Error (L1, L2 distance in latent space z)
"""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F

from ddsp_mastering.loss.loudness import DifferentiableLoudnessLoss
from ddsp_mastering.loss.stft import MultiResolutionSTFTLoss


class MasteringBenchmarkSuite:
    """Evaluates mastered audio outputs against targets and acoustic standards."""

    def __init__(self, sample_rate: float = 48000.0, latency_samples: int = 159):
        self.sample_rate = sample_rate
        self.latency_samples = latency_samples
        self.loudness_calc = DifferentiableLoudnessLoss(sample_rate=sample_rate)
        self.stft_loss = MultiResolutionSTFTLoss()

    def align_latency(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Slice latency-aligned regions."""
        D = self.latency_samples
        T = pred.shape[-1]
        if D <= 0 or T <= D:
            return pred, target
        return pred[..., D:], target[..., : T - D]

    @staticmethod
    def calculate_log_spectral_distance(
        pred: torch.Tensor,
        target: torch.Tensor,
        n_fft: int = 2048,
        hop_length: int = 512,
        eps: float = 1e-8,
    ) -> float:
        """Compute Log-Spectral Distance (LSD) in dB between two audio signals."""
        if pred.dim() == 3:
            pred = pred.reshape(-1, pred.shape[-1])
            target = target.reshape(-1, target.shape[-1])

        window = torch.hann_window(n_fft, device=pred.device, dtype=pred.dtype)
        P_stft = torch.stft(pred, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        T_stft = torch.stft(target, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)

        P_power = torch.clamp(torch.abs(P_stft) ** 2, min=eps)
        T_power = torch.clamp(torch.abs(T_stft) ** 2, min=eps)

        # Log power in dB: 10 * log10(P)
        P_db = 10.0 * torch.log10(P_power)
        T_db = 10.0 * torch.log10(T_power)

        # Mean squared difference across frequency bins per frame, then sqrt, then mean over frames
        frame_lsd = torch.sqrt(torch.mean((P_db - T_db) ** 2, dim=-2))  # (B*C, num_frames)
        mean_lsd = torch.mean(frame_lsd).item()
        return mean_lsd

    @staticmethod
    def calculate_mid_side_energy_ratio(audio: torch.Tensor, eps: float = 1e-8) -> float:
        """Compute the Mid-to-Side energy ratio in dB."""
        assert audio.shape[-2] == 2, "Expected stereo audio"
        left = audio[..., 0, :]
        right = audio[..., 1, :]
        mid = (left + right) / 1.41421356
        side = (left - right) / 1.41421356

        e_mid = torch.mean(mid ** 2).item()
        e_side = torch.mean(side ** 2).item()
        ratio_db = 10.0 * math.log10((e_mid + eps) / (e_side + eps))
        return ratio_db

    def evaluate(
        self,
        pred_audio: torch.Tensor,
        target_audio: torch.Tensor,
        z_pred: Optional[torch.Tensor] = None,
        z_target: Optional[torch.Tensor] = None,
        align_latency: bool = True,
    ) -> Dict[str, float]:
        """Compute full benchmark metric dictionary.

        Args:
            pred_audio: Audio tensor (1, 2, T) or (2, T).
            target_audio: Audio tensor (1, 2, T) or (2, T).
            z_pred: Normalized parameter vector (41,) or (1, 41).
            z_target: Target parameter vector (41,) or (1, 41).
            align_latency: Whether to slice off chain latency for spectral comparisons.

        Returns:
            Dictionary containing all objective benchmark metrics.
        """
        if pred_audio.dim() == 2:
            pred_audio = pred_audio.unsqueeze(0)
        if target_audio.dim() == 2:
            target_audio = target_audio.unsqueeze(0)

        # 1. Aligned audio for spectral metrics
        if align_latency:
            p_al, t_al = self.align_latency(pred_audio, target_audio)
        else:
            p_al, t_al = pred_audio, target_audio

        # Spectral Metrics
        _, stft_metrics = self.stft_loss(p_al, t_al)
        sc = stft_metrics["stft_sc_lr"]
        lsd = self.calculate_log_spectral_distance(p_al, t_al)

        # Loudness Metrics
        pred_lufs = self.loudness_calc.calculate_lufs(pred_audio).item()
        target_lufs = self.loudness_calc.calculate_lufs(target_audio).item()
        delta_lufs = abs(pred_lufs - target_lufs)

        # Peak & Ceiling Compliance
        pred_peak_linear = torch.max(torch.abs(pred_audio)).item()
        pred_peak_dbfs = 20.0 * math.log10(pred_peak_linear + 1e-12)
        ceiling_violation_db = max(0.0, pred_peak_dbfs - (-0.3000))

        # Dynamic Range / PLR (Peak to Loudness Ratio)
        pred_plr = pred_peak_dbfs - pred_lufs
        target_peak_linear = torch.max(torch.abs(target_audio)).item()
        target_peak_dbfs = 20.0 * math.log10(target_peak_linear + 1e-12)
        target_plr = target_peak_dbfs - target_lufs
        delta_plr = abs(pred_plr - target_plr)

        # Stereo Imaging: Mid-to-Side Energy Ratio
        pred_ms_ratio = self.calculate_mid_side_energy_ratio(pred_audio)
        target_ms_ratio = self.calculate_mid_side_energy_ratio(target_audio)
        delta_ms_ratio = abs(pred_ms_ratio - target_ms_ratio)

        results = {
            "spectral_convergence": sc,
            "log_spectral_distance_db": lsd,
            "pred_lufs": pred_lufs,
            "target_lufs": target_lufs,
            "delta_lufs": delta_lufs,
            "pred_peak_dbfs": pred_peak_dbfs,
            "ceiling_violation_db": ceiling_violation_db,
            "pred_plr_db": pred_plr,
            "target_plr_db": target_plr,
            "delta_plr_db": delta_plr,
            "pred_ms_ratio_db": pred_ms_ratio,
            "target_ms_ratio_db": target_ms_ratio,
            "delta_ms_ratio_db": delta_ms_ratio,
        }

        # Parameter recovery metrics (if ground truth z is provided)
        if z_pred is not None and z_target is not None:
            zp = z_pred.reshape(-1)
            zt = z_target.reshape(-1)
            results["param_l1_error"] = torch.mean(torch.abs(zp - zt)).item()
            results["param_l2_error"] = torch.norm(zp - zt).item()
            results["param_max_error"] = torch.max(torch.abs(zp - zt)).item()

        return results
