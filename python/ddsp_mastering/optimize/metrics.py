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
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torchaudio.functional as taf

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

    @staticmethod
    def calculate_aggregate_spectral_distance(
        pred: torch.Tensor,
        target: torch.Tensor,
        sample_rate: float = 48000.0,
        n_mels: int = 40,
        n_fft: int = 2048,
        hop_length: int = 512,
    ) -> float:
        """Compute L1 distance between normalized long-term log-Mel spectral distributions.

        Used for stylistic evaluation where timelines and musical arrangements differ.
        """
        n_freqs = n_fft // 2 + 1
        fb = taf.melscale_fbanks(
            n_freqs=n_freqs,
            f_min=20.0,
            f_max=min(20000.0, sample_rate * 0.499),
            n_mels=n_mels,
            sample_rate=int(sample_rate),
        ).to(pred.device)
        window = torch.hann_window(n_fft, device=pred.device)

        def _get_mel_shape(sig: torch.Tensor) -> torch.Tensor:
            if sig.ndim == 2:
                sig = sig.unsqueeze(0)  # [1, C, T]
            B, C, T = sig.shape
            sig_flat = sig.reshape(B * C, T)
            stft_res = torch.stft(
                sig_flat,
                n_fft=n_fft,
                hop_length=hop_length,
                window=window,
                return_complex=True,
            )
            mag_sq_all = (torch.abs(stft_res) ** 2).reshape(B, C, -1, stft_res.shape[-1])
            if C == 2:
                mag_sq = 0.5 * (mag_sq_all[:, 0, :, :] + mag_sq_all[:, 1, :, :])
            else:
                mag_sq = mag_sq_all[:, 0, :, :]
            mel_e = torch.matmul(mag_sq.transpose(-1, -2), fb)
            avg_mel = torch.mean(mel_e, dim=-2)
            log_mel = torch.log(avg_mel + 1e-7)
            return log_mel - torch.mean(log_mel, dim=-1, keepdim=True)

        shape_p = _get_mel_shape(pred)
        shape_t = _get_mel_shape(target)
        return torch.mean(torch.abs(shape_p - shape_t)).item()

    def evaluate(
        self,
        pred_audio: torch.Tensor,
        target_audio: torch.Tensor,
        z_pred: Optional[torch.Tensor] = None,
        z_target: Optional[torch.Tensor] = None,
        align_latency: bool = True,
        reference_mode: str = "same_track",
    ) -> Dict[str, Union[float, str, List[str], None]]:
        """Compute objective benchmark metric dictionary.

        Args:
            pred_audio: Audio tensor (1, 2, T_pred) or (2, T_pred).
            target_audio: Audio tensor (1, 2, T_target) or (2, T_target).
            z_pred: Normalized parameter vector (41,) or (1, 41).
            z_target: Target parameter vector (41,) or (1, 41).
            align_latency: Whether to slice off chain latency for same_track spectral comparisons.
            reference_mode: "same_track" | "stylistic". Unrecognized modes raise ValueError.

        Returns:
            Dictionary containing applicable benchmark metrics. For 'stylistic', frame-by-frame
            MR-STFT and Log-Spectral Distance are strictly omitted (set to None) and replaced
            with aggregate spectral envelope distance.
        """
        valid_modes = {"same_track", "stylistic"}
        if reference_mode not in valid_modes:
            raise ValueError(
                f"Unrecognized reference_mode: '{reference_mode}'. "
                f"Must be one of {sorted(list(valid_modes))}."
            )

        if pred_audio.dim() == 2:
            pred_audio = pred_audio.unsqueeze(0)
        if target_audio.dim() == 2:
            target_audio = target_audio.unsqueeze(0)

        # Common Aggregate Descriptors (valid across tracks of differing lengths)
        pred_lufs = self.loudness_calc.calculate_lufs(pred_audio).item()
        target_lufs = self.loudness_calc.calculate_lufs(target_audio).item()
        delta_lufs = abs(pred_lufs - target_lufs)

        pred_peak_linear = torch.max(torch.abs(pred_audio)).item()
        pred_peak_dbfs = 20.0 * math.log10(pred_peak_linear + 1e-12)
        ceiling_violation_db = max(0.0, pred_peak_dbfs - (-0.3000))

        pred_plr = pred_peak_dbfs - pred_lufs
        target_peak_linear = torch.max(torch.abs(target_audio)).item()
        target_peak_dbfs = 20.0 * math.log10(target_peak_linear + 1e-12)
        target_plr = target_peak_dbfs - target_lufs
        delta_plr = abs(pred_plr - target_plr)

        pred_ms_ratio = self.calculate_mid_side_energy_ratio(pred_audio)
        target_ms_ratio = self.calculate_mid_side_energy_ratio(target_audio)
        delta_ms_ratio = abs(pred_ms_ratio - target_ms_ratio)

        results: Dict[str, Union[float, str, List[str], None]] = {
            "reference_mode": reference_mode,
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

        if reference_mode == "same_track":
            # Aligned audio for spectral metrics
            if align_latency:
                p_al, t_al = self.align_latency(pred_audio, target_audio)
            else:
                p_al, t_al = pred_audio, target_audio

            min_len = min(p_al.shape[-1], t_al.shape[-1])
            p_al = p_al[..., :min_len]
            t_al = t_al[..., :min_len]

            _, stft_metrics = self.stft_loss(p_al, t_al)
            sc = stft_metrics["stft_sc_lr"]
            lsd = self.calculate_log_spectral_distance(p_al, t_al)

            results["spectral_convergence"] = sc
            results["log_spectral_distance_db"] = lsd
            results["aggregate_spectral_distance"] = None
            results["applicable_metrics"] = [
                "spectral_convergence",
                "log_spectral_distance_db",
                "delta_lufs",
                "pred_peak_dbfs",
                "ceiling_violation_db",
                "delta_plr_db",
                "delta_ms_ratio_db",
            ]
        else:
            # Stylistic mode: strictly reject frame-by-frame STFT / LSD
            agg_spec_dist = self.calculate_aggregate_spectral_distance(
                pred_audio, target_audio, sample_rate=self.sample_rate
            )
            results["spectral_convergence"] = None
            results["log_spectral_distance_db"] = None
            results["aggregate_spectral_distance"] = agg_spec_dist
            results["applicable_metrics"] = [
                "delta_lufs",
                "pred_peak_dbfs",
                "ceiling_violation_db",
                "delta_plr_db",
                "delta_ms_ratio_db",
                "aggregate_spectral_distance",
            ]

        # Parameter recovery metrics (if ground truth z is provided)
        if z_pred is not None and z_target is not None:
            zp = z_pred.reshape(-1)
            zt = z_target.reshape(-1)
            results["param_l1_error"] = torch.mean(torch.abs(zp - zt)).item()
            results["param_l2_error"] = torch.norm(zp - zt).item()
            results["param_max_error"] = torch.max(torch.abs(zp - zt)).item()

        return results
