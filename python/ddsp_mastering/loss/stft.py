"""Multi-Resolution STFT Loss with Mid/Side Stereo Decomposition.

Computes Spectral Convergence and Log-Magnitude L1 losses across multiple FFT
resolutions simultaneously for Left/Right and normalized Mid/Side channels.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleResolutionSTFTLoss(nn.Module):
    """Computes Spectral Convergence and Log-Magnitude loss for a single FFT resolution."""

    def __init__(
        self,
        fft_size: int = 2048,
        hop_size: int = 512,
        win_length: int = 2048,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.eps = eps
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute spectral convergence and log magnitude loss.

        Args:
            x: Predicted audio tensor of shape (B, C, T) or (B, T).
            y: Target audio tensor of shape (B, C, T) or (B, T).

        Returns:
            Tuple of (spectral_convergence_loss, log_magnitude_loss).
        """
        if x.dim() == 3:
            B, C, T = x.shape
            x = x.reshape(B * C, T)
            y = y.reshape(B * C, T)

        # Compute STFT magnitude
        window = self.window.to(dtype=x.dtype, device=x.device)
        X_stft = torch.stft(
            x,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )
        Y_stft = torch.stft(
            y,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )

        X_mag = torch.abs(X_stft)
        Y_mag = torch.abs(Y_stft)

        # Spectral convergence loss: || |Y| - |X| ||_F / || |Y| ||_F
        sc_loss = torch.norm(Y_mag - X_mag, p="fro") / (
            torch.norm(Y_mag, p="fro") + self.eps
        )

        # Log-magnitude L1 loss: mean( | log(|Y| + eps) - log(|X| + eps) | )
        log_mag_loss = F.l1_loss(
            torch.log(Y_mag + self.eps), torch.log(X_mag + self.eps)
        )

        return sc_loss, log_mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-Resolution STFT Loss supporting Left/Right and Mid/Side evaluation.

    Default resolutions:
      - 512 points  (~10.7 ms at 48 kHz): Fine temporal transients
      - 2048 points (~42.7 ms at 48 kHz): Mid-range balance
      - 8192 points (~170.7 ms at 48 kHz, df ~ 5.86 Hz): Low-frequency precision
    """

    def __init__(
        self,
        fft_sizes: Tuple[int, ...] = (512, 2048, 8192),
        hop_sizes: Tuple[int, ...] = (128, 512, 2048),
        win_lengths: Tuple[int, ...] = (512, 2048, 8192),
        factor_sc: float = 1.0,
        factor_mag: float = 1.0,
        use_mid_side: bool = True,
        mid_side_weight: float = 0.5,
    ):
        super().__init__()
        assert (
            len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        ), "FFT configuration lengths must match."

        self.loss_layers = nn.ModuleList(
            [
                SingleResolutionSTFTLoss(
                    fft_size=f, hop_size=h, win_length=w
                )
                for f, h, w in zip(fft_sizes, hop_sizes, win_lengths)
            ]
        )
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag
        self.use_mid_side = use_mid_side
        self.mid_side_weight = mid_side_weight

    @staticmethod
    def to_mid_side(x: torch.Tensor) -> torch.Tensor:
        """Convert stereo signal (B, 2, T) to normalized Mid/Side representation."""
        left = x[:, 0:1, :]
        right = x[:, 1:2, :]
        mid = (left + right) / 1.4142135623730951  # sqrt(2)
        side = (left - right) / 1.4142135623730951  # sqrt(2)
        return torch.cat([mid, side], dim=1)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Compute multi-resolution STFT loss across channels.

        Args:
            pred: Predicted audio tensor of shape (B, 2, T).
            target: Target audio tensor of shape (B, 2, T).

        Returns:
            Tuple of (total_loss, metrics_dict).
        """
        assert pred.shape == target.shape, (
            f"Shape mismatch: {pred.shape} vs {target.shape}"
        )

        total_sc = 0.0
        total_mag = 0.0

        # 1. Left/Right STFT loss
        for layer in self.loss_layers:
            sc, mag = layer(pred, target)
            total_sc += sc
            total_mag += mag

        total_sc /= len(self.loss_layers)
        total_mag /= len(self.loss_layers)
        lr_loss = self.factor_sc * total_sc + self.factor_mag * total_mag

        metrics = {
            "stft_sc_lr": total_sc.item(),
            "stft_mag_lr": total_mag.item(),
            "stft_loss_lr": lr_loss.item(),
        }

        total_loss = lr_loss

        # 2. Mid/Side STFT loss (if stereo and enabled)
        if self.use_mid_side and pred.shape[1] == 2:
            pred_ms = self.to_mid_side(pred)
            target_ms = self.to_mid_side(target)

            total_sc_ms = 0.0
            total_mag_ms = 0.0
            for layer in self.loss_layers:
                sc, mag = layer(pred_ms, target_ms)
                total_sc_ms += sc
                total_mag_ms += mag

            total_sc_ms /= len(self.loss_layers)
            total_mag_ms /= len(self.loss_layers)
            ms_loss = (
                self.factor_sc * total_sc_ms + self.factor_mag * total_mag_ms
            )

            metrics["stft_sc_ms"] = total_sc_ms.item()
            metrics["stft_mag_ms"] = total_mag_ms.item()
            metrics["stft_loss_ms"] = ms_loss.item()

            total_loss = (
                total_loss + self.mid_side_weight * ms_loss
            ) / (1.0 + self.mid_side_weight)

        metrics["stft_loss_total"] = total_loss.item()
        return total_loss, metrics
