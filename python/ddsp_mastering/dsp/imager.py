import torch
import torch.nn as nn
from typing import Tuple

INV_SQRT2 = 0.70710678118654752440

class DifferentiableMidSideImager(nn.Module):
    """
    Stereo Imager Mid/Side differenziabile per banda.
    Isomorfo al C++ MidSideImager.
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def process_band(x: torch.Tensor, width: torch.Tensor) -> torch.Tensor:
        """
        Applica lo stereo width a una banda audio.
        x: [..., 2, N] (stereo)
        width: [...] o scalare
        """
        orig_dtype = x.dtype
        x_f64 = x.to(torch.float64)
        w_f64 = torch.clamp(width.to(torch.float64), min=0.0).unsqueeze(-1) # broadcast su N

        l = x_f64[..., 0, :]
        r = x_f64[..., 1, :]

        # Codifica Mid/Side
        mid = (l + r) * INV_SQRT2
        side = (l - r) * INV_SQRT2 * w_f64

        # Decodifica Left/Right
        out_l = (mid + side) * INV_SQRT2
        out_r = (mid - side) * INV_SQRT2

        out = torch.stack([out_l, out_r], dim=-2)
        return out.to(orig_dtype)

    def forward(
        self,
        low: torch.Tensor,
        mid_band: torch.Tensor,
        high: torch.Tensor,
        width_low: torch.Tensor,
        width_mid: torch.Tensor,
        width_high: torch.Tensor,
    ) -> torch.Tensor:
        """
        Applica l'imaging a ciascuna delle 3 bande e le somma in x_sum.
        """
        out_low = self.process_band(low, width_low)
        out_mid = self.process_band(mid_band, width_mid)
        out_high = self.process_band(high, width_high)

        return out_low + out_mid + out_high
