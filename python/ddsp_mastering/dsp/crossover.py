import math
import torch
import torch.nn as nn
import torchaudio.functional as taf
from typing import Tuple

SQRT2 = 1.4142135623730950488

class DifferentiableLR4Crossover(nn.Module):
    """
    Crossover Stereo a 3 Vie Linkwitz-Riley del 4° ordine compensato in fase.
    Isomorfo al C++ LinkwitzRiley4_3Way.
    
    Split:
      y_low:  LP4(flow) + AP2(fhigh)
      y_mid:  HP4(flow) + LP4(fhigh)
      y_high: HP4(flow) + HP4(fhigh)
      
    Garantisce somma piatta: y_low + y_mid + y_high = x * AP2(flow) * AP2(fhigh) (0.000 dB).
    """
    def __init__(self, sample_rate: float = 48000.0):
        super().__init__()
        self.sample_rate = float(sample_rate)

    def compute_biquad_butterworth(
        self, fc: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calcola coefficienti LP2, HP2, AP2 per frequenza di taglio fc.
        fc: [...]
        Returns:
          b_lp: [..., 3], b_hp: [..., 3], b_ap: [..., 3], a_coeffs: [..., 3]
        """
        fc_f64 = fc.to(torch.float64)
        wc = (2.0 * math.pi * fc_f64) / self.sample_rate
        k = torch.tan(wc * 0.5)
        k2 = k * k
        a0 = 1.0 + SQRT2 * k + k2
        inv_a0 = 1.0 / a0

        a1 = 2.0 * (k2 - 1.0) * inv_a0
        a2 = (1.0 - SQRT2 * k + k2) * inv_a0

        # LP2
        b0_lp = k2 * inv_a0
        b_lp = torch.stack([b0_lp, 2.0 * b0_lp, b0_lp], dim=-1)

        # HP2
        b0_hp = inv_a0
        b_hp = torch.stack([b0_hp, -2.0 * b0_hp, b0_hp], dim=-1)

        # AP2
        b_ap = torch.stack([a2, a1, torch.ones_like(a2)], dim=-1)

        a_coeffs = torch.stack([torch.ones_like(a1), a1, a2], dim=-1)

        return b_lp, b_hp, b_ap, a_coeffs

    def forward(
        self, x: torch.Tensor, f_low: torch.Tensor, f_high: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Divide l'audio x in 3 bande.
        x: [..., C, N]
        f_low: [...]
        f_high: [...]
        Returns:
          (y_low, y_mid, y_high)
        """
        orig_shape = x.shape
        orig_dtype = x.dtype
        N = x.shape[-1]
        C = x.shape[-2]
        B = x.shape[:-2].numel() if len(x.shape) > 2 else 1

        x_flat = x.to(torch.float64).reshape(B * C, N)

        # Assicura che f_low e f_high abbiano dimensione batch B
        if f_low.ndim == 0:
            f_low = f_low.expand(B)
        else:
            f_low = f_low.reshape(B)

        if f_high.ndim == 0:
            f_high = f_high.expand(B)
        else:
            f_high = f_high.reshape(B)

        # Coefficienti per flow e fhigh
        b_lp_low, b_hp_low, _, a_low = self.compute_biquad_butterworth(f_low)
        b_lp_high, b_hp_high, b_ap_high, a_high = self.compute_biquad_butterworth(f_high)

        # Espansione su tutti i canali C
        def expand_coeffs(c_tensor):
            return c_tensor.reshape(B, 1, 3).expand(B, C, 3).reshape(B * C, 3)

        a_low_rep = expand_coeffs(a_low)
        b_lp_low_rep = expand_coeffs(b_lp_low)
        b_hp_low_rep = expand_coeffs(b_hp_low)

        a_high_rep = expand_coeffs(a_high)
        b_lp_high_rep = expand_coeffs(b_lp_high)
        b_hp_high_rep = expand_coeffs(b_hp_high)
        b_ap_high_rep = expand_coeffs(b_ap_high)

        # 1. Split a flow
        # LP4 a flow = 2 stadi di LP2
        u_low = taf.lfilter(x_flat, a_low_rep, b_lp_low_rep, clamp=False)
        u_low = taf.lfilter(u_low, a_low_rep, b_lp_low_rep, clamp=False)

        # HP4 a flow = 2 stadi di HP2
        u_mh = taf.lfilter(x_flat, a_low_rep, b_hp_low_rep, clamp=False)
        u_mh = taf.lfilter(u_mh, a_low_rep, b_hp_low_rep, clamp=False)

        # 2. Compensazione All-Pass su Low branch a fhigh
        y_low = taf.lfilter(u_low, a_high_rep, b_ap_high_rep, clamp=False)

        # 3. Split a fhigh sul ramo Mid-High
        # Mid: LP4 a fhigh
        y_mid = taf.lfilter(u_mh, a_high_rep, b_lp_high_rep, clamp=False)
        y_mid = taf.lfilter(y_mid, a_high_rep, b_lp_high_rep, clamp=False)

        # High: HP4 a fhigh
        y_high = taf.lfilter(u_mh, a_high_rep, b_hp_high_rep, clamp=False)
        y_high = taf.lfilter(y_high, a_high_rep, b_hp_high_rep, clamp=False)

        return (
            y_low.reshape(orig_shape).to(orig_dtype),
            y_mid.reshape(orig_shape).to(orig_dtype),
            y_high.reshape(orig_shape).to(orig_dtype),
        )
