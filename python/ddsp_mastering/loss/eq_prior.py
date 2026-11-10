"""Composite Parametric EQ Response Prior Loss Module (Sprint 3 Step V3a).

Evaluates the exact analytical composite frequency response of the 6-band
biquad EQ cascade in float64 precision and applies shape and offset priors:
  1. Centered shape roughness / slope penalty (L_slope_raw)
  2. Dead-zone excess shape deviation beyond tau_shape (L_shape_exc)
  3. Dead-zone global gain offset penalty beyond tau_offset (L_offset_raw)
"""

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ddsp_mastering.dsp.eq import DifferentiableEQ
from ddsp_mastering.param_space import NUM_PARAMS, Z_NEUTRAL, map_z_to_theta


class CompositeEQPriorLoss(nn.Module):
    """Composite Frequency Response Prior for 6-Band Parametric EQ.

    Computes:
      G_EQ(w_q) = sum_{i=1}^6 10 * log10( (|B_i|^2 + eps) / (|A_i|^2 + eps) )
      G_bar = mean_q( G_EQ(w_q) )
      G_tilde(w_q) = G_EQ(w_q) - G_bar
      L_shape_exc = mean_q( relu(|G_tilde| - tau_shape)^2 )
      L_slope_raw = mean_q( (G_tilde_{q+1} - G_tilde_q)^2 )
      L_offset_raw = relu(|G_bar| - tau_offset)^2
      L_EQ_shape = L_shape_exc + w_slope * L_slope_raw
      L_EQ_offset = w_offset * L_offset_raw
      L_EQ_prior = L_EQ_shape + L_EQ_offset
    """

    def __init__(
        self,
        sample_rate: float = 48000.0,
        f_min: float = 20.0,
        f_max: float = 20000.0,
        K_eq: int = 128,
        tau_shape: float = 1.5,
        tau_offset: float = 1.0,
        w_slope: float = 2.0,
        w_offset: float = 1.0,
        eps_eq: float = 1e-12,
    ):
        super().__init__()
        self.sample_rate = float(sample_rate)
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.K_eq = int(K_eq)
        self.tau_shape = float(tau_shape)
        self.tau_offset = float(tau_offset)
        self.w_slope = float(w_slope)
        self.w_offset = float(w_offset)
        self.eps_eq = float(eps_eq)

        # Precompute logarithmic frequency grid and phasors in float64
        log_f = torch.linspace(
            math.log10(self.f_min),
            math.log10(self.f_max),
            self.K_eq,
            dtype=torch.float64,
        )
        freqs = torch.pow(10.0, log_f)
        omega = (2.0 * math.pi * freqs) / self.sample_rate

        self.register_buffer("freqs", freqs)
        self.register_buffer("omega", omega)
        self.register_buffer("cos_w", torch.cos(omega))
        self.register_buffer("sin_w", torch.sin(omega))
        self.register_buffer("cos_2w", torch.cos(2.0 * omega))
        self.register_buffer("sin_2w", torch.sin(2.0 * omega))

        self.eq_dsp = DifferentiableEQ(sample_rate=self.sample_rate)

    def compute_composite_response_db(
        self, z: torch.Tensor, eps_eq: Optional[float] = None
    ) -> torch.Tensor:
        """Compute the composite dB frequency response G_EQ(w_q) in float64.

        Args:
            z: Normalized parameter vector of shape [..., 18] or [..., 41].
               If 18 params are passed, non-EQ parameters are padded with neutral values.
            eps_eq: Optional epsilon regularization. If None, self.eps_eq is used.

        Returns:
            G_EQ tensor of shape [..., K_eq] (or [K_eq] if 1D input).
        """
        eff_eps = self.eps_eq if eps_eq is None else float(eps_eq)
        z_f64 = z.to(torch.float64)

        if z.shape[-1] == 18:
            neutral_non_eq = Z_NEUTRAL[18:].to(
                device=z.device, dtype=torch.float64
            )
            if z.dim() == 1:
                z_full = torch.cat([z_f64, neutral_non_eq], dim=-1)
            else:
                expanded_non_eq = neutral_non_eq.expand(*z.shape[:-1], 23)
                z_full = torch.cat([z_f64, expanded_non_eq], dim=-1)
        elif z.shape[-1] == NUM_PARAMS:
            z_full = z_f64
        else:
            raise ValueError(
                f"Expected z to have 18 or {NUM_PARAMS} parameters, got {z.shape[-1]}"
            )

        is_1d = z_full.dim() == 1
        z_in = z_full.unsqueeze(0) if is_1d else z_full

        theta = map_z_to_theta(z_in)
        f0 = theta[..., 0:18:3]
        gain_db = theta[..., 1:18:3]
        q = theta[..., 2:18:3]

        b_norm, a_norm = self.eq_dsp.compute_biquad_coeffs(f0, gain_db, q)

        b0 = b_norm[..., 0].unsqueeze(-1)
        b1 = b_norm[..., 1].unsqueeze(-1)
        b2 = b_norm[..., 2].unsqueeze(-1)
        a1 = a_norm[..., 1].unsqueeze(-1)
        a2 = a_norm[..., 2].unsqueeze(-1)

        cos_w = self.cos_w.to(device=z.device, dtype=torch.float64).view(
            1, 1, self.K_eq
        )
        sin_w = self.sin_w.to(device=z.device, dtype=torch.float64).view(
            1, 1, self.K_eq
        )
        cos_2w = self.cos_2w.to(device=z.device, dtype=torch.float64).view(
            1, 1, self.K_eq
        )
        sin_2w = self.sin_2w.to(device=z.device, dtype=torch.float64).view(
            1, 1, self.K_eq
        )

        num_re = b0 + b1 * cos_w + b2 * cos_2w
        num_im = -(b1 * sin_w + b2 * sin_2w)
        num_sq = num_re ** 2 + num_im ** 2

        den_re = 1.0 + a1 * cos_w + a2 * cos_2w
        den_im = -(a1 * sin_w + a2 * sin_2w)
        den_sq = den_re ** 2 + den_im ** 2

        if eff_eps > 0.0:
            G_bands = 10.0 * torch.log10(
                (num_sq + eff_eps) / (den_sq + eff_eps)
            )
        else:
            G_bands = 10.0 * torch.log10(num_sq / den_sq)

        G_eq = torch.sum(G_bands, dim=-2)  # [..., K_eq]
        if is_1d:
            G_eq = G_eq.squeeze(0)
        return G_eq

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the composite EQ prior loss and diagnostic metrics.

        Args:
            z: Latent parameter vector (1D [41] or batched [B, 41] or [18]).

        Returns:
            Tuple of (scalar_loss_tensor, metrics_dict).
        """
        G_eq = self.compute_composite_response_db(z, eps_eq=self.eps_eq)
        is_1d = G_eq.dim() == 1
        G_eq_2d = G_eq.unsqueeze(0) if is_1d else G_eq

        G_bar = torch.mean(G_eq_2d, dim=-1, keepdim=True)  # [B, 1]
        G_tilde = G_eq_2d - G_bar  # [B, K_eq]

        shape_exc_raw = torch.relu(torch.abs(G_tilde) - self.tau_shape)
        L_shape_exc = torch.mean(shape_exc_raw ** 2, dim=-1)  # [B]

        diff_G = G_tilde[:, 1:] - G_tilde[:, :-1]  # [B, K_eq - 1]
        L_slope_raw = torch.mean(diff_G ** 2, dim=-1)  # [B]

        offset_exc_raw = torch.relu(
            torch.abs(G_bar.squeeze(-1)) - self.tau_offset
        )
        L_offset_raw = offset_exc_raw ** 2  # [B]

        L_eq_shape = L_shape_exc + self.w_slope * L_slope_raw  # [B]
        L_eq_offset = self.w_offset * L_offset_raw  # [B]
        L_eq_total = L_eq_shape + L_eq_offset  # [B]

        if is_1d:
            loss = L_eq_total.squeeze(0)
        else:
            loss = torch.mean(L_eq_total)

        metrics = {
            "G_bar": float(G_bar.mean().item()),
            "G_tilde_max": float(torch.max(G_tilde).item()),
            "G_tilde_min": float(torch.min(G_tilde).item()),
            "max_abs_G_tilde": float(torch.max(torch.abs(G_tilde)).item()),
            "L_shape_exc": float(L_shape_exc.mean().item()),
            "L_slope_raw": float(L_slope_raw.mean().item()),
            "L_offset_raw": float(L_offset_raw.mean().item()),
            "L_eq_shape": float(L_eq_shape.mean().item()),
            "L_eq_offset": float(L_eq_offset.mean().item()),
            "L_eq_prior_unweighted": float(L_eq_total.mean().item()),
        }

        return loss, metrics
