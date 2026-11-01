import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# 31 Tap simmetrici FIR Half-Band (Kaiser window beta=8.5, stopband > 85 dB)
# Identici a Constants.hpp
FIR_HALFBAND_COEFFS = [
    -3.106101348136368259e-05,
     0.000000000000000000e+00,
     4.991001722697966693e-04,
     0.000000000000000000e+00,
    -2.328433331169654612e-03,
     0.000000000000000000e+00,
     7.254618576925952328e-03,
     0.000000000000000000e+00,
    -1.814484155598338616e-02,
     0.000000000000000000e+00,
     4.036219779069402658e-02,
     0.000000000000000000e+00,
    -9.030448489275624824e-02,
     0.000000000000000000e+00,
     3.126929042535008807e-01,
     5.000000000000000000e-01, # Centro
     3.126929042535008807e-01,
     0.000000000000000000e+00,
    -9.030448489275624824e-02,
     0.000000000000000000e+00,
     4.036219779069402658e-02,
     0.000000000000000000e+00,
    -1.814484155598338616e-02,
     0.000000000000000000e+00,
     7.254618576925952328e-03,
     0.000000000000000000e+00,
    -2.328433331169654612e-03,
     0.000000000000000000e+00,
     4.991001722697966693e-04,
     0.000000000000000000e+00,
    -3.106101348136368259e-05
]

class DifferentiableHarmonicSaturator(nn.Module):
    """
    Saturatore Armonico con Oversampling Polifase 2x e Compensazione Latenza Dry (15 campioni).
    Isomorfo al C++ HarmonicSaturator.
    """
    def __init__(self):
        super().__init__()
        # Registriamo i coefficienti FIR come buffer persistente float64
        h = torch.tensor(FIR_HALFBAND_COEFFS, dtype=torch.float64).view(1, 1, -1)
        self.register_buffer("h", h)

    def forward(
        self, x: torch.Tensor, drive_db: torch.Tensor, mix: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [..., C, N] (audio stereo o multicanale)
        drive_db: [...] o scalare in dB [0, 18]
        mix: [...] o scalare in [0, 1]
        """
        orig_dtype = x.dtype
        orig_shape = x.shape
        N = orig_shape[-1]
        # Appiattiamo a [B*C, 1, N] per conv1d
        x_flat = x.reshape(-1, 1, N).to(torch.float64)
        num_channels = x_flat.shape[0]

        # Parametri drive e mix
        drive_f64 = drive_db.to(torch.float64).reshape(-1, 1, 1)
        mix_f64 = torch.clamp(mix.to(torch.float64).reshape(-1, 1, 1), 0.0, 1.0)

        gd = torch.pow(10.0, drive_f64 / 20.0)
        inv_sqrt_gd = 1.0 / torch.sqrt(gd)

        # 1. Upsampling 2x (Zero-stuffing + Causal FIR)
        x_up = torch.zeros((num_channels, 1, 2 * N), device=x.device, dtype=torch.float64)
        x_up[:, :, 0::2] = 2.0 * x_flat

        # Convoluzione causale: padding sinistro di 30 campioni
        u_up = F.conv1d(x_up, self.h, padding=30)[:, :, : 2 * N]

        # 2. Waveshaper tanh compensato
        v_up = torch.tanh(gd * u_up) * inv_sqrt_gd

        # 3. Downsampling 2x (Causal FIR con stride 2)
        y_wet = F.conv1d(v_up, self.h, padding=30, stride=2)[:, :, :N]

        # 4. Compensazione Latenza Dry (esattamente 15 campioni a fs)
        x_dry = F.pad(x_flat, (15, 0))[:, :, :N]

        # 5. Mix Dry/Wet perfettamente coerente in fase
        y_sat = (1.0 - mix_f64) * x_dry + mix_f64 * y_wet

        return y_sat.reshape(orig_shape).to(orig_dtype)
