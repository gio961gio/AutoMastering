import torch
import torch.nn as nn
from typing import Optional, Dict, Union

from .eq import DifferentiableEQ as DifferentiableParametricEQ
from .crossover import DifferentiableLR4Crossover
from .compressor import DifferentiableMultibandCompressor
from .imager import DifferentiableMidSideImager
from .saturator import DifferentiableHarmonicSaturator
from .limiter import DifferentiableLookaheadLimiter
from ..param_space import map_z_to_theta, unpack_theta_dict, NUM_PARAMS

class DifferentiableMasteringChain(nn.Module):
    """
    Catena Completa di Mastering DDSP White-Box Differenziabile (6 stadi).
    Isomorfa al C++ core_dsp::MasteringChain.

    Stadi:
      1. EQ Parametrico 6-Bande (IIR o Fast FFT)
      2. Crossover 3-Vie Linkwitz-Riley 4° ordine compensato AP2
      3. Compressore Multibanda 3-Bande con Log-Domain Ballistics
      4. Stereo Imager M/S 3-Bande & Summing
      5. Saturatore Armonico con Oversampling 2x e Dry Delay di 15 campioni
      6. Lookahead Brickwall Limiter con finestra NL=144 campioni e ceiling -0.3 dBFS

    Latenza Totale della Catena: esattamente 159 campioni (15 sat + 144 limiter).
    """
    def __init__(self, sample_rate: float = 48000.0, default_eq_mode: str = "iir"):
        super().__init__()
        self.sample_rate = float(sample_rate)
        self.default_eq_mode = default_eq_mode

        self.eq = DifferentiableParametricEQ(sample_rate=self.sample_rate, default_mode=self.default_eq_mode)
        self.crossover = DifferentiableLR4Crossover(sample_rate=self.sample_rate)
        self.compressor = DifferentiableMultibandCompressor(sample_rate=self.sample_rate)
        self.imager = DifferentiableMidSideImager()
        self.saturator = DifferentiableHarmonicSaturator()
        self.limiter = DifferentiableLookaheadLimiter(sample_rate=self.sample_rate)

    @property
    def latency_samples(self) -> int:
        return 159

    def forward(
        self,
        x: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        eq_mode: Optional[str] = None,
        return_diagnostics: bool = False,
    ):
        """
        Elabora l'audio attraverso l'intera catena di mastering.

        Args:
            x: [..., 2, N] Segnale audio stereo
            z: [..., 41] Vettore parametri normalizzati in (0, 1) (opzionale se passato theta)
            theta: [..., 41] Vettore parametri fisici reali (opzionale se passato z)
            eq_mode: 'iir' (numerical parity with C++ reference within tested tolerances) o 'fft' (fast training approximation)
            return_diagnostics: Se True restituisce (y_out, diagnostics_dict)
        Returns:
            y_out: [..., 2, N] Audio masterizzato finale
            oppure tupla (y_out, diagnostics_dict) se return_diagnostics=True
        """
        if z is None and theta is None:
            raise ValueError("Must provide either normalized parameters 'z' or physical parameters 'theta'")

        if theta is None:
            theta = map_z_to_theta(z)

        params = unpack_theta_dict(theta)
        eq_mode = eq_mode or self.default_eq_mode

        # 1. Parametric EQ 6-Band
        eq_out = self.eq(
            x,
            params["eq_f0"],
            params["eq_gain_db"],
            params["eq_q"],
            mode=eq_mode
        )

        # 2. Crossover LR4 (3 vie stereo)
        low, mid, high = self.crossover(
            eq_out,
            params["crossover_f_low"],
            params["crossover_f_high"]
        )

        # 3. Multiband Compressor (3 bande parallele)
        comp_res = self.compressor(
            low, mid, high,
            params["comp_threshold_db"],
            params["comp_ratio"],
            params["comp_attack_ms"],
            params["comp_release_ms"],
            params["comp_makeup_db"],
            return_gr=return_diagnostics,
        )
        if return_diagnostics:
            (comp_low, comp_mid, comp_high), comp_diag = comp_res
        else:
            comp_low, comp_mid, comp_high = comp_res

        # 4. Stereo Imager & 3-Band Summing
        w = params["imager_width"] # [..., 3]
        sum_out = self.imager(
            comp_low, comp_mid, comp_high,
            w[..., 0], w[..., 1], w[..., 2]
        )

        # 5. Harmonic Saturator (2x oversampled)
        sat_out = self.saturator(
            sum_out,
            params["saturator_drive_db"],
            params["saturator_mix"]
        )

        # 6. Lookahead Brickwall Limiter (144 campioni lookahead)
        lim_res = self.limiter(
            sat_out,
            params["limiter_push_gain_db"],
            return_gain=return_diagnostics,
        )

        if return_diagnostics:
            lim_out, lim_diag = lim_res
            diagnostics = {
                "gr_low_db": comp_diag["gr_low_db"],
                "gr_mid_db": comp_diag["gr_mid_db"],
                "gr_high_db": comp_diag["gr_high_db"],
                "lim_g_smooth": lim_diag["g_smooth"],
                "lim_atten_db": lim_diag["a_lim_db"],
            }
            return lim_out, diagnostics

        return lim_res
