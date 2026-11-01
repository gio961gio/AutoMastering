import math
import torch
import torch.nn as nn
import torchaudio.functional as taf
from typing import Tuple, Optional

class DifferentiableEQ(nn.Module):
    """
    EQ Parametrico a 6 Bande Differenziabile (Dual Mode: 'iir' e 'fft').
    Isomorfo al motore C++ BiquadCascade6Band.
    
    Bande:
      0: Low-Shelf
      1..4: Peaking
      5: High-Shelf
    """
    def __init__(self, sample_rate: float = 48000.0, default_mode: str = "iir"):
        super().__init__()
        self.sample_rate = float(sample_rate)
        self.default_mode = default_mode

    def compute_biquad_coeffs(
        self, f0: torch.Tensor, gain_db: torch.Tensor, q: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcola i coefficienti [b0, b1, b2] e [a0=1, a1, a2] normalizzati per le 6 bande.
        Tutti i calcoli interni avvengono in torch.float64 per massima precisione numerica.
        
        Input shape:
          f0: [..., 6]
          gain_db: [..., 6]
          q: [..., 6]
        Output:
          b_coeffs: [..., 6, 3] (b0, b1, b2)
          a_coeffs: [..., 6, 3] (1.0, a1, a2)
        """
        f0_f64 = f0.to(torch.float64)
        gain_f64 = gain_db.to(torch.float64)
        q_f64 = q.to(torch.float64)

        w0 = (2.0 * math.pi * f0_f64) / self.sample_rate
        cos_w0 = torch.cos(w0)
        sin_w0 = torch.sin(w0)
        alpha = sin_w0 / (2.0 * q_f64)
        A = torch.pow(10.0, gain_f64 / 40.0)

        # Inizializziamo tensori coefficienti
        b0 = torch.zeros_like(w0)
        b1 = torch.zeros_like(w0)
        b2 = torch.zeros_like(w0)
        a0 = torch.zeros_like(w0)
        a1 = torch.zeros_like(w0)
        a2 = torch.zeros_like(w0)

        # Banda 0: Low-Shelf
        A_0 = A[..., 0]
        alpha_0 = alpha[..., 0]
        cos_w0_0 = cos_w0[..., 0]
        s_shelf_0 = 2.0 * torch.sqrt(A_0) * alpha_0
        Ap1_0 = A_0 + 1.0
        Am1_0 = A_0 - 1.0

        b0[..., 0] = A_0 * (Ap1_0 - Am1_0 * cos_w0_0 + s_shelf_0)
        b1[..., 0] = 2.0 * A_0 * (Am1_0 - Ap1_0 * cos_w0_0)
        b2[..., 0] = A_0 * (Ap1_0 - Am1_0 * cos_w0_0 - s_shelf_0)
        a0[..., 0] = Ap1_0 + Am1_0 * cos_w0_0 + s_shelf_0
        a1[..., 0] = -2.0 * (Am1_0 + Ap1_0 * cos_w0_0)
        a2[..., 0] = Ap1_0 + Am1_0 * cos_w0_0 - s_shelf_0

        # Bande 1..4: Peaking
        for b in range(1, 5):
            A_b = A[..., b]
            alpha_b = alpha[..., b]
            cos_w0_b = cos_w0[..., b]

            b0[..., b] = 1.0 + alpha_b * A_b
            b1[..., b] = -2.0 * cos_w0_b
            b2[..., b] = 1.0 - alpha_b * A_b
            a0[..., b] = 1.0 + alpha_b / A_b
            a1[..., b] = -2.0 * cos_w0_b
            a2[..., b] = 1.0 - alpha_b / A_b

        # Banda 5: High-Shelf
        A_5 = A[..., 5]
        alpha_5 = alpha[..., 5]
        cos_w0_5 = cos_w0[..., 5]
        s_shelf_5 = 2.0 * torch.sqrt(A_5) * alpha_5
        Ap1_5 = A_5 + 1.0
        Am1_5 = A_5 - 1.0

        b0[..., 5] = A_5 * (Ap1_5 + Am1_5 * cos_w0_5 + s_shelf_5)
        b1[..., 5] = -2.0 * A_5 * (Am1_5 + Ap1_5 * cos_w0_5)
        b2[..., 5] = A_5 * (Ap1_5 + Am1_5 * cos_w0_5 - s_shelf_5)
        a0[..., 5] = Ap1_5 - Am1_5 * cos_w0_5 + s_shelf_5
        a1[..., 5] = 2.0 * (Am1_5 - Ap1_5 * cos_w0_5)
        a2[..., 5] = Ap1_5 - Am1_5 * cos_w0_5 - s_shelf_5

        # Normalizzazione per a0
        inv_a0 = 1.0 / a0
        b_norm = torch.stack([b0 * inv_a0, b1 * inv_a0, b2 * inv_a0], dim=-1)
        a_norm = torch.stack([torch.ones_like(a0), a1 * inv_a0, a2 * inv_a0], dim=-1)

        return b_norm, a_norm

    def forward(
        self,
        x: torch.Tensor,
        f0: torch.Tensor,
        gain_db: torch.Tensor,
        q: torch.Tensor,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Applica la cascata EQ all'audio x.
        x: [..., C, N] (es. [2, N] stereo o [B, 2, N])
        mode: 'iir' (numerical parity with C++ reference within tested tolerances) o 'fft' (fast training approximation)
        """
        mode = mode or self.default_mode
        b_coeffs, a_coeffs = self.compute_biquad_coeffs(f0, gain_db, q)

        if mode == "iir":
            return self._forward_iir(x, b_coeffs, a_coeffs)
        elif mode == "fft":
            return self._forward_fft(x, b_coeffs, a_coeffs)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _forward_iir(
        self, x: torch.Tensor, b_coeffs: torch.Tensor, a_coeffs: torch.Tensor
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        orig_shape = x.shape
        N = orig_shape[-1]

        # Flatten audio to [num_ch, N] per compatibilità con taf.lfilter
        y = x.to(torch.float64).reshape(-1, N)
        num_ch = y.shape[0]

        for b in range(6):
            b_b = b_coeffs[..., b, :]
            a_b = a_coeffs[..., b, :]

            # Broadcast coefficienti su tutti i canali
            b_b = b_b.reshape(-1, 3)
            a_b = a_b.reshape(-1, 3)
            if b_b.shape[0] != num_ch:
                b_b = b_b.expand(num_ch, 3)
                a_b = a_b.expand(num_ch, 3)

            y = taf.lfilter(y, a_b, b_b, clamp=False, batching=True)

        return y.reshape(orig_shape).to(orig_dtype)

    def _forward_fft(
        self, x: torch.Tensor, b_coeffs: torch.Tensor, a_coeffs: torch.Tensor
    ) -> torch.Tensor:
        """
        Modalità Fast FFT per il training su GPU.
        Valutazione analitica della risposta in frequenza H(e^jw).
        """
        N = x.shape[-1]
        # Lunghezza FFT potenza di 2 con zero-padding per evitare circular convolution wrap
        M = 2 ** math.ceil(math.log2(2 * N))

        # Frequenze normalizzate w in [0, pi]
        w = torch.fft.rfftfreq(M, d=1.0, device=x.device, dtype=torch.float64) * (2.0 * math.pi)

        # Fasori
        cos_w = torch.cos(w)
        sin_w = torch.sin(w)
        cos_2w = torch.cos(2.0 * w)
        sin_2w = torch.sin(2.0 * w)

        z1 = torch.complex(cos_w, -sin_w)  # e^(-jw)
        z2 = torch.complex(cos_2w, -sin_2w) # e^(-j2w)

        # H_tot = prodotto sulle 6 bande
        H_tot = torch.ones_like(z1, dtype=torch.complex128)

        # Portiamo a complex128
        for b in range(6):
            b0 = b_coeffs[..., b, 0].unsqueeze(-1)
            b1 = b_coeffs[..., b, 1].unsqueeze(-1)
            b2 = b_coeffs[..., b, 2].unsqueeze(-1)
            a1 = a_coeffs[..., b, 1].unsqueeze(-1)
            a2 = a_coeffs[..., b, 2].unsqueeze(-1)

            num = b0 + b1 * z1 + b2 * z2
            den = 1.0 + a1 * z1 + a2 * z2
            H_b = num / den
            H_tot = H_tot * H_b

        # Convertiamo a complex64 e applichiamo all'audio
        H_tot_c64 = H_tot.to(torch.complex64)
        X_fft = torch.fft.rfft(x, n=M)
        Y_fft = X_fft * H_tot_c64
        y_padded = torch.fft.irfft(Y_fft, n=M)

        # Tronchiamo ai primi N campioni
        return y_padded[..., 0:N].to(x.dtype)
