import math
import torch
import torch.nn as nn
import torch.nn.functional as F

LOOKAHEAD_SAMPLES = 144
CEILING_DBFS = -0.3
C_LIN = 10.0 ** (CEILING_DBFS / 20.0) # ~0.96605087870305
RELEASE_MS = 80.0

class DifferentiableLookaheadLimiter(nn.Module):
    """
    Lookahead Brickwall Limiter Differenziabile.
    Isomorfo al C++ LookaheadLimiter con Min-Filter su finestra NL=144,
    One-Pole Release Envelope (80 ms) e Boxcar Moving Average Smoothing (144 campioni).
    """
    def __init__(self, sample_rate: float = 48000.0):
        super().__init__()
        self.sample_rate = float(sample_rate)
        self.lookahead = LOOKAHEAD_SAMPLES
        self.c_lin = C_LIN
        tau_sec = RELEASE_MS * 0.001
        self.alpha_rel = math.exp(-1.0 / (tau_sec * self.sample_rate))

    def forward(self, x: torch.Tensor, push_gain_db: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., 2, N] Segnale audio stereo in ingresso
            push_gain_db: [..., 1] o [..., ] Guadagno di push in dB [0.0, 12.0]
        Returns:
            y_out: [..., 2, N] Segnale limitato a ceiling -0.3 dBFS, ritardato di 144 campioni
        """
        orig_shape = x.shape
        orig_dtype = x.dtype
        N = x.shape[-1]
        x_f64 = x.to(torch.float64).reshape(-1, 2, N)
        B = x_f64.shape[0]

        # 1. Gain Push in float64
        push_db = push_gain_db.reshape(B, 1, 1).to(torch.float64)
        push_lin = torch.pow(10.0, push_db / 20.0)
        u = x_f64 * push_lin # [B, 2, N]

        # 2. Peak Detection & Target Gain Istantaneo in float64
        peak = torch.clamp(
            torch.max(torch.abs(u[:, 0, :]), torch.abs(u[:, 1, :])),
            min=1e-7
        ) # [B, N]
        g_target = torch.clamp(self.c_lin / peak, max=1.0) # [B, N]

        # 3. Stadio 1: Min-Filter su finestra NL=144 via -max_pool1d(-padded)
        g_target_padded = F.pad(g_target.unsqueeze(1), (self.lookahead - 1, 0), mode='constant', value=1.0)
        g_min = -F.max_pool1d(-g_target_padded, kernel_size=self.lookahead, stride=1).squeeze(1) # [B, N]

        # 4. Stadio 2: One-Pole Release Envelope
        # Calcolo sequenziale differenziabile in float64
        g_rel = torch.empty_like(g_min)
        alpha = self.alpha_rel
        g_prev = torch.ones(B, dtype=torch.float64, device=x.device)
        for n in range(N):
            gm = g_min[:, n]
            cond = gm <= g_prev
            g_curr = torch.where(cond, gm, alpha * g_prev + (1.0 - alpha) * gm)
            g_rel[:, n] = g_curr
            g_prev = g_curr

        # 5. Stadio 3: Boxcar Moving Average Smoothing di lunghezza NL=144
        g_rel_padded = F.pad(g_rel.unsqueeze(1), (self.lookahead - 1, 0), mode='constant', value=1.0)
        g_smooth = F.avg_pool1d(g_rel_padded, kernel_size=self.lookahead, stride=1).squeeze(1) # [B, N]

        # 6. Audio ritardato di NL=144 campioni e applicazione guadagno
        u_delayed = F.pad(u, (self.lookahead, 0))[..., :N] # [B, 2, N]
        y_out = u_delayed * g_smooth.unsqueeze(1) # [B, 2, N]

        return y_out.reshape(orig_shape).to(orig_dtype)
