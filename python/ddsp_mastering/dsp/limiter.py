import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import py_core_dsp
    _HAS_PY_CORE_DSP = True
except ImportError:
    _HAS_PY_CORE_DSP = False

LOOKAHEAD_SAMPLES = 144
CEILING_DBFS = -0.3                # Legacy sample-peak ceiling / delivery contract
INTERNAL_CEILING_DBTP = -0.50      # Sprint 4 internal True Peak target ceiling
DELIVERY_CEILING_DBTP = -0.30      # Final PCM_24 contractual delivery ceiling
C_LIN = 10.0 ** (CEILING_DBFS / 20.0) # ~0.96605087870305 (legacy reference)
C_LIN_TP = 10.0 ** (INTERNAL_CEILING_DBTP / 20.0) # ~0.9440608762859234
RELEASE_MS = 80.0

TP_OVERSAMPLE_FACTOR = 4
TP_FIR_TAPS_PER_PHASE = 12
TP_FIR_PROTOTYPE_TAPS = TP_OVERSAMPLE_FACTOR * TP_FIR_TAPS_PER_PHASE # 48
TP_FIR_GROUP_DELAY_BASE_SAMPLES = (TP_FIR_PROTOTYPE_TAPS - 1) / (2.0 * TP_OVERSAMPLE_FACTOR) # 5.875
TP_FIR_ALIGNED_SAMPLE_DELAY = 6    # ceil(5.875)
TP_BOXCAR_SAMPLES = 136            # Zero-release-leak boxcar width matched to NL=144 and 6-sample FIR alignment

# Canonical 48-tap Type-II symmetric WLS polyphase FIR prototype coefficients (mirror of Constants.hpp)
TP_FIR_PROTOTYPE_COEFFS = (
    0.021101060308359995,   0.00033638122803119523, -0.02290233358771963,  -0.032003183981007287,
   -0.017567715780417031,   0.014601578655383312,    0.04401592198233905,   0.047373416913966925,
    0.016016879513805326,  -0.035378170875529531,   -0.073317895426556409, -0.065697780182872964,
   -0.006455635993611754,   0.075332650672725376,    0.12563247533387883,   0.097651474034789229,
   -0.012387342575628709,  -0.15327403357677791,    -0.23299736782631375,  -0.16527746191083259,
    0.077837917251192004,   0.44505196611893527,     0.81289882730160423,   1.0394083724022567,
    1.0394083724022567,     0.81289882730160401,     0.44505196611893533,   0.077837917251192004,
   -0.16527746191083259,   -0.23299736782631369,    -0.15327403357677793,  -0.012387342575628709,
    0.097651474034789229,   0.12563247533387881,     0.07533265067272539,  -0.006455635993611754,
   -0.065697780182872964,  -0.073317895426556395,   -0.035378170875529538,  0.016016879513805326,
    0.047373416913966925,   0.044015921982339043,    0.014601578655383316, -0.017567715780417031,
   -0.032003183981007287,  -0.022902333587719623,    0.00033638122803119529, 0.021101060308359995,
)


def get_tp_fir_prototype_coeffs():
    """Restituisce i 48 coefficienti prototipo 4x FIR dalla C++ single source of truth se compilata."""
    if _HAS_PY_CORE_DSP and hasattr(py_core_dsp, "get_true_peak_fir_coeffs"):
        return tuple(float(v) for v in py_core_dsp.get_true_peak_fir_coeffs())
    return TP_FIR_PROTOTYPE_COEFFS


def _build_polyphase_conv_weight() -> torch.Tensor:
    """Costruisce il kernel Conv1d causale [4, 1, 12] (float64) dai 4 rami polifase."""
    proto = list(get_tp_fir_prototype_coeffs())
    if len(proto) != TP_FIR_PROTOTYPE_TAPS:
        raise ValueError(f"Expected {TP_FIR_PROTOTYPE_TAPS} TP FIR coeffs, got {len(proto)}")
    weight = torch.empty((TP_OVERSAMPLE_FACTOR, 1, TP_FIR_TAPS_PER_PHASE), dtype=torch.float64)
    for phi in range(TP_OVERSAMPLE_FACTOR):
        taps = proto[phi::TP_OVERSAMPLE_FACTOR]
        # Causal conv1d: F.conv1d computes cross-correlation sum(w[j] * x[n - (K-1) + j]),
        # so reverse phase taps to evaluate sum(h_p[k] * u[n - k]).
        weight[phi, 0, :] = torch.tensor(list(reversed(taps)), dtype=torch.float64)
    return weight


class FastLimiterReleaseFunction(torch.autograd.Function):
    """
    Funzione autograd C++ per scansione ricorrente ad altissima velocità del release envelope del limiter.
    """
    @staticmethod
    def forward(ctx, g_min: torch.Tensor, alpha_rel: float) -> torch.Tensor:
        B, N = g_min.shape
        if _HAS_PY_CORE_DSP and not g_min.is_cuda:
            gm_np = g_min.detach().cpu().numpy()
            grel_np, mask_np = py_core_dsp.fast_forward_limiter_envelope(gm_np, float(alpha_rel))
            g_rel = torch.from_numpy(grel_np).to(device=g_min.device, dtype=g_min.dtype)
            mask_rel = torch.from_numpy(mask_np).to(device=g_min.device, dtype=torch.bool)
        else:
            g_rel = torch.empty_like(g_min)
            mask_rel = torch.empty((B, N), dtype=torch.bool, device=g_min.device)
            g_prev = torch.ones(B, dtype=torch.float64, device=g_min.device)
            for n in range(N):
                gm = g_min[:, n]
                is_rel = gm > g_prev
                mask_rel[:, n] = is_rel
                g_curr = torch.where(is_rel, alpha_rel * g_prev + (1.0 - alpha_rel) * gm, gm)
                g_rel[:, n] = g_curr
                g_prev = g_curr

        ctx.save_for_backward(mask_rel)
        ctx.alpha_rel = alpha_rel
        return g_rel

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (mask_rel,) = ctx.saved_tensors
        alpha_rel = ctx.alpha_rel
        B, N = grad_output.shape

        if _HAS_PY_CORE_DSP and not grad_output.is_cuda:
            dg_np = grad_output.detach().cpu().numpy()
            mask_np = mask_rel.detach().cpu().numpy()
            dgmin_np = py_core_dsp.fast_backward_limiter_envelope(dg_np, mask_np, float(alpha_rel))
            d_gmin = torch.from_numpy(dgmin_np).to(device=grad_output.device, dtype=grad_output.dtype)
        else:
            d_gmin = torch.empty_like(grad_output)
            lambda_next = torch.zeros(B, dtype=grad_output.dtype, device=grad_output.device)
            for n in range(N - 1, -1, -1):
                dg_curr = grad_output[:, n] + lambda_next
                is_rel = mask_rel[:, n]
                d_gmin[:, n] = torch.where(is_rel, (1.0 - alpha_rel) * dg_curr, dg_curr)
                lambda_next = torch.where(is_rel, alpha_rel * dg_curr, torch.zeros_like(lambda_next))

        return d_gmin, None


class DifferentiableLookaheadLimiter(nn.Module):
    """
    Lookahead Brickwall Limiter Differenziabile.
    Supporta:
      - 'true_peak_4x' (default Sprint 4): Detector polifase FIR 4x causale stereo-linked + 1x allineato,
        ceiling interno -0.50 dBTP, boxcar 136 campioni privo di release leak su finestra NL=144.
      - 'legacy_sample_1x': Modalità storica 1x sample-peak a -0.30 dBFS e boxcar 144 campioni
        per riproducibilità esplicita dei render precedenti.
    """
    def __init__(
        self,
        sample_rate: float = 48000.0,
        detector_mode: str = "true_peak_4x",
        ceiling_db: Optional[float] = None,
    ):
        super().__init__()
        if detector_mode not in ("true_peak_4x", "legacy_sample_1x"):
            raise ValueError(
                f"Unsupported detector_mode='{detector_mode}' (expected 'true_peak_4x' or 'legacy_sample_1x')"
            )
        self.sample_rate = float(sample_rate)
        self.detector_mode = detector_mode
        self.lookahead = LOOKAHEAD_SAMPLES
        if detector_mode == "legacy_sample_1x":
            self.ceiling_db = float(CEILING_DBFS if ceiling_db is None else ceiling_db)
            self.boxcar_samples = LOOKAHEAD_SAMPLES
        else:
            self.ceiling_db = float(INTERNAL_CEILING_DBTP if ceiling_db is None else ceiling_db)
            self.boxcar_samples = TP_BOXCAR_SAMPLES
        self.c_lin = 10.0 ** (self.ceiling_db / 20.0)
        tau_sec = RELEASE_MS * 0.001
        self.alpha_rel = math.exp(-1.0 / (tau_sec * self.sample_rate))
        self.register_buffer("tp_polyphase_weight", _build_polyphase_conv_weight(), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        push_gain_db: torch.Tensor = None,
        release_ms: float = None,
        return_gain: bool = False,
    ):
        """
        Args:
            x: [..., 2, N] Segnale audio stereo in ingresso
            push_gain_db: [..., 1] o [..., ] Guadagno di push in dB [0.0, 12.0]
            release_ms: Costante di tempo del release in ms (opzionale, default 80.0 ms)
            return_gain: Se True restituisce (y_out, {"g_smooth": g_smooth, "a_lim_db": a_lim_db})
        Returns:
            y_out: [..., 2, N] Segnale limitato, ritardato di esattamente 144 campioni
        """
        orig_shape = x.shape
        orig_dtype = x.dtype
        N = x.shape[-1]
        x_f64 = x.to(torch.float64).reshape(-1, 2, N)
        B = x_f64.shape[0]

        # 1. Gain Push in float64
        if push_gain_db is None:
            push_db = torch.zeros((B, 1, 1), dtype=torch.float64, device=x.device)
        else:
            push_db = push_gain_db.reshape(B, 1, 1).to(torch.float64)
        push_lin = torch.pow(10.0, push_db / 20.0)
        u = x_f64 * push_lin # [B, 2, N]

        # 2. Peak Detection & Target Gain Istantaneo in float64
        if self.detector_mode == "legacy_sample_1x":
            peak = torch.clamp(
                torch.max(torch.abs(u[:, 0, :]), torch.abs(u[:, 1, :])),
                min=1e-7
            ) # [B, N]
        else:
            # 4 rami polifase causali da 12 tap per canale: [B*2, 1, N+11] -> [B*2, 4, N]
            u_flat = u.reshape(B * 2, 1, N)
            u_pad = F.pad(u_flat, (TP_FIR_TAPS_PER_PHASE - 1, 0), mode="constant", value=0.0)
            w = self.tp_polyphase_weight.to(device=u.device, dtype=torch.float64)
            tp_phases = F.conv1d(u_pad, w, stride=1).reshape(B, 2, TP_OVERSAMPLE_FACTOR, N)

            # Include il sample peak 1x allineato al centro del filtro FIR (~5.875 -> 6 campioni)
            u_aligned = F.pad(u, (TP_FIR_ALIGNED_SAMPLE_DELAY, 0), mode="constant", value=0.0)[..., :N].unsqueeze(2)
            all_peaks = torch.cat([tp_phases, u_aligned], dim=2) # [B, 2, 5, N]
            peak = torch.clamp(all_peaks.abs().amax(dim=(1, 2)), min=1e-7) # [B, N]

        g_target = torch.clamp(self.c_lin / peak, max=1.0) # [B, N]

        # 3. Stadio 1: Min-Filter su finestra NL=144 via -max_pool1d(-padded)
        g_target_padded = F.pad(g_target.unsqueeze(1), (self.lookahead - 1, 0), mode='constant', value=1.0)
        g_min = -F.max_pool1d(-g_target_padded, kernel_size=self.lookahead, stride=1).squeeze(1) # [B, N]

        # 4. Stadio 2: One-Pole Release Envelope (C++ Fast Autograd Scan)
        if release_ms is not None:
            tau_sec = float(release_ms) * 0.001
            alpha_rel = math.exp(-1.0 / (tau_sec * self.sample_rate))
        else:
            alpha_rel = self.alpha_rel
        g_rel = FastLimiterReleaseFunction.apply(g_min, alpha_rel)

        # 5. Stadio 3: Boxcar Moving Average Smoothing (136 campioni TruePeak4x, 144 campioni Legacy)
        g_rel_padded = F.pad(g_rel.unsqueeze(1), (self.boxcar_samples - 1, 0), mode='constant', value=1.0)
        g_smooth = F.avg_pool1d(g_rel_padded, kernel_size=self.boxcar_samples, stride=1).squeeze(1) # [B, N]

        # 6. Audio ritardato di NL=144 campioni e applicazione guadagno
        u_delayed = F.pad(u, (self.lookahead, 0))[..., :N] # [B, 2, N]
        y_out = u_delayed * g_smooth.unsqueeze(1) # [B, 2, N]

        y_out_res = y_out.reshape(orig_shape).to(orig_dtype)
        if return_gain:
            batch_shape = orig_shape[:-2]
            g_smooth_2d = g_smooth.reshape(batch_shape + (N,)).to(orig_dtype)
            a_lim_db = -20.0 * torch.log10(torch.clamp(g_smooth_2d, min=1e-12))
            return y_out_res, {"g_smooth": g_smooth_2d, "a_lim_db": a_lim_db}

        return y_out_res
