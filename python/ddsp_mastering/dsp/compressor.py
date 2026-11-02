import math
import torch
import torch.nn as nn
from typing import Tuple

try:
    import py_core_dsp
    _HAS_PY_CORE_DSP = True
except ImportError:
    _HAS_PY_CORE_DSP = False

EPSILON = 1e-12
LN10 = 2.30258509299404568402
C_LN = LN10 / 20.0 # ~0.11512925464970228
KNEE_DB = 6.0
HALF_KNEE = 3.0
INV_TWO_KNEE = 1.0 / 12.0

class DifferentiableSingleBandCompressorFunction(torch.autograd.Function):
    """
    Funzione autograd custom per compressore con ballistics ricorsive nel dominio logaritmico.
    Calcola il forward e il backward pass analitico esatto senza loop autograd lenti.
    """
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        threshold_db: torch.Tensor,
        ratio: torch.Tensor,
        attack_ms: torch.Tensor,
        release_ms: torch.Tensor,
        makeup_db: torch.Tensor,
        sample_rate: float = 48000.0,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f64 = x.to(torch.float64) # [..., 2, N]
        N = x_f64.shape[-1]
        
        # Flatten eventuale batch: [B, 2, N]
        batch_shape = x_f64.shape[:-2]
        x_flat = x_f64.reshape(-1, 2, N)
        B = x_flat.shape[0]

        T = threshold_db.to(torch.float64).reshape(-1)
        R = torch.clamp(ratio.to(torch.float64).reshape(-1), min=1.0)
        tau_a = torch.clamp(attack_ms.to(torch.float64).reshape(-1), min=0.1) * 0.001 # sec
        tau_r = torch.clamp(release_ms.to(torch.float64).reshape(-1), min=1.0) * 0.001 # sec
        M = makeup_db.to(torch.float64).reshape(-1)

        alpha_a = torch.exp(-1.0 / (tau_a * sample_rate))
        alpha_r = torch.exp(-1.0 / (tau_r * sample_rate))

        # 1. Stereo-linked energy detection in dB
        # energy = 0.5 * (L^2 + R^2) + EPSILON
        energy = 0.5 * (x_flat[:, 0, :] ** 2 + x_flat[:, 1, :] ** 2) + EPSILON
        x_db = 10.0 * torch.log10(energy) # [B, N]

        # 2. C1 Soft-Knee Gain Computer
        # delta = x_db - T
        delta = x_db - T.unsqueeze(-1) # [B, N]
        ratio_factor = (1.0 - 1.0 / R).unsqueeze(-1) # [B, 1]

        # gc piecewise
        gc = torch.zeros_like(delta)
        mask_below = (2.0 * delta) <= -KNEE_DB
        mask_knee = torch.abs(2.0 * delta) < KNEE_DB
        mask_above = (2.0 * delta) >= KNEE_DB

        gc[mask_knee] = ratio_factor.expand_as(delta)[mask_knee] * ((delta[mask_knee] + HALF_KNEE) ** 2) * INV_TWO_KNEE
        gc[mask_above] = ratio_factor.expand_as(delta)[mask_above] * delta[mask_above]

        # 3. Ballistics ricorsiva log-domain
        # y_g[n] = alpha[n] * y_g[n-1] + (1 - alpha[n]) * gc[n]
        can_use_fast_comp = _HAS_PY_CORE_DSP and not x_flat.is_cuda and (
            B == 1 or (torch.allclose(alpha_a, alpha_a[0]) and torch.allclose(alpha_r, alpha_r[0]))
        )
        if can_use_fast_comp:
            gc_np = gc.detach().cpu().numpy()
            yg_np, mask_np = py_core_dsp.fast_forward_ballistics(gc_np, float(alpha_a[0]), float(alpha_r[0]))
            yg = torch.from_numpy(yg_np).to(device=x.device, dtype=torch.float64)
            mask_attack = torch.from_numpy(mask_np).to(device=x.device, dtype=torch.bool)
        else:
            yg = torch.zeros((B, N), dtype=torch.float64, device=x.device)
            mask_attack = torch.zeros((B, N), dtype=torch.bool, device=x.device)
            yg_prev = torch.zeros(B, dtype=torch.float64, device=x.device)
            for n in range(N):
                gc_n = gc[:, n]
                m_n = gc_n > yg_prev
                mask_attack[:, n] = m_n
                alpha_n = torch.where(m_n, alpha_a, alpha_r)
                yg_curr = alpha_n * yg_prev + (1.0 - alpha_n) * gc_n
                yg[:, n] = yg_curr
                yg_prev = yg_curr

        # 4. Guadagno lineare applicato + Makeup
        glin = torch.exp(C_LN * (M.unsqueeze(-1) - yg)) # [B, N]
        y_out = x_flat * glin.unsqueeze(1) # [B, 2, N]

        # Salvataggio contesto per backward
        ctx.save_for_backward(x_flat, T, R, tau_a, tau_r, M, alpha_a, alpha_r, energy, delta, gc, yg, mask_attack, glin)
        ctx.sample_rate = sample_rate
        ctx.batch_shape = batch_shape
        ctx.t_shape = threshold_db.shape
        ctx.t_dtype = threshold_db.dtype
        ctx.r_shape = ratio.shape
        ctx.r_dtype = ratio.dtype
        ctx.att_shape = attack_ms.shape
        ctx.att_dtype = attack_ms.dtype
        ctx.rel_shape = release_ms.shape
        ctx.rel_dtype = release_ms.dtype
        ctx.m_shape = makeup_db.shape
        ctx.m_dtype = makeup_db.dtype

        return y_out.reshape(x.shape).to(orig_dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        (x_flat, T, R, tau_a, tau_r, M, alpha_a, alpha_r, energy, delta, gc, yg, mask_attack, glin) = ctx.saved_tensors
        sample_rate = ctx.sample_rate
        B, _, N = x_flat.shape

        grad_out = grad_output.to(torch.float64).reshape(B, 2, N)

        # 1. Gradiente rispetto a glin e contributo diretto a x
        # y_out = x * glin -> dL/dglin = sum_c (grad_out_c * x_c)
        d_glin = grad_out[:, 0, :] * x_flat[:, 0, :] + grad_out[:, 1, :] * x_flat[:, 1, :] # [B, N]
        d_x_direct = grad_out * glin.unsqueeze(1) # [B, 2, N]

        # 2. Gradiente rispetto a M (makeup) e yg (envelope)
        # glin = exp(C_LN * (M - yg))
        d_exponent = d_glin * glin * C_LN # [B, N]
        d_M = torch.sum(d_exponent, dim=-1) # [B]
        d_yg_direct = -d_exponent # [B, N]

        # 3. Scansione all'indietro per ballistics (psi[n])
        can_use_fast_comp = _HAS_PY_CORE_DSP and not x_flat.is_cuda and (
            B == 1 or (torch.allclose(alpha_a, alpha_a[0]) and torch.allclose(alpha_r, alpha_r[0]))
        )
        if can_use_fast_comp:
            dyg_np = d_yg_direct.detach().cpu().numpy()
            mask_np = mask_attack.detach().cpu().numpy()
            psi_np = py_core_dsp.fast_backward_psi(dyg_np, mask_np, float(alpha_a[0]), float(alpha_r[0]))
            psi = torch.from_numpy(psi_np).to(device=x_flat.device, dtype=torch.float64)
        else:
            psi = torch.zeros((B, N + 1), dtype=torch.float64, device=x_flat.device)
            for n in range(N - 1, -1, -1):
                if n + 1 < N:
                    m_next = mask_attack[:, n + 1]
                    alpha_next = torch.where(m_next, alpha_a, alpha_r)
                    psi[:, n] = d_yg_direct[:, n] + alpha_next * psi[:, n + 1]
                else:
                    psi[:, n] = d_yg_direct[:, n]

        # 4. Derivate dL/dgc, dL/dalpha_a, dL/dalpha_r (Vettorizzate)
        alpha = torch.where(mask_attack, alpha_a.unsqueeze(1), alpha_r.unsqueeze(1))
        psi_n = psi[:, :N]
        d_gc = (1.0 - alpha) * psi_n

        yg_shifted = torch.cat([torch.zeros((B, 1), dtype=torch.float64, device=x_flat.device), yg[:, :-1]], dim=-1)
        diff_yg_gc = yg_shifted - gc
        term = psi_n * diff_yg_gc
        d_alpha_a = torch.sum(torch.where(mask_attack, term, 0.0), dim=-1)
        d_alpha_r = torch.sum(torch.where(~mask_attack, term, 0.0), dim=-1)

        # Derivate verso costanti di tempo (tau_a, tau_r in ms)
        # alpha = exp(-1 / (tau_sec * fs)) -> dalpha/dtau_sec = alpha / (tau_sec^2 * fs)
        # dtau_ms = dtau_sec * 0.001
        d_tau_a_ms = d_alpha_a * (alpha_a / (tau_a ** 2 * sample_rate)) * 0.001
        d_tau_r_ms = d_alpha_r * (alpha_r / (tau_r ** 2 * sample_rate)) * 0.001

        # 5. Derivate attraverso il Gain Computer (gc -> delta, R, T)
        d_delta = torch.zeros_like(delta)
        d_R = torch.zeros(B, dtype=torch.float64, device=x_flat.device)

        mask_knee = torch.abs(2.0 * delta) < KNEE_DB
        mask_above = (2.0 * delta) >= KNEE_DB

        # Ramo knee: gc = (1 - 1/R) * (delta + W/2)^2 / (2W)
        if mask_knee.any():
            ratio_factor = (1.0 - 1.0 / R).unsqueeze(-1)
            d_delta_knee = ratio_factor * (delta + HALF_KNEE) / KNEE_DB
            d_delta[mask_knee] = (d_gc * d_delta_knee)[mask_knee]

            inv_R2 = (1.0 / (R ** 2)).unsqueeze(-1)
            term_R_knee = d_gc * inv_R2 * ((delta + HALF_KNEE) ** 2) * INV_TWO_KNEE
            d_R += torch.sum(torch.where(mask_knee, term_R_knee, 0.0), dim=-1)

        # Ramo above: gc = (1 - 1/R) * delta
        if mask_above.any():
            ratio_factor = (1.0 - 1.0 / R).unsqueeze(-1)
            d_delta[mask_above] = (d_gc * ratio_factor)[mask_above]

            inv_R2 = (1.0 / (R ** 2)).unsqueeze(-1)
            term_R_above = d_gc * inv_R2 * delta
            d_R += torch.sum(torch.where(mask_above, term_R_above, 0.0), dim=-1)

        # dL/dT = - sum_n (dL/ddelta)
        d_T = -torch.sum(d_delta, dim=-1)

        # 6. Derivate attraverso l'energy detection (delta -> x_db -> energy -> x)
        # x_db = (10 / ln(10)) * ln(energy) -> dx_db/denergy = 10 / (ln(10) * energy)
        # denergy/dx_c = x_c
        d_x_detection_factor = d_delta * (10.0 / (LN10 * energy)) # [B, N]
        d_x_detection = x_flat * d_x_detection_factor.unsqueeze(1) # [B, 2, N]

        d_x_total = (d_x_direct + d_x_detection).reshape(ctx.batch_shape + (2, N))

        return (
            d_x_total.to(grad_output.dtype),
            d_T.reshape(ctx.t_shape).to(ctx.t_dtype),
            d_R.reshape(ctx.r_shape).to(ctx.r_dtype),
            d_tau_a_ms.reshape(ctx.att_shape).to(ctx.att_dtype),
            d_tau_r_ms.reshape(ctx.rel_shape).to(ctx.rel_dtype),
            d_M.reshape(ctx.m_shape).to(ctx.m_dtype),
            None, # sample_rate
        )


class DifferentiableMultibandCompressor(nn.Module):
    """
    Compressore Dinamico a 3 Bande Differenziabile (Low, Mid, High).
    Isomorfo al C++ MultibandCompressor3Band.
    """
    def __init__(self, sample_rate: float = 48000.0):
        super().__init__()
        self.sample_rate = float(sample_rate)

    def forward(
        self,
        low: torch.Tensor,
        mid: torch.Tensor,
        high: torch.Tensor,
        threshold_db: torch.Tensor, # [..., 3]
        ratio: torch.Tensor,        # [..., 3]
        attack_ms: torch.Tensor,    # [..., 3]
        release_ms: torch.Tensor,   # [..., 3]
        makeup_db: torch.Tensor,    # [..., 3]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Elabora le 3 bande in parallelo con parametri differenziabili.
        """
        out_low = DifferentiableSingleBandCompressorFunction.apply(
            low, threshold_db[..., 0], ratio[..., 0], attack_ms[..., 0], release_ms[..., 0], makeup_db[..., 0], self.sample_rate
        )
        out_mid = DifferentiableSingleBandCompressorFunction.apply(
            mid, threshold_db[..., 1], ratio[..., 1], attack_ms[..., 1], release_ms[..., 1], makeup_db[..., 1], self.sample_rate
        )
        out_high = DifferentiableSingleBandCompressorFunction.apply(
            high, threshold_db[..., 2], ratio[..., 2], attack_ms[..., 2], release_ms[..., 2], makeup_db[..., 2], self.sample_rate
        )
        return out_low, out_mid, out_high
