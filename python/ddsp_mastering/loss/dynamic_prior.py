"""Differentiable Dynamic Prior Loss for DDSP Mastering (Sprint 3 V3b).

Implements:
1. Exact temporal alignment (D_tot = 159 samples = 15 oversampled FIR + 144 limiter lookahead).
2. Active sample mask A_smp from non-overlapping 480-sample (10 ms) blocks with partial remainder handling.
3. L_lim: Penalizes excessive average attenuation and CVaR_0.95 of the lookahead limiter.
4. L_diff: Penalizes excessive gain reduction disparity across adjacent compressor bands.
5. L_conflict: Penalizes compressor makeup gain driving the lookahead limiter into clipping/suppression.
"""

import math
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn

TOTAL_DSP_DELAY_SAMPLES = 159  # 15 samples (2x oversampling FIR in saturator) + 144 samples (limiter lookahead)


def compute_active_sample_mask(
    mix_eval: torch.Tensor,
    block_size: int = 480,
    silence_threshold_db: float = -70.0,
    min_active_blocks: int = 8,
    dynamic_range_db: float = 35.0,
    min_threshold_db: float = -60.0,
) -> Tuple[torch.Tensor, bool, Dict[str, Any]]:
    """Compute boolean sample-level active mask A_smp(x_mix^eval).

    Args:
        mix_eval: Tensor of shape (..., 2, T_eval) or (..., T_eval)
        block_size: Block size in samples (default: 480 = 10 ms at 48 kHz)
        silence_threshold_db: Absolute silence threshold in dBFS (default: -70.0)
        min_active_blocks: Minimum active blocks fallback M_min (default: 8)
        dynamic_range_db: Dynamic range below peak block in dB (default: 35.0)
        min_threshold_db: Floor threshold for block activity in dBFS (default: -60.0)

    Returns:
        mask_smp: Boolean tensor of shape (..., T_eval), detached from autograd graph.
        is_silent: Boolean indicating if maximum block level is below silence threshold.
        meta: Dictionary of diagnostic metadata.
    """
    T_eval = mix_eval.shape[-1]

    # Compute instantaneous stereo power
    if mix_eval.ndim >= 2 and mix_eval.shape[-2] == 2:
        power = 0.5 * (mix_eval[..., 0, :] ** 2 + mix_eval[..., 1, :] ** 2)
    else:
        power = mix_eval ** 2
        if power.ndim > 1 and mix_eval.shape[-2] != 1:
            power = power.mean(dim=-2)

    # Decompose into full and remainder blocks
    n_full = T_eval // block_size
    n_rem = T_eval % block_size

    rms_list = []
    if n_full > 0:
        p_full = power[..., :n_full * block_size].reshape(*power.shape[:-1], n_full, block_size)
        rms_full = torch.mean(p_full, dim=-1)
        rms_list.append(rms_full)

    if n_rem > 0:
        p_rem = power[..., n_full * block_size:]
        rms_rem = torch.mean(p_rem, dim=-1, keepdim=True)
        rms_list.append(rms_rem)

    rms_blocks = torch.cat(rms_list, dim=-1)
    m_blocks = rms_blocks.shape[-1]

    l_blk = 10.0 * torch.log10(torch.clamp(rms_blocks, min=1e-12))
    l_blk_max = torch.max(l_blk)

    if l_blk_max < silence_threshold_db:
        mask_smp = torch.zeros(power.shape[:-1] + (T_eval,), dtype=torch.bool, device=mix_eval.device)
        return mask_smp, True, {
            "l_blk_max": float(l_blk_max.item()),
            "theta_blk": float(l_blk_max.item()),
            "m_active": 0,
            "m_total": m_blocks,
        }

    theta_blk = torch.clamp(l_blk_max - dynamic_range_db, min=min_threshold_db)
    mask_blk = (l_blk >= theta_blk)
    m_active = int(torch.sum(mask_blk).item())

    if m_active < min_active_blocks:
        k_top = min(m_blocks, min_active_blocks)
        _, topk_idx = torch.topk(l_blk, k=k_top, dim=-1)
        mask_blk = torch.zeros_like(l_blk, dtype=torch.bool)
        mask_blk.scatter_(-1, topk_idx, True)
        m_active = int(torch.sum(mask_blk).item())

    # Expand to sample-level mask
    mask_parts = []
    if n_full > 0:
        mask_full_blocks = mask_blk[..., :n_full]
        mask_full_smp = torch.repeat_interleave(mask_full_blocks, block_size, dim=-1)
        mask_parts.append(mask_full_smp)
    if n_rem > 0:
        mask_rem_blocks = mask_blk[..., n_full:]
        mask_rem_smp = torch.repeat_interleave(mask_rem_blocks, n_rem, dim=-1)
        mask_parts.append(mask_rem_smp)

    mask_smp = torch.cat(mask_parts, dim=-1).detach()
    meta = {
        "l_blk_max": float(l_blk_max.item()),
        "theta_blk": float(theta_blk.item()),
        "m_active": m_active,
        "m_total": m_blocks,
    }
    return mask_smp, False, meta


class V3bDynamicPriorLoss(nn.Module):
    """V3b Differentiable Dynamic Prior Loss."""

    def __init__(
        self,
        tau_avg: float = 1.5,
        tau_95: float = 3.5,
        delta_max: float = 4.0,
        delta_ref: float = 2.0,
        delta_m: float = 1.25,
        lambda_95: float = 1.0,
        w_lim: float = 0.05,
        w_diff: float = 0.02,
        w_conflict: float = 0.05,
        sample_rate: float = 48000.0,
        block_size: int = 480,
        silence_threshold_db: float = -70.0,
        min_active_blocks: int = 8,
        delay_dsp_samples: int = TOTAL_DSP_DELAY_SAMPLES,
    ):
        super().__init__()
        self.tau_avg = float(tau_avg)
        self.tau_95 = float(tau_95)
        self.delta_max = float(delta_max)
        self.delta_ref = float(delta_ref)
        self.delta_m = float(delta_m)
        self.lambda_95 = float(lambda_95)
        self.w_lim = float(w_lim)
        self.w_diff = float(w_diff)
        self.w_conflict = float(w_conflict)
        self.sample_rate = float(sample_rate)
        self.block_size = int(block_size)
        self.silence_threshold_db = float(silence_threshold_db)
        self.min_active_blocks = int(min_active_blocks)
        self.delay_dsp_samples = int(delay_dsp_samples)

    def forward(
        self,
        mix_ext: torch.Tensor,
        diagnostics: Dict[str, torch.Tensor],
        makeup_db: torch.Tensor,
        n_pre: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute aligned dynamic prior losses over common evaluation window T_eval = T - 159.

        Args:
            mix_ext: Audio input before mastering chain [..., 2, N_pre + T]
            diagnostics: Trace dictionary from chain with 'gr_low_db', 'gr_mid_db',
                         'gr_high_db', and 'lim_atten_db'.
            makeup_db: Multiband compressor makeup gains [..., 3]
            n_pre: Pre-roll sample count

        Returns:
            loss_dyn: Scalar weighted dynamic prior loss tensor.
            metrics: Dictionary containing individual losses, statistics, and flags.
        """
        N_tot = mix_ext.shape[-1]
        T = N_tot - n_pre
        D = self.delay_dsp_samples
        if T <= D:
            raise ValueError(f"Chunk length T={T} must exceed total delay D={D}")

        T_eval = T - D

        # Temporal alignment and pre-roll slicing
        x_mix_eval = mix_ext[..., n_pre : n_pre + T_eval]
        gr_low_eval = diagnostics["gr_low_db"][..., n_pre : n_pre + T_eval]
        gr_mid_eval = diagnostics["gr_mid_db"][..., n_pre : n_pre + T_eval]
        gr_high_eval = diagnostics["gr_high_db"][..., n_pre : n_pre + T_eval]
        a_lim_eval = diagnostics["lim_atten_db"][..., n_pre + D : n_pre + T]

        # Active sample mask
        mask_smp, is_silent, meta = compute_active_sample_mask(
            x_mix_eval,
            block_size=self.block_size,
            silence_threshold_db=self.silence_threshold_db,
            min_active_blocks=self.min_active_blocks,
        )

        dummy_zero = 0.0 * (mix_ext.sum() + makeup_db.sum() + a_lim_eval.sum() + gr_low_eval.sum())

        if is_silent:
            metrics = {
                "loss_lim": 0.0,
                "loss_diff": 0.0,
                "loss_conflict": 0.0,
                "loss_dyn": 0.0,
                "bar_a_lim_db": 0.0,
                "p95_a_lim_db": 0.0,
                "cvar95_a_lim_db": 0.0,
                "max_a_lim_db": 0.0,
                "bar_gr_low_db": 0.0,
                "bar_gr_mid_db": 0.0,
                "bar_gr_high_db": 0.0,
                "diff_lm_max": 0.0,
                "diff_mh_max": 0.0,
                "h_lim": 0.0,
                "is_silent_chunk": True,
                "n_active_samples": 0,
                "active_ratio": 0.0,
            }
            return dummy_zero, metrics

        n_act = float(mask_smp.sum().item())
        act_mask = mask_smp

        # 1. L_lim: mean and CVaR_0.95 of limiter attenuation on active samples
        a_lim_act = a_lim_eval[act_mask]
        bar_a_lim = torch.mean(a_lim_act)

        k_tail = max(1, int(math.ceil(0.05 * n_act)))
        topk_vals, _ = torch.topk(a_lim_act, k=k_tail, dim=-1)
        cvar_95 = torch.mean(topk_vals)

        term_avg = torch.relu(bar_a_lim - self.tau_avg) ** 2
        term_95 = torch.relu(cvar_95 - self.tau_95) ** 2
        loss_lim = term_avg + self.lambda_95 * term_95

        # 2. L_diff: adjacent band compressor disparity
        diff_lm = torch.abs(gr_low_eval - gr_mid_eval)[act_mask]
        diff_mh = torch.abs(gr_mid_eval - gr_high_eval)[act_mask]
        term_lm = torch.relu(diff_lm - self.delta_max) ** 2
        term_mh = torch.relu(diff_mh - self.delta_max) ** 2
        loss_diff = torch.mean(term_lm + term_mh)

        # 3. L_conflict: compressor makeup gain driving limiter
        r = torch.relu(bar_a_lim - self.tau_avg)
        h_lim = (r / (r + self.delta_ref)).detach()

        bar_gr_low = torch.mean(gr_low_eval[act_mask])
        bar_gr_mid = torch.mean(gr_mid_eval[act_mask])
        bar_gr_high = torch.mean(gr_high_eval[act_mask])

        m_l = makeup_db[..., 0]
        m_m = makeup_db[..., 1]
        m_h = makeup_db[..., 2]

        conf_l = torch.relu(m_l - bar_gr_low.detach() - self.delta_m) ** 2
        conf_m = torch.relu(m_m - bar_gr_mid.detach() - self.delta_m) ** 2
        conf_h = torch.relu(m_h - bar_gr_high.detach() - self.delta_m) ** 2
        loss_conflict = h_lim * (1.0 / 3.0) * (conf_l + conf_m + conf_h)

        loss_dyn = self.w_lim * loss_lim + self.w_diff * loss_diff + self.w_conflict * loss_conflict

        # Diagnostic metadata
        with torch.no_grad():
            p95_val = float(torch.quantile(a_lim_act.float(), 0.95).item()) if a_lim_act.numel() > 0 else 0.0
            max_a_lim_val = float(torch.max(a_lim_act).item()) if a_lim_act.numel() > 0 else 0.0
            diff_lm_max_val = float(torch.max(diff_lm).item()) if diff_lm.numel() > 0 else 0.0
            diff_mh_max_val = float(torch.max(diff_mh).item()) if diff_mh.numel() > 0 else 0.0

        metrics = {
            "loss_lim": float(loss_lim.item()),
            "loss_diff": float(loss_diff.item()),
            "loss_conflict": float(loss_conflict.item()),
            "loss_dyn": float(loss_dyn.item()),
            "bar_a_lim_db": float(bar_a_lim.item()),
            "p95_a_lim_db": p95_val,
            "cvar95_a_lim_db": float(cvar_95.item()),
            "max_a_lim_db": max_a_lim_val,
            "bar_gr_low_db": float(bar_gr_low.item()),
            "bar_gr_mid_db": float(bar_gr_mid.item()),
            "bar_gr_high_db": float(bar_gr_high.item()),
            "diff_lm_max": diff_lm_max_val,
            "diff_mh_max": diff_mh_max_val,
            "h_lim": float(h_lim.item()),
            "is_silent_chunk": False,
            "n_active_samples": int(n_act),
            "active_ratio": float(n_act / T_eval),
        }

        return loss_dyn, metrics
