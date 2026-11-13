"""Differentiable Stereo & Mono Compatibility Prior Loss for DDSP Mastering (Sprint 3 V3c).

Standalone module implementing the frozen pre-implementation design (Q1-Q5):
1. Fixed non-trainable DifferentiableLR4Crossover(sample_rate=48000.0) at 160.0 Hz / 3200.0 Hz
   always applied to full signals including pre-roll BEFORE latency/eval window slicing:
   - mix_eval slice:  [n_pre : n_pre + T - delay]
   - pred_eval slice: [n_pre + delay : n_pre + T]
2. Active sample mask A_smp(x_mix^eval) with 480-sample blocks (10 ms at 48 kHz),
   -70.0 dBFS silence threshold, M_min=8 fallback, and partial remainder block handling.
3. Minimum per-band energy validation E_min = 1e-7 (-70.0 dBFS): evaluation validity is
   determined EXCLUSIVELY by the original mix energy (E_tot_mix >= E_min). When mix is active,
   a silent or sub-threshold prediction stays valid, with pred's eps_rel anchored to
   detach(max(E_tot_mix, E_tot_pred)) * 1e-12 + 1e-30 for finiteness, continuity, and scale invariance.
4. Explicit handling of single-channel silent / hard-panned bands (min(E_L, E_R) < 1e-6 * E_tot).
5. Relative detached epsilon floor eps_rel = 1e-12 * detach(E_anchor) + 1e-30 for quantitative
   scale invariance across [-60 dBFS, +20 dBFS].
6. Frozen Candidate A thresholds & same_track pilot weights:
   - Candidate A: tau_drop_db=0.50, tau_corr=(0.03, 0.05, 0.08), tau_smr_db=(2.0, 2.0, 2.0),
     tau_sub_db=0.50, alpha_sub=1.0, tau_w_low_max=1.00, tau_w_mid=0.15, tau_w_high=0.15
   - Pilot same_track weights: w_mono_drop=1.00, w_band_corr=10.00, w_band_smr=0.05, w_width_param=0.25
7. Diagnostic-only metrics (never entering the loss):
   - L/R balance (dB)
   - Multi-window short-time local correlation (50 ms, 100 ms, 250 ms) with mean, min, P05, CVaR05
   - Spectral anti-comb-filtering diagnostics (fold-down ripple & notch depth)
"""

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ddsp_mastering.dsp.crossover import DifferentiableLR4Crossover
from ddsp_mastering.loss.dynamic_prior import TOTAL_DSP_DELAY_SAMPLES, compute_active_sample_mask
from ddsp_mastering.param_space import map_z_to_theta

INV_SQRT2 = 0.70710678118654752440
FIXED_CROSSOVER_LOW_HZ = 160.0
FIXED_CROSSOVER_HIGH_HZ = 3200.0
MIN_ACTIVE_BAND_ENERGY = 1e-7  # -70.0 dBFS threshold for E_tot = E_L + E_R
SINGLE_CHANNEL_ENERGY_RATIO = 1e-6  # eta_ch threshold for hard-panned / silent channel
REL_EPS_SCALE = 1e-12
ABS_EPS_FLOOR = 1e-30
SHORT_TIME_WINDOWS_MS: Tuple[int, ...] = (50, 100, 250)
BAND_NAMES: Tuple[str, ...] = ("low", "mid", "high")

# Frozen Candidate A thresholds & pilot same_track weights (Step V3c Hardening)
CANDIDATE_A_THRESHOLDS: Dict[str, Any] = {
    "tau_drop_db": 0.50,
    "tau_corr": (0.03, 0.05, 0.08),
    "tau_smr_db": (2.0, 2.0, 2.0),
    "tau_sub_db": 0.50,
    "alpha_sub": 1.0,
    "tau_w_low_max": 1.00,
    "tau_w_mid": 0.15,
    "tau_w_high": 0.15,
}
PILOT_SAME_TRACK_WEIGHTS: Dict[str, float] = {
    "w_mono_drop": 1.00,
    "w_band_corr": 10.00,
    "w_band_smr": 0.05,
    "w_width_param": 0.25,
}


def _ensure_batch_stereo(x: torch.Tensor) -> torch.Tensor:
    """Ensure shape [B, 2, N] in float64."""
    if x.ndim == 2:
        if x.shape[0] != 2:
            raise ValueError(f"Expected stereo shape (2, N) or (B, 2, N), got {tuple(x.shape)}")
        return x.unsqueeze(0).to(torch.float64)
    if x.ndim == 3:
        if x.shape[1] != 2:
            raise ValueError(f"Expected stereo shape (B, 2, N), got {tuple(x.shape)}")
        return x.to(torch.float64)
    raise ValueError(f"Expected 2D or 3D stereo tensor, got shape {tuple(x.shape)}")


def compute_masked_stereo_stats(
    audio_eval_f64: torch.Tensor,
    mask_smp: torch.Tensor,
    min_energy: float = MIN_ACTIVE_BAND_ENERGY,
    eta_ch: float = SINGLE_CHANNEL_ENERGY_RATIO,
    ref_energy_anchor: Optional[torch.Tensor] = None,
    eps_rel_factor: float = REL_EPS_SCALE,
) -> Dict[str, Any]:
    """Compute exact float64 stereo correlation, mono drop, SMR, and L/R balance on active samples.

    Args:
        audio_eval_f64: Tensor of shape (B, 2, T_eval) in float64.
        mask_smp: Boolean tensor of shape (B, T_eval).
        min_energy: Minimum active E_tot = E_L + E_R for band/broadband validity.
        eta_ch: Threshold on min(E_L, E_R) / E_tot for hard-panned / silent single channel flag.
        ref_energy_anchor: Optional reference mix e_tot tensor [B] used when computing
            predicted signal stats so eps_rel stays anchored to mix energy even if pred -> 0.
        eps_rel_factor: Relative floor multiplier (default 1e-12).

    Returns:
        Dictionary containing differentiable scalar/batch tensors and boolean validity metadata.
    """
    mask_f64 = mask_smp.to(dtype=torch.float64, device=audio_eval_f64.device)
    n_act = torch.clamp(mask_f64.sum(dim=-1), min=1.0)  # [B]

    l_sig = audio_eval_f64[:, 0, :] * mask_f64
    r_sig = audio_eval_f64[:, 1, :] * mask_f64
    m_sig = (l_sig + r_sig) * INV_SQRT2
    s_sig = (l_sig - r_sig) * INV_SQRT2

    e_l = torch.sum(l_sig * l_sig, dim=-1) / n_act
    e_r = torch.sum(r_sig * r_sig, dim=-1) / n_act
    e_m = torch.sum(m_sig * m_sig, dim=-1) / n_act
    e_s = torch.sum(s_sig * s_sig, dim=-1) / n_act
    e_tot = e_l + e_r
    c_lr = torch.sum(l_sig * r_sig, dim=-1) / n_act

    if ref_energy_anchor is not None:
        e_anchor = torch.maximum(e_tot.detach(), ref_energy_anchor.detach()).clamp(min=ABS_EPS_FLOOR)
    else:
        e_anchor = torch.clamp(e_tot.detach(), min=ABS_EPS_FLOOR)
    eps_rel = float(eps_rel_factor) * e_anchor + ABS_EPS_FLOOR

    mono_drop_db = 10.0 * torch.log10((e_m + e_s + eps_rel) / (e_m + eps_rel))
    smr_db = 10.0 * torch.log10((e_s + eps_rel) / (e_m + eps_rel))
    lr_balance_db = 10.0 * torch.log10((e_l + eps_rel) / (e_r + eps_rel))
    corr_lr = c_lr / torch.sqrt((e_l + eps_rel) * (e_r + eps_rel))

    e_tot_val = float(e_tot.min().item())
    e_min_ch = float(torch.minimum(e_l, e_r).min().item())
    has_active = bool((mask_smp.sum(dim=-1) > 0).all().item())
    valid = has_active and (e_tot_val >= min_energy)
    hard_panned = bool(valid and (e_min_ch < eta_ch * max(e_tot_val, ABS_EPS_FLOOR)))

    return {
        "e_l": e_l,
        "e_r": e_r,
        "e_m": e_m,
        "e_s": e_s,
        "e_tot": e_tot,
        "e_tot_tensor": e_tot,
        "c_lr": c_lr,
        "eps_rel": eps_rel,
        "mono_drop_db": mono_drop_db,
        "smr_db": smr_db,
        "lr_balance_db": lr_balance_db,
        "corr_lr": corr_lr,
        "valid": bool(valid),
        "energy_valid": bool(valid),
        "channel_silent_or_hard_panned": bool(hard_panned),
    }


@torch.no_grad()
def compute_short_time_window_stats(
    audio_eval_f64: torch.Tensor,
    mask_smp: torch.Tensor,
    sample_rate: float = 48000.0,
    window_ms: int = 100,
    min_energy: float = MIN_ACTIVE_BAND_ENERGY,
) -> Dict[str, Any]:
    """Compute diagnostic-only short-time correlation stats (mean, min, P05, CVaR05)."""
    B, _, T_eval = audio_eval_f64.shape
    w_len = max(1, int(round(float(window_ms) * float(sample_rate) / 1000.0)))
    hop = max(1, w_len // 2)

    if T_eval < w_len:
        # Evaluate single whole-segment fallback if shorter than requested window
        win_l = audio_eval_f64[:, 0, :].unsqueeze(1)
        win_r = audio_eval_f64[:, 1, :].unsqueeze(1)
        win_mask = mask_smp.to(torch.float64).unsqueeze(1)
    else:
        win_l = audio_eval_f64[:, 0, :].unfold(-1, w_len, hop)
        win_r = audio_eval_f64[:, 1, :].unfold(-1, w_len, hop)
        win_mask = mask_smp.to(torch.float64).unfold(-1, w_len, hop)

    # Active coverage per window
    cov = win_mask.mean(dim=-1)  # [B, n_win]
    n_act_w = torch.clamp(win_mask.sum(dim=-1), min=1.0)

    wl = win_l * win_mask
    wr = win_r * win_mask
    el_w = (wl * wl).sum(dim=-1) / n_act_w
    er_w = (wr * wr).sum(dim=-1) / n_act_w
    etot_w = el_w + er_w
    clr_w = (wl * wr).sum(dim=-1) / n_act_w
    eps_w = REL_EPS_SCALE * etot_w + ABS_EPS_FLOOR
    corr_w = clr_w / torch.sqrt((el_w + eps_w) * (er_w + eps_w))

    # Select windows with >= 50% active mask and sufficient energy (fallback to any active sample if none)
    sel = (cov >= 0.5) & (etot_w >= min_energy)
    if not bool(sel.any().item()):
        sel = (win_mask.sum(dim=-1) > 0) & (etot_w >= min_energy)

    valid_vals = corr_w[sel]
    n_valid = int(valid_vals.numel())
    if n_valid == 0:
        return {
            "valid": False,
            "window_ms": int(window_ms),
            "window_samples": int(w_len),
            "num_windows": 0,
            "corr_mean": None,
            "corr_min": None,
            "corr_p05": None,
            "corr_cvar05": None,
        }

    sorted_vals, _ = torch.sort(valid_vals)
    k_tail = max(1, int(math.ceil(0.05 * n_valid)))
    cvar05 = float(sorted_vals[:k_tail].mean().item())
    p05 = float(torch.quantile(sorted_vals, 0.05).item())

    return {
        "valid": True,
        "window_ms": int(window_ms),
        "window_samples": int(w_len),
        "num_windows": n_valid,
        "corr_mean": float(sorted_vals.mean().item()),
        "corr_min": float(sorted_vals[0].item()),
        "corr_p05": p05,
        "corr_cvar05": cvar05,
    }


@torch.no_grad()
def compute_comb_filtering_diagnostics(
    audio_eval_f64: torch.Tensor,
    mask_smp: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int = 512,
    min_energy: float = MIN_ACTIVE_BAND_ENERGY,
) -> Dict[str, Any]:
    """Compute diagnostic-only spectral anti-comb-filtering fold-down ripple & notch metrics."""
    B, _, T_eval = audio_eval_f64.shape
    if T_eval < n_fft:
        return {"valid": False, "comb_fold_mean_db": None, "comb_fold_std_db": None, "comb_fold_min_notch_db": None, "comb_ripple_p95_p05_db": None}

    win_l = audio_eval_f64[:, 0, :].unfold(-1, n_fft, hop_length)
    win_r = audio_eval_f64[:, 1, :].unfold(-1, n_fft, hop_length)
    win_mask = mask_smp.to(torch.float64).unfold(-1, n_fft, hop_length)

    cov = win_mask.mean(dim=-1)
    frame_e = 0.5 * ((win_l * win_l).mean(dim=-1) + (win_r * win_r).mean(dim=-1))
    sel = (cov >= 0.5) & (frame_e >= min_energy)
    if not bool(sel.any().item()):
        sel = (win_mask.sum(dim=-1) > 0) & (frame_e >= min_energy)
    if not bool(sel.any().item()):
        return {"valid": False, "comb_fold_mean_db": None, "comb_fold_std_db": None, "comb_fold_min_notch_db": None, "comb_ripple_p95_p05_db": None}

    hann = torch.hann_window(n_fft, dtype=torch.float64, device=audio_eval_f64.device)
    fl = win_l[sel] * hann
    fr = win_r[sel] * hann
    fm = (fl + fr) * INV_SQRT2

    spec_l = torch.fft.rfft(fl, dim=-1)
    spec_r = torch.fft.rfft(fr, dim=-1)
    spec_m = torch.fft.rfft(fm, dim=-1)

    p_mean_lr = 0.5 * ((spec_l.abs() ** 2) + (spec_r.abs() ** 2)).mean(dim=0)
    p_m = (spec_m.abs() ** 2).mean(dim=0)

    # Focus on bins above -50 dB relative to peak mean bin (skip DC bin 0)
    p_ref = p_mean_lr[1:]
    p_mid = p_m[1:]
    p_peak = float(p_ref.max().item())
    if p_peak <= ABS_EPS_FLOOR:
        return {"valid": False, "comb_fold_mean_db": None, "comb_fold_std_db": None, "comb_fold_min_notch_db": None, "comb_ripple_p95_p05_db": None}

    active_bins = p_ref >= (1e-5 * p_peak)
    if int(active_bins.sum().item()) < 4:
        active_bins = p_ref >= (1e-8 * p_peak)
    if int(active_bins.sum().item()) == 0:
        return {"valid": False, "comb_fold_mean_db": None, "comb_fold_std_db": None, "comb_fold_min_notch_db": None, "comb_ripple_p95_p05_db": None}

    eps_s = REL_EPS_SCALE * p_peak + ABS_EPS_FLOOR
    fold_db = 10.0 * torch.log10((p_mid[active_bins] + eps_s) / (p_ref[active_bins] + eps_s))
    return {
        "valid": True,
        "comb_fold_mean_db": float(fold_db.mean().item()),
        "comb_fold_std_db": float(fold_db.std(unbiased=False).item()),
        "comb_fold_min_notch_db": float(fold_db.min().item()),
        "comb_ripple_p95_p05_db": float((torch.quantile(fold_db, 0.95) - torch.quantile(fold_db, 0.05)).item()),
    }


class V3cStereoCompatibilityLoss(nn.Module):
    """V3c Differentiable Stereo & Mono Compatibility Prior Loss (Standalone)."""

    def __init__(
        self,
        sample_rate: float = 48000.0,
        f_low_hz: float = FIXED_CROSSOVER_LOW_HZ,
        f_high_hz: float = FIXED_CROSSOVER_HIGH_HZ,
        delay_dsp_samples: int = TOTAL_DSP_DELAY_SAMPLES,
        block_size: int = 480,
        silence_threshold_db: float = -70.0,
        min_active_blocks: int = 8,
        min_band_energy: float = MIN_ACTIVE_BAND_ENERGY,
        eta_ch: float = SINGLE_CHANNEL_ENERGY_RATIO,
        eps_rel_factor: float = REL_EPS_SCALE,
        # Frozen Candidate A thresholds
        tau_drop_db: float = 0.50,
        tau_corr: Tuple[float, float, float] = (0.03, 0.05, 0.08),
        tau_smr_db: Tuple[float, float, float] = (2.0, 2.0, 2.0),
        tau_sub_db: float = 0.50,
        alpha_sub: float = 1.0,
        tau_w_low_max: float = 1.00,
        tau_w_mid: float = 0.15,
        tau_w_high: float = 0.15,
        # Frozen pilot same_track weights
        w_mono_drop: float = 1.00,
        w_band_corr: float = 10.00,
        w_band_smr: float = 0.05,
        w_width_param: float = 0.25,
        window_durations_ms: Tuple[int, ...] = SHORT_TIME_WINDOWS_MS,
    ):
        super().__init__()
        self.sample_rate = float(sample_rate)
        self.f_low_hz = float(f_low_hz)
        self.f_high_hz = float(f_high_hz)
        self.delay_dsp_samples = int(delay_dsp_samples)
        self.block_size = int(block_size)
        self.silence_threshold_db = float(silence_threshold_db)
        self.min_active_blocks = int(min_active_blocks)
        self.min_band_energy = float(min_band_energy)
        self.eta_ch = float(eta_ch)
        self.eps_rel_factor = float(eps_rel_factor)

        self.tau_drop_db = float(tau_drop_db)
        self.tau_corr = tuple(float(v) for v in tau_corr)
        self.tau_smr_db = tuple(float(v) for v in tau_smr_db)
        self.tau_sub_db = float(tau_sub_db)
        self.alpha_sub = float(alpha_sub)
        self.tau_w_low_max = float(tau_w_low_max)
        self.tau_w_mid = float(tau_w_mid)
        self.tau_w_high = float(tau_w_high)

        self.w_mono_drop = float(w_mono_drop)
        self.w_band_corr = float(w_band_corr)
        self.w_band_smr = float(w_band_smr)
        self.w_width_param = float(w_width_param)
        self.window_durations_ms = tuple(int(w) for w in window_durations_ms)

        self.crossover = DifferentiableLR4Crossover(sample_rate=self.sample_rate)
        self.register_buffer("fixed_f_low", torch.tensor(self.f_low_hz, dtype=torch.float64))
        self.register_buffer("fixed_f_high", torch.tensor(self.f_high_hz, dtype=torch.float64))

    def _compute_signal_stereo_stats(
        self,
        x_stereo: torch.Tensor,
        active_mask: torch.Tensor,
        ref_energy_anchor: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        return compute_masked_stereo_stats(
            audio_eval_f64=x_stereo,
            mask_smp=active_mask,
            min_energy=self.min_band_energy,
            eta_ch=self.eta_ch,
            ref_energy_anchor=ref_energy_anchor,
            eps_rel_factor=self.eps_rel_factor,
        )

    def _filter_and_slice_pair(
        self,
        mix_ext: torch.Tensor,
        pred_ext: torch.Tensor,
        n_pre: int,
        delay_samples: Optional[int] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        """Run fixed 160/3200 Hz LR4 crossover on full pre-roll signals, then slice aligned eval windows."""
        delay = self.delay_dsp_samples if delay_samples is None else int(delay_samples)
        mix_f64 = _ensure_batch_stereo(mix_ext)
        pred_f64 = _ensure_batch_stereo(pred_ext)

        if mix_f64.shape[-1] != pred_f64.shape[-1]:
            raise ValueError(
                f"mix_ext and pred_ext must have matching extended length, got {mix_f64.shape[-1]} vs {pred_f64.shape[-1]}"
            )
        n_tot = mix_f64.shape[-1]
        T = n_tot - int(n_pre)
        if T <= delay:
            raise ValueError(f"Chunk post-preroll length T={T} must exceed delay={delay}")
        T_eval = T - delay

        f_low = self.fixed_f_low.to(device=pred_f64.device, dtype=torch.float64)
        f_high = self.fixed_f_high.to(device=pred_f64.device, dtype=torch.float64)

        with torch.no_grad():
            m_low, m_mid, m_high = self.crossover(mix_f64, f_low, f_high)
        p_low, p_mid, p_high = self.crossover(pred_f64, f_low, f_high)

        m_slice = slice(int(n_pre), int(n_pre) + T_eval)
        p_slice = slice(int(n_pre) + delay, int(n_pre) + T)

        mix_eval = mix_f64[..., m_slice].detach()
        pred_eval = pred_f64[..., p_slice]

        mix_bands = {
            "low": m_low[..., m_slice].detach(),
            "mid": m_mid[..., m_slice].detach(),
            "high": m_high[..., m_slice].detach(),
        }
        pred_bands = {
            "low": p_low[..., p_slice],
            "mid": p_mid[..., p_slice],
            "high": p_high[..., p_slice],
        }
        return mix_eval, pred_eval, mix_bands, pred_bands

    def compute_width_param_loss(
        self,
        z: Optional[torch.Tensor] = None,
        widths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute direct parameter prior L_width-param on imager_width_low/mid/high."""
        if widths is None:
            if z is None:
                raise ValueError("Either z (41D) or widths (3D) must be provided for loss_width_param")
            theta = map_z_to_theta(z)
            widths_t = theta[..., 35:38].to(torch.float64)
        else:
            widths_t = widths.to(torch.float64)

        w_low = widths_t[..., 0]
        w_mid = widths_t[..., 1]
        w_high = widths_t[..., 2]

        loss_low = F.relu(w_low - self.tau_w_low_max) ** 2
        loss_mid = F.relu(torch.abs(w_mid - 1.0) - self.tau_w_mid) ** 2
        loss_high = F.relu(torch.abs(w_high - 1.0) - self.tau_w_high) ** 2
        loss_width = (loss_low + 0.5 * (loss_mid + loss_high)).mean()

        meta = {
            "imager_width_low": float(w_low.mean().item()),
            "imager_width_mid": float(w_mid.mean().item()),
            "imager_width_high": float(w_high.mean().item()),
            "loss_width_param": float(loss_width.item()),
        }
        return loss_width, meta

    def forward(
        self,
        mix_ext: torch.Tensor,
        pred_ext: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        widths: Optional[torch.Tensor] = None,
        n_pre: int = 0,
        delay_samples: Optional[int] = None,
        compute_diagnostics: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Evaluate all 4 V3c candidate terms and comprehensive diagnostics."""
        mix_eval, pred_eval, mix_bands, pred_bands = self._filter_and_slice_pair(
            mix_ext=mix_ext,
            pred_ext=pred_ext,
            n_pre=n_pre,
            delay_samples=delay_samples,
        )

        # Active sample mask determined strictly on post-preroll input mix
        with torch.no_grad():
            mask_smp, is_silent, mask_meta = compute_active_sample_mask(
                mix_eval,
                block_size=self.block_size,
                silence_threshold_db=self.silence_threshold_db,
                min_active_blocks=self.min_active_blocks,
            )

        dummy_zero = 0.0 * pred_eval.sum()

        # Parameter term works even when audio is silent if z or widths are provided
        if z is not None or widths is not None:
            loss_width_param, width_meta = self.compute_width_param_loss(z=z, widths=widths)
        else:
            loss_width_param = torch.zeros((), dtype=torch.float64, device=pred_eval.device)
            width_meta = {}

        if is_silent:
            loss_mono_drop = dummy_zero
            loss_band_corr = dummy_zero
            loss_band_smr = dummy_zero
            total_loss = (
                self.w_mono_drop * loss_mono_drop
                + self.w_band_corr * loss_band_corr
                + self.w_band_smr * loss_band_smr
                + self.w_width_param * loss_width_param
            )
            band_valid_map = {b: False for b in BAND_NAMES}
            diagnostics: Dict[str, Any] = {
                "is_silent": True,
                "mask_meta": mask_meta,
                "broadband_valid": False,
                "pred_energy_below_emin_broadband": True,
                "band_valid": band_valid_map,
                "pred_energy_below_emin": {b: True for b in BAND_NAMES},
                "channel_silent_or_hard_panned": {b: False for b in BAND_NAMES},
                "loss_mono_drop": 0.0,
                "loss_band_corr": 0.0,
                "loss_band_smr": 0.0,
                "loss_width_param": float(loss_width_param.item()),
                "loss_v3c_total": float(total_loss.item()),
                **width_meta,
            }
            return total_loss, diagnostics

        # Broadband stats (mix & pred; validity determined strictly by mix energy, eps_rel anchored to mix)
        with torch.no_grad():
            bb_mix = self._compute_signal_stereo_stats(mix_eval, mask_smp, ref_energy_anchor=None)
        bb_pred = self._compute_signal_stereo_stats(
            pred_eval, mask_smp, ref_energy_anchor=bb_mix["e_tot_tensor"]
        )

        broadband_valid = (not is_silent) and bool(bb_mix["energy_valid"])
        pred_bb_below_emin = bool(float(bb_pred["e_tot"].min().item()) < self.min_band_energy)
        if broadband_valid:
            mono_excess = bb_pred["mono_drop_db"] - bb_mix["mono_drop_db"].detach() - self.tau_drop_db
            loss_mono_drop = (F.relu(mono_excess) ** 2).mean()
        else:
            loss_mono_drop = dummy_zero

        # Per-band stats & losses
        corr_terms: List[torch.Tensor] = []
        smr_terms: List[torch.Tensor] = []
        band_valid: Dict[str, bool] = {}
        pred_below_emin_map: Dict[str, bool] = {}
        hard_panned_flags: Dict[str, bool] = {}
        band_metrics: Dict[str, Any] = {}

        for idx, b in enumerate(BAND_NAMES):
            with torch.no_grad():
                st_mix = self._compute_signal_stereo_stats(
                    mix_bands[b], mask_smp, ref_energy_anchor=None
                )
            st_pred = self._compute_signal_stereo_stats(
                pred_bands[b], mask_smp, ref_energy_anchor=st_mix["e_tot_tensor"]
            )

            # Validity depends EXCLUSIVELY on original mix energy validity (never pred energy!)
            is_valid_b = (not is_silent) and bool(st_mix["energy_valid"])
            pred_below_emin_b = bool(float(st_pred["e_tot"].min().item()) < self.min_band_energy)
            band_valid[b] = is_valid_b
            pred_below_emin_map[b] = pred_below_emin_b
            hard_panned_flags[b] = bool(
                st_mix["channel_silent_or_hard_panned"] or st_pred["channel_silent_or_hard_panned"]
            )

            if is_valid_b:
                corr_drop = st_mix["corr_lr"].detach() - st_pred["corr_lr"] - self.tau_corr[idx]
                corr_terms.append((F.relu(corr_drop) ** 2).mean())

                delta_smr = st_pred["smr_db"] - st_mix["smr_db"].detach()
                smr_excess = F.relu(torch.abs(delta_smr) - self.tau_smr_db[idx]) ** 2
                smr_terms.append(smr_excess.mean())
                if b == "low" and self.alpha_sub > 0.0:
                    sub_excess = F.relu(delta_smr - self.tau_sub_db) ** 2
                    sub_term = self.alpha_sub * sub_excess.mean()
                elif b == "low":
                    sub_term = dummy_zero
            else:
                corr_terms.append(dummy_zero)
                smr_terms.append(dummy_zero)
                if b == "low":
                    sub_term = dummy_zero

            if compute_diagnostics:
                d_corr = float((st_pred["corr_lr"] - st_mix["corr_lr"]).mean().item()) if is_valid_b else 0.0
                d_smr = float((st_pred["smr_db"] - st_mix["smr_db"]).mean().item()) if is_valid_b else 0.0
                d_mono = float((st_pred["mono_drop_db"] - st_mix["mono_drop_db"]).mean().item()) if is_valid_b else 0.0
                band_metrics[b] = {
                    "band_valid": is_valid_b,
                    "band_valid_mix": bool(st_mix["energy_valid"]),
                    "band_valid_pred": bool(st_pred["energy_valid"]),
                    "pred_energy_below_emin": pred_below_emin_b,
                    "channel_silent_or_hard_panned": hard_panned_flags[b],
                    "e_tot_mix": float(st_mix["e_tot"].mean().item()),
                    "e_tot_pred": float(st_pred["e_tot"].mean().item()),
                    "smr_db_mix": float(st_mix["smr_db"].mean().item()) if st_mix["energy_valid"] else None,
                    "smr_db_pred": float(st_pred["smr_db"].mean().item()) if (is_valid_b or st_pred["energy_valid"]) else None,
                    "delta_smr_db": d_smr if is_valid_b else None,
                    "corr_lr_mix": float(st_mix["corr_lr"].mean().item()) if st_mix["energy_valid"] else None,
                    "corr_lr_pred": float(st_pred["corr_lr"].mean().item()) if (is_valid_b or st_pred["energy_valid"]) else None,
                    "delta_corr_lr": d_corr if is_valid_b else None,
                    "mono_drop_db_mix": float(st_mix["mono_drop_db"].mean().item()) if st_mix["energy_valid"] else None,
                    "mono_drop_db_pred": float(st_pred["mono_drop_db"].mean().item()) if (is_valid_b or st_pred["energy_valid"]) else None,
                    "delta_mono_drop_db": d_mono if is_valid_b else None,
                    "lr_balance_db_mix": float(st_mix["lr_balance_db"].mean().item()) if st_mix["energy_valid"] else None,
                    "lr_balance_db_pred": float(st_pred["lr_balance_db"].mean().item()) if (is_valid_b or st_pred["energy_valid"]) else None,
                }

        loss_band_corr = (corr_terms[0] + corr_terms[1] + corr_terms[2]) / 3.0
        loss_band_smr = (smr_terms[0] + smr_terms[1] + smr_terms[2]) / 3.0 + sub_term

        total_loss = (
            self.w_mono_drop * loss_mono_drop
            + self.w_band_corr * loss_band_corr
            + self.w_band_smr * loss_band_smr
            + self.w_width_param * loss_width_param
        )

        diagnostics = {
            "is_silent": False,
            "mask_meta": mask_meta,
            "broadband_valid": broadband_valid,
            "pred_energy_below_emin_broadband": pred_bb_below_emin,
            "channel_silent_or_hard_panned_broadband": bool(
                bb_mix["channel_silent_or_hard_panned"] or bb_pred["channel_silent_or_hard_panned"]
            ),
            "band_valid": band_valid,
            "pred_energy_below_emin": pred_below_emin_map,
            "channel_silent_or_hard_panned": hard_panned_flags,
            "loss_mono_drop": float(loss_mono_drop.item()),
            "loss_band_corr": float(loss_band_corr.item()),
            "loss_band_smr": float(loss_band_smr.item()),
            "loss_width_param": float(loss_width_param.item()),
            "loss_v3c_total": float(total_loss.item()),
            **width_meta,
        }

        if compute_diagnostics:
            with torch.no_grad():
                diagnostics["broadband"] = {
                    "valid": broadband_valid,
                    "pred_energy_below_emin": pred_bb_below_emin,
                    "e_tot_mix": float(bb_mix["e_tot"].mean().item()),
                    "e_tot_pred": float(bb_pred["e_tot"].mean().item()),
                    "smr_db_mix": float(bb_mix["smr_db"].mean().item()),
                    "smr_db_pred": float(bb_pred["smr_db"].mean().item()),
                    "delta_smr_db": float((bb_pred["smr_db"] - bb_mix["smr_db"]).mean().item()),
                    "corr_lr_mix": float(bb_mix["corr_lr"].mean().item()),
                    "corr_lr_pred": float(bb_pred["corr_lr"].mean().item()),
                    "delta_corr_lr": float((bb_pred["corr_lr"] - bb_mix["corr_lr"]).mean().item()),
                    "mono_drop_db_mix": float(bb_mix["mono_drop_db"].mean().item()),
                    "mono_drop_db_pred": float(bb_pred["mono_drop_db"].mean().item()),
                    "delta_mono_drop_db": float((bb_pred["mono_drop_db"] - bb_mix["mono_drop_db"]).mean().item()),
                    "lr_balance_db_mix": float(bb_mix["lr_balance_db"].mean().item()),
                    "lr_balance_db_pred": float(bb_pred["lr_balance_db"].mean().item()),
                }
                diagnostics["bands"] = band_metrics

                # Multi-window short-time correlation diagnostics (50 ms, 100 ms, 250 ms)
                st_diag: Dict[str, Any] = {}
                for w_ms in self.window_durations_ms:
                    w_key = f"{w_ms}ms"
                    w_entry: Dict[str, Any] = {
                        "broadband_mix": compute_short_time_window_stats(
                            mix_eval, mask_smp, sample_rate=self.sample_rate, window_ms=w_ms, min_energy=self.min_band_energy
                        ),
                        "broadband_pred": compute_short_time_window_stats(
                            pred_eval, mask_smp, sample_rate=self.sample_rate, window_ms=w_ms, min_energy=self.min_band_energy
                        ),
                        "bands_mix": {},
                        "bands_pred": {},
                    }
                    for b in BAND_NAMES:
                        w_entry["bands_mix"][b] = compute_short_time_window_stats(
                            mix_bands[b], mask_smp, sample_rate=self.sample_rate, window_ms=w_ms, min_energy=self.min_band_energy
                        )
                        w_entry["bands_pred"][b] = compute_short_time_window_stats(
                            pred_bands[b], mask_smp, sample_rate=self.sample_rate, window_ms=w_ms, min_energy=self.min_band_energy
                        )
                    st_diag[w_key] = w_entry
                diagnostics["short_time_windows"] = st_diag

                # Spectral anti-comb-filtering diagnostics
                comb_mix = compute_comb_filtering_diagnostics(
                    mix_eval, mask_smp, min_energy=self.min_band_energy
                )
                comb_pred = compute_comb_filtering_diagnostics(
                    pred_eval, mask_smp, min_energy=self.min_band_energy
                )
                diagnostics["comb_filtering"] = {
                    "mix": comb_mix,
                    "pred": comb_pred,
                    "delta_comb_min_notch_db": (
                        comb_pred["comb_fold_min_notch_db"] - comb_mix["comb_fold_min_notch_db"]
                        if comb_mix["valid"] and comb_pred["valid"]
                        else None
                    ),
                    "delta_comb_std_db": (
                        comb_pred["comb_fold_std_db"] - comb_mix["comb_fold_std_db"]
                        if comb_mix["valid"] and comb_pred["valid"]
                        else None
                    ),
                }

        return total_loss, diagnostics
