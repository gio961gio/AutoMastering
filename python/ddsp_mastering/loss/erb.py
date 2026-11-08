"""ERB-48 Perceptual Spectral Filterbank, Active Masking, and Spectral Loss (Sprint 3 — Task 3.1).

Implements:
  1. Active frame selection and smooth sigmoid weighting anchored to the unoptimized
     signal `x_0` (`with torch.no_grad()`), with deterministic, NaN-free fallbacks
     for sparse (`|A_raw| < M_min`) and silent (`max L_rms < -70 dBFS`) chunks.
  2. Explicit conversion from frame/block-level active masks (10 ms / 480 samples)
     to sample-level active masks `A_smp(x_0)` via `repeat_interleave`.
  3. 48-band Glasberg & Moore (1990) ERB-rate filterbank (`20 Hz - 20 kHz`, unit-area
     normalized triangular filters on `N_fft = 4096, hop = 1024`).
  4. `same_track` active-frame Huber ERB spectral loss (paired alongside MR-STFT in V1).
  5. `stylistic` multi-statistic ERB spectral loss (level-invariant mean shape `mu_tilde_b`,
     temporal standard deviation `sigma_b`, and active-frame `P90 - P10` spread `Delta P_b^act`).
  6. Optional asymmetric relative de-harshness penalty (`2.5 - 5.0 kHz` vs `0.4 - 2.0 kHz`
     presence tilt surplus over reference, `V1b`).
"""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def freq_to_erb_rate(freq_hz: torch.Tensor) -> torch.Tensor:
    """Convert frequency in Hz to Glasberg & Moore (1990) ERB_N rate."""
    return 21.4 * torch.log10(0.00437 * freq_hz + 1.0)


def erb_rate_to_freq(erb_rate: torch.Tensor) -> torch.Tensor:
    """Convert Glasberg & Moore (1990) ERB_N rate back to frequency in Hz."""
    return (torch.pow(10.0, erb_rate / 21.4) - 1.0) / 0.00437


def build_erb_filterbank(
    n_freqs: int,
    sample_rate: float = 48000.0,
    n_bands: int = 48,
    f_min: float = 20.0,
    f_max: float = 20000.0,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct a unit-area normalized triangular ERB-rate filterbank.

    Args:
        n_freqs: Number of rFFT bins (`n_fft // 2 + 1`).
        sample_rate: Sampling rate in Hz (default: 48000.0).
        n_bands: Number of ERB bands `B` (default: 48).
        f_min: Minimum center/edge frequency in Hz (default: 20.0).
        f_max: Maximum frequency in Hz (default: 20000.0, clamped below Nyquist).
        dtype: Output tensor dtype.

    Returns:
        Tuple of:
          - `fbanks`: Tensor of shape `(n_freqs, n_bands)` with `sum_k fbanks[k, b] == 1.0`.
          - `center_freqs_hz`: Tensor of shape `(n_bands,)` with center frequencies in Hz.
    """
    f_max_eff = min(float(f_max), float(sample_rate) * 0.499)
    fft_freqs = torch.linspace(0.0, float(sample_rate) * 0.5, n_freqs, dtype=torch.float64)

    erb_min = freq_to_erb_rate(torch.tensor(float(f_min), dtype=torch.float64))
    erb_max = freq_to_erb_rate(torch.tensor(f_max_eff, dtype=torch.float64))

    erb_points = torch.linspace(erb_min.item(), erb_max.item(), n_bands + 2, dtype=torch.float64)
    hz_points = erb_rate_to_freq(erb_points)

    f_low = hz_points[:-2]
    f_center = hz_points[1:-1]
    f_high = hz_points[2:]

    fbanks = torch.zeros((n_freqs, n_bands), dtype=torch.float64)
    for b in range(n_bands):
        fl = f_low[b]
        fc = f_center[b]
        fh = f_high[b]

        up_slope = (fft_freqs - fl) / torch.clamp(fc - fl, min=1e-9)
        down_slope = (fh - fft_freqs) / torch.clamp(fh - fc, min=1e-9)
        tri = torch.clamp(torch.minimum(up_slope, down_slope), min=0.0)

        weight_sum = torch.sum(tri)
        if weight_sum > 1e-12:
            fbanks[:, b] = tri / weight_sum
        else:
            # Fallback to nearest FFT bin if a narrow band falls between two bin centers
            nearest_idx = int(torch.argmin(torch.abs(fft_freqs - fc)).item())
            fbanks[nearest_idx, b] = 1.0

    return fbanks.to(dtype=dtype), f_center.to(dtype=dtype)


def compute_active_frame_weights_and_mask(
    anchor_audio: torch.Tensor,
    n_fft: int = 4096,
    hop_length: int = 1024,
    min_active_frames: int = 8,
    silence_floor_dbfs: float = -70.0,
    active_floor_dbfs: float = -60.0,
    dynamic_span_db: float = 35.0,
    sigmoid_temp_db: float = 2.0,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute smooth active-frame weights and discrete active-frame mask on `anchor_audio` (`no_grad`).

    Implements the deterministic 3-regime behavior specified in Sprint 3 §2.A.2:
      1. Nominal (`max L_rms >= -70 dBFS` and `|A_raw| >= M_min`):
         `A(x_0) = {t | L_rms[t] >= theta_act}`, `is_valid = 1.0`.
      2. Sparse (`max L_rms >= -70 dBFS` and `|A_raw| < M_min`):
         `A(x_0)` falls back to the top-`M_min` most energetic frames via `topk`, `is_valid = 1.0`.
      3. Silent (`max L_rms < -70 dBFS`):
         `w_tilde[t] = 1 / T_frames`, `A(x_0) = all frames`, and `is_valid = 0.0` so
         downstream losses evaluate to finite `0.0` with `0.0` gradient (never NaN/Inf).

    Args:
        anchor_audio: Unoptimized reference/anchor tensor `(B, C, T)` or `(C, T)`.
        n_fft: STFT window size (default: 4096).
        hop_length: STFT hop size (default: 1024).
        min_active_frames: Minimum active frames required for quantile stability (default: 8).
        silence_floor_dbfs: Digital silence cutoff in dBFS (default: -70.0).
        active_floor_dbfs: Minimum gate threshold in dBFS (default: -60.0).
        dynamic_span_db: Dynamic range below peak considered active in dB (default: 35.0).
        sigmoid_temp_db: Smooth sigmoid transition width in dB (default: 2.0).
        eps: Numerical stability floor (default: 1e-12).

    Returns:
        Tuple of `(w_tilde, active_mask_bool, is_valid_chunk)`:
          - `w_tilde`: Smooth normalized frame weights `(B, T_frames)`, summing to 1.0 along `-1`.
          - `active_mask_bool`: Boolean active frame mask `(B, T_frames)` with at least `M_min` True per batch.
          - `is_valid_chunk`: Float tensor `(B,)` equal to `1.0` for non-silent chunks and `0.0` for silent chunks.
    """
    with torch.no_grad():
        x0 = anchor_audio.detach()
        if x0.dim() == 2:
            x0 = x0.unsqueeze(0)
        B, C, T = x0.shape
        device = x0.device
        dtype = x0.dtype

        window = torch.hann_window(n_fft, device=device, dtype=dtype)
        x0_flat = x0.reshape(B * C, T)
        stft_x0 = torch.stft(
            x0_flat,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True,
        )
        # stft_x0: (B * C, n_freqs, T_frames)
        T_frames = stft_x0.shape[-1]
        mag_sq = (torch.abs(stft_x0) ** 2).reshape(B, C, -1, T_frames)

        # Parseval frame mean-square energy across channels (accounting for one-sided rFFT)
        # DC and Nyquist appear once; interior bins appear twice
        one_sided_energy = (
            mag_sq[:, :, 0, :]
            + mag_sq[:, :, -1, :]
            + 2.0 * torch.sum(mag_sq[:, :, 1:-1, :], dim=-2)
        )  # (B, C, T_frames)
        win_norm = float(n_fft) * torch.sum(window ** 2).clamp(min=eps)
        frame_ms = torch.mean(one_sided_energy, dim=1) / win_norm  # (B, T_frames), normalized by 1/C
        l_rms_db = 10.0 * torch.log10(frame_ms + eps)  # (B, T_frames)

        m_min = min(max(1, int(min_active_frames)), T_frames)
        w_tilde = torch.zeros((B, T_frames), device=device, dtype=dtype)
        active_mask = torch.zeros((B, T_frames), device=device, dtype=torch.bool)
        is_valid = torch.ones((B,), device=device, dtype=dtype)

        for b in range(B):
            l_b = l_rms_db[b]
            l_max = torch.max(l_b).item()
            if l_max < silence_floor_dbfs:
                # Regime 3: Completely silent chunk -> uniform weights, all frames masked True, validity 0.0
                w_tilde[b].fill_(1.0 / float(T_frames))
                active_mask[b].fill_(True)
                is_valid[b] = 0.0
            else:
                theta_act = max(float(active_floor_dbfs), l_max - float(dynamic_span_db))
                w_raw = torch.sigmoid((l_b - theta_act) / float(sigmoid_temp_db))
                w_tilde[b] = w_raw / (torch.sum(w_raw) + eps)

                raw_mask = l_b >= theta_act
                if int(torch.sum(raw_mask).item()) >= m_min:
                    # Regime 1: Nominal chunk
                    active_mask[b] = raw_mask
                else:
                    # Regime 2: Sparse chunk -> deterministic fallback to top-M_min most energetic frames
                    top_idx = torch.topk(l_b, k=m_min, largest=True, sorted=False).indices
                    active_mask[b, top_idx] = True
                is_valid[b] = 1.0

        return w_tilde, active_mask, is_valid


def compute_active_sample_mask(
    anchor_audio: torch.Tensor,
    block_samples: int = 480,
    min_active_blocks: int = 8,
    silence_floor_dbfs: float = -70.0,
    active_floor_dbfs: float = -60.0,
    dynamic_span_db: float = 35.0,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert short-block (10 ms / 480 samples) active detection on `anchor_audio` to sample-level mask `A_smp(x_0)`.

    Follows Sprint 3 §2.A.2:
      1. Partitions `anchor_audio` into `M_b = floor(T / H_smp)` blocks of `H_smp = 480` samples (10 ms at 48 kHz).
      2. Computes block RMS level `L_blk[m; x_0]` and binary active block indicator `b_act[m; x_0]`
         with the exact same sparse (`topk(M_min)`) and silent (`< -70 dBFS -> is_valid = 0.0`) fallbacks.
      3. Expands `b_act` to sample-level mask `m_smp[n; x_0]` of length `T` via `repeat_interleave(H_smp)`,
         extending the last block `M_b - 1` to any trailing remainder samples `T - M_b * H_smp`.

    Args:
        anchor_audio: Unoptimized reference/anchor tensor `(B, C, T)` or `(C, T)`.
        block_samples: Block length `H_smp` in samples (default: 480 = 10 ms at 48 kHz).
        min_active_blocks: Minimum active blocks `M_min` (default: 8).
        silence_floor_dbfs: Digital silence cutoff in dBFS (default: -70.0).
        active_floor_dbfs: Minimum gate threshold in dBFS (default: -60.0).
        dynamic_span_db: Dynamic range below peak considered active in dB (default: 35.0).
        eps: Numerical stability floor (default: 1e-12).

    Returns:
        Tuple of `(sample_mask_bool, is_valid_chunk)`:
          - `sample_mask_bool`: Boolean tensor `(B, T)` with `True` on active samples `A_smp(x_0)`.
          - `is_valid_chunk`: Float tensor `(B,)` (`1.0` if non-silent, `0.0` if silent).
    """
    with torch.no_grad():
        x0 = anchor_audio.detach()
        if x0.dim() == 2:
            x0 = x0.unsqueeze(0)
        B, C, T = x0.shape
        device = x0.device
        dtype = x0.dtype

        h_smp = max(1, min(int(block_samples), T))
        m_b = max(1, T // h_smp)
        trimmed_len = m_b * h_smp

        x_blocks = x0[:, :, :trimmed_len].reshape(B, C, m_b, h_smp)
        block_ms = torch.mean(x_blocks ** 2, dim=(1, 3))  # (B, m_b), normalized by 1/(C * H_smp)
        l_blk_db = 10.0 * torch.log10(block_ms + eps)

        m_min = min(max(1, int(min_active_blocks)), m_b)
        b_act = torch.zeros((B, m_b), device=device, dtype=torch.bool)
        is_valid = torch.ones((B,), device=device, dtype=dtype)

        for b in range(B):
            l_b = l_blk_db[b]
            l_max = torch.max(l_b).item()
            if l_max < silence_floor_dbfs:
                b_act[b].fill_(True)
                is_valid[b] = 0.0
            else:
                theta_blk = max(float(active_floor_dbfs), l_max - float(dynamic_span_db))
                raw_mask = l_b >= theta_blk
                if int(torch.sum(raw_mask).item()) >= m_min:
                    b_act[b] = raw_mask
                else:
                    top_idx = torch.topk(l_b, k=m_min, largest=True, sorted=False).indices
                    b_act[b, top_idx] = True
                is_valid[b] = 1.0

        # Piecewise-constant expansion via repeat_interleave(h_smp)
        sample_mask = torch.repeat_interleave(b_act, repeats=h_smp, dim=-1)  # (B, trimmed_len)
        rem = T - trimmed_len
        if rem > 0:
            tail_mask = b_act[:, -1:].expand(B, rem)
            sample_mask = torch.cat([sample_mask, tail_mask], dim=-1)

        return sample_mask, is_valid


class ERBSpectralLoss(nn.Module):
    """48-band ERB perceptual spectral loss for `same_track` and `stylistic` modes (Task 3.1)."""

    def __init__(
        self,
        sample_rate: float = 48000.0,
        n_bands: int = 48,
        n_fft: int = 4096,
        hop_length: int = 1024,
        f_min: float = 20.0,
        f_max: float = 20000.0,
        alpha_std: float = 0.25,
        alpha_quantiles: float = 0.15,
        de_harsh_margin_db: float = 0.75,
        weight_de_harsh: float = 0.0,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.sample_rate = float(sample_rate)
        self.n_bands = int(n_bands)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.alpha_std = float(alpha_std)
        self.alpha_quantiles = float(alpha_quantiles)
        self.de_harsh_margin_db = float(de_harsh_margin_db)
        self.weight_de_harsh = float(weight_de_harsh)
        self.eps = float(eps)

        n_freqs = self.n_fft // 2 + 1
        fbanks, center_freqs = build_erb_filterbank(
            n_freqs=n_freqs,
            sample_rate=self.sample_rate,
            n_bands=self.n_bands,
            f_min=f_min,
            f_max=f_max,
            dtype=torch.float32,
        )
        self.register_buffer("erb_fbanks", fbanks)  # (n_freqs, B)
        self.register_buffer("erb_center_freqs", center_freqs)  # (B,)
        self.register_buffer("stft_window", torch.hann_window(self.n_fft))

        # Band masks for presence / de-harshness tilt (2.5-5 kHz vs 0.4-2.0 kHz)
        harsh_mask = (center_freqs >= 2500.0) & (center_freqs <= 5000.0)
        mid_mask = (center_freqs >= 400.0) & (center_freqs <= 2000.0)
        self.register_buffer("harsh_band_mask", harsh_mask)
        self.register_buffer("mid_band_mask", mid_mask)

    def extract_erb_band_levels_db(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute time-frequency ERB band levels `L_{b,t}(x)` in dB of shape `(B_batch, n_bands, T_frames)`.

        Normalizes channel power by `1 / C` (`C in {1, 2}`) so mono and dual-mono signals yield identical levels.
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)
        B_batch, C, T = audio.shape
        audio_flat = audio.reshape(B_batch * C, T)

        window = self.stft_window.to(device=audio.device, dtype=audio.dtype)
        fbanks = self.erb_fbanks.to(device=audio.device, dtype=audio.dtype)

        stft_res = torch.stft(
            audio_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
        )
        # Power spectrum averaged across channels: 1/C * sum_c |X_c(k, t)|^2
        mag_sq = (torch.abs(stft_res) ** 2).reshape(B_batch, C, -1, stft_res.shape[-1])
        power_avg = torch.mean(mag_sq, dim=1)  # (B_batch, n_freqs, T_frames)

        # Project onto ERB filterbank: (B_batch, T_frames, n_freqs) @ (n_freqs, n_bands) -> (B_batch, n_bands, T_frames)
        erb_energy = torch.matmul(power_avg.transpose(-1, -2), fbanks).transpose(-1, -2)
        erb_db = 10.0 * torch.log10(erb_energy + self.eps)
        return erb_db

    def compute_presence_tilt_db(
        self,
        erb_db: torch.Tensor,
        w_tilde: torch.Tensor,
    ) -> torch.Tensor:
        """Compute presence-to-mid spectral tilt `tau_harsh(x)` in dB (`[2.5, 5 kHz]` minus `[0.4, 2 kHz]`)."""
        # Weighted temporal mean per band: mu_b(x) = sum_t w_tilde[t] * L_{b,t}(x)
        mu_b = torch.sum(erb_db * w_tilde.unsqueeze(1), dim=-1)  # (B_batch, n_bands)
        harsh_mean = torch.mean(mu_b[:, self.harsh_band_mask], dim=-1)
        mid_mean = torch.mean(mu_b[:, self.mid_band_mask], dim=-1)
        return harsh_mean - mid_mean

    def compute_de_harshness_penalty(
        self,
        erb_pred_db: torch.Tensor,
        w_pred: torch.Tensor,
        erb_ref_db: torch.Tensor,
        w_ref: torch.Tensor,
        valid_mult: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute asymmetric relative de-harshness loss `L_de_harsh(y, y_ref)` (§2.B.2)."""
        tau_pred = self.compute_presence_tilt_db(erb_pred_db, w_pred)
        with torch.no_grad():
            tau_ref = self.compute_presence_tilt_db(erb_ref_db, w_ref)

        surplus_db = tau_pred - tau_ref
        penalty_per_batch = (F.relu(surplus_db - self.de_harsh_margin_db) ** 2) * valid_mult
        loss_val = torch.mean(penalty_per_batch)

        metrics = {
            "erb_presence_tilt_pred_db": tau_pred.mean().item(),
            "erb_presence_tilt_ref_db": tau_ref.mean().item(),
            "erb_presence_surplus_db": surplus_db.mean().item(),
            "erb_de_harsh_loss": loss_val.item(),
        }
        return loss_val, metrics

    def forward_same_track(
        self,
        pred_audio: torch.Tensor,
        target_audio: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute `same_track` active-frame weighted Huber ERB-48 loss (§2.B.1)."""
        if pred_audio.dim() == 2:
            pred_audio = pred_audio.unsqueeze(0)
        if target_audio.dim() == 2:
            target_audio = target_audio.unsqueeze(0)

        w_tilde_ref, _, is_valid_ref = compute_active_frame_weights_and_mask(
            target_audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            eps=self.eps,
        )

        erb_pred_db = self.extract_erb_band_levels_db(pred_audio)
        with torch.no_grad():
            erb_ref_db = self.extract_erb_band_levels_db(target_audio)

        # Elementwise Huber_1.0 across (B_batch, n_bands, T_frames)
        huber_elem = F.huber_loss(erb_pred_db, erb_ref_db, reduction="none", delta=1.0)
        # Weight across frames with w_tilde_ref (which sums to 1.0 over T_frames) and average over B_bands
        weighted_time = torch.sum(huber_elem * w_tilde_ref.unsqueeze(1), dim=-1)  # (B_batch, n_bands)
        band_mean = torch.mean(weighted_time, dim=-1) * is_valid_ref  # (B_batch,)
        erb_loss = torch.mean(band_mean)

        de_harsh_loss, dh_metrics = self.compute_de_harshness_penalty(
            erb_pred_db, w_tilde_ref, erb_ref_db, w_tilde_ref, is_valid_ref
        )
        total = erb_loss + self.weight_de_harsh * de_harsh_loss

        metrics: Dict[str, float] = {
            "erb_same_track_loss": erb_loss.item(),
            **dh_metrics,
        }
        return total, metrics

    def _compute_stylistic_band_stats(
        self,
        erb_db: torch.Tensor,
        w_tilde: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute `(mu_tilde_b, sigma_b, delta_P_b_act)` across active frames for each batch item."""
        # 1. Weighted mean per band: mu_b = sum_t w_tilde[t] * L_{b,t}
        mu_b = torch.sum(erb_db * w_tilde.unsqueeze(1), dim=-1)  # (B_batch, n_bands)
        # Level-invariant spectral shape: mu_tilde_b = mu_b - mean_{b'}(mu_{b'})
        mu_tilde_b = mu_b - torch.mean(mu_b, dim=-1, keepdim=True)

        # 2. Weighted temporal standard deviation per band: sigma_b
        centered_sq = (erb_db - mu_b.unsqueeze(-1)) ** 2
        var_b = torch.sum(centered_sq * w_tilde.unsqueeze(1), dim=-1)
        sigma_b = torch.sqrt(var_b + self.eps)

        # 3. Active-frame inter-quantile range: P90 - P10 over t in A(x_0)
        B_batch = erb_db.shape[0]
        delta_p_list = []
        q_probs = torch.tensor([0.10, 0.90], device=erb_db.device, dtype=erb_db.dtype)
        for b_idx in range(B_batch):
            act_indices = torch.nonzero(active_mask[b_idx], as_tuple=False).squeeze(-1)
            active_frames_db = erb_db[b_idx, :, act_indices]  # (n_bands, |A|)
            q_vals = torch.quantile(active_frames_db, q_probs, dim=-1)  # (2, n_bands)
            delta_p_list.append(q_vals[1] - q_vals[0])
        delta_p_b = torch.stack(delta_p_list, dim=0)  # (B_batch, n_bands)

        return mu_tilde_b, sigma_b, delta_p_b

    def forward_stylistic(
        self,
        pred_audio: torch.Tensor,
        ref_audio: torch.Tensor,
        mix_anchor_audio: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute `stylistic` multi-statistic ERB-48 spectral loss (§2.B.1)."""
        if pred_audio.dim() == 2:
            pred_audio = pred_audio.unsqueeze(0)
        if ref_audio.dim() == 2:
            ref_audio = ref_audio.unsqueeze(0)

        pred_anchor = mix_anchor_audio if mix_anchor_audio is not None else pred_audio.detach()
        if pred_anchor.dim() == 2:
            pred_anchor = pred_anchor.unsqueeze(0)

        w_pred, mask_pred, valid_pred = compute_active_frame_weights_and_mask(
            pred_anchor,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            eps=self.eps,
        )
        w_ref, mask_ref, valid_ref = compute_active_frame_weights_and_mask(
            ref_audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            eps=self.eps,
        )
        valid_mult = valid_pred * valid_ref  # (B_batch,)

        erb_pred_db = self.extract_erb_band_levels_db(pred_audio)
        mu_tilde_p, sigma_p, delta_p_p = self._compute_stylistic_band_stats(
            erb_pred_db, w_pred, mask_pred
        )

        with torch.no_grad():
            erb_ref_db = self.extract_erb_band_levels_db(ref_audio)
            mu_tilde_r, sigma_r, delta_p_r = self._compute_stylistic_band_stats(
                erb_ref_db, w_ref, mask_ref
            )

        shape_loss_per_b = torch.mean(
            F.huber_loss(mu_tilde_p, mu_tilde_r, reduction="none", delta=1.0), dim=-1
        ) * valid_mult
        std_loss_per_b = torch.mean(torch.abs(sigma_p - sigma_r), dim=-1) * valid_mult
        quant_loss_per_b = torch.mean(torch.abs(delta_p_p - delta_p_r), dim=-1) * valid_mult

        shape_loss = torch.mean(shape_loss_per_b)
        std_loss = torch.mean(std_loss_per_b)
        quant_loss = torch.mean(quant_loss_per_b)

        erb_style_loss = shape_loss + self.alpha_std * std_loss + self.alpha_quantiles * quant_loss

        de_harsh_loss, dh_metrics = self.compute_de_harshness_penalty(
            erb_pred_db, w_pred, erb_ref_db, w_ref, valid_mult
        )
        total = erb_style_loss + self.weight_de_harsh * de_harsh_loss

        metrics: Dict[str, float] = {
            "erb_stylistic_shape_loss": shape_loss.item(),
            "erb_stylistic_std_loss": std_loss.item(),
            "erb_stylistic_p90_p10_loss": quant_loss.item(),
            "erb_stylistic_total_loss": erb_style_loss.item(),
            **dh_metrics,
        }
        return total, metrics
