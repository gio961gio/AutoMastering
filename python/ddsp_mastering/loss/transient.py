"""Multi-band Transient / Envelope Loss Module for AutoMastering (Sprint 3 Task 3.2).

Computes fast/slow envelope dynamics, local crest factor C_c[m], and attack onset
velocity Delta^+ e_c^fast[m] across three psychoacoustic analysis bands:
  - Low:  [20, 160] Hz
  - Mid:  [160, 3200] Hz
  - High: [3200, 20000] Hz

Supports:
  1. `same_track`: point-to-point active-frame-weighted Huber_1.0 matching.
  2. `stylistic`: multi-quantile (P50, P90, P95) distributional distance on active frames.
  3. Safe fallback for silent and sparse chunks (no NaN, Inf, or zero division).
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ddsp_mastering.dsp.crossover import DifferentiableLR4Crossover


def compute_transient_active_mask(
    anchor_audio: torch.Tensor,
    sample_rate: float = 48000.0,
    hop_samples: int = 120,
    window_samples: int = 2400,
    min_active_frames: int = 8,
    silence_floor_dbfs: float = -70.0,
    active_floor_dbfs: float = -60.0,
    dynamic_span_db: float = 35.0,
    sigmoid_temp_db: float = 2.0,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute active-frame weights and discrete mask for transient evaluation (§2.A.2).

    Args:
        anchor_audio: Anchor tensor `(B, C, T)` or `(C, T)` outside autograd (`no_grad`).
        sample_rate: Sampling frequency in Hz.
        hop_samples: Hop size between envelope frames (default: 120 samples / 2.5 ms).
        window_samples: Analysis window size for RMS gating (default: 2400 samples / 50 ms).
        min_active_frames: Minimum active frames required (default: 8).
        silence_floor_dbfs: Digital silence threshold in dBFS (default: -70.0).
        active_floor_dbfs: Minimum active floor in dBFS (default: -60.0).
        dynamic_span_db: Range below peak considered active (default: 35.0 dB).
        sigmoid_temp_db: Sigmoid slope temperature (default: 2.0 dB).
        eps: Numerical floor (default: 1e-12).

    Returns:
        `(w_tilde, active_mask_bool, is_valid_chunk)`:
          - `w_tilde`: Smooth weights `(B, M)`, summing to 1.0 along `-1`.
          - `active_mask_bool`: Boolean mask `(B, M)`.
          - `is_valid_chunk`: Float tensor `(B,)` with 1.0 (valid) or 0.0 (silent).
    """
    with torch.no_grad():
        x0 = anchor_audio.detach()
        if x0.dim() == 2:
            x0 = x0.unsqueeze(0)
        B, C, T = x0.shape
        device = x0.device

        # Channel mean square: (B, 1, T)
        s = torch.mean(x0.to(torch.float64) ** 2, dim=1, keepdim=True)

        # Running RMS over window_samples via float64 cumsum
        half_w = window_samples // 2
        s_pad = F.pad(s, (half_w, half_w), mode="replicate")
        cs = torch.cumsum(s_pad, dim=-1)
        cs_pad = F.pad(cs, (1, 0), mode="constant", value=0.0)

        indices = torch.arange(0, s_pad.shape[-1] - window_samples + 1, hop_samples, device=device)
        ms = (cs_pad[..., indices + window_samples] - cs_pad[..., indices]) / float(window_samples)
        # ms: (B, 1, M) -> (B, M)
        ms = ms.squeeze(1)
        M_frames = ms.shape[-1]

        l_rms = 10.0 * torch.log10(ms.clamp(min=eps))  # (B, M)

        w_tilde = torch.zeros((B, M_frames), device=device, dtype=torch.float32)
        active_mask = torch.zeros((B, M_frames), device=device, dtype=torch.bool)
        is_valid = torch.zeros((B,), device=device, dtype=torch.float32)

        m_min = min(int(min_active_frames), M_frames)

        for b in range(B):
            l_b = l_rms[b]
            l_max = float(torch.max(l_b).item())
            if l_max < float(silence_floor_dbfs):
                # Silent chunk: all frames equal weight, is_valid = 0.0
                w_tilde[b].fill_(1.0 / float(max(M_frames, 1)))
                active_mask[b].fill_(True)
                is_valid[b] = 0.0
            else:
                theta_act = max(float(active_floor_dbfs), l_max - float(dynamic_span_db))
                raw_mask = l_b >= theta_act
                n_active = int(torch.sum(raw_mask).item())

                if n_active >= m_min:
                    active_mask[b] = raw_mask
                else:
                    top_idx = torch.topk(l_b, k=m_min, largest=True, sorted=False).indices
                    active_mask[b, top_idx] = True

                w_raw = torch.sigmoid((l_b - theta_act) / float(sigmoid_temp_db))
                w_sum = torch.sum(w_raw).clamp(min=eps)
                w_tilde[b] = (w_raw / w_sum).to(torch.float32)
                is_valid[b] = 1.0

        return w_tilde, active_mask, is_valid


class TransientEnvelopeLoss(nn.Module):
    """Multi-band Transient / Envelope Loss Module (§2.C.1).

    Extracts RMS envelopes on fast (5 ms) and slow (50 ms) timescales, local crest factor
    `C_c[m] = e_c^fast[m] - e_c^slow[m]`, and attack onset velocity
    `Delta^+ e_c^fast[m] = ReLU(e_c^fast[m] - e_c^fast[m-1])` across 3 bands:
      - Low:  [20, 160] Hz
      - Mid:  [160, 3200] Hz
      - High: [3200, 20000] Hz
    """

    def __init__(
        self,
        sample_rate: float = 48000.0,
        f_crossover_low: float = 160.0,
        f_crossover_high: float = 3200.0,
        tau_fast_sec: float = 0.005,
        tau_slow_sec: float = 0.050,
        hop_samples: int = 120,
        beta_atk: float = 0.5,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.sample_rate = float(sample_rate)
        self.f_crossover_low = float(f_crossover_low)
        self.f_crossover_high = float(f_crossover_high)
        self.tau_fast_sec = float(tau_fast_sec)
        self.tau_slow_sec = float(tau_slow_sec)
        self.hop_samples = int(hop_samples)
        self.beta_atk = float(beta_atk)
        self.eps = float(eps)

        self.w_fast_samples = int(round(self.tau_fast_sec * self.sample_rate))
        self.w_slow_samples = int(round(self.tau_slow_sec * self.sample_rate))

        self.crossover = DifferentiableLR4Crossover(sample_rate=self.sample_rate)
        self.register_buffer("fc_low", torch.tensor(self.f_crossover_low, dtype=torch.float32))
        self.register_buffer("fc_high", torch.tensor(self.f_crossover_high, dtype=torch.float32))

    def extract_band_envelopes(
        self, audio: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract multi-band local crest factor and positive onset velocity.

        Args:
            audio: Audio tensor `(B, C, T)` or `(C, T)`.

        Returns:
            Tuple `(crest_factors, onset_slopes)`:
              - `crest_factors`: Shape `(B, 3, M)` in dB.
              - `onset_slopes`: Shape `(B, 3, M)` in dB (non-negative).
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)
        B, C, T = audio.shape
        device = audio.device
        dtype = audio.dtype

        # 3-band split: each band is (B, C, T)
        y_low, y_mid, y_high = self.crossover(audio, self.fc_low, self.fc_high)
        bands = [y_low, y_mid, y_high]

        crest_list: List[torch.Tensor] = []
        onset_list: List[torch.Tensor] = []

        half_f = self.w_fast_samples // 2
        half_s = self.w_slow_samples // 2

        for band in bands:
            # Channel mean-square: (B, 1, T)
            s = torch.mean(band.to(torch.float64) ** 2, dim=1, keepdim=True)

            # Fast envelope via float64 cumsum
            s_pad_f = F.pad(s, (half_f, half_f), mode="replicate")
            cs_f = torch.cumsum(s_pad_f, dim=-1)
            cs_f_pad = F.pad(cs_f, (1, 0), mode="constant", value=0.0)
            idx_f = torch.arange(
                0, s_pad_f.shape[-1] - self.w_fast_samples + 1, self.hop_samples, device=device
            )
            ms_fast = (cs_f_pad[..., idx_f + self.w_fast_samples] - cs_f_pad[..., idx_f]) / float(
                self.w_fast_samples
            )

            # Slow envelope via float64 cumsum
            s_pad_s = F.pad(s, (half_s, half_s), mode="replicate")
            cs_s = torch.cumsum(s_pad_s, dim=-1)
            cs_s_pad = F.pad(cs_s, (1, 0), mode="constant", value=0.0)
            idx_s = torch.arange(
                0, s_pad_s.shape[-1] - self.w_slow_samples + 1, self.hop_samples, device=device
            )
            ms_slow = (cs_s_pad[..., idx_s + self.w_slow_samples] - cs_s_pad[..., idx_s]) / float(
                self.w_slow_samples
            )

            # Align frame counts
            M = min(ms_fast.shape[-1], ms_slow.shape[-1])
            ms_fast = ms_fast[..., :M].squeeze(1).clamp(min=self.eps)
            ms_slow = ms_slow[..., :M].squeeze(1).clamp(min=self.eps)

            e_fast_db = 10.0 * torch.log10(ms_fast)  # (B, M)
            e_slow_db = 10.0 * torch.log10(ms_slow)  # (B, M)

            # Local crest factor: C_c[m] = e_c^fast[m] - e_c^slow[m]
            c_c = e_fast_db - e_slow_db  # (B, M)

            # Positive onset derivative: Delta^+ e_c^fast[m] = ReLU(e_c^fast[m] - e_c^fast[m-1])
            diff_fast = e_fast_db[:, 1:] - e_fast_db[:, :-1]
            onset = F.relu(diff_fast)
            onset_padded = F.pad(onset, (1, 0), mode="constant", value=0.0)  # (B, M)

            crest_list.append(c_c.to(dtype=dtype))
            onset_list.append(onset_padded.to(dtype=dtype))

        crest_factors = torch.stack(crest_list, dim=1)  # (B, 3, M)
        onset_slopes = torch.stack(onset_list, dim=1)  # (B, 3, M)

        return crest_factors, onset_slopes

    def forward_same_track(
        self,
        pred_audio: torch.Tensor,
        target_audio: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute point-to-point active-frame-weighted Huber transient loss (§2.C.1).

        Args:
            pred_audio: Audio generated by the chain `(B, 2, T)`.
            target_audio: Aligned reference audio `(B, 2, T)`.

        Returns:
            Tuple `(loss, metrics_dict)`.
        """
        w_tilde, _, is_valid = compute_transient_active_mask(
            target_audio,
            sample_rate=self.sample_rate,
            hop_samples=self.hop_samples,
            window_samples=self.w_slow_samples,
            eps=self.eps,
        )

        crest_pred, onset_pred = self.extract_band_envelopes(pred_audio)
        with torch.no_grad():
            crest_ref, onset_ref = self.extract_band_envelopes(target_audio)

        # Align frame count between envelopes and w_tilde
        M = min(crest_pred.shape[-1], crest_ref.shape[-1], w_tilde.shape[-1])
        c_p = crest_pred[..., :M]
        c_r = crest_ref[..., :M]
        o_p = onset_pred[..., :M]
        o_r = onset_ref[..., :M]
        w = w_tilde[..., :M]
        # Re-normalize weights over trimmed M
        w = w / torch.sum(w, dim=-1, keepdim=True).clamp(min=self.eps)

        # Huber_1.0 on crest differences: (B, 3, M)
        huber_crest = F.huber_loss(c_p, c_r, reduction="none", delta=1.0)
        # Huber_1.0 on onset slope differences: (B, 3, M)
        huber_onset = F.huber_loss(o_p, o_r, reduction="none", delta=1.0)

        # Weighted average over frames and mean over 3 bands
        loss_crest_bands = torch.sum(huber_crest * w.unsqueeze(1), dim=-1)  # (B, 3)
        loss_onset_bands = torch.sum(huber_onset * w.unsqueeze(1), dim=-1)  # (B, 3)

        loss_crest = torch.mean(loss_crest_bands, dim=-1) * is_valid  # (B,)
        loss_onset = torch.mean(loss_onset_bands, dim=-1) * is_valid  # (B,)

        total_loss = torch.mean(loss_crest + self.beta_atk * loss_onset)

        metrics = {
            "trans_same_crest_loss": torch.mean(loss_crest).item(),
            "trans_same_onset_loss": torch.mean(loss_onset).item(),
            "trans_same_total_loss": total_loss.item(),
        }
        return total_loss, metrics

    def forward_stylistic(
        self,
        pred_audio: torch.Tensor,
        ref_audio: torch.Tensor,
        mix_anchor_audio: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute multi-quantile (P50, P90, P95) distributional transient loss (§2.C.1).

        Args:
            pred_audio: Audio generated by the chain `(B, 2, T_pred)`.
            ref_audio: Stylistic reference audio `(B, 2, T_ref)`.
            mix_anchor_audio: Optional unoptimized mix chunk `(B, 2, T_pred)` for anchor mask.

        Returns:
            Tuple `(loss, metrics_dict)`.
        """
        anchor = mix_anchor_audio if mix_anchor_audio is not None else pred_audio.detach()

        _, mask_pred, is_valid_pred = compute_transient_active_mask(
            anchor,
            sample_rate=self.sample_rate,
            hop_samples=self.hop_samples,
            window_samples=self.w_slow_samples,
            eps=self.eps,
        )
        _, mask_ref, is_valid_ref = compute_transient_active_mask(
            ref_audio,
            sample_rate=self.sample_rate,
            hop_samples=self.hop_samples,
            window_samples=self.w_slow_samples,
            eps=self.eps,
        )
        is_valid = is_valid_pred * is_valid_ref  # (B,)

        crest_p, onset_p = self.extract_band_envelopes(pred_audio)
        with torch.no_grad():
            crest_r, onset_r = self.extract_band_envelopes(ref_audio)

        B = pred_audio.shape[0] if pred_audio.dim() == 3 else 1
        q_probs = torch.tensor([0.50, 0.90, 0.95], device=pred_audio.device, dtype=pred_audio.dtype)

        crest_dist_list: List[torch.Tensor] = []
        onset_dist_list: List[torch.Tensor] = []

        for b in range(B):
            # Active frames for pred
            m_p = min(crest_p.shape[-1], mask_pred.shape[-1])
            act_idx_p = torch.nonzero(mask_pred[b, :m_p], as_tuple=False).squeeze(-1)
            # Active frames for ref
            m_r = min(crest_r.shape[-1], mask_ref.shape[-1])
            act_idx_r = torch.nonzero(mask_ref[b, :m_r], as_tuple=False).squeeze(-1)

            # Compute quantiles across active frames: shape (3_quantiles, 3_bands)
            q_cp = torch.quantile(crest_p[b, :, act_idx_p], q_probs, dim=-1)
            q_cr = torch.quantile(crest_r[b, :, act_idx_r], q_probs, dim=-1)

            q_op = torch.quantile(onset_p[b, :, act_idx_p], q_probs, dim=-1)
            q_or = torch.quantile(onset_r[b, :, act_idx_r], q_probs, dim=-1)

            # L1 difference averaged across 3 quantiles and 3 bands
            diff_c = torch.mean(torch.abs(q_cp - q_cr))
            diff_o = torch.mean(torch.abs(q_op - q_or))

            crest_dist_list.append(diff_c * is_valid[b])
            onset_dist_list.append(diff_o * is_valid[b])

        loss_crest = torch.mean(torch.stack(crest_dist_list))
        loss_onset = torch.mean(torch.stack(onset_dist_list))
        total_loss = loss_crest + self.beta_atk * loss_onset

        metrics = {
            "trans_style_crest_loss": loss_crest.item(),
            "trans_style_onset_loss": loss_onset.item(),
            "trans_style_total_loss": total_loss.item(),
        }
        return total_loss, metrics

    def forward(
        self,
        pred_audio: torch.Tensor,
        target_audio: torch.Tensor,
        reference_mode: str = "same_track",
        mix_anchor_audio: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Dispatch forward pass according to reference mode."""
        if reference_mode == "same_track":
            return self.forward_same_track(pred_audio, target_audio)
        elif reference_mode == "stylistic":
            return self.forward_stylistic(pred_audio, target_audio, mix_anchor_audio=mix_anchor_audio)
        else:
            raise ValueError(
                f"Unsupported reference_mode '{reference_mode}'. Expected 'same_track' or 'stylistic'."
            )
