"""End-to-End Self-Supervised AutoMastering Optimizer with Multi-Chunk Evaluation.

Optimizes 41-dimensional mastering parameters in unconstrained logit space
u in R^41 (mapped via z = sigmoid(u)) using Adam with Cosine Annealing,
gradient clipping, and multi-objective perceptual loss guidance.

Supports both:
  - "same_track": Paired time-aligned chunks with multi-resolution STFT & LUFS loss.
  - "stylistic": Aggregate perceptual feature loss across representative song windows.
"""

import json
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.optim as optim

from ddsp_mastering.dsp.chain import DifferentiableMasteringChain
from ddsp_mastering.loss.combined import MasteringPerceptualLoss
from ddsp_mastering.loss.stylistic import AggregatePerceptualLoss
from ddsp_mastering.loss.eq_prior import CompositeEQPriorLoss
from ddsp_mastering.loss.dynamic_prior import V3bDynamicPriorLoss
from ddsp_mastering.param_space import (
    PARAM_SPECS,
    NUM_PARAMS,
    Z_NEUTRAL,
    map_z_to_theta,
    unpack_theta_dict,
)
from ddsp_mastering.optimize.metrics import MasteringBenchmarkSuite


class AutoMasteringOptimizer:
    """Self-supervised optimization engine for track mastering."""

    def __init__(
        self,
        sample_rate: float = 48000.0,
        lr: float = 0.05,
        num_steps: int = 50,
        target_lufs: Optional[float] = None,
        weight_stft: float = 1.0,
        weight_loudness: float = 0.8,
        weight_param_reg: float = 0.02,
        use_hybrid_eq: bool = True,
        weight_erb: float = 0.0,
        stylistic_spectral_mode: str = "mel40",
        weight_de_harsh: float = 0.0,
        weight_trans: float = 0.0,
        beta_atk: float = 0.5,
        weight_eq_prior: float = 0.0,
        eq_tau_shape: float = 1.5,
        eq_tau_offset: float = 1.0,
        eq_w_slope: float = 2.0,
        eq_w_offset: float = 1.0,
        plr_mode: str = "box",
        plr_d_style: float = 1.0,
        plr_gamma_floor: float = 1.0,
        soft_peak_p: int = 8,
        soft_peak_beta: float = 2.0,
        soft_peak_block_duration_sec: float = 0.010,
        v3b_w_lim: float = 0.0,
        v3b_w_diff: float = 0.0,
        v3b_w_conflict: float = 0.0,
        v3b_tau_avg: float = 0.5,
        v3b_tau_95: float = 2.0,
        v3b_lambda_95: float = 0.5,
        v3b_delta_max: float = 3.0,
        v3b_delta_m: float = 1.25,
        v3b_delta_ref: float = 3.0,
    ):
        self.sample_rate = sample_rate
        self.lr = lr
        self.num_steps = num_steps
        self.target_lufs = target_lufs
        self.use_hybrid_eq = use_hybrid_eq
        self.weight_erb = weight_erb
        self.stylistic_spectral_mode = stylistic_spectral_mode
        self.weight_de_harsh = weight_de_harsh
        self.weight_trans = weight_trans
        self.beta_atk = beta_atk
        self.weight_eq_prior = weight_eq_prior
        self.eq_tau_shape = eq_tau_shape
        self.eq_tau_offset = eq_tau_offset
        self.eq_w_slope = eq_w_slope
        self.eq_w_offset = eq_w_offset
        self.plr_mode = plr_mode
        self.plr_d_style = plr_d_style
        self.plr_gamma_floor = plr_gamma_floor
        self.soft_peak_p = soft_peak_p
        self.soft_peak_beta = soft_peak_beta
        self.soft_peak_block_duration_sec = soft_peak_block_duration_sec
        self.weight_param_reg = weight_param_reg
        self.weight_reg = weight_param_reg

        self.v3b_w_lim = float(v3b_w_lim)
        self.v3b_w_diff = float(v3b_w_diff)
        self.v3b_w_conflict = float(v3b_w_conflict)
        self.v3b_tau_avg = float(v3b_tau_avg)
        self.v3b_tau_95 = float(v3b_tau_95)
        self.v3b_lambda_95 = float(v3b_lambda_95)
        self.v3b_delta_max = float(v3b_delta_max)
        self.v3b_delta_m = float(v3b_delta_m)
        self.v3b_delta_ref = float(v3b_delta_ref)

        self.enable_v3b = bool(v3b_w_lim > 0.0 or v3b_w_diff > 0.0 or v3b_w_conflict > 0.0)
        self.v3b_prior_loss = V3bDynamicPriorLoss(
            sample_rate=sample_rate,
            tau_avg=v3b_tau_avg,
            tau_95=v3b_tau_95,
            lambda_95=v3b_lambda_95,
            delta_max=v3b_delta_max,
            delta_m=v3b_delta_m,
            delta_ref=v3b_delta_ref,
            w_lim=v3b_w_lim,
            w_diff=v3b_w_diff,
            w_conflict=v3b_w_conflict,
        )

        self.chain = DifferentiableMasteringChain(sample_rate=sample_rate)
        self.eq_prior_loss = CompositeEQPriorLoss(
            sample_rate=sample_rate,
            tau_shape=eq_tau_shape,
            tau_offset=eq_tau_offset,
            w_slope=eq_w_slope,
            w_offset=eq_w_offset,
        )
        default_lufs = target_lufs if target_lufs is not None else -14.0

        same_plr_mode = "reference_aware_same" if plr_mode in ("reference_aware", "reference_aware_same") else "box"
        style_plr_mode = "reference_aware" if plr_mode in ("reference_aware", "reference_aware_style") else "mse"

        self.loss_fn = MasteringPerceptualLoss(
            sample_rate=sample_rate,
            target_lufs=default_lufs,
            weight_stft=weight_stft,
            weight_loudness=weight_loudness,
            weight_param_reg=weight_param_reg,
            weight_erb=weight_erb,
            weight_de_harsh=weight_de_harsh,
            weight_trans=weight_trans,
            beta_atk=beta_atk,
            weight_eq_prior=weight_eq_prior,
            eq_tau_shape=eq_tau_shape,
            eq_tau_offset=eq_tau_offset,
            eq_w_slope=eq_w_slope,
            eq_w_offset=eq_w_offset,
            plr_mode=same_plr_mode,
            plr_d_style=plr_d_style,
            plr_gamma_floor=plr_gamma_floor,
            soft_peak_p=soft_peak_p,
            soft_peak_beta=soft_peak_beta,
            soft_peak_block_duration_sec=soft_peak_block_duration_sec,
        )
        self.stylistic_loss_fn = AggregatePerceptualLoss(
            sample_rate=sample_rate,
            weight_spectral=weight_stft,
            weight_loudness=weight_loudness,
            weight_param_reg=weight_param_reg,
            spectral_mode=stylistic_spectral_mode,
            weight_de_harsh=weight_de_harsh,
            weight_trans=weight_trans,
            beta_atk=beta_atk,
            weight_eq_prior=weight_eq_prior,
            eq_tau_shape=eq_tau_shape,
            eq_tau_offset=eq_tau_offset,
            eq_w_slope=eq_w_slope,
            eq_w_offset=eq_w_offset,
            plr_mode=style_plr_mode,
            plr_d_style=plr_d_style,
            plr_gamma_floor=plr_gamma_floor,
            soft_peak_p=soft_peak_p,
            soft_peak_beta=soft_peak_beta,
            soft_peak_block_duration_sec=soft_peak_block_duration_sec,
        )
        self.benchmark = MasteringBenchmarkSuite(sample_rate=sample_rate)

    def optimize_multi_chunk(
        self,
        mix_chunks: List[torch.Tensor],
        ref_chunks: Optional[List[torch.Tensor]] = None,
        reference_mode: str = "same_track",
        chunk_weights: Optional[List[float]] = None,
        initial_z: Optional[torch.Tensor] = None,
        verbose: bool = False,
        align_latency: bool = True,
        mix_preroll_samples: Optional[List[int]] = None,
        target_lufs: Optional[float] = None,
        ref_ext_chunks: Optional[List[torch.Tensor]] = None,
        ref_preroll_list: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, List[float]]]:
        """Run optimization loop across multiple representative audio chunks.

        Args:
            mix_chunks: List of unmastered chunk tensors [(1, 2, T_k), ...].
                        If `mix_preroll_samples` is provided, each `mix_chunks[k]`
                        includes `mix_preroll_samples[k]` causal warm-up samples
                        prior to the evaluation window `[s_k, e_k]`.
            ref_chunks: List of reference chunk tensors [(1, 2, T_k), ...], if available.
            reference_mode: "same_track" (point-to-point) or "stylistic" (aggregate features).
            chunk_weights: Optional weights per chunk. Normalized automatically to sum to 1.0.
            initial_z: Optional initial parameters z in [0, 1]^41.
            verbose: Whether to print progress during optimization.
            align_latency: Whether to slice off chain latency (same_track only).
            mix_preroll_samples: Optional list of causal pre-roll sample counts per chunk.
                                 After running `self.chain(mix_k)`, the first `n_pre_k`
                                 samples are stripped so the loss is evaluated strictly
                                 on the warmed-up evaluation window `[s_k, e_k]`.

        Returns:
            Tuple of (optimal_z, list_of_mastered_chunks, loss_history).
        """
        assert len(mix_chunks) > 0, "Expected at least 1 audio chunk"
        for i, c in enumerate(mix_chunks):
            assert c.dim() == 3 and c.shape[0] == 1, f"Chunk {i} must have shape (1, 2, T)"

        K = len(mix_chunks)
        device = mix_chunks[0].device

        if mix_preroll_samples is None:
            preroll_counts = [0] * K
        else:
            assert len(mix_preroll_samples) == K, "mix_preroll_samples length must match mix_chunks length"
            preroll_counts = [max(0, int(n)) for n in mix_preroll_samples]
            for i, (c, n_pre) in enumerate(zip(mix_chunks, preroll_counts)):
                assert c.shape[-1] > n_pre, (
                    f"Chunk {i} length ({c.shape[-1]}) must exceed its pre-roll ({n_pre})"
                )

        # Setup chunk weighting
        if chunk_weights is None:
            weights = [1.0 / K] * K
        else:
            assert len(chunk_weights) == K, "chunk_weights length must match mix_chunks length"
            total_w = sum(chunk_weights)
            assert total_w > 0, "Sum of chunk weights must be positive"
            weights = [w / total_w for w in chunk_weights]

        # Determine effective target LUFS
        effective_target_lufs = target_lufs if target_lufs is not None else self.target_lufs
        if effective_target_lufs is None:
            if ref_chunks is not None and len(ref_chunks) > 0:
                with torch.no_grad():
                    lufs_vals = [
                        self.loss_fn.loudness_loss.calculate_lufs(rc).item()
                        for rc in ref_chunks
                    ]
                    effective_target_lufs = float(sum(lufs_vals) / len(lufs_vals))
            else:
                effective_target_lufs = -14.0

        # Initialize in unconstrained logit space: u = logit(z)
        if initial_z is None:
            z_init = torch.clamp(Z_NEUTRAL.to(device=device), 1e-4, 1.0 - 1e-4)
            u_init = torch.log(z_init / (1.0 - z_init))
            u = nn.Parameter(u_init.clone().detach())
        else:
            z_clamped = torch.clamp(initial_z.reshape(NUM_PARAMS).to(device=device), 1e-4, 1.0 - 1e-4)
            u_init = torch.log(z_clamped / (1.0 - z_clamped))
            u = nn.Parameter(u_init.clone().detach())

        optimizer = optim.Adam([u], lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.num_steps, eta_min=self.lr * 0.05
        )

        history: Dict[str, List[float]] = {
            "loss": [],
            "stft_loss": [],
            "lufs": [],
            "lr": [],
        }
        if self.enable_v3b:
            history["v3b_loss"] = []
            history["v3b_l_lim_raw"] = []
            history["v3b_l_lim_weighted"] = []
            history["v3b_l_diff_raw"] = []
            history["v3b_l_diff_weighted"] = []
            history["v3b_l_conflict_raw"] = []
            history["v3b_l_conflict_weighted"] = []

        for step in range(self.num_steps):
            optimizer.zero_grad()

            # Map unconstrained logit u to valid normalized space z in (0, 1)
            z = torch.sigmoid(u)

            # Determine EQ mode: fast FFT during coarse steps, IIR during fine steps
            if self.use_hybrid_eq and step < int(self.num_steps * 0.7):
                eq_mode = "fft"
            else:
                eq_mode = "iir"

            step_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            step_stft = 0.0
            step_lufs = 0.0

            if self.enable_v3b:
                step_v3b_total = 0.0
                step_v3b_lim_raw = 0.0
                step_v3b_lim_w = 0.0
                step_v3b_diff_raw = 0.0
                step_v3b_diff_w = 0.0
                step_v3b_conf_raw = 0.0
                step_v3b_conf_w = 0.0

            for k in range(K):
                mix_k = mix_chunks[k]
                n_pre_k = preroll_counts[k]

                if self.enable_v3b:
                    pred_ext_k, diag_k = self.chain(mix_k, z=z, eq_mode=eq_mode, return_diagnostics=True)
                else:
                    pred_ext_k = self.chain(mix_k, z=z, eq_mode=eq_mode)

                pred_k = pred_ext_k[..., n_pre_k:] if n_pre_k > 0 else pred_ext_k
                mix_eval_k = mix_k[..., n_pre_k:] if n_pre_k > 0 else mix_k

                ref_k = ref_chunks[k % len(ref_chunks)] if ref_chunks else None
                ref_ext_k = ref_ext_chunks[k % len(ref_ext_chunks)] if ref_ext_chunks else None
                ref_n_pre_k = ref_preroll_list[k % len(ref_preroll_list)] if ref_preroll_list else 0

                if reference_mode == "stylistic":
                    if ref_k is None:
                        raise ValueError("Reference audio chunk required for stylistic mode")
                    chunk_loss, chunk_metrics = self.stylistic_loss_fn(
                        pred_audio=pred_k,
                        ref_audio=ref_k,
                        normalized_params=z,
                        target_lufs=effective_target_lufs,
                        mix_anchor_audio=mix_eval_k,
                        pred_ext_audio=pred_ext_k,
                        pred_preroll_samples=n_pre_k,
                        ref_ext_audio=ref_ext_k,
                        ref_preroll_samples=ref_n_pre_k,
                        include_eq_prior=False,
                    )
                    step_stft += weights[k] * chunk_metrics.get("stylistic_spectral_loss", 0.0)
                else:
                    chunk_loss, chunk_metrics = self.loss_fn(
                        pred_audio=pred_k,
                        target_audio=ref_k,
                        target_lufs=effective_target_lufs,
                        normalized_params=z,
                        align_latency=align_latency,
                        pred_ext_audio=pred_ext_k,
                        pred_preroll_samples=n_pre_k,
                        ref_ext_audio=ref_ext_k,
                        ref_preroll_samples=ref_n_pre_k,
                        include_eq_prior=False,
                    )
                    step_stft += weights[k] * chunk_metrics.get("stft_loss_total", 0.0)

                if self.enable_v3b:
                    l_lim_k, l_diff_k, l_conf_k, l_v3b_k, v3b_m_k = self.v3b_prior_loss.compute_chunk_terms(
                        mix_audio=mix_k,
                        diagnostics=diag_k,
                        z=z,
                        preroll_samples=n_pre_k,
                    )
                    chunk_loss = chunk_loss + l_v3b_k.to(chunk_loss.dtype)
                    step_v3b_total += weights[k] * float(v3b_m_k["loss_dyn"])
                    step_v3b_lim_raw += weights[k] * float(v3b_m_k["loss_lim"])
                    step_v3b_lim_w += weights[k] * (self.v3b_w_lim * float(v3b_m_k["loss_lim"]))
                    step_v3b_diff_raw += weights[k] * float(v3b_m_k["loss_diff"])
                    step_v3b_diff_w += weights[k] * (self.v3b_w_diff * float(v3b_m_k["loss_diff"]))
                    step_v3b_conf_raw += weights[k] * float(v3b_m_k["loss_conflict"])
                    step_v3b_conf_w += weights[k] * (self.v3b_w_conflict * float(v3b_m_k["loss_conflict"]))

                step_loss = step_loss + weights[k] * chunk_loss
                step_lufs += weights[k] * chunk_metrics.get("lufs_pred", 0.0)

            # Single evaluation of EQ composite prior outside chunk loop
            if self.weight_eq_prior > 0.0:
                eq_prior_val, eq_prior_metrics = self.eq_prior_loss(z)
                step_loss = step_loss + self.weight_eq_prior * eq_prior_val.to(step_loss.dtype)

            if not torch.isfinite(step_loss):
                raise RuntimeError(f"Step {step}: non-finite loss encountered: {step_loss.item()}")

            step_loss.backward()

            if not torch.isfinite(u.grad).all():
                raise RuntimeError(f"Step {step}: non-finite gradients encountered in u.grad")

            # Gradient clipping to prevent sudden spikes
            torch.nn.utils.clip_grad_norm_([u], max_norm=1.0)

            optimizer.step()
            scheduler.step()

            history["loss"].append(step_loss.item())
            history["stft_loss"].append(step_stft)
            history["lufs"].append(step_lufs)
            history["lr"].append(scheduler.get_last_lr()[0])
            if self.enable_v3b:
                history["v3b_loss"].append(step_v3b_total)
                history["v3b_l_lim_raw"].append(step_v3b_lim_raw)
                history["v3b_l_lim_weighted"].append(step_v3b_lim_w)
                history["v3b_l_diff_raw"].append(step_v3b_diff_raw)
                history["v3b_l_diff_weighted"].append(step_v3b_diff_w)
                history["v3b_l_conflict_raw"].append(step_v3b_conf_raw)
                history["v3b_l_conflict_weighted"].append(step_v3b_conf_w)

            if verbose and (step % 5 == 0 or step == self.num_steps - 1):
                print(
                    f"Step {step:02d}/{self.num_steps} | Loss: {step_loss.item():.4f} | "
                    f"LUFS: {step_lufs:.2f} (Target: {effective_target_lufs:.2f}) | "
                    f"LR: {scheduler.get_last_lr()[0]:.4f}",
                    flush=True,
                )

        optimal_z = torch.sigmoid(u).detach()
        with torch.no_grad():
            final_chunks = []
            for k, c in enumerate(mix_chunks):
                n_pre_k = preroll_counts[k]
                out_k = self.chain(c, z=optimal_z, eq_mode="iir")
                final_chunks.append(out_k[..., n_pre_k:] if n_pre_k > 0 else out_k)

        return optimal_z, final_chunks, history

    def _eval_loss_iir(
        self,
        mix_chunks: List[torch.Tensor],
        ref_chunks: Optional[List[torch.Tensor]] = None,
        z: Optional[torch.Tensor] = None,
        reference_mode: str = "same_track",
        chunk_weights: Optional[List[float]] = None,
        target_lufs: Optional[float] = None,
        mix_preroll_samples: Optional[List[int]] = None,
        ref_ext_chunks: Optional[List[torch.Tensor]] = None,
        ref_preroll_list: Optional[List[int]] = None,
        align_latency: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Evaluate full composite objective in IIR mode across chunks with single EQ prior evaluation."""
        if z is None:
            z = Z_NEUTRAL.to(device=mix_chunks[0].device)
        K = len(mix_chunks)
        weights = [1.0 / K] * K if chunk_weights is None else [w / sum(chunk_weights) for w in chunk_weights]
        preroll_counts = [0] * K if mix_preroll_samples is None else [max(0, int(n)) for n in mix_preroll_samples]

        tot_loss = torch.tensor(0.0, device=mix_chunks[0].device, dtype=torch.float32)
        agg_metrics: Dict[str, float] = {}

        for k in range(K):
            mk = mix_chunks[k]
            n_pre = preroll_counts[k]
            if self.enable_v3b:
                y_ext, diag_k = self.chain(mk, z=z, eq_mode="iir", return_diagnostics=True)
            else:
                y_ext = self.chain(mk, z=z, eq_mode="iir")
            y_k = y_ext[..., n_pre:] if n_pre > 0 else y_ext
            ref_k = ref_chunks[k % len(ref_chunks)] if ref_chunks else None
            ref_ext_k = ref_ext_chunks[k % len(ref_ext_chunks)] if ref_ext_chunks else None
            ref_n_pre_k = ref_preroll_list[k % len(ref_preroll_list)] if ref_preroll_list else 0

            if reference_mode == "stylistic":
                anchor_k = mk[..., n_pre:] if n_pre > 0 else mk
                lk, mk_dict = self.stylistic_loss_fn(
                    pred_audio=y_k,
                    ref_audio=ref_k,
                    normalized_params=z,
                    target_lufs=target_lufs,
                    mix_anchor_audio=anchor_k,
                    pred_ext_audio=y_ext,
                    pred_preroll_samples=n_pre,
                    ref_ext_audio=ref_ext_k,
                    ref_preroll_samples=ref_n_pre_k,
                    include_eq_prior=False,
                )
            else:
                lk, mk_dict = self.loss_fn(
                    pred_audio=y_k,
                    target_audio=ref_k,
                    target_lufs=target_lufs,
                    normalized_params=z,
                    align_latency=align_latency,
                    pred_ext_audio=y_ext,
                    pred_preroll_samples=n_pre,
                    ref_ext_audio=ref_ext_k,
                    ref_preroll_samples=ref_n_pre_k,
                    include_eq_prior=False,
                )

            if self.enable_v3b:
                l_lim_k, l_diff_k, l_conf_k, l_v3b_k, v3b_m_k = self.v3b_prior_loss.compute_chunk_terms(
                    mix_audio=mk,
                    diagnostics=diag_k,
                    z=z,
                    preroll_samples=n_pre,
                )
                lk = lk + l_v3b_k.to(lk.dtype)
                for v_key, v_val in v3b_m_k.items():
                    agg_metrics[f"v3b_{v_key}"] = agg_metrics.get(f"v3b_{v_key}", 0.0) + weights[k] * float(v_val)

            tot_loss = tot_loss + weights[k] * lk
            for mk_key, mk_val in mk_dict.items():
                agg_metrics[mk_key] = agg_metrics.get(mk_key, 0.0) + weights[k] * float(mk_val)

        if self.weight_eq_prior > 0.0:
            eq_val, eq_metrics = self.eq_prior_loss(z)
            tot_loss = tot_loss + self.weight_eq_prior * eq_val.to(tot_loss.dtype)
            agg_metrics["eq_prior_loss"] = float(eq_val.item())
            for k_m, v_m in eq_metrics.items():
                agg_metrics[f"eq_prior_{k_m}"] = v_m

        agg_metrics["total_loss"] = float(tot_loss.item())
        return tot_loss, agg_metrics

    def optimize(
        self,
        raw_mix: torch.Tensor,
        target_reference: Optional[torch.Tensor] = None,
        initial_z: Optional[torch.Tensor] = None,
        verbose: bool = False,
        align_latency: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, List[float]]]:
        """Backward-compatible single-window optimization loop."""
        ref_list = [target_reference] if target_reference is not None else None
        opt_z, mastered_chunks, history = self.optimize_multi_chunk(
            mix_chunks=[raw_mix],
            ref_chunks=ref_list,
            reference_mode="same_track",
            initial_z=initial_z,
            verbose=verbose,
            align_latency=align_latency,
        )
        return opt_z, mastered_chunks[0], history

    def master_full_track(self, full_mix: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Apply optimized parameters z to master the full audio track without truncation."""
        with torch.no_grad():
            mastered = self.chain(full_mix, z=z, eq_mode="iir")
        return mastered

    def export_preset_json(self, z: torch.Tensor, output_path: Optional[str] = None) -> str:
        """Export normalized parameters z to canonical JSON preset format interoperable with JUCE."""
        z_cpu = z.detach().cpu().reshape(-1)
        theta = map_z_to_theta(z_cpu)

        preset_data = {
            "format": "AutoMastering_Preset_v1",
            "version": "1.0",
            "sample_rate": float(self.sample_rate),
            "normalized_z": z_cpu.tolist(),
            "parameters": {},
            "parameter_details": {},
        }

        # Populate with parameter IDs: both flat numeric values for JUCE and rich metadata
        for i, spec in enumerate(PARAM_SPECS):
            param_id = spec.name
            val = float(theta[i].item())
            preset_data["parameters"][param_id] = val
            preset_data[param_id] = val  # top-level fallback
            preset_data["parameter_details"][param_id] = {
                "index": i,
                "value": val,
                "min": spec.min_val,
                "max": spec.max_val,
                "is_log": spec.is_log,
                "unit": spec.unit,
            }

        json_str = json.dumps(preset_data, indent=2)
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(json_str)

        return json_str
