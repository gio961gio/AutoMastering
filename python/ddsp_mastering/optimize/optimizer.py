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
        plr_mode: str = "box",
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
        self.plr_mode = plr_mode

        self.chain = DifferentiableMasteringChain(sample_rate=sample_rate)
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
            plr_mode=same_plr_mode,
        )
        self.stylistic_loss_fn = AggregatePerceptualLoss(
            sample_rate=sample_rate,
            weight_spectral=weight_stft,
            weight_loudness=weight_loudness,
            weight_param_reg=weight_param_reg,
            spectral_mode=stylistic_spectral_mode,
            weight_de_harsh=weight_de_harsh,
            weight_trans=weight_trans,
            plr_mode=style_plr_mode,
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
        effective_target_lufs = self.target_lufs
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

            for k in range(K):
                mix_k = mix_chunks[k]
                n_pre_k = preroll_counts[k]
                pred_ext_k = self.chain(mix_k, z=z, eq_mode=eq_mode)
                pred_k = pred_ext_k[..., n_pre_k:] if n_pre_k > 0 else pred_ext_k
                mix_eval_k = mix_k[..., n_pre_k:] if n_pre_k > 0 else mix_k

                if reference_mode == "stylistic":
                    ref_k = ref_chunks[k % len(ref_chunks)] if ref_chunks else None
                    if ref_k is None:
                        raise ValueError("Reference audio chunk required for stylistic mode")
                    chunk_loss, chunk_metrics = self.stylistic_loss_fn(
                        pred_audio=pred_k,
                        ref_audio=ref_k,
                        normalized_params=z,
                        target_lufs=effective_target_lufs,
                        mix_anchor_audio=mix_eval_k,
                    )
                    step_stft += weights[k] * chunk_metrics.get("stylistic_spectral_loss", 0.0)
                else:
                    ref_k = ref_chunks[k] if ref_chunks and k < len(ref_chunks) else None
                    chunk_loss, chunk_metrics = self.loss_fn(
                        pred_audio=pred_k,
                        target_audio=ref_k,
                        target_lufs=effective_target_lufs,
                        normalized_params=z,
                        align_latency=align_latency,
                    )
                    step_stft += weights[k] * chunk_metrics.get("stft_loss_total", 0.0)

                step_loss = step_loss + weights[k] * chunk_loss
                step_lufs += weights[k] * chunk_metrics.get("lufs_pred", 0.0)

            step_loss.backward()

            # Gradient clipping to prevent sudden spikes
            torch.nn.utils.clip_grad_norm_([u], max_norm=1.0)

            optimizer.step()
            scheduler.step()

            history["loss"].append(step_loss.item())
            history["stft_loss"].append(step_stft)
            history["lufs"].append(step_lufs)
            history["lr"].append(scheduler.get_last_lr()[0])

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
