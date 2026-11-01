"""End-to-End Self-Supervised AutoMastering Optimizer.

Optimizes 41-dimensional mastering parameters in unconstrained logit space
u in R^41 (mapped via z = sigmoid(u)) using Adam with Cosine Annealing,
gradient clipping, and multi-objective perceptual loss guidance.
"""

import json
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim

from ddsp_mastering.dsp.chain import DifferentiableMasteringChain
from ddsp_mastering.loss.combined import MasteringPerceptualLoss
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
    ):
        self.sample_rate = sample_rate
        self.lr = lr
        self.num_steps = num_steps
        self.target_lufs = target_lufs
        self.use_hybrid_eq = use_hybrid_eq

        self.chain = DifferentiableMasteringChain(sample_rate=sample_rate)
        default_lufs = target_lufs if target_lufs is not None else -14.0
        self.loss_fn = MasteringPerceptualLoss(
            sample_rate=sample_rate,
            target_lufs=default_lufs,
            weight_stft=weight_stft,
            weight_loudness=weight_loudness,
            weight_param_reg=weight_param_reg,
        )
        self.benchmark = MasteringBenchmarkSuite(sample_rate=sample_rate)

    def optimize(
        self,
        raw_mix: torch.Tensor,
        target_reference: Optional[torch.Tensor] = None,
        initial_z: Optional[torch.Tensor] = None,
        verbose: bool = False,
        align_latency: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, List[float]]]:
        """Run optimization loop to find optimal mastering parameters z*.

        Args:
            raw_mix: Unmastered audio tensor (1, 2, T).
            target_reference: Optional reference target master audio (1, 2, T).
            initial_z: Optional initial parameters z in [0, 1]^41 (defaults to neutral 0.5).
            verbose: Whether to print progress during optimization.
            align_latency: Whether to compensate for chain latency when comparing against target_reference.
                Set to False when target_reference is already delayed by the chain.

        Returns:
            Tuple of (optimal_z, mastered_audio, loss_history).
        """
        assert raw_mix.dim() == 3 and raw_mix.shape[0] == 1, "Expected shape (1, 2, T)"

        # Determine target LUFS: if self.target_lufs is explicitly set, use it;
        # otherwise, if target_reference is provided, match its LUFS; else default to -14.0
        effective_target_lufs = self.target_lufs
        if effective_target_lufs is None:
            if target_reference is not None:
                with torch.no_grad():
                    effective_target_lufs = self.loss_fn.loudness_loss.calculate_lufs(target_reference).item()
            else:
                effective_target_lufs = -14.0

        # Initialize in unconstrained logit space: u = logit(z)
        if initial_z is None:
            # Physically neutral mastering preset baseline
            z_init = torch.clamp(Z_NEUTRAL.to(device=raw_mix.device), 1e-4, 1.0 - 1e-4)
            u_init = torch.log(z_init / (1.0 - z_init))
            u = nn.Parameter(u_init.clone().detach())
        else:
            # Inverse sigmoid: log(z / (1 - z))
            z_clamped = torch.clamp(initial_z.reshape(NUM_PARAMS), 1e-4, 1.0 - 1e-4)
            u_init = torch.log(z_clamped / (1.0 - z_clamped))
            u = nn.Parameter(u_init.clone().detach().to(device=raw_mix.device))

        optimizer = optim.Adam([u], lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_steps, eta_min=self.lr * 0.05)

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
                eq_mode = "iir"  # Fine steps use true IIR biquad filters matching C++

            pred_audio = self.chain(raw_mix, z=z, eq_mode=eq_mode)

            loss, metrics = self.loss_fn(
                pred_audio=pred_audio,
                target_audio=target_reference,
                target_lufs=effective_target_lufs,
                normalized_params=z,
                align_latency=align_latency,
            )

            loss.backward()

            # Gradient clipping to prevent sudden spikes
            torch.nn.utils.clip_grad_norm_([u], max_norm=1.0)

            optimizer.step()
            scheduler.step()

            # Record history
            history["loss"].append(loss.item())
            if "stft_loss_total" in metrics:
                history["stft_loss"].append(metrics["stft_loss_total"])
            if "lufs_pred" in metrics:
                history["lufs"].append(metrics["lufs_pred"])
            history["lr"].append(scheduler.get_last_lr()[0])

            if verbose and (step % 5 == 0 or step == self.num_steps - 1):
                print(
                    f"Step {step:02d}/{self.num_steps} | Loss: {loss.item():.4f} | "
                    f"LUFS: {metrics.get('lufs_pred', 0.0):.2f} (Target: {effective_target_lufs:.2f}) | "
                    f"LR: {scheduler.get_last_lr()[0]:.4f}",
                    flush=True
                )

        optimal_z = torch.sigmoid(u).detach()
        with torch.no_grad():
            final_master = self.chain(raw_mix, z=optimal_z, eq_mode="iir")

        return optimal_z, final_master, history

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
