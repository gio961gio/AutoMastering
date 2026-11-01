"""Self-Supervised Synthetic Mastering Pair Generator.

Generates ground-truth mastering pairs (unmastered mix -> target master, z_target)
using the isomorphic differentiable mastering engine or C++ core DSP engine.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn

from ddsp_mastering.dsp.chain import DifferentiableMasteringChain
from ddsp_mastering.param_space import NUM_PARAMS


class SyntheticMasteringPairGenerator(nn.Module):
    """Generates synthetic (raw_mix, mastered_target, z_target) pairs for self-supervised training.

    Enables high-fidelity training without needing expensive proprietary multi-track stems.
    """

    def __init__(self, sample_rate: float = 48000.0, use_cpp_fast: bool = True):
        super().__init__()
        self.sample_rate = sample_rate
        self.use_cpp_fast = use_cpp_fast
        self.chain = DifferentiableMasteringChain(sample_rate=sample_rate)

    def sample_realistic_z(self, batch_size: int = 1, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Sample realistic mastering parameter vectors z in [0, 1]^41.

        Uses beta/normal priors concentrated around moderate settings [0.3, 0.7] rather
        than extreme edge values, reflecting professional mastering engineering practices.
        """
        # Beta distribution (alpha=4.0, beta=4.0) produces smooth bell curve centered at 0.5
        # We approximate via clamped normal distribution: N(0.5, 0.12^2)
        if generator is not None:
            noise = torch.randn(batch_size, NUM_PARAMS, generator=generator)
        else:
            noise = torch.randn(batch_size, NUM_PARAMS)

        z = 0.5 + 0.12 * noise
        # Clamp to musically valid safe range [0.1, 0.9]
        z = torch.clamp(z, 0.1, 0.9)
        return z

    def generate_pair(
        self,
        raw_mix: torch.Tensor,
        z_target: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate a synthetic mastering training pair.

        Args:
            raw_mix: Normalized pre-master audio tensor of shape (B, 2, T).
            z_target: Optional explicit target parameter vector of shape (B, 41).
            generator: Optional torch.Generator for reproducibility.

        Returns:
            Tuple of (raw_mix, target_master, z_target).
        """
        B = raw_mix.shape[0]
        if z_target is None:
            z_target = self.sample_realistic_z(batch_size=B, generator=generator).to(
                device=raw_mix.device, dtype=raw_mix.dtype
            )

        with torch.no_grad():
            if self.use_cpp_fast and B == 1:
                # Use C++ fast processing
                z_single = z_target[0]
                target_master = self.chain.forward_cpp(raw_mix, z_single)
            else:
                target_master = self.chain(raw_mix, z=z_target, eq_mode="fft")

        return raw_mix, target_master, z_target
