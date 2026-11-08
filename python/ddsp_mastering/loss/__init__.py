"""Loss functions for differentiable audio mastering."""

from .loudness import (
    DifferentiableKWeighting,
    DifferentiableLoudnessLoss,
    get_k_weighting_coefficients,
)
from .stft import (
    SingleResolutionSTFTLoss,
    MultiResolutionSTFTLoss,
)
from .combined import MasteringPerceptualLoss
from .stylistic import AggregatePerceptualLoss
from .erb import (
    ERBSpectralLoss,
    build_erb_filterbank,
    compute_active_frame_weights_and_mask,
    compute_active_sample_mask,
)

__all__ = [
    "DifferentiableKWeighting",
    "DifferentiableLoudnessLoss",
    "get_k_weighting_coefficients",
    "SingleResolutionSTFTLoss",
    "MultiResolutionSTFTLoss",
    "MasteringPerceptualLoss",
    "AggregatePerceptualLoss",
    "ERBSpectralLoss",
    "build_erb_filterbank",
    "compute_active_frame_weights_and_mask",
    "compute_active_sample_mask",
]
