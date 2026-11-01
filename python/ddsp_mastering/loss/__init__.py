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

__all__ = [
    "DifferentiableKWeighting",
    "DifferentiableLoudnessLoss",
    "get_k_weighting_coefficients",
    "SingleResolutionSTFTLoss",
    "MultiResolutionSTFTLoss",
    "MasteringPerceptualLoss",
    "AggregatePerceptualLoss",
]
