"""Loss functions for DDSP AutoMastering."""

from ddsp_mastering.loss.stft import SingleResolutionSTFTLoss, MultiResolutionSTFTLoss
from ddsp_mastering.loss.loudness import (
    DifferentiableKWeighting,
    DifferentiableLoudnessLoss,
    get_k_weighting_coefficients,
)
from ddsp_mastering.loss.combined import MasteringPerceptualLoss

__all__ = [
    "SingleResolutionSTFTLoss",
    "MultiResolutionSTFTLoss",
    "DifferentiableKWeighting",
    "DifferentiableLoudnessLoss",
    "get_k_weighting_coefficients",
    "MasteringPerceptualLoss",
]
