"""Data pipeline, reference protocol, and multi-chunk evaluation modules."""

from .dataset import AudioDataset
from .synthetic import SyntheticMasteringPairGenerator
from .reference_protocol import (
    ReferenceMode,
    AlignmentResult,
    verify_audio_pair,
    estimate_time_alignment,
    apply_same_track_alignment,
    apply_fractional_delay,
    get_same_track_slice_indices,
    assert_stylistic_guards,
)
from .chunking import (
    ChunkSpec,
    MultiChunkSelector,
)

__all__ = [
    "AudioDataset",
    "SyntheticMasteringPairGenerator",
    "ReferenceMode",
    "AlignmentResult",
    "verify_audio_pair",
    "estimate_time_alignment",
    "apply_same_track_alignment",
    "apply_fractional_delay",
    "get_same_track_slice_indices",
    "assert_stylistic_guards",
    "ChunkSpec",
    "MultiChunkSelector",
]
