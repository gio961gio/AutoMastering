"""Mastering Session Manifest & Provenance Schema.

Stores full metadata, alignment parameters, chunk specifications, optimization hyperparameters,
DSP control values, and pre/post quantitative evaluations for 100% reproducible mastering runs.
"""

from dataclasses import dataclass, field, asdict
import json
import os
import sys
from typing import Any, Dict, List, Optional, Union
import torch


@dataclass
class MasteringManifest:
    """Complete provenance and session manifest for an AutoMastering execution."""
    manifest_version: str = "2.0"
    reference_mode: str = "same_track"  # "same_track" | "stylistic"

    # Input track details
    input_track_path: str = ""
    input_sample_rate: float = 48000.0
    input_duration_sec: float = 0.0

    # Reference track details
    reference_track_path: str = ""
    reference_sample_rate: float = 48000.0
    reference_duration_sec: float = 0.0

    # Alignment details (same_track only)
    alignment: Dict[str, Any] = field(default_factory=dict)

    # Multi-chunk details
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    chunk_selection_config: Dict[str, Any] = field(default_factory=dict)

    # Optimization configuration
    optimization_config: Dict[str, Any] = field(default_factory=dict)

    # DSP Parameter outcomes
    normalized_z: List[float] = field(default_factory=list)
    physical_theta: List[float] = field(default_factory=list)
    parameters: Dict[str, float] = field(default_factory=dict)

    # Quantitative Benchmark Metrics
    metrics_pre: Dict[str, float] = field(default_factory=dict)
    metrics_post: Dict[str, float] = field(default_factory=dict)

    # Environment & software info
    software_info: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, output_path: Optional[str] = None, indent: int = 2) -> str:
        """Serialize manifest to JSON string and optionally write to disk."""
        data = self.to_dict()
        json_str = json.dumps(data, indent=indent)
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(json_str)
        return json_str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MasteringManifest":
        """Reconstruct manifest instance from dictionary."""
        return cls(
            manifest_version=data.get("manifest_version", "2.0"),
            reference_mode=data.get("reference_mode", "same_track"),
            input_track_path=data.get("input_track_path", ""),
            input_sample_rate=data.get("input_sample_rate", 48000.0),
            input_duration_sec=data.get("input_duration_sec", 0.0),
            reference_track_path=data.get("reference_track_path", ""),
            reference_sample_rate=data.get("reference_sample_rate", 48000.0),
            reference_duration_sec=data.get("reference_duration_sec", 0.0),
            alignment=data.get("alignment", {}),
            chunks=data.get("chunks", []),
            chunk_selection_config=data.get("chunk_selection_config", {}),
            optimization_config=data.get("optimization_config", {}),
            normalized_z=data.get("normalized_z", []),
            physical_theta=data.get("physical_theta", []),
            parameters=data.get("parameters", {}),
            metrics_pre=data.get("metrics_pre", {}),
            metrics_post=data.get("metrics_post", {}),
            software_info=data.get("software_info", {}),
        )

    @classmethod
    def from_json(cls, json_str_or_path: str) -> "MasteringManifest":
        """Load manifest from JSON string or file path."""
        if os.path.exists(json_str_or_path):
            with open(json_str_or_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(json_str_or_path)
        return cls.from_dict(data)
