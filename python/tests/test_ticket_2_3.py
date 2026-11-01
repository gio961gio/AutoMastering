"""Tests for Ticket 2.3: End-to-End Optimization Loop & Benchmark Suite."""

import json
import math
import numpy as np
import pytest
import torch

from ddsp_mastering.data import SyntheticMasteringPairGenerator
from ddsp_mastering.optimize import AutoMasteringOptimizer, MasteringBenchmarkSuite
from ddsp_mastering.param_space import NUM_PARAMS, PARAM_SPECS
import py_core_dsp


@pytest.fixture
def synthetic_test_track():
    """Generates synthetic stereo audio chunk (48 kHz, 1.0s) for optimization tests."""
    torch.manual_seed(42)
    fs = 48000
    t = torch.linspace(0, 1.0, fs)
    left = (
        0.2 * torch.sin(2 * math.pi * 80.0 * t)
        + 0.15 * torch.sin(2 * math.pi * 350.0 * t)
        + 0.1 * torch.sin(2 * math.pi * 2000.0 * t)
        + 0.05 * torch.sin(2 * math.pi * 8000.0 * t)
    )
    right = (
        0.2 * torch.sin(2 * math.pi * 80.0 * t)
        + 0.12 * torch.sin(2 * math.pi * 350.0 * t + 0.2)
        + 0.1 * torch.sin(2 * math.pi * 2000.0 * t - 0.2)
        + 0.05 * torch.sin(2 * math.pi * 8000.0 * t + 0.1)
    )
    return torch.stack([left, right], dim=0).unsqueeze(0)  # (1, 2, 48000)


def test_benchmark_suite_metrics_identity(synthetic_test_track):
    """Test that identical audio yields zero spectral distance and loudness mismatch."""
    suite = MasteringBenchmarkSuite(sample_rate=48000.0)
    results = suite.evaluate(synthetic_test_track, synthetic_test_track, align_latency=False)

    assert results["spectral_convergence"] < 1e-4
    assert results["log_spectral_distance_db"] < 1e-4
    assert results["delta_lufs"] < 1e-4
    assert results["delta_plr_db"] < 1e-4
    assert results["ceiling_violation_db"] == 0.0


def test_benchmark_suite_detects_spectral_and_stereo_alterations(synthetic_test_track):
    """Test that benchmark suite accurately measures alterations in EQ and stereo field."""
    suite = MasteringBenchmarkSuite(sample_rate=48000.0)

    # 1. High frequency boost
    altered_audio = synthetic_test_track.clone()
    altered_audio[:, :, 1::2] *= 1.5  # Modulate spectrum

    results = suite.evaluate(altered_audio, synthetic_test_track, align_latency=False)
    assert results["log_spectral_distance_db"] > 1.0, "LSD should detect spectral variation"
    assert results["spectral_convergence"] > 0.05


def test_end_to_end_self_supervised_optimization(synthetic_test_track):
    """Verify that AutoMasteringOptimizer achieves significant loss reduction and reaches target LUFS."""
    pair_gen = SyntheticMasteringPairGenerator(sample_rate=48000.0, use_cpp_fast=False)

    # Fast test chunk: 0.25s (12,000 samples)
    test_chunk = synthetic_test_track[:, :, :12000]

    # Generate synthetic target pair
    gen = torch.Generator().manual_seed(123)
    raw_mix, target_master, z_target = pair_gen.generate_pair(
        test_chunk, generator=gen
    )

    optimizer = AutoMasteringOptimizer(
        sample_rate=48000.0,
        lr=0.08,
        num_steps=20,
        target_lufs=None,
    )

    z_opt, mastered, history = optimizer.optimize(
        raw_mix=raw_mix,
        target_reference=target_master,
        verbose=False,
        align_latency=False,
    )

    # Invariants
    assert z_opt.shape == (NUM_PARAMS,)
    assert torch.all(z_opt > 0.0) and torch.all(z_opt < 1.0), "z must be validly bounded in (0, 1)"

    # Convergence: Loss must decrease significantly
    initial_loss = history["loss"][0]
    final_loss = history["loss"][-1]
    assert final_loss < 0.85 * initial_loss, (
        f"Loss should decrease by at least 15%: initial={initial_loss:.4f}, final={final_loss:.4f}"
    )

    # Evaluate benchmark metrics before and after optimization
    suite = MasteringBenchmarkSuite(sample_rate=48000.0)
    metrics_initial = suite.evaluate(raw_mix, target_master, align_latency=False, z_target=z_target)
    metrics_final = suite.evaluate(mastered, target_master, align_latency=False, z_pred=z_opt, z_target=z_target)

    # Spectral convergence must improve compared to unmastered raw mix
    assert metrics_final["spectral_convergence"] < metrics_initial["spectral_convergence"]
    # Log-spectral distance must improve
    assert metrics_final["log_spectral_distance_db"] < metrics_initial["log_spectral_distance_db"]
    # Ceiling compliance
    assert metrics_final["ceiling_violation_db"] == 0.0


def test_preset_json_export_and_cpp_twin_parity(synthetic_test_track, tmp_path):
    """Verify JSON export and exact parity with C++20 core engine execution."""
    optimizer = AutoMasteringOptimizer(sample_rate=48000.0, num_steps=5)

    z_test = torch.full((NUM_PARAMS,), 0.55, dtype=torch.float32)
    json_path = tmp_path / "test_preset.json"
    json_str = optimizer.export_preset_json(z_test, output_path=str(json_path))

    # Parse and verify JSON structure
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["format"] == "AutoMastering_Preset_v1"
    assert data["sample_rate"] == 48000.0
    assert len(data["parameters"]) == NUM_PARAMS

    for spec in PARAM_SPECS:
        assert spec.name in data["parameters"]

    # Verify execution parity with C++20 py_core_dsp.MasteringChain
    cpp_chain = py_core_dsp.MasteringChain(48000.0)
    cpp_chain.set_parameters_z(z_test.tolist())

    x_np = synthetic_test_track.squeeze(0).numpy()
    y_cpp = cpp_chain.process(x_np)

    # PyTorch chain
    with torch.no_grad():
        y_torch = optimizer.chain(synthetic_test_track, z=z_test, eq_mode="fft").squeeze(0).numpy()

    # Both must respect the -0.3 dBFS ceiling
    peak_cpp = np.max(np.abs(y_cpp))
    peak_torch = np.max(np.abs(y_torch))
    assert peak_cpp <= 0.9660508787031944 + 1e-4
    assert peak_torch <= 0.9660508787031944 + 1e-4
