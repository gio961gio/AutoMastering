import json
import pytest
import torch
import numpy as np
import py_core_dsp

from ddsp_mastering.param_space import (
    PARAM_SPECS,
    NUM_PARAMS,
    THETA_NEUTRAL,
    Z_NEUTRAL,
    map_z_to_theta,
    map_theta_to_z,
)
from ddsp_mastering.dsp import (
    DifferentiableHarmonicSaturator,
    DifferentiableSingleBandCompressorFunction,
    DifferentiableEQ,
)
from ddsp_mastering.dsp.limiter import FastLimiterReleaseFunction
from ddsp_mastering.optimize import AutoMasteringOptimizer
from scripts.run_mastering import find_highest_energy_window

def test_neutral_parameters_integrity():
    """Verify THETA_NEUTRAL and Z_NEUTRAL are defined, valid, and roundtrip cleanly."""
    assert len(THETA_NEUTRAL) == NUM_PARAMS == 41
    assert len(Z_NEUTRAL) == NUM_PARAMS == 41

    z_np = Z_NEUTRAL.detach().cpu().numpy()
    theta_np = THETA_NEUTRAL.detach().cpu().numpy()

    # Check Z_NEUTRAL is in valid normalized bounds [0, 1]
    assert np.all(z_np >= 0.0)
    assert np.all(z_np <= 1.0)

    # Check roundtrip: map_theta_to_z(THETA_NEUTRAL) == Z_NEUTRAL
    computed_z = map_theta_to_z(THETA_NEUTRAL)
    np.testing.assert_allclose(computed_z.numpy(), z_np, atol=1e-5)

    # Check roundtrip: map_z_to_theta(Z_NEUTRAL.to(torch.float64)) == THETA_NEUTRAL
    recovered_theta = map_z_to_theta(Z_NEUTRAL.to(torch.float64))
    np.testing.assert_allclose(recovered_theta.numpy(), theta_np, atol=1e-4)

    # Verify that compressor ratios are 1.0 (true identity) in THETA_NEUTRAL
    assert float(theta_np[21]) == 1.0  # low ratio
    assert float(theta_np[26]) == 1.0  # mid ratio
    assert float(theta_np[31]) == 1.0  # high ratio

def test_compressor_ratio_one_identity():
    """Verify that compressor with ratio=1.0 and makeup=0.0 produces mathematical identity."""
    # Test across loud dynamic signals exceeding 0 dBFS or negative levels
    torch.manual_seed(42)
    x = torch.randn(1, 2, 9600, dtype=torch.float64) * 0.8  # loud signal
    threshold_db = torch.tensor([-24.0], dtype=torch.float64)  # well below signal
    ratio = torch.tensor([1.0], dtype=torch.float64)          # 1.0:1
    attack_ms = torch.tensor([10.0], dtype=torch.float64)
    release_ms = torch.tensor([100.0], dtype=torch.float64)
    makeup_db = torch.tensor([0.0], dtype=torch.float64)

    out = DifferentiableSingleBandCompressorFunction.apply(
        x, threshold_db, ratio, attack_ms, release_ms, makeup_db, 48000.0
    )

    # When ratio is 1.0, output must be mathematically identical to input (error 0.0)
    diff = torch.max(torch.abs(out - x)).item()
    assert diff < 1e-12, f"Compressor with ratio=1.0 altered signal by {diff:.2e}, expected 0.0"

    # Parity with C++ SingleBandComp
    comp_cpp = py_core_dsp.SingleBandComp()
    comp_cpp.set_parameters(-24.0, 1.0, 10.0, 100.0, 0.0, 6.0, 48000.0)
    x_np = x.squeeze(0).numpy().astype(np.float32)
    out_cpp = comp_cpp.process(x_np)
    cpp_diff = np.max(np.abs(out_cpp - x_np))
    assert cpp_diff < 1e-6, f"C++ SingleBandComp with ratio=1.0 altered signal by {cpp_diff:.2e}"

def test_fast_limiter_release_gradcheck():
    """Formally verify FastLimiterReleaseFunction analytical gradients with autograd.gradcheck."""
    torch.manual_seed(123)
    # Test on alternating attack and release steps in double precision
    N = 64
    g_min = torch.rand((1, N), dtype=torch.float64, requires_grad=True)
    alpha_rel = 0.995

    # Run torch.autograd.gradcheck (finite differences vs analytical backward)
    test_passed = torch.autograd.gradcheck(
        FastLimiterReleaseFunction.apply,
        (g_min, alpha_rel),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
    )
    assert test_passed, "FastLimiterReleaseFunction failed torch.autograd.gradcheck"

def test_neutral_chain_dynamic_signal_near_bypass():
    """Verify that Z_NEUTRAL in C++ MasteringChain produces transparent behaviour on dynamic signal."""
    chain = py_core_dsp.MasteringChain(48000.0)
    z_list = Z_NEUTRAL.tolist()
    chain.set_parameters_z(z_list)

    # Generate 1s of complex dynamic signal (multi-tone + noise) at -18 dBFS RMS
    np.random.seed(42)
    t = np.linspace(0, 1.0, 48000, endpoint=False)
    bass = 0.3 * np.sin(2 * np.pi * 80.0 * t)
    mid = 0.2 * np.sin(2 * np.pi * 1000.0 * t)
    high = 0.1 * np.sin(2 * np.pi * 8000.0 * t)
    noise = 0.05 * np.random.randn(len(t))
    signal = (bass + mid + high + noise).astype(np.float32)
    stereo_in = np.stack([signal, signal], axis=0)

    out = chain.process(stereo_in)

    # Compare steady-state audio (second half-second) with input
    in_sub = stereo_in[:, 24000:]
    out_sub = out[:, 24000:]
    rms_in = np.sqrt(np.mean(in_sub ** 2))
    rms_out = np.sqrt(np.mean(out_sub ** 2))
    delta_rms_db = abs(20 * np.log10(rms_out / rms_in))

    # Documented tolerance: delta RMS < 0.1 dB
    assert delta_rms_db < 0.1, f"Neutral preset altered RMS by {delta_rms_db:.3f} dB (expected < 0.1 dB)"

def test_saturator_small_signal_unit_gain():
    """Verify that saturator has strictly 0 dB linear small-signal gain (1/gd normalisation)."""
    sat_py = DifferentiableHarmonicSaturator()
    drive_db = torch.tensor([6.0]) # high drive
    mix = torch.tensor([1.0])      # 100% wet

    # Very small signal: -80 dBFS stereo [1, 2, T]
    small_amp = 1e-4
    t = torch.linspace(0, 0.1, 4800, dtype=torch.float32)
    sine = (small_amp * torch.sin(2 * np.pi * 440.0 * t)).unsqueeze(0) # (1, T)
    x = torch.cat([sine, sine], dim=0).unsqueeze(0) # (1, 2, T)

    y = sat_py(x, drive_db, mix)

    # Small signal gain y / x after filter settles (skip FIR group delay)
    gain = (torch.max(torch.abs(y[:, :, 100:])) / torch.max(torch.abs(x[:, :, 100:]))).item()
    gain_db = 20 * np.log10(gain)
    assert abs(gain_db) < 0.05, f"PyTorch small-signal gain was {gain_db:.3f} dB, expected ~0 dB"

    # Parity with C++ saturator
    sat_cpp = py_core_dsp.HarmonicSaturator()
    sat_cpp.set_parameters(6.0, 1.0)
    x_np = x.squeeze(0).numpy() # (2, T)
    y_cpp = sat_cpp.process(x_np)
    cpp_gain = (np.max(np.abs(y_cpp[:, 100:])) / np.max(np.abs(x_np[:, 100:])))
    cpp_gain_db = 20 * np.log10(cpp_gain)
    assert abs(cpp_gain_db) < 0.05, f"C++ small-signal gain was {cpp_gain_db:.3f} dB, expected ~0 dB"

def test_preset_json_roundtrip_interoperability():
    """Verify that exported preset JSON has canonical schema parseable by JUCE and Python."""
    opt = AutoMasteringOptimizer(48000.0)
    json_str = opt.export_preset_json(Z_NEUTRAL)

    data = json.loads(json_str)
    assert data["format"] == "AutoMastering_Preset_v1"
    assert data["version"] == "1.0"
    assert data["sample_rate"] == 48000.0
    assert len(data["normalized_z"]) == 41

    # Check that parameters dict contains all 41 parameters as flat floats (for JUCE)
    assert "parameters" in data
    assert len(data["parameters"]) == 41
    for spec in PARAM_SPECS:
        assert spec.name in data["parameters"]
        val = data["parameters"][spec.name]
        assert isinstance(val, (int, float))
        assert spec.min_val <= val <= spec.max_val

    # Check that parameter_details contains rich metadata
    assert "parameter_details" in data
    assert len(data["parameter_details"]) == 41

def test_pytorch_iir_vs_cpp_parity_with_tolerances():
    """Measure explicit numerical error tolerances between PyTorch IIR cascade and C++ core."""
    torch.manual_seed(99)
    # Generate 0.5s stereo signal
    x_t = torch.randn(1, 2, 24000, dtype=torch.float32)
    x_np = x_t.squeeze(0).numpy()

    eq_py = DifferentiableEQ(48000.0, default_mode="iir")
    f0 = torch.tensor([[60.0, 250.0, 1000.0, 3000.0, 8000.0, 12000.0]])
    gain = torch.tensor([[2.0, -3.0, 1.5, -2.0, 3.0, -1.0]])
    q = torch.tensor([[1.0, 1.4, 0.7, 1.2, 0.8, 0.7]])

    y_py = eq_py(x_t, f0, gain, q, mode="iir").squeeze(0).detach().numpy()

    eq_cpp = py_core_dsp.BiquadCascade6Band()
    eq_cpp.set_parameters(f0[0].tolist(), gain[0].tolist(), q[0].tolist(), 48000.0)
    y_cpp = eq_cpp.process(x_np)

    # Documented numerical tolerances: Max Absolute Error < 1e-4, Null RMS < -80 dBFS
    max_abs_err = np.max(np.abs(y_py - y_cpp))
    null_rms = np.sqrt(np.mean((y_py - y_cpp) ** 2))
    null_rms_dbfs = 20 * np.log10(null_rms + 1e-12)

    assert max_abs_err < 1e-4, f"PyTorch IIR vs C++ Max Abs Error was {max_abs_err:.2e} (expected < 1e-4)"
    assert null_rms_dbfs < -80.0, f"PyTorch IIR vs C++ Null RMS was {null_rms_dbfs:.1f} dBFS (expected < -80 dBFS)"

def test_rolling_rms_window_selection():
    """Verify rolling RMS window finder correctly skips silent/intro sections."""
    np.random.seed(1234)
    fs = 48000
    total_sec = 10
    total_samples = total_sec * fs
    signal = np.zeros((2, total_samples), dtype=np.float32)

    # Put a loud drop at 4.0s - 7.0s
    start_drop = int(4.0 * fs)
    end_drop = int(7.0 * fs)
    signal[:, start_drop:end_drop] = 0.5 * np.random.randn(2, end_drop - start_drop).astype(np.float32)

    opt_sec = 2.0
    opt_len = int(opt_sec * fs)
    best_start = find_highest_energy_window(signal, opt_len, hop_len=1200)

    best_time = best_start / fs
    assert 3.5 <= best_time <= 5.5, f"Best window started at {best_time:.2f}s, expected around 4.0-5.0s"
