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
from ddsp_mastering.dsp import DifferentiableHarmonicSaturator
from ddsp_mastering.optimize import AutoMasteringOptimizer
from scripts.run_mastering import find_highest_energy_window

def test_neutral_parameters_integrity():
    """Verify THETA_NEUTRAL and Z_NEUTRAL are defined, valid, and roundtrip cleanly."""
    assert len(THETA_NEUTRAL) == NUM_PARAMS == 41
    assert len(Z_NEUTRAL) == NUM_PARAMS == 41

    z_np = Z_NEUTRAL.detach().cpu().numpy()
    theta_np = THETA_NEUTRAL.detach().cpu().numpy()

    # Check Z_NEUTRAL is in valid normalized bounds (0, 1)
    assert np.all(z_np >= 0.0)
    assert np.all(z_np <= 1.0)

    # Check roundtrip: map_theta_to_z(THETA_NEUTRAL) == Z_NEUTRAL
    computed_z = map_theta_to_z(THETA_NEUTRAL)
    np.testing.assert_allclose(computed_z.numpy(), z_np, atol=1e-5)

    # Check roundtrip: map_z_to_theta(Z_NEUTRAL.to(torch.float64)) == THETA_NEUTRAL
    recovered_theta = map_z_to_theta(Z_NEUTRAL.to(torch.float64))
    np.testing.assert_allclose(recovered_theta.numpy(), theta_np, atol=1e-4)

def test_neutral_chain_near_bypass():
    """Verify that Z_NEUTRAL in C++ MasteringChain produces steady-state near-bypass behaviour (< 0.1 dB change)."""
    chain = py_core_dsp.MasteringChain(48000.0)
    z_list = Z_NEUTRAL.tolist()
    chain.set_parameters_z(z_list)

    # Generate 1s of 1kHz sine wave at -20 dBFS
    t = np.linspace(0, 1.0, 48000, endpoint=False)
    sine = (10 ** (-20.0 / 20.0) * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    stereo_in = np.stack([sine, sine], axis=0)

    out = chain.process(stereo_in)

    # Steady-state peak amplitude (after startup transients settle, e.g. second half second)
    in_peak = np.max(np.abs(stereo_in[:, 24000:]))
    out_peak = np.max(np.abs(out[:, 24000:]))
    diff_db = 20 * np.log10(out_peak / in_peak)
    assert abs(diff_db) < 0.1, f"Neutral chain altered steady-state peak level by {diff_db:.3f} dB, expected < 0.1 dB"

def test_saturator_small_signal_unit_gain():
    """Verify that saturator has strictly 0 dB linear small-signal gain (no -4.5 dB gain drop)."""
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

    # Parity with C++ saturator (expects [2, num_samples])
    sat_cpp = py_core_dsp.HarmonicSaturator()
    sat_cpp.set_parameters(6.0, 1.0)
    x_np = x.squeeze(0).numpy() # (2, T)
    y_cpp = sat_cpp.process(x_np)
    cpp_gain = (np.max(np.abs(y_cpp[:, 100:])) / np.max(np.abs(x_np[:, 100:])))
    cpp_gain_db = 20 * np.log10(cpp_gain)
    assert abs(cpp_gain_db) < 0.05, f"C++ small-signal gain was {cpp_gain_db:.3f} dB, expected ~0 dB"

def test_rolling_rms_window_selection():
    """Verify rolling RMS window finder correctly skips silent/intro sections."""
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

    # The chosen window must be inside or overlapping the loud drop [4.0s, 7.0s]
    best_time = best_start / fs
    assert 3.5 <= best_time <= 5.5, f"Best window started at {best_time:.2f}s, expected around 4.0-5.0s"
