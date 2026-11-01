import pytest
import numpy as np
import torch
import py_core_dsp
from ddsp_mastering.param_space import (
    NUM_PARAMS,
    PARAM_SPECS,
    map_z_to_theta,
    map_theta_to_z,
)

def test_cpp_constants_and_specs():
    assert py_core_dsp.NUM_PARAMS == NUM_PARAMS == 41
    assert py_core_dsp.DEFAULT_SAMPLE_RATE == 48000.0
    assert py_core_dsp.TOTAL_CHAIN_LATENCY_SAMPLES == 159
    assert py_core_dsp.FIR_HALFBAND_LATENCY == 15
    assert py_core_dsp.LIMITER_LOOKAHEAD_SAMPLES == 144

    specs_cpp = py_core_dsp.get_param_specs()
    assert len(specs_cpp) == 41

    for i, (name_cpp, min_cpp, max_cpp, is_log_cpp, unit_cpp) in enumerate(specs_cpp):
        p_py = PARAM_SPECS[i]
        assert name_cpp == p_py.name
        assert min_cpp == p_py.min_val
        assert max_cpp == p_py.max_val
        assert is_log_cpp == p_py.is_log
        assert unit_cpp == p_py.unit

def test_fir_halfband_coefficients_parity():
    coeffs = py_core_dsp.get_fir_halfband_coeffs()
    assert len(coeffs) == 31
    # Verifica simmetria esatta
    arr = np.array(coeffs, dtype=np.float64)
    assert np.allclose(arr, arr[::-1], atol=1e-15, rtol=1e-15)
    assert np.isclose(arr[15], 0.5, atol=1e-15)
    assert np.isclose(np.sum(arr), 1.0, atol=1e-15)

def test_cpp_python_z_to_theta_bit_parity():
    """
    Quality gate di Argon:
    MSE < 10^(-14) tra C++ e PyTorch su tutti i 41 parametri per z uniformi e casuali.
    """
    rng = np.random.default_rng(777)
    
    # 50 vettori casuali in [0, 1]
    for _ in range(50):
        z_np = rng.uniform(0.0, 1.0, size=NUM_PARAMS).astype(np.float64)
        
        # C++
        theta_cpp = np.array(py_core_dsp.map_z_to_theta(z_np.tolist()), dtype=np.float64)
        
        # PyTorch
        z_torch = torch.from_numpy(z_np).unsqueeze(0)
        theta_torch = map_z_to_theta(z_torch).squeeze(0).numpy()
        
        # Errore quadratico medio e errore massimo
        mse = np.mean((theta_cpp - theta_torch) ** 2)
        max_abs_diff = np.max(np.abs(theta_cpp - theta_torch))
        
        assert mse < 1e-14, f"MSE {mse} exceeds 1e-14"
        assert max_abs_diff < 1e-7, f"Max diff {max_abs_diff} too large"

def test_cpp_python_theta_to_z_bit_parity():
    rng = np.random.default_rng(888)
    
    for _ in range(50):
        z_orig = rng.uniform(0.01, 0.99, size=NUM_PARAMS).astype(np.float64)
        theta_cpp = py_core_dsp.map_z_to_theta(z_orig.tolist())
        
        # Roundtrip C++
        z_rec_cpp = np.array(py_core_dsp.map_theta_to_z(theta_cpp), dtype=np.float64)
        
        # Roundtrip PyTorch
        theta_torch = torch.tensor(theta_cpp, dtype=torch.float64).unsqueeze(0)
        z_rec_torch = map_theta_to_z(theta_torch).squeeze(0).numpy()
        
        mse = np.mean((z_rec_cpp - z_rec_torch) ** 2)
        assert mse < 1e-15, f"Inverse MSE {mse} exceeds 1e-15"
