import pytest
import time
import numpy as np
import torch
import py_core_dsp
from ddsp_mastering.dsp.compressor import (
    DifferentiableSingleBandCompressorFunction,
    DifferentiableMultibandCompressor
)

def compute_null_dbfs(y_cpp: np.ndarray, y_torch: np.ndarray) -> float:
    diff = y_cpp - y_torch
    mse = np.mean(diff ** 2)
    return float(10.0 * np.log10(mse + 1e-16))

def compute_max_abs_err_dbfs(y_cpp: np.ndarray, y_torch: np.ndarray) -> float:
    max_err = np.max(np.abs(y_cpp - y_torch))
    return float(20.0 * np.log10(max_err + 1e-12))


# === 1. PARITÀ SINGLE BAND COMPRESSOR (C++ vs PyTorch) ===
@pytest.mark.parametrize("thresh,ratio,att,rel,makeup,desc", [
    (-15.0, 2.5, 30.0, 100.0, 2.0, "mild_compression"),
    (-28.0, 8.0, 5.0, 40.0, 6.0, "heavy_compression"),
    (0.0, 4.0, 10.0, 100.0, 0.0, "sub_threshold_transparent"),
])
def test_single_band_compressor_parity(sig_pink_music, thresh, ratio, att, rel, makeup, desc):
    x_torch = sig_pink_music # [2, 96000]
    x_np = x_torch.numpy()

    # C++ SingleBandComp
    cpp_comp = py_core_dsp.SingleBandComp()
    cpp_comp.set_parameters(thresh, ratio, att, rel, makeup, 6.0, 48000.0)
    y_cpp = cpp_comp.process(x_np)

    # PyTorch DifferentiableSingleBandCompressorFunction
    t_thresh = torch.tensor(thresh, dtype=torch.float32)
    t_ratio = torch.tensor(ratio, dtype=torch.float32)
    t_att = torch.tensor(att, dtype=torch.float32)
    t_rel = torch.tensor(rel, dtype=torch.float32)
    t_makeup = torch.tensor(makeup, dtype=torch.float32)

    y_torch = DifferentiableSingleBandCompressorFunction.apply(
        x_torch, t_thresh, t_ratio, t_att, t_rel, t_makeup, 48000.0
    ).numpy()

    null_db = compute_null_dbfs(y_cpp, y_torch)
    max_err_db = compute_max_abs_err_dbfs(y_cpp, y_torch)

    print(f"\n[Test Parity {desc}] Null: {null_db:.2f} dBFS, MaxErr: {max_err_db:.2f} dBFS")
    assert null_db <= -110.0, f"SingleBandComp Null {null_db} dBFS > -110 dBFS ({desc})"


# === 2. PARITÀ MULTIBAND COMPRESSOR (C++ vs PyTorch) ===
def test_multiband_compressor_parity(sig_sweep):
    x_np = sig_sweep.numpy()
    # Generiamo 3 bande stereo distinte
    low_np = x_np * 0.6
    mid_np = x_np * 0.4
    high_np = x_np * 0.2

    low_t = torch.from_numpy(low_np)
    mid_t = torch.from_numpy(mid_np)
    high_t = torch.from_numpy(high_np)

    # Parametri per Low, Mid, High: [thresh, ratio, att, rel, makeup]
    low_p = [-18.0, 3.0, 40.0, 150.0, 2.0]
    mid_p = [-14.0, 2.5, 20.0, 80.0, 1.0]
    high_p = [-20.0, 4.0, 8.0, 40.0, 3.0]

    # C++ MultibandCompressor3Band
    cpp_mb = py_core_dsp.MultibandCompressor3Band()
    cpp_mb.set_parameters(low_p, mid_p, high_p, 48000.0)
    y_l_cpp, y_m_cpp, y_h_cpp = cpp_mb.process(low_np, mid_np, high_np)

    # PyTorch DifferentiableMultibandCompressor
    torch_mb = DifferentiableMultibandCompressor(sample_rate=48000.0)
    t_thresh = torch.tensor([low_p[0], mid_p[0], high_p[0]], dtype=torch.float32)
    t_ratio = torch.tensor([low_p[1], mid_p[1], high_p[1]], dtype=torch.float32)
    t_att = torch.tensor([low_p[2], mid_p[2], high_p[2]], dtype=torch.float32)
    t_rel = torch.tensor([low_p[3], mid_p[3], high_p[3]], dtype=torch.float32)
    t_makeup = torch.tensor([low_p[4], mid_p[4], high_p[4]], dtype=torch.float32)

    y_l_torch, y_m_torch, y_h_torch = torch_mb(
        low_t, mid_t, high_t,
        t_thresh, t_ratio, t_att, t_rel, t_makeup
    )

    null_low = compute_null_dbfs(y_l_cpp, y_l_torch.numpy())
    null_mid = compute_null_dbfs(y_m_cpp, y_m_torch.numpy())
    null_high = compute_null_dbfs(y_h_cpp, y_h_torch.numpy())

    print(f"\n[Multiband Parity] Low: {null_low:.2f} dBFS, Mid: {null_mid:.2f} dBFS, High: {null_high:.2f} dBFS")
    assert null_low <= -110.0, f"Low band Null {null_low} dBFS > -110 dBFS"
    assert null_mid <= -110.0, f"Mid band Null {null_mid} dBFS > -110 dBFS"
    assert null_high <= -110.0, f"High band Null {null_high} dBFS > -110 dBFS"


# === 3. NUMERICAL GRADCHECK DEL BACKWARD ANALITICO ===
def test_compressor_autograd_gradcheck():
    # Eseguiamo gradcheck in float64 su un segmento controllato di N=128 campioni
    torch.manual_seed(42)
    N = 128
    x = torch.randn(1, 2, N, dtype=torch.float64, requires_grad=True) * 0.3
    
    t_thresh = torch.tensor([-15.0], dtype=torch.float64, requires_grad=True)
    t_ratio = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
    t_att = torch.tensor([10.0], dtype=torch.float64, requires_grad=True)
    t_rel = torch.tensor([50.0], dtype=torch.float64, requires_grad=True)
    t_makeup = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)

    test_passed = torch.autograd.gradcheck(
        DifferentiableSingleBandCompressorFunction.apply,
        (x, t_thresh, t_ratio, t_att, t_rel, t_makeup, 48000.0),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
        raise_exception=True
    )
    assert test_passed, "Gradcheck failed for DifferentiableSingleBandCompressorFunction"


# === 4. STABILITÀ BACKWARD SU AUDIO COMPLETO (2s / 96000 campioni) ===
def test_compressor_backward_stability(sig_pink_music):
    x = sig_pink_music.clone().detach().requires_grad_(True)
    torch_mb = DifferentiableMultibandCompressor(sample_rate=48000.0)

    # Parametri con gradienti attivi
    t_thresh = torch.tensor([-18.0, -14.0, -20.0], dtype=torch.float32, requires_grad=True)
    t_ratio = torch.tensor([3.0, 2.5, 4.0], dtype=torch.float32, requires_grad=True)
    t_att = torch.tensor([40.0, 20.0, 8.0], dtype=torch.float32, requires_grad=True)
    t_rel = torch.tensor([150.0, 80.0, 40.0], dtype=torch.float32, requires_grad=True)
    t_makeup = torch.tensor([2.0, 1.0, 3.0], dtype=torch.float32, requires_grad=True)

    # Forward
    low = x * 0.6
    mid = x * 0.4
    high = x * 0.2
    y_l, y_m, y_h = torch_mb(low, mid, high, t_thresh, t_ratio, t_att, t_rel, t_makeup)
    y_sum = y_l + y_m + y_h

    # Loss fittizia di matching spettrale / energetico
    loss = torch.mean(y_sum ** 2)
    loss.backward()

    # Verifiche stabilità gradienti (no NaN, no Inf)
    assert x.grad is not None and not torch.isnan(x.grad).any() and not torch.isinf(x.grad).any()
    assert not torch.isnan(t_thresh.grad).any() and not torch.isinf(t_thresh.grad).any()
    assert not torch.isnan(t_ratio.grad).any() and not torch.isinf(t_ratio.grad).any()
    assert not torch.isnan(t_att.grad).any() and not torch.isinf(t_att.grad).any()
    assert not torch.isnan(t_rel.grad).any() and not torch.isinf(t_rel.grad).any()
    assert not torch.isnan(t_makeup.grad).any() and not torch.isinf(t_makeup.grad).any()

    # Tutti i gradienti devono essere diversi da zero
    assert torch.abs(t_thresh.grad).sum() > 0.0
    assert torch.abs(t_ratio.grad).sum() > 0.0
    assert torch.abs(t_att.grad).sum() > 0.0
    assert torch.abs(t_rel.grad).sum() > 0.0
    assert torch.abs(t_makeup.grad).sum() > 0.0


# === 5. BENCHMARK DELLE PRESTAZIONI ===
def test_compressor_speed_benchmark(sig_pink_music):
    x_np = sig_pink_music.numpy()
    x_torch = sig_pink_music

    # C++ SingleBandComp benchmark
    cpp_comp = py_core_dsp.SingleBandComp()
    cpp_comp.set_parameters(-15.0, 3.0, 20.0, 80.0, 1.5, 6.0, 48000.0)

    # Warmup
    for _ in range(2):
        _ = cpp_comp.process(x_np)

    t0 = time.perf_counter()
    iters = 10
    for _ in range(iters):
        _ = cpp_comp.process(x_np)
    cpp_time_ms = ((time.perf_counter() - t0) / iters) * 1000.0

    # PyTorch forward benchmark
    t_thresh = torch.tensor(-15.0)
    t_ratio = torch.tensor(3.0)
    t_att = torch.tensor(20.0)
    t_rel = torch.tensor(80.0)
    t_makeup = torch.tensor(1.5)

    _ = DifferentiableSingleBandCompressorFunction.apply(x_torch, t_thresh, t_ratio, t_att, t_rel, t_makeup, 48000.0)

    t0 = time.perf_counter()
    iters = 2
    for _ in range(iters):
        _ = DifferentiableSingleBandCompressorFunction.apply(x_torch, t_thresh, t_ratio, t_att, t_rel, t_makeup, 48000.0)
    py_time_ms = ((time.perf_counter() - t0) / iters) * 1000.0

    print(f"\n[Benchmark 2.0s Audio (96000 samples)] C++: {cpp_time_ms:.2f} ms | PyTorch Forward: {py_time_ms:.2f} ms")
    assert cpp_time_ms < 20.0, f"C++ compressor exceeded 20 ms: {cpp_time_ms:.2f} ms"
    assert py_time_ms < 100.0, f"PyTorch compressor exceeded 100 ms: {py_time_ms:.2f} ms"
