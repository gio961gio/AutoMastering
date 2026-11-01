import pytest
import numpy as np
import torch
import py_core_dsp
from ddsp_mastering.dsp.eq import DifferentiableEQ
from ddsp_mastering.dsp.crossover import DifferentiableLR4Crossover

def compute_null_dbfs(y_cpp: np.ndarray, y_torch: np.ndarray) -> float:
    """
    Calcola l'errore residuo in dBFS secondo la formula di Argon:
    Null_dBFS = 10 * log10( 1/(2N) * sum_c sum_n (y_cpp - y_torch)^2 + 1e-16 )
    """
    diff = y_cpp - y_torch
    mse = np.mean(diff ** 2)
    null_dbfs = 10.0 * np.log10(mse + 1e-16)
    return float(null_dbfs)

def compute_max_abs_err_dbfs(y_cpp: np.ndarray, y_torch: np.ndarray) -> float:
    max_err = np.max(np.abs(y_cpp - y_torch))
    return float(20.0 * np.log10(max_err + 1e-12))


# === 1. PARITÀ EQ PARAMETRICO 6 BANDE (MODE IIR) ===
@pytest.mark.parametrize("sig_fixture_name", ["sig_impulse", "sig_sweep", "sig_burst", "sig_pink_music"])
def test_eq_iir_parity(sig_fixture_name, request):
    x_torch = request.getfixturevalue(sig_fixture_name) # [2, 96000] float32
    x_np = x_torch.numpy()

    # Parametri realistici di mastering per le 6 bande:
    # 0: Low-Shelf, 1..4: Peaking, 5: High-Shelf
    f0 = [45.0, 180.0, 1200.0, 3500.0, 8000.0, 14000.0]
    gain_db = [2.5, -1.8, 1.2, -2.0, 1.5, 2.0]
    q = [0.8, 1.4, 2.0, 1.8, 1.2, 0.707]

    # C++
    eq_cpp = py_core_dsp.BiquadCascade6Band()
    eq_cpp.set_parameters(f0, gain_db, q, 48000.0)
    y_cpp = eq_cpp.process(x_np)

    # PyTorch (IIR Exact Mode)
    eq_py = DifferentiableEQ(sample_rate=48000.0, default_mode="iir")
    f0_t = torch.tensor(f0, dtype=torch.float32)
    gain_t = torch.tensor(gain_db, dtype=torch.float32)
    q_t = torch.tensor(q, dtype=torch.float32)
    
    y_torch = eq_py(x_torch, f0_t, gain_t, q_t, mode="iir").numpy()

    null_db = compute_null_dbfs(y_cpp, y_torch)
    max_err_db = compute_max_abs_err_dbfs(y_cpp, y_torch)

    print(f"\n[EQ IIR Parity - {sig_fixture_name}] Null: {null_db:.2f} dBFS, MaxErr: {max_err_db:.2f} dBFS")
    # Quality gate di Argon: Null <= -115 dBFS
    assert null_db <= -115.0, f"EQ IIR Null {null_db} dBFS > -115.0 dBFS"


# === 2. PARITÀ EQ PARAMETRICO (MODE FFT FAST GPU) ===
@pytest.mark.parametrize("sig_fixture_name", ["sig_sweep", "sig_pink_music"])
def test_eq_fft_parity(sig_fixture_name, request):
    x_torch = request.getfixturevalue(sig_fixture_name)
    x_np = x_torch.numpy()

    f0 = [60.0, 250.0, 1000.0, 4000.0, 9000.0, 15000.0]
    gain_db = [-3.0, 2.0, -1.5, 3.0, -2.0, 1.0]
    q = [0.9, 1.2, 1.5, 1.8, 1.1, 0.8]

    eq_cpp = py_core_dsp.BiquadCascade6Band()
    eq_cpp.set_parameters(f0, gain_db, q, 48000.0)
    y_cpp = eq_cpp.process(x_np)

    eq_py = DifferentiableEQ(sample_rate=48000.0)
    f0_t = torch.tensor(f0, dtype=torch.float32)
    gain_t = torch.tensor(gain_db, dtype=torch.float32)
    q_t = torch.tensor(q, dtype=torch.float32)

    y_fft = eq_py(x_torch, f0_t, gain_t, q_t, mode="fft").numpy()

    null_db = compute_null_dbfs(y_cpp, y_fft)
    print(f"\n[EQ FFT Parity - {sig_fixture_name}] Null: {null_db:.2f} dBFS")
    # Quality gate di Argon: Null <= -70.0 dBFS
    assert null_db <= -70.0, f"EQ FFT Null {null_db} dBFS > -70.0 dBFS"


# === 3. PARITÀ CROSSOVER LR4 A 3 VIE ===
@pytest.mark.parametrize("sig_fixture_name", ["sig_sweep", "sig_burst"])
def test_lr4_crossover_parity(sig_fixture_name, request):
    x_torch = request.getfixturevalue(sig_fixture_name)
    x_np = x_torch.numpy()

    f_low, f_high = 180.0, 3200.0

    # C++
    xo_cpp = py_core_dsp.LinkwitzRiley4_3Way()
    xo_cpp.set_crossover_frequencies(f_low, f_high, 48000.0)
    low_cpp, mid_cpp, high_cpp = xo_cpp.process(x_np)

    # PyTorch
    xo_py = DifferentiableLR4Crossover(sample_rate=48000.0)
    f_low_t = torch.tensor(f_low, dtype=torch.float32)
    f_high_t = torch.tensor(f_high, dtype=torch.float32)

    low_py, mid_py, high_py = xo_py(x_torch, f_low_t, f_high_t)
    low_py_np = low_py.numpy()
    mid_py_np = mid_py.numpy()
    high_py_np = high_py.numpy()

    null_low = compute_null_dbfs(low_cpp, low_py_np)
    null_mid = compute_null_dbfs(mid_cpp, mid_py_np)
    null_high = compute_null_dbfs(high_cpp, high_py_np)

    print(f"\n[LR4 Parity - {sig_fixture_name}] Low: {null_low:.2f} dBFS, Mid: {null_mid:.2f} dBFS, High: {null_high:.2f} dBFS")

    assert null_low <= -115.0, f"Low band Null {null_low} > -115 dBFS"
    assert null_mid <= -115.0, f"Mid band Null {null_mid} > -115 dBFS"
    assert null_high <= -115.0, f"High band Null {null_high} > -115 dBFS"


# === 4. RICOSTRUZIONE PERFETTAMENTE PIATTA DEL CROSSOVER LR4 ===
def test_lr4_allpass_flat_reconstruction(sig_impulse):
    """
    Verifica l'invariante di Argon:
    La somma Low + Mid + High ha risposta in ampiezza piatta a 0.000 dB (+- 1e-5 dB).
    """
    f_low, f_high = 220.0, 2800.0

    # 1. Test su C++
    xo_cpp = py_core_dsp.LinkwitzRiley4_3Way()
    xo_cpp.set_crossover_frequencies(f_low, f_high, 48000.0)
    low_cpp, mid_cpp, high_cpp = xo_cpp.process(sig_impulse.numpy())
    sum_cpp = low_cpp + mid_cpp + high_cpp

    # FFT magnitudine
    fft_cpp = np.fft.rfft(sum_cpp[0]) # Canale Left
    mag_db_cpp = 20.0 * np.log10(np.abs(fft_cpp) + 1e-12)
    max_dev_cpp = np.max(np.abs(mag_db_cpp))

    print(f"\n[LR4 Flat Reconstruction C++] Max dev: {max_dev_cpp:.7f} dB")
    assert max_dev_cpp < 1e-5, f"C++ LR4 sum deviation {max_dev_cpp} >= 1e-5 dB"

    # 2. Test su PyTorch
    xo_py = DifferentiableLR4Crossover(sample_rate=48000.0)
    low_py, mid_py, high_py = xo_py(sig_impulse, torch.tensor(f_low), torch.tensor(f_high))
    sum_py = (low_py + mid_py + high_py)[0].numpy()

    fft_py = np.fft.rfft(sum_py)
    mag_db_py = 20.0 * np.log10(np.abs(fft_py) + 1e-12)
    max_dev_py = np.max(np.abs(mag_db_py))

    print(f"[LR4 Flat Reconstruction PyTorch] Max dev: {max_dev_py:.7f} dB")
    assert max_dev_py < 1e-5, f"PyTorch LR4 sum deviation {max_dev_py} >= 1e-5 dB"


# === 5. GRADIENT FLOW & AUTOGRAD CHECK ===
def test_eq_and_crossover_gradcheck():
    """
    Verifica che i gradienti fluiscano correttamente attraverso l'EQ e il Crossover
    senza generare NaN o Inf.
    """
    torch.manual_seed(42)
    # Segnale breve di 4096 campioni stereo
    x = torch.randn(2, 4096, dtype=torch.float32)

    # EQ params con gradiente
    f0 = torch.tensor([50.0, 200.0, 1000.0, 3000.0, 7000.0, 12000.0], requires_grad=True)
    gain = torch.tensor([1.0, -1.0, 2.0, -2.0, 0.5, 1.5], requires_grad=True)
    q = torch.tensor([0.7, 1.0, 1.4, 1.2, 0.9, 0.8], requires_grad=True)

    eq = DifferentiableEQ(sample_rate=48000.0)
    y_eq = eq(x, f0, gain, q, mode="fft") # Test differenziabilità su FFT

    loss_eq = torch.sum(y_eq ** 2)
    loss_eq.backward()

    assert f0.grad is not None and not torch.isnan(f0.grad).any()
    assert gain.grad is not None and not torch.isnan(gain.grad).any()
    assert q.grad is not None and not torch.isnan(q.grad).any()

    # Crossover params con gradiente
    f_low = torch.tensor(150.0, requires_grad=True)
    f_high = torch.tensor(3000.0, requires_grad=True)

    xo = DifferentiableLR4Crossover(sample_rate=48000.0)
    y_low, y_mid, y_high = xo(x, f_low, f_high)

    loss_xo = torch.sum(y_low ** 2) + 0.5 * torch.sum(y_mid ** 2) + 2.0 * torch.sum(y_high ** 2)
    loss_xo.backward()

    assert f_low.grad is not None and not torch.isnan(f_low.grad).any()
    assert f_high.grad is not None and not torch.isnan(f_high.grad).any()
