import pytest
import numpy as np
import torch
import py_core_dsp
from ddsp_mastering.dsp.imager import DifferentiableMidSideImager
from ddsp_mastering.dsp.saturator import DifferentiableHarmonicSaturator

def compute_null_dbfs(y_cpp: np.ndarray, y_torch: np.ndarray) -> float:
    diff = y_cpp - y_torch
    mse = np.mean(diff ** 2)
    return float(10.0 * np.log10(mse + 1e-16))

def compute_max_abs_err_dbfs(y_cpp: np.ndarray, y_torch: np.ndarray) -> float:
    max_err = np.max(np.abs(y_cpp - y_torch))
    return float(20.0 * np.log10(max_err + 1e-12))


# === 1. PARITÀ E LOGICA STEREO IMAGER ===
def test_imager_single_band_parity(sig_sweep):
    x_torch = sig_sweep # [2, 96000]
    x_np = x_torch.numpy()

    widths = [0.0, 0.5, 1.0, 1.4, 2.0]
    imager_py = DifferentiableMidSideImager()

    for w in widths:
        # C++
        y_cpp = py_core_dsp.MidSideImager.process_band(x_np, w)
        # PyTorch
        y_torch = imager_py.process_band(x_torch, torch.tensor(w)).numpy()

        null_db = compute_null_dbfs(y_cpp, y_torch)
        assert null_db <= -120.0, f"Imager Null {null_db} dBFS > -120 dBFS for width {w}"

def test_imager_identity_and_mono(sig_pink_music):
    x = sig_pink_music
    imager_py = DifferentiableMidSideImager()

    # Width = 1.0: identità perfetta
    y_ident = imager_py.process_band(x, torch.tensor(1.0))
    assert torch.allclose(x, y_ident, atol=1e-6, rtol=1e-6)

    # Width = 0.0: mono perfetto (L == R)
    y_mono = imager_py.process_band(x, torch.tensor(0.0))
    diff_lr = torch.abs(y_mono[0] - y_mono[1])
    assert torch.max(diff_lr).item() < 1e-7

def test_imager_3band_sum_parity(sig_sweep):
    x_np = sig_sweep.numpy()
    # Generiamo 3 bande fittizie per testare il summing
    low_np = x_np * 0.5
    mid_np = x_np * 0.3
    high_np = x_np * 0.2

    low_t = torch.from_numpy(low_np)
    mid_t = torch.from_numpy(mid_np)
    high_t = torch.from_numpy(high_np)

    w_l, w_m, w_h = 0.2, 1.1, 1.8

    # C++
    sum_cpp = py_core_dsp.MidSideImager.process_3band_and_sum(
        low_np, mid_np, high_np, w_l, w_m, w_h
    )

    # PyTorch
    imager_py = DifferentiableMidSideImager()
    sum_torch = imager_py(
        low_t, mid_t, high_t,
        torch.tensor(w_l), torch.tensor(w_m), torch.tensor(w_h)
    ).numpy()

    null_db = compute_null_dbfs(sum_cpp, sum_torch)
    print(f"\n[Imager 3-Band Sum Parity] Null: {null_db:.2f} dBFS")
    assert null_db <= -120.0, f"Imager 3-Band Null {null_db} > -120 dBFS"


# === 2. PARITÀ HARMONIC SATURATOR CON OVERSAMPLING 2X ===
@pytest.mark.parametrize("sig_fixture_name", ["sig_sweep", "sig_burst", "sig_pink_music"])
def test_saturator_parity(sig_fixture_name, request):
    x_torch = request.getfixturevalue(sig_fixture_name)
    x_np = x_torch.numpy()

    drive_db = 6.0
    mix = 0.75

    # C++
    sat_cpp = py_core_dsp.HarmonicSaturator()
    sat_cpp.set_parameters(drive_db, mix)
    y_cpp = sat_cpp.process(x_np)

    # PyTorch
    sat_py = DifferentiableHarmonicSaturator()
    drive_t = torch.tensor(drive_db)
    mix_t = torch.tensor(mix)
    y_torch = sat_py(x_torch, drive_t, mix_t).numpy()

    # Escludiamo i primi 32 campioni di transitorio iniziale del filtro
    null_db = compute_null_dbfs(y_cpp[:, 32:], y_torch[:, 32:])
    max_err_db = compute_max_abs_err_dbfs(y_cpp[:, 32:], y_torch[:, 32:])

    print(f"\n[Saturator Parity - {sig_fixture_name}] Null: {null_db:.2f} dBFS, MaxErr: {max_err_db:.2f} dBFS")
    assert null_db <= -115.0, f"Saturator Null {null_db} > -115 dBFS"


# === 3. ALLINEAMENTO DI FASE DRY/WET (ASSENZA COMB-FILTERING) ===
def test_saturator_phase_alignment(sig_impulse):
    """
    Verifica che con Drive = 0 dB e Mix = 0.5 (somma 50% dry + 50% wet),
    il ritardo di 15 campioni compensi esattamente la latenza del FIR 2x,
    evitando notch o comb-filtering distruttivi.
    """
    sat_cpp = py_core_dsp.HarmonicSaturator()
    sat_cpp.set_parameters(0.0, 0.5) # Drive 0 dB, Mix 50%
    y_cpp = sat_cpp.process(sig_impulse.numpy())

    # La risposta impulsiva deve avere il picco principale esattamente a n = 100 + 15 = 115
    peak_idx = np.argmax(np.abs(y_cpp[0]))
    assert peak_idx == 115, f"Peak index {peak_idx} != 115 (atteso ritardo di 15 campioni)"

    # Lo spettro tra 20 Hz e 18 kHz deve essere piatto senza zeri
    fft_y = np.fft.rfft(y_cpp[0])
    mag_db = 20.0 * np.log10(np.abs(fft_y) + 1e-12)
    # Calcola ripple massimo nella banda audio fino a 15 kHz
    k_15k = int(15000.0 / (48000.0 / 2.0) * len(mag_db))
    ripple = np.max(mag_db[:k_15k]) - np.min(mag_db[:k_15k])

    print(f"\n[Saturator Dry/Wet Alignment] Ripple in banda audio: {ripple:.4f} dB")
    assert ripple < 0.2, f"Comb-filtering detected! Ripple {ripple} dB >= 0.2 dB"


# === 4. GRADIENT FLOW AUTOGRAD ===
def test_saturator_and_imager_gradcheck():
    torch.manual_seed(42)
    x = torch.randn(2, 2048, dtype=torch.float32)

    # Imager
    w_low = torch.tensor(0.5, requires_grad=True)
    w_mid = torch.tensor(1.2, requires_grad=True)
    w_high = torch.tensor(1.8, requires_grad=True)

    imager = DifferentiableMidSideImager()
    y_imager = imager(x * 0.5, x * 0.3, x * 0.2, w_low, w_mid, w_high)
    loss_im = torch.sum(y_imager ** 2)
    loss_im.backward()

    assert w_low.grad is not None and not torch.isnan(w_low.grad).any()
    assert w_mid.grad is not None and not torch.isnan(w_mid.grad).any()
    assert w_high.grad is not None and not torch.isnan(w_high.grad).any()

    # Saturator
    drive = torch.tensor(6.0, requires_grad=True)
    mix = torch.tensor(0.5, requires_grad=True)

    sat = DifferentiableHarmonicSaturator()
    y_sat = sat(x, drive, mix)
    loss_sat = torch.sum(y_sat ** 2)
    loss_sat.backward()

    assert drive.grad is not None and not torch.isnan(drive.grad).any()
    assert mix.grad is not None and not torch.isnan(mix.grad).any()
