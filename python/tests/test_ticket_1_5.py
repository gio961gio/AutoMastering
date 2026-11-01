import pytest
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import py_core_dsp
from ddsp_mastering.dsp.limiter import DifferentiableLookaheadLimiter, CEILING_DBFS, C_LIN
from ddsp_mastering.dsp.chain import DifferentiableMasteringChain
from ddsp_mastering.param_space import map_z_to_theta, NUM_PARAMS

def compute_null_dbfs(y_cpp: np.ndarray, y_torch: np.ndarray) -> float:
    diff = y_cpp - y_torch
    mse = np.mean(diff ** 2)
    return float(10.0 * np.log10(mse + 1e-16))

def compute_max_abs_err_dbfs(y_cpp: np.ndarray, y_torch: np.ndarray) -> float:
    max_err = np.max(np.abs(y_cpp - y_torch))
    return float(20.0 * np.log10(max_err + 1e-12))


# === 1. PARITÀ & BRICKWALL CEILING DEL LIMITER ===
@pytest.mark.parametrize("push_db", [0.0, 3.0, 6.0, 12.0])
def test_limiter_brickwall_ceiling_and_parity(sig_pink_music, push_db):
    x_torch = sig_pink_music # [2, 96000]
    x_np = x_torch.numpy()

    # C++ LookaheadLimiter
    cpp_lim = py_core_dsp.LookaheadLimiter(48000.0)
    cpp_lim.set_push_gain_db(push_db)
    y_cpp = cpp_lim.process(x_np)

    # PyTorch DifferentiableLookaheadLimiter
    torch_lim = DifferentiableLookaheadLimiter(sample_rate=48000.0)
    t_push = torch.tensor(push_db, dtype=torch.float32)
    y_torch = torch_lim(x_torch, t_push).numpy()

    null_db = compute_null_dbfs(y_cpp, y_torch)
    max_err_db = compute_max_abs_err_dbfs(y_cpp, y_torch)

    print(f"\n[Limiter Push {push_db:+.1f} dB] Null: {null_db:.2f} dBFS, MaxErr: {max_err_db:.2f} dBFS")
    assert null_db <= -110.0, f"Limiter Null {null_db} dBFS > -110 dBFS at push {push_db} dB"

    # Verifica vincolo di sicurezza Brickwall Ceiling: nessun campione deve superare C_LIN (~0.96605)
    max_peak_cpp = float(np.max(np.abs(y_cpp)))
    max_peak_torch = float(np.max(np.abs(y_torch)))
    ceiling_lin = C_LIN + 1e-4

    print(f"[Ceiling Check] C++ peak: {max_peak_cpp:.6f}, Torch peak: {max_peak_torch:.6f} (Ceiling: {C_LIN:.6f})")
    assert max_peak_cpp <= ceiling_lin, f"C++ limiter exceeded ceiling: {max_peak_cpp} > {ceiling_lin}"
    assert max_peak_torch <= ceiling_lin, f"PyTorch limiter exceeded ceiling: {max_peak_torch} > {ceiling_lin}"


# === 2. LATENZA LOOKAHEAD LIMITER (144 CAMPIONI) ===
def test_limiter_lookahead_latency(sig_impulse):
    # sig_impulse ha il delta a n = 100
    x_np = sig_impulse.numpy()
    cpp_lim = py_core_dsp.LookaheadLimiter(48000.0)
    cpp_lim.set_push_gain_db(0.0)
    y_cpp = cpp_lim.process(x_np)

    # Il picco deve essere ritardato esattamente di 144 campioni: n = 100 + 144 = 244
    peak_idx_cpp = int(np.argmax(np.abs(y_cpp[0])))
    assert peak_idx_cpp == 244, f"C++ Limiter latency mismatch: expected 244, got {peak_idx_cpp}"

    torch_lim = DifferentiableLookaheadLimiter(sample_rate=48000.0)
    y_torch = torch_lim(sig_impulse, torch.tensor(0.0)).numpy()
    peak_idx_torch = int(np.argmax(np.abs(y_torch[0])))
    assert peak_idx_torch == 244, f"PyTorch Limiter latency mismatch: expected 244, got {peak_idx_torch}"


# === 3. LATENZA TOTALE CATENA COMPLETA (159 CAMPIONI) ===
def test_mastering_chain_total_latency(sig_impulse):
    # Dtot = Dsat (15) + Dlim (144) = 159 campioni
    # Con impulso a n = 100, il picco deve emergere esattamente a n = 100 + 159 = 259
    x_np = sig_impulse.numpy()

    # Vettore z neutro a 0.5
    z_neutral = [0.5] * NUM_PARAMS

    # C++ MasteringChain
    cpp_chain = py_core_dsp.MasteringChain(48000.0)
    cpp_chain.set_parameters_z(z_neutral)
    y_cpp = cpp_chain.process(x_np)

    assert cpp_chain.get_latency_samples() == 159

    # PyTorch MasteringChain
    torch_chain = DifferentiableMasteringChain(sample_rate=48000.0, default_eq_mode="iir")
    assert torch_chain.latency_samples == 159

    z_t = torch.tensor(z_neutral, dtype=torch.float32)
    y_torch = torch_chain(sig_impulse, z=z_t).numpy()

    peak_idx_cpp = int(np.argmax(np.abs(y_cpp[0])))
    peak_idx_torch = int(np.argmax(np.abs(y_torch[0])))

    print(f"\n[Total Latency Peak Check] C++ peak at n={peak_idx_cpp}, Torch peak at n={peak_idx_torch}")
    assert peak_idx_cpp == peak_idx_torch, f"Latency mismatch between C++ ({peak_idx_cpp}) and Torch ({peak_idx_torch})"
    assert peak_idx_cpp in (259, 260), f"Expected total latency around 159 (+1 AP2 group delay), got {peak_idx_cpp - 100}"


# === 4. PARITÀ CATENA COMPLETA C++ vs PyTorch (IIR Exact Mode <= -100 dBFS) ===
def test_mastering_chain_iir_parity(sig_pink_music):
    torch.manual_seed(123)
    # Parametri normalizzati realistici
    z = torch.tensor([
        0.55, 0.45, 0.50, # EQ Band 0
        0.48, 0.52, 0.50, # EQ Band 1
        0.50, 0.47, 0.50, # EQ Band 2
        0.52, 0.53, 0.50, # EQ Band 3
        0.49, 0.51, 0.50, # EQ Band 4
        0.53, 0.48, 0.50, # EQ Band 5
        0.35, 0.65,       # Crossover f_low, f_high
        0.45, 0.30, 0.40, 0.40, 0.55, # Comp Low
        0.40, 0.25, 0.35, 0.35, 0.52, # Comp Mid
        0.50, 0.35, 0.25, 0.25, 0.58, # Comp High
        0.40, 0.50, 0.60, # Imager widths
        0.30, 0.40,       # Saturator drive, mix
        0.50              # Limiter push
    ], dtype=torch.float32)

    x_np = sig_pink_music.numpy()
    z_list = z.tolist()

    # C++ MasteringChain
    cpp_chain = py_core_dsp.MasteringChain(48000.0)
    cpp_chain.set_parameters_z(z_list)
    y_cpp = cpp_chain.process(x_np)

    # PyTorch MasteringChain (IIR mode)
    torch_chain = DifferentiableMasteringChain(sample_rate=48000.0, default_eq_mode="iir")
    y_torch = torch_chain(sig_pink_music, z=z, eq_mode="iir").numpy()

    null_db = compute_null_dbfs(y_cpp, y_torch)
    max_err_db = compute_max_abs_err_dbfs(y_cpp, y_torch)

    print(f"\n[MasteringChain Full Parity - IIR] Null: {null_db:.2f} dBFS, MaxErr: {max_err_db:.2f} dBFS")
    assert null_db <= -100.0, f"Full Chain IIR Null {null_db} dBFS > -100.0 dBFS"
    assert max_err_db <= -85.0, f"Full Chain IIR MaxErr {max_err_db} dBFS > -85.0 dBFS"


# === 5. PARITÀ CATENA COMPLETA FAST FFT MODE (<= -68 dBFS) ===
def test_mastering_chain_fft_parity(sig_pink_music):
    z = torch.full((NUM_PARAMS,), 0.5, dtype=torch.float32)
    x_np = sig_pink_music.numpy()

    # C++ MasteringChain
    cpp_chain = py_core_dsp.MasteringChain(48000.0)
    cpp_chain.set_parameters_z(z.tolist())
    y_cpp = cpp_chain.process(x_np)

    # PyTorch MasteringChain (Fast FFT mode)
    torch_chain = DifferentiableMasteringChain(sample_rate=48000.0, default_eq_mode="fft")
    y_torch = torch_chain(sig_pink_music, z=z, eq_mode="fft").numpy()

    null_db = compute_null_dbfs(y_cpp, y_torch)
    max_err_db = compute_max_abs_err_dbfs(y_cpp, y_torch)

    print(f"\n[MasteringChain Full Parity - Fast FFT] Null: {null_db:.2f} dBFS, MaxErr: {max_err_db:.2f} dBFS")
    assert null_db <= -68.0, f"Full Chain FFT Null {null_db} dBFS > -68.0 dBFS"


# === 6. TOY OPTIMIZATION / SYSTEM IDENTIFICATION SANITY CHECK ===
def test_toy_optimization_system_identification():
    """
    Sanity Check System Identification con torch.optim.Adam:
    1. Si genera l'audio target elaborando un breve segnale con z_target noto.
    2. Si inizializza z_learnable = 0.5 * 1_41 con requires_grad=True.
    3. Si ottimizza con Adam (lr=0.03) verificando la discesa della loss e il flusso dei gradienti.
    """
    torch.manual_seed(999)
    # Segnale musicale sintetico breve di 4096 campioni (~85 ms) per velocità e convergenza deterministica
    x = torch.randn(1, 2, 4096, dtype=torch.float32) * 0.2

    chain = DifferentiableMasteringChain(sample_rate=48000.0, default_eq_mode="fft")

    # Target: parametri realistici non-banali
    z_target = torch.tensor([
        0.60, 0.55, 0.50,
        0.45, 0.48, 0.50,
        0.52, 0.52, 0.50,
        0.48, 0.46, 0.50,
        0.55, 0.54, 0.50,
        0.50, 0.50, 0.50,
        0.30, 0.70,
        0.40, 0.30, 0.40, 0.40, 0.50,
        0.40, 0.30, 0.40, 0.40, 0.50,
        0.40, 0.30, 0.40, 0.40, 0.50,
        0.45, 0.50, 0.55,
        0.35, 0.30,
        0.40
    ], dtype=torch.float32)

    with torch.no_grad():
        y_target = chain(x, z=z_target, eq_mode="fft")

    # Parametro ottimizzabile inizializzato al centro neutro (0.5)
    z_learnable = nn.Parameter(torch.full((NUM_PARAMS,), 0.5, dtype=torch.float32, requires_grad=True))
    optimizer = optim.Adam([z_learnable], lr=0.04)

    initial_loss = 0.0
    final_loss = 0.0

    print("\n[Toy Optimization System Identification]")
    for step in range(40):
        optimizer.zero_grad()
        y_pred = chain(x, z=torch.clamp(z_learnable, 0.01, 0.99), eq_mode="fft")

        loss = torch.mean((y_pred - y_target) ** 2)
        loss.backward()

        if step == 0:
            initial_loss = loss.item()
            # Verifica gradiente su tutti i 41 parametri
            grad_norm = float(torch.norm(z_learnable.grad).item())
            print(f"Step 0 Initial Loss: {initial_loss:.6e}, Grad Norm: {grad_norm:.6e}")
            assert not torch.isnan(z_learnable.grad).any(), "NaN in gradients"
            assert not torch.isinf(z_learnable.grad).any(), "Inf in gradients"
            assert grad_norm > 1e-5, f"Gradient norm too small: {grad_norm}"

        optimizer.step()
        final_loss = loss.item()

    print(f"Step 40 Final Loss: {final_loss:.6e} (Loss Ratio: {final_loss / initial_loss:.4f})")
    assert final_loss < 0.3 * initial_loss, f"Loss did not decrease sufficiently: {final_loss} >= 0.3 * {initial_loss}"
