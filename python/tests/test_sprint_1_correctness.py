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
from ddsp_mastering.dsp.chain import DifferentiableMasteringChain
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
    torch.manual_seed(42)
    x = torch.randn(1, 2, 9600, dtype=torch.float64) * 0.8  # loud dynamic signal
    threshold_db = torch.tensor([-24.0], dtype=torch.float64)
    ratio = torch.tensor([1.0], dtype=torch.float64)          # 1.0:1
    attack_ms = torch.tensor([10.0], dtype=torch.float64)
    release_ms = torch.tensor([100.0], dtype=torch.float64)
    makeup_db = torch.tensor([0.0], dtype=torch.float64)

    out = DifferentiableSingleBandCompressorFunction.apply(
        x, threshold_db, ratio, attack_ms, release_ms, makeup_db, 48000.0
    )

    # When ratio is 1.0, output must be mathematically identical to input (error < 1e-12)
    diff = torch.max(torch.abs(out - x)).item()
    assert diff < 1e-12, f"Compressor with ratio=1.0 altered signal by {diff:.2e}, expected < 1e-12"

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
    N = 64
    g_min = torch.rand((1, N), dtype=torch.float64, requires_grad=True)
    alpha_rel = 0.995

    test_passed = torch.autograd.gradcheck(
        FastLimiterReleaseFunction.apply,
        (g_min, alpha_rel),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
    )
    assert test_passed, "FastLimiterReleaseFunction failed torch.autograd.gradcheck"


def test_fast_limiter_vs_pure_pytorch_reference():
    """Formally compare FastLimiterReleaseFunction forward and backward with pure PyTorch reference recurrence."""
    def pure_pytorch_limiter_envelope(g_min: torch.Tensor, alpha_rel: float) -> torch.Tensor:
        B, N = g_min.shape
        env_steps = []
        g_prev = torch.ones(B, dtype=g_min.dtype, device=g_min.device)
        for n in range(N):
            gm = g_min[:, n]
            is_rel = gm > g_prev
            g_curr = torch.where(is_rel, alpha_rel * g_prev + (1.0 - alpha_rel) * gm, gm)
            env_steps.append(g_curr)
            g_prev = g_curr
        return torch.stack(env_steps, dim=1)

    torch.manual_seed(42)
    for B in (1, 2):  # Tests B=1 (C++ fast path) and B=2 (PyTorch fallback path)
        N = 128
        alpha_rel = 0.99
        t = torch.linspace(0, 1, N, dtype=torch.float64)
        if B == 1:
            g_min_raw = (0.5 + 0.4 * torch.sin(2 * np.pi * 5 * t)).unsqueeze(0)
        else:
            g_min_raw = torch.stack([
                0.5 + 0.4 * torch.sin(2 * np.pi * 5 * t),
                0.6 + 0.3 * torch.cos(2 * np.pi * 3 * t)
            ], dim=0)

        # Introduce distinct attack steps and release recovery
        g_min_raw[:, 20:25] = 0.2
        g_min_raw[:, 70:75] = 0.1

        # Reference
        g_ref = g_min_raw.clone().detach().requires_grad_(True)
        env_ref = pure_pytorch_limiter_envelope(g_ref, alpha_rel)

        # Fast custom
        g_fast = g_min_raw.clone().detach().requires_grad_(True)
        env_fast = FastLimiterReleaseFunction.apply(g_fast, alpha_rel)

        # Forward match
        fwd_diff = torch.max(torch.abs(env_fast - env_ref)).item()
        assert fwd_diff < 1e-7, f"Batch={B} forward diff {fwd_diff:.2e} exceeds 1e-7"

        # Backward match
        grad_upstream = torch.randn_like(env_fast, dtype=torch.float64)
        env_fast.backward(grad_upstream)
        env_ref.backward(grad_upstream)

        bwd_diff = torch.max(torch.abs(g_fast.grad - g_ref.grad)).item()
        assert bwd_diff < 1e-6, f"Batch={B} backward gradient diff {bwd_diff:.2e} exceeds 1e-6"


def test_true_bypass_latency_and_invariance():
    """Verify True Bypass routes original audio through dedicated 159-sample delay line with exact float copy."""
    fs = 48000.0
    chain = py_core_dsp.MasteringChain(fs)

    # 1. State check
    assert not chain.is_bypass()
    chain.set_bypass(True)
    assert chain.is_bypass()
    latency = chain.get_latency_samples()
    assert latency == 159

    # 2. Non-correlated stereo noise test
    np.random.seed(101)
    N = 2000
    x = np.random.randn(2, N).astype(np.float32) * 0.5
    y = chain.process(x)

    # After 159 samples, y must match x exactly (delay line bit-level float copy)
    max_diff = np.max(np.abs(y[:, latency:] - x[:, : N - latency]))
    assert max_diff == 0.0, f"True Bypass altered delayed audio by {max_diff:.2e}"

    # 3. Impulse / Latency check
    chain.reset()
    imp = np.zeros((2, 512), dtype=np.float32)
    imp[0, 0] = 1.0
    imp[1, 0] = -0.5
    y_imp = chain.process(imp)
    assert y_imp[0, latency] == 1.0
    assert y_imp[1, latency] == -0.5
    assert np.all(y_imp[:, :latency] == 0.0)
    assert np.all(y_imp[:, latency + 1:] == 0.0)

    # 4. Parameter Invariance while in bypass
    chain.set_parameters_z([0.95] * NUM_PARAMS)
    x2 = np.random.randn(2, 512).astype(np.float32) * 0.4
    y2 = chain.process(x2)
    diff_inv = np.max(np.abs(y2[:, latency:] - x2[:, : 512 - latency]))
    assert diff_inv == 0.0, f"Modifying parameters while bypassed altered audio by {diff_inv:.2e}"

    # 5. Toggle back to active DSP
    chain.set_bypass(False)
    assert not chain.is_bypass()


def test_neutral_chain_rigorous_characterization():
    """Verify THETA_NEUTRAL characterization: preserves energy and stereo image under headroom, respects ceiling."""
    chain = py_core_dsp.MasteringChain(48000.0)
    chain.set_parameters_z(Z_NEUTRAL.tolist())

    fs = 48000
    t = np.linspace(0, 1.0, fs, endpoint=False)
    sig_l = 0.2 * np.sin(2 * np.pi * 80.0 * t) + 0.15 * np.sin(2 * np.pi * 1000.0 * t) + 0.1 * np.sin(2 * np.pi * 8000.0 * t)
    sig_r = 0.2 * np.cos(2 * np.pi * 95.0 * t) + 0.15 * np.cos(2 * np.pi * 1200.0 * t) + 0.1 * np.sin(2 * np.pi * 6000.0 * t)
    np.random.seed(123)
    noise = np.random.randn(2, fs).astype(np.float32) * 0.02
    x = np.stack([sig_l, sig_r], axis=0).astype(np.float32) + noise

    # Scale to -6 dBFS peak (safely under -0.3 dBFS limiter ceiling to isolate filter/compressor response)
    x = x * (10.0 ** (-6.0 / 20.0) / np.max(np.abs(x)))

    y = chain.process(x)

    latency = 159
    settle = 480  # 10 ms settle time for filter transients
    x_aligned = x[:, settle : fs - latency]
    y_aligned = y[:, settle + latency : fs]

    # 1. Delta RMS
    rms_x = np.sqrt(np.mean(x_aligned ** 2))
    rms_y = np.sqrt(np.mean(y_aligned ** 2))
    delta_rms_db = abs(20 * np.log10(rms_y / rms_x))
    assert delta_rms_db < 0.05, f"Neutral preset altered RMS by {delta_rms_db:.4f} dB (expected < 0.05 dB)"

    # 2. Stereo Correlation Preservation (no stereo collapse)
    corr_x = np.corrcoef(x_aligned[0], x_aligned[1])[0, 1]
    corr_y = np.corrcoef(y_aligned[0], y_aligned[1])[0, 1]
    delta_corr = abs(corr_x - corr_y)
    assert delta_corr < 0.02, f"Stereo correlation altered by {delta_corr:.4f}"

    # 3. Limiter Safety Check on loud transients (+3 dBFS peak)
    chain.reset()
    x_loud = x * 3.0
    y_loud = chain.process(x_loud)
    max_peak = np.max(np.abs(y_loud))
    ceiling_lin = 10.0 ** (-0.3 / 20.0) + 1e-3  # ~0.967
    assert max_peak <= ceiling_lin, f"Limiter allowed peak {max_peak:.4f} > {ceiling_lin:.4f}"


def test_optimizer_neutral_boundary_initialization():
    """Verify optimizer initialization near Z_NEUTRAL boundary is stable and produces valid non-zero gradients."""
    z_init = torch.clamp(Z_NEUTRAL, 1e-4, 1.0 - 1e-4)
    theta_init = map_z_to_theta(z_init.unsqueeze(0)).squeeze(0)

    # Compressor ratios (indices 21, 26, 31) are initialized extremely close to 1.0 (within 0.02%)
    assert 1.0 <= float(theta_init[21]) < 1.01
    assert 1.0 <= float(theta_init[26]) < 1.01
    assert 1.0 <= float(theta_init[31]) < 1.01

    # Check forward and backward with learnable logit u
    u = torch.logit(z_init).clone().detach().requires_grad_(True)
    chain = DifferentiableMasteringChain(sample_rate=48000.0, default_eq_mode="fft")
    x = torch.randn(1, 2, 4800, dtype=torch.float32) * 0.1
    z = torch.sigmoid(u)
    y = chain(x, z=z, eq_mode="fft")
    loss = torch.mean(y ** 2)
    loss.backward()

    assert not torch.isnan(u.grad).any(), "NaN gradient detected at boundary init"
    assert not torch.isinf(u.grad).any(), "Inf gradient detected at boundary init"
    assert torch.max(torch.abs(u.grad)) > 0.0, "Zero gradient detected at boundary init"


def test_pytorch_iir_vs_cpp_parity_matrix():
    """Systematic parity matrix between PyTorch IIR cascade and C++ core across 7 test conditions."""
    fs = 48000.0
    eq_py = DifferentiableEQ(fs, default_mode="iir")
    eq_cpp = py_core_dsp.BiquadCascade6Band()

    f0_flat = torch.tensor([[60.0, 250.0, 1000.0, 3000.0, 8000.0, 12000.0]])
    q_flat = torch.ones((1, 6)) * 0.707

    # Case 1: Stereo unit impulse (flat EQ)
    imp = np.zeros((1, 2, 512), dtype=np.float32)
    imp[0, 0, 0] = 1.0
    imp[0, 1, 0] = 0.5
    g_flat = torch.zeros((1, 6))
    y_py1 = eq_py(torch.from_numpy(imp), f0_flat, g_flat, q_flat, mode="iir").squeeze(0).numpy()
    eq_cpp.set_parameters(f0_flat[0].tolist(), g_flat[0].tolist(), q_flat[0].tolist(), fs)
    y_cpp1 = eq_cpp.process(imp[0])
    err1 = np.max(np.abs(y_py1 - y_cpp1))
    assert err1 < 1e-6, f"Case 1 (impulse flat) error {err1:.2e} >= 1e-6"

    # Case 2: Sinusoid low (80 Hz) with low-shelf boost (+4 dB)
    t = np.linspace(0, 0.2, int(0.2 * fs), endpoint=False)
    x2 = np.stack([np.sin(2 * np.pi * 80.0 * t), np.cos(2 * np.pi * 80.0 * t)], axis=0).astype(np.float32)
    g_low = torch.tensor([[4.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    f0_low = torch.tensor([[80.0, 250.0, 1000.0, 3000.0, 8000.0, 12000.0]])
    eq_cpp.reset()
    eq_cpp.set_parameters(f0_low[0].tolist(), g_low[0].tolist(), q_flat[0].tolist(), fs)
    y_cpp2 = eq_cpp.process(x2)
    y_py2 = eq_py(torch.from_numpy(x2).unsqueeze(0), f0_low, g_low, q_flat, mode="iir").squeeze(0).numpy()
    err2 = np.max(np.abs(y_py2 - y_cpp2))
    assert err2 < 1e-4, f"Case 2 (sinusoid low) error {err2:.2e} >= 1e-4"

    # Case 3: Sinusoid mid (1000 Hz) with peaking cut (-3 dB)
    x3 = np.stack([np.sin(2 * np.pi * 1000.0 * t), np.cos(2 * np.pi * 1000.0 * t)], axis=0).astype(np.float32)
    g_mid = torch.tensor([[0.0, 0.0, -3.0, 0.0, 0.0, 0.0]])
    eq_cpp.reset()
    eq_cpp.set_parameters(f0_flat[0].tolist(), g_mid[0].tolist(), q_flat[0].tolist(), fs)
    y_cpp3 = eq_cpp.process(x3)
    y_py3 = eq_py(torch.from_numpy(x3).unsqueeze(0), f0_flat, g_mid, q_flat, mode="iir").squeeze(0).numpy()
    err3 = np.max(np.abs(y_py3 - y_cpp3))
    assert err3 < 1e-4, f"Case 3 (sinusoid mid) error {err3:.2e} >= 1e-4"

    # Case 4: Sinusoid high (10000 Hz) with high-shelf boost (+3 dB) near Nyquist
    x4 = np.stack([np.sin(2 * np.pi * 10000.0 * t), np.cos(2 * np.pi * 10000.0 * t)], axis=0).astype(np.float32)
    g_high = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 3.0]])
    eq_cpp.reset()
    eq_cpp.set_parameters(f0_flat[0].tolist(), g_high[0].tolist(), q_flat[0].tolist(), fs)
    y_cpp4 = eq_cpp.process(x4)
    y_py4 = eq_py(torch.from_numpy(x4).unsqueeze(0), f0_flat, g_high, q_flat, mode="iir").squeeze(0).numpy()
    err4 = np.max(np.abs(y_py4 - y_cpp4))
    assert err4 < 1e-4, f"Case 4 (sinusoid high) error {err4:.2e} >= 1e-4"

    # Case 5: Independent stereo broadband noise
    np.random.seed(55)
    x5 = np.random.randn(2, 24000).astype(np.float32) * 0.2
    g_arb = torch.tensor([[2.0, -3.0, 1.5, -2.0, 3.0, -1.0]])
    q_arb = torch.tensor([[1.0, 1.4, 0.7, 1.2, 0.8, 0.7]])
    eq_cpp.reset()
    eq_cpp.set_parameters(f0_flat[0].tolist(), g_arb[0].tolist(), q_arb[0].tolist(), fs)
    y_cpp5 = eq_cpp.process(x5)
    y_py5 = eq_py(torch.from_numpy(x5).unsqueeze(0), f0_flat, g_arb, q_arb, mode="iir").squeeze(0).numpy()
    err5 = np.max(np.abs(y_py5 - y_cpp5))
    null_rms5 = np.sqrt(np.mean((y_py5 - y_cpp5) ** 2))
    null_rms_dbfs5 = 20 * np.log10(null_rms5 + 1e-12)
    assert err5 < 1e-4, f"Case 5 (noise) error {err5:.2e} >= 1e-4"
    assert null_rms_dbfs5 < -80.0, f"Case 5 null {null_rms_dbfs5:.1f} dBFS >= -80 dBFS"

    # Case 6: Block streaming continuity (2 continuous blocks in C++ vs full stream in PyTorch)
    x6 = np.random.randn(2, 2048).astype(np.float32)
    eq_cpp.reset()
    b1 = eq_cpp.process(np.ascontiguousarray(x6[:, :1024]))
    b2 = eq_cpp.process(np.ascontiguousarray(x6[:, 1024:]))
    y_cpp6 = np.concatenate([b1, b2], axis=1)
    y_py6 = eq_py(torch.from_numpy(x6).unsqueeze(0), f0_flat, g_arb, q_arb, mode="iir").squeeze(0).numpy()
    err6 = np.max(np.abs(y_py6 - y_cpp6))
    assert err6 < 1e-4, f"Case 6 (streaming continuity) error {err6:.2e} >= 1e-4"


def test_optimizer_fft_to_iir_transition():
    """Verify optimizer smoothly transitions across the FFT -> IIR threshold without NaNs, Infs, or instability."""
    torch.manual_seed(42)
    num_steps = 10
    opt = AutoMasteringOptimizer(sample_rate=48000.0, num_steps=num_steps, lr=0.03)

    # 0.5s stereo audio for swift deterministic test execution
    t = torch.linspace(0, 0.5, 24000).unsqueeze(0).repeat(2, 1)
    x = (0.2 * torch.sin(2 * np.pi * 440.0 * t)).unsqueeze(0)
    target = x * 1.1

    optimal_z, final_master, history = opt.optimize(raw_mix=x, target_reference=target, verbose=False)

    losses = history["loss"]
    assert len(losses) == num_steps

    # Transition threshold is at step 7 (int(10 * 0.7))
    transition_idx = int(num_steps * 0.7)
    loss_fft_last = losses[transition_idx - 1]
    loss_iir_first = losses[transition_idx]

    # Verify all losses are finite
    for step_idx, l in enumerate(losses):
        assert not np.isnan(l), f"NaN loss at step {step_idx}"
        assert not np.isinf(l), f"Inf loss at step {step_idx}"
        assert l > 0.0

    # Verify transition continuity: delta across mode switch is smooth
    delta_transition = abs(loss_iir_first - loss_fft_last)
    assert delta_transition < 1.0, f"Transition loss delta was {delta_transition:.4f}, expected < 1.0"

    # Verify final master audio is finite and bounded
    assert not torch.isnan(final_master).any()
    assert not torch.isinf(final_master).any()
    assert torch.max(torch.abs(final_master)).item() <= 1.0

    # Verify optimal_z is within valid [0, 1] bounds
    assert (optimal_z >= 0.0).all() and (optimal_z <= 1.0).all()


def test_saturator_small_signal_unit_gain():
    """Verify that saturator has strictly 0 dB linear small-signal gain (1/gd normalisation)."""
    sat_py = DifferentiableHarmonicSaturator()
    drive_db = torch.tensor([6.0])  # high drive
    mix = torch.tensor([1.0])       # 100% wet

    # Very small signal: -80 dBFS stereo [1, 2, T]
    small_amp = 1e-4
    t = torch.linspace(0, 0.1, 4800, dtype=torch.float32)
    sine = (small_amp * torch.sin(2 * np.pi * 440.0 * t)).unsqueeze(0)
    x = torch.cat([sine, sine], dim=0).unsqueeze(0)

    y = sat_py(x, drive_db, mix)

    gain = (torch.max(torch.abs(y[:, :, 100:])) / torch.max(torch.abs(x[:, :, 100:]))).item()
    gain_db = 20 * np.log10(gain)
    assert abs(gain_db) < 0.05, f"PyTorch small-signal gain was {gain_db:.3f} dB, expected ~0 dB"

    # Parity with C++ saturator
    sat_cpp = py_core_dsp.HarmonicSaturator()
    sat_cpp.set_parameters(6.0, 1.0)
    x_np = x.squeeze(0).numpy()
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
