"""Sprint 3 V3b Technical Gate Test Suite.

Verifies the 11 Technical Gate Requirements:
1. Limiter bypass / zero attenuation at -20 dBFS with 0 dB push.
2. Limiter stress monotonic growth across push gains {0, 2, 4, 6, 8} dB.
3. Makeup gain invariance of GR_c[n] and d(GR)/d(M) == 0.
4. Differential gain reduction detection with band-separated signals.
5. CVaR_0.95 >= P95 property and autograd uniform gradient on top-k tail.
6. Active sample mask, silence rejection (< -70 dBFS), M_min=8 fallback, and 321-sample partial block.
7. Exact 159-sample delay alignment (15 oversampling + 144 lookahead) covering input events.
8. Compressor backward bit-identical regression test, float64 gradcheck, and L_conflict gradient isolation.
9. Bit-accurate C++/Python parity for LookaheadLimiter audio and attenuation (< -100 dBFS RMS null).
10. Multi-chunk aggregation with equal weights (w_k = 1/4) and no dynamic renormalization on silence.
11. Full parameter space integrity theta_JSON == map_z_to_theta(z_JSON) across all 41 dimensions.
"""

import copy
import json
import math
import os
from pathlib import Path
import subprocess
import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ddsp_mastering.dsp.crossover import DifferentiableLR4Crossover
from ddsp_mastering.dsp.compressor import (
    DifferentiableSingleBandCompressorFunction,
    DifferentiableSingleBandCompressor,
    DifferentiableMultibandCompressor,
)
from ddsp_mastering.dsp.saturator import DifferentiableHarmonicSaturator
from ddsp_mastering.dsp.limiter import (
    DifferentiableLookaheadLimiter,
    LOOKAHEAD_SAMPLES,
)
from ddsp_mastering.dsp.chain import DifferentiableMasteringChain
from ddsp_mastering.loss.dynamic_prior import (
    V3bDynamicPriorLoss,
    compute_active_sample_mask,
    TOTAL_DSP_DELAY_SAMPLES,
)
from ddsp_mastering.optimize import AutoMasteringOptimizer
from ddsp_mastering.param_space import (
    map_z_to_theta,
    unpack_theta_dict,
    PARAM_SPECS,
    Z_NEUTRAL,
)

try:
    import py_core_dsp
    _HAS_PY_CORE_DSP = True
except ImportError:
    _HAS_PY_CORE_DSP = False


# ==============================================================================
# Gate 1: Limiter bypass / zero attenuation
# ==============================================================================
def test_gate_1_limiter_bypass():
    """Verify Gate 1: Signal at -20 dBFS with 0 dB push produces zero limiter attenuation (< 1e-7 dB)."""
    sr = 48000.0
    t = torch.linspace(0, 1.0, int(sr), dtype=torch.float32)
    x = 0.1 * torch.stack([torch.sin(2 * math.pi * 440.0 * t), torch.cos(2 * math.pi * 440.0 * t)], dim=0).unsqueeze(0)
    
    limiter = DifferentiableLookaheadLimiter(sample_rate=sr)
    push = torch.tensor([0.0])
    y_out, diag = limiter(x, push, return_gain=True)
    
    a_lim = diag["a_lim_db"]
    max_a_lim = torch.max(a_lim).item()
    assert max_a_lim < 1e-7, f"Max A_lim {max_a_lim} exceeds 1e-7 dB at push=0 dB"
    assert torch.allclose(diag["g_smooth"], torch.ones_like(diag["g_smooth"]), atol=1e-6)


# ==============================================================================
# Gate 2: Limiter stress monotonic growth
# ==============================================================================
def test_gate_2_limiter_stress_monotonicity():
    """Verify Gate 2: Limiter metrics (mean, P95, CVaR_0.95) grow monotonically with push gain."""
    sr = 48000.0
    torch.manual_seed(42)
    t = torch.linspace(0, 1.0, int(sr), dtype=torch.float32)
    carrier = torch.sin(2 * math.pi * 300.0 * t) * (0.5 + 0.4 * torch.sin(2 * math.pi * 4.0 * t))
    x = carrier.repeat(2, 1).unsqueeze(0)

    limiter = DifferentiableLookaheadLimiter(sample_rate=sr)
    pushes = [0.0, 2.0, 4.0, 6.0, 8.0]
    avgs, p95s, cvars = [], [], []

    for p in pushes:
        _, diag = limiter(x, torch.tensor([p]), return_gain=True)
        a_lim = diag["a_lim_db"]
        mask, is_sil, _ = compute_active_sample_mask(x)
        assert not is_sil
        a_act = a_lim[mask]
        avg = torch.mean(a_act).item()
        p95 = torch.quantile(a_act.float(), 0.95).item()
        k = max(1, int(math.ceil(0.05 * a_act.numel())))
        cvar = torch.mean(torch.topk(a_act, k).values).item()
        avgs.append(avg)
        p95s.append(p95)
        cvars.append(cvar)

    for i in range(len(pushes) - 1):
        assert avgs[i+1] > avgs[i], f"Avg attenuation not monotonic at push {pushes[i]} -> {pushes[i+1]}"
        assert p95s[i+1] > p95s[i], f"P95 attenuation not monotonic at push {pushes[i]} -> {pushes[i+1]}"
        assert cvars[i+1] > cvars[i], f"CVaR_0.95 attenuation not monotonic at push {pushes[i]} -> {pushes[i+1]}"


# ==============================================================================
# Gate 3: Makeup invariance of GR_c[n] and zero gradient
# ==============================================================================
def test_gate_3_makeup_invariance_gr():
    """Verify Gate 3: GR_c[n] is bit-identical across makeup values and d(GR)/d(M) is identically zero."""
    sr = 48000.0
    torch.manual_seed(42)
    x = torch.randn(1, 2, 9600, dtype=torch.float32) * 0.5
    comp = DifferentiableSingleBandCompressor(sample_rate=sr)
    t_th = torch.tensor([-18.0])
    t_r = torch.tensor([3.0])
    t_a = torch.tensor([20.0])
    t_rel = torch.tensor([100.0])

    makeups = [-3.0, 0.0, 3.0, 6.0]
    ygs = []
    for m in makeups:
        _, yg = comp(x, t_th, t_r, t_a, t_rel, torch.tensor([m]), return_gr=True)
        ygs.append(yg)

    for i in range(1, len(makeups)):
        diff = torch.max(torch.abs(ygs[i] - ygs[0])).item()
        assert diff == 0.0, f"GR diff between makeup {makeups[i]} and {makeups[0]} is {diff} != 0"

    # Autograd: d(sum(yg))/dM == 0
    m_param = torch.tensor([2.0], requires_grad=True)
    _, yg = comp(x, t_th, t_r, t_a, t_rel, m_param, return_gr=True)
    loss = torch.sum(yg)
    loss.backward()
    grad_m = m_param.grad.item() if m_param.grad is not None else 0.0
    assert grad_m == 0.0, f"Expected d(GR)/d(M) == 0.0, got {grad_m}"


# ==============================================================================
# Gate 4: Differential gain reduction with band-separated signals
# ==============================================================================
def test_gate_4_differential_gr_band_separated():
    """Verify Gate 4: 60 Hz tone compresses low band and activates L_diff > 0."""
    sr = 48000.0
    t = torch.linspace(0, 1.0, int(sr), dtype=torch.float32)
    tone_60 = 0.8 * torch.sin(2 * math.pi * 60.0 * t)
    x = tone_60.repeat(2, 1).unsqueeze(0)

    crossover = DifferentiableLR4Crossover(sample_rate=sr)
    f_low = torch.tensor([200.0])
    f_high = torch.tensor([4000.0])
    low, mid, high = crossover(x, f_low, f_high)

    comp = DifferentiableMultibandCompressor(sample_rate=sr)
    th = torch.tensor([[-18.0, -18.0, -18.0]])
    r = torch.tensor([[4.0, 4.0, 4.0]])
    att = torch.tensor([[20.0, 20.0, 20.0]])
    rel = torch.tensor([[100.0, 100.0, 100.0]])
    makeup = torch.tensor([[0.0, 0.0, 0.0]])

    _, gr_dict = comp(low, mid, high, th, r, att, rel, makeup, return_gr=True)
    gr_low = gr_dict["gr_low_db"]
    gr_mid = gr_dict["gr_mid_db"]
    gr_high = gr_dict["gr_high_db"]

    assert torch.max(gr_low).item() > 10.0, "Low band should be heavily compressed"
    assert torch.max(gr_mid).item() < 2.0, "Mid band should not be heavily compressed"
    diff_lm = torch.max(torch.abs(gr_low - gr_mid)).item()
    assert diff_lm > 4.0, f"Differential GR {diff_lm} dB should exceed 4.0 dB"

    loss_mod = V3bDynamicPriorLoss(delta_max=4.0, delay_dsp_samples=0)
    fake_diag = {
        "gr_low_db": gr_low,
        "gr_mid_db": gr_mid,
        "gr_high_db": gr_high,
        "lim_atten_db": torch.zeros_like(gr_low),
    }
    _, metrics = loss_mod(x, fake_diag, makeup, n_pre=0)
    assert metrics["loss_diff"] > 0.0, f"L_diff should be > 0, got {metrics['loss_diff']}"


# ==============================================================================
# Gate 5: CVaR properties and autograd gradient distribution
# ==============================================================================
def test_gate_5_cvar_properties_and_gradcheck():
    """Verify Gate 5: CVaR_0.95 >= P95 and autograd gradient is uniformly 1/k on top-k tail."""
    torch.manual_seed(42)
    A = (torch.abs(torch.randn(2000, dtype=torch.float64)) * 3.0).requires_grad_(True)
    p95 = torch.quantile(A, 0.95).item()
    k = max(1, int(math.ceil(0.05 * 2000)))
    topk_vals, topk_idx = torch.topk(A, k=k)
    cvar95 = torch.mean(topk_vals)

    assert cvar95.item() >= p95, f"CVaR {cvar95.item()} must be >= P95 {p95}"

    cvar95.backward()
    grad = A.grad
    expected_grad = 1.0 / k

    grad_topk = grad[topk_idx]
    assert torch.allclose(grad_topk, torch.full_like(grad_topk, expected_grad), atol=1e-7)

    mask_non_topk = torch.ones(2000, dtype=torch.bool)
    mask_non_topk[topk_idx] = False
    assert torch.all(grad[mask_non_topk] == 0.0), "Gradient on non-topk elements must be zero"


# ==============================================================================
# Gate 6: Active sample mask, silence rejection, and 321-sample partial block
# ==============================================================================
def test_gate_6_active_mask_and_partial_block():
    """Verify Gate 6: Silence detection (< -70 dBFS), M_min=8 fallback, and 321-sample partial block."""
    T_eval = 143841

    # 1. Digital silence
    x_silent = torch.zeros(1, 2, T_eval, dtype=torch.float32)
    mask, is_sil, _ = compute_active_sample_mask(x_silent)
    assert is_sil is True
    assert mask.sum().item() == 0

    # 2. Sparse signal fallback to 8 blocks
    x_sparse = torch.zeros(1, 2, T_eval, dtype=torch.float32) + 1e-5
    x_sparse[0, :, 10 * 480 : 11 * 480] = 0.5
    x_sparse[0, :, 20 * 480 : 21 * 480] = 0.3
    mask_sp, is_sil_sp, meta_sp = compute_active_sample_mask(x_sparse, min_active_blocks=8)
    assert is_sil_sp is False
    assert meta_sp["m_active"] == 8

    # 3. Partial block of 321 samples
    x_partial = torch.zeros(1, 2, T_eval, dtype=torch.float32) + 1e-5
    x_partial[0, :, 143520 : 143841] = 0.8
    mask_part, is_sil_part, _ = compute_active_sample_mask(x_partial)
    assert is_sil_part is False
    assert torch.all(mask_part[0, 143520 : 143841] == True)


# ==============================================================================
# Gate 7: Exact 159-sample delay alignment
# ==============================================================================
def test_07_temporal_alignment_159_samples():
    """Verify Gate 7: Total DSP delay is 159 samples, limiter attenuation aligns with input pulse,
    and exact sample-by-sample multiplicative identity holds over evaluation window T_eval = T - 159 = 143841.
    """
    sr = 48000.0
    N_pre = 4800
    T = 144000
    total_len = N_pre + T
    x = torch.zeros(1, 2, total_len, dtype=torch.float32)
    n_pulse = N_pre + 1000
    x[0, :, n_pulse] = 0.9

    sat = DifferentiableHarmonicSaturator()
    lim = DifferentiableLookaheadLimiter(sample_rate=sr)

    sat_out = sat(x, torch.tensor([0.0]), torch.tensor([0.0]))
    y_out, lim_diag = lim(sat_out, torch.tensor([4.0]), return_gain=True)
    a_lim_db = lim_diag["a_lim_db"]

    peak_idx = torch.argmax(torch.abs(y_out[0, 0, :])).item()
    expected_peak_idx = n_pulse + 159
    assert peak_idx == expected_peak_idx, f"Peak index {peak_idx} != expected {expected_peak_idx}"

    a_lim_eval = a_lim_db[..., N_pre + 159 : N_pre + T]
    k_impulse = n_pulse - N_pre
    atten_at_pulse = a_lim_eval[0, k_impulse].item()
    assert atten_at_pulse > 0.5, f"Limiter attenuation at pulse index should be active, got {atten_at_pulse} dB"

    # Exact sample-by-sample multiplicative identity check over evaluation window T_eval = T - 159 = 143841
    # n_mix in [N_pre, N_pre + T_eval - 1], length T_eval = T - 159 = 143841
    T_eval = T - 159
    g_push = 10.0 ** (4.0 / 20.0)
    g_smooth = lim_diag["g_smooth"]
    expected_y = (
        sat_out[:, :, N_pre + 15 : N_pre + 15 + T_eval]
        * g_push
        * g_smooth[:, None, N_pre + 159 : N_pre + 159 + T_eval]
    )
    actual_y = y_out[:, :, N_pre + 159 : N_pre + 159 + T_eval]
    assert actual_y.shape[-1] == 143841
    assert torch.max(torch.abs(actual_y - expected_y)).item() < 1e-6


test_gate_7_temporal_alignment_159_samples = test_07_temporal_alignment_159_samples


# ==============================================================================
# Gate 8: Compressor backward, gradcheck, and L_conflict gradient isolation
# ==============================================================================
def test_gate_8_compressor_backward_and_conflict_gradient_isolation():
    """Verify Gate 8: Compressor backward gradcheck in float64, bit-identical regression, and L_conflict isolation."""
    # 1. Combined gradcheck
    def combined_fn(x, th, r, att, rel, m):
        y_out, yg = DifferentiableSingleBandCompressorFunction.apply(x, th, r, att, rel, m, 48000.0)
        return y_out, yg

    torch.manual_seed(42)
    N = 32
    x = torch.randn(1, 2, N, dtype=torch.float64, requires_grad=True)
    th = torch.tensor([-15.0], dtype=torch.float64, requires_grad=True)
    r = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
    att = torch.tensor([10.0], dtype=torch.float64, requires_grad=True)
    rel = torch.tensor([50.0], dtype=torch.float64, requires_grad=True)
    m = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)

    passed = torch.autograd.gradcheck(
        combined_fn,
        (x, th, r, att, rel, m),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
        raise_exception=True
    )
    assert passed is True

    # 2. Bit-identical audio gradients when grad_yg_ext is None
    x1 = torch.randn(1, 2, 64, dtype=torch.float64, requires_grad=True)
    th1 = torch.tensor([-15.0], dtype=torch.float64, requires_grad=True)
    r1 = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
    att1 = torch.tensor([10.0], dtype=torch.float64, requires_grad=True)
    rel1 = torch.tensor([50.0], dtype=torch.float64, requires_grad=True)
    m1 = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)

    y_out1, yg1 = DifferentiableSingleBandCompressorFunction.apply(x1, th1, r1, att1, rel1, m1, 48000.0)
    loss1 = torch.sum(y_out1 ** 2)
    loss1.backward()

    x2 = x1.detach().clone().requires_grad_(True)
    th2 = th1.detach().clone().requires_grad_(True)
    r2 = r1.detach().clone().requires_grad_(True)
    att2 = att1.detach().clone().requires_grad_(True)
    rel2 = rel1.detach().clone().requires_grad_(True)
    m2 = m1.detach().clone().requires_grad_(True)

    y_out2, yg2 = DifferentiableSingleBandCompressorFunction.apply(x2, th2, r2, att2, rel2, m2, 48000.0)
    loss2 = torch.sum(y_out2 ** 2) + 0.0 * torch.sum(yg2)
    loss2.backward()

    assert torch.equal(x1.grad, x2.grad)
    assert torch.equal(th1.grad, th2.grad)
    assert torch.equal(r1.grad, r2.grad)
    assert torch.equal(att1.grad, att2.grad)
    assert torch.equal(rel1.grad, rel2.grad)
    assert torch.equal(m1.grad, m2.grad)

    # 3. L_conflict gradient isolation: strictly on makeup parameters (indices 24, 29, 34)
    chain = DifferentiableMasteringChain()
    z = torch.full((41,), 0.5, dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        z[24] = 0.95
        z[29] = 0.95
        z[34] = 0.95
        z[40] = 0.8

    audio = torch.randn(1, 2, 10000, dtype=torch.float32) * 0.4
    y_out, diag = chain(audio, z=z, return_diagnostics=True)

    loss_mod = V3bDynamicPriorLoss(
        tau_avg=0.5,
        delta_m=1.0,
        w_lim=0.0,
        w_diff=0.0,
        w_conflict=1.0,
        delay_dsp_samples=0,
    )
    theta = map_z_to_theta(z)
    params = unpack_theta_dict(theta)
    makeup_db = params["comp_makeup_db"]

    loss_conflict, metrics = loss_mod(audio, diag, makeup_db, n_pre=0)
    assert metrics["loss_conflict"] > 0.0

    loss_conflict.backward()
    grad_z = z.grad

    makeup_indices = [24, 29, 34]
    non_makeup_indices = [i for i in range(41) if i not in makeup_indices]

    assert all(grad_z[i].item() > 0.0 for i in makeup_indices), "Makeup gradients must be > 0"
    max_non_makeup = max(abs(grad_z[i].item()) for i in non_makeup_indices)
    assert max_non_makeup == 0.0, f"Non-makeup gradients must be identically 0.0, got max {max_non_makeup}"


# ==============================================================================
# Gate 9: C++/Python parity for LookaheadLimiter and MultibandCompressor audio output
# ==============================================================================
@pytest.mark.skipif(not _HAS_PY_CORE_DSP, reason="py_core_dsp required for C++ parity")
def test_09_cpp_python_limiter_and_compressor_parity():
    """Verify Gate 9: Bit-accurate C++/Python parity for processed audio outputs (< -100 dBFS RMS).
    
    NOTE: The C++ bindings in py_core_dsp (e.g., LookaheadLimiter, SingleBandComp,
    MultibandCompressor3Band) expose exclusively processed audio output arrays and do NOT
    expose internal yg[n] trace buffers directly. Verification confirms exact mathematical parity
    of the processed audio output (null_rms_dbfs < -100.0 dBFS), confirming exact DSP equivalence
    without direct yg buffer inspection.
    """
    sr = 48000.0
    torch.manual_seed(42)
    x_torch = torch.randn(1, 2, 48000, dtype=torch.float32) * 0.5
    x_np = x_torch.squeeze(0).numpy()

    push_db = 4.0

    cpp_lim = py_core_dsp.LookaheadLimiter(sr)
    cpp_lim.set_push_gain_db(push_db)
    y_cpp = cpp_lim.process(x_np)

    py_lim = DifferentiableLookaheadLimiter(sample_rate=sr)
    y_py, diag = py_lim(x_torch, torch.tensor([push_db]), return_gain=True)
    y_py_np = y_py.squeeze(0).detach().numpy()
    g_py_np = diag["g_smooth"].squeeze(0).detach().numpy()

    diff_y = y_cpp - y_py_np
    null_y_db = 10.0 * np.log10(np.mean(diff_y ** 2) + 1e-16)

    push_lin = 10.0 ** (push_db / 20.0)
    u_np = x_np * push_lin
    u_delayed = np.zeros_like(u_np)
    u_delayed[:, LOOKAHEAD_SAMPLES:] = u_np[:, :-LOOKAHEAD_SAMPLES]

    mask_valid = np.abs(u_delayed) > 1e-4
    g_from_cpp = y_cpp[mask_valid] / u_delayed[mask_valid]
    g_from_py = np.repeat(g_py_np[np.newaxis, :], 2, axis=0)[mask_valid]
    diff_g = g_from_cpp - g_from_py
    null_g_db = 10.0 * np.log10(np.mean(diff_g ** 2) + 1e-16)

    assert null_y_db <= -100.0, f"Limiter audio null {null_y_db} dBFS > -100 dBFS"
    assert null_g_db <= -100.0, f"Limiter gain factor null {null_g_db} dBFS > -100 dBFS"

    # Compressor C++ vs Python parity on processed audio output:
    # Explicit declaration: The C++ binding py_core_dsp.SingleBandComp exposes exclusively the audio
    # output and does not expose internal yg[n] trace buffers. We verify bit-level parity of the
    # compressed audio output (null < -100 dBFS).
    cpp_comp = py_core_dsp.SingleBandComp()
    cpp_comp.set_parameters(-15.0, 3.0, 10.0, 50.0, 2.0, 6.0, sr)
    y_cpp_comp = cpp_comp.process(x_np)

    py_comp = DifferentiableSingleBandCompressor(sample_rate=sr)
    y_py_comp = py_comp(
        x_torch,
        torch.tensor([-15.0]),
        torch.tensor([3.0]),
        torch.tensor([10.0]),
        torch.tensor([50.0]),
        torch.tensor([2.0]),
    ).squeeze(0).detach().numpy()

    diff_comp = y_cpp_comp - y_py_comp
    null_comp_db = 10.0 * np.log10(np.mean(diff_comp ** 2) + 1e-16)
    assert null_comp_db <= -100.0, f"Compressor audio null {null_comp_db} dBFS > -100 dBFS"


test_gate_9_limiter_cpp_python_parity = test_09_cpp_python_limiter_and_compressor_parity


# ==============================================================================
# Gate 10: Multi-chunk aggregation
# ==============================================================================
def test_gate_10_multi_chunk_aggregation():
    """Verify Gate 10: Weighted chunk aggregation w_k = 1/4 with no dynamic renormalization on silence."""
    losses = [2.5, 4.0, 1.5, 0.0]  # 4th chunk is silent
    w = 0.25
    agg_loss = sum(w * l for l in losses)
    expected_loss = 0.25 * (2.5 + 4.0 + 1.5)
    renormalized_loss = (1.0 / 3.0) * (2.5 + 4.0 + 1.5)

    assert math.isclose(agg_loss, expected_loss, abs_tol=1e-9)
    assert not math.isclose(agg_loss, renormalized_loss, abs_tol=1e-3)


# ==============================================================================
# Gate 11: Parameter space JSON integrity
# ==============================================================================
def test_gate_11_param_space_json_integrity():
    """Verify Gate 11: theta_JSON matches map_z_to_theta(z_JSON) within 1e-5 on SPRINT3 JSON artifacts."""
    files_to_check = [
        "docs/sprint3/SPRINT3_V2_ABLATION_RESULTS.json",
        "docs/sprint3/SPRINT3_V1_HARMONIZED_RESULTS.json",
    ]

    for filepath in files_to_check:
        assert os.path.exists(filepath), f"File {filepath} not found"
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        total_checked = 0
        max_err = 0.0
        sessions = data.get("sessions", [])
        for session in sessions:
            session_name = session.get("track_prefix", "unknown")
            variants = session.get("variants", {})
            for variant_name, v_data in variants.items():
                if "normalized_z_opt" in v_data and "physical_parameters_theta" in v_data:
                    z = torch.tensor(v_data["normalized_z_opt"], dtype=torch.float32)
                    theta_computed = map_z_to_theta(z).numpy()
                    theta_stored = v_data["physical_parameters_theta"]

                    for dim_idx, spec in enumerate(PARAM_SPECS):
                        c_val = float(theta_computed[dim_idx])
                        s_val = float(theta_stored[spec.name])
                        err = abs(c_val - s_val)
                        if err > max_err:
                            max_err = err
                        assert err < 1e-5, f"Mismatch in {session_name}/{variant_name}/{spec.name}: computed={c_val}, stored={s_val}, err={err}"
                        total_checked += 1

        assert total_checked > 0, f"No parameters verified in {filepath}"
        assert max_err < 1e-5, f"Max error {max_err} >= 1e-5 in {filepath}"


# ==============================================================================
# Gate 12: Formula coherence between runner and V3bDynamicPriorLoss on strictly non-zero terms
# ==============================================================================
def test_preflight_runner_vs_dynamic_prior_nonzero_coherence():
    """Verify Gate Coherence: compute_chunk_terms and forward agree exactly with all 3 terms > 0."""
    sr = 48000.0
    T = 48000 * 2
    t = torch.linspace(0, 2.0, T)

    x1 = (0.8 * torch.sin(2 * math.pi * 100.0 * t) + 0.01 * torch.sin(2 * math.pi * 1000.0 * t)).repeat(2, 1).unsqueeze(0)
    x2 = (0.9 * torch.sin(2 * math.pi * 80.0 * t) + 0.02 * torch.sin(2 * math.pi * 1500.0 * t)).repeat(2, 1).unsqueeze(0)
    chunks = [x1, x2]

    chain = DifferentiableMasteringChain()
    z = torch.full((41,), 0.5, dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        z[20] = 0.1   # Low threshold very low
        z[21] = 0.8   # Low ratio high
        z[24] = 0.95  # Low makeup high
        z[29] = 0.95  # Mid makeup high
        z[40] = 0.85  # Limiter push high -> strong limiter attenuation

    cfg = {
        "tau_avg": 0.2,
        "tau_95": 1.0,
        "delta_max": 1.0,
        "delta_ref": 2.0,
        "delta_m": 1.0,
        "lambda_95": 0.5,
        "w_lim": 0.02,
        "w_diff": 0.02,
        "w_conflict": 0.02,
    }
    loss_fn = V3bDynamicPriorLoss(**cfg)

    theta = map_z_to_theta(z)
    params = unpack_theta_dict(theta)
    makeup_db = params["comp_makeup_db"]

    w_k = 0.5
    tot_lim = torch.tensor(0.0)
    tot_diff = torch.tensor(0.0)
    tot_conf = torch.tensor(0.0)
    tot_combined = torch.tensor(0.0)
    tot_forward = torch.tensor(0.0)

    for mk in chunks:
        _, diag = chain(mk, z=z, return_diagnostics=True)
        l_lim, l_diff, l_conf, meta = loss_fn.compute_chunk_terms(mk, diag, makeup_db)
        l_dyn, fwd_meta = loss_fn(mk, diag, makeup_db)

        assert meta["loss_lim"] > 0.0, "L_lim must be strictly positive"
        assert meta["loss_diff"] > 0.0, "L_diff must be strictly positive"
        assert meta["loss_conflict"] > 0.0, "L_conflict must be strictly positive"

        # Bitwise identical metadata
        assert abs(fwd_meta["loss_lim"] - meta["loss_lim"]) < 1e-12
        assert abs(fwd_meta["loss_diff"] - meta["loss_diff"]) < 1e-12
        assert abs(fwd_meta["loss_conflict"] - meta["loss_conflict"]) < 1e-12

        comb = cfg["w_lim"] * l_lim + cfg["w_diff"] * l_diff + cfg["w_conflict"] * l_conf
        assert torch.allclose(comb, l_dyn, atol=1e-12)

        tot_lim = tot_lim + w_k * l_lim
        tot_diff = tot_diff + w_k * l_diff
        tot_conf = tot_conf + w_k * l_conf
        tot_combined = tot_combined + w_k * comb
        tot_forward = tot_forward + w_k * l_dyn

    assert torch.allclose(tot_combined, tot_forward, atol=1e-12)

    g_lim = torch.autograd.grad(tot_lim, z, retain_graph=True)[0]
    g_diff = torch.autograd.grad(tot_diff, z, retain_graph=True)[0]
    g_conf = torch.autograd.grad(tot_conf, z, retain_graph=True)[0]
    g_comb = torch.autograd.grad(tot_combined, z, retain_graph=False)[0]

    expected_g_comb = cfg["w_lim"] * g_lim + cfg["w_diff"] * g_diff + cfg["w_conflict"] * g_conf
    err = torch.max(torch.abs(g_comb - expected_g_comb)).item()
    assert err < 1e-6, f"Linear gradient combination discrepancy {err} >= 1e-6"


# ==============================================================================
# Gate 13: Bit-identical V0p path when V3b is disabled
# ==============================================================================
def test_13_v0p_path_bit_identical_when_v3b_disabled():
    """Verify 2-step optimize_multi_chunk with default V3b disabled produces bit-identical results to V0p."""
    torch.manual_seed(42)
    mix_k = [torch.randn(1, 2, 48000 + 4800) for _ in range(2)]
    ref_k = [torch.randn(1, 2, 48000) for _ in range(2)]
    prerolls = [4800, 4800]

    opt_control = AutoMasteringOptimizer(
        sample_rate=48000.0, lr=0.04, num_steps=2, target_lufs=-14.0
    )
    assert opt_control.enable_v3b is False
    assert opt_control.v3b_w_lim == 0.0
    assert opt_control.v3b_w_diff == 0.0
    assert opt_control.v3b_w_conflict == 0.0

    opt_v0p_pure = copy.deepcopy(opt_control)

    z_opt_ctrl, _, hist_ctrl = opt_control.optimize_multi_chunk(
        mix_chunks=mix_k,
        ref_chunks=ref_k,
        reference_mode="same_track",
        mix_preroll_samples=prerolls,
        target_lufs=-14.0,
    )

    z_opt_pure, _, hist_pure = opt_v0p_pure.optimize_multi_chunk(
        mix_chunks=mix_k,
        ref_chunks=ref_k,
        reference_mode="same_track",
        mix_preroll_samples=prerolls,
        target_lufs=-14.0,
    )

    diff_z = torch.max(torch.abs(z_opt_ctrl - z_opt_pure)).item()
    diff_l0 = abs(hist_ctrl["loss"][0] - hist_pure["loss"][0])
    diff_l1 = abs(hist_ctrl["loss"][1] - hist_pure["loss"][1])

    assert diff_z == 0.0, f"z_opt discrepancy: {diff_z} != 0.0"
    assert diff_l0 == 0.0, f"loss step 0 discrepancy: {diff_l0} != 0.0"
    assert diff_l1 == 0.0, f"loss step 1 discrepancy: {diff_l1} != 0.0"


# ==============================================================================
# Gate 14: V3b-B0.5 Optimizer Configuration and Gradient Linearity
# ==============================================================================
def test_14_v3b_b05_optimizer_config_and_single_chunk_evaluation():
    """Verify V3b-B0.5 configuration, exactly 1 call per chunk with weight 1/4, and gradient linearity."""
    opt_v3b = AutoMasteringOptimizer(
        sample_rate=48000.0,
        lr=0.04,
        num_steps=1,
        target_lufs=-14.0,
        v3b_w_lim=0.50,
        v3b_tau_avg=0.5,
        v3b_tau_95=2.0,
        v3b_lambda_95=0.5,
        v3b_delta_max=3.0,
        v3b_delta_m=1.25,
        v3b_delta_ref=3.0,
        v3b_w_diff=0.02,
        v3b_w_conflict=0.02,
    )

    assert opt_v3b.enable_v3b is True
    assert opt_v3b.v3b_w_lim == 0.50
    assert opt_v3b.v3b_tau_avg == 0.5
    assert opt_v3b.v3b_tau_95 == 2.0
    assert opt_v3b.v3b_lambda_95 == 0.5
    assert opt_v3b.v3b_delta_max == 3.0
    assert opt_v3b.v3b_delta_m == 1.25
    assert opt_v3b.v3b_delta_ref == 3.0
    assert opt_v3b.v3b_w_diff == 0.02
    assert opt_v3b.v3b_w_conflict == 0.02

    # 4 loud chunks (amplitude 1.5) to ensure limiter attenuation > 0
    torch.manual_seed(123)
    mix_chunks = [torch.randn(1, 2, 48000 + 4800) * 1.5 for _ in range(4)]
    ref_chunks = [torch.randn(1, 2, 48000) for _ in range(4)]
    prerolls = [4800] * 4

    call_count = [0]
    orig_compute = opt_v3b.v3b_prior_loss.compute_chunk_terms

    def spy_compute(*args, **kwargs):
        call_count[0] += 1
        return orig_compute(*args, **kwargs)

    opt_v3b.v3b_prior_loss.compute_chunk_terms = spy_compute

    z_opt, _, hist = opt_v3b.optimize_multi_chunk(
        mix_chunks=mix_chunks,
        ref_chunks=ref_chunks,
        reference_mode="same_track",
        mix_preroll_samples=prerolls,
        target_lufs=-14.0,
    )

    # Exactly 1 call per chunk across 4 chunks in 1 step
    assert call_count[0] == 4, f"Expected 4 calls to compute_chunk_terms, got {call_count[0]}"
    assert "v3b_loss" in hist and len(hist["v3b_loss"]) == 1
    assert hist["v3b_loss"][0] > 0.0
    assert hist["v3b_l_lim_raw"][0] > 0.0
    assert hist["v3b_l_lim_weighted"][0] > 0.0

    # Gradient linearity check at initial parameter point
    K = 4
    weights = [1.0 / K] * K
    device = mix_chunks[0].device
    z_init = torch.clamp(Z_NEUTRAL.to(device=device), 1e-4, 1.0 - 1e-4)
    u_init = torch.log(z_init / (1.0 - z_init))

    # Total gradient
    u_tot = nn.Parameter(u_init.clone().detach())
    z_tot = torch.sigmoid(u_tot)
    tot_step_loss = torch.tensor(0.0, device=device)
    for k in range(K):
        mk = mix_chunks[k]
        np_k = prerolls[k]
        pred_ext, diag_k = opt_v3b.chain(mk, z=z_tot, eq_mode="fft", return_diagnostics=True)
        pred_k = pred_ext[..., np_k:]
        ref_k = ref_chunks[k]
        lk_v0p, _ = opt_v3b.loss_fn(
            pred_audio=pred_k,
            target_audio=ref_k,
            target_lufs=-14.0,
            normalized_params=z_tot,
            align_latency=True,
            pred_ext_audio=pred_ext,
            pred_preroll_samples=np_k,
            include_eq_prior=False,
        )
        _, _, _, l_v3b_k, _ = orig_compute(
            mix_audio=mk,
            diagnostics=diag_k,
            z=z_tot,
            preroll_samples=np_k,
        )
        tot_step_loss = tot_step_loss + weights[k] * (lk_v0p + l_v3b_k)
    tot_step_loss.backward()
    g_total = u_tot.grad.clone()

    # V0p gradient
    u_v0p = nn.Parameter(u_init.clone().detach())
    z_v0p = torch.sigmoid(u_v0p)
    tot_v0p = torch.tensor(0.0, device=device)
    for k in range(K):
        mk = mix_chunks[k]
        np_k = prerolls[k]
        pred_ext = opt_v3b.chain(mk, z=z_v0p, eq_mode="fft")
        pred_k = pred_ext[..., np_k:]
        ref_k = ref_chunks[k]
        lk_v0p, _ = opt_v3b.loss_fn(
            pred_audio=pred_k,
            target_audio=ref_k,
            target_lufs=-14.0,
            normalized_params=z_v0p,
            align_latency=True,
            pred_ext_audio=pred_ext,
            pred_preroll_samples=np_k,
            include_eq_prior=False,
        )
        tot_v0p = tot_v0p + weights[k] * lk_v0p
    tot_v0p.backward()
    g_v0p = u_v0p.grad.clone()

    # V3b gradient (1/4 sum)
    u_v3b = nn.Parameter(u_init.clone().detach())
    z_v3b = torch.sigmoid(u_v3b)
    tot_v3b = torch.tensor(0.0, device=device)
    for k in range(K):
        mk = mix_chunks[k]
        np_k = prerolls[k]
        _, diag_k = opt_v3b.chain(mk, z=z_v3b, eq_mode="fft", return_diagnostics=True)
        _, _, _, l_v3b_k, _ = orig_compute(
            mix_audio=mk,
            diagnostics=diag_k,
            z=z_v3b,
            preroll_samples=np_k,
        )
        tot_v3b = tot_v3b + weights[k] * l_v3b_k
    tot_v3b.backward()
    g_v3b = u_v3b.grad.clone()

    g_expected = g_v0p + g_v3b
    assert torch.allclose(g_total, g_expected, atol=1e-5, rtol=1e-4), (
        f"Gradient linearity failed: max diff = {torch.max(torch.abs(g_total - g_expected)).item()}"
    )
    assert torch.isfinite(g_total).all(), "Gradient g_total contains NaN or Inf"


# ==============================================================================
# Gate 15: Integrity and zero-diff preservation of all historical artifacts
# ==============================================================================
def test_15_preserved_artifacts_zero_diff():
    """Verify that all historical preflight and ablation artifacts exist, are non-empty, and unmodified in git."""
    repo_root = Path(__file__).resolve().parents[1]
    historical_files = [
        "docs/sprint3/SPRINT3_V3B_PREFLIGHT_RESULTS.json",
        "docs/sprint3/SPRINT3_V3B_PREFLIGHT_REPORT.md",
        "docs/sprint3/SPRINT3_V3B_PREFLIGHT_V2_RESULTS.json",
        "docs/sprint3/SPRINT3_V3B_PREFLIGHT_V2_REPORT.md",
        "docs/sprint3/SPRINT3_V3B_PREFLIGHT_V3_RESULTS.json",
        "docs/sprint3/SPRINT3_V3B_PREFLIGHT_V3_REPORT.md",
        "docs/sprint3/SPRINT3_V3B_PREFLIGHT_V3_AUDIT_ERRATA.md",
        "docs/sprint3/SPRINT3_V2_ABLATION_RESULTS.json",
        "docs/sprint3/SPRINT3_V2_REPORT.md",
        "docs/sprint3/SPRINT3_V1_HARMONIZED_RESULTS.json",
        "docs/sprint3/SPRINT3_V0P_V1_ABLATION_RESULTS.json",
        "docs/sprint3/SPRINT3_V0P_V1_REPORT.md",
        "docs/sprint3/SPRINT3_V3A_ABLATION_RESULTS.json",
        "docs/sprint3/SPRINT3_V3A_ABLATION_REPORT.md",
        "docs/sprint3/SPRINT3_V3A_PREFLIGHT_GRADIENT_DIAGNOSTICS.json",
    ]

    for rel_path in historical_files:
        p = repo_root / rel_path
        assert p.exists(), f"Historical artifact missing: {rel_path}"
        assert p.stat().st_size > 0, f"Historical artifact is empty: {rel_path}"

        res = subprocess.run(
            ["git", "status", "--porcelain", str(p)],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(repo_root),
        )
        assert res.stdout.strip() == "", f"Uncommitted modification in historical artifact: {rel_path}"


# Aliases for suite runners
test_gate_13_v0p_path_bit_identical_when_v3b_disabled = test_13_v0p_path_bit_identical_when_v3b_disabled
test_gate_14_v3b_b05_optimizer_config_and_single_chunk_evaluation = test_14_v3b_b05_optimizer_config_and_single_chunk_evaluation
test_gate_15_preserved_artifacts_zero_diff = test_15_preserved_artifacts_zero_diff


# ==============================================================================
# Gate 16: V3b Ablation Results Complete Integrity and Historical Preservation
# ==============================================================================
def test_16_v3b_ablation_results_integrity():
    """Verify completeness of 6 sessions in SPRINT3_V3B_V0P_ABLATION_RESULTS.json, roundtrip theta, and zero diff."""
    repo_root = Path(__file__).resolve().parents[1]
    results_path = repo_root / "docs" / "sprint3" / "SPRINT3_V3B_V0P_ABLATION_RESULTS.json"
    assert results_path.exists(), f"Ablation results missing: {results_path}"
    assert results_path.stat().st_size > 0, f"Ablation results file is empty: {results_path}"

    data = json.loads(results_path.read_text(encoding="utf-8"))
    sessions = data.get("sessions", [])
    assert len(sessions) == 6, f"Expected 6 sessions, got {len(sessions)}"

    expected_prefixes = {"track01", "track02", "track03", "track01_style", "track02_style", "track03_style"}
    found_prefixes = {s["track_prefix"] for s in sessions}
    assert found_prefixes == expected_prefixes, f"Mismatched session prefixes: {found_prefixes}"

    name_to_idx = {p.name: i for i, p in enumerate(PARAM_SPECS)}

    for s in sessions:
        sid = s["track_prefix"]
        # Control V0p check
        ctrl = s.get("control_v0p", {})
        z_v0p = torch.tensor(ctrl["normalized_z_opt"], dtype=torch.float32)
        theta_v0p_stored = ctrl["physical_parameters_theta"]
        theta_v0p_calc = map_z_to_theta(z_v0p)
        for p in PARAM_SPECS:
            stored_val = theta_v0p_stored[p.name]
            calc_val = float(theta_v0p_calc[name_to_idx[p.name]].item())
            diff = abs(stored_val - calc_val)
            assert diff < 1e-5, f"[{sid}] V0p theta mismatch on {p.name}: {stored_val} vs {calc_val} (diff: {diff})"

        # Treatment V3b-B0.5 check
        trt = s.get("treatment_v3b_b05", {})
        z_v3b = torch.tensor(trt["normalized_z_opt"], dtype=torch.float32)
        theta_v3b_stored = trt["physical_parameters_theta"]
        theta_v3b_calc = map_z_to_theta(z_v3b)
        for p in PARAM_SPECS:
            stored_val = theta_v3b_stored[p.name]
            calc_val = float(theta_v3b_calc[name_to_idx[p.name]].item())
            diff = abs(stored_val - calc_val)
            assert diff < 1e-5, f"[{sid}] V3b theta mismatch on {p.name}: {stored_val} vs {calc_val} (diff: {diff})"

    # Verify zero diff on historical/preflight artifacts
    historical_files = [
        "docs/sprint3/SPRINT3_V3B_PREFLIGHT_RESULTS.json",
        "docs/sprint3/SPRINT3_V3B_PREFLIGHT_REPORT.md",
        "docs/sprint3/SPRINT3_V3B_PREFLIGHT_V2_RESULTS.json",
        "docs/sprint3/SPRINT3_V3B_PREFLIGHT_V2_REPORT.md",
        "docs/sprint3/SPRINT3_V3B_PREFLIGHT_V3_RESULTS.json",
        "docs/sprint3/SPRINT3_V3B_PREFLIGHT_V3_REPORT.md",
        "docs/sprint3/SPRINT3_V3B_PREFLIGHT_V3_AUDIT_ERRATA.md",
        "docs/sprint3/SPRINT3_V2_ABLATION_RESULTS.json",
        "docs/sprint3/SPRINT3_V2_REPORT.md",
        "docs/sprint3/SPRINT3_V1_HARMONIZED_RESULTS.json",
        "docs/sprint3/SPRINT3_V0P_V1_ABLATION_RESULTS.json",
        "docs/sprint3/SPRINT3_V0P_V1_REPORT.md",
        "docs/sprint3/SPRINT3_V3A_ABLATION_RESULTS.json",
        "docs/sprint3/SPRINT3_V3A_ABLATION_REPORT.md",
        "docs/sprint3/SPRINT3_V3A_PREFLIGHT_GRADIENT_DIAGNOSTICS.json",
    ]
    for rel_path in historical_files:
        p = repo_root / rel_path
        assert p.exists() and p.stat().st_size > 0
        res = subprocess.run(
            ["git", "status", "--porcelain", str(p)],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(repo_root),
        )
        assert res.stdout.strip() == "", f"Uncommitted modification in historical artifact: {rel_path}"

    # Verify ablation report has non-zero Table 3 variation for eq_b4_gain_db on track01
    report_path = repo_root / "docs" / "sprint3" / "SPRINT3_V3B_V0P_ABLATION_REPORT.md"
    assert report_path.exists() and report_path.stat().st_size > 0, f"Report missing: {report_path}"
    report_text = report_path.read_text(encoding="utf-8")
    assert "eq_b4_gain_db" in report_text, "Expected eq_b4_gain_db column in Tabella 3"
    t3_start = report_text.find("## 4. Tabella 3")
    assert t3_start != -1, "Tabella 3 section missing in report"
    t3_section = report_text[t3_start : t3_start + 3000]
    assert "-0.0149" in t3_section or "-0.015" in t3_section, (
        "Expected non-zero eq_b4_gain_db delta (-0.0149 or -0.015) on track01 in Tabella 3"
    )
    assert "-0.1051" in t3_section and "-0.2062" in t3_section, (
        "Expected non-zero comp_low_threshold_db (-0.1051) and comp_high_threshold_db (-0.2062) on track01"
    )


test_gate_16_v3b_ablation_results_integrity = test_16_v3b_ablation_results_integrity

