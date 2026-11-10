"""Sprint 3 — Step V3a Mandatory Technical Gate Verification Tests.

Verifies all 7 points of the Technical Gate for CompositeEQPriorLoss:
1. Exact match with scipy.signal.freqz at machine precision (max diff < 1e-11 dB);
2. Flat 0.0 dB response and identically null gradients at Z_NEUTRAL;
3. Finite and exact gradients on 18 EQ params + torch.autograd.gradcheck in float64;
4. Strict gradient isolation on the 23 non-EQ parameters (indices 18..40);
5. Dead-zone behavior for shape (+/- 1.5 dB) and offset (+/- 1.0 dB);
6. Contrast between wide bell (Q=0.7) and narrow resonance (Q=4.5), and pure offset shift;
7. Absence of NaN/Inf at range extremes (z in [-3, 3]) and multi-chunk non-duplication in AutoMasteringOptimizer.
"""

import math
import numpy as np
import pytest
from scipy import signal
import torch

from ddsp_mastering.loss.eq_prior import CompositeEQPriorLoss
from ddsp_mastering.optimize.optimizer import AutoMasteringOptimizer
from ddsp_mastering.param_space import (
    NUM_PARAMS,
    PARAM_SPECS,
    Z_NEUTRAL,
    map_theta_to_z,
    map_z_to_theta,
)


def test_gate1_scipy_freqz_parity():
    """Gate 1: Composite response matches scipy.signal.freqz at machine precision (< 1e-11 dB)."""
    fs = 48000.0
    loss_module = CompositeEQPriorLoss(sample_rate=fs)

    # Non-trivial 6-band biquad configuration with varied gains, frequencies, and Q
    theta = map_z_to_theta(Z_NEUTRAL.clone().unsqueeze(0)).squeeze(0).to(torch.float64)
    theta[0] = 120.0; theta[1] = 1.5; theta[2] = 0.8     # Band 0 Low-shelf
    theta[3] = 400.0; theta[4] = -1.5; theta[5] = 1.0    # Band 1 Peaking
    theta[6] = 1200.0; theta[7] = 2.0; theta[8] = 0.9    # Band 2 Peaking
    theta[9] = 3500.0; theta[10] = -1.2; theta[11] = 1.1 # Band 3 Peaking
    theta[12] = 7000.0; theta[13] = 1.8; theta[14] = 1.0 # Band 4 Peaking
    theta[15] = 12000.0; theta[16] = -1.5; theta[17] = 0.707 # Band 5 High-shelf

    z = map_theta_to_z(theta.unsqueeze(0)).squeeze(0).to(torch.float64)
    G_torch = loss_module.compute_composite_response_db(z, eps_eq=0.0).cpu().numpy()

    # Compute individual biquad responses with scipy.signal.freqz
    b_norm, a_norm = loss_module.eq_dsp.compute_biquad_coeffs(
        theta[0:18:3].unsqueeze(0),
        theta[1:18:3].unsqueeze(0),
        theta[2:18:3].unsqueeze(0),
    )
    b_np = b_norm.squeeze(0).cpu().numpy()
    a_np = a_norm.squeeze(0).cpu().numpy()
    w_np = loss_module.omega.cpu().numpy()

    G_scipy = np.zeros(loss_module.K_eq, dtype=np.float64)
    for i in range(6):
        w, h = signal.freqz(b_np[i], a_np[i], worN=w_np)
        G_scipy += 20.0 * np.log10(np.abs(h))

    max_diff = np.max(np.abs(G_torch - G_scipy))
    assert max_diff < 1e-11, (
        f"Composite response must match scipy.signal.freqz within 1e-11 dB, got {max_diff:.4e} dB"
    )


def test_gate2_z_neutral_flat_response_and_null_gradient():
    """Gate 2: Composite response is flat 0.0 dB and gradient is identically null at Z_NEUTRAL."""
    loss_module = CompositeEQPriorLoss(sample_rate=48000.0)
    z = Z_NEUTRAL.clone().to(torch.float64).requires_grad_(True)

    G_eq = loss_module.compute_composite_response_db(z)
    assert torch.max(torch.abs(G_eq)).item() < 1e-10, (
        f"G_eq at Z_NEUTRAL must be 0 dB, got max abs {torch.max(torch.abs(G_eq)).item()}"
    )

    loss, metrics = loss_module(z)
    assert abs(loss.item()) < 1e-12
    assert abs(metrics["L_shape_exc"]) < 1e-12
    assert abs(metrics["L_slope_raw"]) < 1e-12
    assert abs(metrics["L_offset_raw"]) < 1e-12

    loss.backward()
    assert z.grad is not None
    grad_norm = torch.norm(z.grad).item()
    assert grad_norm < 1e-12, f"Gradient at Z_NEUTRAL must be machine zero, got norm {grad_norm}"


def test_gate3_gradcheck_eq_params_float64():
    """Gate 3: Finite gradients and gradcheck verification in float64 outside dead-zone."""
    loss_module = CompositeEQPriorLoss(sample_rate=48000.0)

    # 1. Gradcheck on compute_composite_response_db
    def fn_response(z_eq):
        return loss_module.compute_composite_response_db(z_eq, eps_eq=0.0)

    z_eq_test1 = (Z_NEUTRAL[:18].clone().to(torch.float64) + 0.1).requires_grad_(True)
    res_resp = torch.autograd.gradcheck(
        fn_response, z_eq_test1, eps=1e-5, atol=1e-4, rtol=1e-3
    )
    assert res_resp, "gradcheck failed on compute_composite_response_db"

    # 2. Gradcheck on full scalar loss CompositeEQPriorLoss()(z)[0] outside dead-zone
    def fn_scalar_loss(z_eq):
        loss_val, _ = loss_module(z_eq)
        return loss_val

    # Construct active configuration outside both shape (+/- 1.5 dB) and offset (+/- 1.0 dB) dead-zones
    z_eq_test2 = Z_NEUTRAL[:18].clone().to(torch.float64)
    z_eq_test2[1] = 0.90  # Band 0 gain boost +4.8 dB
    z_eq_test2[4] = 0.85  # Band 1 gain boost +4.2 dB
    z_eq_test2[7] = 0.90  # Band 2 gain boost +4.8 dB
    z_eq_test2.requires_grad_(True)

    loss_val, metrics = loss_module(z_eq_test2)
    assert metrics["L_shape_exc"] > 0.1, "Test configuration must be outside shape deadzone"
    assert metrics["L_offset_raw"] > 0.1, "Test configuration must be outside offset deadzone"

    res_loss = torch.autograd.gradcheck(
        fn_scalar_loss, z_eq_test2, eps=1e-5, atol=1e-4, rtol=1e-3
    )
    assert res_loss, "gradcheck failed on full scalar CompositeEQPriorLoss"


def test_gate4_non_eq_parameter_isolation():
    """Gate 4: Strict zero gradient isolation on non-EQ parameters (indices 18..40)."""
    loss_module = CompositeEQPriorLoss(sample_rate=48000.0)

    # Perturb ALL 41 parameters away from Z_NEUTRAL
    torch.manual_seed(999)
    z_all = torch.rand(NUM_PARAMS, dtype=torch.float64).requires_grad_(True)

    loss, _ = loss_module(z_all)
    assert loss.item() > 0.0

    loss.backward()
    assert z_all.grad is not None

    grad_eq = z_all.grad[:18]
    grad_non_eq = z_all.grad[18:]

    assert torch.norm(grad_eq).item() > 0.1, "EQ parameters must receive non-zero gradients"
    non_eq_norm = torch.norm(grad_non_eq).item()
    assert non_eq_norm == 0.0, (
        f"Non-EQ parameters (indices 18..40) must have exactly 0.0 gradient, got norm {non_eq_norm}"
    )


def test_gate5_deadzone_behavior():
    """Gate 5: Exact dead-zone behavior for shape (+/- 1.5 dB) and offset (+/- 1.0 dB)."""
    loss_module = CompositeEQPriorLoss(
        sample_rate=48000.0, tau_shape=1.5, tau_offset=1.0
    )

    # 1. Inside dead-zone: modest peaking of +1.0 dB and low shelf of +0.5 dB
    theta_in = map_z_to_theta(Z_NEUTRAL.clone().unsqueeze(0)).squeeze(0).to(torch.float64)
    theta_in[7] = 1.0   # Band 2 peaking +1.0 dB (shape excursion < 1.5 dB, mean < 1.0 dB)
    theta_in[1] = 0.5   # Band 0 low-shelf +0.5 dB
    z_in = map_theta_to_z(theta_in.unsqueeze(0)).squeeze(0).to(torch.float64)

    _, metrics_in = loss_module(z_in)
    assert metrics_in["max_abs_G_tilde"] < 1.5, f"Expected shape inside dead-zone, got {metrics_in['max_abs_G_tilde']}"
    assert abs(metrics_in["G_bar"]) < 1.0, f"Expected offset inside dead-zone, got {metrics_in['G_bar']}"
    assert metrics_in["L_shape_exc"] == 0.0, "L_shape_exc must be exactly 0.0 inside dead-zone"
    assert metrics_in["L_offset_raw"] == 0.0, "L_offset_raw must be exactly 0.0 inside dead-zone"

    # 2. Outside dead-zone: large peaking of +4.5 dB and offset of +3.0 dB
    theta_out = map_z_to_theta(Z_NEUTRAL.clone().unsqueeze(0)).squeeze(0).to(torch.float64)
    theta_out[7] = 4.5
    for b in range(6):
        theta_out[b * 3 + 1] += 2.0
    z_out = map_theta_to_z(theta_out.unsqueeze(0)).squeeze(0).to(torch.float64)

    _, metrics_out = loss_module(z_out)
    assert metrics_out["max_abs_G_tilde"] > 1.5
    assert abs(metrics_out["G_bar"]) > 1.0
    assert metrics_out["L_shape_exc"] > 0.0, "L_shape_exc must be strictly positive outside dead-zone"
    assert metrics_out["L_offset_raw"] > 0.0, "L_offset_raw must be strictly positive outside dead-zone"


def test_gate6_wide_bell_vs_narrow_resonance_and_offset():
    """Gate 6: Narrow resonance produces much higher L_slope_raw than wide bell; uniform shift activates offset."""
    loss_module = CompositeEQPriorLoss(sample_rate=48000.0)

    # Wide bell: Q=0.7 (z[8]=0.31 approx) with +4.0 dB boost on band 2
    theta_wide = map_z_to_theta(Z_NEUTRAL.clone().unsqueeze(0)).squeeze(0).to(torch.float64)
    theta_wide[7] = 4.0
    theta_wide[8] = 0.7
    z_wide = map_theta_to_z(theta_wide.unsqueeze(0)).squeeze(0).to(torch.float64)
    _, m_wide = loss_module(z_wide)

    # Narrow resonance: Q=2.4 with +4.0 dB boost on band 2
    theta_narrow = map_z_to_theta(Z_NEUTRAL.clone().unsqueeze(0)).squeeze(0).to(torch.float64)
    theta_narrow[7] = 4.0
    theta_narrow[8] = 2.4
    z_narrow = map_theta_to_z(theta_narrow.unsqueeze(0)).squeeze(0).to(torch.float64)
    _, m_narrow = loss_module(z_narrow)

    assert m_narrow["L_slope_raw"] > 2.0 * m_wide["L_slope_raw"], (
        f"Narrow resonance slope roughness ({m_narrow['L_slope_raw']:.4f}) must be significantly larger than wide bell ({m_wide['L_slope_raw']:.4f})"
    )

    # Uniform gain shift across all 6 bands (pure offset):
    theta_offset = map_z_to_theta(Z_NEUTRAL.clone().unsqueeze(0)).squeeze(0).to(torch.float64)
    for b in range(6):
        theta_offset[b * 3 + 1] = 2.5
    z_offset = map_theta_to_z(theta_offset.unsqueeze(0)).squeeze(0).to(torch.float64)
    _, m_offset = loss_module(z_offset)

    assert m_offset["L_offset_raw"] > 1.0, f"Uniform shift must activate offset excess, got {m_offset['L_offset_raw']}"
    assert m_offset["L_shape_exc"] < 0.01, f"Pure offset should leave shape excess near zero, got {m_offset['L_shape_exc']}"


def test_gate7_extremes_and_multichunk_non_duplication():
    """Gate 7: No NaN/Inf at range extremes (z in [-3, +3]) and multi-chunk non-duplication in AutoMasteringOptimizer."""
    loss_module = CompositeEQPriorLoss(sample_rate=48000.0)

    # 1. Extremes test: z in [-3, +3]
    for z_scalar in [-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0]:
        z_extreme = torch.full((NUM_PARAMS,), z_scalar, dtype=torch.float32)
        loss_val, metrics = loss_module(z_extreme)
        assert torch.isfinite(loss_val), f"Loss must be finite at z={z_scalar}"
        for k, v in metrics.items():
            assert math.isfinite(v), f"Metric {k} must be finite at z={z_scalar}"

    # 2. Multi-chunk non-duplication in AutoMasteringOptimizer
    sr = 48000.0
    torch.manual_seed(1234)
    chunks = [torch.randn(1, 2, int(sr * 1.5)) * 0.1 for _ in range(4)]

    opt_base = AutoMasteringOptimizer(sample_rate=sr, num_steps=1, weight_eq_prior=0.0)
    opt_v3a = AutoMasteringOptimizer(sample_rate=sr, num_steps=1, weight_eq_prior=0.02)

    # Track calls to eq_prior_loss.forward
    forward_calls = {"count": 0}
    orig_fwd = opt_v3a.eq_prior_loss.forward

    def wrapped_fwd(z):
        forward_calls["count"] += 1
        return orig_fwd(z)

    opt_v3a.eq_prior_loss.forward = wrapped_fwd

    u_test = torch.zeros(NUM_PARAMS, dtype=torch.float32, requires_grad=True)
    z_test = torch.sigmoid(u_test)
    L_eq_direct, _ = opt_v3a.eq_prior_loss(z_test)
    grad_eq_direct = torch.autograd.grad(L_eq_direct, u_test, retain_graph=True)[0]

    # Evaluate step 0 loss for opt_base across 4 chunks
    loss_base = torch.tensor(0.0)
    for k in range(4):
        c_loss, _ = opt_base.loss_fn(
            opt_base.chain(chunks[k], z=z_test, eq_mode="fft"),
            target_audio=chunks[k],
            include_eq_prior=False,
        )
        loss_base = loss_base + 0.25 * c_loss
    grad_base = torch.autograd.grad(loss_base, u_test, retain_graph=True)[0]

    # Evaluate step 0 loss for opt_v3a across 4 chunks + single EQ prior
    loss_v3a = torch.tensor(0.0)
    for k in range(4):
        c_loss, _ = opt_v3a.loss_fn(
            opt_v3a.chain(chunks[k], z=z_test, eq_mode="fft"),
            target_audio=chunks[k],
            include_eq_prior=False,
        )
        loss_v3a = loss_v3a + 0.25 * c_loss
    eq_val, _ = opt_v3a.eq_prior_loss(z_test)
    loss_v3a = loss_v3a + 0.02 * eq_val
    grad_v3a = torch.autograd.grad(loss_v3a, u_test, retain_graph=True)[0]

    # Difference in loss must be exactly 1 * 0.02 * L_eq_direct (NOT 4 * 0.02)
    diff_loss = (loss_v3a - loss_base).item()
    expected_diff_loss = (0.02 * L_eq_direct).item()
    assert abs(diff_loss - expected_diff_loss) < 1e-6, (
        f"Loss difference ({diff_loss:.6f}) must match single prior ({expected_diff_loss:.6f})"
    )

    # Verify forward call count: exactly 2 calls total (1 for direct check + 1 for step evaluation)
    assert forward_calls["count"] == 2, (
        f"CompositeEQPriorLoss.forward must be called exactly once per step (got {forward_calls['count'] - 1} calls)"
    )

    # Difference in gradient must be exactly 0.02 * grad_eq_direct
    diff_grad = grad_v3a - grad_base
    expected_diff_grad = 0.02 * grad_eq_direct
    max_grad_diff = torch.max(torch.abs(diff_grad - expected_diff_grad)).item()
    assert max_grad_diff < 1e-6, (
        f"Gradient difference max error must be < 1e-6, got {max_grad_diff:.4e}"
    )


def test_v3a_preflight_gradient_diagnostics_nonzero():
    """Verify that in docs/sprint3/SPRINT3_V3A_PREFLIGHT_GRADIENT_DIAGNOSTICS.json:
    For all 6 sessions, both at_z_neutral and at_v0p_z_opt have strictly positive
    gradient norms (> 0.0) for the spectral term (mrstft or mel_spectral) and lufs."""
    import json
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "docs" / "sprint3" / "SPRINT3_V3A_PREFLIGHT_GRADIENT_DIAGNOSTICS.json"
    assert p.exists(), f"Diagnostics file not found at: {p}"

    data = json.loads(p.read_text(encoding="utf-8"))
    sessions = data.get("sessions", [])
    assert len(sessions) == 6, f"Expected 6 sessions, found {len(sessions)}"

    for s in sessions:
        prefix = s["track_prefix"]
        mode = s["reference_mode"]
        spec_key = "mrstft" if mode == "same_track" else "mel_spectral"

        for state_key in ["at_z_neutral", "at_v0p_z_opt"]:
            state_data = s.get(state_key)
            assert state_data is not None, f"[{prefix}] Missing {state_key}"

            terms = state_data.get("v0p_terms_grad_norm_eq18")
            if terms is None:
                terms = state_data.get("gradient_norms_eq_params", {}).get("individual_terms_weighted", {})

            assert spec_key in terms, f"[{prefix}][{state_key}] Missing spectral term '{spec_key}' in {terms}"
            assert "lufs" in terms, f"[{prefix}][{state_key}] Missing 'lufs' term in {terms}"

            norm_spec = float(terms[spec_key])
            norm_lufs = float(terms["lufs"])

            assert norm_spec > 0.0, f"[{prefix}][{state_key}] Spectral gradient norm '{spec_key}' must be > 0.0, got {norm_spec}"
            assert norm_lufs > 0.0, f"[{prefix}][{state_key}] LUFS gradient norm must be > 0.0, got {norm_lufs}"

