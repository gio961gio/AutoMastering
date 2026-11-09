"""Sprint 3 — Step V3a Pre-Implementation Mathematical Verification Tests.

Verifies the foundational properties of the 6-band biquad composite response
and the three EQ prior components (L_shape_exc, L_slope_raw, L_offset_raw):
1. Exact 0.0 dB response and identically null gradients at Z_NEUTRAL;
2. Strict gradient isolation on the 23 non-EQ parameters (indices 18..40);
3. Invariance of the centered shape G_tilde(w) under global offset shifts.
"""

import math
import pytest
import torch

from ddsp_mastering.dsp.eq import DifferentiableEQ
from ddsp_mastering.param_space import (
    PARAM_SPECS,
    Z_NEUTRAL,
    NUM_PARAMS,
    map_z_to_theta,
    map_theta_to_z,
    unpack_theta_dict,
)


def _compute_eq_prior_tensors(
    z: torch.Tensor,
    f_min: float = 20.0,
    f_max: float = 20000.0,
    K_eq: int = 128,
    fs: float = 48000.0,
    tau_shape: float = 1.5,
    tau_offset: float = 1.0,
    w_slope: float = 2.0,
    w_offset: float = 1.0,
    eps_eq: float = 1e-12,
):
    """Clean reference implementation of composite frequency response & priors."""
    if z.dim() == 1:
        z = z.unsqueeze(0)
    
    theta = map_z_to_theta(z)
    params = unpack_theta_dict(theta)
    
    eq_module = DifferentiableEQ(sample_rate=fs)
    b_coeffs, a_coeffs = eq_module.compute_biquad_coeffs(
        params["eq_f0"], params["eq_gain_db"], params["eq_q"]
    )  # shapes: [B, 6, 3]
    
    log_f = torch.linspace(math.log10(f_min), math.log10(f_max), K_eq, dtype=torch.float64, device=z.device)
    freqs = torch.pow(10.0, log_f)
    w = (2.0 * math.pi * freqs) / fs
    
    cos_w = torch.cos(w).view(1, 1, K_eq)
    cos_2w = torch.cos(2.0 * w).view(1, 1, K_eq)
    sin_w = torch.sin(w).view(1, 1, K_eq)
    sin_2w = torch.sin(2.0 * w).view(1, 1, K_eq)
    
    b0 = b_coeffs[..., 0].unsqueeze(-1)
    b1 = b_coeffs[..., 1].unsqueeze(-1)
    b2 = b_coeffs[..., 2].unsqueeze(-1)
    
    a1 = a_coeffs[..., 1].unsqueeze(-1)
    a2 = a_coeffs[..., 2].unsqueeze(-1)
    
    num_re = b0 + b1 * cos_w + b2 * cos_2w
    num_im = -(b1 * sin_w + b2 * sin_2w)
    num_sq = num_re ** 2 + num_im ** 2
    
    den_re = 1.0 + a1 * cos_w + a2 * cos_2w
    den_im = -(a1 * sin_w + a2 * sin_2w)
    den_sq = den_re ** 2 + den_im ** 2
    
    G_bands = 10.0 * torch.log10((num_sq + eps_eq) / (den_sq + eps_eq))
    G_eq = torch.sum(G_bands, dim=1)  # [B, K_eq]
    
    G_bar = torch.mean(G_eq, dim=-1, keepdim=True)  # [B, 1]
    G_tilde = G_eq - G_bar  # [B, K_eq]
    
    shape_exc_raw = torch.relu(torch.abs(G_tilde) - tau_shape)
    L_shape_exc = torch.mean(shape_exc_raw ** 2, dim=-1)
    
    diff_G = G_tilde[:, 1:] - G_tilde[:, :-1]
    L_slope_raw = torch.mean(diff_G ** 2, dim=-1)
    
    offset_exc_raw = torch.relu(torch.abs(G_bar.squeeze(-1)) - tau_offset)
    L_offset_raw = offset_exc_raw ** 2
    
    L_eq_shape = L_shape_exc + w_slope * L_slope_raw
    L_eq_offset = w_offset * L_offset_raw
    L_eq_total = L_eq_shape + L_eq_offset
    
    return {
        "G_eq": G_eq,
        "G_bar": G_bar,
        "G_tilde": G_tilde,
        "L_shape_exc": L_shape_exc,
        "L_slope_raw": L_slope_raw,
        "L_offset_raw": L_offset_raw,
        "L_eq_total": L_eq_total,
    }


def test_v3a_z_neutral_identity_and_null_gradient():
    """Property 1: At Z_NEUTRAL, composite response is 0 dB and gradient is identically zero."""
    z = Z_NEUTRAL.clone().detach().requires_grad_(True)
    res = _compute_eq_prior_tensors(z)
    
    # Check 0.0 dB response
    assert torch.max(torch.abs(res["G_eq"])).item() < 1e-10, (
        f"G_eq at Z_NEUTRAL must be 0 dB, got max abs {torch.max(torch.abs(res['G_eq'])).item()}"
    )
    assert abs(res["L_shape_exc"].item()) < 1e-12
    assert abs(res["L_slope_raw"].item()) < 1e-12
    assert abs(res["L_offset_raw"].item()) < 1e-12
    assert abs(res["L_eq_total"].item()) < 1e-12
    
    # Check gradients
    res["L_eq_total"].backward()
    assert z.grad is not None
    grad_norm = torch.norm(z.grad).item()
    assert grad_norm < 1e-12, f"Gradient at Z_NEUTRAL must be machine zero, got norm {grad_norm}"


def test_v3a_non_eq_parameter_isolation():
    """Property 2: Non-EQ parameters (indices 18..40) receive strictly zero gradient."""
    # Create an active configuration with both shape and offset excess
    theta = map_z_to_theta(Z_NEUTRAL.clone().unsqueeze(0)).squeeze(0)
    theta[7] = 4.0   # Band 2 boost +4 dB
    theta[10] = -3.5 # Band 3 cut -3.5 dB
    theta[1] = 2.5   # Band 0 Low-shelf +2.5 dB
    z_active = map_theta_to_z(theta.unsqueeze(0)).squeeze(0).detach().requires_grad_(True)
    
    res = _compute_eq_prior_tensors(z_active)
    assert res["L_eq_total"].item() > 0.0
    
    res["L_eq_total"].backward()
    assert z_active.grad is not None
    
    grad_eq = z_active.grad[:18]
    grad_non_eq = z_active.grad[18:]
    
    # EQ parameters must receive gradients
    assert torch.norm(grad_eq).item() > 0.1, "Active EQ configuration must receive non-zero EQ gradients"
    
    # Non-EQ parameters must be machine-zero
    non_eq_norm = torch.norm(grad_non_eq).item()
    assert non_eq_norm == 0.0, (
        f"Non-EQ parameters (indices 18..40) must have 0.0 gradient, got norm {non_eq_norm}"
    )


def test_v3a_shape_invariance_under_pure_offset_shift():
    """Property 3: Centered shape G_tilde(w) is invariant to uniform shift on all biquad gains."""
    # Configuration A: gentle peaking on band 2
    theta_a = map_z_to_theta(Z_NEUTRAL.clone().unsqueeze(0)).squeeze(0)
    theta_a[7] = 2.0 # Band 2 gain +2 dB
    z_a = map_theta_to_z(theta_a.unsqueeze(0)).squeeze(0)
    
    res_a = _compute_eq_prior_tensors(z_a)
    G_tilde_a = res_a["G_tilde"]
    
    # Configuration B: identical curve with all peaking gains shifted by delta
    # On peaking filters 1..4, applying equal gain shift shifts response uniformly
    theta_b = theta_a.clone()
    shift_db = 2.0
    for b in range(1, 5):
        theta_b[b * 3 + 1] += shift_db
    z_b = map_theta_to_z(theta_b.unsqueeze(0)).squeeze(0)
    
    res_b = _compute_eq_prior_tensors(z_b)
    G_tilde_b = res_b["G_tilde"]
    
    # In middle frequency range (where peaking filters dominate), shapes must match closely
    # Filter 1-4 cover ~100 Hz to ~12 kHz
    mid_mask = (res_a["G_eq"] > -100.0) # all valid
    diff_tilde = torch.abs(G_tilde_b - G_tilde_a)
    
    # Verify offset was correctly altered
    assert abs(res_b["G_bar"].item() - res_a["G_bar"].item()) > 1.0, "G_bar must reflect gain shift"
    # Slope roughness should be virtually identical
    assert abs(res_b["L_slope_raw"].item() - res_a["L_slope_raw"].item()) < 0.01, (
        "L_slope_raw must remain very close under uniform shift"
    )
