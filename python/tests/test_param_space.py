import pytest
import torch
from ddsp_mastering.param_space import (
    NUM_PARAMS,
    PARAM_SPECS,
    map_z_to_theta,
    map_theta_to_z,
    unpack_theta_dict,
)

def test_param_specs_count():
    assert len(PARAM_SPECS) == NUM_PARAMS == 41

def test_boundary_values():
    # z = 0 -> theta = min
    z_zero = torch.zeros((1, NUM_PARAMS), dtype=torch.float64)
    theta_min = map_z_to_theta(z_zero)
    expected_min = torch.tensor([p.min_val for p in PARAM_SPECS], dtype=torch.float64).unsqueeze(0)
    assert torch.allclose(theta_min, expected_min, rtol=1e-12, atol=1e-12)

    # z = 1 -> theta = max
    z_one = torch.ones((1, NUM_PARAMS), dtype=torch.float64)
    theta_max = map_z_to_theta(z_one)
    expected_max = torch.tensor([p.max_val for p in PARAM_SPECS], dtype=torch.float64).unsqueeze(0)
    assert torch.allclose(theta_max, expected_max, rtol=1e-12, atol=1e-12)

def test_bijectivity_roundtrip():
    # Test random uniform z in [0.01, 0.99]
    torch.manual_seed(12345)
    z_rand = torch.rand((100, NUM_PARAMS), dtype=torch.float64)
    theta = map_z_to_theta(z_rand)
    z_rec = map_theta_to_z(theta)

    diff = torch.abs(z_rand - z_rec)
    max_err = torch.max(diff).item()
    assert max_err < 1e-13, f"Bijectivity roundtrip error too large: {max_err}"

def test_differentiability_and_gradient_flow():
    z = torch.linspace(0.05, 0.95, 32, dtype=torch.float64).unsqueeze(1).repeat(1, NUM_PARAMS)
    z.requires_grad_(True)

    theta = map_z_to_theta(z)
    loss = torch.sum(theta)
    loss.backward()

    assert z.grad is not None
    assert not torch.isnan(z.grad).any()
    assert not torch.isinf(z.grad).any()
    # Tutte le derivate d(theta)/d(z) devono essere strettamente positive (funzioni monotone crescenti)
    assert (z.grad > 0.0).all()

def test_unpack_dict():
    z = torch.full((4, NUM_PARAMS), 0.5, dtype=torch.float32)
    theta = map_z_to_theta(z)
    d = unpack_theta_dict(theta)

    assert d["eq_f0"].shape == (4, 6)
    assert d["eq_gain_db"].shape == (4, 6)
    assert d["eq_q"].shape == (4, 6)
    assert d["crossover_f_low"].shape == (4,)
    assert d["crossover_f_high"].shape == (4,)
    assert d["comp_threshold_db"].shape == (4, 3)
    assert d["comp_ratio"].shape == (4, 3)
    assert d["comp_attack_ms"].shape == (4, 3)
    assert d["comp_release_ms"].shape == (4, 3)
    assert d["comp_makeup_db"].shape == (4, 3)
    assert d["imager_width"].shape == (4, 3)
    assert d["saturator_drive_db"].shape == (4,)
    assert d["saturator_mix"].shape == (4,)
    assert d["limiter_push_gain_db"].shape == (4,)
