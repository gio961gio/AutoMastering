"""Sprint 3 V3c Stereo & Mono Compatibility Technical Gate Test Suite.

Verifies all 8 required Technical Gates + 7 corner cases + Scale Invariance:
1. Orthonormal energy conservation (E_L + E_R == E_M + E_S) broadband and LR4-banded + w=1 identity.
2. Canonical correlation and mono fold-down drop (in-phase, quadrature 90 deg, anti-phase).
3. Active sample mask, digital silence rejection, inactive mix band gating (E_min=1e-7),
   321-sample partial block on T_eval=143,841, and LR4 pre-roll warm-up effect.
4. Float64 torch.autograd.gradcheck across all 4 candidate terms (signal + z[35:38]).
5. All 7 corner cases (mono, correlated stereo, decorrelated stereo, anti-phase, hard-panned,
   silent band, total silence) + quantitative scale invariance over [-60, +20] dBFS (< 1e-8).
6. Python/C++ parity for DifferentiableMidSideImager vs py_core_dsp.MidSideImager (< -120 dBFS RMS null).
7. Zero diff preservation of frozen optimizer/prior/param_space/DSP modules vs 9b1fd9115cce6deaffe637c649c4bb7d6159a195
   AND deterministic bit-identity verification (max_diff == 0.0) of V0p & V3b forward/backward paths.
8. Active mix with sub-threshold / completely silent prediction (< E_min=1e-7): mix-only validity,
   non-zero penalties, finite continuous gradients wrt pred & z[35:38], and continuity across E_min.
"""

import math
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch
from torch.autograd import gradcheck

from ddsp_mastering.dsp.crossover import DifferentiableLR4Crossover
from ddsp_mastering.dsp.imager import DifferentiableMidSideImager
from ddsp_mastering.loss.dynamic_prior import TOTAL_DSP_DELAY_SAMPLES, compute_active_sample_mask
from ddsp_mastering.loss.stereo_prior import (
    CANDIDATE_A_THRESHOLDS,
    FIXED_CROSSOVER_HIGH_HZ,
    FIXED_CROSSOVER_LOW_HZ,
    INV_SQRT2,
    MIN_ACTIVE_BAND_ENERGY,
    PILOT_SAME_TRACK_WEIGHTS,
    V3cStereoCompatibilityLoss,
    compute_masked_stereo_stats,
)
from ddsp_mastering.param_space import Z_NEUTRAL, map_z_to_theta

try:
    import py_core_dsp
    _HAS_PY_CORE_DSP = True
except ImportError:
    _HAS_PY_CORE_DSP = False


def _make_tone_stereo(
    freq_hz: float = 1000.0,
    num_samples: int = 48000,
    sr: float = 48000.0,
    amp_l: float = 0.2,
    amp_r: float = 0.2,
    phase_r: float = 0.0,
) -> torch.Tensor:
    t = torch.arange(num_samples, dtype=torch.float64) / sr
    left = amp_l * torch.sin(2.0 * math.pi * freq_hz * t)
    right = amp_r * torch.sin(2.0 * math.pi * freq_hz * t + phase_r)
    return torch.stack([left, right], dim=0).unsqueeze(0)  # (1, 2, N)


# ==============================================================================
# Gate 1: Orthonormal Energy Conservation and Mid/Side Imager Identity at w=1.0
# ==============================================================================
def test_gate_1_orthonormal_energy_conservation_and_identity():
    torch.manual_seed(42)
    x = torch.randn(1, 2, 24000, dtype=torch.float64) * 0.25
    imager = DifferentiableMidSideImager()
    xover = DifferentiableLR4Crossover(sample_rate=48000.0)

    f_low = torch.tensor(FIXED_CROSSOVER_LOW_HZ, dtype=torch.float64)
    f_high = torch.tensor(FIXED_CROSSOVER_HIGH_HZ, dtype=torch.float64)
    b_low, b_mid, b_high = xover(x, f_low, f_high)

    # Identity at w = 1.0 per band and summed
    w1 = torch.ones(1, dtype=torch.float64)
    for band in (x, b_low, b_mid, b_high):
        y = imager.process_band(band, w1)
        assert torch.max(torch.abs(y - band)).item() < 1e-13

        mask = torch.ones(1, band.shape[-1], dtype=torch.bool)
        st = compute_masked_stereo_stats(band, mask, min_energy=1e-12)
        e_lr = (st["e_l"] + st["e_r"]).item()
        e_ms = (st["e_m"] + st["e_s"]).item()
        assert abs(e_lr - e_ms) < 1e-12

    y_sum = imager(b_low, b_mid, b_high, w1, w1, w1)
    assert torch.max(torch.abs(y_sum - (b_low + b_mid + b_high))).item() < 1e-13

    # Mono sum invariance across arbitrary widths w != 1
    y_wide = imager(b_low, b_mid, b_high, torch.tensor([0.5]), torch.tensor([1.3]), torch.tensor([1.4]))
    mono_orig = (b_low[:, 0] + b_low[:, 1] + b_mid[:, 0] + b_mid[:, 1] + b_high[:, 0] + b_high[:, 1]) * INV_SQRT2
    mono_wide = (y_wide[:, 0] + y_wide[:, 1]) * INV_SQRT2
    assert torch.max(torch.abs(mono_orig - mono_wide)).item() < 1e-12


# ==============================================================================
# Gate 2: Canonical Signal Correlation and Mono Fold-Down Drop
# ==============================================================================
def test_gate_2_canonical_correlation_and_mono_drop():
    N = 48000
    mask = torch.ones(1, N, dtype=torch.bool)

    # 1. Pure in-phase (L == R): corr == +1.0, D_mono == 0.0 dB
    x_mono = _make_tone_stereo(freq_hz=1000.0, num_samples=N, amp_l=0.25, amp_r=0.25, phase_r=0.0)
    st_mono = compute_masked_stereo_stats(x_mono, mask)
    assert abs(float(st_mono["corr_lr"].item()) - 1.0) < 1e-10
    assert abs(float(st_mono["mono_drop_db"].item())) < 1e-10

    # 2. Quadrature 90 deg (sin vs cos, integer cycles): corr == 0.0, D_mono == 10*log10(2)
    x_quad = _make_tone_stereo(freq_hz=1000.0, num_samples=N, amp_l=0.25, amp_r=0.25, phase_r=0.5 * math.pi)
    st_quad = compute_masked_stereo_stats(x_quad, mask)
    assert abs(float(st_quad["corr_lr"].item())) < 1e-8
    assert abs(float(st_quad["mono_drop_db"].item()) - 10.0 * math.log10(2.0)) < 1e-6

    # 3. Pure anti-phase (L == -R): corr == -1.0, E_M == 0, finite D_mono
    x_anti = _make_tone_stereo(freq_hz=1000.0, num_samples=N, amp_l=0.25, amp_r=0.25, phase_r=math.pi)
    st_anti = compute_masked_stereo_stats(x_anti, mask)
    assert abs(float(st_anti["corr_lr"].item()) + 1.0) < 1e-10
    assert float(st_anti["e_m"].item()) < 1e-24
    assert math.isfinite(float(st_anti["mono_drop_db"].item()))
    assert float(st_anti["mono_drop_db"].item()) > 100.0  # 10*log10(1e12) = 120 dB


# ==============================================================================
# Gate 3: Active Mask, Silence Rejection, Inactive Band, Partial Block & LR4 Warmup
# ==============================================================================
def test_gate_3_active_mask_silence_band_validity_partial_block_and_warmup():
    criterion = V3cStereoCompatibilityLoss(delay_dsp_samples=0)

    # 3a. Digital silence (< -70 dBFS): signal losses == 0.0 and zero gradient
    mix_sil = torch.full((1, 2, 12000), 1e-5, dtype=torch.float64)
    pred_sil = (torch.full((1, 2, 12000), 1e-5, dtype=torch.float64)).requires_grad_(True)
    loss_s, diag_s = criterion(mix_sil, pred_sil, compute_diagnostics=True)
    loss_s.backward()
    assert diag_s["is_silent"] is True
    assert diag_s["loss_mono_drop"] == 0.0
    assert diag_s["loss_band_corr"] == 0.0
    assert diag_s["loss_band_smr"] == 0.0
    assert torch.max(torch.abs(pred_sil.grad)).item() == 0.0

    # 3b. Single-band signal (20 Hz sine => only low band active; mid & high < E_min=1e-7)
    n_pre = 12000
    t_len = 24000
    mix_20 = _make_tone_stereo(freq_hz=20.0, num_samples=n_pre + t_len, amp_l=0.2, amp_r=0.2, phase_r=0.0)
    pred_20 = _make_tone_stereo(freq_hz=20.0, num_samples=n_pre + t_len, amp_l=0.2, amp_r=0.2, phase_r=1.2).requires_grad_(True)
    loss_20, diag_20 = criterion(mix_20, pred_20, n_pre=n_pre, delay_samples=0, compute_diagnostics=True)
    loss_20.backward()
    assert diag_20["band_valid"]["low"] is True
    assert diag_20["band_valid"]["mid"] is False
    assert diag_20["band_valid"]["high"] is False
    assert not torch.isnan(pred_20.grad).any()

    # 3c. Partial remainder block (321 samples on T_eval = 143,841)
    mix_eval = torch.randn(1, 2, 143841, dtype=torch.float64) * 0.1
    mask_smp, is_sil, mmeta = compute_active_sample_mask(mix_eval)
    assert not is_sil
    assert mask_smp.shape[-1] == 143841
    assert 143841 % 480 == 321
    assert mmeta["m_total"] == (143841 // 480) + 1  # 299 full + 1 partial = 300 blocks

    # 3d. LR4 warm-up check: filtering full pre-roll differs from zero pre-roll near window start
    sr = 48000.0
    t_full = torch.arange(24000, dtype=torch.float64) / sr
    # Low-frequency 100 Hz continuous tone throughout pre-roll and eval window
    sig = 0.25 * torch.sin(2.0 * math.pi * 100.0 * t_full)
    x_ext = torch.stack([sig, sig], dim=0).unsqueeze(0)
    x_nopre = x_ext[..., 12000:].clone()

    _, _, mb_pre, _ = criterion._filter_and_slice_pair(x_ext, x_ext, n_pre=12000, delay_samples=0)
    _, _, mb_nopre, _ = criterion._filter_and_slice_pair(x_nopre, x_nopre, n_pre=0, delay_samples=0)
    # Startup transient difference in first 512 samples due to un-warmed LR4 IIR state
    startup_diff = torch.max(torch.abs(mb_pre["low"][..., :512] - mb_nopre["low"][..., :512])).item()
    assert startup_diff > 1e-2


# ==============================================================================
# Gate 4: Float64 torch.autograd.gradcheck on All 4 Candidate Loss Terms
# ==============================================================================
def test_gate_4_gradcheck_float64_all_four_terms():
    torch.manual_seed(7)
    # 4a. Dense elementwise gradcheck on pred tensor (N=48 broadband noise covering all 3 LR4 bands)
    N_small = 48
    base = torch.randn(1, 2, N_small, dtype=torch.float64) * 0.25
    mix_s = torch.stack([base[:, 0], 0.8 * base[:, 0] + 0.1 * base[:, 1]], dim=1)
    pred_s = torch.stack([base[:, 0] + 0.3 * base[:, 1], 0.8 * base[:, 0] - 0.35 * base[:, 1]], dim=1).requires_grad_(True)

    c_drop = V3cStereoCompatibilityLoss(
        delay_dsp_samples=0, tau_drop_db=0.05,
        w_mono_drop=1.0, w_band_corr=0.0, w_band_smr=0.0, w_width_param=0.0,
    )
    val_d, _ = c_drop(mix_s, pred_s, compute_diagnostics=False)
    assert val_d.item() > 1e-4
    assert gradcheck(lambda p: c_drop(mix_s, p, compute_diagnostics=False)[0], (pred_s,), eps=1e-6, atol=1e-4, rtol=1e-3)

    c_corr = V3cStereoCompatibilityLoss(
        delay_dsp_samples=0, tau_corr=(0.01, 0.01, 0.01),
        w_mono_drop=0.0, w_band_corr=1.0, w_band_smr=0.0, w_width_param=0.0,
    )
    val_c, d_c = c_corr(mix_s, pred_s, compute_diagnostics=False)
    assert all(d_c["band_valid"].values())
    assert val_c.item() > 1e-5
    assert gradcheck(lambda p: c_corr(mix_s, p, compute_diagnostics=False)[0], (pred_s,), eps=1e-6, atol=1e-4, rtol=1e-3)

    c_smr = V3cStereoCompatibilityLoss(
        delay_dsp_samples=0, tau_smr_db=(0.5, 0.5, 0.5), tau_sub_db=0.0, alpha_sub=1.0,
        w_mono_drop=0.0, w_band_corr=0.0, w_band_smr=1.0, w_width_param=0.0,
    )
    val_smr, _ = c_smr(mix_s, pred_s, compute_diagnostics=False)
    assert val_smr.item() > 1e-4
    assert gradcheck(lambda p: c_smr(mix_s, p, compute_diagnostics=False)[0], (pred_s,), eps=1e-6, atol=1e-4, rtol=1e-3)

    # 4b. Full-block (N=960) gradcheck end-to-end through LR4 + MidSideImager wrt z[35:38]
    N_full = 960
    t_ax = torch.arange(N_full, dtype=torch.float64) / 48000.0
    m_sig = 0.2 * (torch.sin(2 * math.pi * 80 * t_ax) + torch.sin(2 * math.pi * 800 * t_ax) + torch.sin(2 * math.pi * 6000 * t_ax))
    s_sig = 0.08 * (torch.cos(2 * math.pi * 80 * t_ax) + torch.cos(2 * math.pi * 800 * t_ax) + torch.cos(2 * math.pi * 6000 * t_ax))
    mix_3b = torch.stack([(m_sig + s_sig) * INV_SQRT2, (m_sig - s_sig) * INV_SQRT2], dim=0).unsqueeze(0)

    xover = DifferentiableLR4Crossover(sample_rate=48000.0)
    imager = DifferentiableMidSideImager()
    f_l = torch.tensor(FIXED_CROSSOVER_LOW_HZ, dtype=torch.float64)
    f_h = torch.tensor(FIXED_CROSSOVER_HIGH_HZ, dtype=torch.float64)
    mb_l, mb_m, mb_h = xover(mix_3b, f_l, f_h)

    z_full = Z_NEUTRAL.to(torch.float64).clone()
    z_full[35] = 0.95  # w_low > 1.0
    z_full[36] = 0.90  # wider mid
    z_full[37] = 0.90  # wider high
    z_full.requires_grad_(True)

    def _end_to_end_v3c(z_in):
        th = map_z_to_theta(z_in)
        pred_out = imager(mb_l, mb_m, mb_h, th[35:36], th[36:37], th[37:38])
        crit = V3cStereoCompatibilityLoss(
            delay_dsp_samples=0, tau_drop_db=0.05, tau_corr=(0.005, 0.005, 0.005),
            tau_smr_db=(0.3, 0.3, 0.3), tau_sub_db=0.0, alpha_sub=1.0,
            tau_w_low_max=1.0, tau_w_mid=0.05, tau_w_high=0.05,
        )
        l_tot, _ = crit(mix_3b, pred_out, z=z_in, compute_diagnostics=False)
        return l_tot

    assert gradcheck(_end_to_end_v3c, (z_full,), eps=1e-6, atol=1e-4, rtol=1e-3)

    # 4c. loss_width_param gradcheck + strict gradient isolation to indices 35:38
    c_wp = V3cStereoCompatibilityLoss(
        delay_dsp_samples=0, tau_w_low_max=1.0, tau_w_mid=0.05, tau_w_high=0.05,
        w_mono_drop=0.0, w_band_corr=0.0, w_band_smr=0.0, w_width_param=1.0,
    )
    val_wp, _ = c_wp.compute_width_param_loss(z=z_full)
    assert val_wp.item() > 1e-4
    assert gradcheck(lambda z_in: c_wp.compute_width_param_loss(z=z_in)[0], (z_full,), eps=1e-6, atol=1e-5, rtol=1e-4)

    val_wp.backward()
    non_imager = torch.cat([z_full.grad[:35], z_full.grad[38:]])
    assert torch.max(torch.abs(non_imager)).item() == 0.0
    assert torch.all(torch.abs(z_full.grad[35:38]) > 0.0)


# ==============================================================================
# Gate 5: 7 Corner Cases + Dead-Zone + Quantitative Scale Invariance
# ==============================================================================
def test_gate_5_seven_corner_cases_dead_zone_and_scale_invariance():
    criterion = V3cStereoCompatibilityLoss(delay_dsp_samples=0)
    sr = 48000.0
    N = 24000
    t = torch.arange(N, dtype=torch.float64) / sr
    # Multi-band carrier spanning low, mid, high
    sig_a = 0.2 * (torch.sin(2 * math.pi * 80 * t) + torch.sin(2 * math.pi * 800 * t) + torch.sin(2 * math.pi * 6000 * t))
    sig_b = 0.2 * (torch.cos(2 * math.pi * 80 * t) + torch.cos(2 * math.pi * 800 * t) + torch.cos(2 * math.pi * 6000 * t))

    cases = {
        "1_pure_mono": torch.stack([sig_a, sig_a], dim=0).unsqueeze(0),
        "2_correlated_stereo": torch.stack([sig_a, 0.7 * sig_a + 0.3 * sig_b], dim=0).unsqueeze(0),
        "3_decorrelated_stereo": torch.stack([sig_a, sig_b], dim=0).unsqueeze(0),
        "4_pure_antiphase": torch.stack([sig_a, -sig_a], dim=0).unsqueeze(0),
        "5_hard_panned_right_silent": torch.stack([sig_a, torch.zeros_like(sig_a)], dim=0).unsqueeze(0),
        "6_band_silent_60hz_only": _make_tone_stereo(freq_hz=60.0, num_samples=N, amp_l=0.2, amp_r=0.15, phase_r=0.4),
        "7_total_silence": torch.full((1, 2, N), 1e-6, dtype=torch.float64),
    }

    for name, audio in cases.items():
        loss, diag = criterion(audio, audio, z=Z_NEUTRAL, compute_diagnostics=True)
        assert math.isfinite(loss.item()), f"Non-finite loss on corner case {name}"
        # Dead-zone: identical pred == mix and Z_NEUTRAL yields strictly zero total loss
        assert loss.item() == 0.0, f"Expected exact zero loss in dead-zone for {name}, got {loss.item()}"

    # Specific check on corner case 5 (hard-panned: R == 0)
    _, d_hp = criterion(cases["5_hard_panned_right_silent"], cases["5_hard_panned_right_silent"], compute_diagnostics=True)
    assert d_hp["channel_silent_or_hard_panned_broadband"] is True
    assert d_hp["band_valid"]["mid"] is True and d_hp["channel_silent_or_hard_panned"]["mid"] is True
    assert abs(d_hp["broadband"]["corr_lr_mix"]) < 1e-12
    assert abs(d_hp["broadband"]["mono_drop_db_mix"] - 10.0 * math.log10(2.0)) < 1e-6

    # Quantitative Scale Invariance over [-60, -40, -20, 0, +10, +20] dBFS relative to baseline
    mix_base = cases["2_correlated_stereo"]
    pred_base = cases["3_decorrelated_stereo"]
    ref_loss, ref_diag = criterion(mix_base, pred_base, compute_diagnostics=True)
    ref_drop = ref_diag["broadband"]["mono_drop_db_pred"]
    ref_smr = ref_diag["broadband"]["smr_db_pred"]
    ref_corr = ref_diag["broadband"]["corr_lr_pred"]

    # RMS level of mix_base; scale jointly across [-60 dBFS .. +20 dBFS]
    rms_curr = float(torch.sqrt(0.5 * torch.mean(mix_base ** 2)).item())
    for target_db in (-60.0, -40.0, -20.0, 0.0, 10.0, 20.0):
        target_rms = 10.0 ** (target_db / 20.0)
        scale = target_rms / rms_curr
        l_sc, d_sc = criterion(mix_base * scale, pred_base * scale, compute_diagnostics=True)
        assert abs(d_sc["broadband"]["mono_drop_db_pred"] - ref_drop) < 1e-8
        assert abs(d_sc["broadband"]["smr_db_pred"] - ref_smr) < 1e-8
        assert abs(d_sc["broadband"]["corr_lr_pred"] - ref_corr) < 1e-8
        assert abs(l_sc.item() - ref_loss.item()) < 1e-8
        assert abs(d_sc["loss_mono_drop"] - ref_diag["loss_mono_drop"]) < 1e-8
        assert abs(d_sc["loss_band_corr"] - ref_diag["loss_band_corr"]) < 1e-8
        assert abs(d_sc["loss_band_smr"] - ref_diag["loss_band_smr"]) < 1e-8


# ==============================================================================
# Gate 6: Python vs C++ MidSideImager Parity (< -120 dBFS RMS Null)
# ==============================================================================
@pytest.mark.skipif(not _HAS_PY_CORE_DSP, reason="py_core_dsp C++ extension not available")
def test_gate_6_cpp_python_mid_side_imager_parity():
    torch.manual_seed(123)
    N = 16384
    low_t = (torch.randn(1, 2, N, dtype=torch.float32) * 0.25).contiguous()
    mid_t = (torch.randn(1, 2, N, dtype=torch.float32) * 0.25).contiguous()
    high_t = (torch.randn(1, 2, N, dtype=torch.float32) * 0.25).contiguous()

    imager_py = DifferentiableMidSideImager()
    combos = [
        (1.0, 1.0, 1.0),
        (0.5, 1.15, 1.35),
        (0.4, 0.85, 1.40),
        (1.05, 1.11, 1.11),
    ]
    for wl, wm, wh in combos:
        y_py = imager_py(
            low_t, mid_t, high_t,
            torch.tensor([wl], dtype=torch.float32),
            torch.tensor([wm], dtype=torch.float32),
            torch.tensor([wh], dtype=torch.float32),
        ).squeeze(0).numpy()

        y_cpp = py_core_dsp.MidSideImager.process_3band_and_sum(
            low_t.squeeze(0).numpy(),
            mid_t.squeeze(0).numpy(),
            high_t.squeeze(0).numpy(),
            wl, wm, wh,
        )
        diff = y_py - y_cpp
        max_abs = float(np.max(np.abs(diff)))
        rms_null_db = 20.0 * math.log10(max(float(np.sqrt(np.mean(diff ** 2))), 1e-12))
        assert max_abs < 1e-6, f"Max abs diff too high: {max_abs}"
        assert rms_null_db < -120.0, f"RMS null above -120 dBFS: {rms_null_db:.2f} dBFS"


# ==============================================================================
# Gate 7: Zero-Diff Preservation + Deterministic Bit-Identity of V0p & V3b Paths
# ==============================================================================
def test_gate_7_frozen_modules_zero_diff_and_v0p_v3b_isolation():
    repo_root = Path(__file__).resolve().parents[1]
    baseline_sha = "9b1fd9115cce6deaffe637c649c4bb7d6159a195"
    protected_paths = [
        "python/ddsp_mastering/loss/dynamic_prior.py",
        "python/ddsp_mastering/param_space.py",
        "python/ddsp_mastering/dsp",
        "cpp/core_dsp",
    ]
    for rel in protected_paths:
        st_res = subprocess.run(
            ["git", "status", "--porcelain", "--", rel],
            capture_output=True, text=True, check=True, cwd=str(repo_root),
        )
        assert st_res.stdout.strip() == "", f"Uncommitted change in protected path {rel}: {st_res.stdout}"
        df_res = subprocess.run(
            ["git", "diff", "--name-only", f"{baseline_sha}..HEAD", "--", rel],
            capture_output=True, text=True, check=True, cwd=str(repo_root),
        )
        assert df_res.stdout.strip() == "", f"Historical diff in protected path {rel}: {df_res.stdout}"

    # Verify stereo_prior.py remains frozen relative to Phase 0 baseline commit
    v3c_baseline_sha = "1bf4954584c70f639484d8318c1990aeabe29040"
    sp_st = subprocess.run(
        ["git", "status", "--porcelain", "--", "python/ddsp_mastering/loss/stereo_prior.py"],
        capture_output=True, text=True, check=True, cwd=str(repo_root),
    )
    assert sp_st.stdout.strip() == "", f"Uncommitted change in stereo_prior.py: {sp_st.stdout}"
    sp_df = subprocess.run(
        ["git", "diff", "--name-only", f"{v3c_baseline_sha}..HEAD", "--", "python/ddsp_mastering/loss/stereo_prior.py"],
        capture_output=True, text=True, check=True, cwd=str(repo_root),
    )
    assert sp_df.stdout.strip() == "", f"Forbidden diff in stereo_prior.py: {sp_df.stdout}"

    # Verify AutoMasteringOptimizer keeps V3c disabled by default with zero V3c coupling
    from ddsp_mastering.dsp.chain import DifferentiableMasteringChain
    from ddsp_mastering.loss.combined import MasteringPerceptualLoss
    from ddsp_mastering.loss.dynamic_prior import V3bDynamicPriorLoss
    from ddsp_mastering.optimize import AutoMasteringOptimizer

    opt = AutoMasteringOptimizer(v3b_w_lim=0.50, v3b_w_diff=0.02, v3b_w_conflict=0.02)
    assert hasattr(opt, "v3b_prior_loss")
    assert opt.enable_v3c is False and opt.v3c_prior_loss is None

    # Deterministic bit-identity check before & after invoking V3cStereoCompatibilityLoss
    n_pre = 4800
    t_eval = 14400
    torch.manual_seed(98765)
    mix_ext = torch.randn(1, 2, n_pre + t_eval + TOTAL_DSP_DELAY_SAMPLES, dtype=torch.float32) * 0.2
    ref_chunk = mix_ext[..., n_pre:].clone()

    def _run_v0p_v3b_snapshot():
        chain = DifferentiableMasteringChain(sample_rate=48000.0)
        loss_v0p = MasteringPerceptualLoss(sample_rate=48000.0, target_lufs=-14.0)
        loss_v3b = V3bDynamicPriorLoss(sample_rate=48000.0, w_lim=0.50, w_diff=0.02, w_conflict=0.02)

        z_a = (Z_NEUTRAL.clone() + 0.03).clamp(0.05, 0.95).to(torch.float32).requires_grad_(True)
        y_ext, diag_k = chain(mix_ext, z=z_a, eq_mode="iir", return_diagnostics=True)
        y_post = y_ext[..., n_pre:]
        l_v0p, _ = loss_v0p(y_post, ref_chunk, z=z_a, align_latency=True)
        _, _, _, l_v3b, _ = loss_v3b.compute_chunk_terms(
            mix_audio=mix_ext, diagnostics=diag_k, z=z_a, preroll_samples=n_pre
        )
        l_comb = l_v0p + l_v3b.to(l_v0p.dtype)
        l_comb.backward()
        return (
            y_ext.detach().clone(),
            float(l_v0p.item()),
            float(l_v3b.item()),
            z_a.grad.detach().clone(),
        )

    y0, v0p_0, v3b_0, grad_0 = _run_v0p_v3b_snapshot()

    # Execute V3c standalone forward + backward between snapshots
    v3c = V3cStereoCompatibilityLoss(delay_dsp_samples=TOTAL_DSP_DELAY_SAMPLES)
    z_v3c = (Z_NEUTRAL.clone() + 0.03).clamp(0.05, 0.95).to(torch.float32).requires_grad_(True)
    y_ext_v3c = DifferentiableMasteringChain(sample_rate=48000.0)(mix_ext, z=z_v3c, eq_mode="iir")
    l_v3c, _ = v3c(mix_ext, y_ext_v3c, z=z_v3c, n_pre=n_pre, compute_diagnostics=True)
    l_v3c.backward()

    y1, v0p_1, v3b_1, grad_1 = _run_v0p_v3b_snapshot()

    assert torch.max(torch.abs(y0 - y1)).item() == 0.0
    assert abs(v0p_0 - v0p_1) == 0.0
    assert abs(v3b_0 - v3b_1) == 0.0
    assert torch.max(torch.abs(grad_0 - grad_1)).item() == 0.0


# ==============================================================================
# Gate 8: Frozen Candidate A / Pilot Weights + Sub-Threshold & Silent Prediction
# ==============================================================================
def test_gate_8_predicted_subthreshold_and_silent_when_mix_active():
    # 8.0 Verify frozen Candidate A defaults and pilot same_track weights
    criterion = V3cStereoCompatibilityLoss(delay_dsp_samples=0)
    assert criterion.tau_drop_db == CANDIDATE_A_THRESHOLDS["tau_drop_db"] == 0.50
    assert criterion.tau_corr == CANDIDATE_A_THRESHOLDS["tau_corr"] == (0.03, 0.05, 0.08)
    assert criterion.tau_smr_db == CANDIDATE_A_THRESHOLDS["tau_smr_db"] == (2.0, 2.0, 2.0)
    assert criterion.tau_sub_db == CANDIDATE_A_THRESHOLDS["tau_sub_db"] == 0.50
    assert criterion.alpha_sub == CANDIDATE_A_THRESHOLDS["alpha_sub"] == 1.0
    assert criterion.tau_w_low_max == CANDIDATE_A_THRESHOLDS["tau_w_low_max"] == 1.00
    assert criterion.tau_w_mid == CANDIDATE_A_THRESHOLDS["tau_w_mid"] == 0.15
    assert criterion.tau_w_high == CANDIDATE_A_THRESHOLDS["tau_w_high"] == 0.15
    assert criterion.w_mono_drop == PILOT_SAME_TRACK_WEIGHTS["w_mono_drop"] == 1.00
    assert criterion.w_band_corr == PILOT_SAME_TRACK_WEIGHTS["w_band_corr"] == 10.00
    assert criterion.w_band_smr == PILOT_SAME_TRACK_WEIGHTS["w_band_smr"] == 0.05
    assert criterion.w_width_param == PILOT_SAME_TRACK_WEIGHTS["w_width_param"] == 0.25

    sr = 48000.0
    N = 24000
    t = torch.arange(N, dtype=torch.float64) / sr
    # Correlated multi-band active mix: low (15 Hz), mid (800 Hz), high (18000 Hz)
    # Minimal crossover skirt leakage between bands so low+high alone leaves mid < 1e-10
    tone_l = 0.05 * (torch.sin(2 * math.pi * 15 * t) + torch.sin(2 * math.pi * 800 * t) + torch.sin(2 * math.pi * 18000 * t))
    tone_q = 0.05 * (torch.cos(2 * math.pi * 15 * t) + torch.cos(2 * math.pi * 800 * t) + torch.cos(2 * math.pi * 18000 * t))
    mix_active = torch.stack([tone_l, 0.90 * tone_l + 0.10 * tone_q], dim=0).unsqueeze(0)

    # ------------------------------------------------------------------
    # Case 8.1: Mix active in mid band, pred quasi-silent in mid band (E_tot_mid^pred ~ 1e-10 < E_min)
    # ------------------------------------------------------------------
    pred_low_high = 0.05 * (
        torch.stack([
            torch.sin(2 * math.pi * 15 * t) + torch.sin(2 * math.pi * 18000 * t),
            0.9 * (torch.sin(2 * math.pi * 15 * t) + torch.sin(2 * math.pi * 18000 * t))
            + 0.1 * (torch.cos(2 * math.pi * 15 * t) + torch.cos(2 * math.pi * 18000 * t)),
        ], dim=0).unsqueeze(0)
    )
    # Decorrelated mid component scaled so E_tot_mid^pred ~ 1e-10
    mid_unit = torch.stack([
        torch.sin(2 * math.pi * 800 * t),
        torch.cos(2 * math.pi * 800 * t),
    ], dim=0).unsqueeze(0)
    pred_quasi_mid = (pred_low_high + 1.0e-5 * mid_unit).detach().clone().requires_grad_(True)

    l_81, d_81 = criterion(mix_active, pred_quasi_mid, compute_diagnostics=True)
    assert d_81["band_valid"]["mid"] is True
    assert d_81["bands"]["mid"]["pred_energy_below_emin"] is True
    assert 1e-12 < d_81["bands"]["mid"]["e_tot_pred"] < MIN_ACTIVE_BAND_ENERGY
    assert d_81["loss_band_corr"] > 0.0
    assert d_81["loss_band_smr"] > 0.0
    assert math.isfinite(l_81.item())
    l_81.backward()
    assert pred_quasi_mid.grad is not None
    assert torch.isfinite(pred_quasi_mid.grad).all()
    assert torch.norm(pred_quasi_mid.grad).item() > 0.0

    # Also verify finite non-zero gradients wrt z[35:38] when mid band into imager yields E_tot_mid ~ 1e-10
    tone_lh_l = 0.05 * (torch.sin(2 * math.pi * 15 * t) + torch.sin(2 * math.pi * 18000 * t))
    tone_lh_q = 0.05 * (torch.cos(2 * math.pi * 15 * t) + torch.cos(2 * math.pi * 18000 * t))
    tone_full_l = tone_lh_l + 0.05 * torch.sin(2 * math.pi * 800 * t)
    tone_full_q = tone_lh_q + 0.05 * torch.cos(2 * math.pi * 800 * t)
    mix_corr_mod = torch.stack([tone_full_l, 0.65 * tone_full_l + 0.76 * tone_full_q], dim=0).unsqueeze(0)
    mix_lh_only = torch.stack([tone_lh_l, 0.65 * tone_lh_l + 0.76 * tone_lh_q], dim=0).unsqueeze(0)
    xover = DifferentiableLR4Crossover(sample_rate=sr)
    imager = DifferentiableMidSideImager()
    f_l = torch.tensor(FIXED_CROSSOVER_LOW_HZ, dtype=torch.float64)
    f_h = torch.tensor(FIXED_CROSSOVER_HIGH_HZ, dtype=torch.float64)
    mb_l, _, mb_h = xover(mix_lh_only, f_l, f_h)
    _, mb_m, _ = xover(mix_corr_mod, f_l, f_h)
    mb_m_tiny = mb_m * 2.0e-4  # E_tot_mid drops by 4e-8 => ~ 1e-10
    z_sub = Z_NEUTRAL.to(torch.float64).clone()
    z_sub[35] = 0.95  # w_low > 1.0
    z_sub[36] = 0.98  # w_mid ~ 1.386
    z_sub[37] = 0.98  # w_high ~ 1.386
    z_sub.requires_grad_(True)
    th_sub = map_z_to_theta(z_sub)
    pred_im = imager(mb_l, mb_m_tiny, mb_h, th_sub[35:36], th_sub[36:37], th_sub[37:38])
    l_sub_z, d_sub_z = criterion(mix_corr_mod, pred_im, z=z_sub, compute_diagnostics=True)
    assert d_sub_z["band_valid"]["mid"] is True
    assert d_sub_z["bands"]["mid"]["pred_energy_below_emin"] is True
    assert 1e-12 < d_sub_z["bands"]["mid"]["e_tot_pred"] < MIN_ACTIVE_BAND_ENERGY
    assert d_sub_z["loss_band_corr"] > 0.0
    assert d_sub_z["loss_band_smr"] > 0.0
    l_sub_z.backward()
    assert z_sub.grad is not None
    assert torch.isfinite(z_sub.grad).all()
    assert torch.all(torch.abs(z_sub.grad[35:38]) > 0.0)

    # ------------------------------------------------------------------
    # Case 8.2: Mix active across all 3 bands, pred completely silent (0.0) OR quasi-silent on both L/R
    # ------------------------------------------------------------------
    pred_zero = torch.zeros_like(mix_active).requires_grad_(True)
    l_zero, d_zero = criterion(mix_active, pred_zero, compute_diagnostics=True)
    assert d_zero["broadband_valid"] is True
    assert all(d_zero["band_valid"].values())
    assert d_zero["pred_energy_below_emin_broadband"] is True
    assert all(d_zero["pred_energy_below_emin"].values())
    assert d_zero["loss_band_corr"] > 0.0
    assert d_zero["loss_band_smr"] > 0.0
    assert math.isfinite(l_zero.item())
    l_zero.backward()
    assert torch.isfinite(pred_zero.grad).all()

    pred_tiny_all = (1e-6 * torch.stack([tone_l, tone_q], dim=0).unsqueeze(0)).requires_grad_(True)
    l_tiny, d_tiny = criterion(mix_active, pred_tiny_all, compute_diagnostics=True)
    assert d_tiny["broadband_valid"] is True
    assert all(d_tiny["band_valid"].values())
    assert d_tiny["pred_energy_below_emin_broadband"] is True
    assert all(d_tiny["pred_energy_below_emin"].values())
    assert d_tiny["loss_band_corr"] > 0.0
    assert d_tiny["loss_band_smr"] > 0.0
    assert math.isfinite(l_tiny.item())
    l_tiny.backward()
    assert torch.isfinite(pred_tiny_all.grad).all()
    assert torch.norm(pred_tiny_all.grad).item() > 0.0

    # ------------------------------------------------------------------
    # Case 8.3: Numerical continuity across E_min = 1e-7 threshold (zero step jump)
    # ------------------------------------------------------------------
    # Uniformly scale a 3-band decorrelated prediction so its mid-band energy crosses
    # [2.0 * E_min, 1.0 * E_min, 0.5 * E_min] with zero step discontinuity at E_min.
    pred_3band_unit = torch.stack([tone_full_l, tone_full_q], dim=0).unsqueeze(0)
    with torch.no_grad():
        _, _, _, unit_b_all = criterion._filter_and_slice_pair(mix_active, pred_3band_unit, n_pre=0, delay_samples=0)
        mask_m, _, _ = compute_active_sample_mask(mix_active)
        st_unit = compute_masked_stereo_stats(unit_b_all["mid"], mask_m)
        scale_unit = 1.0 / math.sqrt(float(st_unit["e_tot"].item()))

    losses_across = []
    for factor_emin in (2.0, 1.0, 0.5):
        target_e = factor_emin * MIN_ACTIVE_BAND_ENERGY
        amp = math.sqrt(target_e) * scale_unit
        pred_cross = amp * pred_3band_unit
        _, d_cross = criterion(mix_active, pred_cross, compute_diagnostics=True)
        assert d_cross["band_valid"]["mid"] is True
        assert abs(d_cross["bands"]["mid"]["e_tot_pred"] - target_e) / target_e < 1e-9
        losses_across.append((d_cross["loss_band_corr"], d_cross["loss_band_smr"]))

    for i in range(len(losses_across) - 1):
        assert abs(losses_across[i][0] - losses_across[i + 1][0]) < 1e-6
        assert abs(losses_across[i][1] - losses_across[i + 1][1]) < 1e-6


# ==============================================================================
# Gate 9: Opt-in Integration of V3c into AutoMasteringOptimizer
# ==============================================================================
def test_gate_9_optimizer_v3c_integration_all_checks():
    from ddsp_mastering.optimize import AutoMasteringOptimizer

    # 9.1 Default opt-in guard: enable_v3c=False and no V3cStereoCompatibilityLoss instantiated
    opt_default = AutoMasteringOptimizer()
    assert opt_default.enable_v3c is False
    assert opt_default.v3c_prior_loss is None

    # Build deterministic 2-chunk stereo audio with pre-roll + DSP delay margin
    sr = 48000.0
    n_pre = 4800
    t_eval = 9600
    t_post = t_eval + TOTAL_DSP_DELAY_SAMPLES
    n_tot = n_pre + t_post
    t_vec = torch.arange(n_tot, dtype=torch.float32) / sr

    sig_l = 0.20 * torch.sin(2.0 * math.pi * 110.0 * t_vec) + 0.15 * torch.sin(2.0 * math.pi * 880.0 * t_vec)
    sig_r = 0.18 * torch.sin(2.0 * math.pi * 110.0 * t_vec + 0.25) + 0.14 * torch.cos(2.0 * math.pi * 880.0 * t_vec)
    chunk_0 = torch.stack([sig_l, sig_r], dim=0).unsqueeze(0)
    chunk_1 = torch.stack([0.85 * sig_l, 0.90 * sig_r], dim=0).unsqueeze(0)
    mix_chunks = [chunk_0, chunk_1]
    ref_chunks = [c[..., n_pre:].clone() for c in mix_chunks]
    prerolls = [n_pre, n_pre]

    z_wide = Z_NEUTRAL.clone().to(torch.float32)
    z_wide[35] = 0.99  # imager_width_low > 1.0
    z_wide[36] = 0.99  # imager_width_mid ~ 1.39 > 1.15
    z_wide[37] = 0.99  # imager_width_high ~ 1.39 > 1.15

    # 9.2 Bit-identity of V0p path when enable_v3c=False (both same_track and stylistic)
    for mode in ("same_track", "stylistic"):
        opt_a = AutoMasteringOptimizer(
            sample_rate=sr, lr=0.02, num_steps=2, target_lufs=-14.0, use_hybrid_eq=False
        )
        assert opt_a.enable_v3c is False and opt_a.v3c_prior_loss is None
        z_a, out_a, hist_a = opt_a.optimize_multi_chunk(
            mix_chunks, ref_chunks, reference_mode=mode, initial_z=z_wide, mix_preroll_samples=prerolls
        )
        assert set(hist_a.keys()) == {"loss", "stft_loss", "lufs", "lr"}

        # Run an interleaved enable_v3c=True same_track pass between V0p snapshots
        opt_inter = AutoMasteringOptimizer(
            sample_rate=sr, lr=0.02, num_steps=2, target_lufs=-14.0, use_hybrid_eq=False, enable_v3c=True
        )
        opt_inter.optimize_multi_chunk(
            mix_chunks, ref_chunks, reference_mode="same_track", initial_z=z_wide, mix_preroll_samples=prerolls
        )

        opt_b = AutoMasteringOptimizer(
            sample_rate=sr, lr=0.02, num_steps=2, target_lufs=-14.0, use_hybrid_eq=False
        )
        z_b, out_b, hist_b = opt_b.optimize_multi_chunk(
            mix_chunks, ref_chunks, reference_mode=mode, initial_z=z_wide, mix_preroll_samples=prerolls
        )
        assert torch.max(torch.abs(z_a - z_b)).item() == 0.0
        for ca, cb in zip(out_a, out_b):
            assert torch.max(torch.abs(ca - cb)).item() == 0.0
        for k_hist in hist_a:
            assert hist_a[k_hist] == hist_b[k_hist]

    # 9.3 enable_v3c=True in same_track adds V3c contribution, populates history, yields finite grads
    opt_v0p = AutoMasteringOptimizer(
        sample_rate=sr, lr=0.02, num_steps=2, target_lufs=-14.0, use_hybrid_eq=False, enable_v3c=False
    )
    opt_v3c = AutoMasteringOptimizer(
        sample_rate=sr, lr=0.02, num_steps=2, target_lufs=-14.0, use_hybrid_eq=False, enable_v3c=True
    )
    assert opt_v3c.enable_v3c is True
    assert isinstance(opt_v3c.v3c_prior_loss, V3cStereoCompatibilityLoss)
    assert opt_v3c.v3c_prior_loss.tau_drop_db == CANDIDATE_A_THRESHOLDS["tau_drop_db"]
    assert opt_v3c.v3c_prior_loss.w_mono_drop == PILOT_SAME_TRACK_WEIGHTS["w_mono_drop"]
    assert opt_v3c.v3c_prior_loss.w_band_corr == PILOT_SAME_TRACK_WEIGHTS["w_band_corr"]
    assert opt_v3c.v3c_prior_loss.w_band_smr == PILOT_SAME_TRACK_WEIGHTS["w_band_smr"]
    assert opt_v3c.v3c_prior_loss.w_width_param == PILOT_SAME_TRACK_WEIGHTS["w_width_param"]

    z_0, _, hist_0 = opt_v0p.optimize_multi_chunk(
        mix_chunks, ref_chunks, reference_mode="same_track", initial_z=z_wide, mix_preroll_samples=prerolls
    )
    z_c, _, hist_c = opt_v3c.optimize_multi_chunk(
        mix_chunks, ref_chunks, reference_mode="same_track", initial_z=z_wide, mix_preroll_samples=prerolls
    )
    expected_v3c_keys = {
        "v3c_loss_total",
        "v3c_mono_drop_raw", "v3c_mono_drop_weighted",
        "v3c_band_corr_raw", "v3c_band_corr_weighted",
        "v3c_band_smr_raw", "v3c_band_smr_weighted",
        "v3c_width_param_raw", "v3c_width_param_weighted",
    }
    assert expected_v3c_keys.issubset(hist_c.keys())
    for k_v3c in expected_v3c_keys:
        assert len(hist_c[k_v3c]) == 2
        assert all(math.isfinite(val) for val in hist_c[k_v3c])
    assert hist_c["v3c_loss_total"][0] > 0.0
    assert hist_c["loss"][0] > hist_0["loss"][0]
    assert torch.isfinite(z_c).all()
    assert torch.max(torch.abs(z_c - z_0)).item() > 0.0

    # 9.4 enable_v3c=True explicitly blocks reference_mode="stylistic"
    with pytest.raises(ValueError, match="same_track"):
        opt_v3c.optimize_multi_chunk(
            mix_chunks, ref_chunks, reference_mode="stylistic", initial_z=z_wide, mix_preroll_samples=prerolls
        )
    with pytest.raises(ValueError, match="same_track"):
        opt_v3c.evaluate_loss_iir(
            z_wide, mix_chunks, ref_chunks, prerolls, -14.0, reference_mode="stylistic"
        )
    with pytest.raises(ValueError, match="same_track"):
        opt_v3c._eval_loss_iir(
            mix_chunks, ref_chunks, z=z_wide, reference_mode="stylistic", mix_preroll_samples=prerolls
        )

    # 9.5 Spy on opt.v3c_prior_loss.forward: verify full mix_k, pred_ext_k, n_pre_k, delay=159, compute_diagnostics=False
    recorded_calls = []
    orig_forward = opt_v3c.v3c_prior_loss.forward

    def _spy_forward(*args, **kwargs):
        mix_arg = kwargs["mix_ext"] if "mix_ext" in kwargs else args[0]
        pred_arg = kwargs["pred_ext"] if "pred_ext" in kwargs else args[1]
        recorded_calls.append({
            "mix_len": int(mix_arg.shape[-1]),
            "pred_len": int(pred_arg.shape[-1]),
            "n_pre": int(kwargs.get("n_pre", args[4] if len(args) > 4 else 0)),
            "delay_samples": kwargs.get("delay_samples"),
            "compute_diagnostics": kwargs.get("compute_diagnostics"),
        })
        return orig_forward(*args, **kwargs)

    opt_v3c.v3c_prior_loss.forward = _spy_forward
    try:
        opt_v3c.optimize_multi_chunk(
            mix_chunks, ref_chunks, reference_mode="same_track", initial_z=z_wide, mix_preroll_samples=prerolls
        )
    finally:
        opt_v3c.v3c_prior_loss.forward = orig_forward

    assert len(recorded_calls) == 4  # 2 steps x 2 chunks
    for rc in recorded_calls:
        assert rc["mix_len"] == n_tot
        assert rc["pred_len"] == n_tot
        assert rc["n_pre"] == n_pre
        assert rc["delay_samples"] == TOTAL_DSP_DELAY_SAMPLES == 159
        assert rc["compute_diagnostics"] is False

    # 9.6 Consistency between optimize_multi_chunk (eq_mode="iir") and evaluate_loss_iir
    z_chk = torch.clamp(z_wide, 1e-4, 1.0 - 1e-4)
    z_chk = torch.sigmoid(torch.log(z_chk / (1.0 - z_chk)))
    for en_v3c in (False, True):
        opt_chk = AutoMasteringOptimizer(
            sample_rate=sr, lr=0.02, num_steps=1, target_lufs=-14.0, use_hybrid_eq=False, enable_v3c=en_v3c
        )
        _, _, h_chk = opt_chk.optimize_multi_chunk(
            mix_chunks, ref_chunks, reference_mode="same_track", initial_z=z_chk, mix_preroll_samples=prerolls
        )
        eval_tot, eval_metrics = opt_chk.evaluate_loss_iir(
            z_chk, mix_chunks, ref_chunks, prerolls, -14.0, reference_mode="same_track"
        )
        assert abs(h_chk["loss"][0] - float(eval_tot.item())) < 1e-6
        assert abs(eval_metrics["total_loss"] - float(eval_tot.item())) < 1e-6

    # 9.7 Non-multiplication of loss_width_param by number of chunks K = 4
    # Pure mono active signal (L == R): loss_mono_drop = loss_band_corr = loss_band_smr = 0.0
    mono_sig = 0.25 * (torch.sin(2.0 * math.pi * 100.0 * t_vec) + torch.sin(2.0 * math.pi * 1000.0 * t_vec))
    mono_chunk = torch.stack([mono_sig, mono_sig], dim=0).unsqueeze(0)
    k4_mix = [mono_chunk.clone() for _ in range(4)]
    k4_ref = [mono_chunk[..., n_pre:].clone() for _ in range(4)]
    k4_pre = [n_pre] * 4

    # Note: imager_width_low has max=1.1 in PARAM_SPECS ([0.4, 1.1]); choose w_low=1.08 -> 1x weighted = 0.25 * 0.08^2 = 0.0016
    z_mono_w = Z_NEUTRAL.clone().to(torch.float32)
    z_mono_w[35] = (1.08 - 0.4) / 0.7  # w_low = 1.08 > tau_w_low_max=1.00; w_mid=w_high=1.00

    opt_k4_off = AutoMasteringOptimizer(
        sample_rate=sr, lr=0.01, num_steps=1, target_lufs=-14.0, use_hybrid_eq=False, enable_v3c=False
    )
    opt_k4_on = AutoMasteringOptimizer(
        sample_rate=sr, lr=0.01, num_steps=1, target_lufs=-14.0, use_hybrid_eq=False, enable_v3c=True
    )
    _, _, h_k4_off = opt_k4_off.optimize_multi_chunk(
        k4_mix, k4_ref, reference_mode="same_track", initial_z=z_mono_w, mix_preroll_samples=k4_pre
    )
    _, _, h_k4_on = opt_k4_on.optimize_multi_chunk(
        k4_mix, k4_ref, reference_mode="same_track", initial_z=z_mono_w, mix_preroll_samples=k4_pre
    )
    assert h_k4_on["v3c_mono_drop_raw"][0] == 0.0
    assert h_k4_on["v3c_band_corr_raw"][0] == 0.0
    assert h_k4_on["v3c_band_smr_raw"][0] == 0.0
    expected_single_width = 0.25 * ((1.08 - 1.00) ** 2)  # 0.0016 (1x, NOT 4x = 0.0064)
    observed_delta = h_k4_on["loss"][0] - h_k4_off["loss"][0]
    assert abs(observed_delta - expected_single_width) < 1e-6
    assert abs(observed_delta - 4.0 * expected_single_width) > 0.004

    # 9.8 Isolation from V1, V2, V3a, V3b and zero diff on frozen modules vs 1bf4954..HEAD
    assert opt_v3c.weight_erb == 0.0
    assert opt_v3c.weight_de_harsh == 0.0
    assert opt_v3c.weight_trans == 0.0
    assert opt_v3c.plr_mode == "box"
    assert opt_v3c.weight_eq_prior == 0.0
    assert opt_v3c.enable_v3b is False

    repo_root = Path(__file__).resolve().parents[1]
    base_sha = "1bf4954584c70f639484d8318c1990aeabe29040"
    locked_paths = [
        "python/ddsp_mastering/loss/stereo_prior.py",
        "python/ddsp_mastering/loss/dynamic_prior.py",
        "python/ddsp_mastering/param_space.py",
        "python/ddsp_mastering/dsp",
        "cpp/core_dsp",
    ]
    for rel in locked_paths:
        st_l = subprocess.run(
            ["git", "status", "--porcelain", "--", rel],
            capture_output=True, text=True, check=True, cwd=str(repo_root),
        )
        assert st_l.stdout.strip() == "", f"Uncommitted change in locked path {rel}: {st_l.stdout}"
        df_l = subprocess.run(
            ["git", "diff", "--name-only", f"{base_sha}..HEAD", "--", rel],
            capture_output=True, text=True, check=True, cwd=str(repo_root),
        )
        assert df_l.stdout.strip() == "", f"Forbidden diff in locked path {rel}: {df_l.stdout}"


def test_gate_10_v3c_ablation_pilot_and_blind_pack_integrity():
    """
    Gate 10: Post-pilot verification of Phase 2 deliverables:
      - Strictly 3 same_track sessions (track01..track03), zero stylistic runs;
      - All floats finite, Control V0p parity < 1e-5 vs V3b harmonized baseline,
        41D map_z_to_theta/map_theta_to_z roundtrip < 1e-5 on both arms;
      - Explicit separation of V0p vs V3c objectives + v3c_eval_on_v0p;
      - All 6 WAVs on disk (48 kHz, 2 ch, PCM_24, SHA-256, BS.1770 LUFS) match manifest;
      - Secret-free blind evaluator zip excludes mapping JSON;
      - Zero diff vs Phase 1 baseline (a98c2d6) on optimizer, stereo_prior, DSP,
        V3c preflight artifacts, and historical baselines.
    """
    import hashlib
    import json
    import math
    import subprocess
    import zipfile
    import soundfile as sf

    from ddsp_mastering.loss.loudness import calculate_integrated_loudness_bs1770
    from ddsp_mastering.param_space import PARAM_SPECS, map_theta_to_z, map_z_to_theta

    repo_root = Path(__file__).resolve().parents[1]
    docs_dir = repo_root / "docs" / "sprint3"

    results_path = docs_dir / "SPRINT3_V3C_ABLATION_PILOT_RESULTS.json"
    report_path = docs_dir / "SPRINT3_V3C_ABLATION_PILOT_REPORT.md"
    manifest_json_path = docs_dir / "SPRINT3_V3C_BLIND_AUDIO_MANIFEST.json"
    manifest_md_path = docs_dir / "SPRINT3_V3C_BLIND_AUDIO_MANIFEST.md"
    worksheet_path = docs_dir / "SPRINT3_V3C_BLIND_LISTENING_WORKSHEET.md"
    mapping_path = docs_dir / "SPRINT3_V3C_BLIND_MAPPING.json"
    wav_dir = docs_dir / "blind_v3c_pilot_wavs"
    zip_path = docs_dir / "blind_v3c_pilot_evaluator_pack.zip"
    v3b_results_path = docs_dir / "SPRINT3_V3B_V0P_ABLATION_RESULTS.json"

    for p in (
        results_path,
        report_path,
        manifest_json_path,
        manifest_md_path,
        worksheet_path,
        mapping_path,
        zip_path,
        v3b_results_path,
    ):
        assert p.exists(), f"Missing required Phase 2 / baseline artifact: {p}"

    results = json.loads(results_path.read_text(encoding="utf-8"))
    v3b_results = json.loads(v3b_results_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_json_path.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))

    # 10.1 Recursive finite check (no NaN/Inf anywhere in pilot JSON)
    def _assert_all_finite(obj, path="root"):
        if isinstance(obj, float):
            assert math.isfinite(obj), f"Non-finite float at {path}: {obj}"
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _assert_all_finite(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for idx, v in enumerate(obj):
                _assert_all_finite(v, f"{path}[{idx}]")

    _assert_all_finite(results)

    # 10.2 Scope & session checks: strictly 3 same_track sessions, zero stylistic
    expected_tracks = ["track01", "track02", "track03"]
    assert results["authorized_scope_sessions"] == expected_tracks
    assert results["excluded_stylistic_sessions"] == ["track01_style", "track02_style", "track03_style"]
    sessions = results["sessions"]
    assert len(sessions) == 3
    assert [s["track_prefix"] for s in sessions] == expected_tracks
    for s in sessions:
        assert s["mode"] == "same_track"
        assert s["num_steps"] == 30

    # 10.3 Control V0p parity (< 1e-5 vs V3b harmonized baseline), roundtrip & separate losses
    v3b_by_id = {
        entry["track_prefix"]: entry
        for entry in v3b_results["sessions"]
        if entry.get("mode") == "same_track"
    }
    param_names = [spec.name for spec in PARAM_SPECS]
    assert len(param_names) == 41

    for s in sessions:
        tid = s["track_prefix"]
        assert s["v0p_harmonized_baseline_max_diff_z"] < 1e-5
        ctrl = s["control_v0p"]
        treat = s["treatment_v3c"]

        ref_z = torch.tensor(v3b_by_id[tid]["control_v0p"]["normalized_z_opt"], dtype=torch.float64)
        ctrl_z = torch.tensor(ctrl["normalized_z_opt"], dtype=torch.float64)
        assert float(torch.max(torch.abs(ctrl_z - ref_z)).item()) < 1e-5

        for arm_name, arm in (("control_v0p", ctrl), ("treatment_v3c", treat)):
            z_vec = torch.tensor(arm["normalized_z_opt"], dtype=torch.float32)
            assert z_vec.numel() == 41
            theta_dict = arm["physical_parameters_theta"]
            theta_from_dict = torch.tensor([theta_dict[name] for name in param_names], dtype=torch.float32)
            theta_mapped = map_z_to_theta(z_vec)
            assert float(torch.max(torch.abs(theta_mapped - theta_from_dict)).item()) < 1e-5, (
                f"{tid}.{arm_name} theta mismatch"
            )
            z_roundtrip = map_theta_to_z(theta_from_dict)
            assert float(torch.max(torch.abs(z_roundtrip - z_vec)).item()) < 1e-5, (
                f"{tid}.{arm_name} z roundtrip mismatch"
            )

        # Objective separation: V0p total and V3c weighted total never conflated
        assert "loss_v0p_total" in ctrl["final_evaluation"]["v0p_objective_iir"]
        assert "loss_v3c_total_weighted" in ctrl["v3c_eval_on_v0p"]
        assert abs(
            ctrl["final_evaluation"]["v3c_objective_and_stereo_iir"]["loss_v3c_total_weighted"]
            - ctrl["v3c_eval_on_v0p"]["loss_v3c_total_weighted"]
        ) < 1e-7
        assert "loss_v0p_total" in treat["final_evaluation"]["v0p_objective_iir"]
        assert "loss_v3c_total_weighted" in treat["final_evaluation"]["v3c_objective_and_stereo_iir"]
        assert len(ctrl["loss_history"]["loss"]) == 30 and len(treat["loss_history"]["loss"]) == 30
        assert len(treat["loss_history"]["v3c_loss_total"]) == 30

    # 10.4 Report check (metrics present + distinction proxy vs unperformed human listening)
    report_txt = report_path.read_text(encoding="utf-8")
    assert "V0p" in report_txt and "V3c" in report_txt
    assert "Guardrail" in report_txt and "-0.30" in report_txt
    assert "ascolto" in report_txt.lower() and "proxy" in report_txt.lower()
    assert "track01_style" in report_txt  # mentioned only in excluded note, zero rows in tables
    assert "| **`track01_style`**" not in report_txt

    # 10.5 Blind audio manifest, 6 WAVs on disk, zero key leakage
    assert manifest["wav_count"] == 6
    assert manifest["scope_sessions"] == expected_tracks
    assert manifest["expected_sample_rate_hz"] == 48000
    assert manifest["expected_channels"] == 2
    assert manifest["expected_subtype"] == "PCM_24"
    assert manifest["contains_secret_mapping"] is False
    assert manifest["max_abs_delta_lufs_across_pairs"] < 0.1

    manifest_raw = manifest_json_path.read_text(encoding="utf-8")
    manifest_md_raw = manifest_md_path.read_text(encoding="utf-8")
    worksheet_raw = worksheet_path.read_text(encoding="utf-8")
    for text_blob in (manifest_raw, manifest_md_raw, worksheet_raw):
        assert "Control_V0p" not in text_blob
        assert "Treatment_V3c" not in text_blob

    files = manifest["files"]
    assert len(files) == 6
    for entry in files:
        wav_p = wav_dir / entry["file_name"]
        assert wav_p.exists(), f"Missing blind WAV: {wav_p}"
        assert wav_p.stat().st_size == entry["bytes"]
        sha = hashlib.sha256(wav_p.read_bytes()).hexdigest()
        assert sha == entry["sha256"]
        info = sf.info(str(wav_p))
        assert info.samplerate == 48000
        assert info.channels == 2
        assert info.subtype == "PCM_24"
        assert info.frames == entry["frames"]
        audio, sr = sf.read(str(wav_p), dtype="float32", always_2d=True)
        assert sr == 48000
        tensor_audio = torch.from_numpy(audio.T).unsqueeze(0).to(torch.float64)
        lufs = float(calculate_integrated_loudness_bs1770(tensor_audio, sample_rate=float(sr)))
        assert abs(lufs - entry["integrated_lufs_bs1770"]) < 1e-3

    # 10.6 Blind mapping completeness & evaluator zip secret exclusion
    assert set(mapping.keys()) == set(expected_tracks)
    for tid in expected_tracks:
        pair = mapping[tid]
        assert set(pair["variant_mapping"].keys()) == {"Control_V0p", "Treatment_V3c"}
        assert set(pair["variant_mapping"].values()) == {"variant_A", "variant_B"}
        assert set(pair["files"].values()) == {"Control_V0p", "Treatment_V3c"}

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        expected_zip_entries = {f"blind_v3c_pilot_wavs/{e['file_name']}" for e in files} | {
            "SPRINT3_V3C_BLIND_LISTENING_WORKSHEET.md",
            "SPRINT3_V3C_BLIND_AUDIO_MANIFEST.md",
        }
        assert names == expected_zip_entries
        assert all("MAPPING" not in n.upper() for n in names)

    # 10.7 Immutability vs Phase 1 baseline commit a98c2d67b87d798d449a1018157d50d2a2e7be20
    phase1_sha = "a98c2d67b87d798d449a1018157d50d2a2e7be20"
    locked_phase2_paths = [
        "python/ddsp_mastering/optimize/optimizer.py",
        "python/ddsp_mastering/loss/stereo_prior.py",
        "python/ddsp_mastering/loss/dynamic_prior.py",
        "python/ddsp_mastering/param_space.py",
        "python/ddsp_mastering/dsp",
        "cpp/core_dsp",
        "docs/sprint3/SPRINT3_V3C_PREFLIGHT_RESULTS.json",
        "docs/sprint3/SPRINT3_V3C_PREFLIGHT_REPORT.md",
        "docs/sprint3/SPRINT3_V3B_V0P_ABLATION_RESULTS.json",
        "docs/sprint3/SPRINT3_V3B_V0P_ABLATION_REPORT.md",
    ]
    for rel in locked_phase2_paths:
        st_p = subprocess.run(
            ["git", "status", "--porcelain", "--", rel],
            capture_output=True, text=True, check=True, cwd=str(repo_root),
        )
        assert st_p.stdout.strip() == "", f"Uncommitted change in locked Phase 2 path {rel}: {st_p.stdout}"
        df_p = subprocess.run(
            ["git", "diff", "--name-only", f"{phase1_sha}..HEAD", "--", rel],
            capture_output=True, text=True, check=True, cwd=str(repo_root),
        )
        assert df_p.stdout.strip() == "", f"Forbidden diff in locked Phase 2 path {rel}: {df_p.stdout}"

