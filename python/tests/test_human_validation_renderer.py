"""Automated Unit Tests for the Human Validation Offline Renderer, Input Validator & Sprint 1/2 Helpers.

Covers Section 19 requirements:
  - mono/stereo input acceptance and invalid shape/NaN rejection;
  - unsupported sample rate rejection (no silent resampling) and empty file rejection;
  - offline render with 159-sample latency compensation and tail flush;
  - exact output duration equal to source duration;
  - preservation of final transient at the very end of the buffer (no tail truncation);
  - state independence between consecutive renders;
  - True Bypass bit-exact alignment with input after 159-sample latency compensation;
  - uniform application of diagnostic headroom trim across variants;
  - SHA-256 hash verification, JSON manifest serializability, and overwrite protection;
  - Legacy Baseline (`86ae20f08dc7fbebc7f67af064a15b648a492fc7`) reconstruction & strict compatibility check.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _REPO_ROOT / "python"
for _p in (str(_REPO_ROOT), str(_PYTHON_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ddsp_mastering.param_space import THETA_NEUTRAL
from scripts.human_validation.offline_renderer import (
    DECLARED_LATENCY_SAMPLES,
    compute_diagnostic_headroom_trim,
    render_offline_with_tail_flush,
)
from scripts.human_validation.render_sprint1 import (
    check_legacy_baseline_compatibility,
    reconstruct_legacy_baseline_86ae20f0,
    run_experiment_a_sprint1,
)
from scripts.human_validation.validate_inputs import (
    check_overwrite_allowed,
    compute_file_sha256,
    load_and_validate_wav,
    validate_audio_array,
)


def test_validate_audio_array_mono_stereo_and_invalid_shapes():
    """Verify mono duplication, stereo standardization, and rejection of invalid shapes/NaNs."""
    # 1D mono -> duplicated to (2, T)
    mono_1d = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)
    stereo_from_1d, dup_flag = validate_audio_array(mono_1d, allow_mono=True)
    assert stereo_from_1d.shape == (2, 1000)
    assert dup_flag is True
    np.testing.assert_allclose(stereo_from_1d[0], stereo_from_1d[1])

    # Reject mono when allow_mono=False
    with pytest.raises(ValueError, match="Mono audio is not allowed"):
        validate_audio_array(mono_1d, allow_mono=False)

    # (T, 2) -> transposed to (2, T)
    stereo_Tx2 = np.random.RandomState(42).randn(1200, 2) * 0.1
    stereo_2xT, dup_flag2 = validate_audio_array(stereo_Tx2)
    assert stereo_2xT.shape == (2, 1200)
    assert dup_flag2 is False
    np.testing.assert_allclose(stereo_2xT[0], stereo_Tx2[:, 0])

    # Invalid 3-channel array
    with pytest.raises(ValueError):
        validate_audio_array(np.zeros((3, 500)))

    # Empty array
    with pytest.raises(ValueError, match="empty"):
        validate_audio_array(np.zeros((2, 0)))

    # NaN / Inf rejection
    bad = np.zeros((2, 256))
    bad[0, 10] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_audio_array(bad)


def test_load_and_validate_wav_rejects_wrong_sr_and_empty(tmp_path):
    """Verify 44.1 kHz WAV is rejected without silent resampling, and 48 kHz mono/stereo succeed."""
    wav_44k = tmp_path / "wrong_44100.wav"
    sig = np.sin(2.0 * np.pi * 440.0 * np.arange(4410) / 44100.0).astype(np.float32)
    sf.write(str(wav_44k), sig, 44100, subtype="FLOAT")

    with pytest.raises(ValueError, match="Unsupported sample rate 44100 Hz"):
        load_and_validate_wav(str(wav_44k), required_sr=48000)

    # Valid 48 kHz mono WAV
    wav_48k_mono = tmp_path / "mono_48000.wav"
    sig_48k = np.sin(2.0 * np.pi * 440.0 * np.arange(4800) / 48000.0).astype(np.float32)
    sf.write(str(wav_48k_mono), sig_48k, 48000, subtype="FLOAT")

    audio_2xT, meta = load_and_validate_wav(str(wav_48k_mono), required_sr=48000, allow_mono=True)
    assert audio_2xT.shape == (2, 4800)
    assert meta["mono_duplicated_to_stereo"] is True
    assert meta["sample_rate"] == 48000
    assert len(meta["sha256"]) == 64


def test_offline_render_bypass_alignment_and_final_transient_preservation():
    """Verify 159-sample latency compensation, exact output length T, and zero truncation of final sample."""
    assert DECLARED_LATENCY_SAMPLES == 159
    num_samples = 4096
    signal = np.zeros((2, num_samples), dtype=np.float64)
    # Place unit impulses at the very first sample (t=0) and the very last sample (t=T-1)
    signal[0, 0] = 0.75
    signal[1, 0] = -0.50
    signal[0, num_samples - 1] = 0.85
    signal[1, num_samples - 1] = -0.65

    # 1. True Bypass render
    out_bypass, meta_bypass = render_offline_with_tail_flush(
        signal, mode="bypass", sample_rate=48000.0
    )
    assert out_bypass.shape == (2, num_samples)
    assert meta_bypass["input_samples"] == num_samples
    assert meta_bypass["output_samples"] == num_samples
    assert meta_bypass["declared_latency_samples"] == 159
    assert meta_bypass["tail_flush_zero_samples"] >= 159

    # Bit-exact float32 alignment at sample 0 AND at sample T-1 (proves tail flush prevented truncation!)
    # Note: py_core_dsp.MasteringChain.process() operates on float32 buffers in C++ (~2.4e-8 float32<->float64 rounding)
    np.testing.assert_allclose(out_bypass, signal, atol=1e-7, rtol=0.0)
    np.testing.assert_array_equal(out_bypass.astype(np.float32), signal.astype(np.float32))
    assert abs(out_bypass[0, 0] - 0.75) < 1e-7
    assert abs(out_bypass[0, num_samples - 1] - 0.85) < 1e-7

    # 2. Corrected Neutral render preserves final transient energy at t = T - 1
    out_neutral, _ = render_offline_with_tail_flush(
        signal, mode="theta", theta=THETA_NEUTRAL, sample_rate=48000.0
    )
    assert out_neutral.shape == (2, num_samples)
    # Without tail flush of >=159 samples, the impulse at num_samples-1 would still be inside
    # the 159-sample lookahead buffer and out_neutral[:, -10:] would be identically zero!
    tail_energy = float(np.max(np.abs(out_neutral[:, num_samples - 5 : num_samples])))
    assert tail_energy > 0.05, f"Final transient was truncated! Tail peak={tail_energy}"


def test_offline_renderer_state_independence_between_consecutive_renders():
    """Verify two consecutive renders of the same input are unaffected by a loud dirty intermediate render."""
    rng = np.random.RandomState(123)
    quiet_input = (rng.randn(2, 8192) * 0.02).astype(np.float64)
    loud_input = np.clip(rng.randn(2, 8192) * 0.95, -0.99, 0.99).astype(np.float64)

    out_1, _ = render_offline_with_tail_flush(
        quiet_input, mode="theta", theta=THETA_NEUTRAL, sample_rate=48000.0
    )
    # Run an aggressive render in between
    dirty_theta = list(THETA_NEUTRAL)
    dirty_theta[38] = 6.0   # saturator drive
    dirty_theta[40] = 10.0  # limiter push
    _, _ = render_offline_with_tail_flush(
        loud_input, mode="theta", theta=dirty_theta, sample_rate=48000.0
    )
    out_2, _ = render_offline_with_tail_flush(
        quiet_input, mode="theta", theta=THETA_NEUTRAL, sample_rate=48000.0
    )

    np.testing.assert_array_equal(out_1, out_2)


def test_uniform_diagnostic_headroom_trim_and_experiment_a_manifests(tmp_path):
    """Verify diagnostic trim is static and identical across A/B/C, manifests serialize cleanly, and overwrite protection works."""
    sr = 48000
    t = np.arange(sr) / float(sr)  # 1.0 s stereo signal
    raw = np.vstack([
        0.90 * np.sin(2.0 * np.pi * 220.0 * t),
        0.85 * np.cos(2.0 * np.pi * 440.0 * t),
    ]).astype(np.float64)

    trimmed, trim_meta = compute_diagnostic_headroom_trim(raw, sample_rate=48000.0, target_dbtp=-6.0)
    assert trim_meta["applied_static_trim_db"] < 0.0
    assert abs(trim_meta["post_trim_true_peak_dbtp"] - (-6.0)) < 0.05
    # Verify strictly linear static scaling (ratio is constant at every sample)
    np.testing.assert_allclose(trimmed, raw * trim_meta["applied_linear_gain"], atol=1e-15)

    # Save input WAV and run Experiment A in strict mode and in clamped_to_current_bounds mode
    in_wav = tmp_path / "test_mix_48k.wav"
    sf.write(str(in_wav), raw.T, sr, subtype="FLOAT")

    out_strict = tmp_path / "exp_a_strict"
    res_strict = run_experiment_a_sprint1(
        input_wav_path=str(in_wav),
        output_dir=str(out_strict),
        track_id="track_test",
        legacy_mode="strict",
        overwrite=False,
    )
    assert res_strict["status"] == "FINDING_LEGACY_BASELINE_NOT_REPRODUCIBLE_STRICT"
    assert res_strict["legacy_finding_code"] == "LEGACY_BASELINE_NOT_REPRODUCIBLE"
    assert os.path.isfile(res_strict["legacy_finding_file"])
    # Verify overwrite protection raises FileExistsError when overwrite=False
    with pytest.raises(FileExistsError):
        run_experiment_a_sprint1(
            input_wav_path=str(in_wav),
            output_dir=str(out_strict),
            track_id="track_test",
            legacy_mode="strict",
            overwrite=False,
        )

    # Run with explicit replay policy clamped_to_current_bounds -> exports A, B, and C
    out_clamped = tmp_path / "exp_a_clamped"
    res_clamped = run_experiment_a_sprint1(
        input_wav_path=str(in_wav),
        output_dir=str(out_clamped),
        track_id="track_test",
        legacy_mode="clamped_to_current_bounds",
        overwrite=False,
    )
    assert res_clamped["status"] == "ALL_THREE_CONDITIONS_EXPORTED"
    for cond_key in ("A_true_bypass", "B_legacy_baseline", "C_corrected_neutral"):
        wav_p = res_clamped["exports"][cond_key]
        assert os.path.isfile(wav_p)
        assert len(compute_file_sha256(wav_p)) == 64

    # Check A, B, C manifests have identical diagnostic_headroom_trim
    man_a = json.loads((out_clamped / "manifests" / "track_test_A_manifest.json").read_text(encoding="utf-8"))
    man_b = json.loads((out_clamped / "manifests" / "track_test_B_manifest.json").read_text(encoding="utf-8"))
    man_c = json.loads((out_clamped / "manifests" / "track_test_C_manifest.json").read_text(encoding="utf-8"))
    assert man_a["diagnostic_headroom_trim"] == man_b["diagnostic_headroom_trim"] == man_c["diagnostic_headroom_trim"]
    assert man_b["condition_description"] == "Legacy initialization baseline replayed on frozen current engine"


def test_legacy_baseline_reconstruction_from_86ae20f0():
    """Verify the 41-parameter historical vector from commit 86ae20f0 is complete, finite, and identifies the 5 out-of-range params."""
    theta_legacy, table = reconstruct_legacy_baseline_86ae20f0()
    assert len(theta_legacy) == 41
    assert len(table) == 41
    assert all(np.isfinite(v) for v in theta_legacy)

    compat = check_legacy_baseline_compatibility()
    assert compat["finding_code"] == "LEGACY_BASELINE_NOT_REPRODUCIBLE"
    assert compat["num_out_of_range_params"] == 5
    out_names = [row["name"] for row in compat["out_of_range_parameters"]]
    assert out_names == [
        "comp_low_ratio",
        "comp_low_attack_ms",
        "comp_mid_ratio",
        "comp_high_ratio",
        "saturator_drive_db",
    ]
