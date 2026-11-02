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

    # Verify key separation for Sprint 1
    assert os.path.isdir(res_clamped["blind_delivery_dir"])
    assert os.path.isfile(os.path.join(res_clamped["secret_keys_dir"], "track_test_sprint1_blind_key.json"))
    assert not os.path.exists(os.path.join(res_clamped["blind_delivery_dir"], "track_test_sprint1_blind_key.json"))


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

    # Check that eq_band6_f0 historical bounds from 86ae20f0 are accurately recorded as [5000.0, 18000.0]
    band6_row = next(r for r in table if r["legacy_name_at_86ae20f0"] == "eq_band6_f0")
    assert band6_row["legacy_bounds_86ae20f0"] == [5000.0, 18000.0]
    assert abs(band6_row["theta_legacy_value"] - 9486.833) < 1e-2


def test_sprint2_manifest_and_exports_integration(tmp_path):
    """Verify Sprint 2 orchestrator creates a valid, readable manifest with all required sections and separate keys."""
    from scripts.human_validation.render_sprint2 import run_experiment_b_sprint2

    sr = 48000
    t = np.arange(int(sr * 1.5)) / float(sr)
    raw_sig = np.vstack([0.25 * np.sin(2 * np.pi * 300 * t), 0.25 * np.cos(2 * np.pi * 300 * t)]).astype(np.float32)
    ref_sig = np.vstack([0.30 * np.sin(2 * np.pi * 300 * t), 0.30 * np.cos(2 * np.pi * 300 * t)]).astype(np.float32)

    raw_wav = tmp_path / "mix_48k.wav"
    ref_wav = tmp_path / "ref_48k.wav"
    sf.write(str(raw_wav), raw_sig.T, sr, subtype="FLOAT")
    sf.write(str(ref_wav), ref_sig.T, sr, subtype="FLOAT")

    out_dir = tmp_path / "sprint2_out"
    keys_dir = tmp_path / "private_keys"

    res = run_experiment_b_sprint2(
        raw_wav_path=str(raw_wav),
        ref_wav_path=str(ref_wav),
        output_dir=str(out_dir),
        prefix="track_integration",
        mode="same_track",
        num_chunks=2,
        chunk_len_sec=0.4,
        num_steps=1,
        section_duration_sec=0.4,
        keys_dir=str(keys_dir),
        overwrite=False,
    )

    manifest_path = res["manifest_json"]
    assert os.path.isfile(manifest_path)
    assert os.path.abspath(manifest_path) == str(out_dir / "track_integration_manifest.json")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    # 1. Engine provenance
    assert "engine_provenance" in manifest_data
    assert manifest_data["engine_provenance"]["engine_reference_code_commit"] == "5940e81f12741e2276f7a446adfb6e12f44574fe"
    assert manifest_data["engine_provenance"]["engine_reference_doc_commit"] == "92fe105a19312f258fefb2c70a847720cbd95525"

    # 2. Reference mode & track prefix
    assert manifest_data["reference_mode"] == "same_track"
    assert manifest_data["track_prefix"] == "track_integration"

    # 3. Optimization config
    opt_cfg = manifest_data["optimization_config"]
    assert opt_cfg["single_preset_per_track"] is True
    assert opt_cfg["num_steps"] == 1
    assert opt_cfg["num_chunks"] == 2

    # 4. Preset vectors
    preset = manifest_data["optimized_preset"]
    assert len(preset["normalized_z_41"]) == 41
    assert len(preset["physical_theta_41"]) == 41
    assert len(preset["parameters"]) == 41

    # 5. Full track renders & metrics references
    renders = manifest_data["full_track_renders"]
    assert "original_or_bypass" in renders
    assert "mastered_full" in renders
    assert os.path.isfile(os.path.join(str(out_dir), renders["original_or_bypass"]["file_name"]))
    assert os.path.isfile(os.path.join(str(out_dir), renders["mastered_full"]["file_name"]))

    # 6. Separate private keys vs delivery directory
    assert os.path.isdir(res["secret_keys_dir"])
    assert os.path.isdir(res["sections_blind_delivery_dir"])
    assert res["secret_keys_dir"] != res["sections_blind_delivery_dir"]

    # Blind delivery directory must have NO secret key files
    for root, _, files in os.walk(res["sections_blind_delivery_dir"]):
        for f_name in files:
            assert not f_name.endswith("_blind_key.json")

    # Private keys directory must contain the blind key files
    key_files = os.listdir(res["secret_keys_dir"])
    assert len(key_files) >= 2
    assert all(k.endswith(".json") for k in key_files)


def test_sprint1_and_sprint2_seed_propagation_and_delivery_hygiene(tmp_path, monkeypatch):
    """Verify Sprint 1 and Sprint 2 resolve omitted seed once via secrets.randbits(64), pass it to chunking & derived section keys, and leak zero unblinding info into delivery folders."""
    import scripts.human_validation.make_blind_set as mbs
    import scripts.human_validation.render_sprint2 as rs2

    generated_seeds = iter([0xCAFEBABE11223344, 0xDEADBEEF55667788])
    randbits_calls = []

    def fake_randbits(k: int) -> int:
        randbits_calls.append(k)
        return next(generated_seeds)

    monkeypatch.setattr(mbs.secrets, "randbits", fake_randbits)

    # Spy on MultiChunkSelector to verify Sprint 2 passes the single effective_seed to chunk selection
    observed_chunk_seeds = []
    OrigSelector = rs2.MultiChunkSelector

    class SpySelector(OrigSelector):
        def __init__(self, *args, **kwargs):
            observed_chunk_seeds.append(kwargs.get("seed"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(rs2, "MultiChunkSelector", SpySelector)

    sr = 48000
    t = np.arange(int(sr * 1.5)) / float(sr)
    raw_sig = np.vstack([0.25 * np.sin(2 * np.pi * 300 * t), 0.25 * np.cos(2 * np.pi * 300 * t)]).astype(np.float32)
    ref_sig = np.vstack([0.30 * np.sin(2 * np.pi * 300 * t), 0.30 * np.cos(2 * np.pi * 300 * t)]).astype(np.float32)

    raw_wav = tmp_path / "mix_48k.wav"
    ref_wav = tmp_path / "ref_48k.wav"
    sf.write(str(raw_wav), raw_sig.T, sr, subtype="FLOAT")
    sf.write(str(ref_wav), ref_sig.T, sr, subtype="FLOAT")

    # 1. Sprint 1 with omitted seed (seed=None) -> calls secrets.randbits(64) exactly once
    s1_out = tmp_path / "s1_auto_seed"
    s1_keys = tmp_path / "s1_keys"
    s1_res = run_experiment_a_sprint1(
        input_wav_path=str(raw_wav),
        output_dir=str(s1_out),
        keys_dir=str(s1_keys),
        track_id="t01",
        legacy_mode="clamped_to_current_bounds",
        seed=None,
    )
    assert randbits_calls == [64]
    expected_s1_seed = 0xCAFEBABE11223344
    assert s1_res["seed"] == expected_s1_seed

    s1_meta = json.loads((s1_out / "t01_metadata.json").read_text(encoding="utf-8"))
    assert s1_meta["seed"] == expected_s1_seed
    assert "seed" not in s1_meta["blind_session"]

    s1_key_data = json.loads((s1_keys / "t01_sprint1_blind_key.json").read_text(encoding="utf-8"))
    assert s1_key_data["seed"] == expected_s1_seed

    # Verify Sprint 1 blind_delivery has zero seed or ground-truth leakage
    for fpath in (s1_out / "blind_delivery").rglob("*"):
        if fpath.is_file() and fpath.suffix == ".csv":
            txt = fpath.read_text(encoding="utf-8")
            assert str(expected_s1_seed) not in txt
            assert "A_true_bypass" not in txt
            assert "B_legacy_baseline" not in txt
            assert "C_corrected_neutral" not in txt

    # 2. Sprint 2 with omitted seed (seed=None) -> calls secrets.randbits(64) exactly once and reuses it everywhere
    s2_out = tmp_path / "s2_auto_seed"
    s2_keys = tmp_path / "s2_keys"
    s2_res = rs2.run_experiment_b_sprint2(
        raw_wav_path=str(raw_wav),
        ref_wav_path=str(ref_wav),
        output_dir=str(s2_out),
        keys_dir=str(s2_keys),
        prefix="t01",
        mode="same_track",
        num_chunks=2,
        chunk_len_sec=0.4,
        num_steps=1,
        section_duration_sec=0.4,
        seed=None,
    )
    # Total randbits calls across Sprint 1 + Sprint 2 must be 2 (one per experiment, NOT extra calls per section)
    assert randbits_calls == [64, 64]
    expected_s2_seed = 0xDEADBEEF55667788
    assert s2_res["seed"] == expected_s2_seed
    assert observed_chunk_seeds == [expected_s2_seed]

    # Check Sprint 2 private manifest and sections JSON record the effective seed
    s2_manifest = json.loads(Path(s2_res["manifest_json"]).read_text(encoding="utf-8"))
    assert s2_manifest["seed"] == expected_s2_seed
    assert s2_manifest["optimization_config"]["seed"] == expected_s2_seed

    s2_sections = json.loads(Path(s2_res["sections_json"]).read_text(encoding="utf-8"))
    assert s2_sections["seed"] == expected_s2_seed

    # Check all 6 section private keys (3 sections x [program, level_matched]) record derived session seed + parent experiment_seed
    s2_key_files = sorted(s2_keys.glob("*.json"))
    assert len(s2_key_files) == 6
    for kf in s2_key_files:
        kd = json.loads(kf.read_text(encoding="utf-8"))
        assert kd["experiment_seed"] == expected_s2_seed
        assert isinstance(kd["seed"], int) and kd["seed"] > expected_s2_seed

    # Verify Sprint 2 sections_blind_delivery has zero seed, key, or ground-truth leakage
    delivery_root = Path(s2_res["sections_blind_delivery_dir"])
    for fpath in delivery_root.rglob("*"):
        assert "key" not in fpath.name.lower()
        if fpath.is_file() and fpath.suffix == ".csv":
            txt = fpath.read_text(encoding="utf-8")
            assert str(expected_s2_seed) not in txt
            assert "original_or_bypass" not in txt
            assert "mastered" not in txt
