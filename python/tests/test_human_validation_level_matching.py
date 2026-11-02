"""Automated Unit Tests for the Human Validation Level-Matching Utility.

Covers Section 19 requirements:
  - level matching applies ONLY a static linear gain per file (zero dynamic compression/EQ/limiting);
  - all matched signals converge to the same Integrated Loudness (BS.1770-5);
  - True Peak ceiling safety margin (`<= -1.0 dBTP` default) is strictly respected across all files;
    when an initial target would push a high-crest signal above the ceiling, the common target
    loudness is lowered uniformly for all files in the group;
  - original files are never modified or overwritten.
"""

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

from scripts.human_validation.make_level_matched import (
    compute_group_level_matching_gains,
    create_level_matched_wavs,
)
from scripts.human_validation.validate_inputs import compute_file_sha256


def test_level_matching_uses_static_gain_only_and_matches_lufs():
    """Verify level matching applies pure static scalar multiplication and equalizes BS.1770-5 LUFS."""
    sr = 48000
    t = np.arange(int(sr * 1.5)) / float(sr)  # 1.5s (> 400ms BS.1770-5 gating block)

    sig_quiet = np.vstack([
        0.10 * np.sin(2.0 * np.pi * 440.0 * t),
        0.10 * np.cos(2.0 * np.pi * 440.0 * t),
    ]).astype(np.float64)
    sig_loud = np.vstack([
        0.45 * np.sin(2.0 * np.pi * 440.0 * t + 0.3),
        0.45 * np.cos(2.0 * np.pi * 440.0 * t + 0.3),
    ]).astype(np.float64)

    matched, report = compute_group_level_matching_gains(
        {"quiet": sig_quiet, "loud": sig_loud},
        sample_rate=48000.0,
        target_lufs=None,
        ceiling_dbtp=-1.0,
    )

    assert report["post_loudness_spread_lu"] < 1e-3
    for label, orig in (("quiet", sig_quiet), ("loud", sig_loud)):
        v_rep = report["variants"][label]
        assert v_rep["loudness_method"] == "itu_bs1770_5_integrated"
        assert v_rep["post_true_peak_dbtp"] <= -1.0 + 1e-3
        # Prove that the transformation is strictly a single static scalar gain
        scalar = v_rep["applied_linear_gain"]
        np.testing.assert_allclose(matched[label], orig * scalar, atol=1e-14, rtol=0.0)


def test_level_matching_lowers_common_target_when_true_peak_ceiling_would_be_exceeded():
    """Verify that if a high-crest signal would exceed -1.0 dBTP at the requested target LUFS,
    the common target LUFS is lowered uniformly so all files stay <= -1.0 dBTP and remain matched."""
    sr = 48000
    n = int(sr * 1.2)
    t = np.arange(n) / float(sr)

    # Signal A: steady sine wave
    sig_sine = np.vstack([
        0.25 * np.sin(2.0 * np.pi * 300.0 * t),
        0.25 * np.sin(2.0 * np.pi * 300.0 * t),
    ]).astype(np.float64)

    # Signal B: quiet bed with a high-crest transient burst (high PLR)
    sig_burst = (0.04 * np.sin(2.0 * np.pi * 300.0 * t)).astype(np.float64)
    sig_burst[2000:2050] = 0.85  # high peak relative to integrated loudness
    sig_high_crest = np.vstack([sig_burst, sig_burst])

    # Request a high target (-10 LUFS) that would push sig_high_crest well above -1.0 dBTP
    matched, report = compute_group_level_matching_gains(
        {"sine": sig_sine, "high_crest": sig_high_crest},
        sample_rate=48000.0,
        target_lufs=-10.0,
        ceiling_dbtp=-1.0,
    )

    assert report["target_lowered_for_true_peak_ceiling"] is True
    assert report["effective_common_target_lufs"] < -10.0
    # Relative loudness matching between the two signals is still exact!
    assert report["post_loudness_spread_lu"] < 1e-3
    # Both signals strictly respect the -1.0 dBTP ceiling
    for label in ("sine", "high_crest"):
        assert report["variants"][label]["post_true_peak_dbtp"] <= -1.0 + 1e-3


def test_create_level_matched_wavs_preserves_originals_and_prevents_overwrite(tmp_path):
    """Verify WAV export writes distinct copies, preserves original hashes, and refuses accidental overwrite."""
    sr = 48000
    t = np.arange(sr) / float(sr)
    s1 = np.vstack([0.15 * np.sin(2 * np.pi * 250 * t), 0.15 * np.sin(2 * np.pi * 250 * t)])
    s2 = np.vstack([0.35 * np.sin(2 * np.pi * 250 * t), 0.35 * np.sin(2 * np.pi * 250 * t)])

    in_dir = tmp_path / "orig"
    out_dir = tmp_path / "lm"
    in_dir.mkdir()

    p1 = in_dir / "var1.wav"
    p2 = in_dir / "var2.wav"
    sf.write(str(p1), s1.T, sr, subtype="FLOAT")
    sf.write(str(p2), s2.T, sr, subtype="FLOAT")

    sha1_before = compute_file_sha256(str(p1))
    sha2_before = compute_file_sha256(str(p2))

    rep = create_level_matched_wavs(
        input_files={"v1": str(p1), "v2": str(p2)},
        output_dir=str(out_dir),
        ceiling_dbtp=-1.0,
        overwrite=False,
    )
    assert os.path.isfile(rep["variants"]["v1"]["output_path"])
    assert os.path.isfile(rep["variants"]["v2"]["output_path"])
    # Originals untouched
    assert compute_file_sha256(str(p1)) == sha1_before
    assert compute_file_sha256(str(p2)) == sha2_before

    # Second call without overwrite=True must raise FileExistsError
    with pytest.raises(FileExistsError):
        create_level_matched_wavs(
            input_files={"v1": str(p1), "v2": str(p2)},
            output_dir=str(out_dir),
            ceiling_dbtp=-1.0,
            overwrite=False,
        )


def test_level_matching_iterative_refinement_and_error_handling(monkeypatch):
    """Verify iterative refinement recalculates all copies and raises RuntimeError if ceiling is impossible."""
    sr = 48000
    t = np.arange(int(sr * 1.5)) / float(sr)
    sig_1 = np.vstack([0.30 * np.sin(2 * np.pi * 440 * t), 0.30 * np.sin(2 * np.pi * 440 * t)]).astype(np.float64)
    sig_2 = np.vstack([0.15 * np.cos(2 * np.pi * 880 * t), 0.15 * np.cos(2 * np.pi * 880 * t)]).astype(np.float64)

    # 1. Normal run: ceiling is respected and LUFS spread is < 1e-3
    matched, report = compute_group_level_matching_gains(
        {"s1": sig_1, "s2": sig_2},
        sample_rate=48000.0,
        ceiling_dbtp=-1.0,
    )
    assert report["post_loudness_spread_lu"] < 1e-3
    for v in report["variants"].values():
        assert v["post_true_peak_dbtp"] <= -1.0 + 1e-3

    # 2. Impossible ceiling: monkeypatch measure_audio_signal to simulate persistent overshoot
    from scripts.human_validation import make_level_matched
    real_measure = make_level_matched.measure_audio_signal

    def mock_overshoot_measure(audio, sample_rate=48000.0, file_path=None):
        res = real_measure(audio, sample_rate=sample_rate, file_path=file_path)
        res["true_peak_dbtp"] = 0.5  # always exceeds -1.0 dBTP
        return res

    monkeypatch.setattr(make_level_matched, "measure_audio_signal", mock_overshoot_measure)

    with pytest.raises(RuntimeError, match="failed to satisfy True Peak ceiling"):
        compute_group_level_matching_gains(
            {"s1": sig_1, "s2": sig_2},
            sample_rate=48000.0,
            ceiling_dbtp=-1.0,
        )
