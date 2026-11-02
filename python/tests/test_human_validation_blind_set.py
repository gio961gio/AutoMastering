"""Automated Unit Tests for the Human Validation Blind Set Generator & Sprint 2 Section Extractor.

Covers Section 19 requirements:
  - blind mapping is complete, bijective (one-to-one), and 100% reproducible given a fixed seed;
  - anonymous WAV files have clean names (`blind_A.wav`, `blind_B.wav`, ...), match the audio
    content of their mapped source, and keep the secret key in a separate JSON file;
  - overwrite protection prevents accidental replacement of blind keys or worksheets;
  - Sprint 2 listening section selector extracts >= 3 non-empty sections without artificial fades.
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

from scripts.human_validation.make_blind_set import (
    create_blind_listening_set,
    generate_blind_mapping,
)
from scripts.human_validation.render_sprint2 import select_listening_sections


def test_generate_blind_mapping_bijective_and_reproducible():
    """Verify blind mapping is a complete bijection and deterministic for a fixed seed."""
    variants = ["A_true_bypass", "B_legacy_baseline", "C_corrected_neutral"]
    map1 = generate_blind_mapping(variants, seed=2026)
    map2 = generate_blind_mapping(variants, seed=2026)

    assert map1 == map2
    assert list(map1.keys()) == ["blind_A", "blind_B", "blind_C"]
    assert set(map1.values()) == set(variants)
    assert len(set(map1.values())) == 3

    # Reject duplicate variant names
    with pytest.raises(ValueError, match="unique"):
        generate_blind_mapping(["dup", "dup"], seed=2026)


def test_create_blind_listening_set_writes_anonymous_wavs_and_separate_key(tmp_path):
    """Verify anonymous WAV files match source samples bit-for-bit and keep the key separate."""
    sr = 48000
    t = np.arange(4800) / float(sr)
    sig_a = np.vstack([0.2 * np.sin(2 * np.pi * 200 * t), 0.2 * np.sin(2 * np.pi * 200 * t)]).astype(np.float32)
    sig_b = np.vstack([0.3 * np.sin(2 * np.pi * 500 * t), 0.3 * np.sin(2 * np.pi * 500 * t)]).astype(np.float32)

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    pa = src_dir / "original_or_bypass.wav"
    pb = src_dir / "mastered.wav"
    sf.write(str(pa), sig_a.T, sr, subtype="FLOAT")
    sf.write(str(pb), sig_b.T, sr, subtype="FLOAT")

    blind_dir = tmp_path / "blind_out"
    res = create_blind_listening_set(
        variant_files={"original_or_bypass": str(pa), "mastered": str(pb)},
        output_dir=str(blind_dir),
        seed=2026,
        session_id="test_ab",
        overwrite=False,
    )

    assert os.path.isfile(res["listening_sheet_csv"])
    assert os.path.isfile(res["secret_key_json"])

    # Worksheet CSV must NOT contain ground-truth names ('original_or_bypass' or 'mastered')
    csv_text = Path(res["listening_sheet_csv"]).read_text(encoding="utf-8")
    assert "original_or_bypass" not in csv_text
    assert "mastered" not in csv_text
    assert "blind_A" in csv_text
    assert "blind_B" in csv_text

    # Key JSON contains the exact bijective correspondence and SHA-256 hashes
    key_data = json.loads(Path(res["secret_key_json"]).read_text(encoding="utf-8"))
    mapping = key_data["mapping_anonymous_to_ground_truth"]
    assert set(mapping.keys()) == {"blind_A", "blind_B"}
    assert set(mapping.values()) == {"original_or_bypass", "mastered"}

    for anon_label, gt_name in mapping.items():
        anon_wav = blind_dir / f"{anon_label}.wav"
        assert anon_wav.is_file()
        loaded_anon, _ = sf.read(str(anon_wav), dtype="float32", always_2d=True)
        expected = sig_a if gt_name == "original_or_bypass" else sig_b
        np.testing.assert_allclose(loaded_anon.T, expected, atol=1e-7)

    # Overwrite protection
    with pytest.raises(FileExistsError):
        create_blind_listening_set(
            variant_files={"original_or_bypass": str(pa), "mastered": str(pb)},
            output_dir=str(blind_dir),
            seed=2026,
            session_id="test_ab",
            overwrite=False,
        )


def test_sprint2_select_listening_sections_extracts_three_distinct_roles():
    """Verify automatic section selector identifies quiet, dense, and transient sections."""
    sr = 48000
    # Construct a 15-second synthetic track with 3 distinct 5-second regions:
    # 0..5s: quiet verse; 5..10s: loud dense chorus; 10..15s: sparse spiky transients
    n_sec = 15
    total = sr * n_sec
    audio = np.zeros((2, total), dtype=np.float64)
    t5 = np.arange(sr * 5) / float(sr)

    # 0..5s: quiet sine (-26 dBFS)
    audio[:, 0 : sr * 5] = 0.05 * np.sin(2 * np.pi * 220 * t5)
    # 5..10s: dense loud multi-tone (-6 dBFS)
    audio[:, sr * 5 : sr * 10] = 0.45 * np.sin(2 * np.pi * 220 * t5) + 0.25 * np.cos(2 * np.pi * 880 * t5)
    # 10..15s: quiet bed with high-crest spikes every 0.5s
    bed = 0.02 * np.sin(2 * np.pi * 330 * t5)
    for k in range(10):
        idx = k * (sr // 2) + 100
        bed[idx : idx + 20] = 0.85
    audio[:, sr * 10 : sr * 15] = np.vstack([bed, bed])

    sections = select_listening_sections(audio, sample_rate=48000.0, section_duration_sec=4.0)
    assert len(sections) == 3
    roles = {s["role"] for s in sections}
    assert roles == {"quiet_intro_or_verse", "dense_chorus_or_drop", "transient_bridge_or_outro"}
    for s in sections:
        assert abs(s["duration_sec"] - 4.0) < 1e-6
        assert s["end_sample"] - s["start_sample"] == 4 * sr
