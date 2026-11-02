"""
Sprint 2.2 Audit Remediation Tests:
- F-10: Clean metrics.py without dead code.
- F-11: C++ THETA_NEUTRAL and MasteringParameters default initialization consistency.
"""

import numpy as np
import pytest
import torch

from ddsp_mastering.param_space import THETA_NEUTRAL, Z_NEUTRAL, NUM_PARAMS
import py_core_dsp


def test_f11_cpp_theta_neutral_matches_python_and_default_struct():
    """Verify F-11: C++ THETA_NEUTRAL and default MasteringParameters match Python THETA_NEUTRAL."""
    cpp_theta_neutral = np.array(py_core_dsp.get_theta_neutral(), dtype=np.float64)
    cpp_default_struct_theta = np.array(py_core_dsp.get_default_mastering_parameters_theta(), dtype=np.float64)
    py_theta_neutral = THETA_NEUTRAL.detach().cpu().numpy()

    assert len(cpp_theta_neutral) == NUM_PARAMS == 41
    assert len(cpp_default_struct_theta) == NUM_PARAMS == 41

    # Check perfect numerical match between C++ THETA_NEUTRAL and Python THETA_NEUTRAL
    np.testing.assert_allclose(cpp_theta_neutral, py_theta_neutral, rtol=1e-12, atol=1e-12)

    # Check that default MasteringParameters{} struct exactly matches THETA_NEUTRAL
    np.testing.assert_allclose(cpp_default_struct_theta, py_theta_neutral, rtol=1e-12, atol=1e-12)


def test_b3_f04_chunk_selection_breakdown_silence_rejection():
    """Verify B3 / F-04: Displaced chunk landing in silent breakdown is strictly rejected (rms_db >= min_rms_db)."""
    from ddsp_mastering.data.chunking import MultiChunkSelector

    sr = 48000
    chunk_dur = 2.0  # 96,000 samples
    min_sep = 1.0    # 48,000 samples
    selector = MultiChunkSelector(
        sample_rate=sr,
        chunk_duration_sec=chunk_dur,
        num_chunks=3,
        min_rms_db=-30.0,
        min_separation_sec=min_sep,
        seed=42,
    )

    # Total duration 10.0 seconds
    total_len = int(10.0 * sr)
    audio = np.zeros((2, total_len), dtype=np.float32)

    # Only 0 to 2.5s has loud audio (candidates at 0.0s and 0.5s)
    t = np.arange(int(2.5 * sr)) / sr
    audio[:, :int(2.5 * sr)] = 0.316 * np.sin(2 * np.pi * 440.0 * t)
    # The rest (2.5s to 10.0s) is SILENCE

    chunks = selector.select_chunks(audio)

    # Every selected chunk MUST strictly have rms_db >= selector.min_rms_db (-30 dBFS)
    assert len(chunks) >= 1
    for c in chunks:
        assert c.rms_db >= selector.min_rms_db, (
            f"Chunk {c.chunk_id} starting at {c.start_sec}s has rms_db={c.rms_db:.1f} dBFS < {selector.min_rms_db} dBFS!"
        )

    # All chunk starts must be strictly separated by at least chunk_len + min_separation_samples
    for i in range(len(chunks) - 1):
        assert chunks[i + 1].start_sample >= chunks[i].end_sample + selector.min_separation_samples


def test_b3_outro_fade_under_threshold_rejection():
    """Verify B3: Outro fading below min_rms_db is not selected."""
    from ddsp_mastering.data.chunking import MultiChunkSelector

    sr = 48000
    chunk_dur = 2.0
    min_sep = 1.0
    selector = MultiChunkSelector(
        sample_rate=sr,
        chunk_duration_sec=chunk_dur,
        num_chunks=3,
        min_rms_db=-25.0,
        min_separation_sec=min_sep,
        seed=123,
    )

    # 15s track: first 8s loud (-9 dBFS), remaining 7s muted / below threshold (-50 dBFS)
    total_len = int(15.0 * sr)
    audio = np.zeros((2, total_len), dtype=np.float32)
    t = np.arange(total_len) / sr
    audio[:, :int(8.0 * sr)] = 0.5 * np.sin(2 * np.pi * 1000.0 * t[:int(8.0 * sr)])
    # Outro from 8s to 15s has tiny noise (-50 dBFS)
    audio[:, int(8.0 * sr):] = 0.001 * np.random.randn(2, total_len - int(8.0 * sr))

    chunks = selector.select_chunks(audio)
    for c in chunks:
        assert c.rms_db >= selector.min_rms_db
        # No chunk should start in the muted outro (>= 8s)
        assert c.start_sec < 8.0

    for i in range(len(chunks) - 1):
        assert chunks[i + 1].start_sample >= chunks[i].end_sample + selector.min_separation_samples


def test_b3_empty_temporal_zones_fallback_no_duplicates():
    """Verify B3: Temporal zones without candidates fallback without selecting duplicate starts."""
    from ddsp_mastering.data.chunking import MultiChunkSelector

    sr = 48000
    chunk_dur = 2.0
    min_sep = 0.5
    selector = MultiChunkSelector(
        sample_rate=sr,
        chunk_duration_sec=chunk_dur,
        num_chunks=4,
        min_rms_db=-20.0,
        min_separation_sec=min_sep,
        seed=999,
    )

    total_len = int(20.0 * sr)
    audio = np.zeros((2, total_len), dtype=np.float32)
    # Only active between 0-3s and 16-19s
    t = np.arange(total_len) / sr
    audio[:, :int(3.0 * sr)] = 0.4 * np.sin(2 * np.pi * 440.0 * t[:int(3.0 * sr)])
    audio[:, int(16.0 * sr):int(19.0 * sr)] = 0.4 * np.sin(2 * np.pi * 440.0 * t[:int(3.0 * sr)])

    chunks = selector.select_chunks(audio)
    starts = [c.start_sample for c in chunks]
    assert len(starts) == len(set(starts)), "Duplicate chunk starts detected!"

    for c in chunks:
        assert c.rms_db >= selector.min_rms_db

    for i in range(len(chunks) - 1):
        assert chunks[i + 1].start_sample >= chunks[i].end_sample + selector.min_separation_samples


def test_b3_deterministic_seed_and_insufficient_duration_warning():
    """Verify B3: Reproducibility with seed and clear warning on insufficient duration."""
    from ddsp_mastering.data.chunking import MultiChunkSelector
    import warnings

    sr = 48000
    selector = MultiChunkSelector(
        sample_rate=sr,
        chunk_duration_sec=3.0,
        num_chunks=5,
        min_rms_db=-30.0,
        min_separation_sec=2.0,
        seed=777,
    )

    # 10s audio: can fit at most 2 chunks (3s + 2s + 3s = 8s; 3 chunks need 13s)
    audio = np.random.randn(2, int(10.0 * sr)).astype(np.float32) * 0.1

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        chunks1 = selector.select_chunks(audio)
        assert len(w) >= 1
        assert "insufficient to accommodate" in str(w[-1].message)

    chunks2 = selector.select_chunks(audio)
    assert len(chunks1) == len(chunks2) == 2
    for c1, c2 in zip(chunks1, chunks2):
        assert c1.start_sample == c2.start_sample
        assert c1.rms_db == c2.rms_db


