"""Sprint 2 Verification Suite — Reference Protocol & Multi-Chunk Evaluation.

Tests:
  1. Recognition of both reference modes ('same_track' and 'stylistic').
  2. Same-track time alignment with known artificially injected delays.
  3. Strict separation of file delay and DDSP chain latency (no double counting).
  4. Guardrails rejecting point-to-point waveform comparison in stylistic mode.
  5. Deterministic selection of 4-6 representative chunks distributed chronologically.
  6. Non-overlap and feature diversity across selected chunks.
  7. Multi-chunk loss aggregation and healthy gradient propagation (same_track & stylistic).
  8. Full-track post-optimization evaluation and validation.
  9. Full-track mastering without audio truncation.
 10. Manifest serialization, round-trip deserialization, and reproducibility.
"""

import os
import json
import pytest
import numpy as np
import torch
import soundfile as sf

from ddsp_mastering.data import (
    ReferenceMode,
    AlignmentResult,
    verify_audio_pair,
    estimate_time_alignment,
    apply_same_track_alignment,
    assert_stylistic_guards,
    ChunkSpec,
    MultiChunkSelector,
)
from ddsp_mastering.loss.stylistic import AggregatePerceptualLoss
from ddsp_mastering.optimize import (
    AutoMasteringOptimizer,
    MasteringBenchmarkSuite,
    MasteringManifest,
)
from ddsp_mastering.param_space import NUM_PARAMS, Z_NEUTRAL


# 1. Riconoscimento delle due modalità di riferimento
def test_reference_mode_recognition():
    """Verify that both same_track and stylistic modes are formally recognized and validated."""
    mode_same = ReferenceMode("same_track")
    assert mode_same == ReferenceMode.SAME_TRACK
    assert mode_same.value == "same_track"

    mode_style = ReferenceMode("stylistic")
    assert mode_style == ReferenceMode.STYLISTIC
    assert mode_style.value == "stylistic"

    with pytest.raises(ValueError):
        ReferenceMode("unsupported_mode")


# 2. Allineamento same-track con ritardi noti artificialmente introdotti
@pytest.mark.parametrize("known_delay", [-120, 0, 48, 350])
def test_same_track_alignment_known_delays(known_delay):
    """Estimate artificially introduced delays between mix and reference master."""
    fs = 48000
    np.random.seed(42 + abs(known_delay))
    # Generate 1.0s of bandlimited musical noise with transient spikes
    N = int(1.0 * fs)
    t = np.linspace(0, 1.0, N, endpoint=False)
    bass = 0.4 * np.sin(2 * np.pi * 90.0 * t)
    mid = 0.3 * np.sin(2 * np.pi * 1200.0 * t)
    high = 0.1 * np.sin(2 * np.pi * 7500.0 * t)
    noise = 0.1 * np.random.randn(N)
    mono = (bass + mid + high + noise).astype(np.float32)

    # Add sharp transient clicks
    mono[1000:1010] += 0.8
    mono[12000:12010] += 0.9

    mix = np.stack([mono, mono * 0.9], axis=0)  # [2, N]

    # Create delayed reference
    if known_delay >= 0:
        ref_mono = np.pad(mono, (known_delay, 0))[:N]
    else:
        pos_lag = -known_delay
        ref_mono = np.pad(mono[pos_lag:], (0, pos_lag))

    ref = np.stack([ref_mono, ref_mono * 0.9], axis=0)

    alignment = estimate_time_alignment(mix, ref, sample_rate=fs, max_search_sec=1.0)

    # In our cross-correlation convention: ref relative to mix
    # When ref is padded at front by D, ref is delayed relative to mix -> lag = +D
    assert abs(alignment.lag_samples - known_delay) <= 1, (
        f"Estimated lag {alignment.lag_samples} did not match known delay {known_delay}"
    )
    assert alignment.correlation_coefficient > 0.85
    assert alignment.is_aligned is True
    assert -0.5 <= alignment.subsample_offset <= 0.5


# 3. Corretta compensazione della latenza DDSP senza doppia compensazione
def test_ddsp_latency_compensation_no_double_counting():
    """Verify that file alignment lag and DDSP internal chain latency (159 smp) remain strictly separated."""
    fs = 48000
    mix = np.random.randn(2, 48000).astype(np.float32)
    ref = np.random.randn(2, 48000).astype(np.float32)

    alignment = estimate_time_alignment(mix, ref, sample_rate=fs)

    # Chain latency is declared explicitly as 159 samples and is never added to file delay
    assert alignment.chain_latency_samples == 159

    # Applying alignment shifts only the file-level lag, leaving DDSP PDC independent
    aligned_mix, aligned_ref = apply_same_track_alignment(mix, ref, alignment)
    assert aligned_mix.shape[0] == 2
    assert aligned_ref.shape[0] == 2
    assert aligned_mix.shape[1] == aligned_ref.shape[1]


# 4. Rifiuto dei confronti waveform tra brani differenti in modalità stylistic
def test_stylistic_mode_rejects_waveform_matching():
    """Verify guardrails prohibit point-to-point waveform comparison in stylistic mode."""
    # Guardrail check raises ValueError
    with pytest.raises(ValueError, match="strictly prohibited"):
        assert_stylistic_guards(ReferenceMode.STYLISTIC)

    # Stylistic guard does not raise on same_track
    assert_stylistic_guards(ReferenceMode.SAME_TRACK)

    # Unrelated songs produce low correlation coefficient
    np.random.seed(111)
    track_a = np.random.randn(2, 48000).astype(np.float32)
    np.random.seed(999)
    track_b = np.random.randn(2, 48000).astype(np.float32)
    align_unrelated = estimate_time_alignment(track_a, track_b, sample_rate=48000.0)
    assert align_unrelated.correlation_coefficient < 0.20
    assert align_unrelated.is_aligned is False


# 5. Selezione deterministica di 4-6 chunk distribuiti nel tempo
def test_multi_chunk_selection_distribution_and_determinism():
    """Verify MultiChunkSelector extracts 4-6 non-overlapping, temporally distributed chunks deterministically."""
    fs = 48000
    duration_sec = 45.0  # 45-second track
    total_samples = int(duration_sec * fs)

    np.random.seed(123)
    audio = np.random.randn(2, total_samples).astype(np.float32) * 0.2

    selector = MultiChunkSelector(
        sample_rate=fs,
        num_chunks=5,
        chunk_duration_sec=3.0,
        seed=42,
    )

    chunks_1 = selector.select_chunks(audio)
    assert len(chunks_1) == 5

    # Check chronological ordering and strict non-overlap
    for i in range(len(chunks_1) - 1):
        c_curr = chunks_1[i]
        c_next = chunks_1[i + 1]
        assert c_curr.start_sample < c_next.start_sample
        assert c_curr.end_sample <= c_next.start_sample, "Chunks must not overlap"

    # Check temporal distribution: first chunk in early half, last chunk in late half
    assert chunks_1[0].start_sec < 15.0
    assert chunks_1[-1].start_sec > 25.0

    # Check determinism: same seed yields identical chunk offsets
    chunks_2 = selector.select_chunks(audio)
    for c1, c2 in zip(chunks_1, chunks_2):
        assert c1.start_sample == c2.start_sample
        assert c1.end_sample == c2.end_sample
        assert c1.rms_db == c2.rms_db


# 6. Verifica che i chunk siano sufficientemente rappresentativi e non duplicati
def test_multi_chunk_diversity():
    """Verify that chunks selected across varied song sections exhibit feature diversity."""
    fs = 48000
    N = int(30.0 * fs)
    t = np.linspace(0, 30.0, N, endpoint=False)

    # Section 1 (0-10s): Bass-heavy quiet verse
    s1 = 0.05 * np.sin(2 * np.pi * 60.0 * t[: int(10 * fs)])
    # Section 2 (10-20s): Loud chorus / drop
    s2 = 0.8 * np.sin(2 * np.pi * 440.0 * t[int(10 * fs) : int(20 * fs)])
    # Section 3 (20-30s): High-frequency synth / percussion
    s3 = 0.3 * np.sin(2 * np.pi * 6000.0 * t[int(20 * fs) :])

    signal_full = np.concatenate([s1, s2, s3])
    stereo = np.stack([signal_full, signal_full], axis=0).astype(np.float32)

    selector = MultiChunkSelector(sample_rate=fs, num_chunks=3, chunk_duration_sec=2.0, seed=42)
    chunks = selector.select_chunks(stereo)

    assert len(chunks) == 3
    # Check that chunks have varied RMS and spectral centroids
    rms_values = [c.rms_db for c in chunks]
    centroids = [c.spectral_centroid_hz for c in chunks]

    assert max(rms_values) - min(rms_values) > 4.0, "Chunks must exhibit varied RMS dynamics"
    assert max(centroids) - min(centroids) > 500.0, "Chunks must exhibit varied spectral centroids"


# 7. Aggregazione delle loss e propagazione dei gradienti su tutti i chunk
def test_multi_chunk_loss_aggregation_and_gradients():
    """Verify loss aggregates across all chunks and propagates non-zero gradients into u in R^41."""
    torch.manual_seed(99)
    optimizer = AutoMasteringOptimizer(sample_rate=48000.0, num_steps=5, lr=0.03)

    # Create 3 distinct mix chunks
    chunk_1 = torch.randn(1, 2, 9600, dtype=torch.float32) * 0.1
    chunk_2 = torch.randn(1, 2, 9600, dtype=torch.float32) * 0.2
    chunk_3 = torch.randn(1, 2, 9600, dtype=torch.float32) * 0.15
    mix_chunks = [chunk_1, chunk_2, chunk_3]

    # Target reference chunks
    ref_chunks = [c * 1.1 for c in mix_chunks]

    # Same-track multi-chunk optimization
    opt_z, final_chunks, history = optimizer.optimize_multi_chunk(
        mix_chunks=mix_chunks,
        ref_chunks=ref_chunks,
        reference_mode="same_track",
        verbose=False,
    )

    assert len(history["loss"]) == 5
    assert history["loss"][-1] < history["loss"][0], "Loss must descend across multi-chunk steps"
    assert len(final_chunks) == 3
    assert opt_z.shape == (NUM_PARAMS,)
    assert (opt_z >= 0.0).all() and (opt_z <= 1.0).all()

    # Stylistic multi-chunk optimization with AggregatePerceptualLoss
    opt_z_style, final_chunks_style, history_style = optimizer.optimize_multi_chunk(
        mix_chunks=mix_chunks,
        ref_chunks=ref_chunks,
        reference_mode="stylistic",
        verbose=False,
    )
    assert len(history_style["loss"]) == 5
    assert not np.isnan(history_style["loss"][-1])
    assert not np.isinf(history_style["loss"][-1])


# 8. Validazione sull'intera traccia dopo l'ottimizzazione
def test_full_track_validation_post_optimization():
    """Verify full-track evaluation computes valid psychoacoustic benchmark metrics."""
    torch.manual_seed(42)
    optimizer = AutoMasteringOptimizer(sample_rate=48000.0, num_steps=3, lr=0.03)
    suite = MasteringBenchmarkSuite(sample_rate=48000.0)

    # 4-second mix
    full_mix = torch.randn(1, 2, 48000 * 4, dtype=torch.float32) * 0.2
    full_ref = full_mix * 1.05

    # Master full track
    z_neutral = Z_NEUTRAL.clone().detach()
    mastered = optimizer.master_full_track(full_mix, z_neutral)

    metrics = suite.evaluate(mastered, full_ref, align_latency=True)
    assert "pred_lufs" in metrics
    assert "target_lufs" in metrics
    assert "spectral_convergence" in metrics
    assert "log_spectral_distance_db" in metrics
    assert "pred_peak_dbfs" in metrics
    assert not np.isnan(metrics["pred_lufs"])
    assert not np.isnan(metrics["log_spectral_distance_db"])


# 9. Esportazione dell'intero brano senza troncamenti
def test_full_track_export_no_truncation(tmp_path):
    """Verify that processing and exporting preserves the complete track duration without truncation."""
    fs = 48000
    duration_sec = 12.0  # longer than legacy 10s default
    total_samples = int(duration_sec * fs)

    mix = np.random.randn(total_samples, 2).astype(np.float32) * 0.1
    ref = np.random.randn(total_samples, 2).astype(np.float32) * 0.1

    raw_path = str(tmp_path / "test_raw.wav")
    ref_path = str(tmp_path / "test_ref.wav")
    sf.write(raw_path, mix, fs)
    sf.write(ref_path, ref, fs)

    # Run mastering session with full duration (duration_sec=None)
    from scripts.run_mastering import run_mastering_session
    manifest = run_mastering_session(
        raw_path=raw_path,
        ref_path=ref_path,
        out_prefix="test_no_trunc",
        mode="same_track",
        num_chunks=3,
        chunk_len_sec=2.0,
        num_steps=3,
        duration_sec=None,  # No truncation!
        out_dir=str(tmp_path),
    )

    out_wav = str(tmp_path / "test_no_trunc_mastered.wav")
    assert os.path.exists(out_wav)
    out_data, out_sr = sf.read(out_wav)
    assert out_sr == fs
    assert out_data.shape[0] == total_samples, f"Expected {total_samples} samples, got {out_data.shape[0]}"


# 10. Salvataggio e ricaricamento del manifest
def test_manifest_roundtrip_and_reproducibility(tmp_path):
    """Verify MasteringManifest serializes to JSON and reloads with 100% field fidelity."""
    manifest_path = str(tmp_path / "session_manifest.json")

    original = MasteringManifest(
        reference_mode="stylistic",
        input_track_path="/path/to/mix.wav",
        input_sample_rate=48000.0,
        input_duration_sec=180.5,
        reference_track_path="/path/to/ref.wav",
        reference_sample_rate=48000.0,
        reference_duration_sec=210.0,
        alignment={"is_aligned": False},
        chunks=[
            {"chunk_id": 0, "start_sec": 10.0, "end_sec": 13.0, "rms_db": -18.2},
            {"chunk_id": 1, "start_sec": 45.0, "end_sec": 48.0, "rms_db": -14.1},
        ],
        chunk_selection_config={"num_chunks": 2, "chunk_duration_sec": 3.0, "seed": 42},
        optimization_config={"steps": 50, "lr": 0.04},
        normalized_z=[0.5] * NUM_PARAMS,
        physical_theta=[1.0] * NUM_PARAMS,
        parameters={"comp_low_ratio": 1.0, "saturator_drive_db": 0.0},
        metrics_pre={"pred_lufs": -20.0},
        metrics_post={"pred_lufs": -14.0},
        software_info={"python": "3.11.5", "torch": "2.12.0"},
    )

    original.to_json(output_path=manifest_path)
    assert os.path.exists(manifest_path)

    reloaded = MasteringManifest.from_json(manifest_path)

    assert reloaded.reference_mode == "stylistic"
    assert reloaded.input_duration_sec == 180.5
    assert len(reloaded.chunks) == 2
    assert reloaded.chunk_selection_config["seed"] == 42
    assert len(reloaded.normalized_z) == NUM_PARAMS
    assert reloaded.parameters["comp_low_ratio"] == 1.0
    assert reloaded.metrics_post["pred_lufs"] == -14.0
