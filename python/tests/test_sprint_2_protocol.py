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
import py_core_dsp

from ddsp_mastering.data import (
    ReferenceMode,
    AlignmentResult,
    verify_audio_pair,
    estimate_time_alignment,
    apply_same_track_alignment,
    apply_fractional_delay,
    get_same_track_slice_indices,
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


# 2. Allineamento same-track con ritardi noti artificialmente introdotti e verifica coincidenza
@pytest.mark.parametrize("known_delay", [-120, 0, 48, 350])
def test_same_track_alignment_known_delays(known_delay):
    """Estimate artificially introduced delays and verify resulting aligned waveforms match within tolerance."""
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

    # Apply alignment and verify resulting waveforms are coincident
    aligned_mix, aligned_ref = apply_same_track_alignment(mix, ref, alignment, enable_subsample=False)
    assert aligned_mix.shape == aligned_ref.shape
    max_diff = float(np.max(np.abs(aligned_mix - aligned_ref)))
    assert max_diff < 1e-4, f"Aligned waveforms differ by {max_diff}"
    corr = float(np.corrcoef(aligned_mix[0], aligned_ref[0])[0, 1])
    assert corr > 0.9999


# 2b. Raffinamento sub-sample e compensazione fractional-delay
@pytest.mark.parametrize("true_delay", [12.35, -8.40, 0.25])
def test_subsample_alignment_fractional_delay_rigorous(true_delay):
    """Verify sub-sample alignment estimates fractional lag and achieves > 60 dB SNR via fractional filtering."""
    fs = 48000
    N = 48000
    t = np.arange(N) / fs
    # Bandlimited multitone audio
    mix_mono = (
        0.5 * np.sin(2 * np.pi * 440.0 * t)
        + 0.3 * np.sin(2 * np.pi * 1200.0 * t)
        + 0.2 * np.cos(2 * np.pi * 3200.0 * t)
    ).astype(np.float64)
    mix = np.stack([mix_mono, mix_mono * 0.8], axis=0).astype(np.float32)

    # Generate ground-truth fractionally delayed reference
    t_del = t - true_delay / fs
    ref_mono = (
        0.5 * np.sin(2 * np.pi * 440.0 * t_del)
        + 0.3 * np.sin(2 * np.pi * 1200.0 * t_del)
        + 0.2 * np.cos(2 * np.pi * 3200.0 * t_del)
    ).astype(np.float64)
    ref = np.stack([ref_mono, ref_mono * 0.8], axis=0).astype(np.float32)

    alignment = estimate_time_alignment(mix, ref, sample_rate=fs, max_search_sec=1.0)

    # Check lag estimation
    assert abs(alignment.total_lag_samples - true_delay) < 0.05, (
        f"Estimated total lag {alignment.total_lag_samples} did not match true delay {true_delay}"
    )

    # Apply alignment with sub-sample compensation enabled
    aligned_mix, aligned_ref = apply_same_track_alignment(mix, ref, alignment, enable_subsample=True)
    assert aligned_mix.shape == aligned_ref.shape

    # Apply alignment with integer-only slicing
    int_mix, int_ref = apply_same_track_alignment(mix, ref, alignment, enable_subsample=False)

    err_sub = float(np.max(np.abs(aligned_mix - aligned_ref)))
    err_int = float(np.max(np.abs(int_mix - int_ref)))

    # Fractional compensation must reduce residual error by at least a factor of 10 (20 dB)
    assert err_sub < 1e-3, f"Sub-sample error {err_sub} exceeds 1e-3 threshold"
    assert err_sub < err_int / 10.0, f"Sub-sample error {err_sub} not significantly better than integer {err_int}"
    corr_sub = float(np.corrcoef(aligned_mix[0], aligned_ref[0])[0, 1])
    assert corr_sub > 0.99999


# 2c. Gestione segnali di diversa durata ed eccezioni per overlap non valido
def test_same_track_unequal_lengths_and_invalid_overlap():
    """Verify alignment on tracks of unequal lengths and error when no valid overlap exists."""
    fs = 48000
    t_mix = np.arange(48000) / fs
    mix_mono = np.sin(2 * np.pi * 440.0 * t_mix).astype(np.float32)
    mix = np.stack([mix_mono, mix_mono], axis=0)  # [2, 48000]

    # Ref is longer (54000 samples) and delayed by +200 samples
    t_ref = np.arange(54000) / fs
    ref_mono = np.pad(np.sin(2 * np.pi * 440.0 * t_ref[:54000 - 200]), (200, 0)).astype(np.float32)
    ref = np.stack([ref_mono, ref_mono], axis=0)  # [2, 54000]

    align = estimate_time_alignment(mix, ref, sample_rate=fs)
    assert align.lag_samples == 200

    aligned_mix, aligned_ref = apply_same_track_alignment(mix, ref, align, enable_subsample=False)
    # Valid overlap is min(48000, 54000 - 200) = 48000
    assert aligned_mix.shape == (2, 48000)
    assert aligned_ref.shape == (2, 48000)
    assert np.max(np.abs(aligned_mix - aligned_ref)) < 1e-4

    # Negative lag with longer mix
    mix_long = np.pad(mix, ((0, 0), (0, 6000)))  # [2, 54000]
    align_neg = AlignmentResult(
        lag_samples=-500,
        lag_seconds=-500 / fs,
        subsample_offset=0.0,
        total_lag_samples=-500.0,
        correlation_coefficient=0.95,
        is_aligned=True,
        reliability_score=0.95,
    )
    al_mix_n, al_ref_n = apply_same_track_alignment(mix_long, ref, align_neg, enable_subsample=False)
    # Valid len is min(54000 - 500, 54000) = 53500
    assert al_mix_n.shape == al_ref_n.shape == (2, 53500)

    # Invalid overlap: lag exceeds signal length -> must raise ValueError
    align_impossible = AlignmentResult(
        lag_samples=60000,
        lag_seconds=60000 / fs,
        subsample_offset=0.0,
        total_lag_samples=60000.0,
        correlation_coefficient=0.1,
        is_aligned=False,
        reliability_score=0.1,
    )
    with pytest.raises(ValueError, match="No valid overlapping temporal region"):
        apply_same_track_alignment(mix, ref, align_impossible)


# 3. Corretta compensazione della latenza DDSP (159 smp) senza doppia compensazione
def test_ddsp_latency_compensation_no_double_counting():
    """Verify that file alignment lag and DDSP internal chain latency (159 smp) remain strictly separated."""
    fs = 48000
    # Step 1: Prove DDSP chain introduces exactly 159 samples of group delay
    chain = py_core_dsp.MasteringChain(fs)
    assert chain.get_latency_samples() == 159

    # Unit impulse at n = 500
    impulse = np.zeros((2, 1000), dtype=np.float32)
    impulse[:, 500] = 1.0
    out_impulse = chain.process(impulse)
    peak_idx = int(np.argmax(np.abs(out_impulse[0])))
    # 500 + 159 = 659 (with LR4 AP2 crossover group delay centering peak at 659 or 660)
    assert peak_idx in (659, 660), f"Expected peak near 659, got {peak_idx}"

    # Step 2: Artificial file lag between mix and reference master
    file_lag = 120
    mix = np.random.randn(2, 48000).astype(np.float32)
    ref = np.pad(mix, ((0, 0), (file_lag, 0)))[:, :48000]

    alignment = estimate_time_alignment(mix, ref, sample_rate=fs)
    # Chain latency is declared explicitly as 159 samples and is NEVER added to file delay
    assert alignment.chain_latency_samples == 159
    assert abs(alignment.lag_samples - file_lag) <= 1

    # Applying alignment shifts ONLY the file-level lag
    aligned_mix, aligned_ref = apply_same_track_alignment(mix, ref, alignment, enable_subsample=False)
    assert aligned_mix.shape[1] == 48000 - file_lag
    assert aligned_ref.shape[1] == 48000 - file_lag

    # Step 3: MasteringBenchmarkSuite latency alignment compensates ONLY the 159 DDSP samples
    suite = MasteringBenchmarkSuite(sample_rate=fs, latency_samples=159)
    pred_t = torch.tensor(out_impulse).unsqueeze(0)
    ref_t = torch.tensor(impulse).unsqueeze(0)
    p_al, t_al = suite.align_latency(pred_t, ref_t)

    # After latency slicing of 159 samples, the impulse peak in p_al aligns with ref_t at n = 500
    p_peak = int(torch.argmax(torch.abs(p_al[0, 0])).item())
    t_peak = int(torch.argmax(torch.abs(t_al[0, 0])).item())
    assert abs(p_peak - t_peak) <= 1, f"Chain latency alignment mismatch: p_peak={p_peak}, t_peak={t_peak}"


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


# 4b. Separazione completa dei benchmark same_track e stylistic
def test_benchmark_suite_mode_separation_and_guardrails():
    """Verify benchmark suite segregates same_track from stylistic and excludes invalid metrics."""
    suite = MasteringBenchmarkSuite(sample_rate=48000.0)

    # Generate 4-second signal A and 6-second signal B (differing durations)
    t_a = np.arange(48000 * 4) / 48000.0
    t_b = np.arange(48000 * 6) / 48000.0
    sig_a = torch.tensor(np.stack([np.sin(2 * np.pi * 440 * t_a), np.sin(2 * np.pi * 440 * t_a)], axis=0), dtype=torch.float32)
    sig_b = torch.tensor(np.stack([np.sin(2 * np.pi * 880 * t_b), np.sin(2 * np.pi * 880 * t_b)], axis=0), dtype=torch.float32)

    # Stylistic mode: handles differing lengths without error, computes aggregate descriptors
    res_style = suite.evaluate(sig_a, sig_b, align_latency=False, reference_mode="stylistic")
    assert res_style["reference_mode"] == "stylistic"
    assert res_style["spectral_convergence"] is None
    assert res_style["log_spectral_distance_db"] is None
    assert res_style["aggregate_spectral_distance"] is not None
    assert res_style["aggregate_spectral_distance"] > 0.0
    assert "pred_lufs" in res_style
    assert "delta_plr_db" in res_style

    # Same-track mode: requires common duration, computes STFT and LSD
    sig_b_sliced = sig_b[:, :sig_a.shape[-1]]
    res_same = suite.evaluate(sig_a, sig_b_sliced, align_latency=False, reference_mode="same_track")
    assert res_same["reference_mode"] == "same_track"
    assert res_same["spectral_convergence"] is not None
    assert res_same["log_spectral_distance_db"] is not None
    assert res_same["aggregate_spectral_distance"] is None

    # Invalid mode raises explicit ValueError
    with pytest.raises(ValueError, match="Unrecognized reference_mode"):
        suite.evaluate(sig_a, sig_b, reference_mode="unsupported_eval_mode")


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


# 5b. Feature extraction su intero chunk, min_separation e gestione tracce anomale
def test_multi_chunk_stft_centroid_and_separation_constraint():
    """Verify STFT centroid spans the full chunk duration and min_separation is strictly obeyed."""
    fs = 48000
    selector = MultiChunkSelector(sample_rate=fs, num_chunks=3, chunk_duration_sec=3.0, min_separation_sec=4.0, seed=42)

    # Signal whose frequency switches after 0.5s: 200 Hz for first 0.5s, 5000 Hz for remaining 2.5s
    t = np.arange(int(3.0 * fs)) / fs
    sig_mono = np.where(t < 0.5, np.sin(2 * np.pi * 200 * t), np.sin(2 * np.pi * 5000 * t)).astype(np.float32)
    sig_stereo = np.stack([sig_mono, sig_mono], axis=0)

    rms_db, cf_db, cent = selector._compute_chunk_features(sig_stereo)
    # The full-chunk centroid must reflect the 5000 Hz majority (> 2500 Hz), not just the first 40ms (< 300 Hz)
    assert cent > 2500.0, f"Centroid {cent} Hz failed to reflect full-chunk STFT spectrum"

    # Test strict min_separation enforcement on a 45s track
    np.random.seed(99)
    track_45s = np.random.randn(2, 45 * fs).astype(np.float32) * 0.2
    chunks = selector.select_chunks(track_45s)
    min_sep_samples = int(4.0 * fs)
    for i in range(len(chunks) - 1):
        sep = chunks[i + 1].start_sample - chunks[i].end_sample
        assert sep >= min_sep_samples, f"Separation {sep} is less than required {min_sep_samples}"

    # Test short track handling with warning and metadata
    short_track = np.random.randn(2, int(2.0 * fs)).astype(np.float32) * 0.2
    with pytest.warns(UserWarning, match="shorter than or equal to requested chunk duration"):
        short_chunks = selector.select_chunks(short_track)
    assert len(short_chunks) == 1
    assert selector.last_selected_chunks == 1
    assert len(selector.selection_warnings) >= 1

    # Test quiet track handling with fallback
    quiet_track = np.random.randn(2, 20 * fs).astype(np.float32) * 1e-5
    with pytest.warns(UserWarning, match="Audio energy across all candidate windows is below threshold"):
        quiet_chunks = selector.select_chunks(quiet_track)
    assert len(quiet_chunks) == 1
    assert selector.last_selected_chunks == 1


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
    duration_sec = 20.0  # longer than legacy 10s default
    total_samples = int(duration_sec * fs)

    mix = np.random.randn(total_samples, 2).astype(np.float32) * 0.1
    ref = (mix * 1.05).astype(np.float32)

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


# 11. S2.1-FINAL: Verifica end-to-end dell'indicizzazione temporale del benchmark full-track
@pytest.mark.parametrize(
    "case_name, file_lag, sub_offset",
    [
        ("positive_lag", 120, 0.0),
        ("negative_lag", -120, 0.0),
        ("zero_lag", 0, 0.0),
        ("fractional_positive", 48, 0.35),
        ("fractional_negative", -48, -0.30),
    ],
)
def test_full_track_benchmark_temporal_indexing_end_to_end(tmp_path, case_name, file_lag, sub_offset):
    """Verify S2.1-FINAL temporal indexing across full-track pipeline:
      - Pre-mastering evaluation without PDC (align_latency=False)
      - Post-mastering evaluation matching raw_aligned temporal support with exactly one PDC (align_latency=True)
      - Exact temporal correspondence of reference
      - Strict absence of double compensation
      - Valid finite metrics across all delay configurations
    """
    fs = 48000
    duration_sec = 10.0
    total_samples = int(duration_sec * fs)
    t = np.arange(total_samples) / fs

    # Synthesize rich bandlimited musical content with sharp transient impulse at 2.0s
    bass = 0.4 * np.sin(2 * np.pi * 100.0 * t)
    mid = 0.3 * np.sin(2 * np.pi * 1000.0 * t)
    high = 0.15 * np.sin(2 * np.pi * 4000.0 * t)
    mono_source = (bass + mid + high).astype(np.float64)
    # Distinct transient impulse at sample 96000 (t = 2.0s)
    click_sample = int(2.0 * fs)
    mono_source[click_sample : click_sample + 5] += 0.8

    # Mix is base audio
    mix = np.stack([mono_source, mono_source * 0.9], axis=0).astype(np.float32)

    # Reference is delayed/advanced by file_lag + sub_offset
    total_delay = float(file_lag + sub_offset)
    if abs(sub_offset) >= 1e-4:
        t_ref = t - total_delay / fs
        ref_mono = (
            0.4 * np.sin(2 * np.pi * 100.0 * t_ref)
            + 0.3 * np.sin(2 * np.pi * 1000.0 * t_ref)
            + 0.15 * np.sin(2 * np.pi * 4000.0 * t_ref)
        )
        # Shift click accurately
        ref_click = int(round(click_sample + total_delay))
        ref_mono[ref_click : ref_click + 5] += 0.8
    else:
        int_delay = int(file_lag)
        if int_delay >= 0:
            ref_mono = np.pad(mono_source, (int_delay, 0))[:total_samples]
        else:
            ref_mono = np.pad(mono_source[-int_delay:], (0, -int_delay))

    ref = np.stack([ref_mono, ref_mono * 0.9], axis=0).astype(np.float32)

    raw_path = str(tmp_path / f"mix_{case_name}.wav")
    ref_path = str(tmp_path / f"ref_{case_name}.wav")
    sf.write(raw_path, mix.T, fs)
    sf.write(ref_path, ref.T, fs)

    from scripts.run_mastering import run_mastering_session
    manifest = run_mastering_session(
        raw_path=raw_path,
        ref_path=ref_path,
        out_prefix=f"test_{case_name}",
        mode="same_track",
        num_chunks=2,
        chunk_len_sec=2.0,
        num_steps=3,
        lr=0.03,
        duration_sec=duration_sec,
        out_dir=str(tmp_path),
    )

    # 1. Verify manifest existence and metrics presence
    assert manifest is not None
    pre = manifest.metrics_pre
    post = manifest.metrics_post

    assert pre is not None and post is not None

    # 2. Pre-mastering verification: NO latency compensation applied (align_latency=False)
    # Spectral convergence and LSD must reflect original unmastered vs reference distance
    assert "spectral_convergence" in pre and "log_spectral_distance_db" in pre
    assert not np.isnan(pre["spectral_convergence"])
    assert not np.isnan(pre["log_spectral_distance_db"])
    assert pre["spectral_convergence"] < 0.8  # In phase, well aligned

    # 3. Post-mastering verification: DDSP latency compensated ONCE (align_latency=True)
    assert "spectral_convergence" in post and "log_spectral_distance_db" in post
    assert not np.isnan(post["spectral_convergence"])
    assert not np.isnan(post["log_spectral_distance_db"])
    assert post["pred_lufs"] > -30.0 and post["pred_lufs"] < 0.0
    assert post["ceiling_violation_db"] == 0.0  # Limiter ceiling respected!

    # 4. Temporal correspondence of reference & no double compensation:
    # Check directly using get_same_track_slice_indices
    align = estimate_time_alignment(mix, ref, sample_rate=fs)
    mix_start, ref_start, eval_len = get_same_track_slice_indices(
        align, mix.shape[1], ref.shape[1], enable_subsample=True
    )
    raw_al, ref_al = apply_same_track_alignment(mix, ref, align, enable_subsample=True)

    assert raw_al.shape[1] == eval_len
    assert ref_al.shape[1] == eval_len

    # Expected mix_start conforms to rule:
    expected_mix_start = 0 if file_lag >= 0 else -file_lag
    if abs(sub_offset) >= 1e-4:
        expected_mix_start += 32  # K = filter_half_length

    assert abs(mix_start - expected_mix_start) <= 1, (
        f"mix_start {mix_start} did not match expected {expected_mix_start}"
    )

    # Read the full mastered WAV output from disk (WAV duration must match input without truncation)
    out_wav_path = str(tmp_path / f"test_{case_name}_mastered.wav")
    assert os.path.exists(out_wav_path)
    out_audio, out_sr = sf.read(out_wav_path)
    assert out_sr == fs
    assert out_audio.shape[0] == total_samples, "Full-track WAV output duration must not be truncated!"
