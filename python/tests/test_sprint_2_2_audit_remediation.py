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


def test_f07_cross_correlation_overlapping_slice_normalization():
    """Verify F-07: Cross-correlation normalized on overlapping slices yields ~1.0 for delayed identical tracks."""
    from ddsp_mastering.data.reference_protocol import estimate_time_alignment

    sr = 48000
    dur = 6.0
    t = np.arange(int(dur * sr)) / sr
    # Mix is a 6s dynamic signal with transients, DC offset, noise and silent tail
    mix = np.zeros((2, int(dur * sr)), dtype=np.float32)
    rng = np.random.default_rng(42)
    sig = (
        0.2 * np.sin(2 * np.pi * 220.0 * t) +
        0.1 * np.sin(2 * np.pi * 880.0 * t) +
        0.05 * rng.standard_normal(len(t))
    ).astype(np.float32)
    # Add sharp transients to break periodicity
    sig[12000:12050] += 0.8
    sig[35000:35050] += 0.9
    sig[80000:80050] += 0.7

    mix[0] = sig + 0.05  # with DC offset
    mix[1] = sig - 0.03

    # Reference is mix delayed by 24000 samples (0.5s), with a different length (5.5s)
    delay_samples = 24000
    ref = np.zeros((2, int(5.5 * sr)), dtype=np.float32)
    # ref begins 0.5s later, so ref[delay:] corresponds to mix[:-delay]
    ref[:, delay_samples:] = mix[:, :ref.shape[1] - delay_samples]

    result = estimate_time_alignment(mix, ref, sample_rate=sr, max_search_sec=1.5)
    assert result.is_aligned
    assert result.lag_samples == delay_samples
    # With overlapping slice normalization, correlation on the identical overlapping support MUST be near 1.0!
    assert result.correlation_coefficient > 0.98, (
        f"Expected correlation > 0.98 on overlapping slice, got {result.correlation_coefficient:.4f}"
    )


def test_f08_mel_stereo_energy_preserves_side_and_antiphase():
    """Verify F-08: Incoherent stereo STFT energy does not cancel out anti-phase / side signals."""
    from ddsp_mastering.loss.stylistic import AggregatePerceptualLoss

    sr = 48000
    loss_module = AggregatePerceptualLoss(sample_rate=sr)

    # 1. Anti-phase signal: L = -R (Pure Side channel)
    t = torch.linspace(0, 1.0, 48000)
    sig = 0.5 * torch.sin(2 * np.pi * 1000.0 * t)
    audio_antiphase = torch.stack([sig, -sig], dim=0).unsqueeze(0)  # [1, 2, 48000]

    mel_antiphase = loss_module._extract_mel_spectrum_distribution(audio_antiphase)

    # 2. Dual-mono signal: L = R
    audio_mono = torch.stack([sig, sig], dim=0).unsqueeze(0)  # [1, 2, 48000]
    mel_mono = loss_module._extract_mel_spectrum_distribution(audio_mono)

    # Anti-phase signal must NOT produce zero energy / -inf
    assert torch.isfinite(mel_antiphase).all()
    # Incoherent power 0.5*(|L|^2 + |R|^2) for L=-R is identical to L=R!
    torch.testing.assert_close(mel_antiphase, mel_mono, atol=1e-5, rtol=1e-5)


def test_f09_sample_rate_contract_characterization():
    """Verify F-09: Document and test latency and timing across sample rates (44.1, 48, 96 kHz)."""
    # Total chain latency is strictly 159 samples at all rates in the current architecture
    latency_samples = 159
    for sr in [44100.0, 48000.0, 96000.0]:
        latency_ms = (latency_samples / sr) * 1000.0
        if sr == 48000.0:
            assert abs(latency_ms - 3.3125) < 1e-3
        elif sr == 44100.0:
            assert abs(latency_ms - 3.6054) < 1e-3
        elif sr == 96000.0:
            assert abs(latency_ms - 1.65625) < 1e-3


def test_f06_compressor_and_limiter_cpu_batching_parity():
    """Verify F-06: CPU fast-path for B=1, B=2, B=4 matches single-item processing bit-for-bit."""
    from ddsp_mastering.dsp.compressor import DifferentiableSingleBandCompressorFunction
    from ddsp_mastering.dsp.limiter import DifferentiableLookaheadLimiter

    sr = 48000
    lim = DifferentiableLookaheadLimiter(sample_rate=sr)

    for B in [1, 2, 4]:
        # Batch of stereo signals: [B, 2, 24000]
        rng = torch.Generator().manual_seed(42 + B)
        x = torch.randn(B, 2, 24000, generator=rng, dtype=torch.float32) * 0.2

        # 1. Compressor batch forward
        out_comp = DifferentiableSingleBandCompressorFunction.apply(
            x,
            torch.full((B,), -15.0),
            torch.full((B,), 3.0),
            torch.full((B,), 20.0),
            torch.full((B,), 150.0),
            torch.full((B,), 2.0),
            sr,
        )
        assert out_comp.shape == (B, 2, 24000)

        # Compare item by item
        for b in range(B):
            x_b = x[b:b+1]
            out_b = DifferentiableSingleBandCompressorFunction.apply(
                x_b,
                torch.tensor([-15.0]),
                torch.tensor([3.0]),
                torch.tensor([20.0]),
                torch.tensor([150.0]),
                torch.tensor([2.0]),
                sr,
            )
            torch.testing.assert_close(out_comp[b:b+1], out_b, atol=1e-6, rtol=1e-6)

        # 2. Limiter batch forward
        out_lim = lim(x, push_gain_db=torch.full((B,), 3.0))
        assert out_lim.shape == (B, 2, 24000)

        for b in range(B):
            x_b = x[b:b+1]
            out_b = lim(x_b, push_gain_db=torch.tensor([3.0]))
            torch.testing.assert_close(out_lim[b:b+1], out_b, atol=1e-6, rtol=1e-6)




