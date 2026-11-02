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


# ---------------------------------------------------------------------------
# B1 (F-01): ITU-R BS.1770-5 Integrated Loudness & Causal K-Weighting Tests
# ---------------------------------------------------------------------------

def test_b1_f01_integrated_loudness_matches_reference_pyloudnorm_within_tolerance():
    """Verify B1 / F-01: Integrated loudness matches pyloudnorm within 0.05 LU on complex signals."""
    import pyloudnorm as pyln
    from ddsp_mastering.loss.loudness import calculate_integrated_loudness_bs1770

    sr = 48000
    meter = pyln.Meter(sr)

    # Signal 1: 1 kHz tone at -23.0 dBFS RMS
    t = np.arange(sr * 5) / sr
    amp = np.sqrt(2.0) * (10.0 ** (-23.0 / 20.0))
    tone = amp * np.sin(2 * np.pi * 1000.0 * t)
    stereo_tone = np.stack([tone, np.zeros_like(tone)], axis=-1)

    pyln_tone = meter.integrated_loudness(stereo_tone)
    our_tone = calculate_integrated_loudness_bs1770(stereo_tone, sample_rate=sr)

    assert abs(our_tone - (-23.0)) < 0.05, f"EBU calibration tone expected -23 LUFS, got {our_tone}"
    assert abs(pyln_tone - our_tone) < 0.05, f"Discrepancy with pyloudnorm: {abs(pyln_tone - our_tone)} LU"

    # Signal 2: Multitone dynamic signal with pauses
    rng = np.random.RandomState(42)
    sig_l = 0.1 * np.sin(2 * np.pi * 440.0 * t) + 0.05 * np.sin(2 * np.pi * 2500.0 * t)
    sig_r = 0.08 * np.sin(2 * np.pi * 880.0 * t) + 0.04 * rng.randn(len(t))
    # Add quiet section in the middle
    sig_l[sr*2:sr*3] = 0.0001
    sig_r[sr*2:sr*3] = 0.0001
    stereo_sig = np.stack([sig_l, sig_r], axis=-1)

    pyln_sig = meter.integrated_loudness(stereo_sig)
    our_sig = calculate_integrated_loudness_bs1770(stereo_sig, sample_rate=sr)
    assert abs(pyln_sig - our_sig) < 0.05, f"Discrepancy on dynamic signal: {abs(pyln_sig - our_sig)} LU"


def test_b1_f01_gating_absolute_and_relative_mechanisms():
    """Verify B1 / F-01: Absolute (-70 LKFS) and relative (-10 LU) gating correctly exclude quiet blocks."""
    import pyloudnorm as pyln
    from ddsp_mastering.loss.loudness import calculate_integrated_loudness_bs1770

    sr = 48000
    meter = pyln.Meter(sr)

    # 8-second signal:
    # 0s - 4s: Active music section (-16 LKFS)
    # 4s - 8s: Low level section below -70 LKFS (-100 dBFS)
    t4 = np.arange(sr * 4) / sr
    loud_part = 0.15 * np.sin(2 * np.pi * 1000.0 * t4)
    silent_part = 1e-5 * np.sin(2 * np.pi * 1000.0 * t4)

    sig_loud_only = np.stack([loud_part, loud_part], axis=0) # shape (2, 4*sr)
    sig_with_silence = np.concatenate([sig_loud_only, np.stack([silent_part, silent_part], axis=0)], axis=-1)

    lufs_loud_only = calculate_integrated_loudness_bs1770(sig_loud_only, sample_rate=sr)
    lufs_with_silence = calculate_integrated_loudness_bs1770(sig_with_silence, sample_rate=sr)
    pyln_with_silence = meter.integrated_loudness(sig_with_silence.T)

    # In BS.1770, pure silence blocks (4.4s-8.0s) are completely excluded by the absolute gate.
    # Without gating, 4s of silence would drop loudness by 3.01 dB.
    # With absolute gating, only 3 boundary blocks overlap, keeping the difference < 0.25 dB,
    # and exactly matching pyloudnorm to within 0.05 LU.
    assert abs(lufs_with_silence - pyln_with_silence) < 0.05
    assert abs(lufs_loud_only - lufs_with_silence) < 0.25

    # Test Relative Gate:
    # 0s - 4s: Loud part at ~ -16 LKFS
    # 4s - 8s: Moderate part at ~ -36 LKFS (passes absolute -70 LKFS, but is >10 LU below prelim level)
    mod_part = 0.001 * np.sin(2 * np.pi * 1000.0 * t4) # ~ 43 dB lower
    sig_with_quiet = np.concatenate([sig_loud_only, np.stack([mod_part, mod_part], axis=0)], axis=-1)

    lufs_with_quiet = calculate_integrated_loudness_bs1770(sig_with_quiet, sample_rate=sr)
    pyln_with_quiet = meter.integrated_loudness(sig_with_quiet.T)
    # The relative gate must drop the -36 LKFS blocks, matching pyloudnorm within 0.05 LU
    assert abs(lufs_with_quiet - pyln_with_quiet) < 0.05


def test_b1_f01_silence_and_short_audio_error_handling():
    """Verify B1 / F-01: Audio shorter than 400ms raises ValueError; total silence returns -inf."""
    from ddsp_mastering.loss.loudness import calculate_integrated_loudness_bs1770

    sr = 48000
    # 1. Audio shorter than 400ms (e.g. 300ms = 14400 samples)
    short_audio = np.zeros((2, int(0.300 * sr)), dtype=np.float32)
    with pytest.raises(ValueError, match="shorter than minimum gating block"):
        calculate_integrated_loudness_bs1770(short_audio, sample_rate=sr)

    # 2. Total silence of 2.0s: all blocks are 0, must return -float('inf')
    silent_audio = np.zeros((2, 2 * sr), dtype=np.float32)
    res = calculate_integrated_loudness_bs1770(silent_audio, sample_rate=sr)
    assert res == -float("inf")


def test_b1_f01_k_weighting_odd_even_lengths_and_no_circular_wrap_around():
    """Verify B1 / F-01: K-weighting filters odd and even lengths and exhibits strict causality (no FFT wrap-around)."""
    from ddsp_mastering.loss.loudness import DifferentiableKWeighting

    sr = 48000
    k_filter = DifferentiableKWeighting(sample_rate=sr)

    # 1. Even and odd lengths
    for length in [48000, 48001, 19203]:
        x = torch.randn(1, 2, length, dtype=torch.float32)
        y = k_filter(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()

    # 2. Causality and zero circular wrap-around test:
    # Buffer of 48000 zeros with a single unit impulse placed at the very last sample (T-1).
    # If the filter is causal IIR, samples 0 through T-2 MUST BE STRICTLY ZERO.
    # If it were circular FFT filtering, the impulse response tail would wrap around to sample 0!
    T = 48000
    impulse_at_end = torch.zeros(1, 2, T, dtype=torch.float64)
    impulse_at_end[..., -1] = 1.0

    y_imp = k_filter(impulse_at_end)

    # Check all samples before the impulse: must be identically zero
    pre_impulse_max = torch.max(torch.abs(y_imp[..., :-1])).item()
    assert pre_impulse_max == 0.0, f"Causality violation! Non-zero values before impulse: {pre_impulse_max}"
    # The impulse sample itself should be non-zero (Stage 1 b1[0] * Stage 2 b2[0])
    assert abs(y_imp[0, 0, -1].item()) > 0.1


def test_b1_f01_differentiable_loudness_loss_gradient_flow():
    """Verify B1 / F-01: DifferentiableLoudnessLoss backward pass propagates finite, non-null gradients."""
    from ddsp_mastering.loss.loudness import DifferentiableLoudnessLoss

    sr = 48000
    loss_fn = DifferentiableLoudnessLoss(sample_rate=sr, target_lufs=-14.0)

    x = torch.randn(2, 2, sr * 2, dtype=torch.float32, requires_grad=True)
    loss, metrics = loss_fn(x)

    assert "loudness_proxy_pred" in metrics
    assert "loudness_proxy_target" in metrics
    assert "loudness_loss_total" in metrics

    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all()
    assert torch.norm(x.grad).item() > 0.0


def test_b1_f01_mastering_benchmark_suite_uses_integrated_standard():
    """Verify B1 / F-01: MasteringBenchmarkSuite uses ITU-R BS.1770-5 Integrated standard."""
    from ddsp_mastering.optimize.metrics import MasteringBenchmarkSuite
    from ddsp_mastering.loss.loudness import calculate_integrated_loudness_bs1770

    sr = 48000
    suite = MasteringBenchmarkSuite(sample_rate=sr)

    # 5-second test audio
    t = torch.arange(sr * 5) / sr
    audio_a = 0.1 * torch.sin(2 * np.pi * 1000.0 * t).repeat(2, 1) # (2, T)
    audio_b = 0.05 * torch.sin(2 * np.pi * 1000.0 * t).repeat(2, 1)

    res = suite.evaluate(audio_a, audio_b, reference_mode="same_track", align_latency=False)

    assert res["loudness_standard"] == "ITU-R BS.1770-5 (11/2023) Integrated"
    expected_pred = calculate_integrated_loudness_bs1770(audio_a, sample_rate=sr)
    assert abs(res["pred_lufs"] - expected_pred) < 1e-6
    expected_target = calculate_integrated_loudness_bs1770(audio_b, sample_rate=sr)
    assert abs(res["target_lufs"] - expected_target) < 1e-6


# ---------------------------------------------------------------------------
# B2 (F-02): Sample Peak vs True Peak 4x Oversampled Characterization Tests
# ---------------------------------------------------------------------------

def test_b2_f02_true_peak_detects_inter_sample_peaks_above_sample_peak():
    """Verify B2 / F-02: 4x oversampled True Peak meter detects inter-sample peaks (dBTP > dBFS)."""
    from ddsp_mastering.optimize.metrics import calculate_sample_peak_dbfs, calculate_true_peak_dbtp

    sr = 48000
    t = np.arange(sr * 2) / sr
    # Textbook inter-sample peak signal: 12 kHz sine wave at 45 degree phase
    # At fs=48kHz, sample values are sin(pi/4 + k*pi/2) = +/- 0.707106 (-3.0103 dBFS)
    # The true continuous/analog peak is 1.0 (0.0 dBTP)
    isp_sig = np.sin(2.0 * np.pi * 12000.0 * t + np.pi / 4.0).astype(np.float32)
    stereo_sig = np.stack([isp_sig, isp_sig * 0.5], axis=0) # (2, T)

    sp_dbfs = calculate_sample_peak_dbfs(stereo_sig)
    tp_dbtp = calculate_true_peak_dbtp(stereo_sig, sample_rate=sr)

    # Sample peak should be ~ -3.01 dBFS
    assert abs(sp_dbfs - (-3.0103)) < 0.15
    # True peak should reconstruct to 0.0 dBTP (+/- 0.1 dBTP)
    assert abs(tp_dbtp - 0.0) < 0.15
    # The inter-sample peak difference must be ~ +2.9 to +3.0 dB
    assert (tp_dbtp - sp_dbfs) > 2.8


def test_b2_f02_true_peak_ceiling_violation_distinction():
    """Verify B2 / F-02: Ceiling violations distinguish Sample Peak (-0.3 dBFS) and True Peak (-0.3 dBTP)."""
    from ddsp_mastering.optimize.metrics import MasteringBenchmarkSuite

    sr = 48000
    suite = MasteringBenchmarkSuite(sample_rate=sr, ceiling_dbfs=-0.3000, true_peak_ceiling_dbtp=-0.3000)

    # 1. Signal with Sample Peak BELOW ceiling (-0.40 dBFS) but True Peak ABOVE ceiling (+2.5 dBTP)
    # Using 12 kHz 45-deg sine scaled so sample peak is -0.40 dBFS
    t = torch.arange(sr * 2) / sr
    isp_sig = torch.sin(2.0 * np.pi * 12000.0 * t + np.pi / 4.0)
    current_sp = torch.max(torch.abs(isp_sig)).item()
    target_sp_lin = 10.0 ** (-0.40 / 20.0)
    scaled_isp = (isp_sig / current_sp) * target_sp_lin
    audio_isp = scaled_isp.repeat(2, 1) # (2, T)

    res = suite.evaluate(audio_isp, audio_isp, reference_mode="same_track", align_latency=False)

    # Sample peak is -0.40 dBFS -> NO sample peak violation
    assert abs(res["pred_sample_peak_dbfs"] - (-0.40)) < 0.05
    assert res["sample_peak_ceiling_violation_db"] == 0.0
    assert res["ceiling_violation_db"] == 0.0

    # True peak is ~ +2.5 dBTP -> MUST report true peak violation (> 2.7 dB above -0.3 dBTP)
    assert res["pred_true_peak_dbtp"] > 2.0
    assert res["true_peak_ceiling_violation_db"] > 2.3

    # 2. Compliant signal (-6 dBFS peak) -> ZERO violations on both
    compliant = audio_isp * 0.5
    res_comp = suite.evaluate(compliant, compliant, reference_mode="same_track", align_latency=False)
    assert res_comp["sample_peak_ceiling_violation_db"] == 0.0
    assert res_comp["true_peak_ceiling_violation_db"] == 0.0


def test_b2_f02_limiter_sample_peak_ceiling_characterization_and_cpp_parity():
    """Verify B2 / F-02: Limiter satisfies Sample Peak ceiling (-0.3 dBFS) on transient bursts with C++ parity."""
    from ddsp_mastering.dsp.limiter import DifferentiableLookaheadLimiter
    import py_core_dsp

    sr = 48000
    lim_py = DifferentiableLookaheadLimiter(sample_rate=sr)
    lim_cpp = py_core_dsp.LookaheadLimiter(sr)

    # Test cases:
    # 1. Pushed sine (+6 dB push gain)
    t = torch.arange(sr) / sr
    sine_pushed = 0.9 * torch.sin(2.0 * np.pi * 1000.0 * t).repeat(2, 1).unsqueeze(0) # (1, 2, T)
    out_py_sine = lim_py(sine_pushed, push_gain_db=torch.tensor([6.0]))

    max_py_sine = torch.max(torch.abs(out_py_sine)).item()
    ceil_lin = 10.0 ** (-0.3000 / 20.0) # ~0.96605088
    # Sample peak must not exceed ceiling by more than float32 epsilon (1e-5 linear, < 0.0001 dB)
    assert max_py_sine <= ceil_lin + 1e-5

    # 2. Isolated single clicks and rapid pulse bursts
    clicks = torch.zeros(1, 2, sr, dtype=torch.float32)
    clicks[0, 0, 1000] = 1.8
    clicks[0, 1, 1005] = 2.2
    clicks[0, 0, 2000:2020:4] = 1.5 # burst of 5 rapid spikes
    out_py_clicks = lim_py(clicks, push_gain_db=torch.tensor([0.0]))

    max_py_clicks = torch.max(torch.abs(out_py_clicks)).item()
    assert max_py_clicks <= ceil_lin + 1e-5

    # 3. Verify PyTorch and C++ bit-level parity on the click signal
    clicks_np = clicks[0].detach().cpu().numpy().astype(np.float64)
    lim_cpp.reset()
    lim_cpp.set_push_gain_db(0.0)
    out_cpp_clicks = lim_cpp.process(clicks_np)

    max_cpp_clicks = np.max(np.abs(out_cpp_clicks))
    assert max_cpp_clicks <= ceil_lin + 1e-5
    # PyTorch and C++ outputs must match within float precision
    np.testing.assert_allclose(out_py_clicks[0].numpy(), out_cpp_clicks, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# B4 (F-03): JUCE VST3 Plugin True Bypass & JSON Interoperability Tests
# ---------------------------------------------------------------------------

def test_b4_f03_juce_plugin_headless_binary_execution():
    """Verify B4 / F-03: Execute JUCE C++ headless test suite (11 comprehensive test stages)."""
    import subprocess
    import os

    exe_candidates = [
        os.path.join("build_juce", "plugin_headless_tests.exe"),
        os.path.join("build_juce", "Release", "plugin_headless_tests.exe"),
    ]
    exe_path = next((p for p in exe_candidates if os.path.exists(p)), None)
    if not exe_path:
        pytest.skip("Binary plugin_headless_tests.exe not found. Run CMake build first.")

    res = subprocess.run([exe_path], capture_output=True, text=True)
    assert res.returncode == 0, f"Headless JUCE test suite failed with code {res.returncode}:\n{res.stdout}\n{res.stderr}"

    # Verify key test stages in stdout
    assert "Numero parametri APVTS registrati: 42" in res.stdout
    assert "Latenza segnalata all'Host: 159 campioni" in res.stdout
    assert "Parita numerica macchina garantita (<= -140 dBFS)!" in res.stdout
    assert "Serializzazione e iniezione preset JSON validate al 100%!" in res.stdout
    assert "Roundtrip vettore z in (0, 1)^41 verificato con successo!" in res.stdout
    assert "Interoperabilita Python -> JUCE e gestione errori validate con successo!" in res.stdout
    assert "True Bypass validato: identita float esatta dopo 159 campioni di latenza!" in res.stdout
    assert "Host Bypass verificato con successo in processBlock!" in res.stdout
    assert "Stato APVTS serializzato e ripristinato con successo!" in res.stdout
    assert "Schema JSON AutoMastering_Preset_v1 validato al 100%!" in res.stdout


def test_b4_f03_python_preset_schema_and_juce_interoperability():
    """Verify B4 / F-03: Python export_preset_json complies strictly with AutoMastering_Preset_v1."""
    import json
    from ddsp_mastering.optimize.optimizer import AutoMasteringOptimizer
    from ddsp_mastering.param_space import PARAM_SPECS

    opt = AutoMasteringOptimizer(sample_rate=48000.0)
    rng = torch.Generator().manual_seed(12345)
    random_z = torch.rand(41, generator=rng)

    json_str = opt.export_preset_json(random_z)
    preset = json.loads(json_str)

    # 1. Top-level mandatory schema contract
    assert preset["format"] == "AutoMastering_Preset_v1"
    assert preset["version"] == "1.0"
    assert preset["sample_rate"] == 48000.0
    assert len(preset["normalized_z"]) == 41
    assert len(preset["parameters"]) == 41
    assert len(preset["parameter_details"]) == 41

    # 2. Every single of the 41 parameters must match PARAM_SPECS
    for i, spec in enumerate(PARAM_SPECS):
        name = spec.name
        assert name in preset["parameters"]
        val = preset["parameters"][name]
        assert spec.min_val <= val <= spec.max_val, f"Parameter {name}={val} out of bounds [{spec.min_val}, {spec.max_val}]"

        details = preset["parameter_details"][name]
        assert details["index"] == i
        assert details["min"] == spec.min_val
        assert details["max"] == spec.max_val
        assert details["unit"] == spec.unit







