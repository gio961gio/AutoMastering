"""Sprint 4 Verification Suite: True Peak / Inter-Sample Peak Guardrail (T1..T7, AC1..AC5)."""

import io
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import py_core_dsp
from ddsp_mastering.dsp.chain import DifferentiableMasteringChain
from ddsp_mastering.dsp.limiter import (
    DELIVERY_CEILING_DBTP,
    INTERNAL_CEILING_DBTP,
    LOOKAHEAD_SAMPLES,
    TP_BOXCAR_SAMPLES,
    TP_FIR_GROUP_DELAY_BASE_SAMPLES,
    TP_FIR_PROTOTYPE_COEFFS,
    TP_FIR_PROTOTYPE_TAPS,
    DifferentiableLookaheadLimiter,
    get_tp_fir_prototype_coeffs,
)
from ddsp_mastering.optimize.metrics import (
    MasteringBenchmarkSuite,
    calculate_sample_peak_dbfs,
    calculate_true_peak_dbtp,
)
from ddsp_mastering.param_space import map_z_to_theta
from scripts.human_validation.offline_renderer import (
    DECLARED_LATENCY_SAMPLES,
    export_validated_pcm24_wav,
    render_offline_with_tail_flush,
)


def _internal_detector_4x_peak_db(audio_2xT: np.ndarray) -> float:
    """Measure peak using the project's internal 4x polyphase FIR + aligned 1x detector.

    Note: This custom Type-II WLS polyphase FIR is an internal DSP detector topology and is
    explicitly distinct from the external normative Kaiser sinc metric (`calculate_true_peak_dbtp`).
    """
    import torch.nn.functional as F

    coeffs = torch.tensor(get_tp_fir_prototype_coeffs(), dtype=torch.float64)
    kernels = torch.stack([coeffs[0::4], coeffs[1::4], coeffs[2::4], coeffs[3::4]], dim=0).unsqueeze(1)
    x = torch.from_numpy(np.ascontiguousarray(audio_2xT, dtype=np.float64))
    y_poly = F.conv1d(F.pad(x.unsqueeze(1), (11, 0)), kernels)  # [2, 4, T]
    x_1x = F.pad(x, (6, 0))[:, : x.shape[-1]].unsqueeze(1)  # [2, 1, T]
    peak_lin = float(torch.cat([y_poly, x_1x], dim=1).abs().max().item())
    return 20.0 * math.log10(max(peak_lin, 1e-12))


def _measure_pcm24_roundtrip(audio_2xT: np.ndarray, sr: int = 48000):
    buf = io.BytesIO()
    sf.write(buf, np.ascontiguousarray(audio_2xT.T, dtype=np.float32), sr, format="WAV", subtype="PCM_24")
    buf.seek(0)
    disk, disk_sr = sf.read(buf, dtype="float32", always_2d=True)
    disk_2xT = np.ascontiguousarray(disk.T, dtype=np.float64)
    tp = float(calculate_true_peak_dbtp(disk_2xT, sample_rate=float(disk_sr)))
    sp = float(calculate_sample_peak_dbfs(disk_2xT))
    return tp, sp, disk_2xT


# ---------------------------------------------------------------------------
# A. Provenance, Single-Source-of-Truth Coefficients & Latency Contract
# ---------------------------------------------------------------------------

def test_fir_coefficients_provenance_symmetry_unity_dc_and_cpp_identity():
    cpp_coeffs = np.array(py_core_dsp.get_true_peak_fir_coeffs(), dtype=np.float64)
    py_coeffs = np.array(TP_FIR_PROTOTYPE_COEFFS, dtype=np.float64)
    runtime_coeffs = np.array(get_tp_fir_prototype_coeffs(), dtype=np.float64)

    assert len(cpp_coeffs) == TP_FIR_PROTOTYPE_TAPS == 48
    assert np.max(np.abs(cpp_coeffs - py_coeffs)) < 1e-15
    assert np.max(np.abs(cpp_coeffs - runtime_coeffs)) == 0.0

    # Exact even half-sample mirror symmetry h[m] == h[47 - m]
    sym_err = float(np.max(np.abs(cpp_coeffs - cpp_coeffs[::-1])))
    assert sym_err < 1e-15

    # Exact per-phase DC normalization == 1.0
    for phi in range(4):
        phase_sum = float(np.sum(cpp_coeffs[phi::4]))
        assert abs(phase_sum - 1.0) < 1e-14

    # Nominal symmetric group delay (48 - 1) / (2 * 4) = 5.875 base samples
    assert math.isclose(float(py_core_dsp.TP_FIR_GROUP_DELAY_BASE_SAMPLES), 5.875, abs_tol=1e-12)
    assert math.isclose(TP_FIR_GROUP_DELAY_BASE_SAMPLES, 5.875, abs_tol=1e-12)
    assert int(py_core_dsp.TOTAL_CHAIN_LATENCY_SAMPLES) == 159
    assert DECLARED_LATENCY_SAMPLES == 159
    assert int(py_core_dsp.LIMITER_LOOKAHEAD_SAMPLES) == LOOKAHEAD_SAMPLES == 144
    assert int(py_core_dsp.LIMITER_BOXCAR_SAMPLES) == TP_BOXCAR_SAMPLES == 136
    assert math.isclose(float(py_core_dsp.LIMITER_INTERNAL_CEILING_DBTP), INTERNAL_CEILING_DBTP, abs_tol=1e-12)
    assert math.isclose(float(py_core_dsp.LIMITER_DELIVERY_CEILING_DBTP), DELIVERY_CEILING_DBTP, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# T1. High-Frequency Inter-Sample Peak Tones (12, 16, 18, 20, 22 kHz)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("freq_hz", [1000.0, 12000.0, 16000.0, 18000.0, 20000.0, 22000.0])
def test_t1_high_frequency_isp_tones_cpp_and_python(freq_hz: float):
    sr = 48000
    n = np.arange(4800, dtype=np.float64)
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(128) / 128.0))
    win = np.ones(4800, dtype=np.float64)
    win[:128] = ramp
    win[-128:] = ramp[::-1]

    py_lim = DifferentiableLookaheadLimiter(sample_rate=float(sr), detector_mode="true_peak_4x")
    cpp_lim = py_core_dsp.LookaheadLimiter(float(sr), "true_peak_4x")

    for amp in (1.2, 3.0):
        for phi0 in (0.0, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi):
            # Test both smooth windowed sine and abrupt step onset
            for smooth in (True, False):
                s = amp * np.sin(2.0 * np.pi * freq_hz * n / sr + phi0)
                if smooth:
                    s = s * win
                u_np = np.pad(np.stack([s, 0.92 * s], axis=0), ((0, 0), (0, 256)))

                cpp_lim.reset()
                y_cpp = cpp_lim.process(u_np.astype(np.float32)).astype(np.float64)
                tp_cpp, sp_cpp, _ = _measure_pcm24_roundtrip(y_cpp, sr=sr)
                assert tp_cpp <= -0.30, (
                    f"T1 C++ PCM_24 TP violation at f={freq_hz}, amp={amp}, phi={phi0:.2f}, smooth={smooth}: {tp_cpp:.4f} dBTP"
                )
                assert sp_cpp <= -0.30

                with torch.no_grad():
                    y_py = py_lim(torch.from_numpy(u_np).unsqueeze(0)).squeeze(0).numpy()
                tp_py, sp_py, _ = _measure_pcm24_roundtrip(y_py, sr=sr)
                assert tp_py <= -0.30
                assert sp_py <= -0.30


@pytest.mark.parametrize(
    ("sr", "freqs_hz"),
    [
        (44100, (21100.0, 21500.0, 21900.0, 22000.0)),
        (48000, (23000.0, 23500.0, 23900.0, 23990.0)),
        (96000, (46000.0, 47000.0, 47800.0, 47980.0)),
    ],
)
def test_t1b_near_nyquist_dense_phases_and_detector_vs_external_pcm24(sr: int, freqs_hz: tuple):
    """FASE B: Verify near-Nyquist tones with dense phase grid & multiple amplitudes across 44.1/48/96 kHz.

    Compares the custom WLS 4x internal detector peak against external PCM_24 `calculate_true_peak_dbtp`
    (kept explicitly distinct from the custom WLS FIR specification) to prove zero dangerous ISP underestimation.
    """
    # 0.25 s covers multiple Nyquist-beating cycles (e.g., 2.5 beats at 23.99 kHz / 48 kHz)
    N = int(round(0.25 * sr))
    n = np.arange(N, dtype=np.float64)
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(256) / 256.0))
    win = np.ones(N, dtype=np.float64)
    win[:256] = ramp
    win[-256:] = ramp[::-1]

    phases = np.linspace(0.0, np.pi, 12, endpoint=False)
    cpp_lim = py_core_dsp.LookaheadLimiter(float(sr), "true_peak_4x")
    py_lim = DifferentiableLookaheadLimiter(sample_rate=float(sr), detector_mode="true_peak_4x")

    for freq_hz in freqs_hz:
        assert freq_hz < 0.5 * sr
        for amp in (1.0, 1.5, 3.0):
            # Test dense phases on C++, and subset of worst-case + orthogonal phases on Python for runtime parity
            for p_idx, phi0 in enumerate(phases):
                s = amp * np.sin(2.0 * np.pi * freq_hz * n / float(sr) + phi0) * win
                u_np = np.pad(np.stack([s, 0.90 * s], axis=0), ((0, 0), (0, 256)))

                cpp_lim.reset()
                y_cpp = cpp_lim.process(u_np.astype(np.float32)).astype(np.float64)
                det_cpp = _internal_detector_4x_peak_db(y_cpp)
                ext_cpp, sp_cpp, _ = _measure_pcm24_roundtrip(y_cpp, sr=sr)

                assert det_cpp <= INTERNAL_CEILING_DBTP + 0.015, (
                    f"Near-Nyquist internal detector breach at sr={sr}, f={freq_hz}, amp={amp}, phi={phi0:.3f}: {det_cpp:.4f}"
                )
                assert ext_cpp <= DELIVERY_CEILING_DBTP, (
                    f"Near-Nyquist PCM_24 external TP breach at sr={sr}, f={freq_hz}, amp={amp}, phi={phi0:.3f}: {ext_cpp:.4f}"
                )
                assert ext_cpp <= det_cpp + 0.06, (
                    f"Dangerous detector underestimation at sr={sr}, f={freq_hz}: det={det_cpp:.4f}, ext={ext_cpp:.4f}"
                )
                assert sp_cpp <= DELIVERY_CEILING_DBTP

                if p_idx % 4 == 0:
                    with torch.no_grad():
                        y_py = py_lim(torch.from_numpy(u_np).unsqueeze(0)).squeeze(0).numpy()
                    ext_py, sp_py, _ = _measure_pcm24_roundtrip(y_py, sr=sr)
                    det_py = _internal_detector_4x_peak_db(y_py)
                    assert det_py <= INTERNAL_CEILING_DBTP + 0.015
                    assert ext_py <= DELIVERY_CEILING_DBTP
                    assert sp_py <= DELIVERY_CEILING_DBTP


def test_t2b_hf_multitone_multiple_phases_across_sample_rates():
    """FASE B: High-frequency multitone test with varying phase constellations across 44.1 / 48 / 96 kHz."""
    phase_sets = (
        (0.0, 0.0, 0.0, 0.0),
        (0.15, 0.85, 1.70, 2.45),
        (0.5 * np.pi, 0.25 * np.pi, 1.1 * np.pi, 1.6 * np.pi),
        (1.2, 2.7, 0.4, 3.0),
    )
    sr_configs = (
        (44100, (16200.0, 18400.0, 20300.0, 21500.0)),
        (48000, (18000.0, 20000.0, 22000.0, 23500.0)),
        (96000, (36000.0, 40000.0, 44000.0, 47000.0)),
    )

    for sr, (f1, f2, f3, f4) in sr_configs:
        N = int(round(0.25 * sr))
        n = np.arange(N, dtype=np.float64)
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(256) / 256.0))
        win = np.ones(N, dtype=np.float64)
        win[:256] = ramp
        win[-256:] = ramp[::-1]

        cpp_lim = py_core_dsp.LookaheadLimiter(float(sr), "true_peak_4x")
        py_lim = DifferentiableLookaheadLimiter(sample_rate=float(sr), detector_mode="true_peak_4x")

        for amp in (0.6, 1.2):
            for p1, p2, p3, p4 in phase_sets:
                s = amp * (
                    np.sin(2.0 * np.pi * f1 * n / float(sr) + p1)
                    + np.sin(2.0 * np.pi * f2 * n / float(sr) + p2)
                    + np.sin(2.0 * np.pi * f3 * n / float(sr) + p3)
                    + np.sin(2.0 * np.pi * f4 * n / float(sr) + p4)
                ) * win
                u_np = np.pad(np.stack([s, -0.88 * s], axis=0), ((0, 0), (0, 256)))

                cpp_lim.reset()
                y_cpp = cpp_lim.process(u_np.astype(np.float32)).astype(np.float64)
                det_cpp = _internal_detector_4x_peak_db(y_cpp)
                ext_cpp, sp_cpp, _ = _measure_pcm24_roundtrip(y_cpp, sr=sr)

                assert det_cpp <= INTERNAL_CEILING_DBTP + 0.015
                assert ext_cpp <= DELIVERY_CEILING_DBTP
                assert ext_cpp <= det_cpp + 0.06
                assert sp_cpp <= DELIVERY_CEILING_DBTP

                if abs(amp - 1.2) < 1e-9 and p1 == 0.0:
                    with torch.no_grad():
                        y_py = py_lim(torch.from_numpy(u_np).unsqueeze(0)).squeeze(0).numpy()
                    ext_py, sp_py, _ = _measure_pcm24_roundtrip(y_py, sr=sr)
                    assert ext_py <= DELIVERY_CEILING_DBTP
                    assert sp_py <= DELIVERY_CEILING_DBTP


# ---------------------------------------------------------------------------
# T2. Transients, Impulses, Spacing (143/144/145) & Off-by-One Release Check
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("amp", [1.0, 2.0, 5.0, 10.0])
def test_t2_impulses_and_spacing_no_off_by_one_release_leak(amp: float):
    sr = 48000
    N = 2048
    u_np = np.zeros((2, N), dtype=np.float64)
    # Isolated impulse + pulse trains spaced 143, 144, 145 samples apart
    u_np[:, 200] = amp
    u_np[:, 600] = amp
    u_np[:, 600 + 143] = amp
    u_np[:, 1000] = amp
    u_np[:, 1000 + 144] = amp
    u_np[:, 1400] = amp
    u_np[:, 1400 + 145] = amp

    cpp_lim = py_core_dsp.LookaheadLimiter(float(sr), "true_peak_4x")
    y_cpp = cpp_lim.process(u_np.astype(np.float32)).astype(np.float64)
    tp_cpp, sp_cpp, _ = _measure_pcm24_roundtrip(y_cpp, sr=sr)

    c_internal_lin = 10.0 ** (INTERNAL_CEILING_DBTP / 20.0)
    assert np.max(np.abs(y_cpp)) <= c_internal_lin + 1e-5
    assert sp_cpp <= INTERNAL_CEILING_DBTP + 0.01
    assert tp_cpp <= INTERNAL_CEILING_DBTP + 0.01

    # Verify exact 144-sample audio alignment: impulse at 200 appears at 344
    peak_idx = int(np.argmax(np.abs(y_cpp[0, 300:400]))) + 300
    assert peak_idx == 200 + LOOKAHEAD_SAMPLES == 344

    # Bipolar sub-sample ISP step [+A, +A, -A, -A]
    u_bip = np.zeros((2, 1024), dtype=np.float64)
    u_bip[:, 300:302] = amp
    u_bip[:, 302:304] = -amp
    cpp_lim.reset()
    y_bip = cpp_lim.process(u_bip.astype(np.float32)).astype(np.float64)
    tp_bip, sp_bip, _ = _measure_pcm24_roundtrip(y_bip, sr=sr)
    assert tp_bip <= -0.30
    assert sp_bip <= -0.50


def test_t2_multitone_and_high_crested_transient():
    sr = 48000
    n = np.arange(8192, dtype=np.float64)
    # Sum of non-harmonically related tones generating dense multi-tone ISP peaks + sharp bursts
    sig = (
        0.55 * np.sin(2.0 * np.pi * 1100.0 * n / sr + 0.3)
        + 0.55 * np.sin(2.0 * np.pi * 6310.0 * n / sr + 1.1)
        + 0.55 * np.sin(2.0 * np.pi * 15430.0 * n / sr + 2.2)
    )
    sig[4000:4004] += np.array([2.5, 2.5, -2.5, -2.5])
    u = np.stack([sig, -sig], axis=0)
    cpp_lim = py_core_dsp.LookaheadLimiter(float(sr), "true_peak_4x")
    y = cpp_lim.process(u.astype(np.float32)).astype(np.float64)
    tp, sp, _ = _measure_pcm24_roundtrip(y, sr=sr)
    assert tp <= -0.30
    assert sp <= -0.50


# ---------------------------------------------------------------------------
# T3. Unbalanced Stereo Peaks & Channel-Link Integrity
# ---------------------------------------------------------------------------

def test_t3_unbalanced_stereo_linked_behavior():
    sr = 48000
    n = np.arange(4096, dtype=np.float64)
    s_hot = 2.0 * np.sin(2.0 * np.pi * 12000.0 * n / sr + 0.785398)
    s_quiet = 0.25 * np.sin(2.0 * np.pi * 12000.0 * n / sr + 0.785398)

    u_lr = np.stack([s_hot, s_quiet], axis=0)
    cpp_lim = py_core_dsp.LookaheadLimiter(float(sr), "true_peak_4x")
    y_lr = cpp_lim.process(u_lr.astype(np.float32)).astype(np.float64)

    # Swapping L and R must produce exact channel-swapped output (stereo symmetry)
    cpp_lim.reset()
    u_rl = np.stack([s_quiet, s_hot], axis=0)
    y_rl = cpp_lim.process(u_rl.astype(np.float32)).astype(np.float64)
    assert np.max(np.abs(y_lr[0] - y_rl[1])) < 1e-6
    assert np.max(np.abs(y_lr[1] - y_rl[0])) < 1e-6

    # L/R amplitude ratio 8.0 preserved where quiet != 0 -> stereo image never skewed by unlinked limiting
    mask = np.abs(y_lr[1, 500:3500]) > 1e-3
    ratio = np.abs(y_lr[0, 500:3500][mask]) / np.abs(y_lr[1, 500:3500][mask])
    assert np.allclose(ratio, 8.0, atol=1e-4)

    tp, _, _ = _measure_pcm24_roundtrip(y_lr, sr=sr)
    assert tp <= -0.30


# ---------------------------------------------------------------------------
# T4. Buffer Boundaries, Arbitrary Chunk Sizes, State Reset & Multiple Rates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sample_rate", [44100.0, 48000.0, 96000.0])
def test_t4_chunk_streaming_reset_and_short_buffers(sample_rate: float):
    rng = np.random.default_rng(42)
    N = 4096
    # Full-band white noise for strict streaming boundary & reset parity
    x_raw = (rng.standard_normal((2, N)) * 0.7).astype(np.float32)

    ref = py_core_dsp.MasteringChain(sample_rate, "true_peak_4x")
    theta = list(py_core_dsp.get_default_mastering_parameters_theta())
    theta[40] = 6.0  # +6 dB limiter push
    ref.set_parameters_theta(theta)
    ref.reset()
    y_full = ref.process(x_raw)

    # Streaming over non-uniform block sizes including edges < FIR kernel length and < lookahead
    stream = py_core_dsp.MasteringChain(sample_rate, "true_peak_4x")
    stream.set_parameters_theta(theta)
    stream.reset()
    chunks = []
    pos = 0
    block_pattern = [1, 3, 7, 13, 64, 127, 256, 512, 1024]
    idx = 0
    while pos < N:
        bs = min(block_pattern[idx % len(block_pattern)], N - pos)
        chunks.append(stream.process(np.ascontiguousarray(x_raw[:, pos : pos + bs])))
        pos += bs
        idx += 1
    y_stream = np.concatenate(chunks, axis=1)
    assert y_stream.shape == y_full.shape
    assert np.max(np.abs(y_stream - y_full)) < 1e-5

    # State reset determinism after hot excitation
    hot = np.ones((2, 1024), dtype=np.float32) * 3.0
    _ = stream.process(hot)
    stream.reset()
    y_after_reset = stream.process(x_raw)
    assert np.max(np.abs(y_after_reset - y_full)) == 0.0

    # Audio-band complex program signal (sum of musical partials up to 20 kHz + transient) with tail flush
    n = np.arange(N, dtype=np.float64)
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(128) / 128.0))
    win = np.ones(N, dtype=np.float64)
    win[:128] = ramp
    win[-128:] = ramp[::-1]
    prog_l = (
        0.55 * np.sin(2.0 * np.pi * 110.0 * n / sample_rate)
        + 0.45 * np.sin(2.0 * np.pi * 3420.0 * n / sample_rate + 0.4)
        + 0.45 * np.sin(2.0 * np.pi * 15800.0 * n / sample_rate + 1.1)
    ) * win
    prog_r = (
        0.55 * np.sin(2.0 * np.pi * 220.0 * n / sample_rate + 0.7)
        + 0.45 * np.sin(2.0 * np.pi * 6120.0 * n / sample_rate + 1.9)
        + 0.45 * np.sin(2.0 * np.pi * 19500.0 * n / sample_rate + 2.3)
    ) * win
    x_prog = np.stack([prog_l, prog_r], axis=0).astype(np.float32)
    x_prog_pad = np.pad(x_prog, ((0, 0), (0, 159)))
    ref.reset()
    y_prog = ref.process(x_prog_pad)[:, 159:]
    tp, sp, _ = _measure_pcm24_roundtrip(y_prog.astype(np.float64), sr=int(round(sample_rate)))
    assert tp <= -0.30
    assert sp <= -0.30

    # Short buffer (< 144 samples) edge handling in Python & C++ without crash or NaN
    short_in = x_raw[:, :37]
    ref.reset()
    y_short_cpp = ref.process(short_in)
    assert y_short_cpp.shape == (2, 37) and np.all(np.isfinite(y_short_cpp))


# ---------------------------------------------------------------------------
# T5. Silence & Low-Level Signal Unity Transparency
# ---------------------------------------------------------------------------

def test_t5_silence_and_low_level_signals():
    py_lim = DifferentiableLookaheadLimiter(48000.0, detector_mode="true_peak_4x")
    cpp_lim = py_core_dsp.LookaheadLimiter(48000.0, "true_peak_4x")

    # 1. Exact silence
    z_in = np.zeros((2, 2048), dtype=np.float32)
    z_cpp = cpp_lim.process(z_in)
    assert np.all(np.isfinite(z_cpp))
    assert float(np.max(np.abs(z_cpp))) == 0.0
    with torch.no_grad():
        z_py, diag = py_lim(torch.zeros((1, 2, 2048)), return_gain=True)
        assert float(z_py.abs().max().item()) == 0.0
        assert torch.allclose(diag["g_smooth"], torch.ones_like(diag["g_smooth"]))

    # 2. Ultra-low level (-60 dBFS & -120 dBFS): pure 144-sample delay, unity gain g == 1.0
    for amp in (1e-3, 1e-6):
        n = np.arange(2048, dtype=np.float64)
        s = amp * np.sin(2.0 * np.pi * 1000.0 * n / 48000.0)
        u = np.stack([s, 0.5 * s], axis=0)
        with torch.no_grad():
            y_py, diag = py_lim(torch.from_numpy(u).unsqueeze(0), return_gain=True)
        assert torch.allclose(diag["g_smooth"], torch.ones_like(diag["g_smooth"]), atol=1e-12)
        expected = np.pad(u, ((0, 0), (LOOKAHEAD_SAMPLES, 0)))[:, :2048]
        assert np.max(np.abs(y_py.squeeze(0).numpy() - expected)) < 1e-12


# ---------------------------------------------------------------------------
# T6. Cross-Engine Parity (Python / C++ / Chain / Autograd / Legacy Mode)
# ---------------------------------------------------------------------------

def test_t6_python_cpp_parity_finite_autograd_and_legacy_mode():
    sr = 48000.0
    rng = np.random.default_rng(7)
    x_np = (rng.standard_normal((2, 4800)) * 0.9).astype(np.float32)

    # 1. Limiter standalone C++ vs Python parity (< -105 dBFS)
    cpp_lim = py_core_dsp.LookaheadLimiter(sr, "true_peak_4x")
    cpp_lim.set_push_gain_db(6.0)
    y_cpp = cpp_lim.process(x_np).astype(np.float64)

    py_lim = DifferentiableLookaheadLimiter(sr, detector_mode="true_peak_4x")
    with torch.no_grad():
        y_py = py_lim(torch.from_numpy(x_np).unsqueeze(0), push_gain_db=torch.tensor([6.0])).squeeze(0).numpy()
    null_db = 20.0 * np.log10(np.max(np.abs(y_cpp - y_py)) + 1e-12)
    assert null_db <= -105.0, f"Limiter parity null {null_db:.2f} dB > -105 dB"

    # 2. Full chain C++ vs Python parity with realistic hot mastering parameters + offline renderer
    z_vec = np.full(41, 0.55, dtype=np.float64)
    z_vec[40] = 0.75  # higher limiter push
    theta = list(py_core_dsp.map_z_to_theta(z_vec.tolist()))

    chain_cpp = py_core_dsp.MasteringChain(sr, "true_peak_4x")
    chain_cpp.set_parameters_theta(theta)
    y_chain_cpp = chain_cpp.process(x_np).astype(np.float64)

    chain_py = DifferentiableMasteringChain(sr, default_eq_mode="iir", limiter_detector_mode="true_peak_4x")
    with torch.no_grad():
        y_chain_py = chain_py(torch.from_numpy(x_np).unsqueeze(0), theta=torch.tensor(theta).unsqueeze(0)).squeeze(0).numpy()
    chain_null_db = 20.0 * np.log10(np.max(np.abs(y_chain_cpp - y_chain_py)) + 1e-12)
    assert chain_null_db <= -100.0, f"Full chain parity null {chain_null_db:.2f} dB > -100 dB"

    # 3. End-to-end offline renderer + latency/duration/M-S preservation
    y_off, meta = render_offline_with_tail_flush(x_np.astype(np.float64), mode="theta", theta=theta)
    assert y_off.shape == x_np.shape
    assert meta["declared_latency_samples"] == 159
    assert meta["limiter_detector_mode"] == "true_peak_4x"
    assert abs(meta["limiter_ceiling_db"] - INTERNAL_CEILING_DBTP) < 1e-9
    tp_off, sp_off, _ = _measure_pcm24_roundtrip(y_off, sr=int(sr))
    assert tp_off <= -0.30
    assert sp_off <= -0.30

    # 4. Autograd differentiability & finite gradients when TP limiter actively limits a dynamic + ISP signal
    # Note: a mixed quiet body + limited HF 16 kHz ISP burst ensures push_gain_db affects non-limited
    # body energy and release transitions as well as limited peaks (on a constant clamped sine u*(c/u)=c).
    n = torch.arange(2048, dtype=torch.float32)
    env = 0.25 * torch.ones(2048, dtype=torch.float32)
    env[900:1200] = 1.6
    isp_in = (env * torch.sin(2.0 * math.pi * 16000.0 * n / sr + 0.5)).unsqueeze(0).repeat(2, 1).unsqueeze(0)
    z_param = torch.tensor(z_vec, dtype=torch.float32, requires_grad=True)
    y_grad = chain_py(isp_in, z=z_param.unsqueeze(0))
    loss = (y_grad ** 2).mean()
    loss.backward()
    assert z_param.grad is not None
    assert torch.all(torch.isfinite(z_param.grad))
    assert float(z_param.grad[40].abs().item()) > 1e-5  # push gain receives strong finite gradient
    assert float(z_param.grad.abs().max().item()) > 1e-5

    # 5. Explicit legacy_sample_1x reproducibility path (boxcar=144, ceiling=-0.30 dBFS)
    cpp_legacy = py_core_dsp.LookaheadLimiter(sr, "legacy_sample_1x")
    py_legacy = DifferentiableLookaheadLimiter(sr, detector_mode="legacy_sample_1x")
    assert cpp_legacy.get_boxcar_samples() == 144
    assert math.isclose(cpp_legacy.get_ceiling_db(), -0.30, abs_tol=1e-12)
    imp = np.zeros((2, 1024), dtype=np.float32)
    imp[:, 200] = 2.0
    y_leg_cpp = cpp_legacy.process(imp).astype(np.float64)
    with torch.no_grad():
        y_leg_py = py_legacy(torch.from_numpy(imp).unsqueeze(0)).squeeze(0).numpy()
    assert np.max(np.abs(y_leg_cpp - y_leg_py)) < 1e-6
    legacy_c_lin = 10.0 ** (-0.30 / 20.0)
    assert math.isclose(float(np.max(np.abs(y_leg_cpp))), legacy_c_lin, abs_tol=1e-5)


# ---------------------------------------------------------------------------
# T7. Offline Export PCM_24 Safety Guard
# ---------------------------------------------------------------------------

def test_t7_offline_export_pcm24_safety_guard(tmp_path: Path):
    sr = 48000
    n = np.arange(4800, dtype=np.float64)
    # 1. Pass case: TP ~ -0.50 dBTP
    sig_ok = (10.0 ** (-0.50 / 20.0)) * np.sin(2.0 * np.pi * 1000.0 * n / sr)
    ok_wav = tmp_path / "ok.wav"
    res_ok = export_validated_pcm24_wav(ok_wav, np.stack([sig_ok, sig_ok]), sample_rate=sr)
    assert res_ok["status"] == "pass"
    assert res_ok["emergency_trim_applied"] is False
    assert res_ok["trim_applied_db"] == 0.0
    assert res_ok["post_trim_true_peak_dbtp"] <= -0.30
    assert res_ok["subtype"] == "PCM_24"

    # 2. Micro-trim fallback case: overshoot <= 0.05 dB (-0.27 dBTP -> needs ~0.032 dB trim)
    sig_micro = (10.0 ** (-0.27 / 20.0)) * np.sin(2.0 * np.pi * 1000.0 * n / sr)
    micro_wav = tmp_path / "micro.wav"
    res_micro = export_validated_pcm24_wav(micro_wav, np.stack([sig_micro, sig_micro]), sample_rate=sr)
    assert res_micro["status"] == "pass_with_micro_trim"
    assert res_micro["emergency_trim_applied"] is True
    assert -0.05 <= res_micro["trim_applied_db"] < 0.0
    assert res_micro["post_trim_true_peak_dbtp"] <= -0.30
    disk_raw, _ = sf.read(str(micro_wav), dtype="float32", always_2d=True)
    assert calculate_true_peak_dbtp(disk_raw.T, sample_rate=float(sr)) <= -0.30

    # 3. Loud fail guard: overshoot > 0.05 dB (-0.10 dBTP) must raise RuntimeError and remove unsafe file
    sig_bad = (10.0 ** (-0.10 / 20.0)) * np.sin(2.0 * np.pi * 1000.0 * n / sr)
    bad_wav = tmp_path / "bad.wav"
    with pytest.raises(RuntimeError, match="safety guard FAILED"):
        export_validated_pcm24_wav(bad_wav, np.stack([sig_bad, sig_bad]), sample_rate=sr)
    assert not bad_wav.exists()
