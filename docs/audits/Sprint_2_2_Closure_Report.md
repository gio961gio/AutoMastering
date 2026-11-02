# POLITECNICO DI MILANO
### Dipartimento di Elettronica, Informazione e Bioingegneria (DEIB)
**Master's Thesis in Sound and Music Engineering**  
**Candidate:** Giovanni Antonecchia  
**Project:** DDSP Audio Mastering — Differentiable White-Box Audio Mastering Suite  
**Milestone:** Sprint 2.2 — Independent Technical Audit & Final Gate Remediation Closure Report  
**Date:** November 2, 2026  
**Baseline Pre-Remediation Commit:** `874550c5a72245a849b69c91fd8493e546662954`  
**Initial Sprint 2.2 Code Verification Commit:** `0e232045033cb2a46fc3a358348a0af3721518ac` (CI Run [#41430543176](https://github.com/gio961gio/AutoMastering/actions/runs/41430543176) — **SUCCESS**)  
**Previous Documentation Commit:** `5d9646d8106716e313221e47cb7dcdd506a74b65` (CI Run [#41431602818](https://github.com/gio961gio/AutoMastering/actions/runs/41431602818) — **SUCCESS**)  
**Final Gate Code Verification Commit (G2–G4):** `5940e81f12741e2276f7a446adfb6e12f44574fe` (CI Run [#41435629557](https://github.com/gio961gio/AutoMastering/actions/runs/41435629557) — **SUCCESS**)  
**Status:** **GO — Sprint 2.2 Final Gate CLOSED** (128/128 Python Tests + 12/12 C++ JUCE Headless Test Stages Passed)

---

## 1. Executive Summary & Git Provenance

This closure report documents the engineering remediation and formal verification conducted for **Sprint 2.2**, addressing the **Independent Technical Audit** performed by Argon (`docs/argon_feedback/argon_independent_technical_audit.md`) on baseline commit `874550c5a72245a849b69c91fd8493e546662954` (97/97 passing tests), followed by the **Sprint 2.2 Final Gate Remediation** (tasks **G1, G2, G3, G4**).

### Exact Git & CI Provenance Table

To guarantee strict audit traceability, code verification commits and documentation commits are tracked separately alongside their exact GitHub Actions `head_sha`:

| Role | Exact 40-Character Commit SHA | GitHub Actions Run | CI `head_sha` Match | CI Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline Pre-Sprint 2.2** | `874550c5a72245a849b69c91fd8493e546662954` | — | — | 97/97 tests pass |
| **Sprint 2.2 Initial Code Commit** | `0e232045033cb2a46fc3a358348a0af3721518ac` | [#41430543176](https://github.com/gio961gio/AutoMastering/actions/runs/41430543176) | `0e232045033cb2a46fc3a358348a0af3721518ac` | **SUCCESS** (117 Pytest + 11 C++) |
| **Sprint 2.2 Initial Doc Commit** | `5d9646d8106716e313221e47cb7dcdd506a74b65` | [#41431602818](https://github.com/gio961gio/AutoMastering/actions/runs/41431602818) | `5d9646d8106716e313221e47cb7dcdd506a74b65` | **SUCCESS** (117 Pytest + 11 C++) |
| **Final Gate Code Commit (G2–G4)** | `5940e81f12741e2276f7a446adfb6e12f44574fe` | [#41435629557](https://github.com/gio961gio/AutoMastering/actions/runs/41435629557) | `5940e81f12741e2276f7a446adfb6e12f44574fe` | **SUCCESS** (128 Pytest + 12 C++) |

---

## 2. Comprehensive Audit Finding Matrix (F-01 to F-12 & G1 to G4)

### 2.1 Argon Audit Findings (F-01 to F-12)

| ID | Category | Severity | Argon Finding | Implemented Resolution | Commit SHA | Verification Tests |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **F-01** | B1 | **Blocking** | K-weighting DC bin evaluation, circular FFT wrap-around, and missing dual-gated BS.1770-5 integrated loudness. | `rfftfreq` exact frequency response; causal time-domain biquad cascade in `float64` (`taf.lfilter`); dual-gated ITU-R BS.1770-5 integrated loudness (`400 ms` blocks, `75%` overlap, `-70 LKFS` absolute gate, `-10 LU` relative gate) in `loudness.py` and `metrics.py`; smooth ungated K-weighted proxy preserved for autograd loss. | `5d3726f37fb62a264d218f8fb33349f29cc6fb1f` | `test_b1_f01_*` (6 tests) |
| **F-02** | B2 | **Blocking** | Limiter is sample-peak lookahead; benchmark lacked 4x oversampled True Peak (`dBTP`) detection. | Implemented 4x polyphase Kaiser-windowed sinc FIR True Peak meter (`calculate_true_peak_dbtp`) alongside sample peak (`calculate_sample_peak_dbfs`) in `metrics.py`; preserved Sample-Peak Lookahead Limiter (`-0.3 dBFS` ceiling). | `e33f77b21bb46b109c7e2c79d6fc6c4490315241` | `test_b2_f02_*` (3 tests) & `test_g3_*` (9 tests) |
| **F-03** | B4 | **Blocking** | JUCE plugin lacked APVTS host bypass integration and validated 41-parameter JSON preset round-trip. | Added `"bypass"` `AudioParameterBool` to APVTS; overrode `getBypassParameter()`; updated `MasteringChain` bypass delay line continuously; implemented `AutoMastering_Preset_v1` JSON export/import. | `ba746784543fa013fb4947565399a1934849f6a1` | `test_b4_f03_*`, `test_g2_*`, `plugin_headless_tests` (Stages 1–12) |
| **F-04** | B3 | **Blocking** | `MultiChunkSelector` displacement could shift candidate chunks into silent breakdowns or produce duplicates on fallback. | Re-validated RMS energy (`rms_db >= min_rms_db`) on displaced candidates; tracked `accepted_indices` to prevent duplicates; emitted `UserWarning` when track duration/energy cannot fill `num_chunks`. | `7c6976f57618602c1b0637c132d2fa5725517222` | `test_b3_*` (4 tests) |
| **F-05** | — | Non-blocking | `PluginProcessor::processBlock` unpacked and recomputed DSP filter coefficients on every sub-block even when static. | Guarded `chain_.set_parameters_theta()` behind `params_changed` check (`isSmoothing()` or target change), tracking updates via `theta_update_count_`. | `35559a4a957cd3cf0e9133a7a939a4dc51691358` | `plugin_headless_tests` |
| **F-06** | — | Non-blocking | PyBind11 calls held the Python GIL; PyTorch CPU compressor/limiter fell back to Python loops for batch $B > 1$. | Added `py::gil_scoped_release` to `PyMasteringChain::process` and scan bindings; enabled C++ scan loop across batch items $B \ge 1$ on CPU in `compressor.py` and `limiter.py` (forward pass). | `35559a4a957cd3cf0e9133a7a939a4dc51691358` | `test_f06_compressor_and_limiter_cpu_batching_parity` |
| **F-07** | — | Non-blocking | `estimate_time_alignment` normalized cross-correlation over full signal energy rather than overlapping slices. | Computed normalization denominator $\sqrt{E_{\text{mix, overlap}} \cdot E_{\text{ref, overlap}}}$ strictly on the overlapping slice at `best_lag`. | `814674b1a0c0292684eb1cd326d2729852002581` | `test_f07_cross_correlation_overlapping_slice_normalization` |
| **F-08** | — | Non-blocking | Stylistic Mel energy used coherent mid sum $(L+R)/2$, cancelling anti-phase stereo content. | Replaced coherent sum with incoherent stereo power average $0.5 \cdot (\|S_L\|^2 + \|S_R\|^2)$ in `stylistic.py` and `metrics.py`. | `814674b1a0c0292684eb1cd326d2729852002581` | `test_f08_mel_stereo_energy_preserves_side_and_antiphase` |
| **F-09** | — | Non-blocking | $48\text{ kHz}$ sample rate contract was implicit in UI and documentation. | Updated `PluginEditor.cpp` to display dynamic host sample rate and latency; documented the $48\text{ kHz}$ contract in `README.md`. | `814674b1a0c0292684eb1cd326d2729852002581` | `test_f09_sample_rate_contract_characterization` |
| **F-10** | — | Non-blocking | Unused helper methods in `metrics.py`. | Removed dead code from `python/ddsp_mastering/optimize/metrics.py`. | `7e9c6ac18eb59318b316e3a269b680bebac00106` | Full pytest suite |
| **F-11** | — | Non-blocking | `MasteringParameters` default struct values in `ParameterSpec.hpp` differed from Python `THETA_NEUTRAL`. | Defined `core_dsp::THETA_NEUTRAL`, aligned `MasteringParameters` default member initializers (EQ `f0`, crossover `160/3200 Hz`, comp `ratio = 1.0`, `threshold = -12.0 dB`), and added `pack_theta()`. | `7e9c6ac18eb59318b316e3a269b680bebac00106` | `test_f11_cpp_theta_neutral_matches_python_and_default_struct` |
| **F-12** | — | Non-blocking | GitHub Actions CI only ran Python tests without compiling or testing the C++ JUCE plugin. | Updated `.github/workflows/ci.yml` to clone JUCE 8.0.6, build `plugin_headless_tests`, and run the native C++ headless test binary before pytest. | `5ec74668dea5a1e001815bb535768ab1470ff9e4`, `0e232045033cb2a46fc3a358348a0af3721518ac` | GitHub Actions Runs `#41430543176`, `#41435629557` |

### 2.2 Sprint 2.2 Final Gate Tasks (G1 to G4)

| Gate ID | Task Description | Files Modified | Commit SHA | Verification Tests | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **G1** | Correct closure report SHAs, canonical parameter indices (`38..40`), and 6-stage DSP chain description. | `docs/audits/Sprint_2_2_Closure_Report.md` | Current doc commit | Manual & automated cross-check against `ParameterSpec.hpp`, `MasteringChain.hpp`, `HarmonicSaturator.hpp` | **CLOSED** |
| **G2** | Authentic Python preset fixture (`export_preset_json`) loaded by JUCE parser (`Mode A` with `normalized_z`, `Mode B` with `parameters` only), schema `version == "1.0"` enforcement, and atomic rejection on error. | `cpp/plugin/source/PluginProcessor.cpp`, `cpp/plugin/tests/test_plugin.cpp`, `python/tests/fixtures/python_exported_preset_v1.json`, `python/tests/test_sprint_2_2_audit_remediation.py` | `c9e2064`, `5940e81` | `plugin_headless_tests` `[TEST 12]`, `test_g2_python_exported_fixture_provenance_and_cpp_parity` | **CLOSED** |
| **G3** | Strengthen True Peak meter validation: explicit zero-padding boundary handling (preserving edge peaks at $n=0$ and $n=T-1$ and short buffers), 8-frequency $\times$ 16-phase sweep vs independent $64\times$ `scipy.signal.upfirdn` sinc reference. | `python/ddsp_mastering/optimize/metrics.py`, `python/tests/test_sprint_2_2_audit_remediation.py` | `1ab2211`, `5940e81` | `test_g3_true_peak_frequency_and_phase_sweep_vs_independent_64x_reference` (8 parameterized tests), `test_g3_true_peak_boundary_samples_impulses_and_short_buffers` | **CLOSED** |
| **G4** | Explicit per-signal loudness method tracking (`pred_loudness_method`, `target_loudness_method`) and accurate `loudness_standard` labeling when short-audio ($<400\text{ ms}$) ungated proxy fallback is triggered. | `python/ddsp_mastering/optimize/metrics.py`, `scripts/run_mastering.py`, `python/tests/test_sprint_2_2_audit_remediation.py` | `1ab2211`, `5940e81` | `test_g4_loudness_fallback_method_labeling_and_400ms_boundaries` | **CLOSED** |

---

## 3. Technical Deep-Dive: Blocking Requirements & Final Gate Remediations

### 3.1 B1 / F-01 & G4 — ITU-R BS.1770-5 Integrated Loudness, Causal K-Weighting & Fallback Labeling

#### Mathematical & Algorithmic Implementation
1. **Causal Time-Domain K-Weighting (`DifferentiableKWeighting`)**:
   - Stage 1 (high-shelf acoustic head pre-filter) and Stage 2 (RLB high-pass filter) are applied in series in the time domain via `torchaudio.functional.lfilter` in `float64` precision.
   - Eliminates circular FFT convolution wrap-around on both even and odd signal lengths ($T = 48,001$ verified against `scipy.signal.lfilter` with maximum absolute error $< 1.5 \times 10^{-12}$).
   - `compute_frequency_response` uses exact `torch.fft.rfftfreq(n_fft, d=1.0/sample_rate)` with zero DC leakage.
2. **Dual-Gated Integrated Loudness (`calculate_integrated_loudness_bs1770`)**:
   - Segments K-weighted audio into $400\text{ ms}$ rectangular gating blocks ($19,200$ samples at $48\text{ kHz}$) with $75\%$ overlap ($100\text{ ms}$ / $4,800$-sample hop).
   - **Absolute gate**: Excludes blocks with $l_j < -70.0\text{ LKFS}$ to compute preliminary loudness $\Gamma$.
   - **Relative gate**: Excludes blocks with $l_j < \Gamma - 10.0\text{ LU}$ and averages the surviving block energies:
     $$\text{LUFS}_{\text{integrated}} = -0.691 + 10 \log_{10} \left( \frac{1}{|J_{\text{rel}}|} \sum_{j \in J_{\text{rel}}} \sum_{i \in \{L, R\}} z_{i, j} \right)$$
   - Audio shorter than $400\text{ ms}$ raises a `ValueError` in `calculate_integrated_loudness_bs1770`.
3. **Explicit Fallback Method Labeling in `MasteringBenchmarkSuite.evaluate()` (G4)**:
   - Because `pred_audio` and `target_audio` may have independent durations (especially in stylistic mode or short test snippets), `evaluate()` tracks the loudness calculation method separately for each signal:
     - `pred_loudness_method`: `"itu_bs1770_5_integrated"` (when $T_{\text{pred}} \ge 400\text{ ms}$) or `"ungated_k_weighted_proxy_short_audio"` (when $T_{\text{pred}} < 400\text{ ms}$).
     - `target_loudness_method`: `"itu_bs1770_5_integrated"` (when $T_{\text{target}} \ge 400\text{ ms}$) or `"ungated_k_weighted_proxy_short_audio"` (when $T_{\text{target}} < 400\text{ ms}$).
   - The summary key `loudness_standard` is set strictly according to the actual methods used:
     - `"ITU-R BS.1770-5 (11/2023) Integrated"` **only** when both `pred` and `target` use `"itu_bs1770_5_integrated"`;
     - `"Ungated K-Weighted Proxy (Short Audio < 400ms Fallback)"` when both signals use the short-audio proxy;
     - `"Mixed (pred: <method>, target: <method>)"` when one signal is $\ge 400\text{ ms}$ and the other is $< 400\text{ ms}$.
   - Legacy keys `pred_lufs` and `target_lufs` are retained for backward compatibility, with their semantic origin explicitly identified by `pred_loudness_method` and `target_loudness_method` in both CLI output (`scripts/run_mastering.py`) and JSON session manifests.

#### Measured Test Results
- **Calibration Tone**: $1\text{ kHz}$ sine at $-23.0\text{ dBFS}$ RMS measures $-22.993\text{ LUFS}$ ($|\Delta| < 0.007\text{ LU}$).
- **Reference Comparison (`pyloudnorm`)**: Matches `pyloudnorm.Meter(48000)` within $0.041\text{ LU}$ (tolerance $< 0.05\text{ LU}$).
- **400 ms Boundary Suite (`test_g4_loudness_fallback_method_labeling_and_400ms_boundaries`)**: Verified at $19,199$ samples ($399.98\text{ ms}$), $19,200$ samples ($400.00\text{ ms}$), $19,201$ samples ($400.02\text{ ms}$), and mixed short/long pairs.

---

### 3.2 B2 / F-02 & G3 — 4x True Peak Meter Validation vs. Sample-Peak Lookahead Limiter

#### Architectural Separation
- The C++20/PyTorch limiter (`LookaheadLimiter`) remains a **Sample-Peak Lookahead Limiter** with a fixed ceiling of `-0.3 dBFS` (`LIMITER_CEILING_DB = -0.3` in `Constants.hpp`). No hard clipper or post-limiter clipping stage was added.
- True Peak (`dBTP`) is measured independently in the evaluation benchmark (`calculate_true_peak_dbtp` in `metrics.py`).

#### True Peak Algorithm, Boundary Condition & Independent Reference (G3)
1. **Algorithm under test (`calculate_true_peak_dbtp`)**:
   - Converts input to `float64` and applies **explicit zero-padding** (`F.pad(x, (64, 64), mode="constant", value=0.0)`) of $64$ samples on both ends before $4\times$ polyphase Kaiser-windowed sinc interpolation (`lowpass_filter_width=64`, `rolloff=1.0`, $\beta = 14.769656459379492$).
   - Because zero-padding is applied externally and no samples of the signal support $[0, T-1]$ are trimmed away:
     - Sub-phase 0 preserves all original discrete samples ($h[0]=1$, $h[4k]=0$ for $k \neq 0$), guaranteeing $\text{True Peak} \ge \text{Sample Peak}$;
     - Peaks at the very first sample ($n=0$), very last sample ($n=T-1$), near boundaries, and on short buffers ($T \in \{1, 4, 8, 16, 32, 64\}$) are captured without attenuation;
     - Inter-sample overshoots occurring in the filter reconstruction tails immediately before $n=0$ or after $n=T-1$ are preserved.
2. **Independent $64\times$ Reference (`_independent_true_peak_reference_64x_dbtp`)**:
   - Implemented via `scipy.signal.upfirdn` at oversampling factor $L = 64$ using an explicit $8,193$-tap ($2 \times 64 \times 64 + 1$) Kaiser-windowed sinc FIR kernel ($\beta = 14.769656459379492$) with zero-padded boundaries.
   - Shares zero code with `torchaudio.functional.resample`.
3. **Theoretical Tolerance Justification (4x vs. 64x / Continuous Time)**:
   - On a $4\times$ oversampled grid (effective sampling rate $4 f_s$), a sinusoid of frequency $f$ can have its continuous-time peak fall at most half a $4\times$ sample period ($\Delta t = \frac{1}{8 f_s}$) away from the nearest $4\times$ grid point.
   - Consequently, a mathematical $4\times$ interpolator exhibits a theoretical maximum cosine sampling droop relative to continuous time ($64\times$ reference) of:
     $$\Delta_{\text{droop}, 4\times}(f) = -20 \log_{10} \left( \cos\left( \frac{\pi f}{4 f_s} \right) \right)$$
   - At $f_s = 48\text{ kHz}$, this theoretical bound equals $0.0001\text{ dB}$ at $1\text{ kHz}$, $0.015\text{ dB}$ at $3.5\text{ kHz}$, $0.169\text{ dB}$ at $12\text{ kHz}$ ($f_s/4$, matching ITU-R BS.1770-5 Annex 2 Table 3), $0.301\text{ dB}$ at $16\text{ kHz}$, and $0.473\text{ dB}$ at $20\text{ kHz}$.
   - In `test_g3_true_peak_frequency_and_phase_sweep_vs_independent_64x_reference`, across 8 frequencies $\in [100\text{ Hz}, 20\text{ kHz}]$ and 16 uniform initial phases $\phi_k = 2\pi k / 16$, every measurement satisfies $|\text{TP}_{4\times} - \text{TP}_{64\times}| \le \Delta_{\text{droop}, 4\times}(f) + 0.025\text{ dB}$.

---

### 3.3 B3 / F-04 — MultiChunkSelector Separation & Minimum Energy Re-Validation

- In `python/ddsp_mastering/data/chunking.py`, whenever a candidate chunk start index is displaced by $\pm \Delta T_{\text{sep}}$ to satisfy `min_separation_sec`, its local RMS energy is recomputed and checked against `min_rms_db` (`rms_db >= min_rms_db`).
- Displaced candidates falling into breakdowns, silent tails, or fade-outs below `min_rms_db` are rejected.
- `accepted_indices` prevents duplicate fallback chunks, and a `UserWarning` is emitted whenever usable high-energy duration is insufficient to provide `num_chunks` separated windows.

---

### 3.4 B4 / F-03 & G2 — JUCE VST3 True Bypass & End-to-End Python → JUCE Preset Interoperability

1. **True Bypass & Latency Invariance**:
   - `MasteringPluginAudioProcessor` exposes a host-automatable `"bypass"` parameter in APVTS and returns it via `getBypassParameter()`.
   - `MasteringChain::process()` continuously updates its 159-sample circular bypass delay lines (`bypass_delay_l_`, `bypass_delay_r_`) on every audio block, ensuring sample-accurate 159-sample latency in both active and bypassed states (`max_diff = 0.0` post-latency in headless tests).
2. **End-to-End Python Fixture → JUCE Parser Verification (G2)**:
   - **Authentic Fixture (`python/tests/fixtures/python_exported_preset_v1.json`)**: Generated directly by `AutoMasteringOptimizer(sample_rate=48000.0).export_preset_json(z)` using 41 distinct interior values $z_i \in [0.13, 0.88]$ ($z_i \neq 0.5$ for all $i$). Provenance is enforced by `test_g2_python_exported_fixture_provenance_and_cpp_parity`, which regenerates the JSON from `AutoMasteringOptimizer.export_preset_json(z)` and asserts exact string equality with the fixture file on disk.
   - **Mode A (Full preset with `normalized_z` and `parameters`) & Mode B (`normalized_z` removed, `parameters` map only)**:
     - In `[TEST 12]` of `cpp/plugin/tests/test_plugin.cpp`, the JUCE headless test loads the Python fixture in both Mode A and Mode B and verifies all 41 APVTS physical parameters (`|actual_theta[i] - expected_theta[i]| <= max(1e-3, 1e-4 * |expected_theta[i]|)`) and all 41 normalized parameters (`|actual_z[i] - expected_z[i]| <= 1e-4`).
   - **Schema Versioning & Atomic Rejection (G2.4)**:
     - `MasteringPluginAudioProcessor::loadParametersFromJson()` validates `format == "AutoMastering_Preset_v1"`, `version == "1.0"`, `sample_rate == 48000.0`, and all 41 parameter bounds/types before touching APVTS.
     - Tested and verified that unsupported versions (`"2.0"`, missing `"version"`), out-of-range values (`-999.0`, `z = 1.25`), non-numeric strings (`"NaN"`), and inconsistent `normalized_z` vs `parameters` are rejected (`return false`) with **zero partial mutation** of the 41 APVTS parameters.

---

## 4. Non-Blocking Audit Remediation Details (F-05 to F-12) & Canonical Parameter Specification (G1.2)

### Canonical 41-Parameter Vector (`ParameterSpec.hpp` & `param_space.py`)

In accordance with `cpp/core_dsp/include/core_dsp/ParameterSpec.hpp`, the canonical parameter vector $\theta \in \mathbb{R}^{41}$ has the exact layout:

| Index Range | Stage | Parameter Names | Physical Bounds & Scale | Neutral Value (`THETA_NEUTRAL`) |
| :--- | :--- | :--- | :--- | :--- |
| `0..17` | **6-Band Parametric EQ** | `eq_b{0..5}_f0`, `eq_b{0..5}_gain_db`, `eq_b{0..5}_q` | $f_0$: Exp (Hz), Gain: $[-6, +6]\text{ dB}$ Lin, $Q$: $[0.4, 2.5]$ Exp | $f_0 = \{60, 250, 1000, 3000, 8000, 12000\}\text{ Hz}$, Gain $= 0.0\text{ dB}$, $Q = 0.707$ |
| `18..19` | **3-Way LR4 Crossover** | `crossover_f_low`, `crossover_f_high` | $[80, 350]\text{ Hz}$ Exp, $[1500, 6000]\text{ Hz}$ Exp | $160.0\text{ Hz}$, $3200.0\text{ Hz}$ |
| `20..34` | **3-Band Compressor** | `comp_{low,mid,high}_{threshold_db, ratio, attack_ms, release_ms, makeup_db}` | Thresh: $[-30, 0]\text{ dB}$, Ratio: $[1, 4]$ Exp, Att/Rel: Exp (ms), Makeup: $[-3, +6]\text{ dB}$ | Thresh $= -12.0\text{ dB}$, Ratio $= 1.0$, Att $= \{40, 25, 15\}\text{ ms}$, Rel $= \{120, 100, 80\}\text{ ms}$, Makeup $= 0.0\text{ dB}$ |
| `35..37` | **3-Band Mid/Side Imager** | `imager_width_low`, `imager_width_mid`, `imager_width_high` | Low: $[0.4, 1.1]$, Mid/High: $[0.7, 1.4]$ Lin | $1.0, 1.0, 1.0$ |
| `38` | **Harmonic Saturator** | `saturator_drive_db` | $[0.0, 6.0]\text{ dB}$ Linear | $0.0\text{ dB}$ |
| `39` | **Harmonic Saturator** | `saturator_mix` | $[0.0, 1.0]$ Linear | $0.0$ ($100\%$ dry) |
| `40` | **Lookahead Limiter** | `limiter_push_gain_db` | $[0.0, 8.0]\text{ dB}$ Linear | $0.0\text{ dB}$ |

> **Correction Note (G1.2 — F-11)**: Index `38` is `saturator_drive_db`, index `39` is `saturator_mix`, and index `40` is `limiter_push_gain_db`. The limiter sample-peak ceiling (`-0.3 dBFS`, `LIMITER_CEILING_DB` in `Constants.hpp`) is a fixed configuration constant of `LookaheadLimiter`, not an element of $\theta$. Finding **F-11** concerned the discrepancy between the default member initializers of the C++ struct `core_dsp::MasteringParameters` (which previously defaulted to `ratio = 2.0`, `threshold_db = -20.0`, `crossover = 150 / 3000 Hz`, and `eq` bands at `1000 Hz`) and Python's `THETA_NEUTRAL`. Commit `7e9c6ac18eb59318b316e3a269b680bebac00106` defined `core_dsp::THETA_NEUTRAL`, aligned `MasteringParameters` default member initializers with `THETA_NEUTRAL`, and added `pack_theta()`, verified by `test_f11_cpp_theta_neutral_matches_python_and_default_struct`.

---

## 5. Core DSP C++20 Architecture & Invariance Statement (G1.3)

In accordance with `cpp/core_dsp/include/core_dsp/MasteringChain.hpp` and its component headers, the 6-stage signal processing chain is:

1. **6-Band Parametric EQ (`BiquadCascade6Band`, `BiquadCascade.hpp`)**: Stereo cascade of 6 Transposed Direct Form II biquad filters (Band 0 Low-Shelf, Bands 1–4 Peaking, Band 5 High-Shelf) controlled by `theta[0..17]`.
2. **3-Way Linkwitz-Riley 4th-Order Crossover (`LinkwitzRiley4_3Way`, `LinkwitzRiley4.hpp`)**: Splits the equalized stereo signal into phase-coherent Low, Mid, and High frequency bands controlled by `theta[18..19]`.
3. **3-Band Parallel Multiband Compressor (`MultibandCompressor3Band`, `MultibandComp.hpp`)**: Processes the Low, Mid, and High stereo bands in parallel with decoupled attack/release ballistic smoothing and soft-knee log-domain gain reduction controlled by `theta[20..34]`.
4. **3-Band Mid/Side Stereo Imager & Band Summing (`MidSideImager::process_3band_and_sum`, `MidSideImager.hpp`)**: Applies per-band Mid/Side stereo width factors (`theta[35..37]`) and sums the three frequency bands back into a single full-bandwidth stereo stream.
5. **Harmonic Saturator (`HarmonicSaturator`, `HarmonicSaturator.hpp`)**: 2x polyphase halfband FIR oversampled stereo saturator applying the small-signal unit-gain normalized hyperbolic tangent nonlinearity $v = \tanh(g_d \cdot u) / g_d$ (`theta[38]` drive in dB, `theta[39]` dry/wet mix), with a 15-sample dry delay line for phase-aligned dry/wet blending.
6. **Sample-Peak Lookahead Brickwall Limiter (`LookaheadLimiter`, `LookaheadLimiter.hpp`)**: 144-sample lookahead buffer ($3.0\text{ ms}$ at $48\text{ kHz}$), two-stage peak hold and release smoother, fixed `-0.3 dBFS` sample-peak ceiling (`LIMITER_CEILING_DB`), driven by `limiter_push_gain_db` (`theta[40]`).

### Scope of Core Invariance
- **Total Chain Latency**: Exactly $159\text{ samples}$ ($15\text{ samples}$ from `HarmonicSaturator` polyphase FIR + $144\text{ samples}$ from `LookaheadLimiter` lookahead $\equiv 3.3125\text{ ms}$ at $48\text{ kHz}$).
- **DSP Stage Invariance**: The mathematical topologies and algorithms of all six DSP stages (`BiquadCascade.hpp`, `LinkwitzRiley4.hpp`, `MultibandComp.hpp`, `MidSideImager.hpp`, `HarmonicSaturator.hpp`, `LookaheadLimiter.hpp`) were preserved without modification.
- **Bypass Delay Line Update**: `MasteringChain.hpp` was modified solely in `process()` (`commit ba74678`) so that the 159-sample `bypass_delay_l_` and `bypass_delay_r_` ring buffers are continuously fed during normal processing, preventing stale buffer discontinuities when toggling True Bypass on or off.

---

## 6. Full Regression Test Matrix

Executed on Windows 11 x64 (Python 3.11.5, PyTorch, MSVC v143 C++20, JUCE 8.0.6) and replicated on GitHub Actions `windows-latest`:

### 6.1 Python Test Suite (`pytest python/tests/ -v`)
- **Total Test Files**: 12 files
- **Total Collected & Passed**: **128 / 128 tests (100% PASS, 0 failures)**

| Test Suite File | Passed | Scope Covered |
| :--- | :--- | :--- |
| `test_cpp_parity.py` | 4 / 4 | C++ constants, FIR halfband coefficients, `z <-> theta` bit-parity |
| `test_param_space.py` | 5 / 5 | 41-parameter bijective space, boundary mapping, autograd flow |
| `test_sprint_1_correctness.py` | 12 / 12 | Gate A baseline: neutral chain, compressor 1:1 identity, limiter gradcheck, IIR/C++ parity |
| `test_sprint_2_2_audit_remediation.py` | **31 / 31** | **Sprint 2.2 Audit & Final Gate (B1–B4, F-06..F-11, G2, G3 [8 freqs $\times$ 16 phases + boundaries], G4 [400 ms boundaries])** |
| `test_sprint_2_protocol.py` | 13 / 13 | Same-track cross-correlation & fractional-delay alignment, stylistic mode, full-track temporal indexing |
| `test_ticket_1_2.py` | 7 / 7 | 6-band Parametric EQ (IIR & FFT), 3-way LR4 crossover all-pass reconstruction |
| `test_ticket_1_3.py` | 6 / 6 | 3-band Mid/Side imager, 2x oversampled `tanh` saturator parity & phase alignment |
| `test_ticket_1_4.py` | 7 / 7 | 3-band compressor C++/PyTorch parity, autograd stability |
| `test_ticket_1_5.py` | 8 / 8 | Sample-peak lookahead limiter ceiling, 159-sample chain latency, system identification |
| `test_ticket_2_1.py` | 9 / 9 | Multi-resolution M/S STFT loss, K-weighting coefficients & filter parity |
| `test_ticket_2_2.py` | 5 / 5 | Dataset loading, energy gating, headroom normalization |
| `test_ticket_2_3.py` | 4 / 4 | Benchmark suite evaluation, self-supervised optimization, preset export parity |
| *Parametrized expansions across suites* | +17 | Parametrized test cases across `test_ticket_1_2..1_5` and `test_sprint_2_protocol` |
| **TOTAL** | **128 / 128** | **Zero regressions across Gate A, Sprint 2, Sprint 2.1, and Sprint 2.2** |

### 6.2 Native C++ JUCE 8 Headless Test Suite (`plugin_headless_tests.exe`)
- **Total Stages Passed**: **12 / 12 stages (100% PASS)**
  - `[TEST 1]`–`[TEST 11]`: 42 APVTS parameters (`41 DSP + 1 Bypass`), `getBypassParameter()`, 159-sample latency, `processBlock` across buffer sizes `32..2048`, `-240 dBFS` null parity vs standalone `core_dsp::MasteringChain`, host bypass bit-exact 159-sample delay, APVTS XML state persistence, `AutoMastering_Preset_v1` schema export.
  - `[TEST 12]` (**G2**): Authentic Python fixture (`python/tests/fixtures/python_exported_preset_v1.json`) loaded in **Mode A** (`normalized_z` + `parameters`) and **Mode B** (`parameters` map only), verifying all 41 APVTS parameters and normalized `z` values, schema `version == "1.0"` enforcement, and strict atomicity (zero partial APVTS updates on rejected presets).

---

## 7. Explicit Hardening Backlog (Out of Scope for Sprint 2.2 Gate Closure)

In accordance with the Sprint 2.2 Final Gate specification, the following items are **not** claimed as completed in Sprint 2.2 and remain tracked in the engineering hardening backlog:

1. **F-06 Extended Scalability & Backward Batching**:
   - Quantitative multi-threaded GIL contention benchmarks, C++ batch parallelization for the custom autograd backward pass ($B > 1$), and formal CPU speedup benchmarking across batch sizes.
2. **Custom JUCE GUI Editor Binding**:
   - `MasteringPluginAudioProcessor::createEditor()` currently returns `juce::GenericAudioProcessorEditor` for host parameter manipulation, while `PluginEditor.cpp` implements the informational header banner. A full custom graphical interface with 41 dedicated rotary/slider attachments (`AudioProcessorValueTreeState::SliderAttachment`) remains in the UI backlog.
3. **Multi-Platform CI & Real-Host VST3 Validation**:
   - Automated CI currently builds and tests on `windows-latest` (MSVC x64). Linux/macOS CI build matrices and external VST3 host validation (`pluginval` / commercial DAW bundle verification) remain in the cross-platform deployment backlog.

---

## 8. Final Gate Verdict & Sign-Off

- **Verdict**: **GO — Sprint 2.2 Final Gate CLOSED**
  - All four Final Gate criteria (**G1, G2, G3, G4**) are implemented, verified by automated tests, and confirmed by a green GitHub Actions CI run (`#41435629557`) on exact code commit `5940e81f12741e2276f7a446adfb6e12f44574fe`.
  - All core DSP invariants (41 parameters, 159-sample latency, 6-stage topology, sample-peak limiter vs. 4x True Peak benchmark meter, differentiable proxy vs. dual-gated BS.1770-5 integrated loudness) are preserved without regression.

**Candidate:**  
Giovanni Antonecchia  
*Master's Thesis in Sound and Music Engineering — Politecnico di Milano (DEIB)*
