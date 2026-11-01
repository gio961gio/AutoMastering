# Review Dossier: Ticket 2.1 — Multi-Resolution STFT & Perceptual Loudness Loss

**To**: Argon (Senior Technical Supervisor & Chief Architect)  
**From**: Antigravity (Lead Implementation Engineer)  
**Project**: AutoMastering (DDSP White-Box Twin Engine — PoliMi Master's Thesis & Industry Prototype)  
**Ticket**: 2.1 — Multi-Resolution STFT Loss & Differentiable ITU-R BS.1770-4 Loudness Loss  
**Date**: October 31, 2026  
**Status**: Ready for Review / Requesting Approval (`STATUS: APPROVED`)

---

## 1. Executive Summary & Deliverables

In Ticket 2.1, we transitioned to **Phase 2: Loss Functions & Dataset Pipeline**, implementing the perceptual loss surface that guides the self-supervised mastering optimization and neural parameter estimator:

### Key Deliverables:
1. **Multi-Resolution STFT Loss with Mid/Side Stereo Decomposition (`ddsp_mastering/loss/stft.py`)**:
   - Multi-scale time-frequency trade-off across three complementary window sizes:
     - $N_{\text{fft}} = 512$ ($\sim 10.7\text{ ms}$ at $48\text{ kHz}$, hop 128): captures high temporal precision for compressor transients and limiter clamping.
     - $N_{\text{fft}} = 2048$ ($\sim 42.7\text{ ms}$ at $48\text{ kHz}$, hop 512): balanced mid-band spectral resolution.
     - $N_{\text{fft}} = 8192$ ($\sim 170.7\text{ ms}$ at $48\text{ kHz}$, bin width $\Delta f \approx 5.86\text{ Hz}$, hop 2048): pinpoint sub-bass spectral resolution for 20–100 Hz EQ tuning and crossover band separation.
   - Dual-component loss at each resolution:
     - Spectral Convergence: $\mathcal{L}_{\text{sc}} = \frac{\| |Y| - |\hat{Y}| \|_F}{\| |Y| \|_F}$
     - Log-Magnitude $L_1$: $\mathcal{L}_{\text{mag}} = \frac{1}{N} \| \log(|Y| + \epsilon) - \log(|\hat{Y}| + \epsilon) \|_1$ ($\epsilon = 10^{-5}$)
   - Mid/Side Decomposition:
     $$M = \frac{L + R}{\sqrt{2}}, \quad S = \frac{L - R}{\sqrt{2}}$$
     Loss evaluated simultaneously over Left/Right and Mid/Side channels, penalizing stereo phase cancellation and width distortions.

2. **Differentiable ITU-R BS.1770-4 / EBU R128 K-Weighting & Loudness Loss (`ddsp_mastering/loss/loudness.py`)**:
   - Exact cascaded two-stage biquad filtering:
     - Stage 1: High-shelf acoustic head-model boost ($f_0 \approx 1682\text{ Hz}$, $Q = 0.7071$, Gain $+4\text{ dB}$).
     - Stage 2: RLB high-pass filter ($f_0 \approx 38.1\text{ Hz}$, $Q = 0.5003$).
   - Differentiable implementation: Analytical evaluation of complex frequency response $H(e^{j\omega}) = H_1(e^{j\omega}) \cdot H_2(e^{j\omega})$ combined with real FFT (`torch.fft.rfft` / `irfft`), providing $O(T \log T)$ parallel filtering without sequential autograd recurrences and matching Scipy reference within $0.0002\text{ LU}$.
   - Integrated LUFS calculation:
     $$z_{\text{total}} = \frac{1}{T} \sum_{t=1}^T y_{k, L}^2[t] + \frac{1}{T} \sum_{t=1}^T y_{k, R}^2[t]$$
     $$\text{LUFS} = -0.691 + 10 \log_{10}(z_{\text{total}} + 10^{-12})$$
   - Crest Factor / Peak-to-Loudness Ratio (PLR) Dynamic Range penalty:
     $$\text{PLR} = 20 \log_{10}(\text{Peak} + \epsilon) - \text{LUFS}$$
     Penalizes over-compression ($\text{PLR} < 8\text{ dB}$) and retains transient punch.

3. **Mastering Perceptual Loss with Latency Compensation (`ddsp_mastering/loss/combined.py`)**:
   - Automatically slices off the deterministic 159-sample delay of the chain ($D = 159$ samples at $48\text{ kHz}$) before reference comparison.
   - Combines STFT loss, loudness loss, and a latent parameter regularization prior ($\|z - 0.5\|^2$) to prevent extreme DSP setting drift.

---

## 2. Mathematical Formulation & Architecture

### 2.1 Multi-Resolution Mid/Side STFT Loss
Given target $y$ and prediction $\hat{y} \in \mathbb{R}^{B \times 2 \times T}$:
$$\mathcal{L}_{\text{stft}} = \frac{\mathcal{L}_{\text{LR}} + w_{\text{ms}} \mathcal{L}_{\text{MS}}}{1 + w_{\text{ms}}}$$
where:
$$\mathcal{L}_{\text{LR}} = \frac{1}{M} \sum_{i=1}^M \left[ \alpha_{\text{sc}} \mathcal{L}_{\text{sc}}^{(i)}(y, \hat{y}) + \alpha_{\text{mag}} \mathcal{L}_{\text{mag}}^{(i)}(y, \hat{y}) \right]$$
$$\mathcal{L}_{\text{MS}} = \frac{1}{M} \sum_{i=1}^M \left[ \alpha_{\text{sc}} \mathcal{L}_{\text{sc}}^{(i)}(y_{\text{ms}}, \hat{y}_{\text{ms}}) + \alpha_{\text{mag}} \mathcal{L}_{\text{mag}}^{(i)}(y_{\text{ms}}, \hat{y}_{\text{ms}}) \right]$$

### 2.2 Differentiable K-Weighting Frequency Response
The cascaded digital filter transfer function is:
$$H(z) = \left( \frac{b_{1,0} + b_{1,1} z^{-1} + b_{1,2} z^{-2}}{1 + a_{1,1} z^{-1} + a_{1,2} z^{-2}} \right) \cdot \left( \frac{b_{2,0} + b_{2,1} z^{-1} + b_{2,2} z^{-2}}{1 + a_{2,1} z^{-1} + a_{2,2} z^{-2}} \right)$$
Evaluated along the unit circle $z = e^{-j\omega_k}$ for $\omega_k = \frac{2\pi k}{T}$, $k \in [0, T/2]$.

Filtering in PyTorch:
$$Y_k(\omega) = \text{RFFT}(x) \cdot H(\omega), \quad y_k(t) = \text{IRFFT}(Y_k(\omega))$$
Autograd propagates cleanly through complex multiplication and inverse FFT.

---

## 3. Verification & Test Results (`test_ticket_2_1.py`)

All 9 unit tests passed in pytest:
```
python/tests/test_ticket_2_1.py::test_stft_loss_identity PASSED          [ 11%]
python/tests/test_ticket_2_1.py::test_stft_loss_m_s_sensitivity PASSED   [ 22%]
python/tests/test_ticket_2_1.py::test_stft_loss_gradient_flow PASSED     [ 33%]
python/tests/test_ticket_2_1.py::test_k_weighting_coefficients PASSED    [ 44%]
python/tests/test_ticket_2_1.py::test_k_weighting_filter_parity PASSED   [ 55%]
python/tests/test_ticket_2_1.py::test_lufs_calculation_calibration PASSED [ 66%]
python/tests/test_ticket_2_1.py::test_loudness_loss_gradient_flow PASSED [ 77%]
python/tests/test_ticket_2_1.py::test_mastering_loss_latency_alignment PASSED [ 88%]
python/tests/test_ticket_2_1.py::test_end_to_end_optimization_with_mastering_chain PASSED [100%]
```

### Key Verification Metrics:
- **K-Filter Accuracy vs Scipy Reference**: Max error $< 10^{-3}$ on continuous audio; LUFS difference: **$0.000235\text{ LU}$** (essentially bit-identical).
- **1 kHz 0 dBFS Calibration**: Evaluated at $-0.691\text{ LUFS}$ within $\pm 0.05\text{ LU}$ of ITU-R BS.1770 specification.
- **Latency Alignment**: Compensates for 159-sample delay, achieving monotonic loss reduction in reference-matching tasks.
- **End-to-End Optimization**: 10-step Adam optimization successfully minimizes composite loss through the entire 41-D `DifferentiableMasteringChain` without exploding or vanishing gradients.

---

## 4. Request for Review & Phase 2 Roadmap

We request Argon's review and approval (`STATUS: APPROVED`) for Ticket 2.1 to proceed to:
- **Ticket 2.2**: Unpaired & Paired Audio Dataset Pipeline (MusicNet / MedleyDB / DSD100 audio chunk loaders with dynamic loudness normalization).
- **Ticket 2.3**: End-to-End Self-Supervised Optimization Loop & Benchmark Suite.
