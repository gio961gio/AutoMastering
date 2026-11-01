# Review Dossier: Ticket 2.2 — Audio Dataset Pipeline & Synthetic Perturbation Engine

**To**: Argon (Senior Technical Supervisor & Chief Architect)  
**From**: Antigravity (Lead Implementation Engineer)  
**Project**: AutoMastering (DDSP White-Box Twin Engine — PoliMi Master's Thesis & Industry Prototype)  
**Ticket**: 2.2 — Audio Dataset Ingestion, Energy Gating, Headroom Normalization & Synthetic Pairing  
**Date**: October 31, 2026  
**Status**: Ready for Review / Requesting Approval (`STATUS: APPROVED`)

---

## 1. Executive Summary & Deliverables

In Ticket 2.2, we established the complete data ingestion, pre-processing, and self-supervised synthetic pairing pipeline required for training and evaluating our DDSP mastering model:

### Key Deliverables:
1. **Audio Dataset Pipeline (`ddsp_mastering/data/dataset.py`)**:
   - **Sample-Rate & Channel Enforcement**: Automatically ingests multi-format audio files, verifies/resamples to $48\text{ kHz}$ float32, and expands mono files to dual-channel stereo.
   - **RMS / LUFS Energy-Gated Chunk Extraction**: Extracts $2.0\text{ s}$ to $4.0\text{ s}$ slices ($N \in [96000, 192000]$ samples). Rejects quiet intros, sparse breakdowns, and fading outros falling $> 12\text{ dB}$ below the track's average RMS energy, guaranteeing that training batches always contain full-band musical dynamics.
   - **Pre-Mastering Headroom Normalization**: Uses our differentiable loudness calculator to normalize raw unmastered input mixes to a uniform pre-mastering loudness target ($-18.0\text{ LUFS}$), with an automatic peak guard limiter ($-1.0\text{ dBFS}$) to ensure that multiband compressor thresholds ($T_b$) and saturator drive operate in an acoustically consistent, predictable regime.
   - **Memory Caching**: Supports zero-disk-bottleneck in-memory preloading for rapid epoch throughput.
2. **Self-Supervised Synthetic Pair Generator (`ddsp_mastering/data/synthetic.py`)**:
   - Enables self-supervised training and ground-truth validation without requiring proprietary multi-track stems.
   - Samples realistic mastering parameter vectors $z_{\text{target}} \in [0.1, 0.9]^{41}$ via Gaussian priors centered around neutral $0.5$ ($\sigma = 0.12$), avoiding extreme unrealistic edge settings.
   - Generates synchronized triplets $(x_{\text{raw}}, y_{\text{mastered}}, z_{\text{target}})$ via the differentiable mastering chain (or C++ fast engine).
   - Fully reproducible worker execution using deterministic `torch.Generator` seeds.
3. **Universal Batching Enhancement in LR4 Crossover (`crossover.py`)**:
   - Enhanced `DifferentiableLR4Crossover` to seamlessly broadcast and process batched multi-channel audio tensors $(B, C, N)$ through `torchaudio.functional.lfilter`, resolving channel-dimension mismatches while maintaining 100% mathematical parity (verified across all 10 unit tests of Ticket 1.2).

---

## 2. Technical Architecture & Ingestion Flow

```
Raw Audio File (.wav / .flac / .mp3)
       │
       ▼
Resample to 48 kHz Stereo (float32)
       │
       ▼
Compute Track-Wide Energy Profile
       │
       ▼
Energy Gate Filter: Reject segments < (Track_RMS - 12 dB)
       │
       ▼
Extract Chunk (L = 96,000 samples / 2.0s)
       │
       ▼
Pre-Mastering Headroom Normalization (Target: -18.0 LUFS, Peak <= -1.0 dBFS)
       │
       ├──► Mode A (Paired / Benchmark): Feed to AutoMastering Optimizer
       │
       └──► Mode B (Synthetic Self-Supervised):
                 │
                 ▼
            Sample z_target ~ N(0.5, 0.12^2) in [0.1, 0.9]^41
                 │
                 ▼
            Apply MasteringChain(z_target) -> (x_raw, y_master, z_target)
```

---

## 3. Verification & Test Results (`test_ticket_2_2.py`)

All 5 unit tests passed cleanly in pytest:
```
python/tests/test_ticket_2_2.py::test_audio_dataset_loading_and_conversion PASSED [ 20%]
python/tests/test_ticket_2_2.py::test_energy_gate_avoids_quiet_sections PASSED [ 40%]
python/tests/test_ticket_2_2.py::test_premastering_headroom_normalization PASSED [ 60%]
python/tests/test_ticket_2_2.py::test_synthetic_pair_generator PASSED    [ 80%]
python/tests/test_ticket_2_2.py::test_generator_reproducibility PASSED   [100%]

============================== 5 passed in 3.11s ==============================
```

### Key Verification Metrics:
- **Audio Loading & Resampling**: Verified 44.1 kHz mono conversion to 48 kHz stereo float32.
- **Energy Gating**: Successfully rejected low-energy intros (tested on tracks with $-40\text{ dBFS}$ intro vs $-9\text{ dBFS}$ drop), keeping selected chunks within the active dynamic region ($>-25\text{ dBFS}$).
- **Headroom Normalization**: Normalized loud ($-6\text{ dBFS}$) and quiet ($-30\text{ dBFS}$) audio chunks to $-18.0\text{ LUFS}$ within $\pm 0.6\text{ LU}$.
- **Synthetic Generator Invariants**: Parameter vectors $z_{\text{target}}$ strictly bounded in $[0.1, 0.9]^{41}$.
- **Reproducibility**: Identical `torch.Generator` seeds produce identical parameter vectors ($0.0\text{ error}$).
- **Full Suite Non-Regression**: 10/10 tests in `test_ticket_1_2.py` passed in $0.67\text{ s}$.

---

## 4. Request for Review & Phase 2 Roadmap

We request Argon's review and approval (`STATUS: APPROVED`) for Ticket 2.2 to proceed to the final ticket of Phase 2:
- **Ticket 2.3**: End-to-End Self-Supervised Optimization Loop & Multi-Metric Benchmark Suite (Evaluating convergence across EQ, Crossover, Dynamics, Stereo Width, Saturation, and Limiting).
