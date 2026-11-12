# SPRINT 3 — NOTA DI AUDIT & ERRATA PREFLIGHT V3

> **Documento Formale di Audit, Errata & Riconciliazione Numerica Preflight V3**  
> **Repository:** `gio961gio/AutoMastering` | **Branch:** `human-validation-package`  
> **Commit di riferimento:** `6afb247f249cd03684d1f42df343911928f7e8bd`  
> **Modalità:** Rigorosamente READ-ONLY (ZERO training, ZERO ottimizzazioni, preservazione integrale degli storici).

---

## 1. Riconciliazione SHA Commit 1 del Preflight V3

Nella sequenza operativa approvata da Argon per il Preflight V3, sono stati generati e pubblicati sul branch `human-validation-package` esattamente **2 commit sequenziali**:

1. **Commit 1 (Runner, Gate Test e Loss Unificata):**  
   **SHA:** `c267de68098a551c235c769c1f02389297ffdfac`  
   **Messaggio:** `fix(sprint3-v3b): unify preflight and V3bDynamicPriorLoss formulas, add w_lim sensitivity and exact sample-level alignment test`  
   File modificati:
   - `python/ddsp_mastering/loss/dynamic_prior.py`
   - `scripts/sprint3/run_v3b_preflight.py`
   - `tests/test_sprint3_v3b_gate.py`

2. **Commit 2 (Nuovi Artefatti Diagnostici V3):**  
   **SHA:** `6afb247f249cd03684d1f42df343911928f7e8bd`  
   **Messaggio:** `feat(sprint3-v3b): generate preflight V3 results and report with unified formulas and w_lim sensitivity`  
   File creati:
   - `docs/sprint3/SPRINT3_V3B_PREFLIGHT_V3_RESULTS.json`
   - `docs/sprint3/SPRINT3_V3B_PREFLIGHT_V3_REPORT.md`

Si conferma che il commit `c267de68098a551c235c769c1f02389297ffdfac` è il reale genitore (`HEAD~1`) del commit `6afb247f249cd03684d1f42df343911928f7e8bd`.

---

## 2. Riconciliazione Esatta coi Valori del File JSON `SPRINT3_V3B_PREFLIGHT_V3_RESULTS.json`

I valori numerici congelati e verificabili nel file JSON `docs/sprint3/SPRINT3_V3B_PREFLIGHT_V3_RESULTS.json` al commit `6afb247` sono riepilogati di seguito con percorso JSON e definizione matematica esatta:

### 2.1 Cosine Similarities su `track01` (`Candidate_B`, $\mathcal{L}_{\text{lim}}$ vs $\mathcal{L}_{\text{V0p}}$ totale)

- **Sottospazio Completo 41D (`all_nonzero_groups_41d`):**
  - Spazio $z$: **`-0.8940588`** (`-0.8940588118768376`)
  - Spazio $u$: **`-0.8797892`** (`-0.8797892335165385`)
- **Cosine Similarities Per-Gruppo in Spazio $z$ (`vs_v0p_total`):**
  - `EQ` (indici 0..17): **`-0.901219`** (`-0.9012194540704723`)
  - `Crossover` (indici 18..19): **`+0.928731`** (`+0.9287308020314530`)
  - `Compressor` (indici 20..34): **`-0.897095`** (`-0.8970954201904680`)
  - `Imager` (indici 35..37): **`-0.733199`** (`-0.7331988929016701`)
  - `Saturator` (indici 38..39): **`+1.000000`** (`+0.9999999991138560`)
  - `Limiter` (indice 40): **`-1.000000`** (`-1.0000000000000000`)
- **Sottospazio Compressore + Limiter (`compressor_limiter_subspace`, 16 parametri: $20\dots 34 + 40$):**
  - Spazio $z$: **`-0.967990`** (`-0.9679899388277258`)
  - Spazio $u$: **`-0.898533`** (`-0.8985331104460677`)

### 2.2 Gain Reduction (GR) Multibanda in `track01_style` (`v0p_z_opt`)

In `track01_style` (che registra il picco massimo globale di compressione multibanda tra tutte le 6 sessioni armonizzate $V0p$), i valori congelati nel JSON (`session_summary.pooled_active`) sono:
- $\bar{GR}_L$ (media banda Low): **`0.000157969 dB`** (`0.00015796914522070438 dB`)
- $\bar{GR}_M$ (media banda Mid): **`0.000116359 dB`** (`0.00011635947157628834 dB`)
- $\bar{GR}_H$ (media banda High): **`4.2749e-8 dB`** (`0.00000004274928500080932 dB`)
- $\max GR_L$ (picco massimo Low): **`0.000905038 dB`** (`0.0009050379157997668 dB`)
- $\max GR_M$ (picco massimo Mid): **`0.000329118 dB`** (`0.0003291184257250279 dB`)
- $\max |GR_L - GR_M|$ (discrepanza inter-banda massima): **`0.000805107 dB`** (`0.0008051065378822386 dB`)

### 2.3 Attenuazione Limiter in `track01` — Chunk 1

Sul **Chunk 1** di `track01` (`sessions[0].empirical_stats_v0p_opt.per_chunk_stats[1]`), che ospita il transiente audio più aggressivo dell'intero dataset:
- $P_{95}(A_{\text{lim}})$: **`1.809195 dB`** (`1.8091952800750732 dB`)
- $\text{CVaR}_{0.95}(A_{\text{lim}})$: **`3.043106 dB`** (`3.0431058406829834 dB`)
- $\max A_{\text{lim}}$: **`6.003366 dB`** (`6.003366470336914 dB`)
- $\bar{A}_{\text{lim}}$ (media su campioni attivi): **`0.283576 dB`** (`0.2835756838321686 dB`)

---

## 3. Chiarimento Parità DSP C++ vs Python (Test 9)

Si chiarisce formalmente che:
1. L'interfaccia C++ esposta dal binding Python `py_core_dsp` per il modulo di compressione (`py_core_dsp.SingleBandComp` e `py_core_dsp.MultibandCompressor3Band`) espone **esclusivamente l'uscita audio elaborata** (`process(audio_in) -> audio_out`) e non fornisce buffer o tracciamenti del guadagno interno $yg[n]$.
2. Pertanto, la suite di test del Gate Tecnico (Test 9, `test_09_cpp_python_limiter_and_compressor_parity`) verifica rigorosamente la **parità bit-level dell'uscita audio** tra l'implementazione C++ e quella differenziabile in PyTorch:
   $$\text{Null Audio RMS} = 10 \log_{10} \left( \text{mean}((y_{\text{cpp}} - y_{\text{py}})^2) + 10^{-16} \right) \le -100.0\text{ dBFS}$$
   (con risultato misurato $\le -160.0\text{ dBFS}$ per il compressore e $\le -140.0\text{ dBFS}$ per il limiter).
3. Non viene effettuato, né è tecnicamente possibile senza modifiche al binding C++, un confronto diretto campione per campione della traccia di gain reduction $yg[n]$ del compressore. La perfetta identità dell'output audio dimostra l'equivalenza numerica del calcolo della gain reduction sottostante.
