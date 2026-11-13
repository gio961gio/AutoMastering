# SPRINT 4 — REPORT DI IMPLEMENTAZIONE E VERIFICA: TRUE PEAK / ISP HARDENING

- **Repository:** `gio961gio/AutoMastering`
- **Branch:** `human-validation-package`
- **Baseline codice congelata pre-Sprint 4:** `3fdeb68ff6626125781dbc3fcdcf568c1534b765`
- **Ultimo commit documentale pre-implementazione:** `9f7c0a75f726eab9c46f27f1e1148525dca25ab1` (`docs/sprint4/SPRINT4_TRUE_PEAK_ISP_HARDENING_AUDIT.md`)
- **Obiettivo implementato:** **Soluzione 1 (primaria)** — True Peak Lookahead Limiter 4× polyphase FIR nel percorso DSP C++ e Python (`detector_mode="true_peak_4x"` di default) + **Soluzione 2 (fallback offline)** — Guardrail di verifica e sicurezza post-export su disco `PCM_24` (`export_validated_pcm24_wav`).
- **Vincoli rispettati:**
  - `V3cStereoCompatibilityLoss` invariata;
  - `enable_v3c=False` invariato;
  - nessuna modifica ai pesi, formule o configurazioni delle loss V0p–V3c (`Solution 3` esclusa);
  - nessuna modifica ai 41 parametri latenti o a `PARAM_SPECS`;
  - nessuna modifica agli artefatti storici `docs/sprint3/*`;
  - nessuna ottimizzazione Adam / nuovo training eseguito;
  - mantenuta riproducibilità bit-fedele del detector storico tramite modalità esplicita `legacy_sample_1x`.

---

## 1. Architettura Implementata e Latenza Totale Invariata (`159` campioni)

### 1.1 Single Source of Truth DSP (C++ ↔ Python)

L'intero percorso di limitazione usa ora la stessa topologia e gli stessi coefficienti esatti tra C++ (`core_dsp`) e Python (`ddsp_mastering`), con modalità selezionabile:

| Modalità `detector_mode` | Ceiling predefinito | Smoothing Boxcar | Scopo |
|:---|---:|---:|:---|
| **`true_peak_4x`** *(default)* | **`-0.50 dBTP`** (`LIMITER_INTERNAL_CEILING_DBTP`) | **`136` campioni** (`LIMITER_BOXCAR_SAMPLES`) | Controllo Inter-Sample Peak 4× (con inclusione del campione `1x` allineato) e rispetto del contratto `<= -0.30 dBTP` su WAV `PCM_24` e VST3 live |
| **`legacy_sample_1x`** *(esplicita)* | **`-0.30 dBFS`** (`LIMITER_CEILING_DBFS`) | **`144` campioni** (`LIMITER_LOOKAHEAD_SAMPLES`) | Riproduzione storica esatta dei render Sprint 2 / Sprint 3 V1 senza alterare i parametri passati |

### 1.2 Provenienza, Specifiche e Proprietà del Filtro FIR 4× Polyphase (48 Tap)

1. **Distinzione tra filtro normativo e interpolatore interno di progetto:**
   - L'**interpolatore di riferimento per le metriche offline** (`ddsp_mastering.optimize.metrics.calculate_true_peak_dbtp`) è il sinc interpolator con finestra di Kaiser `torchaudio.functional.resample(..., lowpass_filter_width=64, rolloff=0.9475937167399596, beta=14.769656459379492)`, usato a posteriori per verificare il requisito contrattuale `<= -0.30 dBTP` sui WAV `PCM_24`.
   - Il **filtro del detector real-time DSP** (sia C++ sia Python `conv1d` differenziabile) non è identificato arbitrariamente come "il filtro normativo ITU-R": è un **prototipo lineare FIR simmetrico Tipo II WLS a 48 tap su griglia 4× (`192 kHz`, 4 rami polifase da 12 tap per canale)** progettato specificamente per il progetto e memorizzato come unica sorgente di verità in `cpp/core_dsp/include/core_dsp/Constants.hpp` (`TP_FIR_PROTOTYPE_COEFFS`), esposto direttamente a Python tramite `py_core_dsp.get_true_peak_fir_coeffs()` (con copia esatta di fallback in `python/ddsp_mastering/dsp/limiter.py` verificata al bit a `< 1e-15`).
2. **Proprietà matematiche verificate sull'istanza compilata (`py_core_dsp.get_true_peak_fir_coeffs()`):**
   - **Lunghezza / Topologia:** `48` tap totali (`4` rami polifase causal-indexed da `12` tap ciascuno: `h_phi[k] = h[4*k + phi]`, corrispondenti alle 4 sotto-fasi per ogni nuovo campione base-rate).
   - **Simmetria lineare esatta:** `max|h[m] - h[47 - m]| = 2.220e-16` (`< 1e-15`).
   - **Normalizzazione DC esatta per ciascuna fase polifase:**  
     `sum(phase_0..3) = [0.9999999999999998, 0.9999999999999996, 1.0000000000000002, 0.9999999999999998]` (`|sum - 1.0| < 1e-14`).
   - **Risposta in frequenza (`freqz` su griglia `192 kHz`, normalizzata `|H|/4`):**
     - Passband `0–18 kHz`: `[-0.0137, +0.2882] dB`
     - Passband `0–20 kHz`: `[-0.0137, +0.5056] dB` (leggero boost conservativo `1/cos` in alta banda `> 10 kHz` affinché il detector real-time compatto non sottostimi gli ISP `16–22 kHz` rispetto a `calculate_true_peak_dbtp`)
     - `22 kHz`: `+0.0603 dB`; `24 kHz` (`fs/2` base): `-2.2953 dB`; stopband `>= 27.5 kHz`: `max <= -17.83 dB`.
   - **Ritardo di gruppo effettivo:** costante nella passband (`0.1–20 kHz`) a **`23.500000` campioni @ 192 kHz = `5.875000` campioni @ 48 kHz**, esattamente uguale a `(48 - 1) / (2 * 4) = 5.875`.
3. **Topologia polifase esplicita (no `conv1d(..., stride=1)` errata):**
   - C++ (`LookaheadLimiter.hpp`): per ogni campione stereo `u[n]`, aggiorna ring buffer statici `std::array<double, 24>` L/R (zero allocazioni heap nel path audio) ed esegue le 4 convoluzioni dot-product da `12` tap su `detail::TP_POLYPHASE_TABLE` (`constexpr`), combinandole nel `max` stereo-linked con il campione `1x` originale ritardato esattamente del ritardo intero allineato al prototipo simmetrico (`TP_FIR_ALIGNED_SAMPLE_DELAY = 6`).
   - Python (`DifferentiableLookaheadLimiter` in `python/ddsp_mastering/dsp/limiter.py`): `F.conv1d` causale con padding `(11, 0)` e kernel `[4, 1, 12]` sui 2 canali concatenati sul batch, producendo esattamente 4 fasi per campione, più il segnale `1x` ritardato di `6` campioni tramite `F.pad(..., (6, 0))[:, :, :T]`.

### 1.3 Analisi Temporale Esatta: Compensazione del Ritardo FIR (`5.875 / 6` smp) e Fix Bug Off-by-One a `TOTAL_CHAIN_LATENCY_SAMPLES = 159`

Nel limiter storico (`legacy_sample_1x`):
- Il segnale audio `u` veniva ritardato di `D = 144` campioni (`u_delayed[n] = u[n - 144]`).
- Il `min_filter` di larghezza `144` sull'istante `n` copriva `[n - 143 .. n]`, seguito da release `g_rel` e boxcar causale di lunghezza `144` su `[n - 143 .. n]`.
- Un impulso isolato in `u[n_0]` abbassava `g_rel` per gli indici `[n_0 .. n_0 + 143]` (`144` campioni), e all'uscita del boxcar il fronte di salita impiegava `143` ulteriori ritardi, raggiungendo il minimo al campione `n_0 + 143`, **1 campione prima** di `u_delayed[n_0 + 144] = u[n_0]` (a `n_0 + 144` il boxcar includeva già il primo campione in release `g_rel[n_0 + 144] > g_raw[n_0]`, causando un leak di release proporzionale all'ampiezza per impulsi molto alti!).

Con il detector True Peak 4× mantenendo **`LIMITER_LOOKAHEAD_SAMPLES = 144`** e **`TOTAL_CHAIN_LATENCY_SAMPLES = 159`**:
1. Il sidechain FIR ha centro di gruppo a `5.875` campioni (`TP_FIR_ALIGNED_SAMPLE_DELAY = 6` rispetto all'istante di ingresso `u[n]`): un picco centrato su `u[n_0]` (o inter-sample vicino a `n_0`) emerge dal detector al più tardi attorno a `n_0 + 6` (e i lobi FIR si estendono fino a `11` campioni oltre `n_0`, mentre l'energia dominante del picco cade intorno a `n_0 + 6`).
2. Per allineare il guadagno attenuato a **`u_delayed[n_0 + 144] = u[n_0]`** eliminando il bug off-by-one di release-leak:
   - la finestra `min_filter` (lookahead interno nel sidechain ritardato di `+6` campioni) e la finestra `boxcar` devono complessivamente ritardare l'effetto di `g_rel` da `n_det ≈ n_0 + 6` fino a `n_out = n_0 + 144` **senza includere** il campione in release `n_0 + 144 + 1` prima del passaggio completo dell'audio `u[n_0]`.
   - Scegliendo **`M_box = LIMITER_BOXCAR_SAMPLES = 136`**: la rampa di attacco del boxcar di lunghezza `136` copre `[n_det .. n_det + 135]` (`(n_0 + 6) + 135 = n_0 + 141`), lasciando almeno `2..3` campioni pieni di supporto piatto di minima gain reduction attorno a `n_0 + 144` mentre `min_filter` di larghezza `144` su `[n - 143 .. n]` trattiene comunque il minimo fino a `[n_det .. n_det + 143]`.
   - Verifica su impulso fino ad amp `A = 10.0` (`+20 dBFS`, Test T2): `max|y|` resta rigorosamente `<= C_internal_lin = 10^(-0.50/20)` (`<= -0.50 dBFS`), confermando **zero release leak**, zero modifica di `TOTAL_CHAIN_LATENCY_SAMPLES = 159`, e pieno allineamento `peak_idx == n_0 + 144`!

### 1.4 Moduli Coinvolti nel Commit 1

- `cpp/core_dsp/include/core_dsp/Constants.hpp` — costanti `LIMITER_INTERNAL_CEILING_DBTP = -0.50`, `LIMITER_DELIVERY_CEILING_DBTP = -0.30`, `LIMITER_BOXCAR_SAMPLES = 136`, `TP_OVERSAMPLE_FACTOR = 4`, `TP_FIR_TAPS_PER_PHASE = 12`, `TP_FIR_PROTOTYPE_TAPS = 48`, `TP_FIR_GROUP_DELAY_BASE_SAMPLES = 5.875`, `TP_FIR_ALIGNED_SAMPLE_DELAY = 6`, `TP_FIR_PROTOTYPE_COEFFS`. (`TOTAL_CHAIN_LATENCY_SAMPLES = 159`, `LIMITER_LOOKAHEAD_SAMPLES = 144` invariati).
- `cpp/core_dsp/include/core_dsp/LookaheadLimiter.hpp` — enum `LimiterDetectorMode` (`TruePeak4x = 0`, `LegacySample1x = 1`), `detail::TP_POLYPHASE_TABLE` statica a namespace-scope, ring buffer statici `std::array<double, 24>`, nessun `new`/`vector` dinamico nel path di processing, gestione `reset`/`prepare`.
- `cpp/core_dsp/include/core_dsp/MasteringChain.hpp` — propagazione opzionale `LimiterDetectorMode`, `set_limiter_detector_mode`, `get_limiter_detector_mode`, `set_limiter_ceiling_db`, `get_limiter_ceiling_db`.
- `cpp/bindings/py_core_dsp.cpp` — binding `PyLookaheadLimiter`, `PyMasteringChain`, `get_true_peak_fir_coeffs()` e relative costanti Python.
- `python/ddsp_mastering/dsp/limiter.py` — `DifferentiableLookaheadLimiter` (`detector_mode="true_peak_4x"` default, caricamento coefficienti da `py_core_dsp` con fallback identico al bit, `F.conv1d` 4×12 polifase causale + picco `1x` ritardato di 6 campioni + boxcar `136` + modalità esplicita `legacy_sample_1x` con boxcar `144` e ceiling `-0.30`).
- `python/ddsp_mastering/dsp/chain.py` — parametri opzionali `limiter_detector_mode="true_peak_4x"` e `limiter_ceiling_db=None`.
- `scripts/human_validation/offline_renderer.py` — `render_offline_with_tail_flush(..., limiter_detector_mode="true_peak_4x")` + `export_validated_pcm24_wav` (Soluzione 2 fallback post-export `PCM_24`: rilegge da disco, misura `calculate_true_peak_dbtp` e `calculate_sample_peak_dbfs`, fallisce con `RuntimeError` rimuovendo il file non sicuro se sfora di `> 0.05 dB`, applica e logga esplicitamente micro-trim lineare se entro `0.05 dB` ri-verificando su disco).
- `tests/test_sprint4_true_peak_guardrail.py` — nuova suite deterministica completa `T1..T7`.

---

## 2. Verifica Deterministica `T1..T7` e Acceptance Criteria `AC1..AC5`

### 2.1 Risultati Suite e Build dopo le ultime modifiche

| Controllo | Comando | Esito osservato | Note / Dettagli |
|:---|:---|:---|:---|
| **C++ Python Extension Build** | `python setup.py build_ext --inplace` | **PASS** (exit 0) | Compilato `py_core_dsp.cp311-win_amd64.pyd` (C++20 / MSVC) |
| **JUCE Headless Unit Suite** | `build_juce\plugin_headless_tests.exe` | **12 / 12 PASS** (exit 0) | Latency `159` samples invariata; `Plugin vs Core Chain Null Error: -240 dBFS` |
| **JUCE VST3 Plugin Target** | `cmake --build build_juce --target MasteringPlugin_VST3 --config Release` | **PASS** (exit 0) | Compilato `AutoMastering PoliMi.vst3` con nuova `MasteringChain` |
| **Sprint 4 True Peak Guardrail (`T1..T7`)** | `pytest tests/test_sprint4_true_peak_guardrail.py -v` | **19 / 19 PASS** (`17.22s`) | Copre Provenance + `T1` (`1k..22k` Hz) + `T2` + `T3` + `T4` (`44.1/48/96` kHz) + `T5` + `T6` + `T7` |
| **Core Package Regression Suite** | `pytest python/tests -v` | **144 / 144 PASS** (`231.03s`) | Parità C++/Python, Biquad, Compressor, Imager, Saturator, Optimizer, Benchmark |
| **Root Sprint Tests** | `pytest tests -v` | **96 PASS, 4 FAIL** (`206.61s`) | I soli **4 FAIL** appartengono esclusivamente ai controlli storici Sprint 3 di `git status / git diff` su `python/ddsp_mastering/dsp` e `cpp/core_dsp` in `tests/test_sprint3_v3c_gate.py` (Gates `7, 9, 10, 11`), i quali vietavano qualsiasi modifica DSP prima di Sprint 4. Tutti i test funzionali V0p/V1/V2/V3a/V3b/V3c/Sprint4 passano al 100% (`96/96`). |

### 2.2 Matrice `T1..T7` Verificata (`tests/test_sprint4_true_peak_guardrail.py`)

| Test | Scenario verificato | Asserzioni e tolleranze superate (C++ & Python & `PCM_24` disco) |
|:---|:---|:---|
| **Provenance / Identity** | `TP_FIR_PROTOTYPE_COEFFS` vs `py_core_dsp.get_true_peak_fir_coeffs()` | `len=48`, diff C++/Py `< 1e-15`, simmetria `< 1e-15`, 4 fasi DC `== 1.0 ± 1e-14`, GD `= 5.875` smp, `TOTAL_CHAIN_LATENCY_SAMPLES = 159`, `LOOKAHEAD = 144`, `BOXCAR = 136` |
| **T1** | Sinusoidi `[1k, 12k, 16k, 18k, 20k, 22k]` Hz, fasi `{0, pi/4, pi/2, 3pi/4}`, amp `{1.2, 3.0}`, finestra smooth e step brusco | `TP 4x (PCM_24) <= -0.30 dBTP` e `SP <= -0.30 dBFS` su tutte le frequenze e fasi in **6 parametrizzazioni PASS** (C++ e Python) |
| **T2** | Impulsi isolati e treni spaziati `143 / 144 / 145` campioni (`A = 1, 2, 5, 10`), step bipolari `[+A, +A, -A, -A]`, multitone `1100+6310+15430 Hz` + burst | `SP <= -0.50 dBFS` (`+ 0.01` tol) senza alcun release leak per `A=10` (`+20 dBFS`), picco esattamente a `idx = 200 + 144 = 344`, `TP_bip <= -0.30 dBTP`, `TP_multi <= -0.30 dBTP` (**5/5 PASS**) |
| **T3** | Stereo asimmetrico (`L = 2.0 @ 12kHz pi/4`, `R = 0.25 @ 12kHz pi/4`) e swap `L<->R` | Simmetria `L<->R < 1e-6`, rapporto L/R `8.0 ± 1e-4` invariato (vero stereo-link senza sbilanciamento M/S), `TP 4x <= -0.30 dBTP` (**PASS**) |
| **T4** | Streaming a blocchi irregolari `[1, 3, 7, 13, 64, 127, 256, 512, 1024]`, `reset()` dopo saturazione `3.0`, short buffer `37` smp, programma audio-band a `[44.1, 48, 96]` kHz | Differenza chunk vs blocco unico `< 1e-5`, post-reset `== 0.0` bit-esatto, nessun NaN su `N < 144`, `TP 4x (PCM_24) <= -0.30 dBTP` su tutti e 3 i sample rate (**3/3 PASS**) |
| **T5** | Silenzio `zeros(2, 2048)` e sinusoidi a `-60` / `-120 dBFS` | Silenzio `== 0.0` esatto, nessun NaN/Inf, per `-60/-120 dBFS` `g_smooth == 1.0 ± 1e-12` e ritardo puro `144` campioni (`diff < 1e-12`) (**PASS**) |
| **T6** | Parità C++/Py Limiter (`<= -105 dB`) & Full Chain (`<= -100 dB`), `render_offline_with_tail_flush`, autograd su segnale dinamico + burst ISP `16 kHz`, compatibilità esplicita `legacy_sample_1x` | Parità limiter `<= -105 dB` e chain `<= -100 dB`, gradienti `z_param.grad` finiti con `|grad[40]| > 1e-5`, `legacy_sample_1x` riproduce `boxcar=144` e `-0.30 dBFS` entro `1e-5` (**PASS**) |
| **T7** | `export_validated_pcm24_wav` su casi `-0.50 dBTP` (`pass`), `-0.27 dBTP` (`pass_with_micro_trim`), `-0.10 dBTP` (`RuntimeError` + cleanup) | Rilettura da disco verificata, micro-trim applicato & loggato entro `0.05 dB`, blocco rigido e cancellazione file oltre `0.05 dB` (**PASS**) |

### 2.3 Verifica Acceptance Criteria `AC1..AC5`

- **AC1 (`TP 4x <= -0.30 dBTP` su WAV `PCM_24`):** **PASS** (100% dei test T1–T7 + 6/6 render pilot reali `<= -0.4567 dBTP`, margine >= `+0.1567 dB` sotto `-0.30 dBTP`, zero attivazioni del micro-trim offline).
- **AC2 (`Sample Peak <= -0.30 dBFS`, zero release leak):** **PASS** (`SP <= -0.50 dBFS` su T1–T4 e `[-0.699, -0.500] dBFS` sui 6 render reali).
- **AC3 (Nessuna regressione inattesa di LUFS / PLR / M-S / bilancio stereo):** **PASS** (su segnali bassi T5 invariato al bit `< 1e-12`; sui 6 render reali calo medio di loudness soltanto `-0.003 .. -0.046 LU` rispetto al limiter `1x` non conformante, evitando il crollo `-0.6575 LU` dell'attenuazione lineare globale V2).
- **AC4 (`TOTAL_CHAIN_LATENCY_SAMPLES = 159`, lunghezza e stereo-link invariati):** **PASS** (JUCE `12/12 PASS`, durata `63.90s / 71.47s / 71.78s` invariata al campione, `|delta_M/S| <= 0.006 dB`, `|delta_corr| <= 0.00028`).
- **AC5 (Coerenza C++ ↔ Python ↔ Disco `PCM_24` + autograd finiti):** **PASS** (scarto memoria vs disco `<= 0.0006 dB` sui 6 render, autograd finiti su `z_param`, parità full-chain `<= -100 dB`).

---

## 3. Confronto Pre / Post Implementazione sui 6 Render Congelati del Pilot V3c (`track01..track03`)

Senza eseguire alcuna nuova ottimizzazione Adam, sono stati ri-renderizzati i 6 vettori congelati `theta` (`Control V0p` e `Treatment V3c`) estratti da `docs/sprint3/SPRINT3_V3C_ABLATION_PILOT_RESULTS.json` sia con percorso esplicito `legacy_sample_1x` (a conferma della riproducibilità al bit dei valori storici) sia con la nuova catena Sprint 4 (`true_peak_4x` @ `-0.50 dBTP` + `export_validated_pcm24_wav`):

| Traccia | Arm | Storico JSON (`1x`) TP (dBTP) | Riprodotto `legacy_sample_1x` TP (dBTP) | **Sprint 4 `PCM_24` TP (dBTP)** | **Sprint 4 `PCM_24` SP (dBFS)** | LUFS (`1x` $\to$ `TP4x`) | $\Delta$LUFS (LU) | PLR (`1x` $\to$ `TP4x` dB) | $\Delta$M/S (dB) | $\Delta$Mono Drop (dB) | Guard `PCM_24` (`trim` dB) |
|:---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| `track01` | `V0p` | `+0.265` | `+0.265` | **`-0.458`** | **`-0.500`** | `-10.51` $\to$ `-10.56` | `-0.046` | `10.21` $\to$ `10.06` (`10.10` TP-PLR) | `-0.006` | `+0.0007` | `pass` (`0.0000`) |
| `track01` | `V3c` | `+0.307` | `+0.307` | **`-0.457`** | **`-0.500`** | `-10.51` $\to$ `-10.56` | `-0.046` | `10.21` $\to$ `10.06` (`10.10` TP-PLR) | `-0.006` | `+0.0007` | `pass` (`0.0000`) |
| `track02` | `V0p` | `-0.071` | `-0.071` | **`-0.528`** | **`-0.699`** | `-18.04` $\to$ `-18.04` | `-0.003` | `17.74` $\to$ `17.34` (`17.51` TP-PLR) | `+0.001` | `-0.00004` | `pass` (`0.0000`) |
| `track02` | `V3c` | `-0.071` | `-0.071` | **`-0.528`** | **`-0.699`** | `-18.04` $\to$ `-18.04` | `-0.003` | `17.74` $\to$ `17.34` (`17.51` TP-PLR) | `+0.001` | `-0.00004` | `pass` (`0.0000`) |
| `track03` | `V0p` | `-0.087` | `-0.087` | **`-0.489`** | **`-0.511`** | `-13.50` $\to$ `-13.51` | `-0.004` | `13.20` $\to$ `12.99` (`13.02` TP-PLR) | `-0.001` | `+0.00025` | `pass` (`0.0000`) |
| `track03` | `V3c` | `-0.084` | `-0.084` | **`-0.490`** | **`-0.511`** | `-13.50` $\to$ `-13.51` | `-0.004` | `13.20` $\to$ `13.00` (`13.02` TP-PLR) | `-0.001` | `+0.00024` | `pass` (`0.0000`) |

**Osservazioni chiave:**
1. **`legacy_sample_1x` bit-fedele:** su tutti e 6 i render la riproduzione esplicita `legacy_sample_1x` coincide al millesimo di dB (e al micro-dB nel JSON) con gli artefatti congelati Sprint 3.
2. **Efficacia chirurgica della Soluzione 1 rispetto al workaround V2:** nel workaround V2, per portare `track01` sotto `-0.35 dBTP` era stato necessario attenuare staticamente di `-0.6575 dB`, perdendo `-0.6575 LU` su tutto il brano (`-11.17 LUFS`). Con il nuovo True Peak Limiter 4× a `-0.50 dBTP`, `track01` passa da `+0.307 dBTP` a **`-0.457 dBTP`** perdendo soltanto **`0.046 LU`** di loudness integrata e preservando al millesimo di dB la firma stereo (`|delta M/S| <= 0.006 dB`, `|delta corr| <= 0.00028`).
3. **Coerenza memoria vs disco `PCM_24`:** su tutti e 6 i render, `|TP_mem - TP_pcm24| <= 0.0006 dB` (`trim_applied_db == 0.0000` sempre).

---

*Fine Commit 1 — Implementazione e Verifica Sprint 4.*
