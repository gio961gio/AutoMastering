# SPRINT 4 — AUDIT ARCHITETTURALE E PROGETTO DI HARDENING TRUE PEAK / INTER-SAMPLE PEAK

- **Repository:** `gio961gio/AutoMastering`
- **Branch / Baseline codice congelata:** `human-validation-package` @ `3fdeb68ff6626125781dbc3fcdcf568c1534b765`
- **Stato V3c:** pilot `same_track` chiuso su baseline congelata; `enable_v3c=False` invariato; nessun training o ottimizzazione eseguita.
- **Requisito contrattuale di consegna:** **True Peak 4x sul WAV `PCM_24` finale $\le -0.30\text{ dBTP}$** (`-0.50 dBTP` valutato come margine interno di progetto).
- **Modalità documento:** **SOLO AUDIT READ-ONLY E DESIGN** — nessuna modifica a codice DSP/loss/pesi/parametri latenti e nessun commit di codice (solo consegna documentale su GitHub).

---

## FASE A — AUDIT READ-ONLY DELL'ARCHITETTURA ESISTENTE

### 1. Mappa completa del percorso audio (Input $\to$ Export `PCM_24`)

Il sistema esegue la stessa catena a 6 stadi su tre superfici:
1. **Training / Autograd Python** (`DifferentiableMasteringChain`, `python/ddsp_mastering/dsp/chain.py:11-137`);
2. **Engine C++ Streaming & Binding Offline** (`core_dsp::MasteringChain`, `cpp/core_dsp/include/core_dsp/MasteringChain.hpp`, esposto tramite `PyMasteringChain` in `cpp/bindings/py_core_dsp.cpp:405-460` e orchestrato da `render_offline_with_tail_flush` in `scripts/human_validation/offline_renderer.py`);
3. **Plugin Live JUCE VST3/Standalone** (`MasteringPluginAudioProcessor::processBlock`, `cpp/plugin/source/PluginProcessor.cpp:142-193`).

```text
Input Stereo [2, N]
  ├─► Python: torch.float32 (promosso internamente a float64 nei biquad/comp/imager/limiter)
  └─► Offline C++ / JUCE: float32 ([2, N], tail flush +159 campioni zero in offline_renderer.py)
        │
        ▼
[Stage 1] 6-Band Parametric EQ (IIR Biquad Cascade / FFT in early steps)
          f0 [20..20k Hz], gain_db [-6.0, +6.0] dB, Q [0.3, 8.0]  | Latenza: 0 smp
        │
        ▼  (C++ inter-stage scratch float32)
[Stage 2] 3-Way Phase-Compensated Linkwitz-Riley 4th-Order Crossover (LR4)
          crossover_f_low [40, 600 Hz], crossover_f_high [800, 12000 Hz] | Latenza: 0 smp
        │  └─► band_low [2, N], band_mid [2, N], band_high [2, N]
        ▼
[Stage 3] 3-Band Stereo-Linked Log-Domain Soft-Knee Compressor
          thresh [-40, 0] dB, ratio [1, 12], att [1, 100] ms, rel [10, 500] ms,
          makeup_db [-3.0, +6.0] dB per banda (knee fissa 6.0 dB)  | Latenza: 0 smp
        │
        ▼
[Stage 4] 3-Band Mid/Side Stereo Imager + Somma delle 3 bande
          width_low [0.4, 1.1], width_mid [0.7, 1.4], width_high [0.7, 1.4]
          L' = (M + w*S)/sqrt(2), R' = (M - w*S)/sqrt(2); y_sum = low' + mid' + high'
        │  └─► Gain combinato potenziale da width > 1 su segnale decorrelato/S + somme di fase LR4
        ▼
[Stage 5] 2x Oversampled Harmonic Saturator (`DifferentiableHarmonicSaturator` / `HarmonicSaturator`)
          drive_db [0.0, 6.0] dB (spec), saturator_mix [0.0, 1.0],
          31-tap Kaiser halfband FIR (beta=8.5), tanh(gd*u)/gd, percorso dry ritardato di 15 smp
        │  └─► Latenza: 15 campioni (FIR_HALFBAND_LATENCY)
        ▼
[Stage 6] Lookahead Limiter (`DifferentiableLookaheadLimiter` / `LookaheadLimiter`)
          push_gain_db [0.0, 8.0] dB (lineare = 10^(push_db/20))
          Detector: SOLO SAMPLE PEAK 1x: max(|u_L|, |u_R|, 1e-7) vs C_LIN = 10^(-0.3/20) = 0.966050879
          Smoothing: Min-Filter (144) -> Release 80ms -> Boxcar Moving Average (144)
          Audio Delay: 144 campioni (`LIMITER_LOOKAHEAD_SAMPLES`)
        │  └─► Latenza totale dichiarata catena: 159 campioni (15 + 144)
        ▼
Uscita Catena DSP
  ├─► VST3 Live: JUCE AudioBuffer<float> -> DAW / DAC (True Peak non controllato)
  └─► Offline Renderer (`offline_renderer.py`):
        py_core_dsp ritorna float32 -> slice [:, 159 : 159 + N] -> cast float64 ->
        `evaluate_v3c_ablation.py`:
          • Benchmark interno: `.astype(np.float32)` su `[mix_start : mix_start + eval_len]`
          • Export disco: `sf.write(..., np.transpose(audio_64), 48000, subtype="PCM_24")`
```

---

### 2. Inventario di tutti gli stadi di gain, non linearità, limiter e conversione formato

Dalla verifica di `cpp/core_dsp/include/core_dsp/ParameterSpec.hpp`, `Constants.hpp`, `HarmonicSaturator.hpp`, `MultibandComp.hpp`, `MidSideImager.hpp`, `LookaheadLimiter.hpp`, `MasteringChain.hpp`, `py_core_dsp.cpp`, `offline_renderer.py` e `evaluate_v3c_ablation.py`:

| # | Modulo / File | Tipo di operazione | Range / Costante osservata | Rischio verso TP / Clipping |
|:---|:---|:---|:---|:---|
| 1 | `BiquadCascade.hpp` / `eq.py` | Boost/cut EQ IIR 6 bande (Low-Shelf, 4 Peaking, High-Shelf) | `gain_db` $\in [-6.0, +6.0]\text{ dB}$, `Q` $\in [0.3, 8.0]$ | Può aumentare il picco e lo sfasamento di banda prima del crossover |
| 2 | `LinkwitzRiley4.hpp` / `crossover.py` | Split 3 bande LP4/HP4 + All-Pass 2° ordine su High | `f_low` $[40, 600]$, `f_high` $[800, 12000]\text{ Hz}$ | Rotazione di fase all-pass modifica la cresta transitoria quando le bande hanno gain/compressione diversi |
| 3 | `MultibandComp.hpp` / `compressor.py` | Compressione + Makeup gain lineare per banda | `comp_{low,mid,high}_makeup_db` $\in [-3.0, +6.0]\text{ dB}$ | Nei 3 pilot V1 contribuisce $+1.37 \div +1.42\text{ dB}$ per banda |
| 4 | `MidSideImager.hpp` / `imager.py` | Scaling componente Side M/S + somma 3 bande | `w_low` $[0.4, 1.1]$, `w_mid/high` $[0.7, 1.4]$ | Allargamento `w > 1` aumenta l'energia Side e i picchi L/R su segnali stereo fuori fase |
| 5 | `HarmonicSaturator.hpp` / `saturator.py` | Zero-stuffing 2x + $\tanh(g_d u)/g_d$ + FIR 31-tap + blend dry/wet (delay 15 smp) | `drive_db` $\in [0, 6]\text{ dB}$ (`ParameterSpec.hpp:102`), `mix` $\in [0, 1]$ | Soft-clipping riduce i picchi istantanei ma genera armoniche HF che aumentano l'energia sopra $f_s/4$; *rimane prima del limiter* |
| 6 | `LookaheadLimiter.hpp` / `limiter.py` | Pre-gain `push_gain_db` + lookahead gain reduction | `push_gain_db` $\in [0.0, 8.0]\text{ dB}$ (`ParameterSpec.hpp:106`); target `C_LIN = -0.30 dBFS` | **Unico stadio di controllo picchi**; opera **solo su sample peak 1x**, non su True Peak |
| 7 | `MasteringChain.hpp` / `py_core_dsp.cpp` | Narrowing `double -> float32` ad ogni bordo di stadio | IEEE-754 `float32` (`~1.19e-7` ULP) | Errore max ~1e-6 dB al ceiling; trascurabile rispetto agli ISP, ma rende il sample peak non esattamente $\le C_{\text{LIN}}$ al bit |
| 8 | `evaluate_v3c_ablation.py:548-549` (`sf.write`) | Quantizzazione 24-bit lineare PCM senza dithering né safety trim | `subtype="PCM_24"` (`[-1.0, 1.0 - 2^-23]`), zero clipping duro sotto `0 dBFS` se `max <= 0.966` | Nessuno stadio genera nuovi DSP overshoot dopo il limiter, ma quantizzazione + filtro DAC/4x ricostruiscono gli ISP già presenti nel treno di campioni |

---

### 3. Analisi dettagliata del Limiter attuale (Python vs C++)

Dall'audit diretto di `python/ddsp_mastering/dsp/limiter.py` e `cpp/core_dsp/include/core_dsp/LookaheadLimiter.hpp`:

1. **Detector esclusivamente Sample Peak 1x (zero oversampling):**
   - Sia Python (`limiter.py:88-90`) che C++ (`LookaheadLimiter.hpp:132-135`) calcolano:
     $$p[n] = \max(|u_L[n]|, |u_R[n]|, 10^{-7}), \qquad g_{\text{target}}[n] = \min\!\left(1.0,\; \frac{C_{\text{LIN}}}{p[n]}\right)$$
     con $C_{\text{LIN}} = 10^{-0.3/20} \approx 0.96605087870305$ (`-0.3000 dBFS`).
   - Nessun filtro di interpolazione (4x o superiore) è applicato nel detector. Qualsiasi segnale i cui **campioni discreti** rispettino $|u[n]| \le 0.96605$ riceve $0.000\text{ dB}$ di gain reduction anche quando il picco continuo inter-sample (ISP) supera $0\text{ dBTP}$.
2. **Lookahead, smoothing e release:**
   - Lookahead fisso: `LIMITER_LOOKAHEAD_SAMPLES = 144` (`3.0 ms` a `48 kHz`), finestra causale `(143, 0)` su `min_filter` (`std::deque` monotona O(1) ammortizzato in C++, `unfold(144)` in Python).
   - Release esponenziale one-pole: `LIMITER_RELEASE_MS = 80.0 ms`, $\alpha_{\text{rel}} = \exp\!\left(-1/(0.080 \cdot f_s)\right) \approx 0.999739617$ a `48 kHz`. Scan C++ custom `fast_forward_limiter_envelope` / `fast_backward_limiter_envelope` in `py_core_dsp.cpp:615-685` per parità e backprop esatto.
   - Moving average (boxcar) finale: finestra `144` campioni (`(143, 0)` in Python, circular buffer inizializzato a `1.0` in C++).
   - Linea di ritardo audio: `F.pad(u, (144, 0))[..., :N]` in Python; lettura da `delay_l_[delay_idx_]` di lunghezza `144` prima della scrittura in C++.
3. **Bug analitico di allineamento causale di 1 campione (`143` vs `144`) verificato sperimentalmente:**
   - Su `boxcar` + `min_filter` entrambi lunghi `144` campioni (che coprono indici `[n-143 .. n]`), mentre l'audio `u_delayed[n]` legge `u[n-144]`: quando un singolo impulso isolato di ampiezza $A > C_{\text{LIN}}$ cade a $k$, `g_min` vale $g_{\text{target}}[k]$ su $[k, k+143]$. Al campione $n = k + 144$, in cui l'impulso esce da `u_delayed`, il filtro boxcar media $g_{\text{rel}}[k+1 \dots k+144]$: i primi 143 termini valgono $g_{\text{target}}[k]$, ma l'ultimo termine $g_{\text{rel}}[k+144]$ **è già entrato in fase di release** ($g_{\text{rel}}[k+144] > g_{\text{target}}[k]$).
   - Questo introduce un leak deterministico esattamente pari a $+\frac{(1 - \alpha_{\text{rel}})(A - C_{\text{LIN}})}{144}$ lineare sull'uscita!
   - **Verifica empirica eseguita su Python & C++ (`audit_probe2.py`):**
     - Impulso $A = 1.0$: `max = 0.966050923`, `SP = -0.300000 dBFS`
     - Impulso $A = 2.0$: `max = 0.966052771`, `SP = -0.299983 dBFS` (`+1.7e-5 dB` sopra `-0.30`)
     - Impulso $A = 5.0$: `max = 0.966058195`, `SP = -0.299934 dBFS`
     - Impulso $A = 10.0$: `max = 0.966067195`, `SP = -0.299853 dBFS` (predizione analitica: `0.96606721` $\to$ match a 7 cifre significative!).
     - Se `u_delayed` usasse ritardo `143` campioni (o se `g_min` usasse finestra `145` / padding simmetrico allineato), la media coprirebbe esattamente $[k \dots k+143]$ garantendo $g_{\text{smooth}}[k+143] = g_{\text{target}}[k]$ e `SP <= -0.300000 dBFS` senza leak.
4. **Parità Python vs C++ e dipendenza da Sample Rate nel plugin VST3:**
   - Nei test deterministici eseguiti (`audit_probe.py`), Python `DifferentiableLookaheadLimiter` e C++ `LookaheadLimiter` (sia full-block che chunked 64 campioni) hanno dato scarto massimo **`|Py - Cpp| = 0.00e+00`** e **`|full - chunk64| = 0.00e+00`** su uscita `float32`.
   - **Nota architetturale per il VST3 live (`PluginProcessor.cpp:112-115` e `Constants.hpp:61`):** `MasteringChain::get_latency_samples()` restituisce la costante compile-time `TOTAL_CHAIN_LATENCY_SAMPLES = 159` e `LookaheadLimiter` mantiene `LOOKAHEAD = 144` campioni fissi anche chiamando `prepare(sampleRate, ...)` a `44.1 kHz` o `96 kHz` (verificato via `py_core_dsp.MasteringChain(96000)` $\to$ `159` campioni). A `96 kHz` il lookahead reale si dimezzerebbe a `1.5 ms` pur restando funzionale, mentre la release `alpha_rel` si adatta correttamente a `sample_rate`.
5. **Esistono stadi successivi che causano nuovi overshoot?**
   - **No a livello DSP interno** (il limiter è l'ultimo blocco della catena), **eccetto**:
     1. il **bug di 1 campione (`+1e-5` a `+1.5e-4 dB`) + arrotondamento `double -> float32`** al bordo di uscita (`pred_sample_peak_dbfs` osservato sui 6 render V1 pari a `-0.299999` / `-0.299994 dBFS`, cioè $+0.6 \div +5.8 \times 10^{-6}\text{ dB}$ sopra `-0.30 dBFS`);
     2. la **ricostruzione continua/4x tra i campioni** (overshoot fino a `+0.607 dB` sui render reali e fino a `+3.83 dB` su segnali HF sintetici);
     3. l'eventuale **quantizzazione `PCM_24`** (`LSB = 2^-23 ≈ 1.19e-7`), che comunque su questi segnali altera il True Peak di `< 1e-5 dB`.

---

### 4. Audit della misura `calculate_true_peak_dbtp` e confronto con `PCM_24`

Dall'ispezione di `python/ddsp_mastering/optimize/metrics.py:130-220` e `scripts/sprint3/evaluate_v3c_ablation.py:607-617`:

- **Implementazione (`calculate_true_peak_dbtp`):**
  - Esegue resampling `4x` (`192 kHz` da `48 kHz`) tramite `torchaudio.functional.resample(..., resampling_method="sinc_interp_kaiser", lowpass_filter_width=64, rolloff=1.0, beta=14.769656459379492)` in precisione **`torch.float64`**.
  - Applica zero-padding esplicito di `64` campioni ai bordi sinistro e destro del segnale e processa in chunk da `480,000` campioni con halo simmetrico di `128` campioni per evitare artefatti di concatenazione.
  - Restituisce `20 * log10(max(|x_4x|) + 1e-12)`.
- **Differenza rispetto al filtro ITU-R BS.1770-4 standard (48-tap 4x polyphase FIR a 4 fasi):**
  - Il filtro Kaiser a 64 zero-crossing (`beta ≈ 14.77`, attenuazione lobi laterali `> 120 dB`, `rolloff = 1.0`) è matematicamente un interpolatore sinc quasi ideale: misura un True Peak molto vicino al picco continuo reale dell'inviluppo banda-limitato, ma può risultare **leggermente più severo** (fino a `+0.1 .. +0.3 dB` vicino a Nyquist o sui fronti ripidi con Gibbs ringing) rispetto al filtro 4x da 48 tap di BS.1770-4 (`12 tap/fase`) che attenua leggermente sopra `~19 kHz`.
  - Prova empirica (`audit_probe.py` / `audit_probe2.py`): una sinusoide `Fs/4` (`12 kHz`, fase $\pi/4$) con finestra Hann priva di discontinuità ai bordi e ampiezza tale da avere `SP = -0.300000 dBFS` misura esattamente **`TP = +2.710300 dBTP`** (teorico $-0.30 + 3.0103 = +2.7103\text{ dBTP}$), mentre la stessa sinusoide con inizio rettangolare netto ai bordi raggiunge **`+3.5336 dBTP`** per effetto Gibbs della troncatura.
- **Coerenza in-memory (`float32`/`float64`) vs WAV `PCM_24` su disco:**
  - `evaluate_v3c_ablation.py:822` misura `bm_v0p`/`bm_v3c` sul segmento `[mix_start : mix_start + eval_len]` castato a `float32`, mentre l'export su disco (`sf.write`) scrive l'intero render `mastered_64` in `PCM_24`.
  - La verifica incrociata sui 6 WAV originali V1 (`docs/sprint3/blind_v3c_pilot_wavs/`) mostra coincidenza entro **`<= 0.000001 dBTP`** fra `SPRINT3_V3C_ABLATION_PILOT_RESULTS.json`, `SPRINT3_V3C_BLIND_AUDIO_MANIFEST.json` e la rilettura diretta in `float64` dal file `PCM_24`. La quantizzazione a 24 bit non introduce quindi derive apprezzabili di True Peak rispetto al buffer pre-export, ma **fissa nel file il picco inter-sample già presente all'uscita del limiter**.

---

### 5. Ricostruzione dei valori True Peak dei sei render originali V1 (nativi, senza V2)

Tutti i valori sottostanti provengono esclusivamente dagli artefatti nativi congelati V1 (`docs/sprint3/SPRINT3_V3C_ABLATION_PILOT_RESULTS.json`, `docs/sprint3/SPRINT3_V3C_BLIND_MAPPING.json`, `docs/sprint3/SPRINT3_V3C_BLIND_AUDIO_MANIFEST.json`) e dalla rilettura diretta in `float64` dei 6 WAV in `docs/sprint3/blind_v3c_pilot_wavs/`. **Nessun dato dei WAV attenuati V2 è stato utilizzato.**

| Sessione | Braccio | File blind V1 nativo | Integrated LUFS (BS.1770-5) | PLR (dB) | Sample Peak (dBFS) | True Peak 4x nativo (dBTP) | Violazione soglia `-0.30 dBTP` | Parametri chiave (`push` / `sat_drive, mix` / `comp_makeup [L,M,H]` / `imager [L,M,H]`) |
|:---|:---|:---|---:|---:|---:|---:|---:|:---|
| **`track01`** | **Control `V0p`** | `track01_variant_A.wav` | `-10.5121` | `10.2121` | `-0.299995` | **`+0.264710`** | **`+0.5647 dB` FAIL** | `0.00155 dB` / `0.00058 dB, 0.00006` / `[1.372, 1.398, 1.416]` / `[1.042, 1.112, 1.110]` |
| **`track01`** | **Treatment `V3c`** | `track01_variant_B.wav` | `-10.5103` | `10.2103` | `-0.299994` | **`+0.307430`** | **`+0.6074 dB` FAIL** | `0.00155 dB` / `0.00058 dB, 0.00006` / `[1.374, 1.397, 1.417]` / `[1.042, 1.111, 1.110]` |
| **`track02`** | **Control `V0p`** | `track02_variant_B.wav` | `-18.0399` | `17.7399` | `-0.2999996` | **`-0.071335`** | **`+0.2287 dB` FAIL** | `0.00156 dB` / `0.00066 dB, 0.00020` / `[1.405, 1.400, 1.410]` / `[1.023, 1.112, 1.111]` |
| **`track02`** | **Treatment `V3c`** | `track02_variant_A.wav` | `-18.0399` | `17.7399` | `-0.2999996` | **`-0.071320`** | **`+0.2287 dB` FAIL** | `0.00156 dB` / `0.00066 dB, 0.00020` / `[1.405, 1.400, 1.410]` / `[1.019, 1.112, 1.111]` |
| **`track03`** | **Control `V0p`** | `track03_variant_B.wav` | `-13.5017` | `13.2017` | `-0.299997` | **`-0.086810`** | **`+0.2132 dB` FAIL** | `0.00155 dB` / `0.00064 dB, 0.00020` / `[1.391, 1.398, 1.393]` / `[1.043, 1.112, 1.111]` |
| **`track03`** | **Treatment `V3c`** | `track03_variant_A.wav` | `-13.5018` | `13.2018` | `-0.299997` | **`-0.083792`** | **`+0.2162 dB` FAIL** | `0.00155 dB` / `0.00064 dB, 0.00020` / `[1.391, 1.398, 1.393]` / `[1.042, 1.112, 1.111]` |

**Osservazioni diagnostiche chiave dai 6 render nativi:**
1. **Tutti e 6 i render nativi violano `-0.30 dBTP`** (minimo `+0.213 dB` su `track03 V0p`, massimo `+0.607 dB` su `track01 V3c`), mentre tutti e 6 hanno `Sample Peak` bloccato a `-0.3000 dBFS` (entro `+5.8e-6 dB`).
2. **`push_gain_db` e `saturator` erano quasi inattivi** (`~0.00155 dB` push, `~0.0006 dB` drive): la spinta verso il limiter deriva essenzialmente da EQ + ~`1.40 dB` di `comp_makeup_db` + `imager_width` (`~1.11` mid/high) guidati da `weight_loudness = 0.8` verso target loudness più spinti.
3. Il fatto che su `track01` il solo riallineamento di fase/ampiezza fra bande tra `V0p` e `V3c` sposti il True Peak da `+0.2647` a `+0.3074 dBTP` (`+0.0427 dB` a parità di sample peak `-0.3000 dBFS`) dimostra in modo definitivo che la causa dominante è l'assenza di rilevazione oversampled nel limiter.

---

### 6. Causa più probabile degli overshoot (Classifica e diagnosi)

1. **Causa primaria (99.9% dell'overshoot osservato — `+0.21` a `+0.61 dB` reali, fino a `+3.83 dB` worst-case):**
   Il detector del `LookaheadLimiter` (Python e C++) opera esclusivamente a `1x` sui campioni discreti contro `C_LIN = -0.30 dBFS`. I picchi inter-sample reali delle forme d'onda banda-limitate sono invisibili al detector.
2. **Causa secondaria strutturale (fino a `+1.5e-4 dB` su impulsi estremi, `~6e-6 dB` sui render):**
   Disallineamento causale di `1` campione nel `LookaheadLimiter`: la combinazione `min_filter` `(143, 0)` + `boxcar` `(143, 0)` applica a `u[n-144]` una media di gain che include il primo campione già in release `g_rel[n]`, sommata al rounding `double -> float32` tra gli stadi C++ e l'uscita.
3. **Amplificatori operativi:**
   La loss `DifferentiableLoudnessLoss` (`weight_loudness = 0.8`) spinge il compressore (`makeup_db ~ +1.4 dB`) e il limiter contro il ceiling per recuperare LUFS verso i target commerciali (`~ -7 LUFS`), generando molti campioni prossimi a `-0.30 dBFS` con alta probabilità di ISP sulle creste ad alta frequenza e sui transienti di somma multi-banda.

---

## FASE B — CONFRONTO DELLE SOLUZIONI

### Soluzione 1 — True Peak Limiter nel percorso DSP (Python + C++)

**Architettura tecnica:**
- Sostituire il detector `1x` con un **detector oversampled `4x` causale polyphase FIR** conforme a **ITU-R BS.1770-4 Annex 2** (filtro prototipo 48-tap diviso in 4 fasi polifase da 12 tap per campione in ingresso $\to$ 4 valori interpolati $y_0, y_1, y_2, y_3$ per campione stereo L/R, senza allocazioni dinamiche, con stato ring-buffer di 12 campioni per canale).
- Calcolare il picco inter-sample su ogni intervallo campione:
  $$p_{\text{TP}}[n] = \max_{c \in \{L, R\},\, m \in \{0,1,2,3\}} |y_{c, m}[n]| \cup \{|u_L[n]|, |u_R[n]|\}$$
  (includendo anche il campione discreto originale `u[n]` per evitare qualsiasi piccola attenuazione passband `0.1 dB` del filtro 48-tap vicino a $f_s/2$, oppure usando direttamente il filtro Kaiser ad alta fedeltà / BS.1770-4 con **margine di progetto interno `CEILING_TP_DB`**, v. sotto).
- **Correzione del ritardo di 1 campione e compatibilità FIR polyphase:**
  - Il filtro 48-tap a 4 fasi introduce un ritardo di gruppo esatto di $5.5 / 6$ campioni `1x` nel sidechain, oppure con lookahead audio fisso a `144` campioni la finestra di `min_filter` può coprire comodamente sia la latenza del detector sidechain (~6 campioni) sia il lookahead principale, **allineando esattamente `u_delayed` in modo che nessuna finestra di smoothing includa un campione già in release** (es. ritardo audio `144` con sidechain compensato e `min_filter` + `boxcar` calibrati in modo che per ogni impulso $g_{\text{smooth}}[n_{\text{peak}}] \le g_{\text{target}}[n_{\text{peak}}]$ al bit).
  - Nota: se il sidechain FIR polifase è puro **detector** (non modifica il percorso audio principale tranne che tramite `g_smooth[n]`), la fase dell'audio non viene alterata se la compensazione di ritardo resta dentro la finestra lookahead di `144` campioni (`3.0 ms`), preservando intatta `TOTAL_CHAIN_LATENCY_SAMPLES = 159`! *Verifica attenta*: con 6 campioni di ritardo nel detector polyphase rispetto all'ingresso `u[n]`, per avere lookahead pieno prima che l'impulso arrivi all'uscita dell'audio delay line, l'audio delay line può compensare con `144 + 6 = 150` campioni (portando latenza a `165`) **oppure** sfruttare `144` campioni sul percorso audio e `min_filter` di lunghezza `138 + 6` (mantenendo `159` campioni invariati con `2.875 ms` di pre-attacco reale invece di `3.0 ms`!).
- **Valutazione punti chiave:**
  - *Trasparenza sui transienti:* la 3-stage envelope (min-filter + 80ms release + boxcar 3 ms) mantiene l'attacco graduale dolce senza hard-clipping né wave-shaping sulle alte frequenze; i transienti vengono attenuati dolcemente in anticipo sull'inviluppo ISP anziché tosati.
  - *Costo computazionale:* 4 fasi $\times$ 12 MAC per canale $\times$ 2 canali = **96 MAC/campione**. A `48 kHz` real-time (~`4.6 MFLOP/s`) il carico è **< 0.5% di un singolo core CPU moderno**, pienamente compatibile con VST3 zero-allocation; in Python autograd il detector 4x è implementabile con un singolo `F.conv1d(..., groups=2)` di kernel `48` e stride `1` (con peso congelato, o stop-gradient parziale/pieno a seconda della stabilità) molto veloce su CPU/GPU.
  - *Isomorfismo Python/C++:* lo stesso kernel FIR statico (es. 48 coefficienti BS.1770-4 o 64 coefficienti Kaiser `float64`) garantisce parità numerica stretta tra training PyTorch, renderer offline e plugin VST3 live.
  - *Limiti residui:* la discrepanza fra un FIR `48-tap` leggero in DSP e il sinc Kaiser `64 zero-crossing` (`beta=14.77`) di `calculate_true_peak_dbtp` può arrivare fino a `~0.05 – 0.15 dB` su segnali patologici vicino a `20–23 kHz`, più eventuali piccoli overshoot sui fronti con lookahead finito o sotto `float32` rounding. Per questo serve un margine interno sotto `-0.30 dBTP`.

### Soluzione 2 — Post-processing Safety Stage

**Architettura tecnica:**
- Dopo il rendering DSP completo (`render_offline_with_tail_flush`), misurare `TP_4x = calculate_true_peak_dbtp(mastered_64)` e, se `TP_4x > target_internal_dbtp`, applicare una **attenuazione lineare uniforme** (static gain trim):
  $$g_{\text{trim}} = 10^{(\text{target\_internal\_dbtp} - \text{TP}_{4\times}) / 20}$$
  seguita da esportazione `PCM_24` e ri-verifica su disco (`calculate_true_peak_dbtp(read_pcm24)`), eventualmente iterando con micro-step di sicurezza in caso di rounding LSB `PCM_24`.
- **Valutazione punti chiave:**
  - *Qualità audio:* **distorsione armonica esattamente nulla** (cambio di volume puro), nessuna alterazione di immagine stereo, timbro, fase o transienti relativi.
  - *Affidabilità del ceiling offline:* **100% garantita matematicamente** sul WAV `PCM_24` esportato, poiché la verifica finale avviene con lo stesso misuratore sul file letto da disco.
  - *Compromessi su Loudness e PLR:* il PLR (`TP - LUFS` o `SP - LUFS`) resta quasi invariato se misurato su True Peak, ma **l'Integrated LUFS scende esattamente di quanto si attenua**: ad esempio su `track01 V3c` (`TP = +0.3074 dBTP`, `LUFS = -10.51`), per scendere a `-0.30 dBTP` si perdono `-0.61 dB` di LUFS (`-11.12 LUFS`); per scendere a `-0.50 dBTP` si perdono **`-0.81 dB` di LUFS (`-11.32 LUFS`)**, aumentando l'errore rispetto al `target_lufs`!
  - *Compatibilità DSP nativo / Live VST3:* **NON applicabile in streaming real-time** nel plugin VST3 perché richiede di conoscere il picco massimo dell'intero brano prima dell'uscita (non-causale) e **non protegge il percorso live DAW/DAC** né migliora il limiter stesso (l'ottimizzatore continuerebbe a generare master fuori specifica che poi vengono abbassati passivamente di volume).

### Soluzione 3 — True Peak Penalty nella Loss

**Architettura tecnica:**
- Aggiungere nella loss totale (`MasteringPerceptualLoss` o helper separato) un termine differenziabile basato su `resample 4x` (o `conv1d` 4-phase) che penalizza l'eccesso rispetto a $\tau_{\text{TP}}$ (es. `-0.50 dBTP`):
  $$\mathcal{L}_{\text{TP}} = \mathbb{E}\!\left[\operatorname{ReLU}(\text{TP}_{4\times}(y) - \tau_{\text{TP}})^2\right] \quad \text{oppure via SoftMax/LogSumExp sui blocchi}$$
- **Valutazione punti chiave e conflitti quantificati con le loss esistenti:**
  1. **Impossibile come garanzia contrattuale:** l'ottimizzatore gira per **30 step Adam (`lr=0.04`) su chunk parziali con pre-roll**, mentre la validazione `MasteringBenchmarkSuite` valuta sull'intera traccia/eval window e sull'export `PCM_24`. Anche un singolo transiente non visto nei chunk di training o l'arresto prima della convergenza esatta può sforare `-0.30 dBTP`.
  2. **Conflitto diretto con `DifferentiableLoudnessLoss` (`weight_loudness = 0.8`):** `lufs_loss` spinge verso target commerciali alti (`-7 dBFS/-7 LUFS` nei test V1), alzando `makeup_db` e spingendo il segnale contro il ceiling del limiter. Se il limiter DSP ha detector `1x`, la loss TP dovrebbe contemporaneamente ridurre il livello spingendo in direzione opposta a `lufs_loss` e alla `plr_loss` (`plr_min = 8.0, plr_max = 14.0` in `box` mode e `Huber(pred_plr - ref_plr)` in V2).
  3. **Conflitto con `V3bDynamicPriorLoss`:** `L_lim` (`tau_avg`, `tau_95`) e `L_conflict` già penalizzano l'eccesso di gain reduction sul limiter e il makeup eccessivo.
  4. **Costo computazionale in autograd:** un resampling 4x con kernel `64`-tap o `48`-tap su tutti i chunk `[B, 2, T]` per ogni step backward quadruplica la memoria e i tempi di attivazione dello stadio finale rispetto al solo costo di forward C++.
  5. **Ruolo appropriato:** utile solo come *eventuale* regolarizzatore morbido futuro (se il limiter dovesse saturarsi troppo su ISP), ma non necessario né sicuro come barriera primaria finché `V0p` e `V3c` restano congelati.

### Matrice di confronto sintetico

| Criterio | Soluzione 1 (True Peak Limiter DSP) | Soluzione 2 (Post-Processing Safety Trim) | Soluzione 3 (True Peak Loss Penalty) |
|:---|:---|:---|:---|
| **Affidabilità ceiling `-0.30 dBTP`** | **Alta** (con margine interno `-0.50` o `-0.40 dBTP` + fix boxcar) | **Assoluta (100%) offline** sul file export; **0% live VST3** | **Insufficiente** (ottimizzazione stocastica a 30 step su chunk) |
| **Qualità audio / Transienti** | **Alta** (lookahead trasparente 3 ms + release 80 ms sugli ISP reali) | **Massima** (gain lineare statico), ma penalizza inutilmente la loudness di tutto il brano | Buona se debole, ma può sotto-comprimere o deformare il bilanciamento tonale per evitare picchi |
| **Costo computazionale** | **Basso** (<0.5% CPU live C++; minimo overhead in Python) | **Medio offline** (un passaggio `calculate_true_peak_dbtp` + ri-export eventuale); N/A real-time | **Alto in training** (4x attivazioni in autograd per ogni step) |
| **Stabilità numerica & Determinismo** | **Deterministica**, parità Python/C++ con kernel polyphase identico | **Deterministica** su `float64` / `PCM_24` | Dipende da lr, step, chunk sampling, peso loss |
| **Impatto su Loudness / PLR** | **Ottimale**: attenua *solo* gli istanti con ISP, preservando il LUFS integrato del resto del brano | **Penalizzante**: abbassa l'intera traccia dello sforo massimo (es. `-0.81 LU` su `track01`) | Indiretto, confligge con `weight_loudness=0.8` |
| **Compatibilità Python + C++ + VST3** | **Totale** (unica sorgente di verità DSP) | **Solo Offline Exporter** (il VST3 rimane privo di TP guard) | **Solo Training** (VST3/parametri esterni/manuali non protetti) |

---

## ARCHITETTURA RACCOMANDATA, TARGET NUMERICI E MARGINI

### Architettura raccomandata: Ibrida difensiva (`Soluzione 1` primaria + `Soluzione 2` guard rail offline di verifica/emergenza)

1. **Livello 1 (Core DSP nativo — C++ & Python allineati): True Peak Lookahead Limiter 4x**
   - Dotare `LookaheadLimiter` (C++) e `DifferentiableLookaheadLimiter` (Python) di un **detector 4x polyphase FIR causale** condiviso (coefficienti costanti esatti in `Constants.hpp` / `limiter.py`), includendo nel `max` di picco sia i 4 sotto-campioni interpolati sia il campione `1x` originale per sicurezza passband vicino a Nyquist.
   - **Correggere il bug causale di 1 campione tra `boxcar` e `u_delayed`**, garantendo che per qualsiasi impulso o gradino la gain reduction copra esattamente l'intero supporto del picco senza leak del primo campione di release.
   - Esporre eventuale modalità detector selezionabile o bump di versione controllato dopo approvazione, preservando identica interfaccia a 41 parametri (`PARAM_SPECS` / `THETA_NEUTRAL` invariati).
   - **Gestione `calculate_true_peak_dbtp` vs detector DSP:** Poiché il limiter DSP real-time userà un FIR causale compatto (es. 48-tap BS.1770-4, 12 tap/fase) mentre il benchmark finale usa un sinc Kaiser lungo (`lowpass_filter_width=64, beta=14.77`), può esistere un residuo scarto di filtro sui segnali molto vicini a `20 kHz` o ai transienti ripidi, oltre all'arrotondamento `double -> float32` tra gli stadi C++ e alla quantizzazione `PCM_24`.
2. **Livello 2 (Offline Export Guardrail — Defense-in-Depth su disco):**
   - Nello script/pipeline di export `PCM_24`, **non sostituire il DSP**, ma verificare sempre `calculate_true_peak_dbtp` direttamente dal `PCM_24` scritto su disco (`<= -0.30 dBTP`).
   - Solo se una tolleranza estrema di filtro (`sinc_kaiser` vs `polyphase_48`) producesse un micro-residuo sopra `-0.30 dBTP` (evento atteso raro grazie al margine interno DSP), applicare e **loggare esplicitamente** un micro-safety-trim di emergenza + flag diagnostico (mai silenziato come se non fosse successo nulla).
3. **Livello 3 (Loss): Nessuna modifica ai pesi `V0p` / `V3c` / `V3cStereoCompatibilityLoss`.**
   - La Soluzione 3 resta **non raccomandata** in questa fase per rispettare i vincoli di congelamento di V0p/V3c ed evitare regressioni su STFT/LUFS/PLR. Quando il limiter DSP agirà direttamente sugli ISP durante il forward pass, l'ottimizzatore riceverà già automaticamente il comportamento reale del limiter True Peak nel grafo differenziabile (se Python e C++ usano lo stesso detector 4x).

### Target numerici e valutazione del margine `-0.50 dBTP` vs `-0.30 dBTP`

- **Requisito contrattuale di consegna (invariato):**
  $$\text{True Peak 4x}_{\text{PCM\_24 on disk}} \le -0.30\text{ dBTP}$$
- **Valutazione del margine interno di progetto a `-0.50 dBTP`:**
  - **Pro di `-0.50 dBTP` (`CEILING_INTERNAL_DBTP = -0.50`, ovvero `C_LIN_TP ≈ 0.944060876`):**
    1. Fornisce **`0.20 dB` di headroom reale** contro: (a) differenza tra filtro detector DSP real-time (48-tap BS.1770-4) e interpolatore severo `sinc_interp_kaiser` (`beta=14.77, width=64`) del benchmark; (b) `float32` inter-stage rounding (`~1e-5 dB`); (c) quantizzazione `PCM_24` (`~1e-6 dB`); (d) lieve sovra-oscillazione di Gibbs su troncamenti netti a bordo blocco o DAC reali consumer non ideali.
    2. Evita praticamente a zero l'attivazione del trim post-export offline su musica reale e rende il plugin **VST3 live** robusto verso DAC hardware / codec streaming lossy (AAC/Opus/MP3 che notoriamente aggiungono `+0.2 .. +0.5 dB` di overshoot).
  - **Contro / Costo di `-0.50 dBTP` vs `-0.40 dBTP`:**
    - Riduce il picco ammesso di soli `0.20 dB` rispetto a `-0.30 dBTP` (`0.10 dB` rispetto a `-0.40 dBTP`). Nei punti dove il brano è limitato dai picchi inter-sample, l'Integrated LUFS massimo raggiungibile prima che intervenga il limiter scende al più di `<= 0.20 LU` (e solo sui picchi reali, **non** sull'intero brano come nella Soluzione 2!).
  - **Raccomandazione numerica:** Adottare **`-0.50 dBTP` come ceiling interno DSP di progetto** (`LIMITER_CEILING_DBTP = -0.50`) e mantenere **`<= -0.30 dBTP` come gate rigido di consegna e test**. Se in benchmark empirico un detector Kaiser-like allineato riducesse lo scarto DSP-vs-metrica sotto `0.02 dB`, il margine `-0.50 dBTP` resterebbe comunque una scelta sicura di mastering professionale (`-0.5` / `-1.0 dBTP` standard EBU/AES/streaming) e verrebbe comunque documentato con dati prima di qualsiasi tuning.

---

## FASE C — PIANO DI VERIFICA DETERMINISTICO E ACCEPTANCE CRITERIA

### 1. Test deterministici proposti (`tests/test_sprint4_true_peak_guardrail.py` & C++ / Python parity)

Tutti i test vanno eseguiti **sia su Python (`DifferentiableLookaheadLimiter` / `DifferentiableMasteringChain`) sia su C++ (`py_core_dsp.LookaheadLimiter` / `MasteringChain`)**, includendo il round-trip su disco **`PCM_24`** tramite `calculate_true_peak_dbtp(..., oversampling_factor=4)`:

| # | Categoria segnale | Specifica deterministica proposta (`48 kHz`, stereo) | Comportamento atteso |
|:---|:---|:---|:---|
| T1 | **Sinusoidi HF multi-fase** | Frequenze `12 kHz` ($f_s/4$), `16 kHz`, `18 kHz`, `20 kHz`, `22 kHz`; fasi $\{0, \pi/8, \pi/4, \pi/2\}$; ampiezze `[0.966, 1.35, 1.50]` con ramp-in/ramp-out Hann breve per isolare il comportamento steady-state dal bordo rettangolare | `TP 4x <= -0.30 dBTP` su tutte le frequenze e fasi (target interno `<= -0.50 dBTP`); nessun campione singolo sfuggito al detector a fase $\pi/4$ |
| T2 | **Transienti & Multitone** | (a) Impulsi isolati $A \in \{1.0, 2.0, 5.0, 10.0\}$; (b) treno di impulsi bipolari distanziati di `143`, `144`, `145` campioni; (c) multitone denso (somma 8 toni $100\text{ Hz} \dots 18\text{ kHz}$ a fasi coerenti e random) | `SP <= target_lin` senza leak del campione di release (`A=10.0` deve restare strettamente $\le \text{target}$); `TP 4x <= -0.30 dBTP`; zero distorsione da discontinuità |
| T3 | **Near-limit ISP (zero GR a `1x`)** | `Fs/4` fase $\pi/4$ scalata così che `max(|u[n]|) = -0.3001 dBFS` (nel vecchio limiter attivava `GR = 0 dB` generando `+2.71 dBTP` steady-state / `+3.53 dBTP` con edge) | Il nuovo detector **deve rilevare** l'ISP oltre soglia, applicare attenuazione lookahead pulita e portare `TP 4x <= -0.30 dBTP` |
| T4 | **Stereo asimmetrico L/R** | Canale L = `12 kHz` $\pi/4$ ad alta ampiezza (`1.4`), Canale R = `440 Hz` a `-26 dBFS`; più caso inverso (R caldo, L silenzioso) | Stereo-linked peak detection applica identica riduzione L/R, nessun sbilanciamento M/S sui segnali stereo bilanciati e `TP 4x <= -0.30 dBTP` su entrambi i canali |
| T5 | **Buffer & Chunk Boundaries** | Elaborazione in un blocco unico vs streaming a blocchi da `1`, `16`, `64`, `127`, `512`, `1024` campioni; verifica across-block degli ISP a cavallo del confine di blocco; verifica a `44.1 / 48 / 96 kHz` | Differenza `|single_block - chunked| <= 1e-6` (`float32`), zero click o discontinuità di stato polyphase/lookahead ai bordi, `TP 4x <= -0.30 dBTP` |
| T6 | **Silenzio e bassissimo livello** | (a) Zeri esatti `[2, 8192]`; (b) DC/sinusoide a `-90 dBFS` e `-120 dBFS` | Silenzio $\to$ uscita esattamente `0.0` senza `NaN/Inf`; segnale `-90 dBFS` $\to$ `gain == 1.0` bit-esatto entro tolleranza di arrotondamento (zero pompaggio/alterazione) |
| T7 | **Pre-export vs Post-`PCM_24`** | Confronto `TP 4x` & `SP` tra output DSP in memoria (`float32`/`float64`) e file WAV `PCM_24` riletto da disco (`sf.read`) | Delta `|TP_mem - TP_pcm24| <= 0.005 dB` e **`TP_pcm24 <= -0.30 dBTP`** sempre soddisfatto |

### 2. Test di regressione e Acceptance Criteria (5 Gate obbligatori)

Quando l'implementazione verrà autorizzata, la PR/commit dovrà superare con 100% di successo tutti e 5 i criteri seguenti oltre all'intera suite esistente (`225 passed` baseline senza regressioni sui parametri neutri e sotto soglia):

1. **Compliance True Peak finale:** `True Peak 4x <= -0.30 dBTP` su tutti i WAV `PCM_24` finali (target interno di progetto `-0.50 dBTP` verificato con margine positivo almeno `>= +0.20 dB` sotto `-0.30 dBTP` sui segnali di test standard).
2. **Assenza totale di sample-peak clipping:** `Sample Peak <= -0.30 dBFS` (nessun singolo campione `>= 1.0` e zero leak di release su impulsi isolati fino a `+20 dBFS` / `A = 10.0`).
3. **Nessuna regressione inattesa di Loudness e PLR:** su segnali che *non* violano il ceiling True Peak (`TP < -0.50 dBTP`), la differenza rispetto al DSP baseline è **nulla / < 1e-4 dB** su `LUFS`, `PLR`, `M/S ratio` e `STFT`; sui segnali con ISP reali, nessun calo di `LUFS` superiore al necessario fisico per riportare il picco sotto il ceiling interno.
4. **Invarianza stereo/mono, latenza e durata:** latenza di catena allineata e deterministica (`TOTAL_CHAIN_LATENCY_SAMPLES` esatta tra Python, C++ `offline_renderer.py`, e VST3 `setLatencySamples`), lunghezza campioni preservata, zero alterazione del bilanciamento stereo L/R o `M/S` (`gain` strictly stereo-linked), bypass bit-exact latency-aligned confermato.
5. **Coerenza misura interna vs file esportato:** parità Python vs C++ (`max|Py - Cpp|` entro tolleranza `float32` deterministica) e coerenza stretta (`<= 0.005 dB`) tra `MasteringBenchmarkSuite` in memoria e misura indipendente su `PCM_24` riletto da disco, inclusa verifica autograd finiti (`loss.backward()` privo di `NaN/Inf`) in Python.

---

## INVENTARIO FILE E MODULI COINVOLTI NELL'EVENTUALE IMPLEMENTAZIONE

*(Elenco puramente informativo — **nessuno** di questi file è stato modificato in questo audit)*

1. **Costanti & Spec Core C++ (Source of Truth condivisa):**
   - `cpp/core_dsp/include/core_dsp/Constants.hpp` — costanti `LIMITER_CEILING_DBFS` / nuovo `LIMITER_CEILING_DBTP`, eventuali coefficienti 4x polyphase FIR e verifica costante latenza.
   - `cpp/core_dsp/include/core_dsp/LookaheadLimiter.hpp` — aggiunta ring buffer / detector 4x polyphase causale senza allocazioni heap + correzione allineamento causale di 1 campione tra `boxcar`/`min_filter` e linea di ritardo audio.
   - `cpp/core_dsp/include/core_dsp/MasteringChain.hpp` — eventuale propagazione costante latenza aggiornata (se variata) e verifica reset/prepare a multi-sample-rate.
2. **Moduli Python DDSP (Training & Parità):**
   - `python/ddsp_mastering/dsp/limiter.py` — detector 4x polyphase identico al C++ (`F.conv1d` / padding allineato) + fix allineamento `(143, 0)` vs `(144, 0)` + verifica gradienti autograd.
   - `python/ddsp_mastering/dsp/chain.py` — solo se cambia `latency_samples` (auspicabile mantenerlo a `159` compensando la fase polyphase dentro il lookahead `144`, evitando impatti su tutti i test/loss che usano `159`!).
   - `cpp/bindings/py_core_dsp.cpp` — eventuale esposizione helper/coefficienti True Peak per verificare parità esatta Python/C++ nei test unitari (es. `get_tp_fir_coeffs`).
3. **Exporter / Benchmark / Plugin:**
   - `scripts/human_validation/offline_renderer.py` — eventuale validazione opzionale / reporting latenza.
   - `python/ddsp_mastering/optimize/metrics.py` — verifica coerenza `true_peak_ceiling_dbtp` e reportistica (mantenendo `-0.30 dBTP` come ceiling di consegna e distinguendo l'eventuale soglia interna).
   - `cpp/plugin/source/PluginProcessor.cpp` — eredità automatica da `core_dsp::MasteringChain`; verifica `setLatencySamples` rispetto al sample rate host se diverso da `48 kHz`.
4. **Nuovi test dedicati (Phase C):**
   - `tests/test_sprint4_true_peak_guardrail.py` (e/o estensione mirata `python/tests/test_cpp_parity.py`) — test deterministici `T1..T7`, parità Python/C++, autograd e round-trip `PCM_24`.
5. **File esplicitamente ESCLUSI (invariati per vincolo):**
   - `python/ddsp_mastering/loss/stereo_prior.py` (`V3cStereoCompatibilityLoss`), `python/ddsp_mastering/loss/dynamic_prior.py`, `python/ddsp_mastering/loss/combined.py`, `python/ddsp_mastering/loss/loudness.py`, `python/ddsp_mastering/optimize/optimizer.py` (default `enable_v3c=False`), `ParameterSpec.hpp` (41 parametri latenti invariati), tutti gli artefatti storici `docs/sprint3/*`.

---

## STATO FINALE — STOP OPERATIVO

- **Documento prodotto:** `docs/sprint4/SPRINT4_TRUE_PEAK_ISP_HARDENING_AUDIT.md`
- **Modifiche a codice DSP, loss, parametri, pesi o risultati storici:** **Nessuna.**
- **Training o ottimizzazioni eseguite:** **Nessuna.**
- **Commit di codice creati:** **Nessuno** (solo questo commit documentale `.md` sopra la baseline codice congelata `3fdeb68ff6626125781dbc3fcdcf568c1534b765`).
- In attesa di revisione tecnica e autorizzazione esplicita prima di avviare l'implementazione.
