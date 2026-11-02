# AutoMastering_Independent_Technical_Audit.md

**Politecnico di Milano — DEIB | Master of Science in Sound and Music Engineering**

**Project**: AutoMastering (White-Box DDSP Differentiable Twin Engine)

**Candidate**: Giovanni Antonecchia

**Audited Commit**: `874550c5a72245a849b69c91fd8493e546662954`

**Audit Scope**: Full Codebase Static & Mathematical Verification (Gate A, Sprint 2, Sprint 2.1, S2.1-FINAL)

**Date**: November 2, 2026

**Overall Verdict**: **CONDITIONAL GO** (0 Critical P0, 4 High P1, 5 Medium P2, 3 Low P3)

---

## 1. Executive Verdict

Il progetto **AutoMastering** al commit `874550c5a72245a849b69c91fd8493e546662954` presenta un'architettura ingegneristica solida, ben modulare e matematicamente rigorosa sul piano dell'isomorfismo tra il motore **C++20 Core DSP** e il gemello differenziabile **PyTorch DDSP**. L'introduzione del protocollo duale (`same_track` vs `stylistic`), la stratificazione multi-chunk e l'indicizzazione temporale corretta in **S2.1-FINAL** dimostrano una maturità metodologica elevata, con una separazione netta tra il ritardo di file esterno e la latenza interna deterministica di 159 campioni ($3.31\text{ ms}$ a $48\text{ kHz}$).

Tuttavia, un'analisi critica indipendente che va oltre il semplice superamento dei 97 test automatizzati su GitHub Actions ha portato alla luce **4 difetti di priorità P1 (High)** e **5 debolezze di priorità P2 (Medium)**. In particolare, permangono discrepanze sostanziali tra quanto dichiarato nei report/commenti e quanto effettivamente calcolato dal codice su **ITU-R BS.1770-4 LUFS** (assenza di gating assoluto/relativo e convoluzione circolare non paddata), **True Peak** (misurato come mero Sample Peak a $1\times f_s$ senza oversampling nel limiter e nella suite di benchmark) e sull'integrazione del plugin **JUCE 8 VST3** (il `MasteringPluginAudioProcessor` istanzia `GenericAudioProcessorEditor` ignorando la GUI custom `MasteringPluginAudioProcessorEditor`, non espone il parametro di True Bypass all'host e presenta la latenza hardcoded a 159 campioni anche per sample rate diversi da $48\text{ kHz}$).

Il passaggio allo **Sprint 3** è autorizzato sotto condizione (**CONDITIONAL GO**): le criticità **P1** devono essere sanate prima o contestualmente all'apertura dello Sprint 3 per garantire che la nuova funzione di loss psicoacustica e i benchmark della tesi poggino su metriche conformi agli standard internazionali.

---

## 2. Architecture Review

### 2.1 Ricostruzione del Flusso Dati End-to-End

1. **Audio Input & Verification**: `scripts/run_mastering.py` carica i file stereo a $48\text{ kHz}$ in formato float32 `[2, T]` senza troncamento (`duration_sec=None` di default). `verify_audio_pair` valida dimensionalità e frequenza di campionamento.


2. **Reference Protocol (`reference_protocol.py`)**:


* In modalità `same_track`, `estimate_time_alignment` calcola la cross-correlazione normalizzata (primi 15s, media mono zero-mean) e raffina il picco con interpolazione parabolica sub-sample $\delta \in [-0.5, 0.5]$. `apply_same_track_alignment` allinea i supporti temporali, applica il filtro FIR sinc finestrato Blackman a 65 tap (`apply_fractional_delay`) e rifila simmetricamente i bordi di $K=32$ campioni.


* In modalità `stylistic`, qualsiasi allineamento temporale o confronto punto-a-punto viene inibito (`assert_stylistic_guards`).




3. **Multi-Chunk Selection (`chunking.py`)**: `MultiChunkSelector` esplora una griglia densa (hop $0.5\text{ s}$), applica un gate energetico ($\text{RMS} \ge -42\text{ dBFS}$), suddivide la timeline in $K$ zone stratificate alternando la massimizzazione di RMS, Crest Factor e diversità di Spectral Centroid (calcolato via STFT multi-frame 2048/512 sull'intero chunk) e impone `min_separation_samples` ricalcolando le feature in caso di spostamento.


4. **PyTorch DDSP & Optimization (`optimizer.py`, `chain.py`)**: L'ottimizzatore inizializza i 41 parametri nello spazio logit non vincolato $u = \text{logit}(\text{clamp}(z_{\text{neutral}}, 10^{-4}, 1-10^{-4}))$ e aggiorna $z = \sigma(u)$ tramite Adam + CosineAnnealingLR e gradient clipping ($\text{norm} \le 1.0$). Usa l'approssimazione EQ FFT per il primo 70% degli step e commuta in modalità IIR esatta per il 30% finale.


5. **Full-Track Rendering (`py_core_dsp.MasteringChain`)**: I parametri ottimali $z^*$ vengono iniettati nel motore C++20 che elabora l'intera traccia `raw_full` in doppia precisione interna.


6. **Benchmark & Manifest (`metrics.py`, `manifest.py`)**: `MasteringBenchmarkSuite.evaluate` calcola le metriche pre-mastering (`align_latency=False`) e post-mastering (`align_latency=True` sul segmento allineato estratto con `get_same_track_slice_indices`), salvando WAV completo, preset JSON canonico e `MasteringManifest`.



### 2.2 Punti di Forza e Punti di Fragilità Architetturale

* **Pro**:
* Eccellente disaccoppiamento tra fase di ottimizzazione (multi-chunk in PyTorch) e fase di rendering finale (full-track in C++20).


* Tabelle parametriche canoniche (`PARAM_SPECS`) perfettamente speculari tra C++ (`ParameterSpec.hpp`) e Python (`param_space.py`).
* Scansioni ricorsive custom in C++ (`fast_forward_ballistics`, `fast_backward_psi`, `fast_forward_limiter_envelope`, `fast_backward_limiter_envelope`) che abbattono il collo di bottiglia dei loop Python in autograd.


* **Contro / Rischi di Consistenza**:
* **Accoppiamento rigido a $48\text{ kHz}$**: Mentre i filtri biquad e le costanti di tempo accettano `sample_rate` variabile, il lookahead del limiter ($144\text{ campioni} = 3\text{ ms}$ solo a $48\text{ kHz}$) e i tap del saturatore sono fissi in campioni, rendendo il comportamento psicoacustico e la latenza in millisecondi dipendenti dalla frequenza di campionamento.
* **Asimmetria degli iperparametri di default**: `AutoMasteringOptimizer.__init__` imposta di default `weight_param_reg = 0.02`, `lr = 0.05`, `num_steps = 50`, mentre `scripts/run_mastering.py` istanzia l'ottimizzatore con `weight_param_reg = 0.01`, `lr = 0.04` e CLI default `--steps 30`.



---

## 3. Findings (Rilievi Tecnici Classificati P0–P3)

### 3.1 Tabella Riepilogativa dei Rilievi

| ID | Severità | Modulo / File | Titolo Sintetico del Rilievo | Impatto Principale |
| --- | --- | --- | --- | --- |
| **F-01** | **P1** | `loudness.py`, `metrics.py` | Misura LUFS priva di gating BS.1770-4 e affetta da convoluzione circolare FFT | Metriche benchmark LUFS non conformi allo standard ITU-R BS.1770-4 |
| **F-02** | **P1** | `LookaheadLimiter.hpp`, `metrics.py` | Assenza di True Peak oversampling (Sample Peak spacciato per True Peak) | Possibili inter-sample peaks oltre $-0.3\text{ dBFS}$ e metrica benchmark imprecisa |
| **F-03** | **P1** | `PluginProcessor.cpp`, `PluginEditor.cpp` | Editor GUI custom inutilizzato, True Bypass non esposto e stato JSON incompleto | Plugin VST3 mostra GUI generica JUCE, manca di bypass host e non salva i preset in DAW |
| **F-04** | **P1** | `chunking.py` | Possibile violazione di `min_separation_samples` e bypass di `min_rms_db` nello spostamento | Chunk consecutivi troppo vicini o riposizionati su code silenziose |
| **F-05** | **P2** | `MasteringChain.hpp`, `PluginProcessor.cpp` | Potenziale riallocazione heap in `process()` e overhead eccessivo in `processBlock` | Rischio realtime safety se l'host eccede `max_block_size` + ricalcolo trigonometrico inutile |
| **F-06** | **P2** | `compressor.py`, `limiter.py`, `py_core_dsp.cpp` | Rilascio mancante del GIL in `pybind11` e fallback lento per $B > 1$ o CUDA | Scalabilità ridotta per training batch/multi-thread nello Sprint 3 |
| **F-07** | **P2** | `reference_protocol.py` | Bias del denominatore nella cross-correlazione su segnali con offset elevati o code | Sottostima del coefficiente di correlazione `norm_corr` per lag grandi |
| **F-08** | **P2** | `stylistic.py` | Calcolo Mel-spectrum solo su mixdown mono e mismatch con il crossover M/S | Perdita di informazione timbrica laterale (Side) nello spectral matching stilistico |
| **F-09** | **P2** | `Constants.hpp`, `LookaheadLimiter.hpp` | Latenza e finestra lookahead cablate in campioni anziché scalate con `sample_rate` | Variazione della costante di tempo lookahead/boxcar se l'host DAW opera a $44.1$ o $96\text{ kHz}$ |
| **F-10** | **P3** | `metrics.py` | Codice duplicato (dead code) dopo il `return` in `MasteringBenchmarkSuite.evaluate` | Pulizia e manutenibilità del codice |
| **F-11** | **P3** | `PluginProcessor.cpp`, `ParameterSpec.hpp` | Inconsistenza nei valori di fallback in `MasteringParameters` vs `THETA_NEUTRAL` | Rischio di inizializzazione non neutra se si usa la struct C++ di default |
| **F-12** | **P3** | `.github/workflows/ci.yml` | CI limitata a Windows e mancata esecuzione di `plugin_headless_tests` nel workflow | Regressioni C++/JUCE o incompatibilità Linux/macOS non intercettate in CI |

---

### 3.2 Dettaglio Analitico dei Findings

#### **F-01 [P1 — High]: Misura LUFS priva di gating BS.1770-4 e affetta da convoluzione circolare non paddata**

* **File e righe**: `python/ddsp_mastering/loss/loudness.py` (righe 86–142), `python/ddsp_mastering/optimize/metrics.py` (righe 149–150).
* **Spiegazione tecnica**:
1. In `DifferentiableKWeighting.forward` (righe 101–104), il filtraggio K-weighting viene eseguito nel dominio della frequenza tramite `X = torch.fft.rfft(x, n=T, dim=-1)` senza zero-padding, applicando una **convoluzione circolare** anziché lineare. Inoltre, la risposta in frequenza viene valutata con `omega = torch.linspace(0, math.pi, n_fft // 2 + 1)` (riga 74), che per $T$ dispari diverge dalle frequenze esatte dei bin di `rfft` (`2 * pi * k / T`).
2. In `DifferentiableLoudnessLoss.calculate_lufs` (righe 130–142), la loudness integrata viene calcolata come semplice media quadratica globale sull'intero segnale (`torch.mean(y_k ** 2, dim=-1)`). Mancano completamente i blocchi scorrevoli da $400\text{ ms}$ con overlap del $75\%$ ($100\text{ ms}$ step), il **gating assoluto a $-70\text{ LUFS}$** e il **gating relativo a $-10\text{ LU}$** prescritti dallo standard **ITU-R BS.1770-4 / EBU R128**.
3. Sebbene una media non-gated sia accettabile (e spesso preferibile per continuità dei gradienti) come *proxy loss* su chunk pre-filtrati da $3\text{ s}$, la stessa funzione `self.loudness_calc.calculate_lufs()` viene usata in `MasteringBenchmarkSuite.evaluate()` sull'intera traccia per certificare i risultati finali.


* **Conseguenza pratica**: Su un brano completo con intro, outro o pause dinamiche, la misura `pred_lufs` e `target_lufs` nel manifest e nei report di benchmark sottostima la loudness reale BS.1770-4 anche di $1.5 - 4.0\text{ LU}$, invalidando la conformità allo standard dichiarata nei report.
* **Fix raccomandato**: Distinguere esplicitamente la funzione proxy differenziabile (`calculate_ungated_k_lkfs`) dalla metrica di benchmark conforme allo standard. In `MasteringBenchmarkSuite`, implementare il calcolo ITU-R BS.1770-4 completo (filtraggio IIR o FFT zero-paddata con `torch.fft.rfftfreq(M) * 2 * math.pi`, finestre da $400\text{ ms}$, gate a $-70\text{ LUFS}$ e gate relativo a $-10\text{ LU}$).

#### **F-02 [P1 — High]: Assenza di True Peak oversampling nel Limiter e nel Benchmark**

* **File e righe**: `cpp/core_dsp/LookaheadLimiter.hpp` (righe 65–72), `python/ddsp_mastering/dsp/limiter.py` (righe 86–91), `python/ddsp_mastering/optimize/metrics.py` (righe 153–155), `PluginEditor.cpp` (riga 25).
* **Spiegazione tecnica**:
* Sia il limiter C++ (`LookaheadLimiter::process_sample`) sia il limiter PyTorch (`DifferentiableLookaheadLimiter.forward`) rilevano il picco esclusivamente sui campioni a frequenza nominale $1\times f_s$ (`std::max(std::abs(u_l), std::abs(u_r))`). Non vi è alcuno stadio di oversampling $4\times$ (o $2\times$) nel rilevatore di inviluppo del limiter.
* Inoltre, poiché il guadagno `g_smooth` (media mobile boxcar su 144 campioni di `g_rel`) varia nel tempo durante la fase di rilascio, quando un secondo picco ravvicinato cade nella finestra di lookahead mentre i primi campioni del buffer boxcar contengono ancora valori in risalita ($g_{\text{rel}} > g_{\text{min}}$), la media aritmetica `g_smooth` può risultare leggermente superiore a $g_{\text{target}}$, permettendo lievi overshoot anche sul sample peak (motivo per cui `test_sprint_1_correctness.py` riga 202 ha dovuto aggiungere una tolleranza `+ 1e-3` al ceiling lineare).
* In `MasteringBenchmarkSuite.evaluate` (riga 153), `pred_peak_linear = torch.max(torch.abs(pred_audio)).item()` calcola il **Sample Peak (dBFS)** e non il **True Peak (dBTP)** sovracampionato a $4\times$ come richiesto da ITU-R BS.1770-4, pur dichiarando nel docstring `4. True Peak & Ceiling Compliance (<= -0.3 dBFS)`.


* **Conseguenza pratica**: Segnali masterizzati con contenuto ad alta frequenza possono generare **inter-sample peaks (ISP)** fino a $+0.5 / +1.5\text{ dBTP}$ sopra $-0.3\text{ dBFS}$ durante la conversione D/A o la codifica lossy, senza che il benchmark lo rilevi.
* **Fix raccomandato**: Implementare in `MasteringBenchmarkSuite` una misura True Peak reale con interpolazione polifase FIR $4\times$ (conforme BS.1770-4) riportando sia `pred_sample_peak_dbfs` sia `pred_true_peak_dbtp`. Per il limiter, chiarire nella documentazione della tesi che si tratta di un *Sample-Peak Lookahead Brickwall Limiter* (con margine di sicurezza a $-0.3\text{ dBFS}$ per assorbire gli ISP), oppure aggiungere un rilevatore di picco $2\times/4\times$ o un hard-clipper di sicurezza finale a $-0.3\text{ dBFS}$ post-boxcar.

#### **F-03 [P1 — High]: Editor GUI custom ignorato, True Bypass assente da APVTS e serializzazione DAW incompleta**

* **File e righe**: `cpp/plugin/PluginProcessor.cpp` (righe 253–255, 89–159, 239–251), `cpp/plugin/PluginEditor.cpp`.
* **Spiegazione tecnica**:
1. **GUI Custom ignorata**: Esistono `PluginEditor.h` e `PluginEditor.cpp` che definiscono `MasteringPluginAudioProcessorEditor`, ma in `PluginProcessor.cpp` (riga 254) `createEditor()` restituisce `new juce::GenericAudioProcessorEditor(*this);`. La classe `MasteringPluginAudioProcessorEditor` è codice morto mai istanziato.
2. **True Bypass non esposto nel Plugin**: Il requisito di separazione True Bypass vs Neutral Preset è implementato in `core_dsp::MasteringChain::set_bypass(bool)`, ma `MasteringPluginAudioProcessor` non registra alcun parametro booleano di bypass in APVTS né implementa `AudioProcessor::getBypassParameter()`. L'utente o l'host VST3 non possono attivare il True Bypass compensato a 159 campioni dal plugin.
3. **Incoerenza nella serializzazione JSON**: `MasteringPluginAudioProcessor::getParametersAsJson()` (righe 212–222) esporta un JSON piatto privo dei campi `"format"`, `"version"`, `"sample_rate"`, `"normalized_z"` e `"parameters"`, divergendo dallo schema canonico `AutoMastering_Preset_v1` generato da Python.


* **Conseguenza pratica**: Caricando il VST3 in una DAW (Ableton Live, FL Studio, Reaper), viene mostrata l'interfaccia generica a slider di JUCE invece dell'editor progettato, e il pulsante di bypass nativo della DAW non utilizza la linea di ritardo compensata interna di `MasteringChain`.
* **Fix raccomandato**: Sostituire `GenericAudioProcessorEditor` con `new MasteringPluginAudioProcessorEditor(*this)` (arricchendolo con i controlli APVTS o pulsante Load Preset JSON), aggiungere un parametro `bypass` collegato a `getBypassParameter()` e a `chain_.set_bypass()`, e allineare `getParametersAsJson()` allo schema `AutoMastering_Preset_v1`.

#### **F-04 [P1 — High]: Bug logico nello spostamento dei chunk in `MultiChunkSelector` (`min_separation` e `min_rms_db`)**

* **File e righe**: `python/ddsp_mastering/data/chunking.py` (righe 185–208).
* **Spiegazione tecnica**:
Nel post-processing dei chunk selezionati (righe 187–207):
```python
for c in selected_candidates:
    ...
    prev_start = filtered[-1]["start"]
    required_start = prev_start + self.chunk_len + self.min_separation_samples
    if c["start"] >= required_start:
        filtered.append(c)
    else:
        new_start = required_start
        if new_start + self.chunk_len <= T:
            new_chunk = audio[:, new_start : new_start + self.chunk_len]
            rms_db, cf_db, cent = self._compute_chunk_features(new_chunk)
            filtered.append({"start": new_start, ...})

```


Emergono due difetti:
1. **Mancata verifica di `min_rms_db` dopo lo spostamento**: Quando il chunk viene spostato forzosamente a `new_start = required_start`, le feature vengono ricalcolate ma **non viene verificato** se `rms_db >= self.min_rms_db`. Il chunk spostato può quindi cadere esattamente in una pausa silenziosa o in un breakdown sotto $-42\text{ dBFS}$.
2. **Possibile disordinamento o duplicazione in zone vuote**: Se una zona temporale non ha candidati validi, il fallback `zone_pool = sorted(candidates, key=lambda c: abs(c["start"] - zone_center))[:3]` (riga 171) può pescare un candidato già selezionato in una zona adiacente. Lo spostamento cieco a `required_start` ignora completamente i criteri di qualità (RMS, Crest Factor, Centroid) dei candidati reali presenti più avanti nella traccia.


* **Conseguenza pratica**: Su brani con breakdown centrali o outro sfumati, lo spostamento automatico può inserire nell'ottimizzazione un chunk silenzioso ($\text{RMS} < -42\text{ dBFS}$), falsando il calcolo di `effective_target_lufs` e i gradienti di loudness.
* **Fix raccomandato**: Quando `c["start"] < required_start`, cercare il miglior candidato valido pre-filtrato in `candidates` che soddisfi `cand["start"] >= required_start` (che garantisce già `rms_db >= min_rms_db`). Solo se non esiste, scartare il chunk riducendo il numero di chunk con warning.

#### **F-05 [P2 — Medium]: Potenziale allocazione heap in `MasteringChain::process` e ricalcolo coefficienti in `processBlock**`

* **File e righe**: `cpp/core_dsp/MasteringChain.hpp` (righe 119, 159–180), `cpp/plugin/PluginProcessor.cpp` (righe 145–147).
* **Spiegazione tecnica**:
1. `MasteringChain::process()` chiama ad ogni blocco `ensure_scratch_buffers(num_samples)`, che invoca `std::vector::resize()` su 18 buffer qualora `num_samples` superi la dimensione allocata in `prepare()`. Sebbene `PluginProcessor::processBlock` suddivida il buffer in sub-blocchi di `SUB_BLOCK_SIZE = 32` campioni (evitando il resize nel plugin), una chiamata diretta a `MasteringChain::process` in un contesto C++ realtime con buffer variabile può innescare 18 allocazioni heap sul thread audio.
2. In `PluginProcessor::processBlock` (riga 145):
```cpp
if (params_changed || samples_processed == 0) {
    chain_.set_parameters_theta(current_theta_);
}

```


La condizione `|| samples_processed == 0` forza il ricalcolo completo di tutti i coefficienti trigonometrici (`std::sin`, `std::cos`, `std::tan`, `std::pow`, `std::exp`) di 6 biquad EQ, 6 biquad crossover LR4, 3 compressori, saturatore e limiter **all'inizio di ogni singolo buffer audio**, anche quando nessun parametro è cambiato (`params_changed == false`).


* **Conseguenza pratica**: Spreco di cicli CPU sul thread real-time ad ogni callback audio DAW quando i parametri sono statici.
* **Fix raccomandato**: Rimuovere `|| samples_processed == 0` in `processBlock` (poiché `prepareToPlay` inizializza già `chain_.set_parameters_theta(current_theta_)`). In `MasteringChain`, marcare `process()` come `noexcept` e usare `jassert` / pre-allocazione stretta in `prepare()`.

#### **F-06 [P2 — Medium]: Mancato rilascio del GIL in `pybind11` e limitazione del fast-path C++ a `B == 1**`

* **File e righe**: `cpp/bindings/py_core_dsp.cpp` (righe 319–426), `python/ddsp_mastering/dsp/compressor.py` (righe 64, 113), `python/ddsp_mastering/dsp/limiter.py` (righe 22, 41).
* **Spiegazione tecnica**:
* I loop C++ in `py_core_dsp.cpp` (`fast_forward_ballistics`, `fast_backward_psi`, `fast_forward_limiter_envelope`, `fast_backward_limiter_envelope`) supportano già una dimensione batch generica `size_t B = buf.shape[0]`. Tuttavia, in Python (`compressor.py` e `limiter.py`), la guardia condizionale impone `if _HAS_PY_CORE_DSP and not x_flat.is_cuda and B == 1:`. Se si valuta un batch di chunk in parallelo ($B = K = 4$), Python cade nel lentissimo ciclo `for n in range(N):` interpretato (144.000 iterazioni per chunk da 3s).
* Inoltre, in `py_core_dsp.cpp`, né `MasteringChain::process` né le funzioni di scansione rilasciano il Global Interpreter Lock (`py::call_guard<py::gil_scoped_release>()`).


* **Conseguenza pratica**: In `optimize_multi_chunk`, i $K$ chunk devono essere processati sequenzialmente in un ciclo `for k in range(K):` anziché come singolo tensore batch `(K, 2, T)`, moltiplicando per $K$ il tempo di forward/backward.
* **Fix raccomandato**: Rimuovere la restrizione `and B == 1` in `compressor.py` (passando gli array di `alpha_a` e `alpha_r` per elemento batch o vettore, poiché i parametri $z$ sono condivisi tra tutti i chunk dello stesso brano: `alpha_a` è identico per tutti i chunk!) e in `limiter.py`. Ciò consente di impilare i chunk in un unico tensore `(K, 2, T)` ed eseguire un solo forward/backward vettorizzato per step.

#### **F-07 [P2 — Medium]: Normalizzazione imprecisa della cross-correlazione per lag elevati**

* **File e righe**: `python/ddsp_mastering/data/reference_protocol.py` (righe 82–109).
* **Spiegazione tecnica**:
`norm_mix = np.linalg.norm(mix_mono)` e `norm_ref = np.linalg.norm(ref_mono)` vengono calcolati sull'intera finestra `analysis_len` (15 secondi). Quando `best_lag` è diverso da zero, `signal.correlate(ref_mono, mix_mono, mode='full')` somma solo gli $N - |\text{lag}|$ campioni sovrapposti, ma divide il picco per le norme complete su $N$ campioni (`peak_val / (norm_mix * norm_ref)`).
* **Conseguenza pratica**: Per ritardi di $2 - 3\text{ secondi}$ su clip brevi (es. $5 - 10\text{ s}$) o quando la parte non sovrapposta contiene energia elevata, `norm_corr` viene artificialmente sottostimato e può scendere sotto `min_correlation_threshold = 0.35` anche per tracce identiche.
* **Fix raccomandato**: Calcolare il coefficiente di correlazione di Pearson `norm_corr` ricalcolando la norma euclidea esattamente sulle due slice sovrapposte corrispondenti a `best_lag`.

#### **F-08 [P2 — Medium]: Perdita di informazione spaziale in `AggregatePerceptualLoss._extract_mel_spectrum_distribution**`

* **File e righe**: `python/ddsp_mastering/loss/stylistic.py` (righe 56–77), `python/ddsp_mastering/optimize/metrics.py` (righe 96–111).
* **Spiegazione tecnica**:
In `_extract_mel_spectrum_distribution`, l'audio stereo viene convertito in mono tramite somma coerente `mono = torch.mean(audio, dim=1)` ($M = (L+R)/2$) prima di calcolare la STFT. Qualsiasi energia presente in controfase o nel canale Side ($S = (L-R)/2$, es. synth stereo wide, riverberi, piatti panoramizzati) subisce cancellazione di fase nel mixdown mono.
* **Conseguenza pratica**: Se l'ottimizzatore allarga l'immagine stereo tramite i parametri `imager_width_mid` o `imager_width_high`, l'energia del canale Side aumenta, ma la funzione di loss timbrica `_extract_mel_spectrum_distribution` (che vede solo il canale Mid) non percepisce l'aumento di energia sugli alti/medi, spingendo l'EQ ad enfatizzare eccessivamente le alte frequenze.
* **Fix raccomandato**: Calcolare lo spettro di potenza come media incoerente delle energie dei due canali: `mag_sq = 0.5 * (torch.abs(stft_L)**2 + torch.abs(stft_R)**2)` anziché fare la STFT della media mono.

#### **F-09 [P2 — Medium]: Costanti di latenza e lookahead cablate in campioni anziché dipendenti da `sample_rate**`

* **File e righe**: `cpp/core_dsp/Constants.hpp` (righe 47–51), `cpp/core_dsp/LookaheadLimiter.hpp` (riga 12), `python/ddsp_mastering/dsp/limiter.py` (riga 11).
* **Spiegazione tecnica**:
`LIMITER_LOOKAHEAD_SAMPLES = 144` e `FIR_HALFBAND_LATENCY = 15` sono costanti `constexpr` fisse. Se il plugin JUCE viene istanziato in una sessione DAW a $96\text{ kHz}$, 144 campioni corrispondono a $1.5\text{ ms}$ anziché $3.0\text{ ms}$ (modificando la pendenza di attacco del filtro boxcar), e il filtro FIR half-band (progettato in frequenza normalizzata $\pi/2$) mantiene 15 campioni di ritardo a $f_s$. In `PluginEditor.cpp` (riga 20), la label mostra una stringa statica `"Reported Host Latency: 159 samples (3.31 ms @ 48 kHz)"`.
* **Conseguenza pratica**: Comportamento dinamico del limiter variabile al variare del sample rate dell'host DAW.
* **Fix raccomandato**: Documentare chiaramente che il motore è progettato per operare a $48\text{ kHz}$ (o scalare dinamicamente la dimensione della finestra lookahead $\text{round}(0.003 \cdot f_s)$ in `prepareToPlay` aggiornando `setLatencySamples`).

#### **F-10 [P3 — Low]: Codice duplicato irraggiungibile (Dead Code) in `MasteringBenchmarkSuite.evaluate**`

* **File e righe**: `python/ddsp_mastering/optimize/metrics.py` (righe 223–230).
* **Spiegazione tecnica**: Le righe 223–230 duplicano esattamente il blocco `if z_pred is not None and z_target is not None: ... return results` già presente alle righe 214–221 subito dopo l'istruzione `return results`.
* **Conseguenza pratica**: Nessun impatto funzionale, ma riduce la pulizia del codice.
* **Fix raccomandato**: Rimuovere le righe duplicate 223–230 in `metrics.py`.

#### **F-11 [P3 — Low]: Valori di default di `MasteringParameters` non allineati a `THETA_NEUTRAL**`

* **File e righe**: `cpp/core_dsp/ParameterSpec.hpp` (righe 86–118).
* **Spiegazione tecnica**: La struct `MasteringParameters` inizializza `BandComp::ratio = 2.0` e `threshold_db = -20.0`, `crossover_f_low = 150.0`, `crossover_f_high = 3000.0`, mentre `THETA_NEUTRAL` (e `PluginProcessor.cpp`) usa `ratio = 1.0`, `threshold_db = -12.0`, `f_low = 160.0`, `f_high = 3200.0`.
* **Fix raccomandato**: Allineare i default member initializers di `MasteringParameters` in `ParameterSpec.hpp` ai valori esatti di `THETA_NEUTRAL`.

#### **F-12 [P3 — Low]: Workflow GitHub Actions CI non compila né esegue il target C++/JUCE `plugin_headless_tests**`

* **File e righe**: `.github/workflows/ci.yml` (righe 1–43).
* **Spiegazione tecnica**: Il file `.github/workflows/ci.yml` compila solo l'estensione `pybind11` (`python setup.py build_ext --inplace`) ed esegue `pytest python/tests/`. Non esegue CMake per compilare il plugin JUCE 8 né lancia l'eseguibile C++ `plugin_headless_tests` (`test_plugin.cpp`).
* **Conseguenza pratica**: Eventuali rotture di compilazione C++20/JUCE in `cpp/plugin/` non vengono rilevate dalla CI automatica su GitHub.
* **Fix raccomandato**: Aggiungere uno step CMake in `.github/workflows/ci.yml` per compilare ed eseguire `test_plugin.cpp` ad ogni push.

---

## 4. Gate A / Sprint 2 / Sprint 2.1 Audit (Dichiarazioni vs Realtà del Codice)

Verifica rigorosa della catena **Requisito → Codice effettivo → Test effettivo → Evidenza → Limiti residui**:

| Milestone / Ticket | Requisito Dichiarato nei Closure Report | Stato nel Codice (`874550c`) | Qualità del Test | Verdetto Audit & Limiti Residui |
| --- | --- | --- | --- | --- |
| **Gate A (A-02)** | Compressore con `ratio = 1.0` produce identità matematica ($\text{err} < 10^{-12}$ PyTorch, $< 10^{-6}$ C++). | `SingleBandComp` e `compressor.py` hanno `ratio_factor = 1 - 1/R = 0` per $R=1.0 \implies gc=0, y_g=0, g_{\text{lin}}=1$. | `test_compressor_ratio_one_identity` verifica input random ad alto livello. | **VERIFICATO (100%)**. Nessun limite residuo. |
| **Gate A (A-03/04/05)** | Separazione True Bypass (linea di ritardo 159 smp, zero diff) vs Neutral Preset (`THETA_NEUTRAL`). | `MasteringChain.hpp` implementa `bypass_delay_l_/r_` a 159 campioni. `THETA_NEUTRAL` ha ratio 1.0, gain 0, mix 0. | `test_true_bypass_latency_and_invariance` e `test_neutral_chain_rigorous_characterization`. | **PARZIALE**: Verificato nel core C++ e in `py_core_dsp`, ma il True Bypass **non è esposto nel plugin JUCE** (**F-03**). |
| **Gate A (A-06)** | Backward analitico del limiter verificato con `gradcheck` e loop PyTorch puro. | `FastLimiterReleaseFunction` implementa la ricorrenza aggiunta $\lambda_n = \nabla y_n + \alpha_{\text{rel}} \mathbb{I}_{\text{rel}}[n] \lambda_{n+1}$ (con reset su attack). | `test_fast_limiter_release_gradcheck` e `test_fast_limiter_vs_pure_pytorch_reference`. | **VERIFICATO (100%)**. Derivata analitica esatta rispetto alla maschera di stato. |
| **Gate A (A-07/08)** | Parità numerica IIR C++/PyTorch su 7 casi e continuità transizione FFT $\to$ IIR al 70%. | `eq.py` usa `torchaudio.functional.lfilter` in `float64`; `optimizer.py` commuta a `step == int(0.7 * num_steps)`. | `test_pytorch_iir_vs_cpp_parity_matrix` ($\text{err} < 10^{-4}$, null $< -80\text{ dBFS}$). | **VERIFICATO (100%)**. Nota: in `eq.py` il test copre 6 casi numerati (manca il 7° caso "full chain" nel corpo della funzione, coperto in `test_cpp_parity.py`). |
| **Sprint 2 / 2.1 (A-01)** | Allineamento `same_track` con direzione corretta del lag e gestione lunghezze disuguali. | `reference_protocol.py` usa `signal.correlate(ref, mix)`: lag $D > 0 \implies$ `ref` ritardato $\implies$ `ref[D:]`. | `test_same_track_alignment_known_delays` e `test_same_track_unequal_lengths_and_invalid_overlap`. | **VERIFICATO**. Residuo minore: normalizzazione della correlazione su lunghezze intere (**F-07**). |
| **Sprint 2.1 (A-02)** | Raffinamento sub-sample e filtraggio fractional delay FIR sinc Blackman a 65 tap ($K=32$). | `apply_fractional_delay` usa `h = np.sinc(k + delta) * blackman(65)` e rifila $K=32$ campioni ai bordi. | `test_subsample_alignment_fractional_delay_rigorous` su $\pm 12.35, -8.40, 0.25$ smp. | **VERIFICATO (100%)**. Convenzione del segno di $\delta$ e fase di `np.sinc(k + delta)` matematicamente esatte. |
| **Sprint 2.1 (A-04)** | Separazione rigorosa dei benchmark `same_track` vs `stylistic` (nessuna STFT frame-by-frame su stylistic). | `metrics.py` imposta `spectral_convergence = None` e `log_spectral_distance_db = None` in `stylistic`. | `test_benchmark_suite_mode_separation_and_guardrails`. | **VERIFICATO**. Residuo: lo spettro Mel stilistico usa il mixdown mono anziché la potenza stereo (**F-08**). |
| **Sprint 2.1 (A-05)** | Multi-chunk con centroide STFT su intero chunk, `min_separation_samples` e ricalcolo feature. | `chunking.py` usa `scipy.signal.stft` (2048/512) sull'intero chunk e sposta i chunk troppo vicini. | `test_multi_chunk_stft_centroid_and_separation_constraint`. | **PARZIALE**: Lo spostamento a `required_start` non ricontrolla la soglia `min_rms_db` (**F-04**). |
| **S2.1-FINAL** | Indicizzazione temporale benchmark full-track: pre-mastering `align_latency=False`, post-mastering con offset `mix_start` e singola PDC (159 smp). | `run_mastering.py` (righe 185–208) usa `get_same_track_slice_indices` ed estrae `mastered_full_np[:, mix_start : mix_start + eval_len]`. | `test_full_track_benchmark_temporal_indexing_end_to_end` (5 topologie di ritardo). | **VERIFICATO**. Piccola imprecisione residua: per lag frazionari, `mastered_eval_segment` non subisce `apply_fractional_delay` (ma `ref_aligned` è già stato filtrato per allinearsi a `raw_aligned`, quindi il confronto di fase è comunque corretto!). |

---

## 5. DSP & Numerical Correctness

### 5.1 Analisi Stadio per Stadio e Parità C++20 vs PyTorch

1. **6-Band Parametric EQ (`BiquadCascade.hpp` vs `eq.py`)**:
* **Equazioni**: Formule RBJ Audio EQ Cookbook per Low-Shelf (banda 0), Peaking (bande 1–4) e High-Shelf (banda 5). Identiche bit-a-bit in `float64` tra C++ e PyTorch.
* **Stabilità**: Poli sempre interni al cerchio unitario grazie ai vincoli $f_0 \in [20, 18000]\text{ Hz}$ ($< f_s/2$) e $Q \in [0.4, 2.5]$. La struttura Direct Form II Transposed (`BiquadDF2T`) in `double` garantisce assenza di overflow interni e rumore di quantizzazione trascurabile ($< -140\text{ dBFS}$).
* **Approssimazione FFT vs IIR**: In `eq.py`, `_forward_fft` fa zero-padding a $M = 2^{\lceil \log_2(2N) \rceil}$, evitando l'aliasing temporale da convoluzione circolare finché la coda dell'impulso IIR decade entro $N$ campioni (ampiamente soddisfatto per $N = 144.000$ e $Q \le 2.5$).


2. **4th-Order Linkwitz-Riley Crossover (`LinkwitzRiley4.hpp` vs `crossover.py`)**:
* **Topologia**: Split a 3 vie ottenuto cascando uno split LR4 a $f_{\text{low}}$ ($\text{LP}_4, \text{HP}_4$) seguito da uno split LR4 a $f_{\text{high}}$ sul ramo medio-alto, con filtro All-Pass del 2° ordine ($\text{AP}_2$ a $f_{\text{high}}$) sul ramo basso.
* **Ricostruzione**: Poiché $\text{LP}_4(s) + \text{HP}_4(s) = \text{AP}_2(s)$, la somma delle 3 bande ricostruisce un filtro All-Pass globale perfettamente piatto in ampiezza ($0.00\text{ dB}$ di ripple).


3. **3-Band Stereo-Linked Compressor (`MultibandComp.hpp` vs `compressor.py`)**:
* **Rilevatore ed Equivalenza**: Rilevamento energetico stereo-linked $E[n] = 0.5(L^2 + R^2) + 10^{-12}$, gain computer soft-knee $C^1$ con ampiezza fissa $W = 6\text{ dB}$, e balistica ricorsiva one-pole nel dominio dei dB.
* **Autograd & Gradient Flow**: La funzione custom `DifferentiableSingleBandCompressorFunction` calcola analiticamente le derivate rispetto a $x, T, R, \tau_a, \tau_r, M$. L'uso del soft-knee quadratica assicura la continuità della derivata prima $\partial g_c / \partial \delta$ ai bordi del ginocchio ($\delta = \pm 3\text{ dB}$), prevenendo oscillazioni del gradiente attorno alla soglia.


4. **Mid/Side Stereo Imager (`MidSideImager.hpp` vs `imager.py`)**:
* **Ortonormalità**: Usa il fattore $1/\sqrt{2}$ sia in codifica sia in decodifica. Per $w = 1.0$, $(M+S)/\sqrt{2} = L$ e $(M-S)/\sqrt{2} = R$ con identità algebrica esatta.


5. **2x Oversampled Harmonic Saturator (`HarmonicSaturator.hpp` vs `saturator.py`)**:
* **Architettura**: Interpolazione $2\times$ zero-stuffing con guadagno $2.0$ tramite filtro FIR Half-Band simmetrico a 31 tap (Kaiser $\beta = 8.5$), non-linearità normalizzata $\tanh(g_d u) / g_d$ (guadagno unitario per piccoli segnali: $\lim_{u \to 0} \tanh(g_d u)/g_d = u$), e decimazione $2\times$ tramite il medesimo FIR.
* **Allineamento di fase Dry/Wet**: I due filtri FIR a 31 tap ad alta frequenza ($2 f_s$) hanno ciascuno un ritardo di gruppo di $(31-1)/2 = 15$ campioni a $2 f_s$, per un totale di $30$ campioni a $2 f_s$, che corrispondono esattamente a **$15$ campioni alla frequenza base $f_s$**. La linea di ritardo `delay_dry` ritarda il segnale dry di esattamente 15 campioni, eliminando qualsiasi comb-filtering nel mix parallelo.


6. **Lookahead Brickwall Limiter (`LookaheadLimiter.hpp` vs `limiter.py`)**:
* **Catena del Guadagno**: Min-filter su finestra di $N_L = 144$ campioni (implementato in $O(1)$ ammortizzato tramite Monotonic Deque circolare in C++ e tramite `-F.max_pool1d(-g)` in PyTorch), seguito da rilascio one-pole ($\tau_{\text{rel}} = 80\text{ ms}$) e media mobile boxcar su $N_L = 144$ campioni.
* **Verifica della Latenza PDC (159 campioni)**: La convoluzione tra il min-filter di lunghezza $N_L$ e il filtro boxcar di lunghezza $N_L$ genera una rampa di attacco lineare anticipata esattamente di $N_L = 144$ campioni, che raggiunge il valore minimo in corrispondenza del picco ritardato di 144 campioni. Sommando i 15 campioni del saturatore e i 144 campioni del limiter (gli stadi IIR hanno ritardo di gruppo dipendente dalla frequenza ma zero latenza di lookahead), la latenza causale deterministica della catena è **esattamente $15 + 144 = 159$ campioni**.



---

## 6. Plugin & Production Readiness (JUCE 8 / VST3)

### 6.1 Realtime Safety nel `processBlock`

* **Protezione Denormali**: `juce::ScopedNoDenormals noDenormals;` è correttamente istanziato in cima a `MasteringPluginAudioProcessor::processBlock` (riga 117).
* **Zero Lock / Zero Allocazioni sul Percorso Nominale**: La lettura dei parametri avviene tramite `std::atomic<float>*` con `std::memory_order_relaxed` e rampe lineari `juce::SmoothedValue` di $20\text{ ms}$ aggiornate ogni `SUB_BLOCK_SIZE = 32` campioni. Finché `SUB_BLOCK_SIZE` (32) è inferiore a `max_block_size_` ($\ge 256$), `ensure_scratch_buffers` non effettua allocazioni.
* **Difetti da Correggere per Production Readiness**:
1. Collegare `MasteringPluginAudioProcessorEditor` in `createEditor()` (**F-03**).
2. Esporre il parametro di **True Bypass** all'host DAW (**F-03**).
3. Rimuovere `|| samples_processed == 0` dal check `if (params_changed)` in `processBlock` (**F-05**).
4. Includere un meccanismo nella GUI per caricare un file `_preset.json` generato dallo script Python di ottimizzazione.



---

## 7. Test & CI Quality

### 7.1 Valutazione Critica dei 97 Test

La test suite (`97/97` test passati in $\approx 107\text{ s}$) è ben strutturata e copre unit test DSP, parità C++/PyTorch, `autograd.gradcheck`, allineamento intero/frazionario e pipeline CLI end-to-end. Tuttavia, l'analisi del codice dei test evidenzia alcuni **punti ciechi e tolleranze accomodanti**:

1. **Tolleranza accomodante sul Limiter Ceiling**: In `test_sprint_1_correctness.py` (riga 202), il test del ceiling sul limiter definisce `ceiling_lin = 10.0 ** (-0.3 / 20.0) + 1e-3`, mascherando il fatto che il boxcar smoothing post-release può causare lievi overshoot sopra $-0.3000\text{ dBFS}$ su transienti ravvicinati (**F-02**).
2. **Assenza di test di validazione LUFS contro riferimenti EBU R128 / ITU-R BS.1770-4**: I test su `loudness.py` verificano che la loss scenda e i gradienti esistano, ma non confrontano mai l'output di `calculate_lufs` con i segnali di test ufficiali EBU Tech 3341 (es. tono sinusoidale stereo a $1\text{ kHz}$, $-20\text{ dBFS}$ con gating) (**F-01**).
3. **Mancato test dello spostamento chunk su zone silenziose**: In `test_sprint_2_protocol.py`, `test_multi_chunk_stft_centroid_and_separation_constraint` testa `min_separation` su rumore bianco uniforme (`np.random.randn * 0.2`), dove qualsiasi posizione spostata `required_start` ha sempre energia valida, non scoprendo il bug **F-04**.
4. **Mancata esecuzione di `test_plugin.cpp` in GitHub Actions**: Come rilevato in **F-12**, i test C++ del plugin JUCE sono stati eseguiti localmente ma non fanno parte della pipeline `.github/workflows/ci.yml`.

---

## 8. Sprint 3 Readiness

### Verdetto Finale: **CONDITIONAL GO**

Il progetto possiede fondamenta matematiche e ingegneristiche di alto livello per una Tesi Magistrale al Politecnico di Milano, ma **non può dichiarare la conformità agli standard ITU-R BS.1770-4 (LUFS e True Peak) né la completezza del plugin VST3** senza prima risolvere le 4 condizioni bloccanti **P1** elencate di seguito.

#### Condizioni Bloccanti Obbligatorie per lo Sprint 3:

1. **[Blocco B1 — da F-01] Conformità ITU-R BS.1770-4 nel Benchmark**: Implementare il gating assoluto ($-70\text{ LUFS}$) e relativo ($-10\text{ LU}$) su finestre da $400\text{ ms}$ (overlap $75\%$) e correggere la convoluzione circolare FFT in `DifferentiableKWeighting` (usando zero-padding lineare e `rfftfreq`). Mantenere la versione ungated/soft-gated solo come proxy differenziabile nella loss.
2. **[Blocco B2 — da F-02] True Peak Oversampled ($4\times$) e Hard Ceiling Safety**: Aggiungere il calcolo del vero True Peak ($4\times$ oversampled) in `MasteringBenchmarkSuite.evaluate` e aggiungere uno stadio di safety clipping/limiting post-boxcar a $-0.3\text{ dBFS}$ nel limiter per garantire $0.00\text{ dB}$ di sample-peak overshoot senza tolleranze artificiali.
3. **[Blocco B3 — da F-04] Robustezza di `MultiChunkSelector` su Spostamento**: Garantire che in `chunking.py` i candidati spostati per rispettare `min_separation_samples` soddisfino ancora il gate `rms_db >= min_rms_db`.
4. **[Blocco B4 — da F-03] Allineamento Plugin JUCE**: Collegare `MasteringPluginAudioProcessorEditor`, esporre il parametro di True Bypass all'host VST3 e allineare `getParametersAsJson()` allo schema `AutoMastering_Preset_v1`.

---

## 9. Recommended Action Plan

Di seguito la roadmap ordinata degli interventi prioritari da eseguire prima di sviluppare i nuovi moduli dello Sprint 3:

1. **Step 1 — Remediation Metriche Standard (`loudness.py` & `metrics.py`) [P1 / F-01, F-02, F-10]**:
* Correggere `DifferentiableKWeighting` sostituendo `linspace` con `torch.fft.rfftfreq(M, d=1.0) * 2 * math.pi` e aggiungendo zero-padding a $M = 2^{\lceil \log_2(2T) \rceil}$.
* Aggiungere in `loudness.py` (o `metrics.py`) `calculate_integrated_lufs_bs1770(x)` con blocchi da $400\text{ ms}$, step $100\text{ ms}$, gate assoluto $-70\text{ LUFS}$ e gate relativo $-10\text{ LU}$ per la valutazione oggettiva in `MasteringBenchmarkSuite`.
* Aggiungere `calculate_true_peak_dbtp(x)` tramite interpolazione sinc/polifase $4\times$ in `MasteringBenchmarkSuite`.
* Rimuovere le 8 righe di dead code in fondo a `metrics.py`.


2. **Step 2 — Hardening di `MultiChunkSelector` e `reference_protocol.py` [P1 & P2 / F-04, F-07]**:
* In `chunking.py`, quando un candidato viola `required_start`, selezionare il candidato successivo dalla lista pre-filtrata `candidates` (che rispetta già `min_rms_db`) con `start >= required_start`.
* In `reference_protocol.py`, normalizzare il picco di cross-correlazione dividendo per le norme calcolate sulle slice effettivamente sovrapposte a `best_lag`.


3. **Step 3 — Completamento Integrazione JUCE VST3 e Core DSP [P1 & P2 / F-03, F-05]**:
* In `PluginProcessor.cpp`, rimuovere `|| samples_processed == 0` nel loop dei sub-blocchi, esporre un `AudioParameterBool` per il True Bypass collegato a `chain_.set_bypass()`, e restituire `MasteringPluginAudioProcessorEditor` in `createEditor()`.
* In `LookaheadLimiter.hpp` e `limiter.py`, applicare un clamp finale di sicurezza `std::clamp(out, -c_lin_, c_lin_)` (e `torch.clamp` / soft-clamp) dopo la moltiplicazione per `g_smooth` per eliminare qualsiasi micro-overshoot dovuto alla media mobile boxcar durante la fase di release.


4. **Step 4 — Vettorizzazione Batch PyTorch/C++ e CI [P2 & P3 / F-06, F-08, F-12]**:
* In `compressor.py` e `limiter.py`, abilitare il fast-path C++ anche per $B > 1$, permettendo a `optimize_multi_chunk` di processare tutti i $K$ chunk in un singolo batch `(K, 2, T)`.
* In `stylistic.py`, calcolare lo spettro Mel sulla somma incoerente delle potenze `0.5 * (|STFT(L)|^2 + |STFT(R)|^2)`.
* Aggiungere la build CMake e l'esecuzione di `plugin_headless_tests` in `.github/workflows/ci.yml`.