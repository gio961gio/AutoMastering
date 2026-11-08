# Scheda di Valutazione d'Ascolto — Sprint 1 (Esperimento A)

> [!NOTE]
> Sessione esplorativa *single-listener* ($N = 1$, autore della tesi) completata in data 08/11/2026. I dati grezzi compilati e decodificati a posteriori sono archiviati in [`docs/human_validation/SPRINT1_LISTENING_RESULTS.csv`](SPRINT1_LISTENING_RESULTS.csv), mentre [`docs/human_validation/SPRINT1_LISTENING_TEMPLATE.csv`](SPRINT1_LISTENING_TEMPLATE.csv) è stato mantenuto intatto come template vuoto.

## 1. Metadati Sessione e Verifica Tecnica Preliminare

| Campo | Valore effettivo verificato |
|---|---|
| **ID Anonimo Brano** | `track01` |
| **Data e Ora Sessione** | `2026-11-08 10:54` |
| **Ascoltatore** | Giovanni Antonecchia ($N = 1$, valutazione esplorativa *single-listener*) |
| **Sistema di Ascolto** | Cuffie / Monitor Studio (DAW) |
| **Policy Legacy Baseline usata (`--legacy-mode`)** | `clamped_to_current_bounds` (5 parametri fuori range limitati ai bound correnti) |
| **Conferma stesso input e trim diagnostico uniforme (`-6.0 dBTP`)** | Sì (Trim statico applicato: `-9.975 dB`, True Peak post-trim: `-6.000 dBTP`) |
| **Conferma identica durata e allineamento temporale (159 smp compensati)** | Sì (Campioni: `3066985`, Durata: `63.8955 s` a `48000 Hz`) |
| **Null test `A (True Bypass)` vs Input Trimmato** | `max_err_dbfs = -156.56 dBFS`, `rms_null_dbfs = -178.43 dBFS` *(dimostra l'allineamento bit-exact float32 del solo True Bypass dopo compensazione dei 159 campioni + tail flush)* |
| **Differenza oggettiva `C (Corrected Neutral)` vs `A (True Bypass)`** | `max_err_dbfs = -2.44 dBFS`, `rms_null_dbfs = -22.67 dBFS` *(dimostra che `THETA_NEUTRAL` **non** è numericamente identico al True Bypass, a causa della risposta di fase all-pass del crossover LR4 e dei filtri FIR half-band di oversampling 2x)* |
| **Seed deterministico sessione** | `13607737023179710879` (64-bit CSPRNG, salvato nella chiave privata non pubblicata) |

---

## 2. Valutazione Cieca — Copie Loudness-Matched (`blind_A.wav`, `blind_B.wav`, `blind_C.wav`)

*Scala 1–5 per ciascun criterio:*  
`1` = Molto scarso / difetto grave | `3` = Neutro / accettabile / trasparente | `5` = Eccellente / nessun difetto percepito

| Criterio (1–5) | `blind_A.wav` | `blind_B.wav` | `blind_C.wav` |
|---|:---:|:---:|:---:|
| **Tonal Balance** (`tonal_balance_1_5`) | **5** | **5** | **5** |
| **Bass Control** (`bass_control_1_5`) | **5** | **5** | **5** |
| **Vocal Clarity** (`vocal_clarity_1_5`) | **5** | **5** | **5** |
| **Transient Punch** (`transient_punch_1_5`) | **5** | **5** | **5** |
| **Stereo Image / Mono Compatibility** (`stereo_image_1_5`) | **5** | **5** | **5** |
| **Dynamic Naturalness** (`dynamic_naturalness_1_5`) | **5** | **5** | **5** |
| **Distortion / Artifacts** (`distortion_artifacts_1_5`, `5` = zero artefatti) | **5** | **5** | **5** |
| **Overall Preference** (`overall_preference_1_5`) | **5** | **5** | **5** |
| **Media aritmetica 8 criteri (`mean_8_criteria`)** | **`5.000`** | **`5.000`** | **`5.000`** |

### Note Libere Originali dell'Ascoltatore (registrate prima dell'unblinding)
- `blind_A`: *"Nessun difetto"*
- `blind_B`: *"Nessun Difetto"*
- `blind_C`: *"Nessun difetto, ma ho notato che la traccia inizia prima delle altre due (probabilmente nelle altre due è presente un pò di latenza)"*

---

## 3. Unblinding e Analisi Critica (Limiti Metodologici Inclusi)

| Etichetta Anonima | Condizione Reale | Media 8 Criteri | Interpretazione Tecnica e Metodologica |
|---|---|:---:|---|
| **`blind_A`** | **`A_true_bypass`** | `5.000 / 5.0` | Riferimento trasparente (null test vs input trimmato: `-178.43 dBFS` RMS). |
| **`blind_B`** | **`C_corrected_neutral`** | `5.000 / 5.0` | Pur presentando una differenza oggettiva misurabile rispetto a `True Bypass` (`-22.67 dBFS` RMS null dovuta alla fase del crossover LR4 e del filtro FIR 2x), ha ricevuto `5/5` su tutte le dimensioni dall'unico ascoltatore, senza artefatti o colorazioni percepite. |
| **`blind_C`** | **`B_legacy_baseline`** | `5.000 / 5.0` | Replay della baseline storica (`86ae20f0`) con 5 parametri clampati ai bound attuali e level-matching a `-24.19 LUFS` (da `-15.26 LUFS` pre-matching). |

### Limiti Metodologici Rilevati nello Sprint 1
1. **Valutazione esplorativa *single-listener* ($N = 1$):** Il test è stato condotto da un singolo ascoltatore (l'autore della tesi) su un singolo brano (`track01`); pertanto i punteggi `5/5` rappresentano una verifica qualitativa di assenza di difetti macroscopici e non una prova statistica di indistinguibilità psicoacustica (che richiederebbe un test ABX multi-ascoltatore).
2. **Anomalia procedurale sulla traccia `blind_C` (possibile compromissione parziale della cecità per la variante B):** L'ascoltatore ha annotato che `blind_C` (*Legacy Baseline*) iniziava leggermente prima delle altre due tracce. Sebbene i file WAV generati da `render_sprint1.py` nella cartella `blind_delivery/` abbiano identica durata (`3066985` campioni) e allineamento sincrono verificato, durante il caricamento/esportazione manuale in DAW da parte dell'ascoltatore una porzione del silenzio iniziale di `blind_C` è stata involontariamente tagliata. Tale disallineamento accidentale in DAW costituisce un indizio esterno che ha reso riconoscibile `blind_C` rispetto a `blind_A` e `blind_B`, e viene pertanto segnalato come limite procedurale della sessione blind relativamente alla condizione `B_legacy_baseline` (mentre il confronto tra `blind_A` e `blind_B` è rimasto rigorosamente allineato e cieco).
