# Report Conclusivo di Validazione Umana — Sprint 1 & Sprint 2

**Candidato:** Giovanni Antonecchia  
**Corso di Laurea:** Master's Thesis in Sound and Music Engineering — Politecnico di Milano (DEIB)  
**Supervisore Tecnico:** Antigravity / Argon Bridge  
**Commit motore congelato (Codice):** `5940e81f12741e2276f7a446adfb6e12f44574fe`  
**Commit motore congelato (Documentazione):** `92fe105a19312f258fefb2c70a847720cbd95525`  
**Commit pacchetto validazione umana:** `ffafb9848bea375195973795df83af6fe0970e86`  
**Stato corrente:** `[X] Human Experiment Completed` (Completato in data 08/11/2026 — Revisione di coerenza scientifica e documentale)

---

## 1. Riepilogo Esecutivo Esperimento A (Sprint 1: Neutralità, True Bypass e Legacy Baseline)

L'Esperimento A ha indagato il comportamento numerico e percettivo del motore DDSP a parametri neutri (`THETA_NEUTRAL`) rispetto al True Bypass e alla Legacy Baseline storica (`z = 0.5`), prima dell'ottimizzazione automatica.

### 1.1 Misure Oggettive e Distinzione tra Null-Test e Catena Attiva
- **Brano sorgente utilizzato:** `track01_mix.wav`  
  - SHA-256: `56c57e4fe5cb799e6364056e7fa5cf795e28db7794196804bcd58ada3086203d`
  - Durata: `3066985 campioni` (`63.8955 s` a 48000 Hz, stereo).
- **Trim Diagnostico Statico Headroom:**
  - True Peak iniziale: `+3.975 dBTP` (inter-sample peak da decodifica MP3).
  - Trim statico applicato: **`-9.975 dB`** (linear gain `0.317126`).
  - True Peak post-trim uniforme sulle 3 condizioni: **`-6.000 dBTP`**.
- **Distinzione tra Equivalenza Numerica (`A` vs Input) e Differenza Oggettiva (`C` vs `A`):**
  1. **Null-Test `A (True Bypass)` vs Input Trimmato (compensazione latenza 159 campioni + tail flush):**
     - `max_err_dbfs`: **`-156.56 dBFS`**
     - `rms_null_dbfs`: **`-178.43 dBFS`**
     - *Significato:* conferma esclusivamente il perfetto allineamento temporale e l'assenza di alterazioni nel ramo di True Bypass rispetto al file di input trimmato. **Questo dato non misura né dimostra l'equivalenza numerica tra `C (Corrected Neutral)` e `A (True Bypass)`.**
  2. **Differenza `C (Corrected Neutral / THETA_NEUTRAL)` vs `A (True Bypass)`:**
     - `max_err_dbfs`: **`-2.44 dBFS`**
     - `rms_null_dbfs`: **`-22.67 dBFS`**
     - *Significato:* pur con guadagni EQ a `0 dB`, compressori a ratio `1:1` e drive nulli, la catena DDSP attiva attraversa i filtri di crossover Linkwitz-Riley del 4° ordine (che introducono una rotazione di fase all-pass alle frequenze di incrocio) e il filtro FIR anti-aliasing dell'oversampling 2x nel saturatore/limiter. Pertanto **`C_corrected_neutral` e `A_true_bypass` NON sono numericamente equivalenti** e presentano una differenza oggettiva misurabile di **`-22.67 dBFS` RMS**.
- **Stato ricostruzione Legacy Baseline (`86ae20f08dc7fbebc7f67af064a15b648a492fc7`):**
  - Vettore storico `z = 0.5` ricostruito integralmente (41 parametri).
  - 5 parametri su 41 eccedono i limiti fisici del motore corrente congelato (`comp_low_ratio`, `comp_low_attack_ms`, `comp_mid_ratio`, `comp_high_ratio`, `saturator_drive_db`) -> emesso finding `LEGACY_BASELINE_NOT_REPRODUCIBLE` in modalità `strict`.
  - Policy adottata per l'ascolto della variante B: **`clamped_to_current_bounds`** (5 parametri clampati ai bound contrattuali correnti, 36 parametri legacy mantenuti invariati).
- **Seed deterministico sessione:** `13607737023179710879` (64-bit CSPRNG isolato in `human_validation/private/keys/track01_sprint1_blind_key.json`).

### 1.2 Esito dell'Ascolto Cieco (Unblinded) e Nota Metodologica sul Blind Test
I risultati compilati sono archiviati in `docs/human_validation/SPRINT1_LISTENING_RESULTS.csv` (lasciando intatto il template vuoto `docs/human_validation/SPRINT1_LISTENING_TEMPLATE.csv`):
- **`blind_A` = `A_true_bypass`:** Voto **`5 / 5`** su tutte le 8 metriche. Note: *"Nessun difetto"*.
- **`blind_B` = `C_corrected_neutral`:** Voto **`5 / 5`** su tutte le 8 metriche. Note: *"Nessun Difetto"*.
- **`blind_C` = `B_legacy_baseline`:** Voto **`5 / 5`** su tutte le 8 metriche. Note: *"Nessun difetto, ma ho notato che la traccia inizia prima delle altre due... a posteriori era leggermente più schiacciata dinamicamente"*.

> **Conclusione e Limite Metodologico Sprint 1:**  
> 1. **Trasparenza percettiva in presenza di differenza oggettiva:** Sia `A_true_bypass` sia `C_corrected_neutral` hanno ricevuto il punteggio massimo (`5/5` su tutte le 8 dimensioni) dall'unico ascoltatore ($N=1$), dimostrando che la differenza oggettiva misurabile tra i due segnali (`-22.67 dBFS` RMS, dovuta alla fase all-pass dei crossover LR4 e al filtro FIR 2x) non ha prodotto artefatti o colorazioni timbriche percepite dall'ascoltatore.  
> 2. **Anomalia di allineamento su `blind_C` (`B_legacy_baseline`):** Sebbene i 3 file WAV generati dal runner abbiano durata identica al campione (`3066985 campioni`), durante l'importazione/esportazione nella DAW locale per la sessione d'ascolto la traccia `blind_C` ha subito un trim accidentale del silenzio iniziale. L'ascoltatore ha correttamente annotato tale partenza anticipata; questo elemento costituisce un **limite metodologico alla completa cecità della condizione `B_legacy_baseline`**, rendendo quella traccia distinguibile per l'offset temporale iniziale.

---

## 2. Riepilogo Esecutivo Esperimento B (Sprint 2: Valutazione Multi-Brano e Multi-Sezione)

L'Esperimento B ha valutato il mastering automatico DDSP su 3 brani musicali differenti, testando sia la modalità `--mode same_track` (reference: master dello stesso brano, 3 sezioni per brano) sia la modalità `--mode stylistic` (reference: brano commerciale finito dello stesso genere, 1 sezione sul Ritornello/Drop per brano), sia a parità di loudness (`level_matched`) sia a livello di programma (`program_level`), per un totale di **24 sessioni d'ascolto cieco** (12 `level_matched` + 12 `program_level`). I dati compilati sono archiviati in `docs/human_validation/SPRINT2_SAME_TRACK_LISTENING_RESULTS.csv` e `docs/human_validation/SPRINT2_STYLISTIC_DROP_LISTENING_RESULTS.csv`.

### 2.1 Tabella Sintetica Misure Oggettive Pre / Post Mastering

| Track ID | Modalità | Seed (64-bit) | Pre LUFS | Post LUFS | Δ LUFS | Pre TP (`dBTP`) | Post TP (`dBTP`) | Pre PLR (`dB`) | Post PLR (`dB`) | Loss Composita Totale*: Iniziale (`z_0`) -> `step 29` -> Finale su `z_opt` (`IIR`) | Riduzione Loss (`z_0` -> `z_opt`) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`track01`** | `same_track` | `5405783196615117109` | `-14.22` | **`-10.51`** | `+3.71` | `+3.98` | `+0.28` | `18.12` | `10.21` | `6.6669` -> `3.0966` -> **`3.0841`** | **`-53.7%`** |
| **`track02`** | `same_track` | `3048363069421082382` | `-22.03` | **`-18.04`** | `+3.99` | `-4.03` | `-0.07` | `18.01` | `17.74` | `14.2843` -> `10.4312` -> **`10.4173`** | **`-27.1%`** |
| **`track03`** | `same_track` | `12990700764526553868` | `-17.86` | **`-13.50`** | `+4.36` | `-3.38` | `+0.02` | `14.45` | `13.20` | `10.4493` -> `6.4696` -> **`6.4545`** | **`-38.2%`** |
| **`track01_style`** | `stylistic` | `14887125631601849244` | `-14.22` | **`-17.84`** | `-3.62` | `+3.98` | `-2.38` | `18.12` | `15.46` | `28.0920` -> `8.5858` -> **`8.5529`** | **`-69.6%`** |
| **`track02_style`** | `stylistic` | `3606996967096283602` | `-22.03` | **`-18.05`** | `+3.98` | `-4.03` | `-0.07` | `18.01` | `17.75` | `159.9057` -> `82.9988` -> **`82.7613`** | **`-48.2%`** |
| **`track03_style`** | `stylistic` | `15136152743686100324` | `-17.86` | **`-13.52`** | `+4.34` | `-3.38` | `+0.05` | `14.45` | `13.22` | `110.5956` -> `46.7759` -> **`46.5655`** | **`-57.9%`** |

*\*Nota sulla Loss Composita Totale (`num_steps = 30`, `lr = 0.04`, `4 chunk x 3.0 s`):* La metrica registrata è la **loss composita completa** dell'ottimizzatore (in `same_track`: `1.0 * MR-STFT + 0.8 * Loudness/PLR + 0.01 * ParamReg`; in `stylistic`: `1.0 * MelSpectral + 0.8 * Loudness + 0.5 * PLR + 0.3 * Stereo + 0.01 * ParamReg`), e non la sola somma STFT + Loudness. Nella tabella sono riportati il valore iniziale a `z_0 = Z_NEUTRAL` (`step 0`), il valore registrato in `history["loss"][-1]` all'ultimo passo di ciclo (`step 29`, pre-update finale) e il valore finale esatto ricalcolato sul vettore restituito `z_opt` (`step 30`, con `eq_mode="iir"`).

---

### 2.2 Riconciliazione Conteggi e Medie Riproducibili dello Sprint 2

Come dettagliato nella Sezione 3 e Sezione 6 di `docs/human_validation/SPRINT2_LISTENING_TEMPLATE.md`, i conteggi delle sessioni sono riportati secondo due regole esplicite:
- **Regola Primaria (`OP` pura):** basata esclusivamente sul voto `overall_preference_1_5`.
- **Regola Secondaria (`OP` + spareggio `Mean8`):** in caso di parità su `overall_preference_1_5`, assegna la vittoria alla variante con media aritmetica superiore sulle 8 dimensioni qualitative (`Mean8`); se anche `Mean8` è identica, mantiene il pareggio.

| Condizione / Sottoinsieme | N. Sessioni | Conteggio Regola Primaria (`OP` pura) | Conteggio Regola Secondaria (`OP` + `Mean8`) | Master $\ge$ Orig (Vittorie + Pareggi) | Media 8 Criteri (`Mean8`) Master vs Orig | Media `OP` Master vs Orig |
|---|:---:|---|---|---|---|---|
| **`same_track` — `level_matched`** | `9` | `2 Master, 5 Orig, 2 Pareggi` | `4 Master, 5 Orig, 0 Pareggi` | **`4 / 9`** | `4.63` (`333/72`) vs `4.81` (`346/72`) | `4.22` (`38/9`) vs `4.67` (`42/9`) |
| **`stylistic` — `level_matched`** | `3` | `0 Master, 0 Orig, 3 Pareggi` | `0 Master, 0 Orig, 3 Pareggi` | **`3 / 3`** *(tutti pareggi `5/5`)* | `5.00` (`120/24`) vs `5.00` (`120/24`) | `5.00` (`15/3`) vs `5.00` (`15/3`) |
| **Totale `level_matched`** | **`12`** | **`2 Master, 5 Orig, 5 Pareggi`** | **`4 Master, 5 Orig, 3 Pareggi`** | **`7 / 12`** (`58.3%`) | **`4.72` (`453/96`) vs `4.85` (`466/96`)** | **`4.42` (`53/12`) vs `4.75` (`57/12`)** |
| **`same_track` — `program_level`** | `9` | `2 Master, 5 Orig, 2 Pareggi` | `4 Master, 5 Orig, 0 Pareggi` | **`4 / 9`** | `4.50` (`324/72`) vs `4.64` (`320/69`*) | `3.78` (`34/9`) vs `4.44` (`40/9`) |
| **`stylistic` — `program_level`** | `3` | `1 Master, 2 Orig, 0 Pareggi` | `1 Master, 2 Orig, 0 Pareggi` | **`1 / 3`** | `4.63` (`111/24`) vs `4.75` (`114/24`) | `4.00` (`12/3`) vs `4.33` (`13/3`) |
| **Totale `program_level`** | **`12`** | **`3 Master, 7 Orig, 2 Pareggi`** | **`5 Master, 7 Orig, 0 Pareggi`** | **`5 / 12`** (`41.7%`) | **`4.53` (`435/96`) vs `4.67` (`434/93`*)** | **`3.83` (`46/12`) vs `4.42` (`53/12`)** |

*\*Nota sulle celle non valutate (`NV`):* in `same_track program_level`, 3 singole celle dell'Originale su 72 non sono state compilate dall'ascoltatore (`track01 sec03 TB`, `track03 sec01 DN`, `track03 sec03 DA`) e sono escluse dal denominatore (`69` celle valide in `same_track program_level`, `93` nel totale `program_level`). Tutte le altre serie hanno il 100% delle celle compilate (`96/96`).

---

### 2.3 Sintesi Analitica dei Giudizi Umani Ciechi

#### A. Sessioni `same_track` (18 sessioni d'ascolto cieco: 9 `level_matched` + 9 `program_level`)
- **`track02` (Prevalenza netta del Master DDSP in 4 sessioni su 6):**
  - Il mix grezzo partiva con ampio headroom (`-22.03 LUFS`). Il Master DDSP ha vinto con `OP: 5 vs 4` e `Mean8: 5.000` in `sec01 program_level`, `sec02 level_matched`, `sec02 program_level` e `sec03 level_matched`, recuperando il punch su kick e snare senza degradare il `PLR` (`17.74 dB`). L'Originale è stato preferito solo in `sec01 level_matched` e `sec03 program_level`.
- **`track03` (Pareggi su `OP` con vantaggio del Master su `Mean8` in 3 sessioni su 6):**
  - In `sec01 level_matched`, `sec03 level_matched` e `sec02 program_level`, Master DDSP e Originale hanno pareggiato su `overall_preference_1_5` (`5 vs 5`), ma il Master DDSP ha ottenuto `5.000` su tutte le 8 dimensioni mentre l'Originale ha subito leggere penalizzazioni su `tonal_balance_1_5` (`4/5`, *"le basse sono un po' sotto"*) o `transient_punch_1_5` / `dynamic_naturalness_1_5` (`4/5` e `3/5`, *"meno brillante, si sente poco il punch"*). Nelle restanti 3 sessioni (`sec02 level_matched`, `sec01 program_level`, `sec03 program_level`), l'Originale ha vinto per la maggiore naturalezza dinamica rispetto al Master spinto a `-13.50 LUFS`.
- **`track01` (Prevalenza dell'Originale su un mix di partenza già compresso e in clipping):**
  - Il mix di partenza era già a `-14.22 LUFS` con picchi a `+3.98 dBTP`. Per avvicinare lo spettro del reference in `same_track`, l'ottimizzatore ha aperto l'imager e spinto le alte frequenze, portando il master a `-10.51 LUFS` (`PLR` ridotto a `10.21 dB`). L'Originale ha vinto in 5 sessioni su 6, mentre `sec03 program_level` si è chiusa in pareggio su `OP` (`3 vs 3`, con `Mean8: 4.000 vs 3.857`).

#### B. Sessioni `stylistic` (6 sessioni d'ascolto cieco sul Ritornello / Drop Denso)
- **Distinzione tra i 3 confronti `stylistic` `level_matched` e i 3 confronti `stylistic` `program_level`:**
  1. **Condizione `level_matched` (3 pareggi pieni `5/5` vs `5/5`):**
     - Nei tre confronti a loudness pareggiata (`track01_style`, `track02_style`, `track03_style`), **sia il Master DDSP sia l'Originale hanno ricevuto `5/5` su tutte le 8 dimensioni qualitative (`Mean8 = 5.000 vs 5.000`)**. Questi esiti rappresentano **3 pareggi al massimo punteggio** (non vittorie del Master DDSP) e indicano che, una volta compensata la differenza di volume, l'adattamento timbrico verso un reference esterno non ha introdotto artefatti o squilibri percepiti dall'ascoltatore.
  2. **Condizione `program_level` (esiti differenziati sui tre brani: `1 vittoria Master DDSP, 2 vittorie Originale`):**
     - **`track01_style` (`program_level` — Vittoria Master DDSP `5 vs 3`):** Il reference commerciale esterno aveva un profilo più morbido e dinamico; l'ottimizzatore ha ridotto la loudness da `-14.22 LUFS` a `-17.84 LUFS`, eliminando i picchi oltre `0 dBFS` (`-2.38 dBTP`). All'ascolto a livello di programma, il Master DDSP ha vinto nettamente (`OP: 5 vs 3`, `Mean8: 5.000 vs 4.250`), mentre l'Originale è stato penalizzato come *"harsh e un po' troppo compresso"*.
     - **`track02_style` (`program_level` — Vittoria Originale `5 vs 4`):** L'Originale è stato preferito (`OP: 5 vs 4`, `Mean8: 5.000 vs 4.625`) poiché nel Master DDSP a livello di programma *"si sentono molto le alte rispetto alle basse"*.
     - **`track03_style` (`program_level` — Vittoria Originale `5 vs 3`):** L'Originale è stato preferito (`OP: 5 vs 3`, `Mean8: 5.000 vs 4.250`) poiché la spinta di `+4.34 LUFS` del Master DDSP ha prodotto una *"loudness molto forte, harsh"*.

---

## 3. Limiti Metodologici e Conclusioni per la Tesi

### 3.1 Limiti Metodologici della Campagna Sperimentale
1. **Natura Esplorativa Single-Listener ($N=1$):**
   - L'intera campagna di ascolto (Sprint 1 e Sprint 2) è stata eseguita da un **unico ascoltatore** ($N=1$, l'autore della tesi) ed ha carattere di **valutazione esplorativa preliminare**, non di studio statistico multi-ascoltatore.
2. **Non-Indipendenza delle Sessioni Ripetute:**
   - Le 24 sessioni dello Sprint 2 derivano da 3 brani analizzati su più estratti temporali (`sec01`, `sec02`, `sec03`), due modalità di ottimizzazione (`same_track`, `stylistic`) e due regimi di guadagno (`level_matched`, `program_level`). Tali osservazioni costituiscono misure ripetute sullo stesso soggetto e sugli stessi brani, e **non possono essere trattate come campioni indipendenti di ascoltatori**.
3. **Limite alla Cecità della Condizione `B_legacy_baseline` nello Sprint 1:**
   - Come riportato nella Sezione 1.2, il taglio accidentale del silenzio iniziale avvenuto durante il passaggio in DAW locale sul file `blind_C` (`B_legacy_baseline`) ha reso identificabile l'anticipo temporale di quella variante rispetto a `blind_A` e `blind_B`.

### 3.2 Conclusioni Tecniche Consolidate
1. **Trasparenza Percettiva di `THETA_NEUTRAL` (Sprint 1):**
   - Pur presentando una differenza oggettiva misurabile di **`-22.67 dBFS` RMS** rispetto al True Bypass (dovuta ai filtri crossover LR4 e all'oversampling FIR 2x), il preset `THETA_NEUTRAL` ha ottenuto `5/5` su tutte le dimensioni qualitative al pari del True Bypass, confermando l'assenza di colorazioni o distorsioni percepite nello stato iniziale dell'ottimizzatore.
2. **Coerenza del Preset Unico Multi-Sezione (Sprint 2):**
   - A parità di loudness (`level_matched`), il Master DDSP ha raggiunto una media di **`4.72 / 5.0`** (`453/96`) sulle 8 dimensioni qualitative (`4.42 / 5.0` sulla sola `overall_preference_1_5`) e ha ottenuto una valutazione superiore o pari all'Originale in **7 sessioni su 12**.
3. **Interazione tra Headroom del Mix, Loudness Target e Percezione Umana:**
   - I risultati a `program_level` dimostrano che la minimizzazione della loss spettrale e di loudness produce miglioramenti percettivi netti quando il mix ha margine dinamico (`track02`) o quando il reference guida un'attenuazione correttiva di un mix saturato (`track01_style`), mentre può indurre affaticamento d'ascolto (*harshness* e sovra-compressione) quando forza un incremento di loudness su mix già fortemente compressi (`track01 same_track`, `track03`). Questo risultato motiva l'introduzione futura di termini di penalizzazione sul fattore di cresta (*crest-factor / dynamic naturalness regularization*) nella funzione di loss.
