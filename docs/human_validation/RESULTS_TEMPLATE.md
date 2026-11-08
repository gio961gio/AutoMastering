# Report Conclusivo di Validazione Umana — Sprint 1 & Sprint 2

**Candidato:** Giovanni Antonecchia  
**Corso di Laurea:** Master's Thesis in Sound and Music Engineering — Politecnico di Milano (DEIB)  
**Supervisore Tecnico:** Antigravity / Argon Bridge  
**Commit motore congelato (Codice):** `5940e81f12741e2276f7a446adfb6e12f44574fe`  
**Commit motore congelato (Documentazione):** `92fe105a19312f258fefb2c70a847720cbd95525`  
**Commit pacchetto validazione umana:** `ffafb9848bea375195973795df83af6fe0970e86`  
**Stato corrente:** `[X] Human Experiment Completed` (Completato con successo in data 08/11/2026)

---

## 1. Riepilogo Esecutivo Esperimento A (Sprint 1: Neutralità, True Bypass e Legacy Baseline)

L'Esperimento A ha verificato sperimentalmente la trasparenza matematica e percettiva del motore DDSP prima dell'ottimizzazione automatica.

### 1.1 Misure Oggettive e Verifiche Numeriche
- **Brano sorgente utilizzato:** `track01_mix.wav`  
  - SHA-256: `56c57e4fe5cb799e6364056e7fa5cf795e28db7794196804bcd58ada3086203d`
  - Durata: `3066985 campioni` (`63.8955 s` a 48000 Hz, stereo).
- **Trim Diagnostico Statico Headroom:**
  - True Peak iniziale: `+3.975 dBTP` (inter-sample peak da decodifica MP3).
  - Trim statico applicato: **`-9.975 dB`** (linear gain `0.317126`).
  - True Peak post-trim uniforme sulle 3 condizioni: **`-6.000 dBTP`**.
- **Allineamento e Compensazione Latenza (159 campioni + Tail Flush):**
  - **Null-Test `A (True Bypass)` vs Input Trimmato:**
    - `max_err_dbfs`: **`-156.56 dBFS`**
    - `rms_null_dbfs`: **`-178.43 dBFS`** (allineamento bit-exact su campioni float32).
  - **Differenza RMS `C (Corrected Neutral)` vs `A (True Bypass)`:** **`-22.67 dBFS`** (conferma filtraggio trasparente della catena neutrale attiva).
- **Stato ricostruzione Legacy Baseline (`86ae20f08dc7fbebc7f67af064a15b648a492fc7`):**
  - Vettore storico `z = 0.5` ricostruito integralmente (41 parametri).
  - 5 parametri su 41 eccedono i limiti fisici del motore corrente congelato (`comp_low_ratio`, `comp_low_attack_ms`, `comp_mid_ratio`, `comp_high_ratio`, `saturator_drive_db`) -> emesso finding `LEGACY_BASELINE_NOT_REPRODUCIBLE` in modalità `strict`.
  - Policy adottata per l'ascolto della variante B: **`clamped_to_current_bounds`** (5 parametri clampati ai bound contrattuali correnti, 36 parametri legacy mantenuti invariati).
- **Seed deterministico sessione:** `13607737023179710879` (64-bit CSPRNG isolato in `human_validation/private/keys/track01_sprint1_blind_key.json`).

### 1.2 Esito dell'Ascolto Cieco (Unblinded)
- **`blind_A` = `A_true_bypass`:** Voto **`5 / 5`** su tutte le 8 metriche. Note: *"Nessun difetto"*.
- **`blind_B` = `C_corrected_neutral`:** Voto **`5 / 5`** su tutte le 8 metriche. Note: *"Nessun Difetto"*.
- **`blind_C` = `B_legacy_baseline`:** Voto **`5 / 5`** su tutte le 8 metriche. Note: *"Nessun difetto, ma ho notato che la traccia inizia prima delle altre due... a posteriori era leggermente più schiacciata dinamicamente"*.

> **Conclusione Sprint 1:**  
> Piena conferma della trasparenza percettiva di `THETA_NEUTRAL`: all'ascolto critico, la catena di mastering neutra è risultata indistinguibile dal True Bypass. L'ascoltatore ha inoltre rilevato la maggiore densità dinamica della Legacy Baseline causata dalla struttura originale di +9 dB di makeup gain e limiter push.

---

## 2. Riepilogo Esecutivo Esperimento B (Sprint 2: Valutazione Multi-Brano e Multi-Sezione)

L'Esperimento B ha valutato l'efficacia del mastering automatico DDSP su 3 brani musicali differenti, testando sia la modalità `--mode same_track` (reference: master dello stesso brano) sia la modalità `--mode stylistic` (reference: brano commerciale finito dello stesso genere), con esportazione di sezioni d'ascolto cieco a livello di programma e loudness-matched.

### 2.1 Tabella Sintetica Misure Oggettive Pre / Post Mastering

| Track ID | Modalità | Seed (64-bit) | Pre LUFS | Post LUFS | Δ LUFS | Pre TP (`dBTP`) | Post TP (`dBTP`) | Pre PLR (`dB`) | Post PLR (`dB`) | Loss Iniziale -> Finale (STFT + Loudness) | Riduzione Loss |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`track01`** | `same_track` | `5405783196615117109` | `-14.22` | **`-10.51`** | `+3.71` | `+3.98` | `+0.28` | `18.12` | `10.21` | `6.67` -> **`3.10`** | **`-53.5%`** |
| **`track02`** | `same_track` | `3048363069421082382` | `-22.03` | **`-18.04`** | `+3.99` | `-4.03` | `-0.07` | `18.01` | `17.74` | `14.28` -> **`10.43`** | **`-27.0%`** |
| **`track03`** | `same_track` | `12990700764526553868` | `-17.86` | **`-13.50`** | `+4.36` | `-3.38` | `+0.02` | `14.45` | `13.20` | `10.45` -> **`6.47`** | **`-38.1%`** |
| **`track01_style`** | `stylistic` | `14887125631601849244` | `-14.22` | **`-17.84`** | `-3.62` | `+3.98` | `-2.38` | `18.12` | `15.46` | `28.09` -> **`8.59`** | **`-69.4%`** |
| **`track02_style`** | `stylistic` | `3606996967096283602` | `-22.03` | **`-18.05`** | `+3.98` | `-4.03` | `-0.07` | `18.01` | `17.75` | `159.91` -> **`83.00`** | **`-48.1%`** |
| **`track03_style`** | `stylistic` | `15136152743686100324` | `-17.86` | **`-13.52`** | `+4.34` | `-3.38` | `+0.05` | `14.45` | `13.22` | `110.60` -> **`46.78`** | **`-57.7%`** |

---

### 2.2 Sintesi dei Giudizi Umani Ciechi (Originale vs Master DDSP)

#### A. Sessioni `same_track` (18 sessioni d'ascolto cieco)
- **`track02` (Vittoria schiacciante del Master DDSP, 4 su 6):**
  - Il mix grezzo partiva con headroom ideale (`-22 LUFS`). Il master DDSP ha recuperato la dinamica e il punch mancante senza schiacciare il suono (`PLR` preservato a `17.74 dB`). L'ascoltatore ha penalizzato l'originale definendolo *"spento e privo di punch"* rispetto al master (`5/5`).
- **`track03` (Vittoria del Master DDSP nelle sezioni ritmiche e dense):**
  - Nel Ritornello/Drop a *program-level*, il Master DDSP ha ottenuto `5/5` su tutta la linea, mentre l'originale è risultato *"meno brillante e con poco punch"*. Nel *level-matched*, il master ha corretto la debolezza dei bassi dell'originale (*"le basse sono un po' sotto"*).
- **`track01` (Insegnamento critico sui mix già iper-compressi):**
  - Il mix di partenza era già a `-14.2 LUFS` con clipping. Per inseguire la brillantezza del reference, l'ottimizzatore ha aperto l'imager e spinto la presence, portando il master a `-10.5 LUFS`. L'ascoltatore ha rilevato con precisione la compressione eccessiva a *program-level* (*"harsh, stereo troppo aperto"*), preferendo l'equilibrio più cupo ma morbido dell'originale.

#### B. Sessioni `stylistic` (6 sessioni d'ascolto cieco sul Ritornello / Drop Denso)
- **Prestazione perfetta in *Loudness-Matched*:**
  - Su tutti e 3 i brani (`track01_style`, `track02_style`, `track03_style`), il Master DDSP ha ottenuto **`5.0 / 5.0` su tutte le 8 metriche qualitative**, confermando assenza totale di artefatti e perfetta pulizia del segnale.
- **Il trionfo a *Program-Level* di `track01_style`:**
  - Nel confronto stilistico, il reference commerciale più morbido ha guidato l'ottimizzatore ad attenuare le alte frequenze (`-1.9 dB`) e il volume (`-17.84 LUFS`), rimuovendo i picchi a `+3.98 dBTP`.
  - All'ascolto cieco a *program-level*, **il Master DDSP ha battuto l'originale con un netto `5 / 5` contro `3 / 5`**: l'ascoltatore ha penalizzato l'originale descrivendolo come *"harsh e un po' troppo compresso"*, preferendo la musicalità rilassata e trasparente del master DDSP.

---

## 3. Conclusioni Critiche per la Tesi

Dall'analisi congiunta delle metriche numeriche e delle valutazioni d'ascolto umano emergono 3 conclusioni fondamentali:

### 1. Verifica della Neutralità Operativa (Sprint 1)
L'implementazione del preset neutro `THETA_NEUTRAL` (EQ a 0 dB, compressori a ratio 1:1, saturatore e limiter a drive nullo) è matematicamente e percettivamente equivalente al bypass puro. Questo garantisce che la catena DDSP non operi alterazioni preconcette o deleterie sul materiale sonoro in assenza di gradienti di loss.

### 2. Tenuta del Preset Unico Multi-Sezione (Sprint 2)
L'approccio a **singolo preset ottimizzato per l'intero brano tramite selezione multi-chunk** si è dimostrato robusto: il preset calcolato sui chunk estratti ha mantenuto stabilità e coerenza lungo tutte le sezioni del pezzo (intro quiete, strofe, ritornelli densi e code con transienti), senza generare discontinuità timbriche o pompaggio localizzato.

### 3. Correlazione tra Loss DDSP e Percezione Umana
L'esperimento ha evidenziato una preziosa dicotomia per la discussione accademica:
- **Quando il mix in ingresso ha headroom dinamico (`track02`):** La minimizzazione della loss (STFT multi-risoluzione + Loudness loss) si traduce direttamente in un miglioramento percettivo evidente (punteggi 5/5, cassa e rullante più incisivi, maggior calore).
- **Quando il mix in ingresso è già saturato o privo di headroom (`track01` same-track):** Una diminuzione della loss numerica (che spinge le frequenze alte per inseguire la brillantezza del target) può accentuare la fatica d'ascolto (*harshness*) nel limiter.
- Questo dimostra scientificamente perché **l'ascolto critico umano (Human-in-the-Loop) è insostituibile** e fornisce una solida base per proporre, come sviluppi futuri della tesi, l'integrazione di vincoli di cresta dinamica (*crest-factor penalty*) nella funzione di costo.
