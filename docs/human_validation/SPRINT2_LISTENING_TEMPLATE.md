# Scheda di Valutazione d'Ascolto — Sprint 2 (Esperimento B: Multi-Brano & Multi-Sezione)

> [!NOTE]
> Sessioni completate dall'ascoltatore singolo ($N=1$) in data 08/11/2026. I dati grezzi anonimizzati sono archiviati in `docs/human_validation/SPRINT2_SAME_TRACK_LISTENING_RESULTS.csv` e `docs/human_validation/SPRINT2_STYLISTIC_DROP_LISTENING_RESULTS.csv` (mantenendo `docs/human_validation/SPRINT2_LISTENING_TEMPLATE.csv` come template vuoto), e la decodifica (unblinding) è stata effettuata tramite le 24 chiavi segrete locali in `human_validation/private/keys/`.

## 1. Metadati Sessioni

| Campo | Valore effettivo verificato |
|---|---|
| **Data e Ora Valutazioni** | `2026-11-08 11:00 – 14:00` |
| **Ascoltatore** | Giovanni Antonecchia (valutazione esplorativa single-listener, $N=1$) |
| **Sistema di Ascolto** | Cuffie / Monitor Studio (DAW & Web App interattiva) |
| **Commit del Motore Congelato** | Codice: `5940e81f12741e2276f7a446adfb6e12f44574fe` — Documentazione: `92fe105a19312f258fefb2c70a847720cbd95525` |
| **File Risultati Compilati** | `docs/human_validation/SPRINT2_SAME_TRACK_LISTENING_RESULTS.csv`, `docs/human_validation/SPRINT2_STYLISTIC_DROP_LISTENING_RESULTS.csv` |

---

## 2. Profilo dei 3 Brani e Modalità Sperimentali

| Track ID | Stile / Tipologia Mix | Durata Mix | Modalità Reference Testate | Unico Preset per Brano |
|---|---|---|---|:---:|
| **`track01`** | Pop / Vocale / Beat moderno | `63.90 s` | 1. `same_track` (`track01_ref.wav`) — 3 sezioni (`sec01`, `sec02`, `sec03`) <br> 2. `stylistic` (`track01_ref_altro_brano.wav`) — 1 sezione (`sec01` Drop) | **Sì** |
| **`track02`** | Hip-Hop / Elettronica / Trap | `71.47 s` | 1. `same_track` (`track02_ref.wav`) — 3 sezioni (`sec01`, `sec02`, `sec03`) <br> 2. `stylistic` (`track02_ref_altro_brano.wav`) — 1 sezione (`sec01` Drop) | **Sì** |
| **`track03`** | Rock / Strumentale / Acustico | `71.78 s` | 1. `same_track` (`track03_ref.wav`) — 3 sezioni (`sec01`, `sec02`, `sec03`) <br> 2. `stylistic` (`track03_ref_altro_brano.wav`) — 1 sezione (`sec01` Drop) | **Sì** |

---

## 3. Regola di Determinazione del Vincitore per Sessione

Ogni sessione d'ascolto cieco confronta `blind_A` e `blind_B` su **8 dimensioni qualitative** valutate su scala intera $1\text{–}5$:
1. `tonal_balance_1_5` (`TB`)
2. `bass_control_1_5` (`BC`)
3. `vocal_clarity_1_5` (`VC`)
4. `transient_punch_1_5` (`TP`)
5. `stereo_image_1_5` (`SI`)
6. `dynamic_naturalness_1_5` (`DN`)
7. `distortion_artifacts_1_5` (`DA`)
8. `overall_preference_1_5` (`OP`)

Per garantire la massima trasparenza metodologica e riconciliare ogni possibile ambiguità di conteggio, nelle matrici seguenti vengono riportate esplicitamente due regole di decisione:

- **Regola Primaria (`OP` pura — `overall_preference_1_5`):**
  $$\text{Vincitore}_{\text{OP}} = \begin{cases} \text{Master DDSP} & \text{se } \text{OP}(\text{Master}) > \text{OP}(\text{Orig}) \\ \text{Originale} & \text{se } \text{OP}(\text{Orig}) > \text{OP}(\text{Master}) \\ \text{Pareggio} & \text{se } \text{OP}(\text{Master}) = \text{OP}(\text{Orig}) \end{cases}$$
- **Regola Secondaria (`OP` + spareggio su `Mean8`):**
  In caso di parità su `overall_preference_1_5` ($\text{OP}(\text{Master}) = \text{OP}(\text{Orig})$), si confronta la media aritmetica sulle dimensioni compilate della sessione ($\text{Mean8}$, calcolata sulle 8 dimensioni o sulle 7 dimensioni non vuote qualora una cella sia rimasta non valutata `NV`). Se anche $\text{Mean8}(\text{Master}) = \text{Mean8}(\text{Orig})$, la sessione rimane **Pareggio**.

> [!IMPORTANT]
> **Riconciliazione rispetto al commit `587252a072526860f9a972a729d0758db8de85e7`:**
> - Nel commit `587252a072526860f9a972a729d0758db8de85e7`, la colonna `"Vincitore Decodificato"` applicava implicitamente lo spareggio su `Mean8` a 3 sessioni con `OP` pari (`track03 sec01 level_matched`, `track03 sec03 level_matched`, `track03 sec02 program_level`), ma lasciava come `Pareggio` `track01 sec03 program_level` (`OP: 3 vs 3`, `Mean8: 4.000 vs 3.857` con `TB` dell'Originale non valutato), producendo i conteggi di tabella `4 Master, 5 Orig, 3 Pareggi` (`level_matched`) e `4 Master, 7 Orig, 1 Pareggio` (`program_level`).
> - Inoltre, il testo narrativo della Sezione 5 riportava erroneamente la frase *"10 sessioni su 12"* in `level_matched`: tale numero era un refuso di sintesi non corrispondente alla somma effettiva (`4 vittorie Master + 3 pareggi = 7 sessioni su 12`, sia con Regola Primaria `2 + 5 = 7/12` sia con Regola Secondaria `4 + 3 = 7/12`). La dicitura errata *"10 sessioni su 12"* è stata rimossa e sostituita con i conteggi esatti verificati dai file CSV grezzi.

---

## 4. Matrice Completa Risultati Ciechi (Unblinded) — Parte 1: `same_track` (18 Sessioni)

Confronti ciechi A/B su 3 sezioni per brano (*Intro/Strofa*, *Ritornello/Drop*, *Bridge/Transienti*) a volume pareggiato (`level_matched`, 9 sessioni) e a livello reale di programma (`program_level`, 9 sessioni).

| Track ID | Sezione & Ruolo | Intervallo | Modalità | `blind_A` (ID, `OP`, `Mean8`) | `blind_B` (ID, `OP`, `Mean8`) | Esito Regola Primaria (`OP` pura) | Esito Regola Secondaria (`OP` + `Mean8`) | Note Cieche dell'Ascoltatore |
|---|---|---|---|---|---|:---:|:---:|---|
| **`track01`** | `sec01` (Transienti) | `0.0–12.0s` | `level_matched` | Orig (`OP=5`, `4.875`) | Master (`OP=3`, `4.250`) | **Originale** | **Originale** | *Master: "Basse poco presenti, presence alta"* |
| **`track01`** | `sec01` (Transienti) | `0.0–12.0s` | `program_level` | Master (`OP=3`, `4.250`) | Orig (`OP=4`, `4.625`) | **Originale** | **Originale** | *Master: "Alte affaticanti, stereo aperto" vs Orig: "Spenta, manca punch"* |
| **`track01`** | `sec02` (Strofa) | `21.5–33.5s` | `level_matched` | Master (`OP=4`, `4.625`) | Orig (`OP=5`, `4.875`) | **Originale** | **Originale** | *Master: "Bassi poco controllati"* |
| **`track01`** | `sec02` (Strofa) | `21.5–33.5s` | `program_level` | Orig (`OP=5`, `5.000`) | Master (`OP=3`, `4.000`) | **Originale** | **Originale** | *Master: "Molto compresso, troppa presence, harsh, stereo aperto"* |
| **`track01`** | `sec03` (Drop) | `49.0–61.0s` | `level_matched` | Master (`OP=3`, `4.000`) | Orig (`OP=4`, `4.625`) | **Originale** | **Originale** | *Master: "Sbilanciato sulle alte, stereo aperto" vs Orig: "Mudness, poco punch"* |
| **`track01`** | `sec03` (Drop) | `49.0–61.0s` | `program_level` | Master (`OP=3`, `4.000`) | Orig (`OP=3`, `3.857`*) | **Pareggio (`3 vs 3`)** | **Master DDSP** (spareggio `4.000 vs 3.857`*) | *Master: "Harsh, stereo aperto" vs Orig: "Impastato sulle medie"* (*`TB` Orig = `NV`) |
| **`track02`** | `sec01` (Drop) | `5.5–17.5s` | `level_matched` | Orig (`OP=5`, `5.000`) | Master (`OP=4`, `4.250`) | **Originale** | **Originale** | *Master: "Bassi abbastanza compressi e poco dinamici"* |
| **`track02`** | `sec01` (Drop) | `5.5–17.5s` | `program_level` | Master (`OP=5`, `5.000`) | Orig (`OP=4`, `4.625`) | **Master DDSP** | **Master DDSP** | *Orig: "Risulta un po' spento perché manca punch"* |
| **`track02`** | `sec02` (Transienti) | `32.5–44.5s` | `level_matched` | Master (`OP=5`, `5.000`) | Orig (`OP=4`, `4.625`) | **Master DDSP** | **Master DDSP** | *Orig: "Manca il punch, specie kick e snare"* |
| **`track02`** | `sec02` (Transienti) | `32.5–44.5s` | `program_level` | Master (`OP=5`, `5.000`) | Orig (`OP=4`, `4.625`) | **Master DDSP** | **Master DDSP** | *Orig: "Manca il punch, specie kick e snare"* |
| **`track02`** | `sec03` (Strofa) | `59.0–71.0s` | `level_matched` | Master (`OP=5`, `5.000`) | Orig (`OP=4`, `4.500`) | **Master DDSP** | **Master DDSP** | Master valutato `5/5` su tutte le 8 metriche |
| **`track02`** | `sec03` (Strofa) | `59.0–71.0s` | `program_level` | Orig (`OP=5`, `4.750`) | Master (`OP=4`, `4.625`) | **Originale** | **Originale** | *Orig: "Manca punch" / Master: "Un po' harsh"* |
| **`track03`** | `sec01` (Strofa) | `0.0–12.0s` | `level_matched` | Orig (`OP=5`, `4.875`) | Master (`OP=5`, `5.000`) | **Pareggio (`5 vs 5`)** | **Master DDSP** (spareggio `5.000 vs 4.875`) | *Orig: "Le basse sono un po' sotto" (`TB=4`)* |
| **`track03`** | `sec01` (Strofa) | `0.0–12.0s` | `program_level` | Orig (`OP=5`, `4.857`*) | Master (`OP=3`, `4.375`) | **Originale** | **Originale** | *Master: "Troppo compresso, risulta un po' harsh"* (*`DN` Orig = `NV`) |
| **`track03`** | `sec02` (Drop) | `21.5–33.5s` | `level_matched` | Orig (`OP=5`, `5.000`) | Master (`OP=4`, `4.500`) | **Originale** | **Originale** | *Master: "Un po' troppo compresso"* |
| **`track03`** | `sec02` (Drop) | `21.5–33.5s` | `program_level` | Master (`OP=5`, `5.000`) | Orig (`OP=5`, `4.500`) | **Pareggio (`5 vs 5`)** | **Master DDSP** (spareggio `5.000 vs 4.500`) | *Orig: "Meno brillante, si sente poco il punch" (`TP=4, DN=3`)* |
| **`track03`** | `sec03` (Transienti) | `59.5–71.5s` | `level_matched` | Master (`OP=5`, `5.000`) | Orig (`OP=5`, `4.875`) | **Pareggio (`5 vs 5`)** | **Master DDSP** (spareggio `5.000 vs 4.875`) | *Orig: "Poco punch dovuto a un po' di compressione in più" (`TP=4`)* |
| **`track03`** | `sec03` (Transienti) | `59.5–71.5s` | `program_level` | Master (`OP=3`, `4.250`) | Orig (`OP=5`, `4.857`*) | **Originale** | **Originale** | *Master: "Harsh, stereo aperto, overcompresso e schiacciato"* (*`DA` Orig = `NV`) |

---

## 5. Matrice Risultati Ciechi (Unblinded) — Parte 2: `stylistic` (6 Sessioni sul Drop Denso)

Valutazione dell'adattamento stilistico da reference commerciale esterno (`trackXX_ref_altro_brano.wav`) sui Ritornelli/Drop dei 3 brani. I risultati sono suddivisi nettamente tra la condizione a parità di loudness (`level_matched`) e la condizione a livello di programma (`program_level`).

### 5.1 Sotto-Tabella `stylistic` — Condizione `level_matched` (3 Sessioni)
In tutti e tre i confronti `stylistic` a loudness pareggiata, **sia il Master DDSP sia l'Originale hanno ottenuto `5/5` su tutte le 8 dimensioni qualitative (`Mean8 = 5.000`)**, configurando **3 pareggi pieni** (e non vittorie del Master DDSP):

| Track ID | Sezione | Modalità | `blind_A` (ID, `OP`, `Mean8`) | `blind_B` (ID, `OP`, `Mean8`) | Esito Regola Primaria (`OP` pura) | Esito Regola Secondaria (`OP` + `Mean8`) | Note Cieche dell'Ascoltatore |
|---|---|---|---|---|---|:---:|:---:|---|
| **`track01_style`** | Drop (`49.0–61.0s`) | `level_matched` | Master (`OP=5`, `5.000`) | Orig (`OP=5`, `5.000`) | **Pareggio (`5 vs 5`)** | **Pareggio (`5.000 vs 5.000`)** | Entrambe valutate `5/5` su tutte le 8 dimensioni |
| **`track02_style`** | Drop (`5.5–17.5s`) | `level_matched` | Master (`OP=5`, `5.000`) | Orig (`OP=5`, `5.000`) | **Pareggio (`5 vs 5`)** | **Pareggio (`5.000 vs 5.000`)** | Entrambe valutate `5/5` su tutte le 8 dimensioni |
| **`track03_style`** | Drop (`21.5–33.5s`) | `level_matched` | Master (`OP=5`, `5.000`) | Orig (`OP=5`, `5.000`) | **Pareggio (`5 vs 5`)** | **Pareggio (`5.000 vs 5.000`)** | Entrambe valutate `5/5` su tutte le 8 dimensioni |

### 5.2 Sotto-Tabella `stylistic` — Condizione `program_level` (3 Sessioni)
A livello reale di programma (`program_level`), i tre brani hanno mostrato esiti divergenti legati alla differenza di loudness tra mix originale e reference stilistico esterno (`1 vittoria Master DDSP, 2 vittorie Originale`):

| Track ID | Sezione | Modalità | `blind_A` (ID, `OP`, `Mean8`) | `blind_B` (ID, `OP`, `Mean8`) | Esito Regola Primaria (`OP` pura) | Esito Regola Secondaria (`OP` + `Mean8`) | Note Cieche dell'Ascoltatore |
|---|---|---|---|---|---|:---:|:---:|---|
| **`track01_style`** | Drop (`49.0–61.0s`) | `program_level` | Orig (`OP=3`, `4.250`) | Master (`OP=5`, `5.000`) | **Master DDSP (`5 vs 3`)** | **Master DDSP (`5.000 vs 4.250`)** | *Originale penalizzato per: "Harsh e un po' troppo compresso"* |
| **`track02_style`** | Drop (`5.5–17.5s`) | `program_level` | Orig (`OP=5`, `5.000`) | Master (`OP=4`, `4.625`) | **Originale (`5 vs 4`)** | **Originale (`5.000 vs 4.625`)** | *Master: "Si sentono molto le alte rispetto alle basse"* |
| **`track03_style`** | Drop (`21.5–33.5s`) | `program_level` | Orig (`OP=5`, `5.000`) | Master (`OP=3`, `4.250`) | **Originale (`5 vs 3`)** | **Originale (`5.000 vs 4.250`)** | *Master: "Loudness molto forte, harsh"* |

---

## 6. Riconciliazione Conteggi e Verifica Riproducibile dei Punteggi Medi

### 6.1 Conteggi Vittorie / Pareggi (24 Sessioni Totali = 12 `level_matched` + 12 `program_level`)

| Sottoinsieme Sessioni | N. Sessioni | Regola Primaria (`overall_preference_1_5` pura) | Regola Secondaria (`OP` + spareggio `Mean8`) | Sessioni con Master $\ge$ Originale (Vittorie + Pareggi) |
|---|:---:|---|---|---|
| **`same_track` — `level_matched`** | `9` | **2 Master**, **5 Originale**, **2 Pareggi** | **4 Master**, **5 Originale**, **0 Pareggi** | **`4 / 9`** (`44.4%`) |
| **`stylistic` — `level_matched`** | `3` | **0 Master**, **0 Originale**, **3 Pareggi** | **0 Master**, **0 Originale**, **3 Pareggi** | **`3 / 3`** (`100.0%`, tutti pareggi `5/5`) |
| **Totale `level_matched`** | **`12`** | **2 Master**, **5 Originale**, **5 Pareggi** | **4 Master**, **5 Originale**, **3 Pareggi** | **`7 / 12`** (`58.3%`) |
| **`same_track` — `program_level`** | `9` | **2 Master**, **5 Originale**, **2 Pareggi** | **4 Master**, **5 Originale**, **0 Pareggi** | **`4 / 9`** (`44.4%`) |
| **`stylistic` — `program_level`** | `3` | **1 Master**, **2 Originale**, **0 Pareggi** | **1 Master**, **2 Originale**, **0 Pareggi** | **`1 / 3`** (`33.3%`) |
| **Totale `program_level`** | **`12`** | **3 Master**, **7 Originale**, **2 Pareggi** | **5 Master**, **7 Originale**, **0 Pareggi** | **`5 / 12`** (`41.7%`) |
| **Totale Complessivo Sprint 2** | **`24`** | **5 Master**, **12 Originale**, **7 Pareggi** | **9 Master**, **12 Originale**, **3 Pareggi** | **`12 / 24`** (`50.0%`) |

### 6.2 Derivazione Esatta e Riproducibile del Punteggio Medio `4.72 / 5.0` (e Tutte le Medie Aggregate)

- **Definizione del valore `4.72 / 5.0`:**
  Il punteggio medio **`4.72 / 5.0`** rappresenta la **media aritmetica semplice di tutte le 8 dimensioni qualitative** (`tonal_balance_1_5`, `bass_control_1_5`, `vocal_clarity_1_5`, `transient_punch_1_5`, `stereo_image_1_5`, `dynamic_naturalness_1_5`, `distortion_artifacts_1_5`, `overall_preference_1_5`) ottenute dal **Master DDSP** nelle **12 sessioni `level_matched`** (`9 same_track + 3 stylistic`, totale $12 \times 8 = 96$ valutazioni, senza alcuna cella vuota):
  $$\mu_{\text{Mean8, LM}}(\text{Master DDSP}) = \frac{\sum_{s=1}^{12} \sum_{k=1}^{8} x_{s,k}^{\text{Master}}}{96} = \frac{333_{\text{same\_track}} + 120_{\text{stylistic}}}{72 + 24} = \frac{453}{96} = 4.71875 \approx \mathbf{4.72 / 5.0}$$
- **Trattamento di celle non valutate (`NV`) e pareggi nel calcolo delle medie:**
  - I punteggi appartenenti a sessioni concluse in pareggio sono inclusi integralmente nel numeratore e nel denominatore di ciascuna variante.
  - Nelle sessioni `level_matched` (`9 same_track + 3 stylistic`) e `stylistic program_level` (`3 sessioni`) **non vi sono celle vuote** (`0 NV` su `96` celle per variante in `level_matched`).
  - Nelle sessioni `same_track program_level` (`9 sessioni`), il Master DDSP ha `72/72` celle compilate (`0 NV`), mentre l'Originale presenta **3 singole celle lasciate vuote (`NV`)** dall'ascoltatore su 72 (`track01 sec03 program_level` -> `tonal_balance_1_5` su `blind_B`; `track03 sec01 program_level` -> `dynamic_naturalness_1_5` su `blind_A`; `track03 sec03 program_level` -> `distortion_artifacts_1_5` su `blind_B`). Tali 3 celle vuote sono escluse dal denominatore dell'Originale (media calcolata sulle `69` celle valide in `same_track program_level`, ossia `93` celle valide su `96` nel totale `program_level`), senza alcuna imputazione o modifica dei dati originali.

| Sottoinsieme Sessioni | Metrica Aggregata | Formula e Frazione Esatta — **Master DDSP** | Formula e Frazione Esatta — **Originale (Mix)** |
|---|---|---|---|
| **Totale `level_matched` (12 sess.)** | **Media 8 dimensioni (`Mean8`)** | $\frac{333 + 120}{72 + 24} = \frac{453}{96} = \mathbf{4.7188 \approx 4.72}$ | $\frac{346 + 120}{72 + 24} = \frac{466}{96} = \mathbf{4.8542 \approx 4.85}$ |
| `same_track` — `level_matched` (9 sess.) | Media 8 dimensioni (`Mean8`) | $\frac{333}{72} = \mathbf{4.6250 \approx 4.63}$ | $\frac{346}{72} = \mathbf{4.8056 \approx 4.81}$ |
| `stylistic` — `level_matched` (3 sess.) | Media 8 dimensioni (`Mean8`) | $\frac{120}{24} = \mathbf{5.0000}$ | $\frac{120}{24} = \mathbf{5.0000}$ |
| **Totale `level_matched` (12 sess.)** | **Media sola `overall_preference_1_5`** | $\frac{38 + 15}{9 + 3} = \frac{53}{12} = \mathbf{4.4167 \approx 4.42}$ | $\frac{42 + 15}{9 + 3} = \frac{57}{12} = \mathbf{4.7500}$ |
| `same_track` — `level_matched` (9 sess.) | Media sola `overall_preference_1_5` | $\frac{38}{9} = \mathbf{4.2222 \approx 4.22}$ | $\frac{42}{9} = \mathbf{4.6667 \approx 4.67}$ |
| `stylistic` — `level_matched` (3 sess.) | Media sola `overall_preference_1_5` | $\frac{15}{3} = \mathbf{5.0000}$ | $\frac{15}{3} = \mathbf{5.0000}$ |
| **Totale `program_level` (12 sess.)** | **Media 8 dimensioni (`Mean8`)** | $\frac{324 + 111}{72 + 24} = \frac{435}{96} = \mathbf{4.5313 \approx 4.53}$ | $\frac{320 + 114}{69 + 24} = \frac{434}{93} = \mathbf{4.6667 \approx 4.67}$ |
| `same_track` — `program_level` (9 sess.) | Media 8 dimensioni (`Mean8`) | $\frac{324}{72} = \mathbf{4.5000}$ | $\frac{320}{69} = \mathbf{4.6377 \approx 4.64}$ *(3 celle `NV` escluse)* |
| `stylistic` — `program_level` (3 sess.) | Media 8 dimensioni (`Mean8`) | $\frac{111}{24} = \mathbf{4.6250 \approx 4.63}$ | $\frac{114}{24} = \mathbf{4.7500}$ |
| **Totale `program_level` (12 sess.)** | **Media sola `overall_preference_1_5`** | $\frac{34 + 12}{9 + 3} = \frac{46}{12} = \mathbf{3.8333 \approx 3.83}$ | $\frac{40 + 13}{9 + 3} = \frac{53}{12} = \mathbf{4.4167 \approx 4.42}$ |
| `same_track` — `program_level` (9 sess.) | Media sola `overall_preference_1_5` | $\frac{34}{9} = \mathbf{3.7778 \approx 3.78}$ | $\frac{40}{9} = \mathbf{4.4444 \approx 4.44}$ |
| `stylistic` — `program_level` (3 sess.) | Media sola `overall_preference_1_5` | $\frac{12}{3} = \mathbf{4.0000}$ | $\frac{13}{3} = \mathbf{4.3333 \approx 4.33}$ |

---

## 7. Discussione Critica e Limiti Metodologici

1. **Valutazione Esplorativa Single-Listener ($N=1$) e Non-Indipendenza delle Sessioni:**
   - La campagna d'ascolto è stata condotta come **valutazione esplorativa single-listener ($N=1$)** dall'autore della tesi, e non costituisce uno studio statistico inferenziale multi-ascoltatore (tipo MUSHRA/ITU-R BS.1534 con panel di soggetti indipendenti).
   - Le 24 sessioni dello Sprint 2 rappresentano misure ripetute effettuate dallo stesso ascoltatore su sezioni diverse (`sec01`, `sec02`, `sec03`) e modalità diverse (`level_matched`, `program_level`, `same_track`, `stylistic`) estratte dagli stessi 3 brani; pertanto **non devono essere interpretate come campioni statistici indipendenti**.
2. **Confronto a Loudness Pareggiata (`level_matched`) vs Livello di Programma (`program_level`):**
   - In `level_matched`, il Master DDSP ottiene una media complessiva sulle 8 dimensioni di `4.72 / 5.0` (`453/96`, contro `4.85 / 5.0` dell'Originale) e una `overall_preference_1_5` media di `4.42 / 5.0` (contro `4.75 / 5.0` dell'Originale), superando o pareggiando l'Originale in **`7 sessioni su 12`** (`2 vittorie + 5 pareggi` con `OP` pura; `4 vittorie + 3 pareggi` includendo lo spareggio su `Mean8`). In particolare, nei 3 confronti `stylistic` `level_matched` entrambe le versioni hanno ricevuto `5/5` su tutte le dimensioni (`3 pareggi`), mentre in `same_track` il Master DDSP ha vinto nettamente su `track02` (`sec02`, `sec03`) e ha prevalso allo spareggio su `Mean8` in `track03` (`sec01`, `sec03`).
   - In `program_level`, l'esito dipende fortemente dall'headroom del mix di partenza rispetto al target di loudness del reference: su `track02` (`same_track`, mix con alto headroom a `-22.03 LUFS`) e su `track01_style` (`stylistic`, dove il reference commerciale ha guidato un'attenuazione morbida a `-17.84 LUFS` eliminando il clipping dell'originale), il Master DDSP supera nettamente l'Originale (`5 vs 4` e `5 vs 3`). Al contrario, quando l'ottimizzatore spinge fortemente il limiter e le alte frequenze su mix già densi (`track01 same_track` a `-10.51 LUFS`, `track03 same_track` e `track03_style`), l'ascoltatore penalizza l'*harshness* e la compressione eccessiva a favore dell'Originale (`3 vittorie Master, 7 Originale, 2 Pareggi` con `OP` pura; `5 Master, 7 Originale` con spareggio `Mean8`).
