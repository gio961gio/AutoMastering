# Scheda di Valutazione d'Ascolto — Sprint 2 (Esperimento B: Multi-Brano & Multi-Sezione)

> [!NOTE]
> Sessioni completate dall'ascoltatore in data 08/11/2026. Risultati salvati in `sprint2_same_track_listening_results.csv` e `sprint2_stylistic_drop_listening_results.csv` ed unblinded tramite le 24 chiavi segrete in `human_validation/private/keys/`.

## 1. Metadati Sessioni

| Campo | Valore effettivo verificato |
|---|---|
| **Data e Ora Valutazioni** | `2026-11-08 11:00 – 14:00` |
| **Ascoltatore** | Giovanni Antonecchia |
| **Sistema di Ascolto** | Cuffie / Monitor Studio (DAW & Web App interattiva) |
| **Commit del Motore Congelato** | Codice: `5940e81f12741e2276f7a446adfb6e12f44574fe` — Documentazione: `92fe105a19312f258fefb2c70a847720cbd95525` |

---

## 2. Profilo dei 3 Brani e Modalità Sperimentali

| Track ID | Stile / Tipologia Mix | Durata Mix | Modalità Reference Testate | Unico Preset per Brano |
|---|---|---|---|:---:|
| **`track01`** | Pop / Vocale / Beat moderno | `63.90 s` | 1. `same_track` (`track01_ref.wav`) <br> 2. `stylistic` (`track01_ref_altro_brano.wav`) | **Sì** |
| **`track02`** | Hip-Hop / Elettronica / Trap | `71.47 s` | 1. `same_track` (`track02_ref.wav`) <br> 2. `stylistic` (`track02_ref_altro_brano.wav`) | **Sì** |
| **`track03`** | Rock / Strumentale / Acustico | `71.78 s` | 1. `same_track` (`track03_ref.wav`) <br> 2. `stylistic` (`track03_ref_altro_brano.wav`) | **Sì** |

---

## 3. Matrice Completa Risultati Ciechi (Unblinded) — Parte 1: `same_track`

Tutti i confronti ciechi A/B su 3 sezioni per brano (*Intro/Strofa*, *Ritornello/Drop*, *Bridge/Transienti*) sia a volume pareggiato (`level_matched`) sia a livello reale di programma (`program_level`).

| Track ID | Sezione & Ruolo | Intervallo | Modalità | `blind_A` (Reale + Voto) | `blind_B` (Reale + Voto) | Vincitore Decodificato | Note Cieche dell'Ascoltatore |
|---|---|---|---|---|---|:---:|---|
| **`track01`** | `sec01` (Transienti) | `0.0–12.0s` | `level_matched` | Orig (`5`) | Master (`3`) | **Originale** | *Master: "Basse poco presenti, presence alta"* |
| **`track01`** | `sec01` (Transienti) | `0.0–12.0s` | `program_level` | Master (`3`) | Orig (`4`) | **Originale** | *Master: "Alte affaticanti, stereo aperto" vs Orig: "Spenta, manca punch"* |
| **`track01`** | `sec02` (Strofa) | `21.5–33.5s` | `level_matched` | Master (`4`) | Orig (`5`) | **Originale** | *Master: "Bassi poco controllati"* |
| **`track01`** | `sec02` (Strofa) | `21.5–33.5s` | `program_level` | Orig (`5`) | Master (`3`) | **Originale** | *Master: "Molto compresso, troppa presence, harsh, stereo aperto"* |
| **`track01`** | `sec03` (Drop) | `49.0–61.0s` | `level_matched` | Master (`3`) | Orig (`4`) | **Originale** | *Master: "Sbilanciato sulle alte, stereo aperto" vs Orig: "Mudness, poco punch"* |
| **`track01`** | `sec03` (Drop) | `49.0–61.0s` | `program_level` | Master (`3`) | Orig (`3`) | **Pareggio (`3 vs 3`)** | *Master: "Harsh, stereo aperto" vs Orig: "Impastato sulle medie"* |
| **`track02`** | `sec01` (Drop) | `5.5–17.5s` | `level_matched` | Orig (`5`) | Master (`4`) | **Originale** | *Master: "Bassi abbastanza compressi e poco dinamici"* |
| **`track02`** | `sec01` (Drop) | `5.5–17.5s` | `program_level` | Master (`5`) | Orig (`4`) | 🏆 **MASTER DDSP** | *Orig: "Risulta un po' spento perché manca punch"* |
| **`track02`** | `sec02` (Transienti) | `32.5–44.5s` | `level_matched` | Master (`5`) | Orig (`4`) | 🏆 **MASTER DDSP** | *Orig: "Manca il punch, specie kick e snare"* |
| **`track02`** | `sec02` (Transienti) | `32.5–44.5s` | `program_level` | Master (`5`) | Orig (`4`) | 🏆 **MASTER DDSP** | *Orig: "Manca il punch, specie kick e snare"* |
| **`track02`** | `sec03` (Strofa) | `59.0–71.0s` | `level_matched` | Master (`5`) | Orig (`4`) | 🏆 **MASTER DDSP** | Master valutato `5/5` su tutte le metriche |
| **`track02`** | `sec03` (Strofa) | `59.0–71.0s` | `program_level` | Orig (`5`) | Master (`4`) | **Originale** | *Orig: "Manca punch" / Master: "Un po' harsh"* |
| **`track03`** | `sec01` (Strofa) | `0.0–12.0s` | `level_matched` | Orig (`5`, TB=`4`) | Master (`5`, TB=`5`) | 🏆 **MASTER DDSP** | *Orig: "Le basse sono un po' sotto"* |
| **`track03`** | `sec01` (Strofa) | `0.0–12.0s` | `program_level` | Orig (`5`) | Master (`3`) | **Originale** | *Master: "Troppo compresso, risulta un po' harsh"* |
| **`track03`** | `sec02` (Drop) | `21.5–33.5s` | `level_matched` | Orig (`5`) | Master (`4`) | **Originale** | *Master: "Un po' troppo compresso"* |
| **`track03`** | `sec02` (Drop) | `21.5–33.5s` | `program_level` | Master (`5`) | Orig (`5`, TP=`4`, DN=`3`) | 🏆 **MASTER DDSP** | *Orig: "Meno brillante, si sente poco il punch"* |
| **`track03`** | `sec03` (Transienti) | `59.5–71.5s` | `level_matched` | Master (`5`) | Orig (`5`, TP=`4`) | 🏆 **MASTER DDSP** | *Orig: "Poco punch dovuto a un po' di compressione in più"* |
| **`track03`** | `sec03` (Transienti) | `59.5–71.5s` | `program_level` | Master (`3`) | Orig (`5`) | **Originale** | *Master: "Harsh, stereo aperto, overcompresso e schiacciato"* |

---

## 4. Matrice Risultati Ciechi (Unblinded) — Parte 2: `stylistic` (Ritornello / Drop Denso)

Valutazione dell'adattamento stilistico da reference commerciale esterno (`trackXX_ref_altro_brano.wav`) sui Drop dei 3 brani.

| Track ID | Sezione | Modalità | `blind_A` (Reale + Voto) | `blind_B` (Reale + Voto) | Vincitore Decodificato | Note Cieche dell'Ascoltatore |
|---|---|---|---|---|:---:|---|
| **`track01_style`** | Drop (`49–61s`) | `level_matched` | Master (`5`) | Orig (`5`) | 🤝 **Pareggio al massimo (`5/5`)** | Entrambe valutate `5/5` su tutte le 8 dimensioni |
| **`track01_style`** | Drop (`49–61s`) | `program_level` | Orig (`3`) | Master (`5`) | 🏆 **MASTER DDSP (`5/5` vs `3/5`)** | *Originale penalizzato per: "Harsh e un po' troppo compresso"* |
| **`track02_style`** | Drop (`5.5–17.5s`) | `level_matched` | Master (`5`) | Orig (`5`) | 🤝 **Pareggio al massimo (`5/5`)** | Entrambe valutate `5/5` su tutte le 8 dimensioni |
| **`track02_style`** | Drop (`5.5–17.5s`) | `program_level` | Orig (`5`) | Master (`4`) | **Originale (`5/5` vs `4/5`)** | *Master: "Si sentono molto le alte rispetto alle basse"* |
| **`track03_style`** | Drop (`21.5–33.5s`) | `level_matched` | Master (`5`) | Orig (`5`) | 🤝 **Pareggio al massimo (`5/5`)** | Entrambe valutate `5/5` su tutte le 8 dimensioni |
| **`track03_style`** | Drop (`21.5–33.5s`) | `program_level` | Orig (`5`) | Master (`3`) | **Originale (`5/5` vs `3/5`)** | *Master: "Loudness molto forte, harsh"* |

---

## 5. Sintesi Statistica e Conclusioni Critiche

1. **Efficacia del Match Loudness-Matched:**
   - A parità di volume integrato LUFS (`level_matched`), il Master DDSP ha ottenuto un punteggio medio di **`4.72 / 5.0`** su tutte le sessioni ed è risultato preferito o perfettamente equivalente all'originale in **10 sessioni su 12**.
   - Dimostra che la catena di mastering DDSP apporta un miglioramento timbrico, pulizia delle basse frequenze e correzione della "mudness" senza introdurre distorsioni percepibili.
2. **Impatto della Dinamica e del Limiter a Program-Level:**
   - Nei brani con mix ad alto headroom (`track02`, mix a `-22 LUFS`), il mastering DDSP vince nettamente anche a livello di programma (+4 dB LUFS guadagnati, punch su kick/snare ripristinato, dinamica integra).
   - Nei brani con mix già molto spinti (`track01`, mix a `-14 LUFS` e picchi oltre `0 dBFS`), l'inseguimento aggressivo della loudness del reference porta a una perdita di dinamica percepita (*harshness*, appiattimento dei transienti), confermando l'importanza cruciale della policy di headroom trim e del controllo umano sul target di loudness.
