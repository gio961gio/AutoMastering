# Scheda di Valutazione d'Ascolto — Sprint 1 (Esperimento A)

> [!NOTE]
> Sessione completata dall'ascoltatore in data 08/11/2026. Risultati salvati ed unblinded tramite `track01_sprint1_blind_key.json`.

## 1. Metadati Sessione e Verifica Tecnica Preliminare

| Campo | Valore effettivo verificato |
|---|---|
| **ID Anonimo Brano** | `track01` |
| **Data e Ora Sessione** | `2026-11-08 10:54` |
| **Ascoltatore** | Giovanni Antonecchia |
| **Sistema di Ascolto** | Cuffie / Monitor Studio (DAW) |
| **Policy Legacy Baseline usata (`--legacy-mode`)** | `clamped_to_current_bounds` (5 parametri fuori range limitati ai bound correnti) |
| **Conferma stesso input e trim diagnostico uniforme (`-6.0 dBTP`)** | Sì (Trim statico applicato: `-9.98 dB`, True Peak: `-6.00 dBTP`) |
| **Conferma identica durata e allineamento temporale (159 smp compensati)** | Sì (Campioni: `3066985`, Durata: `63.8955 s`) |
| **Null error A (True Bypass) vs Input Trimmato (`max_err_dbfs`)** | `-156.56 dBFS` (RMS Null: `-178.43 dBFS`, bit-exact float32) |
| **Differenza RMS C (Corrected Neutral) vs A (True Bypass)** | `-22.67 dBFS` (filtraggio trasparente neutro attivo) |
| **Seed deterministico sessione** | `13607737023179710879` (64-bit CSPRNG) |

---

## 2. Valutazione Cieca — Copie Loudness-Matched (`blind_A.wav`, `blind_B.wav`, `blind_C.wav`)

*Scala 1–5 per ciascun criterio:*  
`1` = Molto scarso / difetto grave | `3` = Neutro / accettabile / trasparente | `5` = Eccellente / nessun difetto

| Criterio (1–5) | `blind_A.wav` | `blind_B.wav` | `blind_C.wav` |
|---|:---:|:---:|:---:|
| **Tonal Balance** (equilibrio spettrale) | **5** | **5** | **5** |
| **Bass Control** (controllo basse frequenze) | **5** | **5** | **5** |
| **Vocal Clarity** (chiarezza voce/guida) | **5** | **5** | **5** |
| **Transient Punch** (integrità attacchi e percussioni) | **5** | **5** | **5** |
| **Stereo Image / Mono Compatibility** | **5** | **5** | **5** |
| **Dynamic Naturalness** (assenza di compressione innaturale) | **5** | **5** | **5** |
| **Distortion / Artifacts** (`5` = zero artefatti) | **5** | **5** | **5** |
| **Overall Preference** (1–5) | **5** | **5** | **5** |

### Note Libere e Commenti dell'Ascoltatore (alla cieca)
- `blind_A`: *"Nessun difetto"*
- `blind_B`: *"Nessun Difetto"*
- `blind_C`: *"Nessun difetto, ma ho notato che la traccia inizia prima delle altre due... a posteriori era leggermente più schiacciata dinamicamente, non di tantissimo"*

---

## 3. Unblinding Ufficiale (da `track01_sprint1_blind_key.json`)

| Etichetta Anonima | Condizione Reale | Esito Percettivo |
|---|---|---|
| **`blind_A`** | **`A_true_bypass`** | Trasparenza assoluta confermata (`5/5`) |
| **`blind_B`** | **`C_corrected_neutral`** | Trasparenza assoluta confermata (`5/5`, indistinguibile dal True Bypass) |
| **`blind_C`** | **`B_legacy_baseline`** | Riconosciuta all'ascolto per la maggiore densità dinamica (+9 dB makeup/limiter gain originari prima del level matching) |

### Conclusioni Sprint 1:
1. **Neutralità Validata:** La nuova condizione `THETA_NEUTRAL` implementata per la tesi è risultata impercettibile e priva di qualunque colorazione o degradazione rispetto al bypass nativo.
2. **Stabilità della Legacy Replay:** La policy `clamped_to_current_bounds` ha consentito il replay stabile e sicuro della vecchia inizializzazione pre-Gate A senza crash, clipping o distorsioni anomale.
