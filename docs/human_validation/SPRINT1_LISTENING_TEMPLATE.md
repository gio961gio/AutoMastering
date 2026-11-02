# Scheda di Valutazione d'Ascolto — Sprint 1 (Esperimento A)

> [!WARNING]
> Compila le tabelle di ascolto cieco **prima** di aprire il file `track01_blind_key.json`.

## 1. Metadati Sessione e Verifica Tecnica Preliminare

| Campo | Valore compilato dall'ascoltatore |
|---|---|
| **ID Anonimo Brano** | `track01` |
| **Data e Ora Sessione** | `YYYY-MM-DD HH:MM` |
| **Ascoltatore** | Giovanni Antonecchia |
| **Sistema di Ascolto (Cuffie / Monitor + Interfaccia)** | *Da compilare* |
| **Livello di Ascolto Indicativo** | *Da compilare* |
| **Policy Legacy Baseline usata (`--legacy-mode`)** | `[ ] strict (solo A e C)` / `[ ] clamped_to_current_bounds` / `[ ] raw_theta_on_current_engine` |
| **Conferma stesso input e trim diagnostico uniforme (`-6.0 dBTP`)** | `[ ] Sì` / `[ ] No` (Trim applicato: `_____ dB`) |
| **Conferma identica durata e allineamento temporale (159 smp compensati)** | `[ ] Sì` / `[ ] No` (Campioni: `_________`, Durata: `_____ s`) |
| **Null error A (True Bypass) vs Input Trimmato (`max_err_dbfs`)** | `_________ dBFS` |

---

## 2. Valutazione Cieca — Copie Loudness-Matched (`blind/blind_A.wav`, `blind_B.wav`, `blind_C.wav`)

*Scala 1–5 per ciascun criterio:*  
`1` = Decisamente peggiore / problema evidente | `2` = Leggermente peggiore | `3` = Neutro / trasparente | `4` = Leggermente migliore | `5` = Decisamente migliore | `NV` = Non valutabile

| Criterio (1–5 o NV) | `blind_A.wav` | `blind_B.wav` | `blind_C.wav` |
|---|---|---|---|
| **Tonal Balance** (equilibrio spettrale) | | | |
| **Bass Control** (controllo basse frequenze) | | | |
| **Vocal Clarity** (chiarezza voce/strumento guida) | | | |
| **Transient Punch** (integrità attacchi e percussioni) | | | |
| **Stereo Image / Mono Compatibility** | | | |
| **Dynamic Naturalness** (assenza di compressione innaturale) | | | |
| **Distortion / Artifacts** (`5` = zero artefatti, `1` = distorsione evidente) | | | |
| **Overall Preference** (1–5) | | | |

### Confronto diretto e udibilità delle differenze (Loudness-Matched)

- **Differenza percepibile tra le varianti?**
  - `blind_A` vs `blind_B`: `[ ] Nessuna differenza percepibile` / `[ ] Differenza sottile` / `[ ] Differenza netta`
  - `blind_A` vs `blind_C`: `[ ] Nessuna differenza percepibile` / `[ ] Differenza sottile` / `[ ] Differenza netta`
  - `blind_B` vs `blind_C`: `[ ] Nessuna differenza percepibile` / `[ ] Differenza sottile` / `[ ] Differenza netta`
- **Variante più neutra / fedele all'originale:** `[ ] blind_A` / `[ ] blind_B` / `[ ] blind_C` / `[ ] Indistinguibili`
- **Fiducia nella valutazione (1–5):** `___`
- **Presenza di artefatti specifici (click, dropout, pumping, distorsione, alterazione stereo):**
  - `blind_A`: *Da compilare*
  - `blind_B`: *Da compilare*
  - `blind_C`: *Da compilare*

---

## 3. Ascolto a Livello di Programma (`exports/track01_A_true_bypass.wav`, `B`, `C`)

| Variante | Differenza di volume percepita | Comportamento dinamico / saturazione / spinta | Note su click, pumping o distorsione |
|---|---|---|---|
| **A — True Bypass** | | | |
| **B — Legacy Baseline** | | | |
| **C — Corrected Neutral** | | | |

---

## 4. Svelamento Chiave (`track01_blind_key.json`) e Conclusioni Sprint 1

*(Compilare solo dopo aver completato la Sezione 2)*

- `blind_A.wav` corrisponde a: `____________________`
- `blind_B.wav` corrisponde a: `____________________`
- `blind_C.wav` corrisponde a: `____________________`

**Osservazioni conclusive sull'efficacia di `THETA_NEUTRAL` (C) rispetto al True Bypass (A) e alla Legacy Baseline (B):**

*Da compilare dall'ascoltatore dopo l'esperimento.*
