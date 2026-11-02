# Report Conclusivo di Validazione Umana — Sprint 1 & Sprint 2

**Candidato:** Giovanni Antonecchia  
**Corso di Laurea:** Master's Thesis in Sound and Music Engineering — Politecnico di Milano (DEIB)  
**Commit motore congelato (Codice):** `5940e81f12741e2276f7a446adfb6e12f44574fe`  
**Stato corrente:** `[ ] Human Experiment Pending` / `[ ] Human Experiment Completed`

> [!IMPORTANT]
> Non marcare questo report come `Human Experiment Completed` finché tutte le sessioni di ascolto reali non sono state eseguite e le schede `SPRINT1_LISTENING_TEMPLATE` e `SPRINT2_LISTENING_TEMPLATE` non sono state compilate integralmente dall'utente.

---

## 1. Riepilogo Esecutivo Esperimento A (Sprint 1: Neutralità, True Bypass e Legacy Baseline)

- **Brano utilizzato (ID anonimo e SHA-256 sorgente):** `track01` — SHA-256: `________________`
- **Stato ricostruzione Legacy Baseline (`86ae20f08dc7fbebc7f67af064a15b648a492fc7`):**
  - Vettore storico `z = 0.5` ricostruito integralmente (41 parametri).
  - 5 parametri su 41 eccedono i limiti fisici del motore corrente congelato (`comp_low_ratio`, `comp_low_attack_ms`, `comp_mid_ratio`, `comp_high_ratio`, `saturator_drive_db`) -> emesso finding `LEGACY_BASELINE_NOT_REPRODUCIBLE` in modalità `strict`.
  - Policy adottata per l'ascolto della variante B: `________________`
- **Risultati oggettivi delle esportazioni A / B / C:**
  - Durata comune (campioni / secondi): `________________`
  - Trim diagnostico statico applicato: `________________ dB`
  - Null test `A (True Bypass)` vs input trimmato (`max_err_dbfs`): `________________ dBFS`
  - Differenza `C (Corrected Neutral)` vs `A (True Bypass)` (`rms_null_dbfs`): `________________ dBFS`
- **Esito dell'ascolto cieco e program-level (Sprint 1):**
  - *Da compilare dall'ascoltatore dopo l'esperimento.*

---

## 2. Riepilogo Esecutivo Esperimento B (Sprint 2: Valutazione Multi-Brano e Multi-Sezione)

### 2.1 Tabella Sintetica Misure Oggettive Pre/Post Mastering

| Track ID | Mode (`same_track` / `stylistic`) | Pre LUFS (BS.1770-5) | Post LUFS (BS.1770-5) | Pre True Peak (`dBTP`) | Post True Peak (`dBTP`) | Pre PLR (`dB`) | Post PLR (`dB`) | Metrica Spettrale Applicabile (Pre -> Post) |
|---|---|---|---|---|---|---|---|---|
| `track01` | | | | | | | | |
| `track02` | | | | | | | | |
| `track03` | | | | | | | | |

### 2.2 Sintesi dei Giudizi Umani Ciechi (Originale vs Master Ottimizzato)

| Track ID | Sezione 01 (Quiet) — Pref. LM / Prog | Sezione 02 (Dense) — Pref. LM / Prog | Sezione 03 (Transient) — Pref. LM / Prog | Coerenza multi-sezione del preset unico | Artefatti rilevati |
|---|---|---|---|---|---|
| `track01` | | | | | |
| `track02` | | | | | |
| `track03` | | | | | |

---

## 3. Conclusioni Critiche per la Tesi

1. **Verifica della neutralità operativa (Sprint 1):** *Da compilare.*
2. **Tenuta del preset unico su più sezioni del brano (Sprint 2):** *Da compilare.*
3. **Confronto tra metriche oggettive e percezione d'ascolto (casi in cui una loss più bassa non ha coinciso con un suono migliore o viceversa):** *Da compilare.*
