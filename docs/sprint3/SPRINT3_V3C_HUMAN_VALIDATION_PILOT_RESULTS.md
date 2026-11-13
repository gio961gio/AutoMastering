# SPRINT 3 — V3C PILOT HUMAN VALIDATION RESULTS (BLIND PACK V2)

**Repository:** `gio961gio/AutoMastering` | **Branch:** `human-validation-package`
**Baseline Tecnica Precedente:** `53bd123d19bba57a18bd0a5970e6a1554329b5ec` (`5709c3f` per risultati oggettivi V3c Pilot)
**Data Completamento Ascolto Cieco:** `2026-11-13T16:42:21.684Z`
**Ambiente di Ascolto:** Ableton Live 12 Suite (`48 kHz`, `PCM_24`, clip allineate `1.1.1`, fader `0.0 dB`, `Warp = OFF`, controllo Stereo + Mono Fold-Down sul Master)

> **Nota Procedurale:** La decodifica rispetto al mapping privato V2 è stata eseguita **esclusivamente dopo** la registrazione delle risposte cieche del valutatore (`blind_listening_v3c_v2_responses.json`). Lo stato del repository (`enable_v3c=False`, zero nuovi training/ottimizzazioni) resta invariato in attesa della decisione del supervisore.

---

## 1. Esito dell'Ascolto Cieco e Decodifica Post-Valutazione

| Pair | Traccia | `A` | `B` | Immagine Stereo | Solidità Mono | Controllo Basso & Sub | Trasparenza & Timbro | **Preferenza Globale** | Note Valutatore |
|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| `PAIR01` | `track03` | `Treatment_V3c` | `Control_V0p` | Pari | Pari | Pari | Pari | **Pari** | Sembrano uguali. |
| `PAIR02` | `track01` | `Treatment_V3c` | `Control_V0p` | Pari | Pari | Pari | Pari | **Pari** | Sembrano uguali. |
| `PAIR03` | `track02` | `Treatment_V3c` | `Control_V0p` | Pari | Pari | Pari | Pari | **Pari** | Sembrano uguali. |

**Riepilogo Conteggi (`N = 3` coppie `same_track`):**
- Preferenze `Treatment_V3c`: `0 / 3`
- Preferenze `Control_V0p`: `0 / 3`
- Pareggi (`Pari`): **`3 / 3`**
- Regressioni percettive segnalate: **`0 / 3`**

---

## 2. Null-Test Fisico `(A - B)` sui WAV V2 & Contesto Parametrico

| Pair | Traccia | Attenuazione Comune V2 | Max TP `4x` (dBTP) | `|ΔLUFS|` Coppia | Picco `(A - B)` | **RMS `(A - B)`** | Principali Variazioni Parametriche (`V3c - V0p`) |
|:---|:---|---:|---:|---:|---:|---:|:---|
| `PAIR01` | `track03` | -0.2663 dB | -0.3501 | 0.000077 LU | -49.48 dBFS | **-65.24 dBFS** | `w_low -0.00026`, `crossover_f_high -4.42 Hz`, `||Δz||₂=0.00208` |
| `PAIR02` | `track01` | -0.6575 dB | -0.3501 | 0.001739 LU | -27.91 dBFS | **-46.97 dBFS** | `w_low -0.00071`, `crossover_f_high -49.97 Hz`, `||Δz||₂=0.01458` |
| `PAIR03` | `track02` | -0.2787 dB | -0.3500 | 0.000038 LU | -77.04 dBFS | **-98.63 dBFS** | `w_low -0.00439`, `crossover_f_high +0.01 Hz`, `||Δz||₂=0.00628` |

---

## 3. Interpretazione Sperimentale Coniugata (Umano + Oggettivo)

1. **Coerenza totale tra ascolto umano e misura fisica:** su `PAIR01` (`track03`, RMS diff `-65.24 dBFS`) e `PAIR03` (`track02`, RMS diff `-98.63 dBFS`), la differenza energetica tra `Treatment_V3c` e `Control_V0p` è decine di decibel sotto la soglia di udibilità. Su `PAIR02` (`track01`, RMS diff `-46.97 dBFS`), quasi tutto il segnale differenza deriva dallo spostamento di `crossover_f_high` (`-49.97 Hz`, lieve rotazione di fase vicino a `2.65 kHz`) con larghezze Imager variate di `< 0.001`, risultando anch'esso indistinguibile in ascolto A/B.
2. **Causa nell'ottimizzazione (`same_track`):** sulle 3 tracce `same_track`, `Control_V0p` presentava già `loss_mono_drop = 0.0` e una `loss_v3c_total_weighted` compresa tra `0.0008` e `0.0025` contro una `loss_v0p_total` tra `2.82` e `10.36` (`< 0.1%` dell'obiettivo totale).
3. **Stato operativo attuale:** zero regressioni uditive introdotte da `V3c`, ma impatto percettivo nullo sul sottoinsieme `same_track` con gli attuali pesi pilota. Restano aperti i passi successivi da decidere con il supervisore (mantenere `enable_v3c=False` su `same_track`, affrontare il ceiling **True Peak 4x** nel DSP/loss visto lo sforamento di `-0.30 dBTP` in `Control_V0p`, oppure valutare `V3c` in scenari `stylistic` con pesi dedicati).
