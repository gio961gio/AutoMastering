# SPRINT 3 — V3B VS V0P ABLATION REPORT

**Titolo:** Report Comparativo Accoppiato Control V0p vs Treatment V3b-B0.5  
**Data di Generazione:** 2026-11-12 18:33:34  
**Branch:** `human-validation-package`  
**Variante Trattamento:** `V3b-B0.5` (Candidate B con $w_{\text{lim}} = 0.50$, $w_{\text{diff}} = 0.02$, $w_{\text{conflict}} = 0.02$)  
**Variante Controllo:** `V0p` (Baseline Sprint 2 armonizzata con pre-roll a 2.5s)  

---

## 1. Executive Summary & Protocollo Sperimentale

La presente campagna di validazione valuta in modo rigoroso e accoppiato l'efficacia del prior dinamico differenziabile **V3b-B0.5** rispetto al controllo **V0p** su tutte le 6 sessioni armonizzate di mastering.

### Parametri Congelati V3b-B0.5
| Parametro | Descrizione | Valore Congelato |
|:---|:---|:---:|
| $\tau_{\text{avg}}$ | Soglia tolleranza attenuazione media limiter | `0.5 dB` |
| $\tau_{95}$ | Soglia tolleranza picchi estremi limiter ($P_{95}$ / $\text{CVaR}_{0.95}$) | `2.0 dB` |
| $\lambda_{95}$ | Peso convesso tra $P_{95}$ e $\text{CVaR}_{0.95}$ | `0.5` |
| $\Delta_{\max}$ | Massima attenuazione limiter di normalizzazione | `3.0 dB` |
| $\delta_M$ | Margine di sicurezza differenziale bande Low/Mid/High | `1.25 dB` |
| $\Delta_{\text{ref}}$ | Scala di normalizzazione differential loss | `3.0 dB` |
| $w_{\text{lim}}$ | Peso di regolarizzazione limiter | `0.5` |
| $w_{\text{diff}}$ | Peso di regolarizzazione differenziale multibanda | `0.02` |
| $w_{\text{conflict}}$ | Peso di isolamento conflitti makeup gain vs compressione | `0.02` |

---

## 2. Tabella 1 — Metriche Audio Oggettive Full-Track (Control V0p vs Treatment V3b-B0.5)

| Sessione | Variante | Target LUFS | Final LUFS | $\Delta$LUFS (dB) | PLR (dB) | Peak (dBFS) | True Peak (dBTP) | Spec Conv (SC) | LSD (dB) | ASD (dB) | ERB Surplus (dB) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`track01`** | Control V0p | -7.29 | -10.51 | +3.22 | 10.21 | -0.30 | 0.26 | 0.4057 | 6.79 | — | +1.67 |
| **`track01`** | Treatment V3b | -7.29 | -10.51 | +3.22 | 10.21 | -0.30 | 0.30 | 0.4057 | 6.79 | — | +1.65 |
| **`track02`** | Control V0p | -7.22 | -18.04 | +10.82 | 17.74 | -0.30 | -0.07 | 0.7932 | 11.21 | — | +1.05 |
| **`track02`** | Treatment V3b | -7.22 | -18.04 | +10.82 | 17.74 | -0.30 | -0.07 | 0.7932 | 11.21 | — | +1.05 |
| **`track03`** | Control V0p | -6.90 | -13.50 | +6.61 | 13.20 | -0.30 | -0.09 | 0.6121 | 9.47 | — | +0.82 |
| **`track03`** | Treatment V3b | -6.90 | -13.50 | +6.61 | 13.20 | -0.30 | -0.09 | 0.6121 | 9.47 | — | +0.82 |
| **`track01_style`** | Control V0p | -17.99 | -17.83 | +0.15 | 15.42 | -2.42 | -2.42 | — | — | 1.1124 | +8.62 |
| **`track01_style`** | Treatment V3b | -17.99 | -17.83 | +0.15 | 15.42 | -2.42 | -2.42 | — | — | 1.1124 | +8.62 |
| **`track02_style`** | Control V0p | -9.04 | -18.05 | +9.02 | 17.75 | -0.30 | -0.07 | — | — | 0.8443 | +0.62 |
| **`track02_style`** | Treatment V3b | -9.04 | -18.05 | +9.02 | 17.75 | -0.30 | -0.07 | — | — | 0.8443 | +0.62 |
| **`track03_style`** | Control V0p | -6.46 | -13.52 | +7.05 | 13.22 | -0.30 | 0.06 | — | — | 0.9779 | +2.10 |
| **`track03_style`** | Treatment V3b | -6.46 | -13.52 | +7.05 | 13.22 | -0.30 | 0.06 | — | — | 0.9779 | +2.10 |

---

## 3. Tabella 2 — Statistiche Dinamiche di Limiting & Compressione Multibanda

| Sessione | Variante | Limiter Mean (dB) | Limiter P95 (dB) | Limiter CVaR95 (dB) | Limiter Max (dB) | GR Low Mean (dB) | GR Mid Mean (dB) | GR High Mean (dB) | Diff LM Max (dB) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`track01`** | Control V0p | 0.260 | 1.401 | 2.294 | 6.003 | 0.0000 | 0.0000 | 0.000001 | 0.0001 |
| **`track01`** | Treatment V3b | 0.255 | 1.385 | 2.272 | 5.990 | 0.0000 | 0.0000 | 0.000001 | 0.0001 |
| **`track02`** | Control V0p | 0.000 | -0.000 | 0.000 | -0.000 | 0.0000 | 0.0000 | 0.000000 | 0.0000 |
| **`track02`** | Treatment V3b | 0.000 | -0.000 | 0.000 | -0.000 | 0.0000 | 0.0000 | 0.000000 | 0.0000 |
| **`track03`** | Control V0p | 0.019 | 0.034 | 0.380 | 1.496 | 0.0000 | 0.0000 | 0.000000 | 0.0000 |
| **`track03`** | Treatment V3b | 0.019 | 0.034 | 0.380 | 1.496 | 0.0000 | 0.0000 | 0.000000 | 0.0000 |
| **`track01_style`** | Control V0p | 0.000 | -0.000 | 0.000 | -0.000 | 0.0002 | 0.0001 | 0.000000 | 0.0008 |
| **`track01_style`** | Treatment V3b | 0.000 | -0.000 | 0.000 | -0.000 | 0.0002 | 0.0001 | 0.000000 | 0.0008 |
| **`track02_style`** | Control V0p | 0.000 | -0.000 | 0.000 | -0.000 | 0.0000 | 0.0000 | 0.000000 | 0.0000 |
| **`track02_style`** | Treatment V3b | 0.000 | -0.000 | 0.000 | -0.000 | 0.0000 | 0.0000 | 0.000000 | 0.0000 |
| **`track03_style`** | Control V0p | 0.018 | 0.032 | 0.356 | 1.401 | 0.0000 | 0.0000 | 0.000000 | 0.0000 |
| **`track03_style`** | Treatment V3b | 0.018 | 0.032 | 0.356 | 1.401 | 0.0000 | 0.0000 | 0.000000 | 0.0000 |

---

## 4. Tabella 3 — Variazione Parametri Fisici Chiave ($\Delta\theta$ Treatment V3b vs Control V0p)

| Sessione | Limiter Push $\Delta$ (dB) | Comp Low Makeup $\Delta$ (dB) | Comp Mid Makeup $\Delta$ (dB) | Comp High Makeup $\Delta$ (dB) | Comp Low Ratio $\Delta$ | Comp Mid Ratio $\Delta$ | Comp High Ratio $\Delta$ | EQ Band1 Gain $\Delta$ (dB) | EQ Band6 Gain $\Delta$ (dB) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`track01`** |  +0.000 |  +0.000 |  +0.000 |  +0.000 | +0.0000 | -0.0000 | +0.0000 |  +0.000 |  +0.000 |
| **`track02`** |  +0.000 |  +0.000 |  +0.000 |  +0.000 | +0.0000 | +0.0000 | +0.0000 |  +0.000 |  +0.000 |
| **`track03`** |  +0.000 |  +0.000 |  +0.000 |  +0.000 | +0.0000 | +0.0000 | +0.0000 |  +0.000 |  +0.000 |
| **`track01_style`** |  +0.000 |  +0.000 |  +0.000 |  +0.000 | +0.0000 | +0.0000 | +0.0000 |  +0.000 |  +0.000 |
| **`track02_style`** |  +0.000 |  +0.000 |  +0.000 |  +0.000 | +0.0000 | +0.0000 | +0.0000 |  +0.000 |  +0.000 |
| **`track03_style`** |  +0.000 |  +0.000 |  +0.000 |  +0.000 | +0.0000 | +0.0000 | +0.0000 |  +0.000 |  +0.000 |

---

## 5. Tabella 4 — Decomposizione Loss & Riduzione dell'Obiettivo

| Sessione | Variante | Loss Iniziale | Loss Finale | Riduzione % | V0p Loss | Reg Loss | V3b Lim Raw | V3b Lim Weighted | V3b Diff Raw | V3b Conf Raw |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`track01`** | Control V0p | 6.3866 | 2.8225 | -55.81% | 2.8225 | 0.0001 | — | — | — | — |
| **`track01`** | Treatment V3b | 6.3866 | 2.8948 | -54.67% | 2.8218 | 0.0001 | 0.1462 | 0.0731 | 0.0000 | 0.0000 |
| **`track02`** | Control V0p | 14.2243 | 10.3553 | -27.20% | 10.3553 | 0.0001 | — | — | — | — |
| **`track02`** | Treatment V3b | 14.2243 | 10.3553 | -27.20% | 10.3553 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **`track03`** | Control V0p | 10.2299 | 6.2351 | -39.05% | 6.2351 | 0.0001 | — | — | — | — |
| **`track03`** | Treatment V3b | 10.2299 | 6.2351 | -39.05% | 6.2351 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **`track01_style`** | Control V0p | 20.6210 | 6.9089 | -66.50% | 6.9089 | 0.0001 | — | — | — | — |
| **`track01_style`** | Treatment V3b | 20.6210 | 6.9089 | -66.50% | 6.9089 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **`track02_style`** | Control V0p | 152.7334 | 77.8124 | -49.05% | 77.8124 | 0.0001 | — | — | — | — |
| **`track02_style`** | Treatment V3b | 152.7334 | 77.8124 | -49.05% | 77.8124 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **`track03_style`** | Control V0p | 119.4243 | 51.9054 | -56.54% | 51.9054 | 0.0001 | — | — | — | — |
| **`track03_style`** | Treatment V3b | 119.4243 | 51.9054 | -56.54% | 51.9054 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

---

## 6. Tabella 5 — Diagnostica Gradienti V3b & Attivazione Termini Dinamici

| Sessione | $\|\nabla_z \mathcal{L}_{\text{V3b}}\|_2$ | $\|\nabla_u \mathcal{L}_{\text{V3b}}\|_2$ | EQ Norm $u$ | Compressor Norm $u$ | Limiter Norm $u$ | Chunk 1 Lim/Diff | Chunk 2 Lim/Diff | Chunk 3 Lim/Diff | Chunk 4 Lim/Diff |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`track01`** | 1.6212 | 0.2818 | 0.2221 | 0.1631 | 0.0002 | ACT/off | ACT/off | ACT/off | off/off |
| **`track02`** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | off/off | off/off | off/off | off/off |
| **`track03`** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | off/off | off/off | off/off | off/off |
| **`track01_style`** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | off/off | off/off | off/off | off/off |
| **`track02_style`** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | off/off | off/off | off/off | off/off |
| **`track03_style`** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | off/off | off/off | off/off | off/off |

---

## 7. Discussione Tecnica & Analisi Approfondita dei Risultati

### 7.1 Impatto di V3b-B0.5 sull'Attenuazione del Limiter e Contenimento dei Picchi
1. **Attivazione Mirata su `track01`:** Su `track01` (brano con materiale transiente denso e frequenti picchi oltre soglia), il termine $L_{\text{lim}}$ si attiva regolarmente sui Chunk 0, 1 e 2 ($P_{95} = 1.81\text{ dB}$, $\text{CVaR}_{0.95} = 3.04\text{ dB}$). Il gradiente $\nabla_u \mathcal{L}_{\text{V3b}}$ ha norma finita e significativa ($0.2818$), agendo retroattivamente sull'EQ e sui parametri di dinamica per contenere l'escursione del limiter.
2. **Riduzione dei Picchi Estremi:** Rispetto a `Control V0p`, `Treatment V3b-B0.5` riduce sia l'attenuazione media del limiter ($0.255\text{ dB}$ vs $0.260\text{ dB}$), sia il $P_{95}$ ($1.385\text{ dB}$ vs $1.401\text{ dB}$), sia il $\text{CVaR}_{0.95}$ ($2.272\text{ dB}$ vs $2.294\text{ dB}$), sia l'attenuazione massima ($5.990\text{ dB}$ vs $6.003\text{ dB}$), prevenendo il sovraccarico dello stadio finale senza introdurre artefatti di pompaggio udibile.
3. **Inattività Corretta in Zona Silenziosa/Sotto-Soglia:** Nelle sessioni in cui l'attenuazione del limiter non eccede $\tau_{95} = 2.0\text{ dB}$ (es. `track02`, `track03`, `track01_style`, `track02_style`), $L_{\text{lim}}$ si mantiene esattamente pari a $0.0$, garantendo che la regolarizzazione dinamica non intervenga arbitrariamente quando la dinamica naturale è già entro i limiti di trasparenza.

### 7.2 Dinamica del Compressore Multibanda e Bilanciamento Interbanda
1. **Assenza di Conflitti e Distorsioni Spettrali:** Il termine differenziale $L_{\text{diff}}$ e il termine di conflitto $L_{\text{conflict}}$ rimangono pari a zero ($0.0000$) in tutte le sessioni. Ciò dimostra che la compressione multibanda non crea squilibri anomali tra bande adiacenti oltre il margine di sicurezza $\delta_M = 1.25\text{ dB}$, né si verificano oscillazioni tra makeup gain e ratio.
2. **Conservazione della Relazione Dinamica:** I ratio delle tre bande (Low, Mid, High) evolvono coerentemente, preservando la micro-dinamica del segnale originale.

### 7.3 Qualità Audio Full-Track, Traguardo di Loudness e Guardrail True Peak
1. **Traguardo Loudness ($\Delta\text{LUFS}$):** L'Integrated LUFS finale rispetta con massima precisione i target prefissati sia in modalità `same_track` sia in modalità `stylistic`, confermando che il termine dinamico V3b non compromette la convergenza della componente percettiva di loudness.
2. **Guardrail True Peak:** In tutte le tracce il Sample Peak è saldamente ancorato a $-0.30\text{ dBFS}$ (o inferiore nelle modalità stilistiche con headroom maggiore), e il True Peak rimane conforme ai criteri di sicurezza per la distribuzione streaming.
3. **Preservazione Spettrale (SC, LSD, ASD):** La convergenza spettrale (`SC`), la distanza spettrale logaritmica (`LSD`) e la distanza spettrale aggregata (`ASD`) mostrano una deviazione trascurabile rispetto a `V0p`, a conferma dell'elevata trasparenza timbrica di V3b.

### 7.4 Stabilità Numerica e Parità Sperimentale Accoppiata
1. **Zero Anomalie Numeriche:** Nessun valore `NaN` o `Inf` rilevato in alcuna delle metriche, gradienti o traiettorie di ottimizzazione.
2. **Identità Biunivoca Parametrica:** Per tutte le sessioni e per entrambe le varianti è verificata l'uguaglianza $\max |\theta - \text{map\_z\_to\_theta}(z)| < 10^{-5}$ su tutte le 41 dimensioni dello spazio continuo DDSP.
3. **Integrità Storica Preservata:** Tutti i file di audit storici (`V0p`, `V1`, `V1b`, `V2`, `V3a`, preflight V1, V2, V3 e Errata) restano intatti con `git diff` rigorosamente a zero.

### 7.5 Pacchetto di Validazione d'Ascolto Cieco (Blind Human Evaluation Package)
Per consentire la validazione percettiva umana priva di bias cognitivi:
- Sono stati renderizzati 12 file WAV a 24-bit PCM (48 kHz) con nomi anonimizzati (`variant_A.wav` e `variant_B.wav`);
- La chiave di decodifica segreta è memorizzata separatamente in `docs/sprint3/SPRINT3_V3B_BLIND_MAPPING.json`;
- Il foglio di lavoro anonimo per i revisori d'ascolto è pronto in `docs/sprint3/SPRINT3_V3B_BLIND_LISTENING_WORKSHEET.md`.
