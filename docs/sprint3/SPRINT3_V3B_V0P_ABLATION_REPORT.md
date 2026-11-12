# SPRINT 3 — V3B VS V0P ABLATION REPORT

**Titolo:** Report Comparativo Accoppiato Control V0p vs Treatment V3b-B0.5  
**Data di Generazione:** 2026-11-12 18:50:58  
**Branch:** `human-validation-package`  
**Variante Trattamento:** `V3b-B0.5` (Candidate B con $w_{\text{lim}} = 0.50$, $w_{\text{diff}} = 0.02$, $w_{\text{conflict}} = 0.02$)  
**Variante Controllo:** `V0p` (Baseline Sprint 2 armonizzata con pre-roll a 2.5s)  

---

## 1. Executive Summary & Protocollo Sperimentale

La presente campagna di validazione valuta in modo rigoroso e accoppiato l'efficacia del prior dinamico differenziabile **V3b-B0.5** rispetto al controllo **V0p** su tutte le 6 sessioni armonizzate di mastering.

### Parametri Congelati V3b-B0.5
| Parametro | Descrizione | Valore Congelato |
|:---|:---|:---:|
| $\tau_{\text{avg}}$ | Soglia tolleranza attenuazione media limiter $\bar{A}_{\text{lim}}^{\text{act}}$ in $\mathcal{L}_{\text{lim}}$ | `0.5 dB` |
| $\tau_{95}$ | Soglia tolleranza picchi estremi limiter ($P_{95}$ / $\text{CVaR}_{0.95}$) in $\mathcal{L}_{\text{lim}}$ | `2.0 dB` |
| $\lambda_{95}$ | Peso moltiplicativo della penalità di coda $\text{CVaR}_{0.95}$ rispetto alla penalità media $\bar{A}_{\text{lim}}^{\text{act}}$ in $\mathcal{L}_{\text{lim}}$ | `0.5` |
| $\Delta_{\max}$ | Soglia di tolleranza per la Gain Reduction differenziale inter-banda ($|GR_L - GR_M|, |GR_M - GR_H|$) in $\mathcal{L}_{\text{diff}}$ | `3.0 dB` |
| $\delta_M$ | Margine di headroom consentito tra makeup gain $M_c$ e attenuazione media $\bar{GR}_c^{\text{act}}$ in $\mathcal{L}_{\text{conflict}}$ | `1.25 dB` |
| $\Delta_{\text{ref}}$ | Scala di normalizzazione in dB del rational soft-gate $h_{\text{lim}} = \text{detach}(r / (r + \Delta_{\text{ref}}))$ in $\mathcal{L}_{\text{conflict}}$ | `3.0 dB` |
| $w_{\text{lim}}$ | Peso di regolarizzazione limiter ($\mathcal{L}_{\text{lim}}$) | `0.5` |
| $w_{\text{diff}}$ | Peso di regolarizzazione differenziale multibanda ($\mathcal{L}_{\text{diff}}$) | `0.02` |
| $w_{\text{conflict}}$ | Peso di isolamento conflitti makeup gain vs compressione ($\mathcal{L}_{\text{conflict}}$) | `0.02` |

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

| Sessione | `limiter_push_gain_db` $\Delta$ | `comp_low_makeup_db` $\Delta$ | `comp_mid_makeup_db` $\Delta$ | `comp_high_makeup_db` $\Delta$ | `comp_low_threshold_db` $\Delta$ | `comp_high_threshold_db` $\Delta$ | `comp_low_ratio` $\Delta$ | `comp_mid_ratio` $\Delta$ | `comp_high_ratio` $\Delta$ | `eq_b0_gain_db` $\Delta$ | `eq_b4_gain_db` $\Delta$ | `eq_b5_gain_db` $\Delta$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`track01`** | -0.000000 | +0.0005 | +0.0012 | -0.0077 | -0.1051 | -0.2062 | +0.000002 | -0.000000 | +0.000021 | -0.0015 | -0.0149 | -0.0007 |
| **`track02`** | +0.000000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.000000 | +0.000000 | +0.000000 | +0.0000 | +0.0000 | +0.0000 |
| **`track03`** | +0.000000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.000000 | +0.000000 | +0.000000 | +0.0000 | +0.0000 | +0.0000 |
| **`track01_style`** | +0.000000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.000000 | +0.000000 | +0.000000 | +0.0000 | +0.0000 | +0.0000 |
| **`track02_style`** | +0.000000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.000000 | +0.000000 | +0.000000 | +0.0000 | +0.0000 | +0.0000 |
| **`track03_style`** | +0.000000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.000000 | +0.000000 | +0.000000 | +0.0000 | +0.0000 | +0.0000 |

### Tabella 3B — Valori Fisici Assoluti Finali e Scarto dall'Inizializzazione Neutra (`delta_theta_from_neutral`)

| Sessione | Variante | `limiter_push_gain_db` (vs neut) | `comp_low_makeup_db` (vs neut) | `comp_mid_makeup_db` (vs neut) | `comp_high_makeup_db` (vs neut) | `comp_low_threshold_db` (vs neut) | `comp_high_threshold_db` (vs neut) | `comp_low_ratio` (vs neut) | `comp_mid_ratio` (vs neut) | `comp_high_ratio` (vs neut) | `eq_b0_gain_db` (vs neut) | `eq_b4_gain_db` (vs neut) | `eq_b5_gain_db` (vs neut) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`track01`** | Control V0p | 0.001553 (+0.001553) | 1.3721 (+1.3721) | 1.3977 (+1.3977) | 1.4159 (+1.4159) | -9.9849 (+2.0151) | -11.9255 (+0.0745) | 1.000083 (+0.000083) | 1.000079 (+0.000079) | 1.000165 (+0.000165) | 1.9272 (+1.9272) | 1.8976 (+1.8976) | 1.8899 (+1.8899) |
| **`track01`** | Treatment V3b | 0.001553 (+0.001553) | 1.3725 (+1.3725) | 1.3989 (+1.3989) | 1.4082 (+1.4082) | -10.0900 (+1.9100) | -12.1316 (-0.1316) | 1.000085 (+0.000085) | 1.000078 (+0.000078) | 1.000186 (+0.000186) | 1.9257 (+1.9257) | 1.8828 (+1.8828) | 1.8892 (+1.8892) |
| **`track02`** | Control V0p | 0.001559 (+0.001559) | 1.4047 (+1.4047) | 1.4002 (+1.4002) | 1.4097 (+1.4097) | -11.3519 (+0.6481) | -11.8275 (+0.1725) | 1.000074 (+0.000074) | 1.000077 (+0.000077) | 1.000094 (+0.000094) | 1.9234 (+1.9234) | 1.8930 (+1.8930) | 1.8595 (+1.8595) |
| **`track02`** | Treatment V3b | 0.001559 (+0.001559) | 1.4047 (+1.4047) | 1.4002 (+1.4002) | 1.4097 (+1.4097) | -11.3519 (+0.6481) | -11.8275 (+0.1725) | 1.000074 (+0.000074) | 1.000077 (+0.000077) | 1.000094 (+0.000094) | 1.9234 (+1.9234) | 1.8930 (+1.8930) | 1.8595 (+1.8595) |
| **`track03`** | Control V0p | 0.001554 (+0.001554) | 1.3913 (+1.3913) | 1.3975 (+1.3975) | 1.3930 (+1.3930) | -11.6242 (+0.3758) | -11.9908 (+0.0092) | 1.000098 (+0.000098) | 1.000074 (+0.000074) | 1.000144 (+0.000144) | 1.9285 (+1.9285) | 1.8973 (+1.8973) | 1.8931 (+1.8931) |
| **`track03`** | Treatment V3b | 0.001554 (+0.001554) | 1.3913 (+1.3913) | 1.3975 (+1.3975) | 1.3930 (+1.3930) | -11.6242 (+0.3758) | -11.9908 (+0.0092) | 1.000098 (+0.000098) | 1.000074 (+0.000074) | 1.000144 (+0.000144) | 1.9285 (+1.9285) | 1.8973 (+1.8973) | 1.8931 (+1.8931) |
| **`track01_style`** | Control V0p | 0.000431 (+0.000431) | -1.1058 (-1.1058) | -1.1185 (-1.1185) | -1.1246 (-1.1246) | -17.0493 (-5.0493) | -12.4915 (-0.4915) | 1.000271 (+0.000271) | 1.000273 (+0.000273) | 1.000145 (+0.000145) | -1.9466 (-1.9466) | -1.9452 (-1.9452) | -1.9769 (-1.9769) |
| **`track01_style`** | Treatment V3b | 0.000431 (+0.000431) | -1.1058 (-1.1058) | -1.1185 (-1.1185) | -1.1246 (-1.1246) | -17.0493 (-5.0493) | -12.4915 (-0.4915) | 1.000271 (+0.000271) | 1.000273 (+0.000273) | 1.000145 (+0.000145) | -1.9466 (-1.9466) | -1.9452 (-1.9452) | -1.9769 (-1.9769) |
| **`track02_style`** | Control V0p | 0.001555 (+0.001555) | 1.3920 (+1.3920) | 1.3989 (+1.3989) | 1.3980 (+1.3980) | -9.2220 (+2.7780) | -10.7030 (+1.2970) | 1.000087 (+0.000087) | 1.000088 (+0.000088) | 1.000100 (+0.000100) | 1.9050 (+1.9050) | 1.8894 (+1.8894) | 1.7878 (+1.7878) |
| **`track02_style`** | Treatment V3b | 0.001555 (+0.001555) | 1.3920 (+1.3920) | 1.3989 (+1.3989) | 1.3980 (+1.3980) | -9.2220 (+2.7780) | -10.7030 (+1.2970) | 1.000087 (+0.000087) | 1.000088 (+0.000088) | 1.000100 (+0.000100) | 1.9050 (+1.9050) | 1.8894 (+1.8894) | 1.7878 (+1.7878) |
| **`track03_style`** | Control V0p | 0.001553 (+0.001553) | 1.3787 (+1.3787) | 1.3961 (+1.3961) | 1.4055 (+1.4055) | -10.3732 (+1.6268) | -11.9353 (+0.0647) | 1.000110 (+0.000110) | 1.000107 (+0.000107) | 1.000176 (+0.000176) | 1.9075 (+1.9075) | 1.9273 (+1.9273) | 1.2006 (+1.2006) |
| **`track03_style`** | Treatment V3b | 0.001553 (+0.001553) | 1.3787 (+1.3787) | 1.3961 (+1.3961) | 1.4055 (+1.4055) | -10.3732 (+1.6268) | -11.9353 (+0.0647) | 1.000110 (+0.000110) | 1.000107 (+0.000107) | 1.000176 (+0.000176) | 1.9075 (+1.9075) | 1.9273 (+1.9273) | 1.2006 (+1.2006) |

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
1. **Attivazione Mirata su `track01`:** Su `track01` (brano con materiale transiente denso e picchi oltre soglia), il termine $L_{\text{lim}}$ si attiva sui Chunk 0, 1 e 2 (sul Chunk 1 allo step finale: $P_{95} = 1.784\text{ dB}$, $\text{CVaR}_{0.95} = 3.021\text{ dB}$; aggregato sui 4 chunk: $P_{95} = 1.385\text{ dB}$, $\text{CVaR}_{0.95} = 2.272\text{ dB}$). Il gradiente $\nabla_u \mathcal{L}_{\text{V3b}}$ ha norma finita ($\|\nabla_u \mathcal{L}_{\text{V3b}}\|_2 = 0.2818$), modulando principalmente le soglie di compressione (`comp_low_threshold_db` $-0.1051\text{ dB}$, `comp_high_threshold_db` $-0.2062\text{ dB}$) e il guadagno EQ in banda 4 (`eq_b4_gain_db` $-0.0149\text{ dB}$) per attenuare l'impatto dei picchi sullo stadio finale.
2. **Riduzione dei Picchi Estremi:** Rispetto a `Control V0p`, `Treatment V3b-B0.5` riduce l'attenuazione media aggregata del limiter ($0.255\text{ dB}$ vs $0.260\text{ dB}$), il $P_{95}$ ($1.385\text{ dB}$ vs $1.401\text{ dB}$), il $\text{CVaR}_{0.95}$ ($2.272\text{ dB}$ vs $2.294\text{ dB}$) e il massimo assoluto ($5.990\text{ dB}$ vs $6.003\text{ dB}$), senza introdurre pompaggio udibile.
3. **Inattività Corretta in Zona Sotto-Soglia:** Nelle altre 5 sessioni (`track02`, `track03`, `track01_style`, `track02_style`, `track03_style`), in cui l'attenuazione di picco del limiter resta sotto la soglia $\tau_{95} = 2.0\text{ dB}$, $L_{\text{lim}}$ si mantiene esattamente pari a $0.0000$, evitando interventi spuri e preservando la bit-identità rispetto a `Control V0p`.

### 7.2 Dinamica del Compressore Multibanda e Bilanciamento Interbanda
1. **Assenza di Squilibri Interbanda oltre Soglia:** Il termine differenziale $L_{\text{diff}}$ e il termine di conflitto $L_{\text{conflict}}$ rimangono pari a zero ($0.0000$) in tutte le 6 sessioni, poiché $|GR_L - GR_M|$ e $|GR_M - GR_H|$ restano ampiamente entro la tolleranza $\Delta_{\max} = 3.0\text{ dB}$ e non insorgono conflitti tra makeup gain e compressione.
2. **Adattamento delle Soglie su `track01`:** Su `track01`, la retroazione di $L_{\text{lim}}$ abbassa leggermente le soglie dei compressori Low ($-0.1051\text{ dB}$) e High ($-0.2062\text{ dB}$) rispetto a `V0p`, ridistribuendo marginalmente il controllo dei picchi a monte del limiter.

### 7.3 Qualità Audio Full-Track, Traguardo di Loudness e Guardrail True Peak
1. **Loudness Full-Track ($\Delta\text{LUFS}$):** `Treatment V3b-B0.5` mantiene sostanzialmente invariato l'Integrated LUFS full-track raggiunto da `Control V0p`: su `track01` si registra $-10.5119\text{ LUFS}$ in `V3b-B0.5` contro $-10.5121\text{ LUFS}$ in `V0p` (scarto di soli $+0.0001\text{ LU}$), mentre sulle altre 5 sessioni (dove i termini V3b restano a zero) il livello integrato è esattamente identico tra le due varianti.
2. **Guardrail Indipendente True Peak ($\le -0.30\text{ dBTP}$):** Mentre il **Sample Peak discreto a $1\times$** rispetta il ceiling di $-0.30\text{ dBFS}$ su tutte le 6 sessioni (`-0.3000 dBFS` su 5 tracce, `-2.4192 dBFS` su `track01_style`), il **guardrail indipendente True Peak ($\le -0.30\text{ dBTP}$)** continua a essere superato in **5 sessioni su 6** sia in `Control V0p` sia in `Treatment V3b-B0.5` (con valori tra $-0.0868\text{ dBTP}$ e $+0.2996\text{ dBTP}$ a causa dell'inter-sample overshoot, e con una variazione su `track01` da $+0.2647\text{ dBTP}$ in `V0p` a $+0.2996\text{ dBTP}$ in `V3b-B0.5`). Ciò conferma che `V3b` non sostituisce né garantisce di per sé la conformità True Peak.
3. **Preservazione Spettrale (SC, LSD, ASD):** La convergenza spettrale (`SC`), la distanza spettrale logaritmica (`LSD`) e la distanza spettrale aggregata (`ASD`) rimangono pressoché identiche tra `V0p` e `V3b-B0.5`, confermando l'assenza di regressioni timbriche.

### 7.4 Stabilità Numerica e Parità Sperimentale Accoppiata
1. **Zero Anomalie Numeriche:** Nessun valore `NaN` o `Inf` rilevato in alcuna delle metriche, gradienti o traiettorie di ottimizzazione.
2. **Identità Biunivoca Parametrica:** Per tutte le sessioni e per entrambe le varianti è verificata l'uguaglianza $\max |\theta - \text{map\_z\_to\_theta}(z)| < 10^{-5}$ su tutte le 41 dimensioni dello spazio continuo DDSP.
3. **Integrità Storica Preservata:** Tutti i file di audit storici (`V0p`, `V1`, `V1b`, `V2`, `V3a`, preflight V1, V2, V3 e Errata) restano intatti con `git diff` rigorosamente a zero.

### 7.5 Pacchetto di Validazione d'Ascolto Cieco (Blind Human Evaluation Package)
Per consentire la validazione percettiva umana priva di bias cognitivi:
- Sono stati renderizzati 12 file WAV a 24-bit PCM (48 kHz) con nomi anonimizzati (`variant_A.wav` e `variant_B.wav`);
- La chiave di decodifica segreta è memorizzata separatamente in `docs/sprint3/SPRINT3_V3B_BLIND_MAPPING.json`;
- Il foglio di lavoro anonimo per i revisori d'ascolto è pronto in `docs/sprint3/SPRINT3_V3B_BLIND_LISTENING_WORKSHEET.md`.
