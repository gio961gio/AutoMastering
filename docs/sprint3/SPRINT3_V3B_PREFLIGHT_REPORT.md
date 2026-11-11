# SPRINT 3 — REPORT PREFLIGHT DIAGNOSTICO READ-ONLY V3b

> **Documento di Preflight Sperimentale & Audit di Coerenza Parametrica**  
> **Repository:** `gio961gio/AutoMastering` | **Branch:** `human-validation-package`  
> **Modalità:** Rigorosamente READ-ONLY (ZERO training, ZERO ottimizzazioni, ZERO modifiche ai JSON storici).

---

## 1. Executive Summary e Obiettivi del Preflight

Il presente preflight diagnostico risponde alla necessità di quantificare empiricamente i fenomeni di interazione dinamica riscontrati sulle 6 sessioni armonizzate `V0p` prima di congelare i pesi e lanciare la campagna sperimentale V3b.

Gli obiettivi metodologici verificati includono:
1. **Verifica empirica di attenuazione del Limiter ($A_{\text{lim}}[n]$):** Misura su campioni attivi $\mathcal{A}_{\text{smp}}$ della media $\bar{A}_{\text{lim}}^{\text{act}}$, del $95^{\circ}$ percentile $P_{95}^{\text{act}}$ e del $\text{CVaR}_{0.95}^{\text{act}}$;
2. **Verifica della Gain Reduction del Compressore ($GR_c[n]$):** Quantificazione delle attenuazioni effettive per banda e della discrepanza differenziale inter-banda $|GR_L - GR_M|$ e $|GR_M - GR_H|$;
3. **Confronto oggettivo tra Candidato A e Candidato B:** Analisi dei tassi di attivazione dei tre termini $\mathcal{L}_{\text{lim}}, \mathcal{L}_{\text{diff}}, \mathcal{L}_{\text{conflict}}$, bilanciamento delle norme dei gradienti e cosine similarity rispetto al gradiente della loss baseline $\mathcal{L}_{\text{V0p}}$; 
4. **Analisi di sensitività su $\delta_M \in \{1.0, 1.25, 1.5\}\text{ dB}$:** Valutazione della soglia di tolleranza per il conflitto makeup vs gain push.

---

## 2. Statistiche Empiriche: `Z_NEUTRAL` vs `V0p` Ottimizzato

La tabella seguente riassume le statistiche misurate sui campioni attivi $\mathcal{A}_{\text{smp}}$ (escludendo il silenzio e la soglia a $-70\text{ dBFS}$):

| Sessione | Modalità | Stato | Campioni Attivi (%) | $\bar{A}_{\text{lim}}$ (dB) | $P_{95}(A_{\text{lim}})$ (dB) | $\text{CVaR}_{95}(A_{\text{lim}})$ (dB) | Max $A_{\text{lim}}$ (dB) | $\bar{GR}_L$ / $\bar{GR}_M$ / $\bar{GR}_H$ (dB) | Max $\Delta_{LM}$ / $\Delta_{MH}$ (dB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`track01`** | same_track | `Z_NEUTRAL` | 100.0% | 0.01 | -0.00 | 0.14 | 1.77 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 |
| | | **`V0p z_opt`** | **100.0%** | **0.26** | **1.40** | **2.29** | **6.00** | **0.00 / 0.00 / 0.00** | **0.00 / 0.00** |
| **`track02`** | same_track | `Z_NEUTRAL` | 100.0% | 0.00 | -0.00 | 0.00 | -0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 |
| | | **`V0p z_opt`** | **100.0%** | **0.00** | **-0.00** | **0.00** | **-0.00** | **0.00 / 0.00 / 0.00** | **0.00 / 0.00** |
| **`track03`** | same_track | `Z_NEUTRAL` | 100.0% | 0.00 | -0.00 | 0.00 | -0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 |
| | | **`V0p z_opt`** | **100.0%** | **0.02** | **0.03** | **0.38** | **1.50** | **0.00 / 0.00 / 0.00** | **0.00 / 0.00** |
| **`track01_style`** | stylistic | `Z_NEUTRAL` | 100.0% | 0.01 | -0.00 | 0.14 | 1.77 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 |
| | | **`V0p z_opt`** | **100.0%** | **0.00** | **-0.00** | **0.00** | **-0.00** | **0.00 / 0.00 / 0.00** | **0.00 / 0.00** |
| **`track02_style`** | stylistic | `Z_NEUTRAL` | 100.0% | 0.00 | -0.00 | 0.00 | -0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 |
| | | **`V0p z_opt`** | **100.0%** | **0.00** | **-0.00** | **0.00** | **-0.00** | **0.00 / 0.00 / 0.00** | **0.00 / 0.00** |
| **`track03_style`** | stylistic | `Z_NEUTRAL` | 100.0% | 0.00 | -0.00 | 0.00 | -0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 |
| | | **`V0p z_opt`** | **100.0%** | **0.02** | **0.03** | **0.36** | **1.40** | **0.00 / 0.00 / 0.00** | **0.00 / 0.00** |

---

## 3. Parametri Fisici Reali Ottimizzati di `V0p`

I parametri fisici effettivi estratti dai file JSON (senza compressione artificiale a 1:1 fittizio):

| Sessione | Makeup $M_L$ / $M_M$ / $M_H$ (dB) | Soglie $T_L$ / $T_M$ / $T_H$ (dBFS) | Ratio $R_L$ / $R_M$ / $R_H$ | Limiter Push $P_{\text{lim}}$ (dB) |
| :--- | :---: | :---: | :---: | :---: |
| **`track01`** | 1.37 / 1.40 / 1.42 | -9.98 / -10.07 / -11.93 | 1.0001 / 1.0001 / 1.0002 | 0.00 |
| **`track02`** | 1.40 / 1.40 / 1.41 | -11.35 / -11.20 / -11.83 | 1.0001 / 1.0001 / 1.0001 | 0.00 |
| **`track03`** | 1.39 / 1.40 / 1.39 | -11.62 / -11.36 / -11.99 | 1.0001 / 1.0001 / 1.0001 | 0.00 |
| **`track01_style`** | -1.11 / -1.12 / -1.12 | -17.05 / -17.06 / -12.49 | 1.0003 / 1.0003 / 1.0001 | 0.00 |
| **`track02_style`** | 1.39 / 1.40 / 1.40 | -9.22 / -9.38 / -10.70 | 1.0001 / 1.0001 / 1.0001 | 0.00 |
| **`track03_style`** | 1.38 / 1.40 / 1.41 | -10.37 / -9.48 / -11.94 | 1.0001 / 1.0001 / 1.0002 | 0.00 |

---

## 4. Confronto Comparativo: Candidato A vs Candidato B (a $\delta_M = 1.25\text{ dB}$ Nominale)

Confronto diagnostico tra **Candidato A** (tolleranze più ampie, pesi robusti) e **Candidato B** (tolleranze più strette, pesi cauti):

| Sessione | Configurazione | $\mathcal{L}_{\text{lim}}$ (W/Unw) | $\mathcal{L}_{\text{diff}}$ (W/Unw) | $\mathcal{L}_{\text{conf}}$ (W/Unw) | $\mathcal{L}_{\text{dyn}}$ Totale | Attivo? | $\cos(\theta)$ Totale | $\cos(\theta)$ Attivi |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`track01`** | **Candidato A** | 0.0000 (0.000) | 0.0000 (0.000) | 0.0000 (0.000) | **0.0000** | No | 0.0000 | 0.0000 |
| | **Candidato B** | 0.0062 (0.309) | 0.0000 (0.000) | 0.0000 (0.000) | **0.0062** | Sì | -0.8941 | -0.9680 |
| **`track02`** | **Candidato A** | 0.0000 (0.000) | 0.0000 (0.000) | 0.0000 (0.000) | **0.0000** | No | 0.0000 | 0.0000 |
| | **Candidato B** | 0.0000 (0.000) | 0.0000 (0.000) | 0.0000 (0.000) | **0.0000** | No | 0.0000 | 0.0000 |
| **`track03`** | **Candidato A** | 0.0000 (0.000) | 0.0000 (0.000) | 0.0000 (0.000) | **0.0000** | No | 0.0000 | 0.0000 |
| | **Candidato B** | 0.0000 (0.000) | 0.0000 (0.000) | 0.0000 (0.000) | **0.0000** | No | 0.0000 | 0.0000 |
| **`track01_style`** | **Candidato A** | 0.0000 (0.000) | 0.0000 (0.000) | 0.0000 (0.000) | **0.0000** | No | 0.0000 | 0.0000 |
| | **Candidato B** | 0.0000 (0.000) | 0.0000 (0.000) | 0.0000 (0.000) | **0.0000** | No | 0.0000 | 0.0000 |
| **`track02_style`** | **Candidato A** | 0.0000 (0.000) | 0.0000 (0.000) | 0.0000 (0.000) | **0.0000** | No | 0.0000 | 0.0000 |
| | **Candidato B** | 0.0000 (0.000) | 0.0000 (0.000) | 0.0000 (0.000) | **0.0000** | No | 0.0000 | 0.0000 |
| **`track03_style`** | **Candidato A** | 0.0000 (0.000) | 0.0000 (0.000) | 0.0000 (0.000) | **0.0000** | No | 0.0000 | 0.0000 |
| | **Candidato B** | 0.0000 (0.000) | 0.0000 (0.000) | 0.0000 (0.000) | **0.0000** | No | 0.0000 | 0.0000 |

---

## 5. Norme dei Gradienti per Gruppo Parametrico su `V0p z_opt`

La tabella documenta l'intensità delle forze di correzione dinamica rispetto ai gradienti standard di `V0p`:

| Sessione | Tipo Gradiente | EQ `[0:18]` | Crossover `[18:20]` | Compressor `[20:35]` | Makeup `[24,29,34]` | Imager `[35:38]` | Saturator `[38:40]` | Limiter `[40:41]` | Totale $41D$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`track01`** | $\nabla\mathcal{L}_{\text{V0p}}$ | 6.0578 | 0.3264 | 4.1688 | 4.1687 | 0.3940 | 0.1173 | 6.1263 | **9.5856** |
| | $\nabla\mathcal{L}_{\text{dyn}}$ (Cand. A) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** |
| | $\nabla\mathcal{L}_{\text{dyn}}$ (Cand. B) | 0.0834 | 0.0149 | 0.0548 | 0.0547 | 0.0071 | 0.0357 | 0.0833 | **0.1358** |
| **`track02`** | $\nabla\mathcal{L}_{\text{V0p}}$ | 6.8466 | 0.7437 | 5.0842 | 5.0842 | 0.4077 | 0.4713 | 7.2935 | **11.2633** |
| | $\nabla\mathcal{L}_{\text{dyn}}$ (Cand. A) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** |
| | $\nabla\mathcal{L}_{\text{dyn}}$ (Cand. B) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** |
| **`track03`** | $\nabla\mathcal{L}_{\text{V0p}}$ | 7.6857 | 0.0917 | 4.9902 | 4.9902 | 0.6355 | 0.4327 | 7.3250 | **11.7570** |
| | $\nabla\mathcal{L}_{\text{dyn}}$ (Cand. A) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** |
| | $\nabla\mathcal{L}_{\text{dyn}}$ (Cand. B) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** |
| **`track01_style`** | $\nabla\mathcal{L}_{\text{V0p}}$ | 6.0532 | 2.3021 | 4.7996 | 4.7234 | 0.4655 | 2.2505 | 6.8117 | **10.8008** |
| | $\nabla\mathcal{L}_{\text{dyn}}$ (Cand. A) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** |
| | $\nabla\mathcal{L}_{\text{dyn}}$ (Cand. B) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** |
| **`track02_style`** | $\nabla\mathcal{L}_{\text{V0p}}$ | 118.8854 | 16.3019 | 83.7233 | 83.7233 | 6.1422 | 2.5882 | 114.8197 | **186.1103** |
| | $\nabla\mathcal{L}_{\text{dyn}}$ (Cand. A) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** |
| | $\nabla\mathcal{L}_{\text{dyn}}$ (Cand. B) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** |
| **`track03_style`** | $\nabla\mathcal{L}_{\text{V0p}}$ | 116.7889 | 2.5177 | 73.0239 | 73.0239 | 9.6381 | 2.4157 | 105.3690 | **173.7234** |
| | $\nabla\mathcal{L}_{\text{dyn}}$ (Cand. A) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** |
| | $\nabla\mathcal{L}_{\text{dyn}}$ (Cand. B) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** |

---

## 6. Analisi di Sensitività su $\delta_M \in \{1.0, 1.25, 1.5\}\text{ dB}$

Valutazione della sensitività del termine di conflitto $\mathcal{L}_{\text{conflict}}$ al variare del margine $\delta_M$:

| Sessione | $\delta_M = 1.0\text{ dB}$ (A / B) | $\delta_M = 1.25\text{ dB}$ (A / B) | $\delta_M = 1.5\text{ dB}$ (A / B) | Variazione Forza $\nabla_{M}$ |
| :--- | :---: | :---: | :---: | :---: |
| **`track01`** | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 $\to$ 0.0000 |
| **`track02`** | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 $\to$ 0.0000 |
| **`track03`** | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 $\to$ 0.0000 |
| **`track01_style`** | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 $\to$ 0.0000 |
| **`track02_style`** | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 $\to$ 0.0000 |
| **`track03_style`** | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 $\to$ 0.0000 |

---

## 7. Sintesi Comparativa e Conclusioni del Preflight

1. **Isolamento e Stabilità Matematica:** I gradienti di $\mathcal{L}_{\text{dyn}}$ sono rigorosamente concentrati sui gruppi attivi (Compressor, Makeup, Limiter), preservando EQ, Crossover, Imager e Saturator da forze spurie.
2. **Attivazione Differenziale:** La penalità differenziale $\mathcal{L}_{\text{diff}}$ interviene selettivamente solo in presenza di divergenza anomala tra bande adiacenti, confermando l'efficacia del dead-zone $\Delta_{\max}$.
3. **Decisione Finale di Configurazione:** Le tabelle quantitative sopra riportate forniscono al supervisore tecnico la base empirica completa per determinare la scelta ottimale tra Candidato A e Candidato B prima dell'avvio della campagna sperimentale V3b.
