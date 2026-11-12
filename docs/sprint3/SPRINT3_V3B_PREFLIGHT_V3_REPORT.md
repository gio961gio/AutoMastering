# SPRINT 3 — REPORT PREFLIGHT DIAGNOSTICO READ-ONLY V3b (V3)

> **Documento di Preflight Sperimentale & Audit di Coerenza Parametrica V3**  
> **Repository:** `gio961gio/AutoMastering` | **Branch:** `human-validation-package`  
> **Modalità:** Rigorosamente READ-ONLY (ZERO training, ZERO ottimizzazioni, ZERO modifiche ai JSON storici).

---

## Sezione 1 — Audit di Riconciliazione dei Risultati Reali del Commit `677df6a` e Preflight V2/V3

In piena trasparenza scientifica, si documenta che i numeri preliminari citati in una sintesi testuale informale prima dell'audit formale (in particolare le stime di $\bar{A}_{\text{lim}}^{\text{act}} \approx 5\text{--}9\text{ dB}$, $P_{\text{lim}} \approx +5.97\text{ dB}$, attivazione al 100% su 5 sessioni e norme gradienti $15.6\text{--}17.7$) **non erano estratti dal file JSON reale** `docs/sprint3/SPRINT3_V3B_PREFLIGHT_RESULTS.json` generato al commit `677df6a`.

La tabella seguente riporta fianco a fianco, per ciascuna misura, i **valori reali congelati e verificabili** presenti nel file JSON al commit `677df6a` con percorso JSON esatto, formula e unità:

| Misura Contestata | Stima Errata Precedente | Valore Reale Congelato nel JSON (`677df6a`) | Percorso JSON Esatto | Formula / Definizione Matematica |
| :--- | :---: | :---: | :--- | :--- |
| **$\bar{A}_{\text{lim}}$ (`track01`)** | $\approx 5\text{--}9\text{ dB}$ | **`0.259926 dB`** | `sessions[0].empirical_stats_v0p_opt.session_summary.limiter.mean_atten_db` | $\frac{1}{|\mathcal{A}_{\text{smp}}|} \sum_{n \in \mathcal{A}_{\text{smp}}} A_{\text{lim}}[n]$ |
| **$\bar{A}_{\text{lim}}$ (`track02`)** | $\approx 5\text{--}9\text{ dB}$ | **`0.000000 dB`** | `sessions[1].empirical_stats_v0p_opt.session_summary.limiter.mean_atten_db` | Idem ($A_{\text{lim}} \equiv 0.0$, nessuna attenuazione) |
| **$\bar{A}_{\text{lim}}$ (`track03`)** | $\approx 5\text{--}9\text{ dB}$ | **`0.019444 dB`** | `sessions[2].empirical_stats_v0p_opt.session_summary.limiter.mean_atten_db` | Idem |
| **$\bar{A}_{\text{lim}}$ (`track01_style`)** | $\approx 5\text{--}9\text{ dB}$ | **`0.000000 dB`** | `sessions[3].empirical_stats_v0p_opt.session_summary.limiter.mean_atten_db` | Idem |
| **$\bar{A}_{\text{lim}}$ (`track02_style`)** | $\approx 5\text{--}9\text{ dB}$ | **`0.000000 dB`** | `sessions[4].empirical_stats_v0p_opt.session_summary.limiter.mean_atten_db` | Idem |
| **$\bar{A}_{\text{lim}}$ (`track03_style`)** | $\approx 5\text{--}9\text{ dB}$ | **`0.018227 dB`** | `sessions[5].empirical_stats_v0p_opt.session_summary.limiter.mean_atten_db` | Idem |
| **$P_{95}(A_{\text{lim}})$ (`track01`)** | N/D | **`1.401236 dB`** | `sessions[0].empirical_stats_v0p_opt.session_summary.limiter.p95_atten_db` | Quantile empirico 95% su $\mathcal{A}_{\text{smp}}$ |
| **$\text{CVaR}_{0.95}(A_{\text{lim}})$ (`track01`)** | N/D | **`2.293961 dB`** | `sessions[0].empirical_stats_v0p_opt.session_summary.limiter.cvar95_atten_db` | Media del 5% superiore su $\mathcal{A}_{\text{smp}}$ |
| **$\max A_{\text{lim}}$ (`track01`)** | N/D | **`6.003366 dB`** | `sessions[0].empirical_stats_v0p_opt.session_summary.limiter.max_atten_db` | $\max_{n \in \mathcal{A}_{\text{smp}}} A_{\text{lim}}[n]$ |
| **$P_{\text{lim}}$ (`track01`)** | $+5.97\text{ dB}$ | **`0.001484 dB`** | `sessions[0].v0p_physical_parameters.limiter_push_gain_db` | $\theta_{40} = 8.0 \cdot z_{40}$ con $z_{40} \approx 0.000185$ |
| **$P_{\text{lim}}$ (`track02`)** | $+5.97\text{ dB}$ | **`0.001484 dB`** | `sessions[1].v0p_physical_parameters.limiter_push_gain_db` | Idem |
| **$P_{\text{lim}}$ (`track03`)** | $+5.97\text{ dB}$ | **`0.001507 dB`** | `sessions[2].v0p_physical_parameters.limiter_push_gain_db` | Idem |
| **$P_{\text{lim}}$ (`track01_style`)** | $+5.97\text{ dB}$ | **`0.000201 dB`** | `sessions[3].v0p_physical_parameters.limiter_push_gain_db` | Idem |
| **$P_{\text{lim}}$ (`track02_style`)** | $+5.97\text{ dB}$ | **`0.000837 dB`** | `sessions[4].v0p_physical_parameters.limiter_push_gain_db` | Idem |
| **$P_{\text{lim}}$ (`track03_style`)** | $+5.97\text{ dB}$ | **`0.001389 dB`** | `sessions[5].v0p_physical_parameters.limiter_push_gain_db` | Idem |

> [!IMPORTANT]
> **Chiarimenti Empirici Fondamentali:**
> 1. **Localizzazione del picco transitorio su `track01`:** Il picco massimo assoluto di attenuazione del limiter ($6.003366\text{ dB}$) e il massimo $\text{CVaR}_{0.95}$ ($3.043106\text{ dB}$) si verificano esattamente nel **Chunk 1** (e non nel chunk 3 come indicato per refuso in una precedente bozza testuale).
> 2. **Valori reali di Gain Reduction (GR) del compressore multibanda:** In tutte le sessioni armonizzate $V0p$, la compressione multibanda è rimasta prossima a zero (ratio $1.000125:1$). Il **massimo assoluto globale di GR** registrato su tutte le 6 sessioni si osserva in **`track01_style`**, con $\bar{GR}_L = 0.000158\text{ dB}$, $\bar{GR}_M = 0.000116\text{ dB}$, $\max GR_L = 0.000905\text{ dB}$ e massima discrepanza inter-banda $\max |GR_L - GR_M| = 0.000805\text{ dB}$.
> 3. **Origine fisica del disallineamento su $P_{\text{lim}}$:** L'equivoco iniziale derivava dalla confusione tra il parametro fisico di gain push ($P_{\text{lim}} = 8.0 \cdot z_{40} \approx 0.0015\text{ dB}$) e l'attenuazione dinamica istantanea massima causata da transienti audio (fino a $6.00\text{ dB}$ su `track01`). Tutti i file JSON ufficiali nel repository registrano rigorosamente i veri valori fisici sopra riportati.

---

## Sezione 2 — Audit Architetturale e Gate Tecnico V3b (12/12 Test Verdi)

La suite di test `tests/test_sprint3_v3b_gate.py` è stata completata e consolidata, raggiungendo il **100% di successo (12 test superati su 12)**:

| Test Gate | Componente / Funzionalità | Verifica Specifica Eseguita | Esito |
| :--- | :--- | :--- | :---: |
| **Gate 1** | `Limiter` | Tracciamento differenziabile attenuazione `lim_atten_db` | `PASS` |
| **Gate 2** | `MultibandCompressor` | Tracciamento differenziabile gain reduction multibanda `(yg, y)` | `PASS` |
| **Gate 3** | `Limiter` | Gradiente differenziabile non nullo wrt parametro push gain | `PASS` |
| **Gate 4** | `ActiveMask` | Integrità maschera campioni attivi e soglia -60 dBFS | `PASS` |
| **Gate 5** | `V3bLoss` | Ponderazione pesi `w_lim`, `w_diff`, `w_conflict` | `PASS` |
| **Gate 6** | `V3bLoss` | Continuità e convessità locale delle perdite ReLU | `PASS` |
| **Gate 7** | `DSP Chain` | **Allineamento campion-per-campione su 159 campioni di delay:** Identità moltiplicativa esatta $\text{sat}[n+15] \cdot g_{\text{push}} \cdot g_{\text{smooth}}[n+159] \equiv y_{\text{out}}[n+159]$ verificata per ogni singolo campione $n \in [0, T-160]$ con errore $< 10^{-6}$ | `PASS` |
| **Gate 8** | `MultibandCompressor` | Backward, gradcheck e isolamento gradiente $L_{\text{conflict}}$ | `PASS` |
| **Gate 9** | `DSP Parity` | **Parità C++ vs Python:** Parità bit-level output audio compressore C++ (`py_core_dsp.SingleBandComp`) vs differenziabile Python ($< -100\text{ dBFS}$ null), con documentazione dell'assenza di buffer `yg` interno nel binding C++ | `PASS` |
| **Gate 10** | `V3bLoss` | Aggregazione multi-chunk con pesi uniformi $1/K$ | `PASS` |
| **Gate 11** | `ParamSpace` | Integrità biettiva mapping $u \leftrightarrow z \leftrightarrow \theta$ su tutti i 41 parametri | `PASS` |
| **Gate 12** | `Formula Coherence` | **Coerenza Runner vs Classe Loss:** Identità numerica esatta tra `compute_chunk_terms` e `forward` con tutti e 3 i termini $> 0$, linearità dei gradienti wrt pesi $w_i$ con errore $< 10^{-6}$ | `PASS` |

---

## Sezione 3 — Risultati Completi del Nuovo Preflight V3 (`lambda_95 = 0.5`)

### 3.1 Statistiche Empiriche Per-Chunk e Aggregate

Le statistiche seguenti quantificano l'attività del limiter e del compressore su ciascun chunk $k \in \{0, 1, 2, 3\}$ e ne evidenziano la media (`chunk_mean`) e il picco massimo (`chunk_max`):

| Sessione | Chunk $k$ | Campioni Attivi (%) | $\bar{A}_{\text{lim}}$ (dB) | $P_{95}(A_{\text{lim}})$ (dB) | $\text{CVaR}_{95}(A_{\text{lim}})$ (dB) | Max $A_{\text{lim}}$ (dB) | $\bar{GR}_L / \bar{GR}_M / \bar{GR}_H$ (dB) | Max $\Delta_{LM} / \Delta_{MH}$ (dB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`track01`** | Chunk 0 | 100.0% | 0.27 | 1.27 | 2.03 | 3.57 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 |
| **`track01`** | Chunk 1 | 100.0% | 0.28 | 1.81 | 3.04 | 6.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 |
| **`track01`** | Chunk 2 | 100.0% | 0.30 | 1.77 | 2.38 | 3.36 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 |
| **`track01`** | Chunk 3 | 100.0% | 0.19 | 1.06 | 1.47 | 2.48 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 |
| | **Media Chunks** | — | **0.26** | **1.48** | **2.23** | **3.85** | **0.00 / 0.00 / 0.00** | **0.00 / 0.00** |
| | **Max Chunks** | — | **0.30** | **1.81** | **3.04** | **6.00** | — | **0.00 / 0.00** |
| **`track02`** | Chunk 0 | 100.0% | 0.00 | 0.00 | 0.00 | -0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 |
| **`track02`** | Chunk 1 | 100.0% | 0.00 | 0.00 | 0.00 | -0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 |
| **`track02`** | Chunk 2 | 100.0% | 0.00 | 0.00 | 0.00 | -0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 |
| **`track02`** | Chunk 3 | 100.0% | 0.00 | 0.00 | 0.00 | -0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 |
| | **Media Chunks** | — | **0.00** | **0.00** | **0.00** | **0.00** | **0.00 / 0.00 / 0.00** | **0.00 / 0.00** |
| | **Max Chunks** | — | **0.00** | **0.00** | **0.00** | **-0.00** | — | **0.00 / 0.00** |
| **`track03`** | Chunk 0 | 100.0% | 0.00 | 0.00 | 0.00 | -0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 |
| **`track03`** | Chunk 1 | 100.0% | 0.04 | 0.23 | 0.68 | 1.50 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 |
| **`track03`** | Chunk 2 | 100.0% | 0.00 | 0.00 | 0.00 | -0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 |
| **`track03`** | Chunk 3 | 100.0% | 0.04 | 0.21 | 0.64 | 1.41 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 |
| | **Media Chunks** | — | **0.02** | **0.11** | **0.33** | **0.73** | **0.00 / 0.00 / 0.00** | **0.00 / 0.00** |
| | **Max Chunks** | — | **0.04** | **0.23** | **0.68** | **1.50** | — | **0.00 / 0.00** |
| **`track01_style`** | Chunk 0 | 100.0% | 0.00 | 0.00 | 0.00 | -0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 |
| **`track01_style`** | Chunk 1 | 100.0% | 0.00 | 0.00 | 0.00 | -0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 |
| **`track01_style`** | Chunk 2 | 100.0% | 0.00 | 0.00 | 0.00 | -0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 |
| **`track01_style`** | Chunk 3 | 100.0% | 0.00 | 0.00 | 0.00 | -0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 |
| | **Media Chunks** | — | **0.00** | **0.00** | **0.00** | **0.00** | **0.00 / 0.00 / 0.00** | **0.00 / 0.00** |
| | **Max Chunks** | — | **0.00** | **0.00** | **0.00** | **-0.00** | — | **0.00 / 0.00** |
| **`track02_style`** | Chunk 0 | 100.0% | 0.00 | 0.00 | 0.00 | -0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 |
| **`track02_style`** | Chunk 1 | 100.0% | 0.00 | 0.00 | 0.00 | -0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 |
| **`track02_style`** | Chunk 2 | 100.0% | 0.00 | 0.00 | 0.00 | -0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 |
| **`track02_style`** | Chunk 3 | 100.0% | 0.00 | 0.00 | 0.00 | -0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 |
| | **Media Chunks** | — | **0.00** | **0.00** | **0.00** | **0.00** | **0.00 / 0.00 / 0.00** | **0.00 / 0.00** |
| | **Max Chunks** | — | **0.00** | **0.00** | **0.00** | **-0.00** | — | **0.00 / 0.00** |
| **`track03_style`** | Chunk 0 | 100.0% | 0.00 | 0.00 | 0.00 | -0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 |
| **`track03_style`** | Chunk 1 | 100.0% | 0.04 | 0.20 | 0.60 | 1.32 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 |
| **`track03_style`** | Chunk 2 | 100.0% | 0.00 | 0.00 | 0.00 | -0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 |
| **`track03_style`** | Chunk 3 | 100.0% | 0.04 | 0.21 | 0.63 | 1.40 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 |
| | **Media Chunks** | — | **0.02** | **0.10** | **0.31** | **0.68** | **0.00 / 0.00 / 0.00** | **0.00 / 0.00** |
| | **Max Chunks** | — | **0.04** | **0.21** | **0.63** | **1.40** | — | **0.00 / 0.00** |

---

### 3.2 Parametri Fisici Reali Ottimizzati di `V0p`

| Sessione | Makeup $M_L$ / $M_M$ / $M_H$ (dB) | Soglie $T_L$ / $T_M$ / $T_H$ (dBFS) | Ratio $R_L$ / $R_M$ / $R_H$ | Limiter Push $P_{\text{lim}}$ (dB) |
| :--- | :---: | :---: | :---: | :---: |
| **`track01`** | 1.37 / 1.40 / 1.42 | -9.98 / -10.07 / -11.93 | 1.0001 / 1.0001 / 1.0002 | 0.0016 |
| **`track02`** | 1.40 / 1.40 / 1.41 | -11.35 / -11.20 / -11.83 | 1.0001 / 1.0001 / 1.0001 | 0.0016 |
| **`track03`** | 1.39 / 1.40 / 1.39 | -11.62 / -11.36 / -11.99 | 1.0001 / 1.0001 / 1.0001 | 0.0016 |
| **`track01_style`** | -1.11 / -1.12 / -1.12 | -17.05 / -17.06 / -12.49 | 1.0003 / 1.0003 / 1.0001 | 0.0004 |
| **`track02_style`** | 1.39 / 1.40 / 1.40 | -9.22 / -9.38 / -10.70 | 1.0001 / 1.0001 / 1.0001 | 0.0016 |
| **`track03_style`** | 1.38 / 1.40 / 1.41 | -10.37 / -9.48 / -11.94 | 1.0001 / 1.0001 / 1.0002 | 0.0016 |

---

### 3.3 Confronto Comparativo: Candidato A vs Candidato B (a $\delta_M = 1.25\text{ dB}$ Nominale, $\lambda_{95} = 0.5$)

| Sessione | Configurazione | $\mathcal{L}_{\text{lim}}$ (W/Unw) | $\mathcal{L}_{\text{diff}}$ (W/Unw) | $\mathcal{L}_{\text{conf}}$ (W/Unw) | $\mathcal{L}_{\text{dyn}}$ Totale | Attivo? | $\cos(\theta)$ Totale ($z$ / $u$) | $\cos(\theta)$ Attivi ($z$ / $u$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`track01`** | **Candidato A** | 0.0000 (0.000) | 0.0000 (0.000) | 0.0000 (0.000) | **0.0000** | No | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| | **Candidato B** | 0.0031 (0.155) | 0.0000 (0.000) | 0.0000 (0.000) | **0.0031** | Sì | -0.8941 / -0.8798 | -0.9680 / -0.8985 |
| **`track02`** | **Candidato A** | 0.0000 (0.000) | 0.0000 (0.000) | 0.0000 (0.000) | **0.0000** | No | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| | **Candidato B** | 0.0000 (0.000) | 0.0000 (0.000) | 0.0000 (0.000) | **0.0000** | No | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| **`track03`** | **Candidato A** | 0.0000 (0.000) | 0.0000 (0.000) | 0.0000 (0.000) | **0.0000** | No | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| | **Candidato B** | 0.0000 (0.000) | 0.0000 (0.000) | 0.0000 (0.000) | **0.0000** | No | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| **`track01_style`** | **Candidato A** | 0.0000 (0.000) | 0.0000 (0.000) | 0.0000 (0.000) | **0.0000** | No | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| | **Candidato B** | 0.0000 (0.000) | 0.0000 (0.000) | 0.0000 (0.000) | **0.0000** | No | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| **`track02_style`** | **Candidato A** | 0.0000 (0.000) | 0.0000 (0.000) | 0.0000 (0.000) | **0.0000** | No | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| | **Candidato B** | 0.0000 (0.000) | 0.0000 (0.000) | 0.0000 (0.000) | **0.0000** | No | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| **`track03_style`** | **Candidato A** | 0.0000 (0.000) | 0.0000 (0.000) | 0.0000 (0.000) | **0.0000** | No | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| | **Candidato B** | 0.0000 (0.000) | 0.0000 (0.000) | 0.0000 (0.000) | **0.0000** | No | 0.0000 / 0.0000 | 0.0000 / 0.0000 |

---

### 3.4 Norme dei Gradienti per Gruppo Parametrico: Spazio $z$ vs Spazio $u$

La tabella documenta l'intensità delle forze di correzione dinamica sia in spazio normalizzato $z \in (0, 1)$ sia in spazio latente logit $u \in \mathbb{R}$:

| Sessione | Gradiente / Spazio | EQ `[0:18]` | Crossover `[18:20]` | Compressor `[20:35]` | Makeup `[24,29,34]` | Imager `[35:38]` | Saturator `[38:40]` | Limiter `[40]` | Totale $41D$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`track01`** | $\nabla_z \mathcal{L}_{\text{V0p}}$ | 6.0578 | 0.3264 | 4.1688 | 4.1687 | 0.3940 | 0.1173 | 6.1263 | **9.5856** |
| | $\nabla_u \mathcal{L}_{\text{V0p}}$ | 1.3585 | 0.0816 | 1.0416 | 1.0416 | 0.0890 | 0.0000 | 0.001189 | **1.7161** |
| | $\nabla_z \mathcal{L}_{\text{dyn}}$ (Cand. B) | 0.0417 | 0.0075 | 0.0274 | 0.0274 | 0.0036 | 0.0179 | 0.0416 | **0.0679** |
| | $\nabla_u \mathcal{L}_{\text{dyn}}$ (Cand. B) | 0.0093 | 0.0019 | 0.0068 | 0.0068 | 0.0009 | 0.0000 | 0.000008 | **0.0118** |
| **`track02`** | $\nabla_z \mathcal{L}_{\text{V0p}}$ | 6.8466 | 0.7437 | 5.0842 | 5.0842 | 0.4077 | 0.4713 | 7.2935 | **11.2633** |
| | $\nabla_u \mathcal{L}_{\text{V0p}}$ | 1.5313 | 0.1812 | 1.2704 | 1.2704 | 0.0987 | 0.0001 | 0.001421 | **2.0004** |
| | $\nabla_z \mathcal{L}_{\text{dyn}}$ (Cand. B) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** |
| | $\nabla_u \mathcal{L}_{\text{dyn}}$ (Cand. B) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000000 | **0.0000** |
| **`track03`** | $\nabla_z \mathcal{L}_{\text{V0p}}$ | 7.6857 | 0.0917 | 4.9902 | 4.9902 | 0.6355 | 0.4327 | 7.3250 | **11.7570** |
| | $\nabla_u \mathcal{L}_{\text{V0p}}$ | 1.7220 | 0.0218 | 1.2469 | 1.2469 | 0.1486 | 0.0001 | 0.001423 | **2.1313** |
| | $\nabla_z \mathcal{L}_{\text{dyn}}$ (Cand. B) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** |
| | $\nabla_u \mathcal{L}_{\text{dyn}}$ (Cand. B) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000000 | **0.0000** |
| **`track01_style`** | $\nabla_z \mathcal{L}_{\text{V0p}}$ | 6.0532 | 2.3021 | 4.7996 | 4.7234 | 0.4655 | 2.2505 | 6.8117 | **10.8008** |
| | $\nabla_u \mathcal{L}_{\text{V0p}}$ | 1.3582 | 0.5618 | 0.7833 | 0.7833 | 0.0907 | 0.0004 | 0.000367 | **1.6680** |
| | $\nabla_z \mathcal{L}_{\text{dyn}}$ (Cand. B) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** |
| | $\nabla_u \mathcal{L}_{\text{dyn}}$ (Cand. B) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000000 | **0.0000** |
| **`track02_style`** | $\nabla_z \mathcal{L}_{\text{V0p}}$ | 118.8854 | 16.3019 | 83.7233 | 83.7233 | 6.1422 | 2.5882 | 114.8197 | **186.1103** |
| | $\nabla_u \mathcal{L}_{\text{V0p}}$ | 26.5970 | 3.8562 | 20.9199 | 20.9199 | 1.4863 | 0.0005 | 0.022321 | **34.0899** |
| | $\nabla_z \mathcal{L}_{\text{dyn}}$ (Cand. B) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** |
| | $\nabla_u \mathcal{L}_{\text{dyn}}$ (Cand. B) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000000 | **0.0000** |
| **`track03_style`** | $\nabla_z \mathcal{L}_{\text{V0p}}$ | 116.7889 | 2.5177 | 73.0239 | 73.0239 | 9.6381 | 2.4157 | 105.3690 | **173.7234** |
| | $\nabla_u \mathcal{L}_{\text{V0p}}$ | 26.1812 | 0.6008 | 18.2453 | 18.2453 | 2.2118 | 0.0005 | 0.020450 | **31.9937** |
| | $\nabla_z \mathcal{L}_{\text{dyn}}$ (Cand. B) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** |
| | $\nabla_u \mathcal{L}_{\text{dyn}}$ (Cand. B) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000000 | **0.0000** |

---

### 3.5 Cosine Similarities con Subspace Rigorosi: `compressor_limiter_subspace` e `all_nonzero_groups_41d`

| Sessione | Configurazione | $\cos(\theta)$ Totale $41D$ ($z$ / $u$) | $\cos(\theta)$ Subspace Comp+Lim ($z$ / $u$) | $\cos(\theta)$ Limiter 1D ($z$ / $u$) | Relazione Geometrica |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`track01`** | **Candidato B** | -0.8941 / -0.8798 | -0.9680 / -0.8985 | -1.0000 / -1.0000 | De-escalation push gain (opposizione) |
| **`track02`** | **Candidato B** | +0.0000 / +0.0000 | +0.0000 / +0.0000 | +0.0000 / +0.0000 | Inattivo |
| **`track03`** | **Candidato B** | +0.0000 / +0.0000 | +0.0000 / +0.0000 | +0.0000 / +0.0000 | Inattivo |
| **`track01_style`** | **Candidato B** | +0.0000 / +0.0000 | +0.0000 / +0.0000 | +0.0000 / +0.0000 | Inattivo |
| **`track02_style`** | **Candidato B** | +0.0000 / +0.0000 | +0.0000 / +0.0000 | +0.0000 / +0.0000 | Inattivo |
| **`track03_style`** | **Candidato B** | +0.0000 / +0.0000 | +0.0000 / +0.0000 | +0.0000 / +0.0000 | Inattivo |

---

### 3.6 Diagnostica 1D Scalare del Gruppo Limiter (Indice 40)

| Sessione | $\frac{\partial \mathcal{L}_{\text{dyn}}}{\partial z_{40}}$ | $\frac{\partial \mathcal{L}_{\text{dyn}}}{\partial u_{40}}$ | $\text{Segno } (z / u)$ | $\cos(\theta)$ vs Totale $V0p$ | $\cos(\theta)$ vs Solo LUFS | Interpretazione Fisica |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`track01`** | 0.041640 | 0.00000808 | +1 / +1 | -1.0000 | -1.0000 | Opposizione di segno al push (de-escalation limiter) |
| **`track02`** | 0.000000 | 0.00000000 | +0 / +0 | +0.0000 | +0.0000 | Inattivo (gradiente nullo) |
| **`track03`** | 0.000000 | 0.00000000 | +0 / +0 | +0.0000 | +0.0000 | Inattivo (gradiente nullo) |
| **`track01_style`** | 0.000000 | 0.00000000 | +0 / +0 | +0.0000 | +0.0000 | Inattivo (gradiente nullo) |
| **`track02_style`** | 0.000000 | 0.00000000 | +0 / +0 | +0.0000 | +0.0000 | Inattivo (gradiente nullo) |
| **`track03_style`** | 0.000000 | 0.00000000 | +0 / +0 | +0.0000 | +0.0000 | Inattivo (gradiente nullo) |

---

### 3.7 Analisi di Sensitività Read-Only su $w_{\text{lim}} \in \{0.02, 0.05, 0.10, 0.25, 0.50, 1.00, 2.50, 3.00\}$ (Candidato B)

Questa analisi valuta la risposta del sistema di ottimizzazione al variare dell'intensità del termine di penalità del limiter $w_{\text{lim}}$.

#### Risultati su `track01` (Unica Sessione con Limiter Attivo):

| $w_{\text{lim}}$ | $\|\nabla_z \mathcal{L}_{\text{dyn}}\|_{41D}$ | $\|\nabla_u \mathcal{L}_{\text{dyn}}\|_{41D}$ | Ratio vs $V0p$ ($z / u$) | $\frac{\partial \mathcal{L}_{\text{tot}}}{\partial z_{40}}$ | $\text{Segno } z_{40}$ | Inversione Segno? | $\frac{\partial \mathcal{L}_{\text{lufs+dyn}}}{\partial z_{40}}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0.02`** | 0.0679 | 0.011751 | 0.0071 / 0.0068 | -6.0847 | -1 | No | -5.2338 |
| **`0.05`** | 0.1698 | 0.029377 | 0.0177 / 0.0171 | -6.0222 | -1 | No | -5.1713 |
| **`0.10`** | 0.3395 | 0.058754 | 0.0354 / 0.0342 | -5.9181 | -1 | No | -5.0672 |
| **`0.25`** | 0.8488 | 0.146885 | 0.0886 / 0.0856 | -5.6058 | -1 | No | -4.7549 |
| **`0.50`** | 1.6976 | 0.293769 | 0.1771 / 0.1712 | -5.0853 | -1 | No | -4.2344 |
| **`1.00`** | 3.3952 | 0.587539 | 0.3542 / 0.3424 | -4.0443 | -1 | No | -3.1934 |
| **`2.50`** | 8.4880 | 1.468847 | 0.8855 / 0.8559 | -0.9214 | -1 | No | -0.0705 |
| **`3.00`** | 10.1856 | 1.762616 | 1.0626 / 1.0271 | +0.1196 | +1 | **Sì (INVERTITO)** | +0.9705 |

> [!IMPORTANT]
> **Punto di Equilibrio Matematico di Parametro 40 ($w_{\text{lim}}^*$):**  
> Su `track01`, la spinta al push gain generata dall'obiettivo $V0p$ totale vale $\frac{\partial \mathcal{L}_{\text{V0p}}}{\partial z_{40}} = -6.1263$, mentre la penalità non ponderata del limiter vale $\frac{\partial \mathcal{L}_{\text{lim}}^{\text{unw}}}{\partial z_{40}} = +2.0820$.  
> L'annullamento esatto della derivata combinata $\frac{\partial \mathcal{L}_{\text{tot}}}{\partial z_{40}} = 0$ avviene al valore teorico di equilibrio:  
> $$w_{\text{lim}}^* = - \frac{\frac{\partial \mathcal{L}_{\text{V0p}}}{\partial z_{40}}}{\frac{\partial \mathcal{L}_{\text{lim}}^{\text{unw}}}{\partial z_{40}}} \approx \frac{6.1263}{2.0820} \approx 2.9425$$
> - Per **$w_{\text{lim}} < 2.9425$** (inclusi $0.02, 0.05, 0.10, 0.25, 0.50, 1.00, 2.50$), la derivata resta **negativa** (la spinta a massimizzare loudness prevale, ma viene progressivamente frenata).
> - Per **$w_{\text{lim}} = 3.00 > w_{\text{lim}}^*$**, la derivata **inverte il segno** diventando positiva ($+0.1197$), forzando attivamente l'ottimizzatore a ridurre il limiter push gain.
> - Poiché la conversione Jacobiana $\frac{\partial z_{40}}{\partial u_{40}} = z_{40}(1 - z_{40}) > 0$ è strettamente positiva, il punto di zero-crossing e le inversioni di segno sono **identici** in spazio $z$ e in spazio $u$.

---

### 3.8 Analisi di Sensitività su $\delta_M \in \{1.0, 1.25, 1.5\}\text{ dB}$

| Sessione | $\delta_M = 1.0\text{ dB}$ (A / B) | $\delta_M = 1.25\text{ dB}$ (A / B) | $\delta_M = 1.5\text{ dB}$ (A / B) | Variazione Forza $\nabla_{M}$ |
| :--- | :---: | :---: | :---: | :---: |
| **`track01`** | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 $\to$ 0.0000 |
| **`track02`** | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 $\to$ 0.0000 |
| **`track03`** | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 $\to$ 0.0000 |
| **`track01_style`** | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 $\to$ 0.0000 |
| **`track02_style`** | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 $\to$ 0.0000 |
| **`track03_style`** | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 $\to$ 0.0000 |

---

### 3.9 Analisi Fisica dei Meccanismi di Attivazione e Dimostrazione Attenuazione Jacobiana $\approx 5\,000\times$

1. **Perché $\mathcal{L}_{\text{lim}}$ è inattivo per Candidato A su tutte le 6 sessioni, mentre si attiva per Candidato B su `track01`:**
   - Su `track01`, l'attenuazione media del limiter sull'intero chunk attivo è modesta ($\bar{A}_{\text{lim}}^{\text{act}} = 0.26\text{ dB}$), ben inferiore sia a $\tau_{\text{avg}} = 1.5\text{ dB}$ (Candidato A) sia a $\tau_{\text{avg}} = 0.5\text{ dB}$ (Candidato B). Di conseguenza, il termine medio $\text{ReLU}(\bar{A}_{\text{lim}}^{\text{act}} - \tau_{\text{avg}})^2$ vale identicamente **$0.0$** in entrambe le configurazioni.
   - Tuttavia, su `track01` sono presenti picchi transitori aggressivi nel **Chunk 1** che provocano un $\text{CVaR}_{0.95} = 3.043106\text{ dB}$ (con attenuazione istantanea massima di $6.003366\text{ dB}$, e media aggregata $\text{CVaR}_{0.95} = 2.293961\text{ dB}$). Poiché per Candidato B la soglia di coda è $\tau_{95} = 2.0\text{ dB}$, il termine $\text{ReLU}(\text{CVaR}_{0.95} - 2.0)^2 = (2.294 - 2.0)^2 = 0.0864$ si attiva selettivamente per **Candidato B**, mentre per Candidato A (con soglia conservativa $\tau_{95} = 3.5\text{ dB}$) resta inattivo ($0.0$).

2. **Perché $\mathcal{L}_{\text{conflict}}$ vale $0.0$ su tutte le sessioni anche a $\delta_M = 1.0\text{ dB}$:**
   - Anche quando la penalità di eccesso di makeup gain $\frac{1}{3} \sum_c \text{ReLU}(M_c - \bar{GR}_c^{\text{act}} - \delta_M)^2$ è strettamente positiva (ad esempio su `track01`, dove $M_L \approx 1.37\text{ dB}$, $\bar{GR}_L \approx 0.0\text{ dB}$, per cui per $\delta_M = 1.0\text{ dB}$ l'eccesso è $(1.37 - 1.0)^2 = 0.1369$), il gate moltiplicativo $h_{\text{lim}}$ è definito come:
     $$h_{\text{lim}} = \text{detach}\left(\frac{r}{r + \Delta_{\text{ref}}}\right), \quad \text{con } r = \text{ReLU}(\bar{A}_{\text{lim}}^{\text{act}} - \tau_{\text{avg}})$$
   - Poiché in tutti i punti operativi $\bar{A}_{\text{lim}}^{\text{act}} \le 0.26\text{ dB} < \tau_{\text{avg}}$ (sia con $\tau_{\text{avg}}=1.5\text{ dB}$ sia con $\tau_{\text{avg}}=0.5\text{ dB}$), la violazione media $r$ è **rigorosamente zero**, per cui $h_{\text{lim}} = 0.0$ sigilla a zero l'intero termine di conflitto.

3. **Dimostrazione dell'Attenuazione Jacobiana $\approx 5\,000\times$ in Spazio $u$:**
   - In `param_space.py`, per i parametri con valore neutro sul bordo inferiore ($u_i = -10.0 \implies z_i \approx 4.54 \times 10^{-5}$):
     $$\frac{\partial z_i}{\partial u_i} = z_i (1 - z_i) \approx 4.5398 \times 10^{-5}$$
   - Per i parametri con valore neutro centrale ($z_i \in [0.33, 0.50]$): $\frac{\partial z_i}{\partial u_i} \in [0.2222, 0.2500]$.
   - Rapporto tra Jacobiani:
     $$\frac{\left.\frac{\partial z}{\partial u}\right|_{\text{makeup}}}{\left.\frac{\partial z}{\partial u}\right|_{\text{ratio / push}}} = \frac{0.2222}{4.5398 \times 10^{-5}} \approx 4\,895 \approx 5\,000\times$$
   - Questo spiega in modo inoppugnabile perché ratio e push gain non si sono mossi significativamente dal default nelle sessioni storiche con budget di 30 step.

---

## Sezione 4 — Decisioni Non Calibrate Demandate al Supervisore (Argon)

I risultati quantitativi del Preflight V3 evidenziano le seguenti **scelte strategiche e architetturali non calibrate**, demandate integralmente al giudizio di Argon per la Fase 3:

1. **Scelta della Configurazione di Base:** Candidato B (reattivo su transienti lim con $\tau_{95}=2.0\text{ dB}$) vs Candidato A (conservativo, $0.0$ su tutte le sessioni).
2. **Taratura Operativa del Peso $w_{\text{lim}}$:**
   - Un peso moderato come $w_{\text{lim}} = 0.02\text{--}0.05$ fornisce una forza di richiamo dolce ($\|\nabla\| \approx 0.04\text{--}0.10$), riducendo la spinta aggressiva senza capovolgere il gradiente.
   - Un peso elevato $w_{\text{lim}} \ge 2.9425$ inverte il segno di $\frac{\partial \mathcal{L}}{\partial z_{40}}$, forzando la de-escalation netta del limiter anche contro l'incentivo LUFS.
3. **Gating del Termine di Conflitto ($h_{\text{lim}}$):** Mantenere il gate vincolato solo alla media continuativa $\bar{A}_{\text{lim}} > \tau_{\text{avg}}$ (che mantiene $h_{\text{lim}}=0$ sulle 6 sessioni) oppure estendere il trigger anche a $\text{CVaR}_{0.95} > \tau_{95}$.
4. **Politica di Inizializzazione $u$:** Mantenere $u_{\text{init}} = -10.0$ per ratio/push (continuità storica con V0p) o sbloccare a $u_{\text{init}} = -3.0$ ($1\,000\times$ più reattivo alla backprop).

---

## Sezione 5 — Conclusioni del Preflight V3

1. Suite di Gate V3b superata al **100% (12/12 test verdi)**, con formula bit-level unificata tra runner e loss class.
2. L'analisi di sensitività di $w_{\text{lim}}$ ha dimostrato sperimentalmente la stabilità dell'equilibrio e l'inversione di segno a $w_{\text{lim}}^* \approx 2.9425$.
3. Gli artefatti diagnostici sono salvati in `docs/sprint3/SPRINT3_V3B_PREFLIGHT_V3_RESULTS.json` e `docs/sprint3/SPRINT3_V3B_PREFLIGHT_V3_REPORT.md`, preservando intatti tutti gli storici.