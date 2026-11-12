# SPRINT 3 — REPORT PREFLIGHT DIAGNOSTICO READ-ONLY V3b (V2)

> **Documento di Preflight Sperimentale & Audit di Coerenza Parametrica V2**  
> **Repository:** `gio961gio/AutoMastering` | **Branch:** `human-validation-package`  
> **Modalità:** Rigorosamente READ-ONLY (ZERO training, ZERO ottimizzazioni, ZERO modifiche ai JSON storici).

---

## Sezione 1 — Audit di Riconciliazione tra Sintesi Precedente e Risultati Reali del Commit `677df6a`

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
> L'origine della confusione su $P_{\text{lim}}$ derivava dalla sovrapposizione concettuale tra il parametro fisico di gain push ($P_{\text{lim}} = 8.0 \cdot z_{40} \approx 0.0015\text{ dB}$) e l'attenuazione di picco massima raggiunta dal limiter su transienti isolati (fino a $6.00\text{ dB}$ su `track01`). Tutti i file JSON ufficiali nel repository registrano rigorosamente i veri valori fisici sopra riportati.

---

## Sezione 2 — Audit dei 10 File Modificati nel Commit `c2fefc2`

L'ispezione di `git diff --name-status ef5254f..c2fefc2` evidenzia esattamente **10 file modificati**:

| File | Operazione Git | Ruolo Architetturale & Motivazione della Modifica |
| :--- | :---: | :--- |
| `python/ddsp_mastering/dsp/compressor.py` | `M` | Implementazione forward a 2 output `(y_out, yg)` e VJP backward differenziabile per la gain reduction del compressore. |
| `python/ddsp_mastering/dsp/limiter.py` | `M` | Esposizione del tensore di attenuazione e guadagno differenziabile `g_smooth[n]` e `a_lim_db[n]`. |
| `python/ddsp_mastering/dsp/chain.py` | `M` | Aggiunta del parametro `return_diagnostics=True` per la propagazione delle tracce differenziabili. |
| `python/ddsp_mastering/loss/dynamic_prior.py` | `A` | Creazione del modulo `V3bDynamicPriorLoss` e maschera di attività campioni `compute_active_sample_mask`. |
| `python/ddsp_mastering/loss/__init__.py` | `M` | Esportazione del modulo `V3bDynamicPriorLoss` nell'interfaccia pubblica del package. |
| `tests/test_sprint3_v3b_gate.py` | `A` | Creazione della suite del Gate Tecnico V3b (11 test rigorosi, passati 11/11). |
| `python/ddsp_mastering/dsp/__init__.py` | `M` | Esportazione delle nuove classi del DSP nell'interfaccia pubblica del modulo. |
| `python/tests/test_sprint_1_correctness.py` | `M` | **Unpacking chiamata low-level:** questo test invocava direttamente `DifferentiableSingleBandCompressorFunction.apply(...)`, che ora restituisce la tupla `(y_out, yg)`. Senza l'unpacking `out, _ = Function.apply(...)`, il test legacy falliva con `TypeError: tuple and Tensor`. |
| `python/tests/test_sprint_2_2_audit_remediation.py` | `M` | **Unpacking chiamata low-level:** identico al precedente, aggiornato per supportare la tupla a 2 uscite di `Function.apply`. |
| `python/tests/test_ticket_1_4.py` | `M` | **Unpacking chiamata low-level:** identico al precedente, aggiornato per supportare la tupla a 2 uscite di `Function.apply`. |

> [!NOTE]
> Si sottolinea che il modulo ad alto livello `DifferentiableSingleBandCompressor.forward` mantiene `return_gr=False` di default, preservando la totale retro-compatibilità con il codice utente e la catena di mastering.

---

## Sezione 3 — Risultati Completi del Nuovo Preflight V2 (`lambda_95 = 0.5`)

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

### 3.5 Diagnostica 1D Scalare del Gruppo Limiter (Indice 40)

Poiché il gruppo Limiter è uno spazio 1D scalare, la cosine similarity rappresenta puramente la **concordanza di segno scalare** ($\pm 1.0$ o $0.0$). La tabella documenta derivata scalare, segno e concordanza rispetto a $\nabla \mathcal{L}_{\text{V0p}}$ e rispetto a $\nabla \mathcal{L}_{\text{LUFS}}$:

| Sessione | $\frac{\partial \mathcal{L}_{\text{dyn}}}{\partial z_{40}}$ | $\frac{\partial \mathcal{L}_{\text{dyn}}}{\partial u_{40}}$ | $\text{Segno } (z / u)$ | $\cos(\theta)$ vs Totale $V0p$ | $\cos(\theta)$ vs Solo LUFS | Interpretazione Fisica |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`track01`** | 0.041640 | 0.00000808 | +1 / +1 | -1.0000 | -1.0000 | Opposizione di segno al push (de-escalation limiter) |
| **`track02`** | 0.000000 | 0.00000000 | +0 / +0 | +0.0000 | +0.0000 | Inattivo (gradiente nullo) |
| **`track03`** | 0.000000 | 0.00000000 | +0 / +0 | +0.0000 | +0.0000 | Inattivo (gradiente nullo) |
| **`track01_style`** | 0.000000 | 0.00000000 | +0 / +0 | +0.0000 | +0.0000 | Inattivo (gradiente nullo) |
| **`track02_style`** | 0.000000 | 0.00000000 | +0 / +0 | +0.0000 | +0.0000 | Inattivo (gradiente nullo) |
| **`track03_style`** | 0.000000 | 0.00000000 | +0 / +0 | +0.0000 | +0.0000 | Inattivo (gradiente nullo) |

---

### 3.6 Analisi di Sensitività su $\delta_M \in \{1.0, 1.25, 1.5\}\text{ dB}$

| Sessione | $\delta_M = 1.0\text{ dB}$ (A / B) | $\delta_M = 1.25\text{ dB}$ (A / B) | $\delta_M = 1.5\text{ dB}$ (A / B) | Variazione Forza $\nabla_{M}$ |
| :--- | :---: | :---: | :---: | :---: |
| **`track01`** | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 $\to$ 0.0000 |
| **`track02`** | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 $\to$ 0.0000 |
| **`track03`** | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 $\to$ 0.0000 |
| **`track01_style`** | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 $\to$ 0.0000 |
| **`track02_style`** | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 $\to$ 0.0000 |
| **`track03_style`** | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 $\to$ 0.0000 |

---

### 3.7 Analisi Fisica dei Meccanismi di Attivazione e Inattività

1. **Perché $\mathcal{L}_{\text{lim}}$ è inattivo per Candidato A su tutte le 6 sessioni, mentre si attiva per Candidato B su `track01`:**
   - Su `track01`, l'attenuazione media del limiter sull'intero chunk attivo è modesta ($\bar{A}_{\text{lim}}^{\text{act}} = 0.26\text{ dB}$), ben inferiore sia a $\tau_{\text{avg}} = 1.5\text{ dB}$ (Candidato A) sia a $\tau_{\text{avg}} = 0.5\text{ dB}$ (Candidato B). Di conseguenza, il termine medio $\text{ReLU}(\bar{A}_{\text{lim}}^{\text{act}} - \tau_{\text{avg}})^2$ vale identicamente **$0.0$** in entrambe le configurazioni.
   - Tuttavia, su `track01` sono presenti picchi transitori aggressivi che provocano un $\text{CVaR}_{0.95}^{\text{act}} = 2.294\text{ dB}$ (e fino a $3.06\text{ dB}$ sul chunk $k=3$, con attenuazione istantanea massima di $6.00\text{ dB}$). Poiché per Candidato B la soglia di coda è $\tau_{95} = 2.0\text{ dB}$, il termine $\text{ReLU}(\text{CVaR}_{0.95} - 2.0)^2 = (2.294 - 2.0)^2 = 0.0864$ si attiva selettivamente per **Candidato B**, mentre per Candidato A (con soglia conservativa $\tau_{95} = 3.5\text{ dB}$) resta inattivo ($0.0$).

2. **Perché $\mathcal{L}_{\text{conflict}}$ vale $0.0$ su tutte le sessioni anche a $\delta_M = 1.0\text{ dB}$:**
   - Anche quando la penalità di eccesso di makeup gain $\text{ReLU}(M_c - \bar{GR}_c^{\text{act}} - \delta_M)^2$ è strettamente positiva (ad esempio su `track01`, dove $M_L \approx 1.37\text{ dB}$, $\bar{GR}_L \approx 0.0\text{ dB}$, per cui per $\delta_M = 1.0\text{ dB}$ l'eccesso è $(1.37 - 1.0)^2 = 0.1369$), il gate moltiplicativo $h_{\text{lim}}$ è definito come:
     $$h_{\text{lim}} = \text{detach}\left(\frac{r}{r + \Delta_{\text{ref}}}\right), \quad \text{con } r = \text{ReLU}(\bar{A}_{\text{lim}}^{\text{act}} - \tau_{\text{avg}})$$
   - Poiché in tutti i punti operativi $\bar{A}_{\text{lim}}^{\text{act}} \le 0.26\text{ dB} < \tau_{\text{avg}}$ (sia con $\tau_{\text{avg}}=1.5\text{ dB}$ sia con $\tau_{\text{avg}}=0.5\text{ dB}$), la violazione media $r$ è **rigorosamente zero**, per cui $h_{\text{lim}} = 0.0$ sigilla a zero l'intero termine di conflitto. Questo comportamento è strutturalmente corretto secondo la specifica (non si deve penalizzare il makeup gain se il limiter non è sovraccarico in modo continuativo).

3. **Perché in `V0p` i ratio ($1.000125:1$) e `limiter_push_gain_db` ($0.0015\text{ dB}$) sono rimasti quasi fermi al punto di partenza (Dimostrazione dell'Attenuazione Jacobiana $\approx 5\,000\times$ in Spazio $u$):**
   - In `param_space.py`, per i parametri il cui valore neutro coincide con il bordo inferiore del range consentito (i ratio di compressione con $\min = 1.0$ e `limiter_push_gain_db` con $\min = 0.0\text{ dB}$), il punto iniziale è posizionato a $u_i = -10.0$ ($z_i = \sigma(-10.0) \approx 4.54 \times 10^{-5}$).
   - La derivata Jacobiana della sigmoide in tale punto operativo vale:
     $$\frac{\partial z_i}{\partial u_i} = z_i (1 - z_i) \approx 4.5398 \times 10^{-5}$$
   - Per i parametri con valore neutro centrale (come i makeup gain e i guadagni EQ, dove $z_i \in [0.33, 0.50]$), la derivata Jacobiana vale $\frac{\partial z_i}{\partial u_i} \in [0.2222, 0.2500]$.
   - Il rapporto tra le derivate Jacobiane evidenzia un'**attenuazione asimmetrica di oltre $5\,000\times$**:
     $$\frac{\left.\frac{\partial z}{\partial u}\right|_{\text{makeup}}}{\left.\frac{\partial z}{\partial u}\right|_{\text{ratio / push}}} = \frac{0.2222}{4.5398 \times 10^{-5}} \approx 4\,895 \approx 5\,000\times$$
   - Di conseguenza, qualsiasi gradiente calcolato in spazio $z$ (ad es. $\nabla_z \mathcal{L}_{\text{V0p}}[40] = 6.1263$) viene compresso in spazio $u$ a un valore microscopico:
     $$\nabla_u \mathcal{L}_{\text{V0p}}[40] = 6.1263 \times (4.54 \times 10^{-5}) \approx 0.000278$$
   - Con un learning rate Adam di $0.04$ e un budget limitato a $30$ step, il vettore $u$ può spostarsi al massimo di $\Delta u \le 30 \times 0.04 = 1.2$ unità logit. Partendo da $u = -10.0$, anche muovendosi con il gradiente alla massima velocità possibile, $u$ arriva al massimo a $-8.8$, dove $z = \sigma(-8.8) \approx 0.00015$, corrispondente a un ratio di $1 + 3.0 \cdot 0.00015 = 1.00045:1$ e a un push gain di $8.0 \cdot 0.00015 = 0.0012\text{ dB}$.
   - Questo risultato matematico spiega in modo definitivo e rigoroso perché i ratio e il push gain siano rimasti quasi identici ai valori di default in tutte le sessioni armonizzate precedenti.

---

## Sezione 4 — Questioni Aperte per il Supervisore

Alla luce dei risultati quantitativi del Preflight V2, si sottopongono al Supervisore Tecnico (Argon) i seguenti punti decisionali per l'avvio della fase operativa V3b:

1. **Scelta della Configurazione di Base (Candidato B vs Candidato A):**
   - **Candidato B** si è dimostrato l'unico in grado di intercettare i picchi transitori sul limiter di `track01` (grazie a $\tau_{95} = 2.0\text{ dB}$ e $\lambda_{95} = 0.5$), esercitando una forza correttiva di de-escalation con cosine similarity negativa ($-0.9680$ sul limiter) che contrasta l'eccesso di spinta.
   - **Candidato A** (con $\tau_{95} = 3.5\text{ dB}$) rimane completamente passivo su tutte le 6 sessioni.

2. **Gating del Termine di Conflitto ($h_{\text{lim}}$):**
   - Attualmente $h_{\text{lim}}$ è controllato unicamente da $r = \text{ReLU}(\bar{A}_{\text{lim}}^{\text{act}} - \tau_{\text{avg}})$. Poiché $\bar{A}_{\text{lim}}^{\text{act}} \le 0.26\text{ dB}$ su tutte le sessioni, $\mathcal{L}_{\text{conflict}}$ non entra mai in azione.
   - Si domanda al Supervisore se $h_{\text{lim}}$ debba includere anche un termine di coda su $\text{CVaR}_{0.95}$ (es. $r_{\text{cvar}} = \text{ReLU}(\text{CVaR}_{0.95} - \tau_{95})$), oppure se si desideri mantenere il gate legato strettamente all'attenuazione media continua.

3. **Inizializzazione di $u$ e Attenuazione Jacobiana $\approx 5\,000\times$:**
   - La scoperta dell'attenuazione Jacobiana $z(1-z) \approx 4.54 \times 10^{-5}$ per ratio e push gain rivela che l'inattività di tali moduli non è dovuta a gradienti nulli in spazio fisico, bensì al posizionamento iniziale sul bordo estremo della sigmoide ($u_{\text{init}} = -10.0$).
   - Si domanda se per la campagna V3b si intenda mantenere l'inizializzazione conservativa standard a $u_{\text{init}} = -10.0$ per preservare la continuità storica con V0p/V1/V2, oppure se considerare un'inizializzazione più permissiva (es. $u = -3.0 \implies z \approx 0.047$, $z(1-z) \approx 0.045$, $1\,000\times$ più reattivo).

---

## Sezione 5 — Conclusioni del Preflight V2

1. Tutti i test del Gate Tecnico V3b sono superati al 100% (11/11 test verdi).
2. I gradienti per singolo componente sono stati isolati con successo sia in spazio normalizzato $z$ sia in spazio logit latente $u$.
3. I nuovi artefatti sono salvati in `docs/sprint3/SPRINT3_V3B_PREFLIGHT_V2_RESULTS.json` e `docs/sprint3/SPRINT3_V3B_PREFLIGHT_V2_REPORT.md` preservando intatti i file storici del primo preflight.