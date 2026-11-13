# SPRINT 3 — V3C PILOT ABLATION REPORT (`V3c vs V0p`, `same_track` only)

**Campagna:** Ablation Pilota Accoppiata `Control V0p` vs `Treatment V3c` (`Candidate A` + pesi pilota congelati)  
**Data di Generazione:** 2026-11-13 11:27:34  
**Durata Complessiva:** 2804.85 s  
**Branch:** `human-validation-package` | **Baseline Commit Fase 1:** `a98c2d67b87d798d449a1018157d50d2a2e7be20`  
**Scope Autorizzato:** esclusivamente le 3 sessioni `same_track` (`track01`, `track02`, `track03`); `track01_style..03_style` escluse dal pilot.

> **Nota Metodologica Fondamentale:** Tutte le tabelle seguenti riportano **separatamente** le componenti dell'obiettivo `V0p` (comuni a entrambi i bracci) e i termini `V3c` (calcolati anche in sola lettura su `Control V0p` come `v3c_eval_on_v0p`), senza mai confondere `V0p + V3c` con il solo totale `V0p`. Tutti i risultati qui presentati sono **metriche quantitative oggettive/proxy**; nessun giudizio percettivo d'ascolto umano è stato ancora svolto.

---

## 1. Riepilogo Integrazione Optimizer (Fase 1) & Protocollo Sperimentale (Fase 2)

- **Integrazione Opt-In (`a98c2d67b87d798d449a1018157d50d2a2e7be20`):** `AutoMasteringOptimizer` espone `enable_v3c: bool = False` (con `self.v3c_prior_loss = None` di default, preservando bit-identità V0p/V3b), blocca esplicitamente con `ValueError` qualsiasi modalità diversa da `same_track`, invoca `v3c_prior_loss(mix_ext=mix_k, pred_ext=pred_ext_k, z=z, n_pre=n_pre_k, delay_samples=159, compute_diagnostics=False)` prima della media pesata `1/K` dei chunk (conservando `1x` su `loss_width_param`), ed espone `evaluate_loss_iir` / `_eval_loss_iir`. Validato dai Gate 1–9 (`223 passed`).
- **Soglie Frozen `Candidate A`:** `tau_drop_db=0.50`, `tau_corr=(0.03, 0.05, 0.08)`, `tau_smr_db=(2.0, 2.0, 2.0)`, `tau_sub_db=0.50`, `alpha_sub=1.0`, `tau_w_low_max=1.00`, `tau_w_mid=0.15`, `tau_w_high=0.15`.
- **Pesi Frozen `same_track`:** `w_mono_drop=1.00`, `w_band_corr=10.00`, `w_band_smr=0.05`, `w_width_param=0.25`.
- **Parità Sperimentale:** stesso `prepare_harmonized_session`, 4 chunk da `144,000` campioni con `120,000` campioni di pre-roll, delay DSP `159`, LR4 fisso `160/3200 Hz`, init `Z_NEUTRAL`, `30` step Adam (`lr=0.04`), schedule `fft (1–20) -> iir (21–30)`.

---

## 2. Tabella 1 — Obiettivi Separati (`V0p` IIR vs `V3c` Weighted & Raw + `v3c_eval_on_v0p`) e Larghezze Imager

| Sessione | Braccio | Max Diff vs V0p Base | `loss_v0p_total` (IIR) | `mrstft` (u) | `lufs` (u) | `plr_box` (u) | `loss_v3c_w` | `mono_drop` (u/w) | `band_corr` (u/w) | `band_smr` (u/w) | `width_param` (u/w) | Imager `(L, M, H)` |
|:---|:---|:---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|
| **`track01`** | Control V0p | 0.00e+00 | 2.822382 | 1.0088 | 2.2668 | 0.0000 | **0.002483** | 0.00000/0.00000 | 0.000015/0.000146 | 0.03784/0.00189 | 0.00178/0.00045 | (1.0422, 1.1117, 1.1098) |
| **`track01`** | Treatment V3c | — | 2.819971 | 1.0085 | 2.2642 | 0.0000 | **0.002306** | 0.00000/0.00000 | 0.000014/0.000138 | 0.03473/0.00174 | 0.00172/0.00043 | (1.0415, 1.1114, 1.1098) |
| **`track02`** | Control V0p | 0.00e+00 | 10.355217 | 1.9794 | 10.2735 | 0.9806 | **0.000807** | 0.00000/0.00000 | 0.000064/0.000641 | 0.00066/0.00003 | 0.00053/0.00013 | (1.0230, 1.1119, 1.1112) |
| **`track02`** | Treatment V3c | — | 10.355348 | 1.9794 | 10.2736 | 0.9808 | **0.000727** | 0.00000/0.00000 | 0.000063/0.000634 | 0.00012/0.00001 | 0.00035/0.00009 | (1.0186, 1.1117, 1.1112) |
| **`track03`** | Control V0p | 0.00e+00 | 6.234956 | 1.5033 | 5.9129 | 0.0074 | **0.000805** | 0.00000/0.00000 | 0.000027/0.000269 | 0.00167/0.00008 | 0.00181/0.00045 | (1.0425, 1.1117, 1.1105) |
| **`track03`** | Treatment V3c | — | 6.234995 | 1.5033 | 5.9130 | 0.0073 | **0.000786** | 0.00000/0.00000 | 0.000026/0.000265 | 0.00149/0.00007 | 0.00179/0.00045 | (1.0423, 1.1116, 1.1105) |

---

## 3. Tabella 2 — Compatibilità Stereo/Mono Chunk-Level IIR (Broadband & Bande: Mix vs Master)

| Sessione | Braccio | BB $\rho_{LR}$ (Mix $\to$ Master, $\Delta$) | Low $\rho_{LR}$ / $\Delta\text{SMR}$ dB | Mid $\rho_{LR}$ / $\Delta\text{SMR}$ dB | High $\rho_{LR}$ / $\Delta\text{SMR}$ dB | BB $D_{\text{mono}}$ (Mix $\to$ Master) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **`track01`** | Control V0p | 0.7855 -> 0.7444 (-0.0411) | 0.8804 (-0.0161) / +0.66 | 0.6032 (-0.0452) / +0.64 | 0.6824 (-0.0498) / +0.85 | 0.502 -> 0.603 |
| **`track01`** | Treatment V3c | 0.7855 -> 0.7447 (-0.0407) | 0.8807 (-0.0158) / +0.65 | 0.6035 (-0.0449) / +0.64 | 0.6824 (-0.0498) / +0.85 | 0.502 -> 0.602 |
| **`track02`** | Control V0p | 0.9013 -> 0.8632 (-0.0381) | 0.9985 (-0.0001) / +0.39 | 0.7704 (-0.0422) / +1.06 | 0.8467 (-0.0257) / +0.88 | 0.222 -> 0.312 |
| **`track02`** | Treatment V3c | 0.9013 -> 0.8633 (-0.0380) | 0.9986 (-0.0001) / +0.36 | 0.7705 (-0.0421) / +1.06 | 0.8468 (-0.0257) / +0.88 | 0.222 -> 0.312 |
| **`track03`** | Control V0p | 0.7292 -> 0.6763 (-0.0529) | 0.8426 (-0.0168) / +0.53 | 0.5701 (-0.0569) / +0.79 | 0.5453 (-0.0681) / +0.94 | 0.633 -> 0.769 |
| **`track03`** | Treatment V3c | 0.7292 -> 0.6764 (-0.0528) | 0.8427 (-0.0167) / +0.53 | 0.5702 (-0.0568) / +0.79 | 0.5454 (-0.0681) / +0.94 | 0.633 -> 0.768 |

### 3B. Diagnostiche Short-Time (`50 ms`, `100 ms`, `250 ms`) e Anti-Comb-Filtering (Chunk-Level & Full-Track C++)

| Sessione | Braccio | BB Corr Mean (50/100/250 ms) | BB Corr P05 (50/100/250 ms) | BB Corr Min / CVaR05 (100 ms) | Comb Min Notch (Mix $\to$ Pred, $\Delta$ dB) | Full-Track Comb $\Delta$ Notch / $\Delta$ Std dB |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **`track01`** | Control V0p | 0.7415/0.7450/0.7436 | 0.5724/0.6161/0.6364 | 0.3817 / 0.5657 | -0.51 -> -1.13 (-0.57) | -0.38 / +0.06 |
| **`track01`** | Treatment V3c | 0.7418/0.7453/0.7440 | 0.5727/0.6165/0.6368 | 0.3821 / 0.5662 | -0.51 -> -1.13 (-0.57) | -0.37 / +0.06 |
| **`track02`** | Control V0p | 0.8541/0.8543/0.8484 | 0.5323/0.5514/0.5727 | -0.0266 / 0.4454 | 0.30 -> -0.15 (-0.33) | -0.14 / +0.03 |
| **`track02`** | Treatment V3c | 0.8542/0.8544/0.8485 | 0.5324/0.5515/0.5728 | -0.0264 / 0.4455 | 0.30 -> -0.15 (-0.33) | -0.14 / +0.03 |
| **`track03`** | Control V0p | 0.6635/0.6683/0.6707 | 0.4527/0.4875/0.5551 | 0.3246 / 0.4424 | -0.73 -> -1.28 (-0.47) | -0.47 / +0.08 |
| **`track03`** | Treatment V3c | 0.6635/0.6684/0.6708 | 0.4528/0.4877/0.5553 | 0.3250 / 0.4426 | -0.73 -> -1.28 (-0.47) | -0.47 / +0.08 |

---

## 4. Tabella 3 — Metriche Audio Full-Track Post-Render C++ & Guardrail True Peak (`<= -0.30 dBTP`)

| Sessione | Braccio | Target LUFS | Final LUFS | $\Delta$LUFS | PLR (dB) | Sample Peak (dBFS) | **True Peak (dBTP)** | Guardrail $\le -0.30$ | SC | LSD (dB) | ERB Surplus (dB) | $\Delta$ M/S Ratio (dB) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`track01`** | Control V0p | -7.29 | -10.51 | +3.22 | 10.21 | -0.3000 | **+0.2647** | EXCEED (+0.56) | 0.4057 | 6.788 | +1.671 | +0.4380 |
| **`track01`** | Treatment V3c | -7.29 | -10.51 | +3.22 | 10.21 | -0.3000 | **+0.3074** | EXCEED (+0.61) | 0.4057 | 6.787 | +1.673 | +0.4318 |
| **`track02`** | Control V0p | -7.22 | -18.04 | +10.82 | 17.74 | -0.3000 | **-0.0713** | EXCEED (+0.23) | 0.7932 | 11.213 | +1.048 | +0.3491 |
| **`track02`** | Treatment V3c | -7.22 | -18.04 | +10.82 | 17.74 | -0.3000 | **-0.0713** | EXCEED (+0.23) | 0.7932 | 11.213 | +1.048 | +0.3468 |
| **`track03`** | Control V0p | -6.90 | -13.50 | +6.61 | 13.20 | -0.3000 | **-0.0868** | EXCEED (+0.21) | 0.6121 | 9.470 | +0.818 | +1.4059 |
| **`track03`** | Treatment V3c | -6.90 | -13.50 | +6.61 | 13.20 | -0.3000 | **-0.0838** | EXCEED (+0.22) | 0.6120 | 9.470 | +0.818 | +1.4047 |

---

## 5. Tabella 4 — Spostamenti Parametrici $\Delta\theta = \theta_{\text{V3c}} - \theta_{\text{V0p}}$, Norme dei 6 Gruppi e Gradienti Finali

| Sessione | $\Delta w_{\text{low}}$ | $\Delta w_{\text{mid}}$ | $\Delta w_{\text{high}}$ | $\|\Delta z\|_2$ (41D) | EQ $\|\Delta z\|_2$ | Xover $\|\Delta z\|_2$ | Comp $\|\Delta z\|_2$ | Imager $\|\Delta z\|_2$ | Sat $\|\Delta z\|_2$ | Lim $\|\Delta z\|_2$ | Final $\|\nabla_u V_{3c}\|_2$ / $\|\nabla_u \text{Tot}\|_2$ |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| **`track01`** | -0.0007 | -0.0003 | +0.0000 | 0.01458 | 0.00021 | 0.01440 | 0.00203 | 0.00111 | 0.00000 | 0.00000 | 0.01655 / 1.70395 |
| **`track02`** | -0.0044 | -0.0002 | -0.0000 | 0.00628 | 0.00019 | 0.00004 | 0.00006 | 0.00627 | 0.00000 | 0.00000 | 0.00538 / 2.00018 |
| **`track03`** | -0.0003 | -0.0001 | -0.0000 | 0.00208 | 0.00015 | 0.00203 | 0.00010 | 0.00039 | 0.00000 | 0.00000 | 0.00676 / 2.12933 |

---

## 6. Analisi delle Regressioni Proxy, Limiti Osservati e Pacchetto Blind

- **`track01`:** `loss_v3c_total_weighted` varia da `0.002483` (`V0p`) a `0.002306` (`V3c`, delta `-0.000178`), mentre le metriche spettrali comuni registrano $\Delta\text{MRSTFT} = -0.000304$ e $\Delta\text{LSD} = -0.0011\text{ dB}$ (nessun valore NaN/Inf in 30 step).
- **`track02`:** `loss_v3c_total_weighted` varia da `0.000807` (`V0p`) a `0.000727` (`V3c`, delta `-0.000079`), mentre le metriche spettrali comuni registrano $\Delta\text{MRSTFT} = +0.000020$ e $\Delta\text{LSD} = +0.0001\text{ dB}$ (nessun valore NaN/Inf in 30 step).
- **`track03`:** `loss_v3c_total_weighted` varia da `0.000805` (`V0p`) a `0.000786` (`V3c`, delta `-0.000019`), mentre le metriche spettrali comuni registrano $\Delta\text{MRSTFT} = +0.000021$ e $\Delta\text{LSD} = +0.0001\text{ dB}$ (nessun valore NaN/Inf in 30 step).

- **Distinzione Proxy vs Ascolto Umano:** Gli scostamenti su $\rho_{LR}$, SMR, $D_{\text{mono}}$, MRSTFT e LSD rappresentano esclusivamente proxy differenziabili/oggettivi; la validazione percettiva finale dipende dall'ascolto cieco umano sui 6 file WAV in `docs/sprint3/blind_v3c_pilot_wavs/`.
- **Guardrail True Peak:** Come atteso (poiché `V3c` opera sulla compatibilità stereo/mono e non sostituisce un ceiling True Peak 4x), l'eventuale superamento di `-0.30 dBTP` già presente in `Control V0p` viene monitorato esplicitamente come guardrail indipendente nella Tabella 3.
- **Pacchetto Blind Audio:** `6` WAV (`48 kHz`, stereo, `PCM_24`), massimo scarto di coppia $|\Delta\text{LUFS}| = 0.0017 LU`, manifest pubblico in `docs/sprint3/SPRINT3_V3C_BLIND_AUDIO_MANIFEST.md`, archivio condivisibile privo di mapping in `docs/sprint3/blind_v3c_pilot_evaluator_pack.zip`.
