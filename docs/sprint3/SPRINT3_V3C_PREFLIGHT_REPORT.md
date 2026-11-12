# SPRINT 3 — V3C READ-ONLY PREFLIGHT DIAGNOSTICS REPORT

**Data di Generazione:** 2026-11-12 20:48:37  
**Branch:** `human-validation-package`  
**Modalità:** Read-Only Preflight (`LR4` fisso `160.0 / 3200.0 Hz` su pre-roll completo, `D_tot = 159` campioni, $T_{\text{eval}} = 143\,841$, pesi chunk `1/4`, ZERO training, ZERO loss V3b attiva)  
**Punti Valutati per Sessione:** `Z_NEUTRAL`, `V0p_z_opt`, `V3b_z_opt`  

### Configurazioni Candidate Valutate (Non Pesate)
| Config | $\tau_{\text{drop}}$ (dB) | $\tau_{\text{corr}}$ `(L, M, H)` | $\tau_{\text{smr}}$ `(L, M, H)` (dB) | $\tau_{\text{sub}}$ (dB) | $\tau_{w,\text{low,max}}$ | $\tau_{w,\text{mid/high}}$ |
|:---|---:|:---:|:---:|---:|---:|---:|
| `Zero_Threshold` | `0.0` | `(0.0, 0.0, 0.0)` | `(0.0, 0.0, 0.0)` | `0.0` | `1.0` | `0.0` |
| `Candidate_A` | `0.5` | `(0.03, 0.05, 0.08)` | `(2.0, 2.0, 2.0)` | `0.5` | `1.0` | `0.15` |
| `Candidate_B` | `0.25` | `(0.015, 0.025, 0.04)` | `(1.0, 1.0, 1.0)` | `0.0` | `1.0` | `0.1` |

---

## 1. Tabella 1 — Metriche Broadband & Parametri Fisici dell'Imager (`Z_NEUTRAL`, `V0p_z_opt`, `V3b_z_opt`)

| Sessione | Punto | `w_low / mid / high` | `SMR mix -> pred` (dB) | $\Delta$SMR (dB) | `corr mix -> pred` | $\Delta\rho_{LR}$ | `D_mono mix -> pred` (dB) | $\Delta D_{\text{mono}}$ (dB) | `L/R bal pred` (dB) |
|:---|:---|:---:|:---:|---:|:---:|---:|:---:|---:|---:|
| **`track01`** | `Z_NEUTRAL` | `1.0000 / 1.0000 / 1.0000` | -9.121 $\to$ -9.127 | -0.006 | 0.7855 $\to$ 0.7856 | +0.0001 | 0.502 $\to$ 0.501 | -0.001 | +0.724 |
| **`track01`** | `V0p_z_opt` | `1.0422 / 1.1117 / 1.1098` | -9.121 $\to$ -8.272 | +0.849 | 0.7855 $\to$ 0.7444 | -0.0411 | 0.502 $\to$ 0.603 | +0.101 | +0.730 |
| **`track01`** | `V3b_z_opt` | `1.0423 / 1.1116 / 1.1081` | -9.121 $\to$ -8.278 | +0.843 | 0.7855 $\to$ 0.7447 | -0.0408 | 0.502 $\to$ 0.602 | +0.100 | +0.729 |
| **`track02`** | `Z_NEUTRAL` | `1.0000 / 1.0000 / 1.0000` | -14.733 $\to$ -14.730 | +0.003 | 0.9013 $\to$ 0.9013 | -0.0000 | 0.222 $\to$ 0.222 | +0.000 | -0.039 |
| **`track02`** | `V0p_z_opt` | `1.0230 / 1.1119 / 1.1112` | -14.733 $\to$ -13.167 | +1.566 | 0.9013 $\to$ 0.8632 | -0.0381 | 0.222 $\to$ 0.312 | +0.090 | -0.061 |
| **`track02`** | `V3b_z_opt` | `1.0230 / 1.1119 / 1.1112` | -14.733 $\to$ -13.167 | +1.566 | 0.9013 $\to$ 0.8632 | -0.0381 | 0.222 $\to$ 0.312 | +0.090 | -0.061 |
| **`track03`** | `Z_NEUTRAL` | `1.0000 / 1.0000 / 1.0000` | -8.088 $\to$ -8.084 | +0.003 | 0.7292 $\to$ 0.7291 | -0.0002 | 0.633 $\to$ 0.633 | +0.000 | -0.141 |
| **`track03`** | `V0p_z_opt` | `1.0425 / 1.1117 / 1.1105` | -8.088 $\to$ -7.182 | +0.906 | 0.7292 $\to$ 0.6763 | -0.0529 | 0.633 $\to$ 0.769 | +0.136 | -0.154 |
| **`track03`** | `V3b_z_opt` | `1.0425 / 1.1117 / 1.1105` | -8.088 $\to$ -7.182 | +0.906 | 0.7292 $\to$ 0.6763 | -0.0529 | 0.633 $\to$ 0.769 | +0.136 | -0.154 |
| **`track01_style`** | `Z_NEUTRAL` | `1.0000 / 1.0000 / 1.0000` | -9.122 $\to$ -9.129 | -0.008 | 0.7855 $\to$ 0.7857 | +0.0002 | 0.502 $\to$ 0.501 | -0.001 | +0.723 |
| **`track01_style`** | `V0p_z_opt` | `0.9455 / 0.8970 / 0.8938` | -9.122 $\to$ -9.996 | -0.875 | 0.7855 $\to$ 0.8215 | +0.0361 | 0.502 $\to$ 0.415 | -0.087 | +0.722 |
| **`track01_style`** | `V3b_z_opt` | `0.9455 / 0.8970 / 0.8938` | -9.122 $\to$ -9.996 | -0.875 | 0.7855 $\to$ 0.8215 | +0.0361 | 0.502 $\to$ 0.415 | -0.087 | +0.722 |
| **`track02_style`** | `Z_NEUTRAL` | `1.0000 / 1.0000 / 1.0000` | -14.732 $\to$ -14.732 | -0.000 | 0.9013 $\to$ 0.9013 | +0.0000 | 0.222 $\to$ 0.222 | -0.000 | -0.039 |
| **`track02_style`** | `V0p_z_opt` | `1.0434 / 1.1127 / 1.1116` | -14.732 $\to$ -13.151 | +1.581 | 0.9013 $\to$ 0.8630 | -0.0383 | 0.222 $\to$ 0.313 | +0.091 | -0.061 |
| **`track02_style`** | `V3b_z_opt` | `1.0434 / 1.1127 / 1.1116` | -14.732 $\to$ -13.151 | +1.581 | 0.9013 $\to$ 0.8630 | -0.0383 | 0.222 $\to$ 0.313 | +0.091 | -0.061 |
| **`track03_style`** | `Z_NEUTRAL` | `1.0000 / 1.0000 / 1.0000` | -8.086 $\to$ -8.083 | +0.003 | 0.7292 $\to$ 0.7290 | -0.0002 | 0.633 $\to$ 0.634 | +0.000 | -0.140 |
| **`track03_style`** | `V0p_z_opt` | `1.0422 / 1.1104 / 1.1101` | -8.086 $\to$ -7.182 | +0.904 | 0.7292 $\to$ 0.6764 | -0.0528 | 0.633 $\to$ 0.768 | +0.135 | -0.154 |
| **`track03_style`** | `V3b_z_opt` | `1.0422 / 1.1104 / 1.1101` | -8.086 $\to$ -7.182 | +0.904 | 0.7292 $\to$ 0.6764 | -0.0528 | 0.633 $\to$ 0.768 | +0.135 | -0.154 |

---

## 2. Tabella 2 — Analisi Per Banda (`low < 160 Hz`, `mid 160–3200 Hz`, `high > 3200 Hz`) al Punto `V0p_z_opt` vs `V3b_z_opt`

| Sessione | Punto | Banda | `valid` | `hard_pan` | `SMR mix -> pred` (dB) | $\Delta$SMR (dB) | `corr mix -> pred` | $\Delta\rho_{LR}$ | `D_mono mix -> pred` (dB) | $\Delta D_{\text{mono}}$ (dB) |
|:---|:---|:---:|:---:|:---:|:---:|---:|:---:|---:|:---:|---:|
| **`track01`** | `V0p_z_opt` | `low` | OK | no | -12.316 $\to$ -11.656 | +0.660 | 0.8965 $\to$ 0.8804 | -0.0161 | 0.252 $\to$ 0.292 | +0.041 |
| **`track01`** | `V0p_z_opt` | `mid` | OK | no | -6.769 $\to$ -6.125 | +0.644 | 0.6484 $\to$ 0.6032 | -0.0452 | 0.848 $\to$ 0.971 | +0.123 |
| **`track01`** | `V0p_z_opt` | `high` | OK | no | -8.118 $\to$ -7.272 | +0.846 | 0.7321 $\to$ 0.6824 | -0.0498 | 0.633 $\to$ 0.759 | +0.125 |
| **`track01`** | `V3b_z_opt` | `low` | OK | no | -12.316 $\to$ -11.664 | +0.652 | 0.8965 $\to$ 0.8806 | -0.0159 | 0.252 $\to$ 0.292 | +0.040 |
| **`track01`** | `V3b_z_opt` | `mid` | OK | no | -6.769 $\to$ -6.128 | +0.641 | 0.6484 $\to$ 0.6034 | -0.0450 | 0.848 $\to$ 0.971 | +0.122 |
| **`track01`** | `V3b_z_opt` | `high` | OK | no | -8.118 $\to$ -7.281 | +0.837 | 0.7321 $\to$ 0.6829 | -0.0492 | 0.633 $\to$ 0.757 | +0.124 |
| **`track02`** | `V0p_z_opt` | `low` | OK | no | -31.806 $\to$ -31.416 | +0.390 | 0.9987 $\to$ 0.9985 | -0.0001 | 0.003 $\to$ 0.003 | +0.000 |
| **`track02`** | `V0p_z_opt` | `mid` | OK | no | -11.021 $\to$ -9.956 | +1.065 | 0.8127 $\to$ 0.7704 | -0.0422 | 0.437 $\to$ 0.544 | +0.107 |
| **`track02`** | `V0p_z_opt` | `high` | OK | no | -12.512 $\to$ -11.630 | +0.881 | 0.8724 $\to$ 0.8467 | -0.0257 | 0.292 $\to$ 0.354 | +0.062 |
| **`track02`** | `V3b_z_opt` | `low` | OK | no | -31.806 $\to$ -31.416 | +0.390 | 0.9987 $\to$ 0.9985 | -0.0001 | 0.003 $\to$ 0.003 | +0.000 |
| **`track02`** | `V3b_z_opt` | `mid` | OK | no | -11.021 $\to$ -9.956 | +1.065 | 0.8127 $\to$ 0.7704 | -0.0422 | 0.437 $\to$ 0.544 | +0.107 |
| **`track02`** | `V3b_z_opt` | `high` | OK | no | -12.512 $\to$ -11.630 | +0.881 | 0.8724 $\to$ 0.8467 | -0.0257 | 0.292 $\to$ 0.354 | +0.062 |
| **`track03`** | `V0p_z_opt` | `low` | OK | no | -11.272 $\to$ -10.739 | +0.533 | 0.8594 $\to$ 0.8426 | -0.0168 | 0.318 $\to$ 0.357 | +0.040 |
| **`track03`** | `V0p_z_opt` | `mid` | OK | no | -6.480 $\to$ -5.688 | +0.792 | 0.6270 $\to$ 0.5701 | -0.0569 | 0.902 $\to$ 1.057 | +0.155 |
| **`track03`** | `V0p_z_opt` | `high` | OK | no | -6.503 $\to$ -5.567 | +0.936 | 0.6134 $\to$ 0.5453 | -0.0681 | 0.948 $\to$ 1.140 | +0.192 |
| **`track03`** | `V3b_z_opt` | `low` | OK | no | -11.272 $\to$ -10.739 | +0.533 | 0.8594 $\to$ 0.8426 | -0.0168 | 0.318 $\to$ 0.357 | +0.040 |
| **`track03`** | `V3b_z_opt` | `mid` | OK | no | -6.480 $\to$ -5.688 | +0.792 | 0.6270 $\to$ 0.5701 | -0.0569 | 0.902 $\to$ 1.057 | +0.155 |
| **`track03`** | `V3b_z_opt` | `high` | OK | no | -6.503 $\to$ -5.567 | +0.936 | 0.6134 $\to$ 0.5453 | -0.0681 | 0.948 $\to$ 1.140 | +0.192 |
| **`track01_style`** | `V0p_z_opt` | `low` | OK | no | -12.319 $\to$ -13.038 | -0.719 | 0.8965 $\to$ 0.9117 | +0.0152 | 0.251 $\to$ 0.214 | -0.037 |
| **`track01_style`** | `V0p_z_opt` | `mid` | OK | no | -6.770 $\to$ -7.489 | -0.720 | 0.6484 $\to$ 0.6942 | +0.0458 | 0.848 $\to$ 0.728 | -0.120 |
| **`track01_style`** | `V0p_z_opt` | `high` | OK | no | -8.117 $\to$ -9.041 | -0.923 | 0.7321 $\to$ 0.7778 | +0.0457 | 0.634 $\to$ 0.519 | -0.115 |
| **`track01_style`** | `V3b_z_opt` | `low` | OK | no | -12.319 $\to$ -13.038 | -0.719 | 0.8965 $\to$ 0.9117 | +0.0152 | 0.251 $\to$ 0.214 | -0.037 |
| **`track01_style`** | `V3b_z_opt` | `mid` | OK | no | -6.770 $\to$ -7.489 | -0.720 | 0.6484 $\to$ 0.6942 | +0.0458 | 0.848 $\to$ 0.728 | -0.120 |
| **`track01_style`** | `V3b_z_opt` | `high` | OK | no | -8.117 $\to$ -9.041 | -0.923 | 0.7321 $\to$ 0.7778 | +0.0457 | 0.634 $\to$ 0.519 | -0.115 |
| **`track02_style`** | `V0p_z_opt` | `low` | OK | no | -31.809 $\to$ -31.282 | +0.527 | 0.9987 $\to$ 0.9985 | -0.0002 | 0.003 $\to$ 0.003 | +0.000 |
| **`track02_style`** | `V0p_z_opt` | `mid` | OK | no | -11.021 $\to$ -9.954 | +1.067 | 0.8127 $\to$ 0.7704 | -0.0423 | 0.437 $\to$ 0.545 | +0.108 |
| **`track02_style`** | `V0p_z_opt` | `high` | OK | no | -12.515 $\to$ -11.633 | +0.883 | 0.8725 $\to$ 0.8468 | -0.0257 | 0.291 $\to$ 0.354 | +0.062 |
| **`track02_style`** | `V3b_z_opt` | `low` | OK | no | -31.809 $\to$ -31.282 | +0.527 | 0.9987 $\to$ 0.9985 | -0.0002 | 0.003 $\to$ 0.003 | +0.000 |
| **`track02_style`** | `V3b_z_opt` | `mid` | OK | no | -11.021 $\to$ -9.954 | +1.067 | 0.8127 $\to$ 0.7704 | -0.0423 | 0.437 $\to$ 0.545 | +0.108 |
| **`track02_style`** | `V3b_z_opt` | `high` | OK | no | -12.515 $\to$ -11.633 | +0.883 | 0.8725 $\to$ 0.8468 | -0.0257 | 0.291 $\to$ 0.354 | +0.062 |
| **`track03_style`** | `V0p_z_opt` | `low` | OK | no | -11.271 $\to$ -10.740 | +0.532 | 0.8593 $\to$ 0.8426 | -0.0168 | 0.318 $\to$ 0.357 | +0.039 |
| **`track03_style`** | `V0p_z_opt` | `mid` | OK | no | -6.480 $\to$ -5.697 | +0.783 | 0.6270 $\to$ 0.5708 | -0.0562 | 0.902 $\to$ 1.055 | +0.153 |
| **`track03_style`** | `V0p_z_opt` | `high` | OK | no | -6.503 $\to$ -5.561 | +0.942 | 0.6134 $\to$ 0.5454 | -0.0681 | 0.948 $\to$ 1.140 | +0.192 |
| **`track03_style`** | `V3b_z_opt` | `low` | OK | no | -11.271 $\to$ -10.740 | +0.532 | 0.8593 $\to$ 0.8426 | -0.0168 | 0.318 $\to$ 0.357 | +0.039 |
| **`track03_style`** | `V3b_z_opt` | `mid` | OK | no | -6.480 $\to$ -5.697 | +0.783 | 0.6270 $\to$ 0.5708 | -0.0562 | 0.902 $\to$ 1.055 | +0.153 |
| **`track03_style`** | `V3b_z_opt` | `high` | OK | no | -6.503 $\to$ -5.561 | +0.942 | 0.6134 $\to$ 0.5454 | -0.0681 | 0.948 $\to$ 1.140 | +0.192 |

---

## 3. Tabella 3 — Correlazione Locale Multi-Finestra (`50 ms / 100 ms / 250 ms`) & Diagnostica Comb-Filtering (`mix` vs `pred @ V0p_z_opt`)

| Sessione | Banda / Scope | `50ms` mix min/CVaR $\to$ pred | `100ms` mix min/CVaR $\to$ pred | `250ms` mix min/CVaR $\to$ pred | `comb_min_notch` pred (worst dB) | $\Delta$comb notch (dB) |
|:---|:---:|:---:|:---:|:---:|---:|---:|
| **`track01`** | `broadband` | 0.416/0.482 $\to$ 0.345/0.403 | 0.448/0.540 $\to$ 0.382/0.474 | 0.616/0.622 $\to$ 0.551/0.554 | -1.13 | -0.57 |
| **`track01`** | `low` | 0.031/0.484 $\to$ 0.019/0.461 | 0.459/0.626 $\to$ 0.473/0.622 | 0.703/0.735 $\to$ 0.668/0.705 | — | — |
| **`track01`** | `mid` | -0.169/0.012 $\to$ -0.208/-0.068 | -0.044/0.113 $\to$ -0.089/0.027 | 0.236/0.254 $\to$ 0.145/0.164 | — | — |
| **`track01`** | `high` | 0.051/0.199 $\to$ -0.060/0.083 | 0.115/0.333 $\to$ 0.004/0.234 | 0.371/0.375 $\to$ 0.274/0.274 | — | — |
| **`track02`** | `broadband` | -0.088/0.077 $\to$ -0.238/-0.066 | 0.104/0.128 $\to$ -0.027/-0.007 | 0.182/0.227 $\to$ 0.038/0.087 | -0.15 | -0.33 |
| **`track02`** | `low` | 0.731/0.877 $\to$ -0.578/0.412 | 0.730/0.859 $\to$ 0.437/0.687 | 0.931/0.931 $\to$ 0.378/0.654 | — | — |
| **`track02`** | `mid` | -0.393/-0.147 $\to$ -0.483/-0.250 | -0.082/-0.041 $\to$ -0.201/-0.153 | -0.018/-0.001 $\to$ -0.127/-0.109 | — | — |
| **`track02`** | `high` | 0.133/0.194 $\to$ 0.019/0.080 | 0.196/0.204 $\to$ 0.088/0.095 | 0.602/0.611 $\to$ 0.523/0.536 | — | — |
| **`track03`** | `broadband` | 0.177/0.325 $\to$ 0.141/0.328 | 0.393/0.419 $\to$ 0.325/0.386 | 0.513/0.538 $\to$ 0.500/0.505 | -1.28 | -0.47 |
| **`track03`** | `low` | -0.250/-0.145 $\to$ -0.352/-0.171 | -0.118/-0.036 $\to$ -0.160/-0.133 | 0.553/0.568 $\to$ 0.521/0.534 | — | — |
| **`track03`** | `mid` | 0.143/0.207 $\to$ 0.061/0.130 | 0.179/0.244 $\to$ 0.106/0.184 | 0.333/0.403 $\to$ 0.271/0.329 | — | — |
| **`track03`** | `high` | 0.109/0.208 $\to$ 0.011/0.109 | 0.223/0.241 $\to$ 0.127/0.144 | 0.292/0.292 $\to$ 0.199/0.200 | — | — |
| **`track01_style`** | `broadband` | 0.411/0.471 $\to$ 0.520/0.574 | 0.445/0.539 $\to$ 0.571/0.623 | 0.618/0.619 $\to$ 0.662/0.672 | -0.02 | +0.43 |
| **`track01_style`** | `low` | 0.029/0.485 $\to$ 0.055/0.535 | 0.459/0.628 $\to$ 0.541/0.682 | 0.705/0.736 $\to$ 0.736/0.769 | — | — |
| **`track01_style`** | `mid` | -0.116/0.029 $\to$ -0.008/0.162 | -0.013/0.127 $\to$ 0.107/0.235 | 0.235/0.254 $\to$ 0.323/0.342 | — | — |
| **`track01_style`** | `high` | 0.064/0.202 $\to$ 0.183/0.317 | 0.114/0.333 $\to$ 0.228/0.429 | 0.371/0.375 $\to$ 0.466/0.472 | — | — |
| **`track02_style`** | `broadband` | -0.073/0.078 $\to$ -0.236/-0.065 | 0.105/0.129 $\to$ -0.030/-0.008 | 0.181/0.226 $\to$ 0.037/0.083 | -0.16 | -0.33 |
| **`track02_style`** | `low` | 0.739/0.882 $\to$ -0.580/0.008 | 0.737/0.865 $\to$ 0.447/0.688 | 0.934/0.934 $\to$ 0.376/0.652 | — | — |
| **`track02_style`** | `mid` | -0.386/-0.141 $\to$ -0.474/-0.245 | -0.087/-0.043 $\to$ -0.192/-0.153 | -0.009/0.004 $\to$ -0.121/-0.107 | — | — |
| **`track02_style`** | `high` | -0.022/0.195 $\to$ -0.140/0.078 | 0.203/0.203 $\to$ 0.095/0.097 | 0.604/0.611 $\to$ 0.524/0.536 | — | — |
| **`track03_style`** | `broadband` | 0.183/0.327 $\to$ 0.133/0.330 | 0.393/0.419 $\to$ 0.321/0.386 | 0.512/0.538 $\to$ 0.502/0.508 | -1.28 | -0.47 |
| **`track03_style`** | `low` | -0.250/-0.131 $\to$ -0.370/-0.173 | -0.099/-0.049 $\to$ -0.164/-0.133 | 0.552/0.567 $\to$ 0.526/0.537 | — | — |
| **`track03_style`** | `mid` | 0.144/0.209 $\to$ 0.065/0.129 | 0.179/0.245 $\to$ 0.107/0.182 | 0.334/0.405 $\to$ 0.272/0.330 | — | — |
| **`track03_style`** | `high` | 0.110/0.211 $\to$ 0.016/0.114 | 0.225/0.242 $\to$ 0.133/0.148 | 0.291/0.293 $\to$ 0.200/0.202 | — | — |

---

## 4. Tabella 4 — Loss Candidate Non Pesate (`Zero_Threshold` vs `Candidate_A` vs `Candidate_B`) su `Z_NEUTRAL`, `V0p_z_opt` e `V3b_z_opt`

| Sessione | Punto | Config | `loss_mono_drop` | `loss_band_corr` | `loss_band_smr` | `loss_width_param` |
|:---|:---|:---|---:|---:|---:|---:|
| **`track01`** | `Z_NEUTRAL` | `Zero_Threshold` | 0.000000 | 0.000000 | 0.000335 | 0.000000 |
| **`track01`** | `Z_NEUTRAL` | `Candidate_A` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| **`track01`** | `Z_NEUTRAL` | `Candidate_B` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| **`track01`** | `V0p_z_opt` | `Zero_Threshold` | 0.010251 | 0.001668 | 0.976180 | 0.014044 |
| **`track01`** | `V0p_z_opt` | `Candidate_A` | 0.000000 | 0.000015 | 0.037837 | 0.001783 |
| **`track01`** | `V0p_z_opt` | `Candidate_B` | 0.000000 | 0.000240 | 0.448038 | 0.001899 |
| **`track01`** | `V3b_z_opt` | `Zero_Threshold` | 0.010107 | 0.001641 | 0.956141 | 0.013867 |
| **`track01`** | `V3b_z_opt` | `Candidate_A` | 0.000000 | 0.000014 | 0.035528 | 0.001789 |
| **`track01`** | `V3b_z_opt` | `Candidate_B` | 0.000000 | 0.000233 | 0.437571 | 0.001890 |
| **`track02`** | `Z_NEUTRAL` | `Zero_Threshold` | 0.000000 | 0.000000 | 0.001674 | 0.000000 |
| **`track02`** | `Z_NEUTRAL` | `Candidate_A` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| **`track02`** | `Z_NEUTRAL` | `Candidate_B` | 0.000000 | 0.000000 | 0.000029 | 0.000000 |
| **`track02`** | `V0p_z_opt` | `Zero_Threshold` | 0.011426 | 0.001080 | 0.882812 | 0.012972 |
| **`track02`** | `V0p_z_opt` | `Candidate_A` | 0.000000 | 0.000064 | 0.000664 | 0.000529 |
| **`track02`** | `V0p_z_opt` | `Candidate_B` | 0.000000 | 0.000280 | 0.186387 | 0.000663 |
| **`track02`** | `V3b_z_opt` | `Zero_Threshold` | 0.011426 | 0.001080 | 0.882812 | 0.012972 |
| **`track02`** | `V3b_z_opt` | `Candidate_A` | 0.000000 | 0.000064 | 0.000664 | 0.000529 |
| **`track02`** | `V3b_z_opt` | `Candidate_B` | 0.000000 | 0.000280 | 0.186387 | 0.000663 |
| **`track03`** | `Z_NEUTRAL` | `Zero_Threshold` | 0.000000 | 0.000000 | 0.000101 | 0.000000 |
| **`track03`** | `Z_NEUTRAL` | `Candidate_A` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| **`track03`** | `Z_NEUTRAL` | `Candidate_B` | 0.000000 | 0.000000 | 0.000057 | 0.000000 |
| **`track03`** | `V0p_z_opt` | `Zero_Threshold` | 0.019099 | 0.002803 | 0.888731 | 0.014152 |
| **`track03`** | `V0p_z_opt` | `Candidate_A` | 0.000000 | 0.000027 | 0.001672 | 0.001809 |
| **`track03`** | `V0p_z_opt` | `Candidate_B` | 0.000000 | 0.000687 | 0.285269 | 0.001932 |
| **`track03`** | `V3b_z_opt` | `Zero_Threshold` | 0.019099 | 0.002803 | 0.888731 | 0.014152 |
| **`track03`** | `V3b_z_opt` | `Candidate_A` | 0.000000 | 0.000027 | 0.001672 | 0.001809 |
| **`track03`** | `V3b_z_opt` | `Candidate_B` | 0.000000 | 0.000687 | 0.285269 | 0.001932 |
| **`track01_style`** | `Z_NEUTRAL` | `Zero_Threshold` | 0.000000 | 0.000000 | 0.000318 | 0.000000 |
| **`track01_style`** | `Z_NEUTRAL` | `Candidate_A` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| **`track01_style`** | `Z_NEUTRAL` | `Candidate_B` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| **`track01_style`** | `V0p_z_opt` | `Zero_Threshold` | 0.000000 | 0.000000 | 0.632064 | 0.010944 |
| **`track01_style`** | `V0p_z_opt` | `Candidate_A` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| **`track01_style`** | `V0p_z_opt` | `Candidate_B` | 0.000000 | 0.000000 | 0.000000 | 0.000024 |
| **`track01_style`** | `V3b_z_opt` | `Zero_Threshold` | 0.000000 | 0.000000 | 0.632064 | 0.010944 |
| **`track01_style`** | `V3b_z_opt` | `Candidate_A` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| **`track01_style`** | `V3b_z_opt` | `Candidate_B` | 0.000000 | 0.000000 | 0.000000 | 0.000024 |
| **`track02_style`** | `Z_NEUTRAL` | `Zero_Threshold` | 0.000000 | 0.000000 | 0.002777 | 0.000000 |
| **`track02_style`** | `Z_NEUTRAL` | `Candidate_A` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| **`track02_style`** | `Z_NEUTRAL` | `Candidate_B` | 0.000000 | 0.000000 | 0.000014 | 0.000000 |
| **`track02_style`** | `V0p_z_opt` | `Zero_Threshold` | 0.011585 | 0.001084 | 1.055584 | 0.014460 |
| **`track02_style`** | `V0p_z_opt` | `Candidate_A` | 0.000000 | 0.000065 | 0.013750 | 0.001883 |
| **`track02_style`** | `V0p_z_opt` | `Candidate_B` | 0.000000 | 0.000283 | 0.314350 | 0.002031 |
| **`track02_style`** | `V3b_z_opt` | `Zero_Threshold` | 0.011585 | 0.001084 | 1.055584 | 0.014460 |
| **`track02_style`** | `V3b_z_opt` | `Candidate_A` | 0.000000 | 0.000065 | 0.013750 | 0.001883 |
| **`track02_style`** | `V3b_z_opt` | `Candidate_B` | 0.000000 | 0.000283 | 0.314350 | 0.002031 |
| **`track03_style`** | `Z_NEUTRAL` | `Zero_Threshold` | 0.000000 | 0.000000 | 0.000105 | 0.000000 |
| **`track03_style`** | `Z_NEUTRAL` | `Candidate_A` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| **`track03_style`** | `Z_NEUTRAL` | `Candidate_B` | 0.000000 | 0.000000 | 0.000045 | 0.000000 |
| **`track03_style`** | `V0p_z_opt` | `Zero_Threshold` | 0.018994 | 0.002761 | 0.887096 | 0.013944 |
| **`track03_style`** | `V0p_z_opt` | `Candidate_A` | 0.000000 | 0.000023 | 0.001604 | 0.001785 |
| **`track03_style`** | `V0p_z_opt` | `Candidate_B` | 0.000000 | 0.000659 | 0.284241 | 0.001890 |
| **`track03_style`** | `V3b_z_opt` | `Zero_Threshold` | 0.018994 | 0.002761 | 0.887096 | 0.013944 |
| **`track03_style`** | `V3b_z_opt` | `Candidate_A` | 0.000000 | 0.000023 | 0.001604 | 0.001785 |
| **`track03_style`** | `V3b_z_opt` | `Candidate_B` | 0.000000 | 0.000659 | 0.284241 | 0.001890 |

---

## 5. Tabella 5 — Diagnostica Norme L2 dei Gradienti (`z` vs `u`, 41D + 6 Gruppi) a `V0p_z_opt` & `V3b_z_opt` (`Zero_Threshold`)

| Sessione | Punto | Termine | $\|\nabla_z\|_2$ (41D) | $\|\nabla_u\|_2$ (41D) | EQ $u$ | Xover $u$ | Comp $u$ | Imager $u$ | Sat $u$ | Lim $u$ | Ratio $\|\nabla_u\|/\|\nabla_u V_{0p}\|$ |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`track01`** | `V0p_z_opt` | `V0p_total` | 9.5856 | 1.7161 | 1.3585 | 0.0816 | 1.0416 | 0.0890 | 0.0000 | 0.0012 | 1.0000 |
| **`track01`** | `V0p_z_opt` | `LUFS_only` | 8.5028 | 1.5512 | 1.2367 | 0.0804 | 0.9298 | 0.0780 | 0.0000 | 0.0010 | 0.9040 |
| **`track01`** | `V0p_z_opt` | `loss_mono_drop` | 0.13769 | 0.03095 | 0.01528 | 0.00287 | 0.01663 | 0.02096 | 0.00000 | 0.00000 | 0.0180 |
| **`track01`** | `V0p_z_opt` | `loss_band_corr` | 0.01734 | 0.00410 | 0.00135 | 0.00008 | 0.00022 | 0.00387 | 0.00000 | 0.00000 | 0.0024 |
| **`track01`** | `V0p_z_opt` | `loss_band_smr` | 12.20486 | 2.21180 | 1.40530 | 0.35917 | 0.78794 | 1.47219 | 0.00002 | 0.00015 | 1.2889 |
| **`track01`** | `V0p_z_opt` | `loss_width_param` | 0.12454 | 0.02695 | 0.00000 | 0.00000 | 0.00000 | 0.02695 | 0.00000 | 0.00000 | 0.0157 |
| **`track01`** | `V3b_z_opt` | `V0p_total` | 9.5671 | 1.7110 | 1.3561 | 0.0601 | 1.0379 | 0.0868 | 0.0000 | 0.0012 | 1.0000 |
| **`track01`** | `V3b_z_opt` | `loss_mono_drop` | 0.13596 | 0.03044 | 0.01516 | 0.00285 | 0.01634 | 0.02054 | 0.00000 | 0.00000 | 0.0178 |
| **`track01`** | `V3b_z_opt` | `loss_band_corr` | 0.01714 | 0.00405 | 0.00132 | 0.00012 | 0.00023 | 0.00382 | 0.00000 | 0.00000 | 0.0024 |
| **`track01`** | `V3b_z_opt` | `loss_band_smr` | 12.06502 | 2.16550 | 1.39133 | 0.33720 | 0.75091 | 1.44084 | 0.00002 | 0.00014 | 1.2657 |
| **`track01`** | `V3b_z_opt` | `loss_width_param` | 0.12387 | 0.02678 | 0.00000 | 0.00000 | 0.00000 | 0.02678 | 0.00000 | 0.00000 | 0.0157 |
| **`track02`** | `V0p_z_opt` | `V0p_total` | 11.2633 | 2.0004 | 1.5313 | 0.1812 | 1.2705 | 0.0987 | 0.0001 | 0.0014 | 1.0000 |
| **`track02`** | `V0p_z_opt` | `LUFS_only` | 10.1275 | 1.8322 | 1.4008 | 0.1761 | 1.1646 | 0.0860 | 0.0000 | 0.0012 | 0.9159 |
| **`track02`** | `V0p_z_opt` | `loss_mono_drop` | 0.15176 | 0.03600 | 0.02181 | 0.00017 | 0.01972 | 0.02076 | 0.00000 | 0.00000 | 0.0180 |
| **`track02`** | `V0p_z_opt` | `loss_band_corr` | 0.01027 | 0.00247 | 0.00067 | 0.00002 | 0.00022 | 0.00237 | 0.00000 | 0.00000 | 0.0012 |
| **`track02`** | `V0p_z_opt` | `loss_band_smr` | 9.17627 | 1.91582 | 0.85621 | 0.11755 | 0.71794 | 1.55177 | 0.00002 | 0.00000 | 0.9577 |
| **`track02`** | `V0p_z_opt` | `loss_width_param` | 0.11503 | 0.02694 | 0.00000 | 0.00000 | 0.00000 | 0.02694 | 0.00000 | 0.00000 | 0.0135 |
| **`track02`** | `V3b_z_opt` | `V0p_total` | 11.2633 | 2.0004 | 1.5313 | 0.1812 | 1.2705 | 0.0987 | 0.0001 | 0.0014 | 1.0000 |
| **`track02`** | `V3b_z_opt` | `loss_mono_drop` | 0.15176 | 0.03600 | 0.02181 | 0.00017 | 0.01972 | 0.02076 | 0.00000 | 0.00000 | 0.0180 |
| **`track02`** | `V3b_z_opt` | `loss_band_corr` | 0.01027 | 0.00247 | 0.00067 | 0.00002 | 0.00022 | 0.00237 | 0.00000 | 0.00000 | 0.0012 |
| **`track02`** | `V3b_z_opt` | `loss_band_smr` | 9.17627 | 1.91582 | 0.85621 | 0.11755 | 0.71794 | 1.55177 | 0.00002 | 0.00000 | 0.9577 |
| **`track02`** | `V3b_z_opt` | `loss_width_param` | 0.11503 | 0.02694 | 0.00000 | 0.00000 | 0.00000 | 0.02694 | 0.00000 | 0.00000 | 0.0135 |
| **`track03`** | `V0p_z_opt` | `V0p_total` | 11.7570 | 2.1313 | 1.7220 | 0.0218 | 1.2469 | 0.1486 | 0.0001 | 0.0014 | 1.0000 |
| **`track03`** | `V0p_z_opt` | `LUFS_only` | 10.3871 | 1.9150 | 1.5563 | 0.0105 | 1.1087 | 0.1257 | 0.0000 | 0.0012 | 0.8985 |
| **`track03`** | `V0p_z_opt` | `loss_mono_drop` | 0.22961 | 0.05215 | 0.02620 | 0.00246 | 0.02669 | 0.03627 | 0.00000 | 0.00000 | 0.0245 |
| **`track03`** | `V0p_z_opt` | `loss_band_corr` | 0.02583 | 0.00619 | 0.00059 | 0.00019 | 0.00026 | 0.00616 | 0.00000 | 0.00000 | 0.0029 |
| **`track03`** | `V0p_z_opt` | `loss_band_smr` | 10.67918 | 1.86273 | 1.14869 | 0.14656 | 0.43598 | 1.39238 | 0.00000 | 0.00002 | 0.8740 |
| **`track03`** | `V0p_z_opt` | `loss_width_param` | 0.12507 | 0.02703 | 0.00000 | 0.00000 | 0.00000 | 0.02703 | 0.00000 | 0.00000 | 0.0127 |
| **`track03`** | `V3b_z_opt` | `V0p_total` | 11.7570 | 2.1313 | 1.7220 | 0.0218 | 1.2469 | 0.1486 | 0.0001 | 0.0014 | 1.0000 |
| **`track03`** | `V3b_z_opt` | `loss_mono_drop` | 0.22961 | 0.05215 | 0.02620 | 0.00246 | 0.02669 | 0.03627 | 0.00000 | 0.00000 | 0.0245 |
| **`track03`** | `V3b_z_opt` | `loss_band_corr` | 0.02583 | 0.00619 | 0.00059 | 0.00019 | 0.00026 | 0.00616 | 0.00000 | 0.00000 | 0.0029 |
| **`track03`** | `V3b_z_opt` | `loss_band_smr` | 10.67918 | 1.86273 | 1.14869 | 0.14656 | 0.43598 | 1.39238 | 0.00000 | 0.00002 | 0.8740 |
| **`track03`** | `V3b_z_opt` | `loss_width_param` | 0.12507 | 0.02703 | 0.00000 | 0.00000 | 0.00000 | 0.02703 | 0.00000 | 0.00000 | 0.0127 |
| **`track01_style`** | `V0p_z_opt` | `V0p_total` | 10.8010 | 1.6680 | 1.3583 | 0.5618 | 0.7833 | 0.0907 | 0.0004 | 0.0004 | 1.0000 |
| **`track01_style`** | `V0p_z_opt` | `LUFS_only` | 11.4029 | 1.8737 | 1.6440 | 0.0010 | 0.8950 | 0.0845 | 0.0000 | 0.0004 | 1.1233 |
| **`track01_style`** | `V0p_z_opt` | `loss_mono_drop` | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.0000 |
| **`track01_style`** | `V0p_z_opt` | `loss_band_corr` | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.0000 |
| **`track01_style`** | `V0p_z_opt` | `loss_band_smr` | 6.74505 | 1.34507 | 0.48154 | 0.07941 | 0.12679 | 1.24698 | 0.00000 | 0.00000 | 0.8064 |
| **`track01_style`** | `V0p_z_opt` | `loss_width_param` | 0.10356 | 0.02084 | 0.00000 | 0.00000 | 0.00000 | 0.02084 | 0.00000 | 0.00000 | 0.0125 |
| **`track01_style`** | `V3b_z_opt` | `V0p_total` | 10.8010 | 1.6680 | 1.3583 | 0.5618 | 0.7833 | 0.0907 | 0.0004 | 0.0004 | 1.0000 |
| **`track01_style`** | `V3b_z_opt` | `loss_mono_drop` | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.0000 |
| **`track01_style`** | `V3b_z_opt` | `loss_band_corr` | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.0000 |
| **`track01_style`** | `V3b_z_opt` | `loss_band_smr` | 6.74505 | 1.34507 | 0.48154 | 0.07941 | 0.12679 | 1.24698 | 0.00000 | 0.00000 | 0.8064 |
| **`track01_style`** | `V3b_z_opt` | `loss_width_param` | 0.10356 | 0.02084 | 0.00000 | 0.00000 | 0.00000 | 0.02084 | 0.00000 | 0.00000 | 0.0125 |
| **`track02_style`** | `V0p_z_opt` | `V0p_total` | 186.1105 | 34.0900 | 26.5970 | 3.8562 | 20.9200 | 1.4863 | 0.0005 | 0.0223 | 1.0000 |
| **`track02_style`** | `V0p_z_opt` | `LUFS_only` | 181.8074 | 32.8407 | 25.5131 | 0.0110 | 20.6390 | 1.2753 | 0.0004 | 0.0223 | 0.9634 |
| **`track02_style`** | `V0p_z_opt` | `loss_mono_drop` | 0.15284 | 0.03625 | 0.02191 | 0.00013 | 0.01994 | 0.02090 | 0.00000 | 0.00000 | 0.0011 |
| **`track02_style`** | `V0p_z_opt` | `loss_band_corr` | 0.01028 | 0.00247 | 0.00067 | 0.00001 | 0.00023 | 0.00237 | 0.00000 | 0.00000 | 0.0001 |
| **`track02_style`** | `V0p_z_opt` | `loss_band_smr` | 11.03912 | 2.16160 | 1.12681 | 0.10826 | 0.87097 | 1.62250 | 0.00002 | 0.00000 | 0.0634 |
| **`track02_style`** | `V0p_z_opt` | `loss_width_param` | 0.12656 | 0.02726 | 0.00000 | 0.00000 | 0.00000 | 0.02726 | 0.00000 | 0.00000 | 0.0008 |
| **`track02_style`** | `V3b_z_opt` | `V0p_total` | 186.1105 | 34.0900 | 26.5970 | 3.8562 | 20.9200 | 1.4863 | 0.0005 | 0.0223 | 1.0000 |
| **`track02_style`** | `V3b_z_opt` | `loss_mono_drop` | 0.15284 | 0.03625 | 0.02191 | 0.00013 | 0.01994 | 0.02090 | 0.00000 | 0.00000 | 0.0011 |
| **`track02_style`** | `V3b_z_opt` | `loss_band_corr` | 0.01028 | 0.00247 | 0.00067 | 0.00001 | 0.00023 | 0.00237 | 0.00000 | 0.00000 | 0.0001 |
| **`track02_style`** | `V3b_z_opt` | `loss_band_smr` | 11.03912 | 2.16160 | 1.12681 | 0.10826 | 0.87097 | 1.62250 | 0.00002 | 0.00000 | 0.0634 |
| **`track02_style`** | `V3b_z_opt` | `loss_width_param` | 0.12656 | 0.02726 | 0.00000 | 0.00000 | 0.00000 | 0.02726 | 0.00000 | 0.00000 | 0.0008 |
| **`track03_style`** | `V0p_z_opt` | `V0p_total` | 173.7236 | 31.9938 | 26.1812 | 0.6009 | 18.2453 | 2.2119 | 0.0005 | 0.0204 | 1.0000 |
| **`track03_style`** | `V0p_z_opt` | `LUFS_only` | 141.4308 | 25.9227 | 21.0721 | 0.0609 | 15.0075 | 1.6510 | 0.0003 | 0.0168 | 0.8102 |
| **`track03_style`** | `V0p_z_opt` | `loss_mono_drop` | 0.22752 | 0.05168 | 0.02609 | 0.00249 | 0.02641 | 0.03586 | 0.00000 | 0.00000 | 0.0016 |
| **`track03_style`** | `V0p_z_opt` | `loss_band_corr` | 0.02544 | 0.00611 | 0.00057 | 0.00019 | 0.00026 | 0.00607 | 0.00000 | 0.00000 | 0.0002 |
| **`track03_style`** | `V0p_z_opt` | `loss_band_smr` | 10.62440 | 1.85376 | 1.14725 | 0.14748 | 0.43355 | 1.38222 | 0.00000 | 0.00002 | 0.0579 |
| **`track03_style`** | `V0p_z_opt` | `loss_width_param` | 0.12415 | 0.02686 | 0.00000 | 0.00000 | 0.00000 | 0.02686 | 0.00000 | 0.00000 | 0.0008 |
| **`track03_style`** | `V3b_z_opt` | `V0p_total` | 173.7236 | 31.9938 | 26.1812 | 0.6009 | 18.2453 | 2.2119 | 0.0005 | 0.0204 | 1.0000 |
| **`track03_style`** | `V3b_z_opt` | `loss_mono_drop` | 0.22752 | 0.05168 | 0.02609 | 0.00249 | 0.02641 | 0.03586 | 0.00000 | 0.00000 | 0.0016 |
| **`track03_style`** | `V3b_z_opt` | `loss_band_corr` | 0.02544 | 0.00611 | 0.00057 | 0.00019 | 0.00026 | 0.00607 | 0.00000 | 0.00000 | 0.0002 |
| **`track03_style`** | `V3b_z_opt` | `loss_band_smr` | 10.62440 | 1.85376 | 1.14725 | 0.14748 | 0.43355 | 1.38222 | 0.00000 | 0.00002 | 0.0579 |
| **`track03_style`** | `V3b_z_opt` | `loss_width_param` | 0.12415 | 0.02686 | 0.00000 | 0.00000 | 0.00000 | 0.02686 | 0.00000 | 0.00000 | 0.0008 |

---

## 6. Tabella 6 — Cosine Similarity in Spazio `u` (`V3c Zero_Threshold` vs `V0p_total` & `stylistic_stereo`) a `V0p_z_opt`

| Sessione | Termine V3c | `cos_u (41D vs V0p)` | `cos_u (Imager vs V0p)` | `cos_u (EQ vs V0p)` | `cos_u (41D vs style_stereo)` | `cos_u (Imager vs style_stereo)` |
|:---|:---|---:|---:|---:|---:|---:|
| **`track01`** | `loss_mono_drop` | -0.2200 | -0.9785 | -0.3171 | N/A | N/A |
| **`track01`** | `loss_band_corr` | +0.0124 | -0.9484 | +0.2319 | N/A | N/A |
| **`track01`** | `loss_band_smr` | -0.2621 | -0.9472 | -0.3312 | N/A | N/A |
| **`track01`** | `loss_width_param` | -0.0475 | -0.9169 | N/A | N/A | N/A |
| **`track02`** | `loss_mono_drop` | -0.3663 | -0.9647 | -0.5373 | N/A | N/A |
| **`track02`** | `loss_band_corr` | -0.1135 | -0.9985 | -0.2555 | N/A | N/A |
| **`track02`** | `loss_band_smr` | -0.2182 | -0.9412 | -0.3785 | N/A | N/A |
| **`track02`** | `loss_width_param` | -0.0445 | -0.9018 | N/A | N/A | N/A |
| **`track03`** | `loss_mono_drop` | -0.2932 | -0.9592 | -0.4467 | N/A | N/A |
| **`track03`** | `loss_band_corr` | -0.0620 | -0.9944 | +0.1957 | N/A | N/A |
| **`track03`** | `loss_band_smr` | -0.1934 | -0.9469 | -0.2141 | N/A | N/A |
| **`track03`** | `loss_width_param` | -0.0666 | -0.9561 | N/A | N/A | N/A |
| **`track01_style`** | `loss_mono_drop` | N/A | N/A | N/A | N/A | N/A |
| **`track01_style`** | `loss_band_corr` | N/A | N/A | N/A | N/A | N/A |
| **`track01_style`** | `loss_band_smr` | -0.0049 | -0.5826 | -0.0241 | +0.1932 | +0.3595 |
| **`track01_style`** | `loss_width_param` | -0.0477 | -0.8773 | N/A | +0.0766 | +0.6413 |
| **`track02_style`** | `loss_mono_drop` | -0.4093 | -0.9975 | -0.5882 | +0.9499 | +0.9996 |
| **`track02_style`** | `loss_band_corr` | -0.1108 | -0.9687 | -0.2598 | +0.8367 | +0.9455 |
| **`track02_style`** | `loss_band_smr` | -0.2557 | -0.9030 | -0.3969 | +0.7853 | +0.8677 |
| **`track02_style`** | `loss_width_param` | -0.0348 | -0.7980 | N/A | +0.5692 | +0.7448 |
| **`track03_style`** | `loss_mono_drop` | -0.2475 | -0.9886 | -0.4029 | +0.9804 | +0.9858 |
| **`track03_style`** | `loss_band_corr` | -0.0544 | -0.9667 | +0.2126 | +0.5985 | +0.9260 |
| **`track03_style`** | `loss_band_smr` | -0.1723 | -0.9450 | -0.1935 | +0.7495 | +0.9782 |
| **`track03_style`** | `loss_width_param` | -0.0634 | -0.9170 | N/A | +0.5663 | +0.8941 |

---

## 7. Diagnostica Casi Limite & Maschera Energetica (24 Chunk Totali a `V0p_z_opt`)

| Sessione | Chunk Silenti | `broadband_valid` | `band_valid` `(L/M/H)` | `hard_panned` `(L/M/H)` | `m_active / m_total` medio |
|:---|---:|---:|:---:|:---:|---:|
| **`track01`** | `0/4` | `4/4` | `4/4/4` | `0/0/0` | `1.000` |
| **`track02`** | `0/4` | `4/4` | `4/4/4` | `0/0/0` | `1.000` |
| **`track03`** | `0/4` | `4/4` | `4/4/4` | `0/0/0` | `1.000` |
| **`track01_style`** | `0/4` | `4/4` | `4/4/4` | `0/0/0` | `1.000` |
| **`track02_style`** | `0/4` | `4/4` | `4/4/4` | `0/0/0` | `1.000` |
| **`track03_style`** | `0/4` | `4/4` | `4/4/4` | `0/0/0` | `1.000` |

---

## 8. Analisi Tecnica, Osservabilità Scale & Questioni Aperte per il Supervisore (`Argon`)

1. **Confronto `Z_NEUTRAL` vs `V0p_z_opt` vs `V3b_z_opt`:** Tutti i valori sopra riportano misure read-only esatte sui 4 chunk armonizzati per sessione (pre-roll completo fino a $120\,000$ campioni, warm-up LR4 $160/3200\text{ Hz}$, finestra comune $T_{\text{eval}}=143\,841$, zero training e zero loss V3b attiva). I width fisici reali di `V0p_z_opt` e `V3b_z_opt` in `SPRINT3_V3B_V0P_ABLATION_RESULTS.json` (`w_low` ~ `1.023–1.043`, `w_mid/high` ~ `1.110–1.113`, tranne `track01_style` dove `V3b_z_opt` riduce leggermente i width a `0.9455 / 0.8970 / 0.8938`) non producono ancora degradazioni catastrofiche sul programma reale ma evidenziano un comportamento differenziale fra `Candidate_A` e `Candidate_B` sulle soglie di guardia e sul termine parametrico.
2. **Sensibilità Finestre Short-Time (`50 ms` / `100 ms` / `250 ms`) & Comb-Filtering:** Il confronto affiancato `mix -> pred` sulle 3 finestre consente di separare le fluttuazioni transitorie/sub-periodo in banda `low` ($<160\text{ Hz}$, dove $50\text{ ms}$ abbraccia pochi cicli) da eventuali cali di correlazione locale in banda `mid/high` e notch spettrali in somma mono. In particolare su `track02` / `track02_style`, la banda `low` mostra correlazione globale integrata quasi unitaria (`0.9985`) ma scende puntualmente fino a `-0.578` / `-0.580` su finestra minima da `50 ms` (rispetto a `+0.437` / `+0.447` a `100 ms`), confermando la correttezza della decisione `Q5` di mantenere le metriche multi-finestra e anti-comb strettamente diagnostiche.
3. **Scale dei Gradienti (`z` vs `u`) & Allineamento Direzionale:** Le norme $\|\nabla_u\|_2$ (Tabella 5) e le cosine similarity rispetto a $\nabla_u V_{0p}$ e $\nabla_u \mathcal{L}_{\text{stereo}}^{\text{style}}$ (Tabella 6) quantificano l'impatto relativo di ciascuno dei 4 termini V3c prima di fissare i pesi finali (`loss_band_smr` non pesata ha scala $O(1)$ confrontabile o superiore a $V_{0p}$ in `same_track`, mentre `loss_mono_drop`, `loss_width_param` e `loss_band_corr` hanno rapporto naturale $0.1\%$–$2.5\%$).
4. **Questioni Aperte per il Supervisore (`Argon`):**
   - **`Q_A` (Soglie `Candidate_A` vs `Candidate_B`):** Alla luce delle loss non pesate misurate in Tabella 4 sui punti `V0p_z_opt` e `V3b_z_opt`, si preferisce adottare **`Candidate_A`** (`tau_drop_db=0.5`, `tau_corr=(0.03, 0.05, 0.08)`, `tau_smr_db=(2.0, 2.0, 2.0)`, `tau_sub_db=0.5`, `tau_w_low_max=1.00`, `tau_w_mid/high=0.15`) oppure **`Candidate_B`** (`tau_drop_db=0.25`, `tau_corr=(0.015, 0.025, 0.04)`, `tau_smr_db=(1.0, 1.0, 1.0)`, `tau_sub_db=0.0`, `tau_w_low_max=1.00`, `tau_w_mid/high=0.10`)?
   - **`Q_B` (Calibrazione Pesi `w_mono_drop`, `w_band_corr`, `w_band_smr`, `w_width_param`):** Sulla base dei rapporti $\|\nabla_u \mathcal{L}_{\text{term}}\|_2 / \|\nabla_u V_{0p}\|_2$ osservati in Tabella 5, quale target di rapporto di norma in spazio $u$ (es. `1%–10%` di $\|\nabla_u V_{0p}\|_2$ nei regimi attivi) deve guidare la scelta dei pesi per l'eventuale step di ablation V3c?
   - **`Q_C` (Gestione `track01` / `track01_style` in `stylistic` mode):** Si conferma che in modalità `stylistic` i pesi V3c devono agire come guardrail simmetrico anche quando il reference target (`stylistic_stereo`) spinge verso maggiore apertura in banda alta?

---

## 9. Manifest Completo dei File & Verifica Integrità (`SHA`)

- **Baseline Git pre-V3c (`9b1fd9115cce6deaffe637c649c4bb7d6159a195`):** `9b1fd9115cce6deaffe637c649c4bb7d6159a195`
- **Commit 1 (Specifica `V3c`):** `5028e61ad2ab334e2a883603fa3ffccd747257a0`
- **Commit 2 (Modulo Loss + 7 Gate Tests):** `6a2ecd3374f54212f5a403f829c373e1065c99b2`
- **SHA-256 `docs/sprint3/SPRINT3_V3C_STEREO_MONO_COMPATIBILITY_DESIGN.md`:** `6c0907d3da075e417ef25e7917ecef4cf82c29eeeca141a717d82ed0bcd414d2`
- **SHA-256 `python/ddsp_mastering/loss/stereo_prior.py`:** `64ca20047fd2ba783f515122b063217ada08341315114fa6df62d705626376dd`
- **SHA-256 `tests/test_sprint3_v3c_gate.py`:** `a52e116bb370a196e048ba80f2a62f1773c34c782ce9b330a979f43990291829`
- **SHA-256 `scripts/sprint3/run_v3c_preflight.py`:** `bfd328e7e3974ca2f31c71c2b0daf8a4be1ef4c17119129fc17c3a21ca703b4f`
- **SHA-256 `docs/sprint3/SPRINT3_V3C_PREFLIGHT_RESULTS.json`:** `f16600fbed7db1c6e951472f7b33d1c3293e27cf522c9dc88a08c40073d2c970`
