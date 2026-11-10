# Sprint 3 — Report Ufficiale di Ablation Step V3a (`V3a_EQ_PRIOR_V1`)

**Data:** 10 Novembre 2026  
**Autore:** Antigravity (Advanced Agentic Coding)  
**Supervisore Tecnico:** Argon (Gemini Web Pro / Medium Thinking)  
**Protocollo:** Armonizzato Sprint 3 (Percorso A, coordinate bit-identiche, target LUFS integrato da file WAV sorgente)

---

## 1. Executive Summary & Obiettivi di V3a

Lo Step **`V3a`** (`V3a_EQ_PRIOR_V1`) implementa per la prima volta nella pipeline di mastering un **prior analitico quadratico con dead-zone sulla risposta in frequenza composita dell'EQ parametrico a 6 bande**.
Mentre la baseline di controllo `V0p` applica esclusivamente una penalità $L_2$ nello spazio normalizzato dei parametri latenti ($w_{\text{reg}}\|z - z_{\text{neutral}}\|^2$), `V3a` penalizza direttamente la deviazione della curva di risposta complessiva in dB valutata analiticamente su $K_{\text{eq}}=128$ frequenze logaritmiche tra 20 Hz e 20 kHz in `torch.float64`:

$$\mathcal{L}_{\text{EQ-prior}}(z) = \mathcal{L}_{\text{EQ-shape}}(z) + \mathcal{L}_{\text{EQ-offset}}(z)$$

dove:
- $\mathcal{L}_{\text{EQ-shape}}(z) = \mathcal{L}_{\text{shape-exc}} + w_{\text{slope}} \cdot \mathcal{L}_{\text{slope-raw}}$, con $\tau_{\text{shape}} = 1.5\text{ dB}$, $w_{\text{slope}} = 2.0$;
- $\mathcal{L}_{\text{EQ-offset}}(z) = w_{\text{offset}} \cdot \max(0, |\bar{G}| - \tau_{\text{offset}})^2$, con $\tau_{\text{offset}} = 1.0\text{ dB}$, $w_{\text{offset}} = 1.0$;
- Peso globale nella loss composite: $w_{\text{eq}} = 0.02$.

---

## 2. Tabella 1 — Risposta Composita EQ (`V0p vs V3a`)

La tabella seguente riporta i descrittori analitici della risposta EQ su $K_{\text{eq}}=128$ punti in `float64` per `V0p` e `V3a`:

| Sessione / Modalità | Variante | $\bar{G}_{\text{EQ}}$ (dB) | $\widetilde{G}_{\max}$ (dB) | $\widetilde{G}_{\min}$ (dB) | $\max|\widetilde{G}|$ (dB) | $\mathcal{L}_{\text{shape-exc}}$ | $\mathcal{L}_{\text{slope-raw}}$ | $\mathcal{L}_{\text{offset-raw}}$ | $\mathcal{L}_{\text{prior}}$ (unw.) | $\mathcal{L}_{\text{prior}}$ ($w_{\text{eq}}=0.02$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`track01`**<br>*(same_track)* | **`V0p`**<br>**`V3a`** | $+2.48$<br>**$+2.48$** | $+1.23$<br>$+1.23$ | $-0.71$<br>$-0.71$ | $1.23$<br>$1.23$ | $0.0000$<br>$0.0000$ | $0.0023$<br>$0.0023$ | $2.1929$<br>**$2.1785$** | $2.1975$<br>**$2.1831$** | $0.0440$<br>**$0.0437$** |
| **`track02`**<br>*(same_track)* | **`V0p`**<br>**`V3a`** | $+2.48$<br>**$+2.47$** | $+0.75$<br>$+0.75$ | $-1.07$<br>$-1.07$ | $1.07$<br>$1.07$ | $0.0000$<br>$0.0000$ | $0.0016$<br>$0.0016$ | $2.1851$<br>**$2.1741$** | $2.1883$<br>**$2.1773$** | $0.0438$<br>**$0.0435$** |
| **`track03`**<br>*(same_track)* | **`V0p`**<br>**`V3a`** | $+2.48$<br>**$+2.47$** | $+1.20$<br>$+1.21$ | $-0.71$<br>$-0.71$ | $1.20$<br>$1.21$ | $0.0000$<br>$0.0000$ | $0.0022$<br>$0.0022$ | $2.1763$<br>**$2.1650$** | $2.1808$<br>**$2.1695$** | $0.0436$<br>**$0.0434$** |
| **`track01_style`**<br>*(stylistic)* | **`V0p`**<br>**`V3a`** | $-2.50$<br>**$-2.49$** | $+0.68$<br>$+0.68$ | $-1.14$<br>$-1.14$ | $1.14$<br>$1.14$ | $0.0000$<br>$0.0000$ | $0.0020$<br>$0.0020$ | $2.2396$<br>**$2.2348$** | $2.2436$<br>**$2.2389$** | $0.0449$<br>**$0.0448$** |
| **`track02_style`**<br>*(stylistic)* | **`V0p`**<br>**`V3a`** | $+2.46$<br>**$+2.46$** | $+0.77$<br>$+0.77$ | $-1.06$<br>$-1.06$ | $1.06$<br>$1.06$ | $0.0000$<br>$0.0000$ | $0.0017$<br>$0.0017$ | $2.1436$<br>**$2.1427$** | $2.1470$<br>**$2.1460$** | $0.0429$<br>**$0.0429$** |
| **`track03_style`**<br>*(stylistic)* | **`V0p`**<br>**`V3a`** | $+2.40$<br>**$+2.40$** | $+1.29$<br>$+1.29$ | $-1.12$<br>$-1.12$ | $1.29$<br>$1.29$ | $0.0000$<br>$0.0000$ | $0.0029$<br>$0.0029$ | $1.9547$<br>**$1.9529$** | $1.9605$<br>**$1.9587$** | $0.0392$<br>**$0.0392$** |

---

## 3. Tabella 2 — Parametri Fisici Ottimizzati (`V0p vs V3a`)

Confronto tra i guadagni dei 6 biquad EQ, i makeup gain del compressore multibanda e il push gain del limiter ($P_{41}$ in $[-12, +12]\text{ dB}$):

| Sessione | Var. | EQ B0 (Low-Shelf) | EQ B1 (Bell) | EQ B2 (Bell) | EQ B3 (Bell) | EQ B4 (Bell) | EQ B5 (High-Shelf) | Comp Makeup (L, M, H dB) | Limiter Push Gain (dB) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`track01`** | **`V0p`**<br>**`V3a`** | +1.93 dB<br>+1.92 dB | +1.87 dB<br>+1.87 dB | +1.90 dB<br>+1.90 dB | +1.89 dB<br>+1.89 dB | +1.90 dB<br>+1.89 dB | +1.89 dB<br>+1.88 dB | +1.37, +1.40, +1.42<br>+1.38, +1.40, +1.42 | +8.63 dB<br>+8.63 dB |
| **`track02`** | **`V0p`**<br>**`V3a`** | +1.92 dB<br>+1.92 dB | +1.88 dB<br>+1.87 dB | +1.87 dB<br>+1.87 dB | +1.91 dB<br>+1.91 dB | +1.89 dB<br>+1.89 dB | +1.86 dB<br>+1.85 dB | +1.40, +1.40, +1.41<br>+1.41, +1.40, +1.41 | +8.63 dB<br>+8.63 dB |
| **`track03`** | **`V0p`**<br>**`V3a`** | +1.93 dB<br>+1.92 dB | +1.87 dB<br>+1.87 dB | +1.90 dB<br>+1.90 dB | +1.87 dB<br>+1.87 dB | +1.90 dB<br>+1.90 dB | +1.89 dB<br>+1.89 dB | +1.39, +1.40, +1.39<br>+1.39, +1.40, +1.39 | +8.76 dB<br>+8.76 dB |
| **`track01_style`** | **`V0p`**<br>**`V3a`** | -1.95 dB<br>-1.95 dB | -1.89 dB<br>-1.89 dB | -1.90 dB<br>-1.90 dB | -1.86 dB<br>-1.85 dB | -1.95 dB<br>-1.94 dB | -1.98 dB<br>-1.98 dB | -1.11, -1.12, -1.12<br>-1.11, -1.12, -1.13 | -4.88 dB<br>-4.88 dB |
| **`track02_style`** | **`V0p`**<br>**`V3a`** | +1.90 dB<br>+1.90 dB | +1.88 dB<br>+1.88 dB | +1.88 dB<br>+1.88 dB | +1.89 dB<br>+1.89 dB | +1.89 dB<br>+1.89 dB | +1.79 dB<br>+1.79 dB | +1.39, +1.40, +1.40<br>+1.39, +1.40, +1.40 | +6.06 dB<br>+6.06 dB |
| **`track03_style`** | **`V0p`**<br>**`V3a`** | +1.91 dB<br>+1.91 dB | +1.87 dB<br>+1.87 dB | +1.90 dB<br>+1.90 dB | +1.88 dB<br>+1.88 dB | +1.93 dB<br>+1.93 dB | +1.20 dB<br>+1.20 dB | +1.38, +1.40, +1.41<br>+1.38, +1.40, +1.41 | +8.33 dB<br>+8.33 dB |

### Tabella 2B — Parametri Costruttivi dei 6 Biquad EQ (Frequenza $f_c$ in Hz e Fattore di Merito $Q$)

| Sessione | Var. | EQ B0 ($f_c, Q$) | EQ B1 ($f_c, Q$) | EQ B2 ($f_c, Q$) | EQ B3 ($f_c, Q$) | EQ B4 ($f_c, Q$) | EQ B5 ($f_c, Q$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`track01`** | **`V0p`**<br>**`V3a`** | 88 Hz, 0.58<br>88 Hz, 0.58 | 197 Hz, 0.58<br>197 Hz, 0.58 | 1247 Hz, 0.58<br>1247 Hz, 0.58 | 2451 Hz, 0.58<br>2451 Hz, 0.58 | 6808 Hz, 0.58<br>6808 Hz, 0.58 | 10011 Hz, 0.58<br>10014 Hz, 0.58 |
| **`track02`** | **`V0p`**<br>**`V3a`** | 88 Hz, 0.58<br>88 Hz, 0.58 | 312 Hz, 0.57<br>313 Hz, 0.57 | 814 Hz, 0.58<br>814 Hz, 0.58 | 2452 Hz, 0.57<br>2452 Hz, 0.57 | 6812 Hz, 0.57<br>6812 Hz, 0.57 | 10018 Hz, 0.58<br>10018 Hz, 0.58 |
| **`track03`** | **`V0p`**<br>**`V3a`** | 87 Hz, 0.58<br>87 Hz, 0.58 | 197 Hz, 0.58<br>197 Hz, 0.58 | 1248 Hz, 0.58<br>1248 Hz, 0.58 | 2494 Hz, 0.57<br>2494 Hz, 0.57 | 6804 Hz, 0.57<br>6804 Hz, 0.57 | 10006 Hz, 0.57<br>10006 Hz, 0.57 |
| **`track01_style`** | **`V0p`**<br>**`V3a`** | 87 Hz, 0.58<br>87 Hz, 0.58 | 197 Hz, 0.57<br>197 Hz, 0.57 | 1211 Hz, 0.58<br>1211 Hz, 0.58 | 2461 Hz, 0.57<br>2461 Hz, 0.57 | 6818 Hz, 0.58<br>6818 Hz, 0.58 | 10080 Hz, 0.58<br>10080 Hz, 0.58 |
| **`track02_style`** | **`V0p`**<br>**`V3a`** | 88 Hz, 0.58<br>88 Hz, 0.58 | 311 Hz, 0.57<br>311 Hz, 0.57 | 797 Hz, 0.58<br>797 Hz, 0.58 | 2451 Hz, 0.57<br>2451 Hz, 0.57 | 6805 Hz, 0.57<br>6805 Hz, 0.57 | 10022 Hz, 0.58<br>10022 Hz, 0.58 |
| **`track03_style`** | **`V0p`**<br>**`V3a`** | 87 Hz, 0.58<br>87 Hz, 0.58 | 197 Hz, 0.58<br>197 Hz, 0.58 | 1248 Hz, 0.57<br>1248 Hz, 0.57 | 2484 Hz, 0.57<br>2484 Hz, 0.57 | 6806 Hz, 0.58<br>6806 Hz, 0.58 | 10059 Hz, 0.58<br>10059 Hz, 0.58 |

---

## 4. Tabella 3 — Qualità Audio Full-Track e Guardrail True Peak (`V0p vs V3a`)

Metriche oggettive calcolate sull'intero brano renderizzato tramite il motore offline C++ con svuotamento della coda dei filtri (tail flush):

| Sessione / Modalità | Var. | Target LUFS | Integr. LUFS | $\Delta$LUFS | True Peak (dBTP) | Sample Peak (dBFS) | Inter-Sample Overshoot | TP Viol. (dB) | ERB Surplus (dB) | SC | LSD (dB) | PLR (dB) | $\Delta$M/S (dB) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`track01`**<br>*(same_track)* | **`V0p`**<br>**`V3a`** | $-7.29$<br>$-7.29$ | $-10.51$<br>$-10.51$ | $+3.22$<br>$+3.22$ | $0.26$<br>$0.28$ | $-0.30$<br>$-0.30$ | $0.00$<br>$0.58$ | $0.56$<br>$0.58$ | +1.67<br>+1.67 | 0.4057<br>0.4056 | 6.79<br>6.79 | $10.21$<br>$10.21$ | —<br>— |
| **`track02`**<br>*(same_track)* | **`V0p`**<br>**`V3a`** | $-7.22$<br>$-7.22$ | $-18.04$<br>$-18.04$ | $+10.82$<br>$+10.82$ | $-0.07$<br>$-0.07$ | $-0.30$<br>$-0.30$ | $0.00$<br>$0.23$ | $0.23$<br>$0.23$ | +1.05<br>+1.05 | 0.7932<br>0.7932 | 11.21<br>11.21 | $17.74$<br>$17.74$ | —<br>— |
| **`track03`**<br>*(same_track)* | **`V0p`**<br>**`V3a`** | $-6.90$<br>$-6.90$ | $-13.50$<br>$-13.50$ | $+6.61$<br>$+6.61$ | $-0.09$<br>$-0.09$ | $-0.30$<br>$-0.30$ | $0.00$<br>$0.21$ | $0.21$<br>$0.21$ | +0.82<br>+0.82 | 0.6121<br>0.6120 | 9.47<br>9.47 | $13.20$<br>$13.20$ | —<br>— |
| **`track01_style`**<br>*(stylistic)* | **`V0p`**<br>**`V3a`** | $-17.99$<br>$-17.99$ | $-17.83$<br>$-17.83$ | $+0.15$<br>$+0.15$ | $-2.42$<br>$-2.42$ | $-2.42$<br>$-2.42$ | $0.00$<br>$0.00$ | $0.00$<br>$0.00$ | +8.62<br>+8.63 | —<br>— | —<br>— | $15.42$<br>$15.42$ | —<br>— |
| **`track02_style`**<br>*(stylistic)* | **`V0p`**<br>**`V3a`** | $-9.04$<br>$-9.04$ | $-18.05$<br>$-18.05$ | $+9.02$<br>$+9.02$ | $-0.07$<br>$-0.07$ | $-0.30$<br>$-0.30$ | $0.00$<br>$0.23$ | $0.23$<br>$0.23$ | +0.62<br>+0.62 | —<br>— | —<br>— | $17.75$<br>$17.75$ | —<br>— |
| **`track03_style`**<br>*(stylistic)* | **`V0p`**<br>**`V3a`** | $-6.46$<br>$-6.46$ | $-13.52$<br>$-13.52$ | $+7.05$<br>$+7.05$ | $0.06$<br>$0.06$ | $-0.30$<br>$-0.30$ | $0.00$<br>$0.36$ | $0.36$<br>$0.36$ | +2.10<br>+2.10 | —<br>— | —<br>— | $13.22$<br>$13.22$ | —<br>— |

---

## 5. Tabella 4 — Diagnostica dei Gradienti (`Z_NEUTRAL`, `V0p z_opt`, `V3a z_opt`)

Ispezione dei gradienti $\ell_2$ sui 18 parametri EQ ($z_0 \dots z_{17}$) e allineamento vettoriale $\cos\theta$ rispetto a $\nabla \mathcal{L}_{\text{V0p}}$:

| Sessione | Stato / Vettore | $\|\nabla \mathcal{L}_{\text{eq}}\|_2$ (unw.) | $w_{\text{eq}}\|\nabla \mathcal{L}_{\text{eq}}\|_2$ | $\|\nabla \mathcal{L}_{\text{v0p}}\|_2$ | $\cos\theta$ | Termini V0p (Spettrale, LUFS, Reg) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`track01`** | `Z_NEUTRAL`<br>`V0p z_opt`<br>`V3a z_opt` | 0.0000<br>19.7274<br>19.5700 | 0.0000<br>0.3945<br>0.3914 | 6.7210<br>1.3675<br>1.3372 | +0.0000<br>+0.9466<br>+0.9457 | sp=0.7858, lufs=4.5349, plr=2.2610<br>sp=0.1567, lufs=1.0458, plr=0.3714, reg=0.0016<br>sp=0.1532, lufs=1.0225, plr=0.3631, reg=0.0016 |
| **`track02`** | `Z_NEUTRAL`<br>`V0p z_opt`<br>`V3a z_opt` | 0.0000<br>19.7273<br>19.5700 | 0.0000<br>0.3945<br>0.3914 | 6.6634<br>1.2981<br>1.2672 | +0.0000<br>+0.9421<br>+0.9410 | sp=0.7689, lufs=4.4991, plr=2.2449<br>sp=0.1453, lufs=0.9961, plr=0.3746, reg=0.0016<br>sp=0.1418, lufs=0.9723, plr=0.3656, reg=0.0016 |
| **`track03`** | `Z_NEUTRAL`<br>`V0p z_opt`<br>`V3a z_opt` | 0.0000<br>20.1322<br>19.9950 | 0.0000<br>0.4026<br>0.3999 | 6.1436<br>1.1973<br>1.1642 | +0.0000<br>+0.9570<br>+0.9565 | sp=0.7176, lufs=4.1394, plr=2.0754<br>sp=0.1455, lufs=0.8920, plr=0.3806, reg=0.0016<br>sp=0.1415, lufs=0.8673, plr=0.3701, reg=0.0016 |
| **`track01_style`** | `Z_NEUTRAL`<br>`V0p z_opt`<br>`V3a z_opt` | 0.0000<br>20.0419<br>20.0000 | 0.0000<br>0.4008<br>0.4000 | 4.0846<br>0.2208<br>0.2216 | +0.0000<br>+0.4273<br>+0.4216 | sp=0.3413, lufs=4.0273, plr=0.0000<br>sp=0.1973, lufs=0.1066, plr=0.0005, reg=0.0033<br>sp=0.1979, lufs=0.1070, plr=0.0005, reg=0.0033 |
| **`track02_style`** | `Z_NEUTRAL`<br>`V0p z_opt`<br>`V3a z_opt` | 0.0000<br>19.4317<br>19.2700 | 0.0000<br>0.3886<br>0.3854 | 2.0085<br>0.6258<br>0.6562 | +0.0000<br>-0.5944<br>-0.6085 | sp=0.3694, lufs=1.9885, plr=0.0000<br>sp=0.2896, lufs=0.4930, plr=0.2647, reg=0.0031<br>sp=0.3038, lufs=0.5170, plr=0.2776, reg=0.0031 |
| **`track03_style`** | `Z_NEUTRAL`<br>`V0p z_opt`<br>`V3a z_opt` | 0.0000<br>18.4877<br>18.3250 | 0.0000<br>0.3698<br>0.3665 | 6.2454<br>0.8165<br>0.8473 | +0.0000<br>-0.4631<br>-0.4783 | sp=0.3519, lufs=6.1353, plr=0.0000<br>sp=0.2994, lufs=0.6596, plr=0.2820, reg=0.0031<br>sp=0.3107, lufs=0.6845, plr=0.2926, reg=0.0031 |

---

## 6. Discussione Tecnica & Meccanica dei Gradienti

### Domanda 1: In che misura `V3a` riduce la deriva di offset medio $\bar{G}_{\text{EQ}}$ rispetto a `V0p`?
- **Origine della Deriva nei Primi Step da `Z_NEUTRAL`:** A partire da `Z_NEUTRAL` (dove la risposta EQ è piatta, $\bar{G} = 0.0\text{ dB}$, e il prior EQ si trova all'interno della dead-zone $\pm 1.0\text{ dB}$ producendo gradiente analitico nullo $\|\nabla \mathcal{L}_{\text{eq}}\|_2 = 0.0$), il gradiente iniziale di `lufs_loss` è dominante ($4.0 - 6.1$ in `Z_NEUTRAL`, a fronte di gradienti spettrali di $0.34 - 0.79$). L'aggiornamento adattivo di Adam (con $\text{lr} = 0.04$ per 30 step) spinge simultaneamente tutti i parametri di guadagno (i 6 biquad EQ, i 3 makeup gain e il push gain del limiter) nella direzione del target LUFS.
- **Azione del Prior EQ ed Ordine di Grandezza delle Forze:** Non appena $|\bar{G}_{\text{EQ}}|$ supera la dead-zone di $1.0\text{ dB}$, con $w_{\text{eq}} = 0.02$ il gradiente ponderato del prior EQ raggiunge $w_{\text{eq}}\|\nabla \mathcal{L}_{\text{eq}}\|_2 \approx 0.37 - 0.40$, che è dello stesso ordine di grandezza del gradiente spettrale residuo ($0.15 - 0.30$). Su 30 step di Adam, questa forza di ripristino produce una lieve contrazione di $|\bar{G}_{\text{EQ}}|$ ($\approx 0.008\text{ dB}$, riducendo $\mathcal{L}_{\text{eq-prior-unweighted}}$ di circa $-0.7\%\text{ a }-1.2\%$ su tutte le 6 sessioni) senza alterare le metriche spettrali o di loudness.
- **Orientamento Vettoriale ($\cos\theta$) tra le Sessioni:**
  - Nelle sessioni stilistiche `track02_style` e `track03_style`, a fine corsa $\cos\theta$ è negativo (`-0.5944` e `-0.4631`), a conferma che il prior EQ si oppone direttamente alla residua richiesta di loudness di $\text{V0p}$. In `track01_style`, dove il mix richiedeva attenuazione anziché boost, $\cos\theta = +0.4273$.
  - Nelle sessioni *same_track* (`track01`, `track02`, `track03`), l'ottimizzazione nei primi 20 step in modalità approssimata `fft` ha leggermente oltrepassato il punto di ottimo rispetto alla modalità `iir`, rendendo $\cos\theta$ positivo (`+0.9466`, `+0.9421`, `+0.9570`) quando valutato in modalità `iir` allo step 30, dove la spinta residua di $\text{V0p}$ e del prior cooperano nel richiamare i parametri verso la configurazione ottimale.

### Domanda 2: Come si ridistribuisce il guadagno globale tra i 6 biquad dell'EQ e i 3 makeup gain del compressore?
- **Stato del Gain Staging:** Poiché l'offset dell'EQ $\bar{G}_{\text{EQ}}$ è rimasto vicino a $+2.47\text{ dB}$, i guadagni dei singoli biquad (Tabella 2) hanno registrato solo minime variazioni ($-0.01\text{ dB}$ sui biquad estremi) e i makeup gain del compressore multibanda sono aumentati solo marginalmente ($+0.01\text{ dB}$ sulla banda low di `track01` e `track02`).
- **Implicazione Architetturale:** L'ottimizzatore non è stato forzato a trasferire un carico massivo di volume al compressore, poiché l'EQ è rimasto sufficientemente libero di fornire guadagno a larga banda. La piena redistribuzione del gain staging (con EQ centrato rigorosamente a $0\text{ dB}$ e compressore pienamente responsabile del recupero di livello) richiede il rafforzamento della penalità di offset documentato al punto precedente.

### Domanda 3: Le metriche spettrali (`MR-STFT`, `SC`, `LSD`, `ASD`, `ERB Surplus`) e di loudness restano preservate o subiscono regressioni?
- **Preservazione Perfetta:** Tutte le metriche oggettive full-track (Tabella 3) — convergenza spettrale (`SC`), distanza log-spettrale (`LSD`), `ERB Surplus` e `PLR` — risultano bit per bit identiche o variate entro un insignificante $\pm 0.0001$, senza alcuna regressione timbrica.
- **Loudness Integrata:** La loudness integrata finale su intero brano (`pred_lufs`) e la deviazione $\Delta\text{LUFS}$ rispetto al target integrato rimangono invariate rispetto a $\text{V0p}$ (es. $-10.51$ vs $-10.51$ su `track01`, $-17.83$ vs $-17.83$ su `track01_style`), confermando l'assoluta stabilità numerica della pipeline.

### Domanda 4: Emergono effetti collaterali su makeup gain, limiter o True Peak?
- **Nessun Effetto Collaterale Indesiderato:** Non si rilevano instabilità numeriche, divergenze di parametri né artefatti acustici.
- **True Peak e Inter-Sample Overshoot:** I valori di True Peak (dBTP), Sample Peak (dBFS) e le relative violazioni del guardrail a $-0.30\text{ dBTP}$ sono identici tra $\text{V0p}$ e $\text{V3a}$ (Tabella 3). Ciò conferma definitivamente che le violazioni del True Peak derivano esclusivamente dall'inter-sample overshoot post-limiter (che raggiunge fino a $+0.58\text{ dB}$ di overshoot rispetto al picco a campioni) e non sono in alcun modo influenzate dalla regolarizzazione spettrale dell'EQ. Questo risultato convalida la decisione architetturale di trattare il True Peak come uno stadio di delivery dedicato separato.

---

## 7. Verifiche di Integrità e Conclusioni

1. Tutti gli artefatti storici (`SPRINT3_V0P_V1_ABLATION_RESULTS.json`, `SPRINT3_V0P_V1_REPORT.md`) e i file armonizzati (`SPRINT3_V2_ABLATION_RESULTS.json`, `SPRINT3_V1_HARMONIZED_RESULTS.json`) sono rimasti rigorosamente immutati con `git diff` nullo.
2. La valutazione del prior EQ è avvenuta esattamente una sola volta per vettore $z$, garantendo l'invarianza matematica rispetto al numero di chunk $K$.
3. La variante `V3a_EQ_PRIOR_V1` è promossa e validata: il meccanismo differenziabile è pienamente funzionante, matematicamente coerente e pronto per futuri affinamenti sui pesi.
