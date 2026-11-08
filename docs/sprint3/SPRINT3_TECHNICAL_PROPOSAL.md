# AutoMastering — Valutazione Tecnica e Piano Sperimentale per lo Sprint 3

**Candidato:** Giovanni Antonecchia  
**Corso di Laurea:** Master's Thesis in Sound and Music Engineering — Politecnico di Milano (DEIB)  
**Baseline di Riferimento (Risultati Sprint 1–2 riconciliati):** `0fd86b4c4bbfde8d8bb364b38172040c6ebdf7a7`  
**Roadmap Approvata (Invariata):** `docs/AutoMastering_Audit_Roadmap_Antigravity.md` (§10, Sprint 3)  
**Vincolo Architetturale:** Architettura DDSP white-box a 41 parametri invariata; nessuna modifica al codice DSP/optimizer applicata in questo documento; nessuna riscrittura della roadmap approvata.

---

## 1. Matrice di Associazione: Proposte vs 5 Task Approvati dello Sprint 3

La tabella seguente associa ciascuna proposta tecnica ai cinque task dello Sprint 3 stabiliti nella roadmap (`Task 3.1 – Task 3.5`), specificando se è **già prevista**, se è una **raffinazione interna** a un task previsto, o se è un **esperimento diagnostico / requisito di delivery separato**, alla luce dei risultati esplorativi single-listener ($N=1$, 24 sessioni Sprint 2) riconciliati nel commit `0fd86b4`:
- **`level_matched` (12 sessioni):** `Mean8` Master `4.72/5` (`453/96`) vs Originale `4.85/5` (`466/96`); `overall_preference_1_5` (`OP`) media Master `4.42/5` (`53/12`) vs Originale `4.75/5` (`57/12`); conteggio con `OP` pura: **2 vittorie Master, 5 Originale, 5 pareggi** (`7/12` sessioni con Master $\ge$ Originale).
- **`program_level` (12 sessioni):** `Mean8` Master `4.53/5` (`435/96`) vs Originale `4.67/5` (`434/93`); `OP` media Master `3.83/5` (`46/12`) vs Originale `4.42/5` (`53/12`); conteggio con `OP` pura: **3 vittorie Master, 7 Originale, 2 pareggi** (`5/12` sessioni con Master $\ge$ Originale).

| Proposta / Osservazione | Task Sprint 3 di Riferimento | Classificazione | Motivazione Tecnica ed Evidenza Empirica (Sprint 1–2) |
|---|---|---|---|
| **A.1 Rappresentazione spettrale ERB / Mel multi-statistica + conservazione MR-STFT** | **Task 3.1** (*ERB o Mel spectral statistics*) | **Già previsto** (Core Task 3.1) | In `python/ddsp_mastering/loss/combined.py` la modalità `same_track` usa solo MR-STFT lineare; in `python/ddsp_mastering/loss/stylistic.py` è presente solo la media temporale `n_mels=40` a singola risoluzione (`n_fft=2048`). Serve un banco ERB unificato con statistiche temporali ($\mu, \sigma, P_{90}-P_{10}$) affiancato alla MR-STFT in `same_track`. |
| **A.2 Normalizzazione dei termini e diagnostica delle norme dei gradienti per banda/gruppo** | **Task 3.1 & Task 3.4** (*Metric & gradient diagnostics*) | **Già previsto** (Roadmap §3.1, §3.5) | Non assumiamo che il conteggio dei bin FFT determini da solo la dominanza degli acuti: va misurata empiricamente la norma $\|\nabla_{u_g} \mathcal{L}_b\|_2$ per banda $b$ e gruppo di parametri $g$ su `track01`, `track02`, `track03`. |
| **B.1 Correzione PLR reference-aware (`same_track`) e prior/quantili dinamici (`stylistic`)** | **Task 3.2 & Task 3.4** (*Transient/envelope loss* & *Dinamica*) | **Già previsto** (Roadmap §1.3, §3.3) | In `python/ddsp_mastering/loss/loudness.py` `target_plr=10.0` è ancora **inutilizzato**; `same_track` applica solo il box `[8, 14] dB`, permettendo al PLR di `track01` di scendere da `18.12 dB` a `10.21 dB` senza alcuna penalità. |
| **B.2 Transient / Envelope Loss e distinzione tra compressione musicale e limiting stress** | **Task 3.2 & Task 3.3** (*Transient/envelope loss* & *Prior fisico-acustici*) | **Raffinazione interna** ai Task 3.2 e 3.3 | Su `track02` (`sec01/sec02`) la compressione multibanda ha migliorato il punch di kick/snare (`5/5`), mentre su `track01` e `track03` a `program_level` la spinta sul limiter ha schiacciato i transienti. Va separata la preservazione dell'inviluppo veloce dalla penalità di *limiter stress* ($A_{\text{lim}}[n]$). |
| **C. Regolarizzazione della risposta complessiva EQ $|H_{\text{EQ}}(f)|$ e disaccoppiamento del guadagno globale** | **Task 3.3** (*Prior fisico-acustici*) | **Raffinazione interna** al Task 3.3 | In `combined.py` e `stylistic.py` il prior è ancora isotropo nello spazio latente: `mean((z - Z_NEUTRAL)**2)`. Regolarizzare la curva composita $20\log_{10}|H_{\text{EQ}}(e^{j\omega_k})|$ su griglia logaritmica evita la fallacia della media dei 6 gain e disaccoppia tonalità (EQ a media zero in dB) e loudness (makeup/limiter). |
| **D. Penalità asimmetrica e relativa alla reference per l'eccesso in banda $2.5\text{–}5\text{ kHz}$ (De-Harshness)** | **Task 3.1** (*ERB/Mel spectral statistics*) | **Raffinazione opzionale** dentro Task 3.1 (da validare in ablation) | Le note cieche segnalano *harshness* in `track01 same_track` e `track03 program_level`, ma `track02` non presenta il problema. Va formulata come termine **asimmetrico e relativo alla reference** (attivo solo se il tilt tra $2.5\text{–}5\text{ kHz}$ e mid-band supera quello della reference), mai come attenuazione fissa. |
| **E. Analisi di sensibilità all'Input Gain Staging / Headroom in ingresso** | **Protocollo diagnostico in Task 3.5** | **Esperimento diagnostico opzionale** (nessun target universale) | Non si introduce alcun target universale a `-20/-22 LUFS`. Si esegue solo un test diagnostico controllato (con riproduzione/compensazione esatta del trim nel render C++) per verificare come l'headroom d'ingresso sposti il punto di lavoro dei processori non lineari. |
| **F. Conformità del True Peak sull'export C++ finale (`+0.28 dBTP`, `+0.02 dBTP`, `+0.05 dBTP`)** | **Requisito di Delivery / Task 3.4** (*Metric diagnostics & export gate*) | **Guardrail di verifica/export** (separato dalla loss percettiva) | `limiter.py` e `LookaheadLimiter.hpp` limitano il *sample peak* a `-0.30 dBFS`, ma le oscillazioni inter-sample superano `0.0 dBTP` in 3 export su 6. Va gestito come requisito di delivery e verifica sul WAV C++ finale, non confuso con la funzione obiettivo percettiva. |

---

## 2. Audit del Codice Attuale: Problemi Già Coperti vs Realmente Mancanti

| Modulo / File | Cosa è GIÀ implementato nel codice (`0fd86b4`) | Cosa è REALMENTE mancante per lo Sprint 3 |
|---|---|---|
| **Preset Neutro & Inizializzazione** (`python/ddsp_mastering/param_space.py`) | `THETA_NEUTRAL` e `Z_NEUTRAL` sono già definiti correttamente (EQ 0 dB, ratio 1:1, makeup 0 dB, imager 1.0, sat mix 0, limiter push 0 dB) e usati come init in `optimizer.py`. | `param_reg` in `combined.py` e `stylistic.py` usa ancora una L2 uniforme `mean((z - z_target)**2)` con peso unico (`0.01`/`0.02`), che non distingue tra spostare una frequenza $f_0$, spingere il limiter o aprire lo stereo imager, né valuta la risposta effettiva dell'EQ. |
| **Spettro in `same_track` vs `stylistic`** (`stft.py`, `stylistic.py`) | `same_track` ha `MultiResolutionSTFTLoss` (`512, 2048, 8192`) L/R + M/S; `stylistic` ha `AggregatePerceptualLoss` con 40 bande Mel (solo media temporale del log-spettro centrato). | 1) In `same_track` manca un termine percettivo ERB/Mel affiancato alla MR-STFT. <br> 2) In `stylistic` mancano statistiche temporali per banda (deviazione standard / range interdecile $P_{90}-P_{10}$) e una scala percettiva ad alta risoluzione alle basse frequenze (ERB). |
| **Loudness & PLR** (`python/ddsp_mastering/loss/loudness.py`) | K-weighting causale float64 (`DifferentiableKWeighting`) e misuratore standard gated ITU-R BS.1770-5 (`calculate_integrated_loudness_bs1770`). | 1) In `loudness.py` (riga 266), `target_plr = 10.0` è dichiarato ma **mai usato**. <br> 2) In `same_track`, `DifferentiableLoudnessLoss.forward` non riceve `target_audio` né `target_plr`, applicando solo la penalità a intervallo fisso `[8.0, 14.0] dB`. <br> 3) In `stylistic.py` (riga 140) la loss LUFS è quadratica `(pred - ref)**2` anziché Huber, dominando il gradiente quando $\Delta\text{LUFS}$ iniziale è elevato (`track02`, `track03`). |
| **Transienti, Dinamica & Stato Interno DSP** (`chain.py`, `compressor.py`, `limiter.py`) | Il compressore calcola internamente la gain reduction in dB `yg` (`compressor.py`, righe 79–93) e il limiter calcola `g_smooth` (`limiter.py`, riga 116). | 1) Manca completamente un modulo di **Transient / Envelope Loss** (`Task 3.2`). <br> 2) `DifferentiableMasteringChain.forward` restituisce solo `lim_out`, senza esporre opzionalmente i segnali interni diagnostici (`yg_low, yg_mid, yg_high`, `g_smooth_limiter`, `H_eq`) necessari per i prior fisici (`Task 3.3`) e la diagnostica (`Task 3.4`). |
| **Stereo & Imager** (`stylistic.py`, `metrics.py`) | Misura globale del rapporto energetico M/S sull'intera banda. | Manca il controllo del rapporto M/S **per macro-bande** (in particolare evitare widening sub-bass e iper-apertura su `imager_width_high`, che ha causato la nota *"stereo troppo aperto"* su `track01` e `track03`) e la metrica di **correlazione di fase L/R** e **mono fold-down loss**. |
| **True Peak vs Sample Peak** (`metrics.py`, `limiter.py`) | `calculate_true_peak_dbtp` (4x Kaiser sinc FIR) è implementato nel benchmark `metrics.py`. | Il detector del limiter (`limiter.py` e `LookaheadLimiter.hpp`) opera a $1\times$ sample peak con `CEILING_DBFS = -0.3`, lasciando passare fino a `+0.58 dB` di inter-sample overshoot (`+0.28 dBTP` su `track01 same_track`). |

---

## 3. Formulazioni Matematiche delle Nuove Loss e dei Prior (Task 3.1 – 3.4)

### A. Task 3.1 — Rappresentazione Spettrale ERB vs Mel, Statistiche Temporali e Opzione Asimmetrica De-Harshness

#### 1. Scelta tecnica: Scala ERB (*Equivalent Rectangular Bandwidth*, Glasberg & Moore 1990) rispetto a Mel standard
- **Motivazione:** La scala Mel (`torchaudio.functional.melscale_fbanks`) è lineare fino a $1\text{ kHz}$ (con spaziatura $\sim 66\text{ Hz}$ per 40 bande), risultando troppo grossolana tra $30\text{ Hz}$ e $250\text{ Hz}$ dove operano la banda Low-Shelf (`eq_b0`, $20\text{–}250\text{ Hz}$), la prima Peaking (`eq_b1`, $100\text{–}500\text{ Hz}$) e il crossover basso (`crossover_f_low`, $80\text{–}350\text{ Hz}$).
- La scala $\text{ERB}_N(f)$ modella la larghezza di banda rettangolare equivalente dei filtri uditivi cocleari lungo l'intero spettro:
  $$\text{ERB}(f) = 24.7 \left(4.37 \frac{f}{1000} + 1\right)\text{ Hz}, \qquad E(f) = 21.4 \log_{10}(0.00437 f + 1)$$
  Costruendo un banco di $B = 48$ filtri triangolari (o $\cos^2$ ad area unitaria) uniformemente spaziati in coordinate $E(f)$ tra $20\text{ Hz}$ e $20\text{ kHz}$ su $N_{\text{fft}} = 4096$ ($\Delta f = 11.72\text{ Hz}$ a $48\text{ kHz}$), otteniamo una risoluzione sub/low-mid nettamente superiore senza perdere la compressione percettiva alle alte frequenze.

#### 2. Log-Energy e Statistiche Temporali per Banda
Sia $E_{b, t}(x) = \sum_{k} W_{b,k} \cdot \frac{1}{2}\left(|X_L(k,t)|^2 + |X_R(k,t)|^2\right)$ l'energia nella banda ERB $b \in \{1,\dots,B\}$ al frame $t$, e sia:
$$L_{b,t}(x) = 10 \log_{10}(E_{b,t}(x) + \varepsilon)$$
- **In modalità `same_track` (affiancata alla MR-STFT esistente, non in sostituzione):**
  Manteniamo $\mathcal{L}_{\text{MR-STFT}}$ per la coerenza fine multi-risoluzione e aggiungiamo il termine di inviluppo spettrale ERB:
  $$\mathcal{L}_{\text{ERB}}^{\text{same}}(y, y_{\text{ref}}) = \frac{1}{B \cdot T} \sum_{b=1}^{B} \sum_{t=1}^{T} \text{Huber}_\delta\!\left(L_{b,t}(y) - L_{b,t}(y_{\text{ref}})\right)$$
- **In modalità `stylistic` (statistiche distribuzionali invarianti al timing):**
  Per isolare il profilo timbrico dal livello assoluto, definiamo lo spettro centrato $\tilde{L}_{b,t}(x) = L_{b,t}(x) - \frac{1}{B}\sum_{b'=1}^B \mu_{b'}(x)$, dove $\mu_b(x) = \text{mean}_t(L_{b,t}(x))$, $\sigma_b(x) = \text{std}_t(L_{b,t}(x))$ e $\Delta P_b(x) = P_{90,t}(L_{b,t}(x)) - P_{10,t}(L_{b,t}(x))$ (escursione dinamica intra-banda calcolata sui frame attivi sopra il gate relativo di $-20\text{ dB}$):
  $$\mathcal{L}_{\text{ERB}}^{\text{style}}(y, y_{\text{ref}}) = \frac{1}{B}\sum_{b=1}^{B} \Big[ \big|\tilde{\mu}_b(y) - \tilde{\mu}_b(y_{\text{ref}})\big| + \alpha_\sigma \big|\sigma_b(y) - \sigma_b(y_{\text{ref}})\big| + \alpha_P \big|\Delta P_b(y) - \Delta P_b(y_{\text{ref}})\big| \Big]$$
  *(Assunzione: confrontare $\sigma_b$ e $P_{90}-P_{10}$ per banda impedisce all'ottimizzatore di raggiungere la stessa energia media schiacciando la dinamica di una banda con il compressore).*

#### 3. Opzione D — Penalità Asimmetrica e Relativa di De-Harshness ($2.5\text{–}5\text{ kHz}$)
Non applichiamo mai un peso statico tipo ISO 226 né penalizziamo l'energia assoluta tra $2.5$ e $5\text{ kHz}$. Definiamo invece il **rapporto di prominenza locale (Presence-to-Mid Tilt)** rispetto alla banda vocale/fondamentale di riferimento $\mathcal{B}_{\text{mid}} = [400\text{ Hz}, 2000\text{ Hz}]$:
$$\tau_{\text{harsh}}(x) = \left(\frac{1}{|\mathcal{B}_{\text{harsh}}|}\sum_{b \in [2.5, 5\text{ kHz}]} \mu_b(x)\right) - \left(\frac{1}{|\mathcal{B}_{\text{mid}}|}\sum_{b \in [0.4, 2\text{ kHz}]} \mu_b(x)\right)$$
La penalità è **unilaterale (hinge) e relativa alla reference** con margine di tolleranza $m_{\text{tol}} = 0.75\text{ dB}$:
$$\mathcal{L}_{\text{de-harsh}}(y, y_{\text{ref}}) = \left[\max\!\Big(0,\; \big(\tau_{\text{harsh}}(y) - \tau_{\text{harsh}}(y_{\text{ref}})\big) - m_{\text{tol}}\Big)\right]^2$$
- **Calibrazione e protezione dei mix morbidi:** Se il master predetto ha $\tau_{\text{harsh}}(y) \le \tau_{\text{harsh}}(y_{\text{ref}}) + m_{\text{tol}}$, il termine vale **esattamente zero** e ha gradiente nullo. Interviene solo quando la catena (ad es. tramite saturatore + limiter + `eq_b3/b4`) gonfia la regione $2.5\text{–}5\text{ kHz}$ *oltre* il bilanciamento della reference.

#### 4. Diagnostica e Normalizzazione dei Gradienti
Ad ogni esperimento, prima di fissare i pesi $\lambda_i$, misuriamo al passo $t=0$ ($z = Z_{\text{NEUTRAL}}$) e lungo la traiettoria la norma del gradiente per ciascun termine $\mathcal{L}_i$ rispetto ai 6 gruppi fisici $\mathcal{G} = \{\text{EQ}, \text{Xover}, \text{Comp}, \text{Imager}, \text{Sat}, \text{Lim}\}$:
$$G_{i, g} = \left\| \nabla_{u_g} \mathcal{L}_i \right\|_2$$

---

### B. Task 3.2 — Transient / Envelope Loss e Dinamica Reference-Aware

#### 1. Inviluppi Multi-Scala e Derivata d'Attacco (Onset Flux)
Su 3 macro-bande fisse di valutazione ($\text{Low}: 20\text{–}160\text{ Hz}$, $\text{Mid}: 160\text{–}3200\text{ Hz}$, $\text{High}: 3.2\text{–}20\text{ kHz}$) calcoliamo l'inviluppo RMS in dB su finestra **veloce** ($\tau_{\text{fast}} = 5\text{ ms}$) e **lenta** ($\tau_{\text{slow}} = 50\text{ ms}$):
$$e_{c}^{\text{fast}}[m] = 10\log_{10}\!\left(\text{AvgPool}_{\tau_{\text{fast}}}(x_c^2)[m] + \varepsilon\right), \qquad e_{c}^{\text{slow}}[m] = 10\log_{10}\!\left(\text{AvgPool}_{\tau_{\text{slow}}}(x_c^2)[m] + \varepsilon\right)$$
Definiamo:
- **Crest Factor Locale (Micro-Dinamica / Punch):** $C_c[m] = e_c^{\text{fast}}[m] - e_c^{\text{slow}}[m]$ (misura la sporgenza dei transienti rispetto al letto medio locale, indipendente dal guadagno statico di banda).
- **Flusso d'Attacco Positivo (Onset Slope):** $\Delta^+ e_c^{\text{fast}}[m] = \text{ReLU}\!\left(e_c^{\text{fast}}[m] - e_c^{\text{fast}}[m-1]\right)$.

#### 2. Distinzione rigorosa tra `same_track` e `stylistic`
- **Modalità `same_track` (allineata temporalmente):** confrontiamo punto-a-punto il profilo di micro-dinamica $C_c[m]$ e le pendenze d'attacco $\Delta^+ e_c^{\text{fast}}[m]$:
  $$\mathcal{L}_{\text{trans}}^{\text{same}}(y, y_{\text{ref}}) = \frac{1}{3}\sum_{c \in \{L,M,H\}} \Big[ \text{Huber}_1\!\big(C_c(y) - C_c(y_{\text{ref}})\big) + \beta_{\text{atk}} \text{Huber}_1\!\big(\Delta^+ e_c^{\text{fast}}(y) - \Delta^+ e_c^{\text{fast}}(y_{\text{ref}})\big) \Big]$$
- **Modalità `stylistic` (brani diversi, nessun confronto frame-by-frame):** confrontiamo esclusivamente i **quantili ($P_{50}, P_{90}, P_{95}$) della distribuzione** di $C_c[m]$ e di $\Delta^+ e_c^{\text{fast}}[m]$ sui chunk:
  $$\mathcal{L}_{\text{trans}}^{\text{style}}(y, y_{\text{ref}}) = \frac{1}{3}\sum_{c \in \{L,M,H\}} \sum_{q \in \{50, 90, 95\}} \Big| P_q\big(C_c(y)\big) - P_q\big(C_c(y_{\text{ref}})\big) \Big|$$

#### 3. Correzione della Loss PLR in `loudness.py`
Usiamo la medesima definizione $\text{PLR}_{\text{proxy}}(x) = \text{Peak}_{\text{dBFS}}(x) - \text{LUFS}_{\text{K-ungated}}(x)$ per entrambi i segnali:
- **In `same_track`:** sostituiamo il box cieco `[8, 14]` con il matching diretto del PLR della reference:
  $$\mathcal{L}_{\text{PLR}}^{\text{same}}(y, y_{\text{ref}}) = \text{Huber}_{1.0}\!\left(\text{PLR}_{\text{proxy}}(y) - \text{PLR}_{\text{proxy}}(y_{\text{ref}})\right)$$
- **In `stylistic`:** usiamo Huber rispetto a $\text{PLR}_{\text{proxy}}(y_{\text{ref}})$ con una zona morta (dead-zone) di $\pm 1.0\text{ dB}$ per accomodare le differenze di arrangiamento tra brani diversi, mantenendo una barriera inferiore morbida contro l'iper-compressione ($\text{PLR} < \min(8.0, \text{PLR}_{\text{ref}} - 1.5)$).

---

### C. Task 3.3 — Prior Fisico-Acustici: Risposta EQ su Griglia Log-Frequenza, Disaccoppiamento Gain e Limiter Stress

#### 1. Risposta Complessiva dell'EQ su Griglia Logaritmica e Disaccoppiamento dal Guadagno Globale
Poiché i 6 filtri biquad dell'EQ si sovrappongono in frequenza e hanno larghezze di banda dipendenti da $Q_b$, valutiamo la risposta complessa analitica della cascata a 6 stadi su una griglia logaritmica fissa di $K = 128$ frequenze $\omega_k = 2\pi f_k / f_s$ con $f_k \in [20\text{ Hz}, 20\text{ kHz}]$:
$$G_{\text{EQ}}(f_k; \theta_{\text{EQ}}) = 20 \log_{10} \left| \prod_{b=0}^{5} H_b(e^{j\omega_k}; f_{0,b}, g_b, Q_b) \right| \quad [\text{dB}]$$
Scomponiamo $G_{\text{EQ}}(f_k)$ nel suo **offset medio (guadagno broadband)** $\bar{G}_{\text{EQ}} = \frac{1}{K}\sum_{k=1}^K G_{\text{EQ}}(f_k)$ e nella sua **curva di forma (tonal shaping)** $\tilde{G}_{\text{EQ}}(f_k) = G_{\text{EQ}}(f_k) - \bar{G}_{\text{EQ}}$:
$$\mathcal{L}_{\text{prior-EQ}}(\theta_{\text{EQ}}) = \underbrace{w_{\text{dc}} \left(\bar{G}_{\text{EQ}}\right)^2}_{\substack{\text{1. Disaccoppiamento:} \\ \text{EQ a media 0 dB}}} + \underbrace{w_{\text{energy}} \frac{1}{K}\sum_{k=1}^{K} \left(\tilde{G}_{\text{EQ}}(f_k)\right)^2}_{\substack{\text{2. Moderazione d'intervento:} \\ \text{energia della deviazione spettrale}}} + \underbrace{w_{\text{smooth}} \frac{1}{K-1}\sum_{k=1}^{K-1} \left(\tilde{G}_{\text{EQ}}(f_{k+1}) - \tilde{G}_{\text{EQ}}(f_k)\right)^2}_{\substack{\text{3. Regolarità spettrale:} \\ \text{penalizza picchi ad alto } Q \text{ e cancellazioni tra bande}}}$$

#### 2. Distinzione tra Compressione Musicale Multibanda e Limiter Stress
Esponendo opzionalmente dal forward di `DifferentiableMasteringChain` un dizionario `aux_signals` con le curve di gain reduction in dB delle 3 bande del compressore $y_{g, c}[n] \ge 0$ e l'attenuazione in dB del limiter $A_{\text{lim}}[n] = -20\log_{10}(g_{\text{smooth}}[n]) \ge 0$:
1. **Penalità di Gain Reduction Differenziale tra Bande Adiacenti del Compressore:**
   $$\mathcal{L}_{\text{prior-comp-diff}} = \frac{1}{N}\sum_{n=1}^{N} \Big[ \text{ReLU}\!\big(|y_{g,\text{low}}[n] - y_{g,\text{mid}}[n]| - \Delta_{\text{max}}\big)^2 + \text{ReLU}\!\big(|y_{g,\text{mid}}[n] - y_{g,\text{high}}[n]| - \Delta_{\text{max}}\big)^2 \Big], \quad \Delta_{\text{max}} = 4.0\text{ dB}$$
2. **Penalità di Limiter Stress (distinta dalla compressione):**
   Penalizziamo l'**attenuazione media** ($\bar{A}_{\text{lim}}$, che indica che il limiter è usato come compressore statico anziché come guardia sui picchi) e il **95° percentile** ($P_{95}(A_{\text{lim}})$) oltre una soglia morbida ($\tau_{\text{avg}} = 1.5\text{ dB}$, $\tau_{95} = 3.5\text{ dB}$):
   $$\mathcal{L}_{\text{prior-lim}} = \text{ReLU}\!\big(\bar{A}_{\text{lim}} - \tau_{\text{avg}}\big)^2 + 0.5\,\text{ReLU}\!\big(P_{95}(A_{\text{lim}}) - \tau_{95}\big)^2$$
3. **Prior di Redundancy Gain-Staging & Imager:**
   Penalizza soluzioni che usano makeup positivo simultaneo sulle 3 bande mentre il limiter sta già attenuando, e applica un prior quadratico morbido centrato su $w = [1.0, 1.0, 1.0]$ con peso maggiore su `imager_width_low` (protezione mono del basso) e `imager_width_high` (prevenzione dell'effetto *"stereo troppo aperto"* osservato su `track01` e `track03`).

---

### D. Task 3.4 — Dinamica, Stereo Multi-Banda e Conformità True Peak all'Export

1. **Stereo Multi-Banda e Mono Compatibility:**
   - Misura (e regolarizzazione morbida) del rapporto energetico Side/Mid separatamente nelle 3 bande del crossover, del **Mono Fold-Down Loss** $\Delta_{\text{mono}}(x) = \text{LUFS}_{\text{K}}(x_{\text{stereo}}) - \text{LUFS}_{\text{K}}(x_{\text{mid}})$, e della correlazione di fase inter-canale $\rho_{LR}$.
2. **Requisito di Delivery sul True Peak dell'Export C++ (separato dalla loss percettiva):**
   - Nello Sprint 2, 3 export hanno mostrato True Peak $> 0.0\text{ dBTP}$ (`track01 same_track`: `+0.28 dBTP`, `track03 same_track`: `+0.02 dBTP`, `track03 stylistic`: `+0.05 dBTP`).
   - Trattiamo il ceiling di delivery ($\text{TP}_{\text{max}} \le -0.30\text{ dBTP}$ o valore configurabile di progetto) come **verifica obbligatoria sul WAV C++ finale** (`mastered_full_np`), con applicazione opzionale di un *post-limiter True-Peak safety trim* statico $\Delta_{\text{TP}} = \min(0,\; \text{ceiling}_{\text{delivery}} - \text{TP}_{\text{measured}})$ in fase di export finale, senza inquinare la loss percettiva dello Sprint 3.

---

## 4. Ordine di Implementazione e Piano di Ablation (Task 3.5)

Manteniamo **congelati e identici allo Sprint 2**: i 3 brani (`track01`, `track02`, `track03`), le reference (`same_track` e `stylistic`), i chunk e i seed a 64 bit registrati nei manifest dello Sprint 2, il numero di step (`40`), il learning rate (`0.04`) e la pipeline di render C++ finale.

| ID Variante | Configurazione | Task Sprint 3 | Ipotesi Specifica da Verificare |
|---|---|:---:|---|
| **`V0 (Baseline Sprint 2)`** | `MR-STFT` (`same_track`) / `Mel-40 mean` (`stylistic`) + `LUFS + PLR box [8,14]` + `L2(z - Z_NEUTRAL)` | — | Riferimento congelato (`0fd86b4`). |
| **`V1 (+ ERB Spectral Stats)`** | `V0` + banco `ERB-48` (affiancato a MR-STFT in `same_track`; con $\mu, \sigma, P_{90}-P_{10}$ in `stylistic`) | **Task 3.1** | Migliora il controllo di sub/low-mid (`track01 sec02`, `track03 sec01`) e stabilizza i gradienti spettrali tra basse e alte frequenze. |
| **`V1b (+ Asymmetric De-Harshness)`** | `V1` + penalità asimmetrica relativa $\mathcal{L}_{\text{de-harsh}}$ ($2.5\text{–}5\text{ kHz}$) | **Task 3.1 (opz.)** | Riduce l'eccesso di presence su `track01 same_track` senza scurire `track02` o `track01_style`. Se non migliora l'ascolto rispetto a `V1 + V3`, viene scartata. |
| **`V2 (+ Transient/Envelope & Ref-PLR)`** | `V0` + $\mathcal{L}_{\text{trans}}$ (crest locale + onset slope) + $\mathcal{L}_{\text{PLR}}$ reference-aware | **Task 3.2** | Impedisce il crollo del PLR su `track01 same_track` (`18.1 -> 10.2 dB`) e `track03`, preservando il punch di kick/snare su `track02`. |
| **`V3 (+ Acoustic Priors)`** | `V0` + $\mathcal{L}_{\text{prior-EQ}}(H(f))$ + $\mathcal{L}_{\text{prior-lim}}(A_{\text{lim}})$ + $\mathcal{L}_{\text{prior-comp-diff}}$ + prior M/S | **Task 3.3** | Disaccoppia EQ e loudness ($\bar{G}_{\text{EQ}} \approx 0\text{ dB}$), frena `limiter_push_gain_db` e impedisce l'apertura eccessiva di `imager_width_high`. |
| **`V4 (Combined Perceptual Loss)`** | Combinazione dei soli componenti `V1 + V2 + V3` (ed eventualmente `V1b`) validati positivamente singolarmente | **Task 3.1–3.5** | Verifica la sinergia complessiva sia a `level_matched` sia a `program_level`. |
| **`Diag-E (Input Headroom Sensitivity)`** | Esperimento diagnostico separato su `track01` (con e senza pre-attenuazione statica riprodotta nel render C++) | **Diagnostica** | Isola l'effetto del clipping in ingresso (`+3.98 dBTP` su `track01`) dall'effetto della nuova loss `V4`. |

---

## 5. Test di Correttezza Numerica, Parità PyTorch/C++ e Validazione Audio

1. **Invarianza e Identità delle Nuove Loss:** $\mathcal{L}_{\text{ERB}}(x, x) = 0$, $\mathcal{L}_{\text{trans}}(x, x) = 0$, $\mathcal{L}_{\text{de-harsh}}(x, x) = 0$; a $z = Z_{\text{NEUTRAL}}$, $\mathcal{L}_{\text{prior-EQ}}(\theta_{\text{neutral}}) = 0$, $\mathcal{L}_{\text{prior-comp-diff}} = 0$.
2. **Test di Asimmetria di $\mathcal{L}_{\text{de-harsh}}$:** Su un segnale $y$ attenuato tra $2.5$ e $5\text{ kHz}$ rispetto a $y_{\text{ref}}$, verifica che $\mathcal{L}_{\text{de-harsh}}(y, y_{\text{ref}}) == 0.0$ e $\nabla_\theta \mathcal{L}_{\text{de-harsh}} == \mathbf{0}$.
3. **Test del Prior EQ contro la Fallacia della Media Aritmetica:** Verifica che filtri con media dei 6 gain pari a zero ma risposta composita $G_{\text{EQ}}(f_k) \ne 0$ producano $\mathcal{L}_{\text{prior-EQ}} > 0$, e che un offset broadband $\bar{G}_{\text{EQ}} \ne 0$ sia penalizzato da $w_{\text{dc}} (\bar{G}_{\text{EQ}})^2$.
4. **Gradcheck e Finiteness:** `torch.autograd.gradcheck` in `float64` su tutti i nuovi termini e sui segnali ausiliari (`yg`, `g_smooth`).
5. **Parità PyTorch $\leftrightarrow$ C++ Invariata:** Verifica che l'esposizione opzionale `return_aux=True` in `DifferentiableMasteringChain` lasci **bit-identical** il segnale d'uscita rispetto a `py_core_dsp.MasteringChain`.
6. **Gate True Peak sul WAV C++ Finale:** Test automatico sul file C++ esportato (`calculate_true_peak_dbtp(cpp_wav) <= ceiling_dbtp`).

---

## 6. Analisi dei Rischi e Mitigazioni

1. **Conflitto tra Target LUFS e Limiter Stress Penalty ($\mathcal{L}_{\text{LUFS}}$ vs $\mathcal{L}_{\text{prior-lim}}$):**
   - *Rischio:* Per guadagnare LUFS senza spingere il limiter, l'ottimizzatore potrebbe gonfiare la regione $1.5\text{–}5\text{ kHz}$ sfruttando la curva K-weighting ($+4\text{ dB}$ high-shelf).
   - *Mitigazione:* Uso di Huber loss su LUFS (anche in `stylistic`), vincolo $\bar{G}_{\text{EQ}} \approx 0\text{ dB}$ sull'EQ, e prior asimmetrico relativo $\mathcal{L}_{\text{de-harsh}}$.
2. **Over-Smoothing (Mix Spento) da Prior Troppo Forti:**
   - *Rischio:* Pesi eccessivi sui prior bloccano i parametri vicino a `THETA_NEUTRAL`, perdendo il guadagno di punch ottenuto su `track02`.
   - *Mitigazione:* Formulazione tramite zone di tolleranza (*hinge / dead-zone*) entro cui i prior hanno pendenza quasi nulla per interventi moderati.
3. **Overfitting Percettivo sui 3 Brani:**
   - *Rischio:* Calibrazione manuale degli iperparametri specifica per `track01–03`.
   - *Mitigazione:* Normalizzazione oggettiva tramite norme dei gradienti al passo iniziale ($t=0$) e selezione strettamente sequenziale tramite il protocollo di ablation `V0 -> V1 -> V2 -> V3 -> V4`.
