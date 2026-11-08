# AutoMastering — Specifica Tecnica e Piano Sperimentale per lo Sprint 3 (Revisione v2)

**Candidato:** Giovanni Antonecchia  
**Corso di Laurea:** Master's Thesis in Sound and Music Engineering — Politecnico di Milano (DEIB)  
**Baseline di Riferimento (Risultati Sprint 1–2 riconciliati):** `0fd86b4c4bbfde8d8bb364b38172040c6ebdf7a7`  
**Roadmap Approvata (Invariata):** `docs/AutoMastering_Audit_Roadmap_Antigravity.md` (§10, Sprint 3)  
**Vincolo Architetturale:** Architettura DDSP white-box a 41 parametri invariata; nessuna modifica al codice DSP/optimizer applicata in questo documento; nessuna riscrittura della roadmap approvata.

---

## 0. Verifica della Baseline `V0` e Riproducibilità Esatta dello Sprint 2

Dall'ispezione diretta dei 6 manifest ufficiali dello Sprint 2 (`human_validation/private/sprint2_same_track/track0{1,2,3}/track0{1,2,3}_manifest.json` e `human_validation/private/sprint2_stylistic/track0{1,2,3}_style/track0{1,2,3}_style_manifest.json`) e dei relativi file `*_sections.json` generati da `scripts/human_validation/render_sprint2.py`:

- **Budget effettivo utilizzato nello Sprint 2:** **`num_steps = 30`** (default di `render_sprint2.py`, riga 223 e 625; **non `40`**, che era rimasto solo nella firma storica di `scripts/run_mastering.py`).
- **Configurazione esatta congelata per `V0` (e per tutte le varianti `V1..V4`):**
  - `num_steps`: **`30`** (transizione hybrid EQ `fft -> iir` al passo `int(30 * 0.7) = 21`, ossia passi `0..20` in `fft` e passi `21..29` in `iir`).
  - `learning_rate`: **`0.04`** (Adam con `CosineAnnealingLR(T_max=30, eta_min=0.002)` e `clip_grad_norm_(max_norm=1.0)`).
  - `num_chunks`: **`4`**, `chunk_duration_sec`: **`3.0 s`** (`144000` campioni a `48000 Hz`), pesi uniformi `1/4 = 0.25`.
  - Pesi loss baseline in `render_sprint2.py` (riga 318–327): `weight_stft = 1.0`, `weight_loudness = 0.8`, `weight_param_reg = 0.01` (con `weight_dynamics = 0.5` e `weight_stereo = 0.3` interni ad `AggregatePerceptualLoss`).

### Tabella di Riproducibilità `V0` (Manifest Sprint 2 e Rivalutazione Esatta su `z_opt`)

Poiché in `AutoMasteringOptimizer.optimize_multi_chunk` `history["loss"][-1]` registra la loss composita al passo `29` prima dell'ultimo `optimizer.step()` che produce `z_opt`, la tabella seguente riporta sia il valore `step 29` salvato nel manifest sia il valore esatto della **loss composita totale** ricalcolata direttamente sul vettore finale restituito `z_opt` (`step 30`, valutato con `eq_mode="iir"`):

| Sessione (`prefix`) | Modalità | Seed (64-bit) | Training Chunks Mix (`start_sec – end_sec`) | Reference Chunks (`start_sec – end_sec`) | Loss Composita Iniziale (`z_0 = Z_NEUTRAL`) | Loss Composita `step 29` (`history[-1]`) | Loss Composita Finale su `z_opt` (`step 30`, `IIR`) |
|---|---|:---:|---|---|:---:|:---:|:---:|
| **`track01`** | `same_track` | `5405783196615117109` | `c0: 1.0–4.0s`, `c1: 30.0–33.0s`, `c2: 47.0–50.0s`, `c3: 54.0–57.0s` | Accoppiati agli stessi offset su `ref_aligned` (`lag = +152 smp`) | `6.6669` | `3.0966` | **`3.0841`** (`-53.7%`) |
| **`track02`** | `same_track` | `3048363069421082382` | `c0: 8.0–11.0s`, `c1: 29.0–32.0s`, `c2: 40.0–43.0s`, `c3: 59.0–62.0s` | Accoppiati agli stessi offset su `ref_aligned` | `14.2843` | `10.4312` | **`10.4173`** (`-27.1%`) |
| **`track03`** | `same_track` | `12990700764526553868` | `c0: 12.0–15.0s`, `c1: 25.5–28.5s`, `c2: 36.0–39.0s`, `c3: 65.0–68.0s` | Accoppiati agli stessi offset su `ref_aligned` | `10.4493` | `6.4696` | **`6.4545`** (`-38.2%`) |
| **`track01_style`** | `stylistic` | `14887125631601849244` | `c0: 1.0–4.0s`, `c1: 30.0–33.0s`, `c2: 47.0–50.0s`, `c3: 54.0–57.0s` | `c0: 59.0–62.0s`, `c1..c3` estratti da `track01_ref_altro_brano.wav` | `28.0920` | `8.5858` | **`8.5529`** (`-69.6%`) |
| **`track02_style`** | `stylistic` | `3606996967096283602` | `c0: 8.0–11.0s`, `c1: 29.0–32.0s`, `c2: 40.0–43.0s`, `c3: 59.0–62.0s` | `c0: 15.0–18.0s`, `c1..c3` estratti da `track02_ref_altro_brano.wav` | `159.9057` | `82.9988` | **`82.7613`** (`-48.2%`) |
| **`track03_style`** | `stylistic` | `15136152743686100324` | `c0: 12.0–15.0s`, `c1: 25.5–28.5s`, `c2: 36.0–39.0s`, `c3: 65.0–68.0s` | `c0: 49.5–52.5s`, `c1..c3` estratti da `track03_ref_altro_brano.wav` | `110.5956` | `46.7759` | **`46.5655`** (`-57.9%`) |

---

## 1. Matrice di Associazione: Proposte vs 5 Task Approvati dello Sprint 3

| Proposta / Osservazione | Task Sprint 3 di Riferimento | Classificazione & Variante Ablation | Motivazione Tecnica ed Evidenza Empirica (Sprint 1–2) |
|---|---|---|---|
| **A.1 Rappresentazione spettrale ERB-48 multi-statistica + conservazione MR-STFT in `same_track`** | **Task 3.1** (*ERB o Mel spectral statistics*) | **Già previsto** (`V1`) | Affianca un banco percettivo ERB a 48 bande alla MR-STFT in `same_track` e sostituisce la sola media Mel-40 in `stylistic` con statistiche temporali ($\mu, \sigma, P_{90}-P_{10}$) calcolate sui frame attivi. |
| **A.2 Normalizzazione dei termini e diagnostica delle norme dei gradienti per banda/gruppo** | **Task 3.1 & Task 3.4** (*Metric & gradient diagnostics*) | **Già previsto** (Tutte le varianti `V0..V4`) | Misura empiricamente $\|\nabla_{u_g} \mathcal{L}_i\|_2$ per ogni termine di loss $i$ e gruppo di parametri $g$ senza assumere a priori una dominanza basata sul solo conteggio dei bin FFT. |
| **D. Penalità asimmetrica e relativa alla reference per l'eccesso in banda $2.5\text{–}5\text{ kHz}$ (De-Harshness)** | **Task 3.1** (*ERB/Mel spectral statistics*) | **Raffinazione opzionale** (`V1b`) | Interviene solo quando il tilt $2.5\text{–}5\text{ kHz}$ vs $0.4\text{–}2\text{ kHz}$ supera quello della reference oltre una tolleranza $m_{\text{tol}}$. Confrontata direttamente con `V1` secondo criteri espliciti prima di eventuale inclusione in `V4`. |
| **B.1 Transient / Envelope Loss multi-banda ($\tau_{\text{fast}}, \tau_{\text{slow}}$, crest locale $C_c$ e onset slope $\Delta^+ e_c^{\text{fast}}$)** | **Task 3.2** (*Transient/envelope loss*) | **Già previsto** (`V2`) | Punto-a-punto in `same_track`; tramite quantili ($P_{50}, P_{90}, P_{95}$) sia di $C_c$ sia di $\Delta^+ e_c^{\text{fast}}$ in `stylistic`, con gestione esplicita di pre-roll/warm-up e selezione dei frame attivi ancorata all'input. |
| **B.2 Correzione PLR reference-aware con proxy di picco differenziabile ($\mathcal{M}_\beta$) distinta da Sample Peak e True Peak** | **Task 3.2 & Task 3.4** (*Transient/envelope & Dinamica*) | **Già previsto** (`V2`) | Sostituisce il box `[8, 14] dB` in `loudness.py` con matching Huber in `same_track` e dead-zone + floor dinamico in `stylistic`, usando la stessa definizione $\text{PLR}_{\text{proxy}}$ per predizione e reference. |
| **C. Prior sulla risposta composita dell'EQ $G_{\text{EQ}}(f_k)$ (forma spettrale, offset medio log-frequenza e gestione guadagno)** | **Task 3.3** (*Prior fisico-acustici*) | **Raffinazione interna** (`V3a`) | Isolato nella variante di ablation `V3a`: regolarizza la forma della risposta dei 6 biquad e limita derive sull'offset medio log-frequenza senza impedire il raggiungimento della loudness target. |
| **B.3 Prior dinamici sui processori: Limiter Stress, Gain Reduction differenziale tra bande adiacenti e consistenza makeup** | **Task 3.3** (*Prior fisico-acustici*) | **Raffinazione interna** (`V3b`) | Isolato nella variante di ablation `V3b`: distingue la compressione musicale multibanda dallo schiacciamento sul limiter ($\bar{A}_{\text{lim}}, P_{95}(A_{\text{lim}})$ sui frame attivi). |
| **S. Prior e diagnostica Stereo multi-banda, correlazione $\rho_{LR}$ e Mono Fold-Down Loss ($\Delta_{\text{mono}}$)** | **Task 3.3 & Task 3.4** (*Prior & Dinamica/Stereo*) | **Già previsto / Raffinazione** (`V3c`) | Isolato nella variante di ablation `V3c`: previene l'iper-apertura di `imager_width_high` e protegge la monofonia delle basse frequenze. |
| **E. Analisi di sensibilità all'Input Gain Staging / Headroom in ingresso** | **Diagnostica separata (Task 3.5)** | **Esperimento diagnostico separato** (`Diag-E`) | Nessun target universale a `-20/-22 LUFS`. Verifica controllata dell'effetto del trim statico d'ingresso (riprodotto/compensato fedelmente nel render C++). |
| **F. Guardrail di Delivery sul True Peak dell'export C++ finale** | **Requisito di Delivery / Task 3.4** | **Guardrail di verifica/export** (Tutte le varianti) | Policy configurabile di progetto verificata sul WAV C++ finale (con eventuale safety trim statico e rimisura obbligatoria post-trim), separata dalla loss percettiva. |

---

## 2. Audit del Codice Attuale: Problemi Già Coperti vs Realmente Mancanti

| Modulo / File | Cosa è GIÀ implementato nel codice (`0fd86b4`) | Cosa è REALMENTE mancante per lo Sprint 3 |
|---|---|---|
| **Preset Neutro & Inizializzazione** (`python/ddsp_mastering/param_space.py`) | `THETA_NEUTRAL` e `Z_NEUTRAL` definiti correttamente e usati come init in `optimizer.py`. | `param_reg` in `combined.py` e `stylistic.py` usa solo una L2 isotropa `mean((z - z_target)**2)` che non valuta né la risposta composita dell'EQ né lo stress dei processori dinamici. |
| **Spettro in `same_track` vs `stylistic`** (`stft.py`, `stylistic.py`) | `same_track` ha `MultiResolutionSTFTLoss` (`512, 2048, 8192`) L/R + M/S; `stylistic` ha `AggregatePerceptualLoss` con 40 bande Mel (solo media temporale su tutti i frame). | 1) In `same_track` manca il termine percettivo ERB affiancato alla MR-STFT. <br> 2) In `stylistic` mancano la scala ERB-48, la selezione/pesatura dei frame attivi e le statistiche temporali di dispersione ($\sigma_b$, $P_{90}-P_{10}$). |
| **Loudness & PLR** (`python/ddsp_mastering/loss/loudness.py`) | K-weighting causale float64 (`DifferentiableKWeighting`) e misuratore standard gated ITU-R BS.1770-5 (`calculate_integrated_loudness_bs1770`). | 1) `target_plr = 10.0` è dichiarato ma **mai usato**. <br> 2) In `same_track` la loss PLR usa solo il box fisso `[8.0, 14.0] dB` e non conosce il PLR della reference. <br> 3) `calculate_peak_dbfs` usa `torch.max` (hard peak), concentrando il gradiente su un singolo campione. |
| **Transienti, Pre-Roll & Segnali Interni DSP** (`chain.py`, `compressor.py`, `limiter.py`) | I calcoli interni di gain reduction `yg` (`compressor.py`) e `g_smooth` (`limiter.py`) sono già presenti nei moduli DSP. | 1) Manca il modulo di **Transient / Envelope Loss** (`Task 3.2`). <br> 2) Manca la gestione del warm-up / pre-roll sui chunk per evitare falsi transienti ai bordi dovuti allo stato zero iniziale dei filtri/inviluppi. <br> 3) `DifferentiableMasteringChain.forward` non espone ancora il dizionario opzionale `aux_signals`. |
| **Stereo & True Peak Delivery** (`stylistic.py`, `metrics.py`, `limiter.py`) | `calculate_true_peak_dbtp` (4x Kaiser sinc FIR) è presente in `metrics.py`; rapporto M/S globale presente in `stylistic.py`. | Mancano il controllo M/S per macro-banda, la correlazione $\rho_{LR}$, il Mono Fold-Down Loss $\Delta_{\text{mono}}$ e il guardrail sistematico sul True Peak del WAV C++ esportato. |

---

## 3. Formulazioni Matematiche Complete (Task 3.1 – 3.4)

### A. Gestione di Pre-Roll, Warm-Up dei Chunk e Selezione dei Frame Attivi (Comune a Task 3.1, 3.2, 3.3)

#### 1. Pre-Roll / Warm-Up e Bordi dei Chunk (Prevenzione di Transienti Artificiali da Reset dello Stato DSP)
Quando `DifferentiableMasteringChain` elabora un chunk $x_k$, i filtri IIR (EQ e crossover LR4), i 3 detector del compressore (con release fino a $600\text{ ms}$) e il limiter (lookahead $144$ campioni, release $80\text{ ms}$) partono da stato nullo ($0\text{ dB}$ di gain reduction). Ciò può generare uno spike transitorio artificiale nei primi millisecondi del chunk.
Per evitare che le loss spettrali, di inviluppo e di gain reduction vengano falsate dai bordi:
1. **Con Pre-Roll dal brano completo (modalità standard per `V1..V4` quando l'indice d'inizio del chunk lo consente):**
   Per ciascun chunk $[s_k, e_k]$, si estrae una finestra estesa all'indietro $[s_k - N_{\text{pre}}, e_k]$ con $N_{\text{pre}} = \min(s_k, \lfloor 0.250 \cdot f_s \rfloor)$ ($250\text{ ms} = 12000$ campioni). La catena DDSP elabora l'intero segmento esteso; prima di calcolare qualsiasi loss, l'uscita predetta viene allineata della latenza fissa $D = 159$ campioni e **vengono scartati i primi $N_{\text{pre}}$ campioni di warm-up**, valutando la loss esattamente sulla finestra utile $[s_k, e_k - D]$.
2. **Guardia ai bordi quando $s_k = 0$ (o su chunk già pre-estratti senza pre-roll):**
   Se $N_{\text{pre}} = 0$, nelle derivate d'inviluppo ($\Delta^+ e_c^{\text{fast}}[m]$) e nelle statistiche di picco/gain reduction si applica una maschera di esclusione del bordo iniziale $T_{\text{trim-start}} = 50\text{ ms}$ ($2400$ campioni) e del bordo finale $D = 159$ campioni. *(Nota: la sola variante `V0` mantiene il comportamento esatto dello Sprint 2 per riprodurre al centesimo i numeri della Tabella 0).*

#### 2. Criterio per la Selezione dei Frame Attivi senza Discontinuità di Gradiente
Un hard gate calcolato direttamente sul segnale ottimizzato $y(\theta)$ (es. $\mathbb{I}[L_t(y(\theta)) > \gamma]$) introduce discontinuità di passo (gradiente nullo o impulsivo) ogni volta che un frame attraversa la soglia durante l'ottimizzazione.
Risolviamo il problema alla radice sfruttando il fatto che la catena DDSP (una volta compensata la latenza fissa $D = 159$ campioni) è **sincrona nel tempo con il mix di input $x_{\text{mix}}$**:
- **Ancoraggio al segnale non ottimizzato (`no_grad`):**
  Per ogni sequenza temporale (frame STFT o frame d'inviluppo $t \in \{1,\dots,T\}$), l'energia di riferimento per decidere se un frame contiene programma musicale attivo è calcolata:
  - sul chunk di input $x_{\text{mix}}$ (allineato) per valutare $y(\theta)$;
  - sul chunk reference $y_{\text{ref}}$ per valutare $y_{\text{ref}}$.
- **Definizione dell'insieme dei frame attivi $\mathcal{A}(x_0)$ e dei pesi morbidi $w_{\text{act}}[t]$:**
  Sia $L_{\text{rms}}[t] = 10\log_{10}(\text{mean}_{n \in \text{frame } t} \frac{1}{2}(x_{0,L}^2[n] + x_{0,R}^2[n]) + \varepsilon)$ il livello RMS in dB del frame $t$ calcolato su $x_0 \in \{x_{\text{mix}}, y_{\text{ref}}\}$ (`with torch.no_grad()`), e sia:
  $$\theta_{\text{act}}(x_0) = \max\!\Big(-60.0\text{ dBFS},\; \max_{t'} L_{\text{rms}}[t'] - 35.0\text{ dB}\Big)$$
  - Per le **medie e deviazioni standard ponderate**, usiamo pesi sigmoidali lisci (costanti rispetto a $\theta$):
    $$w_{\text{act}}[t; x_0] = \sigma\!\left(\frac{L_{\text{rms}}[t] - \theta_{\text{act}}(x_0)}{\tau_{\text{gate}}}\right), \quad \tau_{\text{gate}} = 2.0\text{ dB}, \quad \tilde{w}_{\text{act}}[t; x_0] = \frac{w_{\text{act}}[t; x_0]}{\sum_{t'} w_{\text{act}}[t'; x_0] + \varepsilon}$$
  - Per il **calcolo dei quantili ($P_q^{\text{act}}$)**, selezioniamo il sottoinsieme di indici $\mathcal{A}(x_0) = \{t \mid L_{\text{rms}}[t] \ge \theta_{\text{act}}(x_0)\}$ (con fallback a tutti i frame se $|\mathcal{A}(x_0)| < 4$). Poiché gli indici $\mathcal{A}(x_{\text{mix}})$ dipendono **solo dall'input $x_{\text{mix}}$ e non da $\theta$**, l'insieme dei frame attivi su cui si calcola $P_q(y(\theta))$ non cambia mai da uno step di ottimizzazione all'altro, eliminando ogni discontinuità da selezione variabile dei frame.

---

### B. Task 3.1 — Banco Spettrale ERB-48, Statistiche Temporali e Opzione Asimmetrica De-Harshness

#### 1. Proiezione ERB-48 e Statistiche per Banda
Su $N_{\text{fft}} = 4096$, $\text{hop} = 1024$ ($\sim 21.3\text{ ms}$ a $48\text{ kHz}$), costruiamo $B = 48$ filtri triangolari normalizzati ad area unitaria spaziati uniformemente sulla scala $\text{ERB}_N$:
$$E(f) = 21.4 \log_{10}(0.00437 f + 1), \quad f \in [20\text{ Hz}, 20000\text{ Hz}]$$
Sia $L_{b,t}(x) = 10 \log_{10}\!\left(\sum_{k} W_{b,k} \frac{1}{2}(|X_L(k,t)|^2 + |X_R(k,t)|^2) + \varepsilon\right)$ il livello in dB della banda ERB $b$ al frame $t$:
- **Modalità `same_track` (ERB affiancata alla MR-STFT esistente):**
  $$\mathcal{L}_{\text{spec}}^{\text{same}}(y, y_{\text{ref}}) = \mathcal{L}_{\text{MR-STFT}}(y, y_{\text{ref}}) + w_{\text{erb}} \sum_{b=1}^{B} \sum_{t=1}^{T} \tilde{w}_{\text{act}}[t; y_{\text{ref}}] \cdot \frac{1}{B}\,\text{Huber}_{1.0}\!\left(L_{b,t}(y) - L_{b,t}(y_{\text{ref}})\right)$$
- **Modalità `stylistic` (Statistiche distribuzionali sui frame attivi):**
  Siano $\mu_b(x) = \sum_{t} \tilde{w}_{\text{act}}[t; x_0] L_{b,t}(x)$ la media attiva di banda, $\tilde{\mu}_b(x) = \mu_b(x) - \frac{1}{B}\sum_{b'=1}^B \mu_{b'}(x)$ il profilo timbrico centrato (invariante al guadagno globale), $\sigma_b(x) = \sqrt{\sum_{t} \tilde{w}_{\text{act}}[t; x_0] (L_{b,t}(x) - \mu_b(x))^2 + \varepsilon}$ la deviazione standard temporale, e $\Delta P_b^{\text{act}}(x) = P_{90, t \in \mathcal{A}(x_0)}(L_{b,t}(x)) - P_{10, t \in \mathcal{A}(x_0)}(L_{b,t}(x))$ l'escursione interdecile sui frame attivi:
  $$\mathcal{L}_{\text{spec}}^{\text{style}}(y, y_{\text{ref}}) = \frac{1}{B}\sum_{b=1}^{B} \Big[ \text{Huber}_{1.0}\!\big(\tilde{\mu}_b(y) - \tilde{\mu}_b(y_{\text{ref}})\big) + \alpha_\sigma \big|\sigma_b(y) - \sigma_b(y_{\text{ref}})\big| + \alpha_P \big|\Delta P_b^{\text{act}}(y) - \Delta P_b^{\text{act}}(y_{\text{ref}})\big| \Big]$$
  con valori iniziali $\alpha_\sigma = 0.25$, $\alpha_P = 0.15$.

#### 2. Opzione `V1b` — Penalità Asimmetrica e Relativa di De-Harshness ($2.5\text{–}5\text{ kHz}$)
Definiamo il tilt di presenza rispetto alla banda media fondamentale $\mathcal{B}_{\text{mid}} = [400\text{ Hz}, 2000\text{ Hz}]$:
$$\tau_{\text{harsh}}(x) = \left(\frac{1}{|\mathcal{B}_{\text{harsh}}|}\sum_{b \in [2.5, 5\text{ kHz}]} \mu_b(x)\right) - \left(\frac{1}{|\mathcal{B}_{\text{mid}}|}\sum_{b \in [0.4, 2\text{ kHz}]} \mu_b(x)\right)$$
La penalità è una hinge quadratica unilaterale con margine $m_{\text{tol}} = 0.75\text{ dB}$:
$$\mathcal{L}_{\text{de-harsh}}(y, y_{\text{ref}}) = \left[\text{ReLU}\!\Big(\big(\tau_{\text{harsh}}(y) - \tau_{\text{harsh}}(y_{\text{ref}})\big) - m_{\text{tol}}\Big)\right]^2$$

---

### C. Task 3.2 — Transient / Envelope Loss e Loss PLR Reference-Aware

#### 1. Formulazione Completa di $\mathcal{L}_{\text{trans}}^{\text{same}}$ e $\mathcal{L}_{\text{trans}}^{\text{style}}$ (Inclusi i Quantili dell'Onset Slope)
Su 3 macro-bande fisse di analisi ($\text{Low}: [20, 160]\text{ Hz}$, $\text{Mid}: [160, 3200]\text{ Hz}$, $\text{High}: [3200, 20000]\text{ Hz}$ ottenute con un crossover LR4 fisso di misura, indipendente dai 2 parametri ottimizzabili del crossover DDSP), calcoliamo gli inviluppi RMS in dB con passo $\text{hop}_{\text{env}} = 2.5\text{ ms}$ ($120$ campioni) su finestra veloce ($\tau_{\text{fast}} = 5\text{ ms}$, $240$ campioni) e lenta ($\tau_{\text{slow}} = 50\text{ ms}$, $2400$ campioni):
$$e_{c}^{\text{fast}}[m] = 10\log_{10}\!\left(\text{AvgPool}_{\tau_{\text{fast}}}(x_c^2)[m] + \varepsilon\right), \qquad e_{c}^{\text{slow}}[m] = 10\log_{10}\!\left(\text{AvgPool}_{\tau_{\text{slow}}}(x_c^2)[m] + \varepsilon\right)$$
Da cui ricaviamo:
- **Crest Factor Locale (Micro-Dinamica):** $C_c[m] = e_c^{\text{fast}}[m] - e_c^{\text{slow}}[m]$;
- **Positive Onset Slope (Derivata d'Attacco):** $\Delta^+ e_c^{\text{fast}}[m] = \text{ReLU}\!\left(e_c^{\text{fast}}[m] - e_c^{\text{fast}}[m-1]\right)$.

Le loss nelle due modalità sono:
- **Modalità `same_track` (confronto temporale allineato sui frame attivi $\tilde{w}_{\text{act}}[m; y_{\text{ref}}]$):**
  $$\mathcal{L}_{\text{trans}}^{\text{same}}(y, y_{\text{ref}}) = \frac{1}{3}\sum_{c \in \{L,M,H\}} \sum_{m \in \mathcal{A}_{\text{valid}}} \tilde{w}_{\text{act}}[m; y_{\text{ref}}] \Big[ \text{Huber}_{1.0}\!\big(C_c(y)[m] - C_c(y_{\text{ref}})[m]\big) + \beta_{\text{atk}}\,\text{Huber}_{1.0}\!\big(\Delta^+ e_c^{\text{fast}}(y)[m] - \Delta^+ e_c^{\text{fast}}(y_{\text{ref}})[m]\big) \Big]$$
- **Modalità `stylistic` (confronto dei quantili $q \in \{50, 90, 95\}$ sia del crest locale $C_c$ sia dell'onset slope $\Delta^+ e_c^{\text{fast}}$ sui frame attivi $\mathcal{A}(x_0)$):**
  $$\mathcal{L}_{\text{trans}}^{\text{style}}(y, y_{\text{ref}}) = \frac{1}{3}\sum_{c \in \{L,M,H\}} \frac{1}{3}\sum_{q \in \{50, 90, 95\}} \Big[ \big| P_{q, m \in \mathcal{A}(x_{\text{mix}})}\big(C_c(y)\big) - P_{q, m \in \mathcal{A}(y_{\text{ref}})}\big(C_c(y_{\text{ref}})\big) \big| + \beta_{\text{atk}} \big| P_{q, m \in \mathcal{A}(x_{\text{mix}})}\big(\Delta^+ e_c^{\text{fast}}(y)\big) - P_{q, m \in \mathcal{A}(y_{\text{ref}})}\big(\Delta^+ e_c^{\text{fast}}(y_{\text{ref}})\big) \big| \Big]$$
  con $\beta_{\text{atk}} = 1.0$.

#### 2. Distinzione Rigorosa tra Sample Peak, True Peak e Smooth Peak Proxy ($\mathcal{M}_\beta$) e Formulazione Completa di $\mathcal{L}_{\text{PLR}}$
Nel progetto sono definite tre grandezze distinte che non vengono mai confuse:
1. **`sample_peak_dbfs(x)` (Misura oggettiva discreta $1\times$):** $20\log_{10}(\max_{c,n}|x_c[n]|)$.
2. **`true_peak_dbtp(x)` (Misura oggettiva sovracampionata $4\times$ Kaiser-sinc, ITU-R BS.1770-5 Annex 2):** usata esclusivamente nel benchmark indipendente (`metrics.py`) e nel guardrail di delivery sull'export C++ finale. **La proxy differenziabile di training non viene mai denominata True Peak.**
3. **`soft_peak_dbfs_proxy(x)` (Proxy differenziabile liscia per il gradiente di training):**
   Per evitare che `torch.max` faccia passare il gradiente attraverso un solo campione isolato, suddividiamo il segnale in sotto-blocchi corti $m \in \{1,\dots,M\}$ di $10\text{ ms}$ ($480$ campioni), calcoliamo l'energia di picco smussata di blocco $p_m(x) = 10\log_{10}\!\left(\big(\frac{1}{2N_b}\sum_{c, n \in m} x_c^{2p}[n]\big)^{1/p} + \varepsilon\right)$ con $p = 8$, e applichiamo l'**operatore di media di Boltzmann (Mellowmax)** con temperatura inversa $\beta = 2.0\text{ dB}^{-1}$:
   $$\text{Peak}_{\text{soft-dBFS}}(x) = \mathcal{M}_\beta\big(p(x)\big) = \frac{\sum_{m=1}^{M} p_m(x) \exp\!\big(\beta (p_m(x) - p_{\max}(x))\big)}{\sum_{m=1}^{M} \exp\!\big(\beta (p_m(x) - p_{\max}(x))\big)}$$
   - *Proprietà matematiche:* (a) $C^\infty$ differenziabile ovunque per segnali non nulli; (b) **esente dal bias di lunghezza $-\frac{1}{\beta}\ln M$** del semplice Log-Sum-Exp ($\mathcal{M}_\beta(c\mathbf{1}) = c$ per qualunque durata $M$); (c) distribuisce il gradiente su tutti i picchi musicali entro $\sim 1.5\text{ dB}$ dal picco massimo.

Definiamo quindi in modo **identico per predizione e reference**:
$$\text{PLR}_{\text{proxy}}(x) = \text{Peak}_{\text{soft-dBFS}}(x) - \text{LUFS}_{\text{K-ungated}}(x)$$
Le equazioni complete per la loss PLR nelle due modalità sono:
- **Modalità `same_track`:**
  $$\mathcal{L}_{\text{PLR}}^{\text{same}}(y, y_{\text{ref}}) = \text{Huber}_{1.0}\!\left(\text{PLR}_{\text{proxy}}(y) - \text{PLR}_{\text{proxy}}(y_{\text{ref}})\right)$$
- **Modalità `stylistic`:**
  Sia $\Delta_{\text{PLR}}(y, y_{\text{ref}}) = \text{PLR}_{\text{proxy}}(y) - \text{PLR}_{\text{proxy}}(y_{\text{ref}})$, con semi-ampiezza della dead-zone $d_{\text{style}} = 1.0\text{ dB}$ e pavimento dinamico anti-schiacciamento $\text{PLR}_{\text{floor}}(y_{\text{ref}}) = \min\!\big(8.0\text{ dB},\; \text{PLR}_{\text{proxy}}(y_{\text{ref}}) - 1.5\text{ dB}\big)$:
  $$\mathcal{L}_{\text{PLR}}^{\text{style}}(y, y_{\text{ref}}) = \text{Huber}_{1.0}\!\Big(\text{ReLU}\!\big(|\Delta_{\text{PLR}}(y, y_{\text{ref}})| - d_{\text{style}}\big)\Big) + \gamma_{\text{floor}} \Big[\text{ReLU}\!\big(\text{PLR}_{\text{floor}}(y_{\text{ref}}) - \text{PLR}_{\text{proxy}}(y)\big)\Big]^2$$
  con $\gamma_{\text{floor}} = 0.5$. Inoltre, in `stylistic.py` anche il termine di loudness $\mathcal{L}_{\text{LUFS}}$ viene uniformato a $\text{Huber}_{1.0}(\text{LUFS}_{\text{K-ungated}}(y) - \text{target\_lufs})$ come in `loudness.py`, evitando gradienti esplosivi all'inizio dell'ottimizzazione.

---

### D. Task 3.3 — Prior Fisico-Acustici Separati (`V3a`, `V3b`, `V3c`)

#### 1. Variante `V3a` — Prior sulla Risposta Composita dell'EQ (Forma, Offset Medio Log-Frequenza e Gestione del Guadagno)
Valutiamo la risposta in ampiezza in dB della cascata dei 6 biquad su una griglia logaritmica di $K = 128$ frequenze $f_k \in [20\text{ Hz}, 20000\text{ Hz}]$ ($\omega_k = 2\pi f_k / f_s$):
$$G_{\text{EQ}}(f_k; \theta_{\text{EQ}}) = 20 \log_{10} \left| \prod_{b=0}^{5} H_b(e^{j\omega_k}; f_{0,b}, g_b, Q_b) \right| \quad [\text{dB}]$$
Chiamiamo:
$$\bar{G}_{\text{EQ}}^{\log} = \frac{1}{K}\sum_{k=1}^{K} G_{\text{EQ}}(f_k), \qquad \tilde{G}_{\text{EQ}}(f_k) = G_{\text{EQ}}(f_k) - \bar{G}_{\text{EQ}}^{\log}$$
dove $\bar{G}_{\text{EQ}}^{\log}$ è l'**offset medio della risposta in dB per passo logaritmico di frequenza** (e non un vero guadagno broadband sul segnale, che dipende dallo spettro dell'audio), mentre $\tilde{G}_{\text{EQ}}(f_k)$ è la **forma della risposta EQ**.

Definiamo in modo indipendente:
1. **Prior sulla forma della risposta EQ ($\mathcal{L}_{\text{EQ-shape}}$):**
   Con dead-zone di tolleranza $d_{\text{shape}} = 1.5\text{ dB}$ (entro cui correzioni tonali morbide non subiscono penalità d'ampiezza) e regolarizzazione della derivata prima log-frequenziale contro risonanze ad alto $Q$ o cancellazioni tra bande:
   $$\mathcal{L}_{\text{EQ-shape}}(\theta_{\text{EQ}}) = \frac{1}{K}\sum_{k=1}^{K} \Big[\text{ReLU}\!\big(|\tilde{G}_{\text{EQ}}(f_k)| - d_{\text{shape}}\big)\Big]^2 + w_{\text{slope}} \frac{1}{K-1}\sum_{k=1}^{K-1} \Big(\tilde{G}_{\text{EQ}}(f_{k+1}) - \tilde{G}_{\text{EQ}}(f_k)\Big)^2$$
2. **Prior sull'offset medio della risposta EQ ($\mathcal{L}_{\text{EQ-offset}}$):**
   Per evitare che i 6 filtri EQ vengano spinti tutti nella stessa direzione come semplice controllo di volume, ma **senza bloccare l'EQ né impedire il raggiungimento della loudness target**, applichiamo una dead-zone morbida $d_{\text{offset}} = 1.0\text{ dB}$:
   $$\mathcal{L}_{\text{EQ-offset}}(\theta_{\text{EQ}}) = w_{\text{offset}} \Big[\text{ReLU}\!\big(|\bar{G}_{\text{EQ}}^{\log}| - d_{\text{offset}}\big)\Big]^2$$
3. **Gestione del guadagno globale / makeup gain e raggiungimento della loudness:**
   Il raggiungimento di `target_lufs` (sia in aumento, es. `+4 LU` su `track02`, sia in attenuazione, es. `-3.6 LU` su `track01_style`) è affidato ai controlli di guadagno di stadio della catena: i 3 `comp_{low,mid,high}_makeup_db` ($\in [-3, +6]\text{ dB}$) e `limiter_push_gain_db` ($\in [0, +8]\text{ dB}$). Poiché $\mathcal{L}_{\text{EQ-offset}}$ agisce solo oltre $\pm 1.0\text{ dB}$ sull'offset medio dell'EQ e i makeup gain hanno piena libertà nel range $[-3, +6]\text{ dB}$, la catena può sia attenuare sia amplificare il segnale per raggiungere la loudness corretta senza costringere l'EQ a compensare il livello.

#### 2. Variante `V3b` — Prior su Limiter Stress, Gain Reduction Differenziale e Consistenza Makeup (Soglie come Ipotesi Sperimentali Iniziali e Trattamento dei Frame Silenziosi)
Tutti i segnali di gain reduction — $y_{g,c}[n] \ge 0$ (in dB) per i 3 compressori ($c \in \{L,M,H\}$) e $A_{\text{lim}}[n] = -20\log_{10}(g_{\text{smooth}}[n]) \ge 0$ (in dB) per il limiter — vengono valutati **esclusivamente sull'insieme dei campioni attivi $\mathcal{A}_{\text{smp}}(x_{\text{mix}})$** (determinati sui frame di $x_{\text{mix}}$ sopra la soglia $\theta_{\text{act}}(x_{\text{mix}})$ ed escludendo il warm-up e la coda di latenza $D = 159$). In questo modo i frame silenziosi (dove $y_{g,c}[n] = 0$ e $A_{\text{lim}}[n] = 0$) non diluiscono mai le medie né spostano i percentili.

Le soglie seguenti ($\tau_{\text{avg}} = 1.5\text{ dB}$, $\tau_{95} = 3.5\text{ dB}$, $\Delta_{\text{max}} = 4.0\text{ dB}$) sono definite come **ipotesi sperimentali iniziali** configurabili, non come costanti universali:
1. **Penalità di Limiter Stress sui campioni attivi $\mathcal{A}_{\text{smp}}$:**
   $$\bar{A}_{\text{lim}}^{\text{act}} = \frac{1}{|\mathcal{A}_{\text{smp}}|}\sum_{n \in \mathcal{A}_{\text{smp}}} A_{\text{lim}}[n], \qquad P_{95}^{\text{act}}(A_{\text{lim}}) = \text{Percentile}_{95, n \in \mathcal{A}_{\text{smp}}}(A_{\text{lim}}[n])$$
   $$\mathcal{L}_{\text{prior-lim}} = \Big[\text{ReLU}\!\big(\bar{A}_{\text{lim}}^{\text{act}} - \tau_{\text{avg}}\big)\Big]^2 + 0.5 \Big[\text{ReLU}\!\big(P_{95}^{\text{act}}(A_{\text{lim}}) - \tau_{95}\big)\Big]^2$$
2. **Penalità di Gain Reduction Differenziale tra Bande Adiacenti sui campioni attivi $\mathcal{A}_{\text{smp}}$:**
   $$\mathcal{L}_{\text{prior-comp-diff}} = \frac{1}{|\mathcal{A}_{\text{smp}}|}\sum_{n \in \mathcal{A}_{\text{smp}}} \Big( \Big[\text{ReLU}\!\big(|y_{g,L}[n] - y_{g,M}[n]| - \Delta_{\text{max}}\big)\Big]^2 + \Big[\text{ReLU}\!\big(|y_{g,M}[n] - y_{g,H}[n]| - \Delta_{\text{max}}\big)\Big]^2 \Big)$$
3. **Prior di Non-Conflitto Makeup vs Limiter:**
   Per evitare che i makeup gain spingano inutilmente quando il limiter è già oltre la soglia di stress $\tau_{\text{avg}}$, penalizziamo il prodotto incrociato solo quando entrambi sono positivi:
   $$\mathcal{L}_{\text{prior-gain-conflict}} = \text{ReLU}\!\big(\bar{A}_{\text{lim}}^{\text{act}} - \tau_{\text{avg}}\big) \cdot \frac{1}{3}\sum_{c \in \{L,M,H\}} \Big[\text{ReLU}\!\big(M_c - \bar{y}_{g,c}^{\text{act}} - 1.5\text{ dB}\big)\Big]^2$$

---

### E. Task 3.4 & Variante `V3c` — Definizioni Operative Complete di Stereo Multi-Banda, Correlazione L/R, Mono Fold-Down e Guardrail True Peak

#### 1. Convenzione di Normalizzazione Mid/Side e Somma Mono
In tutto il repository (`stft.py`, `imager.py`, `metrics.py`), la trasformazione Mid/Side ortonormale (che conserva l'energia totale $M^2[n] + S^2[n] = L^2[n] + R^2[n]$) e la somma di ascolto monofonico (che conserva il livello di un segnale centrale in fase $L = R$) sono definite esattamente come:
$$M[n] = \frac{L[n] + R[n]}{\sqrt{2}}, \qquad S[n] = \frac{L[n] - R[n]}{\sqrt{2}}, \qquad x_{\text{mono}}[n] = \frac{L[n] + R[n]}{2} = \frac{M[n]}{\sqrt{2}}$$
Quando $x_{\text{mono}}[n]$ viene riprodotto su un sistema a 2 canali in modalità dual-mono ($L_{\text{fold}}[n] = R_{\text{fold}}[n] = x_{\text{mono}}[n]$), la sua potenza complessiva sui due canali è:
$$P_{\text{fold}}[n] = L_{\text{fold}}^2[n] + R_{\text{fold}}^2[n] = 2 \left(\frac{L[n] + R[n]}{2}\right)^2 = M^2[n]$$

#### 2. Definizioni Operative delle 3 Metriche Stereo (Valutate sui Campioni Attivi $\mathcal{A}_{\text{smp}}(x_0)$)
1. **Side/Mid Energy Ratio per Macro-Banda $c \in \{\text{Low } [20, 160\text{ Hz}], \text{Mid } [160, 3200\text{ Hz}], \text{High } [3200, 20000\text{ Hz}]\}$:**
   Siano $M_c[n]$ e $S_c[n]$ le componenti Mid e Side della banda $c$:
   $$R_{S/M, c}(x) = 10 \log_{10} \left( \frac{\frac{1}{|\mathcal{A}_{\text{smp}}|}\sum_{n \in \mathcal{A}_{\text{smp}}} S_c^2[n] + \varepsilon}{\frac{1}{|\mathcal{A}_{\text{smp}}|}\sum_{n \in \mathcal{A}_{\text{smp}}} M_c^2[n] + \varepsilon} \right) \quad [\text{dB}]$$
2. **Correlazione di Fase Inter-Canale $\rho_{LR}$ (Globale e per Macro-Banda $c$):**
   $$\rho_{LR, c}(x) = \frac{\sum_{n \in \mathcal{A}_{\text{smp}}} L_c[n] R_c[n]}{\sqrt{\left(\sum_{n \in \mathcal{A}_{\text{smp}}} L_c^2[n] + \varepsilon\right)\left(\sum_{n \in \mathcal{A}_{\text{smp}}} R_c^2[n] + \varepsilon\right)}} \in [-1, +1]$$
3. **Mono Fold-Down Loss ($\Delta_{\text{mono}}$ in dB / LU):**
   Definiamo la perdita di livello percepito K-weighted passando dall'ascolto stereo $(L, R)$ all'ascolto in fold-down dual-mono $(L_{\text{fold}}, R_{\text{fold}}) = (\frac{L+R}{2}, \frac{L+R}{2})$. Poiché il filtro K-weighting $\mathcal{K}\{\cdot\}$ è lineare e tempo-invariante, $\mathcal{K}\{\frac{L+R}{\sqrt{2}}\} = M_K[n]$ e $\mathcal{K}\{\frac{L-R}{\sqrt{2}}\} = S_K[n]$. La differenza esatta di loudness K-weighted tra stereo e mono fold-down sui campioni attivi è:
   $$\Delta_{\text{mono}}(x) = \text{LUFS}_{\text{K}}^{\text{act}}(L, R) - \text{LUFS}_{\text{K}}^{\text{act}}(L_{\text{fold}}, R_{\text{fold}}) = 10 \log_{10} \left( \frac{\sum_{n \in \mathcal{A}_{\text{smp}}} \big(M_K^2[n] + S_K^2[n]\big) + \varepsilon}{\sum_{n \in \mathcal{A}_{\text{smp}}} M_K^2[n] + \varepsilon} \right) \ge 0\text{ dB}$$
   - *Interpretazione fisica:* $\Delta_{\text{mono}} = 0\text{ dB}$ per un segnale puramente mono ($S_K = 0$); $\Delta_{\text{mono}} = 3.01\text{ dB}$ quando l'energia Side eguaglia l'energia Mid ($E_{S,K} = E_{M,K}$, correlazione $\rho_{LR} \approx 0$); $\Delta_{\text{mono}} > 3.01\text{ dB}$ indica anti-fase / eccesso di energia Side che subisce forte cancellazione in mono.

#### 3. Formulazione della Loss/Prior Stereo per la Variante `V3c`
Nella variante `V3c` combiniamo:
- Il matching per banda del rapporto Side/Mid rispetto alla reference:
  $$\mathcal{L}_{\text{stereo-band}}(y, y_{\text{ref}}) = \frac{1}{3}\sum_{c \in \{L,M,H\}} \text{Huber}_{1.0}\!\left(R_{S/M, c}(y) - R_{S/M, c}(y_{\text{ref}})\right)$$
- La penalità di protezione Mono Fold-Down e anti-widening estremo:
  $$\mathcal{L}_{\text{mono-compat}}(y, y_{\text{ref}}, w_{\text{imager}}) = \Big[\text{ReLU}\!\Big(\Delta_{\text{mono}}(y) - \max\!\big(\Delta_{\text{mono}}(y_{\text{ref}}) + 0.5\text{ dB},\; 2.2\text{ dB}\big)\Big)\Big]^2 + \sum_{c \in \{L,M,H\}} \gamma_{w,c}\,(w_{\text{imager},c} - 1.0)^2$$
  con pesi $\gamma_w = [0.5, 0.2, 0.5]$ più stretti su Low (bassi mono-compatibili) e High (prevenzione dell'effetto *"stereo troppo aperto"* osservato su `track01` e `track03`).

#### 4. Guardrail di Delivery sul True Peak dell'Export C++ Finale
Indipendentemente dalla loss percettiva e applicato **con la stessa identica regola a tutte le varianti `V0..V4`**:
- Si definisce il parametro configurabile di progetto `delivery_tp_ceiling_dbtp` (default `-0.30 dBTP`, coerente con `true_peak_ceiling_dbtp = -0.3000` già dichiarato in `MasteringBenchmarkSuite`, oppure `-1.00 dBTP` se richiesto per delivery streaming).
- Dopo il render full-track tramite il motore C++ (`render_offline_with_tail_flush`), si misura `tp_pre = calculate_true_peak_dbtp(mastered_full_64)`.
- Se `tp_pre > delivery_tp_ceiling_dbtp`:
  1. Si calcola il guadagno statico di sicurezza $\Delta_{\text{trim-db}} = \text{delivery\_tp\_ceiling\_dbtp} - \text{tp\_pre} - 0.02\text{ dB}$ e si scala l'array C++ prima del salvataggio del WAV di delivery (`mastered_delivery = mastered_full_64 * 10**(Delta_trim_db / 20)`).
  2. Si **rimisura obbligatoriamente** `tp_post = calculate_true_peak_dbtp(mastered_delivery)` e `lufs_post = calculate_integrated_loudness_bs1770(mastered_delivery)` sul file WAV finale scritto su disco, verificando con un'asserzione che `tp_post <= delivery_tp_ceiling_dbtp` e registrando nel manifest sia le metriche raw post-C++ sia le metriche post-guardrail.

---

## 4. Protocollo Sperimentale e Sequenza di Ablation (`V0 → V1 → V2 → V3a → V3b → V3c → V4`)

Tutte le varianti utilizzano i medesimi 3 brani, reference, 4 chunk da $3.0\text{ s}$, seed a 64 bit, `num_steps = 30` e `lr = 0.04` documentati nella Tabella 0. Ogni ablation da `V1` a `V3c` viene valutata **individualmente rispetto a `V0`** per isolarne il contributo causale.

| ID Variante | Configurazione Esatta | Task Sprint 3 | Criterio Oggettivo e Percettivo di Valutazione |
|---|---|:---:|---|
| **`V0`** | Baseline Sprint 2 esatta (`num_steps=30`, `lr=0.04`, `MR-STFT` / `Mel-40`, `PLR box [8,14]`, `L2(z - Z_NEUTRAL)`). | Baseline | Riproduce esattamente i 6 manifest della Tabella 0. |
| **`V1`** | `V0` + Banco spettrale **`ERB-48`** (affiancato a MR-STFT in `same_track`; con $\tilde{\mu}_b, \sigma_b, \Delta P_b^{\text{act}}$ in `stylistic`) + pre-roll/active frames. | **Task 3.1** | Verifica miglioramento del bilanciamento sub/low-mid e riduzione della varianza spettrale rispetto a `V0`. |
| **`V1b` *(opz.)*** | `V1` + Penalità asimmetrica relativa **$\mathcal{L}_{\text{de-harsh}}$** ($2.5\text{–}5\text{ kHz}$ vs $0.4\text{–}2\text{ kHz}$). | **Task 3.1 (opz.)** | **Confronto diretto `V1b` vs `V1`:** viene promossa per l'inclusione in `V4` **solo se** riduce l'eccesso misurato $\tau_{\text{harsh}}(y) - \tau_{\text{harsh}}(y_{\text{ref}})$ su `track01 same_track` e `track03` senza ridurre l'energia $2.5\text{–}5\text{ kHz}$ o il voto d'ascolto su `track02` e `track01_style`. |
| **`V2`** | `V0` + **$\mathcal{L}_{\text{trans}}$** (crest locale $C_c$ + onset slope $\Delta^+ e_c^{\text{fast}}$, incluso nei quantili `stylistic`) + **$\mathcal{L}_{\text{PLR}}$ reference-aware** con proxy $\mathcal{M}_\beta$. | **Task 3.2** | **Confronto diretto `V2` vs `V0`:** verifica la conservazione del PLR (`track01 same_track`, `track03`) e la tenuta dei transienti ($\Delta C_c$) senza perdere il punch su `track02`. |
| **`V3a`** | `V0` + **Prior sulla risposta EQ** ($\mathcal{L}_{\text{EQ-shape}} + \mathcal{L}_{\text{EQ-offset}}$ su griglia log a 128 punti). | **Task 3.3 (a)** | **Confronto diretto `V3a` vs `V0`:** verifica che $|\bar{G}_{\text{EQ}}^{\log}|$ resti contenuto ($\lesssim 1\text{ dB}$), che la curva EQ sia più regolare e che `delta_lufs` non peggiori. |
| **`V3b`** | `V0` + **Prior dinamici sui processori** ($\mathcal{L}_{\text{prior-lim}}$ + $\mathcal{L}_{\text{prior-comp-diff}}$ + $\mathcal{L}_{\text{prior-gain-conflict}}$ sui campioni attivi). | **Task 3.3 (b)** | **Confronto diretto `V3b` vs `V0`:** verifica la riduzione di $\bar{A}_{\text{lim}}^{\text{act}}$ e $P_{95}^{\text{act}}(A_{\text{lim}})$ nei brani densi (`track01`, `track03`). |
| **`V3c`** | `V0` + **Prior e loss Stereo** ($\mathcal{L}_{\text{stereo-band}} + \mathcal{L}_{\text{mono-compat}}$). | **Task 3.3 (c) & 3.4** | **Confronto diretto `V3c` vs `V0`:** verifica il contenimento di `imager_width_high`, la riduzione di $\Delta_{\text{mono}}$ e la correlazione $\rho_{LR}$. |
| **`V4`** | **Combinazione selettiva** dei soli moduli tra `{V1, (V1b), V2, V3a, V3b, V3c}` che hanno superato positivamente le rispettive ablation individuali, con pesi calibrati tramite le norme dei gradienti al passo iniziale. | **Task 3.1–3.5** | Confronto finale completo quantitativo e d'ascolto (`level_matched` e `program_level`) contro `V0`. |
| **`Diag-E`** | Esperimento diagnostico separato sull'headroom d'ingresso (`track01` con trim statico riprodotto/compensato nel render C++), senza alcun target universale a `-20/-22 LUFS`. | **Diagnostica separata** | Isola l'impatto del livello d'ingresso sul punto di lavoro non lineare a parità di loss `V4`. |

---

## 5. Test di Correttezza Numerica, Gradcheck, Parità PyTorch/C++ e Validazione Audio

1. **Test di Identità, Invarianza e Non-Invasività:**
   - $\mathcal{L}_{\text{ERB}}(x, x) = 0$, $\mathcal{L}_{\text{trans}}^{\text{same}}(x, x) = 0$, $\mathcal{L}_{\text{trans}}^{\text{style}}(x, x) = 0$, $\mathcal{L}_{\text{PLR}}^{\text{same}}(x, x) = 0$, $\mathcal{L}_{\text{PLR}}^{\text{style}}(x, x) = 0$.
   - A $z = Z_{\text{NEUTRAL}}$, $\mathcal{L}_{\text{EQ-shape}} = 0$ e $\mathcal{L}_{\text{EQ-offset}} = 0$.
   - Per $\mathcal{L}_{\text{de-harsh}}$: verifica analitica che per $\tau_{\text{harsh}}(y) \le \tau_{\text{harsh}}(y_{\text{ref}}) + m_{\text{tol}}$ si abbia $\mathcal{L}_{\text{de-harsh}} == 0.0$ e $\nabla_\theta \mathcal{L}_{\text{de-harsh}} == \mathbf{0}$.
2. **Strategia di `gradcheck` e Gestione dei Punti Non Differenziabili:**
   - Per le funzioni lisce ($\mathcal{L}_{\text{ERB}}^{\text{same}}$, $\mathcal{M}_\beta$, $\mathcal{L}_{\text{EQ-shape}}$, $\mathcal{L}_{\text{EQ-offset}}$ lontano dalle soglie di `ReLU`/`Huber`), si esegue `torch.autograd.gradcheck` in `float64` in **regioni regolari** lontane dai punti angolosi (`|x - threshold| > 10 * eps`).
   - Per gli operatori con punti angolosi o d'ordine (`ReLU` sui margini di dead-zone, quantili empirici $P_q$ che dipendono dall'ordinamento, `max` sui frame), **non si pretende la differenziabilità classica nei punti di kink/tie**: si testano separatamente (a) la finitezza e limitatezza del sottogradiente esattamente sui punti di soglia ($\text{arg} = 0$), (b) l'assenza di `NaN`/`Inf` su segnali silenziosi o costanti, e (c) la consistenza direzionale per differenze finite unilaterali ($h > 0$ vs $h < 0$).
3. **Contratto di Parità PyTorch $\leftrightarrow$ C++:**
   - Si applicano rigorosamente le tolleranze già stabilite nella test suite del progetto (`python/tests/test_cpp_parity.py`, `test_ticket_1_4.py`, `test_ticket_1_5.py`, `test_sprint_1_correctness.py`):
     - `map_z_to_theta` / `map_theta_to_z`: $\text{MSE} < 10^{-14}$, $\max|\Delta\theta| < 10^{-7}$;
     - Catena completa `DifferentiableMasteringChain(eq_mode="iir")` vs `py_core_dsp.MasteringChain`: $\text{Null}_{\text{RMS}} \le -100.0\text{ dBFS}$ e $\text{MaxErr} \le -85.0\text{ dBFS}$.
   - Inoltre, si verifica che in PyTorch `chain(x, z, return_aux=False)` e `chain(x, z, return_aux=True)[0]` producano esattamente lo stesso tensore d'uscita (`torch.equal`).
4. **Test del Guardrail True Peak sull'Export C++:**
   - Verifica automatica che per ciascuna variante `V0..V4` il file WAV C++ finale dopo il guardrail rispetti `calculate_true_peak_dbtp(wav_path) <= delivery_tp_ceiling_dbtp`.

---

## 6. Analisi dei Rischi e Mitigazioni

1. **Conflitto tra Loudness Target e Limiter Stress ($\mathcal{L}_{\text{LUFS}}$ vs $\mathcal{L}_{\text{prior-lim}}$):**
   - *Mitigazione:* Uniformazione di $\mathcal{L}_{\text{LUFS}}$ a Huber loss (`beta = 1.0`) anche in `stylistic`, controllo dell'offset medio log-frequenza dell'EQ ($\mathcal{L}_{\text{EQ-offset}}$ con dead-zone $\pm 1.0\text{ dB}$ per impedire che l'EQ gonfi le medio-alte per sfruttare lo shelf $+4\text{ dB}$ del K-weighting), e separazione dell'ablation `V3a` / `V3b`.
2. **Over-Smoothing (Mix Spento) da Prior Troppo Forti:**
   - *Mitigazione:* Tutte le penalità di prior (`V3a`, `V3b`, `V3c`) utilizzano dead-zone esplicite ($d_{\text{shape}} = 1.5\text{ dB}$, $d_{\text{offset}} = 1.0\text{ dB}$, $\tau_{\text{avg}} = 1.5\text{ dB}$, $\Delta_{\text{max}} = 4.0\text{ dB}$) entro cui i gradienti di regolarizzazione sono nulli, lasciando piena libertà di azione per interventi musicali moderati.
3. **Attribuzione Confusa degli Effetti tra Moduli:**
   - *Mitigazione:* Separazione rigorosa di `V1`, `V2`, `V3a`, `V3b`, `V3c` in esperimenti singoli confrontati ciascuno contro `V0` prima di assemblare `V4`.
