# AutoMastering — Specifica Tecnica e Piano Sperimentale per lo Sprint 3 (Revisione Finale v3)

**Candidato:** Giovanni Antonecchia  
**Corso di Laurea:** Master's Thesis in Sound and Music Engineering — Politecnico di Milano (DEIB)  
**Artefatto Pubblico di Riproducibilità `V0`:** `docs/human_validation/SPRINT2_V0_REPRODUCIBILITY_MANIFEST.json`  
**Roadmap Approvata (Invariata):** `docs/AutoMastering_Audit_Roadmap_Antigravity.md` (§10, Sprint 3)  
**Vincolo Architetturale:** Architettura DDSP white-box a 41 parametri invariata; nessuna modifica al codice DSP/optimizer applicata in questo documento; nessuna riscrittura della roadmap approvata.

---

## 0. Verifica della Baseline `V0`, Artefatto Pubblico e Distinzione Esatta tra `same_track` e `stylistic`

Tutti i dati pubblici e depurati necessari per riprodurre esattamente le 6 sessioni dello Sprint 2 (seed a 64 bit, hash SHA-256 dei file mix e reference, lag di allineamento, coordinate esatte dei 4 chunk di training e dei chunk reference, 41 parametri fisici e normalizzati `z_opt`, e scomposizione di tutti i sotto-termini della loss) sono pubblicati in [`docs/human_validation/SPRINT2_V0_REPRODUCIBILITY_MANIFEST.json`](../human_validation/SPRINT2_V0_REPRODUCIBILITY_MANIFEST.json), senza esporre percorsi privati o chiavi blind.

- **Serializzazione sicura dei seed `uint64` nel manifest JSON:** Poiché i seed a 64 bit senza segno superano `Number.MAX_SAFE_INTEGER` ($2^{53} - 1 = 9007199254740991$) dello standard IEEE-754 double utilizzato dai parser JavaScript/RFC-8259, nel file JSON sia il campo top-level `seed` di ciascuna sessione sia `optimization_config.seed` sono serializzati come **stringhe decimali esatte** (es. `"5405783196615117109"`). In fase di utilizzo e riproduzione, il consumatore deve convertirli esplicitamente a intero a 64 bit prima di inizializzare il generatore pseudocasuale (`int(seed_str)` in Python, `BigInt(seed_str)` in JavaScript, `std::stoull(seed_str)` in C++).

### 0.1 Configurazione Congelata di `V0` e Distinzione tra le Due Modalità
In `V0` (baseline storica dello Sprint 2, eseguita da `scripts/human_validation/render_sprint2.py`):
- **Iperparametri di ottimizzazione comuni:** `num_steps = 30` (transizione hybrid EQ `fft -> iir` al passo `21`), `learning_rate = 0.04` (Adam + `CosineAnnealingLR(T_max=30, eta_min=0.002)` + `clip_grad_norm_(1.0)`), `num_chunks = 4`, `chunk_duration_sec = 3.0 s` (`144000` campioni a `48 kHz`), pesi chunk uniformi `1/4`.
- **Funzione obiettivo `V0` in modalità `same_track`** (`MasteringPerceptualLoss` in `combined.py` + `DifferentiableLoudnessLoss` in `loudness.py`):
  $$\mathcal{L}_{V0}^{\text{same}}(y, y_{\text{ref}}, z) = 1.0 \cdot \mathcal{L}_{\text{MR-STFT}}(y_{\text{al}}, y_{\text{ref,al}}) + 0.8 \cdot \Big(1.0 \cdot \text{Huber}_{1.0}\big(\text{LUFS}_{\text{K}}(y) - \text{target\_lufs}\big) + 0.2 \cdot \mathcal{L}_{\text{PLR-box}[8,14]}(y)\Big) + 0.01 \cdot \frac{1}{41}\|z - Z_{\text{NEUTRAL}}\|_2^2$$
  dove **solo in `same_track`** il termine dinamico è il box fisso $\mathcal{L}_{\text{PLR-box}[8,14]}(y) = \text{ReLU}(8.0 - \text{PLR}(y))^2 + \text{ReLU}(\text{PLR}(y) - 14.0)^2$.
- **Funzione obiettivo `V0` in modalità `stylistic`** (`AggregatePerceptualLoss` in `stylistic.py`):
  $$\mathcal{L}_{V0}^{\text{style}}(y, y_{\text{ref}}, z) = 1.0 \cdot \big\|\tilde{\mu}_{\text{Mel40}}(y) - \tilde{\mu}_{\text{Mel40}}(y_{\text{ref}})\big\|_1 + 0.8 \cdot \big(\text{LUFS}_{\text{K}}(y) - \text{target\_lufs}\big)^2 + 0.5 \cdot \big(\text{PLR}(y) - \text{PLR}(y_{\text{ref}})\big)^2 + 0.3 \cdot \big|R_{S/M}^{\ln}(y) - R_{S/M}^{\ln}(y_{\text{ref}})\big| + 0.01 \cdot \frac{1}{41}\|z - Z_{\text{NEUTRAL}}\|_2^2$$
  dove **`stylistic` usa già in `V0` un matching quadratico diretto del PLR rispetto alla reference** con `weight_dynamics = 0.5` (e **non** il box `[8, 14]`), oltre al termine stereo globale con `weight_stereo = 0.3`.

### 0.2 Tabella di Riproducibilità `V0` (Valutazione Omogenea in Modalità `eq_mode="iir"` sugli Stessi 4 Chunk)

Per garantire il confronto rigorosamente omogeneo richiesto, sia la loss composita iniziale a `z_0 = Z_NEUTRAL` sia la loss composita finale sul vettore restituito `z_opt` (`step 30`) utilizzate per calcolare le percentuali di riduzione sono valutate **nella medesima modalità `eq_mode="iir"` e sullo stesso identico insieme di 4 chunk** (riportando per completezza anche `initial_loss_z0_fft` e `step 29`):

| Sessione (`prefix`) | Modalità | Seed (64-bit) | Training Chunks Mix (`start_sec – end_sec`) | Loss Iniziale `z_0` (`FFT` / **`IIR`**) | Loss `step 29` (`IIR`) | Loss Finale `z_opt` (`step 30`, **`IIR`**) | Riduzione (`z_0 IIR -> z_opt IIR`) |
|---|---|:---:|---|:---:|:---:|:---:|:---:|
| **`track01`** | `same_track` | `5405783196615117109` | `1.0–4.0s`, `30.0–33.0s`, `47.0–50.0s`, `54.0–57.0s` | `6.6669` / **`6.6670`** | `3.0966` | **`3.0841`** | **`-53.74%`** |
| **`track02`** | `same_track` | `3048363069421082382` | `8.0–11.0s`, `29.0–32.0s`, `40.0–43.0s`, `59.0–62.0s` | `14.2843` / **`14.2845`** | `10.4312` | **`10.4173`** | **`-27.07%`** |
| **`track03`** | `same_track` | `12990700764526553868` | `12.0–15.0s`, `25.5–28.5s`, `36.0–39.0s`, `65.0–68.0s` | `10.4493` / **`10.4494`** | `6.4696` | **`6.4545`** | **`-38.23%`** |
| **`track01_style`** | `stylistic` | `14887125631601849244` | `1.0–4.0s`, `30.0–33.0s`, `47.0–50.0s`, `54.0–57.0s` | `28.0920` / **`28.0920`** | `8.5858` | **`8.5529`** | **`-69.55%`** |
| **`track02_style`** | `stylistic` | `3606996967096283602` | `8.0–11.0s`, `29.0–32.0s`, `40.0–43.0s`, `59.0–62.0s` | `159.9058` / **`159.9058`** | `82.9988` | **`82.7613`** | **`-48.24%`** |
| **`track03_style`** | `stylistic` | `15136152743686100324` | `12.0–15.0s`, `25.5–28.5s`, `36.0–39.0s`, `65.0–68.0s` | `110.5957` / **`110.5956`** | `46.7759` | **`46.5655`** | **`-57.90%`** |

---

## 1. Matrice di Associazione: Proposte vs 5 Task Approvati dello Sprint 3

| Proposta / Osservazione | Task Sprint 3 di Riferimento | Classificazione & Variante Ablation | Motivazione Tecnica ed Evidenza Empirica (Sprint 1–2) |
|---|---|---|---|
| **Baseline di controllo del contesto `V0p` (Pre-Roll `2.5 s` + warm-up su loss `V0`)** | **Controllo Metodologico** (Task 3.5) | **Baseline di controllo** (`V0p`) | Isola l'effetto del nuovo contesto di warm-up (`2.5 s` di pre-roll coerente con le costanti di release fino a `600 ms`) rispetto alla baseline storica `V0`, così che `V1..V3c` siano confrontate a parità di stato DSP con `V0p`. |
| **A.1 Rappresentazione spettrale ERB-48 multi-statistica + conservazione MR-STFT in `same_track`** | **Task 3.1** (*ERB o Mel spectral statistics*) | **Già previsto** (`V1`) | Affianca un banco percettivo ERB a 48 bande alla MR-STFT in `same_track` e sostituisce la sola media Mel-40 in `stylistic` con statistiche temporali ($\tilde{\mu}_b, \sigma_b, \Delta P_b^{\text{act}}$) sui frame attivi. |
| **A.2 Normalizzazione dei termini e diagnostica delle norme dei gradienti per banda/gruppo** | **Task 3.1 & Task 3.4** (*Metric & gradient diagnostics*) | **Già previsto** (Tutte le varianti `V0..V4`) | Misura empiricamente $\|\nabla_{u_g} \mathcal{L}_i\|_2$ per ogni termine di loss $i$ e gruppo di parametri $g$. |
| **D. Penalità asimmetrica e relativa alla reference per l'eccesso in banda $2.5\text{–}5\text{ kHz}$ (De-Harshness)** | **Task 3.1** (*ERB/Mel spectral statistics*) | **Raffinazione opzionale** (`V1b`) | Riduce l'eccesso di presence quando $\tau_{\text{harsh}}(y) > \tau_{\text{harsh}}(y_{\text{ref}}) + m_{\text{tol}}$, evitando al contempo un'attenuazione eccessiva o un suono troppo scuro sui brani già equilibrati. |
| **B.1 Transient / Envelope Loss multi-banda ($\tau_{\text{fast}}, \tau_{\text{slow}}$, crest locale $C_c$ e onset slope $\Delta^+ e_c^{\text{fast}}$)** | **Task 3.2** (*Transient/envelope loss*) | **Già previsto** (`V2`) | Punto-a-punto in `same_track`; tramite quantili ($P_{50}, P_{90}, P_{95}$) sia di $C_c$ sia di $\Delta^+ e_c^{\text{fast}}$ in `stylistic`, sui frame attivi post-warm-up. |
| **B.2 Loss PLR reference-aware con `Boltzmann-weighted soft-peak mean` distinta da Sample Peak e True Peak** | **Task 3.2 & Task 3.4** (*Transient/envelope & Dinamica*) | **Già previsto** (`V2`) | Sostituisce il box `[8, 14] dB` in `same_track` con Huber reference-aware e la MSE pura in `stylistic` con dead-zone $\pm 1.0\text{ dB}$ + floor dinamico, usando la stessa definizione di $\text{PLR}_{\text{proxy}}$ per predizione e reference. |
| **C. Prior sulla risposta composita dell'EQ $G_{\text{EQ}}(f_k)$ (forma spettrale, offset medio log-frequenza e gestione guadagno)** | **Task 3.3** (*Prior fisico-acustici*) | **Raffinazione interna** (`V3a`) | Isolato in `V3a`: regolarizza la forma della risposta dei 6 biquad e limita derive sull'offset medio log-frequenza senza ostacolare il raggiungimento della loudness target. |
| **B.3 Prior dinamici sui processori: Limiter Stress, Gain Reduction differenziale e non-conflitto makeup** | **Task 3.3** (*Prior fisico-acustici*) | **Raffinazione interna** (`V3b`) | Isolato in `V3b`: valuta $\bar{A}_{\text{lim}}^{\text{act}}$ e $P_{95}^{\text{act}}(A_{\text{lim}})$ sui campioni attivi con soglie trattate come ipotesi sperimentali iniziali. |
| **S. Prior e loss Stereo distinti tra `same_track` (matching per banda) e `stylistic` (prior conservativo e mono guardrail)** | **Task 3.3 & Task 3.4** (*Prior & Dinamica/Stereo*) | **Già previsto / Raffinazione** (`V3c`) | Isolato in `V3c`: in `same_track` esegue matching S/M per banda verso la reference dello stesso brano; in `stylistic` **non** forza l'immagine stereo di un altro brano ma applica un prior conservativo e il guardrail sulla differenza di energia K-weighted in mono $\Delta E_{\text{mono,K}}$. |
| **E. Analisi di sensibilità all'Input Gain Staging / Headroom in ingresso** | **Diagnostica separata (Task 3.5)** | **Esperimento diagnostico separato** (`Diag-E`) | Nessun target universale a `-20/-22 LUFS`. Verifica controllata dell'effetto del trim statico d'ingresso (riprodotto/compensato fedelmente nel render C++). |
| **F. Guardrail di Delivery sul True Peak dell'export C++ finale** | **Requisito di Delivery / Task 3.4** | **Guardrail di verifica/export** (Tutte le varianti) | Policy configurabile di progetto verificata sul WAV C++ finale con rimisura obbligatoria post-trim, separata dalla loss percettiva. |

---

## 2. Formulazioni Matematiche Complete (Task 3.1 – 3.4)

### A. Pre-Roll / Warm-Up di `2.5 s` (`V0p..V4`), Test di Convergenza e Selezione dei Frame Attivi

#### 1. Durata del Pre-Roll Coerente con le Costanti di Tempo Massime ($600\text{ ms}$) e Gestione dei Chunk Vicini a $t=0$
I compressori Low e Mid possono raggiungere costanti di tempo $\tau_{\text{release}} \le 600\text{ ms}$ e $\tau_{\text{attack}} \le 120\text{ ms}$ (`param_space.py`, righe 43–50). Poiché un inviluppo esponenziale del primo ordine impiega $4\tau \approx 2.4\text{ s}$ per esaurire oltre il $98.2\%$ del transitorio iniziale ($5\tau = 3.0\text{ s}$ per il $99.3\%$), un pre-roll di $250\text{ ms}$ non è sufficiente.
Pertanto, a partire dalla baseline di controllo `V0p` e per tutte le varianti `V1..V4`:
- **Durata nominale di pre-roll:** $T_{\text{pre}} = 2.5\text{ s}$ ($N_{\text{pre}} = 120000$ campioni a $48\text{ kHz}$).
- **Regola esatta per i chunk che iniziano prima di $T_{\text{pre}}$:**
  Per un chunk con intervallo di valutazione $[s_k, e_k]$, l'indice di partenza dell'estrazione dal brano completo è:
  $$s_{k,\text{ext}} = \max(0,\; s_k - N_{\text{pre}}), \qquad N_{\text{pre,eff}}(k) = s_k - s_{k,\text{ext}}$$
  Quando $s_k < N_{\text{pre}}$ (ad esempio `chunk 0` di `track01`, che inizia a $s_0 = 1.0\text{ s} = 48000$ campioni), viene utilizzato **tutto il contesto disponibile dall'inizio fisico del brano ($0 \dots s_k - 1$)**. Poiché a $t = 0\text{ s}$ (inizio reale del file audio) anche il render full-track C++ parte legittimamente da stato DSP zero, usare l'intero segmento $[0, e_k]$ riproduce **esattamente** lo stato interno che il motore C++ ha all'istante $s_k$, senza mai inizializzare artificialmente lo stato DSP a metà traccia.
- **Slicing post-catena:**
  La catena DDSP elabora $x_{\text{ext}} = x[:, s_{k,\text{ext}} : e_k]$. Compensata la latenza deterministica $D = 159$ campioni e rimosso il pre-roll $N_{\text{pre,eff}}(k)$, la loss viene valutata esclusivamente sulla regione utile allineata di lunghezza $T_{\text{eval}} = (e_k - s_k) - D$.
- **Test obbligatorio di convergenza del pre-roll (`test_preroll_convergence`):**
  Su chunk situati a $s_k \ge 10\text{ s}$ e con parametri di release massimi ($\tau_{\text{release}} = 600\text{ ms}$), si confrontano le uscite e i valori delle loss ottenuti con $T_{\text{pre}} \in \{0.25\text{ s}, 1.0\text{ s}, 2.0\text{ s}, 2.5\text{ s}, 3.0\text{ s}\}$ rispetto allo stato esatto del rendering continuo dall'inizio del brano ($T_{\text{pre}} = s_k$). Si verifica che passando da $2.5\text{ s}$ a $3.0\text{ s}$ la variazione relativa della loss sia $< 10^{-3}$ e la differenza RMS del segnale nella finestra utile $[s_k, e_k]$ sia $\le -60\text{ dBFS}$.

#### 2. Selezione dei Frame Attivi Ancorata al Segnale Non Ottimizzato (`no_grad`), Gestione dei Chunk Silenziosi/Sparsi e Conversione Frame-Level $\to$ Sample-Level
Per evitare discontinuità di gradiente da hard gate dipendenti da $y(\theta)$, la maschera dei frame attivi $\mathcal{A}(x_0)$ e i pesi sigmoidali lisci $\tilde{w}_{\text{act}}[t; x_0]$ sono calcolati fuori dal grafo autograd (`with torch.no_grad()`) usando $x_0 = x_{\text{mix}}$ per la predizione $y(\theta)$ e $x_0 = y_{\text{ref}}$ per la reference ($t \in \{1, \dots, T_{\text{frames}}\}$, con $\varepsilon = 10^{-12}$):
$$\theta_{\text{act}}(x_0) = \max\!\Big(-60.0\text{ dBFS},\; \max_{t'} L_{\text{rms}}[t'; x_0] - 35.0\text{ dB}\Big), \qquad w_{\text{act}}[t; x_0] = \sigma\!\left(\frac{L_{\text{rms}}[t; x_0] - \theta_{\text{act}}(x_0)}{2.0\text{ dB}}\right)$$
$$\tilde{w}_{\text{act}}[t; x_0] = \frac{w_{\text{act}}[t; x_0]}{\sum_{t'=1}^{T_{\text{frames}}} w_{\text{act}}[t'; x_0] + \varepsilon}, \qquad \mathcal{A}_{\text{raw}}(x_0) = \big\{t \;\big|\; L_{\text{rms}}[t; x_0] \ge \theta_{\text{act}}(x_0)\big\}$$

- **Comportamento deterministico per chunk silenziosi o con meno del numero minimo di frame attivi ($M_{\min}$):**
  Sia $M_{\min} = \min(8,\; T_{\text{frames}})$ il numero minimo di frame attivi richiesto per stimare stabilmente i quantili ($P_{10}, P_{50}, P_{90}, P_{95}$) e le statistiche temporali senza degenerazioni:
  1. **Regime nominale ($|\mathcal{A}_{\text{raw}}(x_0)| \ge M_{\min}$ e $\max_{t'} L_{\text{rms}}[t'; x_0] \ge -70.0\text{ dBFS}$):** $\mathcal{A}(x_0) = \mathcal{A}_{\text{raw}}(x_0)$ e $\mathbb{I}_{\text{valid}}(x_0) = 1.0$.
  2. **Chunk sparsamente attivo ($0 \le |\mathcal{A}_{\text{raw}}(x_0)| < M_{\min}$ con $\max_{t'} L_{\text{rms}}[t'; x_0] \ge -70.0\text{ dBFS}$):** l'insieme discreto $\mathcal{A}(x_0)$ seleziona deterministicamente i **top-$M_{\min}$ frame più energetici** di $x_0$ tramite $\text{topk}\big(L_{\text{rms}}[\cdot; x_0],\; k=M_{\min}\big)$, così che $|\mathcal{A}(x_0)| = M_{\min} \ge 1$ sia sempre garantito; i pesi continui $\tilde{w}_{\text{act}}[t; x_0]$ restano strettamente positivi e ben condizionati poiché $\sigma(\cdot) \in (0, 1)$ e il denominatore include $+\varepsilon = 10^{-12}$.
  3. **Chunk completamente silenzioso ($\max_{t'} L_{\text{rms}}[t'; x_0] < -70.0\text{ dBFS}$):** quando l'intera finestra utile del chunk è sotto $-70.0\text{ dBFS}$ (silenzio digitale), si impongono $\tilde{w}_{\text{act}}[t; x_0] = \frac{1}{T_{\text{frames}}}$, $\mathcal{A}(x_0) = \{1, \dots, T_{\text{frames}}\}$ e un moltiplicatore di validità del chunk $\mathbb{I}_{\text{valid}}(x_0) = 0.0$ che annulla il contributo delle loss ERB, transient, limiter e stereo su quel chunk silenzioso. In combinazione con $+\varepsilon = 10^{-12}$ in ogni logaritmo (`10 * log10(E + eps)`), radice quadrata (`sqrt(var + eps)`) e rapporto energetico (`(num + eps) / (den + eps)`), **nessuna tra le loss ERB, transient, limiter o stereo può mai produrre divisioni per zero, `Inf` o `NaN`**.

- **Conversione esplicita della maschera frame-level in maschera sample-level $\mathcal{A}_{\text{smp}}(x_0)$:**
  Per i termini nel dominio dei campioni valutati sui $T_{\text{eval}}$ campioni post-warm-up (`V3b`: limiter stress $\bar{A}_{\text{lim}}^{\text{act}}, P_{95}^{\text{act}}(A_{\text{lim}})$ e GR differenziale; `V3c`: rapporti energetici $R_{S/M,c}, \rho_{LR,c}, \Delta E_{\text{mono,K}}$), il segnale di ancoraggio $x_0$ viene partizionato in $M_b = \lfloor T_{\text{eval}} / H_{\text{smp}} \rfloor$ blocchi non sovrapposti da $H_{\text{smp}} = 480$ campioni ($10\text{ ms}$ a $48\text{ kHz}$), con livello RMS di blocco $L_{\text{blk}}[m; x_0]$ ($m \in \{0, \dots, M_b - 1\}$). Calcolato l'indicatore binario di blocco attivo $b_{\text{act}}[m; x_0] \in \{0, 1\}$ con la medesima soglia $\theta_{\text{blk}}(x_0) = \max(-60.0\text{ dBFS}, \max_{m'} L_{\text{blk}}[m'; x_0] - 35.0\text{ dB})$ e la stessa regola di fallback sui top-$M_{\min}$ blocchi se $\sum_m b_{\text{act}}[m; x_0] < M_{\min}$ (o tutti i blocchi con $\mathbb{I}_{\text{valid}}(x_0) = 0.0$ se $\max_{m'} L_{\text{blk}}[m'; x_0] < -70.0\text{ dBFS}$), la maschera frame-level viene convertita in maschera sample-level tramite **espansione piecewise-constant (`repeat_interleave(H_smp)`)** su ciascun blocco $[m H_{\text{smp}}, (m+1)H_{\text{smp}} - 1]$, estendendo l'ultimo blocco $M_b - 1$ agli eventuali $T_{\text{eval}} - M_b H_{\text{smp}}$ campioni residui di coda:
  $$m_{\text{smp}}[n; x_0] = b_{\text{act}}\!\big[\min(\lfloor n / H_{\text{smp}} \rfloor,\; M_b - 1);\; x_0\big] \in \{0, 1\}, \qquad \mathcal{A}_{\text{smp}}(x_0) = \big\{n \in [0, T_{\text{eval}} - 1] \;\big|\; m_{\text{smp}}[n; x_0] = 1\big\}$$
  Questo garantisce $|\mathcal{A}_{\text{smp}}(x_0)| \ge M_{\min} \cdot H_{\text{smp}} = 3840$ campioni ($80\text{ ms}$) per ogni chunk standard e denominatori strettamente positivi $\max(|\mathcal{A}_{\text{smp}}(x_0)|, 1)$ in tutte le medie campionarie.

---

### B. Task 3.1 — Banco Spettrale ERB-48 (`V1`) e Penalità Asimmetrica Relativa De-Harshness (`V1b`)

#### 1. Proiezione ERB-48 e Statistiche per Banda (`V1`)
Su $N_{\text{fft}} = 4096$, $\text{hop} = 1024$, proiettiamo la potenza media dei canali $\frac{1}{C}\sum_{c=1}^{C} |X_c(k,t)|^2$ ($C \in \{1, 2\}$) su $B = 48$ filtri triangolari ad area unitaria spaziati uniformemente sulla scala $\text{ERB}_N(f) = 21.4 \log_{10}(0.00437 f + 1)$ tra $20\text{ Hz}$ e $20\text{ kHz}$, ottenendo $L_{b,t}(x) = 10\log_{10}(E_{b,t}(x) + \varepsilon)$:
- **In `same_track` (ERB affiancata alla MR-STFT esistente):**
  $$\mathcal{L}_{\text{spec}}^{\text{same}}(y, y_{\text{ref}}) = \mathcal{L}_{\text{MR-STFT}}(y, y_{\text{ref}}) + w_{\text{erb}} \sum_{b=1}^{B} \sum_{t=1}^{T} \tilde{w}_{\text{act}}[t; y_{\text{ref}}] \cdot \frac{1}{B}\,\text{Huber}_{1.0}\!\left(L_{b,t}(y) - L_{b,t}(y_{\text{ref}})\right)$$
- **In `stylistic` (statistiche temporali sui frame attivi):**
  Con $\mu_b(x) = \sum_t \tilde{w}_{\text{act}}[t; x_0] L_{b,t}(x)$, $\tilde{\mu}_b(x) = \mu_b(x) - \frac{1}{B}\sum_{b'} \mu_{b'}(x)$, $\sigma_b(x) = \sqrt{\sum_t \tilde{w}_{\text{act}}[t; x_0](L_{b,t}(x) - \mu_b(x))^2 + \varepsilon}$ e $\Delta P_b^{\text{act}}(x) = P_{90, t \in \mathcal{A}(x_0)}(L_{b,t}(x)) - P_{10, t \in \mathcal{A}(x_0)}(L_{b,t}(x))$:
  $$\mathcal{L}_{\text{spec}}^{\text{style}}(y, y_{\text{ref}}) = \frac{1}{B}\sum_{b=1}^{B} \Big[ \text{Huber}_{1.0}\!\big(\tilde{\mu}_b(y) - \tilde{\mu}_b(y_{\text{ref}})\big) + \alpha_\sigma \big|\sigma_b(y) - \sigma_b(y_{\text{ref}})\big| + \alpha_P \big|\Delta P_b^{\text{act}}(y) - \Delta P_b^{\text{act}}(y_{\text{ref}})\big| \Big]$$

#### 2. Opzione `V1b` — Penalità Asimmetrica Relativa De-Harshness ($2.5\text{–}5\text{ kHz}$) e Criterio di Selezione Corretto
Definiamo il tilt di presenza rispetto alla banda media $\mathcal{B}_{\text{mid}} = [400\text{ Hz}, 2000\text{ Hz}]$:
$$\tau_{\text{harsh}}(x) = \left(\frac{1}{|\mathcal{B}_{\text{harsh}}|}\sum_{b \in [2.5, 5\text{ kHz}]} \mu_b(x)\right) - \left(\frac{1}{|\mathcal{B}_{\text{mid}}|}\sum_{b \in [0.4, 2\text{ kHz}]} \mu_b(x)\right)$$
$$\mathcal{L}_{\text{de-harsh}}(y, y_{\text{ref}}) = \left[\text{ReLU}\!\Big(\big(\tau_{\text{harsh}}(y) - \tau_{\text{harsh}}(y_{\text{ref}})\big) - m_{\text{tol}}\Big)\right]^2, \quad m_{\text{tol}} = 0.75\text{ dB}$$
- **Criterio di selezione e inclusione di `V1b`:**
  Una riduzione moderata di energia tra $2.5$ e $5\text{ kHz}$ è **desiderata** quando il master presenta un eccesso di presence/harshness rispetto alla reference; ciò che va evitato è una **sovra-attenuazione della presence che renda il timbro scuro o cupo**. Pertanto `V1b` viene confrontata direttamente contro `V1` e viene inclusa in `V4` se:
  1. riduce il surplus $\tau_{\text{harsh}}(y) - \tau_{\text{harsh}}(y_{\text{ref}})$ e attenua l'*harshness* percepita nelle sessioni critiche (`track01 same_track`, `track03 program_level`);
  2. **evita un'attenuazione eccessiva della presence o uno scurimento timbrico** (verificando che $\tau_{\text{harsh}}(y)$ non scenda mai oltre $-1.0\text{ dB}$ sotto $\tau_{\text{harsh}}(y_{\text{ref}})$ e che non peggiorino chiarezza vocale e bilanciamento tonale all'ascolto su `track02` e `track01_style`).

---

### C. Task 3.2 — Transient / Envelope Loss e Loss PLR Reference-Aware (`V2`)

#### 1. Formulazione Completa di $\mathcal{L}_{\text{trans}}^{\text{same}}$ e $\mathcal{L}_{\text{trans}}^{\text{style}}$ (Inclusi i Quantili dell'Onset Slope)
Su 3 macro-bande fisse di analisi ($\text{Low}: [20, 160]\text{ Hz}$, $\text{Mid}: [160, 3200]\text{ Hz}$, $\text{High}: [3200, 20000]\text{ Hz}$), calcoliamo gli inviluppi RMS in dB su finestra veloce ($\tau_{\text{fast}} = 5\text{ ms}$) e lenta ($\tau_{\text{slow}} = 50\text{ ms}$), il crest factor locale $C_c[m] = e_c^{\text{fast}}[m] - e_c^{\text{slow}}[m]$ e la derivata d'attacco positiva $\Delta^+ e_c^{\text{fast}}[m] = \text{ReLU}(e_c^{\text{fast}}[m] - e_c^{\text{fast}}[m-1])$:
- **Modalità `same_track`:**
  $$\mathcal{L}_{\text{trans}}^{\text{same}}(y, y_{\text{ref}}) = \frac{1}{3}\sum_{c \in \{L,M,H\}} \sum_{m} \tilde{w}_{\text{act}}[m; y_{\text{ref}}] \Big[ \text{Huber}_{1.0}\!\big(C_c(y)[m] - C_c(y_{\text{ref}})[m]\big) + \beta_{\text{atk}}\,\text{Huber}_{1.0}\!\big(\Delta^+ e_c^{\text{fast}}(y)[m] - \Delta^+ e_c^{\text{fast}}(y_{\text{ref}})[m]\big) \Big]$$
- **Modalità `stylistic` (quantili $q \in \{50, 90, 95\}$ sia di $C_c$ sia di $\Delta^+ e_c^{\text{fast}}$ sui frame attivi):**
  $$\mathcal{L}_{\text{trans}}^{\text{style}}(y, y_{\text{ref}}) = \frac{1}{3}\sum_{c \in \{L,M,H\}} \frac{1}{3}\sum_{q \in \{50, 90, 95\}} \Big[ \big| P_{q, m \in \mathcal{A}(x_{\text{mix}})}\big(C_c(y)\big) - P_{q, m \in \mathcal{A}(y_{\text{ref}})}\big(C_c(y_{\text{ref}})\big) \big| + \beta_{\text{atk}} \big| P_{q, m \in \mathcal{A}(x_{\text{mix}})}\big(\Delta^+ e_c^{\text{fast}}(y)\big) - P_{q, m \in \mathcal{A}(y_{\text{ref}})}\big(\Delta^+ e_c^{\text{fast}}(y_{\text{ref}})\big) \big| \Big]$$

#### 2. `Boltzmann-Weighted Soft-Peak Mean` (Valida per Mono e Stereo) e Loss PLR Reference-Aware
Manteniamo rigorosamente separate:
1. **`sample_peak_dbfs(x)`:** massimo campionario discreto $1\times$.
2. **`true_peak_dbtp(x)`:** misura oggettiva $4\times$ Kaiser-sinc (ITU-R BS.1770-5 Annex 2), usata solo nel benchmark e nel guardrail di delivery.
3. **`boltzmann_soft_peak_dbfs(x)` (`Boltzmann-weighted soft-peak mean` / `softmax-weighted peak proxy`):**
   Sia $C = \text{x.shape}[1] \in \{1, 2\}$ il **numero effettivo di canali** (valida sia per input mono $C=1$ sia per stereo $C=2$). Suddiviso il segnale in $M$ blocchi corti di lunghezza $N_b = 480$ campioni ($10\text{ ms}$), definiamo il livello di picco morbido del blocco $m$ con normalizzazione esatta per $C \cdot N_b$:
   $$p_m(x) = 10\log_{10}\!\left(\left(\frac{1}{C \cdot N_b}\sum_{c=1}^{C}\sum_{n \in m} |x_c[n]|^{2p}\right)^{1/p} + \varepsilon\right), \quad p = 8$$
   e la **media pesata di Boltzmann dei picchi di blocco** con $\beta = 2.0\text{ dB}^{-1}$ e $p_{\max}(x) = \max_{m'} p_{m'}(x)$:
   $$\text{Peak}_{\text{soft-dBFS}}(x) = \frac{\sum_{m=1}^{M} p_m(x) \exp\!\big(\beta (p_m(x) - p_{\max}(x))\big)}{\sum_{m=1}^{M} \exp\!\big(\beta (p_m(x) - p_{\max}(x))\big)}$$
   Definiamo in modo identico per predizione e reference $\text{PLR}_{\text{proxy}}(x) = \text{Peak}_{\text{soft-dBFS}}(x) - \text{LUFS}_{\text{K-ungated}}(x)$, ottenendo:
   - **In `same_track`:** $\mathcal{L}_{\text{PLR}}^{\text{same}}(y, y_{\text{ref}}) = \text{Huber}_{1.0}\!\left(\text{PLR}_{\text{proxy}}(y) - \text{PLR}_{\text{proxy}}(y_{\text{ref}})\right)$.
   - **In `stylistic`:** con $\Delta_{\text{PLR}} = \text{PLR}_{\text{proxy}}(y) - \text{PLR}_{\text{proxy}}(y_{\text{ref}})$, dead-zone $d_{\text{style}} = 1.0\text{ dB}$ e pavimento $\text{PLR}_{\text{floor}}(y_{\text{ref}}) = \min(8.0\text{ dB},\; \text{PLR}_{\text{proxy}}(y_{\text{ref}}) - 1.5\text{ dB})$:
     $$\mathcal{L}_{\text{PLR}}^{\text{style}}(y, y_{\text{ref}}) = \text{Huber}_{1.0}\!\Big(\text{ReLU}\!\big(|\Delta_{\text{PLR}}| - d_{\text{style}}\big)\Big) + \gamma_{\text{floor}} \Big[\text{ReLU}\!\big(\text{PLR}_{\text{floor}}(y_{\text{ref}}) - \text{PLR}_{\text{proxy}}(y)\big)\Big]^2$$

---

### D. Task 3.3 & 3.4 — Prior Fisico-Acustici e Stereo Separati (`V3a`, `V3b`, `V3c`)

#### 1. Variante `V3a` — Prior sulla Risposta Composita dell'EQ
Sulla griglia logaritmica di $K = 128$ frequenze $f_k \in [20\text{ Hz}, 20000\text{ Hz}]$, sia $G_{\text{EQ}}(f_k; \theta_{\text{EQ}}) = 20\log_{10}\big|\prod_{b=0}^5 H_b(e^{j\omega_k})\big|$, $\bar{G}_{\text{EQ}}^{\log} = \frac{1}{K}\sum_{k=1}^K G_{\text{EQ}}(f_k)$ l'**offset medio della risposta per passo logaritmico di frequenza**, e $\tilde{G}_{\text{EQ}}(f_k) = G_{\text{EQ}}(f_k) - \bar{G}_{\text{EQ}}^{\log}$ la forma spettrale:
$$\mathcal{L}_{\text{EQ-shape}}(\theta_{\text{EQ}}) = \frac{1}{K}\sum_{k=1}^{K} \Big[\text{ReLU}\!\big(|\tilde{G}_{\text{EQ}}(f_k)| - 1.5\text{ dB}\big)\Big]^2 + w_{\text{slope}} \frac{1}{K-1}\sum_{k=1}^{K-1} \Big(\tilde{G}_{\text{EQ}}(f_{k+1}) - \tilde{G}_{\text{EQ}}(f_k)\Big)^2$$
$$\mathcal{L}_{\text{EQ-offset}}(\theta_{\text{EQ}}) = w_{\text{offset}} \Big[\text{ReLU}\!\big(|\bar{G}_{\text{EQ}}^{\log}| - 1.0\text{ dB}\big)\Big]^2$$
Il raggiungimento del target di loudness rimane affidato ai 3 makeup gain $[-3, +6]\text{ dB}$ e al limiter push $[0, +8]\text{ dB}$.

#### 2. Variante `V3b` — Limiter Stress, Gain Reduction Differenziale e Non-Conflitto Makeup (sui Campioni Attivi $\mathcal{A}_{\text{smp}}(x_{\text{mix}})$)
Valutate esclusivamente sui campioni attivi post-warm-up $\mathcal{A}_{\text{smp}}(x_{\text{mix}})$ con soglie sperimentali iniziali $\tau_{\text{avg}} = 1.5\text{ dB}$, $\tau_{95} = 3.5\text{ dB}$, $\Delta_{\text{max}} = 4.0\text{ dB}$:
$$\mathcal{L}_{\text{prior-lim}} = \Big[\text{ReLU}\!\big(\bar{A}_{\text{lim}}^{\text{act}} - \tau_{\text{avg}}\big)\Big]^2 + 0.5 \Big[\text{ReLU}\!\big(P_{95}^{\text{act}}(A_{\text{lim}}) - \tau_{95}\big)\Big]^2$$
$$\mathcal{L}_{\text{prior-comp-diff}} = \frac{1}{|\mathcal{A}_{\text{smp}}|}\sum_{n \in \mathcal{A}_{\text{smp}}} \Big( \Big[\text{ReLU}\!\big(|y_{g,L}[n] - y_{g,M}[n]| - \Delta_{\text{max}}\big)\Big]^2 + \Big[\text{ReLU}\!\big(|y_{g,M}[n] - y_{g,H}[n]| - \Delta_{\text{max}}\big)\Big]^2 \Big)$$
$$\mathcal{L}_{\text{prior-gain-conflict}} = \text{ReLU}\!\big(\bar{A}_{\text{lim}}^{\text{act}} - \tau_{\text{avg}}\big) \cdot \frac{1}{3}\sum_{c \in \{L,M,H\}} \Big[\text{ReLU}\!\big(M_c - \bar{y}_{g,c}^{\text{act}} - 1.5\text{ dB}\big)\Big]^2$$

#### 3. Variante `V3c` — Distinzione Esplicita tra `same_track` e `stylistic` per Stereo e Guardrail di Energia K-Weighted in Mono ($\Delta E_{\text{mono, K}}$)
Con normalizzazione $M[n] = \frac{L[n]+R[n]}{\sqrt{2}}$, $S[n] = \frac{L[n]-R[n]}{\sqrt{2}}$ e fold-down dual-mono $L_{\text{fold}}[n] = R_{\text{fold}}[n] = \frac{L[n]+R[n]}{2} = \frac{M[n]}{\sqrt{2}}$:
- **Side/Mid Energy Ratio per macro-banda $c \in \{L, M, H\}$ sui campioni attivi:**
  $$R_{S/M, c}(x) = 10 \log_{10} \left( \frac{\sum_{n \in \mathcal{A}_{\text{smp}}} S_c^2[n] + \varepsilon}{\sum_{n \in \mathcal{A}_{\text{smp}}} M_c^2[n] + \varepsilon} \right) \quad [\text{dB}]$$
- **Correlazione inter-canale $\rho_{LR, c}(x) \in [-1, +1]$:**
  $$\rho_{LR, c}(x) = \frac{\sum_{n \in \mathcal{A}_{\text{smp}}} L_c[n] R_c[n]}{\sqrt{\big(\sum_{n \in \mathcal{A}_{\text{smp}}} L_c^2[n] + \varepsilon\big)\big(\sum_{n \in \mathcal{A}_{\text{smp}}} R_c^2[n] + \varepsilon\big)}}$$
- **Differenza di Energia K-Weighted in dB tra Stereo e Fold-Down Mono ($\Delta E_{\text{mono, K}}$, `k_weighted_mono_folddown_drop_db`):**
  Definita esplicitamente come **differenza di energia K-weighted non-gated sui campioni attivi** (specificando che **non** è una misura di loudness integrata gated conforme a ITU-R BS.1770-5):
  $$\Delta E_{\text{mono, K}}(x) = 10 \log_{10} \left( \frac{\sum_{n \in \mathcal{A}_{\text{smp}}} \big(M_K^2[n] + S_K^2[n]\big) + \varepsilon}{\sum_{n \in \mathcal{A}_{\text{smp}}} M_K^2[n] + \varepsilon} \right) \ge 0\text{ dB}$$

**Distinzione operativa della loss stereo in `V3c` tra `same_track` e `stylistic`:**
1. **In modalità `same_track` (stesso brano e stesso arrangiamento):**
   È fisicamente fondato allineare il bilancio Side/Mid per banda al master di riferimento dello stesso mix, unitamente al guardrail di mono compatibility e al prior conservativo sull'imager:
   $$\mathcal{L}_{V3c}^{\text{same}}(y, y_{\text{ref}}, w_{\text{imager}}) = \frac{1}{3}\sum_{c \in \{L,M,H\}} \text{Huber}_{1.0}\!\Big(R_{S/M,c}(y) - R_{S/M,c}(y_{\text{ref}})\Big) + \Big[\text{ReLU}\!\Big(\Delta E_{\text{mono, K}}(y) - \max\!\big(\Delta E_{\text{mono, K}}(y_{\text{ref}}) + 0.5\text{ dB},\; 2.2\text{ dB}\big)\Big)\Big]^2 + \sum_{c \in \{L,M,H\}} \gamma_{w,c}(w_{\text{imager},c} - 1.0)^2$$
2. **In modalità `stylistic` (brano differente — nessun matching forzato verso il profilo stereo della reference esterna):**
   Per non imporre la larghezza stereo di un'altra produzione musicale (che può avere panning e arrangiamento completamente diversi), in `V3c` **non si confronta $R_{S/M}(y)$ con $R_{S/M}(y_{\text{ref}})$**; al posto del matching verso la reference esterna si applicano un **prior conservativo ancorato al mix originale $x_{\text{mix}}$** (che permette solo variazioni moderate $\pm 1.5\text{ dB}$ rispetto a $R_{S/M,c}(x_{\text{mix}})$ e mantiene focalizzate le basse frequenze) e il **guardrail di mono compatibility**:
   $$\mathcal{L}_{V3c}^{\text{style}}(y, x_{\text{mix}}, w_{\text{imager}}) = \frac{1}{3}\sum_{c \in \{L,M,H\}} \Big[\text{ReLU}\!\Big(\big|R_{S/M,c}(y) - R_{S/M,c}(x_{\text{mix}})\big| - 1.5\text{ dB}\Big)\Big]^2 + \Big[\text{ReLU}\!\Big(\Delta E_{\text{mono, K}}(y) - \max\!\big(\Delta E_{\text{mono, K}}(x_{\text{mix}}) + 0.4\text{ dB},\; 2.0\text{ dB}\big)\Big)\Big]^2 + \sum_{c \in \{L,M,H\}} \gamma_{w,c}(w_{\text{imager},c} - 1.0)^2$$
   con $\gamma_w = [0.6, 0.2, 0.5]$.

---

## 3. Sequenza Completa di Ablation (`V0 → V0p → V1 → V2 → V3a → V3b → V3c → V4`)

| ID Variante | Contesto Pre-Roll | Configurazione della Funzione Obiettivo | Confronto Primario di Attribuzione |
|---|:---:|---|---|
| **`V0`** | Storico Sprint 2 (`0 s`) | Loss originale Sprint 2 (`same_track`: `MR-STFT + LUFS + PLR box [8,14] + L2(z)`; `stylistic`: `Mel-40 + LUFS MSE + PLR MSE + Stereo L1 + L2(z)`). | Baseline storica congelata (Tabella 0). |
| **`V0p`** | **Pre-Roll `2.5 s`** | **Stessa identica funzione obiettivo e pesi di `V0`**, ma con estrazione pre-roll `2.5 s` (o da $t=0$ per `chunk 0` se $s_0 < 2.5\text{ s}$) e warm-up trim. | **`V0p` vs `V0`:** isola l'effetto puro del warm-up dello stato DSP. |
| **`V1`** | Pre-Roll `2.5 s` | `V0p` + Banco spettrale **`ERB-48`** (affiancato a MR-STFT in `same_track`; con $\tilde{\mu}_b, \sigma_b, \Delta P_b^{\text{act}}$ in `stylistic`) sui frame attivi. | **`V1` vs `V0p`:** isola il contributo della rappresentazione spettrale ERB-48. |
| **`V1b` *(opz.)*** | Pre-Roll `2.5 s` | `V1` + Penalità asimmetrica relativa **$\mathcal{L}_{\text{de-harsh}}$** ($2.5\text{–}5\text{ kHz}$ vs $0.4\text{–}2\text{ kHz}$). | **`V1b` vs `V1`:** inclusa in `V4` solo se riduce l'harshness sulle tracce critiche senza causare sovra-attenuazione della presence o scurimento. |
| **`V2`** | Pre-Roll `2.5 s` | `V0p` + **$\mathcal{L}_{\text{trans}}$** ($C_c$ e $\Delta^+ e_c^{\text{fast}}$; quantili di entrambi in `stylistic`) + **$\mathcal{L}_{\text{PLR}}$ reference-aware** con `Boltzmann-weighted soft-peak mean`. | **`V2` vs `V0p`:** isola il contributo di transient loss e PLR reference-aware. |
| **`V3a`** | Pre-Roll `2.5 s` | `V0p` + **Prior sulla risposta EQ** ($\mathcal{L}_{\text{EQ-shape}} + \mathcal{L}_{\text{EQ-offset}}$ su griglia log a 128 punti). | **`V3a` vs `V0p`:** isola l'effetto del prior sulla curva composita EQ. |
| **`V3b`** | Pre-Roll `2.5 s` | `V0p` + **Prior dinamici sui processori** ($\mathcal{L}_{\text{prior-lim}} + \mathcal{L}_{\text{prior-comp-diff}} + \mathcal{L}_{\text{prior-gain-conflict}}$ sui campioni attivi). | **`V3b` vs `V0p`:** isola l'effetto del controllo di limiter stress e GR differenziale. |
| **`V3c`** | Pre-Roll `2.5 s` | `V0p` + **Prior/loss Stereo modalità-specifica** (`same_track`: matching S/M per banda + $\Delta E_{\text{mono,K}}$; `stylistic`: prior ancorato al mix + $\Delta E_{\text{mono,K}}$). | **`V3c` vs `V0p`:** isola il controllo dell'immagine stereo e della mono compatibility. |
| **`V4`** | Pre-Roll `2.5 s` | **Combinazione selettiva** dei soli componenti tra `{V1, (V1b), V2, V3a, V3b, V3c}` che hanno mostrato miglioramenti verificati rispetto a `V0p`. | **`V4` vs `V0p` e `V0`:** valutazione finale quantitativa e d'ascolto (`level_matched` e `program_level`). |
| **`Diag-E`** | Pre-Roll `2.5 s` | Esperimento diagnostico separato sull'headroom d'ingresso (nessun target universale a `-20/-22 LUFS`), con compensazione coerente nel render C++. | Isola la sensibilità del punto di lavoro non lineare al livello d'ingresso. |

---

## 4. Riepilogo dei Test Obbligatori di Correttezza e Validazione

1. **Test di Convergenza del Pre-Roll (`test_preroll_convergence`):** confronto tra $T_{\text{pre}} \in \{0.25\text{ s}, 1.0\text{ s}, 2.0\text{ s}, 2.5\text{ s}, 3.0\text{ s}\}$ e stato di streaming full-track con $\tau_{\text{release}} = 600\text{ ms}$, verificando la stabilizzazione delle metriche e delle loss tra $2.5\text{ s}$ e $3.0\text{ s}$.
2. **Test Mono e Stereo della `Boltzmann-weighted soft-peak mean`:** verifica che per un segnale dual-mono $(x, x)$ e per il corrispondente segnale mono $(x)$ ($C=1$ vs $C=2$) `boltzmann_soft_peak_dbfs` restituisca esattamente lo stesso valore in dBFS ($\Delta < 10^{-12}$) e sia esente da bias di durata.
3. **Test di Identità, Asimmetria e `gradcheck`:** `gradcheck` in `float64` nelle regioni lisce e test dedicati di finitezza/sottogradiente sui punti di soglia (`ReLU`, quantili, `max`).
4. **Parità PyTorch $\leftrightarrow$ C++ entro i Limiti Contrattuali del Progetto:** $\text{Null}_{\text{RMS}} \le -100.0\text{ dBFS}$ e $\text{MaxErr} \le -85.0\text{ dBFS}$ (`test_ticket_1_5.py`), più identità esatta del tensore d'uscita tra `return_aux=False` e `return_aux=True`.
5. **Guardrail True Peak sull'Export C++ Finale:** applicazione uniforme del ceiling configurabile `delivery_tp_ceiling_dbtp` a tutte le varianti `V0..V4`, con rimisura obbligatoria di `calculate_true_peak_dbtp` sul file WAV finale dopo l'eventuale safety trim.
