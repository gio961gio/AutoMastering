# AutoMastering — Follow-up Ingegneristico sul Guardrail True Peak e Inter-Sample Overshoot (Delivery Compliance)

**Autore:** Giovanni Antonecchia  
**Corso di Laurea:** Master's Thesis in Sound and Music Engineering — Politecnico di Milano (DEIB)  
**Standard di Riferimento:** ITU-R BS.1770-4/5 (Annex 2), EBU R128 (Delivery Guardrail: $\text{True Peak} \le -0.30\text{ dBTP}$)  
**Artefatti Dati:** [`docs/sprint3/SPRINT3_V2_ABLATION_RESULTS.json`](./SPRINT3_V2_ABLATION_RESULTS.json), [`docs/sprint3/SPRINT3_V1_HARMONIZED_RESULTS.json`](./SPRINT3_V1_HARMONIZED_RESULTS.json), `scratch/canonical_true_peak_tests.json`  
**Data:** Novembre 2026  
**Stato:** Audit Diagnostico e Analisi Architetturale Offline (**DSP Core C++/PyTorch Congelato**)  

---

## 1. Premessa e Separazione dei Piani Architetturali

Dall'audit comparativo delle 4 varianti armonizzate (`V0p`, `V1`, `V1b`, `V2`) emerge un riscontro oggettivo chiaro:
- In **5 sessioni su 6** (`track01`, `track02`, `track03`, `track02_style`, `track03_style`), il True Peak full-track supera il limite raccomandato dalle specifiche di distribuzione ($-0.30\text{ dBTP}$), attestandosi tra **$-0.09\text{ dBTP}$ e $+0.32\text{ dBTP}$** (con un superamento compreso tra $+0.21\text{ dB}$ e $+0.62\text{ dB}$).
- In **1 sessione su 6** (`track01_style`), caratterizzata da un target di loudness molto conservativo ($-17.99\text{ LUFS}$), il True Peak rimane ampiamente all'interno della soglia ($-2.46\text{ .. }-1.63\text{ dBTP}$, con zero violazioni).

> [!IMPORTANT] **Inquadramento Sistemico: Delivery vs Ottimizzazione**
> 1. **`V3a` non è un True Peak Limiter:** Lo Step `V3a` della roadmap approvata è dedicato unicamente al prior sulla risposta composita dell'EQ parametrico (`SPRINT3_V3A_EQ_PRIOR_DESIGN.md`).
> 2. **L'overshoot inter-sample è un fenomeno di ricostruzione continua:** Il limiter attuale della catena DDSP (`DifferentiableLookaheadLimiter`) rispetta **perfettamente al centesimo di decibel** il proprio ceiling a sample peak ($1\times$ a $48\text{ kHz}$): $\max_n |y[n]| \le -0.3000\text{ dBFS}$.
> 3. Il superamento di $-0.30\text{ dBTP}$ rilevato dal benchmark a $4\times$ oversampling è dovuto interamente all'**Inter-Sample Peak Overshoot (ISP)** generato dall'interpolazione continua tra campioni discreti.

---

## 2. Analisi Fisica e Architetturale del Limiter Attuale

### 2.1 Architettura del `DifferentiableLookaheadLimiter` (`limiter.py` e `cpp/core_dsp/`)
La catena di mastering DDSP include al sesto stadio un lookahead brickwall limiter differenziabile operante a frequenza di campionamento nativa ($f_s = 48000\text{ Hz}$):
1. **Rilevatore di Picco Discreto ($1\times$):**
   Calcola l'ampiezza massima istantanea stereo su campioni discreti:
   $$p[n] = \max(|x_L[n]|, |x_R[n]|)$$
2. **Min-Filter su Finestra di Lookahead ($N_L = 144$ campioni, $\approx 3.0\text{ ms}$):**
   Calcola il guadagno di compressione istantaneo necessario a non superare il ceiling lineare $C_{\text{lin}} = 10^{-0.30 / 20} \approx 0.96605$ ($-0.30\text{ dBFS}$):
   $$g_{\text{raw}}[n] = \min\left(1.0, \frac{C_{\text{lin}}}{p[n] + \varepsilon}\right)$$
   $$g_{\text{min}}[n] = \min_{k \in [0, N_L - 1]} g_{\text{raw}}[n + k]$$
3. **Inviluppo di Release One-Pole ($\tau_{\text{rel}} = 80\text{ ms}$):**
   $$g_{\text{rel}}[n] = \begin{cases} g_{\text{min}}[n] & \text{se } g_{\text{min}}[n] \le g_{\text{rel}}[n-1] \\ \alpha_{\text{rel}} g_{\text{rel}}[n-1] + (1 - \alpha_{\text{rel}}) g_{\text{min}}[n] & \text{se } g_{\text{min}}[n] > g_{\text{rel}}[n-1] \end{cases}$$
4. **Smussamento Boxcar (Moving Average a 144 campioni) e Ritardo di Allineamento:**
   L'inviluppo $g_{\text{rel}}[n]$ viene mediato su $N_L = 144$ campioni per eliminare discontinuità ad alta frequenza, e applicato al segnale ritardato di $144$ campioni:
   $$y_c[n] = x_c[n - N_L] \cdot g_{\text{smooth}}[n], \quad c \in \{L, R\}$$

Grazie a questa topologia, il segnale discreto soddisfa rigorosamente:
$$\max_{n} |y_c[n]| \le C_{\text{lin}} \implies \text{Sample Peak} \le -0.3000\text{ dBFS}$$

### 2.2 Origine Matematica dell'Inter-Sample Peak Overshoot ($4\times$ Oversampling)
La misura oggettiva di conformità True Peak conforme a ITU-R BS.1770-4/5 Annex 2 impone un sovracampionamento a $4\times$ ($192\text{ kHz}$) tramite un filtro polifase FIR sinc finestrato Kaiser a 64 tap per fase (attenuazione banda d'arresto $> 90\text{ dB}$).  
La forma d'onda analogica continua $y_c(t)$ ricostruita dal convertitore D/A o dal filtro di oversampling è data dalla formula di interpolazione Whittaker-Shannon:
$$y_c(t) = \sum_{n} y_c[n] \cdot \text{sinc}\left(\frac{t - n T_s}{T_s}\right)$$

Anche se per ogni campione discreto $|y_c[n]| \le C_{\text{lin}}$, nei punti intermedi $t = (n + \delta) T_s$ con $\delta \in (0, 1)$ la sovrapposizione delle code sinc dei campioni adiacenti ad alta frequenza o ad alta pendenza genera un'escursione costruttiva (fenomeno di Gibbs):
$$\max_{t} |y_c(t)| > \max_{n} |y_c[n]|$$

In particolare, per sequenze alternate ad alta frequenza (prossime a $f_s/4$ o $f_s/2$), l'overshoot teorico massimo raggiunge:
$$\Delta_{\text{ISP}} = 20 \log_{10}(\sqrt{2}) \approx +3.01\text{ dB}$$
e può superare i $+5\text{ dB}$ su segnali clippati o con transienti impulsivi complessi.

---

## 3. Audit Diagnostico in Sola Lettura sulle 6 Sessioni Renderizzate

La seguente tabella raccoglie i dati estratti direttamente da [`SPRINT3_V2_ABLATION_RESULTS.json`](./SPRINT3_V2_ABLATION_RESULTS.json) e [`SPRINT3_V1_HARMONIZED_RESULTS.json`](./SPRINT3_V1_HARMONIZED_RESULTS.json) per tutte le 24 combinazioni di master renderizzati:

| Sessione | Modalità | Variante | Sample Peak ($1\times$) | True Peak ($4\times$) | Inter-Sample Overshoot | Parametro 41 (`limiter_push`) | Esito Guardrail ($-0.30\text{ dBTP}$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `track01` | `same_track` | **`V0p`** | **$-0.30\text{ dBFS}$** | $+0.26\text{ dBTP}$ | **$+0.56\text{ dB}$** | $0.00\text{ dB}$ | ❌ **FAIL** ($+0.56\text{ dB}$) |
| `track01` | `same_track` | **`V1`** | **$-0.30\text{ dBFS}$** | $+0.29\text{ dBTP}$ | **$+0.59\text{ dB}$** | $0.00\text{ dB}$ | ❌ **FAIL** ($+0.59\text{ dB}$) |
| `track01` | `same_track` | **`V1b`** | **$-0.30\text{ dBFS}$** | $+0.32\text{ dBTP}$ | **$+0.62\text{ dB}$** | $0.00\text{ dB}$ | ❌ **FAIL** ($+0.62\text{ dB}$) |
| `track01` | `same_track` | **`V2`** | **$-0.30\text{ dBFS}$** | $+0.15\text{ dBTP}$ | **$+0.45\text{ dB}$** | $0.00\text{ dB}$ | ❌ **FAIL** ($+0.45\text{ dB}$) |
| `track02` | `same_track` | **`V0p`** | **$-0.30\text{ dBFS}$** | $-0.07\text{ dBTP}$ | **$+0.23\text{ dB}$** | $0.00\text{ dB}$ | ❌ **FAIL** ($+0.23\text{ dB}$) |
| `track02` | `same_track` | **`V1`** | **$-0.30\text{ dBFS}$** | $-0.07\text{ dBTP}$ | **$+0.23\text{ dB}$** | $0.00\text{ dB}$ | ❌ **FAIL** ($+0.23\text{ dB}$) |
| `track02` | `same_track` | **`V1b`** | **$-0.30\text{ dBFS}$** | $-0.07\text{ dBTP}$ | **$+0.23\text{ dB}$** | $0.00\text{ dB}$ | ❌ **FAIL** ($+0.23\text{ dB}$) |
| `track02` | `same_track` | **`V2`** | **$-0.30\text{ dBFS}$** | $+0.04\text{ dBTP}$ | **$+0.34\text{ dB}$** | $0.00\text{ dB}$ | ❌ **FAIL** ($+0.34\text{ dB}$) |
| `track03` | `same_track` | **`V0p`** | **$-0.30\text{ dBFS}$** | $-0.09\text{ dBTP}$ | **$+0.21\text{ dB}$** | $0.00\text{ dB}$ | ❌ **FAIL** ($+0.21\text{ dB}$) |
| `track03` | `same_track` | **`V1`** | **$-0.30\text{ dBFS}$** | $-0.09\text{ dBTP}$ | **$+0.21\text{ dB}$** | $0.00\text{ dB}$ | ❌ **FAIL** ($+0.21\text{ dB}$) |
| `track03` | `same_track` | **`V1b`** | **$-0.30\text{ dBFS}$** | $-0.09\text{ dBTP}$ | **$+0.21\text{ dB}$** | $0.00\text{ dB}$ | ❌ **FAIL** ($+0.21\text{ dB}$) |
| `track03` | `same_track` | **`V2`** | **$-0.30\text{ dBFS}$** | $-0.04\text{ dBTP}$ | **$+0.26\text{ dB}$** | $0.00\text{ dB}$ | ❌ **FAIL** ($+0.26\text{ dB}$) |
| `track01_style` | `stylistic` | **`V0p`** | $-2.42\text{ dBFS}$ | $-2.42\text{ dBTP}$ | $+0.00\text{ dB}$ | $0.00\text{ dB}$ |  **PASS** |
| `track01_style` | `stylistic` | **`V1`** | $-2.46\text{ dBFS}$ | $-2.46\text{ dBTP}$ | $+0.00\text{ dB}$ | $0.00\text{ dB}$ |  **PASS** |
| `track01_style` | `stylistic` | **`V1b`** | $-2.33\text{ dBFS}$ | $-2.10\text{ dBTP}$ | $+0.24\text{ dB}$ | $0.00\text{ dB}$ |  **PASS** |
| `track01_style` | `stylistic` | **`V2`** | $-1.84\text{ dBFS}$ | $-1.63\text{ dBTP}$ | $+0.21\text{ dB}$ | $0.00\text{ dB}$ |  **PASS** |
| `track02_style` | `stylistic` | **`V0p`** | **$-0.30\text{ dBFS}$** | $-0.07\text{ dBTP}$ | **$+0.23\text{ dB}$** | $0.00\text{ dB}$ | ❌ **FAIL** ($+0.23\text{ dB}$) |
| `track02_style` | `stylistic` | **`V1`** | **$-0.30\text{ dBFS}$** | $-0.07\text{ dBTP}$ | **$+0.23\text{ dB}$** | $0.00\text{ dB}$ | ❌ **FAIL** ($+0.23\text{ dB}$) |
| `track02_style` | `stylistic` | **`V1b`** | **$-0.30\text{ dBFS}$** | $-0.08\text{ dBTP}$ | **$+0.22\text{ dB}$** | $0.00\text{ dB}$ | ❌ **FAIL** ($+0.22\text{ dB}$) |
| `track02_style` | `stylistic` | **`V2`** | **$-0.30\text{ dBFS}$** | $-0.06\text{ dBTP}$ | **$+0.24\text{ dB}$** | $0.00\text{ dB}$ | ❌ **FAIL** ($+0.24\text{ dB}$) |
| `track03_style` | `stylistic` | **`V0p`** | **$-0.30\text{ dBFS}$** | $+0.06\text{ dBTP}$ | **$+0.36\text{ dB}$** | $0.00\text{ dB}$ | ❌ **FAIL** ($+0.36\text{ dB}$) |
| `track03_style` | `stylistic` | **`V1`** | **$-0.30\text{ dBFS}$** | $+0.05\text{ dBTP}$ | **$+0.35\text{ dB}$** | $0.00\text{ dB}$ | ❌ **FAIL** ($+0.35\text{ dB}$) |
| `track03_style` | `stylistic` | **`V1b`** | **$-0.30\text{ dBFS}$** | $+0.05\text{ dBTP}$ | **$+0.35\text{ dB}$** | $0.00\text{ dB}$ | ❌ **FAIL** ($+0.35\text{ dB}$) |
| `track03_style` | `stylistic` | **`V2`** | **$-0.30\text{ dBFS}$** | $-0.06\text{ dBTP}$ | **$+0.24\text{ dB}$** | $0.00\text{ dB}$ | ❌ **FAIL** ($+0.24\text{ dB}$) |

### Deduzioni Chiave dall'Audit
1. **Precisione Assoluta del Sample-Peak Limiter:** Nelle 20 tracce che saturano il limiter, il Sample Peak è **esattamente $-0.30\text{ dBFS}$**. Non esiste alcun difetto di tenuta o perdita di controllo del limiter discreto;
2. **Entità dell'Inter-Sample Overshoot:** Lo scarto $\Delta_{\text{ISP}} = \text{True Peak} - \text{Sample Peak}$ è sistematicamente compreso tra **$+0.21\text{ dB}$ e $+0.62\text{ dB}$**;
3. **Parametro 41 Inattivo:** Il valore di `limiter_push_gain_db` è $0.00\text{ dB}$ in tutte le configurazioni (l'ottimizzatore non ha spinto ulteriore guadagno nel limiter, poiché il segnale proveniente dai makeup gain dei compressori e dal saturatore era già sufficiente a raggiungere il target di loudness).

---

## 4. Validazione Sperimentale su Segnali Canonici Sintetici

Per caratterizzare in isolamento il comportamento dell'attuale limiter e del rilevatore ITU-R BS.1770 su forme d'onda elementari, abbiamo eseguito il protocollo diagnostico `scratch/test_canonical_signals_true_peak.py`:

| Segnale di Test Sintetico | Sample Peak Ingresso | True Peak Ingresso | Overshoot Ingresso | Limiter Sample Peak | Limiter True Peak | Limiter Overshoot | Violazione Guardrail ($-0.30\text{ dBTP}$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Singolo Impulso (Dirac)** | $-3.01\text{ dBFS}$ | $-3.01\text{ dBTP}$ | $+0.00\text{ dB}$ | **$-0.30\text{ dBFS}$** | **$-0.30\text{ dBTP}$** | $+0.00\text{ dB}$ | **$0.00\text{ dB}$ (CONFORME)** |
| **2. Tono $12\text{ kHz}$ Off-Grid ($\phi = \pi/4$)** | $-5.91\text{ dBFS}$ | $-2.18\text{ dBTP}$ | $+3.73\text{ dB}$ | **$-2.91\text{ dBFS}$** | **$+0.81\text{ dBTP}$** | $+3.73\text{ dB}$ | ❌ **$+1.11\text{ dB}$** |
| **3. Tono $18\text{ kHz}$ Off-Grid ($\phi = \pi/4$)** | $-3.01\text{ dBFS}$ | $-2.59\text{ dBTP}$ | $+0.42\text{ dB}$ | **$-0.30\text{ dBFS}$** | **$+0.12\text{ dBTP}$** | $+0.42\text{ dB}$ | ❌ **$+0.42\text{ dB}$** |
| **4. Percussive Burst (Fast Decay, $1\text{ kHz}$)** | $-3.00\text{ dBFS}$ | $-2.99\text{ dBTP}$ | $+0.01\text{ dB}$ | **$-0.30\text{ dBFS}$** | **$-0.29\text{ dBTP}$** | $+0.01\text{ dB}$ | **$+0.01\text{ dB}$ (MARGINALE)** |
| **5. Gradino Lookahead (Step Transient)** | $-3.00\text{ dBFS}$ | $-1.51\text{ dBTP}$ | $+1.49\text{ dB}$ | **$-0.30\text{ dBFS}$** | **$+1.19\text{ dBTP}$** | $+1.49\text{ dB}$ | ❌ **$+1.49\text{ dB}$** |
| **6. Stereo Decorrelato (Fase $90^\circ$, $4\text{ kHz}$)** | $-3.01\text{ dBFS}$ | $-2.13\text{ dBTP}$ | $+0.88\text{ dB}$ | **$-0.30\text{ dBFS}$** | **$+0.58\text{ dBTP}$** | $+0.88\text{ dB}$ | ❌ **$+0.88\text{ dB}$** |

### Evidenze Fisiche Rilevate
- **Impulso Isolato:** Nessun overshoot. Quando un singolo campione tocca il ceiling, l'inviluppo sinc ha zeri esatti su tutti gli altri punti di sovracampionamento;
- **Segnali con Componenti ad Alta Frequenza Off-Grid:** Quando campioni consecutivi oscillano ad alta frequenza senza cadere sul massimo esatto del seno continuo, l'interpolazione sinc a $4\times$ ricostruisce il vero vertice analogico, evidenziando un overshoot fino a **$+3.73\text{ dB}$**;
- **Gradino / Transiente Ripido:** Lo scalino produce un forte fenomeno di Gibbs nell'interpolatore sinc a 64 tap, generando un overshoot di **$+1.49\text{ dB}$**.

---

## 5. Opzioni Architetturali di Follow-up per la Delivery Conformant

Per garantire la conformità tassativa allo standard EBU R128 ($\le -0.30\text{ dBTP}$) senza compromettere la stabilità e la velocità del loop di differenziazione automatica PyTorch/C++, si delineano 3 opzioni ingegneristiche a disposizione del comitato di supervisione:

```mermaid
flowchart TD
    subgraph Pipeline Attuale
        A[Catena DDSP C++/PyTorch] --> B[Sample-Peak Limiter a 1x (Ceiling = -0.30 dBFS)]
        B --> C[Master Export a 48 kHz]
        C --> D[True Peak Measure a 4x: -0.09 .. +0.32 dBTP (FAIL)]
    end
    
    subgraph Opzione A: Delivery Headroom / Post-Trim Offline
        C --> E[Misura True Peak 4x BS.1770]
        E -->|Se TP > -0.30 dBTP| F[Post-Gain Trim Scalare: g = min(0 dB, -0.30 - TP)]
        F --> G[Master Finale Conforme <= -0.30 dBTP]
    end
    
    subgraph Opzione B: Oversampled Limiter Detector (4x)
        A --> H[Detector Limiter a 4x Sinc Interpolation]
        H --> I[Gain Envelope calcolato su True Peak Reale]
        I --> J[Master Conforme <= -0.30 dBTP nativo]
    end
```

### Opzione A: Headroom Statico di Sicurezza o Post-Trim di Delivery Offline (Raccomandata)
- **Principio:** Mantenere inalterata la catena DDSP differenziabile a $1\times$ ($48\text{ kHz}$) durante l'ottimizzazione Adam. In fase di export/delivery finale, misurare il True Peak a $4\times$ e applicare un guadagno correttivo lineare scalare $g_{\text{trim}} = \min(0.0\text{ dB}, -0.30\text{ dBTP} - \text{True Peak})$ (pari a circa $-0.4\text{ a }-0.6\text{ dB}$ sulle tracce critiche), oppure abbassare il ceiling discreto del limiter a $-1.00\text{ dBFS}$ (come consentito dalle raccomandazioni EBU R128 per materiale distribuito su formati lossy).
- **Vantaggi:**
  - Zero modifiche al DSP core C++/PyTorch congelato;
  - Nessun incremento nei tempi di training Adam;
  - Conformità $100\%$ garantita su ogni brano.
- **Svantaggi:**
  - Riduzione marginale ($0.4\text{–}0.6\text{ LU}$) della loudness integrata finale su tracce dense.

### Opzione B: True Peak Limiter con Detector Sovracampionato a $4\times$ nel DSP Core
- **Principio:** Riscrivere `DifferentiableLookaheadLimiter` e il corrispettivo C++ `LookaheadLimiter.cpp` inserendo un banco di filtri polifase a $4\times$ sul ramo di sidechain/detection, in modo che il min-filter operi direttamente sul picco interpolato a $192\text{ kHz}$.
- **Vantaggi:**
  - Il limiter sopprime i picchi inter-sample all'origine senza abbassare l'intero brano.
- **Svantaggi:**
  - Richiede la rottura del congelamento del DSP core C++ e PyTorch;
  - Incremento del costo computazionale del limiter di circa $4\times$;
  - Necessità di riscrivere la funzione di backward autograd C++ per l'inviluppo sovracampionato.

### Opzione C: Disaccoppiamento Funzionale (Training Proxy + Mastering Render Finale a 2 Stadi)
- **Principio:** Considerare la catena DDSP come un ottimizzatore di *timbro e macro-dinamica* (EQ, multiband comp, imager, saturatore). Il brickwall limiting finale di conformità True Peak viene affidato a un plugin/modulo C++ offline dedicato con oversampling e dithering, escluso dal grafo dei gradienti Adam.
- **Vantaggi:**
  - Perfetta corrispondenza con le pratiche dell'industria professionale (dove i limiting processors differenziabili sono separati dal mastering finalizer).

---

## 6. Conclusioni

1. **Assoluta Integrità del DSP Core:** Il limiter attuale è un perfetto *Sample-Peak Brickwall Limiter*, privo di falle algoritmiche;
2. **Natura della Violazione:** Il superamento del guardrail di $-0.30\text{ dBTP}$ è provocato al $100\%$ dall'inter-sample peak overshoot derivante dalla ricostruzione continua di materiale compresso e denso;
3. **Indipendenza da V3a:** Questo studio conferma formalmente che la risoluzione del True Peak appartiene allo stadio di **delivery/export** e non ha alcuna relazione con lo Step `V3a` (prior sulla risposta in frequenza dell'EQ).
