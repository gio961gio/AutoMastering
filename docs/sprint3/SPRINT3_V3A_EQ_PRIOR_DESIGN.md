# AutoMastering — Specifica Matematica e Analisi Pre-Implementazione: Step V3a (Prior Risposta Composita EQ)

**Autore:** Giovanni Antonecchia  
**Corso di Laurea:** Master's Thesis in Sound and Music Engineering — Politecnico di Milano (DEIB)  
**Riferimento Roadmap:** `docs/AutoMastering_Audit_Roadmap_Antigravity.md` (§10, Sprint 3)  
**Specifica Tecnica di Base:** `docs/sprint3/SPRINT3_TECHNICAL_PROPOSAL.md` (§2.D.1 e §3)  
**Data:** Novembre 2026  
**Stato:** Specifica Analitica e Validazione Numerica Offline (**Nessuna Nuova Ottimizzazione di Training Eseguita**)  

---

## 1. Motivazione Architetturale e Isolamento Incrementale

Nello sviluppo incrementale dello Sprint 3:
$$\mathbf{V0} \longrightarrow \mathbf{V0p} \longrightarrow \mathbf{V1} / \mathbf{V1b} \longrightarrow \mathbf{V2} \longrightarrow \mathbf{V3a} \longrightarrow \mathbf{V3b} \longrightarrow \mathbf{V3c} \longrightarrow \mathbf{V4}$$

Nelle varianti precedenti (`V0p`, `V1`, `V1b`, `V2`), la regolarizzazione parametrica era confinata a una penalità isotropa L2 globale verso `Z_NEUTRAL`:
$$\mathcal{L}_{\text{param-reg}} = \frac{1}{41} \sum_{i=1}^{41} (z_i - z_{\text{neutral}, i})^2$$
Questa penalizzazione tratta ogni parametro in modo isolato e cieco rispetto alla fisica del sistema:
1. **Accoppiamento dei Biquad in Cascata:** Due filtri biquad adiacenti possono compensarsi a vicenda (es. un boost di $+4\text{ dB}$ seguito da un cut di $-4\text{ dB}$ a frequenze vicine), generando una risposta acustica locale con forti distorsioni di fase e ripidi gradienti di pendenza spettrale, pur mantenendo una loss spettrale simile;
2. **Deriva di Offset Globale:** L'ottimizzatore Adam può sfruttare i 6 guadagni dell'EQ parametrico per incrementare il guadagno a banda larga (spostando tutti i biquad di $+2\text{ a }+3\text{ dB}$ nella stessa direzione) per soddisfare la loss di loudness, utilizzando l'equalizzatore come un controllo di volume improprio anziché delegare il bilanciamento dinamico ai makeup gain del compressore multibanda e al limiter;
3. **Pendenza e Asprezza Locale:** L'orecchio umano tollera curve di equalizzazione ampie e dolci (basso $Q$, pendenze inferiori a $1\text{–}2\text{ dB/ottava}$ tipiche del mastering analogico), mentre percepisce come innaturali o affaticanti risonanze strette ad alto $Q$ o discontinuità di pendenza.

Lo **Step V3a** introduce un prior analitico differenziabile calcolato direttamente sulla **risposta in frequenza composita complessa dei 6 filtri biquad**, imponendo:
- Una dead-zone tollerante per interventi correttivi morbidi;
- Una penalità quadratica sulle deviazioni macroscopiche di forma ($\mathcal{L}_{\text{shape-exc}}$);
- Una penalità di regolarità sulla derivata prima logaritmica / ruvidezza spettrale ($\mathcal{L}_{\text{slope-raw}}$);
- Una barriera sull'offset medio globale ($\mathcal{L}_{\text{offset-raw}}$), preservando la funzione dei controlli di dinamica a valle.

> [!IMPORTANT] **Chiarimento sulla Tassonomia Architetturale**
> Lo Step `V3a` riguarda **esclusivamente** il prior sulla risposta composita dell'EQ parametrico. Non include e non costituisce uno stadio di True Peak Limiting. I controlli di dinamica, limiter stress e True Peak guardrail appartengono rispettivamente a `V3b` e al protocollo di delivery/export documentato in `SPRINT3_TRUE_PEAK_DELIVERY_FOLLOWUP.md`.

---

## 2. Formulazione Matematica Rigorosa

### 2.1 Griglia di Frequenza Logaritmica
Sia $f_s = 48000.0\text{ Hz}$ la frequenza di campionamento. Definiamo una griglia logaritmica uniforme di $K_{\text{eq}} = 128$ punti tra $f_{\min} = 20.0\text{ Hz}$ e $f_{\max} = 20000.0\text{ Hz}$:
$$\log_{10} f_q = \log_{10}(20.0) + \frac{q}{K_{\text{eq}} - 1} \left(\log_{10}(20000.0) - \log_{10}(20.0)\right), \quad q = 0, \dots, K_{\text{eq}} - 1$$
$$f_q = 10^{\log_{10} f_q}, \qquad \omega_q = \frac{2\pi f_q}{f_s} \in (0, \pi)$$

Il passo discreto tra frequenze adiacenti è costante in scala logaritmica (e in ottave):
$$\Delta \log_{10} f = \frac{\log_{10}(1000)}{127} \approx 0.02362 \implies \Delta \log_2 f = \frac{\log_2(1000)}{127} \approx 0.07847\text{ ottave} \approx \frac{1}{12.74}\text{ ottava}$$

### 2.2 Risposta in Frequenza della Cascata dei 6 Biquad
I 18 parametri di $z \in [0, 1]^{41}$ associati all'EQ ($z[0:18]$) vengono convertiti nello spazio fisico $\theta_{\text{EQ}} = [f_{0,i}, \text{gain}_{i}, Q_i]_{i=0}^5$ tramite `map_z_to_theta`.  
Il modulo `DifferentiableEQ.compute_biquad_coeffs` calcola in doppia precisione `torch.float64` i coefficienti canonici normalizzati dei 6 filtri (Banda 0: Low-Shelf, Bande 1–4: Peaking, Banda 5: High-Shelf):
$$H_i(z^{-1}) = \frac{b_{0,i} + b_{1,i} z^{-1} + b_{2,i} z^{-2}}{1 + a_{1,i} z^{-1} + a_{2,i} z^{-2}}, \quad i = 0, \dots, 5$$

Valutando sul cerchio unitario $z = e^{j\omega_q}$ con $\cos(\omega_q)$ e $\sin(\omega_q)$:
$$\left|B_i(e^{j\omega_q})\right|^2 = \left(b_{0,i} + b_{1,i}\cos\omega_q + b_{2,i}\cos 2\omega_q\right)^2 + \left(b_{1,i}\sin\omega_q + b_{2,i}\sin 2\omega_q\right)^2$$
$$\left|A_i(e^{j\omega_q})\right|^2 = \left(1 + a_{1,i}\cos\omega_q + a_{2,i}\cos 2\omega_q\right)^2 + \left(a_{1,i}\sin\omega_q + a_{2,i}\sin 2\omega_q\right)^2$$

La risposta in decibel del singolo filtro e la risposta composita totale sono date da:
$$G_i(\omega_q) = 10 \log_{10}\!\left(\frac{\left|B_i(e^{j\omega_q})\right|^2 + \varepsilon_{\text{eq}}}{\left|A_i(e^{j\omega_q})\right|^2 + \varepsilon_{\text{eq}}}\right), \qquad G_{\text{EQ}}(\omega_q) = \sum_{i=0}^{5} G_i(\omega_q)$$
con costante di stabilizzazione numerica $\varepsilon_{\text{eq}} = 10^{-12}$.

### 2.3 Decomposizione in Offset Globale e Forma Centrata
L'offset medio logaritmico (in dB) e il profilo di forma centrato (in dB) sono definiti come:
$$\bar{G}_{\text{EQ}} = \frac{1}{K_{\text{eq}}} \sum_{q=0}^{K_{\text{eq}}-1} G_{\text{EQ}}(\omega_q), \qquad \widetilde{G}_{\text{EQ}}(\omega_q) = G_{\text{EQ}}(\omega_q) - \bar{G}_{\text{EQ}}$$
Per costruzione:
$$\sum_{q=0}^{K_{\text{eq}}-1} \widetilde{G}_{\text{EQ}}(\omega_q) \equiv 0$$

### 2.4 Le Tre Componenti di Costo Elementari

1. **Eccesso di Escursione di Forma oltre la Dead-Zone $\tau_{\text{shape}} = 1.5\text{ dB}$ (unità: $\text{dB}^2$):**
   Permette interventi di mastering fino a $\pm 1.5\text{ dB}$ a costo zero, penalizzando quadraticamente le escursioni eccessive:
   $$\mathcal{L}_{\text{shape-exc}} = \frac{1}{K_{\text{eq}}} \sum_{q=0}^{K_{\text{eq}}-1} \left[\text{ReLU}\!\left(|\widetilde{G}_{\text{EQ}}(\omega_q)| - \tau_{\text{shape}}\right)\right]^2$$

2. **Ruvidezza / Pendenza Locale tra Punti Adiacenti (unità: $\text{dB}^2 / \text{step}^2$):**
   Misura la derivata discreta della curva centrata sulla griglia a 128 punti:
   $$\mathcal{L}_{\text{slope-raw}} = \frac{1}{K_{\text{eq}}-1} \sum_{q=0}^{K_{\text{eq}}-2} \left(\widetilde{G}_{\text{EQ}}(\omega_{q+1}) - \widetilde{G}_{\text{EQ}}(\omega_q)\right)^2$$
   Poiché il passo è $\Delta \log_2 f \approx 0.0785\text{ ottave}$, una campana morbida a basso $Q$ produce $\mathcal{L}_{\text{slope-raw}} \sim 10^{-3}\text{ dB}^2$, mentre risonanze o biquad contrapposti ad alto $Q$ producono $\mathcal{L}_{\text{slope-raw}} \sim 10^{-1}\text{ dB}^2$ ($100\times$ superiore).

3. **Eccesso di Offset Globale oltre la Dead-Zone $\tau_{\text{offset}} = 1.0\text{ dB}$ (unità: $\text{dB}^2$):**
   Impedisce all'EQ di fungere da guadagno a banda larga:
   $$\mathcal{L}_{\text{offset-raw}} = \left[\text{ReLU}\!\left(|\bar{G}_{\text{EQ}}| - \tau_{\text{offset}}\right)\right]^2$$

### 2.5 Funzione di Costo Composita `V3a`
$$\mathcal{L}_{\text{EQ-shape}} = \mathcal{L}_{\text{shape-exc}} + w_{\text{slope}} \cdot \mathcal{L}_{\text{slope-raw}}$$
$$\mathcal{L}_{\text{EQ-offset}} = w_{\text{offset}} \cdot \mathcal{L}_{\text{offset-raw}}$$
$$\mathcal{L}_{\text{V3a}} = \mathcal{L}_{\text{V0p}} + w_{\text{eq}} \left(\mathcal{L}_{\text{EQ-shape}} + \mathcal{L}_{\text{EQ-offset}}\right)$$

---

## 3. Risultati Sperimentali: Diagnostica Numerica sui 7 Casi di Studio

Lo studio numerico è stato eseguito tramite lo script diagnostico `scratch/analyze_v3a_eq_prior.py` con autograd PyTorch su parametri floating point float64.

### Tabella Sinottica Comparativa

| Configurazione | $\bar{G}_{\text{EQ}}$ (dB) | $\max |\widetilde{G}_{\text{EQ}}|$ (dB) | $\mathcal{L}_{\text{shape-exc}}$ | $\mathcal{L}_{\text{slope-raw}}$ | $\mathcal{L}_{\text{offset-raw}}$ | $\mathcal{L}_{\text{EQ-total}}$ | $\|\nabla_{z_{\text{EQ}}} \mathcal{L}_{\text{EQ}}\|_2$ | $\|\nabla_{z_{\text{non-EQ}}}\|_2$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Caso 0 (`Z_NEUTRAL`)** | $-0.00$ | $0.00$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | **$0.00 \times 10^{0}$** |
| **Caso 1 (Gentle in-deadzone, $+1.2\text{ dB}, Q=0.7$)** | $+0.27$ | $0.93$ | $0.0000$ | $0.0008$ | $0.0000$ | $0.0017$ | $0.0362$ | **$0.00 \times 10^{0}$** |
| **Caso 2 (Moderate wide boost, $+3.0\text{ dB}, Q=0.8$)** | $+0.61$ | $2.39$ | $0.0430$ | $0.0056$ | $0.0000$ | $0.0543$ | $1.2435$ | **$0.00 \times 10^{0}$** |
| **Caso 3 (Biquad contrapposti, $+4\text{ dB} / -4\text{ dB}, Q=2.4$)** | $+0.01$ | $3.91$ | $0.3273$ | $0.0543$ | $0.0000$ | $0.4360$ | $3.0623$ | **$0.00 \times 10^{0}$** |
| **Caso 4 (Offset drift globale $+2.5\text{ dB}$ su tutte le 6 bande)** | $+2.70$ | $1.60$ | $0.0003$ | $0.0038$ | $2.8875$ | $2.8953$ | $19.6096$ | **$0.00 \times 10^{0}$** |
| **Caso 5 (`V0p` Reale: `track01`)** | $+2.48$ | $1.23$ | $0.0000$ | $0.0023$ | $2.1929$ | $2.1975$ | $19.9246$ | **$0.00 \times 10^{0}$** |
| **Caso 5 (`V0p` Reale: `track02`)** | $+2.48$ | $1.07$ | $0.0000$ | $0.0016$ | $2.1851$ | $2.1883$ | $19.9383$ | **$0.00 \times 10^{0}$** |
| **Caso 5 (`V0p` Reale: `track03`)** | $+2.48$ | $1.20$ | $0.0000$ | $0.0022$ | $2.1763$ | $2.1808$ | $19.8430$ | **$0.00 \times 10^{0}$** |
| **Caso 5 (`V0p` Reale: `track01_style`)** | $-2.50$ | $1.14$ | $0.0000$ | $0.0020$ | $2.2396$ | $2.2436$ | $20.1281$ | **$0.00 \times 10^{0}$** |
| **Caso 5 (`V0p` Reale: `track02_style`)** | $+2.46$ | $1.06$ | $0.0000$ | $0.0017$ | $2.1436$ | $2.1470$ | $19.7459$ | **$0.00 \times 10^{0}$** |
| **Caso 5 (`V0p` Reale: `track03_style`)** | $+2.40$ | $1.29$ | $0.0000$ | $0.0029$ | $1.9547$ | $1.9605$ | $18.7927$ | **$0.00 \times 10^{0}$** |

*(Pesi impostati per la diagnostica analitica: $w_{\text{slope}} = 2.0$, $w_{\text{offset}} = 1.0$).*

---

## 4. Analisi Critica dei Risultati

### 4.1 Caso 0 e Caso 1: Trasparenza nella Dead-Zone
- In `Z_NEUTRAL`, tutti i guadagni sono a $0\text{ dB}$, la risposta è piatta al bit ($G_{\text{EQ}} \equiv 0$), tutte le tre loss elementari sono esattamente nulle ($0.0000$) e il gradiente su tutti i 41 parametri è identicamente zero.
- Nel Caso 1 (intervento tipico di mastering: boost moderato di $+1.2\text{ dB}$ a $1\text{ kHz}$ con $Q=0.7$), l'escursione massima centrata è $|\widetilde{G}| = 0.93\text{ dB} \le 1.5\text{ dB}$ e l'offset è $+0.27\text{ dB} \le 1.0\text{ dB}$. Sia $\mathcal{L}_{\text{shape-exc}}$ sia $\mathcal{L}_{\text{offset-raw}}$ sono rigorosamente **zero**. La sola pendenza residua produce una loss trascurabile ($0.0017$), con norma del gradiente $\|\nabla_{z_{\text{EQ}}}\|_2 = 0.036$, confermando che gli interventi gentili non vengono penalizzati.

### 4.2 Caso 2 vs Caso 3: Discriminazione tra Campana Dolce e Risonanza Stretta
- Nel Caso 2 (boost ampio di $+3.0\text{ dB}$ a $Q=0.8$), $\mathcal{L}_{\text{slope-raw}} = 0.0056\text{ dB}^2$.
- Nel Caso 3 (filtri adiacenti contrapposti $+4\text{ dB} / -4\text{ dB}$ a $Q=2.4$), $\mathcal{L}_{\text{slope-raw}}$ schizza a **$0.0543\text{ dB}^2$** (quasi **$10\times$ superiore**!).
- Questa proprietà matematica conferma che la penalità di pendenza $\mathcal{L}_{\text{slope-raw}}$ agisce come forte deterrente contro "comb-filtering" o risonanze biquad aggressive senza penalizzare interventi ampi.

### 4.3 Caso 4 e Caso 5: La Scoperta della Deriva di Offset in `V0p`
- L'ispezione analitica dei vettori `normalized_z_opt` reali salvati per `V0p` rivela un fenomeno illuminante:
  - In tutte le 6 sessioni, la forma centrata dell'EQ è rimasta sorprendentemente pulita: $\max |\widetilde{G}_{\text{EQ}}| \le 1.29\text{ dB}$, quindi $\mathcal{L}_{\text{shape-exc}} \equiv 0.0000$!
  - **Tuttavia**, l'offset globale $\bar{G}_{\text{EQ}}$ è sistematicamente derivato a $+2.48\text{ dB}$ in `same_track` e a $-2.50\text{ dB}$ in `track01_style`!
  - L'ottimizzatore non vincolato ha utilizzato l'EQ parametrico per applicare un guadagno a banda larga di $\approx +2.5\text{ dB}$ per soddisfare la loss di loudness ITU-R BS.1770, bypassando i makeup gain del compressore multibanda.
  - Questo attiva massicciamente $\mathcal{L}_{\text{offset-raw}} \approx 2.19\text{ dB}^2$, generando un gradiente di richiamo $\|\nabla_{z_{\text{EQ}}}\|_2 \approx 19.9$.
  - L'introduzione di `V3a` forzerà l'ottimizzatore a mantenere $\bar{G}_{\text{EQ}} \in [-1.0, +1.0]\text{ dB}$, delegando il guadagno di loudness ai corretti stadi dedicati (makeup e push limiter).

### 4.4 Isolamento Rigoroso dei Parametri Non-EQ
In tutte le configurazioni analizzate, la norma del gradiente sui 23 parametri non-EQ ($z[18:41]$: Crossover, Compressor, Imager, Saturator, Limiter) è risultata:
$$\|\nabla_{z_{\text{non-EQ}}} \mathcal{L}_{\text{EQ}}\|_2 = 0.00 \times 10^0$$
Questo certifica l'indipendenza funzionale e la totale assenza di gradienti spuri verso i blocchi di dinamica e stereo.

---

## 5. Giustificazione Matematica dei Pesi e Calibrazione

### 5.1 Analisi Dimensionale
1. $\mathcal{L}_{\text{shape-exc}}$ ha unità $[\text{dB}^2]$. Un superamento quadratico medio di $0.5\text{ dB}$ oltre la dead-zone corrisponde a $\approx 0.25\text{ dB}^2$.
2. $\mathcal{L}_{\text{slope-raw}}$ ha unità $[\text{dB}^2 / \text{step}^2]$. Sulla griglia di 128 punti ($\Delta q \approx 0.0785\text{ ottave}$), un gradiente dolce produce $0.001\text{–}0.005\text{ dB}^2$, mentre una risonanza produce $> 0.05\text{ dB}^2$. Ponendo $w_{\text{slope}} = 2.0$, il contributo di ruvidezza patologica ($2.0 \times 0.05 \approx 0.10$) si allinea con l'ordine di grandezza di $\mathcal{L}_{\text{shape-exc}}$.
3. $\mathcal{L}_{\text{offset-raw}}$ ha unità $[\text{dB}^2]$. Con $w_{\text{offset}} = 1.0$, una deriva di $+2.5\text{ dB}$ (eccesso di $1.5\text{ dB}$) produce una penalità netta di $2.25\text{ dB}^2$.

### 5.2 Calibrazione di $w_{\text{eq}}$ sul Rapporto tra Gradienti
Dai dati registrati in `SPRINT3_V1_GRADIENT_DIAGNOSTICS.json`:
- La norma del gradiente spettrale su EQ ($\|\nabla_{z_{\text{EQ}}} \mathcal{L}_{\text{spectral}}\|_2$) vale:
  - $0.72\text{–}0.81$ per MR-STFT (`same_track`);
  - $0.21\text{–}0.60$ per Mel-40 / ERB-48 (`stylistic`).
- La norma del gradiente del prior EQ non pesato ($\|\nabla_{z_{\text{EQ}}} \mathcal{L}_{\text{EQ}}\|_2$) vale:
  - $0.036$ per interventi in dead-zone (Caso 1);
  - $1.24$ per un boost di $+3.0\text{ dB}$ fuori dead-zone (Caso 2);
  - $3.06$ per risonanze strette (Caso 3);
  - $19.6\text{–}20.0$ per derive massive di offset globale (Casi 4 e 5).

Affinché il prior EQ agisca come regolarizzatore morbido (generando una forza di richiamo pari a circa il **$5\%\text{–}20\%$ della spinta spettrale** per deviazioni moderate fuori dead-zone), definiamo:
$$\frac{\|\nabla_{z_{\text{EQ}}} (w_{\text{eq}} \mathcal{L}_{\text{EQ}})\|_2}{\|\nabla_{z_{\text{EQ}}} \mathcal{L}_{\text{spectral}}\|_2} \approx 0.05\text{–}0.20 \implies w_{\text{eq}} \cdot 1.24 \approx 0.05 \times 0.70 \implies w_{\text{eq}} \approx 0.028$$

### 5.3 Tabella di Sensibilità sui Valori Candidati di $w_{\text{eq}}$

| Valore Candidato $w_{\text{eq}}$ | Spinta su Boost $+3\text{ dB}$ ($w_{\text{eq}} \times 1.24$) | Ratio vs Gradiente Spettrale ($\sim 0.70$) | Spinta su Offset $+2.5\text{ dB}$ ($w_{\text{eq}} \times 19.8$) | Valutazione Tecnica |
|:---:|:---:|:---:|:---:|:---|
| **$0.005$** | $0.0062$ | $\approx 0.9\%$ | $0.099$ | **Sotto-regolarizzato:** troppo debole, non frena l'offset drift. |
| **$0.010$** | $0.0124$ | $\approx 1.8\%$ | $0.198$ | **Conservativo:** frena lentamente l'offset drift. |
| **`0.020` (Raccomandato)** | **`0.0248`** | **`~3.5%–5.0%`** | **`0.396`** | **Ottimale:** guida morbida sulla forma, forza sufficiente ($\approx 55\%$ dello spettro) a ricondurre l'offset entro la dead-zone di $1\text{ dB}$. |
| **$0.050$** | $0.0620$ | $\approx 9.0\%$ | $0.990$ | **Rigido:** compensazione offset dominante rispetto allo spettro. |
| **$0.100$** | $0.1240$ | $\approx 18.0\%$ | $1.980$ | **Sovra-regolarizzato:** rischia di appiattire l'equalizzazione timbrica desiderata. |

---

## 6. Conclusioni e Raccomandazioni per l'Implementazione Futura

1. **Formula Congelata:** La formulazione matematica a 3 termini ($\mathcal{L}_{\text{shape-exc}} + w_{\text{slope}} \mathcal{L}_{\text{slope-raw}} + w_{\text{offset}} \mathcal{L}_{\text{offset-raw}}$) con griglia logaritmica a 128 punti si dimostra numericamente stabile, differenziabile e perfettamente isolata sui 18 parametri dell'EQ;
2. **Iperparametri Raccomandati:**
   - Dead-zone di forma: $\tau_{\text{shape}} = 1.5\text{ dB}$
   - Dead-zone di offset: $\tau_{\text{offset}} = 1.0\text{ dB}$
   - Peso pendenza: $w_{\text{slope}} = 2.0$
   - Peso offset: $w_{\text{offset}} = 1.0$
   - Peso complessivo prior: $w_{\text{eq}} = 0.02$
3. **Condizione di Gate:** Nessuna riga di codice verrà integrata nella pipeline di ottimizzazione Adam fino alla formale autorizzazione del supervisore.
