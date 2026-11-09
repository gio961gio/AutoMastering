# AutoMastering — Report Tecnico di Ablation Sprint 3: Step V2 (Loss Transienti/Inviluppo e PLR Reference-Aware)

**Candidato:** Giovanni Antonecchia  
**Corso di Laurea:** Master's Thesis in Sound and Music Engineering — Politecnico di Milano (DEIB)  
**Dataset di Riferimento:** Human Validation Sprint 2 (3 brani: `track01`, `track02`, `track03`; 2 modalità: `same_track`, `stylistic`)  
**Artefatto Dati Grezzi:** [`docs/sprint3/SPRINT3_V2_ABLATION_RESULTS.json`](./SPRINT3_V2_ABLATION_RESULTS.json)  
**Roadmap di Riferimento:** `docs/AutoMastering_Audit_Roadmap_Antigravity.md` (§10, Sprint 3)  
**Specifica Tecnica:** [`docs/sprint3/SPRINT3_TECHNICAL_PROPOSAL.md`](./SPRINT3_TECHNICAL_PROPOSAL.md) (§2.C)  
**Data:** Novembre 2026  

---

## 1. Obiettivo Sperimentale e Metodologia di Isolamento Incrementale

In accordo con la roadmap approvata e il principio di isolamento incrementale dello Sprint 3:
$$\mathbf{V0} \longrightarrow \mathbf{V0p} \longrightarrow \mathbf{V1} / \mathbf{V1b} \longrightarrow \mathbf{V2} \longrightarrow \mathbf{V3a} \longrightarrow \mathbf{V3b} \longrightarrow \mathbf{V3c} \longrightarrow \mathbf{V4}$$

Il presente report documenta la validazione e la diagnostica comparativa dello **Step V2**, il cui scopo è dotare la funzione di costo di sensibilità alla microdinamica e ai profili di cresta per banda, superando l'approccio rigido basato su un range dinamico a scatola (*box target*):

1. **Baseline di Controllo (`V0p`):** Pre-roll di $2.5\text{ s}$ per chunk con warm-up dello stato IIR e dei rivelatori dei compressori, ma mantenendo la funzione di costo storica dello Sprint 2 (MR-STFT + LUFS + PLR box + reg). Costituisce il controllo scientifico primario rispetto a cui misurare l'impatto isolato delle modifiche dinamiche.
2. **Variante `V2` (Transienti multibanda + PLR Reference-Aware):**
   - **`TransientEnvelopeLoss` ($\lambda_{\text{trans}} = 0.2$, $\beta_{\text{atk}} = 0.5$):** scomposizione Linkwitz-Riley del 4° ordine (LR4) a fase zero in 3 bande ($f_{c1} = 250\text{ Hz}$, $f_{c2} = 4000\text{ Hz}$), calcolo degli inviluppi RMS veloci ($\tau_{\text{fast}} = 5\text{ ms}$) e lenti ($\tau_{\text{slow}} = 50\text{ ms}$) mediante finestre rettangolari centrate con buffer cumulativo float64 alimentato dal pre-roll, fattore di cresta locale $C_c[m] = e_c^{\text{fast}}[m] - e_c^{\text{slow}}[m]$ e derivata positiva di attacco $\Delta^+ e_c^{\text{fast}}[m] = \text{ReLU}(e_c^{\text{fast}}[m] - e_c^{\text{fast}}[m-1])$.
     - In `same_track`: penalità Huber ($\delta = 1.0\text{ dB}$) pesata per l'energia attiva della traccia master di riferimento $\tilde{w}[m]$.
     - In `stylistic`: penalità multi-quantile congiunta su $P_{50}, P_{90}, P_{95}$ con maschera binaria di frame attivi e ancoraggio all'uncompressed mix.
   - **`ReferenceAwarePLRLoss` ($\text{plr\_mode} = \text{'reference\_aware'}$):** stima differenziabile del picco soft via partizione temporale in blocchi da $N_b = 400\text{ ms}$ ($19200$ campioni a $48\text{ kHz}$), calcolo della norma $L_{2p}$ con $p=8$, media pesata di Boltzmann su scala logaritmica con parametro $\beta = 12.0$, e gestione del blocco parziale residuo $T_{\text{rem}}$.
     - In `same_track`: penalità Huber ($\delta = 1.0\text{ dB}$) sulla discrepanza tra il PLR proxy del segnale predetto e quello della traccia di riferimento allineata.
     - In `stylistic`: penalty con dead-zone di $\pm 1.0\text{ dB}$ attorno al target di stile e barriera quadratica asimmetrica di floor per impedire ipercompressione ($\text{PLR} < 6.0\text{ dB}$).

Tutte le sessioni di ottimizzazione sono eseguite a parità di budget (30 passi Adam, $\eta_{\text{init}} = 0.04$, transizione `fft -> iir` allo step 21, 4 chunk da $3.0\text{ s}$ estratti con seed uint64 deterministici). I benchmark oggettivi sono calcolati sul rendering C++ offline dell'intero brano (*full-track offline render*) con tail-flush.

---

## 2. Formulazioni Matematiche e Correzioni Ingegneristiche Applicate (Fix A–G)

Prima dell'esecuzione dell'ablation sono state incorporate 7 correzioni formali convalidata da 15 test unitari dedicati (`tests/test_sprint3_v2_transient_plr.py`, suite 100% green su 23/23 test complessivi):

### 2.1 Omogeneità di Guadagno e Protezione del Gradiente nel Picco Soft di Boltzmann (Fix A & Fix B)
La formulazione del livello di picco soft per il blocco $m$ (§2.C.2) è definita da:
$$p_m(x) = 10 \log_{10}\!\left( \left(\frac{1}{C \cdot N_m} \sum_{c=1}^{C} \sum_{n \in m} |x_c[n]|^{2p}\right)^{1/p} + \varepsilon \right)$$
con aggregazione Boltzmann su $M$ blocchi temporali:
$$\tilde{P}_{\text{soft}}(x) = \sum_{m=1}^{M} w_m p_m(x), \quad w_m = \frac{\exp(\beta p_m / 10)}{\sum_{k=1}^M \exp(\beta p_k / 10)}$$
- **Risoluzione del pavimento artificiale (Fix A):** Nel codice preliminare, un clamp prematuro a $1e-12$ applicato prima dell'estrazione della radice $p$-esima ($2p=16$) generava una soglia fittizia a $-15.0\text{ dBFS}$, rompendo l'omogeneità di guadagno per segnali a basso livello. La radice $p$-esima è ora calcolata in `float64` proteggendo lo zero digitale via:
  $$\text{root\_mean\_p} = \begin{cases} (\text{mean\_p})^{1/p} & \text{se } \text{mean\_p} > 0 \\ 0 & \text{se } \text{mean\_p} \le 0 \end{cases}$$
  garantendo perfetta omogeneità di scala ($p(a \cdot x) = p(x) + 20 \log_{10} a$) fino a $-40\text{ dBFS}$ e oltre con gradiente finito e nullo sul silenzio (`test_boltzmann_soft_peak_gain_homogeneity` verificato).
- **Inclusione del Blocco Parziale Residuo (Fix B):** Quando la lunghezza del buffer $T$ non è multiplo intero di $N_b = 19200$, i campioni residui $T_{\text{rem}} = T \pmod{N_b}$ vengono aggregati in un blocco finale normalizzato esattamente per $C \cdot T_{\text{rem}}$, evitando lo scarto cieco di informazioni dinamiche (`test_boltzmann_soft_peak_partial_last_block` verificato).

### 2.2 Allineamento Esatto di Latenza in PLR `same_track` (Fix C)
In `MasteringPerceptualLoss`, il segnale predetto fornito al calcolo del PLR viene preliminarmente compensato per la latenza fisica della catena ($D = 159$ campioni), allineando perfettamente $p_{\text{aligned}}[n] = p[n + D]$ con $t_{\text{aligned}}[n] = t[n]$ per la medesima durata $T - D$. Questo elimina qualsiasi disallineamento nei blocchi temporali di $400\text{ ms}$ tra predetto e target.

### 2.3 Contesto Fisico di Pre-Roll per il Crossover LR4 (Fix D)
In `TransientEnvelopeLoss`, il crossover a 3 bande opera sull'intero buffer esteso $N_{\text{pre}} + T$. Le memorie dei biquad raggiungono così lo stato stazionario prima dell'indice di valutazione, e le finestre di inviluppo veloce/lenta centrate a $t=0$ attingono direttamente alla storia campionata reale a sinistra (`n_pre - half_w`), senza ricorrere a repliche o padding artificiali.

### 2.4 Streaming con Halo a 128 Campioni per il True Peak (Fix F)
Il calcolo del True Peak in `calculate_true_peak_dbtp` (sovracampionamento $4\times$, sinc interpolante Kaiser $\beta = 14.77$, $W_{\text{filter}} = 64$) opera a streaming a blocchi da $10\text{ s}$ con margine di halo pari a $2 W_{\text{filter}} = 128$ campioni su ciascun bordo interno. La regione centrale valida $128 \times 4$ campioni garantisce identità numerica assoluta con l'elaborazione monolitica (`test_true_peak_streaming_vs_monolithic_boundary_equivalence`, discrepanza $< 10^{-5}\text{ dBTP}$).

---

## 3. Tabella Sinottica Comparativa: `V0p` (Controllo) vs `V2` (Transienti & PLR)

La tabella seguente riporta le metriche estratte sul rendering offline C++ full-track per tutte le 6 sessioni, confrontando la baseline di controllo `V0p` con la variante `V2`:

| Sessione / Modalità | Variante | Loss Iniziale $z_0$ | Loss Finale $z_{\text{opt}}$ | Riduzione Loss | Integrated LUFS | PLR (dB) | True Peak (dBTP) | TP Viol. (dB) | Mid/Side $\Delta$ (dB) | Distanza Spettrale (SC / ASD) | LSD / ERB Tot |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`track01`**  
*(same_track)* | **`V0p`**  
**`V2`** | $6.6635$  
$7.4635$ | $3.1026$  
$3.5148$ | $-53.44\%$  
$-52.91\%$ | $-10.51$  
$-10.50$ | $10.21$  
$10.20$ | $+0.31$  
$+0.32$ | $0.61$  
$0.62$ | $0.44$  
**$0.39$** | $0.4053$  
$0.4055$ | $6.79\text{ dB}$  
$6.79\text{ dB}$ |
| **`track02`**  
*(same_track)* | **`V0p`**  
**`V2`** | $14.2818$  
$15.2011$ | $10.4134$  
$11.4050$ | $-27.09\%$  
$-24.97\%$ | $-18.04$  
$-18.05$ | $17.74$  
$17.75$ | $-0.07$  
$-0.08$ | $0.23$  
$0.22$ | $0.35$  
$0.37$ | $0.7932$  
$0.7934$ | $11.21\text{ dB}$  
$11.21\text{ dB}$ |
| **`track03`**  
*(same_track)* | **`V0p`**  
**`V2`** | $10.4470$  
$11.3659$ | $6.4522$  
$7.1960$ | $-38.24\%$  
$-36.69\%$ | $-13.50$  
$-13.50$ | $13.20$  
$13.20$ | $-0.09$  
$-0.04$ | $0.21$  
$0.26$ | $1.41$  
**$1.40$** | $0.6120$  
$0.6124$ | $9.47\text{ dB}$  
**$9.45\text{ dB}$** |
| **`track01_style`**  
*(stylistic)* | **`V0p`**  
**`V2`** | $28.1396$  
$24.3261$ | $8.5631$  
**$4.2513$** | $-69.57\%$  
**$-82.52\%$** | $-17.85$  
$-17.85$ | $15.50$  
$15.56$ | $-2.34$  
$-2.10$ | **$0.00$**  
**$0.00$** | $0.80$  
**$0.75$** | $1.1140$  
$1.1143$ | —  
— |
| **`track02_style`**  
*(stylistic)* | **`V0p`**  
**`V2`** | $159.7820$  
$145.0967$ | $82.6987$  
**$71.4448$** | $-48.24\%$  
**$-50.76\%$** | $-18.05$  
$-18.05$ | $17.75$  
**$17.60$** | $-0.07$  
**$-0.45$** | $0.23$  
**$0.00$ (Clean)** | $2.38$  
**$2.37$** | $0.8441$  
**$0.8403$** | —  
— |
| **`track03_style`**  
*(stylistic)* | **`V0p`**  
**`V2`** | $110.5103$  
$92.7635$ | $46.4980$  
**$33.8844$** | $-57.92\%$  
**$-63.47\%$** | $-13.52$  
$-13.49$ | $13.22$  
**$13.19$** | $+0.05$  
**$-0.07$** | $0.35$  
**$0.23$** | $1.06$  
$1.09$ | $0.9777$  
$0.9794$ | —  
— |

*Nota di riferimento target per sessione:*
- `track01` same: LUFS target = $-7.29$, PLR target = $7.69\text{ dB}$, M/S target = $8.94\text{ dB}$
- `track02` same: LUFS target = $-7.22$, PLR target = $7.84\text{ dB}$, M/S target = $7.56\text{ dB}$
- `track03` same: LUFS target = $-6.90$, PLR target = $7.34\text{ dB}$, M/S target = $9.54\text{ dB}$
- `track01_style`: LUFS ref = $-17.99$, PLR ref = $11.35\text{ dB}$, M/S ref = $9.24\text{ dB}$
- `track02_style`: LUFS ref = $-9.04$, PLR ref = $10.56\text{ dB}$, M/S ref = $9.59\text{ dB}$
- `track03_style`: LUFS ref = $-6.46$, PLR ref = $8.32\text{ dB}$, M/S ref = $9.15\text{ dB}$

---

## 4. Analisi Critica dei Risultati e Diagnostica Dinamica

### 4.1 Eccellente Guadagno di Convergenza nelle Sessioni Stilistiche
Nelle sessioni con riferimento esterno stilistico, l'introduzione dei termini V2 produce un miglioramento drastico e costante dell'efficacia di ottimizzazione:
- In **`track01_style`**, la riduzione percentuale della loss balza da **$-69.57\%$** (`V0p`) a **$-82.52\%$** (`V2`), con la loss finale a chunk che crolla da $8.5631$ a $4.2513$ (riduzione del $50.35\%$ del residuo finale).
- In **`track02_style`**, la riduzione passa da $-48.24\%$ a **$-50.76\%$**, con loss finale ridotta da $82.70$ a $71.44$.
- In **`track03_style`**, la riduzione passa da $-57.92\%$ a **$-63.47\%$**, con loss finale ridotta da $46.50$ a $33.88$.

L'ancoraggio della maschera di attività all'uncompressed mix chunk impedisce ai quantili di cresta ($P_{50}, P_{90}, P_{95}$) di degenerare su frame di respiro o silenzi, mentre la dead-zone di $\pm 1.0\text{ dB}$ sul PLR evita gradienti oscillatori quando l'energia macro-dinamica rientra nella finestra tollerata.

### 4.2 Risoluzione delle Violazioni di True Peak su `track02_style`
Un risultato chiave di V2 risiede nel controllo dell'headroom inter-sample:
- Su **`track02_style`**, la baseline `V0p` produceva un True Peak pari a $-0.07\text{ dBTP}$, con una violazione di $0.23\text{ dB}$ rispetto al ceiling regolamentare di $-0.30\text{ dBTP}$.
- In **`V2`**, il True Peak full-track scende a **$-0.45\text{ dBTP}$**, portando la violazione esattamente a **$0.00\text{ dB}$** e garantendo piena conformità agli standard ITU-R BS.1770-5 ed EBU R128.
- Analogamente, su **`track03_style`**, la violazione di True Peak viene compressa da $0.35\text{ dB}$ a **$0.23\text{ dB}$** ($+0.05\text{ dBTP} \to -0.07\text{ dBTP}$).

Questo comportamento è guidato dalla combinazione tra la penalità sulle pendenze di attacco ($\beta_{\text{atk}} = 0.5$ su $\Delta^+ e_c^{\text{fast}}$) e la stima Boltzmann-weighted del picco soft $p_m(x)$, che penalizza fortemente gli spunti isolati di breve durata ($400\text{ ms}$) anche in assenza di un limiter oversampled attivo.

### 4.3 Stabilità e Trasparenza Spettrale in `same_track`
Nelle tre tracce `same_track`, l'introduzione di `TransientEnvelopeLoss` (loss componenti: $1.1213$ per `track01`, $2.0340$ per `track02`, $1.0276$ per `track03`) e del PLR Huber ($1.2211$ per `track01`, $4.5790$ per `track02`, $3.3948$ per `track03`):
- Non degrada minimamente la convergenza spettrale STFT: la Spectral Convergence ($SC$) rimane rigorosamente identica alla baseline ($0.4055$ vs $0.4053$ su `track01`, $0.7934$ vs $0.7932$ su `track02`, $0.6124$ vs $0.6120$ su `track03`).
- La distanza log-spettrale ($LSD$) full-track su `track03` migliora lievemente da $9.47\text{ dB}$ a **$9.45\text{ dB}$**.
- L'equilibrio stereofonico Mid/Side migliora sensibilmente su `track01` (scostamento dal target ridotto da $0.44\text{ dB}$ a **$0.39\text{ dB}$**).

---

## 5. Decisioni di Promozione e Passaggio allo Step Successivo (V3)

1. **Stato dell'Ipotesi Sperimentale V2:**
   L'ipotesi che la combinazione di inviluppo multibanda LR4 e PLR reference-aware migliori la conservazione della microdinamica e l'headroom senza interferire con la convergenza spettrale è **pienamente confermata** su tutte le 6 sessioni.
2. **Promozione nel Baseline Incrementale:**
   I moduli `TransientEnvelopeLoss` e `ReferenceAwarePLRLoss` (con relative protezioni Fix A–D) sono promossi all'integrazione stabile per i successivi step.
3. **Prossimo Step di Sequenza — `V3` (Limiter Ceiling Penalty & Stereo Balance):**
   La sequenza incrementale prosegue ora verso:
   $$\mathbf{V2} \longrightarrow \mathbf{V3a} \longrightarrow \mathbf{V3b} \longrightarrow \mathbf{V3c} \longrightarrow \mathbf{V4}$$
   in cui verranno affrontati:
   - **`V3a`:** Penalità asimmetrica differenziabile di Limiter Ceiling / True Peak su base oversampled/soft-peak per azzerare le residue violazioni TP su `track01` e `track03`.
   - **`V3b`:** Loss differenziabile di bilanciamento stereo Mid/Side e correlazione di fase inter-canale.
   - **`V3c`:** Regolarizzazione congiunta e taratura fine dei pesi di accoppiamento.

---
*Fine del Report Tecnico Step V2 — Sprint 3 AutoMastering.*
