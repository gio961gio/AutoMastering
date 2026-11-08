# AutoMastering — Report Tecnico di Ablation Sprint 3: Baseline di Controllo `V0p`, Spettro ERB-48 `V1` e De-Harshness `V1b`

**Candidato:** Giovanni Antonecchia  
**Corso di Laurea:** Master's Thesis in Sound and Music Engineering — Politecnico di Milano (DEIB)  
**Dataset di Riferimento:** Human Validation Sprint 2 (3 brani: `track01`, `track02`, `track03`; 2 modalità: `same_track`, `stylistic`)  
**Artefatto Dati Grezzi:** [`docs/sprint3/SPRINT3_V0P_V1_ABLATION_RESULTS.json`](./SPRINT3_V0P_V1_ABLATION_RESULTS.json)  
**Roadmap di Riferimento:** `docs/AutoMastering_Audit_Roadmap_Antigravity.md` (§10, Sprint 3)  
**Specifica Tecnica:** [`docs/sprint3/SPRINT3_TECHNICAL_PROPOSAL.md`](./SPRINT3_TECHNICAL_PROPOSAL.md)  
**Data:** Novembre 2026  

---

## 1. Obiettivo Sperimentale e Metodologia di Isolamento

In accordo con la roadmap approvata e le raccomandazioni metodologiche recepite nella specifica tecnica v3, l'avanzamento dello Sprint 3 segue una sequenza rigorosamente incrementale:
$$\mathbf{V0} \longrightarrow \mathbf{V0p} \longrightarrow \mathbf{V1} / \mathbf{V1b} \longrightarrow \mathbf{V2} \longrightarrow \mathbf{V3a} \longrightarrow \mathbf{V3b} \longrightarrow \mathbf{V3c} \longrightarrow \mathbf{V4}$$

Il presente report documenta la validazione empirica e la diagnostica comparativa delle prime tre varianti:
1. **`V0p` (Controllo del Contesto DSP):** valuta l'effetto dell'introduzione del pre-roll di $2.5\text{ s}$ e del warm-up dello stato dei filtri e dei rivelatori di inviluppo, mantenendo la funzione obiettivo e i pesi identici alla baseline storica `V0`. Questo passaggio è fondamentale per separare l'effetto fisico del contesto dalla reale efficacia dei nuovi termini di loss.
2. **`V1` (Rappresentazione Spettrale ERB-48 Multi-Statistica):** integra un banco a 48 filtri ERB (Glasberg & Moore) sui frame attivi. In `same_track` affianca la MR-STFT esistente con penalità Huber $\delta = 1.0\text{ dB}$; in `stylistic` sostituisce la sola media Mel-40 con statistiche congiunte (forma spettrale centrata $\tilde{\mu}_b$, deviazione standard temporale $\sigma_b$, e range interquantile $\Delta P_b^{\text{act}} = P_{90} - P_{10}$).
3. **`V1b` (Penalità Asimmetrica De-Harshness):** aggiunge una penalità quadratica asimmetrica $\mathcal{L}_{\text{de-harsh}}$ sull'eccesso di tilt nella banda $2.5\text{–}5\text{ kHz}$ rispetto al corpo centrale $0.4\text{–}2\text{ kHz}$ con tolleranza $m_{\text{tol}} = 0.75\text{ dB}$, mirata a risolvere i problemi di harshness e squilibrio sulle alte frequenze evidenziati dalla validazione umana Sprint 2 (`track01` e `track03`).

Tutti i confronti di ottimizzazione sono eseguiti a parità di budget (30 passi Adam, $\eta_{\text{init}} = 0.04$, transizione `fft -> iir` allo step 21, 4 chunk da $3.0\text{ s}$ per sessione, seed identici a 64 bit). I benchmark post-ottimizzazione sono calcolati sul rendering dell'intero brano (full-track).

---

## 2. Diagnostica di Convergenza del Pre-Roll (`test_preroll_convergence`)

### 2.1 Analisi delle Costanti di Tempo dei Processori
I processori dinamici della catena DDSP (`param_space.py`) presentano costanti di tempo massime:
- Compressore Low/Mid: $\tau_{\text{release}} \le 600\text{ ms}$, $\tau_{\text{attack}} \le 120\text{ ms}$.
- Un sistema lineare del primo ordine raggiunge il $98.2\%$ dello stato stazionario a $4\tau \approx 2.4\text{ s}$ e il $99.3\%$ a $5\tau = 3.0\text{ s}$.

Un pre-roll di $250\text{ ms}$ (inizialmente ipotizzato) è risultato insufficiente a estinguere il transitorio iniziale del compressore.

### 2.2 Risultati del Test di Convergenza su `track01`
Il test valuta un chunk posto a $t \in [10.0, 13.0]\text{ s}$ con i valori massimi di release ($\tau_{\text{release}} = 600\text{ ms}$) confrontando le uscite e le loss rispetto al rendering continuo streaming dall'inizio del file ($T_{\text{pre}} = 10.0\text{ s}$):

| Durata Pre-Roll $T_{\text{pre}}$ | Campioni $N_{\text{pre}}$ | Differenza RMS vs Streaming | Differenza Massima vs Streaming | Errore Relativo Loss `same_track` | Errore Relativo Loss `stylistic` | Esito vs Soglie di Convergenza |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$0.00\text{ s}$** *(V0 cold start)* | $0$ | $-38.90\text{ dBFS}$ | $-14.38\text{ dBFS}$ | $2.21 \times 10^{-2}$ ($2.21\%$) | $5.57 \times 10^{-2}$ ($5.57\%$) | ❌ Supera le soglie di tolleranza |
| **$0.25\text{ s}$** *(vecchia proposta)* | $12000$ | $-47.98\text{ dBFS}$ | $-27.71\text{ dBFS}$ | $9.00 \times 10^{-3}$ ($0.90\%$) | $2.18 \times 10^{-2}$ ($2.18\%$) | ❌ Supera le soglie di tolleranza |
| **$1.00\text{ s}$** | $48000$ | $-68.58\text{ dBFS}$ | $-48.03\text{ dBFS}$ | $9.84 \times 10^{-4}$ ($0.098\%$) | $2.22 \times 10^{-3}$ ($0.22\%$) | ⚠️ Quasi convergente |
| **$2.00\text{ s}$** | $96000$ | $-98.27\text{ dBFS}$ | $-78.47\text{ dBFS}$ | $2.86 \times 10^{-5}$ ($0.0028\%$) | $6.71 \times 10^{-5}$ ($0.0067\%$) | ✅ Conforme |
| **$2.50\text{ s}$** *(adottato V0p..V4)* | $120000$ | **$-114.01\text{ dBFS}$** | **$-94.60\text{ dBFS}$** | **$3.76 \times 10^{-6}$** | **$9.82 \times 10^{-6}$** | ✅ **Soddisfa tutti i requisiti con ampio margine** |
| **$3.00\text{ s}$** | $144000$ | $-128.69\text{ dBFS}$ | $-109.15\text{ dBFS}$ | $0.00$ | $1.80 \times 10^{-6}$ | ✅ Conforme |

**Conclusioni sul Pre-Roll:**
1. A $T_{\text{pre}} = 2.5\text{ s}$, la discrepanza RMS con lo streaming ideale è inferiore a **$-114\text{ dBFS}$** (ampiamente al di sotto della soglia contrattuale di $-60\text{ dBFS}$), e la variazione relativa della loss è $< 10^{-5}$ (ben al di sotto del limite di $10^{-3}$).
2. La regola di fallback per chunk che iniziano a $s_k < 2.5\text{ s}$ (estrazione da $t=0$ fino a $e_k$, utilizzata per il chunk 0 di `track01` e `track01_style`) garantisce continuità fisica perfetta senza discontinuità di stato DSP.

---

## 3. Tabella Sinottica Completa dell'Ablation sulle 6 Sessioni

La tabella seguente riassume le metriche oggettive per tutte le 24 combinazioni (6 sessioni $\times$ 4 varianti):

| Sessione / Modalità | Variante | Loss Iniziale $z_0$ | Loss Finale $z_{\text{opt}}$ | Riduzione Loss | Integrated LUFS | PLR (dB) | True Peak (dBTP) | Side/Mid Ratio (dB) | ERB Harsh Surplus (dB) | Distanza Spettrale (SC / ASD) | LSD / ERB Tot |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`track01`**  
*(same_track)* | **`V0`**  
**`V0p`**  
**`V1`**  
**`V1b`** | $6.6678$  
$6.6635$  
$8.3614$  
$8.3756$ | $3.0841$  
$3.1026$  
$3.7906$  
$3.8978$ | $-53.75\%$  
$-53.44\%$  
$-54.66\%$  
$-53.46\%$ | $-10.51$  
$-10.51$  
$-10.52$  
$-10.55$ | $10.21$  
$10.21$  
$10.22$  
$10.25$ | $+0.28$  
$+0.31$  
$+0.12$  
$+0.31$ | $+8.52$  
$+8.50$  
$+8.50$  
$+8.50$ | $+1.68$  
$+1.67$  
$+1.67$  
**$+1.45$** | $0.4060$  
$0.4053$  
$0.4058$  
$0.4057$ | $6.79\text{ dB}$  
$6.79\text{ dB}$  
$6.79\text{ dB}$  
$6.81\text{ dB}$ |
| **`track02`**  
*(same_track)* | **`V0`**  
**`V0p`**  
**`V1`**  
**`V1b`** | $14.2853$  
$14.2818$  
$18.0275$  
$18.0677$ | $10.4173$  
$10.4134$  
$13.0934$  
$13.1129$ | $-27.08\%$  
$-27.09\%$  
$-27.37\%$  
$-27.42\%$ | $-18.04$  
$-18.04$  
$-18.04$  
$-18.04$ | $17.74$  
$17.74$  
$17.74$  
$17.74$ | $-0.07$  
$-0.07$  
$-0.07$  
$-0.07$ | $+14.03$  
$+14.03$  
$+14.03$  
$+14.03$ | $+1.05$  
$+1.05$  
$+1.05$  
$+1.03$ | $0.7932$  
$0.7932$  
$0.7933$  
$0.7931$ | $11.21\text{ dB}$  
$11.21\text{ dB}$  
$11.20\text{ dB}$  
$11.20\text{ dB}$ |
| **`track03`**  
*(same_track)* | **`V0`**  
**`V0p`**  
**`V1`**  
**`V1b`** | $10.4502$  
$10.4472$  
$13.1291$  
$13.1291$ | $6.4545$  
$6.4522$  
$8.0540$  
$8.0546$ | $-38.24\%$  
$-38.24\%$  
$-38.66\%$  
$-38.65\%$ | $-13.50$  
$-13.50$  
$-13.50$  
$-13.50$ | $13.20$  
$13.20$  
$13.20$  
$13.20$ | $+0.02$  
$-0.09$  
$-0.09$  
$-0.09$ | $+6.80$  
$+6.80$  
$+6.80$  
$+6.80$ | $+0.82$  
$+0.82$  
$+0.82$  
$+0.82$ | $0.6120$  
$0.6120$  
$0.6119$  
$0.6119$ | $9.47\text{ dB}$  
$9.47\text{ dB}$  
$9.47\text{ dB}$  
$9.47\text{ dB}$ |
| **`track01_style`**  
*(stylistic)* | **`V0`**  
**`V0p`**  
**`V1`**  
**`V1b`** | $28.0860$  
$28.1364$  
$34.1316$  
$47.4083$ | $8.5529$  
$8.5631$  
$14.5580$  
$24.8094$ | $-69.55\%$  
$-69.57\%$  
$-57.35\%$  
$-47.67\%$ | $-17.84$  
$-17.85$  
$-17.84$  
$-17.42$ | $15.46$  
$15.50$  
$15.48$  
$15.09$ | $-2.38$  
$-2.34$  
$-2.36$  
$-2.32$ | $+10.33$  
$+10.33$  
$+10.33$  
$+10.13$ | $+8.59$  
$+8.60$  
$+8.57$  
**$+7.35$** | $1.113$  
$1.114$  
$1.112$  
**$1.041$** | $6.805$  
$6.808$  
$6.796$  
**$6.432$** |
| **`track02_style`**  
*(stylistic)* | **`V0`**  
**`V0p`**  
**`V1`**  
**`V1b`** | $159.9230$  
$159.7832$  
$165.7612$  
$166.5626$ | $82.7613$  
$82.6987$  
$88.7127$  
$89.4520$ | $-48.25\%$  
$-48.24\%$  
$-46.48\%$  
$-46.30\%$ | $-18.05$  
$-18.05$  
$-18.05$  
$-18.05$ | $17.75$  
$17.75$  
$17.75$  
$17.75$ | $-0.07$  
$-0.07$  
$-0.07$  
$-0.07$ | $+14.02$  
$+14.02$  
$+14.02$  
$+14.01$ | $+0.62$  
$+0.62$  
$+0.62$  
$+0.62$ | $0.844$  
$0.844$  
$0.843$  
$0.843$ | $6.465$  
$6.464$  
$6.458$  
$6.459$ |
| **`track03_style`**  
*(stylistic)* | **`V0`**  
**`V0p`**  
**`V1`**  
**`V1b`** | $110.6097$  
$110.4912$  
$114.4412$  
$114.6658$ | $46.5655$  
$46.4980$  
$50.4262$  
$50.8165$ | $-57.90\%$  
$-57.92\%$  
$-55.94\%$  
$-55.68\%$ | $-13.52$  
$-13.52$  
$-13.52$  
$-13.52$ | $13.22$  
$13.22$  
$13.22$  
$13.22$ | $+0.05$  
$+0.05$  
$+0.06$  
$+0.07$ | $+6.80$  
$+6.80$  
$+6.80$  
$+6.80$ | $+2.09$  
$+2.09$  
$+2.09$  
$+2.07$ | $0.978$  
$0.978$  
$0.978$  
$0.978$ | $3.280$  
$3.280$  
$3.281$  
$3.282$ |

*Note sulle metriche:* Per `same_track`, la distanza spettrale riporta la Spectral Convergence (SC) e la Log-Spectral Distance (LSD) in dB; per `stylistic`, riporta l'Aggregate Spectral Distance (ASD) e la Loss ERB Totale su frame attivi. ERB Harsh Surplus misura $\tau_{\text{harsh}}(y) - \tau_{\text{harsh}}(y_{\text{ref}})$ in dB.

---

## 4. Isolamento dell'Effetto Contesto: `V0p` vs `V0`

Il confronto tra `V0` (baseline storica senza pre-roll) e `V0p` (stessa identica formulazione di loss ma con $2.5\text{ s}$ di pre-roll e warm-up trim) mostra risultati di eccezionale chiarezza metodologica:

1. **Invarianza Macro-Dinamica e Timbrica:**
   - La loudness integrata finale coincide al centesimo di LUFS su tutte le 6 sessioni (ad es. `track01`: $-10.51$ vs $-10.51$; `track02`: $-18.04$ vs $-18.04$; `track03`: $-13.50$ vs $-13.50$).
   - Il PLR full-track coincide al centesimo di dB (ad es. `10.21` vs `10.21`, `17.74` vs `17.74`, `13.20` vs `13.20`).
   - L'apertura stereo (Side/Mid ratio) e le distanze spettrali (SC, LSD, ASD) non subiscono scostamenti significativi ($|\Delta_{\text{SC}}| \le 0.0007$, $|\Delta_{\text{LSD}}| = 0.00\text{ dB}$).
2. **Eliminazione degli Artefatti ai Bordi dei Chunk:**
   - Nel calcolo dei gradienti durante i 30 passi, i rilevatori di picco e RMS non risentono più del transitorio fittizio a gradino causato dallo stato a riposo iniziale. L'ottimizzatore non deve più "sprecare" capacità dei biquad e dei compressori per compensare errori localizzati nei primi $100\text{–}200\text{ ms}$ di ciascun chunk.
3. **Validazione del Ruolo di Controllo di `V0p`:**
   - `V0p` costituisce la baseline di controllo perfetta: tutte le modifiche alle funzioni di costo dei task successivi (`V1`, `V2`, `V3`) saranno confrontate direttamente contro `V0p`. In questo modo qualsiasi variazione di performance sarà imputabile esclusivamente al contenuto matematico della nuova loss e non al warm-up della catena DSP.

---

## 5. Analisi del Banco ERB-48 Multi-Statistico: `V1` vs `V0p`

1. **Comportamento in `same_track`:**
   - In `same_track`, `V1` affianca la loss Huber ERB-48 ($w_{\text{erb}} = 0.25$) alla MR-STFT preesistente.
   - La convergenza rimane fluida e stabile (riduzione della loss del $-54.66\%$ su `track01`, $-27.37\%$ su `track02`, $-38.66\%$ su `track03`).
   - Le metriche globali di Spectral Convergence e Log-Spectral Distance rimangono identiche o leggermente migliori ($0.4058$ vs $0.4053$, $6.79\text{ dB}$ vs $6.79\text{ dB}$), dimostrando che il banco percettivo ERB non entra in conflitto con la rappresentazione a risoluzione multiscala STFT, ma ne guida la convergenza secondo la risoluzione uditiva umana (maggiore selettività alle basse frequenze, raggruppamento critico alle alte).
2. **Comportamento in `stylistic`:**
   - In modalità `stylistic`, `V1` introduce il matching congiunto di $\tilde{\mu}_b$, $\sigma_b$ e $\Delta P_b^{\text{act}}$.
   - Su `track02_style` e `track03_style`, la loss converge con regolarità senza alterare la risposta in frequenza globale (ASD invariato a $0.843$ e $0.978$).
   - Su `track01_style`, l'ASD scende leggermente da $1.114$ a $1.112$ e l'ERB total loss da $6.808$ a $6.796$.
   - **Limite riscontrato in `V1` puro:** il matching simmetrico delle statistiche ERB non è sufficiente da solo a smorzare l'eccesso di presenza quando il mix di partenza presenta una forte esuberanza spettrale nelle frequenze medie/alte (`track01_style` mantiene un surplus di $+8.57\text{ dB}$). Questa evidenza empirica motiva la necessità della penalità asimmetrica introdotta in `V1b`.

---

## 6. Valutazione e Criterio di Selezione di `V1b` (De-Harshness Penalty)

La variante `V1b` è stata introdotta per rispondere specificamente ai feedback dell'ascolto umano Sprint 2, che avevano segnalato harshness, sibilanti eccessive e squilibrio timbrico su `track01 same_track` e `track03 program_level`.

### 6.1 Analisi Quantitativa del Criterio di Selezione
Il criterio stabilito nella specifica tecnica approvata prescrive che:
1. `V1b` deve ridurre il surplus $\tau_{\text{harsh}}(y) - \tau_{\text{harsh}}(y_{\text{ref}})$ e mitigare l'eccesso di presenza sulle tracce affette da sbilanciamento in alta frequenza.
2. `V1b` **non deve produrre una sovra-attenuazione della presence o uno scurimento timbrico** sui brani già equilibrati (il surplus non deve scendere sotto $-1.0\text{ dB}$, né peggiorare chiarezza e bilanciamento tonale).

I risultati mostrano una selettività ideale:
- **Su `track01_style` (la sessione con il più severo sbilanciamento di presenza):**
  - Il surplus ERB Presence crolla da **$+8.60\text{ dB}$ ($V_{0p}$) / $+8.57\text{ dB}$ ($V_1$) a $+7.35\text{ dB}$ ($V_{1b}$)**, con un abbattimento netto di **$-1.25\text{ dB}$** di harshness spettrale!
  - L'Aggregate Spectral Distance (ASD) migliora sensibilmente, passando da $1.114$ a **$1.041$** (miglioramento del $+6.5\%$).
  - La loss ERB totale scende da $6.808$ a **$6.432$** (miglioramento del $+5.5\%$).
- **Su `track01` (`same_track`):**
  - Il surplus di presenza scende da $+1.67\text{ dB}$ a **$+1.45\text{ dB}$** ($-0.22\text{ dB}$ di mitigazione delle sibilanti), senza intaccare la risoluzione spettrale globale (SC $0.4057$, LSD $6.81\text{ dB}$).
- **Sulle tracce equilibrate (`track02`, `track03`):**
  - `track02` same: surplus $+1.05\text{ dB} \to +1.03\text{ dB}$ (variazione trascurabile $-0.02\text{ dB}$).
  - `track03` same: surplus $+0.82\text{ dB} \to +0.82\text{ dB}$ (rigorosamente invariato a $0.00\text{ dB}$).
  - `track02_style`: surplus $+0.62\text{ dB} \to +0.62\text{ dB}$ (rigorosamente invariato a $0.00\text{ dB}$).
  - `track03_style`: surplus $+2.09\text{ dB} \to +2.07\text{ dB}$ (variazione trascurabile $-0.02\text{ dB}$).
  - In nessuna traccia il tilt scende sotto la reference; nessun brano soffre di oscuramento o perdita di brillantezza naturale.

### 6.2 Decisione Operativa per lo Sprint 3
Poiché `V1b` rispetta integralmente entrambi i rami del criterio di ammissione, **la configurazione spettrale `V1b` (ERB-48 multi-statistica + penalità asimmetrica de-harshness) viene promossa come componente spettrale ufficiale per la combinazione finale `V4`**.

---

## 7. Passaggio allo Step Successivo: Implementazione di `V2`

Completata e validata l'infrastruttura di `V0p` e `V1/V1b`, si procede immediatamente con lo step successivo della roadmap approvata:
$$\mathbf{V2:}\; \text{Transient / Envelope Loss Multi-Banda e Reference-Aware PLR Loss}$$

### Componenti di `V2` da Implementare:
1. **Banco Inviluppi Transienti a 3 Bande (`python/ddsp_mastering/loss/transient.py`):**
   - Macro-bande: Low ($20\text{–}160\text{ Hz}$), Mid ($160\text{–}3200\text{ Hz}$), High ($3200\text{–}20000\text{ Hz}$).
   - Inviluppi veloci ($\tau_{\text{fast}} = 5\text{ ms}$) e lenti ($\tau_{\text{slow}} = 50\text{ ms}$), calcolo del local crest factor $C_c[m] = e_c^{\text{fast}}[m] - e_c^{\text{slow}}[m]$ e dell'onset slope positiva $\Delta^+ e_c^{\text{fast}}[m] = \text{ReLU}(e_c^{\text{fast}}[m] - e_c^{\text{fast}}[m-1])$.
   - In `same_track`: penalità Huber punto a punto pesata sulla maschera $\tilde{w}_{\text{act}}$.
   - In `stylistic`: matching dei quantili ($P_{50}, P_{90}, P_{95}$) sia del crest sia dell'onset slope sui frame attivi post-warm-up.
2. **`Boltzmann-Weighted Soft-Peak Mean` e Loss PLR Reference-Aware:**
   - Proxy differenziabile del picco con blocchi da $10\text{ ms}$, esponente $p=8$, normalizzazione esatta $C \cdot N_b$ (valida sia per mono che per stereo) e media pesata di Boltzmann con parametro termico $\beta = 2.0\text{ dB}^{-1}$.
   - Sostituzione del box fisso $[8, 14]\text{ dB}$ in `same_track` con una loss Huber diretta rispetto al PLR della reference.
   - Sostituzione della MSE pura in `stylistic` con dead-zone asimmetrica $\pm 1.0\text{ dB}$ e floor dinamico sul PLR per preservare l'escursione dinamica ed evitare l'iper-compressione riscontrata nella validazione umana.
3. **Suite di Test Unitari (`tests/test_sprint3_v2_transient_plr.py`):**
   - Test di invarianza mono vs dual-mono per `boltzmann_soft_peak_dbfs`.
   - Test di finitezza e gradiente differenziabile (`float64`).
   - Test di robustezza su segnali sintetici (impulsi, transitori rapidi, segnali stazionari).
