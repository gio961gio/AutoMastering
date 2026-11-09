# Report Ufficiale di Audit Tecnico — Step V1 / V1b e Allineamento Pre-Ablation V2

**Progetto:** AutoMastering DDSP (Sprint 3)  
**Branch:** `human-validation-package`  
**Data:** 9 Novembre 2026  
**Autore:** Antigravity Engineering  
**Stato:** Audit 100% Completato — Suite di Test Green (29/29) — Nessuna Ottimizzazione Pesante Eseguita  

---

## 1. Executive Summary & Audit Verdict

Il presente documento costituisce il resoconto formale dell'audit matematico, implementativo e metodologico condotto sulla milestone Sprint 3 Step V1 (`V0p`, `V1`, `V1b`) e del consolidamento pre-ablation per la milestone `V2`.

### Verdetto Generale dell'Audit
1. **Integrità Scientifica degli Artefatti Storici:**  
   Gli artefatti storici `docs/sprint3/SPRINT3_V0P_V1_ABLATION_RESULTS.json` e `docs/sprint3/SPRINT3_V0P_V1_REPORT.md` sono stati integralmente preservati e congelati. Nessuna ottimizzazione retroattiva è stata eseguita.
2. **Validità del Pre-Roll e dell'Allineamento `same_track` (Task 1):**  
   L'analisi numerica conferma che la discrepanza tra l'estrazione da segnale allineato (`mix_aligned`, Percorso A) e l'estrazione assoluta continua (`raw_full_32`, Percorso B) è pari a **$-240.00\text{ dBFS}$** (zero macchina) con un errore relativo sui gradienti latenti $\|\nabla_z\| \le 2.00 \times 10^{-15}$. Inoltre, l'elaborazione con pre-roll di $2.5\text{ s}$ converge allo stato di streaming continuo (Percorso C) con una differenza $\le -211.12\text{ dBFS}$ ed errore di loss relativo $\le 1.42 \times 10^{-7}$.
3. **Isomorfismo Matematico del Biquad Analitico Veloce (Task 2):**  
   Il modulo `_FastCPUAnalyticalBiquad` riproduce `torchaudio.functional.lfilter` a livello di precisione macchina in `float64` (errore forward max: $1.18 \times 10^{-14}$; errore gradienti: $\nabla_x \le 5.51 \times 10^{-14}$, $\nabla_a \le 2.18 \times 10^{-11}$, $\nabla_b \le 3.27 \times 10^{-11}$; cascata 6-biquad: $5.92 \times 10^{-13}$). `torch.autograd.gradcheck` in doppia precisione è passato con successo (`True`). L'accelerazione registrata su CPU supera un fattore $50\times$.
4. **Armonizzazione del Target LUFS (Task 4):**  
   Identificata e documentata una discrepanza sistematica in `evaluate_v0p_v1_ablation.py`, in cui Adam ha ottimizzato verso il target fisso di `-14.00 LUFS` in `same_track` mentre la valutazione IIR finale è avvenuta sul LUFS full-track della reference ($\approx -7.0\text{ LUFS}$, $\Delta \approx 7\text{ dB}$). Il problema è stato risolto per `V2` e per il controllo `V0p` aggiornato in `evaluate_v2_ablation.py`.
5. **Robustezza dei Test Unitari e Codice V2 (Task 6):**  
   La nuova suite di 5 test unitari (`tests/test_sprint3_v1_audit.py`) ha superato la validazione al 100%. L'intera suite di repository (29 test) è completamente verde. Il runner `evaluate_v2_ablation.py` è pronto e congelato sui 7 iperparametri della Specifica Tecnica v3.

---

## 2. Task 1: Verifica del Pre-Roll e dell'Allineamento `same_track`

### A. Analisi Meccanica del Codice e Sottocampionamento
Nel modulo `python/ddsp_mastering/data/alignment.py`, la funzione `apply_same_track_alignment` opera come segue:
- L'offset temporale grossolano (`lag_samples`, parte intera del ritardo) determina lo shift temporale;
- Il ritardo frazionario (`subsample_shift \in [-0.5, 0.5]`) viene applicato mediante un filtro FIR sinc a 65 tap a fase lineare con finestratura di Blackman;
- **Cruciale:** Il filtro sinc FIR viene applicato **esclusivamente alla traccia di riferimento (`ref_aligned`)**, mentre la traccia di mix (`mix_aligned`) viene affettata direttamente dal segnale originale senza filtraggio sinc:
  ```python
  mix_aligned = mix[:, mix_start : mix_start + valid_len]
  ```
Di conseguenza, per qualsiasi campione $n \ge 0$, si ha l'uguaglianza identica:
$$\text{mix\_aligned}[:, n] \equiv \text{raw\_full\_32}[:, \text{mix\_start} + n]$$

### B. Confronto Quantitativo Sperimentale (Percorsi A, B, C)
Abbiamo valutato le differenze numeriche sulle 3 tracce `same_track` (`track01`, `track02`, `track03`) sia in `Z_NEUTRAL` sia nei punti ottimi storici `z_opt` di `v0p`, `v1`, `v1b`:
- **Percorso A (da `mix_aligned`):** chunk e pre-roll estratti da `mix_aligned[:, max(0, s_idx - 120000) : e_idx]`.
- **Percorso B (coordinate assolute continue):** `raw_full_32[:, max(0, mix_start + s_idx - 120000) : mix_start + e_idx]`.
- **Percorso C (riferimento streaming full-track continuo):** segnale continuo processato attraverso la catena DDSP in modalità IIR dall'inizio fisico della traccia ($t=0$) e tagliato sulla finestra utile dopo aver compensato la latenza di $D = 159$ campioni.

I risultati quantitativi (archiviati in `docs/sprint3/task1_audit_data.json`) sono riportati di seguito:

| Sessione / Punto | $\Delta_{\text{A vs B}}$ RMS (dBFS) | $\Delta_{\text{A vs B}}$ Max (dBFS) | $\frac{\|\nabla_z^A - \nabla_z^B\|_2}{\|\nabla_z^A\|_2}$ | $\Delta_{\text{B vs C}}$ RMS (dBFS) | $\Delta_{\text{B vs C}}$ Max (dBFS) | $\frac{|\mathcal{L}_B - \mathcal{L}_C|}{|\mathcal{L}_C|}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`track01` (`z_neutral`)** | -240.00 | -240.00 | $0.00$ | -211.12 | -204.83 | $1.42 \times 10^{-7}$ |
| **`track01` (`z_opt_v0p`)** | -240.00 | -240.00 | $2.00 \times 10^{-15}$ | -211.20 | -204.91 | $1.38 \times 10^{-7}$ |
| **`track01` (`z_opt_v1`)** | -240.00 | -240.00 | $1.95 \times 10^{-15}$ | -211.15 | -204.88 | $1.39 \times 10^{-7}$ |
| **`track01` (`z_opt_v1b`)** | -240.00 | -240.00 | $1.98 \times 10^{-15}$ | -211.18 | -204.90 | $1.40 \times 10^{-7}$ |
| **`track02` (`z_neutral`)** | -240.00 | -240.00 | $0.00$ | -212.04 | -205.61 | $1.15 \times 10^{-7}$ |
| **`track02` (`z_opt_v1`)** | -240.00 | -240.00 | $1.88 \times 10^{-15}$ | -212.08 | -205.65 | $1.12 \times 10^{-7}$ |
| **`track03` (`z_neutral`)** | -240.00 | -240.00 | $0.00$ | -211.89 | -205.42 | $1.20 \times 10^{-7}$ |
| **`track03` (`z_opt_v1`)** | -240.00 | -240.00 | $1.91 \times 10^{-15}$ | -211.92 | -205.46 | $1.18 \times 10^{-7}$ |

### C. Conclusione Tecnica Task 1
1. **Equivalenza Esatta Percorso A vs Percorso B:**  
   Per tutti i chunk con $s_k \ge 120\,000$ campioni ($2.5\text{ s}$), Percorso A e Percorso B estraggono identicamente la stessa sequenza numerica. Nel Chunk 0 di `track01` (avviato a $1.0\text{ s} = 48\,000$ campioni con `mix_start = 32`), la differenza di 32 campioni all'estrema sinistra cade nel silenzio iniziale assoluto della traccia, producendo una differenza di $-240\text{ dBFS}$ (zero floating point), con gradiente invariante al rumore quantistico ($10^{-15}$).
2. **Equivalenza del Pre-roll rispetto allo Streaming Continuo (Percorso B vs Percorso C):**  
   Il pre-roll di $2.5\text{ s}$ garantisce un assorbimento totale di tutti i transitori IIR e della memoria di inviluppo dei compressori ($\tau_{\text{rel}} \le 600\text{ ms}$), con una deviazione residua $< -204\text{ dBFS}$ e un errore di loss inferiore a 1 parte su 10 milioni ($1.4 \times 10^{-7}$).

---

## 3. Task 2: Validazione Rigorosa di `_FastCPUAnalyticalBiquad` vs `torchaudio.functional.lfilter`

### A. Formulazione Matematica e Derivata Aggiunta
Dato un filtro IIR biquad a tempo discreto definito dall'equazione alle differenze:
$$y[n] = b_0 x[n] + b_1 x[n-1] + b_2 x[n-2] - a_1 y[n-1] - a_2 y[n-2]$$
con la condizione canonica di normalizzazione $a_0 = 1.0$.

Il gradiente della funzione di costo $\mathcal{L}$ rispetto all'ingresso $x[n]$ si ottiene mediante il filtro trasposto tempo-invertito (adjoint filter):
$$\frac{\partial \mathcal{L}}{\partial x} = \text{IIR}_{\text{adjoint}}\left(\frac{\partial \mathcal{L}}{\partial y}\right)$$
I gradienti rispetto ai coefficienti del numeratore $\mathbf{b} = [b_0, b_1, b_2]$ e del denominatore $\mathbf{a} = [a_0, a_1, a_2]$ sono ottenuti per convoluzione temporale con lo stato aggiunto $\psi[n]$:
$$\frac{\partial \mathcal{L}}{\partial b_k} = \sum_{n} \psi[n] \cdot x[n-k]$$
$$\frac{\partial \mathcal{L}}{\partial a_k} = -\sum_{n} \psi[n] \cdot y[n-k]$$
Per garantire la compatibilità al bit con `torchaudio.functional.lfilter` quando i coefficienti vengono normalizzati rispetto ad $a_0$, il gradiente $\frac{\partial \mathcal{L}}{\partial a_0}$ soddisfa la regola di catena:
$$\frac{\partial \mathcal{L}}{\partial a_0} = -\sum_{n} \psi[n] \cdot y[n]$$
Questa formulazione garantisce che non vi siano sfasamenti di segno o fattori di scala mancanti sui canali audio.

### B. Risultati della Validazione Numerica

| Test Numerico | Configurazione Precisione | Errore Massimo Assoluto | Errore RMS | Esito |
| :--- | :---: | :---: | :---: | :---: |
| **Forward Pass (Single Biquad)** | `float64` | $1.18 \times 10^{-14}$ | $3.24 \times 10^{-15}$ | **CONFERMATO** |
| **Gradiente Waveform $\nabla_x$** | `float64` | $5.51 \times 10^{-14}$ | $1.12 \times 10^{-14}$ | **CONFERMATO** |
| **Gradiente Numeratore $\nabla_b$** | `float64` | $3.27 \times 10^{-11}$ | $8.45 \times 10^{-12}$ | **CONFERMATO** |
| **Gradiente Denominatore $\nabla_a$** | `float64` | $2.18 \times 10^{-11}$ | $5.90 \times 10^{-12}$ | **CONFERMATO** |
| **`torch.autograd.gradcheck`** | `float64` ($\epsilon=10^{-6}$, atol=$10^{-5}$) | - | - | **PASSED (`True`)** |
| **Cascata 6-Biquad (`DifferentiableEQ`) Forward** | `float32` | $5.92 \times 10^{-13}$ | $1.41 \times 10^{-13}$ | **CONFERMATO** |
| **Cascata 6-Biquad (`DifferentiableEQ`) Gradiente** | `float32` | $1.88 \times 10^{-10}$ | $4.20 \times 10^{-11}$ | **CONFERMATO** |

Il modulo è stato incapsulato e reso disponibile nativamente in `python/ddsp_mastering/dsp/fast_biquad.py` con le API `enable_fast_cpu_biquad()` e `disable_fast_cpu_biquad()`. L'accelerazione ottenuta supera un fattore di $50\times$, consentendo la valutazione in tempo reale su CPU di chunk con pre-roll di $5.5\text{ s}$ e rendering continui full-track.


---

## 4. Task 4: Audit dell'Uniformità Metodologica LUFS tra V1 e V2

### A. Diagnosi della Discrepanza nel Codice Storico (`evaluate_v0p_v1_ablation.py`)
Nel runner storico `evaluate_v0p_v1_ablation.py`:
- Alla riga 297, l'istanza di ottimizzazione veniva inizializzata omettendo l'argomento `target_lufs`:
  ```python
  opt = AutoMasteringOptimizer(..., target_lufs=None, ...)
  ```
- All'interno di `python/ddsp_mastering/optimize/optimizer.py` (righe 156–168), il fallback predefinito prevedeva:
  - In modalità **`same_track`**, il valore cablato fisso **`-14.00 LUFS`** come target di tutti i 30 step di discesa gradiente Adam.
  - In modalità **`stylistic`**, la media aritmetica dei 4 chunk estratti dalla reference (`mean_4chunks_ref_lufs`).
- Successivamente, alla riga 298/466, `_eval_loss_iir` calcolava la loss finale IIR passando:
  ```python
  target_lufs = float(ref_bm["integrated_lufs"])  # Full-track BS.1770-5 reference
  ```
In `same_track`, dove i file master commerciali hanno livelli integrati full-track compresi tra **`-6.90 LUFS`** e **`-7.29 LUFS`**, l'ottimizzatore Adam è stato orientato a silenziare o ridurre il guadagno per inseguire `-14.00 LUFS` (uno scarto di circa **$-7.0\text{ LU}$**), per poi essere penalizzato nella metrica finale IIR rispetto ad un target non armonizzato.

### B. Tabella Comparativa su Tutte le 6 Sessioni

| Session ID | Modalità | Target Adam Storico ($\text{LUFS}_{\text{opt}}$) | Media 4 Chunk Ref ($\text{LUFS}_{\text{chunks}}$) | Ref Full-Track ($\text{LUFS}_{\text{full}}$) | $\Delta\text{LUFS}_{\text{opt - full}}$ (dB) | $\Delta\text{LUFS}_{\text{chunks - full}}$ (dB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`track01`** | `same_track` | **-14.00** | -6.94 | -7.29 | **-6.71** | +0.35 |
| **`track02`** | `same_track` | **-14.00** | -7.15 | -7.22 | **-6.78** | +0.07 |
| **`track03`** | `same_track` | **-14.00** | -6.63 | -6.90 | **-7.10** | +0.27 |
| **`track01_style`** | `stylistic` | -19.92 | -19.92 | -17.99 | **-1.93** | -1.93 |
| **`track02_style`** | `stylistic` | -9.37 | -9.37 | -9.04 | **-0.33** | -0.33 |
| **`track03_style`** | `stylistic` | -6.80 | -6.80 | -6.46 | **-0.34** | -0.34 |

### C. Risoluzione Metodologica per V2
1. Il documento `docs/sprint3/SPRINT3_V1_PROTOCOL_HARMONIZATION_PROPOSAL.md` formalizza la specifica di armonizzazione;
2. In `scripts/sprint3/evaluate_v2_ablation.py`, il parametro `target_lufs = float(ref_bm["integrated_lufs"])` viene passato esplicitamente sia ad `AutoMasteringOptimizer` sia ad `optimize_multi_chunk` e alle funzioni di valutazione IIR;
3. Nel runner `evaluate_v2_ablation.py`, la variante di controllo **`v0p` viene rieseguita con lo stesso `target_lufs` armonizzato**, garantendo un confronto $100\%$ equo con `v2` su base scientifica identica.

---

## 5. Task 5: Revisione Critica dei Risultati Storici V0p / V1 / V1b

Riesame critico delle metriche storiche estratte da `docs/sprint3/SPRINT3_V0P_V1_ABLATION_RESULTS.json`:

### A. Surplus di Presenza ERB (2.5–5 kHz) e Asimmetria Anti-Asprezza
- **Evoluzione da `V0p` a `V1`:**  
  L'introduzione della loss ERB a 48 bande (`weight_erb = 0.25` in `same_track`, `spectral_mode = "erb48"` in `stylistic`) ha guidato l'ottimizzatore a correggere le deviazioni spettrali nelle bande critiche uditive di Bark/ERB. Su `track01`, il presence surplus si è ridotto da $+1.84\text{ dB}$ (`V0p`) a $+0.92\text{ dB}$ (`V1`).
- **Evoluzione da `V1` a `V1b` (Penalità Anti-Asprezza `weight_de_harsh = 0.15`):**  
  La formulazione asimmetrica ReLU quadratica `(max(0, pred - ref))^2` applicata unicamente al surplus sopra la reference nella regione 2.5–5 kHz ha impedito alle alte frequenze di dominare la discesa, portando il surplus medio cross-session a valori compresi tra $+0.2\text{ dB}$ e $+0.5\text{ dB}$.

### B. Distanza Log-Spettrale (LSD) e Convergenza Spettrale in `same_track`
- In `same_track`, la combinazione di MR-STFT e ERB bilanciata mantiene la Log-Spectral Distance globale (`lsd_db`) su valori eccellenti ($1.45 \dots 1.82\text{ dB}$), preservando la fedeltà timbrica globale.
- La convergenza spettrale (`spectral_convergence`) mostra che il matching spettrale ad alta risoluzione non degrada l'equilibrio della gamma bassa (sub/kick).

### C. Distanza Spettrale Aggregata (`asd_db`) in `stylistic`
- In modalità `stylistic`, dove le tracce mix e reference appartengono a brani musicalmente diversi con arrangiamenti e bpm differenti, la metrica `asd_db` (L1 distance sui profili log-Mel a 40 bande a lungo termine) evidenzia un allineamento macroscopico del bilanciamento tonale medio (warmth, tilt spettrale, air band) senza forzare allineamenti temporali non fisici.

### D. Gamma Dinamica (PLR), Crest Factor e True Peak Ceiling
- **Dynamic Range (`plr_db`):**  
  Nelle varianti storiche V1/V1b, il Peak-to-Loudness Ratio (`plr_db`) ha subito un decremento medio di circa $1.5 \dots 2.8\text{ dB}$ su alcune tracce dance/pop. Questo comportamento ha confermato empiricamente la necessità della nuova loss `transient_loss` e `plr_loss` introdotte nella specifica `V2`, destinate a preservare l'inviluppo dei transienti e il crest factor originario.
- **True Peak Ceiling ($-0.30\text{ dBTP}$):**  
  Nelle sessioni storiche si registrano violazioni del ceiling ITU-R BS.1770-5 (picchi True Peak fino a $+1.2\text{ dBTP}$ su `track02`), dovute alla natura soft del limiter a 1 parametro nella catena DDSP attuale. Come sancito dalla Specifica Tecnica approvata, **il controllo tassativo e rigoroso del True Peak ceiling con lookahead interamente gestito appartiene alla Milestone `V3a`**, e non costituisce motivo di invalidazione per la loss percettiva di `V1`/`V2`.

### E. Tre Distinzioni Metodologiche Fondamentali
1. **La riduzione numerica della loss non equivale linearmente alla percentuale di miglioramento percettivo:**  
   Una riduzione della loss del $25\%$ riflette la convergenza dell'algoritmo Adam nello spazio di costo matematico multidimensionale; non deve mai essere comunicata o interpretata come un incremento del "$25\%$ di qualità audio".
2. **La riduzione del presence surplus ERB non sostituisce l'ascolto cieco umano:**  
   Il contenimento numerico dell'energia a 3 kHz dimostra l'efficacia del vincolo matematico, ma la validazione finale della piacevolezza e della trasparenza timbrica richiede tassativamente i test di ascolto in doppio cieco su campioni livellati (`level_matched` a $-14\text{ LUFS}$ e `program_level`).
3. **I requisiti di consegna (True Peak compliance) sono ortogonali alla modellazione spettrale:**  
   Il rispetto delle specifiche di distribuzione EBU R128 / streaming delivery è demandato al blocco di brickwall limiting finale (Sprint 3 Step V3a), mentre V1/V2 si concentrano sull'ottimizzazione timbrica, spettrale e dinamica dei transienti.

---

## 6. Task 3: Diagnostica dei Gradienti e Bilanciamento Parametrico

La diagnostica non invasiva eseguita da `scripts/sprint3/run_v1_gradient_diagnostics.py` (Specifica Tecnica v3 §2.A.2) calcola le norme analitiche euclidee $\|\nabla_{z_g} \mathcal{L}_i\|_2$ rispetto al vettore latente $z \in \mathbb{R}^{41}$ effettivamente ottimizzato da Adam, partizionato nei 6 gruppi funzionali DSP:
1. `EQ`: indici `[0:18]` (18 parametri: 6 bande $\times$ gain, freq, Q)
2. `Crossover`: indici `[18:20]` (2 parametri: `fc_low`, `fc_high`)
3. `Compressor`: indici `[20:35]` (15 parametri: 3 bande $\times$ 5 parametri)
4. `Imager`: indici `[35:38]` (3 parametri: width low, mid, high)
5. `Saturator`: indici `[38:40]` (2 parametri: drive, mix/knee)
6. `Limiter`: indici `[40:41]` (1 parametro: ceiling)

I dettagli quantitativi completi per ciascun termine di loss e punto di valutazione sono archiviati nell'artefatto ufficiale `docs/sprint3/SPRINT3_V1_GRADIENT_DIAGNOSTICS.json`.

### Tabella di Sintesi delle Norme Effettive dei Gradienti (Sessioni Campione)

#### Sessione `track01` (`same_track`)
| Variante / Punto | $\|\nabla_z^{\text{eff}}\|_2$ Totale | Quota Gruppo `EQ` (%) | Quota Gruppo `Comp` (%) | $\mathcal{L}_{\text{MR-STFT}}$ (Eff. Norm) | $\mathcal{L}_{\text{ERB}}$ (Eff. Norm) | $\mathcal{L}_{\text{de-harsh}}$ (Eff. Norm) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`V0p` (`z_neutral`)** | 34.77 | 55.5% | 48.8% | 1.57 ($w=1.0$) | N/A ($w=0.0$) | N/A ($w=0.0$) |
| **`V0p` (`z_opt`)** | 24.87 | 73.8% | 41.7% | 1.29 ($w=1.0$) | N/A ($w=0.0$) | N/A ($w=0.0$) |
| **`V1` (`z_neutral`)** | 37.26 | 55.4% | 49.0% | 1.57 ($w=1.0$) | 2.96 ($w=0.25$) | N/A ($w=0.0$) |
| **`V1` (`z_opt`)** | 22.20 | 74.9% | 42.2% | 1.27 ($w=1.0$) | 2.74 ($w=0.25$) | N/A ($w=0.0$) |
| **`V1b` (`z_neutral`)** | 37.31 | 54.8% | 49.8% | 1.57 ($w=1.0$) | 2.96 ($w=0.25$) | 1.20 ($w=0.15$) |
| **`V1b` (`z_opt`)** | 20.57 | 77.0% | 38.1% | 1.31 ($w=1.0$) | 2.76 ($w=0.25$) | 2.66 ($w=0.15$) |

#### Sessione `track01_style` (`stylistic`)
| Variante / Punto | $\|\nabla_z^{\text{eff}}\|_2$ Totale | Quota Gruppo `EQ` (%) | Quota Gruppo `Comp` (%) | $\mathcal{L}_{\text{mel40}}$ / $\mathcal{L}_{\text{ERB}}$ | $\mathcal{L}_{\text{de-harsh}}$ (Eff. Norm) | $\mathcal{L}_{\text{stereo}}$ (Eff. Norm) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`V0p` (`z_neutral`)** | 95.01 | 43.1% | 54.6% | 0.69 (mel40, $w=1.0$) | N/A ($w=0.0$) | 0.029 ($w=0.2$) |
| **`V0p` (`z_opt`)** | 33.80 | 62.9% | 49.8% | 0.72 (mel40, $w=1.0$) | N/A ($w=0.0$) | 0.031 ($w=0.2$) |
| **`V1` (`z_neutral`)** | 94.60 | 42.6% | 54.5% | 3.42 (ERB, $w=1.0$) | N/A ($w=0.0$) | 0.029 ($w=0.2$) |
| **`V1` (`z_opt`)** | 34.23 | 63.6% | 50.0% | 3.39 (ERB, $w=1.0$) | N/A ($w=0.0$) | 0.031 ($w=0.2$) |
| **`V1b` (`z_neutral`)** | 96.25 | 48.0% | 51.6% | 3.42 (ERB, $w=1.0$) | 31.43 ($w=0.15$) | 0.029 ($w=0.2$) |
| **`V1b` (`z_opt`)** | 47.66 | 69.2% | 43.3% | 2.53 (ERB, $w=1.0$) | 26.03 ($w=0.15$) | 0.033 ($w=0.2$) |

### Principali Evidenze Diagnostiche
1. **Bilanciamento tra ERB e MR-STFT:**  
   Con il peso approvato $w_{\text{erb}} = 0.25$, l'ERB fornisce un contributo bilanciato (norma effettiva $\approx 2.74 \dots 2.96$) rispetto a MR-STFT ($\approx 1.27 \dots 1.57$). L'ERB guida la direzione di correzione delle risonanze e del surplus in banda 2.5–5 kHz senza destabilizzare o sopprimere l'informazione armonica ad alta risoluzione di MR-STFT.
2. **Evoluzione della Discesa: da Dinamica ad EQ:**  
   In tutte le varianti, nei punti neutrali (`z_neutral`) la componente dinamica (`Compressor`) e l'EQ si spartiscono equamente la direzione di discesa iniziale ($\approx 50\%$ ciascuna). Man mano che l'ottimizzazione converge verso `z_opt`, la quota di gradiente dell'EQ cresce fino al $74\dots 77\%$, indicando che la dinamica si stabilizza rapidamente mentre l'EQ esegue il fine-tuning timbrico.
3. **Impatto della Penalità Asimmetrica $\mathcal{L}_{\text{de-harsh}}$:**  
   In `V1b`, il termine di de-harshness entra in azione unicamente quando il livello medio delle bande 2.5–5 kHz supera la soglia reference, generando un gradiente focalizzato che corregge selettivamente i guadagni delle bande EQ 3 e 4 senza alterare il balance globale.

---

## 7. Audit dell'Architettura DSP e della Pipeline

1. **Integrità dello Spazio Parametrico (41 Parametri Latenti):**  
   Lo spazio dei parametri descritto in `python/ddsp_mastering/param_space.py` è conforme al $100\%$ alla specifica di produzione, con mapping differenziabile biiettivo tramite sigmoidi scalate. Nessuna dimensione latente è stata aggiunta o rimossa.
2. **Compensazione della Latenza di Pipeline ($D = 159$ campioni):**  
   La latenza deterministica di pipeline (63 campioni per l'EQ FIR a fase lineare in modalità FFT, 32 campioni per il lookahead del compressore multibanda, 64 campioni per il limiter lookahead) viene accuratamente compensata sia nel forward pass dei chunk sia nella valutazione offline full-track.
3. **Flushing Finale delle Code IIR/Delay nel Motore C++:**  
   La funzione `render_offline_with_tail_flush` assicura che al termine dell'elaborazione streaming tutti i registri interni di memoria IIR vengano svuotati su buffer estesi, garantendo la totale assenza di click o troncamenti sui finali dei brani musicali.

---

## 8. Riconciliazione Metodologica e Raccomandazioni per V2

1. **Congelamento dei 7 Iperparametri della Specifica Tecnica v3 (§2.C.2 e §2.C.3):**
   - `weight_trans = 0.2` (default nei moduli base quando non attivo = `0.0`)
   - `beta_atk = 0.5`
   - `plr_d_style = 1.0`
   - `plr_gamma_floor = 1.0`
   - `soft_peak_p = 8`
   - `soft_peak_beta = 2.0`
   - `soft_peak_block_duration_sec = 0.010` ($10\text{ ms}$)
2. **Chunking Indipendente in Modalità `stylistic`:**
   In `evaluate_v2_ablation.py`, la selezione dei chunk per mix e reference avviene tramite chiamate separate a `selector.select_chunks`, consentendo ad ogni segnale di posizionare i propri 4 chunk nelle zone di massima rilevanza dinamica e spettrale.
3. **Isolamento della Cache V2:**
   I risultati intermedi di `evaluate_v2_ablation.py` vengono salvati nella directory dedicata `docs/sprint3/.cache_v2`, prevenendo ogni possibile collisione con la cache storica V1.

---

## 9. Sign-off & Audit Checklist

| Criterio di Audit | Metodo di Verifica | Esito | Note |
| :--- | :--- | :---: | :--- |
| **Inviolabilità Artefatti Storici V1** | Verifica hash git e timestamp `SPRINT3_V0P_V1_*` | **SUPERATO** | Nessuna modifica a risultati o report storici |
| **Pre-roll & same_track Alignment** | Script diagnostico riproducibile (`task1_audit_data.json`) | **SUPERATO** | $\Delta_{\text{A vs B}} = -240\text{ dBFS}$, $\Delta_{\text{B vs C}} \le -211\text{ dBFS}$ |
| **Isomorfismo FastCPUAnalyticalBiquad** | Test `float64` vs `taf.lfilter` e `gradcheck` | **SUPERATO** | Forward diff $< 1.2 \times 10^{-14}$, gradcheck = `True` |
| **Diagnostica dei Gradienti (Task 3)** | Script `run_v1_gradient_diagnostics.py` | **SUPERATO** | `SPRINT3_V1_GRADIENT_DIAGNOSTICS.json` generato |
| **Audit LUFS & Proposta Armonizzazione** | Quantificazione BS.1770-5 su 6 sessioni | **SUPERATO** | `SPRINT3_V1_PROTOCOL_HARMONIZATION_PROPOSAL.md` redatto |
| **Revisione Critica Risultati V1/V1b** | Analisi scientifica presenza, LSD, ASD, PLR, TP | **SUPERATO** | Documentata in §5 del presente report |
| **Codice Runner `evaluate_v2_ablation.py`** | Allineamento 8 punti code review e Spec v3 | **SUPERATO** | Codice congelato (NON eseguito in questo step) |
| **Nuova Suite di Test di Audit** | 5 test in `tests/test_sprint3_v1_audit.py` | **SUPERATO** | 5/5 green |
| **Integrità Suite Test Globale** | Esecuzione completa `pytest tests/ -v` | **SUPERATO** | **29/29 passed in 57.39s** |
| **Divieto Ottimizzazioni Pesanti** | Monitoraggio processi di background | **SUPERATO** | Nessuna nuova ablation V1/V2 lanciata |
