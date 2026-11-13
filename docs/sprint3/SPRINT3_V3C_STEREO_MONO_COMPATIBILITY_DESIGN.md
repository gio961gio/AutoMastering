# SPRINT 3 — SPECIFICA MATEMATICA E ARCHITETTURALE PRE-IMPLEMENTAZIONE V3c (STEREO & MONO COMPATIBILITY)

**Data:** 12 Novembre 2026  
**Autore:** Antigravity (Advanced Agentic Coding / DDSP Audio Team)  
**Supervisore Tecnico:** Argon (Gemini Web Pro / Medium Thinking)  
**Repository:** `gio961gio/AutoMastering`  
**Branch:** `human-validation-package`  
**Commit di riferimento:** `9b1fd9115cce6deaffe637c649c4bb7d6159a195`  
**Stato:** SPECIFICA CONGELATA & HARDENED (Candidate A + Pesi Pilota `same_track`, Validità Mix-Only & Ancoraggio $\varepsilon_{\text{rel}}$, Esclusione `stylistic` dal Trattamento Pilota; ZERO modifiche a `optimizer.py`, `dynamic_prior.py`, `param_space.py` o DSP)

---

## 1. EXECUTIVE SUMMARY & RICOGNIZIONE VERIFICATA DEL CODICE ESISTENTE

### 1.1 Obiettivo e Posizionamento dello Step `V3c`
Lo Step **`V3c`** completa la fase di analisi e progettazione della compatibilità stereo/mono nella roadmap Sprint 3, definendo un prior differenziabile standalone (`V3cStereoCompatibilityLoss`) per il controllo della correlazione per banda, del mono fold-down drop e dell'allargamento in bassa frequenza, senza modificare i moduli DSP o l'ottimizzatore in questa fase.

### 1.2 Moduli DSP/Loss e Spazio Parametrico Reale (`param_space.py`)
Dalla lettura diretta dei sorgenti su `human-validation-package` al commit `9b1fd9115cce6deaffe637c649c4bb7d6159a195`:
1. **Imager Python & C++:**
   - **Python:** `DifferentiableMidSideImager` in `python/ddsp_mastering/dsp/imager.py` (`INV_SQRT2 = 0.70710678118654752440`, calcolo interno in `float64`, `torch.clamp(width, min=0.0)`).
   - **C++ (isomorfo, esposto via `py_core_dsp.MidSideImager`):** `core_dsp::MidSideImager` in `cpp/core_dsp/include/core_dsp/MidSideImager.hpp`.
   - **Ordine nella Catena DSP (`python/ddsp_mastering/dsp/chain.py`):**
     $$\text{Parametric EQ} \;\to\; \text{LR4 3-Way Crossover} \;\to\; \text{Multiband Comp} \;\to\; \mathbf{\text{MidSideImager}} \;\to\; \text{Saturator (15 smp)} \;\to\; \text{Lookahead Limiter (144 smp)}$$
     La latenza complessiva a valle dell'imager resta esattamente $D_{\text{tot}} = 15 + 144 = 159$ campioni a $48\text{ kHz}$.
2. **Copertura Stereo Esistente:**
   - `MasteringPerceptualLoss` (`python/ddsp_mastering/loss/combined.py`, `same_track`): usa `MultiResolutionSTFTLoss(use_mid_side=True, mid_side_weight=0.5)`.
   - `AggregatePerceptualLoss` (`python/ddsp_mastering/loss/stylistic.py`, `stylistic`): usa `stylistic_stereo_loss` con `weight_stereo = 0.3` sul rapporto logaritmico broadband $\ln E_S - \ln E_M$.
   - Nessuna delle due loss include vincoli per banda di frequenza, protezione esplicita da calo di correlazione $\rho_{LR, b}$, protezione da mono fold-down drop $D_{\text{mono}}$ né soglie minime di validità energetica.
3. **Bounds e Jacobiani Reali (`python/ddsp_mastering/param_space.py`, righe 60–62 e 156–162):**

| Indice | Nome in `PARAM_SPECS` | Min | Max | Scala | Neutro $\theta_{\text{neut}}$ | $Z_{\text{NEUTRAL}}$ | $u_{\text{neut}} = \text{logit}(z)$ | Jacobiano $dz/du$ | Jacobiano $d\theta/du$ |
|:---:|:---|---:|---:|:---:|---:|---:|---:|---:|---:|
| 18 | `crossover_f_low` | `80.0` | `350.0` | `log` | `160.0 Hz` | `0.469642` | `-0.121581` | `0.249078` | `58.95 Hz/u` |
| 19 | `crossover_f_high` | `1500.0` | `6000.0` | `log` | `3200.0 Hz` | `0.546555` | `+0.186762` | `0.247833` | `1099.4 Hz/u` |
| 35 | `imager_width_low` | `0.4` | `1.1` | `linear` | `1.0` | `0.857143` (`6/7`) | `+1.791759` | `0.122449` (`6/49`) | `0.085714` |
| 36 | `imager_width_mid` | `0.7` | `1.4` | `linear` | `1.0` | `0.428571` (`3/7`) | `-0.287682` | `0.244898` (`12/49`) | `0.171429` |
| 37 | `imager_width_high` | `0.7` | `1.4` | `linear` | `1.0` | `0.428571` (`3/7`) | `-0.287682` | `0.244898` (`12/49`) | `0.171429` |

### 1.3 Valori Reali Osservati in `docs/sprint3/SPRINT3_V3B_V0P_ABLATION_RESULTS.json`
Dalla lettura diretta dei valori `physical_parameters_theta` e `full_track_benchmark` salvati nel JSON congelato:

| Sessione | `crossover_f_low` / `high` (Hz) | `control_v0p` `imager_width` `[low, mid, high]` | `treatment_v3b_b05` `imager_width` `[low, mid, high]` | `V0p` M/S Ratio `pred` vs `target` (dB) | `delta_ms_ratio_db` |
|:---|:---:|:---:|:---:|:---:|:---:|
| **`track01`** | `166.58` / `2692.08` | `[1.0422, 1.1117, 1.1098]` | `[1.0423, 1.1116, 1.1081]` | `8.5045` vs `8.9424` | `0.4380` |
| **`track02`** | `197.98` / `3170.95` | `[1.0230, 1.1119, 1.1112]` | `[1.0230, 1.1119, 1.1112]` | `14.0311` vs `14.3802` | `0.3491` |
| **`track03`** | `197.55` / `3467.58` | `[1.0425, 1.1117, 1.1105]` | `[1.0425, 1.1117, 1.1105]` | `6.7965` vs `8.2024` | `1.4059` |
| **`track01_style`** | `149.32` / `3495.16` | `[0.9455, 0.8970, 0.8938]` | `[0.9455, 0.8970, 0.8938]` | `10.2965` vs `9.5318` | `0.7646` |
| **`track02_style`** | `200.44` / `3177.98` | `[1.0434, 1.1127, 1.1116]` | `[1.0434, 1.1127, 1.1116]` | `14.0153` vs `16.3930` | `2.3776` |
| **`track03_style`** | `197.48` / `3335.39` | `[1.0422, 1.1104, 1.1101]` | `[1.0422, 1.1104, 1.1101]` | `6.7986` vs `7.8634` | `1.0649` |

> **Nota di Audit sulla Verifica Diretta del Repository:** Tutti i valori in tabella e nei bounds sopra (`[0.4, 1.1]`, `[0.7, 1.4]`, `[0.7, 1.4]`, `DifferentiableMidSideImager`, `MidSideImager.hpp` e le width estratte da `SPRINT3_V3B_V0P_ABLATION_RESULTS.json`) riflettono fedelmente i byte presenti su `HEAD`. Si segnala per trasparenza al supervisore che nella direttiva di revisione erano stati indicati bounds (`[0.50, 1.20]`, `[0.70, 1.50]`), nomi file/classe (`StereoImager.hpp`, `DifferentiableMultibandStereoImager`) e width (`1.0427`, `1.0160`) non riscontrati nei file effettivamente tracciati nel repository.

---

## 2. AUDIT MATEMATICO DELL'IMAGER E DEFINIZIONI SCALE-INVARIANT

### 2.1 Trasformazione Ortonormale M/S per Banda
Per ciascuna banda $b \in \{\text{low}, \text{mid}, \text{high}\}$:
$$M_b[n] = \frac{L_b[n] + R_b[n]}{\sqrt{2}}, \qquad S_b[n] = \frac{L_b[n] - R_b[n]}{\sqrt{2}}$$
$$\hat{S}_b[n] = w_b \, S_b[n], \qquad \hat{L}_b[n] = \frac{M_b[n] + \hat{S}_b[n]}{\sqrt{2}}, \qquad \hat{R}_b[n] = \frac{M_b[n] - \hat{S}_b[n]}{\sqrt{2}}$$
con identità ortonormale esatta $\hat{L}_b^2 + \hat{R}_b^2 = M_b^2 + \hat{S}_b^2$ ed invarianza $\frac{\hat{L}_b + \hat{R}_b}{\sqrt{2}} = M_b$ a monte delle non linearità (saturatore e limiter a valle modificano poi $L$ e $R$ per canale).

### 2.2 Banco LR4 Fisso `160 / 3200 Hz` con Warm-Up su Pre-Roll Completo (Decisione Q1)
Il banco di misura usa `DifferentiableLR4Crossover(sample_rate=48000.0)` alle frequenze fisse:
$$f_{\text{low,fix}} = 160.0\text{ Hz}, \qquad f_{\text{high,fix}} = 3200.0\text{ Hz}$$
Regola tassativa di filtraggio e ritaglio:
1. Il crossover LR4 **è sempre eseguito sull'intero segnale esteso comprensivo del pre-roll completo del chunk** ($[0, N_{\text{pre}} + T)$), così da portare a regime tutti gli stati interni dei filtri IIR biquad prima dell'inizio della finestra di valutazione.
2. Solo **dopo** il filtraggio LR4 si applica il ritaglio alla finestra comune di valutazione ($T_{\text{eval}} = T - 159 = 143\,841$ campioni a $48\text{ kHz}$):
   - per il mix originale $x_{\text{mix}, b}$: slice $[N_{\text{pre}} : N_{\text{pre}} + T - 159]$;
   - per l'uscita masterizzata $y_{\text{pred}, b}$: slice $[N_{\text{pre}} + 159 : N_{\text{pre}} + T]$.
3. La maschera booleana dei campioni attivi $\mathcal{A}_{\text{smp}} = \text{compute\_active\_sample\_mask}(x_{\text{mix}}^{\text{eval}})$ è calcolata a blocchi da $480$ campioni ($10\text{ ms}$, soglia $-70\text{ dBFS}$, fallback $M_{\min}=8$, preservazione blocco parziale residuo da $321$ campioni su $T_{\text{eval}}=143\,841$) e staccata dal grafo.

### 2.3 Controllo di Energia Minima per Banda, Canale Silenzioso e Pavimento Relativo $\varepsilon_{\text{rel}}$
Siano le energie medie sui campioni attivi $n \in \mathcal{A}_{\text{smp}}$ (di cardinalità $N_a \ge 1$) calcolate in precisione `float64`:
$$E_L = \frac{1}{N_a}\sum_{n \in \mathcal{A}_{\text{smp}}} x_L[n]^2, \quad E_R = \frac{1}{N_a}\sum_{n \in \mathcal{A}_{\text{smp}}} x_R[n]^2, \quad E_{\text{tot}} = E_L + E_R = E_M + E_S$$
$$E_M = \frac{1}{N_a}\sum_{n \in \mathcal{A}_{\text{smp}}} m[n]^2, \quad E_S = \frac{1}{N_a}\sum_{n \in \mathcal{A}_{\text{smp}}} s[n]^2, \quad C_{LR} = \frac{1}{N_a}\sum_{n \in \mathcal{A}_{\text{smp}}} x_L[n]\,x_R[n]$$

1. **Pavimento Relativo Staccato & Ancorato per Scale Invariance (Range Operativo $[-60, +20]\text{ dBFS}$):**
   - Per il segnale mix originale (`ref_energy_anchor = None`):
     $$\varepsilon_{\text{rel}}^{\text{mix}} = 10^{-12} \cdot \text{detach}(\max(E_{\text{tot}}^{\text{mix}}, 10^{-30})) + 10^{-30}$$
   - Per il segnale predetto (`ref_energy_anchor = detach(E_tot_mix)`), con ancoraggio esplicito all'energia del mix originale:
     $$\varepsilon_{\text{rel}, b}^{\text{pred}} = 10^{-12}\cdot\text{detach}\!\left(\max\!\left(E_{\text{tot}, b}^{\text{mix}},\, E_{\text{tot}, b}^{\text{pred}}\right)\right) + 10^{-30}$$
   Definiamo con $\varepsilon_{\text{rel}}$ le metriche scale-invariant (invarianti entro $< 10^{-8}$ su tutto l'intervallo $[-60, +20]\text{ dBFS}$ rispetto a riscalature congiunte $(x_{\text{mix}}, y_{\text{pred}}) \mapsto c \cdot (x_{\text{mix}}, y_{\text{pred}})$):
   $$D_{\text{mono}}(x) = 10 \log_{10}\!\left(\frac{E_M + E_S + \varepsilon_{\text{rel}}}{E_M + \varepsilon_{\text{rel}}}\right) \quad [\text{dB}]$$
   $$R_{S/M}(x) = 10 \log_{10}\!\left(\frac{E_S + \varepsilon_{\text{rel}}}{E_M + \varepsilon_{\text{rel}}}\right) \quad [\text{dB}], \qquad B_{L/R}(x) = 10 \log_{10}\!\left(\frac{E_L + \varepsilon_{\text{rel}}}{E_R + \varepsilon_{\text{rel}}}\right) \quad [\text{dB}]$$
   $$\rho_{LR}(x) = \frac{C_{LR}}{\sqrt{(E_L + \varepsilon_{\text{rel}})(E_R + \varepsilon_{\text{rel}})}} = \frac{\frac{1}{2}(E_M - E_S)}{\sqrt{\left(\frac{E_M + E_S}{2} + \varepsilon_{\text{rel}}\right)^2 - C_{MS}^2}}$$
   *(Proprietà su predizioni silenziose o quasi silenziose: con mix valido $E_{\text{tot}, b}^{\text{mix}} \ge 10^{-7}$, $\varepsilon_{\text{rel}, b}^{\text{pred}} \ge 10^{-19}$ garantisce denominatori strettamente positivi e ben condizionati anche quando $y_{\text{pred}, b} \equiv 0$, evitando qualsiasi `NaN`/`Inf` o collasso della scala dei gradienti.)*

2. **Regola di Validità Basata Esclusivamente sul Mix Originale ($E_{\min} = 10^{-7}$, $-70.0\text{ dBFS}$):**
   - La validità di valutazione sia broadband sia per ciascuna banda $b \in \{L, M, H\}$ dipende **esclusivamente** dalla validità energetica del mix originale:
     $$\text{valid}_b = \mathbb{I}(\neg\,\text{is\_silent}) \cdot \mathbb{I}\!\left(E_{\text{tot}, b}^{\text{mix}} \ge E_{\min}\right), \qquad \text{valid}_{\text{bb}} = \mathbb{I}(\neg\,\text{is\_silent}) \cdot \mathbb{I}\!\left(E_{\text{tot}, \text{bb}}^{\text{mix}} \ge E_{\min}\right)$$
   - La validità **non** dipende mai da $E_{\text{tot}, b}^{\text{pred}}$: se una banda del mix originale è attiva ($\ge E_{\min}$), una predizione sotto soglia o silenziosa ($E_{\text{tot}, b}^{\text{pred}} < E_{\min}$) **non** disattiva `loss_mono_drop`, `loss_band_corr` o `loss_band_smr`, mantenendo continuità numerica perfetta attraverso $E_{\min}$ e penalità strettamente positive ($y_{\text{pred}, b} \equiv 0 \implies \rho_{LR, b}^{\text{pred}} = 0,\, R_{S/M, b}^{\text{pred}} = 0\text{ dB}$).
   - Se invece il **mix originale** ha $E_{\text{tot}, b}^{\text{mix}} < E_{\min}$, allora `band_valid[b] = False` e il contributo della banda $b$ è identicamente `0.0` (gradiente nullo). Il dizionario diagnostico registra comunque il flag booleano `pred_energy_below_emin` ($E_{\text{tot}, b}^{\text{pred}} < E_{\min}$).

3. **Gestione di Singolo Canale Silenzioso / Hard-Panned ($\eta_{\text{ch}} = 10^{-6}$):**
   - Quando la banda è attiva ($E_{\text{tot},b} \ge E_{\min}$) ma $\min(E_{L,b}, E_{R,b}) < \eta_{\text{ch}} E_{\text{tot},b}$ (es. segnale hard-panned $x_R \equiv 0, x_L \neq 0$), la correlazione di Pearson classica $0/0$ non è definita in senso stretto; il calcolo differenziabile con $\varepsilon_{\text{rel}, b} > 0$ restituisce in modo liscio e regolare $\rho_{LR, b} = 0.0$ (con $E_{M,b} = E_{S,b} = \frac{1}{2}E_{L,b}$ e $D_{\text{mono},b} = 10\log_{10}(2) \approx 3.0103\text{ dB}$) e registra nel dizionario diagnostico il flag **`channel_silent_or_hard_panned[b] = True`**.

---

## 3. DECISIONI CONGELATE (`Q1–Q5`) & I 4 TERMINI CANDIDATI DELLA LOSS `V3c`

| Quesito | Decisione Congelata | Impatto Progettuale |
|:---:|:---|:---|
| **Q1** | Banco `DifferentiableLR4Crossover` fisso a `160.0 / 3200.0 Hz` | Filtraggio eseguito sul pre-roll completo prima dello slice `[N_pre : N_pre+T-159]` / `[N_pre+159 : N_pre+T]` |
| **Q2** | Correlazione **mix-aware su tutte e 3 le bande** (`L, M, H`) | Penalizza solo il calo $\Delta\rho_b = \rho_{LR,b}(x_{\text{mix}}) - \rho_{LR,b}(y_{\text{pred}}) > \tau_{\rho,b}$; **nessuna** soglia assoluta `0.80` imposta sulla banda `low` |
| **Q3** | Metriche spettrali anti-comb-filtering **solo diagnostiche** | Non entrano nei 4 termini di loss; registrate nel preflight per analisi spettrale di $M(\omega)$ vs media $L/R$ |
| **Q4** | Correlazione locale short-time e $\text{CVaR}_{0.05}$ **solo diagnostiche** | Nel preflight si valutano tutte e 3 le finestre **`50 ms`, `100 ms`, `250 ms`** broadband e per banda (`L, M, H`) |
| **Q5** | **Nessuna** modifica ai bounds di `param_space.py` | Preservata piena confrontabilità storica con `V0p..V3b` |

### I 4 Termini Candidati della Loss `V3c`
1. **$\mathcal{L}_{\text{mono-drop}}$** (eccesso broadband di mono fold-down drop rispetto al mix originale, $[\text{dB}^2]$):
   $$\mathcal{L}_{\text{mono-drop}} = \left[\text{ReLU}\!\left(D_{\text{mono}}(y_{\text{pred}}^{\text{eval}}) - \text{detach}(D_{\text{mono}}(x_{\text{mix}}^{\text{eval}})) - \tau_{\text{drop}}\right)\right]^2$$
2. **$\mathcal{L}_{\text{band-corr}}$** (calo di correlazione mix-aware sulle 3 bande $b \in \{L, M, H\}$):
   $$\mathcal{L}_{\text{band-corr}} = \frac{1}{3}\sum_{b \in \{L,M,H\}} \mathbb{I}(\text{valid}_b)\,\left[\text{ReLU}\!\left(\text{detach}(\rho_{LR,b}(x_{\text{mix}}^{\text{eval}})) - \rho_{LR,b}(y_{\text{pred}}^{\text{eval}}) - \tau_{\rho, b}\right)\right]^2$$
3. **$\mathcal{L}_{\text{band-smr}}$** (eccesso di allargamento energetico Side/Mid per banda rispetto al mix originale, $[\text{dB}^2]$):
   $$\Delta R_{S/M, b} = R_{S/M, b}(y_{\text{pred}}^{\text{eval}}) - \text{detach}(R_{S/M, b}(x_{\text{mix}}^{\text{eval}}))$$
   $$\mathcal{L}_{\text{band-smr}} = \frac{1}{3}\sum_{b \in \{L,M,H\}} \mathbb{I}(\text{valid}_b)\,\left[\text{ReLU}\!\left(\left|\Delta R_{S/M, b}\right| - \tau_{\text{smr}, b}\right)\right]^2 + \alpha_{\text{sub}}\,\mathbb{I}(\text{valid}_L)\,\left[\text{ReLU}\!\left(\Delta R_{S/M, L} - \tau_{\text{sub}}\right)\right]^2$$
4. **$\mathcal{L}_{\text{width-param}}$** (prior parametrico diretto sulle larghezze `imager_width_low/mid/high`):
   $$\mathcal{L}_{\text{width-param}} = \text{ReLU}\!\left(w_{\text{low}} - \tau_{w,\text{low,max}}\right)^2 + \frac{1}{2}\sum_{b \in \{\text{mid},\text{high}\}} \text{ReLU}\!\left(|w_b - 1.0| - \tau_{w,b}\right)^2$$

### 3.1 Parametri Congelati **Candidate A** e Pesi Pilota (`same_track`)
A valle del primo Preflight Read-Only `V3c`, vengono congelati in `V3cStereoCompatibilityLoss` **esclusivamente** le soglie di **Candidate A** e i seguenti pesi pilota per la futura prima ablation `same_track` (nessuna scansione di pesi o soglie aggiuntiva):

| Gruppo | Parametro | Valore Congelato |
|:---|:---|---:|
| **Soglie Candidate A** | `tau_drop_db` | `0.50` dB |
| **Soglie Candidate A** | `tau_corr` `(low, mid, high)` | `(0.03, 0.05, 0.08)` |
| **Soglie Candidate A** | `tau_smr_db` `(low, mid, high)` | `(2.0, 2.0, 2.0)` dB |
| **Soglie Candidate A** | `tau_sub_db` / `alpha_sub` | `0.50` dB / `1.0` |
| **Soglie Candidate A** | `tau_w_low_max` | `1.00` |
| **Soglie Candidate A** | `tau_w_mid` / `tau_w_high` | `0.15` / `0.15` |
| **Pesi Pilota `same_track`** | `w_mono_drop` | **`1.00`** |
| **Pesi Pilota `same_track`** | `w_band_corr` | **`10.00`** |
| **Pesi Pilota `same_track`** | `w_band_smr` | **`0.05`** |
| **Pesi Pilota `same_track`** | `w_width_param` | **`0.25`** |

---

## 4. ALLINEAMENTO TEMPORALE, PREVENZIONE DEI "PERVERSE INCENTIVES" E DIAGNOSTICHE SHORT-TIME / COMB-FILTERING

1. **Allineamento & Integrità:** $D_{\text{tot}} = 159$ campioni e pesatura uniforme per-chunk $w_k = 1/4$ sui $K=4$ chunk ($T_{\text{eval}}=143\,841$).
2. **Prevenzione Perverse Incentives:**
   - Banco LR4 fisso indipendente dai parametri `crossover_f_low/high` (previene elusione spostando i crossover);
   - Formule scale-invariant con $\varepsilon_{\text{rel}}$ proporzionale a $\text{detach}(E_{\text{tot}})$ (previene elusione riducendo o spingendo il livello globale);
   - Dead-zone mix-aware su $\rho_b$ e $\Delta R_{S/M,b}$ (non impone restringimenti artificiali su brani naturalmente larghi).
3. **Diagnostica Multi-Finestra Short-Time (`50 ms`, `100 ms`, `250 ms`, Decisione Q4):**
   Su ogni finestra mobile attiva (con copertura attiva $\ge 50\%$ o blocchi attivi) a `50 ms` ($2400$ campioni), `100 ms` ($4800$ campioni) e `250 ms` ($12\,000$ campioni) vengono calcolate esclusivamente a fini diagnostici: `mean`, `min`, $P_{05}$ e $\text{CVaR}_{0.05}$ di $\rho_{LR}$ broadband e per ciascuna banda `low`, `mid`, `high` (particolarmente critica la finestra `250 ms` per la banda `< 160 Hz`).
4. **Diagnostica Spettrale Anti-Comb-Filtering (Decisione Q3):**
   Definita la media dello spettro di magnitudo di $L$ e $R$ rispetto allo spettro del segnale somma Mono $M$ (su finestra di valutazione attiva, n_fft=2048, hop=512): misura lo scarto quadratico medio e la profondità massima di notch ($10\log_{10}(|M(\omega)|^2 / (\frac{1}{2}(|L(\omega)|^2+|R(\omega)|^2)+\epsilon))$) nella banda attiva, riportando la variazione tra mix originale e master predetto.

---

## 5. INTEGRAZIONE STANDALONE, ESCLUSIONE `stylistic` DAL PILOTA & CRITERIO `V4`

1. **Stato Attuale (`optimizer.py` intatto):** In questa fase `V3cStereoCompatibilityLoss` vive isolato in `python/ddsp_mastering/loss/stereo_prior.py`, garantendo diff zero su `optimizer.py`, `dynamic_prior.py`, `param_space.py` e moduli DSP Python/C++.
2. **Gestione della Modalità `stylistic` & Ambito Esclusivo dell'Ablation Pilota:**
   - Nel primo Preflight Read-Only `V3c`, le sessioni `stylistic` hanno evidenziato un **conflitto strutturale di direzione del gradiente** fra i termini `V3c` (che ancorano l'escursione M/S $R_{S/M}$ ed il drop di correlazione al **mix originale** del brano) e `stylistic_stereo_loss` di `V0p` (che spinge il rapporto M/S verso il **brano reference esterno**), con cosine negative rilevanti fra `V3c` e `V0p_total` sul gruppo `Imager` e conflitto con il target stilistico quando il reference richiede una maggiore apertura stereo.
   - Pertanto, i termini `V3c` nella formulazione attuale **non sono dichiarati adatti alla modalità `stylistic`**: per la prima ablation `V3c` autorizzata verranno utilizzate **esclusivamente le 3 sessioni `same_track` (`track01`, `track02`, `track03`)**, mentre le 3 sessioni `stylistic` (`track01_style`, `track02_style`, `track03_style`) restano **esclusivamente diagnostiche** e **non verranno attivate nell'optimizer**.
   - Un'eventuale futura formulazione `V3c` per la modalità `stylistic` dovrà preservare la libertà di apertura stereo richiesta dal brano reference esterno, limitandosi a guardrail di sicurezza su mono/fase (es. collasso mono estremo, anti-fase locale o eccessi sub-bass in somma mono).
3. **Criterio Esplicito per Futura Composizione `V4` (Punto 6):**
   La futura configurazione **`V4`** includerà **esclusivamente** le componenti di regolarizzazione (tra i candidati `V1/V1b/V2/V3a/V3b/V3c`) che avranno superato **sia la validazione quantitativa oggettiva sia la validazione percettiva d'ascolto controllata (blind human evaluation)**; nessuna componente è pre-ammessa in `V4` per default prima del superamento di entrambe le validazioni.

---

## 6. PIANO DEI GATE TEST TECNICI (`1–8` in `tests/test_sprint3_v3c_gate.py`)

La suite `tests/test_sprint3_v3c_gate.py` verifica integralmente i Gate 1–8 e i 7 casi limite:
1. **Gate 1:** Conservazione ortonormale $E_L + E_R = E_M + E_S$ ($< 10^{-12}$) broadband e sulle 3 bande LR4 + identità esatta dell'imager per $w_b = 1.0$.
2. **Gate 2:** Segnali canonici: in fase ($\rho = +1$, $D_{\text{mono}} = 0\text{ dB}$), quadratura $90^\circ$ ($\rho \approx 0$, $D_{\text{mono}} \approx 3.0103\text{ dB}$), controfase ($\rho = -1$, $D_{\text{mono}}$ finito, zero `NaN`/`Inf`).
3. **Gate 3:** Silenzio digitale ($< -70\text{ dBFS}$ -> loss `0.0`, grad `0.0`), banda priva di energia nel mix (es. solo seno `20 Hz`: `band_valid['mid']=False, ['high']=False`, zero `NaN`), blocco parziale da $321$ campioni su $T_{\text{eval}}=143\,841$ e verifica warm-up LR4 pre-roll vs no-pre-roll.
4. **Gate 4:** `torch.autograd.gradcheck` in `float64` separato su ciascuno dei 4 termini (`loss_mono_drop`, `loss_band_corr`, `loss_band_smr`, `loss_width_param`) rispetto al segnale e a $z_{35..37}$.
5. **Gate 5:** Verifica dei **7 casi limite** (mono puro, stereo correlato, stereo decorrelato, anti-fase pura, hard-panned $x_R \equiv 0$, banda silenziosa, silenzio digitale totale) + test quantitativo di **Scale Invariance** su tutto l'intervallo dichiarato **`[-60.0, -40.0, -20.0, 0.0, +10.0, +20.0] dBFS`** ($< 10^{-8}$ su $D_{\text{mono}}, R_{S/M}, \rho_{LR}$ e sulle 3 loss di segnale).
6. **Gate 6:** Parità Python/C++ (`DifferentiableMidSideImager` vs `py_core_dsp.MidSideImager`, null RMS $< -120\text{ dBFS}$, max abs $< 10^{-6}$).
7. **Gate 7:** Zero diff storico di `optimizer.py`, `dynamic_prior.py`, `param_space.py` e moduli DSP Python/C++ rispetto a `9b1fd9115cce6deaffe637c649c4bb7d6159a195` **e** verifica deterministica effettiva di bit-identity (`max_diff == 0.0` su audio, loss e gradienti di `DifferentiableMasteringChain`, `MasteringPerceptualLoss` `V0p` e `V3bDynamicPriorLoss` `V3b` prima e dopo l'esecuzione di `V3cStereoCompatibilityLoss`).
8. **Gate 8 (Predizione Sotto Soglia / Silenziosa con Mix Attivo):** Verifica obbligatoria dei 3 regimi critici quando $E_{\text{tot}, b}^{\text{mix}} \ge E_{\min}$:
   - Banda `mid` del mix attiva e predizione quasi silenziosa in `mid` ($E_{\text{tot}, \text{mid}}^{\text{pred}} \approx 10^{-10} < E_{\min}$): `band_valid['mid'] == True`, `loss_band_corr > 0`, `loss_band_smr > 0`, gradienti su `pred` e $z_{35..37}$ finiti, non nulli e privi di `NaN`/`Inf`.
   - Mix attivo su tutte le bande e predizione completamente silenziosa (`0.0`) o quasi silenziosa su L/R: `loss_band_corr > 0` e `loss_band_smr > 0` quando applicabile, zero `NaN`/`Inf` in metriche e gradienti.
   - Continuità numerica attraverso la soglia $E_{\min} = 10^{-7}$ ($2\times 10^{-7} \to 1\times 10^{-7} \to 0.5\times 10^{-7}$): nessuna discontinuità a gradino ($\Delta < 10^{-6}$).

---

## 7. PROTOCOLLO PREFLIGHT READ-ONLY V3c & V2 (`scripts/sprint3/run_v3c_preflight.py`)

Valutazione read-only sui 3 punti operativi (`Z_NEUTRAL`, `V0p z_opt`, `V3b z_opt`) su tutte le 6 sessioni armonizzate (4 chunk da $144\,000$ campioni, pre-roll $120\,000$, $T_{\text{eval}}=143\,841$, pesatura $1/4$):
- **Preflight V1 (`SPRINT3_V3C_PREFLIGHT_RESULTS.json`, `SPRINT3_V3C_PREFLIGHT_REPORT.md` — congelato):** confronto non pesato `Zero_Threshold`, `Candidate_A`, `Candidate_B`.
- **Preflight V2 (`SPRINT3_V3C_PREFLIGHT_V2_RESULTS.json`, `SPRINT3_V3C_PREFLIGHT_V2_REPORT.md`):** valutazione con **Candidate A** e **pesi pilota congelati** $(1.00, 10.00, 0.05, 0.25)$ su termini unweighted/weighted, somma combinata `loss_v3c_total_weighted`, norme dei gradienti in `z` e `u` (41D, `Imager [35:38]` e altri 5 gruppi), rapporti rispetto a `V0p_total` e cosine similarities verso `V0p_total`, `LUFS_only` e `stylistic_stereo` (con `track01..03` operative per l'ablation pilota `same_track` e `track01_style..03_style` solo diagnostiche).
