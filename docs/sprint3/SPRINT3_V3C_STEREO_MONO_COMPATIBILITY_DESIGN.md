# SPRINT 3 — SPECIFICA MATEMATICA E ARCHITETTURALE PRE-IMPLEMENTAZIONE V3c (STEREO & MONO COMPATIBILITY)

**Data:** 12 Novembre 2026  
**Autore:** Antigravity (Advanced Agentic Coding / DDSP Audio Team)  
**Supervisore Tecnico:** Argon (Gemini Web Pro / Medium Thinking)  
**Repository:** `gio961gio/AutoMastering`  
**Branch:** `human-validation-package`  
**Commit di riferimento:** `9cc448f8c83c0e1361638cffe4d09873e8bf8428`  
**Stato:** SPECIFICA PROGETTUALE PRE-IMPLEMENTAZIONE (ZERO modifiche a DSP, loss, optimizer o test in questa fase)

---

## 1. EXECUTIVE SUMMARY & RICOGNIZIONE DEL CODICE ESISTENTE

### 1.1 Obiettivo e Posizionamento dello Step `V3c`
Lo Step **`V3c`** completa l'asse di regolarizzazione spaziale/fase della roadmap Sprint 3, introducendo un prior differenziabile di **compatibilità stereo e mono-foldback** progettato per lavorare in composizione pulita con `V0p`, `V3a` (EQ Prior) e `V3b` (Dynamic Priors) senza modificare l'architettura DSP sottostante.

### 1.2 Verifica Esatta dei File Reali, Limiti Parametrici e Riconciliazione Analitica
Dall'ispezione diretta dei sorgenti su `human-validation-package` al commit `9cc448f8c83c0e1361638cffe4d09873e8bf8428`:
1. **Moduli DSP e Loss Verificati:**
   - **Imager Python:** `DifferentiableMidSideImager` in `python/ddsp_mastering/dsp/imager.py` (costante ortonormale `INV_SQRT2 = 0.70710678118654752440`, `float64`, `torch.clamp(width, min=0.0)`).
   - **Imager C++ (per parity test):** `core_dsp::MidSideImager` in `cpp/core_dsp/include/core_dsp/MidSideImager.hpp` *(nota nominale: il file nel repository si chiama `MidSideImager.hpp`, non `StereoImager.hpp`)*.
   - **Ordine nella Catena DSP (`python/ddsp_mastering/dsp/chain.py`):**
     $$\text{Parametric EQ} \;\to\; \text{LR4 3-Way Crossover} \;\to\; \text{Multiband Comp} \;\to\; \mathbf{\text{MidSideImager}} \;\to\; \text{Saturator (15 smp)} \;\to\; \text{Lookahead Limiter (144 smp)}$$
     L'imager introduce **0 campioni** di latenza propria; l'intera latenza di catena resta $D_{\text{tot}} = 15 + 144 = 159$ campioni a valle dell'imager.
   - **Copertura Stereo Attuale nelle Loss:**
     - `MasteringPerceptualLoss` (`python/ddsp_mastering/loss/combined.py`, modalità `same_track`): usa `MultiResolutionSTFTLoss(use_mid_side=True, mid_side_weight=0.5)`, confrontando L/R e M/S punto-a-punto rispetto al reference. Non impone vincoli espliciti su correlazione di fase, drop in mono o larghezza in bassa frequenza rispetto al mix di ingresso.
     - `AggregatePerceptualLoss` (`python/ddsp_mastering/loss/stylistic.py`, modalità `stylistic`): calcola `stylistic_stereo_loss` con `weight_stereo = 0.3` come $\left|\left(\ln\bar{E}_{S,\text{pred}} - \ln\bar{E}_{M,\text{pred}}\right) - \left(\ln\bar{E}_{S,\text{ref}} - \ln\bar{E}_{M,\text{ref}}\right)\right|$ sull'intera banda larga. Nessun vincolo per banda di frequenza, nessuna metrica di correlazione $\rho_b$ e nessuna protezione contro cancellazioni di fase in somma mono.

2. **Spazio Parametrico Reale (`python/ddsp_mastering/param_space.py`) vs Ipotesi Generiche:**
   A differenza dell'ipotesi preliminare di bounds simmetrici $[0.0, 2.0]$ centrati a $z = 0.5$ ($u = 0$), il codice reale definisce le dimensioni 35–37 (e crossover 18–19) con limiti stretti e asimmetrici:

| Indice | Nome in `PARAM_SPECS` | Min | Max | Scala | Neutro $\theta_{\text{neut}}$ | $Z_{\text{NEUTRAL}}$ | $u_{\text{neut}} = \text{logit}(z)$ | Jacobiano $dz/du$ | Jacobiano $d\theta/du$ |
|:---:|:---|---:|---:|:---:|---:|---:|---:|---:|---:|
| 18 | `crossover_f_low` | `80.0` | `350.0` | `log` | `160.0 Hz` | `0.469642` | `-0.121581` | `0.249078` | `58.95 Hz/u` |
| 19 | `crossover_f_high` | `1500.0` | `6000.0` | `log` | `3200.0 Hz` | `0.546555` | `+0.186762` | `0.247833` | `1099.4 Hz/u` |
| 35 | `imager_width_low` | `0.4` | `1.1` | `linear` | `1.0` | `0.857143` | `+1.791759` | `0.122449` | `0.085714` |
| 36 | `imager_width_mid` | `0.7` | `1.4` | `linear` | `1.0` | `0.428571` | `-0.287682` | `0.244898` | `0.171429` |
| 37 | `imager_width_high` | `0.7` | `1.4` | `linear` | `1.0` | `0.428571` | `-0.287682` | `0.244898` | `0.171429` |

   - **Implicazione Critica sull'Ottimizzazione:** Poiché $Z_{\text{NEUTRAL}}[35] = 6/7 \approx 0.8571$, la derivata locale $dw_{\text{low}}/du_{35} = 0.7 \cdot z(1-z) = 0.0857$ all'inizializzazione è già esattamente **la metà** rispetto a `mid`/`high` ($0.1714$), e si schiaccia ulteriormente verso zero avvicinandosi a $w_{\text{low}} \to 1.1$. Di converso, l'escursione fisica in bassa frequenza è già confinata in $[0.4, 1.1]$, mentre `mid` e `high` possono spingersi fino a `1.4` ($+40\%$ Side).

### 1.3 Rilevamenti Empirici Reali da `SPRINT3_V3B_V0P_ABLATION_RESULTS.json`
I valori effettivi ottimizzati dalle sessioni armonizzate `Control V0p` e `Treatment V3b-B0.5` nel JSON congelato sono:

| Sessione | `crossover_f_low` / `high` (Hz) | `V0p` `imager_width` `[low, mid, high]` | `V3b` `imager_width` `[low, mid, high]` | `V0p` M/S Ratio `pred` vs `target` (dB) | `delta_ms_ratio_db` |
|:---|:---:|:---:|:---:|:---:|:---:|
| **`track01`** | `166.58` / `2692.08` | `[1.0422, 1.1117, 1.1098]` | `[1.0423, 1.1116, 1.1081]` | `8.5045` vs `8.9424` | `0.4380` |
| **`track02`** | `197.98` / `3170.95` | `[1.0230, 1.1119, 1.1112]` | `[1.0230, 1.1119, 1.1112]` | `14.0311` vs `14.3802` | `0.3491` |
| **`track03`** | `197.55` / `3467.58` | `[1.0425, 1.1117, 1.1105]` | `[1.0425, 1.1117, 1.1105]` | `6.7965` vs `8.2024` | `1.4059` |
| **`track01_style`** | `149.32` / `3495.16` | `[0.9455, 0.8970, 0.8938]` | `[0.9455, 0.8970, 0.8938]` | `10.2965` vs `9.5318` | `0.7646` |
| **`track02_style`** | `200.44` / `3177.98` | `[1.0434, 1.1127, 1.1116]` | `[1.0434, 1.1127, 1.1116]` | `14.0153` vs `16.3930` | `2.3776` |
| **`track03_style`** | `197.48` / `3335.39` | `[1.0422, 1.1104, 1.1101]` | `[1.0422, 1.1104, 1.1101]` | `6.7986` vs `7.8634` | `1.0649` |

**Evidenze Chiave:**
1. In **5 sessioni su 6**, `imager_width_low` supera l'unità (`1.023..1.043`), allargando sistematicamente la banda sub/bassi (`< 200 Hz`) anziché mantenerla centrata o neutra, e `mid`/`high` convergono verso `~1.11`. Solo `track01_style` (dove il reference è più stretto del target) restringe tutte e tre le bande (`0.9455 / 0.8970 / 0.8938`).
2. `V3b-B0.5` non altera in modo apprezzabile lo spazio stereo rispetto a `V0p` (variazioni $\le 0.0016$ solo su `track01`), confermando l'ortogonalità tra `V3b` e la regolarizzazione spaziale demandata a `V3c`.
3. Il dataset corrente non misura né la correlazione $\rho_b$ né l'attenuazione da mono foldback $\Delta_{\text{mono}}$; tali misure saranno il cuore del **Preflight Read-Only V3c** prima di qualsiasi implementazione in training.

---

## 2. AUDIT MATEMATICO DELLO STEREO IMAGER E INTERAZIONE CON LA CATENA NON LINEARE

### 2.1 Trasformazione Ortonormale M/S per Banda
Per ogni banda $b \in \{\text{low}, \text{mid}, \text{high}\}$ prodotta dal crossover LR4 (`DifferentiableLR4Crossover`) e processata dal compressore multibanda (`y_comp,b`):
$$M_b[n] = \frac{L_b[n] + R_b[n]}{\sqrt{2}}, \qquad S_b[n] = \frac{L_b[n] - R_b[n]}{\sqrt{2}}$$
$$\hat{S}_b[n] = w_b \, S_b[n], \qquad \hat{L}_b[n] = \frac{M_b[n] + \hat{S}_b[n]}{\sqrt{2}}, \qquad \hat{R}_b[n] = \frac{M_b[n] - \hat{S}_b[n]}{\sqrt{2}}$$
L'uscita dell'imager è la somma $\hat{x}_{\text{img}} = \sum_{b} \hat{x}_b$.

### 2.2 Invarianti Esatte e Sorgenti Reali di Incompatibilità Mono
Dalla struttura ortonormale discendono tre identità algebriche esatte per l'imager isolato:
1. **Conservazione dell'Energia Ortonormale ($w_b = 1$):**
   $$M_b^2 + \hat{S}_b^2 = \hat{L}_b^2 + \hat{R}_b^2 \quad \text{(esatta campione per campione)}$$
2. **Invarianza della Somma Mono Pura a Monte delle Non-Linearità:**
   $$\frac{\hat{L}_b[n] + \hat{R}_b[n]}{\sqrt{2}} = M_b[n] \quad \forall w_b \ge 0$$
   Ciò significa che l'imager lineare **non cambia** il segnale $M[n]$ di per sé, ma cambia **drasticamente** la quota di energia totale persa quando il segnale stereo viene sommato in mono:
   $$\frac{E_M}{E_{\text{tot}}} = \frac{\sum_n M[n]^2}{\sum_n (M[n]^2 + \hat{S}[n]^2)} = \frac{E_M}{E_M + w^2 E_S}$$
3. **Dipendenza Esatta della Correlazione $\rho_b$ da $w_b$ (a $\langle M_b, S_b\rangle \approx 0$):**
   $$\langle \hat{L}_b, \hat{R}_b \rangle = \frac{1}{2}\left(E_{M,b} - w_b^2 E_{S,b}\right), \qquad \rho_b = \frac{E_{M,b} - w_b^2 E_{S,b}}{\sqrt{(E_{M,b} + w_b^2 E_{S,b})^2 - 4 w_b^2 \langle M_b, S_b\rangle^2}}$$
   Pertanto:
   - Se $w_b^2 E_{S,b} > E_{M,b}$, la correlazione **diventa negativa** ($\rho_b < 0$), causando cancellazione di fase all'ascolto mono o su diffusori acustici ravvicinati;
   - A valle dell'imager operano **due stadi non lineari per-canale** (`DifferentiableHarmonicSaturator` e `DifferentiableLookaheadLimiter`): quando $w_b$ amplifica $S_b$ su $L$ e $R$ in controfase, la saturazione `tanh` su $L$ e $R$ genera distorsione intermodulata $M\text{-}S$ e il limiter reagisce ai picchi stereo spuri di $|L|, |R|$; questo **altera fisicamente** anche il contenuto spettrale e dinamico di $\hat{M}_{\text{post}}$ rispetto a $M_{\text{pre}}$!

---

## 3. FORMULAZIONE MATEMATICA CANDIDATA PER `V3c`

In coerenza con `V3bDynamicPriorLoss`, ogni termine `V3c` deve essere:
1. Calcolato sulle finestre **post pre-roll allineate in latenza** ($\hat{y}_{\text{eval}} = \hat{y}[\dots, N_{\text{pre}}+D_{\text{tot}} : N_{\text{pre}}+T]$ appaiata con $x_{\text{mix}}^{\text{eval}} = x_{\text{mix}}[\dots, N_{\text{pre}} : N_{\text{pre}}+T_{\text{eval}}]$, $T_{\text{eval}} = 143\,841$ a $48\text{ kHz}$);
2. Mascherato su **soli campioni attivi** tramite `compute_active_sample_mask` ($B = 480$ campioni = $10\text{ ms}$, soglia $-70\text{ dBFS}$, fallback $M_{\min} = 8$ blocchi), evitando divisioni su code silenziose;
3. Aggregato su $K = 4$ chunk con pesi uniformi fissi $w_k = 1/4$ (senza rinormalizzazione dinamica).

### 3.1 Banco d'Analisi Fisso (Indipendente dai Parametri Trainable)
Per valutare correlazione e energia mono per banda senza creare incentivi spuri sullo spostamento delle frequenze di crossover (Sezione 4), `V3c` proietta l'audio attivo su $B = 3$ bande di analisi tramite un **banco LR4 fisso non-trainable** (con frequenze nominali ancorate ai neutri $f_{\text{low,fix}} = 160\text{ Hz}$, $f_{\text{high,fix}} = 3200\text{ Hz}$, o $200 / 2500\text{ Hz}$ da congelare in fase di review):
$$\hat{y}_b = \mathcal{H}_b^{\text{fix}}(\hat{y}_{\text{eval}}), \qquad x_{\text{mix}, b} = \mathcal{H}_b^{\text{fix}}(x_{\text{mix}}^{\text{eval}}), \qquad b \in \{\text{low}, \text{mid}, \text{high}\}$$

### 3.2 Termine 1 — Correlazione Inter-Canale per Banda con Dead-Zone e Tail Temporale ($\mathcal{L}_{\text{corr}}$)
Definita su campioni attivi $n \in A_{\text{smp}}$ la correlazione normalizzata per banda ($\epsilon = 10^{-12}$):
$$\rho_b(\hat{y}) = \frac{\sum_{n \in A_{\text{smp}}} \hat{L}_b[n]\hat{R}_b[n]}{\sqrt{\left(\sum_{n \in A_{\text{smp}}} \hat{L}_b[n]^2 + \epsilon\right)\left(\sum_{n \in A_{\text{smp}}} \hat{R}_b[n]^2 + \epsilon\right)}} \in [-1, +1]$$
Per evitare di comprimere brani intrinsecamente larghi più del mix sorgente (modalità mix-aware), la soglia target effettiva può essere definita come:
$$\tau_{\rho, b}^{\text{eff}} = \min\!\left(\tau_{\rho, b}^{\text{abs}},\; \text{detach}(\rho_b(x_{\text{mix}})) - \delta_{\rho, b}\right)$$
(oppure puramente assoluta $\tau_{\rho, b}^{\text{abs}}$ per la banda sub-bassi, es. $\tau_{\rho,\text{low}} \in [0.70, 0.85]$). La penalità quadratica con dead-zone è:
$$\mathcal{L}_{\text{corr}}^{\text{glob}} = \sum_{b \in \{\text{low}, \text{mid}, \text{high}\}} \alpha_b \, \text{ReLU}\!\left(\tau_{\rho, b}^{\text{eff}} - \rho_b(\hat{y})\right)^2$$
*(Opzionale Q4)* Affiancabile a un termine locale su finestre attive brevi (blocchi da $100\text{ ms} = 4800$ campioni) con $\text{CVaR}_{0.95}$ su $\text{ReLU}(\tau_{\rho,\text{loc}} - \rho_{b,m})$ per intercettare transienti in controfase localizzati che spariscono nella media di chunk da 3s.

### 3.3 Termine 2 — Protezione da Mono-Foldback Drop ($\mathcal{L}_{\text{mono\_drop}}$)
Usando la decomposizione ortonormale su campioni attivi:
$$E_M = \sum_{n \in A_{\text{smp}}} \hat{M}[n]^2, \qquad E_S = \sum_{n \in A_{\text{smp}}} \hat{S}[n]^2$$
L'attenuazione energetica nel passaggio da somma stereo totale a somma mono (in dB $\ge 0$) è:
$$\Delta_{\text{mono}}(\hat{y}) = 10 \log_{10}\!\left(\frac{E_M + E_S + \epsilon}{E_M + \epsilon}\right) \quad [\text{dB}]$$
Esempi fisici esatti:
- Segnale mono ($L = R$, $E_S = 0$): $\Delta_{\text{mono}} = 0.00\text{ dB}$;
- Segnale decorrelato ($\rho = 0$, $E_S = E_M$): $\Delta_{\text{mono}} = 10\log_{10}(2) \approx 3.01\text{ dB}$;
- Eccesso di controfase ($E_S = 3 E_M$, $\rho = -0.5$): $\Delta_{\text{mono}} = 6.02\text{ dB}$.

Sia $\tau_{\text{mono}}^{\text{eff}} = \max\!\left(\tau_{\text{mono}}^{\text{abs}},\; \text{detach}(\Delta_{\text{mono}}(x_{\text{mix}})) + \delta_{\text{mono}}\right)$ (con $\tau_{\text{mono}}^{\text{abs}} \approx 3.5\text{ dB}$ e margine $\delta_{\text{mono}} \approx 0.5\text{ dB}$):
$$\mathcal{L}_{\text{mono\_drop}} = \text{ReLU}\!\left(\Delta_{\text{mono}}(\hat{y}) - \tau_{\text{mono}}^{\text{eff}}\right)^2 \quad [\text{dB}^2]$$
Questo termine penalizza qualsiasi perdita di livello in somma mono oltre il pavimento di sicurezza e oltre quanto già presente nel mix di partenza.

### 3.4 Termine 3 — Vincolo di Centratura Low-Band & Guardrail su `imager_width` ($\mathcal{L}_{\text{low\_mono}}$)
Poiché i dati `V0p`/`V3b` mostrano che `imager_width_low` tende sistematicamente a valori $> 1.0$ (`1.023..1.043` in 5 sessioni su 6), si propone un termine parametrico diretto con dead-zone asimmetrica sulla sola espansione eccessiva in bassa frequenza:
$$\mathcal{L}_{\text{low\_mono}} = \text{ReLU}\!\left(w_{\text{low}} - \tau_{w,\text{low}}\right)^2, \qquad \tau_{w,\text{low}} = 1.00$$
che lascia completamente libera la scelta di **stringere** i bassi ($w_{\text{low}} \in [0.4, 1.0]$) verso una maggiore mono-compatibilità, ma scoraggia ogni allargamento artificiale dei sub/bassi sopra `1.0`. Analogamente, in `u`-space, ricordiamo che $w_{\text{low}} = 0.4 + 0.7 \sigma(u_{35})$, per cui l'uso della penalità quadratica produce gradiente nullo ovunque per $w_{\text{low}} \le 1.0$ e regolare crescita quadratica sopra soglia.

### 3.5 Termine 4 — Guardrail Bilanciamento L/R ($\mathcal{L}_{\text{bal}}$)
Per evitare asimmetrie tonali/dinamiche destra-sinistra indotte dall'EQ/compressore/saturatore su materiale asimmetrico:
$$\Delta_{LR}(\hat{y}) = 10\log_{10}\!\left(\frac{\sum_{A_{\text{smp}}}\hat{L}^2 + \epsilon}{\sum_{A_{\text{smp}}}\hat{R}^2 + \epsilon}\right), \qquad \mathcal{L}_{\text{bal}} = \text{ReLU}\!\left(\left|\Delta_{LR}(\hat{y}) - \text{detach}(\Delta_{LR}(x_{\text{mix}}))\right| - \delta_{\text{bal}}\right)^2$$
(es. $\delta_{\text{bal}} = 0.5\text{ dB}$).

### 3.6 Funzione Obiettivo Aggregata `V3c`
$$\mathcal{L}_{\text{V3c}} = w_{\text{corr}} \mathcal{L}_{\text{corr}} + w_{\text{mono}} \mathcal{L}_{\text{mono\_drop}} + w_{\text{low\_mono}} \mathcal{L}_{\text{low\_mono}} + w_{\text{bal}} \mathcal{L}_{\text{bal}}$$

---

## 4. ALLINEAMENTO TEMPORALE, PREVENZIONE DEI "PERVERSE INCENTIVES" E ANALISI DEI GRADIENTI

1. **Allineamento Esatto ($D_{\text{tot}} = 159$ campioni):** Il slicing di `y_hat` post-preroll $[N_{\text{pre}}+159 : N_{\text{pre}}+T]$ garantisce corrispondenza campione-per-campione con `x_mix_eval` $[N_{\text{pre}} : N_{\text{pre}}+T_{\text{eval}}]$ e con `mask_smp` calcolata su `x_mix_eval`. Ogni operatore di fase/correlazione tra $x_{\text{mix}}$ e $\hat{y}$ risulta così privo di bias da shift temporale.
2. **Prevenzione dei "Perverse Incentives" (Lezione Chiave da `V3b`):**
   - *Bypass tramite shift dei crossover:* Usare un banco di analisi **fisso** (`f_low_fix`, `f_high_fix`) anziché `(crossover_f_low, crossover_f_high)` impedisce all'ottimizzatore di spostare le frequenze di taglio per nascondere bande decorrelate.
   - *Bypass tramite abbassamento globale di volume:* Sia $\rho_b$ sia $\Delta_{\text{mono}}$ sono rapporti **scale-invariant** (omogenei di grado zero rispetto a scalature uniformi di ampiezza $c \cdot \hat{y}$), dunque $\mathcal{L}_{\text{corr}}$ e $\mathcal{L}_{\text{mono\_drop}}$ non possono essere elusi abbassando il guadagno di uscita o spingendo sul makeup/limiter!
   - *Bypass tramite appiattimento eccessivo (`mono collapse`):* Le dead-zone ancorate a `tau_rho_eff` e `tau_mono_eff` assicurano gradiente esattamente nullo nella regione stereo sana, evitando qualsiasi pressione a trasformare un mix stereo naturale in mono.
   - *Asimmetria dello Jacobiano in `u`-space:* Come visto in Sezione 1.2, $dw_{\text{low}}/du_{35} = 0.0857$ è metà di $dw_{\text{mid,high}}/du_{36,37} = 0.1714$. In sede di futuro Preflight V3c andranno quindi ispezionate separatamente $\|\nabla_z \mathcal{L}_{\text{V3c}}\|_2$ e $\|\nabla_u \mathcal{L}_{\text{V3c}}\|_2$ per verificare che $w_{\text{low\_mono}}$ e $w_{\text{corr}}$ non richiedano pesi sproporzionati per muovere $u_{35}$.

---

## 5. INTEGRAZIONE ARCHITETTURALE E INTERAZIONE CON `V0p`, `V3a`, `V3b` E `V4`

1. **Isolamento Modulare (Nessuna Regressione su `V0p`/`V3b`):**
   - Modulo dedicato proposto: `python/ddsp_mastering/loss/stereo_prior.py` (`V3cStereoCompatibilityLoss`), agganciato a `AutoMasteringOptimizer` con pesi di default pari a `0.0` (`w_corr=0.0, w_mono=0.0, w_low_mono=0.0, w_bal=0.0`), garantendo **bit-identità al 100%** quando disattivato.
2. **Interazione con `stylistic_stereo_loss` (`AggregatePerceptualLoss`, peso `0.3`):**
   - In `stylistic` mode, `stylistic_stereo_loss` confronta il rapporto Side/Mid globale verso una traccia di riferimento esterna (es. su `track02_style`, `target_ms_ratio_db = 16.3930 dB` contro `pred = 14.0153 dB`). Le soglie dead-zone di `V3c` fungono esclusivamente da **guardrail di compatibilità fisica inferiore**: se il reference ha un Side/Mid più ampio ma comunque compatibile ($\rho_b \ge \tau_{\rho,b}$ e $\Delta_{\text{mono}} \le \tau_{\text{mono}}$), `V3c` resta inattivo (`0.0000`) e lascia governare `stylistic_stereo_loss`; solo in presenza di rischi reali di cancellazione mono o espansione dei bassi interviene in protezione.
3. **Interazione con `V3bDynamicPriorLoss`:**
   - Limitare la divergenza di fase estrema in basso ($w_{\text{low}}$) e la crescita incontrollata del canale Side riduce anche i picchi L/R asimmetrici all'ingresso del saturatore e del limiter, cooperando sinergicamente con $\mathcal{L}_{\text{lim}}$.
4. **Composizione Finale `V4`:**
   $$\mathcal{L}_{V4} = \mathcal{L}_{V0p} + \mathcal{L}_{V1/\text{ERB}} + \mathcal{L}_{V2/\text{trans}} + \mathcal{L}_{V3a/\text{eq}} + \mathcal{L}_{V3b/\text{dyn}} + \mathcal{L}_{V3c/\text{stereo}}$$
   mantenendo il controllo rigoroso del ceiling **True Peak ($\le -0.30\text{ dBTP}$)** in fase di rendering/delivery a valle (come già documentato nella roadmap Sprint 3).

---

## 6. PIANO DI TECHNICAL GATE TEST PRE-IMPLEMENTAZIONE (DA IMPLEMENTARE IN FASE SUCCESSIVA)

Prima di avviare qualsiasi run di preflight o ablation `V3c`, la suite `tests/test_sprint3_v3c_gate.py` dovrà verificare almeno i seguenti 7 Gate obbligatori:
1. **Gate 1 — Conservazione Ortonormale & Invarianza Mono dell'Imager:** Verifica algebrica $M^2 + \hat{S}^2 = \hat{L}^2 + \hat{R}^2$ e invarianza di $\hat{M}$ rispetto a $w \in [0.0, 2.0]$ su segnali reali e sintetici.
2. **Gate 2 — Monotonicità di $\rho_b$ e $\Delta_{\text{mono}}$:** Test su segnali controllati in fase ($\rho = +1$, $\Delta_{\text{mono}} = 0\text{ dB}$), quadratura $90^\circ$ ($\rho \approx 0$, $\Delta_{\text{mono}} \approx 3.01\text{ dB}$) e controfase ($\rho = -1$, $\Delta_{\text{mono}} \gg 10\text{ dB}$).
3. **Gate 3 — Mascheramento Attivo e Silence Rejection:** Verifica su chunk silenziosi ($< -70\text{ dBFS} \implies \mathcal{L}_{\text{V3c}} = 0.0$, zero `NaN`/`Inf`) e su durata reale $T_{\text{eval}} = 143\,841$ campioni con blocco residuo da $321$ campioni.
4. **Gate 4 — Stabilità Analitica dei Gradienti (`gradcheck` in `float64`):** Verifica rigorosa di $\partial \mathcal{L}_{\text{corr}}/\partial z$, $\partial \mathcal{L}_{\text{mono\_drop}}/\partial z$, $\partial \mathcal{L}_{\text{low\_mono}}/\partial z$ e $\partial \mathcal{L}_{\text{bal}}/\partial z$.
5. **Gate 5 — Dead-Zone Esatta & Scale-Invariance:** Zero perdita e gradiente identicamente nullo quando tutti gli indicatori rispettano le soglie; invarianza rispetto a scalature globali $\alpha \cdot \hat{y}$.
6. **Gate 6 — Isomorfismo C++/Python (`core_dsp::MidSideImager` vs `DifferentiableMidSideImager`):** Verifica bit-accurate con residuo RMS $< -120\text{ dBFS}$.
7. **Gate 7 — Bit-Identità di `V0p` e `V3b` quando `V3c` è Disabilitato:** Uguaglianza esatta dei pesi/audio/metriche con i baseline congelati quando tutti i pesi `V3c = 0.0`.

---

## 7. PIANO PREFLIGHT READ-ONLY & QUESTIONARIO APERTO PER ARGON (`Q1–Q5`)

### 7.1 Protocollo Preflight Read-Only (Prossimo Step)
Prima di integrare `V3c` nell'optimizer, si propone uno script read-only (`scripts/sprint3/run_v3c_preflight.py`) che calcoli sulle 6 sessioni armonizzate, ai punti congelati $z = Z_{\text{NEUTRAL}}$ e $z = z_{\text{opt}}^{V0p} / z_{\text{opt}}^{V3b}$:
- I profili baseline di $\rho_{\text{low}}, \rho_{\text{mid}}, \rho_{\text{high}}$, $\Delta_{\text{mono}}$ (dB), $\Delta_{LR}$ (dB) del mix originale vs output ottimizzato;
- I valori delle loss candidate $\mathcal{L}_{\text{corr}}, \mathcal{L}_{\text{mono\_drop}}, \mathcal{L}_{\text{low\_mono}}, \mathcal{L}_{\text{bal}}$ e le norme dei gradienti separati in $z$ e $u$ ($\|\nabla_u \mathcal{L}_{\text{V3c}}\|_2$, rapporti su EQ / Comp / Imager), così da calibrare i pesi senza ottimizzazioni alla cieca.

### 7.2 Quesiti Decisionali per il Supervisore Tecnico (Argon)
Si sottopongono 5 decisioni progettuali per approvazione prima dell'implementazione di `stereo_prior.py`:
- **Q1 (Bande d'Analisi):** Approvi l'uso di un banco fisso LR4 non-trainable (`160 Hz / 3200 Hz`, allineato a `THETA_NEUTRAL[18..19]`) invece dei crossover trainable per prevenire bypass dell'ottimizzatore, oppure preferisci una decomposizione FFT a bande critiche?
- **Q2 (Soglie $\tau_{\rho,b}$ Mix-Aware vs Assolute):** In `stylistic` mode, ritieni opportuno usare soglie mix-aware ($\min(\tau_{\rho,b}^{\text{abs}}, \rho_b^{\text{mix}} - \delta)$) su `mid`/`high` mantenendo invece una soglia più rigida su `low` ($\tau_{\rho,\text{low}}^{\text{abs}} \approx 0.80$, più $\mathcal{L}_{\text{low\_mono}}$ con $\tau_{w,\text{low}} = 1.00$)?
- **Q3 (Metrica di Compatibilità Mono):** Oltre al drop di energia $\Delta_{\text{mono}} = 10\log_{10}((E_M + E_S)/E_M)$, è desiderabile includere già in `V3c` un termine spettrale anti-comb-filtering ($M(f)$ vs $\sqrt{(|L(f)|^2+|R(f)|^2)/2}$), oppure mantenerlo minimale e ortogonale nella sola energia/correlazione per banda rimandando il comb-filtering al preflight?
- **Q4 (Granularità Temporale di $\rho_b$):** È preferibile valutare $\rho_b$ sull'intero insieme attivo del chunk ($3\text{ s}$) oppure aggiungere anche un termine tail $\text{CVaR}_{0.95}$ su blocchi attivi da $100\text{ ms}$ per catturare transienti stereofonici in controfase localizzati?
- **Q5 (Interazione con i Bounds Stretti di `param_space.py`):** Dato che `imager_width_low` è già limitato a `[0.4, 1.1]` con Jacobiano ridotto (`0.0857` a $z=0.857$), confermi di **non** alterare i bounds di `param_space.py` in `V3c` per preservare la piena comparabilità con gli esperimenti `V0p..V3b`?
