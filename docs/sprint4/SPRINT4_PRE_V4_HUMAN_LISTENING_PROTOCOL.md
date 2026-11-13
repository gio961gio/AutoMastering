# SPRINT 4 — PROTOCOLLO DI VALIDAZIONE UMANA PRE-V4 (SOLO SPECIFICA DI PROTOCOLLO)

- **Repository:** `gio961gio/AutoMastering`
- **Branch:** `human-validation-package`
- **Baseline DSP:** Commit 1 Sprint 4 (`db4a7bf` — True Peak Lookahead Limiter 4× `-0.50 dBTP` + Guardrail Offline `PCM_24 <= -0.30 dBTP` sopra `3fdeb68`)
- **Matrice Dati Accoppiata:** [`SPRINT4_V0P_V3C_COMPARATIVE_MATRIX.json`](./SPRINT4_V0P_V3C_COMPARATIVE_MATRIX.json) e [`SPRINT4_V0P_V3C_COMPARATIVE_MATRIX.md`](./SPRINT4_V0P_V3C_COMPARATIVE_MATRIX.md)
- **Stato Operativo:** **SOLO DOCUMENTO DI PROTOCOLLO.** Nessuna nuova ottimizzazione Adam eseguita, nessun pacchetto blind audio V4 generato in questo commit, nessuna modifica a `V3cStereoCompatibilityLoss`, `enable_v3c=False`, `PARAM_SPECS` (41 parametri) o risultati storici Sprint 3 (`V3c same_track` 3/3 Ties preservati integralmente).

---

## §9. Inventario Completo in Sola Lettura: Varianti `V0`..`V3c`

Tutti i dati di questa sezione sono estratti in sola lettura dai manifest e JSON congelati presenti nel repository (`docs/human_validation/SPRINT2_V0_REPRODUCIBILITY_MANIFEST.json` e `docs/sprint3/*.json`).

### 9.1 Registro Architetturale, Iperparametri e Stato Checkpoint (`V0`..`V3c`)

| Variante | Commit Git (verificato) | Intervento rispetto a `V0p` (Control Arm) | Optimizer (`lr` / `steps` / `pre-roll`) | Sessioni & Checkpoint `z`/`theta` congelati | Stato Validazione Umana |
|:---|:---|:---|:---|:---|:---|
| **`V0`** | `ffafb9848bea` (`SPRINT2_V0_REPRODUCIBILITY_MANIFEST.json`) | Baseline storica Sprint 2 pre-armonizzazione (`target_lufs` locale di chunk) | `lr=0.04`, `30` step (`1..20 fft -> 21..30 iir`), `0.0s` pre-roll | `6/6` sessioni (`track01..03` + `_style`) in manifest JSON | Eseguita vs bypass (Sprint 2 CSV) |
| **`V0p`** | `f53f58f` / `ea3f4ea` | **Control Baseline armonizzato:** pre-roll `2.5s` (`120k` smp), full-track target LUFS, `w_stft=1.0`, `w_lufs=0.8`, `w_reg=0.01`, `plr_mode='box'` | `lr=0.04`, `30` step (`1..20 fft -> 21..30 iir`), `2.5s` pre-roll | `6/6` sessioni congelate in `SPRINT3_V1_HARMONIZED_RESULTS.json` e `SPRINT3_V3B_V0P_ABLATION_RESULTS.json` | Usata come braccio `Control_V0p` nel pilot V3c `same_track` (`3/3` Ties) |
| **`V1`** | `4bbf097` / `ea3f4ea` | `V0p` + perdita spettrale multi-risoluzione ERB-48 (`weight_erb=0.25` in `same_track` con `mel40`, `weight_erb=1.0` in `stylistic` con `erb48`, `weight_de_harsh=0.0`) | Identico a `V0p` (`0.04` / `30` / `2.5s`) | `6/6` sessioni congelate in `SPRINT3_V1_HARMONIZED_RESULTS.json` | **Mancante** |
| **`V1b`** | `4bbf097` / `ea3f4ea` | `V1` + penalità asimmetrica anti-asprezza ERB (`weight_de_harsh=0.15` su banda alta relativa) | Identico a `V0p` (`0.04` / `30` / `2.5s`) | `6/6` sessioni congelate in `SPRINT3_V1_HARMONIZED_RESULTS.json` | **Mancante** |
| **`V2`** | `d22405d` / `427be98` / `ea3f4ea` | `V0p` + `TransientEnvelopeLoss` (`weight_trans=0.2`, `beta_atk=0.5`) + `ReferenceAwarePLRLoss` (`plr_mode='reference_aware'`, Boltzmann soft-peak `p=8, beta=2.0`, `10 ms`, `plr_d_style=1.0`, `plr_gamma_floor=1.0`) | Identico a `V0p` (`0.04` / `30` / `2.5s`) | `6/6` sessioni congelate in `SPRINT3_V2_ABLATION_RESULTS.json` | **Mancante** |
| **`V3a`** | `eff946e` / `533549c` | `V0p` + `CompositeEQPriorLoss` (`weight_eq_prior=0.02`, `K_eq=128`, `20–20k Hz float64`, `tau_shape=1.5`, `tau_offset=1.0`, `w_slope=2.0`, `w_offset=1.0`) | Identico a `V0p` (`0.04` / `30` / `2.5s`) | `6/6` sessioni congelate in `SPRINT3_V3A_ABLATION_RESULTS.json` | **Mancante** |
| **`V3b_B05`** | `c2fefc2` / `2123842` / `9cc448f` | `V0p` + `V3bDynamicPriorLoss` (`tau_avg=0.5 dB`, `tau_95=2.0 dB`, `lambda_95=0.5`, `delta_max=3.0 dB`, `delta_m=1.25 dB`, `delta_ref=3.0 dB`, `w_lim=0.5`, `w_diff=0.02`, `w_conflict=0.02`) | Identico a `V0p` (`0.04` / `30` / `2.5s`) | `6/6` sessioni congelate in `SPRINT3_V3B_V0P_ABLATION_RESULTS.json` | Blind pack generato in Sprint 3, ascolto **non loggato** |
| **`V3c`** *(Pilot)* | `a98c2d6` / `5709c3f` / `53bd123` / `3fdeb68` | `V0p` + `V3cStereoCompatibilityLoss` (`enable_v3c=True` solo opt-in esplicito, `tau_drop_db=0.5`, `w_mono_drop=1.0`, `w_band_corr=10.0`, `w_band_smr=0.05`, `w_width_param=0.25`, `160/3200 Hz LR4`) | Identico a `V0p` (`0.04` / `30` / `2.5s`); **solo `same_track`** (stylistic escluso per conflitto loss) | `3/3` sessioni `same_track` congelate in `SPRINT3_V3C_ABLATION_PILOT_RESULTS.json` (`stylistic` escluso) | **Eseguito su 3 coppie V2 (`3/3` Ties/`Pari`)** |

### 9.2 Separabilità `V1b` vs `V1` e Divergenza dallo Stato `V0p`

Dalla matrice estratta in [`SPRINT4_V0P_V3C_COMPARATIVE_MATRIX.md`](./SPRINT4_V0P_V3C_COMPARATIVE_MATRIX.md):

1. **Verdetto su `V1b` vs `V1` (Separabilità Dipendente dal Contenuto & Trade-Off Loudness):**
   - **`stylistic` con spiccata harshness (`track01_style`):** **Fortemente separabile** (`max|dz| = 0.31878`, `||dz||_2 = 0.61305`), guidata da rimodellamento medio/medio-alto (`eq_b2_gain_db` `dz=+0.3188`, `eq_b2_f0` `dz=-0.2787`, `eq_b3_f0` `dz=+0.2665`), abbattimento dell'ERB Presence Surplus di **`-2.645 dB`** (`+8.52 -> +5.87 dB`), ma **con overshoot di loudness pari a `+0.964 LU` rispetto a `V1`** (`-17.83 -> -16.86 LUFS`, portando lo scarto assoluto rispetto al target `-17.99 LUFS` da `0.159 LU` a **`+1.123 LU` più alto**, perché l'`eq_mean_gain_db` sale da `-1.915` a `-1.263 dB` riducendo `pred_plr_db` da `15.37` a `14.53 dB` durante i 30 step Adam per soddisfare il rapporto di banda anti-harshness).
   - **`same_track` (`track01`, `track02`):** **Separabilità moderata/piccola** (`max|dz| = 0.04341` su `track01`, `0.02476` su `track02`), effetto sottile sull'ERB Surplus (`-0.223 dB` su `track01`, `-0.021 dB` su `track02`).
   - **`track03`, `track02_style`, `track03_style`:** **Praticamente indistinguibili** (`max|dz| = 0.00078 .. 0.00468`, `< 0.005`), poiché i mix/riferimenti non presentano surplus nella banda di de-harshness.
   - *Implicazione sperimentale:* Confrontare `V1b` vs `V1` su brani privi di eccesso di presenza produce quasi certamente `Tie` nulli per mancanza di attivazione; il confronto **`V1` vs `V1b`** va quindi indirizzato specificamente su tracce con asprezza/bright reale (`track01_style` + nuove tracce bright/harsh nello screening).

2. **Profilo di Divergenza delle Altre Varianti vs `V0p`:**
   - **`V2`:** movimento parametrico netto su `track03` (`||dz||_2 = 0.22072`), `track03_style` (`0.20934`), `track01` (`0.14043`), `track01_style` (`0.13885`).
   - **`V3a`:** effetto moderato (`||dz||_2 = 0.0010 .. 0.0309`), poiché tutte le soluzioni `V0p` hanno già curve morbide e un offset medio EQ attorno a `±2.5 dB`.
   - **`V3b_B05`:** **identico al bit (`||dz||_2 = 0.00000`) a `V0p` su 5 sessioni su 6!** Si attiva esclusivamente su `track01` (`||dz||_2 = 0.01672`), l'unica traccia i cui chunk superano la dead-zone di limiter stress (`tau_avg=0.5 dB`, `tau_95=2.0 dB`).
   - **`V3c` (pilot storico conservato):** differenze minuscole su `track01..03` (`||dz||_2 = 0.00208 .. 0.01458`, max null `A-B` `-27.91 dBFS` su `track01`, `-49.48 dBFS` su `track03`, `-77.04 dBFS` su `track02`), coerenti al 100% con il risultato storico **`3/3` Ties (`Pari`)** in [`SPRINT3_V3C_HUMAN_VALIDATION_PILOT_RESULTS.json`](../sprint3/SPRINT3_V3C_HUMAN_VALIDATION_PILOT_RESULTS.json). **Tale esito rimane conservato come dato storico consolidato.**

---

## §10. Disegno Incrementale di Comparazione (Coppie Ragionate `H01..H06`)

Per massimizzare il potere diagnostico evitando ridondanze su coppie matematicamente degenerate (`||dz||_2 == 0`), lo screening pre-V4 struttura sei contrasti principali (`Treatment` vs `Control`, cinque atomici rispetto a `V0p`/`V1` e il contrasto `H03` sul pacchetto composito V2):

| ID | Coppia / Contrasto | Ipotesi percettiva testata | Regime / Materiale idoneo | Note di disaccoppiamento |
|:---|:---|:---|:---|:---|
| **`H01`** | **`V0p` vs `V1`** | La metrica spettrale ERB-48 migliora il bilanciamento timbrico rispetto a MR-STFT/Mel-40? | `same_track` & `stylistic` | Isola la sola loss spettrale ERB-48 (`de_harsh=0`) |
| **`H02`** | **`V1` vs `V1b`** | La penalità asimmetrica `weight_de_harsh=0.15` riduce l'affaticamento/asprezza senza rendere il suono spento/innaturale? | **Solo brani bright/harsh** (`track01_style`, `track06`) | Richiede static loudness match rigoroso (`<= 0.01 LU`) per isolare il timbro dall'**overshoot di `+0.964 LU` (`+1.123 LU` sul target)** generato da `V1b` |
| **`H03`** | **`V0p` vs `V2`** *(pacchetto composito V2)* | Il **pacchetto composito `V2`** (`TransientEnvelopeLoss` + `ReferenceAwarePLRLoss` congiunti) preserva meglio cassa/rullante e dinamica naturale? | Brani **percussivi/dinamici** e `stylistic` a PLR diverso | **Contrasto non atomico:** testa congiuntamente l'intero pacchetto `V2` (massima separazione `||dz||_2` fino a `0.22`); un'eventuale dissezione tra solo `TransientEnvelopeLoss` e solo `ReferenceAwarePLRLoss` è demandata a micro-ablazione secondaria se `H03` risultasse positivo o ambiguo |
| **`H04`** | **`V0p` vs `V3a`** | Il prior analitico sull'EQ composito evita boost globali artificiali migliorando naturalezza/trasparenza? | `stylistic` e brani con offset EQ ampio | Verifica percettiva se `||dz|| <= 0.031` sia udibile o sub-soglia |
| **`H05`** | **`V0p` vs `V3b_B05`** | La regolarizzazione del limiter stress riduce pumping/distorsione sui picchi? | **Solo brani ad alta pressione di limiter (`limiter GR > 0.5 dB`)** (`track01`, `track05`, `track07`) | *Critico:* escludere brani in deadzone `V3b == V0p` (`GR ~ 0`), altrimenti il test produce `Tie` banali per identità bit-esatta! |
| **`H06`** | **`V0p` vs `V3c`** *(esplorativa)* | *(Solo `same_track` stress stereo)* Su mix problematici larghi/out-of-phase, `V3c` migliora tenuta in mono? | **Nuova traccia `same_track` wide-stereo (`track08`, `< 2` brani finora)** | I 3 casi `track01..03` restano chiusi come pilot storico (`3/3` Ties). Finché copre `< 2` tracce indipendenti resta **strettamente esplorativa e non promuovibile**. Non applicare a `stylistic`. |

---

## §11. Selezione Tracce (`6–8` Brani) e Copertura Materiale

L'inventario attuale (`human_validation/private/input/`) contiene soltanto **`3` tracce reali** (`track01`: `63.90s`, mix `-14.22 LUFS`, PLR `18.12`, corr `0.795`; `track02`: `71.47s`, `-22.03 LUFS`, PLR `18.01`, corr `0.948`; `track03`: `71.78s`, `-17.86 LUFS`, PLR `14.45`, corr `0.715`).

Per la validazione pre-V4 si specifica un pannello di **`6–8` brani completi a `48 kHz`** (integrando `track01..03` con **`3–5` nuovi brani `track04..track08`** prima della futura generazione audio):

| Slot Traccia | Regime primario testato | Caratteristica musicale / stress mirato | Coppie assegnate (`H0x`) |
|:---|:---|:---|:---|
| `track01` *(esistente)* | `same_track` + `stylistic` (`_style`) | Pop/Voce con presenza alta marcata e limiter attivo (`push +8.6 dB`) | `H02` (`_style`), `H03`, `H05` |
| `track02` *(esistente)* | `same_track` + `stylistic` | Mix dinamico ad alto PLR (`18.0 dB`), centro solido (`corr 0.948`) | `H01`, `H03`, `NULL` |
| `track03` *(esistente)* | `same_track` + `stylistic` | Mix stereofonico più largo (`M/S 7.79 dB`, `corr 0.715`), PLR `14.45` | `H01`, `H03`, `H04` |
| **`track04` *(mancante — intake)*** | `same_track` | **Acustico / Dinamico / Voce-Chitarra-Pianoforte** (nessun clipping desiderato, PLR elevato `> 11 dB`) | `H01`, `H03`, `H04` |
| **`track05` *(mancante — intake)*** | `same_track` + `stylistic` | **Transienti rapidi & Batteria secca** (Drum & Bass / Funk / Rock moderno con kick/snare netti) | `H03` (`V2`), `H05` (`V3b`) |
| **`track06` *(mancante — intake)*** | `stylistic` | **Bright / Harshness / Sibilanza reale** (`3–7 kHz` aggressivi rispetto al reference equilibrato) | `H02` (`V1b`), `H04` (`V3a`), `NULL` |
| **`track07` *(mancante — intake)*** | `same_track` | **Sub-bass 808 / Densità spettrale alta & Limiter Stress** (loudness target `-7 .. -9 LUFS` per forzare `GR > 1 dB`) | `H05` (`V3b`) |
| **`track08` *(mancante — intake)*** | `same_track` | **Wide Stereo / Synth-Pad / Reverb larghi (`corr < 0.55`)** con rischio di cancellazione in mono | `H06` (`V3c`, esplorativa su `< 2` tracce) |

- **Finestra di ascolto per ogni trial:** clip da **`20–30 s`** centrata sulla sezione più discriminante del brano (drop/ritornello + transizione transiente), con loop libero A/B su sotto-frase critica di `8–12 s`.
- **Mix di regime:** almeno `50%` `same_track` (fedeltà/non-regressione rispetto al master obiettivo) e almeno `35–50%` `stylistic` (valore pratico reale su riferimenti esterni, dove `V1/V1b/V2/V3a` mostrano le differenze più grandi).

---

## §12. Specifiche Tecniche di Pre-Render per il Futuro Blind Pack

*(Specifica obbligatoria per il futuro runner di export — nessun file audio blind generato in questo Sprint)*:

1. **Catena DSP obbligatoria:** rendering C++ `render_offline_with_tail_flush(..., limiter_detector_mode="true_peak_4x")` con ceiling interno `-0.50 dBTP` e tail flush `159` campioni esatti (`48000 Hz`, `stereo`, `float64 -> PCM_24`). È vietato usare `legacy_sample_1x` per file destinati all'ascolto umano pre-V4.
2. **Validazione `export_validated_pcm24_wav` su disco:** ogni WAV generato deve superare la rilettura da disco `PCM_24` con:
   - `calculate_true_peak_dbtp(disk_audio, 48000) <= -0.30 dBTP`;
   - `calculate_sample_peak_dbfs(disk_audio) <= -0.30 dBFS`;
   - qualsiasi micro-trim d'emergenza (`<= 0.05 dB`) tracciato nel manifest JSON privato/pubblico; fallimento rigido se richiesto `> 0.05 dB`.
3. **Static Loudness Matching della Coppia (`<= 0.01 LU`):**
   - Dopo il render TP4× di `Variant_1` e `Variant_2`, calcolare l'Integrated Loudness BS.1770-4 su ciascun WAV `PCM_24`;
   - Applicare una compensazione di guadagno **lineare puramente statica** verso la meno forte delle due (attenuando la variante più forte di $|\Delta\text{LUFS}|$ in modo da **non aumentare mai il True Peak** della variante target), oppure comune se necessario per sicurezza, verificando dopo riscrittura su disco `PCM_24` che:
     - `|LUFS_A - LUFS_B| <= 0.01 LU` (target operativo `<= 0.002 LU`);
     - entrambi `TP_A <= -0.30 dBTP` e `TP_B <= -0.30 dBTP`;
     - identica durata al campione (`frames_A == frames_B`), zero click di inizio/fine (fade in/out `10 ms` cosine identico su A e B se estratti da sezione interna).

---

## §13. Blinding, Randomizzazione e Controllo Coerenza (`>= 20%` Ripetizioni Nascoste)

1. **Nomi completamente opachi:** `TRIAL01_A.wav`, `TRIAL01_B.wav`, ..., `TRIAL24_A.wav`, `TRIAL24_B.wav` in una directory evaluator dedicata priva di prefissi `track0x`, `Control`, `Treatment`, `V1`, `V2`, `V3a`, `V3b`, `V3c`.
2. **Assegnazione A/B randomizzata per trial:** moneta indipendente `CSPRNG` (`secrets.SystemRandom`) per ogni trial (`50%` Control=A / `50%` Treatment=A).
3. **Ripetizioni nascoste (`25.0% >= 20%`):**
   - Su **`24` trial principali di screening** (`18` trial attivi + `6` repliche nascoste = `24`), le **`6` repliche nascoste (`25.0%`)** comprendono:
     - **`4` repliche flipped** `(Control, Treatment) -> (Treatment, Control)` (inversione dell'assegnazione rispetto al trial sorgente sulla stessa coppia/sezione);
     - **`2` null anchor (`A == B`)** nascosti (coppia di file bit-identica sotto nomi opachi) per misurare il tasso di falsi positivi/allucinazione percettiva dell'ascoltatore.
   - **Regola esplicita di valutazione dopo l'unblinding (per VARIANTE, non per lettera A/B):** la coerenza intra-ascoltatore sulle repliche *flipped* viene calcolata **dopo lo sblinding confrontando la variante sottostante scelta** (`Control` vs `Treatment` vs `Tie`), **mai** la lettera superficiale `A`/`B`. Esempio: scegliere `A` sul trial base (`A=V2, B=V0p`) e `B` sul corrispondente *flipped* (`A=V0p, B=V2`) equivale a `100%` coerenza (entrambe le scelte indicano `V2`), mentre scegliere `A` in entrambi i casi (`V2` poi `V0p`) costituisce un'inversione inconsistente.
   - **Criterio di validazione affidabilità pre-decisione:** coerenza per variante sulle `4` repliche flipped `>= 75%` sui casi non-Tie (`>= 3/4` coerenti oppure `Tie` coerente) e zero preferenze ad alta confidenza (`confidence >= 2`) sui `2` null anchor `A == B`.
4. **Confinamento segreto:** la chiave di unblinding (`seed`, assegnazione `A/B`, id repliche nascoste e mappatura `V0p..V3c`) deve essere salvata **esclusivamente** in path locale gitignored `human_validation/private/sprint4_pre_v4_blind_mapping.json` e mai inclusa nello ZIP valutatore né committata su Git.

---

## §14. Workflow Operativo Ableton Live 12 Suite e Scheda Strutturata

### 14.1 Setup Ableton Live 12 Suite

1. **Sessione / Arrangement View:** `48000 Hz`, BPM irrilevante ma **Warp = OFF** su tutte le clip audio (verifica obbligatoria per evitare alterazioni di fase/resampling da warping).
2. **Allineamento & Routing:** per il trial `k`, caricare `TRIALk_A.wav` e `TRIALk_B.wav` su due tracce stereo separate allineate esattamente a `1.1.1`, entrambi fader a **`0.0 dB`**, Pan `C`, zero insert/FX sulle tracce.
3. **Switching A/B istantaneo:** attivare **Exclusive Solo** (`Solo` esclusivo in Preferences) e mappare due tasti rapidi (`1` / `2` in Key Map mode) per commutare in tempo reale senza gap di fase.
4. **Master Bus:** **solo `Utility` impostato a `Mono` (mappato su tasto `M`)** per verificare compatibilità mono e crolli tonali M/S; nessun limiter, EQ o compressore sul Master. Catena d'ascolto calibrata (cuffie/monitor di riferimento fissi per tutta la campagna).

### 14.2 Scheda Strutturata di Valutazione (per ciascun `TRIAL_k`)

L'ascoltatore compila una scheda JSON/HTML offline (modello analogo all'interfaccia web locale usata per Sprint 3 V2) registrando per ogni `TRIAL_k`:

| Campo | Valori ammessi | Significato |
|:---|:---|:---|
| `global_choice` | `A` \| `Pari (Tie)` \| `B` | Preferenza complessiva di mastering |
| `confidence` | `1` (appena percettibile/incerto), `2` (moderata), `3` (netta/sicura) | Peso di confidenza percettiva |
| `tonal_harshness` | `A` \| `Pari` \| `B` | Equilibrio timbrico & assenza asprezza `2.5–6 kHz` |
| `punch_transients` | `A` \| `Pari` \| `B` | Impatto e integrità di cassa/rullante/attacchi |
| `bass_control` | `A` \| `Pari` \| `B` | Solidità e pulizia sub/low-end |
| `stereo_mono` | `A` \| `Pari` \| `B` | Ampiezza stereo naturale e stabilità al toggle Mono |
| `dynamics_artifacts` | `A` \| `Pari` \| `B` + flag `[pumping / harsh / squashed / none]` | Naturalezza dinamica ed eventuali artefatti di limiter/EQ |
| `notes` | Testo libero | Osservazioni sintetiche specifiche |

---

## §15. Budget di Esecuzione e Tabella Esplicita dei `24` Trial (`3 × 8` Screening + Conferma)

Per prevenire affaticamento uditivo (*ear fatigue*) e mantenere alta la discriminazione fine (`~0.3–0.5 dB`):

- **Fase 1 — Screening Esplorativo Strutturato su Singolo Ascoltatore Esperto (`24` trial totali):**
  - **`18` trial attivi** distribuiti su `H01..H06` lungo le tracce idonee + **`4` repliche flipped** + **`2` null anchor `A == B`** = **`24` trial totali**.
  - **Natura epistemologica:** Fase 1 è uno **screening esplorativo diagnostico su singolo ascoltatore esperto** (rilevazione di regressioni, segnali forti di qualità e verifica di deadzone/trade-off); **non** costituisce uno studio inferenziale multi-ascoltatore generalizzabile alla popolazione.
  - Suddivisione tassativa in **`3` sessioni da `8` trial ciascuna (`3 × 8 = 24`)**, dopo mescolamento CSPRNG dei nomi opachi `TRIAL01..TRIAL24` (qui sotto gli slot logici pre-randomizzazione `S01..S24`):
  - **Durata massima per sessione:** `<= 25 minuti` (`~2.5–3 min` per trial).
  - **Pausa obbligatoria:** minimo `4 ore` (preferibilmente giorni distinti o mattina/pomeriggio) tra una sessione e la successiva.

### 15.1 Matrice Esplicita di Allocazione dei `24` Trial (Pre-Randomizzazione CSPRNG)

| Slot Logico | Tipo Trial | Ipotesi | Contrasto | Track ID | Regime | Sezione Candidata (`20–30 s`) | Note Metodologiche / Regola Promozione |
|:---|:---|:---|:---|:---|:---|:---|:---|
| `S01` | **Attivo** (`1/3`) | `H01` | `V0p` vs `V1` | `track02` | `same_track` | Sez. principale dinamica | ERB-48 timbrica su PLR alto |
| `S02` | **Attivo** (`2/3`) | `H01` | `V0p` vs `V1` | `track03` | `stylistic` | Ritornello / dense wide | ERB-48 vs Mel-40 su riferimento esterno |
| `S03` | **Attivo** (`3/3`) | `H01` | `V0p` vs `V1` | `track04` *(nuova)* | `same_track` | Strofa/ritornello acustico | Voce/acustica (`>= 2` tracce distinte ok) |
| `S04` | **Flipped** (`S02`) | `H01` | `V0p` vs `V1` *(flip)* | `track03` | `stylistic` | Identica a `S02` (A/B invertiti) | Check coerenza per variante dopo unblinding |
| `S05` | **Attivo** (`1/2`) | `H02` | `V1` vs `V1b` | `track01` | `stylistic` | Ritornello bright (`2.5–6k`) | Loudness match `<= 0.01 LU` vs `+0.964 LU` |
| `S06` | **Attivo** (`2/2`) | `H02` | `V1` vs `V1b` | `track06` *(nuova)* | `stylistic` | Sez. sibilante/harsh `3–7 kHz` | Seconda traccia indipendente harshness |
| `S07` | **Flipped** (`S05`) | `H02` | `V1` vs `V1b` *(flip)* | `track01` | `stylistic` | Identica a `S05` (A/B invertiti) | Check coerenza timbrica vs asprezza |
| `S08` | **Attivo** (`1/5`) | `H03` | `V0p` vs `V2` *(comp.)* | `track01` | `same_track` | Drop/ritornello denso | Pacchetto composito `Transient+PLR` |
| `S09` | **Attivo** (`2/5`) | `H03` | `V0p` vs `V2` *(comp.)* | `track02` | `stylistic` | Passaggio dinamico acustico | Pacchetto composito `Transient+PLR` |
| `S10` | **Attivo** (`3/5`) | `H03` | `V0p` vs `V2` *(comp.)* | `track03` | `stylistic` | Ritornello/percussioni (`||dz||=0.21`) | Pacchetto composito `Transient+PLR` |
| `S11` | **Attivo** (`4/5`) | `H03` | `V0p` vs `V2` *(comp.)* | `track04` *(nuova)* | `same_track` | Attacchi naturali voce/chitarra | Non-regressione su dinamica acustica |
| `S12` | **Attivo** (`5/5`) | `H03` | `V0p` vs `V2` *(comp.)* | `track05` *(nuova)* | `same_track` | Batteria secca / kick-snare | Target primario transienti |
| `S13` | **Flipped** (`S12`) | `H03` | `V0p` vs `V2` *(flip)* | `track05` *(nuova)* | `same_track` | Identica a `S12` (A/B invertiti) | Check coerenza su attacchi batteria |
| `S14` | **Attivo** (`1/3`) | `H04` | `V0p` vs `V3a` | `track03` | `stylistic` | Sezione spettralmente piena | Verifica soglia udibilità `V3a` |
| `S15` | **Attivo** (`2/3`) | `H04` | `V0p` vs `V3a` | `track04` *(nuova)* | `same_track` | Passaggio acustico trasparente | Evita boost artificiali globali |
| `S16` | **Attivo** (`3/3`) | `H04` | `V0p` vs `V3a` | `track06` *(nuova)* | `stylistic` | Passaggio bright / EQ ampio | `3` tracce distinte |
| `S17` | **Attivo** (`1/3`) | `H05` | `V0p` vs `V3b_B05` | `track01` | `same_track` | Climax (`limiter GR > 0.5 dB`) | Unica storica fuori deadzone `V3b` |
| `S18` | **Attivo** (`2/3`) | `H05` | `V0p` vs `V3b_B05` | `track05` *(nuova)* | `stylistic` | Batteria densa spinta su ref | Alta pressione limiter (`GR > 0.5 dB`) |
| `S19` | **Attivo** (`3/3`) | `H05` | `V0p` vs `V3b_B05` | `track07` *(nuova)* | `same_track` | Drop sub-808 / wall-of-sound | Stress limiter mirato (`-7..-9 LUFS`) |
| `S20` | **Flipped** (`S19`) | `H05` | `V0p` vs `V3b_B05` *(flip)* | `track07` *(nuova)* | `same_track` | Identica a `S19` (A/B invertiti) | Check coerenza anti-pumping |
| `S21` | **Attivo** (`1/2`) | `H06` *(espl.)* | `V0p` vs `V3c` | `track08` *(nuova)* | `same_track` | Sezione A wide-stereo (`corr<0.55`) | **Solo esplorativa (`< 2` tracce wide)** |
| `S22` | **Attivo** (`2/2`) | `H06` *(espl.)* | `V0p` vs `V3c` | `track08` *(nuova)* | `same_track` | Sezione B wide-stereo (`corr<0.55`) | **Non promuovibile senza 2a traccia wide** |
| `S23` | **Null Anchor** (`1/2`) | `NULL` | `V0p` vs `V0p` *(bit-identico)* | `track02` | `same_track` | Identica a `S01` | Verifica falsi positivi ascoltatore |
| `S24` | **Null Anchor** (`2/2`) | `NULL` | `V1` vs `V1` *(bit-identico)* | `track06` *(nuova)* | `stylistic` | Identica a `S06` | Verifica falsi positivi in regime harsh |

**Riepilogo bilancio:** `18` attivi (`H01:3`, `H02:2`, `H03:5`, `H04:3`, `H05:3`, `H06:2`) + `4` flipped (`S04, S07, S13, S20`) + `2` null anchor (`S23, S24`) = **`24` trial esatti** (`6/24 = 25.0%` controlli nascosti), distribuiti in **`3` sessioni da `8` trial** dopo permutazione CSPRNG (`>= 9` `same_track`, `>= 7` `stylistic`).

- **Fase 2 — Sessione di Conferma Mirata (`8–12` trial aggiuntivi, solo post-screening):**
  - Riservata esclusivamente ai contrasti che nello screening hanno mostrato preferenza non banale (`score != Tie` con confidenza `>= 2`) o trade-off ambigui (es. `V1b` con overshoot `+0.964 LU` pre-matching vs timbro post-matching, `V2` pacchetto composito su transienti acustici vs elettronici, `V3b` su nuovi brani ad alto GR), confermando la ripetibilità su segmenti/tracce aggiuntive prima di qualsiasi affermazione inferenziale.

---

## §16. Criteri Espliciti di Promozione / Scarto verso `V4` & Gap List

### 16.1 Regole Decisionali di Screening & Promozione verso `V4`

Una singola componente (`V1`, `V1b`, `V2` pacchetto composito, `V3a`, `V3b`, `V3c`) può superare la fase di **screening esplorativo** verso la candidatura `V4` solo se soddisfa **tutti** i requisiti seguenti sui dati unblinded (dopo superamento della coerenza per variante sulle `4` repliche flipped `>= 75%` e dei `2` null anchor):

1. **Copertura Minima di Tracce Indipendenti (`>= 2` brani distinti obbligatoria):**
   - **Nessuna ipotesi o componente testata su meno di `2` brani indipendenti (`track_id` distinti) può essere promossa dallo screening verso `V4`.** Sezioni multiple dello stesso brano (`S21`/`S22` su `track08`) o repliche flipped contano come **`1` solo brano indipendente**.
   - **Stato corrente di `H06` (`V3c`):** nel pannello `track01..track08`, poiché solo `track08` rappresenta il regime `same_track` wide-stereo/out-of-phase (mentre `track01..03` sono già chiuse storicamente a `3/3 Ties`), **`H06` è contrassegnata esplicitamente come strettamente esplorativa e NON promuovibile dallo screening** finché non venga acquisito almeno un secondo brano indipendente wide-stereo idoneo.
2. **Non-Regressione Rigorosa:** tasso di regressione percettiva (vittorie Control sul Treatment con confidenza `>= 2`) `<= 15%` sui brani applicabili e **zero** artefatti uditivi gravi segnalati (pumping severo, harshness aggiunta, collapse mono, clipping).
3. **Segnale di Beneficio Netto (Screening Diagnostico — senza generalizzazione inferenziale indebita):**
   - Lo screening Phase 1 su `1` ascoltatore fornisce evidenza **esplorativa/diagnostica**, non significatività statistica popolazionale (`p`-value formali riservati a campagne multi-traccia/multi-ascoltatore più ampie).
   - **Promozione candidata (Phase 1 -> Phase 2 conferma):** `Net Win Rate = (Wins_Treat - Wins_Ctrl) / Non_Tie >= +30%` con preferenza netta positiva su almeno **`2` brani indipendenti** (`Wins_Treat > Wins_Ctrl`, confidenza `>= 2`) e zero regressioni altrove (nota: in caso di promozione di `H03`, va annotato che la promozione riguarda il pacchetto composito V2 nel suo insieme);
   - **Opzione "Safety Prior" (promozione difensiva condizionata):** ammessa esclusivamente per prior di regolarizzazione (`V3a`, `V3b`, `V3c`) anche in prevalenza di `Tie`, **purché** (a) copra almeno `2` brani indipendenti fuori deadzone, (b) non generi regressioni rispetto a `V0p`, e (c) dimostri utilità misurabile contro failure mode reali su tracce patologiche (in caso di `||dz|| == 0` sulle tracce attive, ricalibrare soglie prima dell'uso o escludere per parsimonia);
   - **Scarto / Ricalibrazione obbligatoria:** se un intervento registra `Wins_Ctrl > Wins_Treat`, peggiora il trade-off oggettivo/percettivo (es. **`V1b` che sposta `+0.964 LU` verso il più forte (`-17.83 -> -16.86 LUFS`), generando `+1.123 LU` di overshoot sopra il target `-17.99 LUFS` e riducendo il PLR di `0.84 dB`**) oppure produce `0` effetto misurabile e percettivo (`Tie` unanimi anche sotto stress test mirato).

### 16.2 Gap List Oggettiva Emersa dall'Audit Read-Only

*(Nessuna azione correttiva/ottimizzazione intrapresa ora; elenco vincolante per il passo successivo quando autorizzato)*:

1. **Gap Dataset (`3` tracce vs `6–8` richieste, di cui `track04..track08` mancanti):** in `human_validation/private/input/` esistono solo `track01..03`. Servono `3–5` nuovi brani reali a `48 kHz` (`acustico/dinamico`, `transiente percussivo`, `bright/harsh`, `sub-bass/limiter stress`, `wide-stereo`) — più almeno una seconda traccia wide-stereo se si desidera rendere `H06` (`V3c`) non meramente esplorativa.
2. **Gap `V3b` Deadzone:** con l'attuale soglia `tau_avg=0.5 dB` e `tau_95=2.0 dB` sui 6 brani storici, `V3b_B05` è identically equal (`||dz||_2 = 0.00000`) a `V0p` su **5 sessioni su 6** (`track02..03` e tutti gli `_style`). Prima di trarre conclusioni su `V3b`, va testato su tracce con limiter fortemente attivo (`track01`, `track05`, `track07`, target loudness `-7..-9 LUFS`).
3. **Gap Trade-Off `V1b` Loudness Overshoot (`+0.964 LU` rispetto a `V1`, `+1.123 LU` oltre il target su `track01_style`):** la penalità `weight_de_harsh=0.15` su `track01_style` abbatte la zona `2.5–5 kHz` (`-2.645 dB` ERB surplus), ma fa salire il guadagno medio EQ (`-1.915 -> -1.263 dB`) e spinge la loudness integrata a `-16.865 LUFS` rispetto al target `-17.988 LUFS` (`+0.964 LU` più forte di `V1`, con calo PLR da `15.37` a `14.53 dB`). In `V4` sarà opportuno verificare se abbinare `V1b` con `V3a`/schedula o peso tarato elimini questo overshoot di loudness.
4. **Gap Accoppiamento DSP `legacy_sample_1x` vs `true_peak_4x` nei Checkpoint Congelati:** tutti i vettori `z_opt` storici `V0..V3c` sono stati ottimizzati sotto il vecchio limiter `1x` (`-0.30 dBFS`, `boxcar=144`). Sebbene il re-render offline con Sprint 4 `true_peak_4x` garantisca immediatamente la conformità `PCM_24 <= -0.30 dBTP` con soli `0.003..0.046 LU` di scarto sui 6 render V3c, **la campagna `V4`** (quando autorizzata dopo l'esito del protocollo) dovrà ottimizzare i checkpoint candidati direttamente dentro la catena `true_peak_4x` affinché Adam veda il comportamento reale del limiter True Peak nel forward/backward pass.
5. **Gap Applicabilità `V3c` (`stylistic` escluso):** `V3cStereoCompatibilityLoss` resta bloccata in `stylistic` (`raise ValueError`) per conflitto strutturale con `stylistic_stereo_loss`, ed è rimasta sub-soglia sui 3 brani `same_track` originali (`3/3` Ties). La eventuale riabilitazione di `V3c` in `V4` resta subordinata a verifica su nuove tracce con rischio reale di collasso mono (`track08`).

---

*Protocollo di Validazione Umana Pre-V4 aggiornato con correzioni FASE D — Pronto per Data Intake `track04..track08` ([`SPRINT4_PRE_V4_TRACK_INTAKE_MANIFEST.json`](./SPRINT4_PRE_V4_TRACK_INTAKE_MANIFEST.json)).*
