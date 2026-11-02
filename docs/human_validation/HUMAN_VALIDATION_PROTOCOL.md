# Protocollo di Validazione Umana — Sprint 1 & Sprint 2

**Progetto:** AutoMastering (Politecnico di Milano — DEIB)  
**Candidato:** Giovanni Antonecchia  
**Commit motore congelato (Codice):** `5940e81f12741e2276f7a446adfb6e12f44574fe`  
**Commit motore congelato (Documentazione):** `92fe105a19312f258fefb2c70a847720cbd95525`  
**Commit storico Legacy Baseline analizzato:** `86ae20f08dc7fbebc7f67af064a15b648a492fc7`

---

## 1. Scopo e regole fondamentali

 I test automatici di regressione (128 test Python + 12 stage C++ JUCE headless) garantiscono la correttezza matematica e numerica del motore, ma **non sostituiscono l'ascolto critico umano su brani musicali reali**.

Questo protocollo guida l'esecuzione di due esperimenti d'ascolto previsti dalla roadmap:
1. **Esperimento A (Sprint 1):** verifica percettiva di **True Bypass (A)**, **Legacy Initialization Baseline (B)** e **Corrected Neutral (C)** su un brano reale.
2. **Esperimento B (Sprint 2):** verifica percettiva A/B (Originale/Bypass vs Master Ottimizzato) su **almeno 3 brani reali** e **almeno 3 sezioni per brano** (intro/strofa tranquilla, ritornello/drop denso, bridge/outro con transienti), sia a livello reale di programma (*program-level*) sia a volume pareggiato staticamente (*loudness-matched*).

> [!IMPORTANT]
> **Nessun file audio privato nel repository Git:** tutti i file WAV sorgente, gli export, i set ciechi e le chiavi devono rimanere esclusivamente sotto `human_validation/private/`, cartella già ignorata da `.gitignore`.

---

## 2. Requisiti dei file audio in ingresso

Prima di iniziare, prepara i tuoi file nella cartella locale `human_validation/private/input/`:

1. **Formato:** WAV (PCM 16/24-bit oppure IEEE Float 32/64-bit), stereo (2 canali) o mono (1 canale; se mono, verrà duplicato in stereo e dichiarato nel manifest).
2. **Sample rate obbligatorio:** **48000 Hz (48 kHz)**. Il sistema rifiuta qualsiasi file a sample rate diverso (es. 44.1 kHz) per evitare ricampionamenti silenziosi. Se hai un file a 44.1 kHz o 96 kHz, convertilo **una sola volta** prima del test:
   ```powershell
   ffmpeg -i "mio_brano_44100.wav" -ar 48000 -c:a pcm_f32le "human_validation/private/input/track01_mix.wav"
   ```
3. **Brani richiesti:**
   - **Per Esperimento A (Sprint 1):** 1 mix stereo non masterizzato (`track01_mix.wav`), preferibilmente con voce, basse frequenze e transienti ben definiti.
   - **Per Esperimento B (Sprint 2):** almeno **3 mix stereo distinti** (`track01_mix.wav`, `track02_mix.wav`, `track03_mix.wav`, es. Pop/Voce, Hip-Hop/Elettronica, Rock/Acustico) accompagnati ciascuno dal proprio file di riferimento a 48 kHz:
     - Usa `--mode same_track` **solo** se il reference è il master dello **stesso identico brano** del mix.
     - Usa `--mode stylistic` se il reference è un **altro brano commerciale** scelto come riferimento timbrico/dinamico.

Puoi verificare in anticipo che tutti i WAV siano conformi a 48 kHz con:
```powershell
python -m scripts.human_validation.validate_inputs "human_validation/private/input/track01_mix.wav"
```

---

## 3. Stato della Legacy Baseline (`86ae20f08dc7fbebc7f67af064a15b648a492fc7`) e scelta della policy per Esperimento A

L'analisi della cronologia Git antecedente al Gate A ha identificato il commit iniziale `86ae20f08dc7fbebc7f67af064a15b648a492fc7`, in cui `AutoMasteringOptimizer` inizializzava `u = torch.zeros(1, 41)` producendo `z = [0.5]^41`.

Con i range fisici di `86ae20f08dc7fbebc7f67af064a15b648a492fc7`, il vettore fisico `theta_legacy` a `z = 0.5` valeva:
- **EQ 6 bande (`0..17`):** tutti i guadagni a `0.0 dB`, `Q = 1.581139` ($\sqrt{0.5 \times 5.0}$), frequenze centrali geometriche `[70.71, 223.61, 894.43, 3000.00, 6928.20, 9486.83] Hz`.
- **Crossover (`18..19`):** `f_low = 167.3320 Hz`, `f_high = 3000.0 Hz`.
- **Compressore 3 bande (`20..34`):** `threshold = -20.0 dB`, **`ratio = 4.472136:1`** ($\sqrt{1 \times 20}$), **`attack = 7.071068 ms`** ($\sqrt{0.5 \times 100}$), `release = 100.0 ms`, **`makeup = +3.0 dB`** su tutte le 3 bande.
- **Stereo Imager (`35..37`):** `width_low = 0.60`, `width_mid = 1.25`, `width_high = 1.25`.
- **Saturatore Armonico (`38..39`):** **`drive = 9.0 dB`**, `mix = 0.50`.
- **Lookahead Limiter (`40`):** **`push_gain = 6.0 dB`**.

### Finding di compatibilità (`LEGACY_BASELINE_NOT_REPRODUCIBLE` in modalità `strict`)

Confrontando `theta_legacy` con il contratto congelato del motore attuale (`5940e81f12741e2276f7a446adfb6e12f44574fe`), emergono **due incompatibilità oggettive**:
1. **5 parametri su 41 escono dai range fisici attuali (`PARAM_SPECS`):**
   - `[21] comp_low_ratio = 4.472136` vs range attuale `[1.0, 4.0]`
   - `[22] comp_low_attack_ms = 7.071068 ms` vs range attuale `[10.0, 120.0] ms`
   - `[26] comp_mid_ratio = 4.472136` vs range attuale `[1.0, 4.0]`
   - `[31] comp_high_ratio = 4.472136` vs range attuale `[1.0, 4.0]`
   - `[38] saturator_drive_db = 9.0 dB` vs range attuale `[0.0, 6.0] dB`
2. Inoltre, nel commit `86ae20f0`, `HarmonicSaturator.hpp` normalizzava `tanh(x * gd)` per `1 / sqrt(gd)`, mentre il motore attuale congelato normalizza per `1 / gd`.

In conformità alla Sezione 3 della specifica:
- La modalità predefinita `--legacy-mode strict` **non forza valori fuori range** e salva il finding strutturato `finding_legacy_baseline.json` (`LEGACY_BASELINE_NOT_REPRODUCIBLE`), esportando A (`True Bypass`) e C (`Corrected Neutral`).
- Per esportare anche la condizione **B ("Legacy initialization baseline replayed on frozen current engine")** sul motore congelato, puoi specificare esplicitamente la policy desiderata:
  - `--legacy-mode clamped_to_current_bounds`: applica `theta_legacy` limitando i soli 5 valori fuori range ai massimi/minimi consentiti dal contratto attuale (`ratio -> 4.0`, `low_attack -> 10.0 ms`, `drive -> 6.0 dB`) e mantenendo invariati gli altri 36 parametri legacy (inclusi `+3 dB` di makeup per banda, `width = 1.25` e `+6 dB` di limiter push).
  - `--legacy-mode raw_theta_on_current_engine`: passa i 41 valori fisici `theta_legacy` direttamente a `MasteringChain.set_parameters_theta`.

---

## 4. Esecuzione passo-passo — Esperimento A (Sprint 1)

### 4.1 Comando Windows PowerShell

```powershell
# 1. Modalità strict predefinita (genera il finding sulla Legacy Baseline ed esporta A e C)
python -m scripts.human_validation.render_sprint1 `
  --input "human_validation/private/input/track01_mix.wav" `
  --out "human_validation/private/sprint1_track01" `
  --track-id track01

# 2. Esportazione completa A / B / C con replay della Legacy Baseline clampata ai bound correnti
python -m scripts.human_validation.render_sprint1 `
  --input "human_validation/private/input/track01_mix.wav" `
  --out "human_validation/private/sprint1_track01" `
  --track-id track01 `
  --legacy-mode clamped_to_current_bounds `
  --overwrite
```

### 4.2 Comando Shell POSIX (Linux / macOS / Git Bash)

```bash
python -m scripts.human_validation.render_sprint1 \
  --input "human_validation/private/input/track01_mix.wav" \
  --out "human_validation/private/sprint1_track01" \
  --track-id track01 \
  --legacy-mode clamped_to_current_bounds \
  --overwrite
```

### 4.3 File generati da Esperimento A

Nella cartella `human_validation/private/sprint1_track01/` troverai:
- `track01_source.sha256` e `track01_metadata.json`
- `finding_legacy_baseline.json` (tabella completa dei 41 parametri storici vs attuali)
- `exports/track01_A_true_bypass.wav`, `track01_B_legacy_baseline.wav`, `track01_C_corrected_neutral.wav` (con trim diagnostico uniforme a `-6.0 dBTP` e compensazione esatta dei 159 campioni + tail flush)
- `exports/*_level_matched.wav` e `exports/level_matching_report.json` (copie per l'ascolto pareggiate con solo gain statico e tetto `<= -1.0 dBTP`)
- `manifests/track01_A_manifest.json`, `track01_B_manifest.json`, `track01_C_manifest.json`
- `blind/blind_A.wav`, `blind/blind_B.wav`, `blind/blind_C.wav`, la scheda vuota `blind/track01_sprint1_listening_sheet.csv` e la chiave segreta `blind/track01_blind_key.json` (**non aprire prima di aver completato l'ascolto!**).

---

## 5. Esecuzione passo-passo — Esperimento B (Sprint 2)

Ripeti il comando seguente per ciascuno dei **3 brani** (`track01`, `track02`, `track03`), scegliendo `--mode same_track` oppure `--mode stylistic` in base alla natura del reference.

### 5.1 Comando Windows PowerShell

```powershell
# Brano 1 (es. Reference dello stesso identico brano -> same_track)
python -m scripts.human_validation.render_sprint2 `
  --raw "human_validation/private/input/track01_mix.wav" `
  --ref "human_validation/private/input/track01_ref_master.wav" `
  --mode same_track `
  --prefix track01 `
  --out "human_validation/private/sprint2/track01" `
  --notes "Pop/Voce: voce presente, controllo sibilanti e punch naturale"

# Brano 2 (es. Reference stilistica di un altro brano -> stylistic)
python -m scripts.human_validation.render_sprint2 `
  --raw "human_validation/private/input/track02_mix.wav" `
  --ref "human_validation/private/input/track02_stylistic_ref.wav" `
  --mode stylistic `
  --prefix track02 `
  --out "human_validation/private/sprint2/track02" `
  --notes "Hip-hop/Elettronica: sub-bass compatto, kick incisivo, densità alta"

# Brano 3 (con sezioni d'ascolto manuali opzionali: ruolo:inizio_sec:fine_sec)
python -m scripts.human_validation.render_sprint2 `
  --raw "human_validation/private/input/track03_mix.wav" `
  --ref "human_validation/private/input/track03_stylistic_ref.wav" `
  --mode stylistic `
  --prefix track03 `
  --out "human_validation/private/sprint2/track03" `
  --sections "intro_strofa:12.0:26.0" "ritornello_denso:58.0:72.0" "bridge_transienti:115.0:130.0" `
  --notes "Rock/Acustico: dinamica aperta, immagine stereo naturale"
```

### 5.2 Comando Shell POSIX (Linux / macOS / Git Bash)

```bash
python -m scripts.human_validation.render_sprint2 \
  --raw "human_validation/private/input/track01_mix.wav" \
  --ref "human_validation/private/input/track01_ref_master.wav" \
  --mode same_track \
  --prefix track01 \
  --out "human_validation/private/sprint2/track01"
```

### 5.3 File generati per ciascun brano in Esperimento B

In `human_validation/private/sprint2/trackXX/`:
- `trackXX_original_or_bypass.wav` e `trackXX_mastered_full.wav` (durata identica al mix originale, latenza di 159 campioni compensata una sola volta, coda preservata integralmente)
- `trackXX_preset.json` (l'unico preset ottimizzato per l'intero brano)
- `trackXX_manifest.json` e `trackXX_metrics_pre_post.json`
- `trackXX_sections.json` (documenta sia i chunk usati dall'ottimizzatore sia le 3 sezioni d'ascolto estratte)
- `trackXX_sections/section01_original.wav`, `section01_mastered.wav`, ... (`section01`, `section02`, `section03` a livello di programma)
- `trackXX_sections_level_matched/sectionXX/` (copie pareggiate con solo gain statico e tetto `<= -1.0 dBTP`)
- `trackXX_sections_blind/sectionXX_program_level/` e `sectionXX_level_matched/` (coppie anonime `blind_A.wav` / `blind_B.wav` + scheda CSV + chiave segreta separata).

---

## 6. Procedura di ascolto e compilazione dei report

1. **Setup di ascolto:** usa cuffie o monitor da studio affidabili, disattiva qualsiasi normalizzatore/equalizzatore del sistema operativo o del player, e annota il sistema utilizzato.
2. **Fase Cieca (Blind Listening):**
   - Apri i file `blind_A.wav`, `blind_B.wav` (e `blind_C.wav` per Sprint 1) nella tua DAW o player senza guardare i file `*_blind_key.json`.
   - Compila i template in:
     - [`docs/human_validation/SPRINT1_LISTENING_TEMPLATE.md`](SPRINT1_LISTENING_TEMPLATE.md) (oppure `.csv`)
     - [`docs/human_validation/SPRINT2_LISTENING_TEMPLATE.md`](SPRINT2_LISTENING_TEMPLATE.md) (oppure `.csv`)
   - Valuta sia la versione **Loudness-Matched** (per giudicare timbro, bilanciamento, punch e artefatti senza l'inganno del volume più alto) sia la versione **Program-Level** (per giudicare la spinta complessiva e il comportamento del limiter).
3. **Svelamento chiave (Unblinding) e Riepilogo Finale:**
   - Solo **dopo** aver salvato tutte le valutazioni cieche, apri i file `*_blind_key.json` e riporta la corrispondenza reale nel documento conclusivo [`docs/human_validation/RESULTS_TEMPLATE.md`](RESULTS_TEMPLATE.md).
