# SPRINT 4 — PRE-V4 TRACK DATA INTAKE REPORT (`track04–track09`)

**Repository:** `gio961gio/AutoMastering`  
**Branch:** `human-validation-package`  
**Stato Intake:** `COMPLETE_VALIDATED_48K_FROM_44K1` *(coppie `same_track` + tutti i `ref_altro_brano` richiesti)*  
**Baseline DSP invariata:** `db4a7bf26db50ecf0401bcac77374eabd535050e`  
**Ceiling interno confermato:** `-0.50 dBTP` | **Gate contrattuale:** `PCM_24 <= -0.30 dBTP`  
**Manifest associato:** [SPRINT4_PRE_V4_TRACK_INTAKE_MANIFEST.json](SPRINT4_PRE_V4_TRACK_INTAKE_MANIFEST.json)  
**Directory locale audio (gitignored):** `human_validation/private/input/`

---

## 1. Stato Complessivo e Trasparenza Normativa

- **Coppie lossless complete (`same_track` + `stylistic`):** tutti gli slot richiesti (`track04`–`track08`, inclusi `track05_ref_altro_brano.wav`, `track06_ref_altro_brano.wav`, `track07_ref_altro_brano.wav`) più una **seconda traccia wide-stereo indipendente (`track09`)** per `H06` sono stati acquisiti da **fonti lossless reali (`WAV` / `FLAC`)** e validati in locale (`15` file WAV stereo `48000 Hz` per `track04–track09`).
- **Conformità anti-streaming lossy:** **nessun** file proviene da YouTube o rip MP3/Opus/AAC (`commercial_stream_rips_forbidden: true` integralmente rispettato).
- **Nessuna ottimizzazione / training / render eseguito:** `dsp_optimization_executed = false`, `blind_pack_generated = false`. Nessun file audio tracciato in Git.
- **Trasparenza sul ricampionamento (Regola 9):** tutti i sorgenti scaricati erano nativi a **`44100 Hz`**. Su esplicita autorizzazione dell'utente, ciascun file è stato ricampionato offline a **`48000 Hz`** con filtro polifase razionale esatto `scipy.signal.resample_poly(160, 147)` e verificato con `scripts.human_validation.validate_inputs.load_and_validate_wav(..., allow_mono=False)`. Gli SHA-256 pre e post conversione sono registrati nel manifest.
- **Subtypes WAV (`PCM_24` e `FLOAT`):** conformemente a `policy.allowed_subtypes` (`["PCM_24", "FLOAT", "DOUBLE"]`), i file il cui picco post-conversione `48 kHz` raggiunge o supera `0.0 dBFS` (`track05_ref_altro_brano`, `track06_ref`, `track06_ref_altro_brano`, `track07_ref`, `track08_mix`, `track08_ref`) usano **IEEE `FLOAT` (32-bit)** per prevenire hard clipping in quantizzazione intera; tutti gli altri usano **`PCM_24`**.

---

## 2. Tabella Riepilogativa delle Metriche (`48000 Hz`, Stereo `2ch`, `0` NaN/Inf)

| File (`human_validation/private/input/`) | Subtype | Durata (s) | LUFS (BS.1770) | Sample Peak (dBFS) | 4x True Peak (dBTP) | PLR (dB) | Stereo Corr | Mono Drop (dB) | Sub 20–80 Hz (dB rel) | Pres 3–7 kHz (dB rel) | SHA-256 (primi 12 hex) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `track04_mix.wav` | `PCM_24` | `84.50` | `-14.044` | `-0.007` | `-0.001` | `14.037` | `+0.8263` | `-0.395` | `-13.788` | `-26.707` | `c51bb33234a5` |
| `track04_ref.wav` | `PCM_24` | `84.50` | `-9.813` | `-0.119` | `-0.105` | `9.694` | `+0.8047` | `-0.446` | `-12.801` | `-21.416` | `8e73b459cb56` |
| `track05_mix.wav` | `PCM_24` | `77.00` | `-17.555` | `-2.168` | `-2.158` | `15.387` | `+0.9470` | `-0.117` | `-6.194` | `-19.270` | `9ba2f5dba96f` |
| `track05_ref.wav` | `PCM_24` | `77.00` | `-10.100` | `-0.189` | `-0.189` | `9.911` | `+0.9221` | `-0.174` | `-5.221` | `-16.272` | `4771979167b6` |
| `track05_ref_altro_brano.wav` | `FLOAT` | `250.566` | `-13.392` | `+0.002` | `+0.199` | `13.394` | `+0.9780` | `-0.049` | `-7.242` | `-15.768` | `bd35ee549873` |
| `track06_mix.wav` | `PCM_24` | `72.00` | `-18.130` | `-0.021` | `-0.013` | `18.109` | `+0.7791` | `-0.508` | `-20.646` | **`-10.302`** | `3ae24037f515` |
| `track06_ref.wav` | `FLOAT` | `72.00` | `-13.015` | `-0.000` | `+0.017` | `13.015` | `+0.5750` | `-1.037` | `-3.671` | `-20.796` | `74c08c803b2e` |
| `track06_ref_altro_brano.wav` | `FLOAT` | `215.680` | `-8.654` | `+1.022` | `+1.466` | `9.676` | `+0.9198` | `-0.189` | `-6.380` | **`-19.974`** | `808199801d43` |
| `track07_mix.wav` | `PCM_24` | `33.306` | `-14.911` | `-0.832` | `-0.747` | `14.079` | `+0.9670` | `-0.073` | **`-2.216`** | `-18.173` | `e7fddd048a5e` |
| `track07_ref.wav` | `FLOAT` | `33.308` | `-7.320` | `+0.131` | `+0.215` | `7.451` | `+0.9614` | `-0.085` | `-2.157` | `-18.066` | `e367bd31399e` |
| `track07_ref_altro_brano.wav` | `PCM_24` | `193.500` | `-11.855` | `-0.401` | `-0.353` | `11.454` | `+0.9210` | `-0.176` | **`-3.338`** | `-14.486` | `fa6c04b627d3` |
| `track08_mix.wav` | `FLOAT` | `334.679` | `-12.938` | `+0.057` | `+0.057` | `12.995` | **`+0.3141`** | `-1.826` | `-2.417` | `-21.828` | `8873275e961d` |
| `track08_ref.wav` | `FLOAT` | `334.679` | `-9.679` | `+1.071` | `+1.110` | `10.750` | `+0.3063` | `-1.851` | `-2.293` | `-20.795` | `7e7ec8f95024` |
| `track09_mix.wav` | `PCM_24` | `33.098` | `-12.772` | `-1.215` | `-1.211` | `11.557` | **`+0.4209`** | `-1.485` | `-11.175` | `-35.078` | `4616c48bc7ce` |
| `track09_ref.wav` | `PCM_24` | `33.098` | `-11.221` | `-1.030` | `-1.030` | `10.191` | `+0.4757` | `-1.324` | `-7.408` | `-37.261` | `9c54c6af2ead` |

---

## 3. Report di Intake per Singola Traccia

### `track04` — Acustico / Dinamico (`H01`, `H03`, `H04`)
- **Brano (`same_track`):** *Tico Da Costa - Lagartixa* (`Track 1`, `0.0–84.5 s`, XARC Mastering `https://www.xarcmastering.com/demos/`).
- **Allineamento & Idoneità:** `env_corr = 0.9615` (`lag = +0.00 s`), `PLR` mix = **`14.037 dB`** (`>= 11.0 dB`). `stylistic_external_reference` facoltativo come da schema.

### `track05` — Transienti rapidi / Batteria (`H03`, `H05`)
- **Brano (`same_track`):** *J-Mac! - Everything* (`Track 1`, `0.0–77.0 s`, XARC Mastering `https://www.xarcmastering.com/demos/`), `env_corr = 0.9422` (`lag = +0.02 s`), headroom mix `SP -2.168 dBFS`, creste 10 ms `p95 = 10.24 dB / p99 = 11.43 dB / max = 13.60 dB`.
- **Riferimento stilistico (`track05_ref_altro_brano.wav`):** *tinear* (Archive.org `https://archive.org/download/tinear/tinear.wav`, licenza **CC-BY 3.0**, `250.566 s`, funk/hip-hop percussivo coerente con *J-Mac!*, `LUFS -13.392`, `TP +0.199 dBTP`, `corr +0.9780`).

### `track06` — Bright / Harshness & Sibilanza `3–7 kHz` (`H02`, `H04`)
- **Mix (`track06_mix.wav`):** *Sound On Sound Mix Rescue — Kirsty Cooke* (`16.0–88.0 s`, `72.0 s` estratti prima dei clip `>= 88.54 s`), `pres3k_7k_rel = -10.302 dB`. Accompagnato anche da `track06_ref.wav` (SOS *Remix* `16.0–88.0 s`, surplus `+10.494 dB` e `ERB surplus +4.8732 dB`).
- **Riferimento stilistico (`track06_ref_altro_brano.wav`):** *Hungry Lucy - A Girl Alone* (`https://archive.org/download/HungryLucy_BeforeWeStand/01_-_Hungry_Lucy_-_A_Girl_Alone.flac`, pop/electronic vocale, `215.680 s`, `LUFS -8.654`, `pres3k_7k_rel = -19.974 dB`).
  - **Verifica `ERBSpectralLoss(sample_rate=48000.0)` su `track06_mix` vs `track06_ref_altro_brano`:** **`erb_presence_surplus_db = +1.4039 dB`**, `erb_de_harsh_loss = 0.4275` (`pres3k_7k` delta `+9.672 dB`), confermando l'attivazione positiva richiesta da `V1b` *(il precedente candidato `jw_01_release.flac` è stato scartato proprio perché produceva tilt surplus `-0.1326 dB`)*.

### `track07` — Sub-bass & Limiter Stress (`H05`, `H03`)
- **Brano (`same_track`):** *SIMON_V - Cloudspotter* (taglio utente da `0:14`, `33.31 s`, `https://www.simonv.com/mastering/`), `env_corr = 0.9007` (`lag = +0.00 s`), sub-bass `-2.216 dB`, target master `-7.320 LUFS` (`+7.591 LU` vs headroom pre-ceiling `+0.247 dB`).
- **Riferimento stilistico (`track07_ref_altro_brano.wav`):** *Chump Change - Footage feat. Deserved* (Archive.org `https://archive.org/download/yarn014/01-Chump_Change-Footage_feat._Deserved.flac`, licenza **CC-BY 4.0**, `193.500 s`, `sub20_80_rel = -3.338 dB`, `LUFS -11.855`, `TP -0.353 dBTP`, `0` clip), che richiede `+3.056 LU` rispetto a `track07_mix` (`SP -0.832 dBFS`) attivando limiter GR `> 0.5 dB` anche in modalità `stylistic`.

### `track08` — Wide Stereo & Rischio Mono (`H06`)
- **Brano (`same_track`):** *Ainst - Drive* (`Unmastered` vs `Club Master`, `334.679 s`, Archive.org `https://archive.org/details/ainst-drive`), `env_corr = 0.9640` (`lag = +0.00 s`), **`stereo_correlation = +0.3141`** (`<= 0.55` ideale) e `mono_sum_drop = -1.826 dB`.

### `track09` — Seconda Traccia Wide Stereo Indipendente (`H06`)
- **Brano (`same_track`):** *Audio Animals - Orchestral 1* (`33.098 s`, `https://www.audioanimals.co.uk/pages/before-after-audio-examples`), `env_corr = 0.7361` (`lag = +0.00 s`), **`stereo_correlation = +0.4209`** (`<= 0.55` ideale) e `mono_sum_drop = -1.485 dB`, `PLR = 11.557 dB`.

---

## 4. Verifica `H06 — Seconda traccia wide-stereo`

- Con `track08` (`+0.3141`) e `track09` (`+0.4209`), il pacchetto dispone di **2 tracce wide-stereo indipendenti**, entrambe sotto la soglia ideale `<= 0.55`.
- **Nota di governance:** la durata di `track09` (`33.10 s`) è leggermente inferiore a `[35.0, 90.0]` (ma `>= 20–25 s`). In questa fase di intake non si dichiara alcuna promozione automatica di `V3c` verso `V4`.

---

## 5. Scostamenti Residui / Note Documentali

1. **Nessun file mancante** per tutti i regimi richiesti (`same_track` + `stylistic` su `track04..track09`).
2. **Conversione `44.1 → 48 kHz`:** tutti i sorgenti erano nativi `44.1 kHz` e sono stati convertiti offline con autorizzazione utente e tracciamento SHA-256 completo.
3. **Subtypes `FLOAT` su 6 file:** usati per evitare clipping di picchi `> 0 dBFS` nativi dei master/mix originali.
4. **Durate sotto `[35.0, 90.0]` (`track07`: `33.31 s` per taglio `>= 0:14`; `track09`: `33.10 s` durata demo):** entrambe superano `>= 20 s`.
5. **Uso strettamente privato:** i brani XARC, SOS, Simon V, Audio Animals e Archive.org (incluse release con `licenseurl=None` come *Ainst - Drive* o *Hungry Lucy - A Girl Alone*, i cui metadati API non erano raggiungibili dall'ambiente offline) sono destinati **esclusivamente alla validazione privata locale** e non alla ridistribuzione.
