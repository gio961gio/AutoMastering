# SPRINT 4 — PRE-V4 TRACK DATA INTAKE REPORT (`track04–track09`)

**Repository:** `gio961gio/AutoMastering`  
**Branch:** `human-validation-package`  
**Baseline DSP invariata:** `db4a7bf26db50ecf0401bcac77374eabd535050e`  
**Ceiling interno confermato:** `-0.50 dBTP` | **Gate contrattuale:** `PCM_24 <= -0.30 dBTP`  
**Manifest associato:** [SPRINT4_PRE_V4_TRACK_INTAKE_MANIFEST.json](SPRINT4_PRE_V4_TRACK_INTAKE_MANIFEST.json)  
**Directory locale audio (gitignored):** `human_validation/private/input/`

---

## 1. Stato Complessivo e Trasparenza Normativa

- **Coppie lossless acquisite:** tutti gli slot richiesti (`track04`–`track08`) più una **seconda traccia wide-stereo indipendente (`track09`)** per `H06` sono stati acquisiti da **fonti lossless reali (`WAV PCM_16/PCM_24`)** — **nessun** rip da streaming lossy o YouTube (`commercial_stream_rips_forbidden: true` rispettato integralmente).
- **Nessuna ottimizzazione / training / render eseguito:** `dsp_optimization_executed = false`, `blind_pack_generated = false`. Nessun file audio tracciato in Git.
- **Trasparenza sul ricampionamento (Regola 9):** tutti i file originali scaricati erano nativi a **`44100 Hz`**. Su esplicita autorizzazione dell'utente (*"possiam riconvertire noi a 48khz"* / taglio `track07` da `0:14`), ciascun estratto è stato ricampionato offline a **`48000 Hz`** con filtro polifase razionale esatto `scipy.signal.resample_poly(160, 147)` e verificato con `scripts.human_validation.validate_inputs.load_and_validate_wav(..., allow_mono=False)`. Gli SHA-256 sia dei file sorgenti `44.1 kHz` sia dei file validati `48 kHz` sono registrati nel manifest.
- **Subtypes WAV usati (`PCM_24` e `FLOAT`):** conformemente a `policy.allowed_subtypes` (`["PCM_24", "FLOAT", "DOUBLE"]`), i file il cui picco post-ricampionamento `48 kHz` raggiunge o supera `0.0 dBFS` (`track06_ref`, `track07_ref`, `track08_mix`, `track08_ref`) sono stati salvati in **IEEE `FLOAT` (32-bit)** per evitare clipping duro in intero fisso; tutti gli altri sono **`PCM_24`**.
- **Riferimenti stilistici esterni (`*_ref_altro_brano.wav`):** come indicato dall'utente (*"non serve, è opzionale secondo la tabella e nel caso posso farlo io"*), i file `track05_ref_altro_brano.wav`, `track06_ref_altro_brano.wav` e `track07_ref_altro_brano.wav` sono lasciati non popolati (`DEFERRED_OPTIONAL_TO_USER`). Tutte le coppie `same_track` (`mix` + `ref`) sono complete e validate.

---

## 2. Tabella Riepilogativa delle Metriche (`48000 Hz`, Stereo `2ch`, `0` NaN/Inf)

| File (`human_validation/private/input/`) | Subtype | Durata (s) | LUFS (BS.1770) | Sample Peak (dBFS) | 4x True Peak (dBTP) | PLR (dB) | Stereo Corr | Mono Drop (dB) | Sub 20–80 Hz (dB rel) | Pres 3–7 kHz (dB rel) | SHA-256 (primi 12 hex) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `track04_mix.wav` | `PCM_24` | `84.50` | `-14.044` | `-0.007` | `-0.001` | `14.037` | `+0.8263` | `-0.395` | `-13.788` | `-26.707` | `c51bb33234a5` |
| `track04_ref.wav` | `PCM_24` | `84.50` | `-9.813` | `-0.119` | `-0.105` | `9.694` | `+0.8047` | `-0.446` | `-12.801` | `-21.416` | `8e73b459cb56` |
| `track05_mix.wav` | `PCM_24` | `77.00` | `-17.555` | `-2.168` | `-2.158` | `15.387` | `+0.9470` | `-0.117` | `-6.194` | `-19.270` | `9ba2f5dba96f` |
| `track05_ref.wav` | `PCM_24` | `77.00` | `-10.100` | `-0.189` | `-0.189` | `9.911` | `+0.9221` | `-0.174` | `-5.221` | `-16.272` | `4771979167b6` |
| `track06_mix.wav` | `PCM_24` | `72.00` | `-18.130` | `-0.021` | `-0.013` | `18.109` | `+0.7791` | `-0.508` | `-20.646` | **`-10.302`** | `3ae24037f515` |
| `track06_ref.wav` | `FLOAT` | `72.00` | `-13.015` | `-0.000` | `+0.017` | `13.015` | `+0.5750` | `-1.037` | `-3.671` | `-20.796` | `74c08c803b2e` |
| `track07_mix.wav` | `PCM_24` | `33.306` | `-14.911` | `-0.832` | `-0.747` | `14.079` | `+0.9670` | `-0.073` | **`-2.216`** | `-18.173` | `e7fddd048a5e` |
| `track07_ref.wav` | `FLOAT` | `33.308` | `-7.320` | `+0.131` | `+0.215` | `7.451` | `+0.9614` | `-0.085` | `-2.157` | `-18.066` | `e367bd31399e` |
| `track08_mix.wav` | `FLOAT` | `334.679` | `-12.938` | `+0.057` | `+0.057` | `12.995` | **`+0.3141`** | `-1.826` | `-2.417` | `-21.828` | `8873275e961d` |
| `track08_ref.wav` | `FLOAT` | `334.679` | `-9.679` | `+1.071` | `+1.110` | `10.750` | `+0.3063` | `-1.851` | `-2.293` | `-20.795` | `7e7ec8f95024` |
| `track09_mix.wav` | `PCM_24` | `33.098` | `-12.772` | `-1.215` | `-1.211` | `11.557` | **`+0.4209`** | `-1.485` | `-11.175` | `-35.078` | `4616c48bc7ce` |
| `track09_ref.wav` | `PCM_24` | `33.098` | `-11.221` | `-1.030` | `-1.030` | `10.191` | `+0.4757` | `-1.324` | `-7.408` | `-37.261` | `9c54c6af2ead` |

---

## 3. Report di Intake per Singola Traccia

### `track04` — Acustico / Dinamico (`H01`, `H03`, `H04`)
- **Brano:** *Tico Da Costa - Lagartixa* (`Track 1`, estratto pulito `0.0–84.5 s` dallo showcase 5-tracce XARC Mastering).
- **Provenienza / Licenza:** Portfolio pubblico XARC Mastering (`https://www.xarcmastering.com/demos/`, `WAV PCM_24` unmastered / `PCM_16` mastered nativi `44.1 kHz`; uso privato di validazione autorizzato dall'utente, non ridistribuibile).
- **Allineamento same-track (48 kHz):** `env_corr = 0.9615`, `lag = +0.00 s`.
- **Idoneità acustica:** `PLR` mix = **`14.037 dB`** (supera ampiamente la soglia `>= 11.0 dB`), dinamica naturale acustica/vocale conservata (`LUFS -14.044`, `SP -0.007 dBFS`).

### `track05` — Transienti rapidi / Batteria (`H03`, `H05`)
- **Brano:** *J-Mac! - Everything* (`Track 1`, estratto pulito `0.0–77.0 s` dallo showcase 3-tracce XARC Mastering).
- **Provenienza / Licenza:** Portfolio pubblico XARC Mastering (`https://www.xarcmastering.com/demos/`, `WAV PCM_16` nativi `44.1 kHz`; uso privato di validazione autorizzato dall'utente, non ridistribuibile).
- **Allineamento same-track (48 kHz):** `env_corr = 0.9422`, `lag = +0.02 s`.
- **Idoneità acustica:** transienti percussivi netti con ampio headroom sul mix (`SP -2.168 dBFS`, `TP -2.158 dBTP`, `PLR 15.387 dB`); creste 10 ms misurate a 48 kHz: `p95 = 10.24 dB`, `p99 = 11.43 dB`, `max = 13.60 dB`.
- **Gap esplicito:** `track05_ref_altro_brano.wav` non popolato su scelta dell'utente (vedi Sezione 5).

### `track06` — Bright / Harshness & Sibilanza `3–7 kHz` (`H02`, `H04`)
- **Brano:** *Sound On Sound Mix Rescue — Kirsty Cooke* (finestra unclipped `16.0–88.0 s`, `72.0 s`, estratta prima dei campioni clippati a `>= 88.54 s` presenti nell'originale).
- **Provenienza / Licenza:** Sound On Sound editorial *Mix Rescue* (`https://www.soundonsound.com/techniques/mix-rescue-kirsty-cooke`, WAV editoriali educativi `44.1 kHz PCM_16`; uso privato di validazione autorizzato dall'utente, non ridistribuibile).
- **Nota same-track:** in *Mix Rescue*, `Remix.wav` è un remix/master dello stesso brano (`env_corr = 0.508`, `lag = -0.01 s`) che interviene anche su sub-bass/arrangiamento oltre alla riduzione asprezza. Per `track06` lo schema non richiede un `same_track_reference` vincolante (`required: false`).
- **Idoneità acustica:** surplus energetico reale in banda **`3–7 kHz`** di **`+10.494 dB`** (`-10.302 dB` mix vs `-20.796 dB` Remix) e **`erb_presence_surplus_db = +4.8732 dB`** (`erb_de_harsh_loss = 17.001`) misurati con `ERBSpectralLoss(sample_rate=48000.0)`.
- **Gap esplicito:** `track06_ref_altro_brano.wav` non popolato su scelta dell'utente (vedi Sezione 5).

### `track07` — Sub-bass & Limiter Stress (`H05`, `H03`)
- **Brano:** *SIMON_V - Cloudspotter* (taglio da `0:14`, `14.0–47.308 s`, `33.31 s`).
- **Provenienza / Licenza:** Demo mastering lossless Simon V (`https://www.simonv.com/mastering/`, `WAV PCM_16` nativi `44.1 kHz`; uso privato di validazione confermato esplicitamente dall'utente anche per taglio `>= 0:14`).
- **Allineamento same-track (48 kHz):** `env_corr = 0.9007`, `lag = +0.00 s`.
- **Idoneità acustica:** elevato contenuto sub-bass (`20–80 Hz = -2.216 dB rel`) e forte richiesta di guadagno rispetto al target master (`-14.911 -> -7.320 LUFS`, `+7.591 LU`, contro headroom pre-ceiling di `-0.5 - (-0.747) = +0.247 dB`), garantendo `limiter GR >> 0.5 dB` senza alterazioni artificiali.
- **Nota durata / Gap esplicito:** `33.31 s` è leggermente sotto l'intervallo indicativo `[35.0, 90.0]` a causa del taglio da `0:14` richiesto dall'utente, restando ampiamente sopra la soglia minima d'ascolto (`>= 20 s`). `track07_ref_altro_brano.wav` omesso su indicazione dell'utente.

### `track08` — Wide Stereo & Rischio Mono (`H06`)
- **Brano:** *Ainst - Drive* (`Unmastered` vs `Club Master`, `0.0–334.679 s` con coda finale di silenzio di 1.92 s allineata a parità di lunghezza).
- **Provenienza / Licenza:** Archive.org (`https://archive.org/details/ainst-drive`, Beriga Recordings `ANST-BR-2023`, `WAVEX PCM_24` nativi `44.1 kHz`; `licenseurl=None` sul record Archive.org — uso strettamente privato di valutazione autorizzato dall'utente, non ridistribuibile).
- **Allineamento same-track (48 kHz):** `env_corr = 0.9640`, `lag = +0.00 s`.
- **Idoneità acustica:** **`stereo_correlation = +0.3141`** (ampiamente sotto la soglia ideale `<= 0.55` e massima `<= 0.65`) e `mono_sum_drop = -1.826 dB`.

### `track09` — Seconda Traccia Wide Stereo Indipendente (`H06`)
- **Brano:** *Audio Animals - Film Library Mastering Orchestral 1* (`33.098 s` completi).
- **Provenienza / Licenza:** Pagina demo pubblica Audio Animals (`https://www.audioanimals.co.uk/pages/before-after-audio-examples`, `WAV PCM_16` nativi `44.1 kHz`; uso privato di valutazione, non ridistribuibile).
- **Allineamento same-track (48 kHz):** `env_corr = 0.7361`, `lag = +0.00 s`.
- **Idoneità acustica:** **`stereo_correlation = +0.4209`** (sotto soglia ideale `<= 0.55`) e `mono_sum_drop = -1.485 dB`, `PLR = 11.557 dB`.

---

## 4. Verifica `H06 — Seconda traccia wide-stereo`

- Con l'inclusione di `track09` (*Audio Animals - Orchestral 1*, brano indipendente rispetto ad *Ainst - Drive* `track08`), il set locale dispone di **2 tracce wide-stereo indipendenti**, entrambe pienamente conformi alla soglia ideale `stereo_correlation <= 0.55` (`+0.3141` e `+0.4209`).
- **Nota di governance:** la durata di `track09` (`33.10 s`) è di poco inferiore all'intervallo indicativo `[35.0, 90.0]` del template (pur superando i `>= 20–25 s` di ascolto). In ogni caso, come previsto dalle regole Sprint 4, in fase di intake **non si dichiara alcuna promozione di `V3c` verso `V4`**: la decisione sull'uscita di `H06` dallo stato esplorativo resta subordinata alla validazione umana effettiva.

---

## 5. Elenco Esplicito di File Mancanti, Scelte Differite o Scostamenti dallo Schema

1. **`ref_altro_brano` mancanti (scelta esplicita utente):**
   - `human_validation/private/input/track05_ref_altro_brano.wav` *(marcato `required: true` nello schema per il regime `stylistic`)*
   - `human_validation/private/input/track06_ref_altro_brano.wav` *(marcato `required: true` nello schema per il regime `stylistic`; opzionale per utente)*
   - `human_validation/private/input/track07_ref_altro_brano.wav` *(marcato `required: true` nello schema per il regime `stylistic`; opzionale per utente)*
   - *(Opzionali non richiesti: `track04_ref_altro_brano.wav`, `track08_ref_altro_brano.wav`, `track09_ref_altro_brano.wav`.)*  
   Se si vorrà eseguire il regime `stylistic` esterno su `track05`/`track06`/`track07`, questi file andranno aggiunti dall'utente (eventuali candidati CC lossless già verificati restano comunque documentati negli appunti di ricerca temporanei).
2. **Conversione `44.1 kHz -> 48 kHz`:** nessuno dei sorgenti lossless trovati era nativo a `48 kHz`; la conversione offline è stata autorizzata dall'utente ed è interamente tracciata campo per campo nel manifest.
3. **Subtypes `FLOAT` su 4 file (`track06_ref`, `track07_ref`, `track08_mix`, `track08_ref`):** usati per preservare fedelmente picchi sopra `0 dBFS` derivanti dai file master/mix sorgenti senza introdurre clipping da quantizzazione.
4. **Durate sotto `[35.0, 90.0]` su 2 slot (`track07`: `33.31 s` per taglio utente `>= 0:14`; `track09`: `33.10 s` durata integrale della demo):** segnalate per trasparenza; entrambe superano il minimo operativo di `>= 20 s`.
5. **Licenze archiviali / demo studio:** `track08` (`archive.org`, `licenseurl=None`), `track04`/`track05` (XARC showcase), `track06` (SOS editorial), `track07` (Simon V), `track09` (Audio Animals) sono destinati **esclusivamente alla valutazione privata locale** e **non** alla ridistribuzione.
