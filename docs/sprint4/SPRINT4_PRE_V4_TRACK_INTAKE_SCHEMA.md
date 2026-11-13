# SPRINT 4 — SCHEMA DI DATA INTAKE & VALIDAZIONE PRE-V4 (`track04..track08`)

- **Repository:** `gio961gio/AutoMastering`
- **Branch:** `human-validation-package`
- **Manifest JSON vuoto associato:** [`SPRINT4_PRE_V4_TRACK_INTAKE_MANIFEST.json`](./SPRINT4_PRE_V4_TRACK_INTAKE_MANIFEST.json) (`schema_version: 1.0.0`, `status: EMPTY_TEMPLATE_PENDING_AUTHORIZED_LOCAL_AUDIO`)
- **Protocollo di riferimento:** [`SPRINT4_PRE_V4_HUMAN_LISTENING_PROTOCOL.md`](./SPRINT4_PRE_V4_HUMAN_LISTENING_PROTOCOL.md)
- **Stato corrente:** Template vuoto predisposto per l'acquisizione locale; **nessun file audio scaricato**, **nessuna ottimizzazione Adam eseguita**, **nessun blind pack generato**.

---

## 1. Politica di Confinamento, Provenienza e Licenze

1. **Zero Audio su Git:** ogni WAV (`mix`, `ref`, `ref_altro_brano`) risiede esclusivamente in locale sotto la directory gitignored `human_validation/private/input/`. Solo i metadati testuali e gli hash SHA-256 vengono registrati nei manifest JSON sotto `docs/sprint4/`.
2. **Origine & Licenze Autorizzate (`provenance_and_licensing`):**
   - Sono ammessi esclusivamente: mix propri/originali dell'autore (`artist_owned_mix`), tracce multitraccia didattiche/ricerca con licenza esplicita (es. `CC-BY`, `OpenMultitrack`, `MedleyDB` o stems commerciali con licenza studio autorizzata), oppure WAV forniti con consenso documentato (`usage_authorization_confirmed: true`).
   - **Severamente vietati (`commercial_stream_rips_forbidden: true`):** rip da piattaforme streaming commerciali (YouTube/Spotify/Tidal/Apple Music), trascodifiche lossy mascherate da WAV (`.mp3 -> .wav`) o materiale privo di autorizzazione documentata.
3. **Requisiti Formato Tecnico (`load_and_validate_wav` contratto `48 kHz`):**
   - **Sample Rate:** tassativamente **`48000 Hz`** nativo (nessun resampling silenzioso consentito; se un master nativo è a `44.1`/`96 kHz`, documentare l'eventuale conversione offline ad alta qualità prima dell'intake).
   - **Canali:** stereo **`2` ch** (`--no-mono` raccomandato per le nuove tracce `track04..track08`).
   - **Codifica:** WAV `PCM_24`, `FLOAT` (32-bit float) o `DOUBLE` (64-bit); durata consigliata **`35–90 s`** completi a clip con sezione discriminante d'ascolto `>= 20 s`.

---

## 2. Specifica dello Schema JSON (`SPRINT4_PRE_V4_TRACK_INTAKE_MANIFEST.json`)

Ogni voce in `intake_slots[]` (`track04`..`track08`) richiede la compilazione esatta dei seguenti blocchi:

| Campo | Tipo | Obbligo | Descrizione |
|:---|:---|:---|:---|
| `track_id` | `string` | Sì | `track04` .. `track08` |
| `status` | `enum` | Sì | `MISSING_AWAITING_USER_SOURCE` -> `VALIDATED_LOCAL_READY` |
| `acoustic_profile` | `string` | Sì | Classe materiale e stress acustico target |
| `target_hypotheses` | `string[]` | Sì | Sottoinsieme di `["H01".."H06"]` coperto |
| `intended_regimes` | `string[]` | Sì | `["same_track"]`, `["stylistic"]` o entrambi |
| `expected_acoustic_targets` | `object` | Sì | Soglie minime attese (es. `plr_min_db`, `expected_limiter_gr_mean_gt_db`, `stereo_correlation_max`) |
| `source_mix` | `AudioSlot` | **Sempre Sì** | Mix non masterizzato (`local_relative_path`, `sha256`, `sample_rate_hz`, `channels`, `duration_sec`, `bit_depth_or_subtype`, `integrated_lufs_bs1770`, `sample_peak_dbfs`, `true_peak_4x_dbtp`, `plr_db`, `stereo_correlation`) |
| `same_track_reference` | `AudioSlot` | Condizionale | Master ufficiale dello stesso brano (`required: true` per `track04, 05, 07, 08`; `false` per `track06`) |
| `stylistic_external_reference` | `AudioSlot` | Condizionale | Master commerciale/studio di riferimento esterno del genere (`required: true` per `track05, 06, 07`; `false` opz. per `track04, 08`) |
| `candidate_listening_section` | `object` | Sì | Finestra `label`, `start_sec`, `end_sec`, `duration_sec` (`20–30 s`) centrata sull'evento critico per i trial d'ascolto |
| `provenance_and_licensing` | `object` | Sì | `source_type`, `license`, `usage_authorization_confirmed`, `authorized_by`, `notes` |

---

## 3. Workflow CLI Standardizzato per Validazione Ingestione (Sola Lettura)

Una volta posizionati i file locali in `human_validation/private/input/`, eseguire i due tool esistenti del repository dalla root per estrarre in modo deterministico hash SHA-256 e metriche acustiche BS.1770 / True Peak 4×:

```powershell
# 1. Validazione integrità WAV (48 kHz nativo, stereo, no NaN/Inf, SHA-256)
python -m scripts.human_validation.validate_inputs `
  human_validation/private/input/track04_mix.wav `
  human_validation/private/input/track04_ref.wav `
  --no-mono

# 2. Estrazione metriche oggettive per compilare AudioSlot nel manifest JSON
python -m scripts.human_validation.analyze_exports `
  human_validation/private/input/track04_mix.wav `
  human_validation/private/input/track04_ref.wav
```

Campi restituiti da copiare nello slot manifest:
- `validate_inputs`: `sha256`, `sample_rate -> sample_rate_hz`, `channels`, `duration_sec`, `subtype -> bit_depth_or_subtype`.
- `analyze_exports`: `integrated_lufs -> integrated_lufs_bs1770`, `sample_peak_dbfs`, `true_peak_dbtp -> true_peak_4x_dbtp`, `plr_db`, `stereo_correlation`.

---

## 4. Gap Report Corrente (`track01..track03` vs `track04..track08`)

### 4.1 Brani Già Presenti (`human_validation/private/input/` — gitignored)

| Track ID | Stato | Durata | Mix LUFS | Mix PLR | Stereo Corr. | Note Copertura |
|:---|:---|---:|---:|---:|---:|:---|
| `track01` | `PRESENT_IN_PRIVATE_INPUT` | `63.90 s` | `-14.22` | `18.12` | `0.795` | Copre `H02` (`_style` bright, `+0.964 LU` `V1b` overshoot), `H03`, `H05` |
| `track02` | `PRESENT_IN_PRIVATE_INPUT` | `71.47 s` | `-22.03` | `18.01` | `0.948` | Copre `H01`, `H03`, `NULL` anchor |
| `track03` | `PRESENT_IN_PRIVATE_INPUT` | `71.78 s` | `-17.86` | `14.45` | `0.715` | Copre `H01`, `H03`, `H04` |

### 4.2 Gap Mancanti Obbligatori Prima dell'Eventuale Generazione Blind Pack V4

| Slot Mancante | Stato | File Locali Attesi (`human_validation/private/input/`) | Criterio Acustico Chiave | Impatto sullo Screening (`§15–§16`) |
|:---|:---|:---|:---|:---|
| **`track04`** | `MISSING_AWAITING_USER_SOURCE` | `track04_mix.wav`, `track04_ref.wav` | Acustico / Dinamico, `PLR >= 11 dB` | Necessaria per `S03 (H01)`, `S11 (H03)`, `S15 (H04)` |
| **`track05`** | `MISSING_AWAITING_USER_SOURCE` | `track05_mix.wav`, `track05_ref.wav`, `track05_ref_altro_brano.wav` | Kick/snare secchi e transienti rapidi | Necessaria per `S12/S13 (H03)`, `S18 (H05)` |
| **`track06`** | `MISSING_AWAITING_USER_SOURCE` | `track06_mix.wav`, `track06_ref_altro_brano.wav` | Bright / sibilante `3–7 kHz` (`ERB surplus > 0`) | 2ª traccia indipendente obbligatoria per `H02` (`S06`) + `S16 (H04)`, `S24 (NULL)` |
| **`track07`** | `MISSING_AWAITING_USER_SOURCE` | `track07_mix.wav`, `track07_ref.wav`, `track07_ref_altro_brano.wav` | Sub-bass/wall-of-sound (`GR > 0.5 dB` su `-7..-9 LUFS`) | Necessaria per testare `H05 (V3b)` fuori deadzone (`S19/S20`) |
| **`track08`** | `MISSING_AWAITING_USER_SOURCE` | `track08_mix.wav`, `track08_ref.wav` | Wide stereo (`corr < 0.55` pref., `<= 0.65` max) | Copre `S21/S22 (H06)`: essendo `1` solo brano wide, **`H06` resta esplorativa** finché non esista un 2° brano wide |

**Azioni bloccate fino al completamento dell'intake (`track04..track08`) ed esplicita autorizzazione:** nessuna nuova ottimizzazione Adam, nessun render WAV blind di ascolto e nessuna modifica a `enable_v3c=False` o ai parametri `V0p..V3c`.
