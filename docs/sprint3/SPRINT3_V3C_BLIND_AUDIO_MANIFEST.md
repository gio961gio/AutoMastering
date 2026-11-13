# SPRINT 3 — V3C PILOT BLIND AUDIO MANIFEST (SHA-256 & INTEGRITÀ WAV)

**Scopo:** Verifica indipendente da disco dei 6 file WAV anonimizzati per l'ascolto cieco pilota `same_track` (`track01`, `track02`, `track03` — `48 kHz`, stereo, `PCM_24`).  
**Nota di Blinding:** Questo documento non contiene la chiave di decodifica (`SPRINT3_V3C_BLIND_MAPPING.json`) ed è incluso nel pacchetto valutatori.

## 1. Inventario File WAV (`docs/sprint3/blind_v3c_pilot_wavs/`)

| File | Sessione | Etichetta | SR (Hz) | Ch | Formato | Frames | Durata (s) | Bytes | Integrated LUFS | Sample Peak (dBFS) | True Peak 4x (dBTP) | SHA-256 |
|:---|:---|:---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|:---|
| `track01_variant_A.wav` | `track01` | `variant_A` | 48000 | 2 | `PCM_24` | 3066985 | 63.8955 | 18401954 | -10.5121 | -0.3000 | +0.2647 | `1560b645c10b381cc37c383ef2b38a08e292c25774a17a3bac1f13e64dd40dd6` |
| `track01_variant_B.wav` | `track01` | `variant_B` | 48000 | 2 | `PCM_24` | 3066985 | 63.8955 | 18401954 | -10.5103 | -0.3000 | +0.3074 | `d16c6f622b292f48b692364fe2311507cdbacb0db907b5e765d1da3960820bbb` |
| `track02_variant_A.wav` | `track02` | `variant_A` | 48000 | 2 | `PCM_24` | 3430609 | 71.4710 | 20583698 | -18.0400 | -0.3000 | -0.0713 | `a7585d8ceed44dd6ba84d56bfaba8614a36efca7f076b8b65fa4e29949bee4c8` |
| `track02_variant_B.wav` | `track02` | `variant_B` | 48000 | 2 | `PCM_24` | 3430609 | 71.4710 | 20583698 | -18.0399 | -0.3000 | -0.0713 | `f9f9587230323c28303301db3593c097210d30c91704b7c26a4bcc7ebd4d71ab` |
| `track03_variant_A.wav` | `track03` | `variant_A` | 48000 | 2 | `PCM_24` | 3445656 | 71.7845 | 20673980 | -13.5018 | -0.3000 | -0.0838 | `a394b0645f39b03162f580919d827a1d74f0ad9fa46e0f4997cd007ce806ee1a` |
| `track03_variant_B.wav` | `track03` | `variant_B` | 48000 | 2 | `PCM_24` | 3445656 | 71.7845 | 20673980 | -13.5017 | -0.3000 | -0.0868 | `99fe7c6dbbd6675217d5b98db28a4a4795065d5bab8497b3ada63f90e4faeb9e` |

## 2. Parità di Livello per Coppia (`variant_A` vs `variant_B` — Nessun Gain Aggiuntivo)

| Sessione | LUFS `variant_A` | LUFS `variant_B` | $\Delta\text{LUFS}_{A-B}$ (LU) | $|\Delta\text{LUFS}|$ (LU) | SHA-256 Uguale nella Coppia |
|:---|---:|---:|---:|---:|:---:|
| **`track01`** | -10.5121 | -10.5103 | -0.0017 | 0.0017 | No |
| **`track02`** | -18.0400 | -18.0399 | -0.0000 | 0.0000 | No |
| **`track03`** | -13.5018 | -13.5017 | -0.0001 | 0.0001 | No |

- **Massimo scarto assoluto di coppia $|\Delta\text{LUFS}|$:** `0.0017 LU` (render naturali senza compensazione esterna).
- **Archivio Condivisibile Evaluators (privo di mapping):** `docs/sprint3/blind_v3c_pilot_evaluator_pack.zip` (generato su disco locale, escluso da Git via `.gitignore` per limite dimensione file > 100 MB).
- **Contenuto dell'archivio `.zip`:** esclusivamente i 6 `.wav` in `blind_v3c_pilot_wavs/`, `SPRINT3_V3C_BLIND_LISTENING_WORKSHEET.md` e `SPRINT3_V3C_BLIND_AUDIO_MANIFEST.md`.
