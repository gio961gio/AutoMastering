# SPRINT 3 — V3B BLIND AUDIO MANIFEST (SHA-256 & INTEGRITÀ WAV)

**Scopo:** Verifica indipendente da disco dei 12 file WAV anonimizzati per l'ascolto cieco (48 kHz, stereo, 24-bit PCM `PCM_24`).  
**Nota di Blinding:** Questo documento non contiene la chiave di decodifica (`SPRINT3_V3B_BLIND_MAPPING.json`) e può essere incluso nel pacchetto condiviso con i valutatori.

## 1. Inventario File WAV (`docs/sprint3/blind_v3b_wavs/`)

| File | Sessione | Etichetta | SR (Hz) | Ch | Formato | Frames | Durata (s) | Bytes | Integrated LUFS | Sample Peak (dBFS) | True Peak 4x (dBTP) | SHA-256 |
|:---|:---|:---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|:---|
| `track01_style_variant_A.wav` | `track01_style` | `variant_A` | 48000 | 2 | `PCM_24` | 3066985 | 63.8955 | 18401954 | -17.8344 | -2.4192 | -2.4192 | `9a7b51ed522b1410f63308d2acd222f071adafe44aa6d8fd7eee4cae3ff7276d` |
| `track01_style_variant_B.wav` | `track01_style` | `variant_B` | 48000 | 2 | `PCM_24` | 3066985 | 63.8955 | 18401954 | -17.8344 | -2.4192 | -2.4192 | `9a7b51ed522b1410f63308d2acd222f071adafe44aa6d8fd7eee4cae3ff7276d` |
| `track01_variant_A.wav` | `track01` | `variant_A` | 48000 | 2 | `PCM_24` | 3066985 | 63.8955 | 18401954 | -10.5121 | -0.3000 | +0.2647 | `1560b645c10b381cc37c383ef2b38a08e292c25774a17a3bac1f13e64dd40dd6` |
| `track01_variant_B.wav` | `track01` | `variant_B` | 48000 | 2 | `PCM_24` | 3066985 | 63.8955 | 18401954 | -10.5119 | -0.3000 | +0.2996 | `d12eed700fb77897b848b0f87b02c4d5940f1b2603d1f2975ab5765b342ee684` |
| `track02_style_variant_A.wav` | `track02_style` | `variant_A` | 48000 | 2 | `PCM_24` | 3430609 | 71.4710 | 20583698 | -18.0522 | -0.3000 | -0.0669 | `afdad3a698b3e475af256872c16133b775754ed4e7a6e9e8876a6c6fac13bb42` |
| `track02_style_variant_B.wav` | `track02_style` | `variant_B` | 48000 | 2 | `PCM_24` | 3430609 | 71.4710 | 20583698 | -18.0522 | -0.3000 | -0.0669 | `afdad3a698b3e475af256872c16133b775754ed4e7a6e9e8876a6c6fac13bb42` |
| `track02_variant_A.wav` | `track02` | `variant_A` | 48000 | 2 | `PCM_24` | 3430609 | 71.4710 | 20583698 | -18.0399 | -0.3000 | -0.0713 | `f9f9587230323c28303301db3593c097210d30c91704b7c26a4bcc7ebd4d71ab` |
| `track02_variant_B.wav` | `track02` | `variant_B` | 48000 | 2 | `PCM_24` | 3430609 | 71.4710 | 20583698 | -18.0399 | -0.3000 | -0.0713 | `f9f9587230323c28303301db3593c097210d30c91704b7c26a4bcc7ebd4d71ab` |
| `track03_style_variant_A.wav` | `track03_style` | `variant_A` | 48000 | 2 | `PCM_24` | 3445656 | 71.7845 | 20673980 | -13.5180 | -0.3000 | +0.0635 | `d903b9d53a1f9e4d559dbcede56f3454278b9d33d19b791a3b16033df6b8854c` |
| `track03_style_variant_B.wav` | `track03_style` | `variant_B` | 48000 | 2 | `PCM_24` | 3445656 | 71.7845 | 20673980 | -13.5180 | -0.3000 | +0.0635 | `d903b9d53a1f9e4d559dbcede56f3454278b9d33d19b791a3b16033df6b8854c` |
| `track03_variant_A.wav` | `track03` | `variant_A` | 48000 | 2 | `PCM_24` | 3445656 | 71.7845 | 20673980 | -13.5017 | -0.3000 | -0.0868 | `99fe7c6dbbd6675217d5b98db28a4a4795065d5bab8497b3ada63f90e4faeb9e` |
| `track03_variant_B.wav` | `track03` | `variant_B` | 48000 | 2 | `PCM_24` | 3445656 | 71.7845 | 20673980 | -13.5017 | -0.3000 | -0.0868 | `99fe7c6dbbd6675217d5b98db28a4a4795065d5bab8497b3ada63f90e4faeb9e` |

## 2. Parità di Livello per Coppia (`variant_A` vs `variant_B` senza Gain Aggiuntivo)

| Sessione | LUFS `variant_A` | LUFS `variant_B` | $\Delta\text{LUFS}_{A-B}$ (LU) | $|\Delta\text{LUFS}|$ (LU) | Bit-Identici (SHA-256 uguale) |
|:---|---:|---:|---:|---:|:---:|
| **`track01`** | -10.5121 | -10.5119 | -0.0001 | 0.0001 | No |
| **`track01_style`** | -17.8344 | -17.8344 | +0.0000 | 0.0000 | Sì |
| **`track02`** | -18.0399 | -18.0399 | +0.0000 | 0.0000 | Sì |
| **`track02_style`** | -18.0522 | -18.0522 | +0.0000 | 0.0000 | Sì |
| **`track03`** | -13.5017 | -13.5017 | +0.0000 | 0.0000 | Sì |
| **`track03_style`** | -13.5180 | -13.5180 | +0.0000 | 0.0000 | Sì |

- **Massimo scarto assoluto di coppia $|\Delta\text{LUFS}|$:** `0.0001 LU` (sotto la soglia di tolleranza `0.0100 LU`, senza applicazione di gain di compensazione esterno).
- **Archivio Condivisibile Evaluators (senza chiave segreta):** `human_validation/private/sprint3_v3b_blind_evaluator_pack.zip`.
- **Contenuto dell'archivio `.zip`:** esclusivamente i 12 file `.wav` in `blind_v3b_wavs/`, `SPRINT3_V3B_BLIND_LISTENING_WORKSHEET.md` e `SPRINT3_V3B_BLIND_AUDIO_MANIFEST.md` (strictly NO `SPRINT3_V3B_BLIND_MAPPING.json`).
