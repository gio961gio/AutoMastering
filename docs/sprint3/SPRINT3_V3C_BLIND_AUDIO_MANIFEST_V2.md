# SPRINT 3 — V3C PILOT BLIND AUDIO MANIFEST V2 (TRUE PEAK COMPLIANT)

**Scopo:** Pacchetto cieco V2 (`48 kHz`, stereo, `PCM_24`) con nomi opachi (`PAIR01`–`PAIR03`) e conformità verificata **True Peak 4x <= -0.35 dBTP** su tutti i file, ottenuta tramite sola attenuazione lineare identica per coppia sui render esistenti.
**Nota di Blinding:** Questo manifest pubblico contiene esclusivamente metriche tecniche di verifica ed è privo di informazioni identificative delle coppie e delle varianti.

## 1. Inventario File WAV (`blind_v3c_pilot_wavs_v2/`)

| File | Coppia | Etichetta | SR / Ch / Formato | Frames | Attenuazione Comune (dB) | TP 4x Prima (dBTP) | **TP 4x Dopo (dBTP)** | LUFS Dopo (BS.1770) | SHA-256 |
|:---|:---:|:---:|:---:|---:|---:|---:|---:|---:|:---|
| `PAIR01_A.wav` | `PAIR01` | `A` | 48000 Hz / 2ch / `PCM_24` | 3445656 | -0.2663 dB | -0.0838 | **-0.3501** | -13.7681 | `1cd5ee45ed2432d13b733c3936d1ec69bac7256937abf2decda5d47ec8b94263` |
| `PAIR01_B.wav` | `PAIR01` | `B` | 48000 Hz / 2ch / `PCM_24` | 3445656 | -0.2663 dB | -0.0868 | **-0.3531** | -13.7680 | `a0b8ee86121184fc00486e4460a998d1c8c790c34c446d9e9dadd03e1b2c6203` |
| `PAIR02_A.wav` | `PAIR02` | `A` | 48000 Hz / 2ch / `PCM_24` | 3066985 | -0.6575 dB | +0.3074 | **-0.3501** | -11.1678 | `4727e616f2023accbcca3c7d35f030c28083006ae7847b8a727713a55afe3e81` |
| `PAIR02_B.wav` | `PAIR02` | `B` | 48000 Hz / 2ch / `PCM_24` | 3066985 | -0.6575 dB | +0.2647 | **-0.3928** | -11.1696 | `12d4467b8a6a7f9a64856c0aee70c36abc50a049161f4aed32ac6894273dbb2f` |
| `PAIR03_A.wav` | `PAIR03` | `A` | 48000 Hz / 2ch / `PCM_24` | 3430609 | -0.2787 dB | -0.0713 | **-0.3500** | -18.3187 | `5217d07d53eb5563fc90fb71494fcb8601f89184c1b3c03d5c8ca4f5adfda8c2` |
| `PAIR03_B.wav` | `PAIR03` | `B` | 48000 Hz / 2ch / `PCM_24` | 3430609 | -0.2787 dB | -0.0713 | **-0.3500** | -18.3186 | `894f22b157deba94132c7c7653c589c7fb8289f4ea035869da06969f57ebccda` |

## 2. Riepilogo per Coppia & Scarto LUFS (`<= 0.01 LU`)

| Coppia | File A | File B | Attenuazione Iniziale -> Finale (dB) | Guadagno Lineare Comune | LUFS A | LUFS B | `|ΔLUFS|` (LU) | Max TP 4x Dopo (dBTP) |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|
| `PAIR01` | `PAIR01_A.wav` | `PAIR01_B.wav` | 0.2663 -> 0.2663 | 0.96980630 | -13.7681 | -13.7680 | 0.000077 | -0.3501 |
| `PAIR02` | `PAIR02_A.wav` | `PAIR02_B.wav` | 0.6575 -> 0.6575 | 0.92709662 | -11.1678 | -11.1696 | 0.001739 | -0.3501 |
| `PAIR03` | `PAIR03_A.wav` | `PAIR03_B.wav` | 0.2787 -> 0.2787 | 0.96842279 | -18.3187 | -18.3186 | 0.000038 | -0.3500 |

- **Massimo True Peak 4x (tutti i 6 WAV V2):** `-0.3500 dBTP` (requisito `<= -0.35 dBTP`).
- **Massimo scarto di coppia `|ΔLUFS|`:** `0.001739 LU` (target controllo `<= 0.01 LU`, nessuna normalizzazione LUFS indipendente).
- **Contenuto archivio evaluators V2:** esclusivamente i 6 WAV in `blind_v3c_pilot_wavs_v2/`, `SPRINT3_V3C_BLIND_LISTENING_WORKSHEET_V2.md` e `SPRINT3_V3C_BLIND_AUDIO_MANIFEST_V2.md`.
