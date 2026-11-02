# AutoMastering — Human Validation Package CLI Utilities

Questo pacchetto (`scripts/human_validation/`) fornisce la strumentazione CLI offline per preparare, verificare e documentare le sessioni di ascolto umano degli Sprint 1 e Sprint 2 senza modificare il motore DSP C++20/PyTorch congelato al commit `5940e81f12741e2276f7a446adfb6e12f44574fe`.

Tutti i file audio privati dell'utente devono risiedere nella cartella locale `human_validation/private/` (ignorata da Git).

## Moduli disponibili

| Modulo | Funzione |
|---|---|
| `validate_inputs.py` | Validazione rigorosa dei file WAV (48 kHz obbligatori senza resampling silenzioso, mono/stereo, NaN/Inf, calcolo SHA-256 e protezione da sovrascrittura). |
| `offline_renderer.py` | Rendering offline tramite nuova istanza `py_core_dsp.MasteringChain` con reset, zero-padding di coda (`>= 159` campioni) per il tail flush completo e compensazione singola della latenza (`[159 : 159 + T]`). Include anche il trim diagnostico statico (`-6.0 dBTP`). |
| `analyze_exports.py` | Misura oggettiva standardizzata: Sample Peak (`dBFS`), True Peak (`dBTP` 4× oversampling), Integrated Loudness ITU-R BS.1770-5 (con tracciamento esplicito del fallback se `< 400 ms`), RMS, PLR, correlazione stereo e rapporto Mid/Side. |
| `make_level_matched.py` | Generazione di copie per il solo ascolto con guadagno lineare **puramente statico** per file per pareggiare l'Integrated Loudness del gruppo rispettando il tetto di sicurezza True Peak `<= -1.0 dBTP` (con raffinamento iterativo automatico e verifica rigorosa pre/post export). |
| `make_blind_set.py` | Generazione di set d'ascolto ciechi (`blind_A.wav`, `blind_B.wav`, ...) con permutazione deterministica da `--seed`, privi di metadati identificativi, con cartella di consegna separata (`blind_delivery/`) e file chiave separato (`keys/*_blind_key.json`) per impedire la rivelazione accidentale. |
| `render_sprint1.py` | **Esperimento A (Sprint 1):** confronto sullo stesso brano tra **A — True Bypass**, **B — Legacy Initialization Baseline** (da commit `86ae20f08dc7fbebc7f67af064a15b648a492fc7`) e **C — Corrected Neutral** (`THETA_NEUTRAL`). |
| `render_sprint2.py` | **Esperimento B (Sprint 2):** ottimizzazione multi-chunk con **un solo preset per brano** (`same_track` o `stylistic`), export full-track con tail flush, ed estrazione di almeno 3 sezioni d'ascolto (10–20 s: intro/strofa, ritornello/drop, bridge/transienti) in versione program-level, level-matched e blind. |

## Esempi rapidi (PowerShell e POSIX)

Consulta la guida completa passo-passo in [`docs/human_validation/HUMAN_VALIDATION_PROTOCOL.md`](../../docs/human_validation/HUMAN_VALIDATION_PROTOCOL.md).
