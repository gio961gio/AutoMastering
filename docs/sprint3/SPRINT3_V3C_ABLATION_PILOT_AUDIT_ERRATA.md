# SPRINT 3 — V3C PILOT ABLATION AUDIT ERRATA

**Documento di Errata Separata (Non Distruttiva)**  
**Repository:** `gio961gio/AutoMastering` | **Branch:** `human-validation-package`  
**Baseline Fase 1:** `a98c2d67b87d798d449a1018157d50d2a2e7be20`  
**HEAD Fase 2 (Fonte Primaria JSON):** `5709c3f402e434855a28c51e3de4c4b8ae532473`

> **Ambito e Immutabilità:** Questo documento corregge alcune sintesi numeriche, interpretazioni e descrizioni dei test riferite al pilot `V3c vs V0p` (`same_track` only: `track01`, `track02`, `track03`), lasciando **intatti e non modificati** [`SPRINT3_V3C_ABLATION_PILOT_RESULTS.json`](SPRINT3_V3C_ABLATION_PILOT_RESULTS.json), [`SPRINT3_V3C_ABLATION_PILOT_REPORT.md`](SPRINT3_V3C_ABLATION_PILOT_REPORT.md), gli artefatti V3c Preflight V1/V2 e il manifest/mapping del pacchetto V1.

---

## 1. Dati Grezzi Estratti dal JSON Fonte Primaria (`HEAD 5709c3f`)

Tutti i valori seguenti provengono direttamente da [`SPRINT3_V3C_ABLATION_PILOT_RESULTS.json`](SPRINT3_V3C_ABLATION_PILOT_RESULTS.json) committato a `5709c3f402e434855a28c51e3de4c4b8ae532473` (nessun ricalcolo né ri-ottimizzazione):

### 1.1 Rettifica Sintesi Loss (`V0p` IIR vs `V3c` Weighted)

| Traccia | `V0p` loss (`control_v0p` $\to$ `treatment_v3c`) | `V3c` weighted (`v3c_eval_on_v0p` $\to$ `treatment_v3c`) | Delta `V3c` (%) |
|:---|---:|---:|---:|
| **`track01`** | `2.8223822117` $\to$ `2.8199706078` | `0.0024833711` $\to$ `0.0023057233` | `-7.1535%` |
| **`track02`** | `10.3552169800` $\to$ `10.3553476334` | `0.0008066148` $\to$ `0.0007272358` | `-9.8410%` |
| **`track03`** | `6.2349562645` $\to$ `6.2349948883` | `0.0008046846` $\to$ `0.0007858747` | `-2.3375%` |

### 1.2 Parametri Fisici Imager (`w_low`, `w_mid`, `w_high`) e Spostamenti Multi-Gruppo

- **`track01`:**
  - Imager widths: `w_low` `1.042223` $\to$ `1.041514`; `w_mid` `1.111668` $\to$ `1.111352`; `w_high` `1.109783` $\to$ `1.109790`.
  - Crossover high (`crossover_f_high`): `2692.076172 Hz` (`V0p`) $\to$ `2642.102539 Hz` (`V3c`), spostamento **`-49.9736 Hz`** ($\approx -49.974\text{ Hz}$).
  - Norme $\|\Delta z\|_2$ per gruppo: `Crossover = 0.014397`, `Compressor = 0.002034`, `Imager = 0.001109`, `EQ = 0.000211`.
- **`track02`:**
  - Imager widths: `w_low` `1.023008` $\to$ `1.018621`; `w_mid` `1.111870` $\to$ `1.111687`; `w_high` `1.111224` $\to$ `1.111217`.
  - Crossover high (`crossover_f_high`): `3170.945801 Hz` (`V0p`) $\to$ `3170.960449 Hz` (`V3c`), spostamento `+0.0146 Hz`.
  - Norme $\|\Delta z\|_2$ per gruppo: `Imager = 0.006272`, `EQ = 0.000186`, `Compressor = 0.000058`, `Crossover = 0.000038`.
- **`track03`:**
  - Imager widths: `w_low` `1.042530` $\to$ `1.042266`; `w_mid` `1.111665` $\to$ `1.111597`; `w_high` `1.110534` $\to$ `1.110528`.
  - Crossover high (`crossover_f_high`): `3467.580811 Hz` (`V0p`) $\to$ `3463.158447 Hz` (`V3c`), spostamento `-4.4224 Hz`.
  - Norme $\|\Delta z\|_2$ per gruppo: `Crossover = 0.002035`, `Imager = 0.000389`, `EQ = 0.000154`, `Compressor = 0.000104`.

### 1.3 Scarto Massimo di Coppia LUFS (Pacchetto V1)

- Nel file [`SPRINT3_V3C_BLIND_AUDIO_MANIFEST.json`](SPRINT3_V3C_BLIND_AUDIO_MANIFEST.json) il massimo scarto assoluto di coppia non arrotondato è **`0.001739 LU`** (riportato arrotondato a 4 decimali come `0.0017 LU` nel report Markdown).

### 1.4 Errata Descrizione Gate 9 (`tests/test_sprint3_v3c_gate.py`)

- Alla sezione 9.7 di `test_gate_9_optimizer_opt_in_v3c_wiring` ([`tests/test_sprint3_v3c_gate.py`](../../tests/test_sprint3_v3c_gate.py), righe 783–786), il test verifica la non-moltiplicazione su `K = 4` chunk utilizzando **`w_low = 1.08`** (`z_mono_w[35]`) e contributo atteso **`0.0016`** (`0.25 * (1.08 - 1.00)^2`, confrontato con `4 * 0.0016 = 0.0064`), **non** `w_low = 1.15` né `0.005625`.

---

## 2. Interpretazioni Quantitative (Esclusivamente Oggettive / Proxy)

1. **L'effetto di `V3c` NON è confinato all'Imager:** poiché `V3cStereoCompatibilityLoss` propaga gradienti attraverso i termini spettrali/stereo (`loss_band_corr`, `loss_band_smr`) lungo l'intera catena DDSP, l'ottimizzatore modifica sensibilmente anche la frequenza di crossover `crossover_f_high` (`-49.974 Hz` su `track01`, `-4.422 Hz` su `track03`) e marginalmente compressore ed EQ, oltre alle larghezze M/S dell'Imager (su `track02` prevale invece `Imager` con `||Delta z||_2 = 0.006272`).
2. **Stato True Peak e Necessità del Pacchetto Blind V2:** i 6 WAV del pacchetto V1 violano il guardrail `-0.30 dBTP` già nel braccio `Control V0p` (`+0.2647` / `+0.3074` su `track01`, `-0.0713` su `track02`, `-0.0868` / `-0.0838` su `track03`). Per la valutazione percettiva si introduce un **Pacchetto Blind V2** applicando una singola attenuazione lineare **identica** ai due membri di ciascuna coppia sui WAV V1 già renderizzati (zero ricalcolo DSP) fino a portare ogni WAV a `<= -0.35 dBTP` su True Peak 4x.
3. **Stato del Blinding sul Repository Storico:** i file `SPRINT3_V3C_BLIND_MAPPING.json` e `SPRINT3_V3C_ABLATION_PILOT_RESULTS.json` (con i seed di sessione da cui erano derivati i nomi `track0X_variant_A/B.wav`) sono già presenti nella storia Git a `5709c3f`; pertanto il pacchetto V1 non è considerato cieco rispetto a chi abbia accesso al repository. Il Pacchetto Blind V2 ripristina l'effettiva cecità mediante nomi opachi (`PAIR01_A.wav`, ..., `PAIR03_B.wav`), nuova randomizzazione indipendente dai seed pubblicati e salvataggio della chiave V2 esclusivamente nel percorso locale ignorato da Git (`human_validation/private/`).

---

## 3. Risultati d'Ascolto Umano

- **Stato attuale:** **NON DISPONIBILI / NON ANCORA ESEGUITI.**
- Nessuna preferenza percettiva o valutazione d'ascolto umano è stata finora raccolta. Qualsiasi decisione su eventuali iterazioni successive (es. `V4`) resta subordinata al completamento della validazione umana sul Pacchetto Blind V2.
