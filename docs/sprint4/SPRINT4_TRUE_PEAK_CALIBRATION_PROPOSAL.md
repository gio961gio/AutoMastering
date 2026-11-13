# SPRINT 4 — PROPOSTA DI CALIBRAZIONE MARGINE INTERNO TRUE PEAK (FASE C — SOLO ANALISI, NESSUNA MODIFICA AL DSP)

- **Repository:** `gio961gio/AutoMastering`
- **Branch:** `human-validation-package`
- **Baseline DSP congelata:** `db4a7bf26db50ecf0401bcac77374eabd535050e` (4× polyphase FIR `-0.50 dBTP` + guard `PCM_24 <= -0.30 dBTP`)
- **Stato di questo documento:** **Analisi e proposta di calibrazione offline (FASE C).** Il ceiling interno vigente **`LIMITER_INTERNAL_CEILING_DBTP = -0.50 dBTP`** e i coefficienti FIR **`TP_FIR_PROTOTYPE_COEFFS`** restano **immutati** in attesa di eventuale autorizzazione esplicita del supervisore.

---

## 1. Perché la Reference Esterna Misura `-0.457 / -0.458 dBTP` con Ceiling Interno `-0.50 dBTP`

Durante la verifica sui 6 render congelati (`track01..track03`, `Control V0p` e `Treatment V3c`), il requisito contrattuale di consegna (**WAV `PCM_24` `<= -0.30 dBTP`**) è soddisfatto su `6/6` file con margine minimo **`+0.1567 dB`** (`-0.4567 dBTP` caso peggiore su `track01 V3c`).

Tuttavia, confrontando direttamente:
1. il **detector interno 4× WLS polyphase FIR + picco 1× allineato** (`_internal_detector_4x_peak_db`, il segnale visto dal loop di gain-reduction in `LookaheadLimiter.hpp` / `limiter.py`);
2. la **reference esterna di consegna** (`calculate_true_peak_dbtp`, basata su `torchaudio.functional.resample` Kaiser `lowpass_filter_width=64, rolloff=0.9475937, beta=14.769656`),

emerge uno scarto contenuto ma sistematico su segnali con forte contenuto transiente/alto:
- **`TP_FIR_PROTOTYPE_COEFFS`** è un filtro WLS Tipo II compatto da **48 tap (`12` tap per fase)** progettato con vincoli real-time di zero allocazioni e latenza totale di catena `159` campioni invariata (`aligned sample delay = 6`); rispetto a un sinc finestrato Kaiser lungo `64` tap per lobo, presenta un leggero roll-off/ripple di interpolazione locale (`[-0.014, +0.506] dB` tra `0–20 kHz`) che su alcuni transienti reali complessi porta `ext_kaiser` a leggere fino a **`+0.0430 dB`** più in alto rispetto a `det_4x` (`-0.4567 dBTP` vs `-0.4997 dBTP`).
- **Su segnali di test near-Nyquist (`44.1 / 48 / 96 kHz`):** grazie alla pendenza conservativa del nostro WLS in banda alta e all'allineamento `1x` a `6` campioni, alle frequenze `23.0–23.99 kHz` (`48 kHz`), `21.1–22.0 kHz` (`44.1 kHz`) e `46.0–47.98 kHz` (`96 kHz`) il detector interno `det_4x` limita a `-0.5000 dBTP`, mantenendo `ext_kaiser` a `-3.32 .. -3.53 dBTP` e perfino una FFT-sinc ideale senza roll-off (`rolloff -> 1.0`) a `-0.7088 .. -3.4656 dBTP`, **senza alcuna sottostima pericolosa degli ISP**.

---

## 2. Riesame dei 6 Render Congelati: `det_4x` vs `ext_kaiser` @ `-0.50 dBTP`

| Traccia | Arm | `det_4x` (dBTP) | `ext_kaiser` (dBTP) | `ext - det` (dB) | Margine vs `-0.30` (dB) | Scarto vs `-0.50` (dB) | LUFS (`1x->4x`) | `TP-PLR` (dB) |
|:---|:---:|---:|---:|---:|---:|---:|---:|---:|
| `track01` | `V0p` | `-0.4997` | `-0.4579` | `+0.0419` | `+0.1579` (**PASS**) | `-0.0421` *(sopra -0.50)* | `-10.5578` | `10.1000` |
| `track01` | `V3c` | `-0.4997` | `-0.4567` | `+0.0430` | `+0.1567` (**PASS**) | `-0.0433` *(sopra -0.50)* | `-10.5562` | `10.0995` |
| `track02` | `V0p` | `-0.5000` | `-0.5278` | `-0.0278` | `+0.2278` (**PASS**) | `+0.0278` *(sotto -0.50)* | `-18.0428` | `17.5150` |
| `track02` | `V3c` | `-0.5000` | `-0.5278` | `-0.0278` | `+0.2278` (**PASS**) | `+0.0278` *(sotto -0.50)* | `-18.0428` | `17.5150` |
| `track03` | `V0p` | `-0.4999` | `-0.4891` | `+0.0108` | `+0.1891` (**PASS**) | `-0.0109` *(sopra -0.50)* | `-13.5057` | `13.0166` |
| `track03` | `V3c` | `-0.4999` | `-0.4897` | `+0.0101` | `+0.1897` (**PASS**) | `-0.0103` *(sopra -0.50)* | `-13.5057` | `13.0160` |

**Lettura sintetica:**
- Tutte e `6` le sessioni rispettano ampiamente il requisito di consegna `<= -0.30 dBTP` (margine minimo `+0.1567 dB`).
- Se si desidera che **anche la metrica esterna `calculate_true_peak_dbtp` resti `<= -0.50 dBTP`** su tutte e `6` le tracce senza attivare alcun micro-trim post-export, occorre compensare il massimo scarto osservato (`+0.0430 dB` su `track01 V3c`).

---

## 3. Sweep Parametrico del Ceiling Interno (`-0.54`, `-0.55`, `-0.56`, `-0.58 dBTP`)

| Ceiling Interno | `track01 V0p` `ext_kaiser` (`dLUFS` / `dPLR_tp`) | `track01 V3c` `ext_kaiser` (`dLUFS` / `dPLR_tp`) | `track02 V0p/V3c` `ext_kaiser` (`dLUFS`) | `track03 V0p/V3c` `ext_kaiser` (`dLUFS`) | Tutti `<= -0.50`? |
|:---|:---|:---|:---|:---|:---:|
| **`-0.50 dBTP`** *(attuale)* | `-0.4579` (`0.0000` / `0.0000`) | `-0.4567` (`0.0000` / `0.0000`) | `-0.5278` (`0.0000`) | `-0.4891` / `-0.4897` (`0.0000`) | `2/6` *(ma `6/6 <= -0.30`)* |
| **`-0.54 dBTP`** | `-0.4979` (`-0.0074` / `-0.0326`) | `-0.4967` (`-0.0073` / `-0.0327`) | `-0.5678` (`-0.0006`) | `-0.5291` / `-0.5297` (`-0.0008`) | `4/6` |
| **`-0.55 dBTP`** *(soglia minima)* | `-0.5079` (`-0.0093` / `-0.0407`) | `-0.5067` (`-0.0092` / `-0.0408`) | `-0.5778` (`-0.0007`) | `-0.5391` / `-0.5397` (`-0.0010`) | **`6/6` PASS** |
| **`-0.56 dBTP`** *(raccomandato con headroom)* | `-0.5179` (`-0.0111` / `-0.0489`) | `-0.5167` (`-0.0111` / `-0.0489`) | `-0.5878` (`-0.0009`) | `-0.5491` / `-0.5497` (`-0.0012`) | **`6/6` PASS (+0.017 dB)** |
| **`-0.58 dBTP`** | `-0.5379` (`-0.0149` / `-0.0651`) | `-0.5367` (`-0.0148` / `-0.0652`) | `-0.6078` (`-0.0012`) | `-0.5691` / `-0.5697` (`-0.0017`) | **`6/6` PASS (+0.037 dB)** |

### Tabella Sweep Near-Nyquist (`44.1 / 48 / 96 kHz` — 12 fasi x 3 ampiezze x 0.25s, worst-case)

| Sample Rate | Frequenza | `det_4x` (dBTP) | `ext_kaiser` (dBTP) | `fft_ideal` (dBTP) | `SP` (dBFS) | `ext - det` (dB) |
|---:|---:|---:|---:|---:|---:|---:|
| `44100 Hz` | `21100.0 Hz` | `-0.5000` | `-3.3267` | `-3.3267` | `-3.3267` | `-2.8267` |
| `44100 Hz` | `21500.0 Hz` | `-0.5000` | `-3.4616` | `-3.4531` | `-3.4616` | `-2.9616` |
| `44100 Hz` | `21900.0 Hz` | `-0.5000` | `-3.5298` | `-3.3649` | `-3.5298` | `-3.0298` |
| `44100 Hz` | `22000.0 Hz` | `-0.5000` | `-3.5330` | `-1.8102` | `-3.5330` | `-3.0330` |
| `48000 Hz` | `20000.0 Hz` | `-0.5000` | `-1.1281` | `-1.1281` | `-1.1281` | `-0.6281` |
| `48000 Hz` | `21500.0 Hz` | `-0.5000` | `-2.4658` | `-2.4658` | `-2.4658` | `-1.9658` |
| `48000 Hz` | `22500.0 Hz` | `-0.5000` | `-3.0763` | `-3.0763` | `-3.0763` | `-2.5763` |
| `48000 Hz` | `23000.0 Hz` | `-0.5000` | `-3.3257` | `-3.3258` | `-3.3257` | `-2.8257` |
| `48000 Hz` | `23500.0 Hz` | `-0.5000` | `-3.4821` | `-3.4655` | `-3.4821` | `-2.9821` |
| `48000 Hz` | `23900.0 Hz` | `-0.5000` | `-3.5321` | `-2.3768` | `-3.5321` | `-3.0321` |
| `48000 Hz` | `23990.0 Hz` | `-0.5000` | `-3.5344` | `-0.7175` | `-3.5344` | `-3.0344` |
| `96000 Hz` | `46000.0 Hz` | `-0.5000` | `-3.3257` | `-3.3257` | `-3.3257` | `-2.8257` |
| `96000 Hz` | `47000.0 Hz` | `-0.5000` | `-3.4821` | `-3.4656` | `-3.4821` | `-2.9821` |
| `96000 Hz` | `47800.0 Hz` | `-0.5000` | `-3.5321` | `-2.3832` | `-3.5321` | `-3.0321` |
| `96000 Hz` | `47980.0 Hz` | `-0.5000` | `-3.5344` | `-0.7088` | `-3.5344` | `-3.0344` |

---

## 4. Raccomandazione Operativa per Eventuale Follow-up Autorizzato

1. **Per il requisito di consegna (`<= -0.30 dBTP`):** `-0.50 dBTP` è già pienamente robusto (+0.157 dB minimo sui 6 render, zero interventi offline). Non è strettamente necessario modificarlo se il target esterno resta `-0.30 dBTP`.
2. **Se si vuole garantire anche il margine interno esterno-misurato `<= -0.50 dBTP` su segnali reali densi senza ritocchi FIR/latenza:**
   - **Candidato consigliato:** abbassare in un eventuale futuro commit autorizzato il solo scalare `LIMITER_INTERNAL_CEILING_DBTP` da `-0.50 dBTP` a **`-0.56 dBTP`** (oppure `-0.55 dBTP` come minimo stretto).
   - **Impatto acustico/metrico:** variazione massima di loudness su `track01` di soltanto **`-0.0111 LU`** (e **`<= -0.0012 LU`** su `track02`/`track03`, impercettibile), con riduzione di True-Peak PLR di `-0.049 .. -0.059 dB` e mantenimento integrale della topologia FIR `48` tap, latenza `159` campioni e test deterministici `T1..T7`.
3. **Stato attuale invariato:** nessuna costante DSP o coefficiente FIR è stata toccata nel presente sprint in attesa di decisione esplicita del supervisore.
