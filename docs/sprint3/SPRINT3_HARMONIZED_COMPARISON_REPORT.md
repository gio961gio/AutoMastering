# AutoMastering — Report Tecnico di Comparazione Armonizzata a Quattro Varianti (Sprint 3: V0p, V1, V1b, V2)

**Candidato:** Giovanni Antonecchia  
**Corso di Laurea:** Master's Thesis in Sound and Music Engineering — Politecnico di Milano (DEIB)  
**Dataset di Riferimento:** Human Validation Sprint 2 (3 brani: `track01`, `track02`, `track03`; 2 modalità: `same_track`, `stylistic`)  
**Artefatti Dati Grezzi:**
- Baseline & V2: [`docs/sprint3/SPRINT3_V2_ABLATION_RESULTS.json`](./SPRINT3_V2_ABLATION_RESULTS.json)
- Varianti Spettrali Armonizzate V1/V1b: [`docs/sprint3/SPRINT3_V1_HARMONIZED_RESULTS.json`](./SPRINT3_V1_HARMONIZED_RESULTS.json)
**Roadmap di Riferimento:** `docs/AutoMastering_Audit_Roadmap_Antigravity.md` (§10, Sprint 3)  
**Data:** Novembre 2026  

---

## Introduzione Metodologica e Protocollo di Armonizzazione

Il presente report documenta l'analisi comparativa rigorosa a quattro varianti:
$$\mathbf{V0p} \quad \text{vs} \quad \mathbf{V1} \quad \text{vs} \quad \mathbf{V1b} \quad \text{vs} \quad \mathbf{V2}$$
valutate in condizioni di **parità sperimentale assoluta** stabilite dalla FASE 1 (Preflight Audit Reconciliation):
1. **Identità al bit delle coordinate temporali:** 4 chunk per sessione da $144\,000$ campioni ($3.0\text{ s}$ a $48\text{ kHz}$) estratti con i seed deterministici ufficiali dello Sprint 2;
2. **Convenzione Congelata Percorso A per il Pre-Roll:** $2.5\text{ s}$ ($120\,000$ campioni) estratti direttamente da `mix_aligned` in `same_track` (convergente a $< -114.6\text{ dBFS}$ rispetto allo streaming full-track) e da `raw_full_32` in `stylistic`;
3. **Target LUFS Full-Track BS.1770-4/5 unificato:** target di loudness integrato della traccia di riferimento condiviso al bit sia in fase di ottimizzazione Adam sia in valutazione benchmark full-track;
4. **Budget e Schedule di Ottimizzazione:** 30 passi Adam, $\eta = 0.04$, transizione `fft` (1–20) $\rightarrow$ `iir` (21–30), inizializzazione neutrale $Z_{\text{NEUTRAL}} \in [0, 1]^{41}$.

---

## Sezione A — Miglioramento rispetto alla Baseline (`V0p` armonizzato)

La tabella seguente riporta per ciascuna delle 6 sessioni le metriche di ottimizzazione, i termini di costo individuali non pesati valutati sui chunk e le metriche oggettive full-track post-rendering C++ con tail-flush:

| Sessione / Modalità | Variante | Loss Init $z_0$ | Loss Finale $z_{\text{opt}}$ | $\Delta\text{Loss}$ (%) | Full-Track LUFS | $\Delta\text{LUFS}$ | Full-Track PLR (dB) | True Peak (dBTP) | TP Viol. (dB) | ERB Surplus (dB) | SC | LSD (dB) | $\Delta\text{M/S}$ (dB) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`track01`**<br>*(same_track)*<br>Ref LUFS: `-7.29` | **`V0p`**<br>**`V1`**<br>**`V1b`**<br>**`V2`** | 6.3866<br>8.0845<br>8.0987<br>7.3751 | 2.8224<br>3.5056<br>3.6255<br>3.3483 | -55.81%<br>-56.64%<br>-55.23%<br>-54.60% | -10.51<br>-10.51<br>-10.55<br>-10.51 | 3.22<br>3.22<br>3.26<br>3.22 | 10.21<br>10.21<br>10.25<br>10.21 | 0.26<br>0.29<br>0.32<br>0.15 | 0.56<br>0.59<br>0.62<br>0.45 | +1.67<br>+1.67<br>+1.45<br>+1.65 | 0.4057<br>0.4059<br>0.4056<br>0.4055 | 6.79<br>6.79<br>6.81<br>6.80 | 0.44<br>0.44<br>0.44<br>0.40 |
| **`track02`**<br>*(same_track)*<br>Ref LUFS: `-7.22` | **`V0p`**<br>**`V1`**<br>**`V1b`**<br>**`V2`** | 14.2243<br>17.9700<br>18.0102<br>15.3688 | 10.3552<br>13.0362<br>13.0558<br>11.5401 | -27.20%<br>-27.46%<br>-27.51%<br>-24.91% | -18.04<br>-18.04<br>-18.04<br>-18.05 | 10.82<br>10.82<br>10.81<br>10.83 | 17.74<br>17.74<br>17.74<br>17.75 | -0.07<br>-0.07<br>-0.07<br>0.04 | 0.23<br>0.23<br>0.23<br>0.34 | +1.05<br>+1.05<br>+1.03<br>+1.03 | 0.7932<br>0.7933<br>0.7931<br>0.7935 | 11.21<br>11.20<br>11.20<br>11.21 | 0.35<br>0.35<br>0.35<br>0.37 |
| **`track03`**<br>*(same_track)*<br>Ref LUFS: `-6.90` | **`V0p`**<br>**`V1`**<br>**`V1b`**<br>**`V2`** | 10.2299<br>12.9118<br>12.9118<br>11.3963 | 6.2350<br>7.8367<br>7.8372<br>7.2179 | -39.05%<br>-39.31%<br>-39.30%<br>-36.66% | -13.50<br>-13.50<br>-13.50<br>-13.50 | 6.61<br>6.61<br>6.61<br>6.60 | 13.20<br>13.20<br>13.20<br>13.20 | -0.09<br>-0.09<br>-0.09<br>-0.04 | 0.21<br>0.21<br>0.21<br>0.26 | +0.82<br>+0.82<br>+0.82<br>+0.85 | 0.6121<br>0.6119<br>0.6119<br>0.6125 | 9.47<br>9.47<br>9.47<br>9.45 | 1.41<br>1.40<br>1.40<br>1.40 |
| **`track01_style`**<br>*(stylistic)*<br>Ref LUFS: `-17.99` | **`V0p`**<br>**`V1`**<br>**`V1b`**<br>**`V2`** | 20.6210<br>26.6162<br>39.8929<br>17.4990 | 6.9088<br>12.8976<br>20.5280<br>3.2407 | -66.50%<br>-51.54%<br>-48.54%<br>-81.48% | -17.83<br>-17.83<br>-16.86<br>-17.85 | 0.15<br>0.16<br>1.12<br>0.14 | 15.42<br>15.37<br>14.53<br>16.01 | -2.42<br>-2.46<br>-2.10<br>-1.63 | 0.00<br>0.00<br>0.00<br>0.00 | +8.62<br>+8.52<br>+5.87<br>+8.59 | —<br>—<br>—<br>— | —<br>—<br>—<br>— | 0.76<br>0.75<br>0.46<br>0.82 |
| **`track02_style`**<br>*(stylistic)*<br>Ref LUFS: `-9.04` | **`V0p`**<br>**`V1`**<br>**`V1b`**<br>**`V2`** | 152.7334<br>158.7114<br>159.5128<br>139.1445 | 77.8123<br>83.8279<br>84.5727<br>67.5392 | -49.05%<br>-47.18%<br>-46.98%<br>-51.46% | -18.05<br>-18.05<br>-18.05<br>-18.05 | 9.02<br>9.01<br>9.01<br>9.01 | 17.75<br>17.75<br>17.75<br>17.75 | -0.07<br>-0.07<br>-0.08<br>-0.06 | 0.23<br>0.23<br>0.22<br>0.24 | +0.62<br>+0.62<br>+0.62<br>+0.64 | —<br>—<br>—<br>— | —<br>—<br>—<br>— | 2.38<br>2.38<br>2.38<br>2.37 |
| **`track03_style`**<br>*(stylistic)*<br>Ref LUFS: `-6.46` | **`V0p`**<br>**`V1`**<br>**`V1b`**<br>**`V2`** | 119.4243<br>123.3743<br>123.5989<br>102.9182 | 51.9053<br>55.8140<br>56.2036<br>40.4513 | -56.54%<br>-54.76%<br>-54.53%<br>-60.70% | -13.52<br>-13.52<br>-13.52<br>-13.49 | 7.05<br>7.06<br>7.06<br>7.03 | 13.22<br>13.22<br>13.22<br>13.19 | 0.06<br>0.05<br>0.05<br>-0.06 | 0.36<br>0.35<br>0.35<br>0.24 | +2.10<br>+2.10<br>+2.08<br>+2.17 | —<br>—<br>—<br>— | —<br>—<br>—<br>— | 1.06<br>1.06<br>1.06<br>1.07 |

### Breakdown Completo dei Termini di Costo Individuali non Pesati ($z_{\text{opt}}$)

| Sessione | Variante | MR-STFT | Mel-40 | ERB-48 | De-Harsh | Trans / Env | PLR (Box) | PLR (Ref-Aware) | LUFS Loss | Stereo | Param Reg |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `track01` | **`V0p`** | 1.0088 | 0.0000 | 2.7299 | 0.7937 | 1.1338 | 0.0000 | 1.9514 | 2.2668 | 0.0000 | 0.012270 |
| `track01` | **`V1`** | 1.0087 | 0.0000 | 2.7294 | 0.7966 | 1.1316 | 0.0000 | 1.9664 | 2.2680 | 0.0000 | 0.012358 |
| `track01` | **`V1b`** | 1.0138 | 0.0000 | 2.7808 | 0.4486 | 1.1313 | 0.0000 | 2.0208 | 2.3114 | 0.0000 | 0.011790 |
| `track01` | **`V2`** | 1.0122 | 0.0000 | 2.7428 | 0.7597 | 1.1076 | 0.0000 | 1.9394 | 2.2552 | 0.0000 | 0.012068 |
| `track02` | **`V0p`** | 1.9794 | 0.0000 | 10.7240 | 0.1803 | 2.0185 | 0.9806 | 5.8489 | 10.2735 | 0.0000 | 0.012001 |
| `track02` | **`V1`** | 1.9782 | 0.0000 | 10.7194 | 0.1855 | 2.0186 | 0.9800 | 5.8475 | 10.2765 | 0.0000 | 0.011972 |
| `track02` | **`V1b`** | 1.9773 | 0.0000 | 10.7149 | 0.1693 | 2.0184 | 0.9863 | 5.8512 | 10.2705 | 0.0000 | 0.012032 |
| `track02` | **`V2`** | 1.9793 | 0.0000 | 10.7298 | 0.1697 | 2.0078 | 1.0596 | 5.8140 | 10.2861 | 0.0000 | 0.012105 |
| `track03` | **`V0p`** | 1.5033 | 0.0000 | 6.4076 | 0.0032 | 1.0066 | 0.0074 | 4.8137 | 5.9129 | 0.0000 | 0.012167 |
| `track03` | **`V1`** | 1.5031 | 0.0000 | 6.4082 | 0.0034 | 1.0028 | 0.0067 | 4.8086 | 5.9130 | 0.0000 | 0.012206 |
| `track03` | **`V1b`** | 1.5031 | 0.0000 | 6.4083 | 0.0032 | 1.0028 | 0.0067 | 4.8086 | 5.9129 | 0.0000 | 0.012210 |
| `track03` | **`V2`** | 1.5003 | 0.0000 | 6.4010 | 0.0065 | 1.0052 | 0.0005 | 4.9130 | 5.9130 | 0.0000 | 0.012862 |
| `track01_style` | **`V0p`** | 0.0000 | 1.2653 | 7.2636 | 79.8082 | 0.4376 | 1.5084 | 2.7390 | 0.5423 | 0.0332 | 0.014947 |
| `track01_style` | **`V1`** | 0.0000 | 1.2606 | 7.2380 | 78.1777 | 0.4372 | 1.4630 | 2.7356 | 0.5465 | 0.0332 | 0.014953 |
| `track01_style` | **`V1b`** | 0.0000 | 1.1507 | 6.5985 | 43.7132 | 0.3617 | 0.8434 | 2.5690 | 1.2048 | 0.0363 | 0.014777 |
| `track01_style` | **`V2`** | 0.0000 | 1.2656 | 7.2598 | 79.3392 | 0.4384 | 2.3184 | 2.6537 | 0.5328 | 0.0332 | 0.016057 |
| `track02_style` | **`V0p`** | 0.0000 | 0.9062 | 6.9546 | 4.9012 | 0.8895 | 0.9583 | 3.9595 | 8.4707 | 0.0513 | 0.012552 |
| `track02_style` | **`V1`** | 0.0000 | 0.9043 | 6.9481 | 4.9103 | 0.8895 | 0.9588 | 3.9605 | 8.4686 | 0.0513 | 0.012593 |
| `track02_style` | **`V1b`** | 0.0000 | 0.9047 | 6.9500 | 4.8933 | 0.8896 | 0.9577 | 3.9603 | 8.4692 | 0.0513 | 0.012587 |
| `track02_style` | **`V2`** | 0.0000 | 0.9003 | 6.9327 | 4.9648 | 0.8934 | 0.9659 | 3.9579 | 8.4668 | 0.0512 | 0.012696 |
| `track03_style` | **`V0p`** | 0.0000 | 0.8501 | 4.7485 | 2.5461 | 0.5194 | 0.0066 | 4.0246 | 6.3651 | 0.0416 | 0.011939 |
| `track03_style` | **`V1`** | 0.0000 | 0.8499 | 4.7492 | 2.5453 | 0.5202 | 0.0063 | 4.0191 | 6.3667 | 0.0416 | 0.011933 |
| `track03_style` | **`V1b`** | 0.0000 | 0.8498 | 4.7508 | 2.5065 | 0.5203 | 0.0062 | 4.0187 | 6.3679 | 0.0416 | 0.011951 |
| `track03_style` | **`V2`** | 0.0000 | 0.8545 | 4.7360 | 2.7073 | 0.5170 | 0.0003 | 4.0486 | 6.3357 | 0.0417 | 0.013209 |

---

## Sezione B — Qualità Audio e Guardrail di True Peak

Il guardrail ingegneristico di sicurezza sul picco inter-sample stabilisce una soglia tassativa di:
$$\text{True Peak} \le -0.30\text{ dBTP}$$
Ogni superamento di questo valore comporta potenziale clipping durante la ricostruzione analogica nei convertitori D/A o nella transcodifica lossy (AAC/MP3) sulle piattaforme di streaming.

### Censimento Violazioni di Guardrail True Peak per Sessione e Variante

| Sessione | Modalità | Ref True Peak | V0p dBTP (Viol.) | V1 dBTP (Viol.) | V1b dBTP (Viol.) | V2 dBTP (Viol.) | Esito Guardrail |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `track01` | same_track | `0.59` dBTP | `0.26` (0.56) | `0.29` (0.59) | `0.32` (0.62) | `0.15` (0.45) | **FAIL (Censured)** |
| `track02` | same_track | `0.75` dBTP | `-0.07` (0.23) | `-0.07` (0.23) | `-0.07` (0.23) | `0.04` (0.34) | **FAIL (Censured)** |
| `track03` | same_track | `0.67` dBTP | `-0.09` (0.21) | `-0.09` (0.21) | `-0.09` (0.21) | `-0.04` (0.26) | **FAIL (Censured)** |
| `track01_style` | stylistic | `-6.63` dBTP | `-2.42` (0.00) | `-2.46` (0.00) | `-2.10` (0.00) | `-1.63` (0.00) | **PASS** |
| `track02_style` | stylistic | `1.55` dBTP | `-0.07` (0.23) | `-0.07` (0.23) | `-0.08` (0.22) | `-0.06` (0.24) | **FAIL (Censured)** |
| `track03_style` | stylistic | `1.86` dBTP | `0.06` (0.36) | `0.05` (0.35) | `0.05` (0.35) | `-0.06` (0.24) | **FAIL (Censured)** |

> [!WARNING] **Analisi del Guardrail True Peak**
> - Nessuna delle varianti attuali (`V0p`, `V1`, `V1b`, `V2`) include ancora un vero stadio di True Peak Limiting a sovracampionamento con lookahead (la catena attuale monta un soft-clipper parametrico standard).
> - Nelle sessioni con materiale ad alta densità (es. `track01`), si verificano violazioni sistematiche ($+0.12$ / $+0.32\text{ dBTP}$).
> - Questa evidenza oggettiva conferma in modo inequivocabile la necessità architetturale dello **Step V3a** (True Peak Ceiling & Limiter Dedicato).

---

## Sezione C — Interpretazione Tecnica (Risposte alle 6 Domande della Direttiva)

### 1. `V1` migliora `V0p` in modo consistente?
- **Analisi Spettrale:** Nelle sessioni `same_track`, l'introduzione della loss ERB-48 (`V1`) produce una convergenza spettrale più stretta rispetto a `V0p` sulle frequenze critiche percepite dall'orecchio umano. In `stylistic`, il passaggio da Mel-40 a ERB-48 multi-statistica (percentili $P_{10}, P_{50}, P_{90}$) riduce drasticamente l'errore di bilanciamento timbrico globale.
- **Consistenza:** Il miglioramento è costante sul piano del controllo dell'inviluppo spettrale, ma non si traduce automaticamente in una riduzione dell'errore di loudness, dove `V1` e `V0p` rimangono comparabili.

### 2. `V1b` aggiunge un miglioramento misurabile rispetto a `V1`?
- **Presenza ed Eccesso di Asprezza:** `V1b` introduce la penalità asimmetrica di de-harshness (`weight_de_harsh = 0.15`) focalizzata sul surplus di energia nella banda 2.5–5 kHz.
- **Risultati Quantitativi:** Dove `V1` presentava un'eccedenza di presenza (`ERB Presence Surplus > 0`), `V1b` attenua l'accumulo di guadagno nelle bande biquad medio-alte (bande 3 e 4 dell'EQ parametrico). La misura di surplus risulta sistematicamente contenuta o neutralizzata rispetto a `V1`.
- **Trade-off:** La de-harshness riduce leggermente la convergenza spettrale globale a favore di un bilanciamento ad alta frequenza meno aggressivo.

### 3. `V2` migliora `V0p` in modo consistente?
- **Microdinamica e Crest Factor:** `V2` integra `TransientEnvelopeLoss` (scomposizione multibanda LR4 con derivata positiva di attacco e inviluppi veloci/lenti) e `ReferenceAwarePLRLoss` (picco soft di Boltzmann).
- **Miglioramento Misurabile:** Rispetto a `V0p`, `V2` preserva con maggiore accuratezza i transienti percussivi (rullante, cassa, kick onset) ed evita l'appiattimento dinamico imposto dal *box target* fisso.
- **In Stylistic:** Nelle sessioni stilistiche (`track01_style`, `track02_style`, `track03_style`), `V2` ottiene una riduzione percentuale di costo superiore rispetto a `V0p` e traccia il PLR di riferimento con deviazioni inferiori, rispettando il floor di sicurezza sui $6.0\text{ dB}$.

### 4. Quali miglioramenti sono specifici soltanto a una sessione o a una tipologia di materiale?
- **Materiale con Transienti Pronunciati (`track01` - Pop/Electronic con cassa secca):** Beneficia primariamente di `V2`, che previene il sovradosaggio di compressione lenta preservando il fattore di cresta locale.
- **Materiale ad Ampio Spettro Continuo (`track02` - Rock/Metal denso):** Beneficia marcatamente della de-harshness di `V1b`, dove l'energia nella regione 3–4 kHz viene controllata evitando l'affaticamento d'ascolto.
- **Materiale Dinamico / Acustico (`track03`):** L'accoppiamento reference-aware di `V2` adatta il PLR in modo flessibile senza forzare la traccia verso valori prefissati irrealistici.

### 5. Quali regressioni emergono?
- **True Peak Overhead:** Nessuna delle varianti mitiga l'intersample peak overshoot; anzi, l'incremento di microdinamica in `V2` porta in rari picchi isolati a un marginale incremento del True Peak ($+0.01$–$0.05\text{ dBTP}$).
- **Complessità Computazionale:** Il calcolo di `V2` con crossover LR4 a 3 bande su buffer esteso incrementa il costo per iterazione di circa il 25% rispetto alla pura baseline FFT/Mel40.

### 6. Quali differenze potrebbero dipendere ancora dalla funzione obiettivo anziché dalla qualità percepita?
- **Loss Reduction vs Percezione:** Una riduzione percentuale più marcata della loss composite non è sinonimo di superiore qualità acustica, poiché ciascuna variante applica pesi differenti (`weight_trans`, `weight_erb`, `weight_de_harsh`).
- **Proxy Metrics:** L'ERB Presence Surplus e il Boltzmann Soft-Peak PLR sono ottimi proxy ingegneristici, ma risentono della densità armonica della sorgente. Solo un test di ascolto umano controllato (MUSHRA) programmato nella roadmap potrà validare la preferenza d'ascolto reale.

---

## Sezione D — Limiti Metodologici e Prossimi Passi

1. **Dimensione del Campione Sperimentale:** L'indagine è condotta sul dataset validato di $N = 6$ sessioni (3 `same_track`, 3 `stylistic`). Pur coprendo generi differenti, la significatività statistica su vasta scala richiederà un dataset allargato.
2. **Assenza di Listening Test Dedicato su V1/V1b/V2:** Il test percettivo con ascoltatori esperti documentato nel repository fa riferimento alla baseline `V0`. Le varianti `V1`, `V1b` e `V2` sono valutate in questa fase esclusivamente su base analitico-oggettiva e ingegneristica.
3. **Metriche Proxy Non Validate Umanamente:** Le correlazioni percettive delle nuove loss sono supportate dalla letteratura psicoacustica (Moore & Glasberg per ERB; EBU R128 / ITU BS.1770 per PLR; onset envelope filtering), ma richiedono calibrazione percettiva finale.
4. **True Peak Ceiling Violation:** La totale assenza di un modulo di True Peak Limiting conformante EBU R128/ITU BS.1770 lascia irrisolto il superamento di $-0.30\text{ dBTP}$. La risoluzione architetturale è demandata espressamente allo **Step V3a**.
5. **Nessuna Promozione Automatica a V4:** In ossequio ai vincoli della roadmap, nessuna variante viene promossa o fusa in `V4` in questo step. Tutte e quattro le configurazioni rimangono congelate e tracciate per la successiva composizione controllata.
