# AutoMastering PoliMi — Audit tecnico, correzioni e roadmap di ricerca

**Destinatario:** Antigravity  
**Repository:** https://github.com/gio961gio/AutoMastering  
**Branch esaminato:** `master`  
**Data della revisione:** 1 novembre 2026  
**Scopo:** migliorare la qualità percepita del mastering, l'affidabilità dell'ottimizzazione PyTorch e la coerenza con il motore C++/JUCE, senza perdere l'architettura white-box a 41 parametri.

---

## 0. Istruzioni operative per questa revisione

Questa nota è un audit statico e una proposta di piano, non una dichiarazione che tutti i problemi siano già riprodotti a runtime. Prima di cambiare il codice:

1. riprodurre ogni finding con un test minimo e salvare input, parametri e output;
2. distinguere i bug certi dalle ipotesi da verificare;
3. modificare una cosa per volta, aggiungendo test di regressione;
4. non ottimizzare ulteriormente le metriche correnti prima di aver verificato che siano coerenti con l'obiettivo sonoro;
5. non dichiarare un miglioramento audio sulla sola base della diminuzione di una loss: mantenere un protocollo di ascolto level-matched e, quando possibile, blind.

**Priorità generale:** prima correttezza numerica e semantica (neutralità, metrica, motore effettivo), poi objective/loss e protocollo temporale, poi ottimizzatori più sofisticati e infine il predictor neurale.

### Riepilogo delle priorità

| Priorità | Area | Problema / obiettivo |
|---|---|---|
| P0 | Parametri | `z=0.5` non è un preset neutro; la regolarizzazione lo tratta comunque come tale |
| P0 | Motore | Il ramo `hybrid EQ` assegna `fft` in entrambi i casi; si ottimizza FFT ma si esporta con C++ IIR |
| P0 | Loss PLR | `target_plr` è dichiarato ma non utilizzato; la penalità 8–14 dB non equivale al matching della reference |
| P0 | Protocollo | `run_mastering.py` ottimizza solo i primi 0,5 s con 15 step e applica i parametri all'intero brano |
| P0 | Reference | La loss STFT frame-by-frame va usata solo con mix/master dello stesso brano ben allineati; non è adatta a brani diversi |
| P0 | Misure | La misura LUFS è una proxy senza gating; il peak misurato è sample peak, non true peak |
| P0/P1 | Limiter | Verificare con test impulsivi se smoothing, lookahead e ceiling garantiscono davvero il ceiling in ogni condizione |
| P1 | Dataset | I parametri sintetici campionati intorno a `z=0.5` generano impostazioni spesso aggressive e poco realistiche |
| P1 | Batch | EQ FFT e saturatore hanno broadcasting da verificare/correggere per `batch_size > 1` |
| P1 | Preset | JSON esportato da Python e parser del plugin non hanno la stessa struttura |
| P1 | Validazione | I test sintetici verificano soprattutto coerenza interna, non qualità di mastering su musica reale |
| P2 | Architettura | Detector compressore, sidechain, crossover multibanda e priors musicali da studiare |
| P2 | Ricerca | ERB/Mel, transient loss, multi-chunk, multi-start, block-coordinate e Adam→L-BFGS |
| P3 | Fase futura | Predictor neurale `audio → 41 parametri`, PCA/Hessian solo quando ci sono dati e baseline solidi |

---

# 1. Findings sul codice: correzioni da fare prima di nuove ottimizzazioni

## 1.1 `z=0.5` non è neutro — P0

**File:** `python/ddsp_mastering/optimize/optimizer.py`, `python/ddsp_mastering/param_space.py`, `python/ddsp_mastering/loss/combined.py`.

L'ottimizzatore inizializza `u=0`, quindi `z=sigmoid(u)=0.5` per tutti i 41 controlli. La loss `param_reg` penalizza inoltre la distanza da `z=0.5`:

```python
param_reg = torch.mean((normalized_params - 0.5) ** 2)
```

Ma `0.5` è il centro dei range normalizzati, non il punto di neutralità fisica. Per esempio, con gli attuali mapping:

| Controllo | Valore fisico a `z=0.5` | Interpretazione |
|---|---:|---|
| EQ gain | 0 dB | neutro per il guadagno; f0/Q sono comunque solo posizioni dei filtri |
| Compressor ratio | circa 4.47:1 | compressione potenziale, non bypass |
| Compressor makeup | +3 dB | aumento di livello |
| Imager Mid/High | 1.25 | allargamento stereo |
| Saturator drive | 9 dB | drive non nullo |
| Saturator mix | 50% | metà segnale wet |
| Limiter push | +6 dB | forte spinta verso il ceiling |

Quindi il sistema parte da una catena già pesantemente elaborata e viene contemporaneamente regolarizzato verso quella configurazione.

### Correzione richiesta

- Costruire `theta_neutral` nello spazio fisico e ricavare `z_neutral = map_theta_to_z(theta_neutral)`.
- Usare `u_neutral = logit(z_neutral)` come inizializzazione dell'ottimizzatore.
- Sostituire la regolarizzazione uniforme verso 0.5 con prior fisici/percettivi, pesati per gruppo e per unità.
- Aggiungere un test che controlli i valori fisici effettivi del preset neutro e la risposta audio della catena.

Valori iniziali indicativi:

- EQ gains = 0 dB;
- compressor ratio = 1:1 e makeup = 0 dB (threshold diventa irrilevante quando ratio=1);
- imager widths = 1.0;
- saturator mix = 0% (drive irrilevante quando wet=0);
- limiter push = 0 dB.

**Caveat:** il preset non è un bypass matematico perfetto dell'intera catena: il limiter ha un ceiling fisso a -0.3 dBFS e può comunque attenuare picchi alti; il saturatore in dry mantiene il ritardo del percorso dry; crossover e filtri possono introdurre fase/latency. Il test deve separare neutralità dei controlli da identità campione-per-campione.

## 1.2 Bug nel selettore `hybrid EQ` — P0

**File:** `python/ddsp_mastering/optimize/optimizer.py`.

Attualmente:

```python
if self.use_hybrid_eq and step < int(self.num_steps * 0.7):
    eq_mode = "fft"
else:
    eq_mode = "fft"
```

Il ramo `else` è identico: la modalità IIR non viene mai attivata. Correggere il selettore solo dopo aver definito e testato il comportamento desiderato.

In più, `final_master` viene sempre calcolato con `eq_mode="fft"`, mentre lo script finale usa `py_core_dsp.MasteringChain`, che esegue l'EQ IIR C++.

### Correzione richiesta

1. Rendere esplicita la modalità di training e quella di validazione.
2. Se si usa FFT per la fase coarse, rifinire con IIR differenziabile oppure con un metodo che abbia gradienti coerenti con l'implementazione IIR effettiva.
3. Dopo l'ottimizzazione, ricalcolare l'audio con il motore C++ e misurare la discrepanza audio/feature rispetto al risultato PyTorch.
4. Non considerare il preset valido finché non passa un test di coerenza PyTorch→C++ su un brano reale e su segnali sintetici difficili.

La modalità FFT non è automaticamente una sostituzione esatta dell'IIR: approssimazione della risposta, troncamento, condizioni al contorno e lunghezza del blocco possono produrre differenze. Non basta che il vettore dei parametri sia identico.

## 1.3 La loss PLR non corrisponde al target reference — P0

**File:** `python/ddsp_mastering/loss/loudness.py`.

La classe dichiara:

```python
target_plr = 10.0
plr_min = 8.0
plr_max = 14.0
```

ma `target_plr` non viene usato. La loss applica invece una penalità soltanto quando il PLR predetto è fuori dall'intervallo `[8, 14]` dB:

```python
plr_under_penalty = F.relu(self.plr_min - plr) ** 2
plr_over_penalty = F.relu(plr - self.plr_max) ** 2
```

Questa non è una loss di matching del PLR della reference. Se la reference ha PLR 7.4 dB, la loss penalizza il sistema anche quando cerca di riprodurne il PLR.

### Correzione richiesta

- Se il target è lo stesso brano masterizzato, calcolare `PLR_ref` con la stessa definizione e usare una distanza diretta, ad esempio Huber: `Huber(PLR_pred - PLR_ref)`.
- Se non esiste una reference dello stesso brano, evitare un target PLR universale: usare un prior condizionato al genere/stile o un intervallo scelto in modo esplicito.
- Dichiarare una sola definizione per il PLR usata sia in loss sia in benchmark.
- Evitare di chiamare PLR una misura che in realtà è un'altra forma di crest factor senza descriverne la definizione.

La metrica `MasteringBenchmarkSuite` calcola già la differenza tra PLR predetto e reference. La loss di training e il benchmark devono essere coerenti.

## 1.4 LUFS: proxy differenziabile, non misuratore integrato BS.1770 completo — P0

**File:** `python/ddsp_mastering/loss/loudness.py`.

L'implementazione esegue K-weighting e una media energetica dell'intero tensore. È utile come proxy differenziabile, ma non implementa il gating assoluto/relativo previsto dalla misura integrata ITU-R. Inoltre il filtro K-weighting viene applicato tramite moltiplicazione in frequenza su una finestra finita: si tratta di una convoluzione circolare/condizioni al contorno periodiche, non della normale applicazione causale con stato del filtro.

Il codice e i commenti fanno riferimento a BS.1770-4; la versione ITU attualmente in vigore è **BS.1770-5** (novembre 2023).

### Correzione richiesta

- Separare `differentiable_loudness_proxy()` dal misuratore di valutazione standard.
- Per i report, usare un misuratore conforme a ITU-R BS.1770-5 / EBU R128 e documentare il tool e la versione.
- Per il training, valutare una proxy differenziabile su finestre temporali sufficientemente lunghe e con filtro causale/stato correttamente gestito; se il gating hard rende difficile il gradiente, usare una sua approssimazione soft soltanto per il training.
- Verificare che il calcolo sia stabile su durate differenti, file silenziosi, intro/outro e segnali con canali sbilanciati.

Non sostituire la misura standard con la proxy di training nel report finale.

## 1.5 Il calcolo del peak è hard max, non soft maximum — P0/P1

**File:** `python/ddsp_mastering/loss/loudness.py`.

Il commento parla di “Soft maximum”, ma il codice usa `torch.max`. La derivata passa quindi soltanto attraverso il campione massimo (o i massimi a parità). È una scelta utilizzabile come proxy, ma non è un soft maximum.

### Correzione richiesta

- Correggere il commento e documentare la scelta, oppure introdurre una softmax/log-sum-exp controllata per una loss più distribuita.
- Tenere separata la misura precisa di sample peak/true peak dal proxy differenziabile usato per il gradiente.

## 1.6 Sample peak non equivale a True Peak — P0

**File:** `python/ddsp_mastering/dsp/limiter.py`, `cpp/core_dsp/include/core_dsp/LookaheadLimiter.hpp`, `python/ddsp_mastering/optimize/metrics.py`.

Il detector e il benchmark misurano `max(abs(samples))`. Questo è sample peak, non true peak: non misura i picchi inter-sample della forma d'onda ricostruita.

### Correzione richiesta

- Non descrivere il limiter attuale come true-peak limiter finché non viene implementato e validato un detector sovracampionato conforme alla misura scelta.
- Implementare oversampling/interpolazione di misura (ad esempio 4x o il metodo richiesto dallo standard) sia nel controllo sia nel benchmark indipendente.
- Mantenere un margine di sicurezza e testare toni ad alta frequenza, transitori e segnali stereo con fasi differenti.
- Nel report denominare la misura corrente `sample_peak_dbfs`, non `true_peak_dbfs`.

## 1.7 Verificare il comportamento brickwall del limiter — P0

**File:** `python/ddsp_mastering/dsp/limiter.py`, `cpp/core_dsp/include/core_dsp/LookaheadLimiter.hpp`.

La catena limiter usa min-filter/monotonic deque, envelope release e boxcar smoothing. È necessario verificare formalmente che la combinazione di lookahead, ritardo audio e smoothing non produca overshoot del ceiling su picchi isolati o transitori molto brevi.

Il test corrente su rumore musicale casuale non è sufficiente per dimostrare un ceiling assoluto. In particolare, la media mobile del gain può aumentare il gain rispetto al minimo istantaneo necessario, e l'allineamento tra finestra di detection, delay line e finestra di smoothing deve essere verificato campione per campione. Non assumere che il nome `brickwall` garantisca il comportamento.

### Test obbligatori

- Impulso unitario e impulsi di ampiezza diversa in punti iniziali, centrali e finali del buffer.
- Singolo sample peak isolato, burst di 1–3 campioni e sinusoide ad alta frequenza con picco inter-sample elevato.
- Picchi posti in corrispondenza dei bordi della finestra lookahead.
- Test con `push_gain_db` 0, 3, 6 e 12 dB.
- Misura separata di massimo sample peak e true peak.
- Verifica di latenza e allineamento con impulsi, inclusi i primi e ultimi campioni.

Se si trova un overshoot, correggere la struttura del gain envelope/lookahead; non risolvere semplicemente abbassando il ceiling senza spiegare la causa.

## 1.8 `forward_cpp` non è presente nella classe Python — P0/P1

**File:** `python/ddsp_mastering/data/synthetic.py`, `python/ddsp_mastering/dsp/chain.py`.

`SyntheticMasteringPairGenerator` usa di default `use_cpp_fast=True` e, per `batch_size == 1`, richiama `self.chain.forward_cpp(...)`. Nella classe `DifferentiableMasteringChain` del repository non risulta definito un metodo `forward_cpp`.

I test del generatore impostano `use_cpp_fast=False`, quindi non esercitano il percorso predefinito.

### Correzione richiesta

- O implementare chiaramente `forward_cpp` come wrapper del binding C++ (con conversione device/shape/dtype e gestione stato), oppure rendere il default `False` finché il percorso non è testato.
- Aggiungere un test che chiami `SyntheticMasteringPairGenerator()` senza override e verifichi l'esecuzione completa.

## 1.9 Broadcasting batch in EQ FFT e saturatore — P1

**File:** `python/ddsp_mastering/dsp/eq.py`, `python/ddsp_mastering/dsp/saturator.py`.

La modalità EQ FFT costruisce `H_tot` con forma compatibile con `[B, F]`, mentre lo spettro audio ha forma `[B, C, F]`. La moltiplicazione diretta può avere broadcasting errato quando `B > 1` (il batch può essere interpretato come dimensione canale) o fallire.

Il saturatore appiattisce audio e canali in `[B*C, 1, N]`, ma i parametri `drive` e `mix` rimangono normalmente `[B, 1, 1]`: per batch maggiori di uno le dimensioni non corrispondono.

### Correzione richiesta

- Definire in modo uniforme shape e broadcasting di tutti i moduli.
- Aggiungere test con `B=1,2,4`, stereo, differenti parametri per ogni elemento batch.
- Verificare che nessun esempio del batch riceva i parametri di un altro.

L'ottimizzatore attuale usa batch size 1; il bug è quindi soprattutto un blocco per training e ottimizzazione multi-chunk vettorizzata.

## 1.10 Preset JSON Python e parser JUCE non sono compatibili — P1

**File:** `python/ddsp_mastering/optimize/optimizer.py`, `cpp/plugin/source/PluginProcessor.cpp`.

Python esporta i parametri sotto la chiave `parameters`, ciascuno come oggetto contenente `value`, `min`, `max`, ecc. Il parser C++ cerca invece gli ID dei parametri direttamente nell'oggetto root e si aspetta valori numerici.

Inoltre il loader restituisce `true` anche se non trova parametri da caricare.

### Correzione richiesta

- Definire uno schema JSON versionato e unico.
- Fare in modo che il plugin legga esattamente `parameters.<id>.value` oppure modificare l'export per aderire allo schema scelto.
- Validare `format`, versione, sample rate, presenza e tipo dei 41 parametri.
- Restituire `false` se il documento non contiene valori validi.
- Aggiungere un round-trip test Python export → C++ load → C++ export e confrontare i valori fisici.

## 1.11 Altri aspetti da controllare

- Il `run_mastering.py` carica il mix e la reference tagliandoli all'inizio alla stessa durata. Non effettua un allineamento temporale automatico. Questo è accettabile solo se i file sono già temporalmente sincronizzati.
- L'allineamento fisso di 159 campioni compensa la latenza prevista della catena, ma non corregge offset differenti tra mix e master né ritardi di gruppo o automazioni differenti.
- La valutazione di loudness in `MasteringBenchmarkSuite` viene eseguita sull'audio predetto non allineato, mentre la STFT usa la porzione allineata: documentare la scelta e valutare un confronto coerente delle regioni valide.
- Il motore C++ esportato non restituisce la coda di latenza: l'audio finale contiene il ritardo iniziale e tronca la parte finale della coda. Per un export offline valutare pre-roll/flush e compensazione della latenza; per il plugin DAW la gestione dell'host è diversa.
- I test di parità non equivalgono a test di qualità percettiva; mantenere entrambe le categorie.

---

# 2. Problema fondamentale: che tipo di reference viene utilizzata?

Prima di cambiare la loss, rendere esplicita una distinzione che cambia radicalmente il problema.

## Caso A — Mix grezzo e master dello stesso identico brano

Una loss temporale o STFT frame-by-frame può essere utile, a condizione che:

- i file siano sincronizzati;
- le versioni abbiano la stessa struttura musicale e lo stesso contenuto;
- la latenza della catena sia compensata correttamente;
- le differenze di fase e le elaborazioni non riproducibili dalla catena siano considerate;
- i segmenti usati per la loss siano rappresentativi dell'intera traccia.

In questo caso conservare una componente MR-STFT, ma non farla diventare l'unico criterio di qualità.

## Caso B — Reference commerciale di un brano differente

Non usare una corrispondenza frame-by-frame come se i due segnali fossero lo stesso contenuto. Arrangiamento, note, transienti e timing sono diversi: l'ottimizzatore potrebbe inseguire eventi che non esistono nel mix sorgente.

In questo caso confrontare impronte statistiche aggregate e caratteristiche percettive: distribuzione dell'energia per bande ERB/Mel, statistiche temporali della loudness, crest factor/PLR, transient density, stereo width/mono compatibility e, con cautela, embedding audio.

**Aggiungere ai metadati di ogni scenario un campo obbligatorio `reference_type = same_track_master | stylistic_reference`.** La loss deve selezionare il percorso corretto in base a questo campo.

---

# 3. Migliorare la funzione obiettivo: da metriche tecniche a qualità sonora

## 3.1 Non rimuovere MR-STFT: affiancarle una rappresentazione percettiva

La MR-STFT esistente (512, 2048, 8192) è una base utile: risoluzioni diverse compensano il trade-off tempo/frequenza. Tuttavia, la griglia FFT lineare non distribuisce i bin secondo la scala percettiva e una loss uniforme può non riflettere l'importanza musicale delle diverse bande.

Proposta sperimentale:

- proiettare le magnitudini STFT su filterbank ERB o Mel;
- calcolare log-energy/Log-Mel distance per banda;
- valutare statistiche temporali (media, deviazione standard, percentili 10/50/90) per ciascuna banda;
- tenere più di una risoluzione temporale, includendo una risoluzione corta per transienti;
- misurare le norme dei gradienti per termine e normalizzare le scale prima di scegliere i pesi `lambda`.

**Nota:** non dichiarare che una percentuale fissa (ad esempio “75% del gradiente”) sia dovuta alle alte frequenze soltanto contando i bin FFT. Il contributo effettivo al gradiente dipende da contenuto, magnitudine, finestra, loss e parametri. Misurare per banda la norma del gradiente e riportare il risultato.

## 3.2 Transient / envelope loss

Aggiungere una componente che osservi l'inviluppo e la sua variazione nel tempo, distinta dal bilanciamento spettrale medio.

Possibili feature:

- inviluppo RMS veloce e lento per bande;
- onset strength o positive envelope slope;
- crest factor locale;
- percentili dei picchi e distribuzione della gain reduction;
- statistiche dei transienti per sezione.

Un termine concettuale è:

`L_transient = distance(envelope_fast_pred, envelope_fast_ref)`

più, se necessario, una distanza sulle variazioni positive dell'inviluppo.

Per reference diverse non cercare di allineare ogni singolo onset: confrontare distribuzioni/statistiche o profili a livello di sezione.

## 3.3 Dinamica: non soltanto un numero PLR

Valutare insieme:

- PLR/crest factor;
- loudness short-term e momentary;
- distribuzione temporale della gain reduction;
- crest factor locale per sezione;
- transient preservation;
- pumping/modulazione di gain.

Il PLR integrato da solo non descrive tutta la dinamica musicale.

## 3.4 Stereo

Aggiungere test e metriche per:

- Mid/Side energy ratio;
- correlazione L/R;
- mono compatibility / fold-down;
- variazioni di width per bande;
- eventuali inversioni/cancellazioni di fase.

Non premiare automaticamente un'immagine stereo più larga: width elevata non equivale a migliore qualità.

## 3.5 Loss composita proposta

Usare come ipotesi iniziale, da calibrare con esperimenti e gradient diagnostics:

`L_total = λ_spec L_ERB/MR-STFT + λ_trans L_transient + λ_loud L_LUFS + λ_dyn L_dynamics + λ_stereo L_stereo + λ_prior L_prior`

- `L_ERB/MR-STFT`: bilanciamento spettrale percettivo e dettaglio multi-risoluzione.
- `L_transient`: preservazione di attacchi/inviluppi.
- `L_LUFS`: raggiungimento della loudness desiderata.
- `L_dynamics`: PLR/crest distribution e gain reduction.
- `L_stereo`: stabilità spaziale e mono compatibility.
- `L_prior`: preferenza per impostazioni non estreme e musicalmente plausibili.

Prima di fissare i `λ`, loggare ciascun termine, la sua scala e la norma del gradiente che induce sui gruppi dei 41 parametri.

---

# 4. Protocollo temporale: sostituire il chunk singolo da 0,5 s

**File:** `scripts/run_mastering.py`.

Il default attuale ottimizza 0,5 secondi e 15 step, poi applica gli stessi 41 parametri al brano intero. È una configurazione troppo fragile per parametri dinamici e per un mastering che deve funzionare su strofa, ritornello, drop e transizioni.

## Proposta

- iniziare con 4–6 segmenti distribuiti lungo il brano;
- durata iniziale suggerita: 2–4 s ciascuno, da aumentare dopo profilazione;
- includere parti dense e meno dense, non soltanto i chunk più forti;
- condividere un unico vettore di 41 parametri tra tutti i segmenti;
- calcolare la media o una combinazione robusta delle loss dei segmenti;
- aumentare il budget iniziale a circa 60–100 step come esperimento, non come valore universale;
- usare early stopping su validazione temporale e selezionare la soluzione con miglior compromesso tra qualità, stabilità e prior.

### Attenzione allo stato DSP

Non trattare ogni chunk come se iniziasse da uno stato DSP neutro. EQ IIR, compressori, limiter e filtri hanno memoria. Due alternative:

1. processare segmenti con pre-roll precedente e calcolare la loss soltanto sulla regione centrale, dopo il warm-up;
2. implementare esplicitamente la gestione/carry dello stato tra blocchi, in modo coerente con il motore streaming.

Il pre-roll deve considerare le costanti di tempo più lunghe del sistema. Un chunk che riparte da zero altera la gain reduction iniziale e può dare gradienti non rappresentativi.

### Allineamento

- Se è lo stesso mix/master, stimare e verificare l'offset con correlazione o onset alignment prima della loss, senza confondere l'offset artistico con la latenza della catena.
- Per reference diverse, non eseguire allineamento di waveform: utilizzare statistiche e feature distribuzionali.

---

# 5. Priors fisico-acustici e range dei parametri

I range attuali sono molto ampi (EQ ±12 dB, Q fino a 5, compressor ratio fino a 20:1, attack fino a 0.5 ms, release fino a 1 s, makeup -6/+12 dB, width fino a 2, saturator drive 18 dB, limiter push 12 dB). Un ottimizzatore non vincolato può sfruttarli per compensare gli errori della loss.

Non fissare limiti “da mastering” come verità universali. Eseguire invece una ricerca controllata con range più conservativi e prior soft.

Come **esperimento iniziale** (non come standard):

- EQ gain: ±4–6 dB;
- EQ Q: 0.5–2.5;
- compressor ratio: 1:1–4:1;
- attack Low/Mid: testare limiti inferiori più lenti, ad esempio 10 ms, mantenendo una fascia distinta per High;
- makeup: centrare il prior vicino a 0 dB;
- imager width: prior centrato su 1.0;
- saturator mix/drive: prior verso interventi moderati;
- limiter push: prior verso 0 e GR contenuta.

Implementare penalità specifiche, non un semplice `mean((z - 0.5)**2)`:

- penalità su EQ gain eccessivi;
- penalità su Q molto stretto, se non necessario;
- penalità su gain reduction media/massima eccessiva;
- penalità su differenze anomale tra GR delle bande adiacenti;
- penalità su stereo width lontana da 1;
- penalità su saturazione/limiter eccessivi.

Se possibile, esporre gain reduction e statistiche dei processori dal DSP per poter misurare questi prior. Non inferire la GR soltanto dalla differenza tra input e output totale.

---

# 6. Architettura di ottimizzazione: quali strategie usare e in quale ordine

## 6.1 Curriculum / continuation — alta priorità

Non abilitare da subito tutti i 41 controlli con la loss completa.

Sequenza suggerita:

1. **Loudness e EQ:** compressore, imager, saturatore in modalità trasparente; ottimizzare bilanciamento e loudness in modo controllato.
2. **Compressione:** abilitare i parametri dinamici con loss transient/dynamics attiva.
3. **Stereo:** abilitare l'imager, imponendo controlli di mono compatibility.
4. **Saturazione:** abilitare drive/mix e misurare armoniche/aliasing.
5. **Limiter:** rifinire ceiling e loudness finale.
6. **Joint fine-tuning:** sbloccare tutti i gruppi con learning rate più basso e prior attivi.

## 6.2 Block Coordinate Descent — alta priorità

Ottimizzare gruppi di parametri alternandoli: EQ/crossover, compressor, imager, saturator, limiter. I gruppi si influenzano, quindi dopo i passaggi separati serve un raffinamento congiunto.

## 6.3 Multi-start — alta priorità

Eseguire più inizializzazioni ragionevoli (neutra, preset leggero, piccole perturbazioni casuali). Confrontare non solo la loss, ma anche:

- metriche indipendenti;
- valori estremi dei parametri;
- gain reduction;
- stabilità tra sezioni;
- ascolto level-matched.

Multi-start riduce il rischio di scegliere un bacino locale sfortunato, ma non garantisce il minimo globale.

## 6.4 Adam → L-BFGS — priorità media/alta

Prima ottenere gradienti affidabili e una loss ben scalata. Poi testare Adam come coarse search e L-BFGS con line search per la rifinitura. Verificare l'effettiva discesa della loss e la stabilità dei parametri. Non assumere che L-BFGS risolva una loss sbagliata.

## 6.5 Learning rate / gradient diagnostics

- Aumentare gli step soltanto dopo aver verificato le scale dei gradienti.
- Loggare norma del gradiente per ogni gruppo dei 41 parametri.
- Verificare se frequenze/Q/attack/release si muovono o se tutto il lavoro viene svolto dai gain.
- Usare clipping come protezione, non come sostituto della normalizzazione.
- Usare parameterization che eviti saturazione prematura della sigmoid quando possibile.

## 6.6 PCA e Hessian: non sono la prima correzione

PCA è utile se esiste un dataset ampio di soluzioni valide e si vuole imparare una rappresentazione ridotta. Non risolve una loss non percettiva, un target errato o l'inizializzazione sbagliata. Mantenere i 41 controlli fisici per l'interpretabilità UI/DAW.

La Hessian 41×41 è analizzabile in fase offline per individuare direzioni rigide/deboli e accoppiamenti locali. Farlo dopo aver corretto la funzione obiettivo, non prima. La covarianza dei parametri nel dataset e la Hessian della loss sono oggetti diversi e non vanno confusi.

Simulated annealing e Bayesian optimization restano strumenti secondari: possono essere utili su obiettivi non differenziabili o come ricerca grossolana, ma non sono la priorità finché il sistema DDSP ha gradienti disponibili.

---

# 7. Criticità del DSP da verificare con misure e ascolto

## 7.1 Detector del compressore

**File:** `python/ddsp_mastering/dsp/compressor.py`, `cpp/core_dsp/include/core_dsp/MultibandComp.hpp`.

Il detector calcola energia stereo per campione e la converte in dB, poi applica attack/release al gain computer. Non è la stessa cosa di un RMS detector con finestra d'integrazione esplicita. È una scelta progettuale, non necessariamente un bug.

Esperimenti:

- confronto tra detector attuale, peak detector e RMS detector;
- sidechain HPF o weighting per ridurre l'eccessiva influenza del sub, soprattutto nei brani con 808/bassi molto forti;
- test su kick/snare, voce, sub e segnali con transienti isolati;
- misurare gain reduction per banda e modulazione del gain nel tempo.

## 7.2 Crossover e compressione indipendente per banda

Il crossover LR4 può ricostruire una somma coerente prima delle elaborazioni. Quando le tre bande subiscono gain reduction differenti, la somma non mantiene necessariamente la stessa risposta ampiezza/fase del crossover non processato.

Esperimenti:

- confronto crossover + bypass contro segnale originale;
- sweep e impulsi per verificare risposta e group delay;
- compressione singola banda per volta;
- analisi della somma in prossimità delle frequenze di crossover;
- penalità o limiti soft su GR differenziale tra bande adiacenti.

## 7.3 Saturatore

La modalità `mix=0` è dry, ma con `z=0.5` drive e mix sono non nulli e il risultato può essere molto colorato. Testare THD, armoniche, aliasing e loudness-match; non giudicare la saturazione solo da LSD.

## 7.4 Limiter e clipping

Il limiter non deve diventare il mezzo con cui l'ottimizzatore risolve una loss spettrale o LUFS incoerente. Rendere visibili gain reduction e true peak. Valutare un soft clipper differenziabile prima del limiter soltanto come esperimento separato, con test d'ascolto e di distorsione.

---

# 8. Dati sintetici e futura rete neurale

## 8.1 Il prior del generatore sintetico è centrato nel punto sbagliato

**File:** `python/ddsp_mastering/data/synthetic.py`.

Il generatore campiona `z` intorno a 0.5 con deviazione 0.12 e lo clampa a [0.1, 0.9]. Poiché 0.5 non significa neutro per diversi controlli, le coppie sintetiche possono contenere regolarmente compressione, makeup gain, widening, saturazione e limiter push non moderati.

### Correzione richiesta

- Campionare i valori fisici per gruppo usando distribuzioni musicali esplicite.
- Generare esempi con molti livelli di intervento: bypass, mastering leggero, moderato, e casi più estremi ma controllati.
- Assicurarsi che la distribuzione dei parametri sia plausibile in unità fisiche, non soltanto centrata nello spazio normalizzato.
- Registrare le distribuzioni dei parametri e le metriche audio generate.
- Usare `map_theta_to_z` solo dopo aver campionato valori fisici validi.

## 8.2 Le coppie sintetiche non bastano per dimostrare mastering realistico

Generare `raw → master` con la stessa catena e poi invertire i parametri è un test eccellente di system identification, ma non dimostra generalizzazione a mix reali o a master professionali prodotti con catene diverse.

Possibili fonti:

- coppie reali mix/master correttamente allineate e con diritti d'uso chiari;
- degrado sintetico controllato di master di qualità per creare coppie pseudo-allineate, documentando il domain gap;
- esempi di effetti reali o plugin come benchmark, se l'interfaccia di controllo e licenze lo consentono;
- dataset di misure e ascolti per la validazione, separati dal training.

## 8.3 La rete neurale non risulta ancora implementata

Nel repository analizzato ci sono motore DSP, twin PyTorch, loss, ottimizzatore per singola traccia, generatore sintetico e plugin. Non risulta ancora il predictor neurale che esegue `audio → z[41]` in un solo passaggio.

Prima di addestrarlo:

1. rendere affidabile l'ottimizzatore offline;
2. produrre soluzioni stabili e musicalmente valide;
3. raccogliere feature audio globali e multi-sezione;
4. addestrare un encoder a predire i parametri, con vincoli e controlli di sicurezza;
5. valutare su brani non presenti nel training set.

Non distillare subito gli attuali `z*`: se l'ottimizzatore produce EQ aggressiva o compressione eccessiva, il predictor imparerà proprio quei comportamenti.

---

# 9. Protocollo sperimentale richiesto

Ogni modifica che pretende di migliorare la qualità deve essere valutata con lo stesso set di brani e gli stessi livelli di confronto.

## 9.1 Baseline da conservare

Per ogni brano salvare:

- mix grezzo;
- output attuale;
- output con neutral initialization corretta;
- output con multi-chunk;
- output con loss ERB/transient;
- reference, se esistente;
- preset JSON e log dei parametri;
- loss per termine e learning-rate history;
- LUFS integrata standard, sample peak, true peak, PLR, crest factor, LRA, M/S e misure di GR.

## 9.2 Confronto d'ascolto

- Level-match le versioni prima del confronto, altrimenti la più forte può sembrare migliore.
- Randomizzare ordine e nomi dei file; idealmente valutazione blind.
- Confrontare sezioni rappresentative: intro/verse/chorus/drop/outro.
- Raccogliere giudizi separati su tonal balance, punch/transients, harshness, bass control, stereo/mono, loudness e artefatti.
- Non chiedere soltanto “quale suona meglio?”: registrare anche perché e su quale sezione.
- Se possibile, fare un test MUSHRA/paired-comparison con un numero adeguato di ascoltatori.

## 9.3 Ablation study

Confrontare un cambiamento alla volta:

A. baseline attuale;  
B. neutral initialization + neutral prior;  
C. B + multi-chunk;  
D. C + ERB/Mel loss;  
E. D + transient/dynamics loss;  
F. E + block-coordinate/multi-start;  
G. F + Adam→L-BFGS.

Riportare sia misure oggettive sia ascolto. Non sommare tutte le modifiche prima di poter attribuire l'effetto.

## 9.4 Test di regressione DSP

Aggiungere test specifici per:

- neutral parameter mapping fisico;
- `B>1` per tutti i moduli;
- gradcheck del compressore e dei filtri;
- parità PyTorch/C++ a più sample rate e parametri;
- latenza e flush del buffer;
- limiter su impulsi/transienti;
- sample peak e true peak;
- preset export/import round trip;
- stabilità su silenzio, mono, stereo correlato, stereo decorrelato, clip corto e clip lungo.

---

# 10. Piano implementativo suggerito

## Sprint 1 — Correctness gate

1. neutral `theta/z` e prior corretti;
2. fix `hybrid EQ` e confronto PyTorch FFT/IIR/C++;
3. fix loss PLR;
4. rinominare/validare LUFS e peak come proxy vs standard;
5. test specifici sul limiter;
6. fix `forward_cpp` o default sintetico.

**Deliverable:** report di test e 3 export di un brano: bypass/neutro, baseline, corrected-neutral.

## Sprint 2 — Protocollo reference e finestre

1. classificare reference same-track vs stylistic;
2. aggiungere allineamento per same-track;
3. multi-chunk con pre-roll/stato;
4. aumentare steps e logging per gruppo;
5. benchmark full-song su C++ dopo ogni ottimizzazione.

**Deliverable:** esperimento A/B su almeno 3 brani e più sezioni.

## Sprint 3 — Loss e priors percettivi

1. ERB o Mel spectral statistics;
2. transient/envelope loss;
3. prior fisico-acustici;
4. dinamica/stereo e metric diagnostics;
5. ablation e ascolto level-matched.

**Deliverable:** confronto quantitativo + ascolto tra baseline e nuova loss.

## Sprint 4 — Ottimizzazione robusta e prestazioni

1. block-coordinate;
2. multi-start;
3. Adam→L-BFGS;
4. profiling per modulo e memoria;
5. batch support dopo fix shape.

**Deliverable:** qualità, tempo, stabilità e varianza tra inizializzazioni.

## Sprint 5 — Dati e predictor

1. prior realistici per synthetic pair generator;
2. dataset e split senza leakage;
3. predictor `audio → 41 parametri`;
4. validazione su tracce non viste;
5. fallback a ottimizzazione offline o preset manuale se il confidence è basso.

**Deliverable:** predictor confrontato con ottimizzazione per-traccia e ascolto.

---

# 11. Letteratura e risorse scientifiche

I lavori seguenti sono selezionati per la loro utilità concreta al progetto. Non sono tutti paper di mastering automatico: alcuni forniscono metodi DDSP, altri loss/evaluation o studi di mastering. Le note sotto distinguono contributi dimostrati e possibili trasferimenti al nostro caso.

## 11.1 Riferimenti prioritari

### A. Hayes et al. (2024) — Review DDSP

**Ben Hayes, Jordie Shier, György Fazekas, Andrew McPherson, Charalampos Saitis.** “A Review of Differentiable Digital Signal Processing for Music and Speech Synthesis.” *Frontiers in Signal Processing*, 3:1284100. Pubblicato 11 gennaio 2024.  
DOI: https://doi.org/10.3389/frsip.2023.1284100  
Testo: https://www.frontiersin.org/journals/signal-processing/articles/10.3389/frsip.2023.1284100/full  
Preprint: https://arxiv.org/abs/2308.15422

**Perché leggerlo:** è il riferimento generale più utile per il razionale DDSP, la differenziabilità di filtri/effetti, le difficoltà dei gradienti e la valutazione. Discute anche MR-STFT, loss Mel, problemi di stabilità/ottimizzazione e il fatto che le scelte di loss siano dipendenti dal task.

**Applicazione al repository:** usare la review per motivare l'uso di rappresentazioni spettrali multiple e percettive, test di stabilità IIR, gradient diagnostics e valutazione con ascolto. Non assume che una loss spettrale standard sia sufficiente per il mastering.

### B. Peladeau & Peeters (ICASSP 2024) — Blind estimation of mastering effects

**Côme Peladeau, Geoffroy Peeters.** “Blind Estimation of Audio Effects Using an Auto-Encoder Approach and Differentiable Digital Signal Processing.” *ICASSP 2024*, pp. 856–860.  
DOI: https://doi.org/10.1109/ICASSP48485.2024.10448301  
Pagina autori: https://researchportal.ip-paris.fr/en/publications/blind-estimation-of-audio-effects-using-an-auto-encoder-approach-/

**Contributo:** studia la stima cieca degli effetti audio a partire dal solo audio processato. Gli autori propongono un approccio autoencoder che ottimizza una metrica di qualità audio invece di affidarsi soltanto alla distanza tra parametri stimati e parametri ground truth. L'abstract riporta che il metodo stima meglio la qualità audio della catena AFX rispetto all'approccio tradizionale basato sui parametri, pur stimando i parametri meno accuratamente.

**Perché è rilevante:** supporta l'idea che il vero obiettivo debba essere la qualità dell'uscita, non il recupero esatto dei knob. È un motivo scientifico per non confondere parameter recovery con mastering quality.

**Limite:** non va interpretato come prova che un autoencoder generico risolva automaticamente il mastering commerciale; occorre verificare task, dataset, AFX e metrica dell'articolo.

### C. Yu et al. (DAFx 2024) — Differentiable All-pole Filters

**Chin-Yun Yu, Christopher Mitcheltree, Alistair Carson, Stefan Bilbao, Joshua D. Reiss, György Fazekas.** “Differentiable All-pole Filters for Time-varying Audio Systems.” *DAFx 2024*.  
Preprint: https://arxiv.org/abs/2404.07970  
Pagina progetto, paper, codice e samples: https://diffapf.github.io/web/

**Contributo:** affronta il problema dei gradienti attraverso filtri IIR ricorsivi tempo-varianti. Gli autori osservano che approssimazioni non ricorsive come frequency sampling/frame-based possono non restituire gradienti fedeli al sistema originale; propongono una formulazione ricorsiva con backpropagation efficiente. Dimostrano il metodo su phaser, synth e compressore feed-forward.

**Perché è rilevante:** è direttamente pertinente alla discrepanza fra ottimizzazione FFT e motore IIR, e al problema di gradienti accurati per sistemi dinamici.

**Azione proposta:** studiare il codice e gli esperimenti, poi confrontare il loro approccio con il nostro compressore custom-autograd e con l'EQ IIR. Non sostituire i moduli attuali senza un benchmark di parità e qualità.

### D. Martínez Ramírez et al. (2021) — Differentiable Signal Processing With Black-Box Audio Effects

**Marco A. Martínez Ramírez, Oliver Wang, Paris Smaragdis, Nicholas J. Bryan.** “Differentiable Signal Processing With Black-Box Audio Effects.” *ICASSP 2021*.  
Preprint: https://arxiv.org/abs/2105.04752  
Pagina IEEE: https://www.2021.ieeeicassp.org/2021.ieeeicassp.org/Papers/ViewPaper_MScbc2.html?PaperNum=2096

**Contributo:** usa effetti audio stateful di terze parti come layer black-box e una stima stocastica/parallelizzata del gradiente. Include un'applicazione di automatic music mastering e una valutazione d'ascolto soggettiva.

**Perché è rilevante:** fornisce un'alternativa quando un effetto commerciale non è differenziabile, o quando si vuole usare un plugin come teacher/benchmark. Nel nostro caso non è prioritario perché la catena white-box è già differenziabile, ma è una possibile strategia ibrida/surrogate.

### E. Mourgela et al. (AES 2024) — Dataset analysis di mix e master

**Angeliki Mourgela, Elio Quinton, Spyridon Bissas, Joshua D. Reiss, David Ronan.** “Exploring Trends in Audio Mixes and Masters: Insights from a Dataset Analysis.” *AES Convention 157*, paper 10186, settembre 2024.  
Pagina AES: https://aes.org/publications/elibrary/elibrary-page/?id=22689

**Contributo:** analisi di metriche su mix e master, con loudness integrata, mono compatibility, clipping/phase, compressione e profilo tonale su 30 generi forniti dalla piattaforma MixCheck. Il paper riporta la rilevanza di problemi di loudness/dinamica e miglioramenti nei parametri di compressione e nella percentuale di problemi stereo/fase nei master.

**Perché è rilevante:** aiuta a progettare un benchmark multi-dimensionale e a non ridurre la qualità a LUFS/LSD. Può anche suggerire stratificazione per genere, con cautela perché i dati derivano da una specifica piattaforma/utenza.

### F. Elliott & Chon (AES 2022) — Master umani e servizi automatici

**Mitchell Elliott, Song Hui Chon.** “A Comparative Study of Music Mastered by Human Engineers and Automated Services.” *Journal of the Audio Engineering Society*, 70(9), 764–776, 2022.  
DOI: https://doi.org/10.17743/jaes.2022.0050  
Pagina AES: https://aes.org/publications/elibrary/elibrary-page/?id=21888

**Contributo:** confronto percettivo tra due mastering engineer e due servizi automatici. I partecipanti non identificavano in modo consistente i master umani; la preferenza dipendeva anche dal genere (il paper riporta una preferenza per i master umani negli estratti jazz, ma non in quelli rock).

**Perché è rilevante:** suggerisce che la valutazione debba essere basata su ascolti controllati, e che le conclusioni dipendano dal contenuto musicale. Non usare l'articolo per sostenere che mastering umano e automatico siano sempre equivalenti.

## 11.2 Riferimenti complementari

### G. Engel et al. (2020) — DDSP originale

**Jesse Engel, Lamtharn Hantrakul, Chenjie Gu, Adam Roberts.** “DDSP: Differentiable Digital Signal Processing.” *ICLR 2020*.  
Pagina Google Research: https://research.google/pubs/ddsp-differentiable-digital-signal-processing/

**Utilità:** fondamento concettuale dell'integrazione fra DSP interpretabile e reti neurali con autograd. È orientato alla sintesi audio, quindi non costituisce una ricetta diretta per mastering.

### H. Pepe et al. (2022) — Deep Optimization of Parametric IIR Filters

**Giovanni Pepe, Leonardo Gabrielli, Stefano Squartini, Carlo Tripodi, Nicolò Strozzi.** “Deep Optimization of Parametric IIR Filters for Audio Equalization.” *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 2022.  
Preprint: https://arxiv.org/abs/2110.02077  
Pagina archivio: https://iris.univpm.it/handle/11566/297428

**Utilità:** rete BiasNet e ottimizzazione differenziabile di parametric EQ IIR; interessante per parametrizzazione, denormalizzazione di frequenza/Q/gain e stabilità del tuning. Il problema applicativo è equalizzazione di ambienti/HRTF, non mastering musicale.

### I. Gui et al. (ICASSP 2024) — Adapting Frechet Audio Distance

**Azalea Gui, Hannes Gamper, Sebastian Braun, Dimitra Emmanouilidou.** “Adapting Frechet Audio Distance for Generative Music Evaluation.” *ICASSP 2024*.  
DOI: https://doi.org/10.1109/ICASSP48485.2024.10446663  
Pagina Microsoft Research: https://www.microsoft.com/en-us/research/publication/adapting-frechet-audio-distance-for-generative-music-evaluation/

**Utilità:** mostra che la correlazione fra metriche audio e qualità percettiva dipende da embedding, dimensione del campione e qualità del reference set. FAD è una metrica a livello di distribuzione/corpus: non usarla come loss principale per ottimizzare una singola traccia.

### J. ITU-R BS.1770-5 — Standard loudness e true peak

**ITU-R Recommendation BS.1770-5 (novembre 2023).** “Algorithms to measure audio programme loudness and true-peak audio level.”  
Pagina ufficiale: https://www.itu.int/rec/R-REC-BS.1770-5-202311-I

**Utilità:** riferimento tecnico per le misure finali di integrated loudness e true peak. Il progetto deve distinguere la proxy differenziabile di training dalla misura conforme usata per il benchmark/report.

### K. Moroșanu et al. (2024) — Automated Personalized Loudness Control

**Bogdan Moroșanu, Marian Negru, Constantin Paleologu.** “Automated Personalized Loudness Control for Multi-Track Recordings.” *Algorithms*, 17(6), 228, 24 maggio 2024.  
DOI: https://doi.org/10.3390/a17060228

**Utilità:** discute leveling multi-track personalizzato tramite algoritmo genetico e rete neurale. È soprattutto un lavoro di mixing/leveling multi-traccia, non un sistema di mastering stereo finale; tenerlo come riferimento laterale per ottimizzazione evolutiva e controllo loudness.

### L. Melechovsky et al. (ICML 2026) — SonicMaster

**Jan Melechovsky, Ambuj Mehrish, Abhinaba Roy, Dorien Herremans.** “SonicMaster: Towards Controllable All-in-One Music Restoration and Mastering.” *Proceedings of Machine Learning Research*, vol. 306, ICML 2026, pp. 87512–87540.  
Pagina PMLR: https://proceedings.mlr.press/v306/melechovsky26a.html  
Codice/dataset/demo: https://github.com/AMAAI-Lab/SonicMaster

**Contributo:** modello generativo basato su flow-matching per restauro e mastering controllabile, con un dataset di coppie degradato/alta qualità create tramite 19 funzioni di degradazione in gruppi EQ, dinamica, riverbero, ampiezza e stereo. L'articolo riporta anche valutazione soggettiva.

**Perché è rilevante:** non è una soluzione da copiare nel nostro progetto white-box, ma suggerisce un paradigma utile per costruire dati sintetici paired e controllare l'ampiezza dei degradi. Da studiare per il dataset, confrontando con il nostro obiettivo di preservare parametri DSP interpretabili.

---

# 12. Decisioni richieste prima di procedere

Antigravity dovrebbe consegnare un breve report che risponda a queste domande, corredato da test:

1. Il target reference negli scenari è lo stesso brano del mix o una reference stilistica differente? Il codice deve trattare esplicitamente i due casi.
2. La catena PyTorch in `fft` produce risultati sufficientemente vicini alla catena C++ `iir` sui parametri effettivamente ottimizzati?
3. Il limiter rispetta il ceiling su impulsi e transienti isolati, non soltanto su rumore musicale?
4. Il preset fisicamente neutro è davvero neutro nei controlli e quali parti della catena mantengono una risposta/ritardo inevitabile?
5. Il benchmark usa un vero misuratore ITU-R BS.1770-5 per LUFS e true peak, oppure una proxy interna?
6. Quanta parte del gradiente sui 41 controlli va a EQ gains, makeup, limiter push, frequenze/Q e tempi attack/release?
7. I risultati migliorano nell'ascolto blind/level-matched e su più sezioni del brano?

**Non avviare la fase di training del predictor neurale finché le risposte ai punti 1–7 non sono documentate.**

---

## Nota finale

La priorità non è rendere l'ottimizzatore più complicato. È fare in modo che il sistema ottimizzi il problema giusto, parta da controlli interpretabili e neutrali, osservi una parte rappresentativa del brano e venga valutato con misure coerenti e ascolto controllato.

Mantenere i 41 parametri espliciti è una scelta sensata per interpretabilità, automazione DAW e controllo umano. L'eventuale PCA o predictor neurale può essere introdotto più avanti come rappresentazione interna, ricostruendo sempre i parametri fisici prima della catena DSP.
