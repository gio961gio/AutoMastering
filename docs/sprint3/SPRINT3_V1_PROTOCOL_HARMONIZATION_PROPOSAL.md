# Proposta Tecnica di Armonizzazione del Protocollo di Valutazione (V0p / V1 / V1b / V2)

**Autore:** Antigravity Engineering  
**Data:** 9 Novembre 2026  
**Stato:** Proposta Tecnica Documentata (Nessun Ricalcolo Pesante Eseguito)  
**Riferimento:** Sprint 3 Task 4 — Audit dell'Uniformità Metodologica LUFS  

---

## 1. Executive Summary & Diagnostica della Discrepanza

Durante l'audit del codice di `scripts/sprint3/evaluate_v0p_v1_ablation.py` è emersa una discrepanza metodologica fondamentale nella gestione del target di loudness (`target_lufs`) tra la fase di ottimizzazione con l'algoritmo Adam e la successiva fase di valutazione della loss finale IIR.

### Meccanismo Tecnico dell'Incoerenza Storica (V0p / V1 / V1b)
1. **Fase di Ottimizzazione (`optimize_multi_chunk`):**
   - Alla riga 297 di `evaluate_v0p_v1_ablation.py`, la chiamata all'ottimizzatore veniva effettuata omettendo il parametro esplicito `target_lufs`:
     ```python
     opt = AutoMasteringOptimizer(..., target_lufs=None, ...)
     z_opt, _, _ = opt.optimize_multi_chunk(mix_chunks, ref_chunks, ...)
     ```
   - All'interno di `python/ddsp_mastering/optimize/optimizer.py` (righe 156–168), quando `target_lufs is None`:
     - In modalità **`same_track`**, il codice imponeva come fallback fisso il valore nominale **`-14.00 LUFS`** per tutti i 30 step di discesa gradiente Adam.
     - In modalità **`stylistic`**, l'ottimizzatore calcolava la media aritmetica dei LUFS dei 4 chunk reference estratti (`mean_4chunks_ref_lufs`).
2. **Fase di Valutazione Loss IIR (`_eval_loss_iir`):**
   - Immediatamente dopo i 30 step di ottimizzazione, per misurare la convergenza e la metrica di perdita finale, `evaluate_v0p_v1_ablation.py` (riga 298/466) calcolava la loss IIR passando come target:
     ```python
     target_lufs = float(ref_bm["integrated_lufs"])  # LUFS Integrato Full-Track BS.1770-5
     ```
   - In `same_track`, dove le tracce reference masterizzate commerciali hanno un volume tipicamente compreso tra **`-6.90 LUFS`** e **`-7.29 LUFS`**, Adam è stato forzato ad ottimizzare la catena DDSP verso `-14.00 LUFS` (un offset di circa **$-7.0\text{ LU}$**), per poi essere penalizzato nella valutazione IIR finale rispetto ad un target non armonizzato.

---

## 2. Quantificazione Quantitativa su Tutte le 6 Sessioni

La seguente tabella riporta i valori esatti estratti mediante esecuzione analitica non invasiva (`scratch/run_task4_lufs.py` / ITU-R BS.1770-5) su tutte le sessioni:

| Session ID | Modalità | Target Adam Storico ($\text{LUFS}_{\text{opt}}$) | Media 4 Chunk Ref ($\text{LUFS}_{\text{chunks}}$) | Ref Full-Track ($\text{LUFS}_{\text{full}}$) | $\Delta\text{LUFS}_{\text{opt - full}}$ (dB) | $\Delta\text{LUFS}_{\text{chunks - full}}$ (dB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`track01`** | `same_track` | **-14.00** | -6.94 | -7.29 | **-6.71** | +0.35 |
| **`track02`** | `same_track` | **-14.00** | -7.15 | -7.22 | **-6.78** | +0.07 |
| **`track03`** | `same_track` | **-14.00** | -6.63 | -6.90 | **-7.10** | +0.27 |
| **`track01_style`** | `stylistic` | -19.92 | -19.92 | -17.99 | **-1.93** | -1.93 |
| **`track02_style`** | `stylistic` | -9.37 | -9.37 | -9.04 | **-0.33** | -0.33 |
| **`track03_style`** | `stylistic` | -6.80 | -6.80 | -6.46 | **-0.34** | -0.34 |

### Analisi dei Risultati
- **Sessioni `same_track` (`track01`, `track02`, `track03`):**  
  La discrepanza tra l'obiettivo di discesa di Adam e la metrica di riferimento full-track è imponente: tra **$-6.71\text{ LU}$** e **$-7.10\text{ LU}$**. Durante l'ottimizzazione, il termine `weight_loudness * (pred_lufs - (-14.0))^2` ha esercitato una spinta violenta verso il basso sul Makeup Gain e sui parametri dinamici. Al contrario, la differenza tra la media dei 4 chunk reference e la reference full-track è trascurabile ($\le 0.35\text{ LU}$), dimostrando che i 4 chunk selezionati sono altamente rappresentativi del livello globale del brano.
- **Sessioni `stylistic` (`track01_style`, `track02_style`, `track03_style`):**  
  In `stylistic`, l'uso di `mean_4chunks_ref_lufs` ha garantito una coerenza molto maggiore con la reference. Su `track02_style` e `track03_style`, l'errore è minimo ($\le 0.34\text{ LU}$). Su `track01_style`, la natura eterogenea dei chunk reference non correlati ha generato un offset moderato di $-1.93\text{ LU}$.

---

## 3. Specifiche del Protocollo Armonizzato (V0p / V1 / V1b / V2)

Per garantire la perfetta comparabilità scientifica cross-variante e rimuovere qualsiasi artefatto dovuto al disallineamento del target di loudness, si definisce il seguente protocollo tecnico armonizzato:

### A. Definizione Univoca del Target di Loudness
Il target di loudness deve essere **univocamente ancorato alla reference full-track ITU-R BS.1770-5**:
$$\text{target\_lufs} = \text{float}(\text{ref\_bm}[\text{"integrated\_lufs"}]) = \text{calculate\_integrated\_loudness\_bs1770}(\text{ref\_full\_64}, 48000.0)$$
Tale valore scalare deve essere passato:
1. Come argomento esplicito `target_lufs=target_lufs` ad `AutoMasteringOptimizer.optimize_multi_chunk(...)`;
2. Come argomento esplicito `target_lufs=target_lufs` ad ogni funzione di valutazione loss IIR intermedia e finale (`_eval_loss_iir`).

### B. Vincoli e Parametri Operativi Immodificabili
1. **Chunk Topology:**
   - Esattamente **4 chunk da $3.0\text{ s}$** ($144\,000$ campioni a $48\text{ kHz}$).
   - Pre-roll contestuale di **$2.5\text{ s}$** ($120\,000$ campioni) per tutti i chunk di mix e reference.
   - Compensazione deterministica della latenza di pipeline ($D = 159$ campioni).
2. **Selezione dei Chunk:**
   - In `same_track`: selezione basata su `mix_aligned`, applicando coordinate temporalmente identiche a mix e reference (`mix_start + s_idx : mix_start + e_idx`).
   - In `stylistic`: selezione indipendente mediante due istanze separate di `MultiChunkSelector` (`mix_sel` e `ref_sel`). **Le coordinate selezionate devono essere congelate e condivise identicamente tra tutte le varianti (`V0p`, `V1`, `V1b`, `V2`).**
3. **Ottimizzatore Adam:**
   - Learning rate: $\alpha = 0.04$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.
   - Numero di step: $N = 30$.
   - Regime ibrido: FFT-based per gli step $1 \dots 20$, switch ad IIR analitico accelerato per gli step $21 \dots 30$.
4. **Rendering Finale:**
   - Rendering full-track continuo offline mediante motore C++ con svuotamento code IIR/delay (`render_offline_with_tail_flush`).
   - Valutazione benchmark oggettiva mediante `MasteringBenchmarkSuite`.

---

## 4. Valutazione Tecnica sulla Necessità di Ricalcolo

1. **Integrità degli Artefatti Storici:**  
   Gli artefatti storici `SPRINT3_V0P_V1_ABLATION_RESULTS.json` e `SPRINT3_V0P_V1_REPORT.md` rappresentano lo stato di fatto congelato della Milestone Sprint 3 Step V1. **Non devono essere sovrascritti né alterati retroattivamente.**
2. **Impatto sul Confronto V1 vs V2:**  
   Il runner `evaluate_v2_ablation.py` include già al suo interno:
   - La variante di controllo **`v0p` aggiornata**, eseguita con il protocollo armonizzato (`target_lufs = ref_bm["integrated_lufs"]`);
   - La variante sperimentale **`v2`**, eseguita con lo stesso identico target armonizzato.
   In questo modo, il confronto scientifico tra la baseline di riferimento e `V2` è **immediatamente e perfettamente equo all'interno di `evaluate_v2_ablation.py`**.
3. **Opportunità di un Benchmark a 4 Varianti Armonizzato (`V0p` / `V1` / `V1b` / `V2`):**  
   - Per le 3 tracce `same_track`, la discrepanza di $\approx 7\text{ dB}$ ha impattato principalmente il livello assoluto di uscita e il punto di lavoro del compressore/limiter. Tuttavia, il delta relativo tra `V1` e `V1b` (l'effetto anti-asprezza di `weight_de_harsh=0.15`) si è manifestato a parità di condizioni di training.
   - Qualora il comitato di revisione richieda una matrice quadri-variante perfettamente armonizzata, la presente proposta fornisce la specifica formale per eseguirla in un unico job batch deterministico e riproducibile, senza alcuna incertezza metodologica.
