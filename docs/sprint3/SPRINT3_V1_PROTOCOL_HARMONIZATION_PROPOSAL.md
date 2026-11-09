# Proposta Tecnica di Armonizzazione del Protocollo di Valutazione (V0p / V1 / V1b / V2)

**Autore:** Antigravity Engineering  
**Data:** 9 Novembre 2026  
**Stato:** Proposta Tecnica Documentata (Nessun Ricalcolo Pesante Eseguito)  
**Riferimento:** Sprint 3 Task 4 — Audit dell'Uniformità Metodologica LUFS  

---

## 1. Executive Summary & Nota di Rettifica Diagnostica

L'audit metodologico del codice storico `scripts/sprint3/evaluate_v0p_v1_ablation.py` e di `python/ddsp_mastering/optimize/optimizer.py` (righe 158–186) chiarisce in modo definitivo il funzionamento del target di loudness (`target_lufs`) nella discesa gradiente Adam rispetto alla successiva valutazione IIR:

### Meccanismo Reale dell'Ottimizzazione Storica (V0p / V1 / V1b)
1. **Fase di Ottimizzazione (`optimize_multi_chunk`):**
   - Nel codice storico, `opt.optimize_multi_chunk(mix_chunks, ref_chunks, ...)` veniva chiamato passando `target_lufs=None` e fornendo la lista `ref_chunks`.
   - All'interno di `optimizer.py` (righe 158–186), quando `effective_target_lufs is None`:
     ```python
     if ref_chunks is not None and len(ref_chunks) > 0:
         with torch.no_grad():
             lufs_vals = [
                 self.loss_fn.loudness_loss.calculate_lufs(rc).item()
                 for rc in ref_chunks
             ]
             effective_target_lufs = float(sum(lufs_vals) / len(lufs_vals))
     ```
   - Poiché `ref_chunks` era sempre presente sia in `same_track` sia in `stylistic`, l'ottimizzatore storico **non usava il default di `-14.00 LUFS`**, bensì calcolava dinamicamente la media aritmetica dei LUFS dei 4 chunk reference (`historical_opt_target_lufs`).
2. **Fase di Valutazione Loss IIR (`_eval_loss_iir`):**
   - Alla riga 298/466 di `evaluate_v0p_v1_ablation.py`, `_eval_loss_iir` valutava la loss IIR finale passando:
     ```python
     target_lufs = float(ref_bm["integrated_lufs"])  # Full-track reference ITU-R BS.1770
     ```
   - Di conseguenza, lo scarto tra l'obiettivo seguito da Adam sui 4 chunk e la valutazione full-track della reference non era affatto di $7\text{ dB}$, ma era un offset microscopico compreso tra **$+0.07\text{ LU}$ e $+0.35\text{ LU}$** in `same_track` e tra **$-0.33\text{ LU}$ e $-1.93\text{ LU}$** in `stylistic`.

---

## 2. Quantificazione Quantitativa su Tutte le 6 Sessioni

La seguente tabella riporta i valori esatti estratti dall'analisi analitica (`docs/sprint3/task4_lufs_data.json`):

| Session ID | Modalità | Target Adam Storico (`historical_opt_target_lufs`) | Media 4 Chunk BS.1770 (`mean_4chunks_bs1770_lufs`) | Ref Full-Track Armonizzato (`harmonized_fulltrack_target_lufs`) | $\Delta\text{LUFS}$ (`historical` vs `harmonized`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`track01`** | `same_track` | **-6.94** | -6.95 | **-7.29** | **+0.35 LU** |
| **`track02`** | `same_track` | **-7.15** | -7.17 | **-7.22** | **+0.07 LU** |
| **`track03`** | `same_track` | **-6.63** | -6.62 | **-6.90** | **+0.27 LU** |
| **`track01_style`** | `stylistic` | **-19.92** | -19.78 | **-17.99** | **-1.93 LU** |
| **`track02_style`** | `stylistic` | **-9.37** | -9.37 | **-9.04** | **-0.33 LU** |
| **`track03_style`** | `stylistic` | **-6.80** | -6.81 | **-6.46** | **-0.34 LU** |

### Analisi dei Risultati
- **Sessioni `same_track` (`track01`, `track02`, `track03`):**  
  La deviazione tra la media dei 4 chunk reference e la reference full-track è inferiore a **$0.35\text{ LU}$**. Questo conferma che la selezione dei 4 chunk storici era già estremamente rappresentativa della dinamica complessiva del brano.
- **Sessioni `stylistic` (`track01_style`, `track02_style`, `track03_style`):**  
  Su `track02_style` e `track03_style` lo scarto è minimo ($\le 0.34\text{ LU}$). Su `track01_style`, la reference stilistica non correlata presentava una differenza di $-1.93\text{ LU}$ tra i chunk e il brano intero.

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
   - Per le 3 tracce `same_track`, come accertato nell'Audit di Fase 1.2, lo scarto storico $\Delta\text{LUFS}$ tra l'ottimizzazione Adam (media chunk reference) e il target full-track era confinato a un valore minimo ($\le +0.35\text{ LU}$, e non $-14.00\text{ LUFS}$). Tuttavia, il protocollo armonizzato azzera rigorosamente anche questa minima discrepanza, garantendo assoluta identità metodologica tra funzione di costo di training e benchmark full-track continuo. Il delta relativo tra `V1` e `V1b` (l'effetto anti-asprezza di `weight_de_harsh=0.15`) si è manifestato a parità assoluta di condizioni di training.
   - Qualora il comitato di revisione richieda una matrice quadri-variante perfettamente armonizzata, la presente proposta fornisce la specifica formale per eseguirla in un unico job batch deterministico e riproducibile, senza alcuna incertezza metodologica.
