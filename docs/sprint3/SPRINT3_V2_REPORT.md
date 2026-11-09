# AutoMastering — Report Tecnico di Ablation Sprint 3: Step V2 (Loss Transienti/Inviluppo e PLR Reference-Aware)

**Candidato:** Giovanni Antonecchia  
**Corso di Laurea:** Master's Thesis in Sound and Music Engineering — Politecnico di Milano (DEIB)  
**Dataset di Riferimento:** Human Validation Sprint 2 (3 brani: `track01`, `track02`, `track03`; 2 modalità: `same_track`, `stylistic`)  
**Artefatto Dati Grezzi:** [`docs/sprint3/SPRINT3_V2_ABLATION_RESULTS.json`](./SPRINT3_V2_ABLATION_RESULTS.json)  
**Roadmap di Riferimento:** `docs/AutoMastering_Audit_Roadmap_Antigravity.md` (§10, Sprint 3)  
**Specifica Tecnica:** [`docs/sprint3/SPRINT3_TECHNICAL_PROPOSAL.md`](./SPRINT3_TECHNICAL_PROPOSAL.md) (§2.C)  
**Data:** Novembre 2026  

---

## 1. Obiettivo Sperimentale e Metodologia di Isolamento Incrementale

In accordo con la roadmap approvata e il principio di isolamento incrementale dello Sprint 3:
$$\mathbf{V0} \longrightarrow \mathbf{V0p} \longrightarrow \mathbf{V1} / \mathbf{V1b} \longrightarrow \mathbf{V2} \longrightarrow \mathbf{V3a} \longrightarrow \mathbf{V3b} \longrightarrow \mathbf{V3c} \longrightarrow \mathbf{V4}$$

Il presente report documenta la validazione e la diagnostica comparativa dello **Step V2**, il cui scopo è dotare la funzione di costo di sensibilità alla microdinamica e ai profili di cresta per banda, superando l'approccio rigido basato su un range dinamico a scatola (*box target*):

1. **Baseline di Controllo (`V0p`):** Pre-roll di $2.5\text{ s}$ per chunk con warm-up dello stato IIR e dei rivelatori dei compressori, con funzione di costo Sprint 2 armonizzata con il `target_lufs` full-track ITU-R BS.1770-4/5 della traccia di riferimento.
2. **Variante `V2` (Transienti multibanda + PLR Reference-Aware):**
   - **`TransientEnvelopeLoss` ($\lambda_{\text{trans}} = 0.2$, $\beta_{\text{atk}} = 0.5$):** scomposizione Linkwitz-Riley del 4° ordine (LR4) a fase zero in 3 bande, inviluppi veloci (5 ms) e lenti (50 ms), fattore di cresta per banda e derivata positiva di attacco.
   - **`ReferenceAwarePLRLoss` ($	ext{plr\_mode} = \text{'reference\_aware'}$ / $\text{'reference\_aware\_same'}$):** picco soft di Boltzmann ($eta = 2.0$, $p = 8$, blocchi da $10\text{ ms}$) con penalità Huber su riferimento in `same_track` e dead-zone $\pm 1.0\text{ dB}$ con barriera di floor in `stylistic`.

---

## 2. Formulazioni Matematiche e Correzioni Ingegneristiche Applicate (Fix A–G)

Tutte le 7 correzioni formali sono state incorporate e validate dalla suite di unit test (`tests/test_sprint3_v2_transient_plr.py`, 100% green su 23/23 test complessivi):
- **Fix A:** Ordine corretto delle operazioni nel picco soft di Boltzmann (radice p-esima prima del clamp).
- **Fix B:** Inclusione del blocco parziale residuo normalizzato.
- **Fix C:** Allineamento esatto di latenza (D=159 campioni) nel PLR same_track.
- **Fix D:** Buffer esteso di pre-roll per il crossover LR4.
- **Fix E:** Inclusione dei parametri di transient/PLR nella firma dell'ottimizzatore.
- **Fix F:** Halo da 128 campioni nello streaming del True Peak.
- **Fix G:** Protocollo armonizzato con target LUFS full-track e Percorso A per tutte le sessioni.

---

## 3. Tabella Sinottica Comparativa: `V0p` (Controllo) vs `V2` (Transienti & PLR)

| Sessione / Modalità | Variante | Loss Iniziale $z_0$ | Loss Finale $z_{\text{opt}}$ | Riduzione Loss | Integrated LUFS | PLR (dB) | True Peak (dBTP) | TP Viol. (dB) | Mid/Side $\Delta$ (dB) | Distanza Spettrale (SC / ASD) | LSD / ERB Tot |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`track01`**<br>*(same_track)* | **`V0p`**<br>**`V2`** | $6.3866$<br>$7.3751$ | $2.8224$<br>**$3.3483$** | $-55.81\%$<br>**$-54.60\%$** | $-10.51$<br>$-10.51$ | $10.21$<br>$10.21$ | $0.26$<br>$0.15$ | $0.56$<br>$0.45$ | $0.44$<br>$0.40$ | 0.4057<br>0.4055 | 6.79 dB<br>6.80 dB |
| **`track02`**<br>*(same_track)* | **`V0p`**<br>**`V2`** | $14.2243$<br>$15.3688$ | $10.3552$<br>**$11.5401$** | $-27.20\%$<br>**$-24.91\%$** | $-18.04$<br>$-18.05$ | $17.74$<br>$17.75$ | $-0.07$<br>$0.04$ | $0.23$<br>$0.34$ | $0.35$<br>$0.37$ | 0.7932<br>0.7935 | 11.21 dB<br>11.21 dB |
| **`track03`**<br>*(same_track)* | **`V0p`**<br>**`V2`** | $10.2299$<br>$11.3963$ | $6.2350$<br>**$7.2179$** | $-39.05\%$<br>**$-36.66\%$** | $-13.50$<br>$-13.50$ | $13.20$<br>$13.20$ | $-0.09$<br>$-0.04$ | $0.21$<br>$0.26$ | $1.41$<br>$1.40$ | 0.6121<br>0.6125 | 9.47 dB<br>9.45 dB |
| **`track01_style`**<br>*(stylistic)* | **`V0p`**<br>**`V2`** | $20.6210$<br>$17.4990$ | $6.9088$<br>**$3.2407$** | $-66.50\%$<br>**$-81.48\%$** | $-17.83$<br>$-17.85$ | $15.42$<br>$16.01$ | $-2.42$<br>$-1.63$ | $0.00$<br>$0.00$ | $0.76$<br>$0.82$ | —<br>— | —<br>— |
| **`track02_style`**<br>*(stylistic)* | **`V0p`**<br>**`V2`** | $152.7334$<br>$139.1445$ | $77.8123$<br>**$67.5392$** | $-49.05\%$<br>**$-51.46\%$** | $-18.05$<br>$-18.05$ | $17.75$<br>$17.75$ | $-0.07$<br>$-0.06$ | $0.23$<br>$0.24$ | $2.38$<br>$2.37$ | —<br>— | —<br>— |
| **`track03_style`**<br>*(stylistic)* | **`V0p`**<br>**`V2`** | $119.4243$<br>$102.9182$ | $51.9053$<br>**$40.4513$** | $-56.54\%$<br>**$-60.70\%$** | $-13.52$<br>$-13.49$ | $13.22$<br>$13.19$ | $0.06$<br>$-0.06$ | $0.36$<br>$0.24$ | $1.06$<br>$1.07$ | —<br>— | —<br>— |

---

## 4. Diagnostica Dettagliata per Sessione

### 4.1 Sessione `track01` (same_track)
- **Target LUFS:** `-7.29` LUFS
- **Tempo di Calcolo:** `947.1 s`
- **Loss V0p:** `6.3866` -> `2.8224` (-55.81%)
- **Loss V2:** `7.3751` -> `3.3483` (-54.60%)
- **Loudness Full-Track:** Pred `-10.51` LUFS (Target `-7.29` LUFS, Delta `3.22` LU)
- **True Peak:** `0.15` dBTP (Soglia -0.30 dBTP, Violazione: `0.45` dB)
- **PLR:** Pred `10.21` dB (Target `7.69` dB, Delta `2.51` dB)

### 4.2 Sessione `track02` (same_track)
- **Target LUFS:** `-7.22` LUFS
- **Tempo di Calcolo:** `905.2 s`
- **Loss V0p:** `14.2243` -> `10.3552` (-27.20%)
- **Loss V2:** `15.3688` -> `11.5401` (-24.91%)
- **Loudness Full-Track:** Pred `-18.05` LUFS (Target `-7.22` LUFS, Delta `10.83` LU)
- **True Peak:** `0.04` dBTP (Soglia -0.30 dBTP, Violazione: `0.34` dB)
- **PLR:** Pred `17.75` dB (Target `7.84` dB, Delta `9.92` dB)

### 4.3 Sessione `track03` (same_track)
- **Target LUFS:** `-6.90` LUFS
- **Tempo di Calcolo:** `948.3 s`
- **Loss V0p:** `10.2299` -> `6.2350` (-39.05%)
- **Loss V2:** `11.3963` -> `7.2179` (-36.66%)
- **Loudness Full-Track:** Pred `-13.50` LUFS (Target `-6.90` LUFS, Delta `6.60` LU)
- **True Peak:** `-0.04` dBTP (Soglia -0.30 dBTP, Violazione: `0.26` dB)
- **PLR:** Pred `13.20` dB (Target `7.34` dB, Delta `5.86` dB)

### 4.4 Sessione `track01_style` (stylistic)
- **Target LUFS:** `-17.99` LUFS
- **Tempo di Calcolo:** `872.1 s`
- **Loss V0p:** `20.6210` -> `6.9088` (-66.50%)
- **Loss V2:** `17.4990` -> `3.2407` (-81.48%)
- **Loudness Full-Track:** Pred `-17.85` LUFS (Target `-17.99` LUFS, Delta `0.14` LU)
- **True Peak:** `-1.63` dBTP (Soglia -0.30 dBTP, Violazione: `0.00` dB)
- **PLR:** Pred `16.01` dB (Target `11.35` dB, Delta `4.66` dB)

### 4.5 Sessione `track02_style` (stylistic)
- **Target LUFS:** `-9.04` LUFS
- **Tempo di Calcolo:** `870.0 s`
- **Loss V0p:** `152.7334` -> `77.8123` (-49.05%)
- **Loss V2:** `139.1445` -> `67.5392` (-51.46%)
- **Loudness Full-Track:** Pred `-18.05` LUFS (Target `-9.04` LUFS, Delta `9.01` LU)
- **True Peak:** `-0.06` dBTP (Soglia -0.30 dBTP, Violazione: `0.24` dB)
- **PLR:** Pred `17.75` dB (Target `10.56` dB, Delta `7.19` dB)

### 4.6 Sessione `track03_style` (stylistic)
- **Target LUFS:** `-6.46` LUFS
- **Tempo di Calcolo:** `857.6 s`
- **Loss V0p:** `119.4243` -> `51.9053` (-56.54%)
- **Loss V2:** `102.9182` -> `40.4513` (-60.70%)
- **Loudness Full-Track:** Pred `-13.49` LUFS (Target `-6.46` LUFS, Delta `7.03` LU)
- **True Peak:** `-0.06` dBTP (Soglia -0.30 dBTP, Violazione: `0.24` dB)
- **PLR:** Pred `13.19` dB (Target `8.32` dB, Delta `4.87` dB)

---

## 5. Conclusioni dello Step V2 e Prossimi Passi (Sprint 3)

1. **Validazione Funzionale di V2:** La combinazione di `TransientEnvelopeLoss` e `ReferenceAwarePLRLoss` con iperparametri della Specifica Tecnica v3 è perfettamente stabile su tutte e 6 le sessioni.
2. **Confronto Armonizzato a 4 Varianti:** I risultati di `V2` confluiscono nel report comparativo quadrilaterale `docs/sprint3/SPRINT3_HARMONIZED_COMPARISON_REPORT.md` insieme a `V0p`, `V1` e `V1b`.
3. **Transizione verso V3a:** Il guardrail di True Peak evidenzia la necessità di uno stadio di peak control e true-peak limiting dedicato pianificato nello Step V3a.
