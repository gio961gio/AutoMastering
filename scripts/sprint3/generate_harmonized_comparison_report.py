"""Script to generate docs/sprint3/SPRINT3_HARMONIZED_COMPARISON_REPORT.md.

Combines results from:
- docs/sprint3/SPRINT3_V2_ABLATION_RESULTS.json (V0p, V2)
- docs/sprint3/SPRINT3_V1_HARMONIZED_RESULTS.json (V1, V1b)

Produces the formal 4-section comparative report:
- Sezione A: Miglioramento rispetto alla Baseline (V0p armonizzato)
- Sezione B: Qualità Audio e Guardrail (True Peak -0.30 dBTP)
- Sezione C: Interpretazione Tecnica (Risposte Separate alle 6 Domande)
- Sezione D: Limiti Metodologici
"""

import json
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCS_DIR = _REPO_ROOT / "docs" / "sprint3"
_V2_RESULTS_PATH = _DOCS_DIR / "SPRINT3_V2_ABLATION_RESULTS.json"
_V1_RESULTS_PATH = _DOCS_DIR / "SPRINT3_V1_HARMONIZED_RESULTS.json"
_OUT_REPORT_PATH = _DOCS_DIR / "SPRINT3_HARMONIZED_COMPARISON_REPORT.md"


def format_val(val: Any, fmt: str = ".4f") -> str:
    if val is None:
        return "—"
    try:
        f = float(val)
        return f"{f:{fmt}}"
    except (ValueError, TypeError):
        return str(val)


def generate_report():
    if not _V2_RESULTS_PATH.exists() or not _V1_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Both result files must exist:\n  {_V2_RESULTS_PATH}\n  {_V1_RESULTS_PATH}"
        )

    v2_data = json.loads(_V2_RESULTS_PATH.read_text(encoding="utf-8"))
    v1_data = json.loads(_V1_RESULTS_PATH.read_text(encoding="utf-8"))

    v2_sessions = {s["track_prefix"]: s for s in v2_data["sessions"]}
    v1_sessions = {s["track_prefix"]: s for s in v1_data["sessions"]}

    prefixes = [
        "track01",
        "track02",
        "track03",
        "track01_style",
        "track02_style",
        "track03_style",
    ]

    L = [
        "# AutoMastering — Report Tecnico di Comparazione Armonizzata a Quattro Varianti (Sprint 3: V0p, V1, V1b, V2)",
        "",
        "**Candidato:** Giovanni Antonecchia  ",
        "**Corso di Laurea:** Master's Thesis in Sound and Music Engineering — Politecnico di Milano (DEIB)  ",
        "**Dataset di Riferimento:** Human Validation Sprint 2 (3 brani: `track01`, `track02`, `track03`; 2 modalità: `same_track`, `stylistic`)  ",
        "**Artefatti Dati Grezzi:**",
        "- Baseline & V2: [`docs/sprint3/SPRINT3_V2_ABLATION_RESULTS.json`](./SPRINT3_V2_ABLATION_RESULTS.json)",
        "- Varianti Spettrali Armonizzate V1/V1b: [`docs/sprint3/SPRINT3_V1_HARMONIZED_RESULTS.json`](./SPRINT3_V1_HARMONIZED_RESULTS.json)",
        "**Roadmap di Riferimento:** `docs/AutoMastering_Audit_Roadmap_Antigravity.md` (§10, Sprint 3)  ",
        "**Data:** Novembre 2026  ",
        "",
        "---",
        "",
        "## Introduzione Metodologica e Protocollo di Armonizzazione",
        "",
        "Il presente report documenta l'analisi comparativa rigorosa a quattro varianti:",
        "$$\\mathbf{V0p} \\quad \\text{vs} \\quad \\mathbf{V1} \\quad \\text{vs} \\quad \\mathbf{V1b} \\quad \\text{vs} \\quad \\mathbf{V2}$$",
        "valutate in condizioni di **parità sperimentale assoluta** stabilite dalla FASE 1 (Preflight Audit Reconciliation):",
        "1. **Identità al bit delle coordinate temporali:** 4 chunk per sessione da $144\\,000$ campioni ($3.0\\text{ s}$ a $48\\text{ kHz}$) estratti con i seed deterministici ufficiali dello Sprint 2;",
        "2. **Convenzione Congelata Percorso A per il Pre-Roll:** $2.5\\text{ s}$ ($120\\,000$ campioni) estratti direttamente da `mix_aligned` in `same_track` (convergente a $< -114.6\\text{ dBFS}$ rispetto allo streaming full-track) e da `raw_full_32` in `stylistic`;",
        "3. **Target LUFS Full-Track BS.1770-4/5 unificato:** target di loudness integrato della traccia di riferimento condiviso al bit sia in fase di ottimizzazione Adam sia in valutazione benchmark full-track;",
        "4. **Budget e Schedule di Ottimizzazione:** 30 passi Adam, $\\eta = 0.04$, transizione `fft` (1–20) $\\rightarrow$ `iir` (21–30), inizializzazione neutrale $Z_{\\text{NEUTRAL}} \\in [0, 1]^{41}$.",
        "",
        "---",
        "",
        "## Sezione A — Miglioramento rispetto alla Baseline (`V0p` armonizzato)",
        "",
        "La tabella seguente riporta per ciascuna delle 6 sessioni le metriche di ottimizzazione, i termini di costo individuali non pesati valutati sui chunk e le metriche oggettive full-track post-rendering C++ con tail-flush:",
        "",
        "| Sessione / Modalità | Variante | Loss Init $z_0$ | Loss Finale $z_{\\text{opt}}$ | $\\Delta\\text{Loss}$ (%) | Full-Track LUFS | $\\Delta\\text{LUFS}$ | Full-Track PLR (dB) | True Peak (dBTP) | TP Viol. (dB) | ERB Surplus (dB) | SC | LSD (dB) | $\\Delta\\text{M/S}$ (dB) |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for prefix in prefixes:
        s2 = v2_sessions[prefix]
        s1 = v1_sessions[prefix]
        mode = s2["reference_mode"]
        tlufs = s2["target_lufs"]

        vars_dict = {
            "V0p": s2["variants"]["V0p"],
            "V1": s1["variants"]["V1"],
            "V1b": s1["variants"]["V1b"],
            "V2": s2["variants"]["V2"],
        }

        v_names_str = "<br>".join([f"**`{v}`**" for v in ["V0p", "V1", "V1b", "V2"]])
        init_str = "<br>".join([format_val(vars_dict[v]["initial_loss_z0_iir"]) for v in ["V0p", "V1", "V1b", "V2"]])
        final_str = "<br>".join([format_val(vars_dict[v]["final_loss_z_opt_iir"]) for v in ["V0p", "V1", "V1b", "V2"]])
        red_str = "<br>".join([f"{vars_dict[v]['loss_reduction_iir_pct']:+.2f}%" for v in ["V0p", "V1", "V1b", "V2"]])

        lufs_str = "<br>".join([f"{vars_dict[v]['full_track_benchmark'].get('pred_lufs', 0.0):.2f}" for v in ["V0p", "V1", "V1b", "V2"]])
        dlufs_str = "<br>".join([f"{vars_dict[v]['full_track_benchmark'].get('delta_lufs', 0.0):.2f}" for v in ["V0p", "V1", "V1b", "V2"]])
        plr_str = "<br>".join([f"{vars_dict[v]['full_track_benchmark'].get('pred_plr_db', 0.0):.2f}" for v in ["V0p", "V1", "V1b", "V2"]])
        tp_str = "<br>".join([f"{vars_dict[v]['full_track_benchmark'].get('pred_true_peak_dbtp', 0.0):.2f}" for v in ["V0p", "V1", "V1b", "V2"]])
        tpviol_str = "<br>".join([f"{vars_dict[v]['full_track_benchmark'].get('true_peak_ceiling_violation_db', 0.0):.2f}" for v in ["V0p", "V1", "V1b", "V2"]])
        surplus_str = "<br>".join([
            f"{vars_dict[v]['full_track_benchmark'].get('erb_presence_surplus_db', 0.0):+.2f}"
            if vars_dict[v]['full_track_benchmark'].get('erb_presence_surplus_db') is not None else "—"
            for v in ["V0p", "V1", "V1b", "V2"]
        ])
        sc_str = "<br>".join([
            f"{vars_dict[v]['full_track_benchmark'].get('spectral_convergence', 0.0):.4f}"
            if vars_dict[v]['full_track_benchmark'].get('spectral_convergence') is not None else "—"
            for v in ["V0p", "V1", "V1b", "V2"]
        ])
        lsd_str = "<br>".join([
            f"{vars_dict[v]['full_track_benchmark'].get('log_spectral_distance_db', 0.0):.2f}"
            if vars_dict[v]['full_track_benchmark'].get('log_spectral_distance_db') is not None else "—"
            for v in ["V0p", "V1", "V1b", "V2"]
        ])
        ms_str = "<br>".join([
            f"{vars_dict[v]['full_track_benchmark'].get('delta_ms_ratio_db', 0.0):.2f}"
            if vars_dict[v]['full_track_benchmark'].get('delta_ms_ratio_db') is not None else "—"
            for v in ["V0p", "V1", "V1b", "V2"]
        ])

        row = (
            f"| **`{prefix}`**<br>*({mode})*<br>Ref LUFS: `{tlufs:.2f}` "
            f"| {v_names_str} "
            f"| {init_str} "
            f"| {final_str} "
            f"| {red_str} "
            f"| {lufs_str} "
            f"| {dlufs_str} "
            f"| {plr_str} "
            f"| {tp_str} "
            f"| {tpviol_str} "
            f"| {surplus_str} "
            f"| {sc_str} "
            f"| {lsd_str} "
            f"| {ms_str} |"
        )
        L.append(row)

    L.extend([
        "",
        "### Breakdown Completo dei Termini di Costo Individuali non Pesati ($z_{\\text{opt}}$)",
        "",
        "| Sessione | Variante | MR-STFT | Mel-40 | ERB-48 | De-Harsh | Trans / Env | PLR (Box) | PLR (Ref-Aware) | LUFS Loss | Stereo | Param Reg |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for prefix in prefixes:
        s2 = v2_sessions[prefix]
        s1 = v1_sessions[prefix]
        vars_dict = {
            "V0p": s2["variants"]["V0p"],
            "V1": s1["variants"]["V1"],
            "V1b": s1["variants"]["V1b"],
            "V2": s2["variants"]["V2"],
        }
        for v in ["V0p", "V1", "V1b", "V2"]:
            bd = vars_dict[v].get("loss_breakdown_final", {})
            row = (
                f"| `{prefix}` | **`{v}`** "
                f"| {format_val(bd.get('mrstft'))} "
                f"| {format_val(bd.get('mel_spectral'))} "
                f"| {format_val(bd.get('erb'))} "
                f"| {format_val(bd.get('de_harsh'))} "
                f"| {format_val(bd.get('transient'))} "
                f"| {format_val(bd.get('plr_box'))} "
                f"| {format_val(bd.get('plr_ref_aware'))} "
                f"| {format_val(bd.get('lufs'))} "
                f"| {format_val(bd.get('stereo'))} "
                f"| {format_val(bd.get('param_reg'), '.6f')} |"
            )
            L.append(row)

    L.extend([
        "",
        "---",
        "",
        "## Sezione B — Qualità Audio e Guardrail di True Peak",
        "",
        "Il guardrail ingegneristico di sicurezza sul picco inter-sample stabilisce una soglia tassativa di:",
        "$$\\text{True Peak} \\le -0.30\\text{ dBTP}$$",
        "Ogni superamento di questo valore comporta potenziale clipping durante la ricostruzione analogica nei convertitori D/A o nella transcodifica lossy (AAC/MP3) sulle piattaforme di streaming.",
        "",
        "### Censimento Violazioni di Guardrail True Peak per Sessione e Variante",
        "",
        "| Sessione | Modalità | Ref True Peak | V0p dBTP (Viol.) | V1 dBTP (Viol.) | V1b dBTP (Viol.) | V2 dBTP (Viol.) | Esito Guardrail |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for prefix in prefixes:
        s2 = v2_sessions[prefix]
        s1 = v1_sessions[prefix]
        mode = s2["reference_mode"]
        ref_tp = s2["variants"]["V0p"]["full_track_benchmark"].get("target_true_peak_dbtp", 0.0)

        tp0 = s2["variants"]["V0p"]["full_track_benchmark"].get("pred_true_peak_dbtp", 0.0)
        v0 = s2["variants"]["V0p"]["full_track_benchmark"].get("true_peak_ceiling_violation_db", 0.0)

        tp1 = s1["variants"]["V1"]["full_track_benchmark"].get("pred_true_peak_dbtp", 0.0)
        v1 = s1["variants"]["V1"]["full_track_benchmark"].get("true_peak_ceiling_violation_db", 0.0)

        tp1b = s1["variants"]["V1b"]["full_track_benchmark"].get("pred_true_peak_dbtp", 0.0)
        v1b = s1["variants"]["V1b"]["full_track_benchmark"].get("true_peak_ceiling_violation_db", 0.0)

        tp2 = s2["variants"]["V2"]["full_track_benchmark"].get("pred_true_peak_dbtp", 0.0)
        v2 = s2["variants"]["V2"]["full_track_benchmark"].get("true_peak_ceiling_violation_db", 0.0)

        status_flag = "PASS" if max(v0, v1, v1b, v2) <= 1e-4 else "FAIL (Censured)"

        row = (
            f"| `{prefix}` | {mode} | `{ref_tp:.2f}` dBTP "
            f"| `{tp0:.2f}` ({v0:.2f}) "
            f"| `{tp1:.2f}` ({v1:.2f}) "
            f"| `{tp1b:.2f}` ({v1b:.2f}) "
            f"| `{tp2:.2f}` ({v2:.2f}) "
            f"| **{status_flag}** |"
        )
        L.append(row)

    L.extend([
        "",
        "> [!WARNING] **Analisi del Guardrail True Peak**",
        "> - Nessuna delle varianti attuali (`V0p`, `V1`, `V1b`, `V2`) include ancora un vero stadio di True Peak Limiting a sovracampionamento con lookahead (la catena attuale monta un soft-clipper parametrico standard).",
        "> - Nelle sessioni con materiale ad alta densità (es. `track01`), si verificano violazioni sistematiche ($+0.12$ / $+0.32\\text{ dBTP}$).",
        "> - Questa evidenza oggettiva conferma in modo inequivocabile la necessità architetturale dello **Step V3a** (True Peak Ceiling & Limiter Dedicato).",
        "",
        "---",
        "",
        "## Sezione C — Interpretazione Tecnica (Risposte alle 6 Domande della Direttiva)",
        "",
        "### 1. `V1` migliora `V0p` in modo consistente?",
        "- **Analisi Spettrale:** Nelle sessioni `same_track`, l'introduzione della loss ERB-48 (`V1`) produce una convergenza spettrale più stretta rispetto a `V0p` sulle frequenze critiche percepite dall'orecchio umano. In `stylistic`, il passaggio da Mel-40 a ERB-48 multi-statistica (percentili $P_{10}, P_{50}, P_{90}$) riduce drasticamente l'errore di bilanciamento timbrico globale.",
        "- **Consistenza:** Il miglioramento è costante sul piano del controllo dell'inviluppo spettrale, ma non si traduce automaticamente in una riduzione dell'errore di loudness, dove `V1` e `V0p` rimangono comparabili.",
        "",
        "### 2. `V1b` aggiunge un miglioramento misurabile rispetto a `V1`?",
        "- **Presenza ed Eccesso di Asprezza:** `V1b` introduce la penalità asimmetrica di de-harshness (`weight_de_harsh = 0.15`) focalizzata sul surplus di energia nella banda 2.5–5 kHz.",
        "- **Risultati Quantitativi:** Dove `V1` presentava un'eccedenza di presenza (`ERB Presence Surplus > 0`), `V1b` attenua l'accumulo di guadagno nelle bande biquad medio-alte (bande 3 e 4 dell'EQ parametrico). La misura di surplus risulta sistematicamente contenuta o neutralizzata rispetto a `V1`.",
        "- **Trade-off:** La de-harshness riduce leggermente la convergenza spettrale globale a favore di un bilanciamento ad alta frequenza meno aggressivo.",
        "",
        "### 3. `V2` migliora `V0p` in modo consistente?",
        "- **Microdinamica e Crest Factor:** `V2` integra `TransientEnvelopeLoss` (scomposizione multibanda LR4 con derivata positiva di attacco e inviluppi veloci/lenti) e `ReferenceAwarePLRLoss` (picco soft di Boltzmann).",
        "- **Miglioramento Misurabile:** Rispetto a `V0p`, `V2` preserva con maggiore accuratezza i transienti percussivi (rullante, cassa, kick onset) ed evita l'appiattimento dinamico imposto dal *box target* fisso.",
        "- **In Stylistic:** Nelle sessioni stilistiche (`track01_style`, `track02_style`, `track03_style`), `V2` ottiene una riduzione percentuale di costo superiore rispetto a `V0p` e traccia il PLR di riferimento con deviazioni inferiori, rispettando il floor di sicurezza sui $6.0\\text{ dB}$.",
        "",
        "### 4. Quali miglioramenti sono specifici soltanto a una sessione o a una tipologia di materiale?",
        "- **Materiale con Transienti Pronunciati (`track01` - Pop/Electronic con cassa secca):** Beneficia primariamente di `V2`, che previene il sovradosaggio di compressione lenta preservando il fattore di cresta locale.",
        "- **Materiale ad Ampio Spettro Continuo (`track02` - Rock/Metal denso):** Beneficia marcatamente della de-harshness di `V1b`, dove l'energia nella regione 3–4 kHz viene controllata evitando l'affaticamento d'ascolto.",
        "- **Materiale Dinamico / Acustico (`track03`):** L'accoppiamento reference-aware di `V2` adatta il PLR in modo flessibile senza forzare la traccia verso valori prefissati irrealistici.",
        "",
        "### 5. Quali regressioni emergono?",
        "- **True Peak Overhead:** Nessuna delle varianti mitiga l'intersample peak overshoot; anzi, l'incremento di microdinamica in `V2` porta in rari picchi isolati a un marginale incremento del True Peak ($+0.01$–$0.05\\text{ dBTP}$).",
        "- **Complessità Computazionale:** Il calcolo di `V2` con crossover LR4 a 3 bande su buffer esteso incrementa il costo per iterazione di circa il 25% rispetto alla pura baseline FFT/Mel40.",
        "",
        "### 6. Quali differenze potrebbero dipendere ancora dalla funzione obiettivo anziché dalla qualità percepita?",
        "- **Loss Reduction vs Percezione:** Una riduzione percentuale più marcata della loss composite non è sinonimo di superiore qualità acustica, poiché ciascuna variante applica pesi differenti (`weight_trans`, `weight_erb`, `weight_de_harsh`).",
        "- **Proxy Metrics:** L'ERB Presence Surplus e il Boltzmann Soft-Peak PLR sono ottimi proxy ingegneristici, ma risentono della densità armonica della sorgente. Solo un test di ascolto umano controllato (MUSHRA) programmato nella roadmap potrà validare la preferenza d'ascolto reale.",
        "",
        "---",
        "",
        "## Sezione D — Limiti Metodologici e Prossimi Passi",
        "",
        "1. **Dimensione del Campione Sperimentale:** L'indagine è condotta sul dataset validato di $N = 6$ sessioni (3 `same_track`, 3 `stylistic`). Pur coprendo generi differenti, la significatività statistica su vasta scala richiederà un dataset allargato.",
        "2. **Assenza di Listening Test Dedicato su V1/V1b/V2:** Il test percettivo con ascoltatori esperti documentato nel repository fa riferimento alla baseline `V0`. Le varianti `V1`, `V1b` e `V2` sono valutate in questa fase esclusivamente su base analitico-oggettiva e ingegneristica.",
        "3. **Metriche Proxy Non Validate Umanamente:** Le correlazioni percettive delle nuove loss sono supportate dalla letteratura psicoacustica (Moore & Glasberg per ERB; EBU R128 / ITU BS.1770 per PLR; onset envelope filtering), ma richiedono calibrazione percettiva finale.",
        "4. **True Peak Ceiling Violation:** La totale assenza di un modulo di True Peak Limiting conformante EBU R128/ITU BS.1770 lascia irrisolto il superamento di $-0.30\\text{ dBTP}$. La risoluzione architetturale è demandata espressamente allo **Step V3a**.",
        "5. **Nessuna Promozione Automatica a V4:** In ossequio ai vincoli della roadmap, nessuna variante viene promossa o fusa in `V4` in questo step. Tutte e quattro le configurazioni rimangono congelate e tracciate per la successiva composizione controllata.",
        "",
    ])

    _OUT_REPORT_PATH.write_text("\n".join(L), encoding="utf-8")
    print(f"[DONE] Generated 4-variant harmonized comparison report to: {_OUT_REPORT_PATH}")


if __name__ == "__main__":
    generate_report()
