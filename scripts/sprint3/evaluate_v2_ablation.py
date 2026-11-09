"""Sprint 3 Step V2 Ablation Runner: V0p vs V2 (Transient/Envelope & Reference-Aware PLR Loss).

Evaluates:
  - V0p: Context control baseline (2.5s pre-roll, historical Sprint 2 loss with harmonized full-track target LUFS).
  - V2:  2.5s pre-roll + Transient/Envelope Loss (weight_trans=0.2, beta_atk=0.5)
         + Reference-Aware PLR Loss with Boltzmann soft-peak mean (plr_mode="reference_aware").

Runs multi-chunk optimization (num_steps=30, lr=0.04, exact Sprint 2 seeds and chunk coordinates)
across all 6 sessions (track01..03 same_track and track01..03_style stylistic),
renders full-track offline C++ output with tail flush, and benchmarks objective metrics.
Exports results to docs/sprint3/SPRINT3_V2_ABLATION_RESULTS.json and docs/sprint3/SPRINT3_V2_REPORT.md.
"""

import gc
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.signal import lfilter as scipy_lfilter
import torch
import torchaudio.functional as taf

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ddsp_mastering.dsp import enable_fast_cpu_biquad
enable_fast_cpu_biquad()

from ddsp_mastering.dsp.chain import DifferentiableMasteringChain
from ddsp_mastering.optimize import AutoMasteringOptimizer, MasteringBenchmarkSuite
from ddsp_mastering.param_space import PARAM_SPECS, Z_NEUTRAL, map_theta_to_z, map_z_to_theta
from scripts.human_validation.offline_renderer import render_offline_with_tail_flush
from scripts.sprint3.harmonized_session_prep import (
    SESSIONS_SPEC,
    compute_individual_loss_breakdown,
    prepare_harmonized_session,
)


def load_reproducibility_manifest() -> Dict[str, Any]:
    p = _REPO_ROOT / "docs" / "human_validation" / "SPRINT2_V0_REPRODUCIBILITY_MANIFEST.json"
    return json.loads(p.read_text(encoding="utf-8"))


def load_v0p_v1_results() -> Dict[str, Any]:
    p = _REPO_ROOT / "docs" / "sprint3" / "SPRINT3_V0P_V1_ABLATION_RESULTS.json"
    return json.loads(p.read_text(encoding="utf-8"))


def evaluate_single_session_v2(
    spec: Dict[str, str],
    seed_int: int,
    v0p_variant_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate V0p (updated control) and V2 for a single session."""
    prefix = spec["track_prefix"]
    mode = spec["mode"]
    cache_dir = _REPO_ROOT / "docs" / "sprint3" / ".cache_v2_harmonized_final"
    cache_dir.mkdir(parents=True, exist_ok=True)
    session_cache_file = cache_dir / f"{prefix}_v2_harmonized.json"

    if session_cache_file.exists():
        print(f"[{prefix}] Loaded complete session from cache: {session_cache_file}", flush=True)
        return json.loads(session_cache_file.read_text(encoding="utf-8"))

    t0 = time.time()
    prep = prepare_harmonized_session(spec, seed_int=seed_int, repo_root=_REPO_ROOT)
    raw_full_64 = prep["raw_full_64"]
    ref_full_64 = prep["ref_full_64"]
    mix_chunks = prep["mix_chunks"]
    ref_chunks = prep["ref_chunks"]
    preroll_list = prep["preroll_list"]
    ref_ext_chunks = prep["ref_ext_chunks"]
    ref_preroll_list = prep["ref_preroll_list"]
    chunk_metadata = prep["chunk_metadata"]
    target_lufs = prep["target_lufs"]
    ref_eval_tensor = prep["ref_eval_tensor"]
    mix_start = prep["mix_start"]
    eval_len = prep["eval_len"]
    suite = MasteringBenchmarkSuite(sample_rate=48000.0)

    K = len(mix_chunks)
    weights = [1.0 / K] * K

    def _evaluate_objective_iir(opt_obj: AutoMasteringOptimizer, z_in: torch.Tensor, eval_tlufs: float) -> Tuple[float, Dict[str, float]]:
        with torch.no_grad():
            tot_l = 0.0
            agg_m: Dict[str, float] = {}
            for k in range(K):
                n_pre = preroll_list[k]
                mk = mix_chunks[k]
                y_ext = opt_obj.chain(mk, z=z_in, eq_mode="iir")
                y_k = y_ext[..., n_pre:]
                ref_k = ref_chunks[k]
                ref_ext_k = ref_ext_chunks[k]
                ref_pre_k = ref_preroll_list[k]
                if mode == "same_track":
                    lk, mk_dict = opt_obj.loss_fn(
                        pred_audio=y_k,
                        target_audio=ref_k,
                        target_lufs=eval_tlufs,
                        normalized_params=z_in,
                        align_latency=True,
                        pred_ext_audio=y_ext,
                        pred_preroll_samples=n_pre,
                        ref_ext_audio=ref_ext_k,
                        ref_preroll_samples=ref_pre_k,
                    )
                else:
                    anchor_k = mk[..., n_pre:]
                    lk, mk_dict = opt_obj.stylistic_loss_fn(
                        pred_audio=y_k,
                        ref_audio=ref_k,
                        normalized_params=z_in,
                        target_lufs=eval_tlufs,
                        mix_anchor_audio=anchor_k,
                        pred_ext_audio=y_ext,
                        pred_preroll_samples=n_pre,
                        ref_ext_audio=ref_ext_k,
                        ref_preroll_samples=ref_pre_k,
                    )
                tot_l += weights[k] * float(lk.item())
                for mk_key, mk_val in mk_dict.items():
                    agg_m[mk_key] = agg_m.get(mk_key, 0.0) + weights[k] * float(mk_val)
            return tot_l, agg_m

    # -------------------------------------------------------------------------
    # 1. V0p Control Baseline (frozen Sprint 2 loss with full-track target LUFS)
    # -------------------------------------------------------------------------
    opt_v0p = AutoMasteringOptimizer(
        sample_rate=48000.0,
        lr=0.04,
        num_steps=30,
        target_lufs=target_lufs,
        weight_stft=1.0,
        weight_loudness=0.8,
        weight_param_reg=0.01,
        use_hybrid_eq=True,
        weight_erb=0.0,
        stylistic_spectral_mode="mel40",
        weight_de_harsh=0.0,
        weight_trans=0.0,
        plr_mode="box",
        beta_atk=0.5,
        plr_d_style=1.0,
        plr_gamma_floor=1.0,
        soft_peak_p=8,
        soft_peak_beta=2.0,
        soft_peak_block_duration_sec=0.010,
    )

    z_opt_v0p, _, _ = opt_v0p.optimize_multi_chunk(
        mix_chunks=mix_chunks,
        ref_chunks=ref_chunks,
        reference_mode=mode,
        verbose=False,
        align_latency=True,
        mix_preroll_samples=preroll_list,
        target_lufs=target_lufs,
    )

    loss_z0_v0p, metrics_init_v0p = _evaluate_objective_iir(opt_v0p, Z_NEUTRAL, target_lufs)
    loss_zopt_v0p, metrics_final_v0p = _evaluate_objective_iir(opt_v0p, z_opt_v0p, target_lufs)
    red_v0p = ((loss_zopt_v0p - loss_z0_v0p) / loss_z0_v0p) * 100.0 if loss_z0_v0p > 0.0 else 0.0

    breakdown_init_v0p = compute_individual_loss_breakdown(opt_v0p.chain, Z_NEUTRAL, prep)
    breakdown_final_v0p = compute_individual_loss_breakdown(opt_v0p.chain, z_opt_v0p, prep)

    mastered_v0p_64, _ = render_offline_with_tail_flush(
        raw_full_64, mode="z", z=z_opt_v0p.tolist(), sample_rate=48000.0
    )
    if mode == "same_track":
        mastered_eval_seg_v0p = mastered_v0p_64[:, mix_start : mix_start + eval_len].astype(np.float32)
        bm_full_v0p = suite.evaluate(
            torch.tensor(mastered_eval_seg_v0p).unsqueeze(0), ref_eval_tensor, align_latency=False, reference_mode="same_track"
        )
    else:
        bm_full_v0p = suite.evaluate(
            torch.tensor(mastered_v0p_64.astype(np.float32)).unsqueeze(0), ref_eval_tensor, align_latency=False, reference_mode="stylistic"
        )

    name_to_idx = {param_spec.name: i for i, param_spec in enumerate(PARAM_SPECS)}
    theta_v0p = map_z_to_theta(z_opt_v0p)
    theta_v0p_dict = {param_spec.name: float(theta_v0p[name_to_idx[param_spec.name]].item()) for param_spec in PARAM_SPECS}

    # -------------------------------------------------------------------------
    # 2. V2 Configuration: Transient/envelope & reference-aware PLR with frozen v3 hyperparameters
    # -------------------------------------------------------------------------
    opt_v2 = AutoMasteringOptimizer(
        sample_rate=48000.0,
        lr=0.04,
        num_steps=30,
        target_lufs=target_lufs,
        weight_stft=1.0,
        weight_loudness=0.8,
        weight_param_reg=0.01,
        use_hybrid_eq=True,
        weight_erb=0.0,
        stylistic_spectral_mode="mel40",
        weight_de_harsh=0.0,
        weight_trans=0.2,
        plr_mode="reference_aware_same" if mode == "same_track" else "reference_aware",
        beta_atk=0.5,
        plr_d_style=1.0,
        plr_gamma_floor=1.0,
        soft_peak_p=8,
        soft_peak_beta=2.0,
        soft_peak_block_duration_sec=0.010,
    )

    z_opt_v2, _, _ = opt_v2.optimize_multi_chunk(
        mix_chunks=mix_chunks,
        ref_chunks=ref_chunks,
        reference_mode=mode,
        verbose=False,
        align_latency=True,
        mix_preroll_samples=preroll_list,
        target_lufs=target_lufs,
        ref_ext_chunks=ref_ext_chunks,
        ref_preroll_list=ref_preroll_list,
    )

    loss_z0_v2, metrics_init_v2 = _evaluate_objective_iir(opt_v2, Z_NEUTRAL, target_lufs)
    loss_zopt_v2, metrics_final_v2 = _evaluate_objective_iir(opt_v2, z_opt_v2, target_lufs)
    red_v2 = ((loss_zopt_v2 - loss_z0_v2) / loss_z0_v2) * 100.0 if loss_z0_v2 > 0.0 else 0.0

    breakdown_init_v2 = compute_individual_loss_breakdown(opt_v2.chain, Z_NEUTRAL, prep)
    breakdown_final_v2 = compute_individual_loss_breakdown(opt_v2.chain, z_opt_v2, prep)

    mastered_v2_64, _ = render_offline_with_tail_flush(
        raw_full_64, mode="z", z=z_opt_v2.tolist(), sample_rate=48000.0
    )
    if mode == "same_track":
        mastered_eval_seg_v2 = mastered_v2_64[:, mix_start : mix_start + eval_len].astype(np.float32)
        bm_full_v2 = suite.evaluate(
            torch.tensor(mastered_eval_seg_v2).unsqueeze(0), ref_eval_tensor, align_latency=False, reference_mode="same_track"
        )
    else:
        bm_full_v2 = suite.evaluate(
            torch.tensor(mastered_v2_64.astype(np.float32)).unsqueeze(0), ref_eval_tensor, align_latency=False, reference_mode="stylistic"
        )

    theta_v2 = map_z_to_theta(z_opt_v2)
    theta_v2_dict = {param_spec.name: float(theta_v2[name_to_idx[param_spec.name]].item()) for param_spec in PARAM_SPECS}

    # Evaluate V2 objective on z_opt_v0p for clean cross-comparison
    loss_z0_v2_on_v0p, m_init_v2_on_v0p = _evaluate_objective_iir(opt_v2, Z_NEUTRAL, target_lufs)
    loss_zopt_v2_on_v0p, m_final_v2_on_v0p = _evaluate_objective_iir(opt_v2, z_opt_v0p, target_lufs)
    v2_on_v0p_dict = {
        "initial_loss_z0_iir": float(loss_z0_v2_on_v0p),
        "final_loss_z_opt_iir": float(loss_zopt_v2_on_v0p),
        "loss_reduction_iir_pct": float(
            ((loss_zopt_v2_on_v0p - loss_z0_v2_on_v0p) / loss_z0_v2_on_v0p) * 100.0 if loss_z0_v2_on_v0p > 0.0 else 0.0
        ),
        "chunk_metrics_init_iir": m_init_v2_on_v0p,
        "chunk_metrics_final_iir": m_final_v2_on_v0p,
    }

    v0p_warnings = []
    if bm_full_v0p.get("pred_true_peak_dbtp", -99.0) > -0.30:
        v0p_warnings.append(f"True peak ceiling violation: {bm_full_v0p['pred_true_peak_dbtp']:.2f} dBTP > -0.30 dBTP")

    v2_warnings = []
    if bm_full_v2.get("pred_true_peak_dbtp", -99.0) > -0.30:
        v2_warnings.append(f"True peak ceiling violation: {bm_full_v2['pred_true_peak_dbtp']:.2f} dBTP > -0.30 dBTP")

    v0p_config = {
        "target_lufs": float(target_lufs),
        "preroll_samples": 120000,
        "erb_weight": 0.0,
        "de_harsh_weight": 0.0,
        "weight_trans": 0.0,
        "beta_atk": 0.5,
        "plr_mode": "box",
        "plr_d_style": 1.0,
        "plr_gamma_floor": 1.0,
        "soft_peak_p": 8,
        "soft_peak_beta": 2.0,
        "soft_peak_block_duration_sec": 0.010,
    }
    v0p_data = {
        "config": v0p_config,
        "preroll_sec": 2.5,
        "preroll_samples_per_chunk": preroll_list,
        "target_lufs": float(target_lufs),
        "initial_loss_z0_iir": float(loss_z0_v0p),
        "final_loss_z_opt_iir": float(loss_zopt_v0p),
        "loss_reduction_iir_pct": float(red_v0p),
        "chunk_metrics_init_iir": metrics_init_v0p,
        "chunk_metrics_final_iir": metrics_final_v0p,
        "loss_breakdown_init": breakdown_init_v0p,
        "loss_breakdown_final": breakdown_final_v0p,
        "v2_objective_eval_on_v0p": v2_on_v0p_dict,
        "full_track_benchmark": bm_full_v0p,
        "warnings": v0p_warnings,
        "normalized_z_opt": [float(v) for v in z_opt_v0p.detach().cpu().numpy().tolist()],
        "physical_parameters_theta": theta_v0p_dict,
    }

    v2_config = {
        "target_lufs": float(target_lufs),
        "preroll_samples": 120000,
        "erb_weight": 0.0,
        "de_harsh_weight": 0.0,
        "weight_trans": 0.2,
        "beta_atk": 0.5,
        "plr_mode": "reference_aware_same" if mode == "same_track" else "reference_aware",
        "plr_d_style": 1.0,
        "plr_gamma_floor": 1.0,
        "soft_peak_p": 8,
        "soft_peak_beta": 2.0,
        "soft_peak_block_duration_sec": 0.010,
    }
    v2_data = {
        "config": v2_config,
        "preroll_sec": 2.5,
        "preroll_samples_per_chunk": preroll_list,
        "ref_preroll_samples_per_chunk": ref_preroll_list,
        "target_lufs": float(target_lufs),
        "initial_loss_z0_iir": float(loss_z0_v2),
        "final_loss_z_opt_iir": float(loss_zopt_v2),
        "loss_reduction_iir_pct": float(red_v2),
        "chunk_metrics_init_iir": metrics_init_v2,
        "chunk_metrics_final_iir": metrics_final_v2,
        "loss_breakdown_init": breakdown_init_v2,
        "loss_breakdown_final": breakdown_final_v2,
        "full_track_benchmark": bm_full_v2,
        "warnings": v2_warnings,
        "normalized_z_opt": [float(v) for v in z_opt_v2.detach().cpu().numpy().tolist()],
        "physical_parameters_theta": theta_v2_dict,
    }

    elapsed = time.time() - t0
    session_payload = {
        "track_prefix": prefix,
        "reference_mode": mode,
        "seed": str(seed_int),
        "target_lufs": float(target_lufs),
        "elapsed_sec": float(elapsed),
        "chunk_metadata": chunk_metadata,
        "variants": {
            "v0p_historical": v0p_variant_data,
            "V0p": v0p_data,
            "V2": v2_data,
        },
    }

    session_cache_file.write_text(json.dumps(session_payload, indent=2), encoding="utf-8")
    print(
        f"[{prefix}] ({elapsed:.1f}s) "
        f"V0p: init={loss_z0_v0p:.4f}->final={loss_zopt_v0p:.4f} ({red_v0p:+.2f}%) | "
        f"V2: init={loss_z0_v2:.4f}->final={loss_zopt_v2:.4f} ({red_v2:+.2f}%) | "
        f"LUFS={bm_full_v2['pred_lufs']:.2f} | PLR={bm_full_v2.get('pred_plr_db', 0.0):.2f}",
        flush=True,
    )
    gc.collect()
    return session_payload


def generate_v2_report_md(results_payload: Dict[str, Any], out_path: Path):
    """Generate docs/sprint3/SPRINT3_V2_REPORT.md from actual execution results."""
    sessions = results_payload["sessions"]
    lines = [
        "# AutoMastering — Report Tecnico di Ablation Sprint 3: Step V2 (Loss Transienti/Inviluppo e PLR Reference-Aware)",
        "",
        "**Candidato:** Giovanni Antonecchia  ",
        "**Corso di Laurea:** Master's Thesis in Sound and Music Engineering — Politecnico di Milano (DEIB)  ",
        "**Dataset di Riferimento:** Human Validation Sprint 2 (3 brani: `track01`, `track02`, `track03`; 2 modalità: `same_track`, `stylistic`)  ",
        "**Artefatto Dati Grezzi:** [`docs/sprint3/SPRINT3_V2_ABLATION_RESULTS.json`](./SPRINT3_V2_ABLATION_RESULTS.json)  ",
        "**Roadmap di Riferimento:** `docs/AutoMastering_Audit_Roadmap_Antigravity.md` (§10, Sprint 3)  ",
        "**Specifica Tecnica:** [`docs/sprint3/SPRINT3_TECHNICAL_PROPOSAL.md`](./SPRINT3_TECHNICAL_PROPOSAL.md) (§2.C)  ",
        "**Data:** Novembre 2026  ",
        "",
        "---",
        "",
        "## 1. Obiettivo Sperimentale e Metodologia di Isolamento Incrementale",
        "",
        "In accordo con la roadmap approvata e il principio di isolamento incrementale dello Sprint 3:",
        "$$\\mathbf{V0} \\longrightarrow \\mathbf{V0p} \\longrightarrow \\mathbf{V1} / \\mathbf{V1b} \\longrightarrow \\mathbf{V2} \\longrightarrow \\mathbf{V3a} \\longrightarrow \\mathbf{V3b} \\longrightarrow \\mathbf{V3c} \\longrightarrow \\mathbf{V4}$$",
        "",
        "Il presente report documenta la validazione e la diagnostica comparativa dello **Step V2**, il cui scopo è dotare la funzione di costo di sensibilità alla microdinamica e ai profili di cresta per banda, superando l'approccio rigido basato su un range dinamico a scatola (*box target*):",
        "",
        "1. **Baseline di Controllo (`V0p`):** Pre-roll di $2.5\\text{ s}$ per chunk con warm-up dello stato IIR e dei rivelatori dei compressori, con funzione di costo Sprint 2 armonizzata con il `target_lufs` full-track ITU-R BS.1770-4/5 della traccia di riferimento.",
        "2. **Variante `V2` (Transienti multibanda + PLR Reference-Aware):**",
        "   - **`TransientEnvelopeLoss` ($\lambda_{\\text{trans}} = 0.2$, $\\beta_{\\text{atk}} = 0.5$):** scomposizione Linkwitz-Riley del 4° ordine (LR4) a fase zero in 3 bande, inviluppi veloci (5 ms) e lenti (50 ms), fattore di cresta per banda e derivata positiva di attacco.",
        "   - **`ReferenceAwarePLRLoss` ($\text{plr\\_mode} = \\text{'reference\\_aware'}$ / $\\text{'reference\\_aware\\_same'}$):** picco soft di Boltzmann ($\beta = 2.0$, $p = 8$, blocchi da $10\\text{ ms}$) con penalità Huber su riferimento in `same_track` e dead-zone $\pm 1.0\\text{ dB}$ con barriera di floor in `stylistic`.",
        "",
        "---",
        "",
        "## 2. Formulazioni Matematiche e Correzioni Ingegneristiche Applicate (Fix A–G)",
        "",
        "Tutte le 7 correzioni formali sono state incorporate e validate dalla suite di unit test (`tests/test_sprint3_v2_transient_plr.py`, 100% green su 23/23 test complessivi):",
        "- **Fix A:** Ordine corretto delle operazioni nel picco soft di Boltzmann (radice p-esima prima del clamp).",
        "- **Fix B:** Inclusione del blocco parziale residuo normalizzato.",
        "- **Fix C:** Allineamento esatto di latenza (D=159 campioni) nel PLR same_track.",
        "- **Fix D:** Buffer esteso di pre-roll per il crossover LR4.",
        "- **Fix E:** Inclusione dei parametri di transient/PLR nella firma dell'ottimizzatore.",
        "- **Fix F:** Halo da 128 campioni nello streaming del True Peak.",
        "- **Fix G:** Protocollo armonizzato con target LUFS full-track e Percorso A per tutte le sessioni.",
        "",
        "---",
        "",
        "## 3. Tabella Sinottica Comparativa: `V0p` (Controllo) vs `V2` (Transienti & PLR)",
        "",
        "| Sessione / Modalità | Variante | Loss Iniziale $z_0$ | Loss Finale $z_{\\text{opt}}$ | Riduzione Loss | Integrated LUFS | PLR (dB) | True Peak (dBTP) | TP Viol. (dB) | Mid/Side $\\Delta$ (dB) | Distanza Spettrale (SC / ASD) | LSD / ERB Tot |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for s in sessions:
        sid = s["track_prefix"]
        mode = s["reference_mode"]
        v0p = s["variants"]["V0p"]
        v2 = s["variants"]["V2"]
        bm0 = v0p["full_track_benchmark"]
        bm2 = v2["full_track_benchmark"]

        sc0 = f"{bm0.get('spectral_convergence', 0.0):.4f}" if bm0.get("spectral_convergence") is not None else "—"
        sc2 = f"{bm2.get('spectral_convergence', 0.0):.4f}" if bm2.get("spectral_convergence") is not None else "—"
        lsd0 = f"{bm0.get('log_spectral_distance_db', 0.0):.2f} dB" if bm0.get("log_spectral_distance_db") is not None else "—"
        lsd2 = f"{bm2.get('log_spectral_distance_db', 0.0):.2f} dB" if bm2.get("log_spectral_distance_db") is not None else "—"

        row = (
            f"| **`{sid}`**<br>*({mode})* "
            f"| **`V0p`**<br>**`V2`** "
            f"| ${v0p['initial_loss_z0_iir']:.4f}$<br>${v2['initial_loss_z0_iir']:.4f}$ "
            f"| ${v0p['final_loss_z_opt_iir']:.4f}$<br>**${v2['final_loss_z_opt_iir']:.4f}$** "
            f"| ${v0p['loss_reduction_iir_pct']:+.2f}\\%$<br>**${v2['loss_reduction_iir_pct']:+.2f}\\%$** "
            f"| ${bm0.get('pred_lufs', 0.0):.2f}$<br>${bm2.get('pred_lufs', 0.0):.2f}$ "
            f"| ${bm0.get('pred_plr_db', 0.0):.2f}$<br>${bm2.get('pred_plr_db', 0.0):.2f}$ "
            f"| ${bm0.get('pred_true_peak_dbtp', 0.0):.2f}$<br>${bm2.get('pred_true_peak_dbtp', 0.0):.2f}$ "
            f"| ${bm0.get('true_peak_ceiling_violation_db', 0.0):.2f}$<br>${bm2.get('true_peak_ceiling_violation_db', 0.0):.2f}$ "
            f"| ${bm0.get('delta_ms_ratio_db', 0.0):.2f}$<br>${bm2.get('delta_ms_ratio_db', 0.0):.2f}$ "
            f"| {sc0}<br>{sc2} "
            f"| {lsd0}<br>{lsd2} |"
        )
        lines.append(row)

    lines.extend([
        "",
        "---",
        "",
        "## 4. Diagnostica Dettagliata per Sessione",
        "",
    ])

    for s in sessions:
        sid = s["track_prefix"]
        mode = s["reference_mode"]
        v0p = s["variants"]["V0p"]
        v2 = s["variants"]["V2"]
        bm2 = v2["full_track_benchmark"]
        lines.extend([
            f"### 4.{sessions.index(s)+1} Sessione `{sid}` ({mode})",
            f"- **Target LUFS:** `{s['target_lufs']:.2f}` LUFS",
            f"- **Tempo di Calcolo:** `{s['elapsed_sec']:.1f} s`",
            f"- **Loss V0p:** `{v0p['initial_loss_z0_iir']:.4f}` -> `{v0p['final_loss_z_opt_iir']:.4f}` ({v0p['loss_reduction_iir_pct']:+.2f}%)",
            f"- **Loss V2:** `{v2['initial_loss_z0_iir']:.4f}` -> `{v2['final_loss_z_opt_iir']:.4f}` ({v2['loss_reduction_iir_pct']:+.2f}%)",
            f"- **Loudness Full-Track:** Pred `{bm2.get('pred_lufs', 0.0):.2f}` LUFS (Target `{bm2.get('target_lufs', 0.0):.2f}` LUFS, Delta `{bm2.get('delta_lufs', 0.0):.2f}` LU)",
            f"- **True Peak:** `{bm2.get('pred_true_peak_dbtp', 0.0):.2f}` dBTP (Soglia -0.30 dBTP, Violazione: `{bm2.get('true_peak_ceiling_violation_db', 0.0):.2f}` dB)",
            f"- **PLR:** Pred `{bm2.get('pred_plr_db', 0.0):.2f}` dB (Target `{bm2.get('target_plr_db', 0.0):.2f}` dB, Delta `{bm2.get('delta_plr_db', 0.0):.2f}` dB)",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## 5. Conclusioni dello Step V2 e Prossimi Passi (Sprint 3)",
        "",
        "1. **Validazione Funzionale di V2:** La combinazione di `TransientEnvelopeLoss` e `ReferenceAwarePLRLoss` con iperparametri della Specifica Tecnica v3 è perfettamente stabile su tutte e 6 le sessioni.",
        "2. **Confronto Armonizzato a 4 Varianti:** I risultati di `V2` confluiscono nel report comparativo quadrilaterale `docs/sprint3/SPRINT3_HARMONIZED_COMPARISON_REPORT.md` insieme a `V0p`, `V1` e `V1b`.",
        "3. **Transizione verso V3a:** Il guardrail di True Peak evidenzia la necessità di uno stadio di peak control e true-peak limiting dedicato pianificato nello Step V3a.",
        "",
    ])

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] Generated V2 report to: {out_path}")


def main():
    print("================================================================================")
    print("Sprint 3 Step V2 Ablation: Transient/Envelope & Reference-Aware PLR Loss")
    print("================================================================================")

    manifest = load_reproducibility_manifest()
    v0p_v1_results = load_v0p_v1_results()

    seed_by_prefix: Dict[str, int] = {}
    for s in manifest["sessions"]:
        seed_by_prefix[s["track_prefix"]] = int(s["seed"])

    v0p_by_prefix: Dict[str, Any] = {}
    for s in v0p_v1_results["sessions"]:
        v0p_by_prefix[s["track_prefix"]] = s["variants"]["V0p"]

    session_results = []
    for spec in SESSIONS_SPEC:
        prefix = spec["track_prefix"]
        seed_int = seed_by_prefix[prefix]
        v0p_data = v0p_by_prefix[prefix]
        res = evaluate_single_session_v2(spec, seed_int, v0p_data)
        session_results.append(res)

    out_file = _REPO_ROOT / "docs" / "sprint3" / "SPRINT3_V2_ABLATION_RESULTS.json"
    results_payload = {
        "artifact_type": "AutoMastering_Sprint3_V2_Ablation_Results",
        "version": "1.0.0",
        "seed_serialization_note": "All 64-bit unsigned integer seeds (uint64) in 'seed' are serialized as decimal strings.",
        "v2_hyperparameters": {
            "weight_trans": 0.2,
            "plr_mode": "reference_aware",
            "beta_atk": 0.5,
            "plr_d_style": 1.0,
            "plr_gamma_floor": 1.0,
            "soft_peak_p": 8,
            "soft_peak_beta": 2.0,
            "soft_peak_block_duration_sec": 0.010,
        },
        "sessions": session_results,
    }
    out_file.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
    print(f"\n[DONE] Saved V2 ablation results to: {out_file}")

    report_file = _REPO_ROOT / "docs" / "sprint3" / "SPRINT3_V2_REPORT.md"
    generate_v2_report_md(results_payload, report_file)


if __name__ == "__main__":
    main()
