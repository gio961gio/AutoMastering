"""Sprint 3 Step V1 Gradient Diagnostics Runner (Specifica Tecnica v3 §2.A.2).

Computes non-invasive analytical gradients of individual loss terms with respect to
latent parameters z in R^41 across all 6 sessions, 3 variants (V0p, V1, V1b),
and 2 evaluation points (z_neutral and historical z_opt).

Partitions the 41 latent parameters into 6 functional DSP groups:
  - EQ: [0:18] (18 params)
  - Crossover: [18:20] (2 params)
  - Compressor: [20:35] (15 params)
  - Imager: [35:38] (3 params)
  - Saturator: [38:40] (2 params)
  - Limiter: [40:41] (1 param)

Exports results to docs/sprint3/SPRINT3_V1_GRADIENT_DIAGNOSTICS.json.
"""

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "python"))
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from ddsp_mastering.data import (
    MultiChunkSelector,
    apply_same_track_alignment,
    estimate_time_alignment,
    get_same_track_slice_indices,
)
from ddsp_mastering.dsp import enable_fast_cpu_biquad
enable_fast_cpu_biquad()
from ddsp_mastering.dsp.chain import DifferentiableMasteringChain
from ddsp_mastering.loss.combined import MasteringPerceptualLoss
from ddsp_mastering.loss.stylistic import AggregatePerceptualLoss
from ddsp_mastering.param_space import PARAM_SPECS, Z_NEUTRAL
from scripts.human_validation.validate_inputs import load_and_validate_wav

PARAM_GROUPS = {
    "EQ": list(range(0, 18)),
    "Crossover": list(range(18, 20)),
    "Compressor": list(range(20, 35)),
    "Imager": list(range(35, 38)),
    "Saturator": list(range(38, 40)),
    "Limiter": list(range(40, 41)),
}

SESSIONS_SPEC = [
    {"track_prefix": "track01", "mode": "same_track", "mix_path": "human_validation/private/input/track01_mix.wav", "ref_path": "human_validation/private/input/track01_ref.wav"},
    {"track_prefix": "track02", "mode": "same_track", "mix_path": "human_validation/private/input/track02_mix.wav", "ref_path": "human_validation/private/input/track02_ref.wav"},
    {"track_prefix": "track03", "mode": "same_track", "mix_path": "human_validation/private/input/track03_mix.wav", "ref_path": "human_validation/private/input/track03_ref.wav"},
    {"track_prefix": "track01_style", "mode": "stylistic", "mix_path": "human_validation/private/input/track01_mix.wav", "ref_path": "human_validation/private/input/track01_ref_altro_brano.wav"},
    {"track_prefix": "track02_style", "mode": "stylistic", "mix_path": "human_validation/private/input/track02_mix.wav", "ref_path": "human_validation/private/input/track02_ref_altro_brano.wav"},
    {"track_prefix": "track03_style", "mode": "stylistic", "mix_path": "human_validation/private/input/track03_mix.wav", "ref_path": "human_validation/private/input/track03_ref_altro_brano.wav"},
]

VARIANTS_WEIGHTS = {
    "same_track": {
        "V0p": {"mrstft": 1.0, "erb": 0.0, "de_harsh": 0.0, "lufs": 0.8, "plr": 0.8, "param_reg": 0.01},
        "V1":  {"mrstft": 1.0, "erb": 0.25, "de_harsh": 0.0, "lufs": 0.8, "plr": 0.8, "param_reg": 0.01},
        "V1b": {"mrstft": 1.0, "erb": 0.25, "de_harsh": 0.15, "lufs": 0.8, "plr": 0.8, "param_reg": 0.01},
    },
    "stylistic": {
        "V0p": {"mel_spectral": 1.0, "erb": 0.0, "de_harsh": 0.0, "lufs": 0.8, "plr": 0.8, "stereo_moment": 0.2, "param_reg": 0.01},
        "V1":  {"mel_spectral": 0.0, "erb": 1.0, "de_harsh": 0.0, "lufs": 0.8, "plr": 0.8, "stereo_moment": 0.2, "param_reg": 0.01},
        "V1b": {"mel_spectral": 0.0, "erb": 1.0, "de_harsh": 0.15, "lufs": 0.8, "plr": 0.8, "stereo_moment": 0.2, "param_reg": 0.01},
    },
}


def main():
    print("================================================================================", flush=True)
    print("Sprint 3 Step V1 Gradient Diagnostics: Spec v3 §2.A.2", flush=True)
    print("================================================================================", flush=True)

    manifest = json.loads((_REPO_ROOT / "docs" / "human_validation" / "SPRINT2_V0_REPRODUCIBILITY_MANIFEST.json").read_text(encoding="utf-8"))
    hist_results = json.loads((_REPO_ROOT / "docs" / "sprint3" / "SPRINT3_V0P_V1_ABLATION_RESULTS.json").read_text(encoding="utf-8"))

    seed_by_prefix = {s["track_prefix"]: int(s["seed"]) for s in manifest["sessions"]}
    hist_by_prefix = {s["track_prefix"]: s for s in hist_results["sessions"]}

    sr = 48000.0
    chain = DifferentiableMasteringChain(sample_rate=sr)
    n_preroll_samples = 120000
    D = 159

    diagnostics_payload: Dict[str, Any] = {
        "artifact_type": "AutoMastering_Sprint3_V1_Gradient_Diagnostics",
        "version": "1.0.0",
        "metadata": {
            "parameter_space": "z in R^41 (normalized in [0, 1]^41)",
            "groups": {name: {"indices": idxs, "count": len(idxs)} for name, idxs in PARAM_GROUPS.items()},
            "evaluation_eq_mode": "iir",
        },
        "sessions": {},
    }

    t0_all = time.time()

    for spec in SESSIONS_SPEC:
        sid = spec["track_prefix"]
        mode = spec["mode"]
        seed_int = seed_by_prefix[sid]
        print(f"\n>>> Processing Session: {sid} ({mode})", flush=True)

        raw_full_64, _ = load_and_validate_wav(str(_REPO_ROOT / spec["mix_path"]), required_sr=48000, allow_mono=True)
        ref_full_64, _ = load_and_validate_wav(str(_REPO_ROOT / spec["ref_path"]), required_sr=48000, allow_mono=True)
        raw_full_32 = np.ascontiguousarray(raw_full_64, dtype=np.float32)
        ref_full_32 = np.ascontiguousarray(ref_full_64, dtype=np.float32)

        selector = MultiChunkSelector(sample_rate=sr, num_chunks=4, chunk_duration_sec=3.0, seed=seed_int)

        if mode == "same_track":
            alignment = estimate_time_alignment(raw_full_64, ref_full_64, sample_rate=sr)
            mix_aligned, ref_aligned = apply_same_track_alignment(raw_full_64, ref_full_64, alignment, enable_subsample=True)
            mix_start, _, eval_len = get_same_track_slice_indices(alignment, mix_len=raw_full_64.shape[1], ref_len=ref_full_64.shape[1], enable_subsample=True)
            chunks = selector.select_chunks(mix_aligned.astype(np.float32))

            mix_chunk_tensors = []
            ref_chunk_tensors = []
            preroll_counts = []
            for ch in chunks:
                s_idx, e_idx = int(ch["start_sample"]), int(ch["end_sample"])
                abs_s, abs_e = mix_start + s_idx, mix_start + e_idx
                pre_start = max(0, abs_s - n_preroll_samples)
                actual_pre = abs_s - pre_start
                mix_chunk_tensors.append(torch.tensor(raw_full_32[:, pre_start:abs_e], dtype=torch.float32).unsqueeze(0))
                ref_chunk_tensors.append(torch.tensor(ref_aligned[:, s_idx:e_idx], dtype=torch.float32).unsqueeze(0))
                preroll_counts.append(int(actual_pre))
        else:
            mix_sel = selector.select_chunks(raw_full_32, None, mode="stylistic")
            ref_sel = selector.select_chunks(ref_full_32, None, mode="stylistic")
            mix_chunk_tensors = []
            ref_chunk_tensors = []
            preroll_counts = []
            for m_ch, r_ch in zip(mix_sel, ref_sel):
                ms, me = int(m_ch["start_sample"]), int(m_ch["end_sample"])
                pre_start = max(0, ms - n_preroll_samples)
                actual_pre = ms - pre_start
                mix_chunk_tensors.append(torch.tensor(raw_full_32[:, pre_start:me], dtype=torch.float32).unsqueeze(0))
                rs, re = int(r_ch["start_sample"]), int(r_ch["end_sample"])
                ref_chunk_tensors.append(torch.tensor(ref_full_32[:, rs:re], dtype=torch.float32).unsqueeze(0))
                preroll_counts.append(int(actual_pre))

        hist_session = hist_by_prefix[sid]
        session_diag: Dict[str, Any] = {"reference_mode": mode, "variants": {}}

        # Loss modules
        same_loss_inst = MasteringPerceptualLoss(sample_rate=sr, weight_stft=1.0, weight_loudness=0.8, weight_param_reg=0.01, weight_erb=0.25, weight_de_harsh=0.15)
        style_loss_inst = AggregatePerceptualLoss(sample_rate=sr, spectral_mode="erb48", weight_spectral=1.0, weight_loudness=0.8, weight_param_reg=0.01)

        variants_to_test = ["V0p", "V1", "V1b"]
        for var_name in variants_to_test:
            z_opt_hist = torch.tensor(hist_session["variants"][var_name]["normalized_z_opt"], dtype=torch.float32)
            eval_points = {"z_neutral": Z_NEUTRAL, "z_opt": z_opt_hist}
            var_diag: Dict[str, Any] = {}

            weights_dict = VARIANTS_WEIGHTS[mode][var_name]

            for pt_name, z_pt in eval_points.items():
                print(f"  [{var_name}] Evaluating at {pt_name}...", flush=True)

                # Collect average gradients per loss term over 4 chunks
                K = len(mix_chunk_tensors)
                term_grads: Dict[str, torch.Tensor] = {}

                for k in range(K):
                    mk = mix_chunk_tensors[k]
                    rk = ref_chunk_tensors[k]
                    n_pre = preroll_counts[k]

                    z_param = z_pt.clone().detach().requires_grad_(True)
                    y_ext = chain(mk, z=z_param, eq_mode="iir")
                    y_eval = y_ext[..., n_pre + D : n_pre + D + (mk.shape[-1] - n_pre - D)]
                    ref_eval = rk[..., : y_eval.shape[-1]]

                    # Compute individual loss terms
                    terms_k: Dict[str, torch.Tensor] = {}

                    if mode == "same_track":
                        # MR-STFT
                        stft_val, _ = same_loss_inst.stft_loss(y_eval, ref_eval)
                        terms_k["mrstft"] = stft_val

                        # ERB & de_harsh
                        w_tilde_ref, _, is_valid_ref = same_loss_inst.erb_loss.compute_active_weights(ref_eval) if hasattr(same_loss_inst.erb_loss, "compute_active_weights") else (None, None, None)
                        from ddsp_mastering.loss.erb import compute_active_frame_weights_and_mask
                        w_tilde_ref, _, is_valid_ref = compute_active_frame_weights_and_mask(
                            ref_eval, n_fft=same_loss_inst.erb_loss.n_fft, hop_length=same_loss_inst.erb_loss.hop_length, eps=same_loss_inst.erb_loss.eps
                        )
                        erb_pred_db = same_loss_inst.erb_loss.extract_erb_band_levels_db(y_eval)
                        with torch.no_grad():
                            erb_ref_db = same_loss_inst.erb_loss.extract_erb_band_levels_db(ref_eval)

                        import torch.nn.functional as F
                        huber_elem = F.huber_loss(erb_pred_db, erb_ref_db, reduction="none", delta=1.0)
                        weighted_time = torch.sum(huber_elem * w_tilde_ref.unsqueeze(1), dim=-1)
                        erb_same_loss = torch.mean(torch.mean(weighted_time, dim=-1) * is_valid_ref)
                        terms_k["erb"] = erb_same_loss

                        de_harsh_loss, _ = same_loss_inst.erb_loss.compute_de_harshness_penalty(
                            erb_pred_db, w_tilde_ref, erb_ref_db, w_tilde_ref, is_valid_ref
                        )
                        terms_k["de_harsh"] = de_harsh_loss

                        # LUFS
                        pred_lufs = same_loss_inst.loudness_loss.calculate_lufs(y_eval)
                        lufs_loss = torch.mean((pred_lufs - (-14.0)) ** 2)
                        terms_k["lufs"] = lufs_loss

                        # PLR (box mode)
                        pred_plr = same_loss_inst.loudness_loss.calculate_plr(y_eval)
                        with torch.no_grad():
                            ref_plr = same_loss_inst.loudness_loss.calculate_plr(ref_eval)
                        plr_loss = torch.mean((pred_plr - ref_plr) ** 2)
                        terms_k["plr"] = plr_loss

                        # Param reg
                        reg_loss = torch.mean((z_param - Z_NEUTRAL.to(z_param.device)) ** 2)
                        terms_k["param_reg"] = reg_loss
                    else:
                        # Mel40
                        mel_pred = style_loss_inst._extract_mel_spectrum_distribution(y_eval)
                        with torch.no_grad():
                            mel_ref = style_loss_inst._extract_mel_spectrum_distribution(ref_eval)
                        mel_loss = torch.mean(torch.abs(mel_pred - mel_ref))
                        terms_k["mel_spectral"] = mel_loss

                        # ERB & de-harsh
                        from ddsp_mastering.loss.erb import compute_active_frame_weights_and_mask
                        anchor_k = mk[..., n_pre : mk.shape[-1]]
                        w_pred, mask_pred, valid_pred = compute_active_frame_weights_and_mask(
                            anchor_k, n_fft=style_loss_inst.erb_loss.n_fft, hop_length=style_loss_inst.erb_loss.hop_length, eps=style_loss_inst.erb_loss.eps
                        )
                        w_ref, mask_ref, valid_ref = compute_active_frame_weights_and_mask(
                            ref_eval, n_fft=style_loss_inst.erb_loss.n_fft, hop_length=style_loss_inst.erb_loss.hop_length, eps=style_loss_inst.erb_loss.eps
                        )
                        valid_mult = valid_pred * valid_ref
                        erb_pred_db = style_loss_inst.erb_loss.extract_erb_band_levels_db(y_eval)
                        mu_tilde_p, sigma_p, delta_p_p = style_loss_inst.erb_loss._compute_stylistic_band_stats(
                            erb_pred_db, w_pred, mask_pred
                        )
                        with torch.no_grad():
                            erb_ref_db = style_loss_inst.erb_loss.extract_erb_band_levels_db(ref_eval)
                            mu_tilde_r, sigma_r, delta_p_r = style_loss_inst.erb_loss._compute_stylistic_band_stats(
                                erb_ref_db, w_ref, mask_ref
                            )
                        import torch.nn.functional as F
                        shape_loss = torch.mean(torch.mean(F.huber_loss(mu_tilde_p, mu_tilde_r, reduction="none", delta=1.0), dim=-1) * valid_mult)
                        std_loss = torch.mean(torch.mean(torch.abs(sigma_p - sigma_r), dim=-1) * valid_mult)
                        quant_loss = torch.mean(torch.mean(torch.abs(delta_p_p - delta_p_r), dim=-1) * valid_mult)
                        erb_style_loss = shape_loss + style_loss_inst.erb_loss.alpha_std * std_loss + style_loss_inst.erb_loss.alpha_quantiles * quant_loss
                        terms_k["erb"] = erb_style_loss

                        de_harsh_loss, _ = style_loss_inst.erb_loss.compute_de_harshness_penalty(
                            erb_pred_db, w_pred, erb_ref_db, w_ref, valid_mult
                        )
                        terms_k["de_harsh"] = de_harsh_loss

                        # LUFS
                        pred_lufs = style_loss_inst.loudness_loss.calculate_lufs(y_eval)
                        with torch.no_grad():
                            ref_lufs = style_loss_inst.loudness_loss.calculate_lufs(ref_eval)
                        terms_k["lufs"] = torch.mean((pred_lufs - ref_lufs) ** 2)

                        # PLR
                        pred_plr = style_loss_inst.loudness_loss.calculate_plr(y_eval)
                        with torch.no_grad():
                            ref_plr = style_loss_inst.loudness_loss.calculate_plr(ref_eval)
                        terms_k["plr"] = torch.mean((pred_plr - ref_plr) ** 2)

                        # Stereo moment
                        left_p = y_eval[:, 0, :]
                        right_p = y_eval[:, 1, :]
                        mid_p = (left_p + right_p) * 0.70710678
                        side_p = (left_p - right_p) * 0.70710678
                        ms_p = torch.log(torch.mean(side_p ** 2, dim=-1) + 1e-6) - torch.log(torch.mean(mid_p ** 2, dim=-1) + 1e-6)
                        with torch.no_grad():
                            left_r = ref_eval[:, 0, :]
                            right_r = ref_eval[:, 1, :]
                            mid_r = (left_r + right_r) * 0.70710678
                            side_r = (left_r - right_r) * 0.70710678
                            ms_r = torch.log(torch.mean(side_r ** 2, dim=-1) + 1e-6) - torch.log(torch.mean(mid_r ** 2, dim=-1) + 1e-6)
                        terms_k["stereo_moment"] = torch.mean(torch.abs(ms_p - ms_r))

                        # Param reg
                        terms_k["param_reg"] = torch.mean((z_param - Z_NEUTRAL.to(z_param.device)) ** 2)

                    # Compute analytical gradients for each term
                    for t_name, t_val in terms_k.items():
                        if t_val.requires_grad:
                            g_term = torch.autograd.grad(t_val, z_param, retain_graph=True, allow_unused=True)[0]
                            g_vec = g_term.detach().cpu() if g_term is not None else torch.zeros(41)
                        else:
                            g_vec = torch.zeros(41)

                        if t_name not in term_grads:
                            term_grads[t_name] = g_vec / float(K)
                        else:
                            term_grads[t_name] = term_grads[t_name] + (g_vec / float(K))

                # Analyze norms per term and per group
                pt_diag: Dict[str, Any] = {"terms": {}, "groups_summary": {}}
                total_weighted_grad = torch.zeros(41)

                for t_name, g_vec in term_grads.items():
                    raw_norm_total = float(torch.norm(g_vec).item())
                    weight = float(weights_dict.get(t_name, 0.0))
                    weighted_g_vec = weight * g_vec
                    total_weighted_grad += weighted_g_vec
                    weighted_norm_total = float(torch.norm(weighted_g_vec).item())

                    group_breakdown = {}
                    for g_name, g_idxs in PARAM_GROUPS.items():
                        sub_raw = g_vec[g_idxs]
                        sub_weighted = weighted_g_vec[g_idxs]
                        group_breakdown[g_name] = {
                            "raw_norm": float(torch.norm(sub_raw).item()),
                            "weighted_norm": float(torch.norm(sub_weighted).item()),
                        }

                    pt_diag["terms"][t_name] = {
                        "applied_weight": weight,
                        "raw_norm_total": raw_norm_total,
                        "weighted_norm_total": weighted_norm_total,
                        "group_breakdown": group_breakdown,
                    }

                # Summary across groups for total weighted gradient
                total_w_norm = float(torch.norm(total_weighted_grad).item())
                for g_name, g_idxs in PARAM_GROUPS.items():
                    sub_w = total_weighted_grad[g_idxs]
                    sub_norm = float(torch.norm(sub_w).item())
                    pct_share = (sub_norm / total_w_norm * 100.0) if total_w_norm > 0 else 0.0
                    pt_diag["groups_summary"][g_name] = {
                        "weighted_norm": sub_norm,
                        "percentage_share": float(pct_share),
                    }
                pt_diag["total_effective_gradient_norm"] = total_w_norm

                var_diag[pt_name] = pt_diag

            session_diag["variants"][var_name] = var_diag

        diagnostics_payload["sessions"][sid] = session_diag

    out_file = _REPO_ROOT / "docs" / "sprint3" / "SPRINT3_V1_GRADIENT_DIAGNOSTICS.json"
    out_file.write_text(json.dumps(diagnostics_payload, indent=2), encoding="utf-8")
    elapsed_all = time.time() - t0_all
    print(f"\n[DONE] Saved Gradient Diagnostics in {elapsed_all:.1f}s to: {out_file}", flush=True)


if __name__ == "__main__":
    main()
