"""Harmonized Session Preparation and Loss Breakdown Module for Sprint 3.

Provides deterministic data preparation and objective evaluation across all Sprint 3
ablation variants (V0p, V1, V1b, V2) on the 6 human validation sessions.

Guaranteeing by construction:
1. Exact bit-identical chunk selection (4 chunks of 144,000 samples / 3.0s at 48 kHz);
2. Bit-identical pre-roll extraction using Percorso A (from mix_aligned in same_track);
3. Bit-identical full-track target LUFS (reference ITU-R BS.1770-4/5 integrated);
4. Complete unweighted individual loss decomposition across all variants.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ddsp_mastering.data import (
    MultiChunkSelector,
    apply_same_track_alignment,
    estimate_time_alignment,
    get_same_track_slice_indices,
)
from ddsp_mastering.dsp.chain import DifferentiableMasteringChain
from ddsp_mastering.loss.combined import MasteringPerceptualLoss
from ddsp_mastering.loss.erb import ERBSpectralLoss
from ddsp_mastering.loss.loudness import DifferentiableLoudnessLoss
from ddsp_mastering.loss.stft import MultiResolutionSTFTLoss
from ddsp_mastering.loss.stylistic import AggregatePerceptualLoss
from ddsp_mastering.loss.transient import TransientEnvelopeLoss
from ddsp_mastering.optimize import MasteringBenchmarkSuite
from ddsp_mastering.param_space import PARAM_SPECS, Z_NEUTRAL
from scripts.human_validation.validate_inputs import load_and_validate_wav

SESSIONS_SPEC = [
    {
        "track_prefix": "track01",
        "mode": "same_track",
        "mix_path": "human_validation/private/input/track01_mix.wav",
        "ref_path": "human_validation/private/input/track01_ref.wav",
    },
    {
        "track_prefix": "track02",
        "mode": "same_track",
        "mix_path": "human_validation/private/input/track02_mix.wav",
        "ref_path": "human_validation/private/input/track02_ref.wav",
    },
    {
        "track_prefix": "track03",
        "mode": "same_track",
        "mix_path": "human_validation/private/input/track03_mix.wav",
        "ref_path": "human_validation/private/input/track03_ref.wav",
    },
    {
        "track_prefix": "track01_style",
        "mode": "stylistic",
        "mix_path": "human_validation/private/input/track01_mix.wav",
        "ref_path": "human_validation/private/input/track01_ref_altro_brano.wav",
    },
    {
        "track_prefix": "track02_style",
        "mode": "stylistic",
        "mix_path": "human_validation/private/input/track02_mix.wav",
        "ref_path": "human_validation/private/input/track02_ref_altro_brano.wav",
    },
    {
        "track_prefix": "track03_style",
        "mode": "stylistic",
        "mix_path": "human_validation/private/input/track03_mix.wav",
        "ref_path": "human_validation/private/input/track03_ref_altro_brano.wav",
    },
]


def prepare_harmonized_session(
    spec: Dict[str, Any],
    seed_int: int,
    repo_root: Path,
    sr: float = 48000.0,
    n_preroll_samples: int = 120000,
) -> Dict[str, Any]:
    """Prepare bit-identical audio tensors and metadata for a session.

    Convention: Percorso A (mix_aligned in same_track, raw_full_32 in stylistic).
    """
    prefix = spec["track_prefix"]
    mode = spec["mode"]

    raw_full_64, _ = load_and_validate_wav(
        str(repo_root / spec["mix_path"]), required_sr=int(sr), allow_mono=True
    )
    ref_full_64, _ = load_and_validate_wav(
        str(repo_root / spec["ref_path"]), required_sr=int(sr), allow_mono=True
    )
    raw_full_32 = np.ascontiguousarray(raw_full_64, dtype=np.float32)
    ref_full_32 = np.ascontiguousarray(ref_full_64, dtype=np.float32)

    suite = MasteringBenchmarkSuite(sample_rate=sr)
    selector = MultiChunkSelector(
        sample_rate=sr,
        num_chunks=4,
        chunk_duration_sec=3.0,
        seed=seed_int,
    )

    mix_chunks: List[torch.Tensor] = []
    ref_chunks: List[torch.Tensor] = []
    preroll_list: List[int] = []
    ref_ext_chunks: List[torch.Tensor] = []
    ref_preroll_list: List[int] = []
    chunk_metadata: List[Dict[str, int]] = []

    if mode == "same_track":
        alignment = estimate_time_alignment(raw_full_64, ref_full_64, sample_rate=sr)
        mix_aligned, ref_aligned = apply_same_track_alignment(
            raw_full_64, ref_full_64, alignment, enable_subsample=True
        )
        mix_start, _, eval_len = get_same_track_slice_indices(
            alignment,
            mix_len=raw_full_64.shape[1],
            ref_len=ref_full_64.shape[1],
            enable_subsample=True,
        )
        ref_eval_tensor = torch.tensor(ref_aligned, dtype=torch.float32).unsqueeze(0)
        ref_bm = suite.evaluate(
            ref_eval_tensor, ref_eval_tensor, align_latency=False, reference_mode="same_track"
        )
        target_lufs = float(ref_bm["pred_lufs"])
        ref_bm["integrated_lufs"] = target_lufs

        chunks = selector.select_chunks(
            mix_aligned.astype(np.float32),
            ref_aligned.astype(np.float32),
            mode="same_track",
        )

        for k, ch in enumerate(chunks):
            s_idx = int(ch["start_sample"])
            e_idx = int(ch["end_sample"])

            # Percorso A: extract from mix_aligned
            pre_start = max(0, s_idx - n_preroll_samples)
            actual_pre = s_idx - pre_start
            mix_seg = torch.tensor(mix_aligned[:, pre_start:e_idx], dtype=torch.float32).unsqueeze(0)

            ref_seg = torch.tensor(ref_aligned[:, s_idx:e_idx], dtype=torch.float32).unsqueeze(0)

            ref_pre_start = max(0, s_idx - n_preroll_samples)
            actual_ref_pre = s_idx - ref_pre_start
            ref_ext_seg = torch.tensor(ref_aligned[:, ref_pre_start:e_idx], dtype=torch.float32).unsqueeze(0)

            mix_chunks.append(mix_seg)
            ref_chunks.append(ref_seg)
            preroll_list.append(int(actual_pre))
            ref_ext_chunks.append(ref_ext_seg)
            ref_preroll_list.append(int(actual_ref_pre))

            chunk_metadata.append({
                "chunk_index": k,
                "mix_start_sample": int(s_idx),
                "mix_end_sample": int(e_idx),
                "mix_abs_start_sample": int(mix_start + s_idx),
                "mix_abs_end_sample": int(mix_start + e_idx),
                "mix_preroll_samples": int(actual_pre),
                "ref_start_sample": int(s_idx),
                "ref_end_sample": int(e_idx),
                "ref_preroll_samples": int(actual_ref_pre),
            })
    else:
        alignment = None
        mix_start = 0
        eval_len = raw_full_64.shape[1]
        mix_aligned = raw_full_32
        ref_aligned = ref_full_32

        ref_eval_tensor = torch.tensor(ref_full_32, dtype=torch.float32).unsqueeze(0)
        ref_bm = suite.evaluate(
            ref_eval_tensor, ref_eval_tensor, align_latency=False, reference_mode="stylistic"
        )
        target_lufs = float(ref_bm["pred_lufs"])
        ref_bm["integrated_lufs"] = target_lufs

        mix_sel = selector.select_chunks(raw_full_32, None, mode="stylistic")
        ref_sel = selector.select_chunks(ref_full_32, None, mode="stylistic")

        for k, (m_ch, r_ch) in enumerate(zip(mix_sel, ref_sel)):
            ms = int(m_ch["start_sample"])
            me = int(m_ch["end_sample"])
            pre_start = max(0, ms - n_preroll_samples)
            actual_pre = ms - pre_start
            mix_seg = torch.tensor(raw_full_32[:, pre_start:me], dtype=torch.float32).unsqueeze(0)

            rs = int(r_ch["start_sample"])
            re = int(r_ch["end_sample"])
            ref_seg = torch.tensor(ref_full_32[:, rs:re], dtype=torch.float32).unsqueeze(0)

            ref_pre_start = max(0, rs - n_preroll_samples)
            actual_ref_pre = rs - ref_pre_start
            ref_ext_seg = torch.tensor(ref_full_32[:, ref_pre_start:re], dtype=torch.float32).unsqueeze(0)

            mix_chunks.append(mix_seg)
            ref_chunks.append(ref_seg)
            preroll_list.append(int(actual_pre))
            ref_ext_chunks.append(ref_ext_seg)
            ref_preroll_list.append(int(actual_ref_pre))

            chunk_metadata.append({
                "chunk_index": k,
                "mix_start_sample": int(ms),
                "mix_end_sample": int(me),
                "mix_abs_start_sample": int(ms),
                "mix_abs_end_sample": int(me),
                "mix_preroll_samples": int(actual_pre),
                "ref_start_sample": int(rs),
                "ref_end_sample": int(re),
                "ref_preroll_samples": int(actual_ref_pre),
            })

    # Strict invariant assertion: exactly 4 chunks of 3.0s (144000 samples)
    assert len(mix_chunks) == 4 and len(ref_chunks) == 4, (
        f"[{prefix}] Expected 4 chunks, got mix={len(mix_chunks)}, ref={len(ref_chunks)}"
    )
    for k_idx in range(4):
        eval_mix_len = mix_chunks[k_idx].shape[-1] - preroll_list[k_idx]
        eval_ref_len = ref_chunks[k_idx].shape[-1]
        eval_ref_ext_len = ref_ext_chunks[k_idx].shape[-1] - ref_preroll_list[k_idx]
        assert eval_mix_len == 144000, f"[{prefix}] Chunk {k_idx} mix eval len != 144000: {eval_mix_len}"
        assert eval_ref_len == 144000, f"[{prefix}] Chunk {k_idx} ref len != 144000: {eval_ref_len}"
        assert eval_ref_ext_len == 144000, f"[{prefix}] Chunk {k_idx} ref ext eval len != 144000: {eval_ref_ext_len}"

    return {
        "track_prefix": prefix,
        "mode": mode,
        "seed": seed_int,
        "target_lufs": float(target_lufs),
        "raw_full_64": raw_full_64,
        "ref_full_64": ref_full_64,
        "raw_full_32": raw_full_32,
        "ref_full_32": ref_full_32,
        "mix_aligned": mix_aligned,
        "ref_aligned": ref_aligned,
        "alignment": alignment,
        "mix_start": mix_start,
        "eval_len": eval_len,
        "ref_eval_tensor": ref_eval_tensor,
        "ref_bm": ref_bm,
        "mix_chunks": mix_chunks,
        "ref_chunks": ref_chunks,
        "preroll_list": preroll_list,
        "ref_ext_chunks": ref_ext_chunks,
        "ref_preroll_list": ref_preroll_list,
        "chunk_metadata": chunk_metadata,
    }


def compute_individual_loss_breakdown(
    chain: DifferentiableMasteringChain,
    z: torch.Tensor,
    prep_data: Dict[str, Any],
    sr: float = 48000.0,
) -> Dict[str, float]:
    """Compute all 10 unweighted individual loss terms on the 4 chunks in IIR mode.

    Terms evaluated:
    - mrstft: Multi-Resolution STFT spectral distance (same_track)
    - mel_spectral: Mel-40 spectral difference (stylistic)
    - erb: ERB-48 loss
    - de_harsh: Asymmetric relative de-harshness loss
    - transient: Transient / envelope loss
    - plr_box: Dynamic range loss with box formulation
    - plr_ref_aware: Reference-aware PLR loss with Boltzmann soft-peak
    - lufs: BS.1770 differentiable LUFS loss (relative to full-track target_lufs)
    - stereo: Mid-Side energy ratio loss (stylistic)
    - param_reg: Parameter regularization loss relative to Z_NEUTRAL
    """
    mode = prep_data["mode"]
    mix_chunks = prep_data["mix_chunks"]
    ref_chunks = prep_data["ref_chunks"]
    preroll_list = prep_data["preroll_list"]
    ref_ext_chunks = prep_data["ref_ext_chunks"]
    ref_preroll_list = prep_data["ref_preroll_list"]
    target_lufs = prep_data["target_lufs"]
    K = len(mix_chunks)

    # Initialize loss components
    same_loss_fn = MasteringPerceptualLoss(sample_rate=sr)
    stft_fn = MultiResolutionSTFTLoss()
    erb_fn = ERBSpectralLoss(sample_rate=sr)
    trans_fn = TransientEnvelopeLoss(sample_rate=sr, beta_atk=0.5)
    loudness_box = DifferentiableLoudnessLoss(
        sample_rate=sr,
        target_lufs=target_lufs,
        soft_peak_p=8,
        soft_peak_beta=2.0,
        soft_peak_block_duration_sec=0.010,
    )
    loudness_ref = DifferentiableLoudnessLoss(
        sample_rate=sr,
        target_lufs=target_lufs,
        soft_peak_p=8,
        soft_peak_beta=2.0,
        soft_peak_block_duration_sec=0.010,
        plr_d_style=1.0,
        plr_gamma_floor=1.0,
    )
    style_fn = AggregatePerceptualLoss(sample_rate=sr, spectral_mode="mel40")

    stft_vals = []
    mel_vals = []
    erb_vals = []
    deharsh_vals = []
    trans_vals = []
    plr_box_vals = []
    plr_ref_vals = []
    lufs_vals = []
    stereo_vals = []

    with torch.no_grad():
        for k in range(K):
            n_pre = preroll_list[k]
            mk = mix_chunks[k]
            rk = ref_chunks[k]
            ref_ext_k = ref_ext_chunks[k]
            n_ref_pre = ref_preroll_list[k]

            pk_ext = chain(mk, z=z, eq_mode="iir")
            pk = pk_ext[..., n_pre:] if n_pre > 0 else pk_ext
            mk_eval = mk[..., n_pre:] if n_pre > 0 else mk

            if mode == "same_track":
                # Latency alignment (D=159)
                p_aligned, t_aligned = same_loss_fn.align_latency(pk, rk)
                stft_v, _ = stft_fn(p_aligned, t_aligned)
                stft_vals.append(stft_v.item())

                erb_v, erb_m = erb_fn.forward_same_track(p_aligned, t_aligned)
                erb_vals.append(erb_v.item())
                deharsh_vals.append(erb_m.get("erb_de_harsh_loss", 0.0))

                eff_pre_pred = n_pre + same_loss_fn.latency_samples
                eff_pre_ref = n_ref_pre
                D_tail = same_loss_fn.latency_samples
                T_orig = rk.shape[-1]
                ref_for_trans = ref_ext_k[..., : n_ref_pre + T_orig - D_tail]
                trans_v, _ = trans_fn.forward_same_track(
                    pred_audio=pk_ext,
                    target_audio=ref_for_trans,
                    pred_preroll_samples=eff_pre_pred,
                    ref_preroll_samples=eff_pre_ref,
                )
                trans_vals.append(trans_v.item())

                # Loudness Box
                _, l_box_m = loudness_box(p_aligned, target_lufs=target_lufs, plr_mode="box")
                plr_box_vals.append(l_box_m.get("plr_loss", 0.0))
                lufs_vals.append(l_box_m.get("lufs_loss", 0.0))

                # Loudness Ref-Aware
                _, l_ref_m = loudness_ref(
                    p_aligned, target_lufs=target_lufs, ref_audio=t_aligned, plr_mode="reference_aware_same"
                )
                plr_ref_vals.append(l_ref_m.get("plr_loss", 0.0))
            else:
                # Stylistic
                mel_pred = style_fn._extract_mel_spectrum_distribution(pk)
                mel_ref = style_fn._extract_mel_spectrum_distribution(rk)
                mel_v = torch.mean(torch.abs(mel_pred - mel_ref)).item()
                mel_vals.append(mel_v)

                erb_v, erb_m = erb_fn.forward_stylistic(pk, rk, mix_anchor_audio=mk_eval)
                erb_vals.append(erb_v.item())
                deharsh_vals.append(erb_m.get("erb_de_harsh_loss", 0.0))

                trans_v, _ = trans_fn.forward_stylistic(
                    pred_audio=pk_ext,
                    ref_audio=ref_ext_k,
                    mix_anchor_audio=mk_eval,
                    pred_preroll_samples=n_pre,
                    ref_preroll_samples=n_ref_pre,
                )
                trans_vals.append(trans_v.item())

                # Loudness Box
                _, l_box_m = loudness_box(pk, target_lufs=target_lufs, plr_mode="box")
                plr_box_vals.append(l_box_m.get("plr_loss", 0.0))
                lufs_vals.append(l_box_m.get("lufs_loss", 0.0))

                # Loudness Ref-Aware
                _, l_ref_m = loudness_ref(
                    pk, target_lufs=target_lufs, ref_audio=rk, plr_mode="reference_aware"
                )
                plr_ref_vals.append(l_ref_m.get("plr_loss", 0.0))

                # Stereo
                mid_p = (pk[:, 0, :] + pk[:, 1, :]) * 0.70710678
                side_p = (pk[:, 0, :] - pk[:, 1, :]) * 0.70710678
                energy_m_p = torch.mean(mid_p ** 2, dim=-1) + 1e-8
                energy_s_p = torch.mean(side_p ** 2, dim=-1) + 1e-8
                ratio_p = energy_s_p / energy_m_p

                mid_r = (rk[:, 0, :] + rk[:, 1, :]) * 0.70710678
                side_r = (rk[:, 0, :] - rk[:, 1, :]) * 0.70710678
                energy_m_r = torch.mean(mid_r ** 2, dim=-1) + 1e-8
                energy_s_r = torch.mean(side_r ** 2, dim=-1) + 1e-8
                ratio_r = energy_s_r / energy_m_r

                stereo_v = torch.mean(torch.abs(ratio_p - ratio_r)).item()
                stereo_vals.append(stereo_v)

        # Parameter regularizer
        z_neutral = Z_NEUTRAL.to(device=z.device, dtype=z.dtype)
        param_reg_val = torch.mean((z - z_neutral) ** 2).item()

    return {
        "mrstft": float(np.mean(stft_vals)) if len(stft_vals) > 0 else 0.0,
        "mel_spectral": float(np.mean(mel_vals)) if len(mel_vals) > 0 else 0.0,
        "erb": float(np.mean(erb_vals)) if len(erb_vals) > 0 else 0.0,
        "de_harsh": float(np.mean(deharsh_vals)) if len(deharsh_vals) > 0 else 0.0,
        "transient": float(np.mean(trans_vals)) if len(trans_vals) > 0 else 0.0,
        "plr_box": float(np.mean(plr_box_vals)) if len(plr_box_vals) > 0 else 0.0,
        "plr_ref_aware": float(np.mean(plr_ref_vals)) if len(plr_ref_vals) > 0 else 0.0,
        "lufs": float(np.mean(lufs_vals)) if len(lufs_vals) > 0 else 0.0,
        "stereo": float(np.mean(stereo_vals)) if len(stereo_vals) > 0 else 0.0,
        "param_reg": float(param_reg_val),
    }


def evaluate_all_10_terms_iir(*args, **kwargs) -> Dict[str, float]:
    """Canonical evaluator for all 10 unweighted individual loss terms in IIR mode.
    
    Accepts either (chain, z, prep_data, [sr]) or (optimizer, z, prep_data, [sr]).
    """
    if len(args) >= 1 and hasattr(args[0], "chain"):
        chain = args[0].chain
        return compute_individual_loss_breakdown(chain, *args[1:], **kwargs)
    return compute_individual_loss_breakdown(*args, **kwargs)

