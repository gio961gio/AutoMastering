"""Task 1 Audit: Pre-roll and Same-Track Alignment Verification (Path A vs Path B vs Path C).

Compares:
  - Path A: mix_aligned[:, max(0, s_idx - 120000) : e_idx]
  - Path B: raw_full_32[:, max(0, mix_start + s_idx - 120000) : mix_start + e_idx]
  - Path C: continuous full-track streaming through DDSP IIR chain, sliced at chunk window.
"""

import json
import math
import sys
from pathlib import Path

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
from ddsp_mastering.param_space import Z_NEUTRAL
from scripts.human_validation.validate_inputs import load_and_validate_wav

manifest = json.loads((_REPO_ROOT / "docs" / "human_validation" / "SPRINT2_V0_REPRODUCIBILITY_MANIFEST.json").read_text(encoding="utf-8"))
hist_results = json.loads((_REPO_ROOT / "docs" / "sprint3" / "SPRINT3_V0P_V1_ABLATION_RESULTS.json").read_text(encoding="utf-8"))

seed_by_prefix = {s["track_prefix"]: int(s["seed"]) for s in manifest["sessions"]}
hist_by_prefix = {s["track_prefix"]: s for s in hist_results["sessions"]}

tracks = [
    ("track01", "human_validation/private/input/track01_mix.wav", "human_validation/private/input/track01_ref.wav"),
    ("track02", "human_validation/private/input/track02_mix.wav", "human_validation/private/input/track02_ref.wav"),
    ("track03", "human_validation/private/input/track03_mix.wav", "human_validation/private/input/track03_ref.wav"),
]

sr = 48000.0
chain = DifferentiableMasteringChain(sample_rate=sr)
loss_fns = {
    "V0p": MasteringPerceptualLoss(sample_rate=sr, weight_stft=1.0, weight_loudness=0.8, weight_param_reg=0.01, weight_erb=0.0, weight_de_harsh=0.0),
    "V1": MasteringPerceptualLoss(sample_rate=sr, weight_stft=1.0, weight_loudness=0.8, weight_param_reg=0.01, weight_erb=0.25, weight_de_harsh=0.0),
    "V1b": MasteringPerceptualLoss(sample_rate=sr, weight_stft=1.0, weight_loudness=0.8, weight_param_reg=0.01, weight_erb=0.25, weight_de_harsh=0.15),
}

task1_report_data = {}
n_preroll_samples = 120000
D = 159

for prefix, mix_path, ref_path in tracks:
    print(f"\n=================== Running Task 1 Audit for {prefix} ===================", flush=True)
    raw_full_64, _ = load_and_validate_wav(str(_REPO_ROOT / mix_path), required_sr=48000, allow_mono=True)
    ref_full_64, _ = load_and_validate_wav(str(_REPO_ROOT / ref_path), required_sr=48000, allow_mono=True)
    raw_full_32 = np.ascontiguousarray(raw_full_64, dtype=np.float32)
    ref_full_32 = np.ascontiguousarray(ref_full_64, dtype=np.float32)

    alignment = estimate_time_alignment(raw_full_64, ref_full_64, sample_rate=sr)
    mix_aligned, ref_aligned = apply_same_track_alignment(raw_full_64, ref_full_64, alignment, enable_subsample=True)
    mix_start, _, eval_len = get_same_track_slice_indices(alignment, mix_len=raw_full_64.shape[1], ref_len=ref_full_64.shape[1], enable_subsample=True)

    seed_int = seed_by_prefix[prefix]
    selector = MultiChunkSelector(sample_rate=sr, num_chunks=4, chunk_duration_sec=3.0, seed=seed_int)
    chunks = selector.select_chunks(mix_aligned.astype(np.float32))

    hist_session = hist_by_prefix[prefix]
    test_points = {
        "z_neutral": Z_NEUTRAL,
        "z_opt_v0p": torch.tensor(hist_session["variants"]["V0p"]["normalized_z_opt"], dtype=torch.float32),
        "z_opt_v1": torch.tensor(hist_session["variants"]["V1"]["normalized_z_opt"], dtype=torch.float32),
        "z_opt_v1b": torch.tensor(hist_session["variants"]["V1b"]["normalized_z_opt"], dtype=torch.float32),
    }

    track_entry = {
        "alignment": {
            "lag_samples": alignment.lag_samples,
            "subsample_offset": float(alignment.subsample_offset),
            "mix_start": mix_start,
            "eval_len": eval_len,
        },
        "chunks": [{"start_sample": int(c.start_sample), "end_sample": int(c.end_sample)} for c in chunks],
        "evaluations": {},
    }

    # Evaluate Chunk 0
    ch0 = chunks[0]
    s_idx = int(ch0.start_sample)
    e_idx = int(ch0.end_sample)
    abs_s = mix_start + s_idx
    abs_e = mix_start + e_idx

    pre_start_a = max(0, s_idx - n_preroll_samples)
    actual_pre_a = s_idx - pre_start_a
    seg_a = torch.tensor(mix_aligned[:, pre_start_a:e_idx], dtype=torch.float32).unsqueeze(0)

    pre_start_b = max(0, abs_s - n_preroll_samples)
    actual_pre_b = abs_s - pre_start_b
    seg_b = torch.tensor(raw_full_32[:, pre_start_b:abs_e], dtype=torch.float32).unsqueeze(0)

    ref_chunk = torch.tensor(ref_aligned[:, s_idx:e_idx], dtype=torch.float32).unsqueeze(0)

    print(f"[{prefix}:ch0] s_idx={s_idx}, abs_s={abs_s}, pre_a={actual_pre_a}, pre_b={actual_pre_b} (diff={actual_pre_b - actual_pre_a} samples)", flush=True)

    for pt_name, z_vec in test_points.items():
        track_entry["evaluations"][pt_name] = {}

        # Path C: Full track continuous streaming through DDSP IIR chain up to abs_e
        with torch.no_grad():
            full_raw_prefix = torch.tensor(raw_full_32[:, :abs_e], dtype=torch.float32).unsqueeze(0)
            y_stream_full = chain(full_raw_prefix, z=z_vec, eq_mode="iir")
            # Useful slice for ch0 in Path C:
            # Chunk starts at abs_s. Accounting for latency D=159:
            y_eval_c = y_stream_full[..., abs_s + D : abs_s + D + (e_idx - s_idx - D)]

        for var_name, loss_fn in loss_fns.items():
            # Path A forward & backward
            z_a = z_vec.clone().detach().requires_grad_(True)
            y_ext_a = chain(seg_a, z=z_a, eq_mode="iir")
            y_eval_a = y_ext_a[..., actual_pre_a + D : actual_pre_a + D + (e_idx - s_idx - D)]
            ref_eval = ref_chunk[..., : y_eval_a.shape[-1]]
            loss_a, _ = loss_fn(pred_audio=y_eval_a, target_audio=ref_eval, target_lufs=-14.0, normalized_params=z_a, align_latency=False)
            loss_a.backward()
            grad_a = z_a.grad.clone()

            # Path B forward & backward
            z_b = z_vec.clone().detach().requires_grad_(True)
            y_ext_b = chain(seg_b, z=z_b, eq_mode="iir")
            y_eval_b = y_ext_b[..., actual_pre_b + D : actual_pre_b + D + (e_idx - s_idx - D)]
            loss_b, _ = loss_fn(pred_audio=y_eval_b, target_audio=ref_eval, target_lufs=-14.0, normalized_params=z_b, align_latency=False)
            loss_b.backward()
            grad_b = z_b.grad.clone()

            # Path C loss
            with torch.no_grad():
                loss_c, _ = loss_fn(pred_audio=y_eval_c, target_audio=ref_eval, target_lufs=-14.0, normalized_params=z_vec, align_latency=False)

            # Differences Path A vs Path B
            diff_ab = (y_eval_a.detach() - y_eval_b.detach()).cpu().numpy()
            rms_ab = np.sqrt(np.mean(diff_ab ** 2))
            rms_ab_dbfs = 20.0 * np.log10(max(rms_ab, 1e-12))
            max_err_ab = np.max(np.abs(diff_ab))
            max_err_ab_dbfs = 20.0 * np.log10(max(max_err_ab, 1e-12))
            rel_loss_ab = abs(loss_a.item() - loss_b.item()) / max(abs(loss_a.item()), 1e-6)
            grad_norm_a = torch.norm(grad_a).item()
            rel_grad_ab = (torch.norm(grad_a - grad_b).item() / max(grad_norm_a, 1e-9)) if grad_norm_a > 0 else 0.0

            # Differences Path B vs Path C (finite pre-roll vs continuous streaming)
            diff_bc = (y_eval_b.detach() - y_eval_c.detach()).cpu().numpy()
            rms_bc = np.sqrt(np.mean(diff_bc ** 2))
            rms_bc_dbfs = 20.0 * np.log10(max(rms_bc, 1e-12))
            max_err_bc = np.max(np.abs(diff_bc))
            max_err_bc_dbfs = 20.0 * np.log10(max(max_err_bc, 1e-12))
            rel_loss_bc = abs(loss_b.item() - loss_c.item()) / max(abs(loss_b.item()), 1e-6)

            eval_res = {
                "loss_a": float(loss_a.item()),
                "loss_b": float(loss_b.item()),
                "loss_c": float(loss_c.item()),
                "rel_loss_diff_ab": float(rel_loss_ab),
                "rel_grad_diff_ab": float(rel_grad_ab),
                "rms_ab_dbfs": float(rms_ab_dbfs),
                "max_err_ab_dbfs": float(max_err_ab_dbfs),
                "rms_bc_dbfs": float(rms_bc_dbfs),
                "max_err_bc_dbfs": float(max_err_bc_dbfs),
                "rel_loss_diff_bc": float(rel_loss_bc),
            }
            track_entry["evaluations"][pt_name][var_name] = eval_res
            print(
                f"  [{pt_name}:{var_name}] "
                f"A vs B: RMS={rms_ab_dbfs:.2f} dBFS, MaxErr={max_err_ab_dbfs:.2f} dBFS, "
                f"rel_loss={rel_loss_ab:.2e}, rel_grad={rel_grad_ab:.2e} | "
                f"B vs C: RMS={rms_bc_dbfs:.2f} dBFS, rel_loss={rel_loss_bc:.2e}",
                flush=True,
            )

    task1_report_data[prefix] = track_entry

out_path = _REPO_ROOT / "docs" / "sprint3" / "task1_audit_data.json"
out_path.write_text(json.dumps(task1_report_data, indent=2), encoding="utf-8")
print(f"\n[DONE] Saved Task 1 audit data to {out_path}", flush=True)
