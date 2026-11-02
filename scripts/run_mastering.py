"""AutoMastering Production Execution Script (Sprint 2 Architecture).

Implements:
  - Both Reference Modes: 'same_track' (with time alignment) and 'stylistic' (aggregate perceptual loss).
  - Multi-Chunk Evaluation across 4-6 diverse, representative windows.
  - Complete full-track mastering (no truncation) via C++20 Core DSP.
  - Full session provenance manifest generation.
"""

import sys
import os

# Ensure root and python directories are in sys.path for py_core_dsp and ddsp_mastering
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_python_dir = os.path.join(_repo_root, "python")
for _p in [_repo_root, _python_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json
import argparse
from typing import Optional
import numpy as np
import soundfile as sf
import torch

import py_core_dsp
from ddsp_mastering.data import (
    ReferenceMode,
    AlignmentResult,
    estimate_time_alignment,
    apply_same_track_alignment,
    get_same_track_slice_indices,
    MultiChunkSelector,
    ChunkSpec,
)
from ddsp_mastering.optimize import (
    AutoMasteringOptimizer,
    MasteringBenchmarkSuite,
    MasteringManifest,
)
from ddsp_mastering.param_space import (
    NUM_PARAMS,
    PARAM_SPECS,
    map_z_to_theta,
    unpack_theta_dict,
)


def find_highest_energy_window(signal: np.ndarray, window_len: int, hop_len: int = 2400) -> int:
    """Find start index of window with highest RMS energy (kept for backward compatibility)."""
    T = signal.shape[-1]
    if T <= window_len:
        return 0
    mono_sq = np.mean(signal ** 2, axis=0)
    cs = np.pad(np.cumsum(mono_sq), (1, 0))
    starts = np.arange(0, T - window_len + 1, hop_len)
    window_energies = cs[starts + window_len] - cs[starts]
    best_idx = starts[np.argmax(window_energies)]
    return int(best_idx)


def run_mastering_session(
    raw_path: str,
    ref_path: str,
    out_prefix: str = "master_session",
    mode: str = "same_track",
    num_chunks: int = 4,
    chunk_len_sec: float = 3.0,
    seed: int = 42,
    num_steps: int = 40,
    lr: float = 0.04,
    duration_sec: Optional[float] = None,
    out_dir: str = "scratch/output_mastering",
):
    print(f"\n=======================================================", flush=True)
    print(f"  AUTOMASTERING POLIMI — SPRINT 2 PRODUCTION ENGINE", flush=True)
    print(f"  Mode: {mode.upper()} | Chunks: {num_chunks} x {chunk_len_sec:.1f}s | Seed: {seed}", flush=True)
    print(f"=======================================================", flush=True)

    ref_mode = ReferenceMode(mode)

    # 1. Load full audio files
    raw_data, raw_sr = sf.read(raw_path)
    ref_data, ref_sr = sf.read(ref_path)

    if raw_sr != 48000 or ref_sr != 48000:
        raise ValueError(f"Expected 48 kHz audio, got raw={raw_sr}Hz, ref={ref_sr}Hz")

    if raw_data.ndim == 1:
        raw_data = np.stack([raw_data, raw_data], axis=-1)
    if ref_data.ndim == 1:
        ref_data = np.stack([ref_data, ref_data], axis=-1)

    # Handle duration truncation if explicitly requested by user; otherwise preserve 100% full length
    if duration_sec is not None and duration_sec > 0:
        max_samples = int(duration_sec * 48000)
        raw_data = raw_data[:max_samples]
        if ref_mode == ReferenceMode.SAME_TRACK:
            ref_data = ref_data[:max_samples]

    raw_full = np.ascontiguousarray(raw_data.T[:2], dtype=np.float32)  # [2, T_raw]
    ref_full = np.ascontiguousarray(ref_data.T[:2], dtype=np.float32)  # [2, T_ref]

    raw_duration = raw_full.shape[1] / 48000.0
    ref_duration = ref_full.shape[1] / 48000.0
    print(f"[AUDIO LOADED] Mix: {raw_duration:.2f}s | Reference: {ref_duration:.2f}s", flush=True)

    # 2. Time Alignment (Same-Track only)
    alignment_meta = {}
    if ref_mode == ReferenceMode.SAME_TRACK:
        print("[ALIGNMENT] Estimating time alignment via normalized cross-correlation...", flush=True)
        alignment = estimate_time_alignment(raw_full, ref_full, sample_rate=48000.0)
        alignment_meta = alignment.to_dict()
        print(
            f"  -> Lag: {alignment.lag_samples} samples ({alignment.lag_seconds * 1000.0:+.2f} ms) | "
            f"Sub-sample offset: {alignment.subsample_offset:+.3f} | "
            f"Corr: {alignment.correlation_coefficient:.4f} (Reliable: {alignment.is_aligned})",
            flush=True,
        )
        # Symmetrically align reference on mix timeline for paired evaluation
        raw_aligned, ref_aligned = apply_same_track_alignment(raw_full, ref_full, alignment)
    else:
        print("[STYLISTIC MODE] Waveform cross-correlation skipped. Aggregate perceptual matching active.", flush=True)
        raw_aligned = raw_full
        ref_aligned = ref_full
        alignment = None

    # 3. Multi-Chunk Representative Selection
    print(f"\n[MULTI-CHUNK SELECTION] Selecting {num_chunks} representative windows ({chunk_len_sec}s each)...", flush=True)
    selector = MultiChunkSelector(
        sample_rate=48000.0,
        num_chunks=num_chunks,
        chunk_duration_sec=chunk_len_sec,
        seed=seed,
    )
    mix_chunks_spec = selector.select_chunks(raw_aligned)
    mix_chunks_np = selector.extract_chunk_tensors(raw_aligned, mix_chunks_spec)

    print(f"  Selected {len(mix_chunks_spec)} diverse chunks across the track:", flush=True)
    for c in mix_chunks_spec:
        print(
            f"   Chunk {c.chunk_id}: {c.start_sec:.2f}s - {c.end_sec:.2f}s | "
            f"RMS: {c.rms_db:.1f} dBFS | Crest: {c.crest_factor_db:.1f} dB | Centroid: {c.spectral_centroid_hz:.0f} Hz",
            flush=True,
        )

    # Convert to PyTorch tensors [1, 2, T_chunk]
    mix_chunk_tensors = [torch.tensor(c).unsqueeze(0) for c in mix_chunks_np]

    if ref_mode == ReferenceMode.SAME_TRACK:
        # Paired chunks extracted at exact same offsets
        ref_chunk_tensors = [
            torch.tensor(ref_aligned[:, c.start_sample : c.end_sample]).unsqueeze(0)
            for c in mix_chunks_spec
        ]
    else:
        # Reference chunks extracted independently from reference track
        ref_chunks_spec = selector.select_chunks(ref_full)
        ref_chunks_np = selector.extract_chunk_tensors(ref_full, ref_chunks_spec)
        ref_chunk_tensors = [torch.tensor(c).unsqueeze(0) for c in ref_chunks_np]

    # 4. Multi-Chunk AutoMastering Optimization Loop
    optimizer = AutoMasteringOptimizer(
        sample_rate=48000.0,
        lr=lr,
        num_steps=num_steps,
        target_lufs=None,
        weight_stft=1.0,
        weight_loudness=0.8,
        weight_param_reg=0.01,
        use_hybrid_eq=True,
    )

    print(f"\n[OPTIMIZING] Running multi-chunk backpropagation ({num_steps} steps)...", flush=True)
    z_opt, _, history = optimizer.optimize_multi_chunk(
        mix_chunks=mix_chunk_tensors,
        ref_chunks=ref_chunk_tensors,
        reference_mode=mode,
        verbose=True,
        align_latency=True,
    )

    # 5. Full Track Mastering via C++20 Core DSP Engine (No Truncation!)
    print(f"\n[C++20 MASTERING ENGINE] Processing full {raw_duration:.2f}s track...", flush=True)
    cpp_chain = py_core_dsp.MasteringChain(48000.0)
    cpp_chain.set_parameters_z(z_opt.tolist())
    mastered_full_np = cpp_chain.process(raw_full)  # [2, T_raw] full audio!

    # 6. Quantitative Pre/Post Evaluation
    suite = MasteringBenchmarkSuite(sample_rate=48000.0)

    if ref_mode == ReferenceMode.SAME_TRACK:
        # Same-track: evaluated on time-aligned signals with exact common temporal support
        mix_start, ref_start, eval_len = get_same_track_slice_indices(
            alignment,
            mix_len=raw_full.shape[1],
            ref_len=ref_full.shape[1],
            enable_subsample=True,
        )
        raw_eval_tensor = torch.tensor(raw_aligned).unsqueeze(0)
        ref_eval_tensor = torch.tensor(ref_aligned).unsqueeze(0)

        # S2.1-FINAL: Extract mastered segment precisely matching raw_aligned temporal support
        mastered_eval_segment = mastered_full_np[:, mix_start : mix_start + eval_len]
        mastered_eval_tensor = torch.tensor(mastered_eval_segment).unsqueeze(0)

        # S2.1-FINAL Problema 1: Pre-mastering raw mix has NO DDSP chain latency -> align_latency=False!
        pre_metrics = suite.evaluate(
            raw_eval_tensor,
            ref_eval_tensor,
            align_latency=False,
            reference_mode="same_track",
        )
        # S2.1-FINAL Problema 2: Post-mastering DDSP output has 159 smp latency -> align_latency=True!
        post_metrics = suite.evaluate(
            mastered_eval_tensor,
            ref_eval_tensor,
            align_latency=True,
            reference_mode="same_track",
        )
    elif ref_mode == ReferenceMode.STYLISTIC:
        # Stylistic mode: do NOT force identical duration; evaluate aggregate distribution descriptors
        ref_eval_tensor = torch.tensor(ref_full).unsqueeze(0)
        raw_eval_tensor = torch.tensor(raw_full).unsqueeze(0)
        mastered_eval_tensor = torch.tensor(mastered_full_np).unsqueeze(0)

        pre_metrics = suite.evaluate(
            raw_eval_tensor,
            ref_eval_tensor,
            align_latency=False,
            reference_mode="stylistic",
        )
        post_metrics = suite.evaluate(
            mastered_eval_tensor,
            ref_eval_tensor,
            align_latency=False,
            reference_mode="stylistic",
        )
    else:
        raise ValueError(f"Unrecognized reference mode: {ref_mode}")

    print(f"\n[BENCHMARK EVALUATION — {ref_mode.value.upper()} MODE]", flush=True)
    print(f"  - Loudness Standard:       {post_metrics['loudness_standard']}", flush=True)
    print(
        f"  - Pre  Mastering Loudness: {pre_metrics['pred_lufs']:.2f} LUFS [{pre_metrics['pred_loudness_method']}] "
        f"(Reference: {pre_metrics['target_lufs']:.2f} LUFS [{pre_metrics['target_loudness_method']}])",
        flush=True,
    )
    print(
        f"  - Post Mastering Loudness: {post_metrics['pred_lufs']:.2f} LUFS [{post_metrics['pred_loudness_method']}] "
        f"(Delta: {post_metrics['delta_lufs']:.2f} LU)",
        flush=True,
    )
    print(f"  - Post Dynamic Range (PLR): {post_metrics['pred_plr_db']:.2f} dB (Reference: {post_metrics['target_plr_db']:.2f} dB, Delta: {post_metrics['delta_plr_db']:.2f} dB)", flush=True)
    print(
        f"  - Sample Peak Output: {post_metrics['pred_sample_peak_dbfs']:.2f} dBFS "
        f"(Sample Ceiling violation: {post_metrics['sample_peak_ceiling_violation_db']:.2f} dB)",
        flush=True,
    )
    print(
        f"  - True Peak Output:   {post_metrics['pred_true_peak_dbtp']:.2f} dBTP "
        f"(True Peak Ceiling violation: {post_metrics['true_peak_ceiling_violation_db']:.2f} dB)",
        flush=True,
    )

    if ref_mode == ReferenceMode.SAME_TRACK:
        print(f"  - Spectral Convergence: {post_metrics['spectral_convergence']:.4f}", flush=True)
        print(f"  - Log-Spectral Distance: {post_metrics['log_spectral_distance_db']:.2f} dB", flush=True)
    else:
        print(f"  - Aggregate Spectral Distance: {post_metrics['aggregate_spectral_distance']:.4f}", flush=True)
        print(f"  - (Frame-by-frame STFT / LSD strictly omitted for stylistic reference)", flush=True)

    # 7. Export Outputs & Manifest
    os.makedirs(out_dir, exist_ok=True)
    mastered_wav_path = os.path.join(out_dir, f"{out_prefix}_mastered.wav")
    preset_json_path = os.path.join(out_dir, f"{out_prefix}_preset.json")
    manifest_path = os.path.join(out_dir, f"{out_prefix}_manifest.json")

    # Save full mastered WAV
    sf.write(mastered_wav_path, mastered_full_np.T, 48000)
    # Save canonical preset JSON
    optimizer.export_preset_json(z_opt, output_path=preset_json_path)

    # Build and save session manifest
    theta = map_z_to_theta(z_opt)
    param_dict = {spec.name: float(theta[i].item()) for i, spec in enumerate(PARAM_SPECS)}

    manifest = MasteringManifest(
        reference_mode=mode,
        input_track_path=raw_path,
        input_sample_rate=48000.0,
        input_duration_sec=raw_duration,
        reference_track_path=ref_path,
        reference_sample_rate=48000.0,
        reference_duration_sec=ref_duration,
        alignment=alignment_meta,
        chunks=[c.to_dict() for c in mix_chunks_spec],
        chunk_selection_config={
            "num_chunks": num_chunks,
            "chunk_duration_sec": chunk_len_sec,
            "seed": seed,
        },
        optimization_config={
            "steps": num_steps,
            "lr": lr,
            "use_hybrid_eq": True,
        },
        normalized_z=z_opt.tolist(),
        physical_theta=theta.tolist(),
        parameters=param_dict,
        metrics_pre=pre_metrics,
        metrics_post=post_metrics,
        software_info={
            "python": sys.version.split()[0],
            "torch": torch.__version__,
        },
    )
    manifest.to_json(output_path=manifest_path)

    print(f"\n[FILES SAVED ON DISK]", flush=True)
    print(f"  -> Full Mastered WAV: {mastered_wav_path}", flush=True)
    print(f"  -> Canonical Preset:  {preset_json_path}", flush=True)
    print(f"  -> Session Manifest:  {manifest_path}", flush=True)

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoMastering Sprint 2 Multi-Chunk Engine")
    parser.add_argument("--raw", type=str, required=True, help="Path to raw mix WAV (48kHz)")
    parser.add_argument("--ref", type=str, required=True, help="Path to reference master WAV (48kHz)")
    parser.add_argument("--prefix", type=str, default="master_session", help="Output prefix name")
    parser.add_argument("--mode", type=str, choices=["same_track", "stylistic"], default="same_track")
    parser.add_argument("--chunks", type=int, default=4, help="Number of representative chunks")
    parser.add_argument("--chunk-len", type=float, default=3.0, help="Duration of each chunk in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic chunk selection seed")
    parser.add_argument("--steps", type=int, default=30, help="Number of Adam optimization steps")
    parser.add_argument("--lr", type=float, default=0.04, help="Learning rate")
    parser.add_argument("--duration", type=float, default=None, help="Optional duration limit in seconds (default: full track)")
    args = parser.parse_args()

    run_mastering_session(
        raw_path=args.raw,
        ref_path=args.ref,
        out_prefix=args.prefix,
        mode=args.mode,
        num_chunks=args.chunks,
        chunk_len_sec=args.chunk_len,
        seed=args.seed,
        num_steps=args.steps,
        lr=args.lr,
        duration_sec=args.duration,
    )
