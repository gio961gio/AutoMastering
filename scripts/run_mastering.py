import sys
sys.path.insert(0, 'python')
import os
import json
import torch
import soundfile as sf
import numpy as np

import py_core_dsp
from ddsp_mastering.optimize import AutoMasteringOptimizer, MasteringBenchmarkSuite

def run_scenario(name, raw_path, ref_path, out_prefix, optimize_sec=0.5, num_steps=15, lr=0.08, duration_sec=10.0):
    print(f"\n=======================================================", flush=True)
    print(f"  RUNNING {name.upper()}", flush=True)
    print(f"=======================================================", flush=True)

    # 1. Load full audio files
    raw_data, raw_sr = sf.read(raw_path)
    ref_data, ref_sr = sf.read(ref_path)

    assert raw_sr == 48000 and ref_sr == 48000, "Expected 48 kHz audio"

    if raw_data.ndim == 1:
        raw_data = np.stack([raw_data, raw_data], axis=-1)
    if ref_data.ndim == 1:
        ref_data = np.stack([ref_data, ref_data], axis=-1)

    min_len = min(raw_data.shape[0], ref_data.shape[0], int(48000 * duration_sec))
    raw_full = np.ascontiguousarray(raw_data[:min_len, :2].T, dtype=np.float32) # (2, T)
    ref_full = np.ascontiguousarray(ref_data[:min_len, :2].T, dtype=np.float32) # (2, T)

    # Representative chunk for gradient optimization
    opt_len = int(optimize_sec * 48000)
    raw_chunk = np.ascontiguousarray(raw_full[:, :opt_len])
    ref_chunk = np.ascontiguousarray(ref_full[:, :opt_len])

    raw_opt_tensor = torch.tensor(raw_chunk, dtype=torch.float32).unsqueeze(0)  # (1, 2, T_opt)
    ref_opt_tensor = torch.tensor(ref_chunk, dtype=torch.float32).unsqueeze(0)  # (1, 2, T_opt)

    raw_full_tensor = torch.tensor(raw_full, dtype=torch.float32).unsqueeze(0)  # (1, 2, T_full)
    ref_full_tensor = torch.tensor(ref_full, dtype=torch.float32).unsqueeze(0)  # (1, 2, T_full)

    # 2. Evaluate full unmastered vs reference metrics
    suite = MasteringBenchmarkSuite(sample_rate=48000.0)
    init_metrics = suite.evaluate(raw_full_tensor, ref_full_tensor, align_latency=False)
    print(f"[INITIAL UNMASTERED vs REFERENCE]", flush=True)
    print(f"  - RAW Loudness: {init_metrics['pred_lufs']:.2f} LUFS (Reference Target: {init_metrics['target_lufs']:.2f} LUFS)", flush=True)
    print(f"  - Spectral Convergence: {init_metrics['spectral_convergence']:.4f}", flush=True)
    print(f"  - Log-Spectral Distance: {init_metrics['log_spectral_distance_db']:.2f} dB", flush=True)
    print(f"  - Delta Dynamic Range (PLR): {init_metrics['delta_plr_db']:.2f} dB", flush=True)
    print(f"  - RAW Peak: {init_metrics['pred_peak_dbfs']:.2f} dBFS", flush=True)

    # 3. AutoMastering Optimization Loop (PyTorch DDSP Twin)
    optimizer = AutoMasteringOptimizer(
        sample_rate=48000.0,
        lr=lr,
        num_steps=num_steps,
        target_lufs=None,
        weight_stft=1.0,
        weight_loudness=0.8,
        weight_param_reg=0.01,
    )

    print(f"\n[OPTIMIZING] Finding optimal 41 parameters via backpropagation ({num_steps} steps on {optimize_sec}s chunk)...", flush=True)
    z_opt, _, history = optimizer.optimize(
        raw_mix=raw_opt_tensor,
        target_reference=ref_opt_tensor,
        verbose=True,
        align_latency=True,
    )

    # 4. Process FULL audio with C++20 Core DSP Engine
    print(f"\n[C++20 MASTERING ENGINE] Processing full {min_len / 48000.0:.1f}s audio at lightning speed...", flush=True)
    cpp_chain = py_core_dsp.MasteringChain(48000.0)
    cpp_chain.set_parameters_z(z_opt.tolist())
    mastered_full_np = cpp_chain.process(raw_full) # (2, T) contiguous float32
    mastered_full_tensor = torch.tensor(mastered_full_np).unsqueeze(0)

    # 5. Evaluate final full metrics
    final_metrics = suite.evaluate(mastered_full_tensor, ref_full_tensor, align_latency=True)
    print(f"\n[FINAL MASTERED vs REFERENCE]", flush=True)
    print(f"  - Mastered Loudness: {final_metrics['pred_lufs']:.2f} LUFS (Reference Target: {final_metrics['target_lufs']:.2f} LUFS)", flush=True)
    print(f"  - Spectral Convergence: {final_metrics['spectral_convergence']:.4f} (improved by {init_metrics['spectral_convergence'] - final_metrics['spectral_convergence']:+.4f})", flush=True)
    print(f"  - Log-Spectral Distance: {final_metrics['log_spectral_distance_db']:.2f} dB (improved by {init_metrics['log_spectral_distance_db'] - final_metrics['log_spectral_distance_db']:+.2f} dB)", flush=True)
    print(f"  - Mastered Peak: {final_metrics['pred_peak_dbfs']:.2f} dBFS (Ceiling violation: {final_metrics['ceiling_violation_db']:.2f} dB)", flush=True)
    print(f"  - Delta Dynamic Range (PLR): {final_metrics['delta_plr_db']:.2f} dB", flush=True)

    # 6. Export full audio WAVs and JSON preset
    out_dir = "scratch/output_mastering"
    os.makedirs(out_dir, exist_ok=True)

    unmastered_wav_path = os.path.join(out_dir, f"{out_prefix}_unmastered.wav")
    mastered_wav_path = os.path.join(out_dir, f"{out_prefix}_mastered.wav")
    ref_wav_path = os.path.join(out_dir, f"{out_prefix}_reference.wav")
    preset_json_path = os.path.join(out_dir, f"{out_prefix}_preset.json")

    sf.write(unmastered_wav_path, raw_full.T, 48000)
    sf.write(mastered_wav_path, mastered_full_np.T, 48000)
    sf.write(ref_wav_path, ref_full.T, 48000)
    optimizer.export_preset_json(z_opt, output_path=preset_json_path)

    print(f"\n[FILES SAVED ON DISK]", flush=True)
    print(f"  -> Unmastered: {unmastered_wav_path}", flush=True)
    print(f"  -> Mastered:   {mastered_wav_path}", flush=True)
    print(f"  -> Reference:  {ref_wav_path}", flush=True)
    print(f"  -> Preset:     {preset_json_path}", flush=True)

    return {
        "name": name,
        "init": init_metrics,
        "final": final_metrics,
        "unmastered": unmastered_wav_path,
        "mastered": mastered_wav_path,
        "reference": ref_wav_path,
        "preset": preset_json_path,
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run AutoMastering scenario matching")
    parser.add_argument("--raw", type=str, required=True, help="Path to raw mix WAV (48kHz)")
    parser.add_argument("--ref", type=str, required=True, help="Path to reference master WAV (48kHz)")
    parser.add_argument("--prefix", type=str, default="scenario_test", help="Output prefix name")
    parser.add_argument("--steps", type=int, default=15, help="Number of Adam optimization steps")
    parser.add_argument("--lr", type=float, default=0.08, help="Adam learning rate")
    args = parser.parse_args()

    run_scenario(
        name=args.prefix,
        raw_path=args.raw,
        ref_path=args.ref,
        out_prefix=args.prefix,
        num_steps=args.steps,
        lr=args.lr
    )
