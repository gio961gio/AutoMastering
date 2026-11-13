"""Build True-Peak-compliant Blind Audio Evaluator Pack V2 for V3c Pilot (same_track only).

This script MUST NOT rerun any optimization or C++ DSP chain. It reads the 6
already-rendered 24-bit/48kHz WAV files in `docs/sprint3/blind_v3c_pilot_wavs/`
together with `docs/sprint3/SPRINT3_V3C_BLIND_MAPPING.json` and:
  1. Applies an identical linear attenuation to both members of each pair,
     starting from the audited initial attenuations:
       - track01: 0.6575 dB
       - track02: 0.2787 dB
       - track03: 0.2663 dB
     and iteratively increases the shared attenuation by tiny uniform steps if
     needed until BOTH PCM_24 files written on disk achieve True Peak 4x <= -0.35 dBTP.
  2. Uses cryptographically independent randomness (`secrets.SystemRandom`) to:
       - shuffle which track goes into PAIR01, PAIR02, PAIR03;
       - independently assign Control_V0p vs Treatment_V3c to A vs B in each pair.
     Never uses published session seeds (`random.Random(session_seed)` is forbidden).
  3. Writes opaque 24-bit/48kHz stereo WAVs (`PAIR01_A.wav` .. `PAIR03_B.wav`) to
     `docs/sprint3/blind_v3c_pilot_wavs_v2/`.
  4. Saves the secret key ONLY to `human_validation/private/SPRINT3_V3C_BLIND_MAPPING_V2.json`
     (gitignored).
  5. Writes public key-free manifest (`SPRINT3_V3C_BLIND_AUDIO_MANIFEST_V2.{json,md}`),
     worksheet (`SPRINT3_V3C_BLIND_LISTENING_WORKSHEET_V2.md`), and the evaluator ZIP at
     `human_validation/private/blind_v3c_pilot_evaluator_pack_v2.zip` containing
     strictly ONLY the 6 WAVs, worksheet V2, and manifest V2 Markdown.
"""

import argparse
import hashlib
import json
import math
import secrets
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from ddsp_mastering.loss.loudness import calculate_integrated_loudness_bs1770
from ddsp_mastering.optimize.metrics import calculate_sample_peak_dbfs, calculate_true_peak_dbtp

TARGET_TRUE_PEAK_DBTP = -0.35
MAX_PAIR_DELTA_LUFS = 0.01
EXPECTED_SR = 48000
EXPECTED_CHANNELS = 2
EXPECTED_SUBTYPE = "PCM_24"

INITIAL_PAIR_ATTENUATION_DB: Dict[str, float] = {
    "track01": 0.6575,
    "track02": 0.2787,
    "track03": 0.2663,
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _measure_pcm24_metrics(wav_path: Path) -> Dict[str, Any]:
    info = sf.info(str(wav_path))
    if info.samplerate != EXPECTED_SR or info.channels != EXPECTED_CHANNELS or info.subtype != EXPECTED_SUBTYPE:
        raise ValueError(
            f"Unexpected WAV format for {wav_path}: sr={info.samplerate}, ch={info.channels}, subtype={info.subtype}"
        )
    audio_np, sr = sf.read(str(wav_path), dtype="float64", always_2d=True)
    tensor_audio = torch.from_numpy(audio_np.T.copy()).unsqueeze(0).to(torch.float64)
    lufs = float(calculate_integrated_loudness_bs1770(tensor_audio, sample_rate=float(sr)))
    sample_peak = float(calculate_sample_peak_dbfs(tensor_audio))
    true_peak = float(calculate_true_peak_dbtp(tensor_audio, sample_rate=float(sr)))
    return {
        "sample_rate_hz": int(info.samplerate),
        "channels": int(info.channels),
        "subtype": str(info.subtype),
        "frames": int(info.frames),
        "duration_sec": float(info.frames / float(info.samplerate)),
        "bytes": int(wav_path.stat().st_size),
        "sha256": _sha256_file(wav_path),
        "integrated_lufs_bs1770": lufs,
        "sample_peak_dbfs": sample_peak,
        "true_peak_4x_dbtp": true_peak,
    }


def _write_attenuated_pcm24(src_audio: np.ndarray, gain_linear: float, dst_path: Path) -> None:
    scaled = np.clip(src_audio * float(gain_linear), -1.0, 1.0 - 2.0 ** -23)
    sf.write(str(dst_path), scaled, EXPECTED_SR, subtype=EXPECTED_SUBTYPE)


def build_blind_pack_v2() -> None:
    docs_dir = _REPO_ROOT / "docs" / "sprint3"
    v1_wav_dir = docs_dir / "blind_v3c_pilot_wavs"
    v1_mapping_path = docs_dir / "SPRINT3_V3C_BLIND_MAPPING.json"
    v2_wav_dir = docs_dir / "blind_v3c_pilot_wavs_v2"
    private_dir = _REPO_ROOT / "human_validation" / "private"
    private_mapping_path = private_dir / "SPRINT3_V3C_BLIND_MAPPING_V2.json"
    zip_path = private_dir / "blind_v3c_pilot_evaluator_pack_v2.zip"

    manifest_json_path = docs_dir / "SPRINT3_V3C_BLIND_AUDIO_MANIFEST_V2.json"
    manifest_md_path = docs_dir / "SPRINT3_V3C_BLIND_AUDIO_MANIFEST_V2.md"
    worksheet_md_path = docs_dir / "SPRINT3_V3C_BLIND_LISTENING_WORKSHEET_V2.md"

    v2_wav_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)

    # Clean stale WAVs if regenerating cleanly
    for stale in v2_wav_dir.glob("*.wav"):
        stale.unlink()

    v1_mapping = json.loads(v1_mapping_path.read_text(encoding="utf-8"))

    # Cryptographically independent randomization (no published session seed)
    rng = secrets.SystemRandom()
    track_order = ["track01", "track02", "track03"]
    rng.shuffle(track_order)

    manifest_files: List[Dict[str, Any]] = []
    manifest_pairs: List[Dict[str, Any]] = []
    private_mapping: Dict[str, Any] = {
        "artifact_type": "sprint3_v3c_blind_mapping_v2",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "randomization_source": "secrets.SystemRandom (independent of session seeds)",
        "target_true_peak_dbtp": TARGET_TRUE_PEAK_DBTP,
        "pairs": {},
    }

    for pair_idx, track_id in enumerate(track_order, start=1):
        pair_id = f"PAIR{pair_idx:02d}"
        track_map = v1_mapping[track_id]
        inv_files = {arm_name: fname for fname, arm_name in track_map["files"].items()}
        src_paths = {
            "Control_V0p": v1_wav_dir / inv_files["Control_V0p"],
            "Treatment_V3c": v1_wav_dir / inv_files["Treatment_V3c"],
        }
        for arm_name, p in src_paths.items():
            if not p.exists():
                raise FileNotFoundError(f"Missing source WAV V1 for {track_id}/{arm_name}: {p}")

        before_metrics = {arm: _measure_pcm24_metrics(p) for arm, p in src_paths.items()}
        src_audios = {arm: sf.read(str(p), dtype="float64", always_2d=True)[0] for arm, p in src_paths.items()}

        arms_order = ["Control_V0p", "Treatment_V3c"]
        rng.shuffle(arms_order)
        label_to_arm = {"A": arms_order[0], "B": arms_order[1]}

        # Apply identical linear attenuation to both members of the pair, iterating if needed
        init_atten_db = INITIAL_PAIR_ATTENUATION_DB[track_id]
        atten_db = init_atten_db
        step_iterations = 0
        after_metrics: Dict[str, Dict[str, Any]] = {}

        for _attempt in range(25):
            gain_lin = 10.0 ** (-atten_db / 20.0)
            out_paths: Dict[str, Path] = {}
            after_metrics = {}
            for lbl in ("A", "B"):
                arm = label_to_arm[lbl]
                out_path = v2_wav_dir / f"{pair_id}_{lbl}.wav"
                _write_attenuated_pcm24(src_audios[arm], gain_lin, out_path)
                out_paths[lbl] = out_path
                after_metrics[lbl] = _measure_pcm24_metrics(out_path)

            max_tp = max(after_metrics["A"]["true_peak_4x_dbtp"], after_metrics["B"]["true_peak_4x_dbtp"])
            if max_tp <= TARGET_TRUE_PEAK_DBTP:
                break
            overshoot = max_tp - TARGET_TRUE_PEAK_DBTP
            atten_db = round(atten_db + max(overshoot + 0.002, 0.005), 4)
            step_iterations += 1
        else:
            raise RuntimeError(f"Failed to bring {pair_id} ({track_id}) <= {TARGET_TRUE_PEAK_DBTP} dBTP")

        final_gain_lin = 10.0 ** (-atten_db / 20.0)
        delta_lufs = float(after_metrics["A"]["integrated_lufs_bs1770"] - after_metrics["B"]["integrated_lufs_bs1770"])
        abs_delta_lufs = abs(delta_lufs)
        if abs_delta_lufs > MAX_PAIR_DELTA_LUFS:
            raise RuntimeError(f"{pair_id} |delta LUFS|={abs_delta_lufs:.6f} exceeds {MAX_PAIR_DELTA_LUFS}")

        for lbl in ("A", "B"):
            arm = label_to_arm[lbl]
            bm = before_metrics[arm]
            am = after_metrics[lbl]
            fname = f"{pair_id}_{lbl}.wav"
            manifest_files.append(
                {
                    "pair_id": pair_id,
                    "blind_label": lbl,
                    "file_name": fname,
                    "relative_path": f"docs/sprint3/blind_v3c_pilot_wavs_v2/{fname}",
                    "sample_rate_hz": am["sample_rate_hz"],
                    "channels": am["channels"],
                    "subtype": am["subtype"],
                    "frames": am["frames"],
                    "duration_sec": round(am["duration_sec"], 6),
                    "bytes": am["bytes"],
                    "sha256": am["sha256"],
                    "initial_common_attenuation_db": round(init_atten_db, 6),
                    "common_attenuation_db": round(atten_db, 6),
                    "common_linear_gain": final_gain_lin,
                    "sample_peak_before_dbfs": round(bm["sample_peak_dbfs"], 6),
                    "sample_peak_after_dbfs": round(am["sample_peak_dbfs"], 6),
                    "true_peak_4x_before_dbtp": round(bm["true_peak_4x_dbtp"], 6),
                    "true_peak_4x_after_dbtp": round(am["true_peak_4x_dbtp"], 6),
                    "integrated_lufs_before_bs1770": round(bm["integrated_lufs_bs1770"], 6),
                    "integrated_lufs_after_bs1770": round(am["integrated_lufs_bs1770"], 6),
                }
            )

        manifest_pairs.append(
            {
                "pair_id": pair_id,
                "file_A": f"{pair_id}_A.wav",
                "file_B": f"{pair_id}_B.wav",
                "initial_common_attenuation_db": round(init_atten_db, 6),
                "common_attenuation_db": round(atten_db, 6),
                "common_linear_gain": final_gain_lin,
                "extra_attenuation_iterations": step_iterations,
                "lufs_A_after": round(after_metrics["A"]["integrated_lufs_bs1770"], 6),
                "lufs_B_after": round(after_metrics["B"]["integrated_lufs_bs1770"], 6),
                "delta_lufs_A_minus_B": round(delta_lufs, 6),
                "abs_delta_lufs": round(abs_delta_lufs, 6),
                "max_true_peak_4x_after_dbtp": round(
                    max(after_metrics["A"]["true_peak_4x_dbtp"], after_metrics["B"]["true_peak_4x_dbtp"]), 6
                ),
                "all_true_peak_compliant": True,
            }
        )

        private_mapping["pairs"][pair_id] = {
            "session": track_id,
            "initial_common_attenuation_db": init_atten_db,
            "common_attenuation_db": atten_db,
            "common_linear_gain": final_gain_lin,
            "extra_attenuation_iterations": step_iterations,
            "files": {
                f"{pair_id}_A.wav": {
                    "arm": label_to_arm["A"],
                    "source_v1_wav": inv_files[label_to_arm["A"]],
                    "true_peak_before_dbtp": before_metrics[label_to_arm["A"]]["true_peak_4x_dbtp"],
                    "true_peak_after_dbtp": after_metrics["A"]["true_peak_4x_dbtp"],
                },
                f"{pair_id}_B.wav": {
                    "arm": label_to_arm["B"],
                    "source_v1_wav": inv_files[label_to_arm["B"]],
                    "true_peak_before_dbtp": before_metrics[label_to_arm["B"]]["true_peak_4x_dbtp"],
                    "true_peak_after_dbtp": after_metrics["B"]["true_peak_4x_dbtp"],
                },
            },
            "label_to_arm": label_to_arm,
        }

    # Write private key JSON (gitignored)
    private_mapping_path.write_text(json.dumps(private_mapping, indent=2) + "\n", encoding="utf-8")

    # Write public key-free manifest JSON
    max_abs_delta_lufs = max(p["abs_delta_lufs"] for p in manifest_pairs)
    max_after_tp = max(f["true_peak_4x_after_dbtp"] for f in manifest_files)
    manifest_json = {
        "artifact_type": "sprint3_v3c_blind_audio_manifest_v2",
        "description": "True-Peak-compliant (<= -0.35 dBTP) key-free blind listening manifest V2 (opaque PAIR01..PAIR03).",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wav_count": len(manifest_files),
        "expected_sample_rate_hz": EXPECTED_SR,
        "expected_channels": EXPECTED_CHANNELS,
        "expected_subtype": EXPECTED_SUBTYPE,
        "target_true_peak_dbtp": TARGET_TRUE_PEAK_DBTP,
        "max_true_peak_4x_after_dbtp": round(max_after_tp, 6),
        "max_abs_delta_lufs_across_pairs": round(max_abs_delta_lufs, 6),
        "pair_lufs_control_target_lu": MAX_PAIR_DELTA_LUFS,
        "independent_lufs_normalization_applied": False,
        "common_pair_attenuation_only": True,
        "contains_secret_mapping": False,
        "evaluator_zip_relative_path": "human_validation/private/blind_v3c_pilot_evaluator_pack_v2.zip",
        "private_mapping_relative_path": "human_validation/private/SPRINT3_V3C_BLIND_MAPPING_V2.json",
        "files": manifest_files,
        "pair_summary": manifest_pairs,
    }
    manifest_json_path.write_text(json.dumps(manifest_json, indent=2) + "\n", encoding="utf-8")

    # Write public key-free manifest MD
    md_lines = [
        "# SPRINT 3 — V3C PILOT BLIND AUDIO MANIFEST V2 (TRUE PEAK COMPLIANT)",
        "",
        "**Scopo:** Pacchetto cieco V2 (`48 kHz`, stereo, `PCM_24`) con nomi opachi (`PAIR01`–`PAIR03`) e conformità verificata **True Peak 4x <= -0.35 dBTP** su tutti i file, ottenuta tramite sola attenuazione lineare identica per coppia sui render esistenti.",
        "**Nota di Blinding:** Questo manifest pubblico contiene esclusivamente metriche tecniche di verifica ed è privo di informazioni identificative delle coppie e delle varianti.",
        "",
        "## 1. Inventario File WAV (`blind_v3c_pilot_wavs_v2/`)",
        "",
        "| File | Coppia | Etichetta | SR / Ch / Formato | Frames | Attenuazione Comune (dB) | TP 4x Prima (dBTP) | **TP 4x Dopo (dBTP)** | LUFS Dopo (BS.1770) | SHA-256 |",
        "|:---|:---:|:---:|:---:|---:|---:|---:|---:|---:|:---|",
    ]
    for f in manifest_files:
        md_lines.append(
            f"| `{f['file_name']}` | `{f['pair_id']}` | `{f['blind_label']}` | {f['sample_rate_hz']} Hz / {f['channels']}ch / `{f['subtype']}` | {f['frames']} | -{f['common_attenuation_db']:.4f} dB | {f['true_peak_4x_before_dbtp']:+.4f} | **{f['true_peak_4x_after_dbtp']:+.4f}** | {f['integrated_lufs_after_bs1770']:.4f} | `{f['sha256']}` |"
        )
    md_lines += [
        "",
        "## 2. Riepilogo per Coppia & Scarto LUFS (`<= 0.01 LU`)",
        "",
        "| Coppia | File A | File B | Attenuazione Iniziale -> Finale (dB) | Guadagno Lineare Comune | LUFS A | LUFS B | `|ΔLUFS|` (LU) | Max TP 4x Dopo (dBTP) |",
        "|:---|:---|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for p in manifest_pairs:
        md_lines.append(
            f"| `{p['pair_id']}` | `{p['file_A']}` | `{p['file_B']}` | {p['initial_common_attenuation_db']:.4f} -> {p['common_attenuation_db']:.4f} | {p['common_linear_gain']:.8f} | {p['lufs_A_after']:.4f} | {p['lufs_B_after']:.4f} | {p['abs_delta_lufs']:.6f} | {p['max_true_peak_4x_after_dbtp']:+.4f} |"
        )
    md_lines += [
        "",
        f"- **Massimo True Peak 4x (tutti i 6 WAV V2):** `{max_after_tp:+.4f} dBTP` (requisito `<= -0.35 dBTP`).",
        f"- **Massimo scarto di coppia `|ΔLUFS|`:** `{max_abs_delta_lufs:.6f} LU` (target controllo `<= 0.01 LU`, nessuna normalizzazione LUFS indipendente).",
        "- **Contenuto archivio evaluators V2:** esclusivamente i 6 WAV in `blind_v3c_pilot_wavs_v2/`, `SPRINT3_V3C_BLIND_LISTENING_WORKSHEET_V2.md` e `SPRINT3_V3C_BLIND_AUDIO_MANIFEST_V2.md`.",
        "",
    ]
    manifest_md_path.write_text("\n".join(md_lines), encoding="utf-8")

    # Write public key-free worksheet V2
    ws_lines = [
        "# SPRINT 3 — V3C PILOT BLIND LISTENING WORKSHEET V2",
        "",
        "**Istruzioni per il Valutatore:**",
        "Per ciascuna delle 3 coppie opache (`PAIR01`, `PAIR02`, `PAIR03`) sono forniti due master `48 kHz`, stereo, `PCM_24` già livellati (`|ΔLUFS| <= 0.01 LU`) e conformi True Peak (`<= -0.35 dBTP`).",
        "Ascoltare sia in stereo che in mono fold-down (se disponibile) prima di esprimere la preferenza.",
        "",
        "| Coppia | File A | File B | Immagine Stereo (A/B/Pari) | Solidità Mono Fold-Down (A/B/Pari) | Controllo Basso & Sub (A/B/Pari) | Trasparenza & Timbro (A/B/Pari) | Preferenza Globale (A/B) | Note d'Ascolto |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|",
        "| **`PAIR01`** | `PAIR01_A.wav` | `PAIR01_B.wav` | [ ] A  [ ] B  [ ] Pari | [ ] A  [ ] B  [ ] Pari | [ ] A  [ ] B  [ ] Pari | [ ] A  [ ] B  [ ] Pari | [ ] A  [ ] B | |",
        "| **`PAIR02`** | `PAIR02_A.wav` | `PAIR02_B.wav` | [ ] A  [ ] B  [ ] Pari | [ ] A  [ ] B  [ ] Pari | [ ] A  [ ] B  [ ] Pari | [ ] A  [ ] B  [ ] Pari | [ ] A  [ ] B | |",
        "| **`PAIR03`** | `PAIR03_A.wav` | `PAIR03_B.wav` | [ ] A  [ ] B  [ ] Pari | [ ] A  [ ] B  [ ] Pari | [ ] A  [ ] B  [ ] Pari | [ ] A  [ ] B  [ ] Pari | [ ] A  [ ] B | |",
        "",
    ]
    worksheet_md_path.write_text("\n".join(ws_lines), encoding="utf-8")

    # Build strict evaluator-only ZIP (ONLY 6 WAVs + worksheet V2 + manifest V2 MD)
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in manifest_files:
            wav_p = v2_wav_dir / f["file_name"]
            zf.write(wav_p, arcname=f"blind_v3c_pilot_wavs_v2/{f['file_name']}")
        zf.write(worksheet_md_path, arcname="SPRINT3_V3C_BLIND_LISTENING_WORKSHEET_V2.md")
        zf.write(manifest_md_path, arcname="SPRINT3_V3C_BLIND_AUDIO_MANIFEST_V2.md")

    print(f"[DONE] Built 6 True-Peak-compliant V2 WAVs in: {v2_wav_dir}")
    print(f"[DONE] Saved public V2 manifest JSON/MD: {manifest_json_path}, {manifest_md_path}")
    print(f"[DONE] Saved public V2 worksheet: {worksheet_md_path}")
    print(f"[DONE] Saved private V2 mapping (gitignored): {private_mapping_path}")
    print(f"[DONE] Built evaluator-only V2 zip (gitignored): {zip_path}")
    for p in manifest_pairs:
        print(
            f"  - {p['pair_id']}: atten={p['common_attenuation_db']:.4f} dB "
            f"(iters={p['extra_attenuation_iterations']}), "
            f"max_tp={p['max_true_peak_4x_after_dbtp']:+.4f} dBTP, "
            f"|dLUFS|={p['abs_delta_lufs']:.6f} LU"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build True-Peak-compliant V3c Blind Pack V2.")
    parser.parse_args()
    build_blind_pack_v2()


if __name__ == "__main__":
    main()
