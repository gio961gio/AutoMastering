"""Blind Listening Set & Separate Key Generator.

Creates anonymized WAV copies (`blind_X.wav`, `blind_Y.wav`, ...) in a deterministic
pseudo-random order seeded by `seed`, strips any identifying tags by writing clean
PCM/Float WAV data from memory, and stores the secret correspondence mapping in a
separate `*_blind_key.json` file alongside an un-keyed listening worksheet.
"""

import argparse
import csv
import json
import os
import random
import string
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import soundfile as sf

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from scripts.human_validation.validate_inputs import (
    check_overwrite_allowed,
    compute_file_sha256,
    load_and_validate_wav,
)


def generate_blind_mapping(
    variant_names: List[str],
    seed: int = 2026,
    prefix: str = "blind_",
) -> Dict[str, str]:
    """Generate a reproducible bijective mapping {blind_label: variant_name}.

    Args:
        variant_names: List of distinct ground-truth variant identifiers.
        seed: Random seed for deterministic permutation (default: 2026).
        prefix: Prefix for anonymous labels (default: 'blind_').

    Returns:
        Dictionary mapping anonymous label (e.g. 'blind_A') -> ground-truth variant name.
    """
    if not variant_names:
        raise ValueError("variant_names cannot be empty.")
    if len(set(variant_names)) != len(variant_names):
        raise ValueError(f"variant_names must be unique, got duplicates in {variant_names}.")

    sorted_variants = sorted(variant_names)
    rng = random.Random(int(seed))
    shuffled = list(sorted_variants)
    rng.shuffle(shuffled)

    mapping: Dict[str, str] = {}
    for idx, var_name in enumerate(shuffled):
        if idx < 26:
            anon_suffix = string.ascii_uppercase[idx]
        else:
            anon_suffix = f"{idx + 1:02d}"
        anon_label = f"{prefix}{anon_suffix}"
        mapping[anon_label] = var_name

    return mapping


def create_blind_listening_set(
    variant_files: Dict[str, str],
    output_dir: str,
    seed: int = 2026,
    session_id: str = "session",
    key_output_path: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Create anonymized WAV files for blind evaluation and save the secret key separately.

    Args:
        variant_files: Mapping of {ground_truth_variant_id: wav_file_path}.
        output_dir: Directory where anonymous WAVs and blank worksheet are written.
        seed: Deterministic RNG seed.
        session_id: Identifier for the blind session.
        key_output_path: Optional path for the secret key JSON. Defaults to
                         `<output_dir>/<session_id>_blind_key.json`.
        overwrite: Allow overwriting existing files.

    Returns:
        Dictionary containing the blind session metadata and mapping.
    """
    if not variant_files:
        raise ValueError("variant_files cannot be empty.")

    os.makedirs(output_dir, exist_ok=True)
    if key_output_path is None:
        key_output_path = os.path.join(output_dir, f"{session_id}_blind_key.json")

    sheet_csv_path = os.path.join(output_dir, f"{session_id}_listening_sheet.csv")
    check_overwrite_allowed(key_output_path, overwrite=overwrite)
    check_overwrite_allowed(sheet_csv_path, overwrite=overwrite)

    mapping = generate_blind_mapping(list(variant_files.keys()), seed=seed, prefix="blind_")

    blind_files_info: Dict[str, Dict[str, Any]] = {}
    public_files_list: List[Dict[str, Any]] = []

    for anon_label, gt_variant in mapping.items():
        src_path = os.path.abspath(variant_files[gt_variant])
        audio_2xT, meta = load_and_validate_wav(src_path, required_sr=48000, allow_mono=True)

        anon_wav_name = f"{anon_label}.wav"
        anon_wav_path = os.path.abspath(os.path.join(output_dir, anon_wav_name))
        check_overwrite_allowed(anon_wav_path, overwrite=overwrite)

        # Write clean audio samples without copying any source container tags/metadata
        sf.write(anon_wav_path, audio_2xT.T, 48000, subtype="FLOAT")
        anon_sha256 = compute_file_sha256(anon_wav_path)

        blind_files_info[anon_label] = {
            "anonymous_file": anon_wav_name,
            "anonymous_sha256": anon_sha256,
            "ground_truth_variant": gt_variant,
            "source_file_name": meta["file_name"],
            "source_sha256": meta["sha256"],
            "duration_sec": meta["duration_sec"],
            "num_samples": meta["num_samples"],
        }
        public_files_list.append({
            "anonymous_label": anon_label,
            "anonymous_file": anon_wav_name,
            "duration_sec": meta["duration_sec"],
            "num_samples": meta["num_samples"],
        })

    # Write un-keyed CSV worksheet for the listener
    fieldnames = [
        "session_id",
        "anonymous_label",
        "anonymous_file",
        "tonal_balance_1_5",
        "bass_control_1_5",
        "vocal_clarity_1_5",
        "transient_punch_1_5",
        "stereo_image_1_5",
        "dynamic_naturalness_1_5",
        "distortion_artifacts_1_5",
        "overall_preference_1_5",
        "audibility_vs_others",
        "confidence_1_5",
        "artifacts_noted",
        "free_notes",
    ]
    with open(sheet_csv_path, "w", encoding="utf-8", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        writer.writeheader()
        for item in public_files_list:
            writer.writerow({
                "session_id": session_id,
                "anonymous_label": item["anonymous_label"],
                "anonymous_file": item["anonymous_file"],
                "tonal_balance_1_5": "",
                "bass_control_1_5": "",
                "vocal_clarity_1_5": "",
                "transient_punch_1_5": "",
                "stereo_image_1_5": "",
                "dynamic_naturalness_1_5": "",
                "distortion_artifacts_1_5": "",
                "overall_preference_1_5": "",
                "audibility_vs_others": "",
                "confidence_1_5": "",
                "artifacts_noted": "",
                "free_notes": "",
            })

    key_payload: Dict[str, Any] = {
        "warning": "DO NOT OPEN BEFORE COMPLETING THE BLIND LISTENING EVALUATION.",
        "session_id": session_id,
        "seed": int(seed),
        "mapping_anonymous_to_ground_truth": mapping,
        "mapping_ground_truth_to_anonymous": {v: k for k, v in mapping.items()},
        "details": blind_files_info,
    }
    with open(key_output_path, "w", encoding="utf-8") as f:
        json.dump(key_payload, f, indent=2)

    return {
        "session_id": session_id,
        "seed": int(seed),
        "output_dir": os.path.abspath(output_dir),
        "listening_sheet_csv": os.path.abspath(sheet_csv_path),
        "secret_key_json": os.path.abspath(key_output_path),
        "anonymous_files": public_files_list,
        "_internal_key": key_payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create anonymized WAV files and a separate secret key for blind listening."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        help="Variant label=path pairs (e.g. A_bypass=path/to/A.wav C_neutral=path/to/C.wav).",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Optional JSON manifest containing 'files' or 'variants' mapping.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for anonymous WAV files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for deterministic blind ordering (default: 2026).",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default="blind_session",
        help="Identifier for this listening session.",
    )
    parser.add_argument(
        "--key-out",
        type=str,
        default=None,
        help="Optional separate path for the secret key JSON.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing files.",
    )
    args = parser.parse_args()

    variant_map: Dict[str, str] = {}
    if args.manifest:
        with open(args.manifest, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "files" in data and isinstance(data["files"], dict):
            for k, v in data["files"].items():
                variant_map[str(k)] = str(v)
        elif "variants" in data and isinstance(data["variants"], dict):
            for k, v in data["variants"].items():
                if isinstance(v, dict) and "output_path" in v:
                    variant_map[str(k)] = str(v["output_path"])
                elif isinstance(v, dict) and "path" in v:
                    variant_map[str(k)] = str(v["path"])
                elif isinstance(v, str):
                    variant_map[str(k)] = v
        else:
            raise ValueError("Manifest must contain a 'files' or 'variants' mapping.")

    if args.inputs:
        for item in args.inputs:
            if "=" in item and not os.path.exists(item):
                label, path = item.split("=", 1)
                variant_map[label.strip()] = path.strip()
            else:
                stem = Path(item).stem
                variant_map[stem] = item

    if not variant_map:
        raise ValueError("Specify variants via --inputs or --manifest.")

    res = create_blind_listening_set(
        variant_files=variant_map,
        output_dir=args.out,
        seed=args.seed,
        session_id=args.session_id,
        key_output_path=args.key_out,
        overwrite=args.overwrite,
    )
    # Print only the public summary so the terminal output does not spoil the blind test
    public_summary = {k: v for k, v in res.items() if k != "_internal_key"}
    print(json.dumps(public_summary, indent=2))


if __name__ == "__main__":
    main()
