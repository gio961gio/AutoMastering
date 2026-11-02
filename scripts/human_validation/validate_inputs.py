"""Input WAV Validation, SHA-256 Provenance, and Safe File Overwrite Utilities."""

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, Tuple

import numpy as np
import soundfile as sf


def compute_file_sha256(path: str) -> str:
    """Compute SHA-256 hex digest of a file on disk."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found for SHA-256 computation: {path}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def check_overwrite_allowed(path: str, overwrite: bool = False) -> None:
    """Prevent accidental overwrite of existing files unless overwrite=True."""
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing path '{path}'. "
            f"Pass --overwrite (or overwrite=True) if you intend to replace it."
        )


def validate_audio_array(
    audio: np.ndarray,
    allow_mono: bool = True,
) -> Tuple[np.ndarray, bool]:
    """Validate an in-memory audio array and standardize to shape (2, T) float64.

    Args:
        audio: Input numpy array of shape (2, T), (T, 2), (1, T), (T, 1), or (T,).
        allow_mono: Whether to duplicate 1-channel mono to 2-channel stereo.

    Returns:
        Tuple of (stereo_audio_2xT_float64, mono_duplicated_to_stereo).
    """
    if not isinstance(audio, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(audio)}")

    if audio.ndim == 1:
        if audio.size == 0:
            raise ValueError("Audio array is empty (0 samples).")
        if not allow_mono:
            raise ValueError("Mono audio is not allowed without allow_mono=True.")
        if not np.isfinite(audio).all():
            raise ValueError("Audio array contains NaN or Inf values.")
        row = audio.astype(np.float64, copy=False)
        return np.stack([row, row], axis=0), True

    if audio.ndim != 2:
        raise ValueError(f"Invalid audio array dimensions: expected 1D or 2D, got shape {audio.shape}")

    if audio.shape[0] == 0 or audio.shape[1] == 0:
        raise ValueError(f"Audio array is empty: shape {audio.shape}")

    if not np.isfinite(audio).all():
        raise ValueError("Audio array contains NaN or Inf values.")

    # Standardize (C, T) vs (T, C)
    if audio.shape[0] in (1, 2) and (audio.shape[1] > 2 or audio.shape[0] <= audio.shape[1]):
        c_first = audio.astype(np.float64, copy=False)
    elif audio.shape[1] in (1, 2) and audio.shape[0] > 2:
        c_first = audio.T.astype(np.float64, copy=False)
    else:
        raise ValueError(
            f"Unsupported channel count or ambiguous shape {audio.shape}. "
            f"Only mono (1 ch) or stereo (2 ch) are supported."
        )

    if c_first.shape[0] == 1:
        if not allow_mono:
            raise ValueError("Mono audio is not allowed without allow_mono=True.")
        return np.vstack([c_first, c_first]), True

    if c_first.shape[0] != 2:
        raise ValueError(f"Expected 1 or 2 channels, got {c_first.shape[0]} channels.")

    return np.ascontiguousarray(c_first, dtype=np.float64), False


def load_and_validate_wav(
    path: str,
    required_sr: int = 48000,
    allow_mono: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load and strictly validate a WAV file for the 48 kHz mastering contract.

    Never resamples silently. If the sample rate is not `required_sr` (48000 Hz),
    raises a ValueError with an explicit one-time conversion command.

    Args:
        path: Path to the input WAV file.
        required_sr: Required sample rate in Hz (default: 48000).
        allow_mono: Whether to accept mono WAVs by duplicating the channel to stereo.

    Returns:
        Tuple of (stereo_audio_2xT_float64, metadata_dict).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input audio file not found: {path}")

    info = sf.info(path)
    if info.frames <= 0:
        raise ValueError(f"Input audio file is empty (0 frames): {path}")

    if info.samplerate != required_sr:
        raise ValueError(
            f"Unsupported sample rate {info.samplerate} Hz in '{os.path.basename(path)}'. "
            f"The AutoMastering engine strictly requires {required_sr} Hz and refuses silent resampling. "
            f"Convert the file once offline using a high-quality resampler, e.g.:\n"
            f"  ffmpeg -i \"{path}\" -ar {required_sr} -c:a pcm_f32le \"output_{required_sr}hz.wav\""
        )

    if info.channels not in (1, 2):
        raise ValueError(
            f"Unsupported channel count {info.channels} in '{os.path.basename(path)}'. "
            f"Only mono (1 ch) or stereo (2 ch) WAV files are supported."
        )

    if info.channels == 1 and not allow_mono:
        raise ValueError(
            f"Mono input '{os.path.basename(path)}' rejected because allow_mono=False."
        )

    raw_data, sr = sf.read(path, dtype="float64", always_2d=True)
    # raw_data has shape (T, C) from soundfile
    if raw_data.shape[0] <= 0:
        raise ValueError(f"Input audio file has 0 readable frames: {path}")

    if not np.isfinite(raw_data).all():
        raise ValueError(f"Input audio file contains NaN or Inf samples: {path}")

    mono_duplicated = False
    if raw_data.shape[1] == 1:
        stereo_2xT = np.vstack([raw_data[:, 0], raw_data[:, 0]])
        mono_duplicated = True
    else:
        stereo_2xT = np.ascontiguousarray(raw_data.T, dtype=np.float64)

    file_sha256 = compute_file_sha256(path)

    metadata: Dict[str, Any] = {
        "file_name": os.path.basename(path),
        "sha256": file_sha256,
        "sample_rate": int(sr),
        "original_channels": int(info.channels),
        "channels": 2,
        "mono_duplicated_to_stereo": mono_duplicated,
        "num_samples": int(stereo_2xT.shape[1]),
        "duration_sec": float(stereo_2xT.shape[1] / float(sr)),
        "format": str(info.format),
        "subtype": str(info.subtype),
    }
    return stereo_2xT, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate WAV input files for AutoMastering Human Validation.")
    parser.add_argument("inputs", nargs="+", help="Path(s) to WAV files to validate.")
    parser.add_argument("--no-mono", action="store_true", help="Reject mono WAV files.")
    args = parser.parse_args()

    results = []
    for p in args.inputs:
        _, meta = load_and_validate_wav(p, required_sr=48000, allow_mono=not args.no_mono)
        results.append(meta)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
