"""Audio Dataset Pipeline with Energy-Gated Chunking & Headroom Normalization.

Handles stereo audio loading, 48 kHz sample-rate validation, high-energy chunk
extraction (filtering out quiet intros/outros), and pre-mastering headroom normalization.
"""

import math
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import soundfile as sf

from ddsp_mastering.loss.loudness import DifferentiableLoudnessLoss


class AudioDataset(Dataset):
    """Dataset for training and evaluating AutoMastering DDSP models.

    Extracts high-energy audio chunks (2.0s to 4.0s), filters out silent/quiet sections,
    and applies pre-mastering headroom normalization (target -18 to -16 LUFS or -6 dBFS peak).
    """

    def __init__(
        self,
        audio_files: Optional[List[Union[str, Path]]] = None,
        sample_rate: int = 48000,
        chunk_duration_sec: float = 2.0,
        target_headroom_lufs: float = -18.0,
        energy_gate_threshold_db: float = 12.0,
        random_chunk: bool = True,
        preload_memory: bool = False,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.chunk_duration_sec = chunk_duration_sec
        self.chunk_samples = int(chunk_duration_sec * sample_rate)
        self.target_headroom_lufs = target_headroom_lufs
        self.energy_gate_threshold_db = energy_gate_threshold_db
        self.random_chunk = random_chunk
        self.preload_memory = preload_memory

        self.audio_files = [Path(f) for f in (audio_files or [])]
        self.loudness_calculator = DifferentiableLoudnessLoss(sample_rate=float(sample_rate))

        # Memory cache if enabled
        self._cached_audio: List[torch.Tensor] = []
        if preload_memory and self.audio_files:
            for p in self.audio_files:
                self._cached_audio.append(self._load_audio(p))

    def _load_audio(self, path: Path) -> torch.Tensor:
        """Load audio file as float32 stereo tensor at self.sample_rate."""
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        tensor = torch.from_numpy(data.T)  # (channels, T)

        # Convert mono to stereo if necessary
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(2, 1)
        elif tensor.shape[0] > 2:
            tensor = tensor[:2, :]

        # Verify or resample to self.sample_rate
        if sr != self.sample_rate:
            import torchaudio.transforms as T
            resampler = T.Resample(sr, self.sample_rate)
            tensor = resampler(tensor)

        return tensor

    @staticmethod
    def compute_energy_profile(audio: torch.Tensor, frame_size: int = 2048, hop_size: int = 512) -> torch.Tensor:
        """Compute RMS energy per frame in dBFS."""
        # Mean square per frame across stereo channels
        power = audio.pow(2).mean(dim=0, keepdim=True)  # (1, T)
        # Unfold into sliding frames
        if power.shape[-1] < frame_size:
            power = F.pad(power, (0, frame_size - power.shape[-1]))

        frames = power.unfold(dimension=-1, size=frame_size, step=hop_size)  # (1, num_frames, frame_size)
        rms = frames.mean(dim=-1).sqrt().clamp(min=1e-8)  # (1, num_frames)
        rms_db = 20.0 * torch.log10(rms)
        return rms_db.squeeze(0)

    def extract_high_energy_chunk(
        self, audio: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Extract a chunk meeting the energy gate threshold."""
        T = audio.shape[-1]
        L = self.chunk_samples

        if T <= L:
            # Pad with zeros if shorter than chunk_samples
            chunk = F.pad(audio, (0, L - T))
            return self.normalize_headroom(chunk)

        # Compute average track energy
        track_rms = audio.pow(2).mean().sqrt().clamp(min=1e-8)
        track_rms_db = 20.0 * math.log10(track_rms.item())
        min_acceptable_rms_db = track_rms_db - self.energy_gate_threshold_db

        max_start = T - L
        num_candidates = min(10, max(1, max_start // (self.sample_rate // 2)))

        best_chunk = None
        best_rms_db = -float("inf")

        # Try up to 5 random chunks, select one that satisfies energy gate
        for _ in range(5):
            if self.random_chunk:
                if generator is not None:
                    start_idx = torch.randint(0, max_start + 1, (1,), generator=generator).item()
                else:
                    start_idx = torch.randint(0, max_start + 1, (1,)).item()
            else:
                start_idx = max_start // 2  # Center slice

            cand_chunk = audio[:, start_idx : start_idx + L]
            cand_rms = cand_chunk.pow(2).mean().sqrt().clamp(min=1e-8)
            cand_rms_db = 20.0 * math.log10(cand_rms.item())

            if cand_rms_db > best_rms_db:
                best_rms_db = cand_rms_db
                best_chunk = cand_chunk

            if cand_rms_db >= min_acceptable_rms_db:
                # Meets energy gate criteria!
                return self.normalize_headroom(cand_chunk)

        # Fallback to the highest energy chunk found
        return self.normalize_headroom(best_chunk)

    def normalize_headroom(self, audio: torch.Tensor) -> torch.Tensor:
        """Normalize audio chunk to target pre-mastering headroom (-18 LUFS)."""
        with torch.no_grad():
            batch_audio = audio.unsqueeze(0)  # (1, 2, T)
            curr_lufs = self.loudness_calculator.calculate_lufs(batch_audio).item()

            if not math.isfinite(curr_lufs) or curr_lufs < -70.0:
                # Inaudible / silent chunk
                return audio

            gain_db = self.target_headroom_lufs - curr_lufs
            linear_gain = 10.0 ** (gain_db / 20.0)

            normalized = audio * linear_gain

            # Safety check: avoid clipping during pre-mastering headroom stage (limit to -1 dBFS)
            peak = torch.max(torch.abs(normalized)).item()
            if peak > 0.89125:  # -1.0 dBFS
                safety_factor = 0.89125 / (peak + 1e-8)
                normalized = normalized * safety_factor

            return normalized

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.preload_memory and self._cached_audio:
            raw_audio = self._cached_audio[idx]
        else:
            raw_audio = self._load_audio(self.audio_files[idx])

        return self.extract_high_energy_chunk(raw_audio)
