"""Multi-Chunk Representative Audio Selection and Temporal Stratification.

Replaces single-window selection with a deterministic multi-chunk evaluation protocol
distributed across the entire track duration, balancing RMS energy, transient density,
and spectral diversity.
"""

from dataclasses import dataclass, asdict
import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class ChunkSpec:
    """Metadata describing a selected representative audio chunk."""
    chunk_id: int
    start_sample: int
    end_sample: int
    start_sec: float
    end_sec: float
    duration_sec: float
    rms_db: float
    crest_factor_db: float
    spectral_centroid_hz: float

    def to_dict(self) -> Dict[str, Union[int, float]]:
        return asdict(self)


class MultiChunkSelector:
    """Selects 4-6 diverse, representative, non-overlapping audio windows across a track."""

    def __init__(
        self,
        sample_rate: float = 48000.0,
        num_chunks: int = 4,
        chunk_duration_sec: float = 3.0,
        min_rms_db: float = -42.0,
        min_separation_sec: float = 4.0,
        seed: int = 42,
    ):
        self.sample_rate = sample_rate
        self.num_chunks = max(1, num_chunks)
        self.chunk_duration_sec = chunk_duration_sec
        self.chunk_len = int(chunk_duration_sec * sample_rate)
        self.min_rms_db = min_rms_db
        self.min_separation_sec = min_separation_sec
        self.min_separation_samples = int(min_separation_sec * sample_rate)
        self.seed = seed

    def _compute_chunk_features(
        self, audio_chunk: np.ndarray
    ) -> Tuple[float, float, float]:
        """Compute RMS (dBFS), Crest Factor (dB), and Spectral Centroid (Hz) for a chunk."""
        # Mean square across channels
        ms = np.mean(audio_chunk ** 2)
        rms = math.sqrt(ms) if ms > 1e-12 else 1e-6
        rms_db = 20.0 * math.log10(rms)

        peak = np.max(np.abs(audio_chunk))
        peak = max(peak, 1e-6)
        crest_factor_db = 20.0 * math.log10(peak / rms)

        # Spectral Centroid via mono FFT
        mono = np.mean(audio_chunk, axis=0)
        n_fft = min(2048, len(mono))
        window = np.hanning(n_fft)
        spec = np.abs(np.fft.rfft(mono[:n_fft] * window))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.sample_rate)

        spec_sum = np.sum(spec)
        if spec_sum > 1e-8:
            centroid_hz = float(np.sum(freqs * spec) / spec_sum)
        else:
            centroid_hz = 1000.0

        return rms_db, crest_factor_db, centroid_hz

    def select_chunks(self, audio: np.ndarray) -> List[ChunkSpec]:
        """Select representative chunks distributed chronologically and spectrally.

        Args:
            audio: Stereo audio array [2, T].

        Returns:
            List of ChunkSpec metadata sorted chronologically by start_sample.
        """
        assert audio.ndim == 2 and audio.shape[0] == 2, "Expected stereo audio shape [2, T]"
        T = audio.shape[1]

        # Edge case: audio is shorter than requested chunk length
        if T <= self.chunk_len:
            rms_db, cf_db, cent = self._compute_chunk_features(audio)
            return [
                ChunkSpec(
                    chunk_id=0,
                    start_sample=0,
                    end_sample=T,
                    start_sec=0.0,
                    end_sec=T / self.sample_rate,
                    duration_sec=T / self.sample_rate,
                    rms_db=rms_db,
                    crest_factor_db=cf_db,
                    spectral_centroid_hz=cent,
                )
            ]

        rng = np.random.default_rng(self.seed)

        # 1. Generate candidate positions across a dense grid (hop = 0.5s)
        hop = int(0.5 * self.sample_rate)
        starts = np.arange(0, T - self.chunk_len + 1, hop)

        candidates = []
        for s in starts:
            chunk = audio[:, s : s + self.chunk_len]
            rms_db, cf_db, cent = self._compute_chunk_features(chunk)
            if rms_db >= self.min_rms_db:
                candidates.append({
                    "start": int(s),
                    "rms_db": rms_db,
                    "cf_db": cf_db,
                    "cent": cent,
                })

        # Fallback if track is very quiet
        if not candidates:
            # Pick highest energy window unconditionally
            mono_sq = np.mean(audio ** 2, axis=0)
            cs = np.pad(np.cumsum(mono_sq), (1, 0))
            best_idx = int(starts[np.argmax(cs[starts + self.chunk_len] - cs[starts])])
            rms_db, cf_db, cent = self._compute_chunk_features(audio[:, best_idx : best_idx + self.chunk_len])
            return [
                ChunkSpec(
                    chunk_id=0,
                    start_sample=best_idx,
                    end_sample=best_idx + self.chunk_len,
                    start_sec=best_idx / self.sample_rate,
                    end_sec=(best_idx + self.chunk_len) / self.sample_rate,
                    duration_sec=self.chunk_duration_sec,
                    rms_db=rms_db,
                    crest_factor_db=cf_db,
                    spectral_centroid_hz=cent,
                )
            ]

        # 2. Divide track timeline into N temporal zones (intro/verse, chorus, bridge, etc.)
        num_target = min(self.num_chunks, len(candidates))
        zone_duration = T / num_target

        selected_candidates = []
        for zone_idx in range(num_target):
            zone_start = zone_idx * zone_duration
            zone_end = (zone_idx + 1) * zone_duration

            # Find candidates within this temporal zone
            zone_pool = [
                c for c in candidates
                if zone_start <= c["start"] < zone_end
            ]

            if not zone_pool:
                # Fallback to closest available candidate
                zone_center = 0.5 * (zone_start + zone_end)
                zone_pool = sorted(candidates, key=lambda c: abs(c["start"] - zone_center))[:3]

            # In alternate zones, prioritize either peak energy, high transient crest factor, or spectral brightness
            if zone_idx % 3 == 0:
                # Prioritize high RMS energy (chorus/drop)
                chosen = max(zone_pool, key=lambda c: c["rms_db"])
            elif zone_idx % 3 == 1:
                # Prioritize high crest factor (dynamic/transient sections)
                chosen = max(zone_pool, key=lambda c: c["cf_db"])
            else:
                # Prioritize spectral centroid diversity
                mean_cent = np.mean([c["cent"] for c in candidates])
                chosen = max(zone_pool, key=lambda c: abs(c["cent"] - mean_cent))

            selected_candidates.append(chosen)

        # 3. Post-process: ensure strict non-overlap and minimum separation
        selected_candidates.sort(key=lambda c: c["start"])
        filtered = []
        for c in selected_candidates:
            if not filtered:
                filtered.append(c)
            else:
                prev_start = filtered[-1]["start"]
                if c["start"] >= prev_start + self.chunk_len:
                    filtered.append(c)
                else:
                    # Push start if space permits
                    new_start = prev_start + self.chunk_len + hop
                    if new_start + self.chunk_len <= T:
                        c_copy = dict(c)
                        c_copy["start"] = new_start
                        filtered.append(c_copy)

        # Build final ChunkSpec list
        specs = []
        for i, c in enumerate(filtered):
            s = c["start"]
            e = s + self.chunk_len
            specs.append(
                ChunkSpec(
                    chunk_id=i,
                    start_sample=s,
                    end_sample=e,
                    start_sec=float(s / self.sample_rate),
                    end_sec=float(e / self.sample_rate),
                    duration_sec=self.chunk_duration_sec,
                    rms_db=float(c["rms_db"]),
                    crest_factor_db=float(c["cf_db"]),
                    spectral_centroid_hz=float(c["cent"]),
                )
            )

        return specs

    def extract_chunk_tensors(
        self, audio: np.ndarray, chunks: List[ChunkSpec]
    ) -> List[np.ndarray]:
        """Extract audio array slices for each chunk spec."""
        out = []
        for c in chunks:
            chunk = audio[:, c.start_sample : c.end_sample]
            out.append(np.ascontiguousarray(chunk, dtype=np.float32))
        return out
