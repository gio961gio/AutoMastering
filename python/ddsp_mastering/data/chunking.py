"""Multi-Chunk Representative Audio Selection and Temporal Stratification.

Replaces single-window selection with a deterministic multi-chunk evaluation protocol
distributed across the entire track duration, balancing RMS energy, transient density,
and spectral diversity.
"""

from dataclasses import dataclass, asdict
import math
from typing import Dict, List, Optional, Tuple, Union
import warnings
import numpy as np
from scipy import signal


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

        # Track selection diagnostic metadata
        self.last_requested_chunks: int = self.num_chunks
        self.last_selected_chunks: int = 0
        self.selection_warnings: List[str] = []

    def _compute_chunk_features(
        self, audio_chunk: np.ndarray
    ) -> Tuple[float, float, float]:
        """Compute RMS (dBFS), Crest Factor (dB), and Spectral Centroid (Hz) for a chunk.

        The spectral centroid is computed across the ENTIRE chunk duration via multi-frame
        STFT (2048-sample Hann window, 512 hop), averaging the magnitude spectrum over all
        frames to avoid localized 40ms bias.
        """
        # Mean square across channels
        ms = np.mean(audio_chunk ** 2)
        rms = math.sqrt(ms) if ms > 1e-12 else 1e-6
        rms_db = 20.0 * math.log10(rms)

        peak = np.max(np.abs(audio_chunk))
        peak = max(peak, 1e-6)
        crest_factor_db = 20.0 * math.log10(peak / rms)

        # Spectral Centroid via full-chunk multi-frame STFT
        mono = np.mean(audio_chunk, axis=0).astype(np.float64)
        n_fft = min(2048, len(mono))
        hop = min(512, max(1, n_fft // 4))

        if len(mono) >= n_fft and n_fft > 64:
            f, _, Zxx = signal.stft(
                mono,
                fs=self.sample_rate,
                nperseg=n_fft,
                noverlap=n_fft - hop,
                window="hann",
                boundary=None,
                padded=False,
            )
            # Average magnitude spectrum across all time frames of chunk
            mag_spec = np.mean(np.abs(Zxx), axis=-1)
            spec_sum = np.sum(mag_spec)
            if spec_sum > 1e-8:
                centroid_hz = float(np.sum(f * mag_spec) / spec_sum)
            else:
                centroid_hz = 1000.0
        else:
            window = np.hanning(n_fft)
            spec = np.abs(np.fft.rfft(mono[:n_fft] * window))
            freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.sample_rate)
            spec_sum = np.sum(spec)
            centroid_hz = float(np.sum(freqs * spec) / spec_sum) if spec_sum > 1e-8 else 1000.0

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

        self.last_requested_chunks = self.num_chunks
        self.selection_warnings = []

        # Edge case: audio is shorter than requested chunk length
        if T <= self.chunk_len:
            warn_msg = (
                f"Audio duration ({T / self.sample_rate:.2f}s) is shorter than or equal to "
                f"requested chunk duration ({self.chunk_duration_sec:.2f}s). Returning single chunk."
            )
            warnings.warn(warn_msg)
            self.selection_warnings.append(warn_msg)
            rms_db, cf_db, cent = self._compute_chunk_features(audio)
            self.last_selected_chunks = 1
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

        # Deterministic RNG initialized with seed for reproducible pool tie-breaking
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
            warn_msg = (
                f"Audio energy across all candidate windows is below threshold ({self.min_rms_db} dBFS). "
                f"Falling back to highest energy available chunk."
            )
            warnings.warn(warn_msg)
            self.selection_warnings.append(warn_msg)
            mono_sq = np.mean(audio ** 2, axis=0)
            cs = np.pad(np.cumsum(mono_sq), (1, 0))
            best_idx = int(starts[np.argmax(cs[starts + self.chunk_len] - cs[starts])])
            rms_db, cf_db, cent = self._compute_chunk_features(audio[:, best_idx : best_idx + self.chunk_len])
            self.last_selected_chunks = 1
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

        used_starts = set()
        selected_candidates = []
        for zone_idx in range(num_target):
            zone_start = zone_idx * zone_duration
            zone_end = (zone_idx + 1) * zone_duration

            # Find candidates within this temporal zone not already selected
            zone_pool = [
                c for c in candidates
                if zone_start <= c["start"] < zone_end and c["start"] not in used_starts
            ]

            if not zone_pool:
                # Fallback to closest available unused candidate
                unused_candidates = [c for c in candidates if c["start"] not in used_starts]
                if not unused_candidates:
                    break
                zone_center = 0.5 * (zone_start + zone_end)
                zone_pool = sorted(unused_candidates, key=lambda c: abs(c["start"] - zone_center))[:3]

            # In alternate zones, prioritize either peak energy, high transient crest factor, or spectral brightness
            # Use deterministic seed-based jitter for tie-breaking candidates with near-identical scores
            if zone_idx % 3 == 0:
                # Prioritize high RMS energy (chorus/drop)
                chosen = max(zone_pool, key=lambda c: c["rms_db"] + rng.uniform(-1e-6, 1e-6))
            elif zone_idx % 3 == 1:
                # Prioritize high crest factor (dynamic/transient sections)
                chosen = max(zone_pool, key=lambda c: c["cf_db"] + rng.uniform(-1e-6, 1e-6))
            else:
                # Prioritize spectral centroid diversity
                mean_cent = np.mean([c["cent"] for c in candidates])
                chosen = max(zone_pool, key=lambda c: abs(c["cent"] - mean_cent) + rng.uniform(-1e-6, 1e-6))

            selected_candidates.append(chosen)
            used_starts.add(chosen["start"])

        # 3. Post-process: ensure strict non-overlap and minimum separation (min_separation_samples)
        selected_candidates.sort(key=lambda c: c["start"])
        filtered = []
        for c in selected_candidates:
            if not filtered:
                filtered.append(c)
            else:
                prev_start = filtered[-1]["start"]
                required_start = prev_start + self.chunk_len + self.min_separation_samples
                if c["start"] >= required_start:
                    filtered.append(c)
                else:
                    # Search among available pre-filtered candidates (which already satisfy rms_db >= min_rms_db)
                    already_selected = {f["start"] for f in filtered}
                    available = [
                        cand for cand in candidates
                        if cand["start"] >= required_start and cand["start"] not in already_selected
                    ]
                    if available:
                        # Pick the earliest candidate satisfying required_start to preserve temporal ordering
                        best_cand = min(available, key=lambda cand: cand["start"])
                        filtered.append(best_cand)
                    else:
                        # Attempt displacement to next available valid position
                        new_start = required_start
                        if new_start + self.chunk_len <= T:
                            new_chunk = audio[:, new_start : new_start + self.chunk_len]
                            rms_db, cf_db, cent = self._compute_chunk_features(new_chunk)
                            # B3 / F-04: Strict gate verification after displacement!
                            if rms_db >= self.min_rms_db:
                                filtered.append({
                                    "start": new_start,
                                    "rms_db": rms_db,
                                    "cf_db": cf_db,
                                    "cent": cent,
                                })

        if len(filtered) < self.num_chunks:
            warn_msg = (
                f"Audio duration or energy insufficient to accommodate "
                f"{self.num_chunks} chunks with min_separation={self.min_separation_sec:.1f}s and min_rms_db={self.min_rms_db} dBFS. "
                f"Selected {len(filtered)} chunks."
            )
            warnings.warn(warn_msg)
            self.selection_warnings.append(warn_msg)

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

        self.last_selected_chunks = len(specs)
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
