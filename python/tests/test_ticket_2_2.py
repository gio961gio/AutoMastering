"""Tests for Ticket 2.2: Audio Dataset Pipeline & Synthetic Pair Generator."""

import math
import numpy as np
import pytest
import soundfile as sf
import torch

from ddsp_mastering.data import AudioDataset, SyntheticMasteringPairGenerator
from ddsp_mastering.dsp.chain import DifferentiableMasteringChain


@pytest.fixture
def temp_audio_dir(tmp_path):
    """Creates temporary synthetic audio files for dataset testing."""
    fs = 48000
    duration = 5.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Track 1: Quiet intro (1.5s) followed by loud chorus (3.5s)
    quiet_part = 0.01 * np.sin(2 * np.pi * 440.0 * t[: int(fs * 1.5)])
    loud_part = 0.5 * np.sin(2 * np.pi * 440.0 * t[int(fs * 1.5) :])
    sig1_left = np.concatenate([quiet_part, loud_part])
    sig1_right = sig1_left * 0.9
    file1 = tmp_path / "track_dynamic.wav"
    sf.write(str(file1), np.stack([sig1_left, sig1_right], axis=1), fs)

    # Track 2: 44.1 kHz mono file (testing sample rate conversion & mono-to-stereo)
    fs2 = 44100
    t2 = np.linspace(0, 3.0, int(fs2 * 3.0), endpoint=False)
    sig2 = 0.3 * np.sin(2 * np.pi * 220.0 * t2)
    file2 = tmp_path / "track_mono_44k.wav"
    sf.write(str(file2), sig2, fs2)

    return [file1, file2]


def test_audio_dataset_loading_and_conversion(temp_audio_dir):
    """Test audio loading, mono-to-stereo expansion, and 48 kHz resampling."""
    dataset = AudioDataset(
        audio_files=temp_audio_dir,
        sample_rate=48000,
        chunk_duration_sec=2.0,
    )

    assert len(dataset) == 2

    # Load Track 1 (48 kHz stereo)
    chunk1 = dataset[0]
    assert chunk1.shape == (2, 48000 * 2)
    assert chunk1.dtype == torch.float32

    # Load Track 2 (44.1 kHz mono -> resampled to 48 kHz stereo)
    chunk2 = dataset[1]
    assert chunk2.shape == (2, 48000 * 2)
    assert chunk2.dtype == torch.float32


def test_energy_gate_avoids_quiet_sections(temp_audio_dir):
    """Verify that the energy gate filters out quiet intros in favor of high-energy sections."""
    dataset = AudioDataset(
        audio_files=[temp_audio_dir[0]],  # track with quiet intro
        sample_rate=48000,
        chunk_duration_sec=1.5,
        energy_gate_threshold_db=12.0,
        random_chunk=True,
    )

    # Extract 10 chunks; none should be purely the quiet intro (RMS < -35 dBFS)
    for i in range(10):
        gen = torch.Generator().manual_seed(i + 100)
        raw_audio = dataset._load_audio(dataset.audio_files[0])
        chunk = dataset.extract_high_energy_chunk(raw_audio, generator=gen)

        rms = chunk.pow(2).mean().sqrt().item()
        rms_db = 20.0 * math.log10(rms)
        # Quiet section was around -40 dBFS, loud section around -9 dBFS (before headroom norm)
        # Even after headroom normalization (-18 LUFS), RMS should be around -18 to -15 dBFS
        assert rms_db > -25.0, f"Chunk was too quiet ({rms_db:.2f} dBFS), energy gate failed!"


def test_premastering_headroom_normalization():
    """Verify pre-mastering headroom normalization brings signals to ~ -18 LUFS."""
    dataset = AudioDataset(chunk_duration_sec=2.0, target_headroom_lufs=-18.0)

    fs = 48000
    t = torch.linspace(0, 2.0, fs * 2)
    # Loud audio: -6 dBFS peak
    loud_audio = 0.5 * torch.stack([torch.sin(2 * math.pi * 1000.0 * t), torch.sin(2 * math.pi * 1000.0 * t)])
    norm_loud = dataset.normalize_headroom(loud_audio)

    # Quiet audio: -30 dBFS
    quiet_audio = 0.03 * torch.stack([torch.sin(2 * math.pi * 1000.0 * t), torch.sin(2 * math.pi * 1000.0 * t)])
    norm_quiet = dataset.normalize_headroom(quiet_audio)

    lufs_loud = dataset.loudness_calculator.calculate_lufs(norm_loud.unsqueeze(0)).item()
    lufs_quiet = dataset.loudness_calculator.calculate_lufs(norm_quiet.unsqueeze(0)).item()

    assert abs(lufs_loud - (-18.0)) < 0.6, f"Loud audio headroom off: {lufs_loud:.2f} LUFS"
    assert abs(lufs_quiet - (-18.0)) < 0.6, f"Quiet audio headroom off: {lufs_quiet:.2f} LUFS"


def test_synthetic_pair_generator():
    """Verify synthetic (raw, master, z) generation and self-consistency."""
    generator = SyntheticMasteringPairGenerator(sample_rate=48000.0, use_cpp_fast=False)

    fs = 48000
    t = torch.linspace(0, 1.0, fs)
    raw_mix = 0.1 * torch.stack([torch.sin(2 * math.pi * 300.0 * t), torch.sin(2 * math.pi * 600.0 * t)]).unsqueeze(0)

    raw, target, z = generator.generate_pair(raw_mix)

    assert raw.shape == raw_mix.shape
    assert target.shape == raw_mix.shape
    assert z.shape == (1, 41)

    # All z parameters within realistic range [0.1, 0.9]
    assert torch.all(z >= 0.1) and torch.all(z <= 0.9)

    # Check that master differs from raw mix (processing was applied)
    diff = torch.norm(target - raw).item()
    assert diff > 0.01, "Target master should reflect DSP processing"


def test_generator_reproducibility():
    """Verify that deterministic torch.Generator produces 100% reproducible pairs."""
    gen1 = torch.Generator().manual_seed(42)
    gen2 = torch.Generator().manual_seed(42)

    pair_gen = SyntheticMasteringPairGenerator()

    z1 = pair_gen.sample_realistic_z(batch_size=2, generator=gen1)
    z2 = pair_gen.sample_realistic_z(batch_size=2, generator=gen2)

    assert torch.allclose(z1, z2), "Identical seeds must yield identical parameter vectors"
