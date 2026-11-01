import pytest
import numpy as np
import torch
import scipy.signal as signal

SAMPLE_RATE = 48000
NUM_SAMPLES = 96000 # 2.0 secondi a 48 kHz

@pytest.fixture(scope="session")
def fs():
    return SAMPLE_RATE

@pytest.fixture(scope="session")
def sig_impulse():
    """
    Impulso unitario per misurare latenza di fase e risposta all'impulso.
    Left: delta[n - 100]
    Right: 0.5 * delta[n - 100]
    Shape: [2, 96000] float32
    """
    x = np.zeros((2, NUM_SAMPLES), dtype=np.float32)
    x[0, 100] = 1.0
    x[1, 100] = 0.5
    return torch.from_numpy(x)

@pytest.fixture(scope="session")
def sig_sweep():
    """
    Log chirp sweep da 20 Hz a 20 kHz a -3 dBFS (amp = 10^(-3/20) ~ 0.70794578)
    con sfasamento di 45 gradi (pi/4) tra Left e Right.
    Shape: [2, 96000] float32
    """
    t = np.linspace(0.0, 2.0, NUM_SAMPLES, endpoint=False, dtype=np.float64)
    amp = 10.0 ** (-3.0 / 20.0)
    # Log chirp
    f0, f1 = 20.0, 20000.0
    # Phase for exponential sweep: 2 * pi * f0 * ( (f1/f0)^(t/T) - 1 ) / ln(f1/f0)
    phi = 2.0 * np.pi * f0 * ((f1 / f0) ** (t / 2.0) - 1.0) / np.log(f1 / f0)
    
    left = amp * np.sin(phi)
    right = amp * np.sin(phi + np.pi / 4.0)
    
    x = np.stack([left, right], axis=0).astype(np.float32)
    return torch.from_numpy(x)

@pytest.fixture(scope="session")
def sig_burst():
    """
    Seno a 1 kHz modulato a onda quadra ogni 250 ms tra -40 dBFS e +2 dBFS.
    Testa specificamente la risposta dinamica, i detector e il lookahead limiter.
    Shape: [2, 96000] float32
    """
    t = np.linspace(0.0, 2.0, NUM_SAMPLES, endpoint=False, dtype=np.float64)
    carrier = np.sin(2.0 * np.pi * 1000.0 * t)
    
    # Inviluppo a onda quadra (periodo 500 ms: 250 ms high, 250 ms low)
    amp_low = 10.0 ** (-40.0 / 20.0) # 0.01
    amp_high = 10.0 ** (+2.0 / 20.0)  # ~1.2589 (+2 dBFS)
    
    # square wave in [0, 1]
    sq = (signal.square(2.0 * np.pi * 2.0 * t) > 0.0).astype(np.float64)
    env = amp_low + sq * (amp_high - amp_low)
    
    left = (carrier * env).astype(np.float32)
    # Sfasiamo leggermente il canale destro per stereo
    carrier_r = np.sin(2.0 * np.pi * 1000.0 * t + 0.2)
    right = (carrier_r * env).astype(np.float32)
    
    x = np.stack([left, right], axis=0)
    return torch.from_numpy(x)

@pytest.fixture(scope="session")
def sig_pink_music():
    """
    Rumore rosa decorrelato stereo con inviluppo percussivo (Crest Factor ~14 dB).
    Generato deterministico con MT19937 seed 42.
    Shape: [2, 96000] float32
    """
    rng = np.random.default_rng(42)
    
    # Generazione rumore rosa tramite filtro IIR Paul Kellet
    def make_pink(n_samples):
        white = rng.standard_normal(n_samples)
        # Filtro per rumore rosa
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1.0, -2.494956002, 2.017265875, -0.522189400]
        return signal.lfilter(b, a, white)
    
    pink_l = make_pink(NUM_SAMPLES)
    pink_r = make_pink(NUM_SAMPLES)
    
    # Inviluppo percussivo ritmico (battiti ogni 0.5 s, decadimento esponenziale)
    t = np.linspace(0.0, 2.0, NUM_SAMPLES, endpoint=False)
    t_mod = np.mod(t, 0.5)
    env = np.exp(-t_mod / 0.08) + 0.15
    
    stereo_l = pink_l * env
    stereo_r = pink_r * env
    
    # Normalizza RMS a -18 dBFS
    rms_l = np.sqrt(np.mean(stereo_l ** 2))
    target_rms = 10.0 ** (-18.0 / 20.0)
    stereo_l = stereo_l * (target_rms / (rms_l + 1e-12))
    stereo_r = stereo_r * (target_rms / (rms_l + 1e-12))
    
    x = np.stack([stereo_l, stereo_r], axis=0).astype(np.float32)
    return torch.from_numpy(x)
