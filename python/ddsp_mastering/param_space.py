import torch
import torch.nn as nn
from typing import Dict, List, NamedTuple

NUM_PARAMS = 41

class ParamSpec(NamedTuple):
    name: str
    min_val: float
    max_val: float
    is_log: bool
    unit: str

PARAM_SPECS: List[ParamSpec] = [
    # 0..17: Parametric EQ (6 bande x 3 parametri) - Mastering Grade ranges
    ParamSpec("eq_b0_f0", 20.0, 250.0, True, "Hz"),
    ParamSpec("eq_b0_gain_db", -6.0, 6.0, False, "dB"),
    ParamSpec("eq_b0_q", 0.4, 2.5, True, ""),
    ParamSpec("eq_b1_f0", 100.0, 500.0, True, "Hz"),
    ParamSpec("eq_b1_gain_db", -6.0, 6.0, False, "dB"),
    ParamSpec("eq_b1_q", 0.4, 2.5, True, ""),
    ParamSpec("eq_b2_f0", 400.0, 2000.0, True, "Hz"),
    ParamSpec("eq_b2_gain_db", -6.0, 6.0, False, "dB"),
    ParamSpec("eq_b2_q", 0.4, 2.5, True, ""),
    ParamSpec("eq_b3_f0", 1500.0, 6000.0, True, "Hz"),
    ParamSpec("eq_b3_gain_db", -6.0, 6.0, False, "dB"),
    ParamSpec("eq_b3_q", 0.4, 2.5, True, ""),
    ParamSpec("eq_b4_f0", 4000.0, 12000.0, True, "Hz"),
    ParamSpec("eq_b4_gain_db", -6.0, 6.0, False, "dB"),
    ParamSpec("eq_b4_q", 0.4, 2.5, True, ""),
    ParamSpec("eq_b5_f0", 5000.0, 18000.0, True, "Hz"),
    ParamSpec("eq_b5_gain_db", -6.0, 6.0, False, "dB"),
    ParamSpec("eq_b5_q", 0.4, 2.5, True, ""),

    # 18..19: Crossover LR4 (2 parametri)
    ParamSpec("crossover_f_low", 80.0, 350.0, True, "Hz"),
    ParamSpec("crossover_f_high", 1500.0, 6000.0, True, "Hz"),

    # 20..34: Multiband Comp (3 bande x 5 parametri) - Mastering Grade ranges
    # Low Band (20..24) - preserves sub/kick punch with slower attack (ratio 1.0 = true identity bypass)
    ParamSpec("comp_low_threshold_db", -30.0, 0.0, False, "dB"),
    ParamSpec("comp_low_ratio", 1.0, 4.0, True, ""),
    ParamSpec("comp_low_attack_ms", 10.0, 120.0, True, "ms"),
    ParamSpec("comp_low_release_ms", 20.0, 600.0, True, "ms"),
    ParamSpec("comp_low_makeup_db", -3.0, 6.0, False, "dB"),
    # Mid Band (25..29) - preserves snare snap
    ParamSpec("comp_mid_threshold_db", -30.0, 0.0, False, "dB"),
    ParamSpec("comp_mid_ratio", 1.0, 4.0, True, ""),
    ParamSpec("comp_mid_attack_ms", 5.0, 80.0, True, "ms"),
    ParamSpec("comp_mid_release_ms", 20.0, 600.0, True, "ms"),
    ParamSpec("comp_mid_makeup_db", -3.0, 6.0, False, "dB"),
    # High Band (30..34) - smooth vocal/cymbal control
    ParamSpec("comp_high_threshold_db", -30.0, 0.0, False, "dB"),
    ParamSpec("comp_high_ratio", 1.0, 4.0, True, ""),
    ParamSpec("comp_high_attack_ms", 2.0, 50.0, True, "ms"),
    ParamSpec("comp_high_release_ms", 20.0, 600.0, True, "ms"),
    ParamSpec("comp_high_makeup_db", -3.0, 6.0, False, "dB"),

    # 35..37: Stereo Imager (3 parametri) - natural width bounds
    ParamSpec("imager_width_low", 0.4, 1.1, False, ""),
    ParamSpec("imager_width_mid", 0.7, 1.4, False, ""),
    ParamSpec("imager_width_high", 0.7, 1.4, False, ""),

    # 38..39: Harmonic Saturator (2 parametri) - musical tape saturation bounds
    ParamSpec("saturator_drive_db", 0.0, 6.0, False, "dB"),
    ParamSpec("saturator_mix", 0.0, 1.0, False, ""),

    # 40: Lookahead Limiter (1 parametro)
    ParamSpec("limiter_push_gain_db", 0.0, 8.0, False, "dB"),
]

# Tensors costanti per calcolo vettorizzato su GPU/CPU
_MIN_VALS = torch.tensor([p.min_val for p in PARAM_SPECS], dtype=torch.float64)
_MAX_VALS = torch.tensor([p.max_val for p in PARAM_SPECS], dtype=torch.float64)
_IS_LOG = torch.tensor([p.is_log for p in PARAM_SPECS], dtype=torch.bool)

# Valori di guardia per evitare valutazioni di log su parametri non-log (evita gradienti NaN)
_SAFE_LOG_MIN = torch.where(_IS_LOG, _MIN_VALS, torch.ones_like(_MIN_VALS))
_SAFE_LOG_MAX = torch.where(_IS_LOG, _MAX_VALS, torch.ones_like(_MAX_VALS))
_SAFE_LOG_RATIO = torch.log(_SAFE_LOG_MAX / _SAFE_LOG_MIN)


def map_z_to_theta(z: torch.Tensor) -> torch.Tensor:
    """
    Mappa il vettore normalizzato z in (0, 1) nello spazio dei parametri fisici theta.
    Supporta tensori di forma [..., 41]. Preserva i gradienti (differenziabile).
    """
    assert z.shape[-1] == NUM_PARAMS, f"Expected last dim {NUM_PARAMS}, got {z.shape[-1]}"
    
    device = z.device
    dtype = z.dtype
    min_vals = _MIN_VALS.to(device=device, dtype=dtype)
    max_vals = _MAX_VALS.to(device=device, dtype=dtype)
    is_log = _IS_LOG.to(device=device)

    safe_log_min = _SAFE_LOG_MIN.to(device=device, dtype=dtype)
    safe_log_ratio = _SAFE_LOG_RATIO.to(device=device, dtype=dtype)

    z_clamped = torch.clamp(z, 0.0, 1.0)

    # Ramo lineare
    theta_lin = min_vals + z_clamped * (max_vals - min_vals)
    # Ramo logaritmico protetto contro NaN
    theta_log = safe_log_min * torch.exp(z_clamped * safe_log_ratio)

    theta = torch.where(is_log, theta_log, theta_lin)
    return theta


def map_theta_to_z(theta: torch.Tensor) -> torch.Tensor:
    """
    Mappatura inversa dai parametri fisici theta a z normalizzato in [0, 1].
    """
    assert theta.shape[-1] == NUM_PARAMS, f"Expected last dim {NUM_PARAMS}, got {theta.shape[-1]}"
    
    device = theta.device
    dtype = theta.dtype
    min_vals = _MIN_VALS.to(device=device, dtype=dtype)
    max_vals = _MAX_VALS.to(device=device, dtype=dtype)
    is_log = _IS_LOG.to(device=device)

    safe_log_min = _SAFE_LOG_MIN.to(device=device, dtype=dtype)
    safe_log_ratio = _SAFE_LOG_RATIO.to(device=device, dtype=dtype)

    theta_clamped = torch.clamp(theta, min=min_vals, max=max_vals)

    z_lin = (theta_clamped - min_vals) / (max_vals - min_vals)
    
    # Ramo logaritmico protetto
    theta_safe = torch.where(is_log, theta_clamped, safe_log_min)
    z_log = torch.log(theta_safe / safe_log_min) / safe_log_ratio

    z = torch.where(is_log, z_log, z_lin)
    return torch.clamp(z, 0.0, 1.0)


# ---------------------------------------------------------------------------
# PRESET FISICO NEUTRO (Conservative Neutral Starting Preset)
# ---------------------------------------------------------------------------
# Valori fisici iniziali conservativi per l'ottimizzatore (separati dal True Bypass):
# - EQ flat (0 dB gain, Q=0.707 Butterworth)
# - Crossover standard 160 Hz e 3200 Hz
# - Compressore ad assetto trasparente (Ratio 1.0:1 identità, makeup 0 dB)
# - Stereo imager neutro (width = 1.0)
# - Saturatore spento (drive = 0 dB, mix = 0.0 dry)
# - Limiter neutro (push gain = 0 dB)
THETA_NEUTRAL: torch.Tensor = torch.tensor([
    # EQ (f0, gain, q) x 6
    60.0, 0.0, 0.707,
    250.0, 0.0, 0.707,
    1000.0, 0.0, 0.707,
    3000.0, 0.0, 0.707,
    8000.0, 0.0, 0.707,
    12000.0, 0.0, 0.707,
    # Crossover (low, high)
    160.0, 3200.0,
    # Multiband comp: thresh, ratio, attack, release, makeup (ratio 1.0 = true identity)
    -12.0, 1.0, 40.0, 120.0, 0.0,
    -12.0, 1.0, 25.0, 100.0, 0.0,
    -12.0, 1.0, 15.0, 80.0, 0.0,
    # Imager: low, mid, high
    1.0, 1.0, 1.0,
    # Saturator: drive, mix
    0.0, 0.0,
    # Limiter push
    0.0
], dtype=torch.float64)

# Vettore normalizzato neutro z_neutral in (0, 1)
Z_NEUTRAL: torch.Tensor = map_theta_to_z(THETA_NEUTRAL).to(dtype=torch.float32)


def unpack_theta_dict(theta: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Scompatta il tensore theta [..., 41] in un dizionario strutturato per i singoli moduli.
    """
    return {
        # EQ: [..., 6]
        "eq_f0": theta[..., 0:18:3],
        "eq_gain_db": theta[..., 1:18:3],
        "eq_q": theta[..., 2:18:3],

        # Crossover: [..., 1]
        "crossover_f_low": theta[..., 18],
        "crossover_f_high": theta[..., 19],

        # Compressore Multibanda: [..., 3]
        "comp_threshold_db": torch.stack([theta[..., 20], theta[..., 25], theta[..., 30]], dim=-1),
        "comp_ratio": torch.stack([theta[..., 21], theta[..., 26], theta[..., 31]], dim=-1),
        "comp_attack_ms": torch.stack([theta[..., 22], theta[..., 27], theta[..., 32]], dim=-1),
        "comp_release_ms": torch.stack([theta[..., 23], theta[..., 28], theta[..., 33]], dim=-1),
        "comp_makeup_db": torch.stack([theta[..., 24], theta[..., 29], theta[..., 34]], dim=-1),

        # Imager: [..., 3]
        "imager_width": theta[..., 35:38],

        # Saturatore: [...]
        "saturator_drive_db": theta[..., 38],
        "saturator_mix": theta[..., 39],

        # Limiter: [...]
        "limiter_push_gain_db": theta[..., 40],
    }
