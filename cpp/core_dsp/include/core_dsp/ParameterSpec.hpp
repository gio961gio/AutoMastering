#pragma once

#include <array>
#include <cmath>
#include <string_view>
#include <algorithm>

namespace core_dsp {

inline constexpr size_t NUM_PARAMS = 41;

enum class ScaleType {
    Linear,
    Exponential // Log-uniforme: theta = min * (max / min)^z
};

struct ParamDescriptor {
    std::string_view name;
    double min_val;
    double max_val;
    ScaleType scale;
    std::string_view unit;

    [[nodiscard]] constexpr double map_z(double z) const noexcept {
        // Clamp z in [0.0, 1.0] per sicurezza numerica
        z = std::clamp(z, 0.0, 1.0);
        if (scale == ScaleType::Linear) {
            return min_val + z * (max_val - min_val);
        } else {
            return min_val * std::pow(max_val / min_val, z);
        }
    }

    [[nodiscard]] constexpr double unmap_theta(double theta) const noexcept {
        theta = std::clamp(theta, min_val, max_val);
        if (scale == ScaleType::Linear) {
            return (theta - min_val) / (max_val - min_val);
        } else {
            return std::log(theta / min_val) / std::log(max_val / min_val);
        }
    }
};

// Tabella canonica dei 41 parametri
inline constexpr std::array<ParamDescriptor, NUM_PARAMS> PARAM_SPECS = {{
    // 0..17: Parametric EQ (6 bande x 3 parametri)
    // Banda 0: Low-Shelf
    {"eq_b0_f0",      20.0,    250.0,   ScaleType::Exponential, "Hz"},
    {"eq_b0_gain_db", -12.0,   +12.0,   ScaleType::Linear,      "dB"},
    {"eq_b0_q",       0.5,     5.0,     ScaleType::Exponential, ""},
    // Banda 1: Peaking Low-Mid
    {"eq_b1_f0",      100.0,   500.0,   ScaleType::Exponential, "Hz"},
    {"eq_b1_gain_db", -12.0,   +12.0,   ScaleType::Linear,      "dB"},
    {"eq_b1_q",       0.5,     5.0,     ScaleType::Exponential, ""},
    // Banda 2: Peaking Mid
    {"eq_b2_f0",      400.0,   2000.0,  ScaleType::Exponential, "Hz"},
    {"eq_b2_gain_db", -12.0,   +12.0,   ScaleType::Linear,      "dB"},
    {"eq_b2_q",       0.5,     5.0,     ScaleType::Exponential, ""},
    // Banda 3: Peaking High-Mid
    {"eq_b3_f0",      1500.0,  6000.0,  ScaleType::Exponential, "Hz"},
    {"eq_b3_gain_db", -12.0,   +12.0,   ScaleType::Linear,      "dB"},
    {"eq_b3_q",       0.5,     5.0,     ScaleType::Exponential, ""},
    // Banda 4: Peaking Presence
    {"eq_b4_f0",      4000.0,  12000.0, ScaleType::Exponential, "Hz"},
    {"eq_b4_gain_db", -12.0,   +12.0,   ScaleType::Linear,      "dB"},
    {"eq_b4_q",       0.5,     5.0,     ScaleType::Exponential, ""},
    // Banda 5: High-Shelf
    {"eq_b5_f0",      5000.0,  18000.0, ScaleType::Exponential, "Hz"},
    {"eq_b5_gain_db", -12.0,   +12.0,   ScaleType::Linear,      "dB"},
    {"eq_b5_q",       0.5,     5.0,     ScaleType::Exponential, ""},

    // 18..19: Crossover LR4 (2 parametri)
    {"crossover_f_low",  80.0,   350.0,  ScaleType::Exponential, "Hz"},
    {"crossover_f_high", 1500.0, 6000.0, ScaleType::Exponential, "Hz"},

    // 20..34: Multiband Comp (3 bande x 5 parametri)
    // Low Band (20..24)
    {"comp_low_threshold_db", -40.0,  0.0,    ScaleType::Linear,      "dB"},
    {"comp_low_ratio",        1.0,    20.0,   ScaleType::Exponential, ""},
    {"comp_low_attack_ms",    0.5,    100.0,  ScaleType::Exponential, "ms"},
    {"comp_low_release_ms",   10.0,   1000.0, ScaleType::Exponential, "ms"},
    {"comp_low_makeup_db",    -6.0,   +12.0,  ScaleType::Linear,      "dB"},
    // Mid Band (25..29)
    {"comp_mid_threshold_db", -40.0,  0.0,    ScaleType::Linear,      "dB"},
    {"comp_mid_ratio",        1.0,    20.0,   ScaleType::Exponential, ""},
    {"comp_mid_attack_ms",    0.5,    100.0,  ScaleType::Exponential, "ms"},
    {"comp_mid_release_ms",   10.0,   1000.0, ScaleType::Exponential, "ms"},
    {"comp_mid_makeup_db",    -6.0,   +12.0,  ScaleType::Linear,      "dB"},
    // High Band (30..34)
    {"comp_high_threshold_db", -40.0, 0.0,    ScaleType::Linear,      "dB"},
    {"comp_high_ratio",        1.0,   20.0,   ScaleType::Exponential, ""},
    {"comp_high_attack_ms",    0.5,   100.0,  ScaleType::Exponential, "ms"},
    {"comp_high_release_ms",   10.0,  1000.0, ScaleType::Exponential, "ms"},
    {"comp_high_makeup_db",    -6.0,  +12.0,  ScaleType::Linear,      "dB"},

    // 35..37: Stereo Imager (3 parametri)
    {"imager_width_low",  0.0, 1.2, ScaleType::Linear, ""},
    {"imager_width_mid",  0.5, 2.0, ScaleType::Linear, ""},
    {"imager_width_high", 0.5, 2.0, ScaleType::Linear, ""},

    // 38..39: Harmonic Saturator (2 parametri)
    {"saturator_drive_db", 0.0, 18.0, ScaleType::Linear, "dB"},
    {"saturator_mix",      0.0, 1.0,  ScaleType::Linear, ""},

    // 40: Lookahead Limiter (1 parametro)
    {"limiter_push_gain_db", 0.0, 12.0, ScaleType::Linear, "dB"}
}};

// Struttura C++ tipizzata e chiara per uso nel motore C++ e nel plugin
struct MasteringParameters {
    struct BandEQ {
        double f0 = 1000.0;
        double gain_db = 0.0;
        double q = 0.7071;
    };

    struct BandComp {
        double threshold_db = -20.0;
        double ratio = 2.0;
        double attack_ms = 10.0;
        double release_ms = 100.0;
        double makeup_db = 0.0;
    };

    // 1. EQ
    std::array<BandEQ, 6> eq{};

    // 2. Crossover
    double crossover_f_low = 150.0;
    double crossover_f_high = 3000.0;

    // 3. Compressore
    std::array<BandComp, 3> comp{};

    // 4. Imager
    double imager_width_low = 1.0;
    double imager_width_mid = 1.0;
    double imager_width_high = 1.0;

    // 5. Saturatore
    double saturator_drive_db = 0.0;
    double saturator_mix = 0.0;

    // 6. Limiter
    double limiter_push_gain_db = 0.0;
};

// Funzione di conversione da array normalizzato z a parametri fisici theta
[[nodiscard]] inline std::array<double, NUM_PARAMS> map_z_to_theta(const std::array<double, NUM_PARAMS>& z) noexcept {
    std::array<double, NUM_PARAMS> theta{};
    for (size_t i = 0; i < NUM_PARAMS; ++i) {
        theta[i] = PARAM_SPECS[i].map_z(z[i]);
    }
    return theta;
}

// Funzione di conversione inversa da parametri fisici theta a z normalizzato
[[nodiscard]] inline std::array<double, NUM_PARAMS> map_theta_to_z(const std::array<double, NUM_PARAMS>& theta) noexcept {
    std::array<double, NUM_PARAMS> z{};
    for (size_t i = 0; i < NUM_PARAMS; ++i) {
        z[i] = PARAM_SPECS[i].unmap_theta(theta[i]);
    }
    return z;
}

// Conversione da array theta a struttura tipizzata MasteringParameters
[[nodiscard]] inline MasteringParameters unpack_theta(const std::array<double, NUM_PARAMS>& theta) noexcept {
    MasteringParameters p{};
    // EQ
    for (size_t b = 0; b < 6; ++b) {
        p.eq[b].f0      = theta[b * 3 + 0];
        p.eq[b].gain_db = theta[b * 3 + 1];
        p.eq[b].q       = theta[b * 3 + 2];
    }
    // Crossover
    p.crossover_f_low  = theta[18];
    p.crossover_f_high = theta[19];
    // Compressore
    for (size_t b = 0; b < 3; ++b) {
        p.comp[b].threshold_db = theta[20 + b * 5 + 0];
        p.comp[b].ratio        = theta[20 + b * 5 + 1];
        p.comp[b].attack_ms    = theta[20 + b * 5 + 2];
        p.comp[b].release_ms   = theta[20 + b * 5 + 3];
        p.comp[b].makeup_db    = theta[20 + b * 5 + 4];
    }
    // Imager
    p.imager_width_low  = theta[35];
    p.imager_width_mid  = theta[36];
    p.imager_width_high = theta[37];
    // Saturatore
    p.saturator_drive_db = theta[38];
    p.saturator_mix      = theta[39];
    // Limiter
    p.limiter_push_gain_db = theta[40];

    return p;
}

} // namespace core_dsp
