#pragma once

#include <cmath>
#include <array>
#include <algorithm>
#include "Constants.hpp"

namespace core_dsp {

// Parametri per singola banda di compressione
struct CompBandParams {
    double threshold_db = -20.0; // [-40.0, 0.0]
    double ratio = 2.0;          // [1.0, 20.0]
    double attack_ms = 10.0;     // [0.5, 100.0]
    double release_ms = 100.0;   // [10.0, 1000.0]
    double makeup_db = 0.0;      // [-6.0, +12.0]
    double knee_db = 6.0;        // Fissa a 6.0 dB
};

// Coefficienti pre-calcolati per elaborazione sample-by-sample
struct CompBandCoeffs {
    double threshold_db = -20.0;
    double ratio = 2.0;
    double inv_ratio = 0.5;
    double ratio_factor = 0.5; // (1 - 1/ratio)
    double knee_db = 6.0;
    double half_knee = 3.0;
    double inv_two_knee = 1.0 / 12.0;
    double alpha_a = 0.0;
    double alpha_r = 0.0;
    double makeup_lin = 1.0;
};

[[nodiscard]] inline CompBandCoeffs compute_comp_coeffs(const CompBandParams& p, double sample_rate = DEFAULT_SAMPLE_RATE) noexcept {
    CompBandCoeffs c{};
    c.threshold_db = p.threshold_db;
    c.ratio = std::max(1.0, p.ratio);
    c.inv_ratio = 1.0 / c.ratio;
    c.ratio_factor = 1.0 - c.inv_ratio;
    c.knee_db = std::max(0.1, p.knee_db);
    c.half_knee = c.knee_db * 0.5;
    c.inv_two_knee = 1.0 / (2.0 * c.knee_db);

    const double tau_a_sec = std::max(0.0001, p.attack_ms * 0.001);
    const double tau_r_sec = std::max(0.001, p.release_ms * 0.001);
    c.alpha_a = std::exp(-1.0 / (tau_a_sec * sample_rate));
    c.alpha_r = std::exp(-1.0 / (tau_r_sec * sample_rate));

    c.makeup_lin = std::pow(10.0, p.makeup_db / 20.0);
    return c;
}

// Compressore a singola banda stereo-linked
class SingleBandComp {
public:
    SingleBandComp() = default;

    void reset() noexcept {
        yg_ = 0.0;
    }

    void set_parameters(const CompBandParams& p, double sample_rate = DEFAULT_SAMPLE_RATE) noexcept {
        coeffs_ = compute_comp_coeffs(p, sample_rate);
    }

    // Processa un singolo campione stereo in-place o out-of-place
    void process_sample(double in_l, double in_r, double& out_l, double& out_r) noexcept {
        // 1. Stereo-linked energy detection in dB
        const double energy = 0.5 * (in_l * in_l + in_r * in_r) + EPSILON;
        const double x_db = 10.0 * std::log10(energy);

        // 2. C1 Soft-Knee Gain Computer
        const double delta = x_db - coeffs_.threshold_db;
        double gc = 0.0;
        if (2.0 * delta <= -coeffs_.knee_db) {
            gc = 0.0;
        } else if (std::abs(2.0 * delta) < coeffs_.knee_db) {
            const double d_plus_half = delta + coeffs_.half_knee;
            gc = coeffs_.ratio_factor * (d_plus_half * d_plus_half) * coeffs_.inv_two_knee;
        } else {
            gc = coeffs_.ratio_factor * delta;
        }

        // 3. Recursive log-domain ballistics
        const double alpha = (gc > yg_) ? coeffs_.alpha_a : coeffs_.alpha_r;
        yg_ = alpha * yg_ + (1.0 - alpha) * gc;

        // 4. Linear gain application + Makeup
        // glin = 10^(-yg / 20) * makeup_lin
        const double glin = std::pow(10.0, -yg_ / 20.0) * coeffs_.makeup_lin;

        out_l = in_l * glin;
        out_r = in_r * glin;
    }

    [[nodiscard]] double get_envelope() const noexcept {
        return yg_;
    }

private:
    CompBandCoeffs coeffs_{};
    double yg_ = 0.0;
};

// Compressore Multibanda a 3 Bande Stereo
class MultibandCompressor3Band {
public:
    MultibandCompressor3Band() = default;

    void reset() noexcept {
        comp_low_.reset();
        comp_mid_.reset();
        comp_high_.reset();
    }

    void set_parameters(
        const CompBandParams& low_params,
        const CompBandParams& mid_params,
        const CompBandParams& high_params,
        double sample_rate = DEFAULT_SAMPLE_RATE
    ) noexcept {
        comp_low_.set_parameters(low_params, sample_rate);
        comp_mid_.set_parameters(mid_params, sample_rate);
        comp_high_.set_parameters(high_params, sample_rate);
    }

    void process(
        const float* in_low_l, const float* in_low_r,
        const float* in_mid_l, const float* in_mid_r,
        const float* in_high_l, const float* in_high_r,
        float* out_low_l, float* out_low_r,
        float* out_mid_l, float* out_mid_r,
        float* out_high_l, float* out_high_r,
        size_t num_samples
    ) noexcept {
        for (size_t n = 0; n < num_samples; ++n) {
            double ol = 0.0, or_ = 0.0;

            // Low
            comp_low_.process_sample(static_cast<double>(in_low_l[n]),
                                     static_cast<double>(in_low_r[n]),
                                     ol, or_);
            out_low_l[n] = static_cast<float>(ol);
            out_low_r[n] = static_cast<float>(or_);

            // Mid
            comp_mid_.process_sample(static_cast<double>(in_mid_l[n]),
                                     static_cast<double>(in_mid_r[n]),
                                     ol, or_);
            out_mid_l[n] = static_cast<float>(ol);
            out_mid_r[n] = static_cast<float>(or_);

            // High
            comp_high_.process_sample(static_cast<double>(in_high_l[n]),
                                      static_cast<double>(in_high_r[n]),
                                      ol, or_);
            out_high_l[n] = static_cast<float>(ol);
            out_high_r[n] = static_cast<float>(or_);
        }
    }

private:
    SingleBandComp comp_low_{};
    SingleBandComp comp_mid_{};
    SingleBandComp comp_high_{};
};

} // namespace core_dsp
