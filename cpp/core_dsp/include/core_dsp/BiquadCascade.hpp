#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>
#include <algorithm>
#include "Constants.hpp"

namespace core_dsp {

enum class FilterType {
    LowShelf,
    Peaking,
    HighShelf
};

struct BiquadCoeffs {
    double b0 = 1.0;
    double b1 = 0.0;
    double b2 = 0.0;
    double a1 = 0.0;
    double a2 = 0.0;
};

// Calcolo dei coefficienti biquad RBJ Cookbook normalizzati per a0
[[nodiscard]] inline BiquadCoeffs compute_rbj_biquad(FilterType type, double f0, double gain_db, double q, double sample_rate = DEFAULT_SAMPLE_RATE) noexcept {
    BiquadCoeffs c{};
    // Clamp parametri di sicurezza
    f0 = std::clamp(f0, 10.0, sample_rate * 0.499);
    q = std::max(q, 0.05);

    const double w0 = (TWO_PI * f0) / sample_rate;
    const double cos_w0 = std::cos(w0);
    const double sin_w0 = std::sin(w0);
    const double alpha = sin_w0 / (2.0 * q);
    const double A = std::pow(10.0, gain_db / 40.0);

    double b0_raw = 1.0, b1_raw = 0.0, b2_raw = 0.0;
    double a0_raw = 1.0, a1_raw = 0.0, a2_raw = 0.0;

    switch (type) {
        case FilterType::Peaking: {
            b0_raw = 1.0 + alpha * A;
            b1_raw = -2.0 * cos_w0;
            b2_raw = 1.0 - alpha * A;
            a0_raw = 1.0 + alpha / A;
            a1_raw = -2.0 * cos_w0;
            a2_raw = 1.0 - alpha / A;
            break;
        }
        case FilterType::LowShelf: {
            const double s_shelf = 2.0 * std::sqrt(A) * alpha;
            const double Ap1 = A + 1.0;
            const double Am1 = A - 1.0;
            b0_raw = A * (Ap1 - Am1 * cos_w0 + s_shelf);
            b1_raw = 2.0 * A * (Am1 - Ap1 * cos_w0);
            b2_raw = A * (Ap1 - Am1 * cos_w0 - s_shelf);
            a0_raw = Ap1 + Am1 * cos_w0 + s_shelf;
            a1_raw = -2.0 * (Am1 + Ap1 * cos_w0);
            a2_raw = Ap1 + Am1 * cos_w0 - s_shelf;
            break;
        }
        case FilterType::HighShelf: {
            const double s_shelf = 2.0 * std::sqrt(A) * alpha;
            const double Ap1 = A + 1.0;
            const double Am1 = A - 1.0;
            b0_raw = A * (Ap1 + Am1 * cos_w0 + s_shelf);
            b1_raw = -2.0 * A * (Am1 + Ap1 * cos_w0);
            b2_raw = A * (Ap1 + Am1 * cos_w0 - s_shelf);
            a0_raw = Ap1 - Am1 * cos_w0 + s_shelf;
            a1_raw = 2.0 * (Am1 - Ap1 * cos_w0);
            a2_raw = Ap1 - Am1 * cos_w0 - s_shelf;
            break;
        }
    }

    const double inv_a0 = 1.0 / a0_raw;
    c.b0 = b0_raw * inv_a0;
    c.b1 = b1_raw * inv_a0;
    c.b2 = b2_raw * inv_a0;
    c.a1 = a1_raw * inv_a0;
    c.a2 = a2_raw * inv_a0;

    return c;
}

// Cella Biquad singola in Direct Form II Transposed (DF2T) con stato a doppia precisione
class BiquadDF2T {
public:
    void reset() noexcept {
        s1_ = 0.0;
        s2_ = 0.0;
    }

    [[nodiscard]] double process_sample(double x, const BiquadCoeffs& c) noexcept {
        const double y = c.b0 * x + s1_;
        s1_ = c.b1 * x - c.a1 * y + s2_;
        s2_ = c.b2 * x - c.a2 * y;
        return y;
    }

private:
    double s1_ = 0.0;
    double s2_ = 0.0;
};

// Cascata di 6 Bande EQ Stereo (Left, Right separati)
class BiquadCascade6Band {
public:
    BiquadCascade6Band() = default;

    void reset() noexcept {
        for (size_t b = 0; b < 6; ++b) {
            cells_l_[b].reset();
            cells_r_[b].reset();
        }
    }

    // Configura i coefficienti per le 6 bande partendo da vettori di f0, gain_db, q
    void set_parameters(const std::array<double, 6>& f0,
                        const std::array<double, 6>& gain_db,
                        const std::array<double, 6>& q,
                        double sample_rate = DEFAULT_SAMPLE_RATE) noexcept {
        // Banda 0: Low-Shelf
        coeffs_[0] = compute_rbj_biquad(FilterType::LowShelf, f0[0], gain_db[0], q[0], sample_rate);
        // Bande 1..4: Peaking
        for (size_t b = 1; b <= 4; ++b) {
            coeffs_[b] = compute_rbj_biquad(FilterType::Peaking, f0[b], gain_db[b], q[b], sample_rate);
        }
        // Banda 5: High-Shelf
        coeffs_[5] = compute_rbj_biquad(FilterType::HighShelf, f0[5], gain_db[5], q[5], sample_rate);
    }

    // Elaborazione stereo in-place o out-of-place (campioni in float32, calcoli interni float64)
    void process(const float* in_l, const float* in_r, float* out_l, float* out_r, size_t num_samples) noexcept {
        for (size_t n = 0; n < num_samples; ++n) {
            double smp_l = static_cast<double>(in_l[n]);
            double smp_r = static_cast<double>(in_r[n]);

            for (size_t b = 0; b < 6; ++b) {
                smp_l = cells_l_[b].process_sample(smp_l, coeffs_[b]);
                smp_r = cells_r_[b].process_sample(smp_r, coeffs_[b]);
            }

            out_l[n] = static_cast<float>(smp_l);
            out_r[n] = static_cast<float>(smp_r);
        }
    }

    [[nodiscard]] const std::array<BiquadCoeffs, 6>& get_coeffs() const noexcept {
        return coeffs_;
    }

private:
    std::array<BiquadCoeffs, 6> coeffs_{};
    std::array<BiquadDF2T, 6> cells_l_{};
    std::array<BiquadDF2T, 6> cells_r_{};
};

} // namespace core_dsp
