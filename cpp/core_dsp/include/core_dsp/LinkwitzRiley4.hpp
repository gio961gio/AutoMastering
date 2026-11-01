#pragma once

#include <cmath>
#include <array>
#include <algorithm>
#include "Constants.hpp"
#include "BiquadCascade.hpp"

namespace core_dsp {

// Coefficienti per Butterworth 2° ordine e All-Pass di compensazione
struct LR4Coeffs {
    BiquadCoeffs lp2{};
    BiquadCoeffs hp2{};
    BiquadCoeffs ap2{};
};

[[nodiscard]] inline LR4Coeffs compute_lr4_coeffs(double fc, double sample_rate = DEFAULT_SAMPLE_RATE) noexcept {
    LR4Coeffs c{};
    fc = std::clamp(fc, 10.0, sample_rate * 0.499);

    const double wc = (TWO_PI * fc) / sample_rate;
    const double k = std::tan(wc * 0.5);
    const double k2 = k * k;
    const double a0 = 1.0 + SQRT2 * k + k2;
    const double inv_a0 = 1.0 / a0;

    const double a1_norm = 2.0 * (k2 - 1.0) * inv_a0;
    const double a2_norm = (1.0 - SQRT2 * k + k2) * inv_a0;

    // LP2
    const double b0_lp = k2 * inv_a0;
    c.lp2.b0 = b0_lp;
    c.lp2.b1 = 2.0 * b0_lp;
    c.lp2.b2 = b0_lp;
    c.lp2.a1 = a1_norm;
    c.lp2.a2 = a2_norm;

    // HP2
    const double b0_hp = inv_a0;
    c.hp2.b0 = b0_hp;
    c.hp2.b1 = -2.0 * b0_hp;
    c.hp2.b2 = b0_hp;
    c.hp2.a1 = a1_norm;
    c.hp2.a2 = a2_norm;

    // AP2: All-Pass 2° ordine che eguaglia la fase di LP4 + HP4
    c.ap2.b0 = a2_norm;
    c.ap2.b1 = a1_norm;
    c.ap2.b2 = 1.0;
    c.ap2.a1 = a1_norm;
    c.ap2.a2 = a2_norm;

    return c;
}

// Crossover Stereo a 3 Vie Linkwitz-Riley del 4° ordine compensato in fase
class LinkwitzRiley4_3Way {
public:
    LinkwitzRiley4_3Way() = default;

    void reset() noexcept {
        // Canale Left
        lp_low_1_l_.reset();
        lp_low_2_l_.reset();
        hp_low_1_l_.reset();
        hp_low_2_l_.reset();
        ap_low_high_l_.reset();
        lp_high_1_l_.reset();
        lp_high_2_l_.reset();
        hp_high_1_l_.reset();
        hp_high_2_l_.reset();

        // Canale Right
        lp_low_1_r_.reset();
        lp_low_2_r_.reset();
        hp_low_1_r_.reset();
        hp_low_2_r_.reset();
        ap_low_high_r_.reset();
        lp_high_1_r_.reset();
        lp_high_2_r_.reset();
        hp_high_1_r_.reset();
        hp_high_2_r_.reset();
    }

    void set_crossover_frequencies(double f_low, double f_high, double sample_rate = DEFAULT_SAMPLE_RATE) noexcept {
        f_low = std::clamp(f_low, 20.0, 1000.0);
        f_high = std::clamp(f_high, f_low + 50.0, sample_rate * 0.45);

        coeffs_low_ = compute_lr4_coeffs(f_low, sample_rate);
        coeffs_high_ = compute_lr4_coeffs(f_high, sample_rate);
    }

    // Processa campioni stereo separando nelle 3 bande: low, mid, high
    void process(const float* in_l, const float* in_r,
                 float* low_l, float* low_r,
                 float* mid_l, float* mid_r,
                 float* high_l, float* high_r,
                 size_t num_samples) noexcept {
        for (size_t n = 0; n < num_samples; ++n) {
            // === CANALE LEFT ===
            double xl = static_cast<double>(in_l[n]);
            // Split low a flow: LP4 (due stadi LP2) e HP4 (due stadi HP2)
            double u_low_l = lp_low_2_l_.process_sample(lp_low_1_l_.process_sample(xl, coeffs_low_.lp2), coeffs_low_.lp2);
            double u_mh_l  = hp_low_2_l_.process_sample(hp_low_1_l_.process_sample(xl, coeffs_low_.hp2), coeffs_low_.hp2);

            // Compensazione All-Pass su Low branch a fhigh
            double y_low_l = ap_low_high_l_.process_sample(u_low_l, coeffs_high_.ap2);

            // Split medio-alto a fhigh
            double y_mid_l  = lp_high_2_l_.process_sample(lp_high_1_l_.process_sample(u_mh_l, coeffs_high_.lp2), coeffs_high_.lp2);
            double y_high_l = hp_high_2_l_.process_sample(hp_high_1_l_.process_sample(u_mh_l, coeffs_high_.hp2), coeffs_high_.hp2);

            low_l[n]  = static_cast<float>(y_low_l);
            mid_l[n]  = static_cast<float>(y_mid_l);
            high_l[n] = static_cast<float>(y_high_l);

            // === CANALE RIGHT ===
            double xr = static_cast<double>(in_r[n]);
            double u_low_r = lp_low_2_r_.process_sample(lp_low_1_r_.process_sample(xr, coeffs_low_.lp2), coeffs_low_.lp2);
            double u_mh_r  = hp_low_2_r_.process_sample(hp_low_1_r_.process_sample(xr, coeffs_low_.hp2), coeffs_low_.hp2);

            double y_low_r = ap_low_high_r_.process_sample(u_low_r, coeffs_high_.ap2);

            double y_mid_r  = lp_high_2_r_.process_sample(lp_high_1_r_.process_sample(u_mh_r, coeffs_high_.lp2), coeffs_high_.lp2);
            double y_high_r = hp_high_2_r_.process_sample(hp_high_1_r_.process_sample(u_mh_r, coeffs_high_.hp2), coeffs_high_.hp2);

            low_r[n]  = static_cast<float>(y_low_r);
            mid_r[n]  = static_cast<float>(y_mid_r);
            high_r[n] = static_cast<float>(y_high_r);
        }
    }

private:
    LR4Coeffs coeffs_low_{};
    LR4Coeffs coeffs_high_{};

    // Celle biquad Left
    BiquadDF2T lp_low_1_l_{}, lp_low_2_l_{};
    BiquadDF2T hp_low_1_l_{}, hp_low_2_l_{};
    BiquadDF2T ap_low_high_l_{};
    BiquadDF2T lp_high_1_l_{}, lp_high_2_l_{};
    BiquadDF2T hp_high_1_l_{}, hp_high_2_l_{};

    // Celle biquad Right
    BiquadDF2T lp_low_1_r_{}, lp_low_2_r_{};
    BiquadDF2T hp_low_1_r_{}, hp_low_2_r_{};
    BiquadDF2T ap_low_high_r_{};
    BiquadDF2T lp_high_1_r_{}, lp_high_2_r_{};
    BiquadDF2T hp_high_1_r_{}, hp_high_2_r_{};
};

} // namespace core_dsp
