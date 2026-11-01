#pragma once

#include <cstddef>
#include <algorithm>
#include "Constants.hpp"

namespace core_dsp {

// Stereo Imager basato su matrice Mid/Side con scaling della componente Side
class MidSideImager {
public:
    MidSideImager() = default;

    // Processa un singolo blocco stereo di una banda b
    // width: 0.0 = mono puro, 1.0 = stereo naturale intatto, > 1.0 = stereo allargato
    static void process_band(const float* in_l, const float* in_r,
                             float* out_l, float* out_r,
                             size_t num_samples,
                             double width) noexcept {
        const double w = std::max(0.0, width);
        for (size_t n = 0; n < num_samples; ++n) {
            const double l = static_cast<double>(in_l[n]);
            const double r = static_cast<double>(in_r[n]);

            // Codifica Mid/Side ortonormale (INV_SQRT2 = 1/sqrt(2))
            const double mid  = (l + r) * INV_SQRT2;
            const double side = (l - r) * INV_SQRT2 * w;

            // Decodifica Left/Right
            out_l[n] = static_cast<float>((mid + side) * INV_SQRT2);
            out_r[n] = static_cast<float>((mid - side) * INV_SQRT2);
        }
    }

    // Processa simultaneamente le 3 bande uscenti dal crossover e calcola la somma x_sum
    static void process_3band_and_sum(
        const float* low_l, const float* low_r,
        const float* mid_l, const float* mid_r,
        const float* high_l, const float* high_r,
        float* sum_l, float* sum_r,
        size_t num_samples,
        double width_low, double width_mid, double width_high
    ) noexcept {
        const double w_low  = std::max(0.0, width_low);
        const double w_mid  = std::max(0.0, width_mid);
        const double w_high = std::max(0.0, width_high);

        for (size_t n = 0; n < num_samples; ++n) {
            // Low
            const double m_l = (low_l[n] + low_r[n]) * INV_SQRT2;
            const double s_l = (low_l[n] - low_r[n]) * INV_SQRT2 * w_low;
            const double out_low_l = (m_l + s_l) * INV_SQRT2;
            const double out_low_r = (m_l - s_l) * INV_SQRT2;

            // Mid
            const double m_m = (mid_l[n] + mid_r[n]) * INV_SQRT2;
            const double s_m = (mid_l[n] - mid_r[n]) * INV_SQRT2 * w_mid;
            const double out_mid_l = (m_m + s_m) * INV_SQRT2;
            const double out_mid_r = (m_m - s_m) * INV_SQRT2;

            // High
            const double m_h = (high_l[n] + high_r[n]) * INV_SQRT2;
            const double s_h = (high_l[n] - high_r[n]) * INV_SQRT2 * w_high;
            const double out_high_l = (m_h + s_h) * INV_SQRT2;
            const double out_high_r = (m_h - s_h) * INV_SQRT2;

            // Somma multibanda
            sum_l[n] = static_cast<float>(out_low_l + out_mid_l + out_high_l);
            sum_r[n] = static_cast<float>(out_low_r + out_mid_r + out_high_r);
        }
    }
};

} // namespace core_dsp
