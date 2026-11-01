#pragma once

#include <cmath>
#include <array>
#include <algorithm>
#include "Constants.hpp"

namespace core_dsp {

// Saturatore Armonico Stereo con Oversampling Polifase 2x e Compensazione Latenza Dry (15 campioni)
class HarmonicSaturator {
public:
    HarmonicSaturator() {
        reset();
    }

    void reset() noexcept {
        // Canale Left
        state_up_l_.fill(0.0);
        state_down_l_.fill(0.0);
        delay_dry_l_.fill(0.0);
        dry_idx_l_ = 0;

        // Canale Right
        state_up_r_.fill(0.0);
        state_down_r_.fill(0.0);
        delay_dry_r_.fill(0.0);
        dry_idx_r_ = 0;
    }

    // drive_db in [0, 18], mix in [0, 1]
    void set_parameters(double drive_db, double mix) noexcept {
        drive_db = std::clamp(drive_db, 0.0, 36.0);
        mix_ = std::clamp(mix, 0.0, 1.0);
        gd_ = std::pow(10.0, drive_db / 20.0);
        inv_sqrt_gd_ = 1.0 / std::sqrt(gd_);
    }

    void process(const float* in_l, const float* in_r,
                 float* out_l, float* out_r,
                 size_t num_samples) noexcept {
        for (size_t n = 0; n < num_samples; ++n) {
            out_l[n] = process_channel(static_cast<double>(in_l[n]),
                                       state_up_l_, state_down_l_,
                                       delay_dry_l_, dry_idx_l_);

            out_r[n] = process_channel(static_cast<double>(in_r[n]),
                                       state_up_r_, state_down_r_,
                                       delay_dry_r_, dry_idx_r_);
        }
    }

private:
    [[nodiscard]] float process_channel(
        double x,
        std::array<double, FIR_HALFBAND_TAPS>& state_up,
        std::array<double, FIR_HALFBAND_TAPS>& state_down,
        std::array<double, FIR_HALFBAND_LATENCY + 1>& delay_dry,
        size_t& dry_idx
    ) noexcept {
        // 1. Dry delay line di 15 campioni
        // Salva x corrente e leggi il campione ritardato di 15 passi
        delay_dry[dry_idx] = x;
        const size_t read_idx = (dry_idx + 1) % (FIR_HALFBAND_LATENCY + 1);
        const double x_dry = delay_dry[read_idx];
        dry_idx = read_idx;

        // 2. Sub-campione 0 a 2*fs (m = 2n, valore 2.0 * x)
        push_sample(state_up, 2.0 * x);
        const double u0 = dot_fir(state_up);
        const double v0 = std::tanh(gd_ * u0) * inv_sqrt_gd_;

        // Inserisci v0 nel filtro di decimazione e valuta l'uscita wet
        push_sample(state_down, v0);
        const double y_wet = dot_fir(state_down);

        // 3. Sub-campione 1 a 2*fs (m = 2n + 1, valore 0.0)
        push_sample(state_up, 0.0);
        const double u1 = dot_fir(state_up);
        const double v1 = std::tanh(gd_ * u1) * inv_sqrt_gd_;
        push_sample(state_down, v1);

        // 4. Mix Dry/Wet perfettamente allineato in fase (zero comb-filtering)
        const double y_sat = (1.0 - mix_) * x_dry + mix_ * y_wet;
        return static_cast<float>(y_sat);
    }

    static void push_sample(std::array<double, FIR_HALFBAND_TAPS>& state, double s) noexcept {
        for (size_t i = 0; i < FIR_HALFBAND_TAPS - 1; ++i) {
            state[i] = state[i + 1];
        }
        state[FIR_HALFBAND_TAPS - 1] = s;
    }

    [[nodiscard]] static double dot_fir(const std::array<double, FIR_HALFBAND_TAPS>& state) noexcept {
        double sum = 0.0;
        // In C++ con flag /fp:precise e coefficienti constexpr, il compilatore unrolla e vettorizza
        for (size_t i = 0; i < FIR_HALFBAND_TAPS; ++i) {
            sum += state[i] * FIR_HALFBAND_COEFFS[i];
        }
        return sum;
    }

    double gd_ = 1.0;
    double inv_sqrt_gd_ = 1.0;
    double mix_ = 0.0;

    std::array<double, FIR_HALFBAND_TAPS> state_up_l_{};
    std::array<double, FIR_HALFBAND_TAPS> state_down_l_{};
    std::array<double, FIR_HALFBAND_LATENCY + 1> delay_dry_l_{};
    size_t dry_idx_l_ = 0;

    std::array<double, FIR_HALFBAND_TAPS> state_up_r_{};
    std::array<double, FIR_HALFBAND_TAPS> state_down_r_{};
    std::array<double, FIR_HALFBAND_LATENCY + 1> delay_dry_r_{};
    size_t dry_idx_r_ = 0;
};

} // namespace core_dsp
