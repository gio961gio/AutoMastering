#pragma once

#include <cmath>
#include <array>
#include <algorithm>
#include "Constants.hpp"

namespace core_dsp {

// Lookahead Brickwall Limiter con Monotonic Deque O(1) e Boxcar Smoothing
class LookaheadLimiter {
public:
    static constexpr size_t LOOKAHEAD = LIMITER_LOOKAHEAD_SAMPLES; // 144
    static constexpr double CEILING_DB = LIMITER_CEILING_DBFS;     // -0.3 dBFS
    static constexpr double RELEASE_MS = LIMITER_RELEASE_MS;       // 80.0 ms

    explicit LookaheadLimiter(double sample_rate = DEFAULT_SAMPLE_RATE) {
        set_sample_rate(sample_rate);
        reset();
    }

    void set_sample_rate(double sample_rate) noexcept {
        sample_rate_ = (sample_rate > 0.0) ? sample_rate : DEFAULT_SAMPLE_RATE;
        const double tau_sec = RELEASE_MS * 0.001;
        alpha_rel_ = std::exp(-1.0 / (tau_sec * sample_rate_));
    }

    void set_push_gain_db(double push_db) noexcept {
        push_db_ = push_db;
        push_lin_ = std::pow(10.0, push_db_ / 20.0);
    }

    [[nodiscard]] double get_push_gain_db() const noexcept {
        return push_db_;
    }

    void reset() noexcept {
        // Reset linea di ritardo audio (144 campioni)
        delay_l_.fill(0.0);
        delay_r_.fill(0.0);
        delay_write_idx_ = 0;

        // Reset boxcar moving average buffer (144 campioni a 1.0)
        boxcar_buf_.fill(1.0);
        boxcar_idx_ = 0;
        boxcar_sum_ = static_cast<double>(LOOKAHEAD) * 1.0;

        // Reset envelope release
        g_rel_ = 1.0;

        // Reset monotonic deque circolare (capacità 256 per LOOKAHEAD=144)
        sample_count_ = 0;
        deque_head_ = 0;
        deque_tail_ = 0;
        deque_count_ = 0;

        // Inserimento elemento iniziale di guardia a 1.0
        push_deque(0, 1.0);
    }

    // Processa un singolo campione stereo in streaming
    void process_sample(double in_l, double in_r, double& out_l, double& out_r) noexcept {
        // 1. Gain Push
        const double u_l = in_l * push_lin_;
        const double u_r = in_r * push_lin_;

        // 2. Lettura audio ritardato di NL campioni dalla linea di ritardo
        const double delayed_l = delay_l_[delay_write_idx_];
        const double delayed_r = delay_r_[delay_write_idx_];
        delay_l_[delay_write_idx_] = u_l;
        delay_r_[delay_write_idx_] = u_r;
        delay_write_idx_ = (delay_write_idx_ + 1) % LOOKAHEAD;

        // 3. Peak Detection & Target Gain Istantaneo
        const double peak = std::max({std::abs(u_l), std::abs(u_r), 1e-7});
        const double g_target = std::min(1.0, c_lin_ / peak);

        // 4. Stadio 1: Min-Filter su finestra NL tramite Monotonic Deque O(1)
        const size_t current_sample = sample_count_++;

        // Rimuovi elementi dal fondo più grandi o uguali a g_target
        while (deque_count_ > 0) {
            const size_t back_idx = (deque_tail_ + DEQUE_CAP - 1) % DEQUE_CAP;
            if (deque_buf_[back_idx].val >= g_target) {
                deque_tail_ = back_idx;
                --deque_count_;
            } else {
                break;
            }
        }

        // Aggiungi nuovo elemento al fondo
        push_deque(current_sample, g_target);

        // Rimuovi elementi obsoleti dalla testa (più vecchi di LOOKAHEAD)
        while (deque_count_ > 0) {
            if (current_sample >= deque_buf_[deque_head_].sample_idx + LOOKAHEAD) {
                deque_head_ = (deque_head_ + 1) % DEQUE_CAP;
                --deque_count_;
            } else {
                break;
            }
        }

        const double g_min = deque_buf_[deque_head_].val;

        // 5. Stadio 2: One-Pole Release Envelope
        if (g_min <= g_rel_) {
            g_rel_ = g_min; // Attacco istantaneo sul minimo
        } else {
            g_rel_ = alpha_rel_ * g_rel_ + (1.0 - alpha_rel_) * g_min;
        }

        // 6. Stadio 3: Boxcar Moving Average di lunghezza NL
        const double old_boxcar_val = boxcar_buf_[boxcar_idx_];
        boxcar_buf_[boxcar_idx_] = g_rel_;
        boxcar_sum_ += (g_rel_ - old_boxcar_val);
        boxcar_idx_ = (boxcar_idx_ + 1) % LOOKAHEAD;

        const double g_smooth = boxcar_sum_ * INV_LOOKAHEAD;

        // 7. Uscita Sincronizzata
        out_l = delayed_l * g_smooth;
        out_r = delayed_r * g_smooth;
    }

    // Processa blocchi contigui di campioni stereo
    void process(
        const float* in_l, const float* in_r,
        float* out_l, float* out_r,
        size_t num_samples
    ) noexcept {
        for (size_t n = 0; n < num_samples; ++n) {
            double ol = 0.0, or_ = 0.0;
            process_sample(static_cast<double>(in_l[n]),
                           static_cast<double>(in_r[n]),
                           ol, or_);
            out_l[n] = static_cast<float>(ol);
            out_r[n] = static_cast<float>(or_);
        }
    }

private:
    struct DequeItem {
        size_t sample_idx = 0;
        double val = 1.0;
    };

    static constexpr size_t DEQUE_CAP = 256;
    static constexpr double INV_LOOKAHEAD = 1.0 / static_cast<double>(LOOKAHEAD);

    void push_deque(size_t sample_idx, double val) noexcept {
        deque_buf_[deque_tail_] = {sample_idx, val};
        deque_tail_ = (deque_tail_ + 1) % DEQUE_CAP;
        ++deque_count_;
    }

    double sample_rate_ = DEFAULT_SAMPLE_RATE;
    double push_db_ = 0.0;
    double push_lin_ = 1.0;
    const double c_lin_ = std::pow(10.0, CEILING_DB / 20.0); // ~0.96605087870305
    double alpha_rel_ = 0.0;

    // Buffer audio di ritardo Lookahead
    std::array<double, LOOKAHEAD> delay_l_{};
    std::array<double, LOOKAHEAD> delay_r_{};
    size_t delay_write_idx_ = 0;

    // Buffer Boxcar Moving Average
    std::array<double, LOOKAHEAD> boxcar_buf_{};
    size_t boxcar_idx_ = 0;
    double boxcar_sum_ = static_cast<double>(LOOKAHEAD);

    // Stato Envelope
    double g_rel_ = 1.0;

    // Struttura Monotonic Deque O(1) circolare
    std::array<DequeItem, DEQUE_CAP> deque_buf_{};
    size_t sample_count_ = 0;
    size_t deque_head_ = 0;
    size_t deque_tail_ = 0;
    size_t deque_count_ = 0;
};

} // namespace core_dsp
