#pragma once

#include <cmath>
#include <array>
#include <algorithm>
#include "Constants.hpp"

namespace core_dsp {

enum class LimiterDetectorMode {
    TruePeak4x = 0,
    LegacySample1x = 1
};

namespace detail {
    using TpPolyphaseTable = std::array<std::array<double, TP_FIR_TAPS_PER_PHASE>, TP_OVERSAMPLE_FACTOR>;

    constexpr TpPolyphaseTable make_tp_polyphase_table() noexcept {
        TpPolyphaseTable table{};
        for (size_t phi = 0; phi < TP_OVERSAMPLE_FACTOR; ++phi) {
            for (size_t k = 0; k < TP_FIR_TAPS_PER_PHASE; ++k) {
                table[phi][k] = TP_FIR_PROTOTYPE_COEFFS[k * TP_OVERSAMPLE_FACTOR + phi];
            }
        }
        return table;
    }

    inline constexpr TpPolyphaseTable TP_POLYPHASE_TABLE = make_tp_polyphase_table();
} // namespace detail

// Lookahead Brickwall Limiter con Detector True Peak 4x Polifase Causale (default)
// o modalità Legacy 1x Sample Peak per riproducibilità storica.
// Monotonic Deque O(1) + One-Pole Release + Boxcar Smoothing compensato senza release leak.
class LookaheadLimiter {
public:
    static constexpr size_t LOOKAHEAD = LIMITER_LOOKAHEAD_SAMPLES; // 144
    static constexpr double CEILING_DB = LIMITER_CEILING_DBFS;     // -0.3 dBFS (legacy / delivery)
    static constexpr double INTERNAL_CEILING_DBTP = LIMITER_INTERNAL_CEILING_DBTP; // -0.50 dBTP
    static constexpr size_t TP_BOXCAR_SAMPLES = LIMITER_BOXCAR_SAMPLES; // 136
    static constexpr double RELEASE_MS = LIMITER_RELEASE_MS;       // 80.0 ms

    explicit LookaheadLimiter(
        double sample_rate = DEFAULT_SAMPLE_RATE,
        LimiterDetectorMode mode = LimiterDetectorMode::TruePeak4x
    ) noexcept {
        set_sample_rate(sample_rate);
        apply_mode_defaults(mode);
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

    void set_detector_mode(LimiterDetectorMode mode) noexcept {
        apply_mode_defaults(mode);
        reset();
    }

    [[nodiscard]] LimiterDetectorMode get_detector_mode() const noexcept {
        return detector_mode_;
    }

    void set_ceiling_db(double ceiling_db) noexcept {
        ceiling_db_ = ceiling_db;
        c_lin_ = std::pow(10.0, ceiling_db_ / 20.0);
    }

    [[nodiscard]] double get_ceiling_db() const noexcept {
        return ceiling_db_;
    }

    [[nodiscard]] size_t get_boxcar_samples() const noexcept {
        return active_boxcar_len_;
    }

    [[nodiscard]] static constexpr size_t get_lookahead_samples() noexcept {
        return LOOKAHEAD;
    }

    void reset() noexcept {
        // Reset linea di ritardo audio (144 campioni)
        delay_l_.fill(0.0);
        delay_r_.fill(0.0);
        delay_write_idx_ = 0;

        // Reset storico FIR polifase doppio-bufferizzato (zero allocazioni)
        fir_hist_l_.fill(0.0);
        fir_hist_r_.fill(0.0);
        fir_write_idx_ = TP_FIR_TAPS_PER_PHASE - 1;

        // Reset boxcar moving average buffer alla finestra attiva
        boxcar_buf_.fill(1.0);
        boxcar_idx_ = 0;
        boxcar_sum_ = static_cast<double>(active_boxcar_len_);

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

    // Processa un singolo campione stereo in streaming (real-time safe, zero allocazioni)
    void process_sample(double in_l, double in_r, double& out_l, double& out_r) noexcept {
        // 1. Gain Push
        const double u_l = in_l * push_lin_;
        const double u_r = in_r * push_lin_;

        // 2. Lettura audio ritardato di NL=144 campioni dalla linea di ritardo
        const double delayed_l = delay_l_[delay_write_idx_];
        const double delayed_r = delay_r_[delay_write_idx_];
        delay_l_[delay_write_idx_] = u_l;
        delay_r_[delay_write_idx_] = u_r;
        delay_write_idx_ = (delay_write_idx_ + 1) % LOOKAHEAD;

        // 3. Peak Detection (True Peak 4x Polifase + aligned 1x, oppure Legacy 1x)
        double peak = 1e-7;
        if (detector_mode_ == LimiterDetectorMode::LegacySample1x) {
            peak = std::max({std::abs(u_l), std::abs(u_r), 1e-7});
        } else {
            const size_t w = fir_write_idx_;
            fir_hist_l_[w] = u_l;
            fir_hist_l_[w + TP_FIR_TAPS_PER_PHASE] = u_l;
            fir_hist_r_[w] = u_r;
            fir_hist_r_[w + TP_FIR_TAPS_PER_PHASE] = u_r;

            // Sample peak allineato al centro del filtro (~5.875 -> 6 campioni base)
            const double align_l = fir_hist_l_[w + TP_FIR_ALIGNED_SAMPLE_DELAY];
            const double align_r = fir_hist_r_[w + TP_FIR_ALIGNED_SAMPLE_DELAY];
            peak = std::max({peak, std::abs(align_l), std::abs(align_r)});

            // 4 rami polifase FIR causali da 12 tap per canale (stereo-linked)
            const double* hist_l = fir_hist_l_.data() + w;
            const double* hist_r = fir_hist_r_.data() + w;
            for (size_t phi = 0; phi < TP_OVERSAMPLE_FACTOR; ++phi) {
                const auto& taps = detail::TP_POLYPHASE_TABLE[phi];
                double acc_l = 0.0;
                double acc_r = 0.0;
                for (size_t k = 0; k < TP_FIR_TAPS_PER_PHASE; ++k) {
                    const double h_k = taps[k];
                    acc_l += h_k * hist_l[k];
                    acc_r += h_k * hist_r[k];
                }
                peak = std::max({peak, std::abs(acc_l), std::abs(acc_r)});
            }

            fir_write_idx_ = (w == 0) ? (TP_FIR_TAPS_PER_PHASE - 1) : (w - 1);
        }

        const double g_target = std::min(1.0, c_lin_ / peak);

        // 4. Stadio 1: Min-Filter su finestra NL tramite Monotonic Deque O(1)
        const size_t current_sample = sample_count_++;

        while (deque_count_ > 0) {
            const size_t back_idx = (deque_tail_ + DEQUE_CAP - 1) % DEQUE_CAP;
            if (deque_buf_[back_idx].val >= g_target) {
                deque_tail_ = back_idx;
                --deque_count_;
            } else {
                break;
            }
        }

        push_deque(current_sample, g_target);

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

        // 6. Stadio 3: Boxcar Moving Average (136 campioni in TruePeak4x, 144 in LegacySample1x)
        const double old_boxcar_val = boxcar_buf_[boxcar_idx_];
        boxcar_buf_[boxcar_idx_] = g_rel_;
        boxcar_sum_ += (g_rel_ - old_boxcar_val);
        boxcar_idx_ = (boxcar_idx_ + 1) % active_boxcar_len_;

        const double g_smooth = boxcar_sum_ / static_cast<double>(active_boxcar_len_);

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

    void apply_mode_defaults(LimiterDetectorMode mode) noexcept {
        detector_mode_ = mode;
        if (mode == LimiterDetectorMode::LegacySample1x) {
            active_boxcar_len_ = LOOKAHEAD;
            set_ceiling_db(CEILING_DB);
        } else {
            active_boxcar_len_ = TP_BOXCAR_SAMPLES;
            set_ceiling_db(INTERNAL_CEILING_DBTP);
        }
    }

    void push_deque(size_t sample_idx, double val) noexcept {
        deque_buf_[deque_tail_] = {sample_idx, val};
        deque_tail_ = (deque_tail_ + 1) % DEQUE_CAP;
        ++deque_count_;
    }

    double sample_rate_ = DEFAULT_SAMPLE_RATE;
    double push_db_ = 0.0;
    double push_lin_ = 1.0;
    LimiterDetectorMode detector_mode_ = LimiterDetectorMode::TruePeak4x;
    double ceiling_db_ = INTERNAL_CEILING_DBTP;
    double c_lin_ = std::pow(10.0, INTERNAL_CEILING_DBTP / 20.0);
    size_t active_boxcar_len_ = TP_BOXCAR_SAMPLES;
    double alpha_rel_ = 0.0;

    // Buffer audio di ritardo Lookahead (144 campioni)
    std::array<double, LOOKAHEAD> delay_l_{};
    std::array<double, LOOKAHEAD> delay_r_{};
    size_t delay_write_idx_ = 0;

    // Ring buffer doppio per convoluzione FIR polifase senza modulo nel loop
    std::array<double, 2 * TP_FIR_TAPS_PER_PHASE> fir_hist_l_{};
    std::array<double, 2 * TP_FIR_TAPS_PER_PHASE> fir_hist_r_{};
    size_t fir_write_idx_ = TP_FIR_TAPS_PER_PHASE - 1;

    // Buffer Boxcar Moving Average (max LOOKAHEAD campioni)
    std::array<double, LOOKAHEAD> boxcar_buf_{};
    size_t boxcar_idx_ = 0;
    double boxcar_sum_ = static_cast<double>(TP_BOXCAR_SAMPLES);

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
