#pragma once

#include <vector>
#include <array>
#include <span>
#include <memory>
#include "Constants.hpp"
#include "ParameterSpec.hpp"
#include "BiquadCascade.hpp"
#include "LinkwitzRiley4.hpp"
#include "MultibandComp.hpp"
#include "MidSideImager.hpp"
#include "HarmonicSaturator.hpp"
#include "LookaheadLimiter.hpp"

namespace core_dsp {

// Catena Completa di Mastering DDSP White-Box
class MasteringChain {
public:
    explicit MasteringChain(double sample_rate = DEFAULT_SAMPLE_RATE)
        : sample_rate_(sample_rate),
          limiter_(sample_rate)
    {
        reset();
    }

    void reset() noexcept {
        eq_.reset();
        crossover_.reset();
        mb_comp_.reset();
        saturator_.reset();
        limiter_.reset();
        bypass_delay_l_.fill(0.0f);
        bypass_delay_r_.fill(0.0f);
        bypass_write_idx_ = 0;
    }

    void set_bypass(bool bypass) noexcept {
        bypass_ = bypass;
    }

    [[nodiscard]] bool is_bypass() const noexcept {
        return bypass_;
    }

    void set_sample_rate(double sample_rate) noexcept {
        sample_rate_ = (sample_rate > 0.0) ? sample_rate : DEFAULT_SAMPLE_RATE;
        limiter_.set_sample_rate(sample_rate_);
    }

    // Pre-alloca i buffer interni ed imposta il sample rate (da chiamare su thread di setup)
    void prepare(double sample_rate, size_t max_block_size) {
        set_sample_rate(sample_rate);
        ensure_scratch_buffers(max_block_size);
        reset();
    }

    [[nodiscard]] double get_sample_rate() const noexcept {
        return sample_rate_;
    }

    [[nodiscard]] static constexpr size_t get_latency_samples() noexcept {
        return TOTAL_CHAIN_LATENCY_SAMPLES; // 159 campioni
    }

    // Configura tutti i 41 parametri a partire dal vettore fisico theta
    void set_parameters_theta(std::span<const double> theta) {
        if (theta.size() != NUM_PARAMS) {
            throw std::invalid_argument("theta must have exactly 41 parameters");
        }

        // 1. EQ 6-Band (theta[0..17])
        std::array<double, 6> f0{};
        std::array<double, 6> gain_db{};
        std::array<double, 6> q{};
        for (size_t b = 0; b < 6; ++b) {
            f0[b] = theta[3 * b + 0];
            gain_db[b] = theta[3 * b + 1];
            q[b] = theta[3 * b + 2];
        }
        eq_.set_parameters(f0, gain_db, q, sample_rate_);

        // 2. Crossover LR4 (theta[18..19])
        crossover_f_low_ = theta[18];
        crossover_f_high_ = theta[19];
        crossover_.set_crossover_frequencies(crossover_f_low_, crossover_f_high_, sample_rate_);

        // 3. Multiband Compressor (theta[20..34])
        CompBandParams p_l{theta[20], theta[21], theta[22], theta[23], theta[24], 6.0};
        CompBandParams p_m{theta[25], theta[26], theta[27], theta[28], theta[29], 6.0};
        CompBandParams p_h{theta[30], theta[31], theta[32], theta[33], theta[34], 6.0};
        mb_comp_.set_parameters(p_l, p_m, p_h, sample_rate_);

        // 4. Stereo Imager Widths (theta[35..37])
        imager_w_low_ = theta[35];
        imager_w_mid_ = theta[36];
        imager_w_high_ = theta[37];

        // 5. Saturator (theta[38..39])
        saturator_.set_parameters(theta[38], theta[39]);

        // 6. Limiter (theta[40])
        limiter_.set_push_gain_db(theta[40]);
    }

    void set_parameters_theta(const std::vector<double>& theta) {
        set_parameters_theta(std::span<const double>(theta.data(), theta.size()));
    }

    // Configura a partire dal vettore normalizzato z in (0, 1)
    void set_parameters_z(std::span<const double> z) {
        if (z.size() != NUM_PARAMS) {
            throw std::invalid_argument("z must have exactly 41 parameters");
        }
        std::array<double, NUM_PARAMS> z_arr;
        std::copy_n(z.begin(), NUM_PARAMS, z_arr.begin());
        const auto theta = map_z_to_theta(z_arr);
        set_parameters_theta(theta);
    }

    void set_parameters_z(const std::vector<double>& z) {
        set_parameters_z(std::span<const double>(z.data(), z.size()));
    }

    // Elabora un blocco stereo contiguo attraverso tutti i 6 stadi (o bypass a latenza compensata)
    void process(
        const float* in_l, const float* in_r,
        float* out_l, float* out_r,
        size_t num_samples
    ) {
        if (bypass_) {
            for (size_t i = 0; i < num_samples; ++i) {
                float delayed_l = bypass_delay_l_[bypass_write_idx_];
                float delayed_r = bypass_delay_r_[bypass_write_idx_];
                bypass_delay_l_[bypass_write_idx_] = in_l[i];
                bypass_delay_r_[bypass_write_idx_] = in_r[i];
                bypass_write_idx_ = (bypass_write_idx_ + 1) % TOTAL_CHAIN_LATENCY_SAMPLES;
                out_l[i] = delayed_l;
                out_r[i] = delayed_r;
            }
            return;
        }

        ensure_scratch_buffers(num_samples);

        // 1. Parametric EQ 6-Band
        eq_.process(in_l, in_r,
                    buf_eq_l_.data(), buf_eq_r_.data(),
                    num_samples);

        // 2. Crossover LR4 (3 vie stereo)
        crossover_.process(buf_eq_l_.data(), buf_eq_r_.data(),
                           buf_low_l_.data(), buf_low_r_.data(),
                           buf_mid_l_.data(), buf_mid_r_.data(),
                           buf_high_l_.data(), buf_high_r_.data(),
                           num_samples);

        // 3. Multiband Compressor (3 bande parallele)
        mb_comp_.process(buf_low_l_.data(), buf_low_r_.data(),
                         buf_mid_l_.data(), buf_mid_r_.data(),
                         buf_high_l_.data(), buf_high_r_.data(),
                         buf_comp_l_l_.data(), buf_comp_l_r_.data(),
                         buf_comp_m_l_.data(), buf_comp_m_r_.data(),
                         buf_comp_h_l_.data(), buf_comp_h_r_.data(),
                         num_samples);

        // 4. Stereo Imager & 3-Band Summing
        MidSideImager::process_3band_and_sum(
            buf_comp_l_l_.data(), buf_comp_l_r_.data(),
            buf_comp_m_l_.data(), buf_comp_m_r_.data(),
            buf_comp_h_l_.data(), buf_comp_h_r_.data(),
            buf_sum_l_.data(), buf_sum_r_.data(),
            num_samples,
            imager_w_low_, imager_w_mid_, imager_w_high_
        );

        // 5. Harmonic Saturator (2x oversampled FIR)
        saturator_.process(buf_sum_l_.data(), buf_sum_r_.data(),
                           buf_sat_l_.data(), buf_sat_r_.data(),
                           num_samples);

        // 6. Lookahead Brickwall Limiter (144 lookahead samples)
        limiter_.process(buf_sat_l_.data(), buf_sat_r_.data(),
                         out_l, out_r,
                         num_samples);
    }

private:
    void ensure_scratch_buffers(size_t num_samples) {
        if (buf_eq_l_.size() < num_samples) {
            buf_eq_l_.resize(num_samples);
            buf_eq_r_.resize(num_samples);
            buf_low_l_.resize(num_samples);
            buf_low_r_.resize(num_samples);
            buf_mid_l_.resize(num_samples);
            buf_mid_r_.resize(num_samples);
            buf_high_l_.resize(num_samples);
            buf_high_r_.resize(num_samples);
            buf_comp_l_l_.resize(num_samples);
            buf_comp_l_r_.resize(num_samples);
            buf_comp_m_l_.resize(num_samples);
            buf_comp_m_r_.resize(num_samples);
            buf_comp_h_l_.resize(num_samples);
            buf_comp_h_r_.resize(num_samples);
            buf_sum_l_.resize(num_samples);
            buf_sum_r_.resize(num_samples);
            buf_sat_l_.resize(num_samples);
            buf_sat_r_.resize(num_samples);
        }
    }

    double sample_rate_ = DEFAULT_SAMPLE_RATE;
    bool bypass_ = false;

    // Linea di ritardo per True Bypass a latenza compensata (159 campioni)
    std::array<float, TOTAL_CHAIN_LATENCY_SAMPLES> bypass_delay_l_{};
    std::array<float, TOTAL_CHAIN_LATENCY_SAMPLES> bypass_delay_r_{};
    size_t bypass_write_idx_ = 0;

    // Stadi DSP
    BiquadCascade6Band eq_;
    LinkwitzRiley4_3Way crossover_;
    MultibandCompressor3Band mb_comp_;
    HarmonicSaturator saturator_;
    LookaheadLimiter limiter_;

    // Parametri intermedi
    double crossover_f_low_ = 250.0;
    double crossover_f_high_ = 2500.0;
    double imager_w_low_ = 1.0;
    double imager_w_mid_ = 1.0;
    double imager_w_high_ = 1.0;

    // Scratch buffer riutilizzabili per evitare allocazioni in process
    std::vector<float> buf_eq_l_, buf_eq_r_;
    std::vector<float> buf_low_l_, buf_low_r_;
    std::vector<float> buf_mid_l_, buf_mid_r_;
    std::vector<float> buf_high_l_, buf_high_r_;
    std::vector<float> buf_comp_l_l_, buf_comp_l_r_;
    std::vector<float> buf_comp_m_l_, buf_comp_m_r_;
    std::vector<float> buf_comp_h_l_, buf_comp_h_r_;
    std::vector<float> buf_sum_l_, buf_sum_r_;
    std::vector<float> buf_sat_l_, buf_sat_r_;
};

} // namespace core_dsp
