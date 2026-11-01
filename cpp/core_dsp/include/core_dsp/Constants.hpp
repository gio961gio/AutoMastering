#pragma once

#include <cstddef>
#include <array>

namespace core_dsp {

// Frequenza di campionamento nominale (Hz)
inline constexpr double DEFAULT_SAMPLE_RATE = 48000.0;

// Costanti matematiche fondamentali
inline constexpr double PI = 3.141592653589793238462643383279502884;
inline constexpr double TWO_PI = 2.0 * PI;
inline constexpr double SQRT2 = 1.414213562373095048801688724209698079;
inline constexpr double INV_SQRT2 = 1.0 / SQRT2; // ~0.70710678118654752440
inline constexpr double LN10 = 2.302585092994045684017991454684364208;
inline constexpr double EPSILON = 1e-12;

// Coefficienti per saturatore e oversampling
inline constexpr size_t FIR_HALFBAND_TAPS = 31;
inline constexpr size_t FIR_HALFBAND_LATENCY = 15; // Campioni di ritardo a fs (30 a 2*fs)

// 31 Tap simmetrici FIR Half-Band (Kaiser window beta=8.5, stopband > 85 dB)
// Tutti i tap dispari diversi da 15 sono rigorosamente 0.0, h[15] = 0.5, somma = 1.0
inline constexpr std::array<double, FIR_HALFBAND_TAPS> FIR_HALFBAND_COEFFS = {
    -3.106101348136368259e-05,
     0.000000000000000000e+00,
     4.991001722697966693e-04,
     0.000000000000000000e+00,
    -2.328433331169654612e-03,
     0.000000000000000000e+00,
     7.254618576925952328e-03,
     0.000000000000000000e+00,
    -1.814484155598338616e-02,
     0.000000000000000000e+00,
     4.036219779069402658e-02,
     0.000000000000000000e+00,
    -9.030448489275624824e-02,
     0.000000000000000000e+00,
     3.126929042535008807e-01,
     5.000000000000000000e-01, // Centro
     3.126929042535008807e-01,
     0.000000000000000000e+00,
    -9.030448489275624824e-02,
     0.000000000000000000e+00,
     4.036219779069402658e-02,
     0.000000000000000000e+00,
    -1.814484155598338616e-02,
     0.000000000000000000e+00,
     7.254618576925952328e-03,
     0.000000000000000000e+00,
    -2.328433331169654612e-03,
     0.000000000000000000e+00,
     4.991001722697966693e-04,
     0.000000000000000000e+00,
    -3.106101348136368259e-05
};

// Costanti Limiter
inline constexpr size_t LIMITER_LOOKAHEAD_SAMPLES = 144; // 3.0 ms a 48 kHz
inline constexpr double LIMITER_CEILING_DBFS = -0.3;
inline constexpr double LIMITER_RELEASE_MS = 80.0;

// Latenza totale fissa della catena di mastering = Dsat + NL = 15 + 144 = 159 campioni
inline constexpr size_t TOTAL_CHAIN_LATENCY_SAMPLES = FIR_HALFBAND_LATENCY + LIMITER_LOOKAHEAD_SAMPLES;

} // namespace core_dsp
