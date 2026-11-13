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

// Costanti Limiter (Sprint 4 True Peak 4x + Legacy Compatibility)
inline constexpr size_t LIMITER_LOOKAHEAD_SAMPLES = 144; // 3.0 ms a 48 kHz
inline constexpr double LIMITER_CEILING_DBFS = -0.3;     // Legacy sample-peak ceiling / delivery contract
inline constexpr double LIMITER_INTERNAL_CEILING_DBTP = -0.50; // Internal DSP True Peak ceiling
inline constexpr double LIMITER_DELIVERY_CEILING_DBTP = -0.30; // Final PCM_24 delivery ceiling
inline constexpr size_t LIMITER_BOXCAR_SAMPLES = 136;    // Compensato per group delay FIR 4x (~5.875 / 6 campioni) senza release leak
inline constexpr double LIMITER_RELEASE_MS = 80.0;

// True Peak 4x Polyphase FIR Detector Constants
// Specifica: Prototipo Type-II simmetrico lineare WLS da 48 tap su griglia 4x (192 kHz nominali),
// passband [0, 22 kHz] con compensazione geometrica inter-sample 4x 1/cos(pi*f/(4*fs)) (f > 10 kHz),
// stopband [27.5 kHz, 96 kHz], normalizzato per singola fase a guadagno DC unitario (1.0).
// Nota normativa: filtro detector interno del progetto, distinto dal riferimento BS.1770 / Kaiser-64 esterno.
inline constexpr size_t TP_OVERSAMPLE_FACTOR = 4;
inline constexpr size_t TP_FIR_TAPS_PER_PHASE = 12;
inline constexpr size_t TP_FIR_PROTOTYPE_TAPS = TP_OVERSAMPLE_FACTOR * TP_FIR_TAPS_PER_PHASE; // 48
inline constexpr double TP_FIR_GROUP_DELAY_BASE_SAMPLES =
    static_cast<double>(TP_FIR_PROTOTYPE_TAPS - 1) / (2.0 * static_cast<double>(TP_OVERSAMPLE_FACTOR)); // 5.875
inline constexpr size_t TP_FIR_ALIGNED_SAMPLE_DELAY = 6; // ceil(5.875) campioni alla frequenza base

inline constexpr std::array<double, TP_FIR_PROTOTYPE_TAPS> TP_FIR_PROTOTYPE_COEFFS = {
    0.021101060308359995,   0.00033638122803119523, -0.02290233358771963,  -0.032003183981007287,
   -0.017567715780417031,   0.014601578655383312,    0.04401592198233905,   0.047373416913966925,
    0.016016879513805326,  -0.035378170875529531,   -0.073317895426556409, -0.065697780182872964,
   -0.006455635993611754,   0.075332650672725376,    0.12563247533387883,   0.097651474034789229,
   -0.012387342575628709,  -0.15327403357677791,    -0.23299736782631375,  -0.16527746191083259,
    0.077837917251192004,   0.44505196611893527,     0.81289882730160423,   1.0394083724022567,
    1.0394083724022567,     0.81289882730160401,     0.44505196611893533,   0.077837917251192004,
   -0.16527746191083259,   -0.23299736782631369,    -0.15327403357677793,  -0.012387342575628709,
    0.097651474034789229,   0.12563247533387881,     0.07533265067272539,  -0.006455635993611754,
   -0.065697780182872964,  -0.073317895426556395,   -0.035378170875529538,  0.016016879513805326,
    0.047373416913966925,   0.044015921982339043,    0.014601578655383316, -0.017567715780417031,
   -0.032003183981007287,  -0.022902333587719623,    0.00033638122803119529, 0.021101060308359995
};

// Latenza totale fissa della catena di mastering = Dsat + NL = 15 + 144 = 159 campioni
inline constexpr size_t TOTAL_CHAIN_LATENCY_SAMPLES = FIR_HALFBAND_LATENCY + LIMITER_LOOKAHEAD_SAMPLES;

} // namespace core_dsp
