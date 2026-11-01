#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include <stdexcept>
#include "core_dsp/Constants.hpp"
#include "core_dsp/ParameterSpec.hpp"
#include "core_dsp/BiquadCascade.hpp"
#include "core_dsp/LinkwitzRiley4.hpp"
#include "core_dsp/MidSideImager.hpp"
#include "core_dsp/HarmonicSaturator.hpp"
#include "core_dsp/MultibandComp.hpp"
#include "core_dsp/LookaheadLimiter.hpp"
#include "core_dsp/MasteringChain.hpp"

namespace py = pybind11;
using namespace core_dsp;

std::vector<double> py_map_z_to_theta(const std::vector<double>& z) {
    if (z.size() != NUM_PARAMS) {
        throw std::invalid_argument("Input z must have length " + std::to_string(NUM_PARAMS));
    }
    std::array<double, NUM_PARAMS> z_arr;
    std::copy_n(z.begin(), NUM_PARAMS, z_arr.begin());

    auto theta_arr = map_z_to_theta(z_arr);
    return std::vector<double>(theta_arr.begin(), theta_arr.end());
}

std::vector<double> py_map_theta_to_z(const std::vector<double>& theta) {
    if (theta.size() != NUM_PARAMS) {
        throw std::invalid_argument("Input theta must have length " + std::to_string(NUM_PARAMS));
    }
    std::array<double, NUM_PARAMS> theta_arr;
    std::copy_n(theta.begin(), NUM_PARAMS, theta_arr.begin());

    auto z_arr = map_theta_to_z(theta_arr);
    return std::vector<double>(z_arr.begin(), z_arr.end());
}

py::list py_get_param_specs() {
    py::list specs;
    for (size_t i = 0; i < NUM_PARAMS; ++i) {
        const auto& p = PARAM_SPECS[i];
        specs.append(py::make_tuple(
            std::string(p.name),
            p.min_val,
            p.max_val,
            p.scale == ScaleType::Exponential,
            std::string(p.unit)
        ));
    }
    return specs;
}

std::vector<double> py_get_fir_halfband_coeffs() {
    return std::vector<double>(FIR_HALFBAND_COEFFS.begin(), FIR_HALFBAND_COEFFS.end());
}

// Wrapper Python per BiquadCascade6Band
class PyBiquadCascade6Band {
public:
    PyBiquadCascade6Band() = default;

    void reset() {
        cascade_.reset();
    }

    void set_parameters(const std::vector<double>& f0,
                        const std::vector<double>& gain_db,
                        const std::vector<double>& q,
                        double sample_rate = DEFAULT_SAMPLE_RATE) {
        if (f0.size() != 6 || gain_db.size() != 6 || q.size() != 6) {
            throw std::invalid_argument("EQ parameters must each have length 6");
        }
        std::array<double, 6> f0_arr, gain_arr, q_arr;
        std::copy_n(f0.begin(), 6, f0_arr.begin());
        std::copy_n(gain_db.begin(), 6, gain_arr.begin());
        std::copy_n(q.begin(), 6, q_arr.begin());

        cascade_.set_parameters(f0_arr, gain_arr, q_arr, sample_rate);
    }

    py::array_t<float> process(py::array_t<float> input) {
        auto buf = input.request();
        if (buf.ndim != 2 || buf.shape[0] != 2) {
            throw std::invalid_argument("Input must have shape [2, num_samples]");
        }
        size_t num_samples = static_cast<size_t>(buf.shape[1]);

        auto result = py::array_t<float>({static_cast<py::ssize_t>(2), static_cast<py::ssize_t>(num_samples)});
        auto res_buf = result.request();

        const float* in_ptr = static_cast<const float*>(buf.ptr);
        const float* in_l = in_ptr;
        const float* in_r = in_ptr + num_samples;

        float* out_ptr = static_cast<float*>(res_buf.ptr);
        float* out_l = out_ptr;
        float* out_r = out_ptr + num_samples;

        cascade_.process(in_l, in_r, out_l, out_r, num_samples);
        return result;
    }

private:
    BiquadCascade6Band cascade_;
};

// Wrapper Python per LinkwitzRiley4_3Way
class PyLinkwitzRiley4_3Way {
public:
    PyLinkwitzRiley4_3Way() = default;

    void reset() {
        crossover_.reset();
    }

    void set_crossover_frequencies(double f_low, double f_high, double sample_rate = DEFAULT_SAMPLE_RATE) {
        crossover_.set_crossover_frequencies(f_low, f_high, sample_rate);
    }

    py::tuple process(py::array_t<float> input) {
        auto buf = input.request();
        if (buf.ndim != 2 || buf.shape[0] != 2) {
            throw std::invalid_argument("Input must have shape [2, num_samples]");
        }
        size_t num_samples = static_cast<size_t>(buf.shape[1]);

        auto res_low = py::array_t<float>({static_cast<py::ssize_t>(2), static_cast<py::ssize_t>(num_samples)});
        auto res_mid = py::array_t<float>({static_cast<py::ssize_t>(2), static_cast<py::ssize_t>(num_samples)});
        auto res_high = py::array_t<float>({static_cast<py::ssize_t>(2), static_cast<py::ssize_t>(num_samples)});

        const float* in_ptr = static_cast<const float*>(buf.ptr);
        const float* in_l = in_ptr;
        const float* in_r = in_ptr + num_samples;

        float* low_ptr = static_cast<float*>(res_low.request().ptr);
        float* mid_ptr = static_cast<float*>(res_mid.request().ptr);
        float* high_ptr = static_cast<float*>(res_high.request().ptr);

        crossover_.process(in_l, in_r,
                           low_ptr, low_ptr + num_samples,
                           mid_ptr, mid_ptr + num_samples,
                           high_ptr, high_ptr + num_samples,
                           num_samples);

        return py::make_tuple(res_low, res_mid, res_high);
    }

private:
    LinkwitzRiley4_3Way crossover_;
};

// Wrapper Python per MidSideImager
class PyMidSideImager {
public:
    static py::array_t<float> process_band(py::array_t<float> input, double width) {
        auto buf = input.request();
        if (buf.ndim != 2 || buf.shape[0] != 2) {
            throw std::invalid_argument("Input must have shape [2, num_samples]");
        }
        size_t num_samples = static_cast<size_t>(buf.shape[1]);

        auto result = py::array_t<float>({static_cast<py::ssize_t>(2), static_cast<py::ssize_t>(num_samples)});
        const float* in_ptr = static_cast<const float*>(buf.ptr);
        float* out_ptr = static_cast<float*>(result.request().ptr);

        MidSideImager::process_band(in_ptr, in_ptr + num_samples,
                                    out_ptr, out_ptr + num_samples,
                                    num_samples, width);
        return result;
    }

    static py::array_t<float> process_3band_and_sum(
        py::array_t<float> low, py::array_t<float> mid, py::array_t<float> high,
        double width_low, double width_mid, double width_high
    ) {
        auto buf_low = low.request();
        auto buf_mid = mid.request();
        auto buf_high = high.request();
        size_t num_samples = static_cast<size_t>(buf_low.shape[1]);

        auto result = py::array_t<float>({static_cast<py::ssize_t>(2), static_cast<py::ssize_t>(num_samples)});
        float* out_ptr = static_cast<float*>(result.request().ptr);

        const float* l_ptr = static_cast<const float*>(buf_low.ptr);
        const float* m_ptr = static_cast<const float*>(buf_mid.ptr);
        const float* h_ptr = static_cast<const float*>(buf_high.ptr);

        MidSideImager::process_3band_and_sum(
            l_ptr, l_ptr + num_samples,
            m_ptr, m_ptr + num_samples,
            h_ptr, h_ptr + num_samples,
            out_ptr, out_ptr + num_samples,
            num_samples, width_low, width_mid, width_high
        );
        return result;
    }
};

// Wrapper Python per HarmonicSaturator
class PyHarmonicSaturator {
public:
    PyHarmonicSaturator() = default;

    void reset() {
        saturator_.reset();
    }

    void set_parameters(double drive_db, double mix) {
        saturator_.set_parameters(drive_db, mix);
    }

    py::array_t<float> process(py::array_t<float> input) {
        auto buf = input.request();
        if (buf.ndim != 2 || buf.shape[0] != 2) {
            throw std::invalid_argument("Input must have shape [2, num_samples]");
        }
        size_t num_samples = static_cast<size_t>(buf.shape[1]);

        auto result = py::array_t<float>({static_cast<py::ssize_t>(2), static_cast<py::ssize_t>(num_samples)});
        const float* in_ptr = static_cast<const float*>(buf.ptr);
        float* out_ptr = static_cast<float*>(result.request().ptr);

        saturator_.process(in_ptr, in_ptr + num_samples,
                           out_ptr, out_ptr + num_samples,
                           num_samples);
        return result;
    }

private:
    HarmonicSaturator saturator_;
};

// Wrapper Python per SingleBandComp
class PySingleBandComp {
public:
    PySingleBandComp() = default;

    void reset() {
        comp_.reset();
    }

    void set_parameters(double threshold_db, double ratio, double attack_ms,
                        double release_ms, double makeup_db, double knee_db = 6.0,
                        double sample_rate = DEFAULT_SAMPLE_RATE) {
        CompBandParams p;
        p.threshold_db = threshold_db;
        p.ratio = ratio;
        p.attack_ms = attack_ms;
        p.release_ms = release_ms;
        p.makeup_db = makeup_db;
        p.knee_db = knee_db;
        comp_.set_parameters(p, sample_rate);
    }

    py::array_t<float> process(py::array_t<float> input) {
        auto buf = input.request();
        if (buf.ndim != 2 || buf.shape[0] != 2) {
            throw std::invalid_argument("Input must have shape [2, num_samples]");
        }
        size_t num_samples = static_cast<size_t>(buf.shape[1]);

        auto result = py::array_t<float>({static_cast<py::ssize_t>(2), static_cast<py::ssize_t>(num_samples)});
        const float* in_ptr = static_cast<const float*>(buf.ptr);
        float* out_ptr = static_cast<float*>(result.request().ptr);

        const float* in_l = in_ptr;
        const float* in_r = in_ptr + num_samples;
        float* out_l = out_ptr;
        float* out_r = out_ptr + num_samples;

        for (size_t n = 0; n < num_samples; ++n) {
            double ol = 0.0, or_ = 0.0;
            comp_.process_sample(static_cast<double>(in_l[n]),
                                 static_cast<double>(in_r[n]),
                                 ol, or_);
            out_l[n] = static_cast<float>(ol);
            out_r[n] = static_cast<float>(or_);
        }
        return result;
    }

private:
    SingleBandComp comp_;
};

// Wrapper Python per MultibandCompressor3Band
class PyMultibandCompressor3Band {
public:
    PyMultibandCompressor3Band() = default;

    void reset() {
        mb_comp_.reset();
    }

    void set_parameters(
        const std::vector<double>& low_p,
        const std::vector<double>& mid_p,
        const std::vector<double>& high_p,
        double sample_rate = DEFAULT_SAMPLE_RATE
    ) {
        if (low_p.size() != 5 || mid_p.size() != 5 || high_p.size() != 5) {
            throw std::invalid_argument("Band parameters must each have 5 values (thresh, ratio, att, rel, makeup)");
        }
        CompBandParams p_l{low_p[0], low_p[1], low_p[2], low_p[3], low_p[4], 6.0};
        CompBandParams p_m{mid_p[0], mid_p[1], mid_p[2], mid_p[3], mid_p[4], 6.0};
        CompBandParams p_h{high_p[0], high_p[1], high_p[2], high_p[3], high_p[4], 6.0};

        mb_comp_.set_parameters(p_l, p_m, p_h, sample_rate);
    }

    py::tuple process(
        py::array_t<float> low, py::array_t<float> mid, py::array_t<float> high
    ) {
        auto b_low = low.request();
        auto b_mid = mid.request();
        auto b_high = high.request();
        size_t num_samples = static_cast<size_t>(b_low.shape[1]);

        auto res_low = py::array_t<float>({static_cast<py::ssize_t>(2), static_cast<py::ssize_t>(num_samples)});
        auto res_mid = py::array_t<float>({static_cast<py::ssize_t>(2), static_cast<py::ssize_t>(num_samples)});
        auto res_high = py::array_t<float>({static_cast<py::ssize_t>(2), static_cast<py::ssize_t>(num_samples)});

        const float* in_low = static_cast<const float*>(b_low.ptr);
        const float* in_mid = static_cast<const float*>(b_mid.ptr);
        const float* in_high = static_cast<const float*>(b_high.ptr);

        float* out_low = static_cast<float*>(res_low.request().ptr);
        float* out_mid = static_cast<float*>(res_mid.request().ptr);
        float* out_high = static_cast<float*>(res_high.request().ptr);

        mb_comp_.process(
            in_low, in_low + num_samples,
            in_mid, in_mid + num_samples,
            in_high, in_high + num_samples,
            out_low, out_low + num_samples,
            out_mid, out_mid + num_samples,
            out_high, out_high + num_samples,
            num_samples
        );

        return py::make_tuple(res_low, res_mid, res_high);
    }

private:
    MultibandCompressor3Band mb_comp_;
};

// Wrapper Python per LookaheadLimiter
class PyLookaheadLimiter {
public:
    PyLookaheadLimiter(double sample_rate = DEFAULT_SAMPLE_RATE)
        : limiter_(sample_rate) {}

    void reset() {
        limiter_.reset();
    }

    void set_sample_rate(double sample_rate) {
        limiter_.set_sample_rate(sample_rate);
    }

    void set_push_gain_db(double push_db) {
        limiter_.set_push_gain_db(push_db);
    }

    double get_push_gain_db() const {
        return limiter_.get_push_gain_db();
    }

    py::array_t<float> process(py::array_t<float> input) {
        auto buf = input.request();
        if (buf.ndim != 2 || buf.shape[0] != 2) {
            throw std::invalid_argument("Input must have shape [2, num_samples]");
        }
        size_t num_samples = static_cast<size_t>(buf.shape[1]);

        auto result = py::array_t<float>({static_cast<py::ssize_t>(2), static_cast<py::ssize_t>(num_samples)});
        const float* in_ptr = static_cast<const float*>(buf.ptr);
        float* out_ptr = static_cast<float*>(result.request().ptr);

        limiter_.process(in_ptr, in_ptr + num_samples,
                         out_ptr, out_ptr + num_samples,
                         num_samples);
        return result;
    }

private:
    LookaheadLimiter limiter_;
};

// Wrapper Python per MasteringChain
class PyMasteringChain {
public:
    PyMasteringChain(double sample_rate = DEFAULT_SAMPLE_RATE)
        : chain_(sample_rate) {}

    void reset() {
        chain_.reset();
    }

    void set_sample_rate(double sample_rate) {
        chain_.set_sample_rate(sample_rate);
    }

    void set_parameters_theta(const std::vector<double>& theta) {
        chain_.set_parameters_theta(theta);
    }

    void set_parameters_z(const std::vector<double>& z) {
        chain_.set_parameters_z(z);
    }

    void set_bypass(bool bypass) {
        chain_.set_bypass(bypass);
    }

    bool is_bypass() const {
        return chain_.is_bypass();
    }

    size_t get_latency_samples() const {
        return chain_.get_latency_samples();
    }

    py::array_t<float> process(py::array_t<float> input) {
        auto buf = input.request();
        if (buf.ndim != 2 || buf.shape[0] != 2) {
            throw std::invalid_argument("Input must have shape [2, num_samples]");
        }
        size_t num_samples = static_cast<size_t>(buf.shape[1]);

        auto result = py::array_t<float>({static_cast<py::ssize_t>(2), static_cast<py::ssize_t>(num_samples)});
        const float* in_ptr = static_cast<const float*>(buf.ptr);
        float* out_ptr = static_cast<float*>(result.request().ptr);

        chain_.process(in_ptr, in_ptr + num_samples,
                       out_ptr, out_ptr + num_samples,
                       num_samples);
        return result;
    }

private:
    MasteringChain chain_;
};

PYBIND11_MODULE(py_core_dsp, m) {
    m.doc() = "C++20 Core DSP Engine and Python Bindings for DDSP AutoMastering";

    m.attr("NUM_PARAMS") = NUM_PARAMS;
    m.attr("DEFAULT_SAMPLE_RATE") = DEFAULT_SAMPLE_RATE;
    m.attr("TOTAL_CHAIN_LATENCY_SAMPLES") = TOTAL_CHAIN_LATENCY_SAMPLES;
    m.attr("FIR_HALFBAND_LATENCY") = FIR_HALFBAND_LATENCY;
    m.attr("LIMITER_LOOKAHEAD_SAMPLES") = LIMITER_LOOKAHEAD_SAMPLES;

    m.def("map_z_to_theta", &py_map_z_to_theta, "Converte z in (0, 1) nello spazio parametri fisici theta");
    m.def("map_theta_to_z", &py_map_theta_to_z, "Converte theta nello spazio normalizzato z");
    m.def("get_param_specs", &py_get_param_specs, "Restituisce le specifiche dei 41 parametri");
    m.def("get_fir_halfband_coeffs", &py_get_fir_halfband_coeffs, "Restituisce i 31 coefficienti FIR Half-Band");

    py::class_<PyBiquadCascade6Band>(m, "BiquadCascade6Band")
        .def(py::init<>())
        .def("reset", &PyBiquadCascade6Band::reset)
        .def("set_parameters", &PyBiquadCascade6Band::set_parameters,
             py::arg("f0"), py::arg("gain_db"), py::arg("q"), py::arg("sample_rate") = DEFAULT_SAMPLE_RATE)
        .def("process", &PyBiquadCascade6Band::process);

    py::class_<PyLinkwitzRiley4_3Way>(m, "LinkwitzRiley4_3Way")
        .def(py::init<>())
        .def("reset", &PyLinkwitzRiley4_3Way::reset)
        .def("set_crossover_frequencies", &PyLinkwitzRiley4_3Way::set_crossover_frequencies,
             py::arg("f_low"), py::arg("f_high"), py::arg("sample_rate") = DEFAULT_SAMPLE_RATE)
        .def("process", &PyLinkwitzRiley4_3Way::process);

    py::class_<PyMidSideImager>(m, "MidSideImager")
        .def_static("process_band", &PyMidSideImager::process_band,
                    py::arg("input"), py::arg("width"))
        .def_static("process_3band_and_sum", &PyMidSideImager::process_3band_and_sum,
                    py::arg("low"), py::arg("mid"), py::arg("high"),
                    py::arg("width_low"), py::arg("width_mid"), py::arg("width_high"));

    py::class_<PyHarmonicSaturator>(m, "HarmonicSaturator")
        .def(py::init<>())
        .def("reset", &PyHarmonicSaturator::reset)
        .def("set_parameters", &PyHarmonicSaturator::set_parameters,
             py::arg("drive_db"), py::arg("mix"))
        .def("process", &PyHarmonicSaturator::process);

    py::class_<PySingleBandComp>(m, "SingleBandComp")
        .def(py::init<>())
        .def("reset", &PySingleBandComp::reset)
        .def("set_parameters", &PySingleBandComp::set_parameters,
             py::arg("threshold_db"), py::arg("ratio"), py::arg("attack_ms"),
             py::arg("release_ms"), py::arg("makeup_db"), py::arg("knee_db") = 6.0,
             py::arg("sample_rate") = DEFAULT_SAMPLE_RATE)
        .def("process", &PySingleBandComp::process);

    py::class_<PyMultibandCompressor3Band>(m, "MultibandCompressor3Band")
        .def(py::init<>())
        .def("reset", &PyMultibandCompressor3Band::reset)
        .def("set_parameters", &PyMultibandCompressor3Band::set_parameters,
             py::arg("low_p"), py::arg("mid_p"), py::arg("high_p"),
             py::arg("sample_rate") = DEFAULT_SAMPLE_RATE)
        .def("process", &PyMultibandCompressor3Band::process);

    py::class_<PyLookaheadLimiter>(m, "LookaheadLimiter")
        .def(py::init<double>(), py::arg("sample_rate") = DEFAULT_SAMPLE_RATE)
        .def("reset", &PyLookaheadLimiter::reset)
        .def("set_sample_rate", &PyLookaheadLimiter::set_sample_rate, py::arg("sample_rate"))
        .def("set_push_gain_db", &PyLookaheadLimiter::set_push_gain_db, py::arg("push_db"))
        .def("get_push_gain_db", &PyLookaheadLimiter::get_push_gain_db)
        .def("process", &PyLookaheadLimiter::process);

    py::class_<PyMasteringChain>(m, "MasteringChain")
        .def(py::init<double>(), py::arg("sample_rate") = DEFAULT_SAMPLE_RATE)
        .def("reset", &PyMasteringChain::reset)
        .def("set_sample_rate", &PyMasteringChain::set_sample_rate, py::arg("sample_rate"))
        .def("set_parameters_theta", &PyMasteringChain::set_parameters_theta, py::arg("theta"))
        .def("set_parameters_z", &PyMasteringChain::set_parameters_z, py::arg("z"))
        .def("get_latency_samples", &PyMasteringChain::get_latency_samples)
        .def("set_bypass", &PyMasteringChain::set_bypass, py::arg("bypass"))
        .def("is_bypass", &PyMasteringChain::is_bypass)
        .def("process", &PyMasteringChain::process);

    // Fast scans per Autograd Function del Compressore
    m.def("fast_forward_ballistics", [](py::array_t<double> gc, double alpha_a, double alpha_r) {
        auto buf = gc.request();
        if (buf.ndim != 2) {
            throw std::invalid_argument("gc must have shape [B, N]");
        }
        size_t B = static_cast<size_t>(buf.shape[0]);
        size_t N = static_cast<size_t>(buf.shape[1]);

        auto yg = py::array_t<double>({static_cast<py::ssize_t>(B), static_cast<py::ssize_t>(N)});
        auto mask = py::array_t<bool>({static_cast<py::ssize_t>(B), static_cast<py::ssize_t>(N)});

        const double* gc_ptr = static_cast<const double*>(buf.ptr);
        double* yg_ptr = static_cast<double*>(yg.request().ptr);
        bool* mask_ptr = static_cast<bool*>(mask.request().ptr);

        for (size_t b = 0; b < B; ++b) {
            const double* gc_b = gc_ptr + b * N;
            double* yg_b = yg_ptr + b * N;
            bool* mask_b = mask_ptr + b * N;

            double yg_prev = 0.0;
            for (size_t n = 0; n < N; ++n) {
                double gc_n = gc_b[n];
                bool m_n = gc_n > yg_prev;
                mask_b[n] = m_n;
                double alpha = m_n ? alpha_a : alpha_r;
                double yg_curr = alpha * yg_prev + (1.0 - alpha) * gc_n;
                yg_b[n] = yg_curr;
                yg_prev = yg_curr;
            }
        }
        return py::make_tuple(yg, mask);
    }, "Scansione ricorsiva ballistics forward ultra-veloce",
       py::arg("gc"), py::arg("alpha_a"), py::arg("alpha_r"));

    m.def("fast_backward_psi", [](py::array_t<double> d_yg_direct, py::array_t<bool> mask_attack, double alpha_a, double alpha_r) {
        auto buf_dyg = d_yg_direct.request();
        auto buf_mask = mask_attack.request();
        if (buf_dyg.ndim != 2) {
            throw std::invalid_argument("d_yg_direct must have shape [B, N]");
        }
        size_t B = static_cast<size_t>(buf_dyg.shape[0]);
        size_t N = static_cast<size_t>(buf_dyg.shape[1]);

        auto psi = py::array_t<double>({static_cast<py::ssize_t>(B), static_cast<py::ssize_t>(N + 1)});
        const double* dyg_ptr = static_cast<const double*>(buf_dyg.ptr);
        const bool* mask_ptr = static_cast<const bool*>(buf_mask.ptr);
        double* psi_ptr = static_cast<double*>(psi.request().ptr);

        for (size_t b = 0; b < B; ++b) {
            const double* dyg_b = dyg_ptr + b * N;
            const bool* mask_b = mask_ptr + b * N;
            double* psi_b = psi_ptr + b * (N + 1);

            psi_b[N] = 0.0;
            psi_b[N - 1] = dyg_b[N - 1];
            for (py::ssize_t n = static_cast<py::ssize_t>(N) - 2; n >= 0; --n) {
                double alpha_next = mask_b[n + 1] ? alpha_a : alpha_r;
                psi_b[n] = dyg_b[n] + alpha_next * psi_b[n + 1];
            }
        }
        return psi;
    }, "Scansione ricorsiva all'indietro psi ultra-veloce",
       py::arg("d_yg_direct"), py::arg("mask_attack"), py::arg("alpha_a"), py::arg("alpha_r"));

    // Fast scans per Autograd Function del Limiter Lookahead Release Envelope
    m.def("fast_forward_limiter_envelope", [](py::array_t<double> g_min, double alpha_rel) {
        auto buf = g_min.request();
        if (buf.ndim != 2) {
            throw std::invalid_argument("g_min must have shape [B, N]");
        }
        size_t B = static_cast<size_t>(buf.shape[0]);
        size_t N = static_cast<size_t>(buf.shape[1]);

        auto g_rel = py::array_t<double>({static_cast<py::ssize_t>(B), static_cast<py::ssize_t>(N)});
        auto mask = py::array_t<bool>({static_cast<py::ssize_t>(B), static_cast<py::ssize_t>(N)});

        const double* gm_ptr = static_cast<const double*>(buf.ptr);
        double* grel_ptr = static_cast<double*>(g_rel.request().ptr);
        bool* mask_ptr = static_cast<bool*>(mask.request().ptr);

        for (size_t b = 0; b < B; ++b) {
            const double* gm_b = gm_ptr + b * N;
            double* grel_b = grel_ptr + b * N;
            bool* mask_b = mask_ptr + b * N;

            double g_prev = 1.0;
            for (size_t n = 0; n < N; ++n) {
                double gm = gm_b[n];
                bool is_release = gm > g_prev;
                mask_b[n] = is_release;
                double g_curr = is_release ? (alpha_rel * g_prev + (1.0 - alpha_rel) * gm) : gm;
                grel_b[n] = g_curr;
                g_prev = g_curr;
            }
        }
        return py::make_tuple(g_rel, mask);
    }, "Scansione forward limiter release envelope in C++",
       py::arg("g_min"), py::arg("alpha_rel"));

    m.def("fast_backward_limiter_envelope", [](py::array_t<double> d_grel, py::array_t<bool> mask_release, double alpha_rel) {
        auto buf_dg = d_grel.request();
        auto buf_mask = mask_release.request();
        if (buf_dg.ndim != 2) {
            throw std::invalid_argument("d_grel must have shape [B, N]");
        }
        size_t B = static_cast<size_t>(buf_dg.shape[0]);
        size_t N = static_cast<size_t>(buf_dg.shape[1]);

        auto d_gmin = py::array_t<double>({static_cast<py::ssize_t>(B), static_cast<py::ssize_t>(N)});

        const double* dg_ptr = static_cast<const double*>(buf_dg.ptr);
        const bool* mask_ptr = static_cast<const bool*>(buf_mask.ptr);
        double* dgmin_ptr = static_cast<double*>(d_gmin.request().ptr);

        for (size_t b = 0; b < B; ++b) {
            const double* dg_b = dg_ptr + b * N;
            const bool* mask_b = mask_ptr + b * N;
            double* dgmin_b = dgmin_ptr + b * N;

            double lambda_next = 0.0;
            for (py::ssize_t n = static_cast<py::ssize_t>(N) - 1; n >= 0; --n) {
                double lambda_curr = dg_b[n] + lambda_next;
                bool is_rel = mask_b[n];
                dgmin_b[n] = is_rel ? ((1.0 - alpha_rel) * lambda_curr) : lambda_curr;
                lambda_next = is_rel ? (alpha_rel * lambda_curr) : 0.0;
            }
        }
        return d_gmin;
    }, "Scansione backward limiter release envelope in C++",
       py::arg("d_grel"), py::arg("mask_release"), py::arg("alpha_rel"));
}
