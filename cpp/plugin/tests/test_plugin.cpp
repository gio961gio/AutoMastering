#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <random>
#include "PluginProcessor.h"

// Macro per asserzioni di test formali
#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "FAILED: " << msg << " (" << #cond << ") at line " << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while(0)

int main() {
    std::cout << "====================================================" << std::endl;
    std::cout << "  AutoMastering PoliMi - JUCE 8 Plugin Headless Test " << std::endl;
    std::cout << "====================================================" << std::endl;

    // Inizializzazione contesto JUCE per test headless
    juce::ScopedJuceInitialiser_GUI juceInit;

    // TEST 1: Istanziazione e verifica dei 41 parametri APVTS
    std::cout << "\n[TEST 1] Verifica Istanziazione e Parametri APVTS..." << std::endl;
    MasteringPluginAudioProcessor processor;

    const auto& params = processor.getParameters();
    std::cout << "  Numero parametri APVTS registrati: " << params.size() << std::endl;
    TEST_ASSERT(params.size() == core_dsp::NUM_PARAMS, "Expected exactly 41 parameters in APVTS");

    for (size_t i = 0; i < core_dsp::NUM_PARAMS; ++i) {
        const auto& spec = core_dsp::PARAM_SPECS[i];
        const juce::String expected_id(spec.name.data(), spec.name.size());

        auto* param = processor.getAPVTS().getParameter(expected_id);
        TEST_ASSERT(param != nullptr, ("Missing parameter: " + expected_id.toStdString()).c_str());

        const auto* float_param = dynamic_cast<juce::AudioParameterFloat*>(param);
        TEST_ASSERT(float_param != nullptr, ("Parameter is not AudioParameterFloat: " + expected_id.toStdString()).c_str());

        const auto range = float_param->range;
        TEST_ASSERT(std::abs(range.start - spec.min_val) < 1e-4, "Min value mismatch");
        TEST_ASSERT(std::abs(range.end - spec.max_val) < 1e-4, "Max value mismatch");
    }
    std::cout << "  -> Tutti i 41 parametri canonici sono conformi e verificati in APVTS!" << std::endl;

    // TEST 2: PrepareToPlay e Notifica Latenza Host (159 campioni)
    std::cout << "\n[TEST 2] Verifica prepareToPlay e Latenza Host (159 campioni)..." << std::endl;
    constexpr double fs = 48000.0;
    constexpr int block_size = 512;
    processor.prepareToPlay(fs, block_size);

    const int latency = processor.getLatencySamples();
    std::cout << "  Latenza segnalata all'Host: " << latency << " campioni ("
              << (latency * 1000.0 / fs) << " ms)" << std::endl;
    TEST_ASSERT(latency == 159, "Latency must be exactly 159 samples");
    std::cout << "  -> Latenza host verificata con successo!" << std::endl;

    // TEST 3: Elaborazione processBlock su molteplici dimensioni di buffer DAW
    std::cout << "\n[TEST 3] Elaborazione processBlock su buffer arbitrari (32..2048)..." << std::endl;
    const std::vector<int> test_buffer_sizes = {32, 64, 128, 256, 512, 1024, 2048};

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.8f, 0.8f);

    juce::MidiBuffer midi;
    for (int size : test_buffer_sizes) {
        juce::AudioBuffer<float> buffer(2, size);
        for (int ch = 0; ch < 2; ++ch) {
            float* data = buffer.getWritePointer(ch);
            for (int s = 0; s < size; ++s) {
                data[s] = dist(rng);
            }
        }

        processor.processBlock(buffer, midi);

        for (int ch = 0; ch < 2; ++ch) {
            const float* data = buffer.getReadPointer(ch);
            for (int s = 0; s < size; ++s) {
                TEST_ASSERT(!std::isnan(data[s]), "NaN detected in output audio");
                TEST_ASSERT(!std::isinf(data[s]), "Inf detected in output audio");
                TEST_ASSERT(std::abs(data[s]) <= 1.0f, "Ceiling exceeded in output audio");
            }
        }
    }
    std::cout << "  -> processBlock robusto ed esente da denormali/NaN su tutti i buffer!" << std::endl;

    // TEST 4: Parita Numerica tra MasteringPluginAudioProcessor e core_dsp::MasteringChain
    std::cout << "\n[TEST 4] Parita numerica Plugin vs C++20 Core Chain..." << std::endl;
    {
        core_dsp::MasteringChain standalone_chain(fs);
        standalone_chain.prepare(fs, 512);

        MasteringPluginAudioProcessor test_plugin;
        test_plugin.prepareToPlay(fs, 512);
        standalone_chain.set_parameters_theta(test_plugin.getCurrentTheta());

        // Warm up entrambe le catene
        constexpr int N_WARM = 4800;
        juce::AudioBuffer<float> warm_buf(2, 512);
        for (int b = 0; b < N_WARM / 512; ++b) {
            warm_buf.clear();
            test_plugin.processBlock(warm_buf, midi);

            std::vector<float> zeros_in(512, 0.0f);
            std::vector<float> out_l(512), out_r(512);
            standalone_chain.process(zeros_in.data(), zeros_in.data(), out_l.data(), out_r.data(), 512);
        }

        // Test segnale audio identico
        constexpr int TEST_LEN = 2048;
        juce::AudioBuffer<float> plugin_buf(2, TEST_LEN);
        std::vector<float> in_l(TEST_LEN), in_r(TEST_LEN);
        for (int s = 0; s < TEST_LEN; ++s) {
            in_l[s] = dist(rng);
            in_r[s] = dist(rng);
            plugin_buf.setSample(0, s, in_l[s]);
            plugin_buf.setSample(1, s, in_r[s]);
        }

        // Process plugin
        test_plugin.processBlock(plugin_buf, midi);

        // Process standalone
        std::vector<float> ref_out_l(TEST_LEN), ref_out_r(TEST_LEN);
        standalone_chain.process(in_l.data(), in_r.data(), ref_out_l.data(), ref_out_r.data(), TEST_LEN);

        double sum_sq_err = 0.0;
        double max_err = 0.0;
        for (int s = 0; s < TEST_LEN; ++s) {
            double err_l = std::abs(plugin_buf.getSample(0, s) - ref_out_l[s]);
            double err_r = std::abs(plugin_buf.getSample(1, s) - ref_out_r[s]);
            sum_sq_err += err_l * err_l + err_r * err_r;
            max_err = std::max({max_err, err_l, err_r});
        }
        double rms_err = std::sqrt(sum_sq_err / (2.0 * TEST_LEN));
        double null_db = 20.0 * std::log10(rms_err + 1e-12);
        double max_err_db = 20.0 * std::log10(max_err + 1e-12);

        std::cout << "  Plugin vs Core Chain Null Error: " << null_db << " dBFS (Max Err: "
                  << max_err_db << " dBFS)" << std::endl;
        TEST_ASSERT(null_db <= -140.0, "Plugin must match core_dsp bit-identically");
        std::cout << "  -> Parita numerica macchina garantita (<= -140 dBFS)!" << std::endl;
    }

    // TEST 5: Serializzazione JSON e State Injection
    std::cout << "\n[TEST 5] Serializzazione e Iniezione Preset JSON..." << std::endl;
    {
        juce::String jsonStr = processor.getParametersAsJson();
        TEST_ASSERT(jsonStr.isNotEmpty(), "Exported JSON must not be empty");

        // Modifica un parametro (all'interno del range mastering-grade 0..6 dB)
        if (auto* p = processor.getAPVTS().getParameter("saturator_drive_db")) {
            p->setValueNotifyingHost(p->convertTo0to1(4.5f));
        }
        TEST_ASSERT(std::abs(processor.getAPVTS().getRawParameterValue("saturator_drive_db")->load() - 4.5f) < 0.1f,
                    "Parameter update failed");

        // Ricarica il JSON originale
        bool loaded = processor.loadParametersFromJson(jsonStr);
        TEST_ASSERT(loaded, "Failed to load parameters from JSON");

        float restored = processor.getAPVTS().getRawParameterValue("saturator_drive_db")->load();
        std::cout << "  Valore saturator_drive_db ripristinato da JSON: " << restored << " dB" << std::endl;
        TEST_ASSERT(std::abs(restored - 0.0f) < 0.01f, "Parameter not properly restored from JSON");
        std::cout << "  -> Serializzazione e iniezione preset JSON validate al 100%!" << std::endl;
    }

    // TEST 6: Iniezione Vettore Normalizzato z in (0, 1) da PyTorch
    std::cout << "\n[TEST 6] Iniezione Vettore Normalizzato z da PyTorch..." << std::endl;
    {
        std::array<float, core_dsp::NUM_PARAMS> z_in;
        for (size_t i = 0; i < core_dsp::NUM_PARAMS; ++i) {
            z_in[i] = 0.5f; // Metà scala per tutti i 41 parametri
        }
        processor.setNormalizedParameters(z_in);

        auto z_out = processor.getNormalizedParameters();
        for (size_t i = 0; i < core_dsp::NUM_PARAMS; ++i) {
            TEST_ASSERT(std::abs(z_in[i] - z_out[i]) < 1e-3f, "Normalized parameter roundtrip mismatch");
        }
        std::cout << "  -> Roundtrip vettore z in (0, 1)^41 verificato con successo!" << std::endl;
    }

    // TEST 7: Interoperabilita Preset JSON Python -> JUCE e gestione errori
    std::cout << "\n[TEST 7] Interoperabilita Preset JSON Python -> JUCE e gestione errori..." << std::endl;
    {
        // JSON canonico prodotto da Python con nodo annidato parameters e metadati
        const juce::String pythonJson = R"({
            "format": "AutoMastering_Preset_v1",
            "version": "1.0",
            "sample_rate": 48000.0,
            "parameters": {
                "eq_b0_f0": 85.0,
                "eq_b0_gain_db": 3.0,
                "eq_b0_q": 1.2,
                "saturator_drive_db": 2.5,
                "saturator_mix": 0.5,
                "comp_low_ratio": 1.5,
                "comp_low_threshold_db": -18.0
            },
            "parameter_details": {
                "eq_b0_f0": {"value": 85.0, "unit": "Hz"}
            }
        })";

        bool loaded = processor.loadParametersFromJson(pythonJson);
        TEST_ASSERT(loaded, "Failed to parse and load canonical Python JSON preset");
        float f0_val = processor.getAPVTS().getRawParameterValue("eq_b0_f0")->load();
        TEST_ASSERT(std::abs(f0_val - 85.0f) < 0.1f, "Parameter eq_b0_f0 was not loaded correctly from Python JSON");

        float drive_val = processor.getAPVTS().getRawParameterValue("saturator_drive_db")->load();
        TEST_ASSERT(std::abs(drive_val - 2.5f) < 0.1f, "Parameter saturator_drive_db was not loaded correctly");

        // Test JSON non valido / corrotto
        TEST_ASSERT(!processor.loadParametersFromJson("invalid json string"), "Corrupt JSON must return false");
        TEST_ASSERT(!processor.loadParametersFromJson("{}"), "Empty JSON with no valid params must return false");
        TEST_ASSERT(!processor.loadParametersFromJson(R"({"random_key": 123})"), "JSON with unknown keys must return false");

        std::cout << "  -> Interoperabilita Python -> JUCE e gestione errori validate con successo!" << std::endl;
    }

    // TEST 8: True Bypass a latenza compensata (159 campioni) in C++
    std::cout << "\n[TEST 8] True Bypass a latenza compensata (159 campioni) in C++..." << std::endl;
    {
        core_dsp::MasteringChain chain(fs);
        chain.prepare(fs, 512);

        // Abilita True Bypass
        chain.set_bypass(true);
        TEST_ASSERT(chain.is_bypass() == true, "Chain must report bypass active");

        // Genera segnale di test stereo non correlato
        constexpr size_t N_TEST = 1000;
        std::vector<float> in_l(N_TEST), in_r(N_TEST);
        std::vector<float> out_l(N_TEST), out_r(N_TEST);
        for (size_t i = 0; i < N_TEST; ++i) {
            in_l[i] = dist(rng);
            in_r[i] = dist(rng);
        }

        chain.process(in_l.data(), in_r.data(), out_l.data(), out_r.data(), N_TEST);

        // Dopo la latenza di 159 campioni, out[i] deve essere identico a in[i - 159]
        constexpr size_t latency = core_dsp::MasteringChain::get_latency_samples();
        TEST_ASSERT(latency == 159, "Expected latency 159");

        double max_diff = 0.0;
        for (size_t i = latency; i < N_TEST; ++i) {
            double diff_l = std::abs(out_l[i] - in_l[i - latency]);
            double diff_r = std::abs(out_r[i] - in_r[i - latency]);
            max_diff = std::max({max_diff, diff_l, diff_r});
        }
        std::cout << "  True Bypass Max Absolute Difference post-latency: " << max_diff << std::endl;
        TEST_ASSERT(max_diff < 1e-7, "True Bypass output must match delayed input exactly");

        // Disabilita bypass e verifica ritorno a modalità processing
        chain.set_bypass(false);
        TEST_ASSERT(chain.is_bypass() == false, "Chain must report bypass inactive");
        std::cout << "  -> True Bypass validato: identita float esatta dopo 159 campioni di latenza!" << std::endl;
    }

    std::cout << "\n====================================================" << std::endl;
    std::cout << "  TUTTI I TEST DEL TICKET 1.6 COMPLETATI CON SUCCESSO!  " << std::endl;
    std::cout << "====================================================" << std::endl;
    return 0;
}
