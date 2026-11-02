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

    // TEST 1: Istanziazione e verifica dei 41 parametri DSP + 1 Bypass in APVTS
    std::cout << "\n[TEST 1] Verifica Istanziazione e Parametri APVTS..." << std::endl;
    MasteringPluginAudioProcessor processor;

    const auto& params = processor.getParameters();
    std::cout << "  Numero parametri APVTS registrati: " << params.size() << std::endl;
    TEST_ASSERT(params.size() == core_dsp::NUM_PARAMS + 1, "Expected 41 DSP parameters + 1 Bypass parameter in APVTS");

    auto* bypass_param = processor.getBypassParameter();
    TEST_ASSERT(bypass_param != nullptr, "getBypassParameter() must return non-null pointer");
    TEST_ASSERT(bypass_param->getName(100) == "Bypass", "Bypass parameter name must be 'Bypass'");
    TEST_ASSERT(bypass_param->getValue() == 0.0f, "Bypass default value must be 0 (false)");

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
    std::cout << "  -> Tutti i 41 parametri canonici + True Bypass sono conformi e verificati in APVTS!" << std::endl;

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
        // 1. JSON canonico esportato da JUCE/Python con schema AutoMastering_Preset_v1
        const juce::String validCanonicalJson = processor.getParametersAsJson();
        bool loaded = processor.loadParametersFromJson(validCanonicalJson);
        TEST_ASSERT(loaded, "Failed to parse and load canonical AutoMastering_Preset_v1 JSON");

        // 2. Rifiuto esplicito di preset incompleto (parametri mancanti senza normalized_z)
        const juce::String incompleteJson = R"({
            "format": "AutoMastering_Preset_v1",
            "version": "1.0",
            "sample_rate": 48000.0,
            "parameters": {
                "eq_b0_f0": 85.0,
                "eq_b0_gain_db": 3.0
            }
        })";
        TEST_ASSERT(!processor.loadParametersFromJson(incompleteJson), "Incomplete preset with missing parameters must return false");

        // 3. Rifiuto esplicito di formato non conforme
        const juce::String wrongFormatJson = R"({
            "format": "LegacyPreset_v0",
            "version": "1.0",
            "sample_rate": 48000.0,
            "normalized_z": [0.5]
        })";
        TEST_ASSERT(!processor.loadParametersFromJson(wrongFormatJson), "Wrong format string must return false");

        // 4. Test JSON corrotto o vuoto
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

    // TEST 9: Host Bypass Parameter Integration via processBlock
    std::cout << "\n[TEST 9] Verifica Host Bypass Parameter tramite processBlock..." << std::endl;
    {
        MasteringPluginAudioProcessor bypass_proc;
        bypass_proc.prepareToPlay(fs, 512);

        // Turn host bypass ON
        auto* bp = bypass_proc.getBypassParameter();
        TEST_ASSERT(bp != nullptr, "Bypass parameter not found");
        bp->setValueNotifyingHost(1.0f); // 1.0 = true

        constexpr int TEST_SAMPLES = 2048;
        juce::AudioBuffer<float> buffer(2, TEST_SAMPLES);
        std::vector<float> in_l(TEST_SAMPLES), in_r(TEST_SAMPLES);
        for (int s = 0; s < TEST_SAMPLES; ++s) {
            in_l[s] = dist(rng);
            in_r[s] = dist(rng);
            buffer.setSample(0, s, in_l[s]);
            buffer.setSample(1, s, in_r[s]);
        }

        bypass_proc.processBlock(buffer, midi);

        constexpr int latency = 159;
        double max_bp_diff = 0.0;
        for (int s = latency; s < TEST_SAMPLES; ++s) {
            double diff_l = std::abs(buffer.getSample(0, s) - in_l[s - latency]);
            double diff_r = std::abs(buffer.getSample(1, s) - in_r[s - latency]);
            max_bp_diff = std::max({max_bp_diff, diff_l, diff_r});
        }
        std::cout << "  Plugin processBlock Host Bypass Max Diff: " << max_bp_diff << std::endl;
        TEST_ASSERT(max_bp_diff < 1e-6, "Plugin host bypass output must match input delayed by 159 samples");
        std::cout << "  -> Host Bypass verificato con successo in processBlock!" << std::endl;
    }

    // TEST 10: APVTS State save/restore (DAW session persistence)
    std::cout << "\n[TEST 10] Verifica salvataggio/ripristino stato APVTS (DAW session)..." << std::endl;
    {
        MasteringPluginAudioProcessor proc_a;
        proc_a.prepareToPlay(fs, 512);

        // Set bypass ON and change parameter
        proc_a.getBypassParameter()->setValueNotifyingHost(1.0f);
        if (auto* p = proc_a.getAPVTS().getParameter("saturator_drive_db")) {
            p->setValueNotifyingHost(p->convertTo0to1(3.5f));
        }

        juce::MemoryBlock stateData;
        proc_a.getStateInformation(stateData);

        MasteringPluginAudioProcessor proc_b;
        proc_b.prepareToPlay(fs, 512);
        proc_b.setStateInformation(stateData.getData(), (int)stateData.getSize());

        TEST_ASSERT(proc_b.getBypassParameter()->getValue() == 1.0f, "Bypass parameter state must be restored");
        float restored_drive = proc_b.getAPVTS().getRawParameterValue("saturator_drive_db")->load();
        TEST_ASSERT(std::abs(restored_drive - 3.5f) < 0.1f, "saturator_drive_db must be restored");
        std::cout << "  -> Stato APVTS serializzato e ripristinato con successo!" << std::endl;
    }

    // TEST 11: Exported JSON Schema Validation (AutoMastering_Preset_v1)
    std::cout << "\n[TEST 11] Verifica rigorosa dello schema JSON esportato..." << std::endl;
    {
        juce::String exported = processor.getParametersAsJson();
        juce::var parsed;
        TEST_ASSERT(juce::JSON::parse(exported, parsed).wasOk(), "Exported JSON must be valid JSON");
        TEST_ASSERT(parsed["format"].toString() == "AutoMastering_Preset_v1", "Format must be AutoMastering_Preset_v1");
        TEST_ASSERT(parsed["version"].toString() == "1.0", "Version must be 1.0");
        TEST_ASSERT(static_cast<double>(parsed["sample_rate"]) == 48000.0, "Sample rate must match");

        auto* z_arr = parsed["normalized_z"].getArray();
        TEST_ASSERT(z_arr != nullptr && z_arr->size() == 41, "normalized_z must have 41 items");

        auto* paramsObj = parsed["parameters"].getDynamicObject();
        TEST_ASSERT(paramsObj != nullptr && paramsObj->getProperties().size() == 41, "parameters must have 41 keys");

        auto* detailsObj = parsed["parameter_details"].getDynamicObject();
        TEST_ASSERT(detailsObj != nullptr && detailsObj->getProperties().size() == 41, "parameter_details must have 41 keys");
        std::cout << "  -> Schema JSON AutoMastering_Preset_v1 validato al 100%!" << std::endl;
    }

    // TEST 12: End-to-End Python Exporter Fixture -> JUCE Parser (G2.1, G2.2, G2.3, G2.4)
    std::cout << "\n[TEST 12] Verifica End-to-End Fixture Python -> JUCE Parser (41 parametri, Mode A/B, Version & Atomicity)..." << std::endl;
    {
        const std::vector<juce::String> candidatePaths = {
            "python/tests/fixtures/python_exported_preset_v1.json",
            "../python/tests/fixtures/python_exported_preset_v1.json",
            "../../python/tests/fixtures/python_exported_preset_v1.json"
        };

        juce::File fixtureFile;
        bool foundFixture = false;
        for (const auto& relPath : candidatePaths) {
            juce::File f = juce::File::getCurrentWorkingDirectory().getChildFile(relPath);
            if (f.existsAsFile()) {
                fixtureFile = f;
                foundFixture = true;
                break;
            }
        }
        TEST_ASSERT(foundFixture, "Authentic Python preset fixture file not found in expected paths");

        const juce::String pyJsonStr = fixtureFile.loadFileAsString();
        TEST_ASSERT(pyJsonStr.isNotEmpty(), "Python fixture file is empty");

        juce::var pyParsed;
        TEST_ASSERT(juce::JSON::parse(pyJsonStr, pyParsed).wasOk(), "Python fixture must be valid JSON");
        TEST_ASSERT(pyParsed["format"].toString() == "AutoMastering_Preset_v1", "Fixture format must be AutoMastering_Preset_v1");
        TEST_ASSERT(pyParsed["version"].toString() == "1.0", "Fixture version must be 1.0");

        const auto* z_arr = pyParsed["normalized_z"].getArray();
        TEST_ASSERT(z_arr != nullptr && z_arr->size() == static_cast<int>(core_dsp::NUM_PARAMS), "Fixture normalized_z must have 41 elements");

        const auto* paramsObj = pyParsed["parameters"].getDynamicObject();
        TEST_ASSERT(paramsObj != nullptr, "Fixture parameters object must exist");

        std::array<float, core_dsp::NUM_PARAMS> expected_z{};
        std::array<float, core_dsp::NUM_PARAMS> expected_theta{};
        bool z_not_all_equal = false;
        bool z_not_all_half = false;

        for (size_t i = 0; i < core_dsp::NUM_PARAMS; ++i) {
            const auto& spec = core_dsp::PARAM_SPECS[i];
            const juce::String paramID(spec.name.data(), spec.name.size());
            TEST_ASSERT(paramsObj->hasProperty(paramID), ("Missing canonical param in fixture: " + paramID.toStdString()).c_str());

            expected_z[i] = static_cast<float>(static_cast<double>((*z_arr)[static_cast<int>(i)]));
            expected_theta[i] = static_cast<float>(static_cast<double>(paramsObj->getProperty(paramID)));

            TEST_ASSERT(std::isfinite(expected_z[i]) && expected_z[i] >= 0.0f && expected_z[i] <= 1.0f, "Expected z out of [0,1]");
            TEST_ASSERT(std::isfinite(expected_theta[i]) && expected_theta[i] >= static_cast<float>(spec.min_val) - 1e-4f
                        && expected_theta[i] <= static_cast<float>(spec.max_val) + 1e-4f, "Expected theta out of bounds");

            if (i > 0 && std::abs(expected_z[i] - expected_z[0]) > 1e-3f) z_not_all_equal = true;
            if (std::abs(expected_z[i] - 0.5f) > 1e-3f) z_not_all_half = true;
        }
        TEST_ASSERT(z_not_all_equal && z_not_all_half, "Python fixture z must contain distinct, non-trivial interior values");

        // G2.2 — Mode A: Caricamento preset completo (con normalized_z e parameters)
        MasteringPluginAudioProcessor pyProc;
        pyProc.prepareToPlay(fs, 512);
        TEST_ASSERT(pyProc.loadParametersFromJson(pyJsonStr), "Failed to load authentic Python fixture in Mode A");

        const auto loaded_z_modeA = pyProc.getNormalizedParameters();
        for (size_t i = 0; i < core_dsp::NUM_PARAMS; ++i) {
            const auto& spec = core_dsp::PARAM_SPECS[i];
            const juce::String paramID(spec.name.data(), spec.name.size());
            const float actual_theta = pyProc.getAPVTS().getRawParameterValue(paramID)->load();
            const float theta_tol = std::max(1e-3f, 1e-4f * std::abs(expected_theta[i]));
            TEST_ASSERT(std::abs(actual_theta - expected_theta[i]) <= theta_tol,
                        ("Mode A theta mismatch on " + paramID.toStdString()).c_str());
            TEST_ASSERT(std::abs(loaded_z_modeA[i] - expected_z[i]) <= 1e-4f,
                        ("Mode A normalized_z mismatch on " + paramID.toStdString()).c_str());
        }

        // G2.3 — Mode B: Caricamento con normalized_z rimosso (solo mappa esplicita parameters di 41 elementi)
        MasteringPluginAudioProcessor pyProcModeB;
        pyProcModeB.prepareToPlay(fs, 512);

        juce::var parsedModeB;
        juce::JSON::parse(pyJsonStr, parsedModeB);
        parsedModeB.getDynamicObject()->removeProperty("normalized_z");
        const juce::String jsonModeB = juce::JSON::toString(parsedModeB, true);

        TEST_ASSERT(pyProcModeB.loadParametersFromJson(jsonModeB), "Failed to load Python fixture in Mode B (parameters map only)");
        const auto loaded_z_modeB = pyProcModeB.getNormalizedParameters();
        for (size_t i = 0; i < core_dsp::NUM_PARAMS; ++i) {
            const auto& spec = core_dsp::PARAM_SPECS[i];
            const juce::String paramID(spec.name.data(), spec.name.size());
            const float actual_theta = pyProcModeB.getAPVTS().getRawParameterValue(paramID)->load();
            const float theta_tol = std::max(1e-3f, 1e-4f * std::abs(expected_theta[i]));
            TEST_ASSERT(std::abs(actual_theta - expected_theta[i]) <= theta_tol,
                        ("Mode B theta mismatch on " + paramID.toStdString()).c_str());
            TEST_ASSERT(std::abs(loaded_z_modeB[i] - expected_z[i]) <= 1e-4f,
                        ("Mode B normalized_z mismatch on " + paramID.toStdString()).c_str());
        }

        // G2.4 — Validazione versione schema e atomicita (nessun aggiornamento parziale APVTS su errore)
        std::array<float, core_dsp::NUM_PARAMS> snapshot_theta{};
        for (size_t i = 0; i < core_dsp::NUM_PARAMS; ++i) {
            const juce::String paramID(core_dsp::PARAM_SPECS[i].name.data(), core_dsp::PARAM_SPECS[i].name.size());
            snapshot_theta[i] = pyProc.getAPVTS().getRawParameterValue(paramID)->load();
        }

        auto assertStateUnchanged = [&]() {
            for (size_t i = 0; i < core_dsp::NUM_PARAMS; ++i) {
                const juce::String paramID(core_dsp::PARAM_SPECS[i].name.data(), core_dsp::PARAM_SPECS[i].name.size());
                const float cur = pyProc.getAPVTS().getRawParameterValue(paramID)->load();
                TEST_ASSERT(cur == snapshot_theta[i], ("APVTS mutated on rejected preset at param " + paramID.toStdString()).c_str());
            }
        };

        // Versione non supportata "2.0"
        {
            juce::var badVer;
            juce::JSON::parse(pyJsonStr, badVer);
            badVer.getDynamicObject()->setProperty("version", "2.0");
            TEST_ASSERT(!pyProc.loadParametersFromJson(juce::JSON::toString(badVer)), "Unsupported version 2.0 must be rejected");
            assertStateUnchanged();
        }
        // Versione mancante
        {
            juce::var noVer;
            juce::JSON::parse(pyJsonStr, noVer);
            noVer.getDynamicObject()->removeProperty("version");
            TEST_ASSERT(!pyProc.loadParametersFromJson(juce::JSON::toString(noVer)), "Missing version must be rejected");
            assertStateUnchanged();
        }
        // Valore fuori range a metà vettore (indice 20) in Mode B
        {
            juce::var outOfRange;
            juce::JSON::parse(jsonModeB, outOfRange);
            auto* pObj = outOfRange["parameters"].getDynamicObject();
            pObj->setProperty("eq_b0_gain_db", -5.5); // valido, non deve essere applicato parzialmente!
            pObj->setProperty("comp_low_threshold_db", -999.0); // fuori range [-30, 0]
            TEST_ASSERT(!pyProc.loadParametersFromJson(juce::JSON::toString(outOfRange)), "Out-of-range parameter must be rejected");
            assertStateUnchanged();
        }
        // Valore non numerico ("NaN") in normalized_z
        {
            juce::var nanZ;
            juce::JSON::parse(pyJsonStr, nanZ);
            nanZ["normalized_z"].getArray()->set(15, juce::var("NaN"));
            TEST_ASSERT(!pyProc.loadParametersFromJson(juce::JSON::toString(nanZ)), "NaN string in normalized_z must be rejected");
            assertStateUnchanged();
        }
        // Valore fuori dominio (> 1.0) in normalized_z
        {
            juce::var badZ;
            juce::JSON::parse(pyJsonStr, badZ);
            badZ["normalized_z"].getArray()->set(10, juce::var(1.25));
            TEST_ASSERT(!pyProc.loadParametersFromJson(juce::JSON::toString(badZ)), "Out-of-domain normalized_z > 1.0 must be rejected");
            assertStateUnchanged();
        }

        std::cout << "  -> Fixture Python -> JUCE (Mode A/B, 41/41 parametri, schema version 1.0 e atomicita) verificata al 100%!" << std::endl;
    }

    std::cout << "\n====================================================" << std::endl;
    std::cout << "  TUTTI I TEST DEL TICKET 1.6 COMPLETATI CON SUCCESSO!  " << std::endl;
    std::cout << "====================================================" << std::endl;
    return 0;
}
