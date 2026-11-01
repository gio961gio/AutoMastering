#include "PluginProcessor.h"
#include "PluginEditor.h"

namespace {

float getDefaultValueForParam(size_t index) {
    // Valori di default trasparenti e musicali conformi alla specifica
    if (index < 18) {
        // EQ
        size_t param_in_band = index % 3;
        size_t band = index / 3;
        if (param_in_band == 0) {
            // f0
            constexpr float f0_defaults[6] = {100.0f, 250.0f, 1000.0f, 3000.0f, 8000.0f, 12000.0f};
            return f0_defaults[band];
        } else if (param_in_band == 1) {
            // gain_db
            return 0.0f;
        } else {
            // Q
            return 0.7071f;
        }
    } else if (index == 18) {
        // crossover_f_low
        return 150.0f;
    } else if (index == 19) {
        // crossover_f_high
        return 3000.0f;
    } else if (index >= 20 && index < 35) {
        // Multiband Comp: thresh, ratio, att, rel, makeup
        size_t p = (index - 20) % 5;
        if (p == 0) return 0.0f;     // threshold_db (0 = trasparente)
        if (p == 1) return 1.0f;     // ratio (1:1 = trasparente)
        if (p == 2) return 10.0f;    // attack_ms
        if (p == 3) return 100.0f;   // release_ms
        return 0.0f;                 // makeup_db
    } else if (index >= 35 && index < 38) {
        // Imager widths: low, mid, high
        return 1.0f; // 1.0 = identità stereo trasparente
    } else if (index == 38) {
        // saturator_drive_db
        return 0.0f;
    } else if (index == 39) {
        // saturator_mix
        return 0.0f; // 100% dry
    } else if (index == 40) {
        // limiter_push_gain_db
        return 0.0f; // 0 dB push
    }
    return 0.0f;
}

} // namespace

juce::AudioProcessorValueTreeState::ParameterLayout MasteringPluginAudioProcessor::createParameterLayout() {
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;
    params.reserve(core_dsp::NUM_PARAMS);

    for (size_t i = 0; i < core_dsp::NUM_PARAMS; ++i) {
        const auto& spec = core_dsp::PARAM_SPECS[i];
        const juce::String paramID(spec.name.data(), spec.name.size());
        const juce::String paramName = paramID;

        const float min_v = static_cast<float>(spec.min_val);
        const float max_v = static_cast<float>(spec.max_val);
        const float default_v = getDefaultValueForParam(i);

        juce::NormalisableRange<float> range(min_v, max_v);
        if (spec.scale == core_dsp::ScaleType::Exponential) {
            range.setSkewForCentre(std::sqrt(min_v * max_v));
        }

        const juce::String unit(spec.unit.data(), spec.unit.size());

        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID(paramID, 1),
            paramName,
            range,
            default_v,
            unit
        ));
    }

    return { params.begin(), params.end() };
}

MasteringPluginAudioProcessor::MasteringPluginAudioProcessor()
    : AudioProcessor(BusesProperties()
                     .withInput("Input", juce::AudioChannelSet::stereo(), true)
                     .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      apvts_(*this, nullptr, "Parameters", createParameterLayout())
{
    for (size_t i = 0; i < core_dsp::NUM_PARAMS; ++i) {
        const auto& spec = core_dsp::PARAM_SPECS[i];
        const juce::String paramID(spec.name.data(), spec.name.size());
        raw_params_[i] = apvts_.getRawParameterValue(paramID);
        current_theta_[i] = static_cast<double>(getDefaultValueForParam(i));
    }
}

void MasteringPluginAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock) {
    max_block_size_ = std::max(samplesPerBlock, 256);
    chain_.prepare(sampleRate, static_cast<size_t>(max_block_size_));

    // Segnalazione esatta della latenza alla DAW (159 campioni a 48 kHz)
    setLatencySamples(static_cast<int>(core_dsp::MasteringChain::get_latency_samples()));

    for (size_t i = 0; i < core_dsp::NUM_PARAMS; ++i) {
        smoothed_params_[i].reset(sampleRate, 0.020); // 20 ms di rampa anti-zipper
        const float val = raw_params_[i]->load(std::memory_order_relaxed);
        smoothed_params_[i].setCurrentAndTargetValue(val);
        current_theta_[i] = static_cast<double>(val);
    }

    chain_.set_parameters_theta(current_theta_);
}

void MasteringPluginAudioProcessor::releaseResources() {
    chain_.reset();
}

bool MasteringPluginAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const {
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

    if (layouts.getMainInputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

    return true;
}

void MasteringPluginAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) {
    juce::ScopedNoDenormals noDenormals;

    const int totalNumInputChannels = getTotalNumInputChannels();
    const int totalNumOutputChannels = getTotalNumOutputChannels();

    for (int i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear(i, 0, buffer.getNumSamples());

    const int num_samples = buffer.getNumSamples();
    if (num_samples <= 0 || totalNumInputChannels < 2)
        return;

    // Sub-block decimation processing (zero heap allocations)
    int samples_processed = 0;
    while (samples_processed < num_samples) {
        const int sub_samples = std::min(SUB_BLOCK_SIZE, num_samples - samples_processed);

        bool params_changed = false;
        for (size_t i = 0; i < core_dsp::NUM_PARAMS; ++i) {
            const float target = raw_params_[i]->load(std::memory_order_relaxed);
            smoothed_params_[i].setTargetValue(target);

            if (smoothed_params_[i].isSmoothing()) {
                params_changed = true;
                current_theta_[i] = static_cast<double>(smoothed_params_[i].skip(sub_samples));
            } else {
                current_theta_[i] = static_cast<double>(target);
            }
        }

        if (params_changed || samples_processed == 0) {
            chain_.set_parameters_theta(current_theta_);
        }

        chain_.process(
            buffer.getReadPointer(0, samples_processed),
            buffer.getReadPointer(1, samples_processed),
            buffer.getWritePointer(0, samples_processed),
            buffer.getWritePointer(1, samples_processed),
            static_cast<size_t>(sub_samples)
        );

        samples_processed += sub_samples;
    }
}

bool MasteringPluginAudioProcessor::loadParametersFromJson(const juce::String& jsonString) {
    juce::var parsedJson;
    if (juce::JSON::parse(jsonString, parsedJson).failed())
        return false;

    if (!parsedJson.isObject())
        return false;

    auto* obj = parsedJson.getDynamicObject();
    if (!obj)
        return false;

    for (size_t i = 0; i < core_dsp::NUM_PARAMS; ++i) {
        const auto& spec = core_dsp::PARAM_SPECS[i];
        const juce::String paramID(spec.name.data(), spec.name.size());

        if (obj->hasProperty(paramID)) {
            const float val = static_cast<float>(obj->getProperty(paramID));
            if (auto* param = apvts_.getParameter(paramID)) {
                param->setValueNotifyingHost(param->convertTo0to1(val));
            }
        }
    }
    return true;
}

juce::String MasteringPluginAudioProcessor::getParametersAsJson() const {
    juce::DynamicObject::Ptr obj = new juce::DynamicObject();

    for (size_t i = 0; i < core_dsp::NUM_PARAMS; ++i) {
        const auto& spec = core_dsp::PARAM_SPECS[i];
        const juce::String paramID(spec.name.data(), spec.name.size());
        const float val = raw_params_[i]->load(std::memory_order_relaxed);
        obj->setProperty(paramID, val);
    }

    return juce::JSON::toString(juce::var(obj.get()), false);
}

void MasteringPluginAudioProcessor::setNormalizedParameters(std::span<const float> z) {
    if (z.size() != core_dsp::NUM_PARAMS)
        return;

    for (size_t i = 0; i < core_dsp::NUM_PARAMS; ++i) {
        const auto& spec = core_dsp::PARAM_SPECS[i];
        const juce::String paramID(spec.name.data(), spec.name.size());
        const double theta_val = spec.map_z(static_cast<double>(z[i]));
        if (auto* param = apvts_.getParameter(paramID)) {
            param->setValueNotifyingHost(param->convertTo0to1(static_cast<float>(theta_val)));
        }
    }
}

std::array<float, core_dsp::NUM_PARAMS> MasteringPluginAudioProcessor::getNormalizedParameters() const {
    std::array<float, core_dsp::NUM_PARAMS> z{};
    for (size_t i = 0; i < core_dsp::NUM_PARAMS; ++i) {
        const auto& spec = core_dsp::PARAM_SPECS[i];
        const float val = raw_params_[i]->load(std::memory_order_relaxed);
        z[i] = static_cast<float>(spec.unmap_theta(static_cast<double>(val)));
    }
    return z;
}

void MasteringPluginAudioProcessor::getStateInformation(juce::MemoryBlock& destData) {
    auto state = apvts_.copyState();
    std::unique_ptr<juce::XmlElement> xml(state.createXml());
    copyXmlToBinary(*xml, destData);
}

void MasteringPluginAudioProcessor::setStateInformation(const void* data, int sizeInBytes) {
    std::unique_ptr<juce::XmlElement> xmlState(getXmlFromBinary(data, sizeInBytes));
    if (xmlState != nullptr && xmlState->hasTagName(apvts_.state.getType())) {
        apvts_.replaceState(juce::ValueTree::fromXml(*xmlState));
    }
}

juce::AudioProcessorEditor* MasteringPluginAudioProcessor::createEditor() {
    return new juce::GenericAudioProcessorEditor(*this);
}

bool MasteringPluginAudioProcessor::hasEditor() const {
    return true;
}

const juce::String MasteringPluginAudioProcessor::getName() const {
    return "AutoMastering PoliMi";
}

bool MasteringPluginAudioProcessor::acceptsMidi() const { return false; }
bool MasteringPluginAudioProcessor::producesMidi() const { return false; }
bool MasteringPluginAudioProcessor::isMidiEffect() const { return false; }
double MasteringPluginAudioProcessor::getTailLengthSeconds() const { return 0.0; }

int MasteringPluginAudioProcessor::getNumPrograms() { return 1; }
int MasteringPluginAudioProcessor::getCurrentProgram() { return 0; }
void MasteringPluginAudioProcessor::setCurrentProgram(int) {}
const juce::String MasteringPluginAudioProcessor::getProgramName(int) { return {}; }
void MasteringPluginAudioProcessor::changeProgramName(int, const juce::String&) {}

// Plugin creation factory
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
    return new MasteringPluginAudioProcessor();
}
