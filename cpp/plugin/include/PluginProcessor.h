#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <array>
#include <span>
#include <atomic>
#include "core_dsp/MasteringChain.hpp"
#include "core_dsp/ParameterSpec.hpp"

class MasteringPluginAudioProcessor : public juce::AudioProcessor {
public:
    MasteringPluginAudioProcessor();
    ~MasteringPluginAudioProcessor() override = default;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;

    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int index, const juce::String& newName) override;

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    // Preset & JSON State Injection per integrazione con il modello di inferenza DDSP
    bool loadParametersFromJson(const juce::String& jsonString);
    juce::String getParametersAsJson() const;

    void setNormalizedParameters(std::span<const float> z);
    std::array<float, core_dsp::NUM_PARAMS> getNormalizedParameters() const;
    [[nodiscard]] std::array<double, core_dsp::NUM_PARAMS> getCurrentTheta() const noexcept { return current_theta_; }

    juce::AudioProcessorValueTreeState& getAPVTS() noexcept { return apvts_; }
    const core_dsp::MasteringChain& getChain() const noexcept { return chain_; }

    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    static constexpr int SUB_BLOCK_SIZE = 32;

private:
    core_dsp::MasteringChain chain_;
    juce::AudioProcessorValueTreeState apvts_;

    std::array<std::atomic<float>*, core_dsp::NUM_PARAMS> raw_params_{};
    std::array<juce::SmoothedValue<float, juce::ValueSmoothingTypes::Linear>, core_dsp::NUM_PARAMS> smoothed_params_{};
    std::array<double, core_dsp::NUM_PARAMS> current_theta_{};

    int max_block_size_ = 2048;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MasteringPluginAudioProcessor)
};
