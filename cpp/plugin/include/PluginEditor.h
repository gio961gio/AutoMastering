#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include "PluginProcessor.h"

class MasteringPluginAudioProcessorEditor : public juce::AudioProcessorEditor {
public:
    explicit MasteringPluginAudioProcessorEditor(MasteringPluginAudioProcessor&);
    ~MasteringPluginAudioProcessorEditor() override = default;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    MasteringPluginAudioProcessor& audioProcessor;

    juce::Label titleLabel;
    juce::Label subtitleLabel;
    juce::Label latencyLabel;
    juce::Label chainInfoLabel;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MasteringPluginAudioProcessorEditor)
};
