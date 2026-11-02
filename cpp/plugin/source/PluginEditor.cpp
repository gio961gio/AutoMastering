#include "PluginEditor.h"

MasteringPluginAudioProcessorEditor::MasteringPluginAudioProcessorEditor(MasteringPluginAudioProcessor& p)
    : AudioProcessorEditor(&p), audioProcessor(p)
{
    titleLabel.setText("AutoMastering PoliMi", juce::dontSendNotification);
    titleLabel.setFont(juce::FontOptions(22.0f, juce::Font::bold));
    titleLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    titleLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(titleLabel);

    subtitleLabel.setText("DDSP Differentiable Twin Engine (C++20 & PyTorch)", juce::dontSendNotification);
    subtitleLabel.setFont(juce::FontOptions(13.0f, juce::Font::italic));
    subtitleLabel.setColour(juce::Label::textColourId, juce::Colours::lightgrey);
    subtitleLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(subtitleLabel);

    double sr = audioProcessor.getSampleRate();
    if (sr > 0.0) {
        double ms = (159.0 / sr) * 1000.0;
        juce::String text = "Reported Host Latency: 159 samples (" + juce::String(ms, 2) + " ms @ " + juce::String(sr / 1000.0, 1) + " kHz)";
        latencyLabel.setText(text, juce::dontSendNotification);
    } else {
        latencyLabel.setText("Reported Host Latency: 159 samples (3.31 ms @ 48 kHz reference)", juce::dontSendNotification);
    }
    latencyLabel.setFont(juce::FontOptions(12.0f));
    latencyLabel.setColour(juce::Label::textColourId, juce::Colour(0xff4caf50));
    latencyLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(latencyLabel);

    chainInfoLabel.setText("Active Stages: 6-Band EQ | LR4 Crossover | 3-Band Comp | M/S Imager | Saturator 2x | Sample-Peak Lookahead Limiter (-0.3 dBFS)", juce::dontSendNotification);
    chainInfoLabel.setFont(juce::FontOptions(11.0f));
    chainInfoLabel.setColour(juce::Label::textColourId, juce::Colour(0xff90caf9));
    chainInfoLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(chainInfoLabel);

    setSize(640, 260);
}

void MasteringPluginAudioProcessorEditor::paint(juce::Graphics& g) {
    g.fillAll(juce::Colour(0xff121214)); // Elegant dark background

    auto bounds = getLocalBounds().toFloat();

    // Subtle header gradient
    juce::ColourGradient grad(juce::Colour(0xff1e1e24), 0, 0, juce::Colour(0xff121214), 0, 80.0f, false);
    g.setGradientFill(grad);
    g.fillRect(bounds.removeFromTop(80.0f));

    // Outer border
    g.setColour(juce::Colour(0xff2c2c34));
    g.drawRect(getLocalBounds(), 1);

    // Modules status box
    auto statusBox = getLocalBounds().reduced(20).withTrimmedTop(90);
    g.setColour(juce::Colour(0xff1a1a20));
    g.fillRoundedRectangle(statusBox.toFloat(), 6.0f);
    g.setColour(juce::Colour(0xff2d2d38));
    g.drawRoundedRectangle(statusBox.toFloat(), 6.0f, 1.0f);
}

void MasteringPluginAudioProcessorEditor::resized() {
    auto bounds = getLocalBounds();
    titleLabel.setBounds(bounds.removeFromTop(40).reduced(10, 0));
    subtitleLabel.setBounds(bounds.removeFromTop(25).reduced(10, 0));
    latencyLabel.setBounds(bounds.removeFromTop(25).reduced(10, 0));

    bounds.removeFromTop(20);
    chainInfoLabel.setBounds(bounds.removeFromTop(30).reduced(25, 0));
}
