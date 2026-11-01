cmake_minimum_required(VERSION 3.22)
project(PolimiDDSPAutoMastering VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Header-only core DSP library
add_library(core_dsp INTERFACE)
target_include_directories(core_dsp INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/core_dsp/include>
    $<INSTALL_INTERFACE:include>
)

# Option to build pybind11 bindings via CMake (default OFF as setuptools handles it)
option(BUILD_PYTHON_BINDINGS "Build python bindings via CMake" OFF)
if(BUILD_PYTHON_BINDINGS)
    set(PYBIND11_FINDPYTHON ON)
    find_package(pybind11 CONFIG QUIET)
    if(pybind11_FOUND)
        pybind11_add_module(py_core_dsp cpp/bindings/py_core_dsp.cpp)
        target_link_libraries(py_core_dsp PRIVATE core_dsp)
        target_compile_features(py_core_dsp PRIVATE cxx_std_20)
    endif()
endif()

# ==============================================================================
# JUCE 8 Plugin & Headless Test Suite (Ticket 1.6)
# ==============================================================================
set(JUCE_DIR "C:/Users/Giovanni/Downloads/juce-8.0.12-windows/JUCE" CACHE PATH "Path to JUCE 8 directory")

if(EXISTS "${JUCE_DIR}/CMakeLists.txt")
    message(STATUS "Found JUCE 8 at: ${JUCE_DIR}")
    add_subdirectory("${JUCE_DIR}" JUCE EXCLUDE_FROM_ALL)

    juce_add_plugin(MasteringPlugin
        COMPANY_NAME "PoliMi"
        BUNDLE_ID "com.polimi.automastering"
        VERSION "1.0.0"
        IS_SYNTH FALSE
        NEEDS_MIDI_INPUT FALSE
        NEEDS_MIDI_OUTPUT FALSE
        IS_MIDI_EFFECT FALSE
        EDITOR_WANTS_KEYBOARD_FOCUS FALSE
        COPY_PLUGIN_AFTER_BUILD FALSE
        PLUGIN_MANUFACTURER_CODE Poli
        PLUGIN_CODE AuMa
        FORMATS VST3 Standalone
        PRODUCT_NAME "AutoMastering PoliMi"
    )

    target_sources(MasteringPlugin PRIVATE
        cpp/plugin/include/PluginProcessor.h
        cpp/plugin/source/PluginProcessor.cpp
        cpp/plugin/include/PluginEditor.h
        cpp/plugin/source/PluginEditor.cpp
    )

    target_include_directories(MasteringPlugin PRIVATE
        cpp/plugin/include
    )

    target_link_libraries(MasteringPlugin PRIVATE
        core_dsp
        juce::juce_audio_utils
        juce::juce_audio_processors
        juce::juce_dsp
        juce::juce_gui_basics
    )

    target_compile_features(MasteringPlugin PRIVATE cxx_std_20)
    target_compile_definitions(MasteringPlugin PUBLIC JUCE_VST3_CAN_REPLACE_VST2=0)

    # Headless Unit Test Executable
    add_executable(plugin_headless_tests
        cpp/plugin/tests/test_plugin.cpp
        cpp/plugin/source/PluginProcessor.cpp
        cpp/plugin/source/PluginEditor.cpp
    )

    target_include_directories(plugin_headless_tests PRIVATE
        cpp/plugin/include
    )

    target_link_libraries(plugin_headless_tests PRIVATE
        core_dsp
        juce::juce_audio_utils
        juce::juce_audio_processors
        juce::juce_dsp
        juce::juce_gui_basics
    )

    target_compile_features(plugin_headless_tests PRIVATE cxx_std_20)
    target_compile_definitions(plugin_headless_tests PRIVATE JUCE_VST3_CAN_REPLACE_VST2=0)
else()
    message(WARNING "JUCE 8 directory not found at ${JUCE_DIR}. Skipping plugin build.")
endif()
