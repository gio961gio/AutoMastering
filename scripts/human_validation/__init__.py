"""Human Validation Package for AutoMastering (Sprint 1 & Sprint 2 Audio Verification).

Provides reproducible offline rendering with latency compensation and tail flush,
input validation, static loudness matching, blind listening set generation,
and structured manifest/report exports without modifying the frozen DSP engine.
"""

ENGINE_REFERENCE_CODE_COMMIT = "5940e81f12741e2276f7a446adfb6e12f44574fe"
ENGINE_REFERENCE_DOC_COMMIT = "92fe105a19312f258fefb2c70a847720cbd95525"
LEGACY_BASELINE_SOURCE_COMMIT = "86ae20f08dc7fbebc7f67af064a15b648a492fc7"
PACKAGE_VERSION = "1.0.0"
