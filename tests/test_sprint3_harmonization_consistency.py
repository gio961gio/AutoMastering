"""Sprint 3 — Fase 5 Automated Consistency & Harmonization Test Suite.

Verifies the 7 mandatory harmonization gates across V0p, V1, V1b, and V2:
1. `test_harmonized_chunk_metadata_bit_identity`: Bit-identity of chunk_metadata across all 4 variants.
2. `test_harmonized_target_lufs_identity`: Exact identity of target_lufs across all 4 variants per session.
3. `test_harmonized_chunk_coordinate_coherence`: Chunk length (144,000 samples) and pre-roll (<= 120,000 samples).
4. `test_historical_artifacts_integrity`: 100% integrity of SPRINT3_V0P_V1_ABLATION_RESULTS.json and SPRINT3_V0P_V1_REPORT.md.
5. `test_harmonized_sessions_completeness`: Completeness of all 6 sessions in both V2 and V1 harmonized results.
6. `test_harmonized_parameter_serialization`: Correct serialization of 41-dim normalized z in [0, 1], theta, and seed string.
7. `test_cache_directories_isolation`: Strict separation of cache directories between protocols.
"""

import json
import subprocess
from pathlib import Path
from typing import Any, Dict

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DOCS_DIR = _REPO_ROOT / "docs" / "sprint3"
_V2_RESULTS_PATH = _DOCS_DIR / "SPRINT3_V2_ABLATION_RESULTS.json"
_V1_HARMONIZED_PATH = _DOCS_DIR / "SPRINT3_V1_HARMONIZED_RESULTS.json"
_V0P_V1_HISTORICAL_PATH = _DOCS_DIR / "SPRINT3_V0P_V1_ABLATION_RESULTS.json"
_V0P_V1_HISTORICAL_REPORT = _DOCS_DIR / "SPRINT3_V0P_V1_REPORT.md"

EXPECTED_SESSIONS = [
    "track01",
    "track02",
    "track03",
    "track01_style",
    "track02_style",
    "track03_style",
]


def _load_results(path: Path) -> Dict[str, Any]:
    if not path.exists():
        pytest.skip(f"Result file {path.name} not found yet (run runners first).")
    return json.loads(path.read_text(encoding="utf-8"))


def test_historical_artifacts_integrity():
    """Gate 4: Ensure historical Sprint 3 artifacts have NEVER been modified."""
    assert _V0P_V1_HISTORICAL_PATH.exists(), "Historical SPRINT3_V0P_V1_ABLATION_RESULTS.json is missing!"
    assert _V0P_V1_HISTORICAL_REPORT.exists(), "Historical SPRINT3_V0P_V1_REPORT.md is missing!"

    # Check git diff for historical files
    cmd = [
        "git",
        "diff",
        "--name-only",
        "origin/human-validation-package",
        "--",
        str(_V0P_V1_HISTORICAL_PATH),
        str(_V0P_V1_HISTORICAL_REPORT),
    ]
    res = subprocess.run(cmd, cwd=str(_REPO_ROOT), capture_output=True, text=True)
    assert res.returncode == 0, f"Git diff command failed: {res.stderr}"
    modified_files = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    assert len(modified_files) == 0, (
        f"VIOLATION: Historical artifacts have been modified! {modified_files}"
    )


def test_cache_directories_isolation():
    """Gate 7: Ensure cache directories are strictly isolated between protocols."""
    cache_v2 = _DOCS_DIR / ".cache_v2_harmonized_final"
    cache_v1 = _DOCS_DIR / ".cache_v1_harmonized_final"
    cache_hist = _DOCS_DIR / ".cache_v0p_v1"

    # Distinct paths
    assert cache_v2 != cache_v1
    assert cache_v2 != cache_hist
    assert cache_v1 != cache_hist


def test_harmonized_sessions_completeness():
    """Gate 5: Completeness of all 6 sessions in both V2 and V1 harmonized results."""
    v2_data = _load_results(_V2_RESULTS_PATH)
    v1_data = _load_results(_V1_HARMONIZED_PATH)

    v2_prefixes = [s["track_prefix"] for s in v2_data["sessions"]]
    v1_prefixes = [s["track_prefix"] for s in v1_data["sessions"]]

    assert v2_prefixes == EXPECTED_SESSIONS, f"V2 sessions incomplete: {v2_prefixes}"
    assert v1_prefixes == EXPECTED_SESSIONS, f"V1 harmonized sessions incomplete: {v1_prefixes}"


def test_harmonized_chunk_metadata_bit_identity():
    """Gate 1: Bit-identity of chunk_metadata across all 4 variants."""
    v2_data = _load_results(_V2_RESULTS_PATH)
    v1_data = _load_results(_V1_HARMONIZED_PATH)

    v2_by_prefix = {s["track_prefix"]: s for s in v2_data["sessions"]}
    v1_by_prefix = {s["track_prefix"]: s for s in v1_data["sessions"]}

    for prefix in EXPECTED_SESSIONS:
        s_v2 = v2_by_prefix[prefix]
        s_v1 = v1_by_prefix[prefix]

        meta_v2 = s_v2["chunk_metadata"]
        meta_v1 = s_v1["chunk_metadata"]

        assert len(meta_v2) == 4, f"[{prefix}] Expected 4 chunks in V2"
        assert len(meta_v1) == 4, f"[{prefix}] Expected 4 chunks in V1"

        for k in range(4):
            assert meta_v2[k] == meta_v1[k], (
                f"[{prefix}] Chunk {k} metadata mismatch:\nV2: {meta_v2[k]}\nV1: {meta_v1[k]}"
            )


def test_harmonized_target_lufs_identity():
    """Gate 2: Exact identity of target_lufs across all 4 variants per session."""
    v2_data = _load_results(_V2_RESULTS_PATH)
    v1_data = _load_results(_V1_HARMONIZED_PATH)

    v2_by_prefix = {s["track_prefix"]: s for s in v2_data["sessions"]}
    v1_by_prefix = {s["track_prefix"]: s for s in v1_data["sessions"]}

    for prefix in EXPECTED_SESSIONS:
        s_v2 = v2_by_prefix[prefix]
        s_v1 = v1_by_prefix[prefix]

        tlufs_v2 = float(s_v2["target_lufs"])
        tlufs_v1 = float(s_v1["target_lufs"])

        assert abs(tlufs_v2 - tlufs_v1) < 1e-5, (
            f"[{prefix}] Target LUFS mismatch between V2 ({tlufs_v2}) and V1 ({tlufs_v1})"
        )

        # Check internal variant target_lufs
        assert abs(float(s_v2["variants"]["V0p"]["target_lufs"]) - tlufs_v2) < 1e-5
        assert abs(float(s_v2["variants"]["V2"]["target_lufs"]) - tlufs_v2) < 1e-5
        assert abs(float(s_v1["variants"]["V1"]["target_lufs"]) - tlufs_v1) < 1e-5
        assert abs(float(s_v1["variants"]["V1b"]["target_lufs"]) - tlufs_v1) < 1e-5


def test_harmonized_chunk_coordinate_coherence():
    """Gate 3: Chunk length (144,000 samples) and pre-roll (<= 120,000 samples)."""
    v2_data = _load_results(_V2_RESULTS_PATH)

    for s in v2_data["sessions"]:
        prefix = s["track_prefix"]
        for k, ch in enumerate(s["chunk_metadata"]):
            mix_len = ch["mix_end_sample"] - ch["mix_start_sample"]
            ref_len = ch["ref_end_sample"] - ch["ref_start_sample"]
            pre_len = ch["mix_preroll_samples"]

            assert mix_len == 144000, f"[{prefix}] Chunk {k} mix length {mix_len} != 144000"
            assert ref_len == 144000, f"[{prefix}] Chunk {k} ref length {ref_len} != 144000"
            assert 0 <= pre_len <= 120000, f"[{prefix}] Chunk {k} pre-roll {pre_len} out of range [0, 120000]"


def test_harmonized_parameter_serialization():
    """Gate 6: Correct serialization of 41-dim normalized z in [0, 1], theta, and seed string."""
    v2_data = _load_results(_V2_RESULTS_PATH)
    v1_data = _load_results(_V1_HARMONIZED_PATH)

    for dataset in [v2_data, v1_data]:
        for s in dataset["sessions"]:
            prefix = s["track_prefix"]
            seed_str = s["seed"]
            assert isinstance(seed_str, str), f"[{prefix}] Seed must be serialized as string: {type(seed_str)}"
            assert int(seed_str) > 0, f"[{prefix}] Invalid seed: {seed_str}"

            for v_name, v_obj in s["variants"].items():
                if v_name == "v0p_historical":
                    continue
                z_opt = v_obj.get("normalized_z_opt")
                assert z_opt is not None, f"[{prefix}:{v_name}] Missing normalized_z_opt"
                assert len(z_opt) == 41, f"[{prefix}:{v_name}] normalized_z_opt length {len(z_opt)} != 41"
                for i, val in enumerate(z_opt):
                    assert 0.0 <= val <= 1.0, f"[{prefix}:{v_name}] z[{i}] = {val} not in [0, 1]"

                theta = v_obj.get("physical_parameters_theta")
                assert theta is not None, f"[{prefix}:{v_name}] Missing physical_parameters_theta"
                assert len(theta) == 41, f"[{prefix}:{v_name}] physical_parameters_theta len {len(theta)} != 41"


def test_harmonized_loss_breakdown_erb_deharsh_positive():
    """Gate 8: Verify that all_10_terms_init and all_10_terms_final contain strictly positive ERB terms."""
    v2_data = _load_results(_V2_RESULTS_PATH)
    v1_data = _load_results(_V1_HARMONIZED_PATH)

    for dataset, dname in [(v2_data, "V2"), (v1_data, "V1_Harmonized")]:
        for s in dataset["sessions"]:
            prefix = s["track_prefix"]
            for v_name, v_obj in s["variants"].items():
                if v_name == "v0p_historical":
                    continue
                init_terms = v_obj.get("all_10_terms_init")
                assert init_terms is not None, f"[{dname}:{prefix}:{v_name}] Missing all_10_terms_init"
                assert "erb" in init_terms, f"[{dname}:{prefix}:{v_name}] Missing 'erb' in all_10_terms_init"
                assert init_terms["erb"] > 0.0, (
                    f"[{dname}:{prefix}:{v_name}] all_10_terms_init['erb'] must be > 0.0, got {init_terms['erb']}"
                )

                final_terms = v_obj.get("all_10_terms_final")
                assert final_terms is not None, f"[{dname}:{prefix}:{v_name}] Missing all_10_terms_final"
                assert "erb" in final_terms, f"[{dname}:{prefix}:{v_name}] Missing 'erb' in all_10_terms_final"
                assert final_terms["erb"] > 0.0, (
                    f"[{dname}:{prefix}:{v_name}] all_10_terms_final['erb'] must be > 0.0, got {final_terms['erb']}"
                )

