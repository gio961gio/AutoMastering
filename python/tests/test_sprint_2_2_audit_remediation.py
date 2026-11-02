"""
Sprint 2.2 Audit Remediation Tests:
- F-10: Clean metrics.py without dead code.
- F-11: C++ THETA_NEUTRAL and MasteringParameters default initialization consistency.
"""

import numpy as np
import pytest
import torch

from ddsp_mastering.param_space import THETA_NEUTRAL, Z_NEUTRAL, NUM_PARAMS
import py_core_dsp


def test_f11_cpp_theta_neutral_matches_python_and_default_struct():
    """Verify F-11: C++ THETA_NEUTRAL and default MasteringParameters match Python THETA_NEUTRAL."""
    cpp_theta_neutral = np.array(py_core_dsp.get_theta_neutral(), dtype=np.float64)
    cpp_default_struct_theta = np.array(py_core_dsp.get_default_mastering_parameters_theta(), dtype=np.float64)
    py_theta_neutral = THETA_NEUTRAL.detach().cpu().numpy()

    assert len(cpp_theta_neutral) == NUM_PARAMS == 41
    assert len(cpp_default_struct_theta) == NUM_PARAMS == 41

    # Check perfect numerical match between C++ THETA_NEUTRAL and Python THETA_NEUTRAL
    np.testing.assert_allclose(cpp_theta_neutral, py_theta_neutral, rtol=1e-12, atol=1e-12)

    # Check that default MasteringParameters{} struct exactly matches THETA_NEUTRAL
    np.testing.assert_allclose(cpp_default_struct_theta, py_theta_neutral, rtol=1e-12, atol=1e-12)
