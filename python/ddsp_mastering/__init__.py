"""DDSP White-Box AutoMastering Package
Politecnico di Milano - Music and Acoustic Engineering
"""

from .param_space import (
    NUM_PARAMS,
    PARAM_SPECS,
    map_z_to_theta,
    map_theta_to_z,
    unpack_theta_dict,
)

__all__ = [
    "NUM_PARAMS",
    "PARAM_SPECS",
    "map_z_to_theta",
    "map_theta_to_z",
    "unpack_theta_dict",
]
