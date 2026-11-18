###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Task layer.

The task layer translates the physical world (plants) into the MDP interface
(reference waveforms, observations, rewards and event handling).
"""

from .events import apply_drive_limits, phase_error_norm
from .observations import (
    build_assist_obs,
    build_dual_loop_obs,
    build_frequency_obs,
    build_vector_sum_obs,
    build_voltage_obs,
)
from .reference import PulseReference
from .rewards import RegionScaler, RewardFactory

__all__ = [
    "PulseReference",
    "RegionScaler",
    "RewardFactory",
    "apply_drive_limits",
    "build_assist_obs",
    "build_dual_loop_obs",
    "build_frequency_obs",
    "build_vector_sum_obs",
    "build_voltage_obs",
    "phase_error_norm",
]
