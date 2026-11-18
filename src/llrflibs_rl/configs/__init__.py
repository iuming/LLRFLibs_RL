###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Configuration layer.

All configuration objects are plain dataclasses so they can be validated,
serialized to JSON/Markdown and loaded back from files — this is what makes
experiments reproducible.
"""

from .agent import AgentConfig
from .cavity import CavityParams, nominal_nc_cavity, nominal_sc_cavity
from .controller import ADRCControllerConfig, PIControllerConfig, TunerActuatorConfig
from .env import (
    DualLoopEnvConfig,
    EnvConfig,
    FrequencyEnvConfig,
    ResidualEnvConfig,
    VectorSumEnvConfig,
)
from .task import RewardConfig

__all__ = [
    "ADRCControllerConfig",
    "AgentConfig",
    "CavityParams",
    "DualLoopEnvConfig",
    "EnvConfig",
    "FrequencyEnvConfig",
    "PIControllerConfig",
    "ResidualEnvConfig",
    "RewardConfig",
    "TunerActuatorConfig",
    "VectorSumEnvConfig",
    "nominal_nc_cavity",
    "nominal_sc_cavity",
]
