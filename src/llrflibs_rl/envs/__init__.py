###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Environment layer.

Importing this package registers the built-in environments in the shared
registry (:mod:`llrflibs_rl.registry`), so ``llrflibs_rl.make(...)`` works
once the package is imported (the top-level ``llrflibs_rl`` import does that).
"""

from .core import BaseLLRFEnv
from .resonance import DualLoopEnv, FrequencyEnv
from .vectormodule import VectorSumEnv
from .voltage import ADRCAssistEnv, CavityVoltageEnv, PIAssistEnv
from .wrappers import EpisodicMonitor

# Register every built-in environment in the package registry.
from .. import registry as _registry

_registry.register(
    "CavityVoltage-v0",
    description="Single-cavity fast RF voltage tracking with RL residual on the analytic feedforward.",
    entry=CavityVoltageEnv,
    kind="voltage",
)
_registry.register(
    "PIAssist-v0",
    description="Complex-PI inner loop with an RL residual assist.",
    entry=PIAssistEnv,
    kind="voltage",
)
_registry.register(
    "ADRCAssist-v0",
    description="ADRC inner loop with an RL residual assist.",
    entry=ADRCAssistEnv,
    kind="voltage",
)
_registry.register(
    "FrequencyControl-v0",
    description="Resonance / detuning compensation with an RL slow tuner actuator.",
    entry=FrequencyEnv,
    kind="resonance",
)
_registry.register(
    "DualLoop-v0",
    description="Fast RF PI loop + RL slow tuner (hierarchical LLRF cooperation).",
    entry=DualLoopEnv,
    kind="resonance",
)
_registry.register(
    "VectorSum-v0",
    description="Multi-cavity vector-sum module control with a shared RF source.",
    entry=VectorSumEnv,
    kind="vectormodule",
)

__all__ = [
    "ADRCAssistEnv",
    "BaseLLRFEnv",
    "CavityVoltageEnv",
    "DualLoopEnv",
    "EpisodicMonitor",
    "FrequencyEnv",
    "PIAssistEnv",
    "VectorSumEnv",
]
