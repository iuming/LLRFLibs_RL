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
from .gain_scheduling import GainSchedulingEnv
from .pulse_to_pulse import AdaptivePulseEnv, BeamLoadingEnv
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
_registry.register(
    "BeamLoading-v0",
    description="Beam-loading transient: the beam steps on during the flattop while the feedforward is beam-free.",
    entry=BeamLoadingEnv,
    kind="voltage",
)
_registry.register(
    "AdaptivePulse-v0",
    description="Pulse-to-pulse adaptive feedforward under a drifting cavity detuning.",
    entry=AdaptivePulseEnv,
    kind="pulse",
)
_registry.register(
    "VectorSumTuner-v0",
    description="Vector-sum module with a shared RF source and per-cavity slow tuner actuation.",
    entry=VectorSumEnv,
    kind="vectormodule",
)
_registry.register(
    "GainScheduling-v0",
    description="Single-cavity tracking where RL schedules the inner-loop PI gains.",
    entry=GainSchedulingEnv,
    kind="voltage",
)

# Short-horizon (episode 1200) resolution variants of the core environments.
_registry.register(
    "CavityVoltage-v1",
    description="CavityVoltage-v0 with a shorter (1400-sample) episode.",
    entry=CavityVoltageEnv,
    kind="voltage",
)
_registry.register(
    "PIAssist-v1",
    description="PIAssist-v0 with a shorter (1400-sample) episode.",
    entry=PIAssistEnv,
    kind="voltage",
)
_registry.register(
    "ADRCAssist-v1",
    description="ADRCAssist-v0 with a shorter (1400-sample) episode.",
    entry=ADRCAssistEnv,
    kind="voltage",
)
_registry.register(
    "DualLoop-v1",
    description="DualLoop-v0 with a shorter (1400-sample) episode.",
    entry=DualLoopEnv,
    kind="resonance",
)
_registry.register(
    "VectorSum-v1",
    description="VectorSum-v0 with a shorter (1400-sample) episode.",
    entry=VectorSumEnv,
    kind="vectormodule",
)
_registry.register(
    "AdaptivePulse-v1",
    description="AdaptivePulse-v0 with a longer 20-pulse adaptation window.",
    entry=AdaptivePulseEnv,
    kind="pulse",
)

__all__ = [
    "ADRCAssistEnv",
    "AdaptivePulseEnv",
    "BaseLLRFEnv",
    "BeamLoadingEnv",
    "CavityVoltageEnv",
    "DualLoopEnv",
    "EpisodicMonitor",
    "FrequencyEnv",
    "GainSchedulingEnv",
    "PIAssistEnv",
    "VectorSumEnv",
]
