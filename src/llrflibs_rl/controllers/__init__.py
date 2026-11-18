###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Controller layer.

Classical baseline controllers (feedforward, PI, ADRC, tuner actuator, ILC)
that are used to close the inner loops of the environments and to benchmark
RL agents against.  These components do not depend on any RL code.
"""

from .baselines import (
    BASELINE_NAMES,
    list_baselines,
    make_baseline_policy,
    make_detuning_pi_policy,
    make_pi_residual_policy,
    zero_policy,
)
from .feedback import ADRCController, ComplexPIController
from .feedforward import AnalyticFeedforward, PTypeILC
from .resonance import FirstOrderActuator, ZeroActuator

__all__ = [
    "ADRCController",
    "AnalyticFeedforward",
    "BASELINE_NAMES",
    "ComplexPIController",
    "FirstOrderActuator",
    "PTypeILC",
    "ZeroActuator",
    "list_baselines",
    "make_adrc_like_policy",
    "make_baseline_policy",
    "make_detuning_pi_policy",
    "make_pi_residual_policy",
    "zero_policy",
]
