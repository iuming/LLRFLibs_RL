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
from .feedforward import (
    AnalyticBeamFeedforward,
    AnalyticFeedforward,
    PTypeILC,
    beam_keepup_correction,
)
from .ilc import AdaptiveFeedforward, ILCFeedforward, design_ilc_gain, impulse_response_matrix
from .notch import NotchFilter, notch_conf_for_frequencies, notch_filter_wave
from .resonance import FirstOrderActuator, ZeroActuator

__all__ = [
    "ADRCController",
    "AdaptiveFeedforward",
    "AnalyticBeamFeedforward",
    "AnalyticFeedforward",
    "BASELINE_NAMES",
    "ComplexPIController",
    "FirstOrderActuator",
    "ILCFeedforward",
    "NotchFilter",
    "PTypeILC",
    "ZeroActuator",
    "beam_keepup_correction",
    "design_ilc_gain",
    "impulse_response_matrix",
    "list_baselines",
    "make_adrc_like_policy",
    "make_baseline_policy",
    "make_detuning_pi_policy",
    "make_pi_residual_policy",
    "notch_conf_for_frequencies",
    "notch_filter_wave",
    "zero_policy",
]
