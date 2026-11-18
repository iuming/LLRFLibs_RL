###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Controller configuration for the classical baseline controllers.

These controllers are used to build the "inner loop" of the assist / dual-loop
environments and as evaluation baselines.  They wrap the primitives in
``llrflibs.rf_control`` and are fully unit-testable without RL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

__all__ = [
    "PIControllerConfig",
    "ADRCControllerConfig",
    "TunerActuatorConfig",
]


@dataclass
class PIControllerConfig:
    """Complex PI controller configuration (fast RF loop).

    Attributes:
        kp:         Proportional gain of the complex PI controller.
        ki:         Integral gain of the complex PI controller.
        notch_conf: Optional notch-filter configuration consumed by
                    ``llrflibs.rf_control.basic_rf_controller``.
    """

    kp: float = 0.8
    ki: float = 2.0e5
    notch_conf: Optional[Dict[str, List[float]]] = None

    def validate(self) -> None:
        if self.kp < 0 or self.ki < 0:
            raise ValueError("kp and ki must be non-negative.")

    def __post_init__(self) -> None:
        self.validate()


@dataclass
class ADRCControllerConfig:
    """Active-disturbance-rejection (ADRC) controller configuration.

    Attributes:
        kp:                 Controller proportional gain used with the ADRC
                            observer (in the error form).
        ki:                 Controller integral gain used with the ADRC
                            observer.
        observer_pole_scale: Observer pole location as a multiple of the
                            cavity half bandwidth.
        apply_to_err:       Apply ADRC to the error (True) or to the full
                            cavity voltage.
    """

    kp: float = 0.6
    ki: float = 1.0e5
    observer_pole_scale: float = 200.0
    apply_to_err: bool = True

    def validate(self) -> None:
        if self.kp < 0 or self.ki < 0:
            raise ValueError("kp and ki must be non-negative.")
        if self.observer_pole_scale <= 0:
            raise ValueError("observer_pole_scale must be positive.")

    def __post_init__(self) -> None:
        self.validate()


@dataclass
class TunerActuatorConfig:
    """First-order tuner / piezo actuator configuration (slow loop).

    Attributes:
        detuning_limit: Max actuator detuning (maps action [-1, 1]), rad/s.
        tau:            First-order time constant, s.
    """

    detuning_limit: float = 2.0 * np.pi * 3.0e3
    tau: float = 8.0e-5

    def validate(self) -> None:
        if self.detuning_limit <= 0:
            raise ValueError("detuning_limit must be positive.")
        if self.tau <= 0:
            raise ValueError("tau must be positive.")

    def __post_init__(self) -> None:
        self.validate()
