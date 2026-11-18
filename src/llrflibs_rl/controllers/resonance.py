###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Slow-loop actuators and rule-based baselines.

The tuner / piezo is modeled as a first-order actuator whose normalized
command (in ``[-1, 1]``) maps to a detuning set-point.  The :class:`ZeroActuator`
is the "no compensation" baseline used by the resonance / dual-loop tasks.
"""

from __future__ import annotations

import numpy as np

from ..configs.controller import TunerActuatorConfig

__all__ = ["FirstOrderActuator", "ZeroActuator"]


class FirstOrderActuator:
    """First-order tuner / piezo actuator.

    Args:
        config: Actuator parameters.
        Ts:     Sampling time, s.
    """

    def __init__(self, config: TunerActuatorConfig, Ts: float) -> None:
        self.config = config
        self.Ts = float(Ts)
        self.alpha = float(np.exp(-self.Ts / max(config.tau, self.Ts)))
        self.detuning_limit = float(config.detuning_limit)
        self.reset()

    def reset(self) -> None:
        """Zero the actuator state."""
        self.state = 0.0

    def step(self, command: float) -> float:
        """Advance the actuator with a normalized command.

        Args:
            command: Normalized actuator command in ``[-1, 1]``.

        Returns:
            The new actuator detuning, rad/s.
        """
        target = float(np.clip(command, -1.0, 1.0)) * self.detuning_limit
        self.state = self.alpha * self.state + (1.0 - self.alpha) * target
        return self.state


class ZeroActuator:
    """No-compensation baseline: always returns zero detuning."""

    def reset(self) -> None:
        """No-op."""
        self.state = 0.0

    def step(self, command: float) -> float:
        """Return zero regardless of the command.

        Args:
            command: Ignored.

        Returns:
            Always ``0.0``.
        """
        self.state = 0.0
        return 0.0
