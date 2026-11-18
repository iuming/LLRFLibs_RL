###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Reward / task semantics configuration.

The reward of every environment is a weighted sum of physical penalty terms
(tracking error, drive effort, reflected power, residual action, saturation
and — for resonance tasks — residual detuning and actuator effort).  The
weights and the pulse-region weighting are collected here so they can be tuned
and swept without touching the environment code.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["RewardConfig"]


@dataclass
class RewardConfig:
    """Weights of the reward terms shared by the LLRF environments.

    The baseline formula used by all voltage/residual envs is::

        reward = -(
            region(step) * (
                error_weight * (amp_weight * amp_err**2 + phase_weight * phase_err**2)
                + drive_weight      * drive_norm**2
                + reflected_weight  * reflected_norm**2
                + residual_weight   * residual_norm**2
            )
            + saturation_weight * (sat_excess**2 + slew_excess**2)
        )

    plus, for resonance / dual-loop envs:
        + detuning_weight * (detuning_residual / detuning_limit)**2
        + action_weight   * (actuator_state   / detuning_limit)**2

    ``region(step)`` is a smooth pulse-region weight (filling / flattop /
    decay) with ``region_ramp`` samples of linear transition.

    Attributes:
        error_weight:      Weight of the tracking error.
        amp_weight:        Weight of the amplitude part of the error.
        phase_weight:      Weight of the phase part of the error.
        drive_weight:      Weight of the drive-effort term.
        reflected_weight:  Weight of the reflected-power term.
        residual_weight:   Weight of the RL residual drive.
        saturation_weight: Weight of the (bounded) saturation / slew term.
        filling_scale:     Region weight during filling.
        flattop_scale:     Region weight during the flattop.
        decay_scale:       Region weight during the decay.
        region_ramp:       Samples of smooth transition between regions.
        detuning_weight:   Weight of the residual-detuning term (resonance envs).
        action_weight:     Weight of the actuator-effort term (resonance envs).
        voltage_weight:    Weight of the amplitude error (dual-loop envs).
    """

    error_weight: float = 10.0
    amp_weight: float = 1.0
    phase_weight: float = 0.35
    drive_weight: float = 0.0
    reflected_weight: float = 0.005
    residual_weight: float = 0.02
    saturation_weight: float = 0.20
    filling_scale: float = 0.5
    flattop_scale: float = 2.2
    decay_scale: float = 0.02
    region_ramp: int = 25
    detuning_weight: float = 3.0
    action_weight: float = 0.02
    voltage_weight: float = 1.0

    @property
    def region_scales(self) -> tuple[float, float, float]:
        """Return ``(filling_scale, flattop_scale, decay_scale)``."""
        return (self.filling_scale, self.flattop_scale, self.decay_scale)
