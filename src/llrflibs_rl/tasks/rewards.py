###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Reward shaping.

A :class:`RegionScaler` provides the smooth pulse-region weight
(filling / flattop / decay) and a :class:`RewardFactory` turns raw physical
quantities into the per-step reward of an environment.  The weights live in
:class:`llrflibs_rl.configs.task.RewardConfig`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..configs.task import RewardConfig
from .events import phase_error_norm

__all__ = ["RegionScaler", "RewardFactory"]


class RegionScaler:
    """Smooth pulse-region weight.

    The weight ramps linearly over ``ramp`` samples around each region
    boundary instead of jumping, which gives the agent's value estimator a
    smoother reward profile at the filling/flattop/decay transitions.

    Args:
        filling_len: Filling length in samples.
        flattop_len: Flattop length in samples.
        scales:      ``(filling_scale, flattop_scale, decay_scale)``.
        ramp:        Transition length in samples (0 disables smoothing).
    """

    def __init__(
        self,
        filling_len: int,
        flattop_len: int,
        scales: Tuple[float, float, float],
        ramp: int = 0,
    ) -> None:
        self.filling_len = int(filling_len)
        self.flattop_len = int(flattop_len)
        self.filling_scale, self.flattop_scale, self.decay_scale = scales
        self.ramp = max(0, int(ramp))

    def __call__(self, step_index: int) -> float:
        """Return the region weight for a given sample index.

        Args:
            step_index: Current sample index.

        Returns:
            The region weight.
        """
        if self.ramp <= 0:
            if step_index < self.filling_len:
                return float(self.filling_scale)
            if step_index < self.filling_len + self.flattop_len:
                return float(self.flattop_scale)
            return float(self.decay_scale)

        r = max(1, self.ramp)
        b0 = self.filling_len
        b1 = self.filling_len + self.flattop_len
        if step_index < b0:
            t = float(np.clip((b0 - step_index) / r, 0.0, 1.0))
            return float(self.filling_scale * t + self.flattop_scale * (1.0 - t))
        if step_index < b1:
            t = float(np.clip((b1 - step_index) / r, 0.0, 1.0))
            return float(self.flattop_scale * t + self.decay_scale * (1.0 - t))
        return float(self.decay_scale)


class RewardFactory:
    """Compute the per-step reward from raw physical quantities.

    Args:
        config:              Reward weights.
        obs_scale:           Voltage normalization scale (e.g. vector-sum
                             target or single-cavity voltage target).
        filling_len:         Filling length in samples.
        flattop_len:         Flattop length in samples.
        drive_limit:         Max residual drive magnitude.
        drive_effort_scale:  Scale used to normalize the drive-effort term.
        drive_effort_reference: ``"ff"`` (deviation from the nominal
                             feedforward) or ``"total"``.
    """

    def __init__(
        self,
        config: RewardConfig,
        obs_scale: float,
        filling_len: int,
        flattop_len: int,
        drive_limit: float,
        drive_effort_scale: float,
        drive_effort_reference: str = "ff",
    ) -> None:
        self.config = config
        self.obs_scale = max(obs_scale, 1.0)
        self.drive_limit = max(drive_limit, 1.0)
        self.drive_effort_scale = max(drive_effort_scale, 1.0)
        self.drive_effort_reference = drive_effort_reference
        self.region = RegionScaler(
            filling_len, flattop_len, config.region_scales, config.region_ramp
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def compute(
        self,
        *,
        step_index: int,
        vc: complex,
        sp: complex,
        drive_total: complex,
        vf_ff: complex,
        reflected: complex,
        residual: complex,
        sat_excess: float = 0.0,
        slew_excess: float = 0.0,
        detuning: Optional[float] = None,
        detuning_limit: Optional[float] = None,
        actuator_state: Optional[float] = None,
    ) -> float:
        """Compute the reward of one step.

        Args:
            step_index:     Current sample index.
            vc:             Controlled voltage (single cavity or vector sum).
            sp:             Set-point at this sample.
            drive_total:    Actually applied drive.
            vf_ff:          Nominal feedforward drive at this sample.
            reflected:      Reflected voltage (complex).
            residual:       RL residual drive.
            sat_excess:     Bounded saturation excess in ``[0, 1]``.
            slew_excess:    Bounded slew excess in ``[0, 1]``.
            detuning:       Residual detuning (rad/s) for resonance tasks.
            detuning_limit: Detuning normalization scale (rad/s).
            actuator_state: Actuator state (rad/s) for the effort term.

        Returns:
            The (non-positive) per-step reward.
        """
        cfg = self.config
        region = self.region(step_index)

        amp_error = float((abs(vc) - abs(sp)) / self.obs_scale)
        phase_error = phase_error_norm(sp, vc, threshold=0.05 * self.obs_scale)
        error_term = cfg.error_weight * (
            cfg.amp_weight * amp_error**2 + cfg.phase_weight * phase_error**2
        )

        if self.drive_effort_reference == "ff":
            drive_norm = abs(drive_total - vf_ff) / self.drive_effort_scale
        else:
            drive_norm = abs(drive_total) / self.drive_effort_scale
        reflected_norm = abs(reflected) / max(abs(drive_total), self.obs_scale, 1.0)
        residual_norm = abs(residual) / self.drive_limit

        reward = -(
            region
            * (
                error_term
                + cfg.drive_weight * drive_norm**2
                + cfg.reflected_weight * reflected_norm**2
                + cfg.residual_weight * residual_norm**2
            )
            + cfg.saturation_weight * (sat_excess**2 + slew_excess**2)
        )

        if detuning is not None and detuning_limit is not None and detuning_limit > 0:
            detuning_norm = detuning / detuning_limit
            reward -= cfg.detuning_weight * float(detuning_norm) ** 2
            if actuator_state is not None:
                action_norm = actuator_state / detuning_limit
                reward -= cfg.action_weight * float(action_norm) ** 2

        return float(reward)
