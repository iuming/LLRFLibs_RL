###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Shared environment machinery.

Every LLRF environment in this package shares the same loop-level mechanics:
measurement noise, transport delay of the drive command, and the
observation/reward decoration hooks.  They live here so the concrete
environments stay thin and consistent.
"""

from __future__ import annotations

from typing import List, Optional

import gymnasium as gym
import numpy as np

__all__ = ["BaseLLRFEnv"]


class BaseLLRFEnv(gym.Env):
    """Base class providing the shared LLRF loop helpers.

    Subclasses must set ``self.config`` to one of the environment
    configurations (which expose at least ``measurement_noise_std`` and
    ``loop_delay_steps``) and initialise ``self._drive_cmd_hist``.
    """

    metadata = {"render_modes": []}

    # ------------------------------------------------------------------
    # Loop helpers
    # ------------------------------------------------------------------

    def _measure(self, value: complex, obs_scale: float) -> complex:
        """Add measurement noise to a complex quantity.

        Args:
            value:     True quantity.
            obs_scale: Normalization scale of the measured quantity.

        Returns:
            Noisy measurement.
        """
        if self.env_cfg.measurement_noise_std <= 0.0:
            return value
        std = self.env_cfg.measurement_noise_std * obs_scale
        noise = complex(
            float(self.np_random.normal(0.0, std)),
            float(self.np_random.normal(0.0, std)),
        )
        return value + noise

    def _delay_drive(self, command: complex) -> complex:
        """Apply the discrete transport delay to a drive command.

        Args:
            command: Latest drive command.

        Returns:
            The command that is actually applied this sample.
        """
        delay = int(self.env_cfg.loop_delay_steps)
        self._drive_cmd_hist.append(command)
        if delay <= 0:
            return command
        if len(self._drive_cmd_hist) <= delay:
            return 0.0 + 0.0j
        return self._drive_cmd_hist[-(delay + 1)]

    def _init_drive_history(self) -> None:
        """Reset the drive-delay buffer (call in ``reset``)."""
        self._drive_cmd_hist: List[complex] = []

    def _clamp_action(self, action: np.ndarray) -> np.ndarray:
        """Clip an action to the action space bounds.

        Args:
            action: Raw action array.

        Returns:
            Clipped float32 action.
        """
        low = np.asarray(self.action_space.low, dtype=np.float32)
        high = np.asarray(self.action_space.high, dtype=np.float32)
        return np.clip(np.asarray(action, dtype=np.float32), low, high)

    # ------------------------------------------------------------------
    # Public status accessors (used by scenario / evaluation helpers)
    # ------------------------------------------------------------------

    @property
    def drive_total(self) -> complex:
        """Latest applied drive (complex)."""
        return getattr(self, "_drive_total", 0.0 + 0.0j)

    @property
    def config(self):
        """Alias of ``env_cfg`` (loop-level configuration)."""
        return self.env_cfg

    # ------------------------------------------------------------------
    # Gymnasium metadata helpers
    # ------------------------------------------------------------------

    def render(self, mode: Optional[str] = None) -> None:
        """Rendering is not implemented for the LLRF environments."""
        raise NotImplementedError("LLRF environments do not implement rendering.")

    def close(self) -> None:
        """No resources to release."""
        return None
