###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Full LLRF control-loop environment.

This task simulates the closed loop of the LLRF system schematic
(``llrf_control_loop_standalone``): the **fast FPGA loop** (PI + feedforward +
RL residual -> DAC -> I/Q modulator -> SSA -> cavity, closed by the
pick-up/mixer/ADC/I-Q-detection feedback chain) together with the **slow tuner
loop** that compensates the cavity detuning.

The RL action space is the two-channel picture described in the schematic:

    ``Box(-1, 1, (3,)) = [tuner, RF_I, RF_Q]``

* ``tuner``   — normalized command of the slow mechanical tuner (actuator in
  rad/s), the resonance/microphonics compensation channel;
* ``RF_I`` / ``RF_Q`` — normalized I/Q residual of the FPGA RF drive signal,
  added on top of the in-loop complex-PI feedback and the analytic feedforward
  (the "FPGA output RF signal" channel).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from ..configs.cavity import CavityParams
from ..configs.env import LoopEnvConfig
from ..configs.task import RewardConfig
from ..tasks.events import apply_drive_limits
from .resonance import DualLoopEnv

__all__ = ["LLRFLoopEnv"]


class LLRFLoopEnv(DualLoopEnv):
    """Fast FPGA RF loop (PI+FF+RL residual) with an RL slow tuner.

    Observation (15,): the 13 dual-loop features
    ``[vc(2), sp(2), err(2), vr(2), detuning, actuator, disturbance, drive_fb,
    t]`` plus the normalized RL RF residual ``[RF_I, RF_Q]``.
    Action (3,):       ``[tuner, RF_I, RF_Q]``.

    Args:
        cavity:     Cavity description (mechanical modes used if present).
        env_cfg:    Loop configuration (inner PI gains + tuner + residual scale).
        reward_cfg: Reward weights.
    """

    def __init__(
        self,
        cavity: CavityParams,
        env_cfg: LoopEnvConfig = LoopEnvConfig(),
        reward_cfg: RewardConfig = RewardConfig(),
    ) -> None:
        super().__init__(cavity, env_cfg, reward_cfg)
        # Both action channels: tuner (slow loop) + FPGA I/Q RF signal (fast loop).
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.drive_limit = float(env_cfg.drive_limit_ratio * self.obs_scale)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        self._drive_rl = 0.0 + 0.0j
        self._detuning_history: list[float] = []

    # ------------------------------------------------------------------
    # Evaluation protocol (inherits DualLoopEnv properties)
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        base = np.asarray(super()._get_obs(), dtype=np.float32)
        limit = max(self.drive_limit, 1.0)
        residual = np.asarray(
            [self._drive_rl.real / limit, self._drive_rl.imag / limit], dtype=np.float32
        )
        return np.concatenate([base, residual])

    def _make_info(self) -> dict[str, float]:
        info = super()._make_info()
        info["drive_rl_abs"] = float(abs(self._drive_rl))
        return info

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        result = super().reset(seed=seed, options=options)
        self._drive_rl = 0.0 + 0.0j
        self._detuning_history = []
        return result

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, float]]:
        action = self._clamp_action(action)
        # ---- slow channel: tuner actuator ----
        act_state = self._actuator.step(float(action[0]))

        # ---- fast channel: FPGA RF signal = PI feedback + FF + RL residual ----
        residual = complex(float(action[1]), float(action[2])) * self.drive_limit
        self._drive_rl = residual

        idx = min(self._step_index, self.episode_len - 1)
        sp = self._reference.sp_at(idx)
        vf_ff = self._reference.vf_ff_at(idx)
        beam = self._reference.vb_at(idx)
        err = sp - self._meas_vc
        drive_pi, drive_fb = self._controller.step(err, vf_ff)
        self._drive_fb = drive_fb

        drive, sat_excess, slew_excess = apply_drive_limits(
            drive_pi + residual,
            self._drive_total,
            self.env_cfg.max_drive_abs,
            self.env_cfg.max_drive_slew_per_step,
        )
        drive = self._delay_drive(drive)

        external = float(self._dist_wave[idx]) + act_state
        vc, vr, dw = self._plant.step(drive, beam, external)
        self._drive_total = drive
        self._meas_vc = self._measure(self._plant.vc, self.obs_scale)
        self._last_error = err
        self._step_index += 1
        self._detuning_history.append(float(dw))

        reward = self.reward.compute(
            step_index=idx,
            vc=vc,
            sp=sp,
            drive_total=self._drive_total,
            vf_ff=vf_ff,
            reflected=vr,
            residual=residual,
            sat_excess=sat_excess,
            slew_excess=slew_excess,
            detuning=dw,
            detuning_limit=self.detuning_limit,
            actuator_state=act_state,
        )
        terminated = False
        truncated = self._episode_ended()
        return self._get_obs(), float(reward), terminated, truncated, self._make_info()
