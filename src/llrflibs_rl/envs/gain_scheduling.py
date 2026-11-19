###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Gain-scheduling environment.

Instead of adding an RL residual on top of a tuned inner loop (which the
strong-baseline experiments showed has little headroom), :class:`GainSchedulingEnv`
lets RL **choose the inner-loop gains** ``kp`` and ``ki`` sample by sample.
The agent therefore *shapes the controller* rather than fighting it — the
channel through which RL can genuinely improve a classical LLRF loop
(raise gain during flattop, back off through transients, adapt to detuning).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from gymnasium import spaces

from ..configs.cavity import CavityParams
from ..configs.env import ResidualEnvConfig
from ..configs.task import RewardConfig
from ..controllers.feedback import ScheduledPIController
from ..tasks.events import apply_drive_limits
from ..tasks.observations import build_assist_obs
from .voltage import CavityVoltageEnv

__all__ = ["GainSchedulingConfig", "GainSchedulingEnv"]


@dataclass
class GainSchedulingConfig(ResidualEnvConfig):
    """Configuration of the gain-scheduling environment.

    Attributes:
        kp_nom:            Nominal proportional gain (the ``pi`` baseline).
        ki_nom:            Nominal integral gain.
        gain_scale_factor: RL gains span ``[nom/scale, nom*scale]``.
        gain_effort_weight: Penalty weight of the mean squared action.
    """

    kp_nom: float = 0.8
    ki_nom: float = 2.0e5
    gain_scale_factor: float = 4.0
    gain_effort_weight: float = 0.1

    def validate(self) -> None:
        super().validate()
        if self.kp_nom <= 0 or self.ki_nom <= 0:
            raise ValueError("kp_nom and ki_nom must be positive.")
        if self.gain_scale_factor <= 1.0:
            raise ValueError("gain_scale_factor must be > 1.")
        if self.gain_effort_weight < 0:
            raise ValueError("gain_effort_weight must be non-negative.")

    def __post_init__(self) -> None:
        self.validate()


class GainSchedulingEnv(CavityVoltageEnv):
    """Single-cavity voltage tracking with RL-scheduled inner-loop gains.

    Observation (19,):
        ``[vc, sp, err, vr, vf_ff, drive_fb, drive_rl(=0), vb, t,
        log-norm(kp), log-norm(ki)]``.
    Action (2,):                   normalized gain factors in ``[-1, 1]``
        mapping to ``kp = kp_nom * scale**a0`` and ``ki = ki_nom * scale**a1``.

    The ``pi`` baseline of :mod:`llrflibs_rl.controllers.baselines` evaluates
    exactly the fixed-gain PI (gain factor 1), so ``RL vs pi`` is the honest
    "does adaptive scheduling beat fixed gains" comparison.

    Args:
        cavity:     Cavity description.
        env_cfg:    Gain-scheduling loop configuration.
        reward_cfg: Reward weights.
    """

    def __init__(
        self,
        cavity: CavityParams,
        env_cfg: GainSchedulingConfig = GainSchedulingConfig(),
        reward_cfg: RewardConfig = RewardConfig(),
    ) -> None:
        super().__init__(cavity, env_cfg, reward_cfg)
        self.gs_cfg = env_cfg
        self._controller = ScheduledPIController(env_cfg.kp_nom, env_cfg.ki_nom, env_cfg.Ts)
        self._meas_vc = 0.0 + 0.0j
        self._drive_fb = 0.0 + 0.0j
        self._drive_rl = 0.0 + 0.0j
        self._kp = float(env_cfg.kp_nom)
        self._ki = float(env_cfg.ki_nom)
        self._last_gain_action = np.zeros(2, dtype=np.float32)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _gains_from_action(self, action: np.ndarray) -> tuple[float, float]:
        scale = float(self.gs_cfg.gain_scale_factor)
        kp = self.gs_cfg.kp_nom * scale ** float(np.clip(action[0], -1.0, 1.0))
        ki = self.gs_cfg.ki_nom * scale ** float(np.clip(action[1], -1.0, 1.0))
        return kp, ki

    def _get_obs(self) -> np.ndarray:
        idx = min(self._step_index, self.episode_len - 1)
        sp = self._reference.sp_at(idx)
        vf_ff = self._reference.vf_ff_at(idx)
        vb = self._reference.vb_at(idx)
        err = sp - self._plant.vc
        obs = build_assist_obs(
            vc=self._plant.vc,
            sp=sp,
            err=err,
            vr=self._plant.vr,
            vf_ff=vf_ff,
            drive_fb=self._drive_fb,
            drive_rl=self._drive_rl,
            vb=vb,
            obs_scale=self.obs_scale,
            drive_limit=self.drive_limit,
            step_index=self._step_index,
            episode_len=self.episode_len,
        )
        scale = float(self.gs_cfg.gain_scale_factor)
        norm_kp = float(np.log(self._kp / self.gs_cfg.kp_nom) / np.log(scale))
        norm_ki = float(np.log(self._ki / self.gs_cfg.ki_nom) / np.log(scale))
        return np.concatenate([obs, np.asarray([norm_kp, norm_ki], dtype=np.float32)])

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        # Reset the controller state BEFORE super().reset() captures the first
        # observation (see PIAssistEnv.reset).
        self._controller.reset()
        self._meas_vc = 0.0 + 0.0j
        self._drive_fb = 0.0 + 0.0j
        self._drive_rl = 0.0 + 0.0j
        self._kp = float(self.gs_cfg.kp_nom)
        self._ki = float(self.gs_cfg.ki_nom)
        self._last_gain_action = np.zeros(2, dtype=np.float32)
        obs, info = super().reset(seed=seed, options=options)
        self._meas_vc = self._measure(self._plant.vc, self.obs_scale)
        self._drive_fb = 0.0 + 0.0j
        self._drive_rl = 0.0 + 0.0j
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, float]]:
        action = self._clamp_action(action)
        self._last_gain_action = action
        self._kp, self._ki = self._gains_from_action(action)

        idx = min(self._step_index, self.episode_len - 1)
        sp = self._reference.sp_at(idx)
        vf_ff = self._reference.vf_ff_at(idx)
        err = sp - self._meas_vc
        drive_pi, drive_fb = self._controller.step(err, vf_ff, kp=self._kp, ki=self._ki)
        self._drive_fb = drive_fb
        self._drive_rl = 0.0 + 0.0j

        unclipped = drive_pi
        drive, sat_excess, slew_excess = apply_drive_limits(
            unclipped, self._drive_total, self.env_cfg.max_drive_abs, self.env_cfg.max_drive_slew_per_step
        )
        drive = self._delay_drive(drive)

        beam = self._beam_at(idx)
        dw = float(self._detuning_wave[idx]) if self._detuning_wave is not None else None
        self._plant.step(drive, beam=beam, detuning=dw)
        self._meas_vc = self._measure(self._plant.vc, self.obs_scale)
        self._drive_total = drive
        self._step_index += 1

        reward = self.reward.compute(
            step_index=idx,
            vc=self._plant.vc,
            sp=sp,
            drive_total=self._drive_total,
            vf_ff=vf_ff,
            reflected=self._plant.vr,
            residual=0.0 + 0.0j,
            sat_excess=sat_excess,
            slew_excess=slew_excess,
        )
        reward -= self.gs_cfg.gain_effort_weight * float(np.mean(action**2))
        terminated = False
        truncated = self._step_index >= self.episode_len
        return self._get_obs(), float(reward), terminated, truncated, self._make_info()
