###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Single-cavity voltage environments.

Three control structures are provided for the fast RF voltage loop:

* :class:`CavityVoltageEnv`   — RL residual on top of the analytic feedforward.
* :class:`PIAssistEnv`        — complex-PI inner loop + RL residual assist.
* :class:`ADRCAssistEnv`      — ADRC inner loop + RL residual assist.

All of them share the same loop mechanics (noise, delay, drive constraints)
and the same physics-aware reward; only the "who provides the inner drive"
changes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..configs.cavity import CavityParams
from ..configs.controller import ADRCControllerConfig, PIControllerConfig
from ..configs.env import ResidualEnvConfig
from ..configs.task import RewardConfig
from ..controllers.feedback import ADRCController, ComplexPIController
from ..plants.cavity_voltage import CavityVoltagePlant
from ..tasks.events import apply_drive_limits
from ..tasks.observations import build_assist_obs, build_voltage_obs
from ..tasks.reference import PulseReference
from ..tasks.rewards import RewardFactory
from .core import BaseLLRFEnv

__all__ = ["CavityVoltageEnv", "PIAssistEnv", "ADRCAssistEnv"]


class CavityVoltageEnv(BaseLLRFEnv):
    """Track a cavity-voltage pulse with an RL residual correction.

    The plant is the discrete cavity model; the action is a normalized I/Q
    residual added on top of the analytic feedforward.

    Observation (15,): ``[vc, sp, vf_ff, vb, vr, err, last_action(2 each), t]``.
    Action (2,):       normalized I/Q residual in ``[-1, 1]``.

    Args:
        cavity:    Cavity description.
        env_cfg:   Loop-level configuration.
        reward_cfg: Reward weights.
    """

    def __init__(
        self,
        cavity: CavityParams,
        env_cfg: ResidualEnvConfig = ResidualEnvConfig(),
        reward_cfg: RewardConfig = RewardConfig(),
    ) -> None:
        super().__init__()
        self.cavity = cavity
        self.env_cfg = env_cfg
        self.reward_cfg = reward_cfg
        self.episode_len = int(env_cfg.episode_len)

        self._reference = PulseReference(
            cavity, env_cfg.Ts, self.episode_len, env_cfg.reference_detuning
        )
        self.obs_scale = self._reference.obs_scale
        self.drive_limit = float(env_cfg.drive_limit_ratio * self.obs_scale)
        self.drive_effort_scale = self._reference.drive_effort_scale

        self._episode_detuning = float(cavity.detuning)
        self._plant = CavityVoltagePlant(cavity, env_cfg.Ts, self._episode_detuning)
        self.reward = RewardFactory(
            reward_cfg,
            obs_scale=self.obs_scale,
            filling_len=cavity.filling_len,
            flattop_len=cavity.flattop_len,
            drive_limit=self.drive_limit,
            drive_effort_scale=self.drive_effort_scale,
            drive_effort_reference=env_cfg.drive_effort_reference,
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)

        self._detuning_wave: Optional[np.ndarray] = None
        self._last_action = 0.0 + 0.0j
        self._drive_total = 0.0 + 0.0j
        self._step_index = 0
        self._init_drive_history()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_episode_detuning(self) -> float:
        detuning = float(self.cavity.detuning)
        if self.env_cfg.randomize_detuning > 0.0:
            detuning += float(
                self.np_random.uniform(-self.env_cfg.randomize_detuning, self.env_cfg.randomize_detuning)
            )
        return detuning

    def _build_detuning_wave(self, detuning: float) -> None:
        if self.env_cfg.microphonics_amp > 0.0:
            phase = (
                float(self.np_random.uniform(0.0, 2.0 * np.pi))
                if self.env_cfg.microphonics_random_phase
                else 0.0
            )
            t = np.arange(self.episode_len) * self.env_cfg.Ts
            self._detuning_wave = (
                detuning
                + self.env_cfg.microphonics_amp
                * np.sin(2.0 * np.pi * self.env_cfg.microphonics_freq * t + phase)
            )
        else:
            self._detuning_wave = None

    def _get_obs(self) -> np.ndarray:
        idx = min(self._step_index, self.episode_len - 1)
        sp = self._reference.sp_at(idx)
        vf_ff = self._reference.vf_ff_at(idx)
        vb = self._reference.vb_at(idx)
        err = sp - self._plant.vc
        return build_voltage_obs(
            vc=self._plant.vc,
            sp=sp,
            err=err,
            vr=self._plant.vr,
            vf_ff=vf_ff,
            vb=vb,
            last_action=self._last_action,
            obs_scale=self.obs_scale,
            drive_limit=self.drive_limit,
            step_index=self._step_index,
            episode_len=self.episode_len,
        )

    def _make_info(self) -> Dict[str, float]:
        idx = min(self._step_index, self.episode_len - 1)
        err = self._reference.sp_at(idx) - self._plant.vc
        return {
            "step_index": float(self._step_index),
            "vc_abs": float(abs(self._plant.vc)),
            "sp_abs": float(abs(self._reference.sp_at(idx))),
            "error_abs": float(abs(err)),
            "drive_abs": float(abs(self._drive_total)),
            "reflection_abs": float(abs(self._plant.vr)),
            "plant_detuning_hz": float(self._episode_detuning / (2.0 * np.pi)),
            "reference_detuning_hz": float(self._reference.reference_detuning / (2.0 * np.pi)),
        }

    # ------------------------------------------------------------------
    # Evaluation protocol (used by llrflibs_rl.evaluation.metrics)
    # ------------------------------------------------------------------

    @property
    def reference_sp(self) -> np.ndarray:
        """Set-point waveform used for the tracking-error metrics."""
        return self._reference.sp

    @property
    def tracked_signal(self) -> complex:
        """Latest tracked voltage (used by the metrics module)."""
        return self._plant.vc

    @property
    def plant(self) -> CavityVoltagePlant:
        """The underlying cavity plant (status introspection)."""
        return self._plant

    @property
    def reference(self) -> PulseReference:
        """The underlying pulse reference (status introspection)."""
        return self._reference

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        super().reset(seed=seed)
        self._episode_detuning = self._sample_episode_detuning()
        self._plant = CavityVoltagePlant(self.cavity, self.env_cfg.Ts, self._episode_detuning)
        self._plant.reset()
        self._build_detuning_wave(self._episode_detuning)
        self._last_action = 0.0 + 0.0j
        self._drive_total = self._reference.vf_ff_at(0)
        self._step_index = 0
        self._init_drive_history()
        return self._get_obs(), self._make_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, float]]:
        action = self._clamp_action(action)
        residual = complex(float(action[0]), float(action[1])) * self.drive_limit
        self._last_action = residual

        idx = min(self._step_index, self.episode_len - 1)
        vf_ff = self._reference.vf_ff_at(idx)
        unclipped = vf_ff + residual
        drive, sat_excess, slew_excess = apply_drive_limits(
            unclipped, self._drive_total, self.env_cfg.max_drive_abs, self.env_cfg.max_drive_slew_per_step
        )
        drive = self._delay_drive(drive)

        beam = self._reference.vb_at(idx)
        dw = float(self._detuning_wave[idx]) if self._detuning_wave is not None else None
        self._plant.step(drive, beam=beam, detuning=dw)
        self._drive_total = drive
        self._step_index += 1

        sp = self._reference.sp_at(idx)
        reward = self.reward.compute(
            step_index=idx,
            vc=self._plant.vc,
            sp=sp,
            drive_total=self._drive_total,
            vf_ff=vf_ff,
            reflected=self._plant.vr,
            residual=residual,
            sat_excess=sat_excess,
            slew_excess=slew_excess,
        )
        terminated = False
        truncated = self._step_index >= self.episode_len
        return self._get_obs(), float(reward), terminated, truncated, self._make_info()


class PIAssistEnv(CavityVoltageEnv):
    """Complex-PI inner loop with an RL residual assist.

    Structure: ``Vf = Vff + V_PI + ΔV_RL``.

    Observation (17,): ``[vc, sp, err, vr, vf_ff, drive_fb, drive_rl, vb, t]``.

    Args:
        cavity:     Cavity description.
        env_cfg:    Loop-level configuration.
        reward_cfg: Reward weights.
        pi_cfg:     Inner-loop PI parameters.
    """

    def __init__(
        self,
        cavity: CavityParams,
        env_cfg: ResidualEnvConfig = ResidualEnvConfig(),
        reward_cfg: RewardConfig = RewardConfig(),
        pi_cfg: PIControllerConfig = PIControllerConfig(),
    ) -> None:
        super().__init__(cavity, env_cfg, reward_cfg)
        self.pi_cfg = pi_cfg
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)

        self._controller = ComplexPIController(pi_cfg, env_cfg.Ts)
        self._meas_vc = 0.0 + 0.0j
        self._drive_fb = 0.0 + 0.0j
        self._drive_rl = 0.0 + 0.0j
        self._last_error = 0.0 + 0.0j

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        obs, info = super().reset(seed=seed, options=options)
        self._controller.reset()
        self._meas_vc = self._measure(self._plant.vc, self.obs_scale)
        self._drive_fb = 0.0 + 0.0j
        self._drive_rl = 0.0 + 0.0j
        self._last_error = self._reference.sp_at(0)
        return obs, info

    def _get_obs(self) -> np.ndarray:
        idx = min(self._step_index, self.episode_len - 1)
        sp = self._reference.sp_at(idx)
        vf_ff = self._reference.vf_ff_at(idx)
        vb = self._reference.vb_at(idx)
        err = sp - self._plant.vc
        return build_assist_obs(
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

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, float]]:
        action = self._clamp_action(action)
        self._drive_rl = complex(float(action[0]), float(action[1])) * self.drive_limit

        idx = min(self._step_index, self.episode_len - 1)
        sp = self._reference.sp_at(idx)
        vf_ff = self._reference.vf_ff_at(idx)
        err = sp - self._meas_vc
        drive_pi, drive_fb = self._controller.step(err, vf_ff)
        self._drive_fb = drive_fb

        unclipped = drive_pi + self._drive_rl
        drive, sat_excess, slew_excess = apply_drive_limits(
            unclipped, self._drive_total, self.env_cfg.max_drive_abs, self.env_cfg.max_drive_slew_per_step
        )
        drive = self._delay_drive(drive)

        beam = self._reference.vb_at(idx)
        dw = float(self._detuning_wave[idx]) if self._detuning_wave is not None else None
        self._plant.step(drive, beam=beam, detuning=dw)
        self._drive_total = drive
        self._meas_vc = self._measure(self._plant.vc, self.obs_scale)
        self._last_error = err
        self._step_index += 1

        reward = self.reward.compute(
            step_index=idx,
            vc=self._plant.vc,
            sp=sp,
            drive_total=self._drive_total,
            vf_ff=vf_ff,
            reflected=self._plant.vr,
            residual=self._drive_rl,
            sat_excess=sat_excess,
            slew_excess=slew_excess,
        )
        terminated = False
        truncated = self._step_index >= self.episode_len
        return self._get_obs(), float(reward), terminated, truncated, self._make_info()


class ADRCAssistEnv(CavityVoltageEnv):
    """ADRC inner loop with an RL residual assist.

    Structure: ``Vf = Vff + V_ADRC + ΔV_RL``.

    Observation (21,): ``[vc, sp, err, vr, vf_ff, drive_fb, drive_rl, vb,
    vc_est(2), f_est(2), t]``.

    Args:
        cavity:     Cavity description.
        env_cfg:    Loop-level configuration.
        reward_cfg: Reward weights.
        adrc_cfg:   ADRC parameters.
    """

    def __init__(
        self,
        cavity: CavityParams,
        env_cfg: ResidualEnvConfig = ResidualEnvConfig(),
        reward_cfg: RewardConfig = RewardConfig(),
        adrc_cfg: ADRCControllerConfig = ADRCControllerConfig(),
    ) -> None:
        super().__init__(cavity, env_cfg, reward_cfg)
        self.adrc_cfg = adrc_cfg
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32)

        self._controller = ADRCController(adrc_cfg, env_cfg.Ts, cavity.half_bw)
        self._meas_vc = 0.0 + 0.0j
        self._meas_vf = 0.0 + 0.0j
        self._drive_fb = 0.0 + 0.0j
        self._drive_rl = 0.0 + 0.0j
        self._vc_est = 0.0 + 0.0j
        self._f_est = 0.0 + 0.0j

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        obs, info = super().reset(seed=seed, options=options)
        self._controller.reset()
        self._meas_vc = self._measure(self._plant.vc, self.obs_scale)
        self._meas_vf = self._measure(self._drive_total, self.obs_scale)
        self._drive_fb = 0.0 + 0.0j
        self._drive_rl = 0.0 + 0.0j
        self._vc_est = 0.0 + 0.0j
        self._f_est = 0.0 + 0.0j
        return obs, info

    def _get_obs(self) -> np.ndarray:
        idx = min(self._step_index, self.episode_len - 1)
        sp = self._reference.sp_at(idx)
        vf_ff = self._reference.vf_ff_at(idx)
        vb = self._reference.vb_at(idx)
        err = sp - self._plant.vc
        return build_assist_obs(
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
            vc_est=self._vc_est,
            f_est=self._f_est,
        )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, float]]:
        action = self._clamp_action(action)
        self._drive_rl = complex(float(action[0]), float(action[1])) * self.drive_limit

        idx = min(self._step_index, self.episode_len - 1)
        sp = self._reference.sp_at(idx)
        vf_ff = self._reference.vf_ff_at(idx)
        drive_adrc, drive_fb, vc_est, f_est = self._controller.step(
            sp, self._meas_vc, self._meas_vf, vf_ff
        )
        self._drive_fb = drive_fb
        self._vc_est = vc_est
        self._f_est = f_est

        unclipped = drive_adrc + self._drive_rl
        drive, sat_excess, slew_excess = apply_drive_limits(
            unclipped, self._drive_total, self.env_cfg.max_drive_abs, self.env_cfg.max_drive_slew_per_step
        )
        drive = self._delay_drive(drive)

        beam = self._reference.vb_at(idx)
        dw = float(self._detuning_wave[idx]) if self._detuning_wave is not None else None
        self._plant.step(drive, beam=beam, detuning=dw)
        self._drive_total = drive
        self._meas_vc = self._measure(self._plant.vc, self.obs_scale)
        self._meas_vf = self._measure(self._drive_total, self.obs_scale)
        self._step_index += 1

        reward = self.reward.compute(
            step_index=idx,
            vc=self._plant.vc,
            sp=sp,
            drive_total=self._drive_total,
            vf_ff=vf_ff,
            reflected=self._plant.vr,
            residual=self._drive_rl,
            sat_excess=sat_excess,
            slew_excess=slew_excess,
        )
        terminated = False
        truncated = self._step_index >= self.episode_len
        return self._get_obs(), float(reward), terminated, truncated, self._make_info()
