###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Resonance / slow-loop environments.

Two tasks act on the cavity detuning:

* :class:`FrequencyEnv` — RL commands a slow tuner actuator; the RF drive is
  constant, so the resonance loop is the only control channel.
* :class:`DualLoopEnv`  — a fast complex-PI loop closes the voltage while RL
  commands the slow tuner (the classical LLRF division of labour).

Both include Lorentz-force mechanical modes and a sinusoidal microphonics
disturbance.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..configs.cavity import CavityParams
from ..configs.controller import PIControllerConfig, TunerActuatorConfig
from ..configs.env import DualLoopEnvConfig, FrequencyEnvConfig
from ..configs.task import RewardConfig
from ..controllers.feedback import ComplexPIController
from ..controllers.resonance import FirstOrderActuator
from ..plants.disturbance import SinusoidalDisturbance, build_disturbance_wave
from ..plants.resonance import ResonancePlant
from ..tasks.events import apply_drive_limits
from ..tasks.observations import build_dual_loop_obs, build_frequency_obs
from ..tasks.reference import PulseReference
from ..tasks.rewards import RewardFactory
from .core import BaseLLRFEnv

__all__ = ["FrequencyEnv", "DualLoopEnv"]


class FrequencyEnv(BaseLLRFEnv):
    """Compensate cavity detuning with a first-order RL tuner actuator.

    Observation (8,): ``[vc(2), vr(2), detuning, actuator, disturbance, t]``.
    Action (1,):      normalized tuner command in ``[-1, 1]``.

    Args:
        cavity:     Cavity description (mechanical modes used if present).
        env_cfg:    Resonance loop configuration.
        reward_cfg: Reward weights.
    """

    def __init__(
        self,
        cavity: CavityParams,
        env_cfg: FrequencyEnvConfig = FrequencyEnvConfig(),
        reward_cfg: RewardConfig = RewardConfig(),
    ) -> None:
        super().__init__()
        self.cavity = cavity
        self.env_cfg = env_cfg
        self.reward_cfg = reward_cfg
        self.episode_len = int(env_cfg.episode_len)
        self.obs_scale = float(cavity.vc_target)
        self.detuning_limit = float(env_cfg.detuning_limit)

        self._plant = ResonancePlant(cavity, env_cfg.Ts)
        self._actuator = FirstOrderActuator(
            TunerActuatorConfig(detuning_limit=env_cfg.detuning_limit, tau=env_cfg.actuator_tau),
            env_cfg.Ts,
        )
        self._disturbance = SinusoidalDisturbance(
            env_cfg.base_detuning, env_cfg.disturbance_amp, env_cfg.disturbance_freq,
            env_cfg.Ts, self.episode_len,
        )
        self.reward = RewardFactory(
            reward_cfg,
            obs_scale=self.obs_scale,
            filling_len=cavity.filling_len,
            flattop_len=cavity.flattop_len,
            drive_limit=1.0,  # no residual drive in this environment
            drive_effort_scale=self.obs_scale,
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

        self._dist_wave = np.zeros(self.episode_len)
        self._step_index = 0
        self._drive_total = complex(env_cfg.drive_level)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_disturbance(self, offset: float) -> None:
        phase = (
            float(self.np_random.uniform(0.0, 2.0 * np.pi))
            if self.env_cfg.randomize_disturbance
            else 0.0
        )
        self._dist_wave = build_disturbance_wave(
            self.env_cfg.disturbance_type,
            self.episode_len,
            self.env_cfg.Ts,
            base=self.env_cfg.base_detuning,
            amplitude=self.env_cfg.disturbance_amp,
            frequency=self.env_cfg.disturbance_freq,
            phase=phase,
            offset=offset,
            onset=self.env_cfg.disturbance_step_onset,
        )

    def _get_obs(self) -> np.ndarray:
        idx = min(self._step_index, self.episode_len - 1)
        return build_frequency_obs(
            vc=self._plant.vc,
            vr=self._plant.vr,
            detuning=self._plant.detuning,
            act_state=self._actuator.state,
            disturbance=self._dist_wave[idx],
            detuning_limit=self.detuning_limit,
            obs_scale=self.obs_scale,
            step_index=self._step_index,
            episode_len=self.episode_len,
        )

    # ------------------------------------------------------------------
    # Evaluation protocol (used by llrflibs_rl.evaluation.metrics)
    # ------------------------------------------------------------------

    @property
    def reference_sp(self):
        """No tracking set-point in the resonance task."""
        return None

    @property
    def tracked_signal(self) -> complex:
        """Latest cavity voltage (used for the amplitude-error metric)."""
        return self._plant.vc

    @property
    def plant(self) -> ResonancePlant:
        """Underlying resonance plant (status introspection)."""
        return self._plant

    @property
    def actuator(self):
        """Underlying tuner actuator (status introspection)."""
        return self._actuator

    @property
    def disturbance_wave(self) -> np.ndarray:
        """Disturbance waveform of the current episode, rad/s."""
        return self._dist_wave

    def _make_info(self) -> Dict[str, float]:
        idx = min(self._step_index, self.episode_len - 1)
        return {
            "step_index": float(self._step_index),
            "vc_abs": float(abs(self._plant.vc)),
            "detuning_hz": float(self._plant.detuning / (2.0 * np.pi)),
            "actuator_detuning_hz": float(self._actuator.state / (2.0 * np.pi)),
            "disturbance_hz": float(self._dist_wave[idx] / (2.0 * np.pi)),
        }

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
        offset = (
            float(self.np_random.uniform(-self.env_cfg.randomize_detuning, self.env_cfg.randomize_detuning))
            if self.env_cfg.randomize_detuning > 0.0
            else 0.0
        )
        self._build_disturbance(offset)
        self._plant.reset()
        self._actuator.reset()
        self._step_index = 0
        return self._get_obs(), self._make_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, float]]:
        action = self._clamp_action(action)
        act_state = self._actuator.step(float(action[0]))

        idx = min(self._step_index, self.episode_len - 1)
        external = float(self._dist_wave[idx]) + act_state
        vc, vr, dw = self._plant.step(complex(self.env_cfg.drive_level), 0.0 + 0.0j, external)
        self._drive_total = complex(self.env_cfg.drive_level)
        self._step_index += 1

        reward = self.reward.compute(
            step_index=idx,
            vc=vc,
            sp=complex(self.cavity.vc_target),
            drive_total=self._drive_total,
            vf_ff=complex(self.env_cfg.drive_level),
            reflected=vr,
            residual=0.0 + 0.0j,
            detuning=dw,
            detuning_limit=self.detuning_limit,
            actuator_state=act_state,
        )
        terminated = False
        truncated = self._step_index >= self.episode_len
        return self._get_obs(), float(reward), terminated, truncated, self._make_info()


class DualLoopEnv(BaseLLRFEnv):
    """Fast RF PI loop with an RL-controlled slow tuner / piezo actuator.

    Observation (13,): ``[vc(2), sp(2), err(2), vr(2), detuning, actuator,
    disturbance, drive_fb, t]``.
    Action (1,):       normalized tuner command in ``[-1, 1]``.

    Args:
        cavity:     Cavity description (mechanical modes used if present).
        env_cfg:    Dual-loop configuration (fast-loop PI gains included).
        reward_cfg: Reward weights.
    """

    def __init__(
        self,
        cavity: CavityParams,
        env_cfg: DualLoopEnvConfig = DualLoopEnvConfig(),
        reward_cfg: RewardConfig = RewardConfig(),
    ) -> None:
        super().__init__()
        self.cavity = cavity
        self.env_cfg = env_cfg
        self.reward_cfg = reward_cfg
        self.episode_len = int(env_cfg.episode_len)
        self.obs_scale = float(cavity.vc_target)
        self.detuning_limit = float(env_cfg.detuning_limit)

        self._reference = PulseReference(cavity, env_cfg.Ts, self.episode_len)
        self._controller = ComplexPIController(
            PIControllerConfig(kp=env_cfg.kp, ki=env_cfg.ki), env_cfg.Ts
        )
        self._plant = ResonancePlant(cavity, env_cfg.Ts)
        self._actuator = FirstOrderActuator(
            TunerActuatorConfig(detuning_limit=env_cfg.detuning_limit, tau=env_cfg.actuator_tau),
            env_cfg.Ts,
        )
        self._disturbance = SinusoidalDisturbance(
            env_cfg.base_detuning, env_cfg.disturbance_amp, env_cfg.disturbance_freq,
            env_cfg.Ts, self.episode_len,
        )
        self.reward = RewardFactory(
            reward_cfg,
            obs_scale=self.obs_scale,
            filling_len=cavity.filling_len,
            flattop_len=cavity.flattop_len,
            drive_limit=1.0,
            drive_effort_scale=max(self._reference.peak_drive, self.obs_scale, 1.0),
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)

        self._dist_wave = np.zeros(self.episode_len)
        self._meas_vc = 0.0 + 0.0j
        self._drive_total = 0.0 + 0.0j
        self._drive_fb = 0.0 + 0.0j
        self._last_error = 0.0 + 0.0j
        self._step_index = 0
        self._init_drive_history()

    # ------------------------------------------------------------------
    # Evaluation protocol (used by llrflibs_rl.evaluation.metrics)
    # ------------------------------------------------------------------

    @property
    def reference_sp(self) -> np.ndarray:
        """Set-point waveform of the fast RF loop."""
        return self._reference.sp

    @property
    def tracked_signal(self) -> complex:
        """Latest cavity voltage."""
        return self._plant.vc

    @property
    def plant(self) -> ResonancePlant:
        """Underlying resonance plant (status introspection)."""
        return self._plant

    @property
    def actuator(self):
        """Underlying tuner actuator (status introspection)."""
        return self._actuator

    @property
    def reference(self) -> PulseReference:
        """Underlying pulse reference (status introspection)."""
        return self._reference

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_disturbance(self, offset: float) -> None:
        phase = (
            float(self.np_random.uniform(0.0, 2.0 * np.pi))
            if self.env_cfg.randomize_disturbance
            else 0.0
        )
        self._dist_wave = build_disturbance_wave(
            self.env_cfg.disturbance_type,
            self.episode_len,
            self.env_cfg.Ts,
            base=self.env_cfg.base_detuning,
            amplitude=self.env_cfg.disturbance_amp,
            frequency=self.env_cfg.disturbance_freq,
            phase=phase,
            offset=offset,
            onset=self.env_cfg.disturbance_step_onset,
        )

    def _get_obs(self) -> np.ndarray:
        idx = min(self._step_index, self.episode_len - 1)
        sp = self._reference.sp_at(idx)
        err = sp - self._plant.vc
        return build_dual_loop_obs(
            vc=self._plant.vc,
            sp=sp,
            err=err,
            vr=self._plant.vr,
            detuning=self._plant.detuning,
            act_state=self._actuator.state,
            disturbance=self._dist_wave[idx],
            drive_fb=self._drive_fb,
            detuning_limit=self.detuning_limit,
            obs_scale=self.obs_scale,
            step_index=self._step_index,
            episode_len=self.episode_len,
        )

    def _make_info(self) -> Dict[str, float]:
        idx = min(self._step_index, self.episode_len - 1)
        err = self._reference.sp_at(idx) - self._plant.vc
        return {
            "step_index": float(self._step_index),
            "vc_abs": float(abs(self._plant.vc)),
            "error_abs": float(abs(err)),
            "drive_abs": float(abs(self._drive_total)),
            "drive_fb_abs": float(abs(self._drive_fb)),
            "reflection_abs": float(abs(self._plant.vr)),
            "detuning_hz": float(self._plant.detuning / (2.0 * np.pi)),
            "actuator_detuning_hz": float(self._actuator.state / (2.0 * np.pi)),
            "disturbance_hz": float(self._dist_wave[idx] / (2.0 * np.pi)),
        }

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
        offset = (
            float(self.np_random.uniform(-self.env_cfg.randomize_detuning, self.env_cfg.randomize_detuning))
            if self.env_cfg.randomize_detuning > 0.0
            else 0.0
        )
        self._build_disturbance(offset)
        self._plant.reset()
        self._actuator.reset()
        self._controller.reset()
        self._meas_vc = self._measure(self._plant.vc, self.obs_scale)
        self._drive_total = self._reference.vf_ff_at(0)
        self._drive_fb = 0.0 + 0.0j
        self._last_error = self._reference.sp_at(0)
        self._step_index = 0
        self._init_drive_history()
        return self._get_obs(), self._make_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, float]]:
        action = self._clamp_action(action)
        act_state = self._actuator.step(float(action[0]))

        idx = min(self._step_index, self.episode_len - 1)
        sp = self._reference.sp_at(idx)
        vf_ff = self._reference.vf_ff_at(idx)
        beam = self._reference.vb_at(idx)
        err = sp - self._meas_vc
        drive_pi, drive_fb = self._controller.step(err, vf_ff)
        self._drive_fb = drive_fb

        drive, sat_excess, slew_excess = apply_drive_limits(
            drive_pi, self._drive_total, self.env_cfg.max_drive_abs, self.env_cfg.max_drive_slew_per_step
        )
        drive = self._delay_drive(drive)

        external = float(self._dist_wave[idx]) + act_state
        vc, vr, dw = self._plant.step(drive, beam, external)
        self._drive_total = drive
        self._meas_vc = self._measure(self._plant.vc, self.obs_scale)
        self._last_error = err
        self._step_index += 1

        reward = self.reward.compute(
            step_index=idx,
            vc=vc,
            sp=sp,
            drive_total=self._drive_total,
            vf_ff=vf_ff,
            reflected=vr,
            residual=0.0 + 0.0j,
            sat_excess=sat_excess,
            slew_excess=slew_excess,
            detuning=dw,
            detuning_limit=self.detuning_limit,
            actuator_state=act_state,
        )
        terminated = False
        truncated = self._step_index >= self.episode_len
        return self._get_obs(), float(reward), terminated, truncated, self._make_info()
