###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Vector-sum (multi-cavity) environment.

:class:`VectorSumEnv` lets an agent regulate the complex vector sum of several
cavities fed by a single shared RF source — the quantity a real multi-cavity
LLRF module controls.  Each cavity adapts an independently randomized detuning
at reset, and the agent may observe every cavity voltage.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from gymnasium import spaces

from ..configs.cavity import CavityParams
from ..configs.controller import TunerActuatorConfig
from ..configs.env import VectorSumEnvConfig
from ..configs.task import RewardConfig
from ..controllers.resonance import FirstOrderActuator
from ..plants.vectormodule import VectorModulePlant
from ..tasks.events import apply_drive_limits
from ..tasks.observations import build_vector_sum_obs
from ..tasks.reference import PulseReference
from ..tasks.rewards import RewardFactory
from .core import BaseLLRFEnv

__all__ = ["VectorSumEnv"]


class VectorSumEnv(BaseLLRFEnv):
    """Track the vector sum of several cavities with a shared RF drive.

    Observation (13 + 2*n if ``obs_per_cavity``, plus n tuner states with
    ``tuner_action``):
        ``[vs(2), sp(2), err(2), vr(2), vf_ff(2), vb(2), t, (vc_i(2), ...),
        (tuner_state_i, ...)]``.
    Action (2, or 2 + n with ``tuner_action``):
        normalized I/Q residual in ``[-1, 1]`` plus one normalized tuner
        command per cavity.

    Args:
        cavities:   One ``CavityParams`` per cavity.
        env_cfg:    Loop-level (vector-sum) configuration.
        reward_cfg: Reward weights.
    """

    def __init__(
        self,
        cavities: Sequence[CavityParams],
        env_cfg: VectorSumEnvConfig = VectorSumEnvConfig(),
        reward_cfg: RewardConfig = RewardConfig(),
    ) -> None:
        super().__init__()
        self.cavities = list(cavities)
        self.env_cfg = env_cfg
        self.reward_cfg = reward_cfg
        if not self.cavities:
            raise ValueError("cavities must not be empty.")
        self.n_cavities = len(self.cavities)
        self.episode_len = int(env_cfg.episode_len)

        ref_cav = self.cavities[0]
        self._reference = PulseReference(
            ref_cav, env_cfg.Ts, self.episode_len, env_cfg.reference_detuning
        )
        self.obs_scale = float(self._reference.obs_scale * self.n_cavities)
        self.drive_limit = float(
            env_cfg.drive_limit_ratio * max(self._reference.peak_drive, ref_cav.vc_target, 1.0)
        )
        self.drive_effort_scale = max(self._reference.peak_drive, self.obs_scale, 1.0)

        self._plant = VectorModulePlant(
            self.cavities, env_cfg.Ts, self._reference.reference_detuning
        )
        self.reward = RewardFactory(
            reward_cfg,
            obs_scale=self.obs_scale,
            filling_len=ref_cav.filling_len,
            flattop_len=ref_cav.flattop_len,
            drive_limit=self.drive_limit,
            drive_effort_scale=self.drive_effort_scale,
        )

        self.tuner_action = bool(env_cfg.tuner_action)
        if self.tuner_action:
            self._tuners = [
                FirstOrderActuator(
                    TunerActuatorConfig(
                        tau=env_cfg.tuner_tau, detuning_limit=env_cfg.tuner_detuning_limit
                    ),
                    env_cfg.Ts,
                )
                for _ in range(self.n_cavities)
            ]
            self.tuner_detuning_limit = float(env_cfg.tuner_detuning_limit)
            n_action = 2 + self.n_cavities
            # voltage-ish layout (+ per-cavity voltages) + tuner states + detunings.
            n_obs = (13 + 2 * self.n_cavities if env_cfg.obs_per_cavity else 13) + 2 * self.n_cavities
        else:
            self._tuners = []
            n_action = 2
            n_obs = 13 + 2 * self.n_cavities if env_cfg.obs_per_cavity else 13

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_action,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)

        self._detunings: list[float] = []
        self._meas_vs = 0.0 + 0.0j
        self._drive_total = 0.0 + 0.0j
        self._last_action = 0.0 + 0.0j
        self._step_index = 0
        self._init_drive_history()

    # ------------------------------------------------------------------
    # Evaluation protocol (used by llrflibs_rl.evaluation.metrics)
    # ------------------------------------------------------------------

    @property
    def reference_sp(self) -> np.ndarray:
        """Vector-sum set-point waveform."""
        return self._sp_ref

    @property
    def tracked_signal(self) -> complex:
        """Latest vector sum."""
        return self._plant.vector_sum

    @property
    def plant(self) -> VectorModulePlant:
        """Underlying vector-module plant (status introspection)."""
        return self._plant

    @property
    def reference(self) -> PulseReference:
        """Underlying single-cavity pulse reference (status introspection)."""
        return self._reference

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_detunings(self) -> list[float]:
        detunings = []
        for cav in self.cavities:
            detuning = float(cav.detuning)
            if self.env_cfg.randomize_detuning > 0.0:
                detuning += float(
                    self.np_random.uniform(-self.env_cfg.randomize_detuning, self.env_cfg.randomize_detuning)
                )
            detunings.append(detuning)
        return detunings

    def _get_obs(self) -> np.ndarray:
        idx = min(self._step_index, self.episode_len - 1)
        sp = self._sp_ref[idx]
        vf_ff = self._reference.vf_ff_at(idx)
        vb = self._reference.vb_at(idx)
        err = sp - self._meas_vs
        obs = build_vector_sum_obs(
            vs=self._plant.vector_sum,
            sp=sp,
            err=err,
            vr_sum=self._plant.reflected_sum,
            vf_ff=vf_ff,
            vb=vb,
            vc_list=self._plant.vc_list,
            obs_scale=self.obs_scale,
            step_index=self._step_index,
            episode_len=self.episode_len,
            obs_per_cavity=self.env_cfg.obs_per_cavity,
        )
        if self.tuner_action:
            tuner_states = np.asarray([t.state / self.tuner_detuning_limit for t in self._tuners])
            detuning_norm = np.asarray(
                [float(np.clip(d / self.tuner_detuning_limit, -1.0, 1.0)) for d in self._detunings]
            )
            obs = np.concatenate([obs, tuner_states.astype(np.float32),
                                  detuning_norm.astype(np.float32)])
        return obs

    def _make_info(self) -> dict[str, float]:
        idx = min(self._step_index, self.episode_len - 1)
        err = self._sp_ref[idx] - self._plant.vector_sum
        info = {
            "step_index": float(self._step_index),
            "vs_abs": float(abs(self._plant.vector_sum)),
            "sp_abs": float(abs(self._sp_ref[idx])),
            "error_abs": float(abs(err)),
            "drive_abs": float(abs(self._drive_total)),
            "reflection_abs": float(abs(self._plant.reflected_sum)),
        }
        for i, dw in enumerate(self._detunings):
            info[f"detuning_{i}_hz"] = float(dw / (2.0 * np.pi))
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
        super().reset(seed=seed)
        self._detunings = self._sample_detunings()
        self._plant.reset(self._detunings)
        self._sp_ref = self.n_cavities * np.asarray(self._reference.sp, dtype=np.complex128)
        self._meas_vs = self._measure(self._plant.vector_sum, self.obs_scale)
        self._drive_total = self._reference.vf_ff_at(0)
        self._last_action = 0.0 + 0.0j
        self._step_index = 0
        for tuner in self._tuners:
            tuner.reset()
        self._init_drive_history()
        return self._get_obs(), self._make_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, float]]:
        action = self._clamp_action(action)
        residual = complex(float(action[0]), float(action[1])) * self.drive_limit
        self._last_action = residual

        tuner_cmd = np.zeros(self.n_cavities, dtype=np.float32)
        detuning_offsets: list[float] | None = None
        if self.tuner_action:
            tuner_cmd = np.clip(action[2: 2 + self.n_cavities], -1.0, 1.0)
            detuning_offsets = []
            for tuner, cmd in zip(self._tuners, tuner_cmd, strict=False):
                detuning_offsets.append(float(tuner.step(cmd)))

        idx = min(self._step_index, self.episode_len - 1)
        vf_ff = self._reference.vf_ff_at(idx)
        unclipped = vf_ff + residual
        drive, sat_excess, slew_excess = apply_drive_limits(
            unclipped, self._drive_total, self.env_cfg.max_drive_abs, self.env_cfg.max_drive_slew_per_step
        )
        drive = self._delay_drive(drive)

        beam = self._reference.vb_at(idx)
        self._plant.step(drive, beam, detuning_offsets=detuning_offsets)
        self._meas_vs = self._measure(self._plant.vector_sum, self.obs_scale)
        self._drive_total = drive
        self._step_index += 1

        sp = self._sp_ref[idx]
        reward = self.reward.compute(
            step_index=idx,
            vc=self._plant.vector_sum,
            sp=sp,
            drive_total=self._drive_total,
            vf_ff=vf_ff,
            reflected=self._plant.reflected_sum,
            residual=residual,
            sat_excess=sat_excess,
            slew_excess=slew_excess,
        )
        if self.tuner_action:
            reward -= self.env_cfg.tuner_effort_weight * float(np.mean(tuner_cmd**2))
        terminated = False
        truncated = self._episode_ended()
        return self._get_obs(), float(reward), terminated, truncated, self._make_info()
