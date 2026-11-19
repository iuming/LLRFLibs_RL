###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Level-L4 environments: beam-loading transients and pulse-to-pulse adaptation.

* :class:`BeamLoadingEnv`  — the beam enters during the flattop, while the
  analytical feedforward is built for a beam-free cavity.  The RL residual
  (or a PI baseline) must compensate the resulting voltage droop: a textbook
  LLRF beam-loading compensation problem.
* :class:`AdaptivePulseEnv` — a pulse-level MDP in which each environment step
  is one full pulse.  The agent chooses per-segment multiplicative corrections
  to the nominal feedforward from the previous pulse's error, and the cavity
  detuning drifts from pulse to pulse.  This is the pulse-to-pulse
  adaptation / learned-feedforward baseline of ILC-type tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..configs.cavity import CavityParams
from ..configs.env import ResidualEnvConfig
from ..configs.task import RewardConfig
from ..plants.beam_schedule import BeamSchedule
from ..plants.cavity_voltage import CavityVoltagePlant
from ..tasks.reference import PulseReference
from ..tasks.rewards import RegionScaler
from .voltage import CavityVoltageEnv

__all__ = ["AdaptivePulseConfig", "AdaptivePulseEnv", "BeamLoadingConfig", "BeamLoadingEnv"]


# ---------------------------------------------------------------------------
# Beam-loading transient environment
# ---------------------------------------------------------------------------

@dataclass
class BeamLoadingConfig(ResidualEnvConfig):
    """Configuration of the beam-loading environment.

    Attributes:
        beam_current_plant:  Beam current of the plant (the actual machine), A.
        beam_phase_deg_plant: Beam phase of the plant, deg.
        beam_on_start:       Sample index at which the beam turns on; before
                             that the beam drive is zero. ``None`` defaults to
                             the cavity filling length.
        reference_beam_current: Beam current used to build the analytical
                             feedforward (0 gives a beam-free feedforward).
        beam_schedule:       Optional :class:`BeamSchedule` overriding the
                             simple onset (transient trips, partial levels).
    """

    beam_current_plant: float = 8e-3
    beam_phase_deg_plant: float = 0.0
    beam_on_start: int | None = None
    reference_beam_current: float = 0.0
    beam_schedule: BeamSchedule | None = None

    def validate(self) -> None:
        super().validate()
        if self.beam_current_plant < 0:
            raise ValueError("beam_current_plant must be non-negative.")
        if self.reference_beam_current < 0:
            raise ValueError("reference_beam_current must be non-negative.")

    def __post_init__(self) -> None:
        self.validate()


class BeamLoadingEnv(CavityVoltageEnv):
    """Beam-loading compensation with RL residual on the feedforward.

    Structure: the reference / feedforward is built for a beam-free cavity
    while the plant carries the actual beam current that steps on at
    ``beam_on_start``.  The voltage droop that follows is what the action
    (residual I/Q) or a classical baseline has to correct.

    Args:
        cavity:     Nominal cavity description (beam parameters are taken from
                    the configuration).
        env_cfg:    Beam-loading loop configuration.
        reward_cfg: Reward weights.
    """

    def __init__(
        self,
        cavity: CavityParams,
        env_cfg: BeamLoadingConfig = BeamLoadingConfig(),
        reward_cfg: RewardConfig = RewardConfig(),
    ) -> None:
        self.beam_cfg = env_cfg
        plant_cavity = replace(
            cavity,
            beam_current=env_cfg.beam_current_plant,
            beam_phase_deg=env_cfg.beam_phase_deg_plant,
        )
        super().__init__(plant_cavity, env_cfg, reward_cfg)

        # Rebuild the reference with the beam-free cavity and mask the beam.
        ref_cavity = replace(cavity, beam_current=env_cfg.reference_beam_current)
        self._reference = PulseReference(
            ref_cavity,
            env_cfg.Ts,
            self.episode_len,
            env_cfg.reference_detuning,
        )
        # The actual beam waveform is computed with the plant beam paramters.
        beam_ref = PulseReference(
            plant_cavity,
            env_cfg.Ts,
            self.episode_len,
            env_cfg.reference_detuning,
        )
        self._beam_wave = np.asarray(beam_ref.vb, dtype=np.complex128).copy()
        if env_cfg.beam_schedule is not None:
            self._beam_wave = env_cfg.beam_schedule.apply_to(self._beam_wave)
        else:
            on_start = (
                int(env_cfg.beam_on_start) if env_cfg.beam_on_start is not None else cavity.filling_len
            )
            self._beam_wave[: int(on_start)] = 0.0

    def _beam_at(self, step_index: int) -> complex:
        idx = min(int(step_index), self.episode_len - 1)
        return complex(self._beam_wave[idx])


# ---------------------------------------------------------------------------
# Pulse-to-pulse adaptive-feedforward environment
# ---------------------------------------------------------------------------

@dataclass
class AdaptivePulseConfig:
    """Configuration of the pulse-to-pulse adaptive environment.

    Attributes:
        cavity:               Cavity description.
        Ts:                   Sampling time, s.
        pulse_len:            Number of samples of one pulse (episode length of
                              the inner simulation).
        n_basis:              Number of feedforward segments the agent can tune.
        n_pulses:             Number of pulses per episode (the MDP horizon).
        detuning_drift_amp:   Linear pulse-to-pulse detuning drift, rad/s
                              (final offset reached on the last pulse).
        randomize_detuning:   Extra random offset of the initial detuning, rad/s.
        reference_detuning:   Detuning used to build the nominal feedforward.
        action_gain_scale:    Max relative feedforward correction per segment
                              (a segment gain is ``1 + clip(action) * scale``).
        reward_error_weight:  Weight of the flat-top tracking error.
        reward_drive_weight:  Weight of the feedforward-change penalty.
        reward_action_weight: Weight of the action magnitude penalty.
    """

    cavity: CavityParams
    Ts: float = 1.0e-6
    pulse_len: int = 1500
    n_basis: int = 8
    n_pulses: int = 10
    detuning_drift_amp: float = 2.0 * np.pi * 4.0e2
    randomize_detuning: float = 0.0
    reference_detuning: float | None = None
    action_gain_scale: float = 0.4
    reward_error_weight: float = 10.0
    reward_drive_weight: float = 0.5
    reward_action_weight: float = 0.01

    def validate(self) -> None:
        if self.Ts <= 0 or self.pulse_len <= 0:
            raise ValueError("Ts and pulse_len must be positive.")
        if self.n_basis <= 0:
            raise ValueError("n_basis must be positive.")
        if self.n_pulses <= 0:
            raise ValueError("n_pulses must be positive.")
        if self.detuning_drift_amp < 0 or self.randomize_detuning < 0:
            raise ValueError("detuning amplitudes must be non-negative.")

    def __post_init__(self) -> None:
        self.validate()


class AdaptivePulseEnv(gym.Env):
    """Pulse-to-pulse adaptive feedforward under drifting detuning.

    Each environment step simulates one complete pulse with a corrected
    feedforward; the correction is a set of per-segment multiplicative gains
    chosen from the observation (the previous pulse's error pattern).  This
    is the RL counterpart of an ILC feedforward table.

    Observation (4 + n_basis):
        ``[pulse_frac, prev_fferr_re, prev_fferr_im, prev_drive_pen,
        err_proj_0, ..., err_proj_{K-1}]``.
    Action (K,):          normalized per-segment correction in ``[-1, 1]``.

    Args:
        config: Adaptive-pulse configuration.
    """

    metadata = {"render_modes": []}
    reward_range = (-float("inf"), float("inf"))

    def __init__(self, config: AdaptivePulseConfig) -> None:
        super().__init__()
        self.config = config
        self.cavity = config.cavity
        self.episode_len = int(config.n_pulses)
        self.pulse_len = int(config.pulse_len)
        self.n_basis = int(config.n_basis)
        self.obs_scale = float(config.cavity.vc_target)

        self._reference = PulseReference(
            config.cavity,
            config.Ts,
            self.pulse_len,
            config.reference_detuning,
        )
        self._plant = CavityVoltagePlant(
            config.cavity, config.Ts, detuning=config.cavity.detuning
        )

        # Segment boundaries over the filling + flattop region.
        active = config.cavity.filling_len + config.cavity.flattop_len
        self._edges = np.linspace(0.0, active, self.n_basis + 1)
        self._region = RegionScaler(
            config.cavity.filling_len, config.cavity.flattop_len, (0.5, 1.0, 0.0), ramp=25
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_basis,), dtype=np.float32)
        n_obs = 4 + self.n_basis
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)

        self.pulse_reward_history: list[float] = []
        self.pulse_error_history: list[complex] = []
        self._prev_drive_pen = 0.0
        self._prev_fferr = 0.0 + 0.0j
        self._prev_proj = np.zeros(self.n_basis, dtype=np.float32)
        self._pulse_index = 0
        self._detuning_pulse = float(config.cavity.detuning)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _segment_index(self, sample_index: int) -> int:
        for k in range(self.n_basis):
            if sample_index <= self._edges[k + 1]:
                return k
        return self.n_basis - 1

    def _flattop_error_projection(self, error_wave: np.ndarray) -> tuple[complex, np.ndarray]:
        """Return the flat-top mean error and its per-segment projection."""
        cav = self.cavity
        sl = slice(cav.filling_len, min(cav.filling_len + cav.flattop_len, self.pulse_len))
        err = error_wave[sl]
        mean_err = complex(np.mean(err)) if len(err) else 0.0 + 0.0j
        proj = np.zeros(self.n_basis, dtype=np.float32)
        for k in range(self.n_basis):
            s0, s1 = int(self._edges[k]), int(self._edges[k + 1])
            seg = error_wave[s0:s1]
            if len(seg):
                proj[k] = float(np.mean(np.abs(seg)) / self.obs_scale)
        return mean_err, proj

    def _simulate_pulse(self, gains: np.ndarray) -> dict[str, float]:
        """Run one pulse and return its metrics.

        Args:
            gains: Per-segment multiplicative gains (length ``n_basis``).

        Returns:
            Metric dict with reward components and the tracking-error wave.
        """
        self._plant.reset()
        error = np.zeros(self.pulse_len, dtype=np.complex128)
        vf_ff = self._reference.vf_ff
        vb = self._reference.vb
        drive_scale = max(self._reference.peak_drive, self.obs_scale, 1.0)
        drive_pen = 0.0
        reward_acc = 0.0
        for i in range(self.pulse_len):
            seg = self._segment_index(i)
            vf = vf_ff[i] * (1.0 + np.clip(float(gains[seg]), -1.0, 1.0) * self.config.action_gain_scale)
            if self.config.cavity.has_beam:
                beam = complex(vb[i])
            else:
                beam = None
            self._plant.step(vf, beam=beam, detuning=self._detuning_pulse)
            sp = self._reference.sp_at(i)
            error[i] = sp - self._plant.vc
            drive_pen += abs(vf - vf_ff[i]) ** 2 / drive_scale**2
            w = self._region(i)
            reward_acc += w * (abs(error[i]) / self.obs_scale) ** 2
        drive_pen_norm = float(drive_pen / self.pulse_len)
        flattop_err_norm = float(reward_acc / self.pulse_len)

        mean_err, proj = self._flattop_error_projection(error)
        return {
            "flattop_err_norm": flattop_err_norm,
            "drive_pen_norm": drive_pen_norm,
            "mean_err": mean_err,
            "proj": proj,
        }

    # ------------------------------------------------------------------
    @property
    def spec(self):
        """The registry :class:`EnvSpec` attached by ``llrflibs_rl.make``."""
        return getattr(self, "_spec", None)

    @property
    def unwrapped(self):
        """The base (unwrapped) environment."""
        return self

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        pulse_frac = min(self._pulse_index, self.episode_len - 1) / max(self.episode_len - 1, 1)
        obs = [
            pulse_frac,
            self._prev_fferr.real / self.obs_scale,
            self._prev_fferr.imag / self.obs_scale,
            self._prev_drive_pen,
        ]
        obs.extend(float(v) for v in self._prev_proj)
        return np.asarray(obs, dtype=np.float32)

    def _make_info(self) -> dict[str, float]:
        return {
            "pulse_index": float(self._pulse_index),
            "pulse_reward": float(self.pulse_reward_history[-1]) if self.pulse_reward_history else 0.0,
            "detuning_hz": float(self._detuning_pulse / (2.0 * np.pi)),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        super().reset(seed=seed)
        offset = (
            float(self.np_random.uniform(-self.config.randomize_detuning, self.config.randomize_detuning))
            if self.config.randomize_detuning > 0.0
            else 0.0
        )
        self._detuning_pulse = float(self.config.cavity.detuning) + offset
        self._pulse_index = 0
        self.pulse_reward_history = []
        self.pulse_error_history = []
        self._prev_fferr = 0.0 + 0.0j
        self._prev_project = np.zeros(self.n_basis, dtype=np.float32)
        self._prev_drive_pen = 0.0
        return self._get_obs(), self._make_info()

    def _episode_ended(self) -> bool:
        """Pulse-mode horizon check: the adaptation window (``n_pulses``).

        Each step is one pulse, so the episode ends after ``episode_len``
        pulses (N * pulse period).  CW mode is not meaningful here and is
        treated as pulse mode.
        """
        cfg = getattr(self, "config", None)
        if cfg is not None and getattr(cfg, "mode", "pulse") == "cw":
            return False
        return int(getattr(self, "_pulse_index", 0)) >= int(self.episode_len)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, float]]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0).reshape(self.n_basis)
        gains = 1.0 + action * self.config.action_gain_scale
        metrics = self._simulate_pulse(gains)

        flattop_err = metrics["flattop_err_norm"]
        drive_pen = metrics["drive_pen_norm"]
        action_pen = float(np.mean(action**2))
        reward = -(
            self.config.reward_error_weight * flattop_err
            + self.config.reward_drive_weight * drive_pen
            + self.config.reward_action_weight * action_pen
        )

        self._prev_fferr = metrics["mean_err"]
        self._prev_proj = metrics["proj"]
        self._prev_drive_pen = drive_pen
        self.pulse_reward_history.append(float(reward))
        self.pulse_error_history.append(self._prev_fferr)

        self._pulse_index += 1
        # Drift the detuning of the next pulse (linear across the episode).
        progress = self._pulse_index / max(self.episode_len, 1)
        self._detuning_pulse = (
            float(self.config.cavity.detuning)
            + self.config.detuning_drift_amp * progress
        )

        terminated = False
        truncated = self._episode_ended()
        return self._get_obs(), float(reward), terminated, truncated, self._make_info()

    # ------------------------------------------------------------------
    # Evaluation protocol
    # ------------------------------------------------------------------

    @property
    def reference_sp(self):
        """No per-step set-point in the pulse-level MDP."""
        return None

    @property
    def tracked_signal(self) -> complex:
        """Latest flat-top mean error of the last completed pulse."""
        return self._prev_fferr

    @property
    def plant(self) -> CavityVoltagePlant:
        """Inner per-pulse cavity plant (status introspection)."""
        return self._plant
