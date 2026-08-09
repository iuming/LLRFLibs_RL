###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
'''
Gymnasium environments for LLRF reinforcement learning.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from llrflibs.rf_control import cav_sp_ff, ss_discrete
from llrflibs.rf_sim import cav_ss, cav_ss_mech, sim_ncav_step, sim_scav_step


@dataclass
class VoltageControlConfig:
    """Configuration for the cavity-voltage control environment."""

    half_bw: float
    Ts: float
    episode_len: int
    vc_target: float = 25e6
    detuning: float = 0.0
    beta: float = 1e4
    filling_len: int = 510
    flattop_len: int = 800
    beam_current: float = 0.0
    beam_phase_deg: float = 0.0
    roQ_or_RoQ: float = 1036.0
    QL: float = 3e6
    machine: str = "linac"
    drive_limit_ratio: float = 0.35
    reward_error_weight: float = 6.0
    reward_drive_weight: float = 0.15
    reward_reflected_weight: float = 0.05
    randomize_detuning: float = 0.0


@dataclass
class FrequencyControlConfig:
    """Configuration for the detuning / frequency-control environment."""

    half_bw: float
    Ts: float
    episode_len: int
    vc_target: float = 25e6
    beta: float = 1e4
    drive_level: float = 12.5e6
    detuning_limit: float = 2 * np.pi * 1.0e4
    actuator_tau: float = 8e-6
    reward_detuning_weight: float = 2.5
    reward_voltage_weight: float = 1.0
    reward_action_weight: float = 0.02
    mech_modes: Optional[Dict[str, list]] = None
    base_detuning: float = 0.0
    disturbance_amp: float = 2 * np.pi * 2.0e3
    disturbance_freq: float = 280.0
    beam_current: float = 0.0
    beam_phase_deg: float = 0.0
    roQ_or_RoQ: float = 1036.0
    QL: float = 3e6
    machine: str = "linac"


def _complex_state_matrix(size: int, dtype: type = complex) -> np.matrix:
    return np.matrix(np.zeros((size, 1), dtype=dtype))


def _scalar(value) -> float:
    return float(np.asarray(value).squeeze())


class CavityVoltageControlEnv(gym.Env):
    """Track a cavity-voltage pulse with an RL correction signal."""

    metadata = {"render_modes": []}

    def __init__(self, config: VoltageControlConfig):
        super().__init__()

        self.config = config
        self.episode_len = int(config.episode_len)
        self.drive_limit = float(config.drive_limit_ratio * config.vc_target)
        self._obs_scale = float(max(config.vc_target, 1.0))

        passband_modes = {
            "freq_offs": [-800e3],
            "gain_rel": [-1.0],
            "half_bw": [2 * np.pi * 216 * 0.5],
        }

        detuning = float(config.detuning)
        if config.randomize_detuning > 0.0:
            detuning += float(np.random.uniform(-config.randomize_detuning, config.randomize_detuning))

        status, Arf, Brf, Crf, Drf, Abm, Bbm, Cbm, Dbm = cav_ss(
            config.half_bw,
            detuning=detuning,
            beta=config.beta,
            passband_modes=passband_modes,
            plot=False,
        )
        if not status:
            raise ValueError("Failed to build cavity model for the voltage-control environment.")

        status, self._Arf_d, self._Brf_d, self._Crf_d, self._Drf_d, _ = ss_discrete(
            Arf, Brf, Crf, Drf, config.Ts, method="zoh", plot=False
        )
        if not status:
            raise ValueError("Failed to discretize the RF cavity model.")

        if config.beam_current > 0.0:
            status, self._Abm_d, self._Bbm_d, self._Cbm_d, self._Dbm_d, _ = ss_discrete(
                Abm, Bbm, Cbm, Dbm, config.Ts, method="bilinear", plot=False
            )
            if not status:
                raise ValueError("Failed to discretize the beam cavity model.")
        else:
            self._Abm_d = self._Bbm_d = self._Cbm_d = self._Dbm_d = None

        status, sp, vf_ff, vb, _ = cav_sp_ff(
            config.half_bw,
            config.filling_len,
            config.flattop_len,
            config.Ts,
            self.episode_len,
            vc0=config.vc_target,
            detuning=detuning,
            beta=config.beta,
            ib0=config.beam_current if config.beam_current > 0.0 else None,
            phib_deg=config.beam_phase_deg,
            beam_ids=config.filling_len,
            beam_ide=min(self.episode_len - 1, config.filling_len + config.flattop_len - 1),
            roQ_or_RoQ=config.roQ_or_RoQ,
            QL=config.QL,
            machine=config.machine,
        )
        if not status:
            raise ValueError("Failed to generate the cavity voltage reference waveform.")

        self._sp = np.asarray(sp, dtype=np.complex128)
        self._vf_ff = np.asarray(vf_ff, dtype=np.complex128)
        self._vb = np.asarray(vb, dtype=np.complex128)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)

        self._state_rf = None
        self._state_bm = None
        self._vc = 0.0 + 0.0j
        self._vr = 0.0 + 0.0j
        self._last_action = 0.0 + 0.0j
        self._drive_total = 0.0 + 0.0j
        self._step_index = 0

    def _get_obs(self) -> np.ndarray:
        idx = min(self._step_index, self.episode_len - 1)
        sp = self._sp[idx]
        vf_ff = self._vf_ff[idx]
        vb = self._vb[idx]
        error = sp - self._vc
        return np.array(
            [
                self._vc.real / self._obs_scale,
                self._vc.imag / self._obs_scale,
                sp.real / self._obs_scale,
                sp.imag / self._obs_scale,
                vf_ff.real / self._obs_scale,
                vf_ff.imag / self._obs_scale,
                vb.real / self._obs_scale,
                vb.imag / self._obs_scale,
                self._vr.real / self._obs_scale,
                self._vr.imag / self._obs_scale,
                error.real / self._obs_scale,
                error.imag / self._obs_scale,
                self._last_action.real / max(self.drive_limit, 1.0),
                self._last_action.imag / max(self.drive_limit, 1.0),
                self._step_index / max(self.episode_len - 1, 1),
            ],
            dtype=np.float32,
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._state_rf = _complex_state_matrix(self._Brf_d.shape[0])
        self._state_bm = _complex_state_matrix(self._Bbm_d.shape[0], dtype=complex) if self._Abm_d is not None else None
        self._vc = 0.0 + 0.0j
        self._vr = 0.0 + 0.0j
        self._last_action = 0.0 + 0.0j
        self._drive_total = self._vf_ff[0]
        self._step_index = 0
        return self._get_obs(), self._make_info()

    def _make_info(self) -> Dict[str, float]:
        idx = min(self._step_index, self.episode_len - 1)
        sp = self._sp[idx]
        error = sp - self._vc
        return {
            "step_index": float(self._step_index),
            "vc_abs": float(np.abs(self._vc)),
            "sp_abs": float(np.abs(sp)),
            "error_abs": float(np.abs(error)),
            "drive_abs": float(np.abs(self._drive_total)),
            "reflection_abs": float(np.abs(self._vr)),
        }

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(2)
        action = np.clip(action, -1.0, 1.0)
        correction = complex(float(action[0]), float(action[1])) * self.drive_limit
        self._last_action = correction

        idx = min(self._step_index, self.episode_len - 1)
        drive = self._vf_ff[idx] + correction
        beam = self._vb[idx] if self._vb.size > 0 else 0.0 + 0.0j

        status, vc_next, vr_next, state_rf, state_bm = sim_ncav_step(
            self._Arf_d,
            self._Brf_d,
            self._Crf_d,
            self._Drf_d,
            drive,
            self._state_rf,
            Abmd=self._Abm_d,
            Bbmd=self._Bbm_d,
            Cbmd=self._Cbm_d,
            Dbmd=self._Dbm_d,
            vb_step=beam if self._Abm_d is not None else None,
            state_bm0=self._state_bm,
        )
        if not status:
            raise RuntimeError("Cavity step simulation failed.")

        self._state_rf = state_rf
        self._state_bm = state_bm
        self._vc = vc_next
        self._vr = vr_next
        self._drive_total = drive
        self._step_index += 1

        sp = self._sp[idx]
        error = sp - self._vc
        error_norm = float(np.abs(error) / max(self._obs_scale, 1.0))
        drive_norm = float(np.abs(correction) / max(self.drive_limit, 1.0))
        refl_norm = float(np.abs(self._vr) / max(np.abs(drive), self._obs_scale, 1.0))
        reward = -(
            self.config.reward_error_weight * error_norm**2
            + self.config.reward_drive_weight * drive_norm**2
            + self.config.reward_reflected_weight * refl_norm**2
        )

        terminated = False
        truncated = self._step_index >= self.episode_len
        return self._get_obs(), float(reward), terminated, truncated, self._make_info()


class CavityFrequencyControlEnv(gym.Env):
    """Compensate cavity detuning with a simple RL actuator model."""

    metadata = {"render_modes": []}

    def __init__(self, config: FrequencyControlConfig):
        super().__init__()

        self.config = config
        self.episode_len = int(config.episode_len)
        self.detuning_limit = float(config.detuning_limit)
        self.actuator_alpha = float(np.exp(-config.Ts / max(config.actuator_tau, config.Ts)))
        self._obs_scale = float(max(config.vc_target, 1.0))

        mech_modes = config.mech_modes or {
            "f": [280, 341, 460, 487, 618],
            "Q": [40, 20, 50, 80, 100],
            "K": [2, 0.8, 2, 0.6, 0.2],
        }
        status, Am, Bm, Cm, Dm = cav_ss_mech(mech_modes)
        if not status:
            raise ValueError("Failed to build the mechanical model for the frequency-control environment.")

        status, self._Am_d, self._Bm_d, self._Cm_d, self._Dm_d, _ = ss_discrete(
            Am, Bm, Cm, Dm, config.Ts, method="zoh", plot=False
        )
        if not status:
            raise ValueError("Failed to discretize the mechanical model.")

        self._step_index = 0
        self._state_m = None
        self._vc = 0.0 + 0.0j
        self._vr = 0.0 + 0.0j
        self._dw = float(config.base_detuning)
        self._act_state = 0.0

        self._disturbance = np.zeros(self.episode_len, dtype=float)
        time_axis = np.arange(self.episode_len) * config.Ts
        self._disturbance[:] = config.base_detuning + config.disturbance_amp * np.sin(
            2.0 * np.pi * config.disturbance_freq * time_axis
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        idx = min(self._step_index, self.episode_len - 1)
        return np.array(
            [
                float(np.real(self._vc)) / self._obs_scale,
                float(np.imag(self._vc)) / self._obs_scale,
                float(np.real(self._vr)) / self._obs_scale,
                float(np.imag(self._vr)) / self._obs_scale,
                float(self._dw) / max(self.detuning_limit, 1.0),
                float(self._act_state) / max(self.detuning_limit, 1.0),
                float(self._disturbance[idx]) / max(self.detuning_limit, 1.0),
                self._step_index / max(self.episode_len - 1, 1),
            ],
            dtype=np.float32,
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._state_m = _complex_state_matrix(self._Bm_d.shape[0], dtype=float)
        self._step_index = 0
        self._vc = 0.0 + 0.0j
        self._vr = 0.0 + 0.0j
        self._dw = float(self.config.base_detuning)
        self._act_state = 0.0
        return self._get_obs(), self._make_info()

    def _make_info(self) -> Dict[str, float]:
        idx = min(self._step_index, self.episode_len - 1)
        return {
            "step_index": float(self._step_index),
            "vc_abs": float(np.abs(self._vc)),
            "detuning_hz": float(self._dw / (2.0 * np.pi)),
            "actuator_detuning_hz": float(self._act_state / (2.0 * np.pi)),
            "disturbance_hz": float(self._disturbance[idx] / (2.0 * np.pi)),
        }

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(1)
        action = float(np.clip(action[0], -1.0, 1.0))
        target_detuning = action * self.detuning_limit
        self._act_state = self.actuator_alpha * self._act_state + (1.0 - self.actuator_alpha) * target_detuning

        idx = min(self._step_index, self.episode_len - 1)
        status, vc_next, vr_next, dw_next, state_m = sim_scav_step(
            self.config.half_bw,
            self._dw,
            self._disturbance[idx] + self._act_state,
            self.config.drive_level,
            0.0 + 0.0j,
            self._vc,
            self.config.Ts,
            beta=self.config.beta,
            state_m0=self._state_m,
            Am=self._Am_d,
            Bm=self._Bm_d,
            Cm=self._Cm_d,
            Dm=self._Dm_d,
            mech_exe=True,
        )
        if not status:
            raise RuntimeError("Frequency-control simulation failed.")

        self._vc = vc_next
        self._vr = vr_next
        self._dw = _scalar(dw_next)
        self._state_m = state_m
        self._step_index += 1

        amp_error = (np.abs(self._vc) - self.config.vc_target) / max(self.config.vc_target, 1.0)
        detuning_error = self._dw / max(self.detuning_limit, 1.0)
        action_penalty = self._act_state / max(self.detuning_limit, 1.0)
        reward = -(
            self.config.reward_voltage_weight * amp_error**2
            + self.config.reward_detuning_weight * detuning_error**2
            + self.config.reward_action_weight * action_penalty**2
        )

        terminated = False
        truncated = self._step_index >= self.episode_len
        return self._get_obs(), float(reward), terminated, truncated, self._make_info()


def make_voltage_control_env(config: VoltageControlConfig) -> CavityVoltageControlEnv:
    """Create a voltage-control environment with a single call."""

    return CavityVoltageControlEnv(config)


def make_frequency_control_env(config: FrequencyControlConfig) -> CavityFrequencyControlEnv:
    """Create a frequency-control environment with a single call."""

    return CavityFrequencyControlEnv(config)