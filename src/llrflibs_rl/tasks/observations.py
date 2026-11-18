###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Observation builders.

These pure functions assemble and physics-normalize the observation vectors of
each environment.  Keeping them separate from the environment classes makes
the observation layouts explicit, testable and easy to extend (e.g. adding a
disturbance estimate or an observer state).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

__all__ = [
    "build_voltage_obs",
    "build_assist_obs",
    "build_frequency_obs",
    "build_dual_loop_obs",
    "build_vector_sum_obs",
]


def _time_frac(step_index: int, episode_len: int) -> float:
    return float(step_index / max(episode_len - 1, 1))


def build_voltage_obs(
    *,
    vc: complex,
    sp: complex,
    err: complex,
    vr: complex,
    vf_ff: complex,
    vb: complex,
    last_action: complex,
    obs_scale: float,
    drive_limit: float,
    step_index: int,
    episode_len: int,
) -> np.ndarray:
    """Observation of the plain residual voltage environment (15-dim).

    Layout: ``[vc(2), sp(2), vf_ff(2), vb(2), vr(2), err(2), last_action(2), t]``.
    """
    return np.array(
        [
            vc.real / max(obs_scale, 1.0), vc.imag / max(obs_scale, 1.0),
            sp.real / max(obs_scale, 1.0), sp.imag / max(obs_scale, 1.0),
            vf_ff.real / max(obs_scale, 1.0), vf_ff.imag / max(obs_scale, 1.0),
            vb.real / max(obs_scale, 1.0), vb.imag / max(obs_scale, 1.0),
            vr.real / max(obs_scale, 1.0), vr.imag / max(obs_scale, 1.0),
            err.real / max(obs_scale, 1.0), err.imag / max(obs_scale, 1.0),
            last_action.real / max(drive_limit, 1.0),
            last_action.imag / max(drive_limit, 1.0),
            _time_frac(step_index, episode_len),
        ],
        dtype=np.float32,
    )


def build_assist_obs(
    *,
    vc: complex,
    sp: complex,
    err: complex,
    vr: complex,
    vf_ff: complex,
    drive_fb: complex,
    drive_rl: complex,
    vb: complex,
    obs_scale: float,
    drive_limit: float,
    step_index: int,
    episode_len: int,
    vc_est: Optional[complex] = None,
    f_est: Optional[complex] = None,
) -> np.ndarray:
    """Observation of the PI/ADRC assist environments (17 or 21-dim).

    Layout: ``[vc(2), sp(2), err(2), vr(2), vf_ff(2), drive_fb(2), drive_rl(2),
    vb(2), t]`` plus, when ``vc_est``/``f_est`` are given (ADRC), ``[vc_est(2), f_est(2)]``.
    """
    s = max(obs_scale, 1.0)
    obs = [
        vc.real / s, vc.imag / s,
        sp.real / s, sp.imag / s,
        err.real / s, err.imag / s,
        vr.real / s, vr.imag / s,
        vf_ff.real / s, vf_ff.imag / s,
        drive_fb.real / s, drive_fb.imag / s,
        drive_rl.real / max(drive_limit, 1.0), drive_rl.imag / max(drive_limit, 1.0),
        vb.real / s, vb.imag / s,
        _time_frac(step_index, episode_len),
    ]
    if vc_est is not None and f_est is not None:
        obs += [vc_est.real / s, vc_est.imag / s, f_est.real / s, f_est.imag / s]
    return np.asarray(obs, dtype=np.float32)


def build_frequency_obs(
    *,
    vc: complex,
    vr: complex,
    detuning: float,
    act_state: float,
    disturbance: float,
    detuning_limit: float,
    obs_scale: float,
    step_index: int,
    episode_len: int,
) -> np.ndarray:
    """Observation of the resonance (frequency) environment (8-dim).

    Layout: ``[vc(2), vr(2), detuning_norm, actuator_norm, disturbance_norm, t]``.
    """
    return np.array(
        [
            vc.real / max(obs_scale, 1.0), vc.imag / max(obs_scale, 1.0),
            vr.real / max(obs_scale, 1.0), vr.imag / max(obs_scale, 1.0),
            detuning / detuning_limit,
            act_state / detuning_limit,
            disturbance / detuning_limit,
            _time_frac(step_index, episode_len),
        ],
        dtype=np.float32,
    )


def build_dual_loop_obs(
    *,
    vc: complex,
    sp: complex,
    err: complex,
    vr: complex,
    detuning: float,
    act_state: float,
    disturbance: float,
    drive_fb: complex,
    detuning_limit: float,
    obs_scale: float,
    step_index: int,
    episode_len: int,
) -> np.ndarray:
    """Observation of the dual-loop environment (13-dim).

    Layout: ``[vc(2), sp(2), err(2), vr(2), detuning_norm, actuator_norm,
    disturbance_norm, drive_fb_norm, t]``.
    """
    return np.array(
        [
            vc.real / max(obs_scale, 1.0), vc.imag / max(obs_scale, 1.0),
            sp.real / max(obs_scale, 1.0), sp.imag / max(obs_scale, 1.0),
            err.real / max(obs_scale, 1.0), err.imag / max(obs_scale, 1.0),
            vr.real / max(obs_scale, 1.0), vr.imag / max(obs_scale, 1.0),
            detuning / detuning_limit,
            act_state / detuning_limit,
            disturbance / detuning_limit,
            abs(drive_fb) / max(obs_scale, 1.0),
            _time_frac(step_index, episode_len),
        ],
        dtype=np.float32,
    )


def build_vector_sum_obs(
    *,
    vs: complex,
    sp: complex,
    err: complex,
    vr_sum: complex,
    vf_ff: complex,
    vb: complex,
    vc_list: Sequence[complex],
    obs_scale: float,
    step_index: int,
    episode_len: int,
    obs_per_cavity: bool = True,
) -> np.ndarray:
    """Observation of the vector-sum environment (13 or 13 + 2*n dim).

    Layout: ``[vs(2), sp(2), err(2), vr(2), vf_ff(2), vb(2), t]`` plus, when
    ``obs_per_cavity``, the per-cavity voltages ``[vc_0(2), vc_1(2), ...]``.
    """
    s = max(obs_scale, 1.0)
    obs = [
        vs.real / s, vs.imag / s,
        sp.real / s, sp.imag / s,
        err.real / s, err.imag / s,
        vr_sum.real / s, vr_sum.imag / s,
        vf_ff.real / s, vf_ff.imag / s,
        vb.real / s, vb.imag / s,
        _time_frac(step_index, episode_len),
    ]
    if obs_per_cavity:
        for vc in vc_list:
            obs += [vc.real / s, vc.imag / s]
    return np.asarray(obs, dtype=np.float32)
