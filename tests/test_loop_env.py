###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Tests of the full control-loop environment (``LLRFLoop-v0``).

The environment simulates the LLRF block diagram with the two RL action
channels: the slow tuner actuator and the FPGA RF drive signal
(``[tuner, RF_I, RF_Q]``).
"""

from __future__ import annotations

import numpy as np

import llrflibs_rl as rl
from llrflibs_rl.configs.env import LoopEnvConfig
from llrflibs_rl.configs.task import RewardConfig
from llrflibs_rl.controllers.baselines import make_baseline_policy


def _cfg(episodes: int = 600) -> LoopEnvConfig:
    return LoopEnvConfig(
        episode_len=episodes,
        base_detuning=2.0 * np.pi * 100.0,
        disturbance_amp=2.0 * np.pi * 50.0,
        disturbance_type="sinusoidal",
        randomize_disturbance=False,
        drive_limit_ratio=0.10,
    )


def _make(fast_cavity, episodes: int = 600):
    return rl.make(
        "LLRFLoop-v0",
        cavity=fast_cavity,
        env_cfg=_cfg(episodes),
        reward_cfg=RewardConfig(),
    )


def test_loop_env_contract_and_run(fast_cavity):
    env = _make(fast_cavity)
    assert env.spec.kind == "loop"
    assert env.action_space.shape == (3,)  # [tuner, RF_I, RF_Q]
    assert env.observation_space.shape == (15,)
    obs, info = env.reset(seed=0)
    assert obs.shape == (15,) and np.isfinite(obs).all()
    act = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    for _ in range(300):
        obs, _r, _term, trunc, info = env.step(act)
        assert np.isfinite(obs).all()
    assert not trunc
    assert "detuning_hz" in info and "drive_rl_abs" in info
    assert info["drive_rl_abs"] > 0.0  # the RF channel had an effect


def test_tuner_channel_drives_actuator_in_sign(fast_cavity):
    env = _make(fast_cavity)
    env.reset(seed=0)
    for _ in range(300):
        env.step(np.array([0.7, 0.0, 0.0], dtype=np.float32))
    assert env.actuator.state > 0.0

    env2 = _make(fast_cavity)
    env2.reset(seed=0)
    for _ in range(300):
        env2.step(np.array([-0.7, 0.0, 0.0], dtype=np.float32))
    assert env2.actuator.state < 0.0


def test_rf_residual_changes_the_drive_signal(fast_cavity):
    def drive_trace(rf: float, n_steps: int = 60) -> float:
        env = _make(fast_cavity)
        env.reset(seed=0)
        total = 0.0
        for _ in range(n_steps):
            _o, _r, _term, _trunc, info = env.step(np.array([0.0, rf, 0.0], dtype=np.float32))
            total += float(info["drive_abs"])
        return total / n_steps

    # The closed PI loop pins the *steady-state* voltage to the setpoint, but
    # the RF residual channel must measurably change the FPGA drive signal.
    d0 = drive_trace(0.0)
    d1 = drive_trace(0.9)
    assert abs(d0 - d1) > 1e-2 * max(d0, 1.0), f"RF residual must change the drive ({d0} vs {d1})"


def test_pi_baseline_runs_on_the_loop_env(fast_cavity):
    env = _make(fast_cavity)
    policy = make_baseline_policy(env, "pi")
    obs, _ = env.reset(seed=0)
    for _ in range(200):
        obs, _r, term, trunc, _info = env.step(policy(obs))
        assert obs.shape == (15,)
        if term or trunc:
            break
    assert env.actuator.state == env.actuator.state  # ran without crashing


def test_reward_is_bounded_for_extreme_actions(fast_cavity):
    """Regression: the RF residual must be normalized by its own scale, not 1 V.

    With the dual-loop base reward the residual was normalized by 1 V, which
    made any RF action explode the reward (~1e11 per step).  The loop env
    rebuilds the reward factory with the real residual scale.
    """
    env = _make(fast_cavity)
    env.reset(seed=0)
    for _ in range(300):
        _o, reward, _term, _trunc, _info = env.step(np.array([1.0, 1.0, 1.0], dtype=np.float32))
        assert reward > -1.0e6, f"reward must stay bounded, got {reward}"
    env.close()
