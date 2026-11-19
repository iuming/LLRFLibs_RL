###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Tests for the pulse / continuous-wave (CW) mode semantics:

* pulse mode   -> ``spec.max_episode_steps == episode_len`` (for pulse envs this
  is the N-*pulse-period* adaptation window, one step per pulse);
* cw mode      -> ``spec.max_episode_steps is None`` (infinite) and the loop is
  not truncated at ``episode_len``, it only ends on a protective trip.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import llrflibs_rl as rl
from llrflibs_rl.configs.env import EnvConfig


def test_pulse_mode_spec_derives_horizon():
    env = rl.make("CavityVoltage-v0")
    assert env.spec.max_episode_steps == int(env.episode_len)
    assert env.env_cfg.mode == "pulse"


def test_pulse_mode_pulse_env_horizon_is_n_pulses():
    env = rl.make("AdaptivePulse-v0")
    # Episode = N pulses (one step per pulse) -> spec horizon equals episode_len.
    assert env.spec.max_episode_steps == int(env.episode_len)


def test_cw_mode_spec_is_infinite():
    base = rl.make("CavityVoltage-v0").env_cfg
    env = rl.make("CavityVoltage-v0", env_cfg=replace(base, mode="cw"))
    assert env.env_cfg.mode == "cw"
    assert env.spec.max_episode_steps is None


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        EnvConfig(mode="bogus")


def test_cw_mode_never_truncates_at_horizon(fast_cavity, fast_env_config):

    config = replace(fast_env_config, mode="cw")
    env = rl.make("CavityVoltage-v0", cavity=fast_cavity, env_cfg=config)
    env.reset(seed=0)
    horizon = config.episode_len
    steps = 0
    for _ in range(2 * horizon):
        _obs, _r, terminated, truncated, _info = env.step(np.array([1.0, 0.0], dtype=np.float32))
        steps += 1
        if truncated:
            pytest.fail("CW mode must not truncate at the episode horizon.")
        if terminated:
            break
    assert steps == 2 * horizon  # ran twice the pulse horizon without ending


def test_cw_mode_ends_only_on_trip(fast_cavity, fast_env_config):

    config = replace(
        fast_env_config,
        mode="cw",
        reflected_power_limit=1.0e4,
        reflected_power_trip_steps=3,
    )
    env = rl.make("CavityVoltage-v0", cavity=fast_cavity, env_cfg=config)
    env.reset(seed=0)
    terminated = truncated = False
    for _ in range(20000):
        _obs, _r, term, trunc, _info = env.step(np.array([0.05, 0.05], dtype=np.float32))
        if term or trunc:
            terminated, truncated = term, trunc
            break
    # The CW episode ends only through the protective trip, never by
    # truncating at the episode horizon.
    assert terminated is True
    assert truncated is False
