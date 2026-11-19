###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""Registry and Gymnasium-API contract of every environment."""

import numpy as np
import pytest

import llrflibs_rl as rl
from llrflibs_rl.configs import RewardConfig
from llrflibs_rl.configs.controller import PIControllerConfig, ADRCControllerConfig


def zero_action(env):
    def _f(obs):
        return np.zeros(env.action_space.shape, dtype=np.float32)
    return _f


@pytest.mark.parametrize("env_id", rl.list_env_ids())
def test_gymnasium_api_contract_and_registry(env_id, fast_cavity, fast_env_config):
    """Every registered env satisfies reset/step and runs an episode finite."""
    from dataclasses import asdict

    from llrflibs_rl.configs.env import (
        DualLoopEnvConfig,
        FrequencyEnvConfig,
        VectorSumEnvConfig,
    )
    from llrflibs_rl.envs.gain_scheduling import GainSchedulingConfig
    from llrflibs_rl.envs.pulse_to_pulse import AdaptivePulseConfig, BeamLoadingConfig

    assert env_id in rl.list_env_ids()
    spec = rl.get_spec(env_id)
    assert spec.entry is not None

    # Resolution variants reuse the constructor of their -v0 base.
    kind = env_id.replace("-v1", "-v0")
    if kind in ("CavityVoltage-v0", "PIAssist-v0", "ADRCAssist-v0"):
        env = rl.make(env_id, cavity=fast_cavity, env_cfg=fast_env_config,
                      reward_cfg=RewardConfig())
    elif kind == "GainScheduling-v0":
        env = rl.make(env_id, cavity=fast_cavity,
                      env_cfg=GainSchedulingConfig(**asdict(fast_env_config)),
                      reward_cfg=RewardConfig())
    elif kind == "BeamLoading-v0":
        env = rl.make(env_id, cavity=fast_cavity,
                      env_cfg=BeamLoadingConfig(**asdict(fast_env_config)),
                      reward_cfg=RewardConfig())
    elif kind == "AdaptivePulse-v0":
        env = rl.make(env_id, config=AdaptivePulseConfig(
            cavity=fast_cavity, pulse_len=600, n_pulses=8))
    elif kind in ("VectorSum-v0", "VectorSumTuner-v0"):
        small_cavities = [fast_cavity, fast_cavity, fast_cavity]
        env = rl.make(
            env_id,
            cavities=small_cavities,
            env_cfg=VectorSumEnvConfig(
                cavities=small_cavities,
                tuner_action=(kind == "VectorSumTuner-v0"),
                episode_len=fast_env_config.episode_len,
                randomize_detuning=0.0, measurement_noise_std=0.0, loop_delay_steps=0,
                max_drive_abs=None, max_drive_slew_per_step=None,
            ),
            reward_cfg=RewardConfig(),
        )
    elif kind == "FrequencyControl-v0":
        env = rl.make(env_id, cavity=fast_cavity, env_cfg=FrequencyEnvConfig(
            episode_len=fast_env_config.episode_len, measurement_noise_std=0.0,
            loop_delay_steps=0, randomize_detuning=0.0, randomize_disturbance=False,
            max_drive_abs=None, max_drive_slew_per_step=None,
        ), reward_cfg=RewardConfig())
    else:  # DualLoop-v0 / DualLoop-v1
        env = rl.make(env_id, cavity=fast_cavity, env_cfg=DualLoopEnvConfig(
            episode_len=fast_env_config.episode_len, measurement_noise_std=0.0,
            loop_delay_steps=0, randomize_detuning=0.0, randomize_disturbance=False,
            max_drive_abs=None, max_drive_slew_per_step=None,
        ), reward_cfg=RewardConfig())

    obs, info = env.reset(seed=0)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape
    assert info is not None

    action_fn = zero_action(env)
    total = 0.0
    steps = 0
    for _ in range(env.episode_len):
        a = action_fn(obs)
        obs, reward, terminated, truncated, info = env.step(a)
        total += float(reward)
        steps += 1
        assert np.isfinite(total)
        assert obs.shape == env.observation_space.shape
        if terminated or truncated:
            break
    assert steps > 0
    assert truncated  # full episode ran without early termination

    obs, info = env.reset(seed=0)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape
    assert info is not None

    action_fn = zero_action(env)
    total = 0.0
    steps = 0
    for _ in range(env.episode_len):
        a = action_fn(obs)
        obs, reward, terminated, truncated, info = env.step(a)
        total += float(reward)
        steps += 1
        assert np.isfinite(total)
        assert obs.shape == env.observation_space.shape
        if terminated or truncated:
            break
    assert steps > 0
    assert truncated  # full episode ran without early termination


def test_seeding_reproducibility():
    env1 = rl.make("CavityVoltage-v0")
    env2 = rl.make("CavityVoltage-v0")
    obs1, _ = env1.reset(seed=123)
    obs2, _ = env2.reset(seed=123)
    np.testing.assert_allclose(obs1, obs2)


def test_inner_loop_configs_accepted():
    env = rl.make("PIAssist-v0", pi_cfg=PIControllerConfig(kp=0.5, ki=1e5))
    env.reset(seed=0)
    env.step(np.zeros(2, dtype=np.float32))
    env2 = rl.make("ADRCAssist-v0", adrc_cfg=ADRCControllerConfig(observer_pole_scale=100.0))
    env2.reset(seed=0)
    env2.step(np.zeros(2, dtype=np.float32))


def test_training_preset_budgets():
    cfg = rl.agent_config_for("DualLoop-v0")
    assert cfg.timesteps == 30000
    quick = rl.agent_config_for("DualLoop-v0", quick=True)
    assert quick.timesteps == 2000
    override = rl.agent_config_for("DualLoop-v0", timesteps=123)
    assert override.timesteps == 123
