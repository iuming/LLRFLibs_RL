###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""Classical baseline policies: each must run any registered environment."""

import numpy as np
import pytest

import llrflibs_rl as rl
from llrflibs_rl.controllers.baselines import list_baselines, make_baseline_policy


@pytest.mark.parametrize("env_id", rl.list_env_ids())
@pytest.mark.parametrize("baseline", list_baselines())
def test_baseline_policies_run_and_finite(env_id, baseline):
    env = rl.make(env_id)
    policy = make_baseline_policy(env, baseline)
    obs, _ = env.reset(seed=0)
    total = 0.0
    for _ in range(60):
        action = np.asarray(policy(obs), dtype=np.float32)
        assert action.shape == env.action_space.shape
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        assert np.isfinite(total)
        if terminated or truncated:
            break


def test_unknown_baseline_raises():
    env = rl.make("CavityVoltage-v0")
    with pytest.raises(ValueError):
        make_baseline_policy(env, "not-a-baseline")


def test_baselines_differ_between_policies():
    """The PI residual policy is not the zero policy on a plain voltage env."""
    env = rl.make("CavityVoltage-v0")
    zero = make_baseline_policy(env, "zero")
    pi = make_baseline_policy(env, "pi")
    obs, _ = env.reset(seed=1)
    # At sample 0 both setpoint and cavity voltage are zero; take one step so
    # the setpoint ramps up while the cavity still lags -> a real error.
    obs, _, _, _, _ = env.step(np.zeros(2, dtype=np.float32))
    a_zero = np.asarray(zero(obs))
    a_pi = np.asarray(pi(obs))
    assert np.all(a_zero == 0.0)
    assert np.any(np.abs(a_pi) > 1e-6)
