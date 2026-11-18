###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""Training and evaluation smoke tests (small budgets)."""

import numpy as np
import pytest

import llrflibs_rl as rl
from llrflibs_rl.agents import agent_config_for, train
from llrflibs_rl.evaluation import action_from_model, evaluate_episode, evaluate_over_seeds


def test_train_and_evaluate_smoke():
    """A tiny PPO training run must produce a model and finite metrics."""
    env_id = "CavityVoltage-v0"
    cfg = agent_config_for(env_id, timesteps=512, quick=False)
    model = train(lambda: rl.make(env_id), cfg)
    policy = action_from_model(model)
    metrics = evaluate_episode(rl.make(env_id), policy, seed=0)
    assert metrics["reward"] == metrics["reward"]  # not NaN
    assert metrics["length"] > 0
    assert set(metrics) >= {"reward", "error_rms", "flattop_error_rms"}


def test_evaluate_over_seeds():
    policy = lambda obs: np.zeros(1, dtype=np.float32)  # noqa: E731
    metrics = evaluate_over_seeds(lambda: rl.make("DualLoop-v0"), policy, seeds=[0, 1])
    assert "reward_mean" in metrics and "reward_std" in metrics


def test_run_artifacts():
    from pathlib import Path
    from llrflibs_rl.utils import RunArtifacts

    tmp = Path("tests/.tmp_artifacts")
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp)
    try:
        arts = RunArtifacts("test", runs_dir=tmp)
        arts.save_config({"cavity_half_bw": 1e3})
        arts.save_metrics({"reward": -1.5})
        assert (arts.dir / "config.json").exists()
        assert (arts.dir / "metrics.json").exists()
    finally:
        if tmp.exists():
            import shutil
            shutil.rmtree(tmp)
