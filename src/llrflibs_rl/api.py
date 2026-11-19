###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Batteries-included public facade.

``llrflibs_rl.api`` exposes the everyday surface of the toolkit in one place:
environment creation, preset-driven training, classical baselines, evaluation,
the scenario runner and the environment verifier.

Example::

    from llrflibs_rl import api

    env = api.make("DualLoop-v0")
    cfg = api.agent_config_for("DualLoop-v0")          # -> SAC preset
    model = api.train(lambda: api.make("DualLoop-v0"), cfg)
    metrics = api.evaluate_episode(api.make("DualLoop-v0"), api.action_from_model(model))
    api.verify_environment("DualLoop-v0")
"""

from __future__ import annotations

from .agents import agent_config_for, train  # noqa: F401
from .agents.model_zoo import ModelZoo  # noqa: F401
from .configs.agent import AgentConfig  # noqa: F401
from .controllers.baselines import (  # noqa: F401
    list_baselines,
    make_baseline_policy,
    zero_policy,
)
from .evaluation import action_from_model, evaluate_episode, evaluate_over_seeds  # noqa: F401
from .registry import (  # noqa: F401
    get_spec,
    list_env_ids,
    make,
    recommended_timesteps,
)
from .scenario import run_experiment  # noqa: F401
from .utils import verify_environment  # noqa: F401

__all__ = [
    "AgentConfig",
    "ModelZoo",
    "action_from_model",
    "agent_config_for",
    "evaluate_episode",
    "evaluate_over_seeds",
    "get_spec",
    "list_baselines",
    "list_env_ids",
    "make",
    "make_baseline_policy",
    "recommended_timesteps",
    "run_experiment",
    "train",
    "verify_environment",
    "zero_policy",
]
