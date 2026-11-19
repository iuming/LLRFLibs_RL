###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
llrflibs_rl — reinforcement learning for LLRF control of particle-accelerator RF cavities.

This is the RL sub-library of **LLRFLibsPy** (National Synchrotron Radiation
Laboratory, USTC).  It builds a modular research stack on top of the physics
and classical-control primitives of ``llrflibs``:

* ``configs``     — strongly-typed configuration (cavity / env / task / controller / agent)
* ``plants``      — physical environment models (cavity voltage, resonance, vector module)
* ``tasks``       — reference waveforms, observations, rewards, event handling
* ``controllers`` — classical baselines (feedforward, PI, ADRC, tuner, ILC)
* ``envs``        — Gymnasium environments (registered in :mod:`llrflibs_rl.registry`)
* ``agents``      — training / evaluation loops and per-task presets
* ``evaluation``  — physics-aware metrics, robustness sweeps, report generation
* ``utils``       — seeding, run artifacts, logging

Importing this package registers all built-in environments, so the canonical
way to obtain one is::

    env = llrflibs_rl.make("DualLoop-v0")
"""

# Importing ``envs`` registers the built-in environments; keep it explicit.
from . import (
    agents,
    configs,
    controllers,
    envs,  # noqa: E402  (side-effect: registration)
    evaluation,
    plants,
    scenario,  # noqa: E402  (scenario flow used by examples / CLI)
    tasks,
    utils,
)
from ._version import __version__
from .agents import AgentConfig, agent_config_for, train, train_ppo, train_sac
from .configs import (
    ADRCControllerConfig,
    CavityParams,
    DualLoopEnvConfig,
    EnvConfig,
    FrequencyEnvConfig,
    PIControllerConfig,
    ResidualEnvConfig,
    RewardConfig,
    TunerActuatorConfig,
    VectorSumEnvConfig,
    nominal_nc_cavity,
    nominal_sc_cavity,
)
from .evaluation import action_from_model, evaluate_episode, evaluate_over_seeds
from .registry import EnvSpec, get_spec, list_env_ids, make, recommended_timesteps, register

__all__ = [
    # registry
    "EnvSpec",
    "get_spec",
    "list_env_ids",
    "make",
    "recommended_timesteps",
    "register",
    # configs
    "ADRCControllerConfig",
    "CavityParams",
    "DualLoopEnvConfig",
    "EnvConfig",
    "FrequencyEnvConfig",
    "PIControllerConfig",
    "ResidualEnvConfig",
    "RewardConfig",
    "TunerActuatorConfig",
    "VectorSumEnvConfig",
    "nominal_nc_cavity",
    "nominal_sc_cavity",
    # agents
    "AgentConfig",
    "agent_config_for",
    "train",
    "train_ppo",
    "train_sac",
    # evaluation
    "action_from_model",
    "evaluate_episode",
    "evaluate_over_seeds",
    # submodules
    "agents",
    "configs",
    "controllers",
    "envs",
    "evaluation",
    "plants",
    "scenario",
    "tasks",
    "utils",
    # version
    "__version__",
]
