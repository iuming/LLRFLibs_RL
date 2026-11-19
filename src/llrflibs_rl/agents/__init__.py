###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""Agent layer: training wrappers and per-task presets."""

from .presets import agent_config_for
from .trainer import (
    AgentConfig,
    make_training_vec_env,
    make_vector_env,
    train,
    train_ppo,
    train_sac,
    train_td3,
)

__all__ = [
    "AgentConfig",
    "agent_config_for",
    "make_training_vec_env",
    "make_vector_env",
    "train",
    "train_ppo",
    "train_sac",
    "train_td3",
]
