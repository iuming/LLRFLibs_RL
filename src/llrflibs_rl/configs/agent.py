###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Agent (RL algorithm) configuration.

A single dataclass drives the stable-baselines3 wrappers in
:mod:`llrflibs_rl.agents`.  Per-environment recommended presets are provided
in :mod:`llrflibs_rl.agents.presets`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

__all__ = ["AgentConfig"]


@dataclass
class AgentConfig:
    """RL algorithm and training configuration.

    Attributes:
        algorithm:     ``"ppo"`` / ``"sac"`` / ``"td3"``.
        timesteps:     Total training timesteps.
        seed:          Global random seed.
        n_steps:       PPO rollout steps per update (PPO only).
        batch_size:    Minibatch size (PPO and SAC).
        gamma:         Discount factor.
        learning_rate: Optimizer learning rate.
        ent_coef:      Entropy coefficient (PPO); SAC manages entropy itself.
        train_freq:    SAC update frequency in steps.
        gradient_steps: SAC gradient steps per update.
        normalize_obs:  Wrap the training environment in ``VecNormalize``
                     (observation normalization; reward normalization kept off
                     so reported rewards stay physical).
        n_envs:         Number of parallel train environments (DummyVecEnv).
        policy:         SB3 policy class name (``"MlpPolicy"`` or
                     ``"MlpLstmPolicy"`` with the ``recurrent`` extra).
        policy_kwargs:  Extra keyword arguments for the policy network.
        verbose:       SB3 verbosity level (0 quiet, 1 tables).
        model_name:    Optional name used for the saved model.
    """

    algorithm: str = "ppo"
    timesteps: int = 20000
    seed: int = 0
    n_steps: int = 256
    batch_size: int = 64
    gamma: float = 0.995
    learning_rate: float = 3.0e-4
    ent_coef: float = 0.0
    train_freq: int = 1
    gradient_steps: int = 1
    normalize_obs: bool = False
    n_envs: int = 1
    policy: str = "MlpPolicy"
    policy_kwargs: Optional[Dict[str, object]] = None
    verbose: int = 0
    model_name: Optional[str] = None

    def validate(self) -> None:
        """Raise on inconsistent configuration."""
        if self.algorithm not in ("ppo", "sac", "td3"):
            raise ValueError("algorithm must be 'ppo', 'sac' or 'td3'.")
        if self.timesteps <= 0:
            raise ValueError("timesteps must be positive.")
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")
        if self.n_steps <= 0 or self.batch_size <= 0:
            raise ValueError("n_steps and batch_size must be positive.")
        if self.n_envs <= 0:
            raise ValueError("n_envs must be positive.")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must lie in (0, 1].")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.ent_coef < 0:
            raise ValueError("ent_coef must be non-negative.")
        if self.policy not in ("MlpPolicy", "MlpLstmPolicy"):
            raise ValueError("policy must be 'MlpPolicy' or 'MlpLstmPolicy'.")

    def __post_init__(self) -> None:
        self.validate()
