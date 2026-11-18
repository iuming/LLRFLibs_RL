###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Training wrappers around stable-baselines3.

A single :class:`AgentConfig` drives the PPO / SAC training loops.  The
wrappers accept an environment *factory* (so every rollout gets a fresh
environment) and return the trained model.  Evaluation lives in
:mod:`llrflibs_rl.evaluation`.
"""

from __future__ import annotations

from typing import Callable

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from ..configs.agent import AgentConfig

__all__ = ["AgentConfig", "make_vector_env", "train", "train_ppo", "train_sac"]


def make_vector_env(
    env_factory: Callable,
    seed: int = 0,
    n_envs: int = 1,
) -> DummyVecEnv:
    """Create a vectorized environment wrapping a factory.

    Args:
        env_factory: Zero-argument callable returning a single environment.
        seed:        Random seed of the vector environment.
        n_envs:      Number of parallel environments.

    Returns:
        A :class:`DummyVecEnv`.
    """
    vec_env = DummyVecEnv([lambda: env_factory() for _ in range(n_envs)])
    if seed is not None:
        vec_env.seed(seed)
    return vec_env


def train_ppo(env_factory: Callable, cfg: AgentConfig):
    """Train a PPO policy.

    Args:
        env_factory: Environment factory.
        cfg:         Agent configuration.

    Returns:
        A trained ``stable_baselines3.PPO`` instance.
    """
    vec_env = make_vector_env(env_factory, seed=cfg.seed)
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=cfg.verbose,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        learning_rate=cfg.learning_rate,
        ent_coef=cfg.ent_coef,
        seed=cfg.seed,
    )
    model.learn(total_timesteps=cfg.timesteps)
    return model


def train_sac(env_factory: Callable, cfg: AgentConfig):
    """Train a SAC policy.

    Args:
        env_factory: Environment factory.
        cfg:         Agent configuration.

    Returns:
        A trained ``stable_baselines3.SAC`` instance.
    """
    vec_env = make_vector_env(env_factory, seed=cfg.seed)
    model = SAC(
        "MlpPolicy",
        vec_env,
        verbose=cfg.verbose,
        train_freq=cfg.train_freq,
        gradient_steps=cfg.gradient_steps,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        learning_rate=cfg.learning_rate,
        seed=cfg.seed,
    )
    model.learn(total_timesteps=cfg.timesteps)
    return model


def train(env_factory: Callable, cfg: AgentConfig):
    """Train an agent using the algorithm selected in the configuration.

    Args:
        env_factory: Environment factory.
        cfg:         Agent configuration.

    Returns:
        The trained model (PPO or SAC).
    """
    if cfg.algorithm == "ppo":
        return train_ppo(env_factory, cfg)
    if cfg.algorithm == "sac":
        return train_sac(env_factory, cfg)
    raise ValueError(f"unsupported algorithm: {cfg.algorithm}")
