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

from typing import Callable, Dict, List, Optional

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

import numpy as np

from ..configs.agent import AgentConfig

__all__ = ["AgentConfig", "make_vector_env", "train", "train_ppo", "train_sac"]


class EvalCurveCallback(BaseCallback):
    """Periodically evaluate the deterministic policy and record the reward.

    The recorded points are appended to the ``curve`` list supplied by the
    caller (or to ``model.eval_curve``), giving the training-convergence plot
    data of one run.

    Args:
        eval_env_factory: Zero-argument callable returning an evaluation env.
        eval_freq:        Number of env steps between evaluations.
        eval_seed:        Fixed seed of the evaluation environment.
        eval_steps:       Max steps per evaluation episode.
        curve:            Optional list to append ``{"timesteps", "reward"}``.
        verbose:          Verbosity.
    """

    def __init__(
        self,
        eval_env_factory: Callable,
        eval_freq: int = 2048,
        eval_seed: int = 0,
        eval_steps: Optional[int] = None,
        curve: Optional[List[Dict[str, float]]] = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_env_factory = eval_env_factory
        self.eval_freq = int(eval_freq)
        self.eval_seed = int(eval_seed)
        self.eval_steps = eval_steps
        self.curve = curve

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        env = self.eval_env_factory()
        obs, _ = env.reset(seed=self.eval_seed)
        total = 0.0
        steps = 0
        while steps < (self.eval_steps or env.episode_len):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            steps += 1
            if terminated or truncated:
                break
        point = {"timesteps": float(self.num_timesteps), "reward": total}
        if self.curve is not None:
            self.curve.append(point)
        else:
            getattr(self.model, "eval_curve", []).append(point)
        if self.verbose:
            print(f"[eval @ {self.num_timesteps} steps] reward = {total:.3f}")
        return True


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


def _attach_curve(model, curve: Optional[List[Dict[str, float]]]) -> None:
    """Attach the evaluation curve list to a model for later plotting."""
    if curve is None:
        curve = []
    model.eval_curve = curve  # type: ignore[attr-defined]


def train_ppo(
    env_factory: Callable,
    cfg: AgentConfig,
    *,
    eval_env_factory: Optional[Callable] = None,
    eval_freq: int = 2048,
    eval_seed: int = 0,
    curve: Optional[List[Dict[str, float]]] = None,
):
    """Train a PPO policy.

    Args:
        env_factory:      Environment factory.
        cfg:              Agent configuration.
        eval_env_factory: Optional evaluation-env factory (enables the
                          training-reward curve).
        eval_freq:        Evaluation frequency in env steps.
        eval_seed:        Seed of the evaluation environment.
        curve:            Optional list that receives the evaluation points.

    Returns:
        A trained ``stable_baselines3.PPO`` instance (with ``eval_curve``).
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
    _attach_curve(model, curve)
    callback = None
    if eval_env_factory is not None:
        callback = EvalCurveCallback(
            eval_env_factory, eval_freq=eval_freq, eval_seed=eval_seed, curve=curve
        )
    model.learn(total_timesteps=cfg.timesteps, callback=callback)
    return model


def train_sac(
    env_factory: Callable,
    cfg: AgentConfig,
    *,
    eval_env_factory: Optional[Callable] = None,
    eval_freq: int = 2048,
    eval_seed: int = 0,
    curve: Optional[List[Dict[str, float]]] = None,
):
    """Train a SAC policy.

    Args:
        env_factory:      Environment factory.
        cfg:              Agent configuration.
        eval_env_factory: Optional evaluation-env factory (see :func:`train_ppo`).
        eval_freq:        Evaluation frequency in env steps.
        eval_seed:        Seed of the evaluation environment.
        curve:            Optional list that receives the evaluation points.

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
    _attach_curve(model, curve)
    callback = None
    if eval_env_factory is not None:
        callback = EvalCurveCallback(
            eval_env_factory, eval_freq=eval_freq, eval_seed=eval_seed, curve=curve
        )
    model.learn(total_timesteps=cfg.timesteps, callback=callback)
    return model


def train(
    env_factory: Callable,
    cfg: AgentConfig,
    *,
    eval_env_factory: Optional[Callable] = None,
    eval_freq: int = 2048,
    eval_seed: int = 0,
    curve: Optional[List[Dict[str, float]]] = None,
):
    """Train an agent using the algorithm selected in the configuration.

    Args:
        env_factory:      Environment factory.
        cfg:              Agent configuration.
        eval_env_factory: Optional evaluation-env factory (see :func:`train_ppo`).
        eval_freq:        Evaluation frequency in env steps.
        eval_seed:        Seed of the evaluation environment.
        curve:            Optional list that receives the evaluation points.

    Returns:
        The trained model (PPO or SAC).
    """
    if cfg.algorithm == "ppo":
        return train_ppo(
            env_factory, cfg, eval_env_factory=eval_env_factory,
            eval_freq=eval_freq, eval_seed=eval_seed, curve=curve,
        )
    if cfg.algorithm == "sac":
        return train_sac(
            env_factory, cfg, eval_env_factory=eval_env_factory,
            eval_freq=eval_freq, eval_seed=eval_seed, curve=curve,
        )
    raise ValueError(f"unsupported algorithm: {cfg.algorithm}")
