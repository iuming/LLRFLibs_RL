###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Gymnasium wrappers for the LLRF environments.

:class:`EpisodicMonitor` records episode rewards and lengths, which is handy
for training callbacks and quick sanity checks.  More wrappers (observation /
reward normalization) can be added here without touching the environments.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

__all__ = ["EpisodicMonitor"]


class EpisodicMonitor(gym.Wrapper[np.ndarray, np.ndarray, float, Any]):
    """Record per-episode total reward and length in ``info``.

    Example:

        >>> env = EpisodicMonitor(raw_env)
        >>> obs, info = env.reset()
        >>> # ... steps ...
        >>> info["episode"]  # {"r": ..., "l": ...} available at termination
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._reset()

    def _reset(self) -> None:
        self.episode_reward = 0.0
        self.episode_length = 0
        self.started = True

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self._reset()
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward += float(reward)
        self.episode_length += 1
        if terminated or truncated:
            info = dict(info)
            info["episode"] = {"r": float(self.episode_reward), "l": int(self.episode_length)}
        return obs, reward, terminated, truncated, info
