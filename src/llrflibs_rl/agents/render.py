###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Live rendering while the RL agent trains.

:class:`LiveRenderCallback` is a stable-baselines3 callback: every
``render_freq`` training steps it rolls out a short episode with the current
(growing) policy through a :class:`RenderableEnv` in ``human`` mode, so you can
*watch the agent learn* in a window — the LLRF equivalent of watching CartPole
during ``model.learn()``.  On headless backends it degrades silently to a
no-op so CI and servers are unaffected.

Wire it up either via the training API (``train(..., render=True,
render_freq=2048)``) or the CLI (``run.py train --render``).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

__all__ = ["LiveRenderCallback"]


class LiveRenderCallback(BaseCallback):
    """Periodically live-render a rollout of the policy being trained.

    Args:
        env_factory:  Factory building a *raw* environment (wrapped in a
                      :class:`~llrflibs_rl.envs.render.RenderableEnv` here).
        render_freq:  Render a live rollout every this many training steps.
        render_steps: Samples per live rollout (cap modulo episode length).
        seed:         Seed of the rendered evaluation episode.
        pause_sec:    GUI refresh interval in seconds (``human`` render).
        verbose:      SB3 verbosity.
    """

    def __init__(
        self,
        env_factory: Callable,
        render_freq: int = 2048,
        render_steps: int = 300,
        seed: int = 0,
        pause_sec: float = 0.02,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.env_factory = env_factory
        self.render_freq = int(render_freq)
        self.render_steps = int(render_steps)
        self.seed = seed
        self.pause_sec = float(pause_sec)
        self._renders = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            self._render_episode()
        return True

    def _render_episode(self) -> None:
        """Roll out the current policy and live-render one episode."""
        from ..envs.render import RenderableEnv

        env = RenderableEnv(self.env_factory(), render_mode="human", pause_sec=self.pause_sec)
        obs, _ = env.reset(seed=self.seed)
        for _ in range(self.render_steps):
            action = np.asarray(self.model.predict(obs, deterministic=True)[0], dtype=np.float32)
            obs, _reward, terminated, truncated, _info = env.step(action)
            env.render("human")
            if terminated or truncated:
                break
        env.close()
        self._renders += 1

    def rendered_count(self) -> int:
        """Number of live-rendered evaluation episodes so far."""
        return self._renders
