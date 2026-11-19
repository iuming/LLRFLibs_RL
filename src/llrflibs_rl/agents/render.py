###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Live rendering while the RL agent trains.

:class:`LiveRenderCallback` is a stable-baselines3 callback: from the very
first training step, and then every ``render_freq`` steps, it rolls out a short
episode with the current (growing) policy through a *persistent*
:class:`RenderableEnv` in ``human`` mode.  The operator-console window is
opened once at the start of training and stays open (updating live) until
training ends — the LLRF equivalent of watching CartPole during
``model.learn()``.  If no interactive matplotlib backend/display is available
the callback prints why and silently disables itself, so headless CI and
servers are unaffected.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

__all__ = ["LiveRenderCallback"]


class LiveRenderCallback(BaseCallback):
    """Persistent live console of the policy being trained.

    Args:
        env_factory:  Factory building a *raw* environment (wrapped in a
                      :class:`~llrflibs_rl.envs.render.RenderableEnv` here).
        render_freq:  Refresh the live rollout every this many training steps
                      (a first window is opened at step 0).
        render_steps: Samples per live refresh (kept short so training is not
                      slowed down).
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
        self._render_env = None
        self._render_ok = True

    # ------------------------------------------------------------------

    def _on_step(self) -> bool:
        due = self.n_calls <= 1 or (self.render_freq > 0 and self.n_calls % self.render_freq == 0)
        if due and self._render_ok:
            self._render_episode()
        return True

    def _on_training_end(self) -> None:
        """Close the persistent console window when training finishes."""
        if self._render_env is not None:
            try:
                self._render_env.close()
            except Exception:
                pass
            self._render_env = None

    # ------------------------------------------------------------------

    def _render_episode(self) -> None:
        """Roll out the current policy in the persistent console window."""
        try:
            if self._render_env is None:
                from ..envs.render import RenderableEnv

                self._render_env = RenderableEnv(
                    self.env_factory(), render_mode="human", pause_sec=self.pause_sec
                )
            env = self._render_env
            obs, _ = env.reset(seed=self.seed)
            for _ in range(self.render_steps):
                action = np.asarray(self.model.predict(obs, deterministic=True)[0], dtype=np.float32)
                obs, _reward, terminated, truncated, _info = env.step(action)
                env.render("human")
                if terminated or truncated:
                    break
            self._renders += 1
            print(
                f"[render] live console updated @ step {self.n_calls}"
                + ("  (window stays open; press Ctrl-C to stop training and close it)" if self._renders == 1 else "")
            )
        except Exception as exc:  # pragma: no cover - environment/display dependent
            self._render_ok = False
            print(f"[render] live window unavailable ({type(exc).__name__}: {exc}); "
                  "continuing training without a window.")

    def rendered_count(self) -> int:
        """Number of live-rendered evaluations so far."""
        return self._renders
