###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Visualization wrapper for the LLRF environments (CartPole-style ``render``).

The package environments do not implement ``render``; this wrapper adds it so
an episode can be watched live or saved to an image, just like the
``human`` / ``rgb_array`` render modes of classic Gymnasium control tasks:

* ``render("human")`` — updates a multi-panel matplotlib figure during an
  episode (amplitude tracking, phase tracking, I/Q error Lissajous and drive /
  detuning telemetry).  With a GUI backend the figure updates live; with the
  Agg backend it is inert (used by the ``demo`` CLI to write an image).
* ``render("rgb_array")`` — returns the current figure as an ``(H, W, 3)``
  uint8 array, ready to be written to a video or inspected in tests.

matplotlib is imported lazily so importing the wrapper never opens a window;
callers in headless processes should ``matplotlib.use("Agg")`` before the
first render.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["RenderableEnv", "ensure_gui_backend"]

_GUI_TOKENS = ("tk", "qt", "gtk", "wx", "macosx")


def ensure_gui_backend() -> bool:
    """Make sure an interactive matplotlib backend is active.

    Called at the entry points (``render_demo``, the ``demo`` / ``train
    --render`` CLIs) *before* ``matplotlib.pyplot`` is imported anywhere: if the
    current backend is headless (e.g. ``Agg`` set by ``MPLBACKEND``) it tries to
    switch to an interactive one (TkAgg -> QtAgg -> wxAgg).  Returns whether a
    GUI backend is now active (``False`` on truly headless systems).

    Returns:
        ``True`` when a GUI backend is usable, ``False`` otherwise.
    """
    import matplotlib

    backend = matplotlib.get_backend().lower()
    if any(token in backend for token in _GUI_TOKENS):
        return True
    for candidate in ("TkAgg", "QtAgg", "Qt5Agg", "wxAgg"):
        try:
            matplotlib.use(candidate, force=True)
            return True
        except Exception:
            continue
    return False


class RenderableEnv:
    """Wrap an LLRF environment with a matplotlib renderer.

    This is deliberately *not* a ``gym.Wrapper`` subclass: the renderer is an
    optional side-channel, so training rolls (which never call ``render``) are
    untouched and the wrapper composes with the environment registry cleanly.
    It forwards ``reset`` / ``step`` to the wrapped environment so it can be
    dropped into the usual loop::

        env = RenderableEnv(rl.make("CavityVoltage-v0"), render_mode="human")
        obs, _ = env.reset(seed=0)
        for _ in range(400):
            obs, _, terminated, truncated, _ = env.step(policy(obs))
            env.render()
            if terminated or truncated:
                break

    Args:
        env:         The environment to visualize.
        render_mode: ``"human"`` (live figure) or ``"rgb_array"``.
        title:       Optional figure title.
        max_steps:   Rolling window length shown on the time axes.
    """

    def __init__(
        self,
        env,
        render_mode: str = "human",
        title: str | None = None,
        max_steps: int = 1500,
        pause_sec: float = 0.02,
    ) -> None:
        if render_mode not in ("human", "rgb_array"):
            raise ValueError("render_mode must be 'human' or 'rgb_array'.")
        self.env = env
        self.render_mode = render_mode
        self.title = title or getattr(env, "spec", None) and str(env.spec.env_id)
        self.max_steps = int(max_steps)
        self.pause_sec = float(pause_sec)
        self._fig = None
        self._ax = None
        self._lines: dict[str, Any] = {}
        self._history: dict[str, list[float]] = {}
        self._reset_history()

    # ------------------------------------------------------------------
    # Environment delegation
    # ------------------------------------------------------------------

    @property
    def unwrapped(self):
        """The base (unwrapped) environment."""
        return self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env

    @property
    def fig(self):
        """The matplotlib figure (created lazily on first render)."""
        if self._fig is None:
            self._ensure_figure()
        return self._fig

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def reward_range(self):
        return getattr(self.env, "reward_range", (-np.inf, np.inf))

    @property
    def spec(self):
        return getattr(self.env, "spec", None)

    def __getattr__(self, name: str):
        """Forward any unknown attribute to the wrapped environment.

        Baselines and policies read live signals (e.g. ``tracked_signal``,
        ``drive_limit``) straight off the environment object; delegating keeps
        them working through the renderer.
        """
        return getattr(self.env, name)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset the environment and clear the render history."""
        self._reset_history()
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        """Step the environment and stash the telemetry for rendering."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._record()
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Close the figure (if any) and the wrapped environment."""
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
            self._fig = None
        close = getattr(self.env, "close", None)
        if close is not None:
            close()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _reset_history(self) -> None:
        self._history = {"t": [], "amp": [], "phase": [], "drive": [], "detuning": []}

    def _record(self) -> None:
        env = self.env
        idx = int(getattr(env, "_step_index", 0) or getattr(env, "step_index", 0))
        tracked = complex(getattr(env, "tracked_signal", 0.0))
        reference = complex(getattr(env, "reference_sp", [0.0])[min(idx, len(getattr(env, "reference_sp", [0])) - 1)])
        drive = complex(getattr(env, "drive_total", 0.0))
        detuning = float(getattr(getattr(env, "plant", None), "detuning", 0.0) or 0.0)
        self._history["t"].append(idx)
        self._history["amp"].append(abs(tracked))
        self._history["phase"].append(np.angle(tracked))
        self._history["drive"].append(abs(drive))
        self._history["detuning"].append(detuning)
        # Trim the rolling window.
        if len(self._history["t"]) > self.max_steps:
            for key in self._history:
                self._history[key] = self._history[key][-self.max_steps:]
        self._reference = reference

    def _ensure_figure(self):
        if self._fig is not None:
            return
        import matplotlib.pyplot as plt

        self._fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        self._fig.suptitle(self.title or "LLRF environment")
        (ax_amp, ax_pha), (ax_lis, ax_tel) = axes

        (self._lines["amp"],) = ax_amp.plot([], [], lw=1.2, label="tracked")
        (self._lines["ref"],) = ax_amp.plot([], [], lw=1.2, ls="--", color="C3", label="reference")
        ax_amp.set_xlabel("sample")
        ax_amp.set_ylabel("|V| (V)")
        ax_amp.grid(alpha=0.3)
        ax_amp.legend(loc="best")

        (self._lines["phase"],) = ax_pha.plot([], [], lw=1.2, label="tracked")
        (self._lines["phref"],) = ax_pha.plot([], [], lw=1.2, ls="--", color="C3", label="reference")
        ax_pha.set_xlabel("sample")
        ax_pha.set_ylabel("phase (rad)")
        ax_pha.grid(alpha=0.3)
        ax_pha.legend(loc="best")

        (self._lines["lis"],) = ax_lis.plot([], [], lw=1.0, color="C2", label="I/Q error")
        ax_lis.set_xlabel("I error")
        ax_lis.set_ylabel("Q error")
        ax_lis.grid(alpha=0.3)
        ax_lis.legend(loc="best")
        ax_lis.axis("equal")

        (self._lines["drive"],) = ax_tel.plot([], [], lw=1.2, label="|drive|")
        ax_tel.set_xlabel("sample")
        ax_tel.set_ylabel("drive (V)")
        ax_tel.grid(alpha=0.3)
        ax_tel.legend(loc="upper left")
        self._ax_tel2 = ax_tel.twinx()
        (self._lines["detuning"],) = self._ax_tel2.plot([], [], lw=1.0, color="C4", label="detuning")
        self._ax_tel2.set_ylabel("detuning (rad/s)")

        self._fig.tight_layout(rect=(0, 0, 1, 0.96))

    def _live_show(self) -> None:
        """Push the figure to the screen and give the GUI event loop time.

        On headless (Agg) backends this is a no-op so ``render("human")`` is
        safe in CI and servers; on GUI backends it opens the window and streams
        updates in real time, exactly like a control-task ``human`` render.
        """
        import matplotlib.pyplot as plt

        backend = plt.get_backend().lower()
        if "agg" in backend:
            return
        if not plt.isinteractive():
            plt.ion()
        if not plt.fignum_exists(self._fig.number):
            self._fig.show()
        self._fig.canvas.draw()
        plt.pause(self.pause_sec)
        self._fig.canvas.flush_events()

    def render(self, mode: str | None = None):
        """Render the current episode state.

        Args:
            mode: Overrides the render mode (``"human"`` / ``"rgb_array"``).

        Returns:
            ``None`` for ``"human"``, or an ``(H, W, 3)`` uint8 array for
            ``"rgb_array"``.
        """
        mode = mode or self.render_mode
        self._ensure_figure()
        t = np.asarray(self._history["t"], dtype=float)
        amp = np.asarray(self._history["amp"], dtype=float)
        phase = np.asarray(self._history["phase"], dtype=float)
        drive = np.asarray(self._history["drive"], dtype=float)
        det = np.asarray(self._history["detuning"], dtype=float)

        self._lines["amp"].set_data(t, amp)
        self._lines["phase"].set_data(t, phase)
        self._lines["drive"].set_data(t, drive)
        self._lines["detuning"].set_data(t, det)
        for key in (self._lines["amp"].axes, self._lines["phase"].axes, self._lines["drive"].axes):
            key.relim()
            key.autoscale_view()

        ref = np.asarray(getattr(self, "_reference", 0.0))
        self._lines["ref"].set_data(t, np.full_like(t, abs(complex(ref))))
        self._lines["phref"].set_data(t, np.full_like(t, np.angle(complex(ref))))

        if len(t) > 1:
            # Phasor view: the tracked signal trajectory in the I/Q plane.
            self._lines["lis"].set_data(
                amp * np.cos(phase), amp * np.sin(phase)
            )
        self._fig.canvas.draw_idle()

        if mode == "rgb_array":
            return self._to_rgb_array()
        if mode == "human":
            self._fig.canvas.flush_events()
            self._live_show()
        return None

    def _to_rgb_array(self) -> np.ndarray:
        """Snapshot the current figure as an ``(H, W, 3)`` uint8 array."""
        self._fig.canvas.draw()
        buf = np.asarray(self._fig.canvas.buffer_rgba(), dtype=np.uint8)
        return buf[..., :3]
