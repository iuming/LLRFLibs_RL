###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Visualization wrapper for the LLRF environments (CartPole-style ``render``).

The package environments do not implement ``render``; this wrapper adds it so
an episode can be watched live or saved to an image, just like the
``human`` / ``rgb_array`` render modes of classic Gymnasium control tasks.

Two layouts are provided:

* ``"console"`` (default) — an LLRF *operator console* in the spirit of the
  CSS/Phoebus EPICS GUIs: a live numeric PV panel (|Vc|, phase, forward /
  reflected, detuning, beam, drive, error) on the left, waveform trend strips
  for amplitude / phase tracking, and a phasor view with a *moving* state
  marker (the RF equivalent of a vehicle driving its trajectory);
* ``"simple"`` — the original 4-panel figure (amplitude / phase / phasor /
  telemetry).

Rendering is lazy (matplotlib is imported only on first ``render``), so
importing the wrapper never opens a window; headless processes stay inert and
can still capture ``rgb_array`` frames.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["RenderableEnv", "ensure_gui_backend"]

_GUI_TOKENS = ("tk", "qt", "gtk", "wx", "macosx")

# Dark operator-console palette.
_BG = "#0d1117"
_PANEL = "#161b22"
_FG = "#e6edf3"
_ACCENT = "#58a6ff"
_ALARM = "#f85149"
_OK = "#3fb950"
_REF_COLOR = "#d29922"


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
    """Wrap an LLRF environment with a live operator-console renderer.

    Deliberately *not* a ``gym.Wrapper``: the renderer is an optional
    side-channel so training rolls (which never call ``render``) are untouched.
    It forwards ``reset`` / ``step`` / attributes to the wrapped environment::

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
        layout:      ``"console"`` (default, EPICS-style) or ``"simple"``.
        max_steps:   Rolling window length shown on the time axes.
        pause_sec:   GUI refresh interval used by ``human`` mode.
    """

    def __init__(
        self,
        env,
        render_mode: str = "human",
        title: str | None = None,
        layout: str = "console",
        max_steps: int = 1500,
        pause_sec: float = 0.02,
    ) -> None:
        if render_mode not in ("human", "rgb_array"):
            raise ValueError("render_mode must be 'human' or 'rgb_array'.")
        if layout not in ("console", "simple"):
            raise ValueError("layout must be 'console' or 'simple'.")
        self.env = env
        self.render_mode = render_mode
        self.title = title or (getattr(env, "spec", None) and str(env.spec.env_id)) or "LLRF environment"
        self.layout = layout
        self.max_steps = int(max_steps)
        self.pause_sec = float(pause_sec)
        self._fig = None
        self._ax = None
        self._lines: dict[str, Any] = {}
        self._pv: dict[str, Any] = {}
        self._marker: dict[str, Any] = {}
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
    # Telemetry recording
    # ------------------------------------------------------------------

    def _reset_history(self) -> None:
        self._history = {
            "t": [], "amp": [], "phase": [], "drive": [], "detuning": [],
            "refl": [], "beam": [], "err_pct": [],
        }

    def _record(self) -> None:
        env = self.env
        idx = int(getattr(env, "_step_index", 0) or getattr(env, "step_index", 0)
                  or getattr(env, "_pulse_index", 0))
        tracked = complex(getattr(env, "tracked_signal", 0.0))
        sp_arr = getattr(env, "reference_sp", None)
        reference = complex(sp_arr[min(idx, len(sp_arr) - 1)]) if sp_arr is not None else tracked
        drive = complex(getattr(env, "drive_total", 0.0))
        plant = getattr(env, "plant", None)
        detuning = float(getattr(plant, "detuning", 0.0) or 0.0)
        refl = float(abs(getattr(plant, "vr", 0.0 + 0.0j)))
        beam_val = 0.0
        if hasattr(env, "_beam_at"):
            try:
                beam_val = abs(complex(env._beam_at(idx)))
            except Exception:
                beam_val = 0.0
        vc_target = float(getattr(getattr(env, "cavity", None), "vc_target", 1.0) or 1.0)
        err_pct = abs(reference - tracked) / vc_target * 100.0

        h = self._history
        h["t"].append(idx)
        h["amp"].append(abs(tracked))
        h["phase"].append(np.angle(tracked))
        h["drive"].append(abs(drive))
        h["detuning"].append(detuning)
        h["refl"].append(refl)
        h["beam"].append(beam_val)
        h["err_pct"].append(float(err_pct))
        if len(h["t"]) > self.max_steps:
            for key in h:
                h[key] = h[key][-self.max_steps:]
        self._reference = reference
        self._tracked = tracked

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------

    def _style_axis(self, ax) -> None:
        ax.set_facecolor(_PANEL)
        ax.tick_params(colors=_FG, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.grid(alpha=0.25, color="#30363d")

    def _ensure_figure(self):
        if self._fig is not None:
            return
        import matplotlib.pyplot as plt

        if self.layout == "simple":
            self._build_simple(plt)
        else:
            self._build_console(plt)

    def _build_simple(self, plt) -> None:
        self._fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        self._fig.suptitle(self.title, fontsize=11)
        (ax_amp, ax_pha), (ax_lis, ax_tel) = axes

        (self._lines["amp"],) = ax_amp.plot([], [], lw=1.2, label="tracked")
        (self._lines["ref"],) = ax_amp.plot([], [], lw=1.2, ls="--", color="C3", label="reference")
        ax_amp.set_xlabel("sample")
        ax_amp.set_ylabel("|V| (V)")
        ax_amp.legend(loc="best")

        (self._lines["phase"],) = ax_pha.plot([], [], lw=1.2, label="tracked")
        (self._lines["phref"],) = ax_pha.plot([], [], lw=1.2, ls="--", color="C3", label="reference")
        ax_pha.set_xlabel("sample")
        ax_pha.set_ylabel("phase (rad)")
        ax_pha.legend(loc="best")

        (self._lines["lis"],) = ax_lis.plot([], [], lw=1.0, color="C2", label="I/Q error")
        ax_lis.set_xlabel("I error")
        ax_lis.set_ylabel("Q error")
        ax_lis.legend(loc="best")
        ax_lis.axis("equal")

        (self._lines["drive"],) = ax_tel.plot([], [], lw=1.2, label="|drive|")
        ax_tel.set_xlabel("sample")
        ax_tel.set_ylabel("drive (V)")
        ax_tel.legend(loc="upper left")
        for ax in axes.ravel():
            self._style_axis(ax)
        self._fig.tight_layout(rect=(0, 0, 1, 0.96))

    def _pv_artist(self, ax, label: str, value: str, unit: str, row: int) -> tuple:
        y = 0.97 - row * 0.088
        ax.text(0.03, y, label, transform=ax.transAxes, color="#8b949e", fontsize=9, fontfamily="monospace")
        text = ax.text(
            0.62, y, value, transform=ax.transAxes, color=_FG, fontsize=12,
            fontfamily="monospace", ha="right", va="top",
        )
        ax.text(0.99, y, unit, transform=ax.transAxes, color="#8b949e",
                fontsize=8, fontfamily="monospace", ha="right", va="top")
        return text

    def _build_console(self, plt) -> None:
        self._fig = plt.figure(figsize=(13, 7.5), facecolor=_BG)
        self._fig.suptitle(f"{self.title}   |   step --", color=_FG, fontsize=12, fontweight="bold")

        grid = self._fig.add_gridspec(1, 2, width_ratios=[1.0, 2.1], wspace=0.25,
                                      left=0.04, right=0.97, top=0.90, bottom=0.08)

        # --- Left: PV panel (EPICS-style readouts) -------------------------
        ax_pv = self._fig.add_subplot(grid[0, 0])
        ax_pv.axis("off")
        ax_pv.set_facecolor(_PANEL)
        ax_pv.text(0.5, 0.995, "RF STATUS  (PV)", transform=ax_pv.transAxes, ha="center", va="top",
                   color=_ACCENT, fontsize=10, fontweight="bold")
        pv_rows = [
            ("|Vc|", "amp", "V"),
            ("Phase", "phase_deg", "deg"),
            ("|Forward|", "drive", "V"),
            ("|Reflected|", "refl", "V"),
            ("Detuning", "detuning_hz", "Hz"),
            ("Beam |Vb|", "beam", "V"),
            ("Tracking err", "err_pct", "%"),
            ("Reward", "reward", ""),
        ]
        for row, (name, key, unit) in enumerate(pv_rows):
            self._pv[key] = self._pv_artist(ax_pv, name, "—", unit, row)

        # --- Right: waveform & phasor axes ---------------------------------
        inner = grid[0, 1].subgridspec(2, 2, hspace=0.45, wspace=0.22)
        ax_amp = self._fig.add_subplot(inner[0, 0])
        ax_pha = self._fig.add_subplot(inner[0, 1])
        ax_lis = self._fig.add_subplot(inner[1, 0])
        ax_tel = self._fig.add_subplot(inner[1, 1])

        (self._lines["amp"],) = ax_amp.plot([], [], lw=1.6, color=_ACCENT, label="|Vc|")
        (self._lines["ref"],) = ax_amp.plot([], [], lw=1.4, ls="--", color=_REF_COLOR, label="setpoint")
        ax_amp.set_title("Amplitude  |Vc|", color=_FG, fontsize=9)
        ax_amp.set_xlabel("sample", color=_FG, fontsize=8)
        ax_amp.set_ylabel("V", color=_FG, fontsize=8)
        ax_amp.legend(loc="upper right", fontsize=7, facecolor=_PANEL, labelcolor=_FG)

        (self._lines["phase"],) = ax_pha.plot([], [], lw=1.6, color=_ACCENT, label="tracked")
        (self._lines["phref"],) = ax_pha.plot([], [], lw=1.4, ls="--", color=_REF_COLOR, label="reference")
        ax_pha.set_title("Phase", color=_FG, fontsize=9)
        ax_pha.set_xlabel("sample", color=_FG, fontsize=8)
        ax_pha.set_ylabel("deg", color=_FG, fontsize=8)
        ax_pha.legend(loc="upper right", fontsize=7, facecolor=_PANEL, labelcolor=_FG)

        (self._lines["lis"],) = ax_lis.plot([], [], lw=1.0, color="#3fb950", alpha=0.7, label="trajectory")
        (self._marker["dot"],) = ax_lis.plot([], [], "o", color=_ACCENT, ms=9, label="state (Vc)")
        (self._marker["ref"],) = ax_lis.plot([], [], "*", color=_REF_COLOR, ms=14, label="setpoint")
        ax_lis.set_title("Phasor plane (moving state)", color=_FG, fontsize=9)
        ax_lis.set_xlabel("I (V)", color=_FG, fontsize=8)
        ax_lis.set_ylabel("Q (V)", color=_FG, fontsize=8)
        ax_lis.legend(loc="upper right", fontsize=7, facecolor=_PANEL, labelcolor=_FG)
        ax_lis.set_aspect("equal", adjustable="datalim")

        (self._lines["drive"],) = ax_tel.plot([], [], lw=1.6, color=_ACCENT, label="|drive|")
        (self._lines["refl"],) = ax_tel.plot([], [], lw=1.2, color=_ALARM, label="|reflected|")
        ax_tel.set_title("Drive / Reflected vs Detuning", color=_FG, fontsize=9)
        ax_tel.set_xlabel("sample", color=_FG, fontsize=8)
        ax_tel.set_ylabel("V", color=_FG, fontsize=8)
        ax_tel.legend(loc="upper left", fontsize=7, facecolor=_PANEL, labelcolor=_FG)
        self._ax_det2 = ax_tel.twinx()
        self._lines["detuning"] = self._ax_det2.plot([], [], lw=1.1, color="#bc8cff", alpha=0.8, label="detuning")[0]
        self._ax_det2.set_ylabel("detuning (Hz)", color="#bc8cff", fontsize=8)
        self._ax_det2.tick_params(axis="y", colors="#bc8cff", labelsize=8)

        for ax in (ax_amp, ax_pha, ax_lis, ax_tel):
            for spine in ax.spines.values():
                spine.set_color("#30363d")
            ax.set_facecolor(_PANEL)
            ax.tick_params(colors=_FG, labelsize=8)
            ax.grid(alpha=0.25, color="#30363d")

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

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
        refl = np.asarray(self._history["refl"], dtype=float)

        self._lines["amp"].set_data(t, amp)
        self._lines["phase"].set_data(t, phase)
        self._lines["drive"].set_data(t, drive)
        if "refl" in self._lines:
            self._lines["refl"].set_data(t, refl)
        if "detuning" in self._lines:
            det_hz = np.asarray(self._history["detuning"]) / (2.0 * np.pi)
            self._lines["detuning"].set_data(t, det_hz)
        for key in ("amp", "phase", "drive"):
            if key in self._lines:
                axes = self._lines[key].axes
                axes.relim()
                axes.autoscale_view()

        ref = complex(getattr(self, "_reference", 0.0))
        self._lines["ref"].set_data(t, np.full_like(t, abs(ref)))
        self._lines["phref"].set_data(t, np.full_like(t, np.degrees(np.angle(ref))))

        if "lis" in self._lines and len(t) > 1:
            # Trajectory + moving state marker (the "vehicle").
            self._lines["lis"].set_data(amp * np.cos(phase), amp * np.sin(phase))
            self._marker["dot"].set_data([amp[-1] * np.cos(phase[-1])], [amp[-1] * np.sin(phase[-1])])
            self._marker["ref"].set_data([abs(ref) * np.cos(np.angle(ref))], [abs(ref) * np.sin(np.angle(ref))])
            ax_lis = self._lines["lis"].axes
            ax_lis.relim()
            ax_lis.autoscale_view()

        self._update_console_pvs()
        total = int(getattr(self.env, "episode_len", 0) or (int(t[-1]) + 1 if len(t) else 1))
        step_txt = f"step {int(t[-1]) + 1:d}/{total}" if len(t) else "step --"
        title_color = _FG if self.layout == "console" else "#000000"
        self._fig.suptitle(f"{self.title}   |   {step_txt}", color=title_color)
        self._fig.canvas.draw_idle()

        if mode == "rgb_array":
            return self._to_rgb_array()
        if mode == "human":
            self._fig.canvas.flush_events()
            self._live_show()
        return None

    def _update_console_pvs(self) -> None:
        if not self._pv:
            return
        tracked = complex(getattr(self, "_tracked", 0.0))
        h = self._history
        err = float(h["err_pct"][-1]) if h["err_pct"] else 0.0
        reward = None
        values = {
            "amp": abs(tracked),
            "phase_deg": np.degrees(np.angle(tracked)),
            "drive": float(h["drive"][-1]) if h["drive"] else 0.0,
            "refl": float(h["refl"][-1]) if h["refl"] else 0.0,
            "detuning_hz": (float(h["detuning"][-1]) if h["detuning"] else 0.0) / (2.0 * np.pi),
            "beam": float(h["beam"][-1]) if h["beam"] else 0.0,
            "err_pct": err,
            "reward": reward if reward is not None else 0.0,
        }
        for key, text in self._pv.items():
            value = values.get(key, 0.0)
            color = _FG
            if key == "err_pct" and value > 1.0:
                color = _ALARM
            if key == "refl":
                refl_limit = getattr(getattr(self.env, "env_cfg", None), "reflected_power_limit", None)
                if refl_limit is not None and value > float(refl_limit):
                    color = _ALARM
            text.set_text(f"{value:,.3g}")
            text.set_color(color)

    def _to_rgb_array(self) -> np.ndarray:
        """Snapshot the current figure as an ``(H, W, 3)`` uint8 array."""
        self._fig.canvas.draw()
        buf = np.asarray(self._fig.canvas.buffer_rgba(), dtype=np.uint8)
        return buf[..., :3]
