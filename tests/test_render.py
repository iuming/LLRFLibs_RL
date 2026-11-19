###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Tests for the CartPole-style visualization wrapper (RenderableEnv).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless CI backend; must precede any pyplot use

import numpy as np

import llrflibs_rl as rl


def _env(fast_cavity, fast_env_config):
    from llrflibs_rl.envs.render import RenderableEnv

    return RenderableEnv(
        rl.make("CavityVoltage-v0", cavity=fast_cavity, env_cfg=fast_env_config),
        render_mode="rgb_array",
    )


def test_renderable_env_rgb_array(fast_cavity, fast_env_config):
    from llrflibs_rl.controllers.baselines import zero_policy

    env = _env(fast_cavity, fast_env_config)
    policy = zero_policy(env)
    env.reset(seed=0)
    for _ in range(40):
        env.step(policy(np.zeros(2, dtype=np.float32)))
    frame = env.render("rgb_array")
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.dtype == np.uint8 and frame.size > 0
    env.close()


def test_renderable_env_human_and_attribute_forwarding(fast_cavity, fast_env_config):
    env = _env(fast_cavity, fast_env_config)
    env.reset(seed=0)
    env.step(np.zeros(2, dtype=np.float32))
    assert env.render("human") is None
    # Unknown attributes are forwarded to the wrapped environment.
    assert hasattr(env, "tracked_signal")
    assert np.isfinite(abs(complex(env.tracked_signal)))
    assert env.spec is not None
    env.close()


def test_live_human_render_is_safe_on_agg(fast_cavity, fast_env_config):
    from llrflibs_rl.envs.render import RenderableEnv

    env = RenderableEnv(
        rl.make("CavityVoltage-v0", cavity=fast_cavity, env_cfg=fast_env_config),
        render_mode="human", pause_sec=0.0,
    )
    env.reset(seed=0)
    for _ in range(10):
        env.step(np.zeros(2, dtype=np.float32))
        assert env.render("human") is None  # no window and no pause on Agg
    env.close()


class _FakeModel:
    """Minimal policy object used by the callback test (no SB3 needed)."""

    num_timesteps = 0

    def predict(self, obs, deterministic: bool = True):
        return np.zeros(2, dtype=np.float32), None


def test_live_render_callback_triggers_periodically(fast_cavity, fast_env_config):
    from llrflibs_rl.agents.render import LiveRenderCallback

    factory = lambda: rl.make(  # noqa: E731
        "CavityVoltage-v0", cavity=fast_cavity, env_cfg=fast_env_config
    )
    callback = LiveRenderCallback(factory, render_freq=4, render_steps=5, pause_sec=0.0)
    callback.init_callback(_FakeModel())
    assert callback.rendered_count() == 0
    callback.on_step()  # first training step -> the persistent window opens
    assert callback.rendered_count() == 1
    for _ in range(3):  # steps 2..4; refresh again at step 4 (4 %% 4 == 0)
        callback.on_step()
    assert callback.rendered_count() == 2
    callback._on_training_end()  # closes the persistent window, no error
    assert callback._render_env is None


def test_console_layout_renders_and_marker_moves(fast_cavity, fast_env_config):
    from llrflibs_rl.envs.render import RenderableEnv

    env = RenderableEnv(
        rl.make("CavityVoltage-v0", cavity=fast_cavity, env_cfg=fast_env_config),
        render_mode="rgb_array", layout="console",
    )
    env.reset(seed=0)
    action = np.array([1.0, 0.0], dtype=np.float32)
    env.step(action)
    env.step(action)
    frame = env.render("rgb_array")
    assert frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8
    x0 = np.asarray(env._marker["dot"].get_xdata())
    env.step(action)
    env.step(action)
    env.render("rgb_array")
    x1 = np.asarray(env._marker["dot"].get_xdata())
    assert x0.shape == x1.shape and not np.array_equal(x0, x1)  # the state marker moves
    assert set(env._pv) >= {"amp", "phase_deg", "detuning_hz", "err_pct", "reward"}  # PV panel present
    env.close()


def test_simple_layout_still_works(fast_cavity, fast_env_config):
    from llrflibs_rl.envs.render import RenderableEnv

    env = RenderableEnv(
        rl.make("CavityVoltage-v0", cavity=fast_cavity, env_cfg=fast_env_config),
        render_mode="rgb_array", layout="simple",
    )
    env.reset(seed=0)
    env.step(np.array([1.0, 0.0], dtype=np.float32))
    frame = env.render("rgb_array")
    assert frame.ndim == 3 and frame.dtype == np.uint8
    env.close()


def test_http_streamer_serves_frames():
    import urllib.request

    from llrflibs_rl.utils.live_http import HttpStreamer

    stream = HttpStreamer(port=0).start()
    try:
        # Placeholder frame is served before the first publish.
        with urllib.request.urlopen(stream.url + "frame") as response:
            assert response.status == 200
            assert response.headers["Content-Type"] == "image/png"
        payload = b"\x89PNG\r\n\x1a\n\x00fake-payload"
        stream.publish(payload)
        with urllib.request.urlopen(stream.url + "frame?q=1") as response:
            assert response.read() == payload
        with urllib.request.urlopen(stream.url) as response:
            html = response.read().decode("utf-8")
        assert "LLRF live console" in html
    finally:
        stream.close()


def test_renderable_pushes_frames_to_streamer(fast_cavity, fast_env_config):
    import time

    from llrflibs_rl.envs.render import RenderableEnv
    from llrflibs_rl.utils.live_http import HttpStreamer

    stream = HttpStreamer(port=0).start()
    try:
        env = RenderableEnv(
            rl.make("CavityVoltage-v0", cavity=fast_cavity, env_cfg=fast_env_config),
            render_mode="rgb_array", streamer=stream,
        )
        env.reset(seed=0)
        env.step(np.array([1.0, 0.0], dtype=np.float32))
        env.render("rgb_array")
        time.sleep(0.2)  # let the push throttle elapse
        env.render("rgb_array")
        assert stream.frame_count >= 1
    finally:
        stream.close()
