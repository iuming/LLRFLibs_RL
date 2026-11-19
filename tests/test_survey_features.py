###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Tests for the repository-survey features: Bayesian auto-tuner, channel/HIL layer,
quench detection, KPI catalog, beam phase keep-up and firmware decimation.
"""

from __future__ import annotations

import numpy as np
import pytest

import llrflibs_rl as rl

# ---------------------------------------------------------------------------
# Bayesian / GP auto-tuner (agents/optimizers)
# ---------------------------------------------------------------------------

def test_tune_scalar_finds_function_peak():
    from llrflibs_rl.agents.optimizers import tune_scalar

    # Smooth single-peak objective (typical tuner setpoint problem).
    def objective(x: float) -> float:
        return -((x - 0.3) ** 2) - 0.1 * abs(x - 0.7)

    best, history = tune_scalar(objective, (0.0, 1.0), n_init=3, n_iters=16, seed=0)
    assert abs(best - 0.3) < 0.1
    assert len(history) == 16
    assert max(history) > -0.05


def test_bayesian_optimizer_api():
    from llrflibs_rl.agents.optimizers import BayesianOptimizer

    opt = BayesianOptimizer((0.0, 1.0), n_init=2, seed=1)
    for _ in range(8):
        x = opt.suggest()
        y = -((x - 0.5) ** 2)
        opt.update(x, y)
    x_best, y_best = opt.best()
    assert abs(x_best - 0.5) < 0.2
    assert y_best > -0.1


# ---------------------------------------------------------------------------
# Channel / HIL layer (utils/hil)
# ---------------------------------------------------------------------------

def test_simulated_channel_get_put_and_policy_loop(fast_cavity, fast_env_config):
    from llrflibs_rl.controllers.baselines import zero_policy
    from llrflibs_rl.utils.hil import SimulatedChannel, emulate_policy_loop

    factory = lambda: rl.make(  # noqa: E731
        "CavityVoltage-v0", cavity=fast_cavity, env_cfg=fast_env_config
    )
    channel = SimulatedChannel("CavityVoltage-v0", get_attr="plant.vc", env_factory=factory)
    channel.reset(seed=0)
    value0 = channel.get()
    assert np.isfinite(abs(value0))
    action = np.array([1.0, 0.0], dtype=np.float32)
    # The environment models a one-sample command transport delay, so the
    # first command only acts from the second sample.
    channel.step(action)
    channel.step(action)
    assert channel.get() != value0

    loop = emulate_policy_loop(channel, zero_policy(factory()), steps=20, seed=1)
    assert loop["length"] == 20
    assert loop["value"].shape == (20,)


def test_epics_channel_requires_pyepics():
    from llrflibs_rl.utils.hil import EPICSChannel

    try:
        import epics  # noqa: F401

        has_epics = True
    except ImportError:
        has_epics = False

    if has_epics:
        channel = EPICSChannel("sim:test")
        channel.put(1.0)  # writes to a non-existent PV are harmless
    else:
        with pytest.raises(ImportError):
            EPICSChannel("sim:test")


# ---------------------------------------------------------------------------
# Quench detection (sensors/quench)
# ---------------------------------------------------------------------------

def test_quench_detector_fires_on_growth_only():
    from llrflibs_rl.sensors import QuenchDetector, detect_quench_index

    dt = 1.0e-6
    n = 100
    flat = np.full(n, 2.0e6, dtype=float)
    assert detect_quench_index(flat, window=8, rate_threshold=1.0e9, dt=dt) == -1

    growth = flat.copy()
    growth[60:] += 2.0e9 * np.arange(40).astype(float) * dt  # rise 2e9 V/s
    detector = QuenchDetector(window=8, rate_threshold=1.0e9, abs_threshold=1.0e6, dt=dt)
    fired = [detector.update(float(v)) for v in growth]
    assert any(fired)
    assert detector.tripped
    idx = detect_quench_index(growth, window=8, rate_threshold=1.0e9, dt=dt)
    assert idx >= 60


def test_quench_detect_env_flag_creates_detector(fast_cavity, fast_env_config):
    from dataclasses import replace

    config = replace(fast_env_config, quench_detect=True, quench_rate_threshold=1.0e9)
    env = rl.make("CavityVoltage-v0", cavity=fast_cavity, env_cfg=config)
    env.reset(seed=0)
    assert getattr(env, "_quench", None) is not None

    config2 = replace(fast_env_config, quench_detect=False)
    env2 = rl.make("CavityVoltage-v0", cavity=fast_cavity, env_cfg=config2)
    env2.reset(seed=0)
    assert getattr(env2, "_quench", None) is None


# ---------------------------------------------------------------------------
# KPI catalog (evaluation/kpis)
# ---------------------------------------------------------------------------

def test_kpi_report_structure(fast_cavity, fast_env_config):
    from llrflibs_rl.controllers.baselines import zero_policy
    from llrflibs_rl.evaluation.kpis import KPIS, kpi_report

    env = rl.make("CavityVoltage-v0", cavity=fast_cavity, env_cfg=fast_env_config)
    report = kpi_report(env, zero_policy(env), seed=0)
    assert set(report) == set(KPIS)
    for key, item in report.items():
        assert isinstance(item["value"], float)
        assert item["threshold"] == KPIS[key].threshold
        assert isinstance(item["passed"], bool)
    assert report["amp_rms_pct"]["value"] >= 0.0
    assert report["phase_rms_deg"]["value"] >= 0.0


# ---------------------------------------------------------------------------
# Beam phase keep-up (controllers/feedforward)
# ---------------------------------------------------------------------------

def test_beam_keepup_correction_holds_cavity_empty(fast_cavity):
    from llrflibs_rl.controllers.feedforward import beam_keepup_correction
    from llrflibs_rl.plants.cavity_voltage import CavityVoltagePlant

    beam = 5.0e5 * (0.9 + 0.2j)
    correction = beam_keepup_correction(beam)
    # |-drive| = (beta+1)/beta * |beam| ~ |beam| for SC coupling.
    assert abs(correction) == pytest.approx(abs(beam) * (10001.0 / 10000.0), rel=1e-6)

    Ts = 1.0e-6
    plant = CavityVoltagePlant(fast_cavity, Ts, fast_cavity.detuning)
    n = 3000
    vc_keepup = np.zeros(n, dtype=np.complex128)
    vc_none = np.zeros(n, dtype=np.complex128)
    plant.reset()
    for i in range(n):
        vc_keepup[i], _ = plant.step(beam_keepup_correction(beam), beam=beam)
    plant.reset()
    for i in range(n):
        vc_none[i], _ = plant.step(0.0 + 0.0j, beam=beam)
    late = slice(2000, 3000)
    rms_keepup = float(np.sqrt(np.mean(np.abs(vc_keepup[late]) ** 2)))
    rms_none = float(np.sqrt(np.mean(np.abs(vc_none[late]) ** 2)))
    assert rms_none > 0.0
    assert rms_keepup < rms_none * 0.05


def test_analytic_beam_feedforward_on_reference(fast_cavity, fast_env_config):
    from llrflibs_rl.controllers.feedforward import AnalyticBeamFeedforward

    env = rl.make("CavityVoltage-v0", cavity=fast_cavity, env_cfg=fast_env_config)
    ff = AnalyticBeamFeedforward(env.reference)
    value = ff.drive_at(10, beam=1.0e6)
    nominal = env.reference.vf_ff_at(10)
    assert np.isfinite(abs(value))
    assert value != nominal  # beam correction shifts the drive


# ---------------------------------------------------------------------------
# Firmware decimation (plants/measurement)
# ---------------------------------------------------------------------------

def test_measurement_chain_decimation(fast_cavity):
    from llrflibs_rl.plants.measurement import MeasurementChain

    chain = MeasurementChain(delay_steps=0, noise_std=0.0, decimation=4)
    wave = np.ones(16, dtype=np.complex128)
    out = chain.process_wave(wave)
    assert out.shape == wave.shape
    assert np.allclose(out, 1.0)  # DC passes the averaging unchanged

    ramp = np.linspace(0.0, 1.0, 16) * (1 + 0j)
    out_ramp = chain.process_wave(ramp)
    assert np.all(np.isfinite(out_ramp))
