###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""Controller-layer behavior tests."""

import numpy as np

from llrflibs_rl.configs.controller import PIControllerConfig, TunerActuatorConfig
from llrflibs_rl.controllers import (
    ADRCController,
    AnalyticFeedforward,
    ComplexPIController,
    FirstOrderActuator,
    PTypeILC,
    ZeroActuator,
)
from llrflibs_rl.tasks.reference import PulseReference

from llrflibs_rl.configs.cavity import nominal_sc_cavity


def test_complex_pi_responses_to_error():
    ctrl = ComplexPIController(PIControllerConfig(kp=1.0, ki=1e5), Ts=1e-6)
    ctrl.reset()
    drive, fb = ctrl.step(error=1e6 + 0j, feedforward=0j)
    assert np.isfinite(drive.real)
    assert fb.real > 0  # proportional term pushes against the error


def test_adrc_builds_and_steps():
    cavity = nominal_sc_cavity()
    ctrl = ADRCController(compound_config(), Ts=1e-6, half_bw=cavity.half_bw)
    ctrl.reset()
    drive, fb, vc_est, f_est = ctrl.step(sp=cavity.vc_target + 0j, vc_meas=0j,
                                         vf_meas=0j, feedforward=0j)
    assert np.isfinite(drive.real) and np.isfinite(f_est.real)


def compound_config():
    from llrflibs_rl.configs.controller import ADRCControllerConfig
    return ADRCControllerConfig(kp=0.6, ki=1e5, observer_pole_scale=200.0)


def test_actuator_converges_to_target():
    actuator = FirstOrderActuator(TunerActuatorConfig(detuning_limit=2e3, tau=100e-6), Ts=1e-6)
    actuator.reset()
    for _ in range(2000):
        actuator.step(0.5)  # command half range
    assert actuator.state == pytest_approx(1e3, 50.0)


def pytest_approx(value, tol):
    import pytest
    return pytest.approx(value, abs=tol)


def test_zero_actuator():
    actuator = ZeroActuator()
    assert actuator.step(1.0) == 0.0


def test_ilc_learns():
    ilc = PTypeILC(episode_len=100, gain=0.5, drive_limit=1e3)
    ilc.learn(np.full(100, 100.0 + 0.0j))
    assert abs(ilc.drive_at(10)) == pytest_approx(50.0, 1e-9)
    ilc.learn(np.full(100, 100.0 + 0.0j))
    assert abs(ilc.drive_at(10)) == pytest_approx(100.0, 1e-9)


def test_analytic_feedforward():
    cavity = nominal_sc_cavity()
    ref = PulseReference(cavity, 1e-6, 1800)
    ff = AnalyticFeedforward(ref)
    assert ff.drive_at(100) == ref.vf_ff_at(100)
