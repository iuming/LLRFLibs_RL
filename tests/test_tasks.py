###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""Reward / region / event-helpers behavior."""

import numpy as np
import pytest

from llrflibs_rl.configs.cavity import nominal_sc_cavity
from llrflibs_rl.configs.task import RewardConfig
from llrflibs_rl.tasks.events import apply_drive_limits, phase_error_norm
from llrflibs_rl.tasks.observations import build_voltage_obs
from llrflibs_rl.tasks.reference import PulseReference
from llrflibs_rl.tasks.rewards import RegionScaler, RewardFactory


def test_region_scaler_ramp():
    scaler = RegionScaler(filling_len=100, flattop_len=200, scales=(0.5, 2.0, 0.02), ramp=10)
    assert scaler(-1) == 0.5
    assert scaler(100) == 2.0
    assert scaler(300) == 0.02
    # Smooth transitions: no jumps of the full step.
    values = [scaler(i) for i in range(-20, 330)]
    assert max(abs(values[i + 1] - values[i]) for i in range(len(values) - 1)) < 0.5


def test_reward_factory_error_dominant():
    cfg = RewardConfig()
    factory = RewardFactory(cfg, obs_scale=25e6, filling_len=100, flattop_len=200,
                            drive_limit=2.5e6, drive_effort_scale=50e6)
    # Perfect tracking gives a near-zero reward.
    r_perfect = factory.compute(
        step_index=150, vc=25e6 + 0j, sp=25e6 + 0j, drive_total=50e6 + 0j,
        vf_ff=50e6 + 0j, reflected=0j, residual=0j,
    )
    # A large tracking error is penalized much more than a modest drive change.
    r_bad = factory.compute(
        step_index=150, vc=20e6 + 0j, sp=25e6 + 0j, drive_total=50e6 + 2e6j,
        vf_ff=50e6 + 0j, reflected=1e6 + 0j, residual=1e6 + 0j,
    )
    assert abs(r_perfect) < 0.1
    assert r_bad < -0.1
    assert r_bad < r_perfect


def test_apply_drive_limits():
    # The slew limit dominates on the very first step: only 1e8 of the 3e8 target
    # can be realized in one sample, so the (post-limit) drive is 1e8.
    drive, sat, slew = apply_drive_limits(complex(3e8), complex(0), max_abs=2e8, max_slew=1e8)
    assert abs(drive) == pytest.approx(1e8)
    assert sat > 0.0 and sat <= 1.0
    assert slew == pytest.approx(1.0)  # clipped to 1
    # With no slew limit, only the amplitude limit applies.
    drive2, sat2, slew2 = apply_drive_limits(complex(3e8), complex(0), max_abs=2e8, max_slew=None)
    assert abs(drive2) == pytest.approx(2e8)
    assert sat2 == pytest.approx(0.5)
    drive3, sat3, slew3 = apply_drive_limits(complex(1e6), complex(0), max_abs=None, max_slew=None)
    assert abs(drive3) == pytest.approx(1e6)
    assert sat3 == 0.0 and slew3 == 0.0


def test_phase_error_norm():
    assert phase_error_norm(1.0 + 0j, 0.0 + 1j, threshold=0.05) == pytest.approx(0.5)
    assert phase_error_norm(1.0 + 0j, 1.0 + 0j, threshold=0.05) == pytest.approx(0.0)


def test_observation_shapes():
    obs = build_voltage_obs(
        vc=1j, sp=1j, err=0j, vr=0j, vf_ff=1j, vb=0j, last_action=0j,
        obs_scale=25e6, drive_limit=2.5e6, step_index=0, episode_len=1800,
    )
    assert obs.shape == (15,)
    assert obs.dtype == np.float32


def test_pulse_reference_waveforms():
    cavity = nominal_sc_cavity()
    ref = PulseReference(cavity, 1e-6, 1800)
    assert ref.sp.shape == (1800,)
    assert abs(ref.sp_at(cavity.filling_len + 400)) > 0.5 * cavity.vc_target
    assert ref.peak_drive > 0.0
