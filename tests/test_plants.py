###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""Physical sanity of the plant layer."""

import numpy as np
import pytest

from llrflibs_rl.configs.cavity import nominal_sc_cavity
from llrflibs_rl.plants import CavityVoltagePlant, ResonancePlant, SinusoidalDisturbance, VectorModulePlant
from llrflibs_rl.tasks.reference import PulseReference


def _plant_defaults():
    cavity = nominal_sc_cavity()
    return cavity, 1e-6, 1800


def test_cavity_plant_charges_with_feedforward():
    """The analytic feedforward must reach the voltage target."""
    cavity, Ts, episode_len = _plant_defaults()
    ref = PulseReference(cavity, Ts, episode_len)
    plant = CavityVoltagePlant(cavity, Ts, detuning=cavity.detuning)
    plant.reset()

    mid = cavity.filling_len + cavity.flattop_len // 2
    vc_mid = 0.0 + 0.0j
    for i in range(episode_len):
        plant.step(ref.vf_ff[i], beam=ref.vb[i])
        if i == mid:
            vc_mid = plant.vc
    # flattop should be near the target within a few percent
    assert abs(vc_mid) >= 0.9 * cavity.vc_target


def test_cavity_plant_euler_matches_zoh_in_steady_state():
    """Constant-drive steady state agrees between the two stepping methods.

    The ZOH and Euler (time-varying) paths differ only by a one-sample phase on
    fast-changing inputs; under a constant drive their steady-state gains agree
    to within the O(Ts*half_bw) Euler error.
    """
    from llrflibs_rl.configs.cavity import CavityParams

    cavity = CavityParams(
        half_bw=2.0 * np.pi * 216.0 * 0.5,
        vc_target=25e6,
        detuning=0.0,
        filling_len=150,
        flattop_len=200,
        beam_current=0.0,
    )
    Ts = 1e-6
    drive = complex(10e6)
    plant_ss = CavityVoltagePlant(cavity, Ts, detuning=0.0)
    plant_ss.reset()
    plant_euler = CavityVoltagePlant(cavity, Ts, detuning=0.0)
    plant_euler.reset()

    # Enough samples for > 8 time constants (tau = 1/half_bw ~ 1470 samples).
    for _ in range(12000):
        plant_ss.step(drive, beam=None)
        plant_euler.step(drive, beam=None, detuning=0.0)
    # Steady-state amplitude: vc = 2*beta/(beta+1) * drive.
    expected = abs(drive) * 2.0 * cavity.beta / (cavity.beta + 1.0)
    assert abs(plant_ss.vc) == pytest.approx(expected, rel=1e-3)
    assert abs(plant_euler.vc) == pytest.approx(expected, rel=1e-2)


def test_resonance_plant_lorentz_detuning():
    """The mechanical (Lorentz) modes produce a detuning response to |vc|."""
    cavity = nominal_sc_cavity()
    plant = ResonancePlant(cavity, 1e-6)
    plant.reset()
    for _ in range(300):
        plant.step(complex(100e3), 0.0 + 0.0j, external_detuning=0.0)
    assert np.isfinite(plant.detuning)
    # An unloaded cavity with drive should develop some detuning from |vc|^2.
    assert abs(plant.detuning) > 0.0


def test_vector_module_layout_validation():
    cavity = nominal_sc_cavity()
    with pytest.raises(ValueError):
        VectorModulePlant([cavity, nominal_sc_cavity(vc_target=10e6)], 1e-6, 0.0)


def test_vector_module_vector_sum():
    n = 3
    cavity = nominal_sc_cavity()
    plant = VectorModulePlant([cavity] * n, 1e-6, reference_detuning=cavity.detuning)
    plant.reset([cavity.detuning] * n)
    vs, vr_sum, vc_list, vr_list = plant.step(complex(10e6), 0.0 + 0.0j)
    assert len(vc_list) == n
    assert vs == sum(vc_list)
    assert np.isfinite(abs(vs))


def test_sinusoidal_disturbance():
    dist = SinusoidalDisturbance(base=100.0, amplitude=50.0, frequency=100.0, Ts=1e-6, episode_len=1000)
    wave = dist.build(phase=0.0, offset=10.0)
    assert wave.shape == (1000,)
    assert np.isclose(wave[0], 110.0)
    assert np.max(wave) <= 160.0 and np.min(wave) >= 60.0
