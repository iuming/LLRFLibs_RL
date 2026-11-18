###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""Validation and presets of the configuration layer."""

import numpy as np
import pytest

from llrflibs_rl.configs import (
    ADRCControllerConfig,
    AgentConfig,
    CavityParams,
    DualLoopEnvConfig,
    FrequencyEnvConfig,
    PIControllerConfig,
    ResidualEnvConfig,
    RewardConfig,
    TunerActuatorConfig,
    VectorSumEnvConfig,
    nominal_nc_cavity,
    nominal_sc_cavity,
)


def test_cavity_validation():
    with pytest.raises(ValueError):
        CavityParams(half_bw=0.0)
    with pytest.raises(ValueError):
        CavityParams(half_bw=1.0, vc_target=-1.0)
    with pytest.raises(ValueError):
        CavityParams(half_bw=1.0, machine="wrong")


def test_cavity_properties():
    cav = CavityParams(half_bw=1.0, beam_current=5e-3, mech_modes={"f": [1], "Q": [1], "K": [1]})
    assert cav.has_beam
    assert not cav.has_passband_modes
    assert cav.has_mech_modes
    cav2 = CavityParams.from_dict(cav.to_dict())
    assert cav2 == cav


def test_cavity_presets():
    sc = nominal_sc_cavity()
    assert sc.half_bw > 0
    assert sc.has_beam
    assert sc.has_mech_modes
    nc = nominal_nc_cavity()
    assert not nc.has_mech_modes
    assert not nc.has_beam


def test_env_config_validation():
    with pytest.raises(ValueError):
        ResidualEnvConfig(Ts=-1.0)
    with pytest.raises(ValueError):
        ResidualEnvConfig(measurement_noise_std=-0.1)
    with pytest.raises(ValueError):
        ResidualEnvConfig(drive_effort_reference="bad")
    with pytest.raises(ValueError):
        FrequencyEnvConfig(detuning_limit=-1.0)
    with pytest.raises(ValueError):
        DualLoopEnvConfig(actuator_tau=-1.0)
    with pytest.raises(ValueError):
        VectorSumEnvConfig(drive_limit_ratio=-1.0)

    cfg = ResidualEnvConfig()
    assert np.isclose(cfg.microphonics_amp, 2.0 * np.pi * 200.0)


def test_controller_and_agent_config_validation():
    with pytest.raises(ValueError):
        PIControllerConfig(kp=-1.0)
    with pytest.raises(ValueError):
        ADRCControllerConfig(observer_pole_scale=0.0)
    with pytest.raises(ValueError):
        TunerActuatorConfig(tau=-1.0)
    with pytest.raises(ValueError):
        AgentConfig(algorithm="unknown")
    with pytest.raises(ValueError):
        AgentConfig(gamma=2.0)


def test_reward_config_defaults():
    cfg = RewardConfig()
    assert cfg.error_weight >= 0.0
    assert cfg.filling_scale <= cfg.flattop_scale
    assert len(cfg.region_scales) == 3
