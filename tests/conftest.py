###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Shared pytest fixtures.

A minimal, fast cavity preset is provided so unit tests do not depend on the
full 1800-sample scenario defaults.  Imports of the package already register
all environments.
"""

from __future__ import annotations

import numpy as np
import pytest

import llrflibs_rl as rl  # noqa: F401  (importing the package registers environments)


@pytest.fixture
def fast_cavity():
    """Small SC cavity tuned for quick tests (fewer flattop samples)."""
    from llrflibs_rl.configs.cavity import CavityParams

    return CavityParams(
        half_bw=2.0 * np.pi * 216.0 * 0.5,
        vc_target=25e6,
        detuning=2.0 * np.pi * 500.0,
        filling_len=200,
        flattop_len=300,
        beam_current=8e-3,
        beam_phase_deg=0.0,
    )


@pytest.fixture
def fast_env_config():
    """Loop config with a short episode and no noise for fast tests."""
    from llrflibs_rl.configs.env import ResidualEnvConfig

    return ResidualEnvConfig(
        episode_len=600,
        measurement_noise_std=0.0,
        loop_delay_steps=0,
        randomize_detuning=0.0,
        microphonics_amp=0.0,
        max_drive_abs=2.0e8,
        max_drive_slew_per_step=1.0e7,
    )
