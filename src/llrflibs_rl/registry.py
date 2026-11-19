###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Environment registry (gymnasium-style).

Built-in environments are registered by id and can be instantiated with
:func:`make`.  Default constructor arguments (a nominal cavity plus the
recommended configuration presets) are provided per environment so that a
single ``make("DualLoop-v0")`` returns a runnable environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

import gymnasium as gym

from .configs import (
    ADRCControllerConfig,
    DualLoopEnvConfig,
    FrequencyEnvConfig,
    PIControllerConfig,
    ResidualEnvConfig,
    RewardConfig,
    VectorSumEnvConfig,
    nominal_sc_cavity,
)

__all__ = ["EnvSpec", "get_spec", "list_env_ids", "make", "recommended_timesteps", "register"]


@dataclass
class EnvSpec:
    """Description of a registered environment.

    Attributes:
        env_id:      Unique environment id (e.g. ``"DualLoop-v0"``).
        description: Human-readable summary.
        entry:       Environment class (or factory) to instantiate.
        kind:        Control level: ``"voltage"`` / ``"resonance"`` / ``"vectormodule"``.
        default_kwargs: Default constructor keyword arguments.
    """

    env_id: str
    description: str
    entry: Callable[..., gym.Env]
    kind: str
    default_kwargs: Dict[str, Any] = field(default_factory=dict)


_REGISTRY: Dict[str, EnvSpec] = {}


# ---------------------------------------------------------------------------
# Registration / lookup
# ---------------------------------------------------------------------------

def register(
    env_id: str,
    entry: Callable[..., gym.Env],
    description: str = "",
    kind: str = "",
    **default_kwargs: Any,
) -> None:
    """Register an environment by id.

    Args:
        env_id:          Unique id.
        entry:           Environment class (or factory).
        description:     Human-readable summary.
        kind:            Control level.
        default_kwargs:  Default constructor arguments.
    """
    if env_id in _REGISTRY:
        raise ValueError(f"environment id already registered: {env_id}")
    _REGISTRY[env_id] = EnvSpec(
        env_id=env_id,
        description=description,
        entry=entry,
        kind=kind,
        default_kwargs=dict(default_kwargs),
    )


def get_spec(env_id: str) -> EnvSpec:
    """Return the spec of a registered environment.

    Args:
        env_id: Environment id.

    Returns:
        The corresponding :class:`EnvSpec`.

    Raises:
        KeyError: If the id is unknown.
    """
    try:
        return _REGISTRY[env_id]
    except KeyError:
        valid = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"unknown environment id {env_id!r}; registered ids: {valid}") from None


def list_env_ids() -> List[str]:
    """Return the sorted ids of all registered environments."""
    return sorted(_REGISTRY)


def make(env_id: str, **overrides: Any) -> gym.Env:
    """Instantiate a registered environment.

    Args:
        env_id:    Environment id.
        overrides: Constructor arguments that replace the defaults.

    Returns:
        A new environment instance.
    """
    spec = get_spec(env_id)
    kwargs = _default_kwargs_for(env_id)
    kwargs.update(overrides)
    return spec.entry(**kwargs)


# ---------------------------------------------------------------------------
# Default construction presets per environment id
# ---------------------------------------------------------------------------

def _default_kwargs_for(env_id: str) -> Dict[str, Any]:
    """Build the default constructor arguments of an environment (lazily).

    The configuration objects of the Level-L4 environments import the env
    classes, so they are imported here instead of at module scope to avoid a
    circular import with ``llrflibs_rl.envs``.
    """
    from .configs import (
        ADRCControllerConfig,
        DualLoopEnvConfig,
        FrequencyEnvConfig,
        PIControllerConfig,
        ResidualEnvConfig,
        RewardConfig,
        VectorSumEnvConfig,
        nominal_sc_cavity,
    )

    vs_cavities = [nominal_sc_cavity() for _ in range(3)]
    table: Dict[str, Dict[str, Any]] = {
        "CavityVoltage-v0": {
            "cavity": nominal_sc_cavity(),
            "env_cfg": ResidualEnvConfig(episode_len=1500, microphonics_random_phase=True),
            "reward_cfg": RewardConfig(),
        },
        "PIAssist-v0": {
            "cavity": nominal_sc_cavity(),
            "env_cfg": ResidualEnvConfig(),
            "reward_cfg": RewardConfig(),
            "pi_cfg": PIControllerConfig(),
        },
        "ADRCAssist-v0": {
            "cavity": nominal_sc_cavity(),
            "env_cfg": ResidualEnvConfig(),
            "reward_cfg": RewardConfig(),
            "adrc_cfg": ADRCControllerConfig(),
        },
        "FrequencyControl-v0": {
            "cavity": nominal_sc_cavity(),
            "env_cfg": FrequencyEnvConfig(),
            "reward_cfg": RewardConfig(),
        },
        "DualLoop-v0": {
            "cavity": nominal_sc_cavity(),
            "env_cfg": DualLoopEnvConfig(),
            "reward_cfg": RewardConfig(),
        },
        "VectorSum-v0": {
            "cavities": vs_cavities,
            "env_cfg": VectorSumEnvConfig(cavities=vs_cavities),
            "reward_cfg": RewardConfig(),
        },
    }
    if env_id in table:
        return table[env_id]

    from .envs.gain_scheduling import GainSchedulingConfig
    from .envs.pulse_to_pulse import AdaptivePulseConfig, BeamLoadingConfig

    l4: Dict[str, Dict[str, Any]] = {
        "BeamLoading-v0": {
            "cavity": nominal_sc_cavity(),
            "env_cfg": BeamLoadingConfig(episode_len=1500, microphonics_random_phase=True),
            "reward_cfg": RewardConfig(),
        },
        "AdaptivePulse-v0": {
            "config": AdaptivePulseConfig(cavity=nominal_sc_cavity(), n_pulses=10),
        },
        "VectorSumTuner-v0": {
            "cavities": vs_cavities,
            "env_cfg": VectorSumEnvConfig(
                cavities=vs_cavities, tuner_action=True, episode_len=1800,
                randomize_detuning=2.0 * np.pi * 150.0,
                tuner_detuning_limit=2.0 * np.pi * 1500.0,
            ),
            "reward_cfg": RewardConfig(),
        },
        "GainScheduling-v0": {
            "cavity": nominal_sc_cavity(),
            "env_cfg": GainSchedulingConfig(episode_len=1500, microphonics_random_phase=True),
            "reward_cfg": RewardConfig(),
        },
    }
    if env_id in l4:
        return l4[env_id]

    # Short-horizon resolution variants (v1 series). Episode length stays
    # above filling+flattop (1130 samples) of the nominal cavity.
    v1: Dict[str, Dict[str, Any]] = {
        "CavityVoltage-v1": {
            "cavity": nominal_sc_cavity(),
            "env_cfg": ResidualEnvConfig(episode_len=1400, microphonics_random_phase=True),
            "reward_cfg": RewardConfig(),
        },
        "PIAssist-v1": {
            "cavity": nominal_sc_cavity(),
            "env_cfg": ResidualEnvConfig(episode_len=1400),
            "reward_cfg": RewardConfig(),
            "pi_cfg": PIControllerConfig(),
        },
        "ADRCAssist-v1": {
            "cavity": nominal_sc_cavity(),
            "env_cfg": ResidualEnvConfig(episode_len=1400),
            "reward_cfg": RewardConfig(),
            "adrc_cfg": ADRCControllerConfig(),
        },
        "DualLoop-v1": {
            "cavity": nominal_sc_cavity(),
            "env_cfg": DualLoopEnvConfig(episode_len=1400),
            "reward_cfg": RewardConfig(),
        },
        "VectorSum-v1": {
            "cavities": vs_cavities,
            "env_cfg": VectorSumEnvConfig(cavities=vs_cavities, episode_len=1400),
            "reward_cfg": RewardConfig(),
        },
        "AdaptivePulse-v1": {
            "config": AdaptivePulseConfig(cavity=nominal_sc_cavity(), n_pulses=20),
        },
    }
    return v1.get(env_id, {})


def recommended_timesteps(env_id: str, quick: bool = False) -> int:
    """Return the recommended training budget of an environment.

    Args:
        env_id: Environment id.
        quick:  Return a small sanity-check budget instead.

    Returns:
        Number of training timesteps.
    """
    if quick:
        return 2000
    return {
        "CavityVoltage-v0": 10000,
        "PIAssist-v0": 40000,
        "ADRCAssist-v0": 40000,
        "FrequencyControl-v0": 2000,
        "DualLoop-v0": 30000,
        "VectorSum-v0": 60000,
        "BeamLoading-v0": 20000,
        "AdaptivePulse-v0": 10000,
        "VectorSumTuner-v0": 20000,
        "GainScheduling-v0": 30000,
        "CavityVoltage-v1": 7000,
        "PIAssist-v1": 25000,
        "ADRCAssist-v1": 25000,
        "DualLoop-v1": 20000,
        "VectorSum-v1": 40000,
        "AdaptivePulse-v1": 15000,
    }.get(env_id, 20000)
