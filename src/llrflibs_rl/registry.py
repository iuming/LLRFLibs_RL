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

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import gymnasium as gym
import numpy as np

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
        env_id:            Unique environment id (e.g. ``"DualLoop-v0"``).
        description:       Human-readable summary.
        entry:             Environment class (or factory) to instantiate.
        kind:              Control level: ``"voltage"`` / ``"resonance"`` /
                           ``"vectormodule"`` / ``"pulse"``.
        default_kwargs:    Default constructor keyword arguments.
        max_episode_steps: Gymnasium-style horizon metadata (informational;
                           the environments truncate at their own length).
    """

    env_id: str
    description: str
    entry: Callable[..., gym.Env]
    kind: str
    default_kwargs: dict[str, Any] = field(default_factory=dict)
    max_episode_steps: int | None = None
    nondeterministic: bool = False

    @property
    def id(self) -> str:
        """Gymnasium-style spec id (parity alias of ``env_id``)."""
        return self.env_id

    def make(self, **overrides: Any) -> gym.Env:
        """Gymnasium-parity factory: instantiate this spec's environment."""
        kwargs = _default_kwargs_for(self.env_id)
        kwargs.update(overrides)
        env = self.entry(**kwargs)
        env._spec = self  # type: ignore[attr-defined]
        return env


_REGISTRY: dict[str, EnvSpec] = {}


# ---------------------------------------------------------------------------
# Registration / lookup
# ---------------------------------------------------------------------------

def register(
    env_id: str,
    entry: Callable[..., gym.Env],
    description: str = "",
    kind: str = "",
    max_episode_steps: int | None = None,
    **default_kwargs: Any,
) -> None:
    """Register an environment by id.

    Args:
        env_id:            Unique id.
        entry:             Environment class (or factory).
        description:       Human-readable summary.
        kind:              Control level.
        max_episode_steps: Optional horizon metadata (gymnasium parity).
        default_kwargs:    Default constructor arguments.
    """
    if env_id in _REGISTRY:
        raise ValueError(f"environment id already registered: {env_id}")
    _REGISTRY[env_id] = EnvSpec(
        env_id=env_id,
        description=description,
        entry=entry,
        kind=kind,
        default_kwargs=dict(default_kwargs),
        max_episode_steps=max_episode_steps,
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


def list_env_ids() -> list[str]:
    """Return the sorted ids of all registered environments."""
    return sorted(_REGISTRY)


def make(
    env_id: str,
    wrappers: list[Callable[[gym.Env], gym.Env]] | None = None,
    **overrides: Any,
) -> gym.Env:
    """Instantiate a registered environment.

    The registry spec is attached as ``env.spec`` and any wrappers are
    applied, matching the gymnasium ``make`` semantics.

    Args:
        env_id:    Environment id.
        wrappers:  Optional list of wrapper factories applied in order.
        overrides: Constructor arguments that replace the defaults.

    Returns:
        A new environment instance (possibly wrapped).
    """
    spec = get_spec(env_id)
    # Copy the spec so the env-derived horizon never mutates the shared
    # registry entry (each ``make`` gets its own derived ``max_episode_steps``).
    spec = replace(spec)
    kwargs = _default_kwargs_for(env_id)
    kwargs.update(overrides)
    env = spec.entry(**kwargs)
    env._spec = spec  # type: ignore[attr-defined]
    # Gymnasium parity: expose the episode horizon on the spec.
    #  * pulse mode -> ``episode_len`` samples (for the pulse envs this is the
    #    N-*pulse period* adaptation window, one step per pulse);
    #  * cw mode    -> ``None`` (infinite: the loop only ends on a protective
    #    trip, mirroring continuous-wave LLRF operation).
    if spec.max_episode_steps is None and hasattr(env, "episode_len"):
        mode = getattr(getattr(env, "env_cfg", None), "mode", "pulse")
        spec.max_episode_steps = None if mode == "cw" else int(env.episode_len)
    wrapped = env
    for wrapper in wrappers or []:
        wrapped = wrapper(wrapped)
    return wrapped


# ---------------------------------------------------------------------------
# Default construction presets per environment id
# ---------------------------------------------------------------------------

def _default_kwargs_for(env_id: str) -> dict[str, Any]:
    """Build the default constructor arguments of an environment (lazily).

    The configuration objects of the Level-L4 environments import the env
    classes, so they are imported here instead of at module scope to avoid a
    circular import with ``llrflibs_rl.envs``.
    """

    vs_cavities = [nominal_sc_cavity() for _ in range(3)]
    table: dict[str, dict[str, Any]] = {
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

    l4: dict[str, dict[str, Any]] = {
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
    v1: dict[str, dict[str, Any]] = {
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
