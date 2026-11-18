###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Baseline policies.

A "baseline" is a classical controller used as the comparison target of an RL
agent.  The following names are understood by the CLI / scenario runner:

* ``zero`` — no compensation (the RL residual / tuner action is zero);
* ``ff``   — nominal analytical feedforward only (identical to ``zero`` for
  the current environments, kept for clarity);
* ``pi``   — a complex-PI residual closed on the tracking error (voltage and
  vector-sum environments) or a PI closed on the detuning for the slow loop
  (resonance / dual-loop environments);
* ``adrc`` — an ADRC-like PD+integral residual using the error and the
  reflected-power proxy (voltage / vector-sum environments).

For the assist environments (``PIAssist-v0`` / ``ADRCAssist-v0``) the
classical controller is already the inner loop, so ``pi`` / ``adrc`` map to
the zero residual action (i.e. "PI only" / "ADRC only").
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

__all__ = ["BASELINE_NAMES", "make_baseline_policy", "zero_policy", "list_baselines"]


BASELINE_NAMES: List[str] = ["zero", "ff", "pi", "adrc"]


def list_baselines() -> List[str]:
    """Return the supported baseline names."""
    return list(BASELINE_NAMES)


def zero_policy(env) -> Callable[[np.ndarray], np.ndarray]:
    """No-compensation policy (all-zero action).

    Args:
        env: Environment whose action space defines the output shape.

    Returns:
        A policy returning the zero action.
    """

    def _policy(obs: np.ndarray) -> np.ndarray:
        return np.zeros(env.action_space.shape, dtype=np.float32)

    return _policy


# ---------------------------------------------------------------------------
# Residual policies (voltage / vector-sum environments)
# ---------------------------------------------------------------------------

def _error_of(env) -> complex:
    """Tracking error at the current sample, read from the environment state."""
    idx = min(env._step_index, env.episode_len - 1)
    sp = env.reference.sp_at(idx)
    tracked = env.plant.vector_sum if hasattr(env.plant, "vector_sum") else env.plant.vc
    return sp - tracked


def make_pi_residual_policy(
    env,
    kp: float = 0.8,
    ki: float = 2.0e5,
) -> Callable[[np.ndarray], np.ndarray]:
    """Complex-PI residual closed on the tracking error.

    Args:
        env: Voltage / vector-sum environment.
        kp:  Proportional gain.
        ki:  Integral gain.

    Returns:
        A policy that maps an observation to a normalized I/Q residual.
    """
    state: Dict[str, complex] = {"integral": 0.0 + 0.0j}

    def _policy(obs: np.ndarray) -> np.ndarray:
        error = _error_of(env)
        state["integral"] += error * env.config.Ts
        correction = -(kp * error + ki * state["integral"])
        limit = max(env.drive_limit, 1.0)
        return np.array(
            [
                np.clip(correction.real / limit, -1.0, 1.0),
                np.clip(correction.imag / limit, -1.0, 1.0),
            ],
            dtype=np.float32,
        )

    return _policy


def make_adrc_like_policy(
    env,
    kp: float = 0.8,
    ki: float = 2.0e5,
    kd: float = 4.0e-2,
    reflection_gain: float = 2.0e-2,
) -> Callable[[np.ndarray], np.ndarray]:
    """ADRC-like PD + integral residual with a reflection-power feed.

    Args:
        env:              Voltage / vector-sum environment.
        kp:               Proportional gain.
        ki:               Integral gain.
        kd:               Derivative gain on the error.
        reflection_gain:  Gain on the reflected voltage (disturbance proxy).

    Returns:
        A policy that maps an observation to a normalized I/Q residual.
    """
    state: Dict[str, complex] = {"integral": 0.0 + 0.0j, "prev": 0.0 + 0.0j}

    def _policy(obs: np.ndarray) -> np.ndarray:
        error = _error_of(env)
        rate = (
            (error - state["prev"]) / max(env.config.Ts, 1.0e-12)
            if "prev" in state and state["prev"] != 0.0 + 0.0j
            else 0.0 + 0.0j
        )
        state["integral"] += error * env.config.Ts
        state["prev"] = error
        reflected = env.plant.vr if hasattr(env.plant, "vr") else env.plant.reflected_sum
        correction = -(kp * error + ki * state["integral"] + kd * rate) - reflection_gain * reflected
        limit = max(env.drive_limit, 1.0)
        return np.array(
            [
                np.clip(correction.real / limit, -1.0, 1.0),
                np.clip(correction.imag / limit, -1.0, 1.0),
            ],
            dtype=np.float32,
        )

    return _policy


# ---------------------------------------------------------------------------
# Slow-loop policies (resonance / dual-loop environments)
# ---------------------------------------------------------------------------

def make_detuning_pi_policy(
    env,
    kp: float = 0.8,
    ki: float = 5.0e3,
) -> Callable[[np.ndarray], np.ndarray]:
    """PI closed on the residual detuning, commanding the slow actuator.

    Args:
        env: Resonance / dual-loop environment (exposes ``plant.detuning``).
        kp:  Proportional gain.
        ki:  Integral gain.

    Returns:
        A policy mapping an observation to a normalized tuner command.
    """
    state: Dict[str, float] = {"integral": 0.0}

    def _policy(obs: np.ndarray) -> np.ndarray:
        detuning = env.plant.detuning
        state["integral"] += detuning * env.config.Ts
        correction = -(kp * detuning + ki * state["integral"])
        return np.array([np.clip(correction / env.detuning_limit, -1.0, 1.0)], dtype=np.float32)

    return _policy


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def make_baseline_policy(
    env,
    name: str,
    kp: float = 0.8,
    ki: float = 2.0e5,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a baseline policy by name and environment type.

    Args:
        env:  Target environment.
        name: One of ``zero`` / ``ff`` / ``pi`` / ``adrc``.
        kp:   Proportional gain used by the PI/ADRC-style baselines.
        ki:   Integral gain used by the PI/ADRC-style baselines.

    Returns:
        A policy ``action(obs) -> np.ndarray``.

    Raises:
        ValueError: If the name is unknown.
    """
    if name not in BASELINE_NAMES:
        raise ValueError(f"unknown baseline {name!r}; available: {BASELINE_NAMES}")

    cls = type(env).__name__
    if name in ("zero", "ff"):
        return zero_policy(env)

    if name == "pi":
        if cls == "CavityVoltageEnv":
            return make_pi_residual_policy(env, kp=kp, ki=ki)
        if cls == "VectorSumEnv":
            return make_pi_residual_policy(env, kp=kp, ki=ki)
        if cls in ("FrequencyEnv", "DualLoopEnv"):
            return make_detuning_pi_policy(env, kp=kp, ki=ki)
        if cls in ("PIAssistEnv", "ADRCAssistEnv"):
            # The classical controller is already the inner loop.
            return zero_policy(env)
        return zero_policy(env)

    if name == "adrc":
        if cls in ("CavityVoltageEnv", "VectorSumEnv"):
            return make_adrc_like_policy(env, kp=kp, ki=ki)
        if cls in ("FrequencyEnv", "DualLoopEnv"):
            return make_detuning_pi_policy(env, kp=kp, ki=ki)
        if cls in ("PIAssistEnv", "ADRCAssistEnv"):
            return zero_policy(env)
        return zero_policy(env)

    return zero_policy(env)
