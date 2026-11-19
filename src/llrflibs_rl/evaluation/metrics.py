###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Physics-aware evaluation metrics.

:func:`evaluate_episode` rolls out one episode with a deterministic policy and
computes the tracking metrics that matter for a real pulse, i.e. the flattop
amplitude / phase error, the whole-pulse RMS error and (for resonance tasks)
the amplitude droop metric.  :func:`evaluate_over_seeds` aggregates several
episodes so that conclusions survive randomness.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

__all__ = ["evaluate_episode", "evaluate_over_seeds", "action_from_model", "cross_check_with_sb3"]


def _wave(env, name: str) -> Optional[np.ndarray]:
    """Fetch a waveform currently stored on the environment (if any)."""
    value = getattr(env, name, None)
    if value is None:
        return None
    return np.asarray(value, dtype=np.complex128).reshape(-1)


def evaluate_episode(
    env,
    action_fn: Callable[[np.ndarray], np.ndarray],
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Roll out one episode and compute physics-aware metrics.

    Args:
        env:       LLRF environment (exposes ``reference_sp`` / ``tracked_signal``).
        action_fn: Policy mapping an observation to an action array.
        seed:      Optional seed for reproducible evaluation.

    Returns:
        Metric dictionary: ``reward``, ``length`` and the error metrics
        relevant to the environment type.
    """
    obs, _ = env.reset(seed=seed)
    rewards = []
    tracked = np.zeros(env.episode_len, dtype=np.complex128)
    while len(rewards) < env.episode_len:
        action = np.asarray(action_fn(obs), dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        tracked[len(rewards) - 1] = env.tracked_signal
        if terminated or truncated:
            break
    n = len(rewards)

    metrics: Dict[str, float] = {
        "reward": float(np.sum(rewards)),
        "length": float(n),
    }
    sp = _wave(env, "reference_sp")
    if sp is not None:
        error = sp[:n] - tracked[:n]
        metrics["error_rms"] = float(np.sqrt(np.mean(np.abs(error) ** 2)))
        cav = getattr(env, "cavity", None)
        if cav is not None and hasattr(cav, "vc_target"):
            metrics["error_rms_pct"] = metrics["error_rms"] / cav.vc_target
            # Flattop-only error metric.
            fl = int(getattr(cav, "filling_len", 0))
            ft = int(getattr(cav, "flattop_len", 0))
            if fl + ft <= n:
                ft_err = error[fl: fl + ft]
                metrics["flattop_error_rms"] = float(np.sqrt(np.mean(np.abs(ft_err) ** 2)))
                metrics["flattop_error_rms_pct"] = metrics["flattop_error_rms"] / cav.vc_target
    else:
        # Resonance-style task: evaluate the amplitude error vs the target.
        cav = getattr(env, "cavity", None)
        target = float(getattr(cav, "vc_target", 1.0))
        amp_err = np.abs(tracked[:n]) - target
        metrics["amp_error_rms"] = float(np.sqrt(np.mean(amp_err**2)))
    return metrics


def action_from_model(model, deterministic: bool = True):
    """Build a deterministic action function from a trained SB3 model.

    Observations are normalized with the model's ``VecNormalize`` wrapper when
    the model was trained with observation normalization.

    Args:
        model:       Trained stable-baselines3 model.
        deterministic: Whether to act deterministically.

    Returns:
        Callable ``action(obs) -> np.ndarray``.
    """
    from stable_baselines3.common.vec_env import VecNormalize

    norm_env = model.env if isinstance(getattr(model, "env", None), VecNormalize) else None

    def _normalize(obs: np.ndarray) -> np.ndarray:
        if norm_env is not None:
            return norm_env.normalize_obs(obs)
        return obs

    def _policy(obs: np.ndarray) -> np.ndarray:
        action, _ = model.predict(_normalize(obs), deterministic=deterministic)
        return np.asarray(action, dtype=np.float32)

    return _policy


def cross_check_with_sb3(
    model,
    env_factory: Callable,
    n_eval_episodes: int = 3,
    rel_tolerance: float = 0.05,
) -> Dict[str, object]:
    """Cross-check our evaluation metrics against ``stable_baselines3``.

    Runs the package evaluation on one seed and SB3's ``evaluate_policy`` on
    the same environment; reports mean rewards and whether they agree within
    a relative tolerance (a sanity check that the metrics pipeline itself is
    consistent). Models trained with observation normalization are skipped
    (the cross-check uses the raw environment).

    Args:
        model:            Trained SB3 model.
        env_factory:      Environment factory.
        n_eval_episodes:  Number of SB3 evaluation episodes.
        rel_tolerance:    Allowed relative difference between the two means.

    Returns:
        Dict with ``ours``, ``sb3``, ``ok`` and (for normalized models)
        ``skipped``.
    """
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.vec_env import VecNormalize

    if isinstance(getattr(model, "env", None), VecNormalize):
        return {"ours": None, "sb3": None, "ok": None, "skipped": True}

    env = env_factory()
    ours = evaluate_episode(env_factory(), action_from_model(model), seed=7)["reward"]
    sb3_mean, sb3_std = evaluate_policy(
        model, env, n_eval_episodes=n_eval_episodes, deterministic=True, warn=False
    )
    scale = max(abs(sb3_mean), 1e-6)
    ok = abs(ours - sb3_mean) <= rel_tolerance * scale
    return {"ours": float(ours), "sb3": float(sb3_mean), "sb3_std": float(sb3_std), "ok": bool(ok)}


def evaluate_over_seeds(
    env_factory: Callable,
    action_fn: Callable[[np.ndarray], np.ndarray],
    seeds: List[int],
) -> Dict[str, float]:
    """Evaluate a policy over multiple seeds (i.e. random disturbances).

    Args:
        env_factory: Environment factory.
        action_fn:   Policy mapping an observation to an action.
        seeds:       List of seeds to evaluate.

    Returns:
        Aggregated metric dictionary with ``mean`` and ``std`` suffixes.
    """
    collected: Dict[str, List[float]] = {}
    for seed in seeds:
        env = env_factory()
        for key, value in evaluate_episode(env, action_fn, seed=seed).items():
            collected.setdefault(key, []).append(float(value))

    result: Dict[str, float] = {}
    for key, values in collected.items():
        arr = np.asarray(values)
        result[f"{key}_mean"] = float(np.mean(arr))
        result[f"{key}_std"] = float(np.std(arr))
    return result
