###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Scenario runner.

A "scenario" is one environment + one comparison flow: roll out a classical
baseline (``zero`` / ``pi`` / ``adrc`` / ``ff``), train an RL agent, track its
evaluation reward curve, evaluate it, plot the pulse waveforms together with
physics-aware error metrics and write a self-contained artifact directory.
Both the CLI and the example scripts are thin front-ends on
:func:`run_experiment`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .agents import train
from .configs.agent import AgentConfig
from .controllers.baselines import list_baselines, make_baseline_policy
from .evaluation import action_from_model, evaluate_episode, evaluate_over_seeds
from .registry import make
from .utils import print_header, print_metrics
from .utils.artifacts import RunArtifacts
from .utils.plotting import plot_comparison

__all__ = ["run_experiment", "rollout_episode", "zero_action"]


def zero_action(env) -> Callable[[np.ndarray], np.ndarray]:
    """Return the no-compensation (all-zero) action policy."""
    from .controllers.baselines import zero_policy

    return zero_policy(env)


# ---------------------------------------------------------------------------
# Rollout helpers
# ---------------------------------------------------------------------------

def rollout_episode(env, action_fn: Callable[[np.ndarray], np.ndarray], seed: int = 0) -> dict[str, np.ndarray]:
    """Roll out one episode and record the physical waveforms.

    Args:
        env:       Environment.
        action_fn: Policy mapping observation -> action.
        seed:      Episode seed (disturbance realization).

    Returns:
        Dict with ``reward``, ``tracked``, ``drive``, and the optional
        per-environment waves ``detuning``, ``actuator``, ``disturbance`` and
        ``per_cavity`` (shape ``(n_cav, steps)``).
    """
    obs, _ = env.reset(seed=seed)
    n = env.episode_len
    tracked = np.zeros(n, dtype=np.complex128)
    drive = np.zeros(n, dtype=np.complex128)
    detuning = np.zeros(n, dtype=float)
    actuator = np.zeros(n, dtype=float)
    disturbance = np.zeros(n, dtype=float)
    per_cavity_history: list[list[complex]] = []

    rewards: list[float] = []
    steps = 0
    for i in range(n):
        action = np.asarray(action_fn(obs), dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        tracked[i] = env.tracked_signal
        drive[i] = getattr(env, "drive_total", 0.0 + 0.0j)
        if hasattr(env, "plant") and hasattr(env.plant, "detuning"):
            detuning[i] = env.plant.detuning
        if hasattr(env, "actuator"):
            actuator[i] = env.actuator.state
        if hasattr(env, "disturbance_wave"):
            disturbance[i] = env.disturbance_wave[i]
        if hasattr(env.plant, "vc_list"):
            per_cavity_history.append(list(env.plant.vc_list))
        steps += 1
        if terminated or truncated:
            break

    result: dict[str, np.ndarray] = {
        "reward": np.asarray([float(np.sum(rewards))]),
        "tracked": tracked[:steps],
        "drive": drive[:steps],
    }
    if not np.all(detuning[:steps] == 0.0):
        result["detuning"] = detuning[:steps]
        result["actuator"] = actuator[:steps]
        result["disturbance"] = disturbance[:steps]
    if per_cavity_history:
        result["per_cavity"] = np.asarray(per_cavity_history, dtype=np.complex128).T
    if hasattr(env, "pulse_reward_history") and env.pulse_reward_history:
        result["pulse_rewards"] = np.asarray(env.pulse_reward_history, dtype=np.float64)
        result["pulse_errors"] = np.asarray(
            [abs(e) for e in env.pulse_error_history], dtype=np.float64
        )
    return result


def run_experiment(
    env_id: str,
    timesteps: int | None = None,
    seed: int = 0,
    algorithm: str | None = None,
    quick: bool = False,
    baseline: str = "zero",
    runs_dir: Path | None = None,
    evaluate_seeds: list[int] | None = None,
    verbose: bool = True,
    save_plots: bool = True,
    render: bool = False,
    render_freq: int = 2048,
    render_steps: int = 300,
    render_http: int = -1,
) -> dict[str, object]:
    """Train an agent on an environment and persist the full experiment.

    Args:
        env_id:         Registry environment id.
        timesteps:      Training budget (recommended when ``None``).
        seed:           Random seed.
        algorithm:      ``"ppo"`` / ``"sac"`` / ``"td3"``; ``None`` uses the
                        per-environment preset (e.g. SAC for
                        ``VectorSumTuner-v0``).
        quick:          Use the small sanity budget.
        baseline:       Classical baseline name: ``zero`` / ``ff`` / ``pi`` / ``adrc``.
        runs_dir:       Base directory for artifacts.
        evaluate_seeds: Seeds used for the multi-seed robustness evaluation.
        verbose:        Print progress/metrics.
        save_plots:     Write the comparison figure.
        render:         Periodically live-render the policy while training.
        render_freq:    Live-render every this many training steps.
        render_steps:   Samples per live-rendered episode.
        render_http:    Port for the browser live view (0 = auto, -1 = off).

    Returns:
        A summary dictionary with baseline / RL metrics and the artifact path.
    """
    if baseline not in list_baselines():
        raise ValueError(f"unknown baseline {baseline!r}; available: {list_baselines()}")

    from .agents.presets import agent_config_for
    from .registry import get_spec

    spec = get_spec(env_id)
    cfg: AgentConfig = agent_config_for(
        env_id, timesteps=timesteps, algorithm=algorithm, seed=seed, quick=quick
    )
    seeds = evaluate_seeds if evaluate_seeds else [seed]

    print_header(f"Experiment: {env_id}  (baseline={baseline}, algorithm={cfg.algorithm})")

    env_factory = lambda: make(env_id)  # noqa: E731
    baseline_policy = make_baseline_policy(make(env_id), baseline)

    # Persist artifacts (created before training so the SB3 best model and
    # periodic checkpoints land in the run directory).
    artifacts = RunArtifacts(run_name=f"{spec.env_id}-{cfg.algorithm}", runs_dir=runs_dir)

    # Baseline: deterministic rollout on the first seed (for figures) plus a
    # multi-seed mean (for honest comparison under random disturbances).
    baseline_metrics = evaluate_episode(make(env_id), baseline_policy, seed=seeds[0])
    baseline_robust = evaluate_over_seeds(env_factory, baseline_policy, seeds)

    # Training with an evaluation curve, best-model saving and checkpoints.
    curve: list[dict[str, float]] = []
    model = train(
        env_factory,
        cfg,
        eval_env_factory=env_factory,
        eval_freq=max(1024, cfg.timesteps // 10),
        eval_seed=seeds[0],
        curve=curve,
        best_model_dir=artifacts.dir,
        render=render,
        render_freq=render_freq,
        render_steps=render_steps,
        render_http=render_http,
    )
    rl_policy = action_from_model(model)
    rl_metrics = evaluate_episode(make(env_id), rl_policy, seed=seeds[0])
    robust = evaluate_over_seeds(env_factory, rl_policy, seeds)

    artifacts.save_config({
        "env_id": env_id,
        "baseline": baseline,
        "algorithm": cfg.algorithm,
        "timesteps": cfg.timesteps,
        "seed": cfg.seed,
        "env_description": spec.description,
    })
    metrics: dict[str, object] = {
        "baseline": baseline,
        "evaluate_seeds": seeds,
        "baseline_reward_mean": baseline_robust["reward_mean"],
        "rl_reward_mean": robust["reward_mean"],
        "rl_reward_std": robust["reward_std"],
        "baseline_reward_seed0": baseline_metrics["reward"],
        "rl_reward_seed0": rl_metrics["reward"],
    }
    for key in ("error_rms", "flattop_error_rms", "amp_error_rms"):
        if f"{key}_mean" in robust:
            metrics[f"baseline_{key}_mean"] = baseline_robust.get(f"{key}_mean")
            metrics[f"rl_{key}_mean"] = robust.get(f"{key}_mean")
    artifacts.save_metrics(metrics)
    artifacts.save_text(json.dumps(curve), "training_curve.json")
    model.save(str(artifacts.dir / "model.zip"))

    if save_plots:
        baseline_wave = rollout_episode(make(env_id), baseline_policy, seed=seeds[0])
        rl_wave = rollout_episode(make(env_id), rl_policy, seed=seeds[0])
        probe = make(env_id)
        probe.reset(seed=seeds[0])
        plot_path = artifacts.dir / f"comparison_{baseline}.png"
        plot_comparison(probe, baseline_wave, rl_wave, baseline, curve, plot_path)

    summary: dict[str, object] = {
        "env_id": env_id,
        "baseline": baseline,
        "evaluate_seeds": seeds,
        "baseline_reward": baseline_robust.get("reward_mean"),
        "rl_reward": robust.get("reward_mean"),
        "rl_reward_std": robust.get("reward_std"),
        "artifact_dir": str(artifacts.dir),
    }
    for key in ("error_rms", "flattop_error_rms"):
        if f"{key}_mean" in robust:
            summary[f"baseline_{key}"] = baseline_robust.get(f"{key}_mean")
            summary[f"rl_{key}"] = robust.get(f"{key}_mean")
    if verbose:
        print_metrics(f"Baseline ({baseline}) — mean over seeds {seeds}", baseline_robust)
        print_metrics(f"RL — mean over seeds {seeds}", {k: v for k, v in robust.items() if k.endswith("_mean")})
        print(f"\nArtifacts written to: {artifacts.dir}")
    return summary
