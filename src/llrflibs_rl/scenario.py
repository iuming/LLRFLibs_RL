###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Scenario runner.

A "scenario" is one environment + one training/evaluation flow: roll out the
no-compensation baseline, train an RL agent, evaluate it, plot the pulse
waveforms and write a self-contained artifact directory.  Both the CLI
(``python -m llrflibs_rl.scripts.run``) and the example scripts are thin
front-ends on :func:`run_experiment`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .agents import agent_config_for, train
from .configs.agent import AgentConfig
from .evaluation import action_from_model, evaluate_episode, evaluate_over_seeds
from .registry import make
from .utils import print_header, print_metrics
from .utils.artifacts import RunArtifacts

__all__ = ["run_experiment", "rollout_episode", "zero_action"]


# ---------------------------------------------------------------------------
# Rollout helpers
# ---------------------------------------------------------------------------

def zero_action(env) -> Callable[[np.ndarray], np.ndarray]:
    """Return the no-compensation (all-zero) action policy.

    Args:
        env: Environment whose action space defines the output shape.

    Returns:
        A policy that always returns the zero action.
    """

    def _zero(obs: np.ndarray) -> np.ndarray:
        return np.zeros(env.action_space.shape, dtype=np.float32)

    return _zero


def rollout_episode(env, action_fn: Callable[[np.ndarray], np.ndarray], seed: int = 0) -> Dict[str, np.ndarray]:
    """Roll out one episode and record the physical waveforms.

    Args:
        env:       Environment.
        action_fn: Policy mapping observation -> action.
        seed:      Episode seed (disturbance realization).

    Returns:
        Dict with ``reward`` (scalar), ``tracked``, ``drive``, and the optional
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
    per_cavity_history: List[List[complex]] = []

    rewards: List[float] = []
    steps = 0
    for i in range(n):
        action = np.asarray(action_fn(obs), dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        tracked[i] = env.tracked_signal
        drive[i] = env.drive_total
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

    n = steps
    result: Dict[str, np.ndarray] = {
        "reward": np.asarray([float(np.sum(rewards))]),
        "tracked": tracked[:n],
        "drive": drive[:n],
    }
    if not np.all(detuning[:n] == 0.0):
        result["detuning"] = detuning[:n]
        result["actuator"] = actuator[:n]
        result["disturbance"] = disturbance[:n]
    if per_cavity_history:
        # shape: (n_steps, n_cav) -> transpose to (n_cav, n_steps)
        result["per_cavity"] = np.asarray(per_cavity_history, dtype=np.complex128).T
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_tracking(env, baseline: Dict[str, np.ndarray], rl: Dict[str, np.ndarray], path: Path) -> None:
    """Amplitude / phase / drive / before-after for voltage-like scenarios."""
    t = np.arange(len(baseline["tracked"])) * env.config.Ts * 1e6
    sp = np.asarray(env.reference_sp[: len(t)], dtype=np.complex128)
    n_cav = baseline.get("per_cavity", np.zeros((0, len(t)))).shape[0]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    fig.suptitle(f"{env.spec_id if hasattr(env, 'spec_id') else type(env).__name__}  —  baseline vs RL",
                 fontsize=13, fontweight="bold")

    axes[0, 0].plot(t, np.abs(sp), color="black", label="Set-point")
    axes[0, 0].plot(t, np.abs(baseline["tracked"]), "--", label="Baseline")
    axes[0, 0].plot(t, np.abs(rl["tracked"]), label="RL")
    for k in range(min(n_cav, 3)):
        axes[0, 0].plot(t, np.abs(baseline["per_cavity"][k]), ":", alpha=0.5, label=f"cavity {k}")
    axes[0, 0].set_title("Amplitude")
    axes[0, 0].set_ylabel("V")
    axes[0, 0].legend(frameon=False, fontsize=8)

    axes[0, 1].plot(t, np.angle(sp, deg=True), color="black", label="Set-point")
    axes[0, 1].plot(t, np.angle(baseline["tracked"], deg=True), "--", label="Baseline")
    axes[0, 1].plot(t, np.angle(rl["tracked"], deg=True), label="RL")
    axes[0, 1].set_title("Phase")
    axes[0, 1].set_ylabel("deg")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(t, np.abs(baseline["drive"]) * 1e-6, "--", label="Baseline drive")
    axes[1, 0].plot(t, np.abs(rl["drive"]) * 1e-6, label="RL drive")
    axes[1, 0].set_title("Forward drive")
    axes[1, 0].set_ylabel("MV")
    axes[1, 0].legend(frameon=False)

    labels = ["Baseline", "RL"]
    rewards = [float(baseline["reward"][0]), float(rl["reward"][0])]
    errs = [np.sqrt(np.mean(np.abs(sp - baseline["tracked"]) ** 2)),
            np.sqrt(np.mean(np.abs(sp - rl["tracked"]) ** 2))]
    x = np.arange(2)
    ax = axes[1, 1]
    ax.bar(x - 0.18, rewards, 0.36, label="Reward")
    ax2 = ax.twinx()
    ax2.bar(x + 0.18, errs, 0.36, color="tab:orange", alpha=0.8, label="RMS error")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Before vs After")
    ax.set_ylabel("Reward")
    ax2.set_ylabel("RMS error (V)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False)

    for a in axes.flat:
        a.grid(True, alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_resonance(env, baseline: Dict[str, np.ndarray], rl: Dict[str, np.ndarray], path: Path) -> None:
    """Detuning / actuator / amplitude plots for resonance scenarios."""
    t = np.arange(len(baseline["tracked"])) * env.config.Ts * 1e6
    target = env.cavity.vc_target

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    fig.suptitle("Resonance control — baseline vs RL", fontsize=13, fontweight="bold")

    axes[0, 0].plot(t, np.abs(baseline["tracked"]) * 1e-6, "--", label="Baseline")
    axes[0, 0].plot(t, np.abs(rl["tracked"]) * 1e-6, label="RL")
    axes[0, 0].axhline(target * 1e-6, color="black", lw=1, label="Target")
    axes[0, 0].set_title("Cavity amplitude")
    axes[0, 0].set_ylabel("MV")
    axes[0, 0].legend(frameon=False)

    hz = 2.0 * np.pi
    axes[0, 1].plot(t, baseline["disturbance"] / hz, color="black", alpha=0.6, label="Disturbance")
    axes[0, 1].plot(t, baseline["detuning"] / hz, "--", label="Baseline detuning")
    axes[0, 1].plot(t, rl["detuning"] / hz, label="RL detuning")
    axes[0, 1].plot(t, rl["actuator"] / hz, ":", label="RL actuator")
    axes[0, 1].set_title("Detuning")
    axes[0, 1].set_ylabel("Hz")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(t, baseline["actuator"] / hz, "--", label="Baseline")
    axes[1, 0].plot(t, rl["actuator"] / hz, label="RL actuator")
    axes[1, 0].set_title("Actuator command")
    axes[1, 0].set_ylabel("Hz")
    axes[1, 0].legend(frameon=False)

    labels = ["Baseline", "RL"]
    rewards = [float(baseline["reward"][0]), float(rl["reward"][0])]
    dms = [np.mean(np.abs(baseline["detuning"]) / hz), np.mean(np.abs(rl["detuning"]) / hz)]
    x = np.arange(2)
    ax = axes[1, 1]
    ax.bar(x - 0.18, rewards, 0.36, label="Reward")
    ax2 = ax.twinx()
    ax2.bar(x + 0.18, dms, 0.36, color="tab:orange", alpha=0.8, label="Mean |detuning| (Hz)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Before vs After")
    ax.set_ylabel("Reward")
    ax2.set_ylabel("Hz")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False)

    for a in axes.flat:
        a.grid(True, alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------------

def run_experiment(
    env_id: str,
    timesteps: Optional[int] = None,
    seed: int = 0,
    algorithm: str = "ppo",
    quick: bool = False,
    runs_dir: Optional[Path] = None,
    evaluate_seeds: Optional[List[int]] = None,
    verbose: bool = True,
    save_plots: bool = True,
) -> Dict[str, object]:
    """Train an agent on an environment and persist the full experiment.

    Args:
        env_id:         Registry environment id.
        timesteps:      Training budget (recommended when ``None``).
        seed:           Random seed.
        algorithm:      ``"ppo"`` or ``"sac"``.
        quick:          Use the small sanity budget.
        runs_dir:       Base directory for artifacts.
        evaluate_seeds: Seeds used for the multi-seed robustness evaluation.
        verbose:        Print progress/metrics.
        save_plots:     Write the waveform figures.

    Returns:
        A summary dictionary with baseline / RL metrics and the artifact path.
    """
    from .registry import get_spec

    spec = get_spec(env_id)
    env = make(env_id)
    cfg: AgentConfig = agent_config_for(env_id, timesteps=timesteps, algorithm=algorithm,
                                        seed=seed, quick=quick)
    seeds = evaluate_seeds if evaluate_seeds else [seed]

    print_header(f"Experiment: {env_id}")

    env_factory = lambda: make(env_id)  # noqa: E731

    # Baseline (no compensation) rollout on the first evaluation seed.
    baseline_metrics = evaluate_episode(make(env_id), zero_action(env), seed=seeds[0])

    # Training.
    model = train(env_factory, cfg)
    rl_policy = action_from_model(model)
    rl_metrics = evaluate_episode(make(env_id), rl_policy, seed=seeds[0])
    robust = evaluate_over_seeds(env_factory, rl_policy, seeds)

    # Persist artifacts.
    artifacts = RunArtifacts(run_name=f"{spec.env_id}-{cfg.algorithm}", runs_dir=runs_dir)
    artifacts.save_config({
        "env_id": env_id,
        "algorithm": cfg.algorithm,
        "timesteps": cfg.timesteps,
        "seed": cfg.seed,
        "env_description": spec.description,
    })
    metrics = {
        "baseline_reward": baseline_metrics["reward"],
        "rl_reward": rl_metrics["reward"],
        "rl_robust": robust,
    }
    artifacts.save_metrics(metrics)
    # Persist the trained model so the run directory is self-contained.
    model.save(str(artifacts.dir / "model.zip"))

    if save_plots:
        baseline_wave = rollout_episode(make(env_id), zero_action(env), seed=seeds[0])
        rl_wave = rollout_episode(make(env_id), rl_policy, seed=seeds[0])
        # Choose the plot style from a freshly-reset probe environment (the
        # tracking signal exists only for set-point-tracking scenarios).
        probe_env = make(env_id)
        probe_env.reset(seed=seeds[0])
        if getattr(probe_env, "reference_sp", None) is None:
            _plot_resonance(probe_env, baseline_wave, rl_wave, artifacts.dir / "comparison.png")
        else:
            _plot_tracking(probe_env, baseline_wave, rl_wave, artifacts.dir / "comparison.png")

    summary: Dict[str, object] = {
        "env_id": env_id,
        "baseline_reward": baseline_metrics.get("reward"),
        "rl_reward": rl_metrics.get("reward"),
        "rl_reward_mean": robust.get("reward_mean"),
        "artifact_dir": str(artifacts.dir),
    }
    if verbose:
        print_metrics("Baseline", baseline_metrics)
        print_metrics("RL (deterministic)", rl_metrics)
        print(f"\nArtifacts written to: {artifacts.dir}")
    return summary
