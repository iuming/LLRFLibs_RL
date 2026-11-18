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
from pathlib import Path
from typing import Callable, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .agents import train
from .configs.agent import AgentConfig
from .controllers.baselines import list_baselines, make_baseline_policy
from .evaluation import action_from_model, evaluate_episode, evaluate_over_seeds
from .registry import make
from .utils import print_header, print_metrics
from .utils.artifacts import RunArtifacts

__all__ = ["run_experiment", "rollout_episode", "zero_action"]


def zero_action(env) -> Callable[[np.ndarray], np.ndarray]:
    """Return the no-compensation (all-zero) action policy."""
    from .controllers.baselines import zero_policy

    return zero_policy(env)


# ---------------------------------------------------------------------------
# Rollout helpers
# ---------------------------------------------------------------------------

def rollout_episode(env, action_fn: Callable[[np.ndarray], np.ndarray], seed: int = 0) -> Dict[str, np.ndarray]:
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
    per_cavity_history: List[List[complex]] = []

    rewards: List[float] = []
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

    result: Dict[str, np.ndarray] = {
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


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _tracking_errors(env, wave: np.ndarray) -> Dict[str, float]:
    """Return (whole-pulse, flattop) RMS errors in % of the voltage target."""
    sp = np.asarray(env.reference_sp[: len(wave)], dtype=np.complex128)
    error = sp - wave
    scale = env.obs_scale
    err_rms = float(np.sqrt(np.mean(np.abs(error) ** 2)))
    cav = getattr(env, "cavity", None)
    if cav is None:
        cav = env.cavities[0]
    fl = int(cav.filling_len)
    ft = int(cav.flattop_len)
    flattop_rms = 0.0
    n = len(wave)
    if fl + ft <= n:
        flattop_rms = float(np.sqrt(np.mean(np.abs(error[fl: fl + ft]) ** 2)))
    return {
        "error_rms": err_rms,
        "error_rms_pct": err_rms / scale * 100.0,
        "flattop_error_rms_pct": flattop_rms / scale * 100.0,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_tracking_curve(curve: Optional[List[Dict[str, float]]]) -> Optional[List[float]]:
    if curve:
        return [float(p["timesteps"]) for p in curve], [float(p["reward"]) for p in curve]
    return None


def _plot_tracking(
    env,
    baseline: Dict[str, np.ndarray],
    rl: Dict[str, np.ndarray],
    baseline_name: str,
    curve: Optional[List[Dict[str, float]]],
    path: Path,
) -> None:
    """Amplitude / phase / drive / error-metrics / convergence for voltage-like scenarios."""
    t = np.arange(len(baseline["tracked"])) * env.config.Ts * 1e6
    sp = np.asarray(env.reference_sp[: len(t)], dtype=np.complex128)
    b_err = _tracking_errors(env, baseline["tracked"])
    r_err = _tracking_errors(env, rl["tracked"])
    n_cav = baseline.get("per_cavity", np.zeros((0, len(t)))).shape[0]

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.5), constrained_layout=True)
    fig.suptitle(f"{type(env).__name__}  —  {baseline_name} (baseline) vs RL", fontsize=13,
                 fontweight="bold")

    axes[0, 0].plot(t, np.abs(sp), color="black", label="Set-point")
    axes[0, 0].plot(t, np.abs(baseline["tracked"]), "--", label=baseline_name)
    axes[0, 0].plot(t, np.abs(rl["tracked"]), label="RL")
    for k in range(min(n_cav, 3)):
        axes[0, 0].plot(t, np.abs(baseline["per_cavity"][k]), ":", alpha=0.5, label=f"cavity {k}")
    axes[0, 0].set_title("Amplitude")
    axes[0, 0].set_ylabel("V")
    axes[0, 0].legend(frameon=False, fontsize=8)

    axes[0, 1].plot(t, np.angle(sp, deg=True), color="black", label="Set-point")
    axes[0, 1].plot(t, np.angle(baseline["tracked"], deg=True), "--", label=baseline_name)
    axes[0, 1].plot(t, np.angle(rl["tracked"], deg=True), label="RL")
    axes[0, 1].set_title("Phase")
    axes[0, 1].set_ylabel("deg")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(t, np.abs(baseline["drive"]) * 1e-6, "--", label=f"{baseline_name} drive")
    axes[1, 0].plot(t, np.abs(rl["drive"]) * 1e-6, label="RL drive")
    axes[1, 0].set_title("Forward drive")
    axes[1, 0].set_ylabel("MV")
    axes[1, 0].legend(frameon=False)

    # Error-metric panel: whole-pulse and flattop RMS error (% of target).
    y = np.arange(2)
    width = 0.35
    axes[1, 1].bar(y - width / 2, [b_err["error_rms_pct"], b_err["flattop_error_rms_pct"]],
                   width, label=baseline_name, color="tab:blue")
    axes[1, 1].bar(y + width / 2, [r_err["error_rms_pct"], r_err["flattop_error_rms_pct"]],
                   width, label="RL", color="tab:orange")
    axes[1, 1].set_xticks(y)
    axes[1, 1].set_xticklabels(["whole pulse", "flattop"])
    axes[1, 1].set_ylabel("RMS error (% of target)")
    axes[1, 1].set_title("Tracking error")
    axes[1, 1].legend(frameon=False)

    # Training convergence curve.
    axes[0, 2].set_title("RL evaluation reward during training")
    curve_data = _plot_tracking_curve(curve)
    if curve_data is not None:
        xs, ys = curve_data
        axes[0, 2].plot(xs, ys, marker=".", markersize=4)
        axes[0, 2].set_xlabel("timesteps")
        axes[0, 2].set_ylabel("reward")
    else:
        axes[0, 2].axis("off")
        axes[0, 2].text(0.5, 0.5, "no training curve", ha="center", va="center")

    # Summary text panel.
    axes[1, 2].axis("off")
    text = (
        f"Baseline ({baseline_name}):\n"
        f"  reward          = {baseline['reward'][0]:,.1f}\n"
        f"  whole-pulse err = {b_err['error_rms_pct']:.3f} %\n"
        f"  flattop err     = {b_err['flattop_error_rms_pct']:.3f} %\n\n"
        f"RL:\n"
        f"  reward          = {rl['reward'][0]:,.1f}\n"
        f"  whole-pulse err = {r_err['error_rms_pct']:.3f} %\n"
        f"  flattop err     = {r_err['flattop_error_rms_pct']:.3f} %"
    )
    axes[1, 2].text(0.05, 0.95, text, transform=axes[1, 2].transAxes, va="top", fontsize=10,
                    family="monospace")

    for a in axes.flat:
        a.grid(True, alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_resonance(
    env,
    baseline: Dict[str, np.ndarray],
    rl: Dict[str, np.ndarray],
    baseline_name: str,
    curve: Optional[List[Dict[str, float]]],
    path: Path,
) -> None:
    """Detuning / actuator / amplitude / convergence for resonance scenarios."""
    t = np.arange(len(baseline["tracked"])) * env.config.Ts * 1e6
    hz = 2.0 * np.pi
    b_dw = float(np.mean(np.abs(baseline["detuning"]) / hz))
    r_dw = float(np.mean(np.abs(rl["detuning"]) / hz))

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.5), constrained_layout=True)
    fig.suptitle(f"{type(env).__name__}  —  {baseline_name} (baseline) vs RL", fontsize=13,
                 fontweight="bold")

    axes[0, 0].plot(t, np.abs(baseline["tracked"]) * 1e-6, "--", label=baseline_name)
    axes[0, 0].plot(t, np.abs(rl["tracked"]) * 1e-6, label="RL")
    axes[0, 0].axhline(env.cavity.vc_target * 1e-6, color="black", lw=1, label="Target")
    axes[0, 0].set_title("Cavity amplitude")
    axes[0, 0].set_ylabel("MV")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(t, baseline["disturbance"] / hz, color="black", alpha=0.6, label="Disturbance")
    axes[0, 1].plot(t, baseline["detuning"] / hz, "--", label=baseline_name)
    axes[0, 1].plot(t, rl["detuning"] / hz, label="RL")
    axes[0, 1].plot(t, rl["actuator"] / hz, ":", label="RL actuator")
    axes[0, 1].set_title("Detuning")
    axes[0, 1].set_ylabel("Hz")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(t, baseline["actuator"] / hz, "--", label=baseline_name)
    axes[1, 0].plot(t, rl["actuator"] / hz, label="RL actuator")
    axes[1, 0].set_title("Actuator command")
    axes[1, 0].set_ylabel("Hz")
    axes[1, 0].legend(frameon=False)

    # Error-metric panel: mean |detuning| in Hz.
    y = np.arange(1)
    width = 0.35
    axes[1, 1].bar(y - width / 2, [b_dw], width, label=baseline_name, color="tab:blue")
    axes[1, 1].bar(y + width / 2, [r_dw], width, label="RL", color="tab:orange")
    axes[1, 1].set_xticks([])
    axes[1, 1].set_ylabel("mean |detuning| (Hz)")
    axes[1, 1].set_title("Residual detuning")
    axes[1, 1].legend(frameon=False)

    # Training convergence curve.
    axes[0, 2].set_title("RL evaluation reward during training")
    curve_data = _plot_tracking_curve(curve)
    if curve_data is not None:
        xs, ys = curve_data
        axes[0, 2].plot(xs, ys, marker=".", markersize=4)
        axes[0, 2].set_xlabel("timesteps")
        axes[0, 2].set_ylabel("reward")
    else:
        axes[0, 2].axis("off")
        axes[0, 2].text(0.5, 0.5, "no training curve", ha="center", va="center")

    axes[1, 2].axis("off")
    text = (
        f"Baseline ({baseline_name}):\n"
        f"  reward         = {baseline['reward'][0]:,.1f}\n"
        f"  mean |detuning| = {b_dw:.1f} Hz\n\n"
        f"RL:\n"
        f"  reward         = {rl['reward'][0]:,.1f}\n"
        f"  mean |detuning| = {r_dw:.1f} Hz"
    )
    axes[1, 2].text(0.05, 0.95, text, transform=axes[1, 2].transAxes, va="top", fontsize=10,
                    family="monospace")

    for a in axes.flat:
        a.grid(True, alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------------

def _plot_adaptive_pulse(
    env,
    baseline: Dict[str, np.ndarray],
    rl: Dict[str, np.ndarray],
    baseline_name: str,
    curve: Optional[List[Dict[str, float]]],
    path: Path,
) -> None:
    """Per-pulse reward/error and convergence for the pulse-level scenario."""
    t = np.arange(len(baseline["pulse_rewards"]))
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    fig.suptitle(f"{type(env).__name__}  —  {baseline_name} (baseline) vs RL", fontsize=13,
                 fontweight="bold")

    axes[0, 0].plot(t, baseline["pulse_rewards"], "--", marker="o", ms=3, label=baseline_name)
    axes[0, 0].plot(t, rl["pulse_rewards"], marker="o", ms=3, label="RL")
    axes[0, 0].set_title("Reward per pulse")
    axes[0, 0].set_xlabel("pulse")
    axes[0, 0].set_ylabel("reward")
    axes[0, 0].legend(frameon=False)

    scale = env.obs_scale
    axes[0, 1].plot(t, baseline["pulse_errors"] / scale, "--", marker="o", ms=3, label=baseline_name)
    axes[0, 1].plot(t, rl["pulse_errors"] / scale, marker="o", ms=3, label="RL")
    axes[0, 1].set_title("Flat-top tracking error (fraction of target)")
    axes[0, 1].set_xlabel("pulse")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].set_title("RL evaluation reward during training")
    curve_data = _plot_tracking_curve(curve)
    if curve_data is not None:
        xs, ys = curve_data
        axes[1, 0].plot(xs, ys, marker=".", markersize=4)
        axes[1, 0].set_xlabel("timesteps (pulses)")
        axes[1, 0].set_ylabel("reward")
    else:
        axes[1, 0].axis("off")
        axes[1, 0].text(0.5, 0.5, "no training curve", ha="center", va="center")

    axes[1, 1].axis("off")
    text = (
        f"Baseline ({baseline_name}):\n"
        f"  summed reward = {baseline['reward'][0]:,.1f}\n"
        f"  mean |fferr|  = {float(np.mean(baseline['pulse_errors']) / scale):.4f}\n\n"
        f"RL:\n"
        f"  summed reward = {rl['reward'][0]:,.1f}\n"
        f"  mean |fferr|  = {float(np.mean(rl['pulse_errors']) / scale):.4f}"
    )
    axes[1, 1].text(0.05, 0.95, text, transform=axes[1, 1].transAxes, va="top", fontsize=10,
                    family="monospace")

    for a in axes.flat:
        a.grid(True, alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)

def run_experiment(
    env_id: str,
    timesteps: Optional[int] = None,
    seed: int = 0,
    algorithm: Optional[str] = None,
    quick: bool = False,
    baseline: str = "zero",
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
        algorithm:      ``"ppo"`` / ``"sac"`` / ``"td3"``; ``None`` uses the
                        per-environment preset (e.g. SAC for
                        ``VectorSumTuner-v0``).
        quick:          Use the small sanity budget.
        baseline:       Classical baseline name: ``zero`` / ``ff`` / ``pi`` / ``adrc``.
        runs_dir:       Base directory for artifacts.
        evaluate_seeds: Seeds used for the multi-seed robustness evaluation.
        verbose:        Print progress/metrics.
        save_plots:     Write the comparison figure.

    Returns:
        A summary dictionary with baseline / RL metrics and the artifact path.
    """
    if baseline not in list_baselines():
        raise ValueError(f"unknown baseline {baseline!r}; available: {list_baselines()}")

    from .registry import get_spec
    from .agents.presets import agent_config_for

    spec = get_spec(env_id)
    env = make(env_id)
    cfg: AgentConfig = agent_config_for(
        env_id, timesteps=timesteps, algorithm=algorithm, seed=seed, quick=quick
    )
    seeds = evaluate_seeds if evaluate_seeds else [seed]

    print_header(f"Experiment: {env_id}  (baseline={baseline}, algorithm={cfg.algorithm})")

    env_factory = lambda: make(env_id)  # noqa: E731
    baseline_policy = make_baseline_policy(make(env_id), baseline)

    # Baseline: deterministic rollout on the first seed (for figures) plus a
    # multi-seed mean (for honest comparison under random disturbances).
    baseline_metrics = evaluate_episode(make(env_id), baseline_policy, seed=seeds[0])
    baseline_robust = evaluate_over_seeds(env_factory, baseline_policy, seeds)

    # Training with an evaluation curve.
    curve: List[Dict[str, float]] = []
    model = train(
        env_factory,
        cfg,
        eval_env_factory=env_factory,
        eval_freq=max(1024, cfg.timesteps // 10),
        eval_seed=seeds[0],
        curve=curve,
    )
    rl_policy = action_from_model(model)
    rl_metrics = evaluate_episode(make(env_id), rl_policy, seed=seeds[0])
    robust = evaluate_over_seeds(env_factory, rl_policy, seeds)

    # Persist artifacts.
    artifacts = RunArtifacts(run_name=f"{spec.env_id}-{cfg.algorithm}", runs_dir=runs_dir)
    artifacts.save_config({
        "env_id": env_id,
        "baseline": baseline,
        "algorithm": cfg.algorithm,
        "timesteps": cfg.timesteps,
        "seed": cfg.seed,
        "env_description": spec.description,
    })
    metrics: Dict[str, object] = {
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
        if type(probe).__name__ == "AdaptivePulseEnv":
            _plot_adaptive_pulse(probe, baseline_wave, rl_wave, baseline, curve, plot_path)
        elif getattr(probe, "reference_sp", None) is None:
            _plot_resonance(probe, baseline_wave, rl_wave, baseline, curve, plot_path)
        else:
            _plot_tracking(probe, baseline_wave, rl_wave, baseline, curve, plot_path)

    summary: Dict[str, object] = {
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
