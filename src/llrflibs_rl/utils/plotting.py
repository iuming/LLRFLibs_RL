###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Reusable comparison plotting.

These helpers render the "baseline vs RL" figures used by the scenario runner,
the CLI ``gallery`` subcommand and notebooks: pulse waveforms, physics-aware
error-metric panels, the training-convergence curve and a summary text panel.
:func:`plot_comparison` dispatches on the environment type.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "plot_adaptive_pulse",
    "plot_comparison",
    "plot_resonance",
    "plot_tracking",
]


def _plot_curve_data(curve: list[dict[str, float]] | None):
    if curve:
        return [float(p["timesteps"]) for p in curve], [float(p["reward"]) for p in curve]
    return None


def _tracking_errors(env, wave: np.ndarray) -> dict[str, float]:
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


def _finish_and_save(fig, axes, path: Path) -> None:
    for a in axes.flat:
        a.grid(True, alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_tracking(
    env,
    baseline: dict[str, np.ndarray],
    rl: dict[str, np.ndarray],
    baseline_name: str,
    curve: list[dict[str, float]] | None,
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

    axes[0, 2].set_title("RL evaluation reward during training")
    curve_data = _plot_curve_data(curve)
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
    _finish_and_save(fig, axes, path)


def plot_resonance(
    env,
    baseline: dict[str, np.ndarray],
    rl: dict[str, np.ndarray],
    baseline_name: str,
    curve: list[dict[str, float]] | None,
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

    y = np.arange(1)
    width = 0.35
    axes[1, 1].bar(y - width / 2, [b_dw], width, label=baseline_name, color="tab:blue")
    axes[1, 1].bar(y + width / 2, [r_dw], width, label="RL", color="tab:orange")
    axes[1, 1].set_xticks([])
    axes[1, 1].set_ylabel("mean |detuning| (Hz)")
    axes[1, 1].set_title("Residual detuning")
    axes[1, 1].legend(frameon=False)

    axes[0, 2].set_title("RL evaluation reward during training")
    curve_data = _plot_curve_data(curve)
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
    _finish_and_save(fig, axes, path)


def plot_adaptive_pulse(
    env,
    baseline: dict[str, np.ndarray],
    rl: dict[str, np.ndarray],
    baseline_name: str,
    curve: list[dict[str, float]] | None,
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
    curve_data = _plot_curve_data(curve)
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
    _finish_and_save(fig, axes, path)


def plot_comparison(
    env,
    baseline: dict[str, np.ndarray],
    rl: dict[str, np.ndarray],
    baseline_name: str,
    curve: list[dict[str, float]] | None,
    path: Path,
) -> None:
    """Render the right comparison figure for an environment.

    Args:
        env:           A freshly-reset probe environment.
        baseline:      Rollout dict of the baseline policy.
        rl:            Rollout dict of the RL policy.
        baseline_name: Baseline label (``zero`` / ``ff`` / ``pi`` / ``adrc``).
        curve:         Training evaluation curve for the convergence panel.
        path:          Output image path.
    """
    if type(env).__name__ == "AdaptivePulseEnv":
        plot_adaptive_pulse(env, baseline, rl, baseline_name, curve, path)
    elif getattr(env, "reference_sp", None) is None:
        plot_resonance(env, baseline, rl, baseline_name, curve, path)
    else:
        plot_tracking(env, baseline, rl, baseline_name, curve, path)
