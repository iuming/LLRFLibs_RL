###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
'''
Compare RL control against the ILC adaptive feedforward example.
'''

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from llrflibs.rf_control import AFF_ilc, AFF_ilc_design
from llrflibs.rf_sim import cav_impulse
from llrflibs_rl.envs import CavityVoltageControlEnv, VoltageControlConfig


def build_env() -> CavityVoltageControlEnv:
    config = VoltageControlConfig(
        half_bw=2 * np.pi * 216 * 0.5,
        Ts=1.0e-6,
        episode_len=2048,
        vc_target=25e6,
        filling_len=510,
        flattop_len=800,
        beam_current=0.008,
        beam_phase_deg=0.0,
        detuning=2 * np.pi * 2.0e3,
        randomize_detuning=0.0,
        reward_error_weight=7.0,
        reward_drive_weight=0.18,
        reward_reflected_weight=0.08,
    )
    return CavityVoltageControlEnv(config)


def rollout_episode(env: CavityVoltageControlEnv, correction: np.ndarray | None = None):
    obs, info = env.reset()
    rewards = []
    vc_wave = np.zeros(env.episode_len, dtype=np.complex128)
    if correction is None:
        correction = np.zeros(env.episode_len, dtype=np.complex128)

    for step in range(env.episode_len):
        corr = correction[step] if step < correction.shape[0] else 0.0 + 0.0j
        action = np.array(
            [
                np.clip(np.real(corr) / max(env.drive_limit, 1.0), -1.0, 1.0),
                np.clip(np.imag(corr) / max(env.drive_limit, 1.0), -1.0, 1.0),
            ],
            dtype=np.float32,
        )
        obs, reward, terminated, truncated, info = env.step(action)
        vc_wave[step] = env._vc
        rewards.append(float(reward))
        if terminated or truncated:
            break

    return {
        "reward": float(np.sum(rewards)),
        "last_info": info,
        "vc_wave": vc_wave,
        "step_count": len(rewards),
        "error_rms": float(np.sqrt(np.mean(np.abs(env._sp[: len(vc_wave)]) - np.abs(vc_wave[: len(env._sp)])) ** 2)),
    }


def run_ilc(env: CavityVoltageControlEnv, iterations: int):
    pulse_len = min(env.episode_len, env._sp.shape[0])
    status, h = cav_impulse(env.config.half_bw, env.config.detuning, env.config.Ts, order=pulse_len)
    if not status:
        raise RuntimeError("Failed to generate cavity impulse response for ILC comparison.")

    status, L = AFF_ilc_design(h * 2.0, pulse_len, P=np.eye(pulse_len) * 50.0, Q=np.eye(pulse_len))
    if not status:
        raise RuntimeError("Failed to design the ILC gain matrix.")

    correction = np.zeros(env.episode_len, dtype=np.complex128)
    iteration_rewards = []
    iteration_error_rms = []
    last_result = None

    for iteration in range(iterations):
        last_result = rollout_episode(env, correction)
        error = env._sp[:pulse_len] - last_result["vc_wave"][:pulse_len]
        correction_update = np.asarray(AFF_ilc(error, L)).reshape(-1)
        correction[:pulse_len] += correction_update[:pulse_len]
        iteration_rewards.append(last_result["reward"])
        iteration_error_rms.append(float(np.sqrt(np.mean(np.abs(error) ** 2))))

    return {
        "rewards": iteration_rewards,
        "error_rms": iteration_error_rms,
        "final_result": last_result,
        "correction": correction,
    }


def train_and_evaluate_rl(env_factory, timesteps: int):
    vec_env = DummyVecEnv([env_factory])
    model = PPO("MlpPolicy", vec_env, verbose=1, n_steps=256, batch_size=64, gamma=0.995)
    model.learn(total_timesteps=timesteps)

    obs = vec_env.reset()
    rewards = []
    vc_wave = np.zeros(vec_env.envs[0].episode_len, dtype=np.complex128)
    info = {}
    for step in range(vec_env.envs[0].episode_len):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vc_wave[step] = vec_env.envs[0]._vc
        rewards.append(float(reward[0]))
        if done[0]:
            break

    return {
        "reward": float(np.sum(rewards)),
        "vc_wave": vc_wave,
        "last_info": info,
        "model": model,
        "error_rms": float(np.sqrt(np.mean(np.abs(vec_env.envs[0]._sp[: len(vc_wave)]) - np.abs(vc_wave[: len(vec_env.envs[0]._sp)])) ** 2)),
    }


def _wave_metrics(env: CavityVoltageControlEnv, result: dict):
    sp = np.asarray(env._sp, dtype=np.complex128)
    vc = np.asarray(result["vc_wave"], dtype=np.complex128)
    n = min(len(sp), len(vc))
    error = sp[:n] - vc[:n]
    return {
        "amplitude": np.abs(vc[:n]),
        "phase": np.angle(vc[:n], deg=True),
        "error_rms": float(np.sqrt(np.mean(np.abs(error) ** 2))),
        "reward": float(result["reward"]),
    }


def plot_comparison(env: CavityVoltageControlEnv, zero_result: dict, ilc_result: dict, rl_result: dict, ilc_correction: np.ndarray, output_path: Path | None):
    sp = np.asarray(env._sp, dtype=np.complex128)
    t = np.arange(sp.shape[0]) * env.config.Ts * 1e6

    zero_metrics = _wave_metrics(env, zero_result)
    ilc_metrics = _wave_metrics(env, ilc_result["final_result"])
    rl_metrics = _wave_metrics(env, rl_result)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle("RL vs. ILC for Pulsed Cavity Voltage Control", fontsize=15, fontweight="bold")

    amp_ax = axes[0, 0]
    amp_ax.plot(t, np.abs(sp), color="black", linewidth=2.0, label="Setpoint")
    amp_ax.plot(t, zero_metrics["amplitude"], color="#7f7f7f", linestyle="--", label="No correction")
    amp_ax.plot(t, ilc_metrics["amplitude"], color="#1f77b4", linewidth=1.8, label="ILC final")
    amp_ax.plot(t, rl_metrics["amplitude"], color="#d62728", linewidth=1.8, label="RL final")
    amp_ax.set_xlabel("Time (us)")
    amp_ax.set_ylabel("|Vc| (V)")
    amp_ax.grid(True, alpha=0.3)
    amp_ax.legend(frameon=False)

    phase_ax = axes[0, 1]
    phase_ax.plot(t, np.angle(sp, deg=True), color="black", linewidth=2.0, label="Setpoint")
    phase_ax.plot(t, np.angle(zero_result["vc_wave"], deg=True), color="#7f7f7f", linestyle="--", label="No correction")
    phase_ax.plot(t, np.angle(ilc_result["final_result"]["vc_wave"], deg=True), color="#1f77b4", linewidth=1.8, label="ILC final")
    phase_ax.plot(t, np.angle(rl_result["vc_wave"], deg=True), color="#d62728", linewidth=1.8, label="RL final")
    phase_ax.set_xlabel("Time (us)")
    phase_ax.set_ylabel("Phase (deg)")
    phase_ax.grid(True, alpha=0.3)
    phase_ax.legend(frameon=False)

    conv_ax = axes[1, 0]
    ilc_iters = np.arange(1, len(ilc_result["rewards"]) + 1)
    conv_ax.plot(ilc_iters, ilc_result["rewards"], color="#1f77b4", marker="o", label="ILC reward")
    conv_ax.set_xlabel("Iteration")
    conv_ax.set_ylabel("Reward")
    conv_ax.grid(True, alpha=0.3)
    conv_ax_twin = conv_ax.twinx()
    conv_ax_twin.plot(ilc_iters, ilc_result["error_rms"], color="#ff7f0e", marker="s", label="ILC RMS error")
    conv_ax_twin.set_ylabel("Error RMS (V)")
    lines_1, labels_1 = conv_ax.get_legend_handles_labels()
    lines_2, labels_2 = conv_ax_twin.get_legend_handles_labels()
    conv_ax.legend(lines_1 + lines_2, labels_1 + labels_2, frameon=False, loc="best")

    summary_ax = axes[1, 1]
    labels = ["No corr.", "ILC", "RL"]
    rewards = [zero_result["reward"], ilc_result["final_result"]["reward"], rl_result["reward"]]
    errors = [zero_metrics["error_rms"], ilc_metrics["error_rms"], rl_metrics["error_rms"]]
    x = np.arange(len(labels))
    width = 0.34
    reward_bars = summary_ax.bar(x - width / 2, rewards, width, label="Reward", color="#2ca02c")
    summary_ax2 = summary_ax.twinx()
    error_bars = summary_ax2.bar(x + width / 2, errors, width, label="Error RMS", color="#ff7f0e", alpha=0.8)
    summary_ax.set_xticks(x)
    summary_ax.set_xticklabels(labels)
    summary_ax.set_ylabel("Reward")
    summary_ax2.set_ylabel("Error RMS (V)")
    summary_ax.grid(True, axis="y", alpha=0.3)
    summary_ax.set_title("Final comparison")
    lines_1, labels_1 = summary_ax.get_legend_handles_labels()
    lines_2, labels_2 = summary_ax2.get_legend_handles_labels()
    summary_ax.legend(lines_1 + lines_2, labels_1 + labels_2, frameon=False, loc="best")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)

    return {
        "zero_metrics": zero_metrics,
        "ilc_metrics": ilc_metrics,
        "rl_metrics": rl_metrics,
        "figure": fig,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare PPO against ILC on the same cavity control task.")
    parser.add_argument("--timesteps", type=int, default=20000)
    parser.add_argument("--ilc-iters", type=int, default=10)
    parser.add_argument("--save-path", type=str, default="ppo_ilc_compare")
    parser.add_argument("--plot-path", type=str, default="rl_ilc_compare.png")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    env = build_env()
    zero_result = rollout_episode(build_env())
    ilc_result = run_ilc(build_env(), args.ilc_iters)
    rl_result = train_and_evaluate_rl(build_env, args.timesteps)
    rl_result["model"].save(args.save_path)

    plot_summary = None
    if not args.no_plot:
        plot_summary = plot_comparison(
            env,
            zero_result,
            ilc_result,
            rl_result,
            ilc_result["correction"],
            Path(args.plot_path),
        )
        plt.close(plot_summary["figure"])

    print("Environment config:")
    print(asdict(env.config))
    print(f"Zero-correction reward: {zero_result['reward']:.3f}")
    print(f"ILC rewards by iteration: {[round(x, 3) for x in ilc_result['rewards']]}")
    print(f"ILC error RMS by iteration: {[round(x, 3) for x in ilc_result['error_rms']]}")
    print(f"ILC final reward: {ilc_result['final_result']['reward']:.3f}")
    print(f"RL rollout reward: {rl_result['reward']:.3f}")
    if plot_summary is not None:
        print(f"Plot saved to: {Path(args.plot_path).resolve()}")
    print(f"Final zero-action info: {zero_result['last_info']}")
    print(f"Final ILC info: {ilc_result['final_result']['last_info']}")
    print(f"Final RL info: {rl_result['last_info']}")


if __name__ == "__main__":
    main()