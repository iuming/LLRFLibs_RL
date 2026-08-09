###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
'''
Compare RL residual control against the analytical feedforward example.
'''

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from llrflibs.rf_control import cav_sp_ff, ss_discrete
from llrflibs.rf_sim import cav_ss, sim_ncav_step


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 220,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.family": "DejaVu Sans",
    }
)


def _complex_state_matrix(size: int, dtype=complex) -> np.matrix:
    return np.matrix(np.zeros((size, 1), dtype=dtype))


@dataclass
class FeedforwardTrackingConfig:
    """Configuration for the RL-vs-feedforward comparison environment."""

    half_bw: float
    Ts: float
    episode_len: int
    vc_target: float = 25e6
    filling_len: int = 510
    flattop_len: int = 800
    beam_current: float = 0.003
    beam_phase_deg: float = -30.0
    nominal_detuning: float = 0.0
    actual_detuning: float = 2.0 * np.pi * 100.0
    beta: float = 1.0e4
    roQ_or_RoQ: float = 1036.0
    QL: float = 3.0e6
    machine: str = "linac"
    drive_limit_ratio: float = 0.30
    reward_error_weight: float = 7.0
    reward_drive_weight: float = 0.12
    reward_reflected_weight: float = 0.05


class FeedforwardTrackingEnv(gym.Env):
    """Track a cavity pulse with residual RL correction on top of nominal feedforward."""

    metadata = {"render_modes": []}

    def __init__(self, config: FeedforwardTrackingConfig):
        super().__init__()

        self.config = config
        self.episode_len = int(config.episode_len)

        status, sp, vf_ff, vb, _ = cav_sp_ff(
            config.half_bw,
            config.filling_len,
            config.flattop_len,
            config.Ts,
            self.episode_len,
            vc0=config.vc_target,
            detuning=config.nominal_detuning,
            const_fpow=True,
            ib0=config.beam_current,
            phib_deg=config.beam_phase_deg,
            beam_ids=config.filling_len,
            beam_ide=min(self.episode_len - 1, config.filling_len + config.flattop_len - 1),
            roQ_or_RoQ=config.roQ_or_RoQ,
            QL=config.QL,
            machine=config.machine,
        )
        if not status:
            raise ValueError("Failed to generate the nominal feedforward waveform.")

        self._sp = np.asarray(sp, dtype=np.complex128)
        self._vf_ff = np.asarray(vf_ff, dtype=np.complex128)
        self._vb = np.asarray(vb, dtype=np.complex128)
        self.drive_limit = float(config.drive_limit_ratio * max(np.max(np.abs(self._vf_ff)), config.vc_target, 1.0))
        self._obs_scale = float(max(np.max(np.abs(self._vf_ff)), config.vc_target, 1.0))

        status, Arf, Brf, Crf, Drf, Abm, Bbm, Cbm, Dbm = cav_ss(
            config.half_bw,
            detuning=config.actual_detuning,
            beta=config.beta,
            plot=False,
        )
        if not status:
            raise ValueError("Failed to build the actual cavity model.")

        status, self._Arf_d, self._Brf_d, self._Crf_d, self._Drf_d, _ = ss_discrete(
            Arf, Brf, Crf, Drf, config.Ts, method="zoh", plot=False
        )
        if not status:
            raise ValueError("Failed to discretize the cavity model.")

        status, self._Abm_d, self._Bbm_d, self._Cbm_d, self._Dbm_d, _ = ss_discrete(
            Abm, Bbm, Cbm, Dbm, config.Ts, method="bilinear", plot=False
        )
        if not status:
            raise ValueError("Failed to discretize the beam model.")

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)

        self._state_rf = None
        self._state_bm = None
        self._vc = 0.0 + 0.0j
        self._vr = 0.0 + 0.0j
        self._last_action = 0.0 + 0.0j
        self._drive_total = 0.0 + 0.0j
        self._step_index = 0

    def _get_obs(self) -> np.ndarray:
        idx = min(self._step_index, self.episode_len - 1)
        sp = self._sp[idx]
        vf_ff = self._vf_ff[idx]
        vb = self._vb[idx]
        error = sp - self._vc
        return np.array(
            [
                self._vc.real / self._obs_scale,
                self._vc.imag / self._obs_scale,
                sp.real / self._obs_scale,
                sp.imag / self._obs_scale,
                vf_ff.real / self._obs_scale,
                vf_ff.imag / self._obs_scale,
                vb.real / self._obs_scale,
                vb.imag / self._obs_scale,
                self._vr.real / self._obs_scale,
                self._vr.imag / self._obs_scale,
                error.real / self._obs_scale,
                error.imag / self._obs_scale,
                self._last_action.real / max(self.drive_limit, 1.0),
                self._last_action.imag / max(self.drive_limit, 1.0),
                self._step_index / max(self.episode_len - 1, 1),
            ],
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._state_rf = _complex_state_matrix(self._Brf_d.shape[0])
        self._state_bm = _complex_state_matrix(self._Bbm_d.shape[0], dtype=complex)
        self._vc = 0.0 + 0.0j
        self._vr = 0.0 + 0.0j
        self._last_action = 0.0 + 0.0j
        self._drive_total = self._vf_ff[0]
        self._step_index = 0
        return self._get_obs(), self._make_info()

    def _make_info(self) -> dict[str, float]:
        idx = min(self._step_index, self.episode_len - 1)
        sp = self._sp[idx]
        error = sp - self._vc
        return {
            "step_index": float(self._step_index),
            "vc_abs": float(np.abs(self._vc)),
            "sp_abs": float(np.abs(sp)),
            "error_abs": float(np.abs(error)),
            "drive_abs": float(np.abs(self._drive_total)),
            "reflection_abs": float(np.abs(self._vr)),
        }

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(2)
        action = np.clip(action, -1.0, 1.0)
        correction = complex(float(action[0]), float(action[1])) * self.drive_limit
        self._last_action = correction

        idx = min(self._step_index, self.episode_len - 1)
        drive = self._vf_ff[idx] + correction
        beam = self._vb[idx] if self._vb.size > 0 else 0.0 + 0.0j

        status, vc_next, vr_next, state_rf, state_bm = sim_ncav_step(
            self._Arf_d,
            self._Brf_d,
            self._Crf_d,
            self._Drf_d,
            drive,
            self._state_rf,
            Abmd=self._Abm_d,
            Bbmd=self._Bbm_d,
            Cbmd=self._Cbm_d,
            Dbmd=self._Dbm_d,
            vb_step=beam,
            state_bm0=self._state_bm,
        )
        if not status:
            raise RuntimeError("Cavity step simulation failed.")

        self._state_rf = state_rf
        self._state_bm = state_bm
        self._vc = vc_next
        self._vr = vr_next
        self._drive_total = drive
        self._step_index += 1

        sp = self._sp[idx]
        error = sp - self._vc
        error_norm = float(np.abs(error) / max(self._obs_scale, 1.0))
        drive_norm = float(np.abs(correction) / max(self.drive_limit, 1.0))
        refl_norm = float(np.abs(self._vr) / max(np.abs(drive), self._obs_scale, 1.0))
        reward = -(
            self.config.reward_error_weight * error_norm**2
            + self.config.reward_drive_weight * drive_norm**2
            + self.config.reward_reflected_weight * refl_norm**2
        )

        terminated = False
        truncated = self._step_index >= self.episode_len
        return self._get_obs(), float(reward), terminated, truncated, self._make_info()


def build_env() -> FeedforwardTrackingEnv:
    config = FeedforwardTrackingConfig(
        half_bw=2.0 * np.pi * 216.0 * 0.5,
        Ts=1.0e-6,
        episode_len=2048,
        vc_target=25e6,
        filling_len=510,
        flattop_len=800,
        beam_current=0.003,
        beam_phase_deg=-30.0,
        nominal_detuning=0.0,
        actual_detuning=2.0 * np.pi * 100.0,
        reward_error_weight=7.0,
        reward_drive_weight=0.12,
        reward_reflected_weight=0.05,
    )
    return FeedforwardTrackingEnv(config)


def rollout_with_correction(env: FeedforwardTrackingEnv, correction: np.ndarray | None = None):
    obs, info = env.reset()
    rewards = []
    vc_wave = np.zeros(env.episode_len, dtype=np.complex128)
    drive_wave = np.zeros(env.episode_len, dtype=np.complex128)
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
        drive_wave[step] = env._drive_total
        rewards.append(float(reward))
        if terminated or truncated:
            break

    return {
        "reward": float(np.sum(rewards)),
        "vc_wave": vc_wave,
        "drive_wave": drive_wave,
        "last_info": info,
        "step_count": len(rewards),
    }


def rollout_rl(env: FeedforwardTrackingEnv, model: PPO):
    obs, info = env.reset()
    rewards = []
    vc_wave = np.zeros(env.episode_len, dtype=np.complex128)
    drive_wave = np.zeros(env.episode_len, dtype=np.complex128)

    for step in range(env.episode_len):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        vc_wave[step] = env._vc
        drive_wave[step] = env._drive_total
        rewards.append(float(reward))
        if terminated or truncated:
            break

    return {
        "reward": float(np.sum(rewards)),
        "vc_wave": vc_wave,
        "drive_wave": drive_wave,
        "last_info": info,
        "step_count": len(rewards),
    }


def compute_metrics(env: FeedforwardTrackingEnv, result: dict):
    sp = np.asarray(env._sp, dtype=np.complex128)
    vc = np.asarray(result["vc_wave"], dtype=np.complex128)
    drive = np.asarray(result["drive_wave"], dtype=np.complex128)
    n = min(len(sp), len(vc), len(drive))
    error = sp[:n] - vc[:n]
    return {
        "amplitude": np.abs(vc[:n]),
        "phase": np.angle(vc[:n], deg=True),
        "drive_amplitude": np.abs(drive[:n]),
        "error_amplitude": np.abs(error),
        "error_rms": float(np.sqrt(np.mean(np.abs(error) ** 2))),
        "reward": float(result["reward"]),
    }


def plot_comparison(
    env: FeedforwardTrackingEnv,
    nominal_result: dict,
    rl_result: dict,
    output_path: Path | None,
):
    sp = np.asarray(env._sp, dtype=np.complex128)
    t = np.arange(sp.shape[0]) * env.config.Ts * 1e6

    nominal_metrics = compute_metrics(env, nominal_result)
    rl_metrics = compute_metrics(env, rl_result)

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.2), constrained_layout=True)
    fig.suptitle("RL residual control versus analytical feedforward", fontsize=15, fontweight="bold", y=1.01)

    panel_tags = ["(a)", "(b)", "(c)", "(d)"]

    def _panel_label(ax, tag: str):
        ax.text(
            0.02,
            0.96,
            tag,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )

    def _annotate_summary(ax, value: float, title: str, color: str):
        ax.text(
            0.98,
            0.04,
            f"{title}: {value:.2e}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color=color,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": color, "alpha": 0.9},
        )

    amp_ax = axes[0, 0]
    amp_ax.plot(t, np.abs(sp), color="black", linewidth=2.1, label="Setpoint")
    amp_ax.plot(t, nominal_metrics["amplitude"], color="#7f7f7f", linestyle="--", linewidth=1.9, label="Nominal FF response")
    amp_ax.plot(t, rl_metrics["amplitude"], color="#c62828", linewidth=2.0, label="RL response")
    amp_ax.set_ylabel("|Vc| (V)")
    amp_ax.set_xlim(t[0], t[-1])
    amp_ax.legend(frameon=False, loc="upper right")
    amp_ax.set_title("Amplitude tracking")
    _panel_label(amp_ax, panel_tags[0])
    _annotate_summary(amp_ax, rl_metrics["error_rms"], "RL RMS error", "#c62828")

    phase_ax = axes[0, 1]
    phase_ax.plot(t, np.angle(sp, deg=True), color="black", linewidth=2.1, label="Setpoint")
    phase_ax.plot(t, np.angle(nominal_result["vc_wave"], deg=True), color="#7f7f7f", linestyle="--", linewidth=1.9, label="Nominal FF response")
    phase_ax.plot(t, np.angle(rl_result["vc_wave"], deg=True), color="#c62828", linewidth=2.0, label="RL response")
    phase_ax.set_ylabel("Phase (deg)")
    phase_ax.set_xlim(t[0], t[-1])
    phase_ax.legend(frameon=False, loc="upper left")
    phase_ax.set_title("Phase tracking")
    _panel_label(phase_ax, panel_tags[1])
    _annotate_summary(phase_ax, nominal_metrics["error_rms"], "Nominal RMS error", "#7f7f7f")

    drive_ax = axes[1, 0]
    drive_ax.plot(t, nominal_metrics["drive_amplitude"], color="#1565c0", linewidth=1.9, label="Nominal FF drive")
    drive_ax.plot(t, rl_metrics["drive_amplitude"], color="#ef6c00", linewidth=1.9, label="RL drive")
    drive_ax.set_ylabel("|Drive| (V)")
    drive_ax.set_xlim(t[0], t[-1])
    drive_ax.legend(frameon=False, loc="upper right")
    drive_ax.set_title("Drive effort")
    _panel_label(drive_ax, panel_tags[2])
    _annotate_summary(drive_ax, float(np.max(rl_metrics["drive_amplitude"])), "RL peak drive", "#ef6c00")

    err_ax = axes[1, 1]
    err_ax.plot(t, nominal_metrics["error_amplitude"], color="#1565c0", linewidth=1.9, label="Nominal FF error")
    err_ax.plot(t, rl_metrics["error_amplitude"], color="#ef6c00", linewidth=1.9, label="RL error")
    err_ax.set_ylabel("|Error| (V)")
    err_ax.set_xlim(t[0], t[-1])
    err_ax.legend(frameon=False, loc="upper right")
    err_ax.set_title("Tracking error")
    _panel_label(err_ax, panel_tags[3])
    _annotate_summary(err_ax, rl_metrics["error_rms"], "RL RMS error", "#ef6c00")

    for ax in axes.flat:
        ax.grid(True, alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[1, :]:
        ax.set_xlabel("Time (us)")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)

    return {
        "nominal_metrics": nominal_metrics,
        "rl_metrics": rl_metrics,
        "figure": fig,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare PPO against the analytical feedforward example.")
    parser.add_argument("--timesteps", type=int, default=16000)
    parser.add_argument("--save-path", type=str, default="ppo_sp_ff_rl")
    parser.add_argument("--plot-path", type=str, default="rl_sp_ff_compare.png")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    env = build_env()
    nominal_result = rollout_with_correction(build_env())

    vec_env = DummyVecEnv([build_env])
    model = PPO("MlpPolicy", vec_env, verbose=1, n_steps=256, batch_size=64, gamma=0.995)
    model.learn(total_timesteps=args.timesteps)
    model.save(args.save_path)

    rl_result = rollout_rl(build_env(), model)

    plot_summary = None
    if not args.no_plot:
        plot_summary = plot_comparison(env, nominal_result, rl_result, Path(args.plot_path))
        plt.close(plot_summary["figure"])

    nominal_metrics = compute_metrics(env, nominal_result)
    rl_metrics = compute_metrics(env, rl_result)

    print("Environment config:")
    print(asdict(env.config))
    print(f"Nominal feedforward reward: {nominal_result['reward']:.3f}")
    print(f"Nominal feedforward error RMS: {nominal_metrics['error_rms']:.3f}")
    print(f"RL rollout reward: {rl_result['reward']:.3f}")
    print(f"RL rollout error RMS: {rl_metrics['error_rms']:.3f}")
    if plot_summary is not None:
        print(f"Plot saved to: {Path(args.plot_path).resolve()}")
    print(f"Final nominal info: {nominal_result['last_info']}")
    print(f"Final RL info: {rl_result['last_info']}")


if __name__ == "__main__":
    main()