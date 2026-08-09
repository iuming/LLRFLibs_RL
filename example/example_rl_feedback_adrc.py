###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
'''
Train an RL controller for ADRC-style feedback control.
'''

from __future__ import annotations

import argparse
from dataclasses import asdict

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from llrflibs_rl.envs import CavityVoltageControlEnv, VoltageControlConfig


def build_env() -> CavityVoltageControlEnv:
    config = VoltageControlConfig(
        half_bw=2 * np.pi * 216 * 0.5,
        Ts=1.0e-6,
        episode_len=1800,
        vc_target=25e6,
        filling_len=510,
        flattop_len=800,
        beam_current=0.008,
        beam_phase_deg=0.0,
        detuning=2 * np.pi * 2.0e3,
        reward_error_weight=7.5,
        reward_drive_weight=0.20,
        reward_reflected_weight=0.08,
        randomize_detuning=2 * np.pi * 700.0,
    )
    return CavityVoltageControlEnv(config)


def evaluate_adrc_style_baseline(env: CavityVoltageControlEnv, kp: float = 0.8, ki: float = 0.002, kd: float = 0.04):
    obs, info = env.reset()
    integral = 0.0 + 0.0j
    previous_error = 0.0 + 0.0j
    rewards = []
    for _ in range(env.episode_len):
        error = complex(float(obs[10]), float(obs[11])) * env.config.vc_target
        error_rate = (error - previous_error) / max(env.config.Ts, 1.0e-12)
        integral += error * env.config.Ts
        disturbance_proxy = complex(float(obs[8]), float(obs[9])) * env.config.vc_target
        correction = -(kp * error + ki * integral + kd * error_rate) - 0.02 * disturbance_proxy
        action = np.array(
            [
                np.clip(correction.real / max(env.drive_limit, 1.0), -1.0, 1.0),
                np.clip(correction.imag / max(env.drive_limit, 1.0), -1.0, 1.0),
            ],
            dtype=np.float32,
        )
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        previous_error = error
        if terminated or truncated:
            break
    return float(np.sum(rewards)), info


def main():
    parser = argparse.ArgumentParser(description="Train PPO on the ADRC-style feedback RL environment.")
    parser.add_argument("--timesteps", type=int, default=14000)
    parser.add_argument("--save-path", type=str, default="ppo_feedback_adrc_rl")
    args = parser.parse_args()

    env = build_env()
    baseline_reward, baseline_info = evaluate_adrc_style_baseline(build_env())

    vec_env = DummyVecEnv([build_env])
    model = PPO("MlpPolicy", vec_env, verbose=1, n_steps=256, batch_size=64, gamma=0.995)
    model.learn(total_timesteps=args.timesteps)
    model.save(args.save_path)

    obs = vec_env.reset()
    cumulative_reward = 0.0
    for _ in range(env.episode_len):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        cumulative_reward += float(reward[0])
        if done[0]:
            break

    print("Environment config:")
    print(asdict(env.config))
    print(f"ADRC-style baseline reward: {baseline_reward:.3f}")
    print(f"RL rollout reward: {cumulative_reward:.3f}")
    print(f"Final baseline info: {baseline_info}")


if __name__ == "__main__":
    main()