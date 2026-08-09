###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
'''
Train an RL controller for the basic cavity-simulation example.
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
        episode_len=1600,
        vc_target=25e6,
        filling_len=510,
        flattop_len=900,
        beam_current=0.0,
        detuning=2 * np.pi * 1.5e3,
        randomize_detuning=0.0,
        reward_error_weight=6.5,
        reward_drive_weight=0.10,
        reward_reflected_weight=0.05,
    )
    return CavityVoltageControlEnv(config)


def evaluate_zero_correction(env: CavityVoltageControlEnv):
    obs, info = env.reset()
    rewards = []
    for _ in range(env.episode_len):
        obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
        rewards.append(float(reward))
        if terminated or truncated:
            break
    return float(np.sum(rewards)), info


def main():
    parser = argparse.ArgumentParser(description="Train PPO on the basic cavity simulation RL environment.")
    parser.add_argument("--timesteps", type=int, default=12000)
    parser.add_argument("--save-path", type=str, default="ppo_sim_cavity_basic_rl")
    args = parser.parse_args()

    env = build_env()
    baseline_reward, baseline_info = evaluate_zero_correction(build_env())

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
    print(f"Zero-correction baseline reward: {baseline_reward:.3f}")
    print(f"RL rollout reward: {cumulative_reward:.3f}")
    print(f"Final baseline info: {baseline_info}")


if __name__ == "__main__":
    main()