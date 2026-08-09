###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
'''
Train an RL controller for cavity frequency-control compensation.
'''

from __future__ import annotations

import argparse
from dataclasses import asdict

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from llrflibs_rl.envs import CavityFrequencyControlEnv, FrequencyControlConfig


def build_env() -> CavityFrequencyControlEnv:
    config = FrequencyControlConfig(
        half_bw=2 * np.pi * 216 * 0.5,
        Ts=2.0e-6,
        episode_len=2000,
        vc_target=25e6,
        drive_level=12.5e6,
        detuning_limit=2 * np.pi * 1.0e4,
        actuator_tau=8.0e-6,
        disturbance_amp=2 * np.pi * 1500.0,
        disturbance_freq=280.0,
    )
    return CavityFrequencyControlEnv(config)


def evaluate_zero_action(env: CavityFrequencyControlEnv):
    obs, info = env.reset()
    rewards = []
    for _ in range(env.episode_len):
        obs, reward, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))
        rewards.append(float(reward))
        if terminated or truncated:
            break
    return float(np.sum(rewards)), info


def main():
    parser = argparse.ArgumentParser(description="Train SAC on the frequency-control RL environment.")
    parser.add_argument("--timesteps", type=int, default=15000)
    parser.add_argument("--save-path", type=str, default="sac_frequency_rl")
    args = parser.parse_args()

    env = build_env()
    zero_reward, zero_info = evaluate_zero_action(build_env())

    vec_env = DummyVecEnv([build_env])
    model = SAC("MlpPolicy", vec_env, verbose=1, train_freq=1, gradient_steps=1, batch_size=64)
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
    print(f"Zero-action baseline reward: {zero_reward:.3f}")
    print(f"RL rollout reward: {cumulative_reward:.3f}")
    print(f"Final baseline info: {zero_info}")


if __name__ == "__main__":
    main()