###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Quickstart example: create environments from the registry and inspect them.

Run from the repository root::

    python examples/quickstart.py
"""

from __future__ import annotations

import numpy as np

import llrflibs_rl as rl


def main() -> None:
    print("llrflibs_rl version:", rl.__version__)
    print("Registered environments:")
    for env_id in rl.list_env_ids():
        spec = rl.get_spec(env_id)
        print(f"  {env_id:<18} [{spec.kind:<11}] {spec.description}")

    env = rl.make("DualLoop-v0")
    print(f"\nDualLoop-v0 observation space: {env.observation_space}")
    print(f"DualLoop-v0 action space:      {env.action_space}")

    obs, info = env.reset(seed=0)
    total = 0.0
    for _ in range(500):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        if terminated or truncated:
            break
    print(f"Zero-action 500-step reward : {total:.2f}")
    print("The RL model can be trained with `python -m llrflibs_rl.scripts.run train --env DualLoop-v0`.")


if __name__ == "__main__":
    main()
