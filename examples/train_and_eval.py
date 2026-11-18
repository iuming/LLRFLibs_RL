###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Train an agent on one registered environment and write the run artifacts.

Example (from the repository root)::

    python examples/train_and_eval.py --env PIAssist-v0 --timesteps 40000
    python examples/train_and_eval.py --env DualLoop-v0  --timesteps 30000
    python examples/train_and_eval.py --env VectorSum-v0 --timesteps 60000
    python examples/train_and_eval.py --env CavityVoltage-v0 --quick

Each run writes config.json, metrics.json, the trained model and a comparison
figure into a timestamped directory under ``runs/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from llrflibs_rl.scenario import run_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate an LLRF RL controller.")
    parser.add_argument("--env", required=True, help="registered environment id")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--algorithm", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--baseline", type=str, default="zero",
                        choices=["zero", "ff", "pi", "adrc"],
                        help="classical baseline compared against RL")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs-dir", type=str, default=None)
    parser.add_argument("--quick", action="store_true", help="small sanity budget")
    return parser


def main() -> None:
    args = _parser().parse_args()
    summary = run_experiment(
        env_id=args.env,
        timesteps=args.timesteps,
        algorithm=args.algorithm,
        baseline=args.baseline,
        seed=args.seed,
        quick=args.quick,
        runs_dir=Path(args.runs_dir) if args.runs_dir else None,
    )
    print(f"\n=== {summary['env_id']} (baseline={summary['baseline']}) ===")
    print(f"baseline reward : {summary['baseline_reward']:,.2f}")
    print(f"RL reward       : {summary['rl_reward']:,.2f}")
    print(f"artifacts       : {summary['artifact_dir']}")


if __name__ == "__main__":
    main()
