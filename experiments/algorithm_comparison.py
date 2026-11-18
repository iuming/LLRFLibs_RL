###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Full-algorithm comparison across every registered environment.

For each environment the recommended training budget is spent on PPO, SAC and
TD3 (off-policy variants use ``train_freq=gradient_steps=64, batch_size=256``)
and the deterministic policies are evaluated over seeds 0-3.  The result table
drives the per-environment algorithm presets in ``agents/presets.py``.

Run from the repository root::

    python experiments/algorithm_comparison.py [--envs env1,env2,...]

Results are appended incrementally to ``benchmarks/algorithm_comparison.json``
so long runs survive interruptions; already-computed environments are skipped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import llrflibs_rl as rl
from llrflibs_rl.agents import train
from llrflibs_rl.configs.agent import AgentConfig
from llrflibs_rl.controllers.baselines import make_baseline_policy
from llrflibs_rl.evaluation import action_from_model, evaluate_over_seeds

ALGORITHMS = ["ppo", "sac", "td3"]
OFFPOLICY_HP = dict(train_freq=64, gradient_steps=64, batch_size=256)
EVAL_SEEDS = [0, 1, 2, 3]


def _run(env_id: str, algorithm: str, seed: int = 0) -> Dict[str, float]:
    budget = rl.recommended_timesteps(env_id)
    kwargs: Dict[str, object] = dict(algorithm=algorithm, timesteps=budget, seed=seed)
    if algorithm != "ppo":
        kwargs.update(OFFPOLICY_HP)
    cfg = AgentConfig(**kwargs)
    fac = lambda: rl.make(env_id)  # noqa: E731
    model = train(fac, cfg, eval_env_factory=fac, eval_freq=max(1024, budget // 8), eval_seed=0)
    rob = evaluate_over_seeds(fac, action_from_model(model), EVAL_SEEDS)
    return {
        "reward_mean": float(rob["reward_mean"]),
        "reward_std": float(rob["reward_std"]),
        "error_rms_mean": float(rob.get("error_rms_mean", float("nan"))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=str, default=None,
                        help="comma-separated environment ids (default: all)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_json = root / "benchmarks" / "algorithm_comparison.json"
    if out_json.exists():
        rows = json.loads(out_json.read_text(encoding="utf-8"))
    else:
        rows = []
    done = {row["env"] for row in rows}

    env_ids = [e.strip() for e in args.envs.split(",") if e.strip()] if args.envs else rl.list_env_ids()
    for env_id in env_ids:
        if env_id in done:
            print(f"=== {env_id}: already computed, skipping", flush=True)
            continue
        base = evaluate_over_seeds(
            lambda: rl.make(env_id),
            make_baseline_policy(rl.make(env_id), "zero"),
            EVAL_SEEDS,
        )
        scores: Dict[str, float] = {}
        for algo in ALGORITHMS:
            print(f"=== {env_id} / {algo} ...", flush=True)
            result = _run(env_id, algo)
            scores[algo] = result["reward_mean"]
            print(f"    {algo}: {result['reward_mean']:,.1f} +/- {result['reward_std']:,.1f}",
                  flush=True)
        best = max(ALGORITHMS, key=lambda a: scores[a])
        print(f"    -> best: {best} (vs baseline {base['reward_mean']:,.1f})", flush=True)
        rows.append({
            "env": env_id,
            "baseline_reward": base["reward_mean"],
            "best_algorithm": best,
            **{f"{a}_reward": scores[a] for a in ALGORITHMS},
        })
        out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    out = root / "benchmarks" / "algorithm_comparison.md"
    lines = ["# Algorithm comparison across all environments", "",
             "Recommended budget per environment, 4-seed deterministic evaluation.",
             "Higher reward is better.", "", "| env | baseline | PPO | SAC | TD3 | best |",
             "|---|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append("| {env} | {baseline_reward:,.1f} | {ppo_reward:,.1f} | {sac_reward:,.1f} "
                     "| {td3_reward:,.1f} | {best_algorithm} |".format(**row))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport written to {out}", flush=True)


if __name__ == "__main__":
    main()
