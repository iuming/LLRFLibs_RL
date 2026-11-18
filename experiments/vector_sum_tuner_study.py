###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Dedicated stability study for ``VectorSumTuner-v0``.

The observability fix (signed per-cavity detuning) made the task learnable but
the default PPO setup still collapses into drive saturation on longer
budgets.  This script compares algorithm / budget / observation-normalization
variants with a 4-seed deterministic evaluation.

Run from the repository root::

    python experiments/vector_sum_tuner_study.py

Writes ``benchmarks/vstuner_study.md`` and prints the result table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

import llrflibs_rl as rl
from llrflibs_rl.agents import train
from llrflibs_rl.configs.agent import AgentConfig
from llrflibs_rl.controllers.baselines import make_baseline_policy
from llrflibs_rl.evaluation import action_from_model, evaluate_over_seeds

ENV_ID = "VectorSumTuner-v0"
EVAL_SEEDS = [0, 1, 2, 3]


def _env_factory():
    return rl.make(ENV_ID)


def _run(cfg: AgentConfig) -> Dict[str, object]:
    model = train(_env_factory, cfg, eval_env_factory=_env_factory, curve=[])
    robust = evaluate_over_seeds(_env_factory, action_from_model(model), EVAL_SEEDS)
    return {
        "reward_mean": float(robust["reward_mean"]),
        "reward_std": float(robust["reward_std"]),
        "error_rms_mean": float(robust.get("error_rms_mean", float("nan"))),
    }


def _configs() -> List[Dict[str, object]]:
    off = dict(timesteps=20000, seed=0, train_freq=64, gradient_steps=64, batch_size=256)
    return [
        {"name": "ppo-4k (current)", "cfg": AgentConfig(algorithm="ppo", timesteps=4000, seed=0),
         "note": "current recommended budget"},
        {"name": "ppo-20k", "cfg": AgentConfig(algorithm="ppo", timesteps=20000, seed=0),
         "note": "default PPO, longer budget"},
        {"name": "ppo-norm-20k", "cfg": AgentConfig(algorithm="ppo", timesteps=20000, seed=0,
                                                     normalize_obs=True),
         "note": "PPO + VecNormalize observations"},
        {"name": "sac-20k", "cfg": AgentConfig(algorithm="sac", **off),
         "note": "off-policy SAC (freq=grad=64)"},
        {"name": "sac-norm-20k", "cfg": AgentConfig(algorithm="sac", **off, normalize_obs=True),
         "note": "SAC + VecNormalize"},
        {"name": "td3-20k", "cfg": AgentConfig(algorithm="td3", **off),
         "note": "off-policy TD3"},
        {"name": "td3-norm-20k", "cfg": AgentConfig(algorithm="td3", **off, normalize_obs=True),
         "note": "TD3 + VecNormalize"},
    ]


def main() -> None:
    baseline = evaluate_over_seeds(
        _env_factory,
        make_baseline_policy(rl.make(ENV_ID), "zero"),
        EVAL_SEEDS,
    )
    rows = [{
        "conf": "baseline (zero)",
        "reward_mean": baseline["reward_mean"],
        "reward_std": baseline["reward_std"],
        "error_rms_mean": baseline.get("error_rms_mean", float("nan")),
        "note": "nominal feedforward only",
    }]
    for item in _configs():
        print(f"=== {item['name']} ...")
        result = _run(item["cfg"])
        rows.append({
            "conf": item["name"],
            **result,
            "note": item["note"],
        })
        print(f"    reward = {result['reward_mean']:,.1f} +/- {result['reward_std']:,.1f}")

    best = max(rows[1:], key=lambda r: r["reward_mean"])
    print("\nBest config:", best["conf"], best["reward_mean"])

    out = Path(__file__).resolve().parents[1] / "benchmarks" / "vstuner_study.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# VectorSumTuner-v0 stability study", "",
             "4-seed deterministic evaluation; reward = mean episode reward "
             "(higher is better, baseline FF only).", "", "| config | reward_mean | reward_std | error_rms | note |",
             "|---|---:|---:|---:|---|"]
    for row in rows:
        lines.append("| {conf} | {reward_mean:,.1f} | {reward_std:,.1f} | {error_rms_mean:,.3e} | {note} |".format(
            conf=row["conf"], reward_mean=row["reward_mean"], reward_std=row["reward_std"],
            error_rms_mean=row["error_rms_mean"], note=row["note"]))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written to {out}")
    (out.with_suffix(".json")).write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
