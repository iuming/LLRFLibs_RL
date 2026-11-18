###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Command-line interface of llrflibs_rl.

Subcommands:

* ``list``       — list the registered environment ids.
* ``train``      — train an agent on one environment and persist the run.
* ``eval``       — evaluate a saved SB3 model and compare with the baseline.
* ``benchmark``  — run every environment (quick or recommended budgets) and
                   write a Markdown / JSON summary.

Example::

    python -m llrflibs_rl.scripts.run train --env DualLoop-v0 --timesteps 30000
    python -m llrflibs_rl.scripts.run benchmark --quick --out benchmarks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from stable_baselines3 import PPO, SAC

from .. import list_env_ids, make
from ..controllers.baselines import list_baselines
from ..evaluation import action_from_model, evaluate_episode
from ..evaluation.reporting import write_markdown_table
from ..scenario import run_experiment
from ..utils import print_header

__all__ = ["main"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m llrflibs_rl.scripts.run",
                                     description="LLRF RL command line interface.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list registered environment ids")
    p_list.set_defaults(func=_cmd_list)

    p_train = sub.add_parser("train", help="train an agent on one environment")
    p_train.add_argument("--env", default=None, help="environment id (overrides --config)")
    p_train.add_argument("--config", type=str, default=None,
                         help="experiment YAML under experiments/ (env, timesteps, ...)")
    p_train.add_argument("--timesteps", type=int, default=None)
    p_train.add_argument("--algorithm", choices=["ppo", "sac"], default=None)
    p_train.add_argument("--baseline", type=str, default=None, choices=list_baselines())
    p_train.add_argument("--seed", type=int, default=None)
    p_train.add_argument("--runs-dir", type=str, default=None)
    p_train.add_argument("--quick", action="store_true")
    p_train.set_defaults(func=_cmd_train)

    p_eval = sub.add_parser("eval", help="evaluate a saved model against the baseline")
    p_eval.add_argument("--env", required=True)
    p_eval.add_argument("--model", required=True, help="path to the saved SB3 model")
    p_eval.add_argument("--baseline", type=str, default="zero", choices=list_baselines())
    p_eval.add_argument("--seed", type=int, default=0)
    p_eval.set_defaults(func=_cmd_eval)

    p_bench = sub.add_parser("benchmark", help="run all environments and write a report")
    p_bench.add_argument("--quick", action="store_true",
                         help="use the small sanity budget for every environment")
    p_bench.add_argument("--baseline", type=str, default="zero", choices=list_baselines())
    p_bench.add_argument("--envs", type=str, default=None,
                         help="comma-separated environment ids (default: all registered)")
    p_bench.add_argument("--seed", type=int, default=0)
    p_bench.add_argument("--seeds", type=int, default=3,
                         help="number of evaluation seeds averaged per environment")
    p_bench.add_argument("--runs-dir", type=str, default=None)
    p_bench.add_argument("--out", type=str, default="benchmarks")
    p_bench.set_defaults(func=_cmd_benchmark)
    return parser


def _cmd_list(_args: argparse.Namespace) -> None:
    for env_id in list_env_ids():
        print(env_id)


def _load_experiment(path: str) -> Dict[str, object]:
    """Load an experiment YAML into a flat parameter dictionary.

    Args:
        path: Path to the YAML experiment specification.

    Returns:
        The ``experiment`` mapping (or the whole document when no such key).
    """
    import yaml

    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if isinstance(data, dict) and "experiment" in data:
        return dict(data["experiment"])
    return dict(data or {})


def _cmd_train(args: argparse.Namespace) -> None:
    overrides: Dict[str, object] = {}
    if args.config:
        overrides = _load_experiment(args.config)
    env_id = args.env or str(overrides.get("env", "DualLoop-v0"))
    timesteps = args.timesteps if args.timesteps is not None else overrides.get("timesteps")
    algorithm = args.algorithm
    if algorithm is None and overrides.get("algorithm") is not None:
        algorithm = str(overrides["algorithm"])
    baseline = args.baseline if args.baseline is not None else str(overrides.get("baseline", "zero"))
    seed = args.seed if args.seed is not None else int(overrides.get("seed", 0))

    summary = run_experiment(
        env_id=env_id,
        timesteps=int(timesteps) if timesteps is not None else None,
        algorithm=algorithm,
        baseline=baseline,
        seed=seed,
        quick=args.quick,
        runs_dir=Path(args.runs_dir) if args.runs_dir else None,
    )
    print(f"\nFinished: {summary['env_id']}  [{summary['baseline']}] baseline={summary['baseline_reward']:,.2f} "
          f"RL={summary['rl_reward']:,.2f}")


def _cmd_eval(args: argparse.Namespace) -> None:
    from ..controllers.baselines import make_baseline_policy

    env = make(args.env)
    try:
        model = PPO.load(args.model)
    except Exception:
        model = SAC.load(args.model)

    baseline = evaluate_episode(make(args.env), make_baseline_policy(make(args.env), args.baseline),
                                seed=args.seed)
    rl = evaluate_episode(make(args.env), action_from_model(model), seed=args.seed)
    print_header(f"Evaluation  (baseline={args.baseline})")
    for key in ("reward",):
        print(f"{key:>20}: baseline={baseline[key]:,.3f}  RL={rl[key]:,.3f}")
    for key in ("error_rms", "flattop_error_rms", "amp_error_rms"):
        if key in rl:
            print(f"{key:>20}: baseline={baseline.get(key, float('nan')):,.4e}  "
                  f"RL={rl.get(key, float('nan')):,.4e}")


def _cmd_benchmark(args: argparse.Namespace) -> None:
    from ..registry import get_spec

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluate_seeds = list(range(args.seed, args.seed + args.seeds))
    env_ids = [e.strip() for e in args.envs.split(",") if e.strip()] if args.envs else list_env_ids()
    for env_id in env_ids:
        get_spec(env_id)  # validate id early
    rows: List[Dict[str, object]] = []
    for env_id in env_ids:
        print_header(f"Benchmarking {env_id}  (baseline={args.baseline}, seeds={evaluate_seeds})")
        summary = run_experiment(
            env_id=env_id,
            seed=args.seed,
            baseline=args.baseline,
            quick=args.quick,
            runs_dir=Path(args.runs_dir) if args.runs_dir else None,
            evaluate_seeds=evaluate_seeds,
            verbose=False,
        )
        row: Dict[str, object] = {
            "env": env_id,
            "baseline_reward": summary["baseline_reward"],
            "rl_reward": summary["rl_reward"],
            "rl_reward_std": summary.get("rl_reward_std"),
            "n_seeds": len(evaluate_seeds),
        }
        for key in ("error_rms", "flattop_error_rms"):
            if f"rl_{key}" in summary:
                row[f"baseline_{key}"] = summary[f"baseline_{key}"]
                row[f"rl_{key}"] = summary[f"rl_{key}"]
        rows.append(row)

    write_markdown_table(rows, out_dir / "benchmark.md", title="LLRF RL benchmark")
    (out_dir / "benchmark.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Benchmark report written to {out_dir / 'benchmark.md'}")


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).
    """
    parser = _parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
