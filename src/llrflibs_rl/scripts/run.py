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
    p_train.add_argument("--render", action="store_true",
                         help="live-render a short episode of the policy while it trains")
    p_train.add_argument("--render-freq", type=int, default=2048,
                         help="live-render every N training steps")
    p_train.add_argument("--render-steps", type=int, default=300,
                         help="samples per live-rendered episode")
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

    p_verify = sub.add_parser("verify", help="check the Gymnasium contract of environments")
    p_verify.add_argument("--envs", type=str, default=None,
                          help="comma-separated ids (default: all)")
    p_verify.add_argument("--n-steps", type=int, default=60)
    p_verify.set_defaults(func=_cmd_verify)

    p_gallery = sub.add_parser("gallery", help="render baseline-vs-RL figures from the zoo")
    p_gallery.add_argument("--envs", type=str, default=None,
                           help="comma-separated ids (default: all with a saved model)")
    p_gallery.add_argument("--out", type=str, default="gallery")
    p_gallery.add_argument("--zoo-dir", type=str, default=None)
    p_gallery.set_defaults(func=_cmd_gallery)

    p_zoo = sub.add_parser("zoo", help="managed trained-model storage")
    zoo_sub = p_zoo.add_subparsers(dest="zoo_command", required=True)
    z_list = zoo_sub.add_parser("list", help="list saved models")
    z_list.add_argument("--zoo-dir", type=str, default=None)
    z_list.set_defaults(func=_cmd_zoo_list)
    z_latest = zoo_sub.add_parser("latest", help="show the latest model of an environment")
    z_latest.add_argument("--env", required=True)
    z_latest.add_argument("--zoo-dir", type=str, default=None)
    z_latest.set_defaults(func=_cmd_zoo_latest)

    p_sysid = sub.add_parser("sysid", help="identify cavity parameters of a simulated plant")
    p_sysid.add_argument("--env", type=str, default="CavityVoltage-v0")
    p_sysid.add_argument("--n-prbs", type=int, default=2048)
    p_sysid.set_defaults(func=_cmd_sysid)

    p_calib = sub.add_parser("calib", help="virtual-probe / forward calibration on a simulated plant")
    p_calib.add_argument("--env", type=str, default="CavityVoltage-v0")
    p_calib.set_defaults(func=_cmd_calib)

    p_loop = sub.add_parser("loop", help="loop stability margins of a cavity + PI controller")
    p_loop.add_argument("--env", type=str, default="CavityVoltage-v0")
    p_loop.add_argument("--delay-us", type=float, default=0.0,
                        help="loop transport delay in microseconds")
    p_loop.set_defaults(func=_cmd_loop)

    p_spec = sub.add_parser("spectrum", help="PSD of the tracking error of the baseline policy")
    p_spec.add_argument("--env", type=str, default="CavityVoltage-v0")
    p_spec.add_argument("--bands", type=str, default=None,
                        help="comma-separated f1-f2 bands for band RMS (e.g. '20-100,100-500')")
    p_spec.set_defaults(func=_cmd_spectrum)

    p_tune = sub.add_parser("tune", help="GP/Bayesian auto-tune of the PI residual gain")
    p_tune.add_argument("--env", type=str, default="CavityVoltage-v0")
    p_tune.add_argument("--n-iters", type=int, default=14)
    p_tune.add_argument("--seed", type=int, default=0)
    p_tune.set_defaults(func=_cmd_tune)

    p_kpi = sub.add_parser("kpi", help="KPI pass/fail report of a baseline policy")
    p_kpi.add_argument("--env", type=str, default="CavityVoltage-v0")
    p_kpi.add_argument("--baseline", type=str, default="zero", choices=list_baselines())
    p_kpi.add_argument("--seed", type=int, default=0)
    p_kpi.set_defaults(func=_cmd_kpi)

    p_diag = sub.add_parser("diag", help="open-loop diagnostics: identified params + loop margins")
    p_diag.add_argument("--env", type=str, default="CavityVoltage-v0")
    p_diag.set_defaults(func=_cmd_diag)

    p_demo = sub.add_parser("demo", help="run a CartPole-style visualization of an environment")
    p_demo.add_argument("--env", type=str, default="CavityVoltage-v0")
    p_demo.add_argument("--policy", type=str, default="zero", choices=list_baselines())
    p_demo.add_argument("--steps", type=int, default=400)
    p_demo.add_argument("--frame", type=int, default=4,
                        help="render every N-th step (keeps headless mode fast)")
    p_demo.add_argument("--seed", type=int, default=0)
    p_demo.add_argument("--show", action="store_true",
                        help="show the figure interactively (blocks until closed)")
    p_demo.add_argument("--out", type=str, default=None,
                        help="PNG path of the rendered episode (default gallery/demo_<env>.png)")
    p_demo.set_defaults(func=_cmd_demo)
    return parser


def _cmd_list(_args: argparse.Namespace) -> None:
    for env_id in list_env_ids():
        print(env_id)


def _load_experiment(path: str) -> dict[str, object]:
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
    overrides: dict[str, object] = {}
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
        render=args.render,
        render_freq=args.render_freq,
        render_steps=args.render_steps,
    )
    print(f"\nFinished: {summary['env_id']}  [{summary['baseline']}] baseline={summary['baseline_reward']:,.2f} "
          f"RL={summary['rl_reward']:,.2f}")


def _cmd_eval(args: argparse.Namespace) -> None:
    from ..controllers.baselines import make_baseline_policy

    make(args.env)  # validate that the environment id is registered
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
    rows: list[dict[str, object]] = []
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
        row: dict[str, object] = {
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


def _cmd_verify(args: argparse.Namespace) -> None:
    from ..utils.verifier import verify_all_environments

    env_ids = _env_list(args.envs)
    reports = verify_all_environments(lambda: env_ids, n_steps=args.n_steps)
    ok = 0
    for report in reports:
        print(report.fmt())
        ok += int(report.passed)
    print(f"\n{ok}/{len(reports)} environments passed.")
    if ok != len(reports):
        raise SystemExit(1)


def _env_list(envs: str | None) -> list[str]:
    if envs is None:
        return list_env_ids()
    return [e.strip() for e in envs.split(",") if e.strip()]


def _cmd_gallery(args: argparse.Namespace) -> None:
    from ..agents.model_zoo import ModelZoo
    from ..controllers.baselines import make_baseline_policy
    from ..evaluation import action_from_model
    from ..scenario import rollout_episode
    from ..utils.plotting import plot_comparison

    zoo = ModelZoo(Path(args.zoo_dir) if args.zoo_dir else None)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0
    for env_id in _env_list(args.envs):
        model = zoo.latest(env_id)
        if model is None:
            print(f"skip {env_id}: no zoo model (train one with `run.py train`)")
            continue
        env = make(env_id)
        baseline_wave = rollout_episode(env, make_baseline_policy(env, "zero"), seed=0)
        rl_wave = rollout_episode(env, action_from_model(model), seed=0)
        probe = make(env_id)
        probe.reset(seed=0)
        path = out_dir / f"{env_id}.png"
        plot_comparison(probe, baseline_wave, rl_wave, "zero", [], path)
        rendered += 1
        print(f"rendered {path}")
    print(f"gallery: {rendered} figure(s) written to {out_dir}")


def _cmd_zoo_list(args: argparse.Namespace) -> None:
    from ..agents.model_zoo import ModelZoo

    rows = ModelZoo(Path(args.zoo_dir) if args.zoo_dir else None).list()
    if not rows:
        print("(zoo is empty)")
        return
    for row in rows:
        print("{env_id}  tag={tag}  algo={algorithm}  {model_path}".format(**row))


def _cmd_zoo_latest(args: argparse.Namespace) -> None:
    from ..agents.model_zoo import ModelZoo

    zoo = ModelZoo(Path(args.zoo_dir) if args.zoo_dir else None)
    model = zoo.latest(args.env)
    if model is None:
        print(f"no zoo model for {args.env}")
        return
    meta = zoo.metadata(args.env)
    print(meta or f"loaded {args.env} model")


def _cmd_sysid(args: argparse.Namespace) -> None:
    from ..plants.sysid import identify_cavity

    env = make(args.env)
    report = identify_cavity(env.cavity, env.env_cfg.Ts, n_prbs=args.n_prbs)
    print_header("System identification of the simulated cavity plant")
    print(f"{'quantity':<24}{'identified':>18}{'truth':>18}{'rel. error':>14}")
    for name, ident, truth, err in (
        ("half bandwidth (rad/s)", report.half_bw_identified, report.half_bw_truth, report.half_bw_rel_error),
        ("detuning (rad/s)", report.detuning_identified, report.detuning_truth, report.detuning_rel_error),
        ("beam amplitude (V)", report.beam_amp_identified, report.beam_amp_truth, report.beam_rel_error),
    ):
        print(f"{name:<24}{ident:>18.4e}{truth:>18.4e}{err:>13.2%}")


def _cmd_calib(args: argparse.Namespace) -> None:
    import numpy as np

    from ..utils.calibration import calibrate_forward_coefficient, calibrate_virtual_probe

    env = make(args.env)
    plant = env.plant
    n = 600
    drive = np.full(n, 1.0e7, dtype=np.complex128)
    vc = np.zeros(n, dtype=np.complex128)
    vf_m = drive.copy()
    vr_m = np.zeros(n, dtype=np.complex128)
    plant.reset()
    for i in range(n):
        vc[i], vr_m[i] = plant.step(complex(drive[i]))
    m, n_coeff = calibrate_virtual_probe(vc, vf_m, vr_m)
    half_bw = env.cavity.half_bw
    detuning = env.env_cfg.randomize_detuning * 0.0 + env.cavity.detuning
    a = calibrate_forward_coefficient(vc, vf_m, 50, 500, half_bw, detuning, env.env_cfg.Ts)
    print_header("Virtual-probe / forward calibration (simulated plant)")
    print(
        "virtual probe:  "
        f"vc ~= ({m.real:.4e} + {m.imag:.4e}j) * vf_m + ({n_coeff.real:.4e} + {n_coeff.imag:.4e}j) * vr_m"
    )
    print(f"forward coeff:  vf = ({a.real:.4e} + {a.imag:.4e}j) * vf_m")


def _cmd_loop(args: argparse.Namespace) -> None:
    from llrflibs.rf_sim import cav_ss

    from ..configs.controller import PIControllerConfig
    from ..evaluation.loop import loop_margins, pi_controller_abcd

    env = make(args.env)
    cavity = env.cavity
    detuning = env.env_cfg.randomize_detuning * 0.0 + env.cavity.detuning
    ok, arf, brf, crf, drf, *_ = cav_ss(
        cavity.half_bw, detuning=detuning, beta=cavity.beta, passband_modes=cavity.passband_modes, plot=False
    )
    if not ok:
        raise SystemExit("cav_ss failed")
    # A representative PI inner loop matching the environment's default gains.
    pi_cfg = PIControllerConfig()
    ctrl = pi_controller_abcd(pi_cfg.kp, pi_cfg.ki, notch_conf=pi_cfg.notch_conf)
    report = loop_margins(
        (arf, brf, crf, drf), ctrl, Ts=None, delay_s=args.delay_us * 1e-6, plot_maxf=2.0e6
    )
    print_header("Loop stability margins (cavity + PI)")
    for key, value in report.as_dict().items():
        text = f"{value:.3f}" if value is not None else "inf"
        print(f"{key:<18}: {text}")


def _cmd_spectrum(args: argparse.Namespace) -> None:
    from ..controllers.baselines import make_baseline_policy
    from ..evaluation.spectral import tracking_error_spectrum

    env = make(args.env)
    policy = make_baseline_policy(env, "zero")
    bands = None
    if args.bands:
        bands = [tuple(float(x) for x in b.split("-")) for b in args.bands.split(",") if "-" in b]
    report = tracking_error_spectrum(lambda: make(args.env), policy, seed=0, bands=bands)
    print_header("Tracking-error spectrum (baseline policy)")
    print(
        "dominant error @ "
        f"{report.signal_freq_hz:.1f} Hz  ({report.signal_mag_db:.1f} dB, snr {report.snr_db:.1f} dB)"
    )
    print(f"overall error RMS  : {report.error_rms:.4e}")
    for band in report.bands:
        print(f"band {band.f_start:>8.1f}-{band.f_end:<8.1f} Hz : {band.rms:.4e}")


def _cmd_tune(args: argparse.Namespace) -> None:
    from dataclasses import replace

    from ..agents.optimizers import tune_scalar
    from ..controllers.baselines import make_pi_residual_policy
    from ..evaluation import evaluate_episode

    base = make(args.env).env_cfg

    def objective(kp: float) -> float:
        env = make(args.env, env_cfg=replace(
            base, randomize_detuning=0.0, microphonics_amp=0.0,
            measurement_noise_std=0.0, loop_delay_steps=0,
        ))
        policy = make_pi_residual_policy(env, kp=float(kp), ki=2.0e5)
        return float(evaluate_episode(env, policy, seed=0)["reward"])

    best, history = tune_scalar(objective, (0.01, 3.0), n_init=4, n_iters=args.n_iters, seed=args.seed)
    print_header(f"GP/Bayesian tuning of the PI residual gain ({args.env})")
    print(f"best kp        : {best:.4f}")
    print(f"best reward    : {max(history):,.2f}")
    print(f"reward history : {[f'{h:,.0f}' for h in history]}")


def _cmd_kpi(args: argparse.Namespace) -> None:
    from ..controllers.baselines import make_baseline_policy
    from ..evaluation.kpis import KPIS, kpi_report

    env = make(args.env)
    policy = make_baseline_policy(env, args.baseline)
    report = kpi_report(env, policy, seed=args.seed)
    print_header(f"KPI report (baseline={args.baseline})")
    print(f"{'KPI':<26}{'value':>14}{'threshold':>12}{'pass':>8}")
    for key, item in report.items():
        kpi = KPIS[key]
        passed = "PASS" if item["passed"] else "FAIL"
        print(f"{kpi.name:<26}{item['value']:>14.4f}{kpi.threshold:>12.4f}{passed:>8}")


def _cmd_diag(args: argparse.Namespace) -> None:
    from ..configs.controller import PIControllerConfig
    from ..evaluation.loop import cavity_plant_abcd, loop_margins, pi_controller_abcd
    from ..plants.sysid import identify_cavity

    env = make(args.env)
    report = identify_cavity(env.cavity, env.env_cfg.Ts, n_prbs=2048)
    plant = cavity_plant_abcd(env.cavity)
    pi_cfg = PIControllerConfig()
    margins = loop_margins(
        plant, pi_controller_abcd(pi_cfg.kp, pi_cfg.ki, notch_conf=pi_cfg.notch_conf), Ts=None, plot_maxf=2.0e6
    )
    print_header("Open-loop diagnostics (simulated plant)")
    print(
        "identified half bandwidth : "
        f"{report.half_bw_identified:,.4e} rad/s (true {report.half_bw_truth:,.4e}, "
        f"err {report.half_bw_rel_error:.2%})"
    )
    print(f"identified detuning       : {report.detuning_identified:,.4e} rad/s (err {report.detuning_rel_error:.2%})")
    print(f"loop S_max / T_max        : {margins.s_max_db:.2f} / {margins.t_max_db:.2f} dB")


def _cmd_demo(args: argparse.Namespace) -> None:
    requested_live = args.show or args.live if hasattr(args, "live") else args.show
    from ..envs.render import ensure_gui_backend

    if requested_live:
        gui = ensure_gui_backend()
        if not gui:
            print("warning: no interactive matplotlib backend available; saving a PNG only.")
    else:
        import matplotlib

        backend = matplotlib.get_backend().lower()
        if any(token in backend for token in ("tk", "qt", "gtk", "wx", "macosx")):
            matplotlib.use("Agg", force=True)
        gui = False

    import matplotlib.pyplot as plt

    from ..controllers.baselines import make_baseline_policy
    from ..envs.render import RenderableEnv

    env = RenderableEnv(
        make(args.env), render_mode="human" if gui else "rgb_array",
        title=f"{args.env} — baseline={args.policy}",
    )
    policy = make_baseline_policy(env, args.policy)
    render_mode = "human" if gui else "rgb_array"
    obs, _ = env.reset(seed=args.seed)
    total_reward = 0.0
    steps = 0
    for i in range(args.steps):
        obs, reward, terminated, truncated, _info = env.step(policy(obs))
        total_reward += float(reward)
        steps += 1
        if i % max(1, args.frame) == 0 or terminated or truncated:
            env.render(render_mode)  # human mode streams live on GUI backends
        if terminated or truncated:
            break

    out = Path(args.out) if args.out else Path("gallery") / f"demo_{args.env}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    env.fig.savefig(out, dpi=110)
    print(f"demo: {steps} steps, reward {total_reward:,.2f} -> {out}")
    if gui:
        plt.show(block=True)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).
    """
    parser = _parser()
    args = parser.parse_args(argv)
    # Switch to an interactive matplotlib backend before any pyplot import
    # whenever the user asked for live rendering (demo / train --render).
    if getattr(args, "render", False) or getattr(args, "live", False) or getattr(args, "show", False):
        from ..envs.render import ensure_gui_backend

        ensure_gui_backend()
    args.func(args)


if __name__ == "__main__":
    main()
