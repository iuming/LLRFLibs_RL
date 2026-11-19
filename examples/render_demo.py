###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
CartPole-style interactive / saved visualization of an LLRF environment.

Self-contained one-shot demo: it fixes the repository ``src`` on ``sys.path``
and the OpenMP duplicated-runtime flag, so it runs without installation:

    python examples/render_demo.py --env CavityVoltage-v0 --policy pi
    python examples/render_demo.py --env DualLoop-v0 --policy pi --show --steps 200
    python examples/render_demo.py --env BeamLoading-v0 --policy ff --out gallery/beam.png

Headless machines save a PNG; GUI terminals get a live window with ``--show``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# --- repository bootstrap (works without pip install) -------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize an LLRF environment.")
    parser.add_argument("--env", default="CavityVoltage-v0", help="registered environment id")
    parser.add_argument("--policy", default="zero", choices=["zero", "ff", "pi", "adrc"],
                        help="classical baseline policy to drive the episode")
    parser.add_argument("--steps", type=int, default=400, help="number of samples to run")
    parser.add_argument("--frame", type=int, default=4, help="render every N-th step")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show", action="store_true", help="show a live/interactive window")
    parser.add_argument("--live", action="store_true",
                        help="render in real time, frame by frame (human mode on GUI)")
    parser.add_argument("--out", default=None, help="PNG path (default gallery/demo_<env>.png)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    import matplotlib

    backend = matplotlib.get_backend().lower()
    gui = any(token in backend for token in ("tk", "qt", "gtk", "wx", "macosx"))
    # Headless-safe default: render into a figure but never open a window.
    if not (args.show or args.live) and gui:
        matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt

    import llrflibs_rl as rl
    from llrflibs_rl.controllers.baselines import make_baseline_policy
    from llrflibs_rl.envs.render import RenderableEnv

    render_mode = "human" if (args.show or args.live) else "rgb_array"
    env = RenderableEnv(
        rl.make(args.env), render_mode=render_mode, title=f"{args.env} — baseline={args.policy}"
    )
    policy = make_baseline_policy(env, args.policy)
    obs, _ = env.reset(seed=args.seed)

    total_reward = 0.0
    steps = 0
    for i in range(args.steps):
        obs, reward, terminated, truncated, _info = env.step(policy(obs))
        total_reward += float(reward)
        steps += 1
        if i % max(1, args.frame) == 0 or terminated or truncated:
            env.render(render_mode)
        if terminated or truncated:
            break

    out = Path(args.out) if args.out else Path("gallery") / f"demo_{args.env}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    env.fig.savefig(out, dpi=110)
    print(f"[demo] {steps} steps, reward {total_reward:,.2f} -> {out}")
    if (args.show or args.live) and gui:
        plt.show(block=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
