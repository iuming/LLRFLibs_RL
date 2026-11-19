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
    parser.add_argument("--frame", type=int, default=2, help="render every N-th step")
    parser.add_argument("--fps", type=float, default=30.0, help="refresh rate of the live window")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show", action="store_true", help="show a live/interactive window")
    parser.add_argument("--live", action="store_true",
                        help="render in real time, frame by frame (human mode on GUI)")
    parser.add_argument("--web", type=int, nargs="?", const=0, default=None,
                        help="serve a browser live view on this port (0 = auto)")
    parser.add_argument("--out", default=None, help="PNG path (default gallery/demo_<env>.png)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    requested_live = args.show or args.live

    import matplotlib

    from llrflibs_rl.envs.render import ensure_gui_backend

    if requested_live:
        # Try to get an interactive backend (TkAgg -> Qt -> wx) before any
        # pyplot import.  On truly headless systems this falls back to PNG.
        gui = ensure_gui_backend()
        print(f"[demo] matplotlib backend = {matplotlib.get_backend()}  live_window = {gui}")
        if not gui:
            print("[demo] warning: no interactive matplotlib backend available; saving a PNG only.")
        else:
            print("[demo] opening live window — watch the step counter in the title; close the window to finish.")
    else:
        # Headless-safe default for non-live runs.
        backend = matplotlib.get_backend().lower()
        if any(token in backend for token in ("tk", "qt", "gtk", "wx", "macosx")):
            matplotlib.use("Agg", force=True)
        gui = False

    import matplotlib.pyplot as plt

    import llrflibs_rl as rl
    from llrflibs_rl.controllers.baselines import make_baseline_policy
    from llrflibs_rl.envs.render import RenderableEnv
    from llrflibs_rl.utils.live_http import HttpStreamer

    streamer = None
    if args.web is not None:
        streamer = HttpStreamer(port=args.web).start()
        print(f"[demo] browser live view open at {streamer.url}  (Ctrl-C to stop)")

    render_mode = "human" if gui else "rgb_array"
    env = RenderableEnv(
        rl.make(args.env), render_mode=render_mode,
        title=f"{args.env} — baseline={args.policy}",
        pause_sec=1.0 / max(1.0, args.fps),
        streamer=streamer,
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
    if args.web is not None:
        print(f"[demo] serving {args.env} at {streamer.url} — press Enter or Ctrl-C to stop")
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass
    elif gui:
        plt.show(block=True)  # keep the window open until the user closes it
    if streamer is not None:
        streamer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
