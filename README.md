# LLRFLibs_RL

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#)
[![BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](#)

**Reinforcement-learning toolkit for LLRF (low-level radio-frequency) control of
particle-accelerator RF cavities.**

`LLRFLibs_RL` is the RL sub-library of **LLRFLibsPy** (National Synchrotron
Radiation Laboratory, USTC). It builds a modular research stack on top of the
physics and classical-control primitives provided by
[`llrflibs`](https://github.com/iuming/LLRFLibsPy):

- `llrflibs.rf_sim`  — cavity / beam / mechanical-mode numerical models
- `llrflibs.rf_control` — analytical feedforward, PI/notch, ADRC, ILC
- `llrflibs.rf_sysid`  — system identification and disturbance estimation

This package adds the **RL layer**: physical plants, MDP tasks, baseline
controllers, Gymnasium environments, agent training and physics-aware
evaluation. It is designed around the natural *hierarchy of LLRF control*:

| Level | Control problem | Environments |
|---|---|---|
| L0 | Single-cavity fast RF voltage/phase (residual & inner-loop assist) | `CavityVoltage-v0`, `PIAssist-v0`, `ADRCAssist-v0`, `GainScheduling-v0`* |
| L1 | Resonance / slow tuner-piezo compensation | `FrequencyControl-v0` |
| L2 | Hierarchical fast RF + slow tuner cooperation | `DualLoop-v0` |
| L3 | Multi-cavity vector-sum module control | `VectorSum-v0`, `VectorSumTuner-v0`* |
| L4 | Beam loading & pulse-to-pulse adaptation | `BeamLoading-v0`*, `AdaptivePulse-v0`* |

\* `GainScheduling-v0`: RL schedules the inner-loop PI gains instead of adding
a residual. `VectorSumTuner-v0`: a shared RF source plus one slow tuner per
cavity. `BeamLoading-v0`: the beam steps on during the flattop while the
feedforward is built beam-free. `AdaptivePulse-v0`: each step is one full
pulse; RL tunes per-segment feedforward corrections while the detuning drifts.

## Installation

```bash
# from the repository root
pip install -e .
```

The parent library must be installed first:

```bash
pip install -e ../LLRFLibsPy        # or pip install llrflibs>=1.0.2
```

## Quick start

```python
import llrflibs_rl as rl

# Create an environment from the registry with its recommended preset config.
env = rl.make("DualLoop-v0")
obs, info = env.reset(seed=0)
print(env.observation_space, env.action_space)

# Train a baseline-free PPO controller in one call.
from llrflibs_rl.agents import AgentConfig, train
model = train(env, AgentConfig(timesteps=30_000))
```

Or from the command line:

```bash
python -m llrflibs_rl.scripts.run list
python -m llrflibs_rl.scripts.run train --env DualLoop-v0 --timesteps 30000
python -m llrflibs_rl.scripts.run benchmark --seeds 4       # full report
python -m llrflibs_rl.scripts.run verify                    # contract-check all envs
python -m llrflibs_rl.scripts.run zoo list                  # trained-model zoo
python -m llrflibs_rl.scripts.run gallery                   # render figures from zoo models
```

Short-horizon resolution variants of the core environments are registered as
`...-v1` ids (e.g. `CavityVoltage-v1`), and a curated
`llrflibs_rl.api` facade bundles the everyday functions.

See `examples/` for scenario scripts, `experiments/` for reproducible
experiment configurations, and `docs/ARCHITECTURE.md` for the package design.
The consolidated benchmark results (final numbers, the PPO/SAC/TD3 study and
the level-by-level reading) live in **`docs/BENCHMARKS.md`**.

## Package layout

```
src/llrflibs_rl/
├── configs/       strongly-typed configuration (cavity / env / task / controller / agent)
├── plants/        physical MDP "world": cavity voltage, resonance, vector module
├── tasks/         reference waveforms, observations, rewards, event handling
├── controllers/   classical baseline controllers (FF, PI, ADRC, tuner, ILC)
├── envs/          Gymnasium environments + wrappers
├── agents/        training / evaluation loops and per-task presets
├── evaluation/    physics-aware metrics, robustness sweeps, report generation
├── registry.py    environment registry (gym-style register / make)
└── utils/         seeding, run artifacts, logging
```

## Examples

After installing, run any scenario:

```bash
python examples/train_and_eval.py --env PIAssist-v0 --timesteps 40000
python examples/train_and_eval.py --env DualLoop-v0   --timesteps 30000
python examples/train_and_eval.py --env VectorSum-v0  --timesteps 60000
python examples/train_and_eval.py --env CavityVoltage-v0
```

### Demo (visualization)

A CartPole-style renderer is available on any registered environment. Watch
the episode live (GUI terminal) or write a PNG on headless machines:

```bash
# save a figure (any policy: zero / ff / pi / adrc)
python -m llrflibs_rl.scripts.run demo --env CavityVoltage-v0 --policy pi --out gallery/demo.png

# interactive window on a GUI terminal
python -m llrflibs_rl.scripts.run demo --env DualLoop-v0 --policy pi --show
```

In Python, wrap any environment with the renderer:

```python
from llrflibs_rl.envs.render import RenderableEnv
import llrflibs_rl as rl

env = RenderableEnv(rl.make("CavityVoltage-v0"), render_mode="human")
obs, _ = env.reset(seed=0)
for _ in range(400):
    obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
    env.render()
    if terminated or truncated:
        break
```

## Classical baselines

Every training run compares RL against a classical controller selected with
`--baseline` (default `zero`):

| baseline | meaning |
|---|---|
| `zero` | no compensation (analytic feedforward / inner loop only) |
| `ff`    | nominal analytical feedforward only (same as `zero` here) |
| `pi`    | complex-PI residual (voltage/vector-sum) or detuning-PI slow loop (resonance) |
| `adrc`  | ADRC-like PD+integral residual (voltage/vector-sum) or detuning-PI (resonance) |

For the assist environments the classical controller is already the inner
loop, so `pi` / `adrc` reduce to "PI-only" / "ADRC-only".

```bash
python -m llrflibs_rl.scripts.run train --env CavityVoltage-v0 --baseline pi --timesteps 15000
python -m llrflibs_rl.scripts.run train --env DualLoop-v0 --baseline pi  --timesteps 30000
python -m llrflibs_rl.scripts.run eval  --env CavityVoltage-v0 --baseline adrc --model runs/<run>/model.zip
```

Each run writes a self-contained artifact directory under `runs/`
(`config.json`, `metrics.json`, the trained model, the training-reward curve
`training_curve.json` and a comparison figure with **tracking-error panels
and the RL convergence curve**).

## Benchmark

```bash
python -m llrflibs_rl.scripts.run benchmark          # recommended budgets, all environments
python -m llrflibs_rl.scripts.run benchmark --quick  # CI-style sanity check
```

Writes `benchmarks/benchmark.md` / `benchmark.json` with reward and error
metrics per environment. Each environment has an **algorithm preset** picked
by the cross-environment study (`experiments/algorithm_comparison.py`):
PPO by default, off-policy SAC for `DualLoop-v0` / `VectorSumTuner-v0`, and
TD3 for `GainScheduling-v0`. Run `python -u experiments/algorithm_comparison.py`
to reproduce the algorithm study.

### Known limitations

`VectorSumTuner-v0` combines a saturated shared-drive residual with three slow
tuners; the default on-policy PPO collapses into drive saturation on longer
budgets. A stable off-policy **SAC** preset (`experiments/vector_sum_tuner_study.py`)
solves it: at 20k steps SAC scores -231 ± 16 vs a -268 ± 114 baseline.
Observation normalization (`normalize_obs`) did not help on this task.

## Testing

```bash
pytest
```

## License

BSD-3-Clause. Copyright (c) 2026, National Synchrotron Radiation Laboratory,
University of Science and Technology of China. Author: Ming Liu.
