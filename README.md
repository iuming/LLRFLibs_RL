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

A CartPole-style renderer is available on any registered environment:

```bash
# live window, frame by frame (GUI terminal) — watch the episode in real time
python -m llrflibs_rl.scripts.run demo --env CavityVoltage-v0 --policy pi --show

# headless: save a figure (any policy: zero / ff / pi / adrc)
python -m llrflibs_rl.scripts.run demo --env CavityVoltage-v0 --policy pi --out gallery/demo.png

# watch the agent learn: live-render the growing policy during training
python -m llrflibs_rl.scripts.run train --env CavityVoltage-v0 --render --render-freq 1024
```

Or use the self-contained example (works without installing the package):

```bash
D:\Software\Anaconda\envs\rl\python.exe examples\render_demo.py --live      # watch live
D:\Software\Anaconda\envs\rl\python.exe examples\render_demo.py              # save PNG
```

In Python, wrap any environment with the renderer:

```python
from llrflibs_rl.envs.render import RenderableEnv
import llrflibs_rl as rl

env = RenderableEnv(rl.make("CavityVoltage-v0"), render_mode="human")
obs, _ = env.reset(seed=0)
for _ in range(400):
    obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
    env.render()  # human mode streams the figure live on GUI backends
    if terminated or truncated:
        break
```

#### What the four console panels mean

The console view (`layout="console"`, dark EPICS-style) shows:
| panel | what it plots | how to read it |
|---|---|---|
| **Amplitude \|Vc\|** (left-top) | cavity voltage magnitude vs time, with the setpoint (dashed) | cyan curve must ride the dashed reference; gap = tracking error |
| **Phase** (right-top) | cavity phase in degrees vs time, reference dashed | flattop should be flat at the reference phase |
| **Phasor plane** (left-bottom) | I/Q trajectory of the cavity state; blue dot = current state, star = setpoint | like a car on a road: the dot drives along the trajectory toward the star |
| **Drive / Reflected vs Detuning** (right-bottom) | \|drive\| and \|reflected\| (left axis), detuning in Hz (right axis, purple) | loading, reflected-power margin and detuning dynamics at a glance |

The **left PV panel** is the numeric readout column (|Vc|, phase, forward,
reflected, detuning, beam, tracking error, reward) — engineers who are used to
CSS/Phoebus will recognize it.

The same renderer also offers a mech2-style physics view that is plotted from
the *reinforcement-learning environment itself* (`--layout mech`) — three
stacked panels |Vc| (MV) / phase (deg) / detuning (Hz), the latter showing the
real time-varying (microphonics / Lorentz) detuning:

```bash
python run.py demo --env CavityVoltage-v0 --policy pi --layout mech --out gallery/mech_rl.png
D:/Software/Anaconda/envs/rl/python.exe examples/render_demo.py --layout mech --out gallery/mech_rl.png
```

#### LLRFLibsPy-style physics plots (`mech2` port)

Prefer the plain "three-stacked-panels" view of `example_sim_cavity_mech2.py`?
A faithful port is included:

```bash
# headless: saves gallery/mech_scan.png (|Vc| MV / phase deg / detuning Hz)
D:/Software/Anaconda/envs/rl/python.exe examples/sim_cavity_mech_scan.py --out gallery/mech_scan.png
# interactive window
D:/Software/Anaconda/envs/rl/python.exe examples/sim_cavity_mech_scan.py --show
```

It simulates the full open-loop RF chain (RF source -> I/Q modulator ->
amplifier -> cavity) with beam current, microphonics and the multi-mode
mechanical (Lorentz) detuning of the canonical 1.3 GHz SC cavity, and prints
the flattop voltage and the detuning statistics.

#### The full LLRF closed-loop environment (`LLRFLoop-v0`)

`LLRFLoop-v0` simulates the closed loop of the LLRF block diagram
(`llrf_control_loop_standalone.tex`): the **fast FPGA loop** — PI + feedforward
+ RL residual -> DAC -> I/Q modulator -> SSA -> cavity, closed through the
pick-up / mixer / LPF / ADC / I-Q-detection chain — together with the **slow
tuner loop** that compensates the cavity detuning.

The RL action is exactly the two-channel picture of the schematic:

| action dim | meaning |
|---|---|
| `0` `tuner` | normalized slow mechanical-tuner command (resonance / detuning channel) |
| `1`, `2` `RF_I`, `RF_Q` | normalized I/Q residual of the FPGA RF drive signal (added on top of the inner complex-PI feedback and analytic feedforward) |

```bash
python run.py demo --env LLRFLoop-v0 --policy pi --layout console   # watch both loops
python run.py train --env LLRFLoop-v0 --render-http 8192 --timesteps 200000
```

The other registered environments map onto the same block diagram: the
`CavityVoltage` / `PIAssist` / `ADRCAssist` tasks are the inner FPGA loop alone,
`FrequencyControl` / `DualLoop` are the tuner (slow) loop alone, and
`VectorSum*` are the multi-cavity version.

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
