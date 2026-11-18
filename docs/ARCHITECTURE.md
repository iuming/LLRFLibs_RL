# LLRFLibs_RL — Architecture

> Copyright (c) 2026, National Synchrotron Radiation Laboratory, USTC.
> Author: Ming Liu.

This document describes the package structure of `llrflibs_rl`, the
reinforcement-learning sub-library of **LLRFLibsPy** for LLRF (low-level RF)
control of particle-accelerator cavities.

## 1. Positioning

`llrflibs_rl` is a **standalone package** that depends one-way on the parent
library `llrflibs`:

```
llrflibs                         (physics & classical control)
  rf_sim      cavity / beam / mechanical-mode models
  rf_control  analytic feedforward, PI/notch, ADRC, ILC
            ▲  consumed only
            │
llrflibs_rl                      (RL research & engineering)
  configs     strongly-typed configuration
  plants      physical MDP "world"
  tasks       reference / observations / rewards / events
  controllers classical baselines (reusable without RL)
  envs        Gymnasium environments (registered)
  agents      PPO/SAC training wrappers, presets
  evaluation  metrics / robustness / reports
  scenario    one-stop train+plot+artifact flow (used by CLI & examples)
```

The parent library keeps its long-lived physics/algorithm code; the RL layer
can evolve independently (new algorithms, new experiment machinery, heavier
dependencies such as `torch` / `stable-baselines3` / `gymnasium`).

## 2. Control-level map

The task zoo follows the natural hierarchy of LLRF control:

| Level | Control problem | Environment |
|---|---|---|
| L0 | Single-cavity fast RF voltage/phase (residual and inner-loop assist) | `CavityVoltage-v0`, `PIAssist-v0`, `ADRCAssist-v0` |
| L1 | Resonance / slow tuner-piezo compensation | `FrequencyControl-v0` |
| L2 | Hierarchical fast + slow cooperation | `DualLoop-v0` |
| L3 | Multi-cavity vector-sum module control | `VectorSum-v0` |

Each environment = **plant + task + optional inner controller + reward**,
assembled in the `envs/` layer from reusable components:

```
envs/voltage.py      CavityVoltageEnv       (L0, residual)
                     PIAssistEnv            (L0, PI inner loop + RL assist)
                     ADRCAssistEnv          (L0, ADRC inner loop + RL assist)
envs/resonance.py    FrequencyEnv           (L1, RL tuner, constant drive)
                     DualLoopEnv            (L2, PI fast loop + RL tuner)
envs/vectormodule.py VectorSumEnv           (L3, shared source)
```

## 3. Package layout

```
pyproject.toml
README.md
src/llrflibs_rl/
  __init__.py        public API, registers environments on import
  registry.py        register / make / list / recommended_timesteps
  scenario.py        run_experiment(...) + plotting + artifact writing
  configs/           cavity.py · env.py · task.py · controller.py · agent.py
  plants/            cavity_voltage.py · resonance.py · vectormodule.py · disturbance.py
  tasks/             reference.py · observations.py · rewards.py · events.py
  controllers/       feedback.py (PI/ADRC) · feedforward.py (FF/ILC) · resonance.py (actuator)
  envs/              core.py · voltage.py · resonance.py · vectormodule.py · wrappers.py
  agents/            trainer.py (train/train_ppo/train_sac) · presets.py
  evaluation/        metrics.py · reporting.py
  scripts/           run.py (CLI: list/train/eval/benchmark)
  utils/             seeding.py · artifacts.py · logging.py
tests/               pytest suite (all layers)
examples/            quickstart.py · train_and_eval.py
experiments/         *.yaml — reproducible experiment specifications
docs/
```

## 4. Key design decisions

### 4.1 Registry instead of scattered factories

Environments are registered once in `envs/__init__.py` and obtained through
`llrflibs_rl.make(env_id)`, mirroring the gymnasium ecosystem.  Each id carries
recommended default construction arguments (nominal cavity + presets), so
`make("DualLoop-v0")` returns a runnable environment.

```python
rl.make("VectorSum-v0")
rl.list_env_ids()
rl.recommended_timesteps("DualLoop-v0")   # -> 30000
```

### 4.2 Strongly-typed, serializable configuration

Every knobs lives in a dataclass that validates eagerly and serializes to
plain JSON (`to_dict` / `from_dict`).  The PRNG is seeded centrally
(`utils/seeding.set_global_seed`) and SB3 is additionally passed the same
seed, so runs are reproducible.

### 4.3 Physics stays in  `plants`, learning signal in `tasks`

* `plants` expose `reset -> step(drive, beam, detuning)` on top of
  `llrflibs.rf_sim`; they know physics but nothing about learning.
* `tasks` translate physics into the MDP: pulse references, normalized
  observations, and the reward factory with smooth region weighting.
* `controllers` provide the classical baseline inner loops (PI, ADRC, tuner,
  ILC) and are unit-tested without any RL.

This separation is what makes a term like the "drive-effort penalty" a
one-line configuration change instead of a physics edit.

### 4.4 Artifacts per run

`scenario.run_experiment(...)` and the CLI write a timestamped directory under
`runs/` containing `config.json`, `metrics.json`, the trained model and a
comparison figure — a complete record of one experiment.

## 5. Guidelines for contributors

* Keep the dependency direction strict: `plants -> llrflibs`,
  `tasks/envs -> plants`, `agents -> envs`, nothing back.
* All code is English with header (copyright, author) and NumPy-style
  function docstrings.
* New environments must be registered and covered by a Gymnasium-contract test
  (`tests/test_envs.py`).
* `python -m pytest` must stay green; the suite is fast (no long training runs).
* Formatting/linting: [ruff](https://github.com/astral-sh/ruff) (see
  `pyproject.toml`).
