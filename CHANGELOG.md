# Changelog

All notable changes to `llrflibs-rl` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Gymnasium parity: `env.spec` / `env.unwrapped` / `env.reward_range` on every
  environment, `make(..., wrappers=[...])`, `max_episode_steps` registry
  metadata, and the official `gymnasium.utils.env_checker.check_env` in the
  `verify` contract checker.
- SB3 training stack: best-model saving (`EvalCallback`), periodic
  `CheckpointCallback`, `n_envs` vectorization, `policy` / `policy_kwargs`
  (including `MlpLstmPolicy`), and `VecNormalize` statistics persistence in
  the model zoo.
- `evaluation.cross_check_with_sb3` cross-validating the package metrics
  against `stable_baselines3.common.evaluation.evaluate_policy`.
- GitHub Actions CI workflow (`pytest` + `verify` + `ruff`), console script
  entry points, and this changelog.

## [0.3.0] — 2026-11-19

### Added
- Engineering toolkit: `llrflibs_rl.api` facade, `utils/verifier` contract
  checker, `utils/plotting` reusable comparison figures, `agents.ModelZoo`
  and the CLI subcommands `verify` / `zoo` / `gallery`.
- Physics modeling: `plants/beam_schedule`, `plants/lorentz` (dynamic
  `|vc|^2` Lorentz detuning), a disturbance factory
  (`sinusoidal` / `step` / `sweep`), and the short-horizon `-v1` environment
  variants (6 more registered ids).
- Fixed a first-frame stale-feedback bug in the PI/ADRC/gain-scheduling
  environment resets (found by the new verifier).

## [0.2.0] — 2026-11-18

### Added
- Restructured `llrflibs-rl` as a standalone src-layout package: registry,
  configs / plants / tasks / controllers / envs / agents / evaluation /
  scenario / scripts layers, pytest suite (94 tests), examples, experiments
  YAML, CLI (`list` / `train` / `eval` / `benchmark`).
- Level-L4 scenarios: `BeamLoading-v0`, `AdaptivePulse-v0`;
  `VectorSumTuner-v0`; `GainScheduling-v0`; classical baselines
  (`zero` / `ff` / `pi` / `adrc`); PPO / SAC / TD3 presets with a
  cross-environment algorithm study; `docs/BENCHMARKS.md`.

## [0.1.0] — 2026-11-17

### Added
- Initial monolith: `llrflibs_rl/` single-package environments and examples
  for LLRF cavity voltage, resonance, feedback, vector-sum and dual-loop RL.
