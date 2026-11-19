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
- **Science-toolkit layer** integrating the remaining ``llrflibs`` physics
  modules into the RL package:
  - `plants/noise` — PSD-driven microphonics (`MicrophonicsProfile`,
    `typical_cryomodule_microphonics`) wired into the voltage environments via
    `EnvConfig.microphonics_psd` (replaces the single-sine model).
  - `controllers/notch` — state-space notch filter + `notch_conf_for_frequencies`
    helper for `PIControllerConfig.notch_conf`.
  - `controllers/ilc` — adaptive feedforward (`AdaptiveFeedforward`) and optimal
    frequency-domain ILC (`ILCFeedforward`, SVD/LS response inversion) as
    classical baselines for the pulse-to-pulse environments.
  - `plants/sysid` — PRBS/ETFE transfer-function estimation, decay-based
    half-bw/detuning identification and the ADRC beam-loading observer
    (`identify_cavity` pipeline; `sysid` CLI).
  - `utils/calibration` — virtual probe, forward coefficient, DAC offset,
    I/Q modulator, vector-sum and system gain/phase calibration (`calib` CLI).
  - `plants/measurement` — demodulators, I/Q<->A/P conversion and a
    delay+noise `MeasurementChain` for front-end realism.
  - `evaluation/loop` — loop-stability margins (gain/phase margin, S_max/T_max)
    of a cavity + PI loop (`loop` CLI).
  - `evaluation/spectral` — tracking-error PSD and band-RMS analysis
    (`spectrum` CLI).
- **Survey-driven features** (benchmarking/classical baselines and deployment):
  - `agents/optimizers` — dependency-free Gaussian-process Bayesian optimizer
    (RBF + UCB) for scalar setpoint / gain auto-tuning, the classical-optimization
    counterpart to RL; `tune` CLI demos PI-gain auto-tuning on an environment.
  - `utils/hil` — channel layer (`Channel` / `SimulatedChannel` / optional
    `EPICSChannel` via `pyepics`) plus `emulate_policy_loop` policy deployment.
  - `sensors/quench` — reflected-amplitude quench-onset detector wired into the
    voltage environments through `EnvConfig.quench_detect`.
  - `evaluation/kpis` — KPI catalog (0.01 % amplitude RMS / 0.01 deg phase RMS)
    with pass/fail `kpi_report`; `kpi` CLI.
  - `controllers/feedforward` — `beam_keepup_correction` / `AnalyticBeamFeedforward`
    (beam phase keep-up feedforward).
  - `plants/measurement` — firmware-style `decimation` in `MeasurementChain`.
  - `diag` CLI — open-loop identification + loop-margin diagnostics.
- **Visualization** — `envs.render.RenderableEnv`: a CartPole-style render
  wrapper for the LLRF environments with ``render("human")`` / ``render("rgb_array")``
  modes (live multi-panel figure: amplitude / phase tracking, I/Q phasor and
  drive/detuning telemetry); `demo` CLI runs an episode with any baseline and
  saves a PNG (interactive ``--show`` on GUI terminals).
- **Training-time live rendering** — `render("human")` now streams the figure
  in real time on GUI backends (per-frame `plt.pause`); `agents.render.LiveRenderCallback`
  periodically plays the policy being trained in a window (`train --render
  --render-freq N`); `examples/render_demo.py --live` watches an episode frame by frame.
- **Episode-horizon semantics (`EnvConfig.mode`)** — ``"pulse"`` (default):
  `spec.max_episode_steps == episode_len` (for the pulse envs this is the
  N-*pulse-period* adaptation window, one step per pulse); ``"cw"``
  (continuous-wave): `spec.max_episode_steps is None` (infinite) and the loop
  is never truncated at the horizon — it only ends on a protective trip,
  mirroring real CW LLRF operation.  `make()` derives the spec per-instance
  (the shared registry entry is no longer mutated).
- **Operator-console render layout** — `RenderableEnv` gains an EPICS
  CSS/Phoebus-style dashboard (`layout="console"`, default): live numeric
  PV panel (|Vc|, phase, forward/reflected, detuning, beam, drive, tracking
  error with alarm colours), waveform trend strips, and a *moving* state
  marker that drives the phasor trajectory; the old 4-panel layout is kept as
  `layout="simple"`.
- **Nominal SC-cavity preset aligned with LLRFLibsPy** — `nominal_sc_cavity`
  now uses the canonical half bandwidth ``half_bw = pi * f0 / QL`` (1.3 GHz,
  QL = 3e6 -> ~216.7 Hz; previously half of it) and a mild static detuning of
  100 Hz (was 500 Hz), exactly like ``example/example_sp_ff.py``.  The
  constant-filling feedforward now stays comfortably within the drive budget
  and the zero-disturbance, zero-residual flattop tracking error drops from
  ~0.28 % to ~0.03 %.  `nominal_nc_cavity` gets the same canonical
  bandwidth fix.

### Changed
- `EnvConfig` gained the additive `microphonics_psd` field (default `None`,
  behaviour unchanged).
- Deprecated `typing.Dict/List/Optional/Tuple` annotations modernized to
  `dict/list/... | None` / `tuple` across the package (ruff UP rules).

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
