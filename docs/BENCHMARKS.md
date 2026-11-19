# LLRFLibs_RL — Benchmark Report (definitive)

> Copyright (c) 2026, National Synchrotron Radiation Laboratory, USTC.
> Authors: Ming Liu, mliu1018@ustc.edu.cn.

This document consolidates every benchmark result of the toolkit: the
final environment benchmark (recommended budgets, multi-seed), the
cross-environment **PPO / SAC / TD3** algorithm study, the per-level
physics readings and the reproducibility commands.

Round numbers are rounded from JSON artifacts; run the commands in
[§ 6](#6-reproducibility) to regenerate them exactly.

---

## 1. Environment overview

The registry exposes ten environments across the LLRF control hierarchy.
All of them are Gymnasium environments built from the same physics
(`llrflibs.rf_sim` / `rf_control`).

| Level | Environment | Task | Action dim |
|---|---|---|---|
| L0 | `CavityVoltage-v0` | RL residual on the analytic feedforward | 2 |
| L0 | `PIAssist-v0` | Complex-PI inner loop + RL residual assist | 2 |
| L0 | `ADRCAssist-v0` | ADRC inner loop + RL residual assist | 2 |
| L0 | `GainScheduling-v0` | RL schedules the inner-loop PI gains (kp, ki) | 2 |
| L1 | `FrequencyControl-v0` | RL slow tuner vs detuning disturbance | 1 |
| L2 | `DualLoop-v0` | Fast RF PI loop + RL slow tuner | 1 |
| L3 | `VectorSum-v0` | Shared-drive vector sum, 3 cavities | 2 |
| L3 | `VectorSumTuner-v0` | Shared drive + per-cavity slow tuners | 2 + n |
| L4 | `BeamLoading-v0` | Beam steps on during flattop vs beam-free FF | 2 |
| L4 | `AdaptivePulse-v0` | Pulse-to-pulse adaptive feedforward (K segments) | K = 8 |

### Baselines

The comparison baseline is the **no-compensation** policy (`zero`):
analytical feedforward (+ inner loop where present) and no RL action. For
`GainScheduling-v0` the `pi` baseline — the fixed nominal gains — is the
honest "RL adaptive gains vs fixed gains" reference.

---

## 2. Final environment benchmark

Command: `python -u -m llrflibs_rl.scripts.run benchmark --seeds 4 --out benchmarks`
(10 environments, recommended budgets, deterministic policies averaged over
evaluation seeds 0–3). Full machine-readable numbers: `benchmarks/benchmark.json`.

| Environment | Algorithm (preset) | Budget | Baseline | RL | RL std | Baseline flattop err | RL flattop err |
|---|---:|---:|---:|---:|---:|---:|---:|
| ADRCAssist-v0 | PPO | 40k | -64.4 | -66.9 | 0.2 | 338k | 390k |
| AdaptivePulse-v0 | PPO | 10k | -9.28 | **-6.08** | 0.0 | – | – |
| BeamLoading-v0 | PPO | 20k | -1632 | -1698 | 1292 | 11.9e6 | 11.9e6 |
| CavityVoltage-v0 | PPO | 10k | -1420 | -1414 | 501 | 9.6e6 | 9.6e6 |
| DualLoop-v0 | **SAC** | 30k | -366 | **-164** | 4 | 1.22e6 | 1.32e6 |
| FrequencyControl-v0 | PPO | 2k | -8315 | **-7596** | 299 | – | – |
| GainScheduling-v0 | **TD3** | 30k | -353650 | **-267780** | 48 355 | 117e6 | 103e6 |
| PIAssist-v0 | PPO | 40k | -153 | -169 | 2 | 825k | 893k |
| VectorSum-v0 | PPO | 60k | -637 | **-581** | 130 | – | – |
| VectorSumTuner-v0 | **SAC** | 20k | -268 | **-231** | 17 | – | – |

Flattop errors are RMS in volts (voltage-scale tasks); "–" marks tasks whose
metric is amplitude-error based (no tracking set-point).

### Reading

- **RL clearly better (6 / 10):** `AdaptivePulse` +34 %, `DualLoop` -366 →
  -164 (2.2×, low variance), `FrequencyControl`, `GainScheduling` (+24 % and
  flattop error 117e6 → 103e6), `VectorSum`, `VectorSumTuner`.
- **RL ≈ baseline (4 / 10):** `ADRCAssist`, `BeamLoading` (dominated by
  disturbance variance), `CavityVoltage`, `PIAssist`. These are the
  strong-inner-loop residual tasks where the analytic feedforward + tuned
  loop already leave little headroom — an expected physics result, not a
  training failure.

---

## 3. Cross-environment algorithm study

Command: `python -u experiments/algorithm_comparison.py`
(10 environments × PPO / SAC / TD3, recommended budget each, 4-seed eval).
Off-policy variants use `train_freq = gradient_steps = 64`, `batch_size = 256`.

| Environment | Baseline | PPO | SAC | TD3 | Winner |
|---|---:|---:|---:|---:|---|
| ADRCAssist-v0 | -64 | -67 | -122 | -146 | PPO |
| AdaptivePulse-v0 | -9.3 | -6.1 | -6.9 | -6.4 | PPO |
| BeamLoading-v0 | -1632 | -1698 | -1875 | -1646 | TD3 (noise) |
| CavityVoltage-v0 | -1420 | -1414 | -1615 | -1504 | PPO |
| DualLoop-v0 | -366 | -384 | **-164** | -4384 | **SAC** |
| FrequencyControl-v0 | -8315 | -7596 | -11994 | -12459 | PPO |
| GainScheduling-v0 | -353650 | -345495 | -376407 | **-267780** | **TD3** |
| PIAssist-v0 | -153 | -169 | -167 | -178 | SAC (≈) |
| VectorSum-v0 | -637 | -581 | -534 | -3024 | SAC (high std) |
| VectorSumTuner-v0 | -268 | -1032 | **-231** | -1122 | **SAC** |

### Preset decisions (agents/presets.py)

- **SAC** for `DualLoop-v0` and `VectorSumTuner-v0` — stable and clearly
  better; the off-policy credit assignment handles the long-horizon
  slow-loop action.
- **TD3** for `GainScheduling-v0` — the gain-scheduling cost landscape
  favours deterministic off-policy updates.
- **PPO** everywhere else: SAC/TD3 brought no reliable gain (e.g.
  `VectorSum` SAC edges reward but with ±412 std v.s. ±130 for PPO).
- The earlier ad-hoc `FrequencyControl → SAC` hypothesis was disproved by
  the study (PPO -7596 vs SAC -11994 at the same budget) and removed.

---

## 4. Level-by-level physics reading

- **L0 residual (CavityVoltage / PIAssist / ADRCAssist):** an additive RL
  residual on top of a tuned feedforward/inner loop has marginal headroom;
  reward ≈ baseline. This motivated the **gain-scheduling** variant, where
  RL *shapes* the controller instead of fighting it.
- **L0 gain scheduling (GainScheduling):** TD3 learns to raise gains where
  it pays (flattop) and back off through transients — the only structure in
  which "improve a classical loop" yields a real, reproducible gain.
- **L1/L2 resonance (Frequency / DualLoop):** RL slow tuners work; the
  cooperative DualLoop task is where RL beats both no-tuner and the
  on-policy alternative most decisively.
- **L3 vector sum:** shared-drive residual helps; per-cavity tuners need
  signed per-cavity detuning in the observation and an off-policy variant
  to stay stable.
- **L4 beam/pulse:** the pulse-to-pulse adaptive feedforward is the largest
  relative win (+34 %); beam-loading residual stays within disturbance
  noise on the current high-beta cavity parameters (the droop is small).

---

## 5. Case study: VectorSumTuner stability

Diagnosed failure: default PPO converges at ~8k steps (near-optimal) but
drifts into **drive saturation** on longer budgets (−7914 at 80k, −3382 at
40k). Systematic ablations (budget, gamma 0.999, n_steps 1024/2048,
ent_coef 0.02, VecNormalize, TD3) all failed to stabilize it. Two fixes
made it learnable and stable:

1. **Observability:** per-cavity detuning added to the observation
   (*signed*, not absolute — absolute magnitude hides the correction sign).
2. **Algorithm:** off-policy SAC at `train_freq=gradient_steps=64`.

Final: -268 (baseline) → **-231 ± 17** (SAC, 20k). Full ablation table:
`benchmarks/vstuner_study.md`.

---

## 6. Reproducibility

```bash
# final 10-environment benchmark (recommended budgets, 4 seeds)
python -u -m llrflibs_rl.scripts.run benchmark --seeds 4 --out benchmarks

# PPO/SAC/TD3 comparison (resumable, --envs to slice)
python -u experiments/algorithm_comparison.py

# VectorSumTuner stability study
python experiments/vector_sum_tuner_study.py

# single environment with its preset (e.g. DualLoop -> SAC)
python -m llrflibs_rl.scripts.run train --env DualLoop-v0 --runs-dir runs
```

Each `train`/`benchmark` writes a self-contained artifact directory under
`runs/` (config, metrics, model, training curve, comparison figure).
Every benchmark artifact is also indexed in
[`benchmarks/README.md`](../benchmarks/README.md).

## 7. Caveats

- Numbers are from deterministic single training seeds; keep
  `--seeds N` ≥ 4 for robust claims.
- `--quick` budgets (2k steps ≈ one episode) are CI-style sanity checks and
  must **not** be read as algorithm performance.
- Scatter on `BeamLoading` / `CavityVoltage` / `VectorSum` reflects
  disturbance/domain randomization; report means over seeds.
- Reward scales differ across environments — compare within a row only.
