# Benchmarks

Machine-readable and human-readable benchmark reports for the registered
environments of `llrflibs-rl`. All numbers are produced by the deterministic
4-seed evaluation pipeline (`evaluate_seeds=0..3`) and can be regenerated with
the commands below.

| Report | Content | Regenerate |
|---|---|---|
| [`benchmark.md`](benchmark.md) / [`benchmark.json`](benchmark.json) | Recommended-budget baseline-vs-RL benchmark across **all** registered environments | `python -u -m llrflibs_rl.scripts.run benchmark` |
| [`algorithm_comparison.md`](algorithm_comparison.md) | PPO / SAC / TD3 cross-environment algorithm study | `python -u experiments/algorithm_comparison.py` |
| [`vstuner_study.md`](vstuner_study.md) / [`vstuner_study.json`](vstuner_study.json) | VectorSumTuner-v0 stability study (budget & normalization sweeps) | `python -u experiments/vector_sum_tuner_study.py` |
| [`l4/benchmark.md`](l4/benchmark.md) / [`l4/benchmark.json`](l4/benchmark.json) | **Archived** Level-L4 snapshot (BeamLoading / AdaptivePulse / VectorSumTuner / GainScheduling) — superseded by `benchmark.md`, kept for record | — |

The definitive reading for the current recommendation is
[`docs/BENCHMARKS.md`](../docs/BENCHMARKS.md).
