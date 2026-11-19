# LLRF RL benchmark — Level-L4 snapshot (archived)

> Historical snapshot from the initial Level-L4 benchmark (BeamLoading /
> AdaptivePulse / VectorSumTuner / GainScheduling). Superseded by the full
> per-environment benchmark in [`../benchmark.md`](../benchmark.md), which covers
> all registered environments with the recommended budgets. Kept for record.

| env | baseline_reward | rl_reward | rl_reward_std | n_seeds | baseline_error_rms | rl_error_rms | baseline_flattop_error_rms | rl_flattop_error_rms |
|---|---|---|---|---|---|---|---|---|
| BeamLoading-v0 | -1632.3448 | -1698.3747 | 1291.7907 | 4 | 1.28e+07 | 1.27e+07 | 1.19e+07 | 1.19e+07 |
| AdaptivePulse-v0 | -9.2839 | -6.0759 | 0.0000 | 4 | - | - | - | - |
| VectorSumTuner-v0 | -268.3060 | -231.2804 | 16.5084 | 4 | 3.81e+07 | 3.37e+07 | - | - |
| GainScheduling-v0 | -353650.3942 | -345494.8820 | 64148.9547 | 4 | 1.08e+08 | 1.07e+08 | 1.17e+08 | 1.16e+08 |

