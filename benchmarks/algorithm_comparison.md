# Algorithm comparison across all environments

Recommended budget per environment, 4-seed deterministic evaluation.
Higher reward is better. Regenerate with:
`python -u experiments/algorithm_comparison.py`
Definitive reading: `docs/BENCHMARKS.md`.

| env | baseline | PPO | SAC | TD3 | best |
|---|---:|---:|---:|---:|---|
| ADRCAssist-v0 | -64.4 | -66.9 | -122.1 | -145.6 | PPO |
| AdaptivePulse-v0 | -9.3 | -6.1 | -6.9 | -6.4 | PPO |
| BeamLoading-v0 | -1632.3 | -1698.4 | -1875.2 | -1646.5 | TD3 (noise) |
| CavityVoltage-v0 | -1419.8 | -1413.9 | -1614.9 | -1504.4 | PPO |
| DualLoop-v0 | -366.4 | -383.6 | -164.3 | -4383.6 | SAC |
| FrequencyControl-v0 | -8315.4 | -7595.9 | -11993.5 | -12459.0 | PPO |
| GainScheduling-v0 | -353650.4 | -345494.9 | -376407.0 | -267779.9 | TD3 |
| PIAssist-v0 | -153.3 | -169.2 | -166.9 | -178.2 | SAC (tie) |
| VectorSum-v0 | -636.5 | -580.6 | -533.7 | -3023.8 | SAC (high std) |
| VectorSumTuner-v0 | -268.3 | -1032.3 | -231.3 | -1121.7 | SAC |
