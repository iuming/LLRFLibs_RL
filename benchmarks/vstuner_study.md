# VectorSumTuner-v0 stability study

4-seed deterministic evaluation; reward = mean episode reward (higher is better, baseline FF only).

| config | reward_mean | reward_std | error_rms | note |
|---|---:|---:|---:|---|
| baseline (zero) | -268.3 | 114.1 | 3.813e+07 | nominal feedforward only |
| ppo-4k (current) | -478.4 | 81.2 | 4.097e+07 | current recommended budget |
| ppo-20k | -1,032.3 | 258.9 | 5.552e+07 | default PPO, longer budget |
| ppo-norm-20k | -2,923.2 | 660.7 | 4.586e+07 | PPO + VecNormalize observations |
| sac-20k | -231.3 | 16.5 | 3.370e+07 | off-policy SAC (freq=grad=64) |
| sac-norm-20k | -494.7 | 205.9 | 3.110e+07 | SAC + VecNormalize |
| td3-20k | -1,121.7 | 93.0 | 5.040e+07 | off-policy TD3 |
| td3-norm-20k | -2,773.9 | 1,249.0 | 4.295e+07 | TD3 + VecNormalize |
