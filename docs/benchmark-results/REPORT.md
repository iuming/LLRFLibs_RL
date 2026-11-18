# Benchmark Report

This report is auto-generated from `generate_results.py`.

## Primary examples

- `example_rl_sp_ff.py`
- `example_rl_ilc_compare.py`
- `example_rl_frequency.py`
- `example_rl_sim_cavity_mech2.py`
- `example_rl_pi_assist.py`
- `example_rl_adrc_assist.py`
- `example_rl_vector_sum.py`
- `example_rl_dual_loop.py`

## Exploratory examples

- `exploratory/example_rl_cavity.py`
- `exploratory/example_rl_feedback.py`
- `exploratory/example_rl_feedback_adrc.py`
- `exploratory/example_rl_sim_cavity_basic.py`

## Summary table

| Example | Baseline | RL reward | Baseline RMS | RL RMS | Conclusion |
|---|---:|---:|---:|---:|---|
| `example_rl_cavity.py` | PID-like | -40195.92 | 5.484e+07 | 5.278e+07 | RL 明显优 |
| `example_rl_frequency.py` | Zero action | -417.47 | 2.065e+07 | 1.140e+07 | RL 明显优 |
| `example_rl_sim_cavity_basic.py` | Zero correction | -3826.20 | 1.501e+07 | 1.501e+07 | RL 明显优 |
| `example_rl_sim_cavity_mech2.py` | Zero action | -1449.67 | 2.129e+07 | 2.011e+07 | RL 明显优 |
| `example_rl_feedback.py` | PI | -1621.56 | 6.966e+06 | 5.941e+06 | RL 明显优 |
| `example_rl_feedback_adrc.py` | ADRC-like | -50709.90 | 5.033e+07 | 4.839e+07 | RL 明显优 |
| `example_rl_ilc_compare.py` | No correction | -55938.80 | - | 2.731e+07 | ILC 仍优于当前 RL |
| `example_rl_sp_ff.py` | Nominal feedforward | -1147.32 | 1.277e+07 | 1.313e+07 | RL 略差 |
| `example_rl_pi_assist.py` | PI only | -136.34 | 8.492e+06 | 8.464e+06 | RL 略差 |
| `example_rl_adrc_assist.py` | ADRC only | -65.17 | 7.854e+06 | 7.853e+06 | 基本持平 |
| `example_rl_vector_sum.py` | Nominal FF | -478.75 | 4.945e+07 | 4.993e+07 | RL 明显优 |
| `example_rl_dual_loop.py` | No tuner | -22.63 | 9.190e+06 | 9.001e+06 | RL 明显优 |
