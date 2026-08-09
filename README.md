# LLRFLibs_RL

Reinforcement-learning environments and examples for LLRF cavity control.

**Author:** Ming Liu (mliu1018@ustc.edu.cn), National Synchrotron Radiation Laboratory, University of Science and Technology of China

This subrepository is meant to sit next to the main `LLRFLibsPy` repository and reuse its physics and RF helpers.

## Contents

- Gymnasium environments for cavity-voltage tracking and frequency-control compensation.
- Stable-Baselines3 training examples.
- A small PID-style baseline comparison in the cavity example.
- RL counterparts for feedback, ADRC-style feedback, cavity simulation, and mechanical detuning control.

## Install

Install the main library first, then install this subrepository in editable mode from this folder:

```bash
pip install -e ..
pip install -e .
```

If you already have `llrflibs`, `gymnasium`, and `stable-baselines3` in your RL conda environment, only the editable install is needed.

## Run examples

```bash
python -m example.example_rl_cavity
python -m example.example_rl_feedback
python -m example.example_rl_feedback_adrc
python -m example.example_rl_sim_cavity_basic
python -m example.example_rl_sim_cavity_mech2
python -m example.example_rl_frequency
python -m example.example_rl_sp_ff
```

## Example mapping

- `example_rl_cavity.py` maps to pulsed cavity-voltage tracking and feedforward correction.
- `example_rl_feedback.py` maps to the basic RF feedback control example.
- `example_rl_feedback_adrc.py` maps to the ADRC feedback control example.
- `example_rl_sim_cavity_basic.py` maps to the basic cavity-response simulation example.
- `example_rl_sim_cavity_mech2.py` maps to the cavity-with-mechanical-modes example.
- `example_rl_frequency.py` maps to the detuning and frequency-control example.
- `example_rl_ilc_compare.py` compares RL against the ILC-style adaptive feedforward example.
- `example_rl_sp_ff.py` compares RL residual control against the analytical feedforward example.