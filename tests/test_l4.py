###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""Physics tests of the Level-L4 additions (A1-A4)."""

from dataclasses import asdict, replace

import numpy as np
import pytest

import llrflibs_rl as rl
from llrflibs_rl.configs.cavity import nominal_sc_cavity
from llrflibs_rl.configs.env import ResidualEnvConfig, VectorSumEnvConfig
from llrflibs_rl.envs.gain_scheduling import GainSchedulingConfig
from llrflibs_rl.envs.pulse_to_pulse import AdaptivePulseConfig, BeamLoadingConfig


def _small_cavity():
    return replace(nominal_sc_cavity(), filling_len=150, flattop_len=200)


# ---------------------------------------------------------------------------
# A1 -- beam loading
# ---------------------------------------------------------------------------

def test_beam_loading_injects_beam_after_onset():
    cav = _small_cavity()
    env = rl.make(
        "BeamLoading-v0",
        cavity=cav,
        env_cfg=BeamLoadingConfig(
            episode_len=600,
            beam_current_plant=0.0, reference_beam_current=0.0,
            randomize_detuning=0.0, microphonics_amp=0.0,
        ),
    )
    assert np.all(env._beam_wave == 0.0)  # no beam anywhere

    env2 = rl.make(
        "BeamLoading-v0",
        cavity=cav,
        env_cfg=BeamLoadingConfig(
            episode_len=600, beam_current_plant=8e-3, reference_beam_current=0.0,
            beam_on_start=150, randomize_detuning=0.0, microphonics_amp=0.0,
        ),
    )
    assert np.all(env2._beam_wave[:150] == 0.0)
    assert np.any(np.abs(env2._beam_wave[150:]) > 1.0)


def test_beam_loading_droop_is_observable():
    """With a beam-free feedforward, a beam that steps on degrades flattop tracking."""
    cav = _small_cavity()

    def _flattop_error(beam_on: bool):
        env = rl.make(
            "BeamLoading-v0",
            cavity=cav,
            env_cfg=BeamLoadingConfig(
                episode_len=600,
                beam_current_plant=8e-3 if beam_on else 0.0,
                reference_beam_current=0.0,
                beam_on_start=150,
                randomize_detuning=0.0, microphonics_amp=0.0,
            ),
        )
        env.reset(seed=0)
        errs = []
        for _ in range(360):
            _, rw, term, trunc, info = env.step(np.zeros(2, dtype=np.float32))
            errs.append(info["error_abs"])
            if term or trunc:
                break
        window = errs[170:340]  # settled flattop (filling 150 + flattop 200)
        return float(np.mean(window)), float(np.max(window))

    no_beam_mean, _ = _flattop_error(False)
    beam_mean, _ = _flattop_error(True)
    assert beam_mean > no_beam_mean  # the beam step adds a measurable droop


# ---------------------------------------------------------------------------
# A1 -- adaptive pulse
# ---------------------------------------------------------------------------

def test_adaptive_pulse_zero_gain_rewards_drift_with_detuning():
    cav = _small_cavity()
    env = rl.make(
        "AdaptivePulse-v0",
        config=AdaptivePulseConfig(
            cavity=cav, pulse_len=600, n_basis=8, n_pulses=12,
            detuning_drift_amp=2.0 * np.pi * 1200.0, randomize_detuning=0.0,
        ),
    )
    obs, _ = env.reset(seed=0)
    rewards = []
    for _ in range(12):
        obs, rw, term, trunc, info = env.step(np.zeros(8, dtype=np.float32))
        rewards.append(rw)
    # More detuning mismatch -> worse reward on the last pulses.
    assert rewards[0] > rewards[-1]
    assert env._detuning_pulse > env.config.cavity.detuning


def test_adaptive_pulse_gain_correction_changes_result():
    cav = _small_cavity()
    env = rl.make(
        "AdaptivePulse-v0",
        config=AdaptivePulseConfig(cavity=cav, pulse_len=400, n_basis=4, n_pulses=6,
                                   randomize_detuning=0.0, detuning_drift_amp=0.0),
    )
    env.reset(seed=0)
    _, r0, *_ = env.step(np.zeros(4, dtype=np.float32))
    env.reset(seed=0)
    _, r1, *_ = env.step(np.ones(4, dtype=np.float32) * -0.9)
    assert r0 != pytest.approx(r1, abs=1e-3)


# ---------------------------------------------------------------------------
# A2 -- vector-sum tuner
# ---------------------------------------------------------------------------

def test_vector_sum_tuner_offsets_change_plant_output():
    from llrflibs_rl.plants.vectormodule import VectorModulePlant

    cav = _small_cavity()
    ts = 1e-6
    plant = VectorModulePlant([cav, cav, cav], ts, reference_detuning=0.0)
    plant.reset([0.0, 0.0, 0.0])
    drive = complex(10e6)
    beam = complex(0.0)
    for _ in range(200):
        vs0, _, _, _ = plant.step(drive, beam)
    plant.reset([0.0, 0.0, 0.0])
    for _ in range(200):
        vs1, _, _, _ = plant.step(drive, beam, detuning_offsets=[2 * np.pi * 500, 0.0, 0.0])
    assert abs(vs0 - vs1) > 1.0  # detuning one cavity changes the vector sum


def test_vector_sum_tuner_env_shapes_and_step():
    cav = _small_cavity()
    env = rl.make(
        "VectorSumTuner-v0",
        cavities=[cav, cav, cav],
        env_cfg=VectorSumEnvConfig(
            cavities=[cav, cav, cav], tuner_action=True, episode_len=400,
            randomize_detuning=0.0, microphonics_amp=0.0, measurement_noise_std=0.0,
            loop_delay_steps=0, max_drive_abs=None, max_drive_slew_per_step=None,
        ),
    )
    obs, _ = env.reset(seed=0)
    assert env.action_space.shape == (5,)
    assert obs.shape == (13 + 6 + 6,)
    obs, rw, term, trunc, info = env.step(np.zeros(5, dtype=np.float32))
    assert obs.shape == (25,)
    assert np.isfinite(rw)


# ---------------------------------------------------------------------------
# A3 -- gain scheduling
# ---------------------------------------------------------------------------

def test_gain_scheduling_high_versus_nominal_gains_differ():
    cav = _small_cavity()
    base = ResidualEnvConfig(
        episode_len=600, randomize_detuning=0.0, microphonics_amp=0.0,
        measurement_noise_std=0.0, loop_delay_steps=0,
        max_drive_abs=None, max_drive_slew_per_step=None,
    )
    kwargs = dict(asdict(base))
    env_nom = rl.make("GainScheduling-v0", cavity=cav,
                      env_cfg=GainSchedulingConfig(**kwargs))
    obs, _ = env_nom.reset(seed=0)
    r_nom = 0.0
    for _ in range(350):
        obs, rw, term, trunc, info = env_nom.step(np.zeros(2, dtype=np.float32))
        r_nom += rw

    env_hi = rl.make("GainScheduling-v0", cavity=cav,
                     env_cfg=GainSchedulingConfig(**kwargs))
    obs, _ = env_hi.reset(seed=0)
    r_hi = 0.0
    for _ in range(350):
        obs, rw, term, trunc, info = env_hi.step(np.ones(2, dtype=np.float32))
        r_hi += rw
    assert r_nom != pytest.approx(r_hi, abs=10.0)


def test_gain_scheduling_pi_baseline_is_nominal_gains():
    from llrflibs_rl.controllers.baselines import make_baseline_policy

    env = rl.make("GainScheduling-v0", cavity=_small_cavity())
    policy = make_baseline_policy(env, "pi")
    obs, _ = env.reset(seed=0)
    action = np.asarray(policy(obs))
    np.testing.assert_allclose(action, 0.0)  # zero action == nominal gains


# ---------------------------------------------------------------------------
# A4 -- reflected-power trip and half-bandwidth drift
# ---------------------------------------------------------------------------

def test_reflected_power_trip_terminates_early():
    cav = _small_cavity()
    cfg = ResidualEnvConfig(
        episode_len=600, randomize_detuning=0.0, microphonics_amp=0.0,
        reflected_power_limit=5e6, reflected_power_trip_steps=5, trip_penalty=-100.0,
    )
    env = rl.make("CavityVoltage-v0", cavity=cav, env_cfg=cfg)
    obs, _ = env.reset(seed=0)
    steps = 0
    terminated = False
    for _ in range(600):
        obs, rw, term, trunc, info = env.step(np.zeros(2, dtype=np.float32))
        steps += 1
        if term:
            terminated = True
            break
        if trunc:
            break
    assert terminated
    assert steps < 600


def test_reflected_power_trip_not_fired_without_limit():
    cav = _small_cavity()
    cfg = ResidualEnvConfig(
        episode_len=600, randomize_detuning=0.0, microphonics_amp=0.0,
    )
    env = rl.make("CavityVoltage-v0", cavity=cav, env_cfg=cfg)
    obs, _ = env.reset(seed=0)
    steps = 0
    for _ in range(600):
        obs, rw, term, trunc, info = env.step(np.zeros(2, dtype=np.float32))
        steps += 1
        if term or trunc:
            break
    assert not term
    assert steps == 600


def test_half_bw_drift_uses_time_varying_plant():
    cav = _small_cavity()
    cfg = ResidualEnvConfig(
        episode_len=600, randomize_detuning=0.0, microphonics_amp=0.0,
        half_bw_drift_frac=0.25,
    )
    env = rl.make("CavityVoltage-v0", cavity=cav, env_cfg=cfg)
    env.reset(seed=0)
    nominal = cav.half_bw
    assert env._current_half_bw() == pytest.approx(nominal, abs=1e-6)
    for _ in range(500):
        env.step(np.zeros(2, dtype=np.float32))
    drifted = env._current_half_bw()
    assert drifted > nominal * 1.1  # the plant really drifts
