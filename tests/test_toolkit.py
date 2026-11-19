###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""Tests of the engineering toolkit: verifier, api, plotting, model zoo,
beam schedules, disturbances and the Lorentz drift model."""

from pathlib import Path

import numpy as np
import pytest

import llrflibs_rl as rl
from llrflibs_rl.agents import AgentConfig, train
from llrflibs_rl.agents.model_zoo import ModelZoo
from llrflibs_rl.plants.beam_schedule import BeamSchedule, BeamSegment
from llrflibs_rl.plants.disturbance import build_disturbance_wave
from llrflibs_rl.plants.lorentz import LorentzDriftDetuning
from llrflibs_rl.utils.verifier import verify_all_environments, verify_environment

WORKSPACE_TMP = Path(__file__).resolve().parent / ".tmp_toolkit"


@pytest.fixture(autouse=True)
def _workspace_tmp(fast_cavity, fast_env_config):
    WORKSPACE_TMP.mkdir(exist_ok=True)
    yield
    import shutil

    shutil.rmtree(WORKSPACE_TMP, ignore_errors=True)


# ---------------------------------------------------------------------------
# Requirement: src enrichment — engineering toolkit
# ---------------------------------------------------------------------------

def test_verify_environment_reports_pass():
    report = verify_environment("CavityVoltage-v0", n_steps=40)
    assert report.passed, report.checks
    assert report.length > 0
    assert "step" in report.checks


def test_verify_all_environments_passes():
    reports = verify_all_environments(lambda: ["CavityVoltage-v0", "DualLoop-v0", "AdaptivePulse-v0"],
                                      n_steps=30)
    for report in reports:
        assert report.passed, report.checks


def test_api_facade_exposes_common_functions():
    from llrflibs_rl import api

    api.make("CavityVoltage-v0")
    cfg = api.agent_config_for("CavityVoltage-v0", timesteps=256)
    model = api.train(lambda: api.make("CavityVoltage-v0"), cfg)
    metrics = api.evaluate_episode(api.make("CavityVoltage-v0"), api.action_from_model(model))
    assert metrics["length"] > 0
    assert api.list_env_ids()


def test_model_zoo_save_load_list(tmp_cavity_preset=None):
    env_id = "CavityVoltage-v0"
    cfg = AgentConfig(timesteps=256, seed=0)
    model = train(lambda: rl.make(env_id), cfg)
    zoo = ModelZoo(WORKSPACE_TMP / "zoo")
    zoo.save(model, env_id, tag="test", timesteps=256)
    loaded = zoo.load(env_id, "test")
    assert loaded is not None
    rows = zoo.list()
    assert any(r["env_id"] == env_id for r in rows)
    meta = zoo.metadata(env_id, "test")
    assert meta["algorithm"] == "ppo"
    assert zoo.latest(env_id) is not None


def test_plot_comparison_writes_file(fast_cavity, fast_env_config):
    from llrflibs_rl.controllers.baselines import zero_policy
    from llrflibs_rl.scenario import rollout_episode
    from llrflibs_rl.utils.plotting import plot_comparison

    env = rl.make("CavityVoltage-v0", cavity=fast_cavity, env_cfg=fast_env_config)
    policy = zero_policy(env)
    wave = rollout_episode(env, policy, seed=0)
    out = WORKSPACE_TMP / "fig.png"
    plot_comparison(env, wave, wave, "zero", [], out)
    assert out.exists() and out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Gymnasium parity (A): spec / unwrapped / reward_range / wrappers / checker
# ---------------------------------------------------------------------------

def test_gymnasium_parity_attributes_and_wrappers():
    env = rl.make("CavityVoltage-v0")
    assert env.spec is not None and env.spec.env_id == "CavityVoltage-v0"
    assert env.unwrapped is env
    assert env.reward_range[0] == float("-inf")
    assert env.spec.max_episode_steps == 1500

    from gymnasium.wrappers import TimeLimit

    wrapped = rl.make("CavityVoltage-v0", wrappers=[lambda e: TimeLimit(e, max_episode_steps=50)])
    assert wrapped.unwrapped is not None  # wrapper chain can be unwrapped
    assert wrapped.spec is not None
    obs, _ = wrapped.reset(seed=0)
    steps = 0
    for _ in range(200):
        obs, rw, term, trunc, info = wrapped.step(np.zeros(2, dtype=np.float32))
        steps += 1
        if term or trunc:
            break
    assert steps == 50  # TimeLimit truncates at 50


def test_verifier_runs_official_gymnasium_checker():
    report = verify_environment("CavityVoltage-v0", n_steps=10)
    assert report.passed, report.checks
    assert report.checks["gymnasium_check_env"] == "ok"


# ---------------------------------------------------------------------------
# Training stack (B): best model / checkpoints / n_envs / zoo vecnorm
# ---------------------------------------------------------------------------

def test_best_model_and_checkpoint_saved():
    env_id = "CavityVoltage-v0"
    cfg = AgentConfig(timesteps=256, seed=0, n_steps=128)
    best_dir = WORKSPACE_TMP / "best"
    ckpt_dir = WORKSPACE_TMP / "ckpt"
    train(
        lambda: rl.make(env_id), cfg,
        eval_env_factory=lambda: rl.make(env_id),
        eval_freq=128, best_model_dir=best_dir,
        checkpoint_dir=ckpt_dir, checkpoint_freq=100,
    )
    assert any((best_dir).glob("best_model.zip"))
    assert any((ckpt_dir).glob("rl_model_*.zip"))


def test_n_envs_and_policy_kwargs_training():
    env_id = "CavityVoltage-v0"
    cfg = AgentConfig(timesteps=256, seed=0, n_envs=2, n_steps=128,
                      policy_kwargs={"net_arch": [32, 32]})
    model = train(lambda: rl.make(env_id), cfg)
    assert model is not None
    with pytest.raises(ValueError):
        AgentConfig(policy="Nope")


def test_model_zoo_round_trips_vecnormalize():
    from stable_baselines3.common.vec_env import VecNormalize

    env_id = "CavityVoltage-v0"
    cfg = AgentConfig(timesteps=256, seed=0, normalize_obs=True)
    model = train(lambda: rl.make(env_id), cfg)
    zoo = ModelZoo(WORKSPACE_TMP / "zoo_norm")
    zoo.save(model, env_id, tag="norm")
    meta = zoo.metadata(env_id, "norm")
    assert meta["normalize_obs"] is True
    loaded = zoo.load(env_id, "norm")
    assert isinstance(getattr(loaded, "env", None), VecNormalize)


def test_cross_check_with_sb3(fast_cavity, fast_env_config):
    from llrflibs_rl.evaluation import cross_check_with_sb3

    factory = lambda: rl.make(  # noqa: E731
        "CavityVoltage-v0", cavity=fast_cavity, env_cfg=fast_env_config
    )
    cfg = AgentConfig(timesteps=256, seed=0)
    model = train(factory, cfg)
    result = cross_check_with_sb3(model, factory, n_eval_episodes=2)
    assert result["ok"] is True, result
    assert result["ours"] is not None and result["sb3"] is not None


# ---------------------------------------------------------------------------
# Requirement: src enrichment — beam schedules / disturbances / Lorentz
# ---------------------------------------------------------------------------

def test_beam_schedule_applies_segments():
    vb = np.ones(100, dtype=np.complex128) * 5.0
    sched = BeamSchedule([BeamSegment(0, 30, 1.0), BeamSegment(50, 100, 0.5)])
    wave = sched.apply_to(vb)
    assert np.all(wave[:30] == 5.0)
    assert np.all(wave[30:50] == 0.0)
    assert np.all(wave[50:] == 2.5)


def test_beam_schedule_on_beam_loading_env():
    from dataclasses import replace

    from llrflibs_rl.configs.cavity import nominal_sc_cavity
    from llrflibs_rl.envs.pulse_to_pulse import BeamLoadingConfig, BeamLoadingEnv

    cav = replace(nominal_sc_cavity(), filling_len=150, flattop_len=200)
    env = BeamLoadingEnv(
        cav,
        BeamLoadingConfig(
            episode_len=400,
            beam_current_plant=8e-3,
            reference_beam_current=0.0,
            beam_schedule=BeamSchedule([BeamSegment(150, 250, 1.0), BeamSegment(250, 400, 0.5)]),
            randomize_detuning=0.0,
            microphonics_amp=0.0,
        ),
    )
    assert np.all(env._beam_wave[:150] == 0.0)
    assert np.any(np.abs(env._beam_wave[150:250]) > 1.0)
    # partial level segment: half the nominal beam drive
    nominal = float(np.abs(env._beam_wave[200]))
    assert float(np.abs(env._beam_wave[300])) == pytest.approx(nominal * 0.5)


def test_disturbance_types_shapes():
    wave_sin = build_disturbance_wave("sinusoidal", 100, 1e-6, base=1.0, amplitude=0.5)
    wave_step = build_disturbance_wave("step", 100, 1e-6, base=1.0, amplitude=2.0,
                                       onset=40e-6)
    wave_sweep = build_disturbance_wave("sweep", 100, 1e-6, base=1.0, amplitude=2.0)
    assert wave_sin.shape == wave_step.shape == wave_sweep.shape == (100,)
    assert np.all(wave_step[:35] == pytest.approx(1.0))
    assert np.all(wave_step[45:] == pytest.approx(3.0))
    assert wave_sweep[-1] > wave_sweep[0]
    with pytest.raises(ValueError):
        build_disturbance_wave("nope", 10, 1e-6, base=0.0, amplitude=0.0)


def test_lorentz_drift_builds_up_and_is_negative():
    model = LorentzDriftDetuning(k_lor=1.0e-12, tau=1e-4, Ts=1e-6)
    model.reset()
    shifts = [model.step(complex(25e6)) for _ in range(2000)]
    assert shifts[0] < 0.0  # starts negative once amplitude is present
    assert shifts[-1] < shifts[0]  # builds up toward -k*V^2
    expected = -1.0e-12 * (25e6) ** 2
    assert shifts[-1] == pytest.approx(expected, rel=0.01)


def test_lorentz_env_wiring_changes_detuning():
    from dataclasses import asdict, replace

    from llrflibs_rl.configs.cavity import nominal_sc_cavity
    from llrflibs_rl.configs.env import ResidualEnvConfig

    cav = replace(nominal_sc_cavity(), filling_len=150, flattop_len=200)
    base = ResidualEnvConfig(
        episode_len=400, randomize_detuning=0.0, microphonics_amp=0.0,
        measurement_noise_std=0.0, loop_delay_steps=0,
        max_drive_abs=None, max_drive_slew_per_step=None,
    )
    plain = rl.make("CavityVoltage-v0", cavity=cav, env_cfg=ResidualEnvConfig(**asdict(base)))
    lor_kwargs = dict(asdict(base))
    lor_kwargs["lorentz_k"] = 1.0e-12
    lor = rl.make("CavityVoltage-v0", cavity=cav, env_cfg=ResidualEnvConfig(**lor_kwargs))
    envs = [plain, lor]
    for env in envs:
        env.reset(seed=0)
    for _ in range(400):
        for env in envs:
            env.step(np.zeros(2, dtype=np.float32))
    # Lorentz detuning is active only in the Lorentz-enabled environment and
    # has drifted the detuning (observed through the model state).
    assert plain._lorentz is None
    assert lor._lorentz is not None
    assert lor._lorentz.state > 0.0  # positive intensity-proportional magnitude


def test_v1_variants_run():
    for env_id in ("CavityVoltage-v1", "PIAssist-v1", "DualLoop-v1", "VectorSum-v1", "AdaptivePulse-v1"):
        env = rl.make(env_id)
        obs, _ = env.reset(seed=0)
        for _ in range(20):
            obs, rw, term, trunc, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
            if term or trunc:
                break
        assert obs.shape == env.observation_space.shape
