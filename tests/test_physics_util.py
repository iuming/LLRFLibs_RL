###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Tests for the science-toolkit layer: PSD noise, notch, ILC, calibration, sysid,
loop stability, spectral analysis and the measurement chain.
"""

from __future__ import annotations

import numpy as np

import llrflibs_rl as rl

# ---------------------------------------------------------------------------
# PSD-driven microphonics (plants/noise + env integration)
# ---------------------------------------------------------------------------

def test_microphonics_profile_sample_and_rms():
    from llrflibs_rl.plants.noise import typical_cryomodule_microphonics

    profile = typical_cryomodule_microphonics()
    wave = profile.sample(fs=1.0e6, n_samples=8192, rng=np.random.default_rng(7))
    assert wave.shape == (8192,)
    assert np.all(np.isfinite(wave))
    assert abs(np.mean(wave)) < 1e-9
    rms_val = profile.rms(10.0, 300.0, fs=1.0e6)
    assert np.isfinite(rms_val) and rms_val > 0.0


def test_microphonics_psd_used_by_voltage_env(fast_cavity, fast_env_config):
    from dataclasses import replace

    from llrflibs_rl.plants.noise import typical_cryomodule_microphonics

    config = replace(fast_env_config, microphonics_psd=typical_cryomodule_microphonics())
    env = rl.make("CavityVoltage-v0", cavity=fast_cavity, env_cfg=config)
    env.reset(seed=0)
    wave = env._detuning_wave  # noqa: SLF001
    assert wave is not None and wave.shape[0] == config.episode_len
    assert np.max(np.abs(wave)) > 0.0


# ---------------------------------------------------------------------------
# Notch filter / controller notch config (controllers/notch)
# ---------------------------------------------------------------------------

def test_notch_filter_rejects_tone():
    from llrflibs_rl.controllers.notch import NotchFilter

    fs = 1.0e6
    # A cavity EMI / microphonics line at tens of kHz (well-conditioned f0/fs).
    f_tone = 50.0e3
    t = np.arange(20000) / fs
    wave = np.cos(2.0 * np.pi * f_tone * t)
    filt = NotchFilter(fnotch=f_tone, q=40.0, fs=fs)
    out = filt.filter_wave(wave)
    # The tone should be strongly attenuated after the transient.
    rms_in = np.sqrt(np.mean(wave[4000:] ** 2))
    rms_out = np.sqrt(np.mean(np.abs(out[4000:]) ** 2))
    assert float(rms_out) < float(rms_in) * 0.2


def test_notch_conf_helper_and_pi_controller():
    from dataclasses import replace

    from llrflibs_rl.configs.controller import PIControllerConfig
    from llrflibs_rl.controllers.feedback import ComplexPIController
    from llrflibs_rl.controllers.notch import notch_conf_for_frequencies

    conf = notch_conf_for_frequencies([250.0], [2.0 * np.pi * 60.0])
    assert set(conf) == {"freq_offs", "half_bw", "gain"}
    pi_cfg = replace(PIControllerConfig(), notch_conf=conf)
    ctrl = ComplexPIController(pi_cfg, Ts=1.0e-6)
    drive, fb = ctrl.step(1.0 + 0.5j, 0.0)
    assert np.isfinite(drive) and np.isfinite(fb)


# ---------------------------------------------------------------------------
# ILC / adaptive feedforward (controllers/ilc)
# ---------------------------------------------------------------------------

def test_ilc_reduces_repeatable_error(fast_cavity):
    from dataclasses import replace

    from llrflibs_rl.controllers.ilc import ILCFeedforward, impulse_response_matrix
    from llrflibs_rl.plants.cavity_voltage import CavityVoltagePlant
    from llrflibs_rl.plants.sysid import identify_impulse_response, prbs_wave

    # Fast-cavity regime so the impulse response decays within the pulse
    # window (keeps the response matrix well conditioned for ILC).
    fast = replace(fast_cavity, half_bw=2.0 * np.pi * 3000.0, detuning=0.0)
    Ts = 1.0e-6
    plant = CavityVoltagePlant(fast, Ts, fast.detuning)
    n = 512

    # Impulse response of the plant identified from PRBS ensembles.
    drives = np.vstack([prbs_wave(n, -1.0e6, 1.0e6).astype(np.complex128) for _ in range(3)])
    outs = np.zeros_like(drives)
    for k in range(3):
        plant.reset()
        for i in range(n):
            outs[k, i], _ = plant.step(complex(drives[k, i]))
    h = identify_impulse_response(drives, outs, order=64)
    assert np.all(np.isfinite(h))
    mat = impulse_response_matrix(h, pulse_width=n)
    assert mat.shape == (n, n)

    # A repeatable drive disturbance: run closed pulses and apply ILC
    # feedforward from the previous pulse error.
    ilc = ILCFeedforward(h, pulse_width=n)
    sp_phase = np.linspace(0.0, np.pi, n)
    sp = fast.vc_target * np.exp(1j * sp_phase) / 1000.0
    dist = 1.2e6 * np.sin(2.0 * np.pi * 250.0 * np.arange(n) * Ts)
    base = np.full(n, 1.0e6, dtype=np.complex128)
    correction = np.zeros(n, dtype=np.complex128)
    rms0 = None
    for _ in range(8):
        drive = base + correction + dist
        plant.reset()
        vc = np.zeros(n, dtype=np.complex128)
        for i in range(n):
            vc[i], _ = plant.step(complex(drive[i]))
        error = sp[:n] - vc
        rms = float(np.sqrt(np.mean(np.abs(error) ** 2)))
        if rms0 is None:
            rms0 = rms
        correction = ilc.update(error)
    assert rms < rms0 * 0.6


# ---------------------------------------------------------------------------
# System identification (plants/sysid)
# ---------------------------------------------------------------------------

def test_identify_cavity_recovers_parameters():
    from llrflibs_rl.configs.cavity import nominal_sc_cavity
    from llrflibs_rl.plants.sysid import identify_cavity

    cavity = nominal_sc_cavity()
    report = identify_cavity(cavity, Ts=1.0e-6, beam_amp=5.0e5, n_prbs=4096)
    assert report.half_bw_rel_error < 0.05
    # The beam estimate on a slow SC cavity is order-of-magnitude accurate
    # (the parent library can only identify beam OR intra-pulse wh/dw).
    assert report.beam_rel_error < 3.0
    assert np.all(np.isfinite(report.etfe_gain))


def test_beam_observer_quantitative_on_fast_cavity(fast_cavity):
    from dataclasses import replace

    from llrflibs_rl.plants.cavity_voltage import CavityVoltagePlant
    from llrflibs_rl.plants.sysid import identify_beam_disturbance

    fast = replace(fast_cavity, half_bw=2.0 * np.pi * 2000.0, detuning=2.0 * np.pi * 500.0)
    Ts = 1.0e-6
    n = 4000
    drive = np.full(n, 1.0e7, dtype=np.complex128)
    beam = 5.0e5 * np.exp(1j * 0.3)
    plant = CavityVoltagePlant(fast, Ts, fast.detuning)
    vc = np.zeros(n, dtype=np.complex128)
    plant.reset()
    for i in range(n):
        vc[i], _ = plant.step(complex(drive[i]), beam=beam)
    vb_est = identify_beam_disturbance(vc[100:], drive[100:], fast.half_bw, -fast.detuning, Ts, pole_scale=50.0)
    rel = abs(float(np.mean(np.abs(vb_est))) - float(beam.real)) / float(abs(beam))
    assert rel < 0.8


# ---------------------------------------------------------------------------
# Calibration (utils/calibration)
# ---------------------------------------------------------------------------

def test_virtual_probe_calibration_recovers_coefficients():
    from llrflibs_rl.utils.calibration import calibrate_virtual_probe

    n = 500
    theta = np.linspace(0.0, 0.6, n)
    vf_m = (1.0 + 0.3 * np.sin(theta)) * np.exp(1j * theta) * 1.0e7
    vr_m = (0.5 + 0.2 * np.cos(2.0 * theta)) * np.exp(1j * 1.3 * theta) * 2.0e6
    vc = 0.8 * vf_m + 0.3j * vr_m
    m, nd = calibrate_virtual_probe(vc, vf_m, vr_m)
    assert abs(m - 0.8) < 1e-9
    assert abs(nd - 0.3j) < 1e-9


def test_vector_sum_calibration_scales_channels():
    from llrflibs_rl.utils.calibration import calibrate_vector_sum

    amp = np.array([1.0, 0.5, 2.0])
    phase_deg = np.array([0.0, 30.0, -20.0])
    scale, phase = calibrate_vector_sum(amp, phase_deg, ref_ch=0)
    assert np.allclose(scale, [1.0, 2.0, 0.5])
    assert np.allclose(phase, [0.0, -30.0, 20.0])


# ---------------------------------------------------------------------------
# Loop stability (evaluation/loop)
# ---------------------------------------------------------------------------

def test_loop_margins_positive_for_pi_loop(fast_cavity):
    from llrflibs_rl.evaluation.loop import cavity_plant_abcd, loop_margins, pi_controller_abcd

    plant = cavity_plant_abcd(fast_cavity)
    # A mild PI loop must be comfortably stable (small peak sensitivity).
    # The cavity+PI loop never crosses -180 deg / 0 dB, so both the gain and
    # phase margins are infinite (reported as None) by design.
    mild = loop_margins(plant, pi_controller_abcd(0.05, 1.0e3), Ts=None, plot_maxf=1.0e6)
    assert mild.gain_margin_db is None and mild.phase_margin_deg is None
    assert mild.s_max_db < 8.0
    # ...while an aggressive one is clearly less robust.
    aggressive = loop_margins(plant, pi_controller_abcd(2.0, 2.0e5), Ts=None, plot_maxf=1.0e6)
    assert aggressive.s_max_db > mild.s_max_db + 3.0
    delayed = loop_margins(plant, pi_controller_abcd(0.05, 1.0e3), Ts=None, delay_s=1.0e-4, plot_maxf=1.0e6)
    assert np.isfinite(delayed.s_max_db)


# ---------------------------------------------------------------------------
# Spectral analysis (evaluation/spectral)
# ---------------------------------------------------------------------------

def test_tracking_error_spectrum_report(fast_cavity, fast_env_config):
    from llrflibs_rl.controllers.baselines import zero_policy
    from llrflibs_rl.evaluation.spectral import tracking_error_spectrum

    factory = lambda: rl.make(  # noqa: E731
        "CavityVoltage-v0", cavity=fast_cavity, env_cfg=fast_env_config
    )
    policy = zero_policy(factory())
    report = tracking_error_spectrum(factory, policy, seed=0, bands=[(50.0, 500.0)])
    assert report.error_rms > 0.0
    assert len(report.bands) == 1
    assert report.bands[0].rms > 0.0
    assert np.isfinite(report.signal_freq_hz)


# ---------------------------------------------------------------------------
# Measurement chain (plants/measurement)
# ---------------------------------------------------------------------------

def test_measurement_chain_delay_and_noise():
    from llrflibs_rl.plants.measurement import MeasurementChain, ap_to_iq, iq_to_ap

    chain = MeasurementChain(delay_steps=2, noise_std=0.0)
    wave = np.ones(8, dtype=np.complex128)
    out = chain.process_wave(wave)
    assert np.all(out[:2] == 0.0) and np.all(out[2:] == 1.0)

    noisy = MeasurementChain(delay_steps=0, noise_std=0.1, seed=3).process_wave(wave)
    assert np.std(noisy.real) > 0.0

    amp = np.ones(8) * 5.0
    phase = np.linspace(0.0, 180.0, 8)
    i_wf, q_wf = ap_to_iq(amp, phase)
    amp2, phase2 = iq_to_ap(i_wf, q_wf)
    assert np.allclose(amp2, amp, atol=1e-9)
    assert np.allclose(np.mod(phase2, 360.0), np.mod(phase, 360.0), atol=1e-6)
