###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Cavity system identification from simulated machine data.

Wraps the identification toolbox in ``llrflibs.rf_sysid`` (PRBS excitation,
empirical transfer function estimation, decay-based half-bandwidth/detuning
fitting and the ADRC beam-loading observer) into a single pipeline that works
directly on the package's plants:

* :func:`identify_cavity` — drive a :class:`CavityVoltagePlant` with a PRBS
  open-loop stimulus, fit the half bandwidth / detuning from the cavity decay
  and estimate the beam drive with the disturbance observer; returns identified
  values next to the ground truth.

The same routines are used by the ``sysid`` CLI subcommand for a quick
"how well can we identify this simulated machine?" report.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from llrflibs.rf_sysid import (
    cav_beam_pulse_obs,
    detuning_decay,
    etfe,
    half_bw_decay,
    iden_impulse,
    prbs,
)

from ..configs.cavity import CavityParams
from .cavity_voltage import CavityVoltagePlant

__all__ = [
    "prbs_wave",
    "etfe_estimate",
    "identify_decay",
    "identify_beam_disturbance",
    "identify_impulse_response",
    "identify_cavity",
    "SysIdReport",
]


def _check(ok: bool, what: str) -> None:
    if not ok:
        raise ValueError(f"llrflibs sysid routine failed: {what}")


def prbs_wave(n: int, lower: float = -1.0, upper: float = 1.0) -> np.ndarray:
    """Deterministic PRBS excitation signal.

    Args:
        n:     Number of samples.
        lower: Lower level.
        upper: Upper level.

    Returns:
        PRBS waveform.
    """
    ok, data = prbs(int(n), lower_b=float(lower), upper_b=float(upper))
    _check(ok, "prbs")
    return np.asarray(data)


def etfe_estimate(input_wave: np.ndarray, output_wave: np.ndarray, fs: float) -> tuple:
    """Empirical transfer function estimation (wrapper).

    Non-finite bins (e.g. the division at the DC bin where the PRBS spectrum
    vanishes) are replaced by zero.

    Args:
        input_wave:  Input waveform.
        output_wave: Output waveform.
        fs:          Sampling frequency, Hz.

    Returns:
        Tuple ``(freq, G)`` of frequency axis and complex response.
    """
    ok, freq, g, _, _ = etfe(
        np.asarray(input_wave, dtype=np.complex128),
        np.asarray(output_wave, dtype=np.complex128),
        r=1,
        fs=float(fs),
    )
    _check(ok, "etfe")
    g = np.asarray(g)
    g[~np.isfinite(g)] = 0.0
    return freq, g


def identify_decay(
    amp_wave: np.ndarray,
    phase_wave_deg: np.ndarray,
    decay_ids: int,
    decay_ide: int,
    Ts: float,
) -> tuple[float, float]:
    """Fit half bandwidth and detuning from the cavity free decay.

    Args:
        amp_wave:      Cavity amplitude waveform.
        phase_wave_deg: Cavity phase waveform, deg.
        decay_ids:     Start index of the decay window.
        decay_ide:     End index of the decay window.
        Ts:            Sampling time, s.

    Returns:
        Tuple ``(half_bw, detuning)`` in rad/s.
    """
    ok, half_bw = half_bw_decay(amp_wave, int(decay_ids), int(decay_ide), float(Ts))
    _check(ok, "half_bw_decay")
    ok, detuning = detuning_decay(phase_wave_deg, int(decay_ids), int(decay_ide), float(Ts))
    _check(ok, "detuning_decay")
    return float(half_bw), float(detuning)


def identify_beam_disturbance(
    vc_wave: np.ndarray,
    vf_wave: np.ndarray,
    half_bw: float,
    detuning: float,
    Ts: float,
    beta: float = 1e4,
    pole_scale: float = 20.0,
) -> np.ndarray:
    """Estimate the beam drive waveform with the ADRC disturbance observer.

    The observer assumes the intra-pulse half bandwidth and detuning are known
    (as the parent LLRF book notes, beam and intra-pulse ``wh``/``dw`` cannot be
    identified simultaneously).  ``detuning`` is passed with the sign expected
    by ``cav_beam_pulse_obs`` (negated w.r.t. the phase-decay measurement).
    The estimate is quantitatively reliable once the cavity has settled (fast
    / normal-conducting regimes); for slow superconducting cavities treat it
    as an order-of-magnitude check.

    Args:
        vc_wave:    Cavity voltage waveform (complex).
        vf_wave:    Cavity drive waveform (complex).
        half_bw:    Intra-pulse half bandwidth, rad/s.
        detuning:   Intra-pulse detuning, rad/s (sign per the observer).
        Ts:         Sampling time, s.
        beta:       Input coupling factor.
        pole_scale: Observer pole scaling.

    Returns:
        Estimated beam-drive waveform (complex).
    """
    n = vc_wave.shape[0]
    wh_pul = np.full(n, float(half_bw), dtype=float)
    dw_pul = np.full(n, float(detuning), dtype=float)
    ok, vb_est, _, _ = cav_beam_pulse_obs(
        np.asarray(vc_wave, dtype=np.complex128),
        np.asarray(vf_wave, dtype=np.complex128),
        wh_pul,
        dw_pul,
        float(half_bw),
        float(Ts),
        beta=float(beta),
        pole_scale=float(pole_scale),
    )
    _check(ok, "cav_beam_pulse_obs")
    return np.asarray(vb_est)


def _run_plant(plant, drive_wave: np.ndarray, beam=None) -> np.ndarray:
    """Simulate ``drive_wave`` through a plant, returning the cavity voltage."""
    n = drive_wave.shape[0]
    vc = np.zeros(n, dtype=np.complex128)
    plant.reset()
    for i in range(n):
        vc[i], _ = plant.step(complex(drive_wave[i]), beam=beam)
    return vc


def identify_impulse_response(input_waves: np.ndarray, output_waves: np.ndarray, order: int = 20) -> np.ndarray:
    """Least-squares impulse response from ensembles of I/O waveforms.

    Args:
        input_waves:  ``(n_wf, N)`` complex input waveforms.
        output_waves: ``(n_wf, N)`` complex output waveforms.
        order:        Impulse response order.

    Returns:
        Identified complex impulse response.
    """
    ok, h = iden_impulse(np.asarray(input_waves), np.asarray(output_waves), int(order))
    _check(ok, "iden_impulse")
    return np.asarray(h)


@dataclass
class SysIdReport:
    """Summary of one cavity identification run.

    Attributes:
        half_bw_identified:   Fitted half bandwidth, rad/s.
        half_bw_truth:        True half bandwidth, rad/s.
        half_bw_rel_error:    Relative identification error.
        detuning_identified:  Fitted detuning, rad/s.
        detuning_truth:       True detuning, rad/s.
        detuning_rel_error:   Relative identification error.
        beam_amp_identified:  Estimated beam-drive amplitude, V.
        beam_amp_truth:       True beam-drive amplitude, V.
        beam_rel_error:       Relative beam estimation error.
        etfe_freq:            ETFE frequency axis, Hz.
        etfe_gain:            ETFE complex response.
    """

    half_bw_identified: float
    half_bw_truth: float
    half_bw_rel_error: float
    detuning_identified: float
    detuning_truth: float
    detuning_rel_error: float
    beam_amp_identified: float
    beam_amp_truth: float
    beam_rel_error: float
    etfe_freq: np.ndarray
    etfe_gain: np.ndarray

    def as_dict(self) -> dict[str, object]:
        """JSON-friendly summary (arrays excluded)."""
        return {
            "half_bw_identified": self.half_bw_identified,
            "half_bw_truth": self.half_bw_truth,
            "half_bw_rel_error": self.half_bw_rel_error,
            "detuning_identified": self.detuning_identified,
            "detuning_truth": self.detuning_truth,
            "detuning_rel_error": self.detuning_rel_error,
            "beam_amp_identified": self.beam_amp_identified,
            "beam_amp_truth": self.beam_amp_truth,
            "beam_rel_error": self.beam_rel_error,
        }


def identify_cavity(
    cavity: CavityParams,
    Ts: float,
    beam_amp: float = 1.0e6,
    n_prbs: int = 4096,
    prbs_scale: float = 1.0e7,
    fill_len: int = 400,
    decay_ids: int = 20,
    decay_ide: int = 160,
) -> SysIdReport:
    """Identify half-bw / detuning / beam of a simulated cavity plant.

    The identification runs entirely on the package's
    :class:`CavityVoltagePlant`: a PRBS open-loop stimulus is used for the ETFE,
    a filled-pulse free decay for half-bw/detuning, and the ADRC observer for
    the beam drive.

    Args:
        cavity:     Cavity parameters (ground truth).
        Ts:         Sampling time, s.
        beam_amp:   Beam-drive amplitude used in the observer step, V.
        n_prbs:     Number of PRBS samples.
        prbs_scale: Drive amplitude of the PRBS, V.
        fill_len:   Pre-pulse length (rad/s names from RF book) of the decay run.
        decay_ids:  Decay window start (samples after the pulse end).
        decay_ide:  Decay window end.

    Returns:
        A :class:`SysIdReport`.
    """
    plant = CavityVoltagePlant(cavity, float(Ts), float(cavity.detuning))

    # 1) ETFE: PRBS through the plant.
    drive = prbs_wave(n_prbs, -prbs_scale, prbs_scale).astype(np.complex128)
    vc_prbs = np.zeros(n_prbs, dtype=np.complex128)
    plant.reset()
    for i in range(n_prbs):
        vc_prbs[i], _ = plant.step(complex(drive[i]))
    freq, g = etfe_estimate(drive, vc_prbs, fs=1.0 / float(Ts))

    # 2) Decay fit: fill pulse then free decay.
    n_total = fill_len + decay_ide + 40
    drive_wave = np.zeros(n_total, dtype=np.complex128)
    drive_wave[:fill_len] = prbs_scale
    vc_amp = np.zeros(n_total)
    vc_pha = np.zeros(n_total)
    plant.reset()
    for i in range(n_total):
        vc, _ = plant.step(complex(drive_wave[i]))
        vc_amp[i] = abs(vc)
        vc_pha[i] = np.angle(vc, deg=True)
    half_bw, detuning = identify_decay(vc_amp, vc_pha, fill_len + decay_ids, fill_len + decay_ide, Ts)

    # 3) Beam observer: use a long run so the slow SC cavity reaches a settled
    #    window, then estimate with the identified constant profiles.
    n_beam = 12000
    late = slice(8000, 12000)
    beam = beam_amp * complex(np.cos(0.3), np.sin(0.3))
    drive_b = np.full(n_beam, prbs_scale * 0.5, dtype=np.complex128)

    vc_off = _run_plant(plant, drive_b)
    vc_on = _run_plant(plant, drive_b, beam=beam)
    vf_b = drive_b

    # Beam off + beam on probe pulse for the observer profiles (unused in the
    # constant-profile estimate but kept as a consistency check).
    from llrflibs.rf_sysid import cav_par_pulse_obs

    ok, _wh_pul, _dw_pul, _, _ = cav_par_pulse_obs(
        vc_off, vf_b, half_bw, float(Ts), beta=1e4
    )
    _check(ok, "cav_par_pulse_obs")
    # For a constant beam the observer converges on the settled flattop; the
    # detuning enters with the sign expected by ``cav_beam_pulse_obs``.
    vb_est = identify_beam_disturbance(
        vc_on[late], vf_b[late], half_bw, -detuning, float(Ts), pole_scale=50.0
    )
    beam_amp_est = float(np.mean(np.abs(vb_est)))

    return SysIdReport(
        half_bw_identified=half_bw,
        half_bw_truth=float(cavity.half_bw),
        half_bw_rel_error=abs(half_bw - float(cavity.half_bw)) / float(cavity.half_bw),
        detuning_identified=detuning,
        detuning_truth=float(cavity.detuning),
        detuning_rel_error=abs(detuning - float(cavity.detuning)) / max(abs(float(cavity.detuning)), 1.0),
        beam_amp_identified=float(beam_amp_est),
        beam_amp_truth=float(beam_amp),
        beam_rel_error=abs(beam_amp_est - float(beam_amp)) / float(beam_amp),
        etfe_freq=freq,
        etfe_gain=g,
    )
