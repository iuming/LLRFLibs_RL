###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
RF measurement-chain calibration utilities.

Wraps the calibration toolbox ``llrflibs.rf_calib`` (virtual probe, forward
coefficient, DAC offset, I/Q modulator, vector-sum channel matching and system
gain/phase) plus the circle fit from ``llrflibs.rf_fit``.  These are the
classical procedures a real LLRF system runs before closed-loop operation; in
simulation they provide realistic calibration coefficients (e.g. for domain
randomization of observations) and reusable reference implementations.
"""

from __future__ import annotations

import numpy as np
from llrflibs.rf_calib import (
    calib_cav_for,
    calib_dac_offs,
    calib_iqmod,
    calib_sys_gain_pha,
    calib_vprobe,
    calib_vsum_poor,
)
from llrflibs.rf_fit import fit_circle

__all__ = [
    "calibrate_virtual_probe",
    "calibrate_forward_coefficient",
    "calibrate_dac_offset",
    "calibrate_iq_modulator",
    "calibrate_vector_sum",
    "system_gain_phase",
    "fit_iq_circle",
]


def _check(ok: bool, what: str) -> None:
    if not ok:
        raise ValueError(f"llrflibs calibration routine failed: {what}")


def calibrate_virtual_probe(vc: np.ndarray, vf_m: np.ndarray, vr_m: np.ndarray) -> tuple[complex, complex]:
    """Infer ``vc = m * vf_m + n * vr_m`` (virtual probe).

    Args:
        vc:   Cavity probe waveform (complex, reference plane).
        vf_m: Measured forward waveform.
        vr_m: Measured reflected waveform.

    Returns:
        Calibration coefficients ``(m, n)``.
    """
    ok, m, n = calib_vprobe(np.asarray(vc), np.asarray(vf_m), np.asarray(vr_m))
    _check(ok, "calib_vprobe")
    return complex(m), complex(n)


def calibrate_forward_coefficient(
    vc: np.ndarray,
    vf_m: np.ndarray,
    pul_ids: int,
    pul_ide: int,
    half_bw: float,
    detuning: float,
    Ts: float,
) -> complex:
    """Calibrate the cavity forward gain ``vf = a * vf_m``.

    Args:
        vc:       Cavity probe waveform (complex).
        vf_m:     Measured forward waveform.
        pul_ids:  Start of the calibration window.
        pul_ide:  End of the calibration window.
        half_bw:  Cavity half bandwidth, rad/s.
        detuning: Cavity detuning, rad/s.
        Ts:       Sampling time, s.

    Returns:
        Complex forward calibration coefficient.
    """
    ok, a = calib_cav_for(
        np.asarray(vc), np.asarray(vf_m), int(pul_ids), int(pul_ide), float(half_bw), float(detuning), float(Ts)
    )
    _check(ok, "calib_cav_for")
    return complex(a)


def calibrate_dac_offset(
    vdac: np.ndarray,
    viqm: np.ndarray,
    sig_ids: int,
    sig_ide: int,
    leak_ids: int,
    leak_ide: int,
    delay: int = 0,
) -> tuple[float, float]:
    """DAC offset calibration against carrier leakage.

    Args:
        vdac:     DAC output waveform (complex).
        viqm:     I/Q modulator output measurement (complex).
        sig_ids:  Signal-window start.
        sig_ide:  Signal-window end.
        leak_ids: Leakage-window start.
        leak_ide: Leakage-window end.
        delay:    I/Q-vs-DAC delay in samples.

    Returns:
        ``(offset_I, offset_Q)`` to add to the DAC output.
    """
    ok, oi, oq = calib_dac_offs(
        np.asarray(vdac), np.asarray(viqm), int(sig_ids), int(sig_ide), int(leak_ids), int(leak_ide), int(delay)
    )
    _check(ok, "calib_dac_offs")
    return float(oi), float(oq)


def calibrate_iq_modulator(vdac: np.ndarray, viqm: np.ndarray) -> np.ndarray:
    """I/Q modulator imbalance calibration (returns ``inv(M)`` predistortion).

    Args:
        vdac: DAC actuation points (complex).
        viqm: I/Q modulator measured points (complex).

    Returns:
        Inverse imbalance matrix to pre-distort the DAC output.
    """
    ok, inv_m, _, _ = calib_iqmod(np.asarray(vdac), np.asarray(viqm))
    _check(ok, "calib_iqmod")
    return np.asarray(inv_m)


def calibrate_vector_sum(amp: np.ndarray, phase_deg: np.ndarray, ref_ch: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Poor-man's vector-sum channel matching.

    Args:
        amp:      Per-channel amplitudes.
        phase_deg: Per-channel phases, deg.
        ref_ch:   Reference channel index.

    Returns:
        ``(scale, phase_offset_deg)`` per channel.
    """
    ok, scale, phase = calib_vsum_poor(np.asarray(amp), np.asarray(phase_deg), int(ref_ch))
    _check(ok, "calib_vsum_poor")
    return np.asarray(scale), np.asarray(phase)


def system_gain_phase(vact: np.ndarray, vf: np.ndarray, beta: float = 1e4) -> tuple[np.ndarray, np.ndarray]:
    """System gain and phase (drive channel to calibrated forward).

    Args:
        vact: RF actuation (DAC) waveform (complex).
        vf:   Calibrated cavity forward waveform (complex).
        beta: Input coupling factor.

    Returns:
        ``(gain, phase_deg)`` waveforms.
    """
    ok, gain, phase = calib_sys_gain_pha(np.asarray(vact), np.asarray(vf), float(beta))
    _check(ok, "calib_sys_gain_pha")
    return np.asarray(gain), np.asarray(phase)


def fit_iq_circle(i_wf: np.ndarray, q_wf: np.ndarray) -> tuple[float, float, float]:
    """Least-squares circle fit of I/Q samples.

    Args:
        i_wf: I samples.
        q_wf: Q samples.

    Returns:
        ``(x0, y0, radius)`` of the fitted circle.
    """
    ok, x0, y0, r = fit_circle(np.asarray(i_wf), np.asarray(q_wf))
    _check(ok, "fit_circle")
    return float(x0), float(y0), float(r)
