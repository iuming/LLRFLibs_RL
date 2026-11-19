###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Closed-loop stability analysis (classical LLRF loop design).

Wraps ``llrflibs.rf_control`` (:func:`loop_analysis`, :func:`ss_cascade`,
:func:`ss_freqresp`) and ``llrflibs.rf_sim.cav_ss`` to report the robustness
metrics of a cavity + controller loop:

* gain margin and phase margin of the negative-feedback loop ``L = -P * C``;
* peak sensitivity ``S_max`` and complementary sensitivity ``T_max`` (dB)

This is the classical counterpart of the RL evaluation: a PI/ADRC assisted
inner loop used by the environments can be checked for stability margins
(including loop delay), and the same routine serves as a per-environment
diagnostic (``loop`` CLI).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from llrflibs.rf_control import loop_analysis, ss_cascade, ss_freqresp
from llrflibs.rf_sim import cav_ss

from ..configs.cavity import CavityParams

__all__ = [
    "cavity_plant_abcd",
    "pi_controller_abcd",
    "loop_margins",
    "LoopReport",
]


def _check(ok: bool, what: str) -> None:
    if not ok:
        raise ValueError(f"llrflibs loop routine failed: {what}")


def cavity_plant_abcd(cavity: CavityParams, detuning: float = 0.0):
    """Continuous state-space of the cavity RF path.

    Args:
        cavity:   Cavity parameters.
        detuning: Analysis detuning (rad/s), defaults to 0.

    Returns:
        Tuple ``(A, B, C, D)`` complex matrices of the RF path.
    """
    ok, arf, brf, crf, drf, *_ = cav_ss(
        cavity.half_bw, detuning=detuning, beta=cavity.beta, passband_modes=cavity.passband_modes, plot=False
    )
    _check(ok, "cav_ss")
    return arf, brf, crf, drf


def pi_controller_abcd(kp: float, ki: float, notch_conf: dict | None = None):
    """Continuous state-space of the basic complex RF controller.

    Args:
        kp:        Proportional gain.
        ki:        Integral gain.
        notch_conf: Optional ``notch_conf`` dict (see
                     ``PIControllerConfig.notch_conf``).

    Returns:
        Tuple ``(A, B, C, D)`` complex matrices.
    """
    from llrflibs.rf_control import basic_rf_controller

    ok, akc, bkc, ckc, dkc = basic_rf_controller(float(kp), float(ki), notch_conf=notch_conf, plot=False)
    _check(ok, "basic_rf_controller")
    return akc, bkc, ckc, dkc


def _margins_from_loop(freq: np.ndarray, l_neg: np.ndarray) -> dict[str, float | None]:
    """Gain / phase margins from the negative-feedback open loop response.

    Args:
        freq:  Frequency axis, Hz (may include negative frequencies).
        l_neg: Open-loop response ``-P*C`` (complex).

    Returns:
        Dict with ``gain_margin_db``, ``phase_margin_deg``, ``crossover_hz``.
    """
    pos = freq > 0.0
    f_pos = freq[pos]
    l_pos = l_neg[pos]
    phase = np.angle(l_pos)  # wrapped to [-pi, pi]
    mag_db = 20.0 * np.log10(np.abs(l_pos) + 1e-300)

    gain_margin_db: float | None = None
    gm_freq: float | None = None
    # Gain margin: first phase crossing of -180 deg (phase rising through -pi).
    for i in range(1, len(f_pos)):
        if phase[i - 1] < -np.pi and phase[i] >= -np.pi:
            # interpolate frequency
            t = (-np.pi - phase[i - 1]) / (phase[i] - phase[i - 1])
            f_gm = f_pos[i - 1] + t * (f_pos[i] - f_pos[i - 1])
            mg = mag_db[i - 1] + t * (mag_db[i] - mag_db[i - 1])
            gain_margin_db = float(-mg)
            gm_freq = float(f_gm)
            break

    phase_margin_deg: float | None = None
    pm_freq: float | None = None
    # Phase margin: magnitude crossing 0 dB (rising).
    for i in range(1, len(f_pos)):
        if mag_db[i - 1] < 0.0 <= mag_db[i]:
            t = -mag_db[i - 1] / (mag_db[i] - mag_db[i - 1])
            f_pm = f_pos[i - 1] + t * (f_pos[i] - f_pos[i - 1])
            ph = phase[i - 1] + t * (phase[i] - phase[i - 1])
            phase_margin_deg = float(np.degrees(ph + np.pi))
            pm_freq = float(f_pm)
            break

    return {
        "gain_margin_db": gain_margin_db,
        "phase_margin_deg": phase_margin_deg,
        "gain_margin_freq_hz": gm_freq,
        "phase_margin_freq_hz": pm_freq,
    }


@dataclass
class LoopReport:
    """Stability metrics of a cavity + controller loop.

    Attributes:
        gain_margin_db:  Gain margin, dB (``None`` when never unstable).
        phase_margin_deg: Phase margin, deg (``None`` when no 0 dB crossing).
        s_max_db:        Peak sensitivity, dB.
        t_max_db:        Peak complementary sensitivity, dB.
        crossover_hz:    Gain-crossover frequency, Hz (``None`` if none).
        delay_s:         Loop delay used in the analysis, s.
    """

    gain_margin_db: float | None
    phase_margin_deg: float | None
    s_max_db: float
    t_max_db: float
    crossover_hz: float | None = None
    delay_s: float = 0.0

    def as_dict(self) -> dict[str, object]:
        """JSON-friendly summary."""
        return {
            "gain_margin_db": self.gain_margin_db,
            "phase_margin_deg": self.phase_margin_deg,
            "s_max_db": self.s_max_db,
            "t_max_db": self.t_max_db,
            "crossover_hz": self.crossover_hz,
            "delay_s": self.delay_s,
        }


def loop_margins(
    plant_abcd,
    controller_abcd,
    *,
    Ts: float | None = None,
    delay_s: float = 0.0,
    plot_maxf: float = 0.0,
) -> LoopReport:
    """Analyse a cavity + controller loop.

    Args:
        plant_abcd:      Plant state-space ``(A, B, C, D)``.
        controller_abcd: Controller state-space ``(A, B, C, D)``.
        Ts:              Sampling time (``None`` = continuous).
        delay_s:         Loop transport delay, s.
        plot_maxf:       Analysis bandwidth, Hz (0 = automatic).

    Returns:
        A :class:`LoopReport`.
    """
    ag, bg, cg, dg = plant_abcd
    ak, bk, ck, dk = controller_abcd

    ok, al, bl, cl, dl, _ = ss_cascade(ak, bk, ck, dk, ag, bg, cg, dg, Ts=Ts)
    _check(ok, "ss_cascade")

    ok, s_max, t_max = loop_analysis(
        ag, bg, cg, dg, ak, bk, ck, dk, Ts=Ts, delay_s=delay_s, plot=False, plot_maxf=plot_maxf
    )
    _check(ok, "loop_analysis")

    result = ss_freqresp(al, bl, cl, dl, Ts=Ts, plot=False, plot_pno=50000, plot_maxf=plot_maxf)
    ok, f_wf, _, _, resp = result[0], result[1], result[2], result[3], result[4]
    _check(ok, "ss_freqresp")
    resp = np.asarray(resp) * np.exp(-1j * 2.0 * np.pi * np.asarray(f_wf) * float(delay_s))

    margins = _margins_from_loop(np.asarray(f_wf), -resp)
    return LoopReport(
        gain_margin_db=margins["gain_margin_db"],
        phase_margin_deg=margins["phase_margin_deg"],
        s_max_db=float(s_max),
        t_max_db=float(t_max),
        crossover_hz=margins["gain_margin_freq_hz"],
        delay_s=float(delay_s),
    )
