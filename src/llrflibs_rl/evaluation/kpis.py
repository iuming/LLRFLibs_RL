###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Key-performance-indicator (KPI) evaluation for LLRF control.

Real LLRF systems quote stability in amplitude and phase RMS (for example
0.01 % of the voltage target and 0.01 deg on modern SRF systems).  This module
turns those engineering targets into a pass/fail report for any policy:

* :func:`kpi_report` — decompose the complex tracking error of one episode
  into amplitude and phase RMS errors (flattop window) and compare each KPI
  against its threshold.

The catalog is deliberately small so the same thresholds can drive the
evaluation baselines and reward shaping.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = ["KPIS", "kpi_report", "kpi_report_from_wave"]


@dataclass(frozen=True)
class Kpi:
    """One key performance indicator.

    Attributes:
        name:      Display name.
        unit:      Display unit (``%`` or ``deg``).
        threshold: Pass threshold (upper limit).
    """

    name: str
    unit: str
    threshold: float


KPIS: dict[str, Kpi] = {
    "amp_rms_pct": Kpi("Amplitude RMS", "%", 0.01),
    "phase_rms_deg": Kpi("Phase RMS", "deg", 0.01),
    "flattop_error_rms_pct": Kpi("Flattop complex error RMS", "%", 1.0),
}


def _rollout(env, action_fn: Callable[[np.ndarray], np.ndarray], seed: int | None):
    obs, _ = env.reset(seed=seed)
    n = env.episode_len
    tracked = np.zeros(n, dtype=np.complex128)
    length = 0
    for i in range(n):
        action = np.asarray(action_fn(obs), dtype=np.float32)
        obs, _reward, terminated, truncated, _info = env.step(action)
        tracked[i] = env.tracked_signal
        length = i + 1
        if terminated or truncated:
            break
    return tracked[:length]


def kpi_report_from_wave(
    reference_wave: np.ndarray,
    tracked_wave: np.ndarray,
    vc_target: float,
    filling_len: int = 0,
    flattop_len: int | None = None,
) -> dict[str, dict[str, float | bool]]:
    """KPI pass/fail report from reference and tracked complex waveforms.

    Args:
        reference_wave: Complex reference waveform.
        tracked_wave:   Complex tracked waveform (same length).
        vc_target:      Voltage target used to normalize the amplitude error.
        filling_len:    Length of the filling region (skipped in the flattop).
        flattop_len:    Length of the flattop window (defaults to the rest).

    Returns:
        Nested dict ``kpi -> {"value", "threshold", "passed"}``.
    """
    n = min(len(reference_wave), len(tracked_wave))
    ref = np.asarray(reference_wave[:n], dtype=np.complex128)
    trk = np.asarray(tracked_wave[:n], dtype=np.complex128)
    if n == 0 or float(vc_target) <= 0.0:
        raise ValueError("kpi_report_from_wave needs non-empty waveforms and a positive target.")

    amplitude_err = np.abs(trk) - np.abs(ref)
    phase_err = np.angle(trk) - np.angle(ref)
    # Wrap phase error into [-180, 180] deg.
    phase_err_deg = np.degrees(np.arctan2(np.sin(phase_err), np.cos(phase_err)))

    ft_start = int(filling_len)
    ft_end = n if flattop_len is None else min(n, ft_start + int(flattop_len))
    if ft_end <= ft_start:
        ft_start, ft_end = 0, n

    amp_rms_pct = float(np.sqrt(np.mean(amplitude_err[ft_start:ft_end] ** 2)) / vc_target * 100.0)
    phase_rms_deg = float(np.sqrt(np.mean(phase_err_deg[ft_start:ft_end] ** 2)))
    complex_err_pct = float(
        np.sqrt(np.mean(np.abs(ref[ft_start:ft_end] - trk[ft_start:ft_end]) ** 2)) / vc_target * 100.0
    )

    values = {
        "amp_rms_pct": amp_rms_pct,
        "phase_rms_deg": phase_rms_deg,
        "flattop_error_rms_pct": complex_err_pct,
    }
    report: dict[str, dict[str, float | bool]] = {}
    for key, value in values.items():
        kpi = KPIS[key]
        report[key] = {
            "value": value,
            "threshold": kpi.threshold,
            "passed": bool(value <= kpi.threshold),
        }
    return report


def kpi_report(
    env,
    action_fn: Callable[[np.ndarray], np.ndarray],
    seed: int | None = 0,
) -> dict[str, dict[str, float | bool]]:
    """KPI report of a policy on an environment.

    Uses the environment's reference waveform and cavity geometry for the
    flattop window.

    Args:
        env:       An LLRF environment exposing ``reference_sp`` and ``cavity``.
        action_fn: Policy mapping an observation to an action.
        seed:      Optional evaluation seed.

    Returns:
        Nested dict ``kpi -> {"value", "threshold", "passed"}``.
    """
    ref = np.asarray(env.reference_sp, dtype=np.complex128)
    tracked = _rollout(env, action_fn, seed)
    cave = getattr(env, "cavity", None)
    vc_target = float(getattr(cave, "vc_target", 1.0)) if cave is not None else 1.0
    filling = int(getattr(cave, "filling_len", 0)) if cave is not None else 0
    flattop = int(getattr(cave, "flattop_len", 0)) if cave is not None else None
    if flattop == 0:
        flattop = None
    return kpi_report_from_wave(ref, tracked, vc_target, filling, flattop)
