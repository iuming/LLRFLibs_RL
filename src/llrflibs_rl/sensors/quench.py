###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Quench-onset detection and protection logic.

A superconducting cavity quench appears in the LLRF signals as a fast detuning
growth together with a rising reflected amplitude.  This module provides a
small detector (in the spirit of accelerator quench-detection systems, e.g.
the Sirius QDS) that classifies the *reflected amplitude rate* so an
environment can trip before the full RF power is dumped.  It is wired into the
voltage environments through ``EnvConfig.quench_detect``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["QuenchDetector", "detect_quench_index"]


@dataclass
class QuenchDetector:
    """Sliding-window quench detector on the reflected amplitude.

    Fires when the reflected amplitude growth rate (per sample, smoothed over
    ``window`` samples) exceeds ``rate_threshold`` *and* the amplitude is above
    ``abs_threshold`` (when set).  This catches the fast reflection growth of a
    quench onset while ignoring slow tuning drift.

    Attributes:
        window:         Moving-average window for the rate, samples.
        rate_threshold: Minimum |d|reflected|/dtick| per sample.
        abs_threshold:  Optional absolute reflected amplitude floor, V.
        dt:             Sample time used to turn the per-sample slope into s.
    """

    window: int = 8
    rate_threshold: float = 1.0e6
    abs_threshold: float | None = None
    dt: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError("window must be >= 2.")
        if self.rate_threshold <= 0.0:
            raise ValueError("rate_threshold must be positive.")
        self._buffer: list[float] = []
        self._trip = False

    def reset(self) -> None:
        """Clear the detector state."""
        self._buffer = []
        self._trip = False

    @property
    def tripped(self) -> bool:
        """Whether the detector has already fired."""
        return self._trip

    def update(self, reflected_abs: float) -> bool:
        """Feed one reflected-amplitude sample.

        Args:
            reflected_abs: Current |reflected| value, V.

        Returns:
            ``True`` the first time a quench-like onset is detected.
        """
        if self._trip:
            return True
        self._buffer.append(float(reflected_abs))
        if len(self._buffer) > self.window:
            self._buffer.pop(0)
        if len(self._buffer) < self.window:
            return False
        slope = (self._buffer[-1] - self._buffer[0]) / ((self.window - 1) * self.dt)
        level_ok = self.abs_threshold is None or self._buffer[-1] >= self.abs_threshold
        if slope >= self.rate_threshold and level_ok:
            self._trip = True
        return self._trip


def detect_quench_index(reflect_wave: np.ndarray, window: int = 8, rate_threshold: float = 1.0e6, dt: float = 1.0e-6):
    """First index where a quench-like onset is detected on a waveform.

    Args:
        reflect_wave: Reflected-amplitude waveform, V.
        window:       Rate estimation window, samples.
        rate_threshold: Per-second slope threshold, V/s.
        dt:           Sample time, s.

    Returns:
        Index of the first detection or ``-1`` when no quench is found.
    """
    detector = QuenchDetector(window, rate_threshold, dt=dt)
    for i, value in enumerate(np.asarray(reflect_wave, dtype=float)):
        if detector.update(float(value)):
            return i
    return -1
