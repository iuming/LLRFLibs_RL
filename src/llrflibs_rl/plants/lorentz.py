###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Dynamic Lorentz-force detuning model.

The Lorentz force detunes a cavity proportionally to the field gradient
squared (the generic formula ``df = -k_lor * V^2``); the mechanical response
acts like a first-order/low-pass dynamic.  This module provides a
sample-wise model that feeds the time-varying detuning path of the cavity
plants, enabling amplitude-dependent intrapulse Lorentz drift on top of any
static/microphonics waveform.
"""

from __future__ import annotations

__all__ = ["LorentzDriftDetuning"]


class LorentzDriftDetuning:
    """First-order Lorentz detuning driven by the cavity amplitude.

    The derivative of the Lorentz detuning is
    ``dL/dt = (k_lor * |vc|^2 - L) / tau``, i.e. ``L`` relaxes towards
    ``k_lor * |vc|^2`` with the mechanical time constant ``tau``.

    Args:
        k_lor: Lorentz coefficient, rad/s per V**2 (negative sign is the
               convention; the returned shift is negative).
        tau:   Mechanical time constant, s (``tau <= Ts`` is treated as
               instantaneous).
        Ts:    Sampling time, s.
    """

    def __init__(self, k_lor: float, tau: float, Ts: float) -> None:
        if k_lor < 0.0:
            raise ValueError("k_lor must be non-negative")
        if tau <= 0.0 or Ts <= 0.0:
            raise ValueError("tau and Ts must be positive")
        self.k_lor = float(k_lor)
        self.tau = float(tau)
        self.Ts = float(Ts)
        self._beta = self.Ts / max(self.tau, self.Ts)  # <= 1 first-order gain
        self.reset()

    def reset(self) -> None:
        """Reset the Lorentz state to zero."""
        self.state = 0.0

    def step(self, vc: complex) -> float:
        """Advance one sample with the current cavity voltage and return the shift.

        Args:
            vc: Cavity voltage of the current sample, V.

        Returns:
            Lorentz detuning contribution, rad/s (negative).
        """
        target = self.k_lor * abs(vc) ** 2
        self.state += self._beta * (target - self.state)
        return -self.state

    def to_dict(self) -> dict:
        """Serializable description."""
        return {"k_lor": self.k_lor, "tau": self.tau, "Ts": self.Ts}
