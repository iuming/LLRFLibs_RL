###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Single-cavity voltage plant.

A :class:`CavityVoltagePlant` wraps the discrete state-space cavity model of
``llrflibs.rf_sim.cav_ss`` (optionally with beam loading) and exposes a simple
``reset -> step(drive, beam, detuning)`` interface.  It supports both a fixed
detuning (state-space stepping) and a time-varying detuning (single-mode Euler
stepping, which is what the microphonics / Lorentz-type tasks need).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from llrflibs.rf_control import ss_discrete
from llrflibs.rf_sim import cav_ss, sim_ncav_step, sim_ncav_step_simple

from ..configs.cavity import CavityParams

__all__ = ["CavityVoltagePlant"]


def _complex_state_matrix(size: int, dtype: type = complex) -> np.matrix:
    return np.matrix(np.zeros((size, 1), dtype=dtype))


class CavityVoltagePlant:
    """Interactable cavity-voltage model of one cavity.

    Args:
        cavity:   Cavity description.
        Ts:       Sampling time, s.
        detuning: (Fixed) detuning used to build the discrete plant, rad/s.

    Note:
        When :meth:`step` is called with a ``detuning`` argument, the plant
        switches to the single-mode Euler stepping so the detuning can vary
        sample by sample.  The two stepping methods agree to numerical
        precision at the small ``Ts * half_bw`` typical of this package.
    """

    def __init__(self, cavity: CavityParams, Ts: float, detuning: float) -> None:
        self.cavity = cavity
        self.Ts = float(Ts)
        self.detuning = float(detuning)
        self._build_plant(self.detuning)

        self._state_rf: Optional[np.matrix] = None
        self._state_bm: Optional[np.matrix] = None
        self._vc = 0.0 + 0.0j
        self._vr = 0.0 + 0.0j

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_plant(self, detuning: float) -> None:
        status, Arf, Brf, Crf, Drf, Abm, Bbm, Cbm, Dbm = cav_ss(
            self.cavity.half_bw,
            detuning=detuning,
            beta=self.cavity.beta,
            passband_modes=self.cavity.passband_modes,
            plot=False,
        )
        if not status:
            raise ValueError("Failed to build the cavity plant.")

        status, self._Arfd, self._Brfd, self._Crfd, self._Drfd, _ = ss_discrete(
            Arf, Brf, Crf, Drf, self.Ts, method="zoh", plot=False
        )
        if not status:
            raise ValueError("Failed to discretize the RF cavity model.")

        if self.cavity.has_beam:
            status, self._Abmd, self._Bbmd, self._Cbmd, self._Dbmd, _ = ss_discrete(
                Abm, Bbm, Cbm, Dbm, self.Ts, method="bilinear", plot=False
            )
            if not status:
                raise ValueError("Failed to discretize the beam cavity model.")
        else:
            self._Abmd = self._Bbmd = self._Cbmd = self._Dbmd = None

    # ------------------------------------------------------------------
    # Simulation API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all states to zero (cavity empty)."""
        self._state_rf = _complex_state_matrix(self._Brfd.shape[0])
        self._state_bm = (
            _complex_state_matrix(self._Bbmd.shape[0], dtype=complex)
            if self._Abmd is not None
            else None
        )
        self._vc = 0.0 + 0.0j
        self._vr = 0.0 + 0.0j

    def step(
        self,
        drive: complex,
        beam: Optional[complex] = None,
        detuning: Optional[float] = None,
        half_bw: Optional[float] = None,
    ) -> Tuple[complex, complex]:
        """Advance the cavity one sample.

        Args:
            drive:    Forward drive voltage, V.
            beam:     Beam-drive voltage, V (required when beam loading is on).
            detuning: Instantaneous detuning, rad/s.  When given, the
                      time-varying (Euler) path is used.
            half_bw:  Optional instantaneous half bandwidth, rad/s (used by the
                      time-varying path; defaults to ``cavity.half_bw``).

        Returns:
            Tuple of ``(vc_next, vr_next)``.
        """
        if self.cavity.has_beam:
            vb_step = 0.0 + 0.0j if beam is None else beam
        else:
            vb_step = 0.0 + 0.0j

        if detuning is None:
            status, vc_next, vr_next, state_rf, state_bm = sim_ncav_step(
                self._Arfd,
                self._Brfd,
                self._Crfd,
                self._Drfd,
                drive,
                self._state_rf,
                Abmd=self._Abmd,
                Bbmd=self._Bbmd,
                Cbmd=self._Cbmd,
                Dbmd=self._Dbmd,
                vb_step=vb_step if self._Abmd is not None else None,
                state_bm0=self._state_bm,
            )
            if not status:
                raise RuntimeError("Cavity state-space step failed.")
            self._state_rf = state_rf
            self._state_bm = state_bm
        else:
            hbw = self.cavity.half_bw if half_bw is None else float(half_bw)
            status, vc_next, vr_next = sim_ncav_step_simple(
                hbw,
                detuning,
                drive,
                vb_step,
                self._vc,
                self.Ts,
                beta=self.cavity.beta,
            )
            if not status:
                raise RuntimeError("Cavity simple-step failed.")

        self._vc = vc_next
        self._vr = vr_next
        return vc_next, vr_next

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vc(self) -> complex:
        """Latest cavity voltage."""
        return self._vc

    @property
    def vr(self) -> complex:
        """Latest reflected voltage."""
        return self._vr

    @property
    def has_beam(self) -> bool:
        """Whether beam loading is enabled."""
        return self.cavity.has_beam
