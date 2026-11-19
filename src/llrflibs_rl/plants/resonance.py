###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Resonance plant (cavity electrical + mechanical modes).

A :class:`ResonancePlant` advances the single-mode cavity together with the
Lorentz-force mechanical modes in one step (via
:func:`llrflibs.rf_sim.sim_scav_step`).  The *external* detuning (tuner
actuator + microphonics) is supplied by the caller, while the mechanical
(Lorentz) contribution is computed internally from the cavity voltage.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from llrflibs.rf_control import ss_discrete
from llrflibs.rf_sim import cav_ss_mech, sim_scav_step

from ..configs.cavity import CavityParams

__all__ = ["ResonancePlant", "DEFAULT_MECH_MODES"]


DEFAULT_MECH_MODES: dict[str, list[float]] = {
    "f": [280.0, 341.0, 460.0, 487.0, 618.0],
    "Q": [40.0, 20.0, 50.0, 80.0, 100.0],
    "K": [2.0, 0.8, 2.0, 0.6, 0.2],
}


def _complex_state_matrix(size: int, dtype: type = float) -> np.matrix:
    return np.matrix(np.zeros((size, 1), dtype=dtype))


class ResonancePlant:
    """Cavity electrical + Lorentz mechanical modes.

    Args:
        cavity:    Cavity description whose ``mech_modes`` (or the class
                   default) define the Lorentz-force mechanical modes.
        Ts:        Sampling time, s.
        mech_modes: Optional override of the mechanical modes.
    """

    def __init__(
        self,
        cavity: CavityParams,
        Ts: float,
        mech_modes: dict[str, Sequence[float]] | None = None,
    ) -> None:
        self.cavity = cavity
        self.Ts = float(Ts)
        self.mech_modes = dict(mech_modes) if mech_modes is not None else dict(DEFAULT_MECH_MODES)

        status, Am, Bm, Cm, Dm = cav_ss_mech(self.mech_modes)
        if not status:
            raise ValueError("Failed to build the mechanical model.")
        status, self._Amd, self._Bmd, self._Cmd, self._Dmd, _ = ss_discrete(
            Am, Bm, Cm, Dm, self.Ts, method="zoh", plot=False
        )
        if not status:
            raise ValueError("Failed to discretize the mechanical model.")

        self.reset()

    # ------------------------------------------------------------------
    # Simulation API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the electrical and mechanical states to zero."""
        self._state_m: np.matrix = _complex_state_matrix(self._Bmd.shape[0], dtype=float)
        self._vc = 0.0 + 0.0j
        self._vr = 0.0 + 0.0j
        self._dw = 0.0

    def step(
        self,
        drive: complex,
        beam: complex,
        external_detuning: float,
    ) -> tuple[complex, complex, float]:
        """Advance the electrical + mechanical system by one sample.

        Args:
            drive:            Forward drive voltage, V.
            beam:             Beam-drive voltage, V.
            external_detuning: Detuning imposed by the tuner actuator and the
                               microphonics disturbance, rad/s.

        Returns:
            Tuple of ``(vc_next, vr_next, dw_next)`` where ``dw_next`` is the
            total detuning (Lorentz + external).
        """
        status, vc_next, vr_next, dw_next, state_m = sim_scav_step(
            self.cavity.half_bw,
            self._dw,
            external_detuning,
            drive,
            beam,
            self._vc,
            self.Ts,
            beta=self.cavity.beta,
            state_m0=self._state_m,
            Am=self._Amd,
            Bm=self._Bmd,
            Cm=self._Cmd,
            Dm=self._Dmd,
            mech_exe=True,
        )
        if not status:
            raise RuntimeError("Resonance plant step failed.")

        self._vc = vc_next
        self._vr = vr_next
        self._dw = float(np.asarray(dw_next).squeeze())
        self._state_m = state_m
        return vc_next, vr_next, self._dw

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
    def detuning(self) -> float:
        """Latest total detuning (Lorentz + external), rad/s."""
        return self._dw
