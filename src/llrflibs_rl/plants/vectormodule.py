###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Vector-sum (multi-cavity) plant.

A :class:`VectorModulePlant` models a set of cavities fed by a single shared
RF forward drive (one klystron / SSA).  Each cavity has its own plant and its
own independently-randomized detuning; the controlled quantity is the complex
vector sum of the cavity voltages.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from ..configs.cavity import CavityParams
from .cavity_voltage import CavityVoltagePlant

__all__ = ["VectorModulePlant"]


class VectorModulePlant:
    """Vector-sum plant of a multi-cavity module.

    Args:
        cavities:            One ``CavityParams`` per cavity.  All cavities
                             must share the RF layout (half bandwidth, voltage
                             target, beta, pulse timing and beam loading) so
                             that the shared-drive feedforward stays
                             well-defined.
        Ts:                  Sampling time, s.
        reference_detuning:  Detuning used to build the initial plants
                             (nominal), rad/s.
    """

    def __init__(
        self,
        cavities: Sequence[CavityParams],
        Ts: float,
        reference_detuning: float,
    ) -> None:
        if not cavities:
            raise ValueError("cavities must not be empty.")
        self.cavities = list(cavities)
        self.n_cavities = len(self.cavities)
        self.Ts = float(Ts)
        self.reference_detuning = float(reference_detuning)
        self._validate_layout()

        self._plants: List[CavityVoltagePlant] = []
        self.reset([reference_detuning] * self.n_cavities)
        self._vs = 0.0 + 0.0j
        self._vr_sum = 0.0 + 0.0j

    # ------------------------------------------------------------------
    # Layout validation
    # ------------------------------------------------------------------

    def _validate_layout(self) -> None:
        ref = self.cavities[0]
        for cav in self.cavities[1:]:
            if not np.isclose(cav.half_bw, ref.half_bw):
                raise ValueError("all cavities must share the same half_bw.")
            if not np.isclose(cav.vc_target, ref.vc_target):
                raise ValueError("all cavities must share the same vc_target.")
            if cav.filling_len != ref.filling_len or cav.flattop_len != ref.flattop_len:
                raise ValueError("all cavities must share the same filling/flattop length.")
            if not np.isclose(cav.beta, ref.beta):
                raise ValueError("all cavities must share the same beta.")
            if not np.isclose(cav.beam_current, ref.beam_current):
                raise ValueError("all cavities must share the same beam_current.")
            if not np.isclose(cav.beam_phase_deg, ref.beam_phase_deg):
                raise ValueError("all cavities must share the same beam_phase_deg.")

    # ------------------------------------------------------------------
    # Simulation API
    # ------------------------------------------------------------------

    def reset(self, detunings: Sequence[float]) -> None:
        """Rebuild every cavity plant with the given per-cavity detunings.

        The per-cavity detuning may be independently randomized by the caller
        (e.g. in an environment's ``reset``).

        Args:
            detunings:  Detuning (rad/s) of every cavity, same order as
                        ``cavities``.
        """
        if len(detunings) != self.n_cavities:
            raise ValueError("detunings length must match the number of cavities.")
        self.detunings = [float(d) for d in detunings]
        self._plants = [
            CavityVoltagePlant(cav, self.Ts, det) for cav, det in zip(self.cavities, self.detunings)
        ]
        for plant in self._plants:
            plant.reset()
        self._vs = 0.0 + 0.0j
        self._vr_sum = 0.0 + 0.0j
        self._vc_list = [0.0 + 0.0j] * self.n_cavities
        self._vr_list = [0.0 + 0.0j] * self.n_cavities

    def step(
        self,
        drive: complex,
        beam: complex,
    ) -> Tuple[complex, complex, List[complex], List[complex]]:
        """Advance every cavity with the shared drive.

        Args:
            drive:  Shared forward drive, V.
            beam:   Beam-drive voltage, V (identical for all cavities).

        Returns:
            Tuple of ``(vs, vr_sum, vc_list, vr_list)``.
        """
        vc_list: List[complex] = []
        vr_list: List[complex] = []
        vs = 0.0 + 0.0j
        vr_sum = 0.0 + 0.0j
        for plant in self._plants:
            vc_next, vr_next = plant.step(drive, beam)
            vc_list.append(vc_next)
            vr_list.append(vr_next)
            vs += vc_next
            vr_sum += vr_next
        self._vs = vs
        self._vr_sum = vr_sum
        self._vc_list = vc_list
        self._vr_list = vr_list
        return vs, vr_sum, vc_list, vr_list

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vector_sum(self) -> complex:
        """Latest vector sum of the cavity voltages."""
        return self._vs

    @property
    def reflected_sum(self) -> complex:
        """Latest sum of the reflected voltages."""
        return self._vr_sum

    @property
    def vc_list(self) -> List[complex]:
        """Latest per-cavity voltages (one entry per cavity)."""
        return list(self._vc_list)

    @property
    def vr_list(self) -> List[complex]:
        """Latest per-cavity reflected voltages."""
        return list(self._vr_list)

    @property
    def vs_target(self) -> float:
        """Sum of the individual cavity voltage targets."""
        return float(sum(cav.vc_target for cav in self.cavities))
