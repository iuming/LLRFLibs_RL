###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Cavity parameter container and machine presets.

A :class:`CavityParams` object is the single source of truth that describes a
RF cavity to be simulated and controlled.  The same object is consumed by the
plant layer (:mod:`llrflibs_rl.plants`), the reference generator
(:mod:`llrflibs_rl.tasks`) and the environment layer
(:mod:`llrflibs_rl.envs`), so all of them stay consistent by construction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["CavityParams", "nominal_sc_cavity", "nominal_nc_cavity"]


@dataclass
class CavityParams:
    """Physical parameters of the cavity to simulate.

    Attributes:
        half_bw:        Half bandwidth of the fundamental mode, rad/s.
        vc_target:      Desired flattop cavity voltage, V.
        detuning:       Static detuning, rad/s.
        beta:           Input coupling factor (use a large value for SC cavities).
        roQ_or_RoQ:     r/Q (linac) or R/Q (circular), Ohm.
        QL:             Loaded quality factor.
        machine:        ``'linac'`` or ``'circular'``.
        filling_len:    Number of samples of the cavity filling time.
        flattop_len:    Number of samples of the flattop.
        beam_current:   Average beam current, A (0 disables beam loading).
        beam_phase_deg: Beam accelerating phase (0 = on-crest), deg.
        passband_modes: Optional dict with ``freq_offs`` (Hz), ``gain_rel`` and
                        ``half_bw`` (rad/s) lists describing pass-band modes.
        mech_modes:     Optional dict with ``f`` (Hz), ``Q`` and ``K`` lists
                        describing mechanical modes (used by resonance envs).
    """

    half_bw: float
    vc_target: float = 25e6
    detuning: float = 0.0
    beta: float = 1e4
    roQ_or_RoQ: float = 1036.0
    QL: float = 3e6
    machine: str = "linac"
    filling_len: int = 510
    flattop_len: int = 800
    beam_current: float = 0.0
    beam_phase_deg: float = 0.0
    passband_modes: dict[str, Sequence[float]] | None = None
    mech_modes: dict[str, Sequence[float]] | None = None

    def __post_init__(self) -> None:
        """Validate the parameter ranges eagerly so that errors surface early."""
        if self.half_bw <= 0:
            raise ValueError("half_bw must be positive.")
        if self.vc_target <= 0:
            raise ValueError("vc_target must be positive.")
        if self.beta <= 0:
            raise ValueError("beta must be positive.")
        if self.machine not in ("linac", "circular"):
            raise ValueError("machine must be 'linac' or 'circular'.")
        if self.beam_current < 0:
            raise ValueError("beam_current must be non-negative.")
        if self.beam_current > 0 and (self.roQ_or_RoQ <= 0 or self.QL <= 0):
            raise ValueError("roQ_or_RoQ and QL must be positive when beam is present.")
        if self.filling_len <= 0 or self.flattop_len <= 0:
            raise ValueError("filling_len and flattop_len must be positive.")

    @property
    def has_beam(self) -> bool:
        """Whether beam loading is enabled."""
        return self.beam_current > 0.0

    @property
    def has_passband_modes(self) -> bool:
        """Whether pass-band modes are defined."""
        return self.passband_modes is not None

    @property
    def has_mech_modes(self) -> bool:
        """Whether mechanical modes are defined."""
        return self.mech_modes is not None

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dict (lists instead of tuples)."""
        data = {
            k: (list(v) if isinstance(v, (tuple, list)) and not isinstance(v, str) else v)
            for k, v in self.__dict__.items()
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CavityParams:
        """Reconstruct from the output of :meth:`to_dict`."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def _default_mech_modes() -> dict[str, list[float]]:
    """Return the default Lorentz-force mechanical modes of an SC cavity."""
    return {
        "f": [280.0, 341.0, 460.0, 487.0, 618.0],
        "Q": [40.0, 20.0, 50.0, 80.0, 100.0],
        "K": [2.0, 0.8, 2.0, 0.6, 0.2],
    }


def nominal_sc_cavity(
    vc_target: float = 25e6,
    detuning: float = 2.0 * np.pi * 500.0,
    beam_current: float = 8e-3,
    beam_phase_deg: float = 0.0,
    filling_len: int = 510,
    flattop_len: int = 800,
    with_mech_modes: bool = True,
) -> CavityParams:
    """Build a nominal superconducting (SC) linac cavity preset.

    The static detuning is chosen at 500 Hz so that the analytical feedforward
    stays within a physically reachable drive budget (about 2.5x the cavity
    voltage), which keeps the voltage setpoint reachable under saturation
    limits.

    Args:
        vc_target:      Desired flattop cavity voltage, V.
        detuning:       Static plant detuning, rad/s.
        beam_current:   Average beam current, A.
        beam_phase_deg: Beam accelerating phase, deg.
        filling_len:    Filling length in samples.
        flattop_len:    Flattop length in samples.
        with_mech_modes: Whether to attach default mechanical (Lorentz) modes.

    Returns:
        A configured :class:`CavityParams`.
    """
    modes: dict[str, list[float]] | None = _default_mech_modes() if with_mech_modes else None
    return CavityParams(
        half_bw=2.0 * np.pi * 216.0 * 0.5,
        vc_target=vc_target,
        detuning=detuning,
        beta=1e4,
        roQ_or_RoQ=1036.0,
        QL=3e6,
        machine="linac",
        filling_len=filling_len,
        flattop_len=flattop_len,
        beam_current=beam_current,
        beam_phase_deg=beam_phase_deg,
        mech_modes=modes,
    )


def nominal_nc_cavity(
    vc_target: float = 15e6,
    detuning: float = 0.0,
    beam_current: float = 0.0,
    beam_phase_deg: float = 0.0,
    filling_len: int = 300,
    flattop_len: int = 700,
) -> CavityParams:
    """Build a nominal normal-conducting (NC) linac cavity preset.

    Args:
        vc_target:      Desired flattop cavity voltage, V.
        detuning:       Static plant detuning, rad/s.
        beam_current:   Average beam current, A.
        beam_phase_deg: Beam accelerating phase, deg.
        filling_len:    Filling length in samples.
        flattop_len:    Flattop length in samples.

    Returns:
        A configured :class:`CavityParams`.
    """
    return CavityParams(
        half_bw=2.0 * np.pi * 216.0 * 0.5,
        vc_target=vc_target,
        detuning=detuning,
        beta=1.0,
        roQ_or_RoQ=1036.0,
        QL=1e5,
        machine="linac",
        filling_len=filling_len,
        flattop_len=flattop_len,
        beam_current=beam_current,
        beam_phase_deg=beam_phase_deg,
    )
