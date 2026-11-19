###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Pulse reference generation.

A :class:`PulseReference` pre-computes the set-point (``sp``), the analytical
feedforward drive (``vf_ff``) and the beam-drive waveform (``vb``) of a cavity
pulse with :func:`llrflibs.rf_control.cav_sp_ff`.  All voltage environments
share this object, which guarantees that the reference, the feedforward and
the beam model stay mutually consistent.
"""

from __future__ import annotations

import numpy as np
from llrflibs.rf_control import cav_sp_ff

from ..configs.cavity import CavityParams

__all__ = ["PulseReference"]


class PulseReference:
    """Waveforms of a single-cavity pulse (single-frequency model).

    Args:
        cavity:             Cavity description.
        Ts:                 Sampling time, s.
        episode_len:        Number of samples of the episode.
        reference_detuning: Detuning used to build the waveforms, rad/s.
                            ``None`` uses ``cavity.detuning``.
    """

    def __init__(
        self,
        cavity: CavityParams,
        Ts: float,
        episode_len: int,
        reference_detuning: float | None = None,
    ) -> None:
        self.cavity = cavity
        self.Ts = float(Ts)
        self.episode_len = int(episode_len)
        self.reference_detuning = float(
            cavity.detuning if reference_detuning is None else reference_detuning
        )

        status, sp, vf_ff, vb, _ = cav_sp_ff(
            cavity.half_bw,
            cavity.filling_len,
            cavity.flattop_len,
            self.Ts,
            self.episode_len,
            vc0=cavity.vc_target,
            detuning=self.reference_detuning,
            beta=cavity.beta,
            ib0=cavity.beam_current if cavity.has_beam else None,
            phib_deg=cavity.beam_phase_deg,
            beam_ids=cavity.filling_len,
            beam_ide=min(self.episode_len - 1, cavity.filling_len + cavity.flattop_len - 1),
            roQ_or_RoQ=cavity.roQ_or_RoQ,
            QL=cavity.QL,
            machine=cavity.machine,
        )
        if not status:
            raise ValueError("Failed to generate the cavity voltage reference waveform.")

        self._sp = np.asarray(sp, dtype=np.complex128).reshape(-1)
        self._vf_ff = np.asarray(vf_ff, dtype=np.complex128).reshape(-1)
        self._vb = np.asarray(vb, dtype=np.complex128).reshape(-1)

    # ------------------------------------------------------------------
    # Waveform accessors
    # ------------------------------------------------------------------

    @property
    def sp(self) -> np.ndarray:
        """Set-point waveform (complex)."""
        return self._sp

    @property
    def vf_ff(self) -> np.ndarray:
        """Analytical feedforward drive waveform (complex)."""
        return self._vf_ff

    @property
    def vb(self) -> np.ndarray:
        """Beam-drive waveform (complex)."""
        return self._vb

    @property
    def peak_drive(self) -> float:
        """Maximum magnitude of the analytical feedforward, V."""
        return float(np.max(np.abs(self._vf_ff)))

    @property
    def obs_scale(self) -> float:
        """Voltage normalization scale (the cavity voltage target)."""
        return float(self.cavity.vc_target)

    @property
    def drive_effort_scale(self) -> float:
        """Scale used to normalize the drive-effort term."""
        return float(max(self.peak_drive, self.obs_scale, 1.0))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def sp_at(self, step_index: int) -> complex:
        """Set-point at a sample index (clamped to the episode)."""
        idx = min(int(step_index), self.episode_len - 1)
        return complex(self._sp[idx])

    def vf_ff_at(self, step_index: int) -> complex:
        """Feedforward at a sample index (clamped to the episode)."""
        idx = min(int(step_index), self.episode_len - 1)
        return complex(self._vf_ff[idx])

    def vb_at(self, step_index: int) -> complex:
        """Beam drive at a sample index (clamped to the episode)."""
        idx = min(int(step_index), self.episode_len - 1)
        return complex(self._vb[idx])

    def to_dict(self) -> dict:
        """Serializable description of the reference."""
        return {
            "reference_detuning_rad": self.reference_detuning,
            "episode_len": self.episode_len,
            "Ts": self.Ts,
            "peak_drive": self.peak_drive,
        }
