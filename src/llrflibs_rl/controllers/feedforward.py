###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Feedforward controllers: analytical feedforward and a pulse-to-pulse ILC.

The analytical feedforward simply re-exposes the reference waveform.  The
:class:`PTypeILC` (P-type iterative learning control) learns a per-sample
drive correction over repeated pulses; it is the classical "data-driven
feedforward" baseline that RL residual controllers are compared against.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..tasks.reference import PulseReference

__all__ = ["AnalyticFeedforward", "PTypeILC"]


class AnalyticFeedforward:
    """Thin adapter that serves the nominal feedforward of a reference.

    Args:
        reference: Pulse reference owning the nominal feedforward waveform.
    """

    def __init__(self, reference: PulseReference) -> None:
        self.reference = reference

    def drive_at(self, step_index: int) -> complex:
        """Return the nominal feedforward at a sample index.

        Args:
            step_index: Sample index.

        Returns:
            Feedforward drive, V.
        """
        return self.reference.vf_ff_at(step_index)


class PTypeILC:
    """P-type iterative learning controller.

    The correction lookup table is updated between pulses as::

        correction_learned[i] += gain * clamped_error[i]

    Args:
        episode_len:  Number of samples per pulse.
        gain:         ILC learning gain (stability requires modest gains).
        drive_limit:  Optional magnitude limit of the learned correction, V.
    """

    def __init__(
        self,
        episode_len: int,
        gain: float = 0.3,
        drive_limit: Optional[float] = None,
    ) -> None:
        self.episode_len = int(episode_len)
        self.gain = float(gain)
        self.drive_limit = float(drive_limit) if drive_limit else None
        self.reset()

    def reset(self) -> None:
        """Clear the learned correction table."""
        self.correction = np.zeros(self.episode_len, dtype=np.complex128)

    def learn(self, error_wave: np.ndarray) -> None:
        """Update the correction table from one pulse's tracking error.

        Args:
            error_wave: 1-D complex error waveform ``sp - vc``.
        """
        n = min(len(self.correction), len(error_wave))
        update = self.gain * np.asarray(error_wave[:n])
        new_correction = self.correction[:n] + update
        if self.drive_limit is not None:
            mag = np.abs(new_correction)
            scale = np.clip(self.drive_limit / np.maximum(mag, 1e-12), None, 1.0)
            new_correction = new_correction * scale
        self.correction[:n] = new_correction

    def drive_at(self, step_index: int) -> complex:
        """Return the learned correction at a sample index.

        Args:
            step_index: Sample index.

        Returns:
            Learned correction drive, V.
        """
        idx = min(int(step_index), self.episode_len - 1)
        return complex(self.correction[idx]) if self.correction.size else 0.0 + 0.0j
