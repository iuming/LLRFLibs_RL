###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Notch filters for periodic-disturbance rejection.

Narrow-band microphonics / EMI lines can be rejected either at the controller
level (the ``notch_conf`` option of :class:`ComplexPIController`) or on a
waveform level.  This module provides:

* :class:`NotchFilter` — a sample-by-sample IIR notch built with
  ``llrflibs.rf_noise.design_notch_filter`` / ``filt_step``;
* :func:`notch_conf_for_frequencies` — builds the ``notch_conf`` dictionary
  consumed by the parent ``basic_rf_controller`` (and hence by
  ``PIControllerConfig.notch_conf``).

Typical use (PI with an extra notch against a 250 Hz microphonics line)::

    cfg = replace(PIControllerConfig(), notch_conf=notch_conf_for_frequencies(
        frequencies=[250.0], half_bws=[2.0 * np.pi * 60.0], gains=[0.02]))
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from llrflibs.rf_noise import design_notch_filter, filt_step

__all__ = ["NotchFilter", "notch_conf_for_frequencies", "notch_filter_wave"]


def notch_conf_for_frequencies(
    frequencies: Sequence[float],
    half_bws: Sequence[float],
    gains: Sequence[float] | None = None,
) -> dict[str, list[float]]:
    """Build a ``notch_conf`` dict for ``PIControllerConfig.notch_conf``.

    The format matches the parent ``llrflibs.rf_control.basic_rf_controller``:
    parallel complex-conjugate notch transfer functions at the given offset
    frequencies.

    Args:
        frequencies: Offset frequencies to be notched, Hz.
        half_bws:    Half bandwidths of each notch, rad/s.
        gains:       Notch depths (inverse suppression ratio), defaults to
                     ``0.01`` (99 % suppression) per notch.

    Returns:
        Dict with keys ``freq_offs``, ``half_bw`` and ``gain``.

    Raises:
        ValueError: On inconsistent list lengths or invalid values.
    """
    freqs = [float(f) for f in frequencies]
    bws = [float(hb) for hb in half_bws]
    if len(freqs) != len(bws):
        raise ValueError("frequencies and half_bws must have equal length.")
    if len(freqs) == 0:
        raise ValueError("at least one notch frequency is required.")
    if gains is None:
        gains = [0.01] * len(freqs)
    gs = [float(g) for g in gains]
    if len(gs) != len(freqs):
        raise ValueError("gains must match the number of frequencies.")
    if any(f <= 0.0 or hb <= 0.0 or g < 0.0 for f, hb, g in zip(freqs, bws, gs, strict=False)):
        raise ValueError("notch frequencies/bandwidths must be positive, gains non-negative.")
    return {"freq_offs": freqs, "half_bw": bws, "gain": gs}


class NotchFilter:
    """Sample-by-sample IIR notch filter (state-space form).

    Args:
        fnotch: Notch center frequency, Hz.
        q:      Quality factor of the notch.
        fs:     Sampling frequency, Hz.
    """

    def __init__(self, fnotch: float, q: float, fs: float) -> None:
        ok, ad, bd, cd, dd = design_notch_filter(float(fnotch), float(q), float(fs))
        if not ok:
            raise ValueError("notch filter design failed (check fnotch/Q/fs).")
        self._ad = np.asmatrix(ad)
        self._bd = np.asmatrix(bd)
        self._cd = np.asmatrix(cd)
        self._dd = np.asmatrix(dd)
        self._state = np.zeros((self._bd.shape[0], 1), dtype=complex)
        self.fs = float(fs)

    def reset(self) -> None:
        """Reset the filter state."""
        self._state = np.zeros_like(self._state)

    def step(self, value: float | complex) -> complex:
        """Apply one sample.

        Args:
            value: Current sample.

        Returns:
            Filtered sample.
        """
        ok, out, state = filt_step(self._ad, self._bd, self._cd, self._dd, complex(value), self._state)
        if not ok:
            raise RuntimeError("notch filter step failed.")
        self._state = state
        return complex(np.asarray(out).squeeze())

    def filter_wave(self, wave: np.ndarray) -> np.ndarray:
        """Filter a whole waveform, resetting internal state first.

        Args:
            wave: Input waveform (real or complex).

        Returns:
            Filtered waveform.
        """
        self.reset()
        out = np.asarray([self.step(v) for v in np.asarray(wave)])
        return out


def notch_filter_wave(wave: np.ndarray, fnotch: float, q: float, fs: float) -> np.ndarray:
    """Batch waveform-level notch filtering.

    Args:
        wave:   Input waveform.
        fnotch: Notch center frequency, Hz.
        q:      Quality factor.
        fs:     Sampling frequency, Hz.

    Returns:
        Notched waveform.
    """
    return NotchFilter(fnotch, q, fs).filter_wave(wave)
