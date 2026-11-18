###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Event / constraint helpers shared by the environments.

These pure functions implement the engineering constraints of a real LLRF
channel: drive amplitude saturation, drive slew-rate limit, phase-error
normalization and complex-arithmetic helpers.  They are unit-testable without
an environment.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = [
    "apply_drive_limits",
    "clip_complex_abs",
    "apply_complex_slew",
    "phase_error_norm",
]


def clip_complex_abs(value: complex, limit: Optional[float]) -> complex:
    """Clip the magnitude of a complex value.

    Args:
        value: Complex quantity.
        limit: Maximum magnitude, or ``None`` to disable clipping.

    Returns:
        The clipped value.
    """
    if limit is None or limit <= 0:
        return value
    mag = abs(value)
    if mag <= limit or mag == 0.0:
        return value
    return value * (limit / mag)


def apply_complex_slew(value: complex, prev_value: complex, max_step: Optional[float]) -> complex:
    """Limit the per-step variation of a complex quantity.

    Args:
        value:      Target quantity.
        prev_value: Previous quantity.
        max_step:   Maximum per-step magnitude change, or ``None`` to disable.

    Returns:
        The slew-limited value.
    """
    if max_step is None or max_step <= 0:
        return value
    delta = value - prev_value
    mag = abs(delta)
    if mag <= max_step or mag == 0.0:
        return value
    return prev_value + delta * (max_step / mag)


def apply_drive_limits(
    drive: complex,
    prev_drive: complex,
    max_abs: Optional[float],
    max_slew: Optional[float],
) -> Tuple[complex, float, float]:
    """Apply saturation and slew-rate limits to a drive-sample command.

    The returned excesses are bounded to ``[0, 1]`` so that the reward term
    stays well-scaled.

    Args:
        drive:      Unconstrained drive command.
        prev_drive: Previous (limited) drive command.
        max_abs:    Maximum drive magnitude (``None`` disables).
        max_slew:   Maximum per-step drive change (``None`` disables).

    Returns:
        Tuple of ``(limited_drive, sat_excess, slew_excess)``.
    """
    raw_slew = abs(drive - prev_drive)
    if max_slew is not None and max_slew > 0:
        slew_excess = min(max(raw_slew - max_slew, 0.0) / max_slew, 1.0)
    else:
        slew_excess = 0.0

    raw_abs = abs(drive)
    if max_abs is not None and max_abs > 0:
        sat_excess = min(max(raw_abs - max_abs, 0.0) / max_abs, 1.0)
    else:
        sat_excess = 0.0

    limited = apply_complex_slew(drive, prev_drive, max_slew)
    limited = clip_complex_abs(limited, max_abs)
    return limited, sat_excess, slew_excess


def phase_error_norm(reference: complex, value: complex, threshold: float) -> float:
    """Normalized phase error in ``[-1, 1]`` (radians / pi).

    Args:
        reference: Reference phasor.
        value:     Measured phasor.
        threshold: Magnitude below which the phase error is undefined (returns 0).

    Returns:
        The normalized phase error.
    """
    if abs(reference) <= threshold:
        return 0.0
    return float(np.angle(value * np.conj(reference)) / np.pi)
