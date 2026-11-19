###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Beam schedules.

A :class:`BeamSegment` describes one time window in which the beam is present
(level 1.0) or partially loaded (level < 1.0).  :class:`BeamSchedule` turns a
list of segments into the beam-drive waveform of one episode, so transient
beam conditions (beam-on, trips, partial fill) can be configured declaratively.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

__all__ = ["BeamSchedule", "BeamSegment", "pulse_beam_schedule"]


@dataclass
class BeamSegment:
    """Beam present on ``[from_sample, to_sample)`` with a relative level.

    Attributes:
        from_sample: First sample of the segment.
        to_sample:   One past the last sample of the segment.
        level:       Relative beam level in ``(0, 1]`` (1 = nominal current).
    """

    from_sample: int
    to_sample: int
    level: float = 1.0

    def __post_init__(self) -> None:
        if self.from_sample < 0 or self.to_sample < self.from_sample:
            raise ValueError("invalid beam segment boundaries")
        if not 0.0 < self.level <= 1.0:
            raise ValueError("beam level must lie in (0, 1]")


def pulse_beam_schedule(onset_sample: int, episode_len: int) -> "BeamSchedule":
    """Beam present from ``onset_sample`` until the end of the episode.

    Args:
        onset_sample: First sample with the beam present.
        episode_len: Episode length.

    Returns:
        A single-segment schedule (the BeamLoading-v0 default behaviour).
    """
    return BeamSchedule([BeamSegment(int(onset_sample), int(episode_len), 1.0)])


class BeamSchedule:
    """Piecewise-constant beam schedule over one episode.

    Args:
        segments: Ordered beam segments (may overlap; later segments override).
    """

    def __init__(self, segments: Sequence[BeamSegment]) -> None:
        if not segments:
            raise ValueError("segments must not be empty")
        self.segments = list(segments)

    @classmethod
    def from_steps(cls, steps: Sequence[Tuple[int, int, float]]) -> "BeamSchedule":
        """Build a schedule from ``(from_sample, to_sample, level)`` tuples."""
        return cls([BeamSegment(int(f), int(t), float(lv)) for f, t, lv in steps])

    def apply_to(self, reference_vb: np.ndarray) -> np.ndarray:
        """Mask a beam-drive waveform with the schedule.

        Args:
            reference_vb: Nominal beam-drive waveform (V).

        Returns:
            The scheduled beam waveform (V): zero outside any segment,
            scaled by the segment level inside.
        """
        wave = np.zeros(reference_vb.shape, dtype=np.complex128)
        for seg in self.segments:
            wave[seg.from_sample: seg.to_sample] = (
                np.asarray(reference_vb, dtype=np.complex128)[seg.from_sample: seg.to_sample]
                * seg.level
            )
        return wave

    def to_dict(self) -> dict:
        """Serializable description."""
        return {
            "segments": [
                {"from_sample": s.from_sample, "to_sample": s.to_sample, "level": s.level}
                for s in self.segments
            ]
        }
