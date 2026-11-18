###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Disturbance generators.

The disturbance layer describes what the agent has to fight: sinusoidal
microphonics and reproducible (fixed-phase) intra-pulse detuning modulation
similar to Lorentz-force detuning.  Generators return a full waveform for an
episode, which keeps the plant simulation deterministic within one episode.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["SinusoidalDisturbance"]


class SinusoidalDisturbance:
    """Sinusoidal detuning disturbance (microphonics-like).

    Args:
        base:        Static detuning offset, rad/s.
        amplitude:   Oscillation amplitude, rad/s.
        frequency:   Oscillation frequency, Hz.
        Ts:          Sampling time, s.
        episode_len: Number of samples per episode.
    """

    def __init__(
        self,
        base: float,
        amplitude: float,
        frequency: float,
        Ts: float,
        episode_len: int,
    ) -> None:
        self.base = float(base)
        self.amplitude = float(amplitude)
        self.frequency = float(frequency)
        self.Ts = float(Ts)
        self.episode_len = int(episode_len)
        self._time = np.arange(self.episode_len) * self.Ts

    def build(self, phase: float = 0.0, offset: float = 0.0) -> np.ndarray:
        """Return the disturbance waveform of one episode.

        Args:
            phase:  Initial phase of the sinusoid, rad.
            offset: Extra static detuning offset, rad/s (e.g. domain
                    randomization of the operating point).

        Returns:
            Detuning waveform, rad/s.
        """
        return (
            self.base
            + offset
            + self.amplitude * np.sin(2.0 * np.pi * self.frequency * self._time + phase)
        )

    def to_dict(self) -> dict:
        """Serializable description (for experiment records)."""
        return {
            "base": self.base,
            "amplitude": self.amplitude,
            "frequency": self.frequency,
            "Ts": self.Ts,
            "episode_len": self.episode_len,
        }
