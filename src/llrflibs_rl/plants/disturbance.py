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

import numpy as np

__all__ = [
    "SinusoidalDisturbance",
    "StepDisturbance",
    "SweepDisturbance",
    "build_disturbance_wave",
]


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


class StepDisturbance:
    """Step detuning disturbance (e.g. an abrupt piezo/host synchronisation kick).

    Args:
        base:        Static detuning offset before the step, rad/s.
        amplitude:   Step height (final value = ``base + amplitude``), rad/s.
        onset:       Step onset time, s.
        Ts:          Sampling time, s.
        episode_len: Number of samples per episode.
    """

    def __init__(
        self,
        base: float,
        amplitude: float,
        onset: float,
        Ts: float,
        episode_len: int,
    ) -> None:
        self.base = float(base)
        self.amplitude = float(amplitude)
        self.onset = float(onset)
        self.Ts = float(Ts)
        self.episode_len = int(episode_len)

    def build(self, phase: float = 0.0, offset: float = 0.0) -> np.ndarray:
        """Return the step disturbance waveform (``phase`` is unused)."""
        t = np.arange(self.episode_len) * self.Ts
        level = float(self.base + offset)
        wave = np.full(self.episode_len, level)
        mask = t >= self.onset
        wave[mask] = level + self.amplitude
        return wave

    def to_dict(self) -> dict:
        """Serializable description."""
        return {
            "base": self.base,
            "amplitude": self.amplitude,
            "onset": self.onset,
            "Ts": self.Ts,
            "episode_len": self.episode_len,
        }


class SweepDisturbance:
    """Piecewise-linear (ramp / chirp-like) detuning disturbance.

    Args:
        start:       Detuning at the first sample, rad/s.
        end:         Detuning at the last sample, rad/s.
        Ts:          Sampling time, s.
        episode_len: Number of samples per episode.
    """

    def __init__(self, start: float, end: float, Ts: float, episode_len: int) -> None:
        self.start = float(start)
        self.end = float(end)
        self.Ts = float(Ts)
        self.episode_len = int(episode_len)

    def build(self, phase: float = 0.0, offset: float = 0.0) -> np.ndarray:
        """Return the linear ramp waveform (``phase`` is unused)."""
        return np.linspace(self.start + offset, self.end + offset, self.episode_len)

    def to_dict(self) -> dict:
        """Serializable description."""
        return {"start": self.start, "end": self.end, "Ts": self.Ts, "episode_len": self.episode_len}


def build_disturbance_wave(
    name: str,
    episode_len: int,
    Ts: float,
    base: float,
    amplitude: float,
    frequency: float = 250.0,
    phase: float = 0.0,
    offset: float = 0.0,
    onset: float | None = None,
    sweep_end: float | None = None,
) -> np.ndarray:
    """Build a disturbance waveform by name.

    Args:
        name:       ``"sinusoidal"``, ``"step"`` or ``"sweep"``.
        episode_len: Number of samples.
        Ts:          Sampling time, s.
        base:        Static detuning offset, rad/s.
        amplitude:   Oscillation/step amplitude, rad/s.
        frequency:   Frequency (sinusoidal only), Hz.
        phase:       Initial phase (sinusoidal only), rad.
        offset:      Extra static offset, rad/s.
        onset:       Step onset (step only), s.
        sweep_end:   Final detuning (sweep only), rad/s.

    Returns:
        Detuning waveform, rad/s.

    Raises:
        ValueError: If the name is unknown.
    """
    if name == "sinusoidal":
        return SinusoidalDisturbance(base, amplitude, frequency, Ts, episode_len).build(phase, offset)
    if name == "step":
        onset = float(onset) if onset is not None else 0.2 * episode_len * Ts
        return StepDisturbance(base, amplitude, onset, Ts, episode_len).build(phase, offset)
    if name == "sweep":
        end = float(sweep_end) if sweep_end is not None else base + amplitude
        return SweepDisturbance(base + offset, end + offset, Ts, episode_len).build()
    raise ValueError(f"unknown disturbance type {name!r}; use 'sinusoidal', 'step' or 'sweep'")
