###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
PSD-driven disturbance noise for the LLRF environments.

Real RF cavities are disturbed by vibration-induced *microphonics* whose detuning
noise cannot be captured by a single sine tone.  This module turns measured or
modelled power-spectral-density (PSD) envelopes into time-domain detuning
realizations, re-using the spectrum synthesis routines in
``llrflibs.rf_noise`` (:func:`gen_noise_from_psd`, :func:`calc_rms_from_psd`,
:func:`rand_sine`).

Typical use::

    profile = typical_cryomodule_microphonics()
    wave = profile.sample(fs=1.0 / env_cfg.Ts, n_samples=env.episode_len)
    rms = profile.rms(f_start=10.0, f_end=500.0, fs=..., n=4096)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from llrflibs.rf_noise import calc_rms_from_psd, gen_noise_from_psd, rand_sine

__all__ = [
    "MicrophonicsProfile",
    "typical_cryomodule_microphonics",
    "noise_from_psd",
    "rms_from_psd",
    "rand_sine_wave",
]

_FLOOR_DB = -140.0  # broadband numerical/physics floor, dB(rad^2/Hz)


def _check(ok: bool, what: str) -> None:
    if not ok:
        raise ValueError(f"llrflibs noise routine failed: {what}")


def noise_from_psd(freq: np.ndarray, psd_db: np.ndarray, fs: float, n_samples: int) -> np.ndarray:
    """Sample a time-domain noise series from a DSB PSD envelope.

    Args:
        freq:     Positive offset frequency vector from the carrier, Hz.
        psd_db:   Double-sideband PSD in dB units (e.g. dB(rad^2/Hz)).
        fs:       Sampling frequency, Hz.
        n_samples: Number of samples to generate.

    Returns:
        Zero-mean noise series (in the PSD unit, e.g. radians of detuning).
    """
    freq = np.asarray(freq, dtype=float)
    psd_db = np.asarray(psd_db, dtype=float)
    ok, series, _, _ = gen_noise_from_psd(freq, psd_db, float(fs), int(n_samples))
    _check(ok, "gen_noise_from_psd")
    return np.asarray(series, dtype=np.float64)


def rms_from_psd(
    freq: np.ndarray, psd_db: np.ndarray, f_start: float, f_end: float, fs: float, n: int = 4096
) -> float:
    """RMS of a disturbance integrated from a PSD over ``[f_start, f_end]``.

    Args:
        freq:     Positive offset frequency vector, Hz.
        psd_db:   DSB PSD in dB units.
        f_start:  Integration start frequency, Hz.
        f_end:    Integration end frequency, Hz.
        fs:       Sampling frequency, Hz.
        n:        Number of integration points.

    Returns:
        Integrated RMS value.
    """
    ok, _, _, jitter = calc_rms_from_psd(
        np.asarray(freq, dtype=float),
        np.asarray(psd_db, dtype=float),
        float(f_start),
        float(f_end),
        float(fs),
        int(n),
    )
    _check(ok, "calc_rms_from_psd")
    return float(np.sqrt(np.sum(np.diff(jitter) ** 2)))


def rand_sine_wave(
    n_samples: int, fs: float, n_sines: int = 1, fmin: float = 0.0, fmax: float = 1e3
) -> np.ndarray:
    """Random sum-of-sines waveform (wrapper of ``llrflibs.rf_noise.rand_sine``).

    Args:
        n_samples: Number of samples.
        fs:        Sampling frequency, Hz.
        n_sines:   Number of sine components.
        fmin:      Minimum component frequency, Hz.
        fmax:      Maximum component frequency, Hz.

    Returns:
        Waveform with unit-amplitude components.
    """
    ok, wf = rand_sine(int(n_samples), float(fs), nfreq=int(n_sines), fmin=float(fmin), fmax=float(fmax))
    _check(ok, "rand_sine")
    return np.asarray(wf, dtype=np.float64)


@dataclass
class MicrophonicsProfile:
    """A detuning PSD envelope, expressed in dB against offset frequency.

    Attributes:
        freq:   Positive offset frequency axis, Hz (strictly increasing).
        psd_db: Double-sideband detuning PSD, dB(rad^2/Hz).
    """

    freq: np.ndarray
    psd_db: np.ndarray = field(default_factory=lambda: np.array([_FLOOR_DB], dtype=float))

    def __post_init__(self) -> None:
        self.freq = np.asarray(self.freq, dtype=float)
        self.psd_db = np.asarray(self.psd_db, dtype=float)
        if self.freq.shape[0] < 2 or self.freq.shape != self.psd_db.shape:
            raise ValueError("MicrophonicsProfile needs >= 2 frequency points matching psd_db.")
        if np.any(self.freq <= 0.0) or np.any(np.diff(self.freq) <= 0.0):
            raise ValueError("MicrophonicsProfile frequency axis must be positive and increasing.")

    # ------------------------------------------------------------------

    @classmethod
    def from_lorentzian_peaks(
        cls,
        peaks: list[tuple[float, float, float]],
        floor_db: float = _FLOOR_DB,
        fmin: float = 5.0,
        fmax: float = 1000.0,
        n_points: int = 2000,
    ) -> MicrophonicsProfile:
        """Build a profile from a set of Lorentzian (resonance) peaks.

        Each peak is ``(f0_hz, width_hz, peak_db)``: ``f0`` is the center offset
        frequency, ``width`` the half-power width and ``peak_db`` the peak PSD
        level in dB.  The spectrum is ``S(f) = 10**(floor/10) + sum_i A_i /
        (1 + ((f - f0_i) / w_i)**2)`` with ``A_i = 10**(peak_db_i/10)``.

        Args:
            peaks:    Iterable of ``(f0, width, peak_db)`` triples.
            floor_db: Broadband floor level, dB.
            fmin:     Low end of the frequency axis, Hz.
            fmax:     High end of the frequency axis, Hz.
            n_points: Number of points on the frequency axis.

        Returns:
            A :class:`MicrophonicsProfile`.
        """
        freq = np.linspace(float(fmin), float(fmax), int(n_points))
        spec = np.full(freq.shape, 10.0 ** (float(floor_db) / 10.0), dtype=float)
        for f0, w, peak_db in peaks:
            w = max(float(w), 1e-3)
            spec += 10.0 ** (float(peak_db) / 10.0) / (1.0 + ((freq - float(f0)) / w) ** 2)
        return cls(freq=freq, psd_db=10.0 * np.log10(spec))

    # ------------------------------------------------------------------

    def sample(self, fs: float, n_samples: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw a zero-mean detuning realization from the profile.

        Args:
            fs:        Sampling frequency, Hz.
            n_samples: Number of samples.
            rng:       Optional RNG seed for reproducibility (phases).

        Returns:
            Detuning noise series, rad/s.
        """
        # Reproducibility: llrflibs uses ``numpy.random`` internally; seed via
        # the legacy global generator so ``rng`` can pin the realization.
        state = np.random.get_state()
        try:
            if rng is not None:
                np.random.seed(rng.integers(0, 2**31 - 1))
            return noise_from_psd(self.freq, self.psd_db, float(fs), int(n_samples))
        finally:
            np.random.set_state(state)

    def rms(
        self, f_start: float, f_end: float, fs: float, n: int = 4096, ref_level: float = 0.0
    ) -> float:
        """Integrated RMS of the profile detuning in a frequency band.

        Args:
            f_start:    Low integration frequency, Hz.
            f_end:      High integration frequency, Hz.
            fs:         Sampling frequency, Hz.
            n:          Integration points.
            ref_level:  Optional added level shift, dB.

        Returns:
            Band RMS value, rad/s.
        """
        return rms_from_psd(self.freq, self.psd_db + float(ref_level), f_start, f_end, float(fs), int(n))


def typical_cryomodule_microphonics() -> MicrophonicsProfile:
    """A representative cryomodule microphonics spectrum.

    A handful of mechanical-resonance peaks (tens of Hz) over a broadband
    floor, matching the qualitative shape of measured SRF cryomodule detuning
    spectra (see e.g. the ERL/CBETA microphonics literature).

    Returns:
        A :class:`MicrophonicsProfile` peaking around 20-130 Hz.
    """
    return MicrophonicsProfile.from_lorentzian_peaks(
        peaks=[
            (24.0, 8.0, -62.0),
            (47.0, 9.0, -58.0),
            (71.0, 11.0, -64.0),
            (93.0, 12.0, -61.0),
            (121.0, 10.0, -66.0),
        ],
        floor_db=_FLOOR_DB,
    )
