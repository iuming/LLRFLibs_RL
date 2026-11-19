###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Spectral analysis of control error waveforms.

Complementing the scalar RMS metrics, this module computes the power-spectral
density of a policy's tracking error (via ``llrflibs.rf_noise.calc_psd``) and
its integrated band RMS (via ``llrflibs.rf_noise.calc_rms_from_psd``) — e.g. to
verify that a policy rejects the microphonics band and to quantify leakage of
disturbance energy into other bands.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from llrflibs.rf_noise import calc_psd, calc_rms_from_psd

from ..scenario import rollout_episode

__all__ = ["tracking_error_spectrum", "BandRms", "SpectralReport"]


@dataclass
class BandRms:
    """Band RMS of an error spectrum.

    Attributes:
        f_start: Low frequency, Hz.
        f_end:   High frequency, Hz.
        rms:     Integrated RMS value.
    """

    f_start: float
    f_end: float
    rms: float


@dataclass
class SpectralReport:
    """PSD analysis of one evaluation rollout.

    Attributes:
        freq:     Frequency axis, Hz.
        amp_db:   Amplitude response, dB(unit^2/Hz).
        signal_freq_hz: Dominant error frequency, Hz.
        signal_mag_db:  Dominant error level, dB.
        snr_db:   Signal-to-noise ratio, dB.
        bands:    Band RMS values.
        error_rms: Overall RMS of the error waveform.
    """

    freq: np.ndarray
    amp_db: np.ndarray
    signal_freq_hz: float
    signal_mag_db: float
    snr_db: float
    bands: list[BandRms]
    error_rms: float

    def as_dict(self) -> dict[str, object]:
        """JSON-friendly summary (arrays excluded)."""
        return {
            "signal_freq_hz": self.signal_freq_hz,
            "signal_mag_db": self.signal_mag_db,
            "snr_db": self.snr_db,
            "error_rms": self.error_rms,
            "bands": [{"f_start": b.f_start, "f_end": b.f_end, "rms": b.rms} for b in self.bands],
        }


def tracking_error_spectrum(
    env_factory: Callable,
    action_fn: Callable[[np.ndarray], np.ndarray],
    *,
    seed: int = 0,
    fs: float | None = None,
    bands: list[tuple[float, float]] | None = None,
) -> SpectralReport:
    """PSD of the tracking-error waveform of one episode.

    Args:
        env_factory: Environment factory.
        action_fn:   Policy mapping an observation to an action.
        seed:        Evaluation seed.
        fs:          Sampling frequency, Hz (defaults to ``1/env.Ts``).
        bands:       Frequency bands for the RMS table, list of
                     ``(f_start, f_end)`` in Hz.

    Returns:
        A :class:`SpectralReport`.
    """
    env = env_factory()
    wave = rollout_episode(env, action_fn, seed=seed)
    tracked = np.asarray(wave["tracked"], dtype=np.complex128)
    if hasattr(env, "reference_sp"):
        reference = np.asarray(env.reference_sp[: tracked.shape[0]], dtype=np.complex128)
        error = np.abs(reference - tracked)
    else:
        error = np.abs(tracked)

    fs_use = float(fs) if fs is not None else 1.0 / float(env.env_cfg.Ts)
    result = calc_psd(error, fs_use)
    if not result["status"]:
        raise ValueError("calc_psd failed on the error waveform.")

    freq = np.asarray(result["freq"])
    amp_db = np.asarray(result["amp_resp"])

    rms_use = float(np.sqrt(np.mean(error**2)))

    band_list: list[BandRms] = []
    integ_n = 4096
    for f_start, f_end in bands or []:
        ok, f_p, pn_p, _ = calc_rms_from_psd(freq, amp_db, float(f_start), float(f_end), fs_use, integ_n)
        if not ok or f_p is None or pn_p is None:
            continue
        # Recover the linear PSD from the dB axis and integrate the RMS.
        df = (float(f_end) - float(f_start)) / integ_n
        rms = float(np.sqrt(np.sum(10.0 ** (np.asarray(pn_p) / 10.0) * df)))
        band_list.append(BandRms(float(f_start), float(f_end), rms))

    return SpectralReport(
        freq=freq,
        amp_db=amp_db,
        signal_freq_hz=float(result["signal_freq"]),
        signal_mag_db=float(result["signal_mag"]),
        snr_db=float(result["snr"]),
        bands=band_list,
        error_rms=rms_use,
    )
