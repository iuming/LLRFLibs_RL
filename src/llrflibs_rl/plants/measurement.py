###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Measurement-chain realism: demodulators, I/Q<->A/P conversions and a
delay+noise measurment wrapper.

Real LLRF front-ends demodulate an intermediate-frequency signal into I/Q with
a finite detector latency and ADC noise.  This module wraps the demodulation /
conversion toolbox ``llrflibs.rf_det_act`` and provides a small
:class:`MeasurementChain` that applies a transport delay, additive noise and an
I/Q<->A/P round-trip the same way a simulator would — useful for domain
randomization and for studying how RL policies tolerate detector imperfections.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from llrflibs.rf_det_act import (
    ap2iq_wf,
    asyn_demod,
    iq2ap_wf,
    noniq_demod,
    pulse_info,
    twop_demod,
)

__all__ = [
    "demodulate_noniq",
    "demodulate_twop",
    "demodulate_asynchronous",
    "iq_to_ap",
    "ap_to_iq",
    "detect_pulse",
    "MeasurementChain",
]


def _check(ok: bool, what: str) -> None:
    if not ok:
        raise ValueError(f"llrflibs detector routine failed: {what}")


def demodulate_noniq(raw_wave: np.ndarray, n: int, m: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Non-I/Q (moving-average) demodulation.

    Args:
        raw_wave: Raw IF waveform.
        n:        Samples covering ``m`` IF cycles.
        m:        IF cycles per super-period.

    Returns:
        ``(I, Q)`` waveforms.
    """
    ok, i_wf, q_wf = noniq_demod(np.asarray(raw_wave), int(n), int(m))
    _check(ok, "noniq_demod")
    return np.asarray(i_wf), np.asarray(q_wf)


def demodulate_twop(raw_wave: np.ndarray, f_if: float, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Two-point demodulation.

    Args:
        raw_wave: Raw IF waveform.
        f_if:     IF frequency, Hz.
        fs:       Sampling frequency, Hz.

    Returns:
        ``(I, Q)`` waveforms.
    """
    ok, i_wf, q_wf = twop_demod(np.asarray(raw_wave), float(f_if), float(fs))
    _check(ok, "twop_demod")
    return np.asarray(i_wf), np.asarray(q_wf)


def demodulate_asynchronous(raw_wave: np.ndarray, ref_wave: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Asynchronous (reference-based) demodulation.

    Args:
        raw_wave: Signal waveform.
        ref_wave: Reference waveform.

    Returns:
        ``(I, Q)`` waveforms.
    """
    ok, i_wf, q_wf = asyn_demod(np.asarray(raw_wave), np.asarray(ref_wave))
    _check(ok, "asyn_demod")
    return np.asarray(i_wf), np.asarray(q_wf)


def iq_to_ap(i_wf: np.ndarray, q_wf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """I/Q to amplitude/phase.

    Args:
        i_wf: I samples.
        q_wf: Q samples.

    Returns:
        ``(amplitude, phase_deg)`` waveforms.
    """
    ok, amp, phase = iq2ap_wf(np.asarray(i_wf), np.asarray(q_wf))
    _check(ok, "iq2ap_wf")
    return np.asarray(amp), np.asarray(phase)


def ap_to_iq(amp_wf: np.ndarray, phase_wf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Amplitude/phase to I/Q.

    Args:
        amp_wf:   Amplitude samples.
        phase_wf: Phase samples (deg).

    Returns:
        ``(I, Q)`` waveforms.
    """
    ok, i_wf, q_wf = ap2iq_wf(np.asarray(amp_wf), np.asarray(phase_wf))
    _check(ok, "ap2iq_wf")
    return np.asarray(i_wf), np.asarray(q_wf)


def detect_pulse(wave: np.ndarray, threshold: float = 0.1) -> dict[str, object]:
    """Detect the offset/width/peak of a pulse in a waveform.

    Args:
        wave:      The pulse data.
        threshold: Edged-detection threshold (fraction of the peak).

    Returns:
        Dict with ``status``, ``offs``, ``pulw`` and ``peak``.
    """
    result = pulse_info(np.asarray(wave), float(threshold))
    return {k: (int(v) if isinstance(v, (int, np.integer)) else float(v)) for k, v in result.items()}


@dataclass
class MeasurementChain:
    """Configurable detector chain: delay + additive noise + decimation.

    Attributes:
        delay_steps: Transport delay of the measurement, samples.
        noise_std:   Additive I/Q noise standard deviation (same units as the
                     signal).
        seed:        Random seed for reproducible noise.
        decimation:  Firmware-style decimation factor (> 1 averages groups of
                     samples, mirroring the decimated control loops of real
                     LLRF FPGAs).  The output length is preserved (hold).
    """

    delay_steps: int = 1
    noise_std: float = 0.0
    seed: int = 0
    decimation: int = 1

    def process_wave(self, wave: np.ndarray) -> np.ndarray:
        """Apply decimation, delay and noise to a complex I/Q waveform.

        Args:
            wave: Input complex waveform.

        Returns:
            Processed waveform (same length as the input).
        """
        wave = np.asarray(wave, dtype=np.complex128)
        out = np.zeros_like(wave)
        if self.decimation > 1:
            d = int(self.decimation)
            m = wave.shape[0] // d * d
            if m >= d:
                decimated = wave[:m].reshape(-1, d).mean(axis=1)
                out[:m] = np.repeat(decimated, d)
        else:
            out[:] = wave
        if self.delay_steps > 0:
            delayed = np.roll(out, self.delay_steps)
            delayed[: self.delay_steps] = 0.0
            out = delayed
        if self.noise_std > 0.0:
            rng = np.random.default_rng(self.seed)
            real_noise = rng.normal(0.0, self.noise_std, wave.shape)
            imag_noise = rng.normal(0.0, self.noise_std, wave.shape)
            out += (real_noise + 1j * imag_noise) / np.sqrt(2.0)
        return out
