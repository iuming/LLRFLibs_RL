###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Pulse-to-pulse learning controllers: adaptive feedforward and ILC.

Repetitive RF pulses suffer from *repeatable* tracking errors (filling rise,
beam-loading transients, Lorentz detuning) that a classical feedback alone
cannot remove.  The controllers here exploit the repetition to cancel them:

* :class:`AdaptiveFeedforward` — time-reversed low-pass filtered feedback
  (``llrflibs.rf_control.AFF_timerev_lpf``), an "up to 90 degrees phase lead"
  correction added on top of the nominal feedforward between pulses;
* :class:`ILCFeedforward` — optimal iterative learning control
  (``llrflibs.rf_control.AFF_ilc_design`` / ``AFF_ilc``) driven by an impulse
  response of the plant, plus SVD / least-square response-inversion helpers.

These are classical baselines for the BeamLoading / AdaptivePulse
environments, comparable against the RL policies.
"""

from __future__ import annotations

import numpy as np
from llrflibs.rf_control import (
    AFF_ilc,
    AFF_ilc_design,
    AFF_timerev_lpf,
    impulse_resp_matrix,
    resp_inv_lsm,
    resp_inv_svd,
)

__all__ = [
    "AdaptiveFeedforward",
    "ILCFeedforward",
    "impulse_response_matrix",
    "invert_response_matrix",
    "design_ilc_gain",
]


def _check(ok: bool, what: str) -> None:
    if not ok:
        raise ValueError(f"llrflibs ILC routine failed: {what}")


def impulse_response_matrix(impulse_response: np.ndarray, pulse_width: int) -> np.ndarray:
    """Build the lower-triangular Toeplitz response matrix of a plant.

    Args:
        impulse_response: Complex impulse response of the plant (shorter
                          impulses are zero-padded to the pulse width).
        pulse_width:      Number of samples of one pulse.

    Returns:
        The (``pulse_width`` x ``pulse_width``) response matrix.
    """
    h = np.asarray(impulse_response, dtype=np.complex128)
    if h.shape[0] < pulse_width:
        h = np.pad(h, (0, int(pulse_width) - h.shape[0]))
    ok, h_mat = impulse_resp_matrix(h, int(pulse_width))
    _check(ok, "impulse_resp_matrix")
    return np.asarray(h_mat)


def invert_response_matrix(response: np.ndarray, *, regu: float = 0.0, singular_val_filt: float = 0.0) -> np.ndarray:
    """Invert a response matrix (SVD or regularized least squares).

    Args:
        response:           Response matrix to invert.
        regu:               Regularization (if > 0 uses the LS method).
        singular_val_filt:  SVD singular-value threshold.

    Returns:
        The (pseudo-)inverse matrix.
    """
    mat = np.asmatrix(response)
    if regu > 0.0:
        return np.asarray(resp_inv_lsm(mat, regu))
    return np.asarray(resp_inv_svd(mat, singular_val_filt))


def design_ilc_gain(
    impulse_response: np.ndarray, pulse_width: int, p_weight=None, q_weight=None
) -> np.ndarray:
    """Design an optimal ILC gain matrix ``L`` from the plant impulse response.

    Args:
        impulse_response: Complex impulse response of the plant (shorter
                          impulses are zero-padded to the pulse width).
        pulse_width:      Pulse length in samples.
        p_weight:         Optional P weight matrix (defaults to identity).
        q_weight:         Optional Q weight matrix (defaults to identity).

    Returns:
        Complex gain matrix ``L`` such that ``vff = L @ error``.
    """
    h = np.asarray(impulse_response, dtype=np.complex128)
    if h.shape[0] < pulse_width:
        h = np.pad(h, (0, int(pulse_width) - h.shape[0]))
    ok, L = AFF_ilc_design(h, int(pulse_width), P=p_weight, Q=q_weight)
    _check(ok, "AFF_ilc_design")
    return np.asarray(L)


class AdaptiveFeedforward:
    """Time-reversed low-pass filtered feedforward correction (AFF).

    Applies :func:`AFF_timerev_lpf` to the previous pulse's feedback drive and
    returns a correction waveform to be added to the next pulse's feedforward.

    Args:
        fcut: Cut-off frequency of the (first-order IIR) low-pass, Hz.
        fs:   Sampling frequency, Hz.
    """

    def __init__(self, fcut: float, fs: float) -> None:
        self.fcut = float(fcut)
        self.fs = float(fs)
        self.correction: np.ndarray | None = None

    def reset(self) -> None:
        """Forget the previous correction."""
        self.correction = None

    def update(self, feedback_wave: np.ndarray, correction_wave: np.ndarray | None = None) -> np.ndarray:
        """Compute the feedforward correction for the next pulse.

        Args:
            feedback_wave:  Feedback drive waveform of the finished pulse.
            correction_wave: Reuse the internal buffer (optional).

        Returns:
            Correction waveform, same length as ``feedback_wave``.
        """
        fb = np.asarray(feedback_wave, dtype=np.complex128)
        ok, corrected = AFF_timerev_lpf(fb, self.fcut, self.fs, vff_cor=correction_wave)
        _check(ok, "AFF_timerev_lpf")
        self.correction = np.asarray(corrected)
        return self.correction


class ILCFeedforward:
    """Optimal iterative-learning-control feedforward.

    ``vff_{k+1} = L @ error_k`` where ``L`` is designed once from the plant
    impulse response (:func:`design_ilc_gain`).

    Args:
        impulse_response: Complex impulse response of the plant.
        pulse_width:      Number of samples of one pulse.
        p_weight:         Optional P weight matrix.
        q_weight:         Optional Q weight matrix.
    """

    def __init__(
        self,
        impulse_response: np.ndarray,
        pulse_width: int,
        p_weight=None,
        q_weight=None,
    ) -> None:
        self.gain = design_ilc_gain(impulse_response, pulse_width, p_weight, q_weight)
        self.correction = np.zeros(int(pulse_width), dtype=np.complex128)

    def reset(self) -> None:
        """Zero the accumulated feedforward."""
        self.correction = np.zeros_like(self.correction)

    def update(self, error_wave: np.ndarray) -> np.ndarray:
        """Add the ILC correction based on the previous pulse error.

        Args:
            error_wave: Error waveform of the finished pulse.

        Returns:
            Accumulated feedforward correction for the next pulse.
        """
        err = np.asarray(error_wave, dtype=np.complex128)
        if err.shape[0] != self.correction.shape[0]:
            raise ValueError("ILC error waveform length must match the pulse width.")
        correction = np.asarray(AFF_ilc(err, np.asmatrix(self.gain))).squeeze()
        self.correction = correction
        return self.correction
