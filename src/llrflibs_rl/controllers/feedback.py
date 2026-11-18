###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Feedback controllers used as the inner loop of the assist / dual-loop
environments.

Both controllers wrap the primitives in ``llrflibs.rf_control``:
:func:`basic_rf_controller` + :func:`control_step` for the complex PI and
:func:`ADRC_controller` + :func:`ADRC_control_step` for the ADRC observer-based
controller.  They run sample-by-sample and keep their own discrete state.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from llrflibs.rf_control import ADRC_control_step, ADRC_controller, basic_rf_controller, control_step, ss_discrete

from ..configs.controller import ADRCControllerConfig, PIControllerConfig

__all__ = ["ADRCController", "ComplexPIController", "ScheduledPIController"]


def _state_matrix(size: int) -> np.matrix:
    return np.matrix(np.zeros((size, 1), dtype=complex))


def _as_complex(value) -> complex:
    return complex(np.asarray(value).squeeze())


class ComplexPIController:
    """Discrete-time complex PI controller with optional notch filter.

    Args:
        config: PI parameters.
        Ts:     Sampling time, s.
    """

    def __init__(self, config: PIControllerConfig, Ts: float) -> None:
        self.config = config
        self.Ts = float(Ts)

        status, Akc, Bkc, Ckc, Dkc = basic_rf_controller(
            config.kp, config.ki, notch_conf=config.notch_conf, plot=False
        )
        if not status:
            raise ValueError("Failed to build the PI RF controller.")
        status, self._Akd, self._Bkd, self._Ckd, self._Dkd, _ = ss_discrete(
            Akc, Bkc, Ckc, Dkc, self.Ts, method="zoh", plot=False
        )
        if not status:
            raise ValueError("Failed to discretize the PI RF controller.")
        self.reset()

    def reset(self) -> None:
        """Reset the controller integrator state."""
        self._state_k: np.matrix = _state_matrix(self._Bkd.shape[0])

    def step(self, error: complex, feedforward: complex) -> Tuple[complex, complex]:
        """Execute one controller step.

        Args:
            error:        Complex voltage error ``sp - vc``.
            feedforward:  Nominal feedforward drive of this sample.

        Returns:
            Tuple of ``(drive_total, drive_feedback)``.
        """
        status, drive_pi, drive_fb, state_k = control_step(
            self._Akd,
            self._Bkd,
            self._Ckd,
            self._Dkd,
            error,
            self._state_k,
            ff_step=feedforward,
        )
        if not status:
            raise RuntimeError("PI controller step failed.")
        self._state_k = state_k
        return _as_complex(drive_pi), _as_complex(drive_fb)


class ScheduledPIController:
    """Direct-form complex PI whose gains can change sample by sample.

    This is the implementation behind the gain-scheduling environment: the RL
    policy selects ``kp`` and ``ki`` at runtime and the controller applies
    ``drive = -(kp*error + ki*integral) + feedforward`` with a fixed
    integrator recurrence ``integral += Ts*error``.  It mirrors the classical
    complex PI at moderate gains while staying trivially differentiable-free
    and sample-wise.

    Args:
        kp:     Nominal proportional gain.
        ki:     Nominal integral gain.
        Ts:     Sampling time, s.
    """

    def __init__(self, kp: float, ki: float, Ts: float) -> None:
        self.Ts = float(Ts)
        self.kp = float(kp)
        self.ki = float(ki)
        self.reset()

    def reset(self) -> None:
        """Reset the integrator state."""
        self.integral = 0.0 + 0.0j
        self.drive_fb = 0.0 + 0.0j

    def step(
        self,
        error: complex,
        feedforward: complex,
        kp: Optional[float] = None,
        ki: Optional[float] = None,
    ) -> Tuple[complex, complex]:
        """Execute one controller step with (optionally scheduled) gains.

        Args:
            error:       Complex voltage error ``sp - vc``.
            feedforward: Nominal feedforward drive.
            kp:          Proportional gain override (defaults to ``self.kp``).
            ki:          Integral gain override (defaults to ``self.ki``).

        Returns:
            Tuple of ``(drive_total, drive_feedback)``.
        """
        kp_use = float(kp) if kp is not None else self.kp
        ki_use = float(ki) if ki is not None else self.ki
        self.integral = self.integral + error * self.Ts
        drive_fb = -(kp_use * error + ki_use * self.integral)
        self.drive_fb = drive_fb
        return drive_fb + complex(feedforward), drive_fb


class ADRCController:
    """Active-disturbance-rejection controller (observer form).

    Args:
        config:  ADRC parameters.
        Ts:      Sampling time, s.
        half_bw: Cavity half bandwidth, rad/s (needed for the observer design).
    """

    def __init__(self, config: ADRCControllerConfig, Ts: float, half_bw: float) -> None:
        self.config = config
        self.Ts = float(Ts)
        self.half_bw = float(half_bw)

        status, Akc, Bkc, Ckc, Dkc = basic_rf_controller(
            config.kp, config.ki, notch_conf=None, plot=False
        )
        if not status:
            raise ValueError("Failed to build the ADRC feedback controller.")
        status, self._Akd, self._Bkd, self._Ckd, self._Dkd, _ = ss_discrete(
            Akc, Bkc, Ckc, Dkc, self.Ts, method="zoh", plot=False
        )
        if not status:
            raise ValueError("Failed to discretize the ADRC feedback controller.")

        status, Aobc, Bobc, Cobc, Dobc, b0 = ADRC_controller(
            half_bw, pole_scale=config.observer_pole_scale
        )
        if not status:
            raise ValueError("Failed to build the ADRC observer.")
        status, self._Aobd, self._Bobd, _, _, _ = ss_discrete(
            Aobc, Bobc, Cobc, Dobc, self.Ts, method="zoh", plot=False
        )
        if not status:
            raise ValueError("Failed to discretize the ADRC observer.")
        self._b0 = float(b0)
        self.reset()

    def reset(self) -> None:
        """Reset the controller and observer states."""
        self._state_k: np.matrix = _state_matrix(self._Bkd.shape[0])
        self._state_ob: np.matrix = _state_matrix(self._Bobd.shape[0])
        self.drive_fb = 0.0 + 0.0j
        self.vc_est = 0.0 + 0.0j
        self.f_est = 0.0 + 0.0j

    def step(
        self,
        sp: complex,
        vc_meas: complex,
        vf_meas: complex,
        feedforward: complex,
    ) -> Tuple[complex, complex, complex, complex]:
        """Execute one ADRC step.

        Args:
            sp:         Set-point of this sample.
            vc_meas:    Measured cavity voltage (previous sample).
            vf_meas:    Measured drive (previous sample).
            feedforward: Nominal feedforward drive of this sample.

        Returns:
            Tuple of ``(drive_adrc, drive_feedback, vc_est, f_est)``.
        """
        status, drive_adrc, drive_fb, state_k, state_ob, vc_est, f_est = ADRC_control_step(
            self._Akd,
            self._Bkd,
            self._Ckd,
            self._Dkd,
            self._Aobd,
            self._Bobd,
            self._b0,
            sp,
            vc_meas,
            vf_meas,
            self._state_k,
            self._state_ob,
            vf_step=feedforward,
            ff_step=feedforward,
            apply_to_err=self.config.apply_to_err,
        )
        if not status:
            raise RuntimeError("ADRC controller step failed.")
        self._state_k = state_k
        self._state_ob = state_ob
        self.drive_fb = _as_complex(drive_fb)
        self.vc_est = _as_complex(vc_est)
        self.f_est = _as_complex(f_est)
        return _as_complex(drive_adrc), self.drive_fb, self.vc_est, self.f_est
