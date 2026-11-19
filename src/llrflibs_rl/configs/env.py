###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Environment-level configuration.

These dataclasses describe the *loop* of an RL environment: sampling time,
episode length, measurement noise, transport delay, drive constraints, domain
randomization and (for the resonance envs) the slow actuator and disturbance.
Physics parameters live in :mod:`llrflibs_rl.configs.cavity`, reward/shaping
parameters live in :mod:`llrflibs_rl.configs.task`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..plants.noise import MicrophonicsProfile
from .cavity import CavityParams

__all__ = [
    "EnvConfig",
    "ResidualEnvConfig",
    "FrequencyEnvConfig",
    "DualLoopEnvConfig",
    "VectorSumEnvConfig",
]


@dataclass
class EnvConfig:
    """Shared loop-level configuration for every LLRF environment.

    Attributes:
        Ts:                      Sampling time, s.
        episode_len:             Number of samples per episode.
        measurement_noise_std:   Measurement noise (relative to the voltage
                     scale), added to the observed voltages.
        loop_delay_steps:        Discrete transport delay of the drive command.
        max_drive_abs:           Optional drive amplitude saturation, V.
        max_drive_slew_per_step: Optional drive slew-rate limit, V/step.
        randomize_detuning:      Half-range of uniform plant-detuning
                     randomization at reset, rad/s (0 disables it).
        microphonics_amp:        Amplitude of the intra-pulse detuning
                     modulation, rad/s (0 disables it).
        microphonics_freq:       Frequency of the intra-pulse detuning
                     modulation, Hz.
        microphonics_random_phase: Randomize the microphonics phase each reset.
        microphonics_psd:        Optional :class:`MicrophonicsProfile` with a
                     realistic detuning PSD. When set it replaces the single
                     sine microphonics (``microphonics_amp``) with a full PSD
                     realization.
        reflected_power_limit:     Optional reflected-voltage threshold, V;
                     exceeding it for ``reflected_power_trip_steps`` consecutive
                     samples ends the episode (quench-like protective trip).
        reflected_power_trip_steps: Consecutive samples above the limit that
                     trigger the trip.
        trip_penalty:              Extra (usually negative) reward added on trip.
        half_bw_drift_frac:        Relative linear drift of the cavity half
                     bandwidth across the episode (robustness test); uses the
                     time-varying plant path.
        seed:                    Base random seed (the np_random is derived from it).
    """

    Ts: float = 1.0e-6
    episode_len: int = 1800
    measurement_noise_std: float = 1.0e-4
    loop_delay_steps: int = 1
    max_drive_abs: float | None = 2.0e8
    max_drive_slew_per_step: float | None = 1.0e7
    randomize_detuning: float = 2.0 * np.pi * 300.0
    microphonics_amp: float = 2.0 * np.pi * 200.0
    microphonics_freq: float = 250.0
    microphonics_random_phase: bool = False
    reflected_power_limit: float | None = None
    reflected_power_trip_steps: int = 10
    trip_penalty: float = 0.0
    half_bw_drift_frac: float = 0.0
    seed: int = 0
    microphonics_psd: MicrophonicsProfile | None = None

    def validate(self) -> None:
        """Raise on inconsistent configuration."""
        if self.Ts <= 0:
            raise ValueError("Ts must be positive.")
        if self.episode_len <= 0:
            raise ValueError("episode_len must be positive.")
        if self.measurement_noise_std < 0:
            raise ValueError("measurement_noise_std must be non-negative.")
        if self.loop_delay_steps < 0:
            raise ValueError("loop_delay_steps must be non-negative.")
        if self.max_drive_abs is not None and self.max_drive_abs <= 0:
            raise ValueError("max_drive_abs must be positive when set.")
        if self.max_drive_slew_per_step is not None and self.max_drive_slew_per_step <= 0:
            raise ValueError("max_drive_slew_per_step must be positive when set.")
        if self.randomize_detuning < 0:
            raise ValueError("randomize_detuning must be non-negative.")
        if self.microphonics_amp < 0:
            raise ValueError("microphonics_amp must be non-negative.")
        if self.microphonics_freq < 0:
            raise ValueError("microphonics_freq must be non-negative.")
        if self.reflected_power_limit is not None and self.reflected_power_limit <= 0:
            raise ValueError("reflected_power_limit must be positive when set.")
        if self.reflected_power_trip_steps <= 0:
            raise ValueError("reflected_power_trip_steps must be positive.")
        if self.half_bw_drift_frac < 0:
            raise ValueError("half_bw_drift_frac must be non-negative.")

    def __post_init__(self) -> None:
        self.validate()


@dataclass
class ResidualEnvConfig(EnvConfig):
    """Configuration for single-cavity residual / inner-loop-assist envs.

    Attributes:
        drive_limit_ratio:   Max RL residual drive as a fraction of the voltage
                     scale (used to map the normalized action to I/Q).
        reference_detuning:  Detuning used to build the nominal reference /
                     feedforward waveform. ``None`` uses ``cavity.detuning``.
        drive_effort_reference: How the drive-effort penalty is computed:
                     ``"ff"`` (deviation from the nominal feedforward) or
                     ``"total"`` (total drive magnitude).
    """

    drive_limit_ratio: float = 0.10
    reference_detuning: float | None = None
    drive_effort_reference: str = "ff"
    lorentz_k: float = 0.0
    lorentz_tau: float = 5.0e-4

    def validate(self) -> None:
        super().validate()
        if self.drive_limit_ratio <= 0:
            raise ValueError("drive_limit_ratio must be positive.")
        if self.drive_effort_reference not in ("ff", "total"):
            raise ValueError("drive_effort_reference must be 'ff' or 'total'.")
        if self.lorentz_k < 0:
            raise ValueError("lorentz_k must be non-negative.")
        if self.lorentz_tau <= 0:
            raise ValueError("lorentz_tau must be positive.")

    def __post_init__(self) -> None:
        self.validate()


@dataclass
class FrequencyEnvConfig(EnvConfig):
    """Configuration for the resonance / detuning-compensation environment.

    Attributes:
        drive_level:        Constant forward drive amplitude, V.
        detuning_limit:     Max actuator detuning (maps action [-1, 1]), rad/s.
        actuator_tau:       First-order actuator time constant, s.
        base_detuning:      Static detuning offset of the disturbance, rad/s.
        disturbance_amp:    Amplitude of the sinusoidal microphonics, rad/s.
        disturbance_freq:   Frequency of the sinusoidal microphonics, Hz.
        disturbance_type:   ``"sinusoidal"`` / ``"step"`` / ``"sweep"``.
        disturbance_step_onset: Onset time of the step disturbance, s.
        randomize_disturbance: Randomize the disturbance phase each reset.
    """

    drive_level: float = 12.5e6
    detuning_limit: float = 2.0 * np.pi * 3.0e3
    actuator_tau: float = 8.0e-5
    base_detuning: float = 2.0 * np.pi * 2.0e2
    disturbance_amp: float = 2.0 * np.pi * 2.0e2
    disturbance_freq: float = 250.0
    disturbance_type: str = "sinusoidal"
    disturbance_step_onset: float = 6.0e-4
    randomize_disturbance: bool = True

    def validate(self) -> None:
        super().validate()
        if self.drive_level <= 0:
            raise ValueError("drive_level must be positive.")
        if self.detuning_limit <= 0:
            raise ValueError("detuning_limit must be positive.")
        if self.actuator_tau <= 0:
            raise ValueError("actuator_tau must be positive.")
        if self.disturbance_amp < 0:
            raise ValueError("disturbance_amp must be non-negative.")
        if self.disturbance_freq < 0:
            raise ValueError("disturbance_freq must be non-negative.")
        if self.disturbance_type not in ("sinusoidal", "step", "sweep"):
            raise ValueError("disturbance_type must be 'sinusoidal', 'step' or 'sweep'.")
        if self.disturbance_step_onset < 0:
            raise ValueError("disturbance_step_onset must be non-negative.")

    def __post_init__(self) -> None:
        self.validate()


@dataclass
class DualLoopEnvConfig(EnvConfig):
    """Configuration for the fast-RF + slow-tuner dual-loop environment.

    Attributes:
        detuning_limit:     Max actuator detuning (maps action [-1, 1]), rad/s.
        actuator_tau:       First-order actuator time constant, s.
        base_detuning:      Static detuning offset of the disturbance, rad/s.
        disturbance_amp / disturbance_freq / randomize_disturbance:
                    Same meaning as in :class:`FrequencyEnvConfig`.
        disturbance_type / disturbance_step_onset:
                    Same meaning as in :class:`FrequencyEnvConfig`.
        kp / ki:            Gain of the inner complex-PI fast loop.
    """

    detuning_limit: float = 2.0 * np.pi * 3.0e3
    actuator_tau: float = 8.0e-5
    base_detuning: float = 2.0 * np.pi * 2.0e2
    disturbance_amp: float = 2.0 * np.pi * 2.0e2
    disturbance_freq: float = 250.0
    disturbance_type: str = "sinusoidal"
    disturbance_step_onset: float = 6.0e-4
    randomize_disturbance: bool = True
    kp: float = 0.8
    ki: float = 2.0e5

    def validate(self) -> None:
        super().validate()
        if self.detuning_limit <= 0:
            raise ValueError("detuning_limit must be positive.")
        if self.actuator_tau <= 0:
            raise ValueError("actuator_tau must be positive.")
        if self.kp < 0 or self.ki < 0:
            raise ValueError("kp and ki must be non-negative.")
        if self.disturbance_amp < 0:
            raise ValueError("disturbance_amp must be non-negative.")
        if self.disturbance_freq < 0:
            raise ValueError("disturbance_freq must be non-negative.")
        if self.disturbance_type not in ("sinusoidal", "step", "sweep"):
            raise ValueError("disturbance_type must be 'sinusoidal', 'step' or 'sweep'.")
        if self.disturbance_step_onset < 0:
            raise ValueError("disturbance_step_onset must be non-negative.")

    def __post_init__(self) -> None:
        self.validate()


@dataclass
class VectorSumEnvConfig(EnvConfig):
    """Configuration for the multi-cavity vector-sum environment.

    Attributes:
        cavities:              One ``CavityParams`` per cavity (must share the
                     RF layout, see ``VectorModulePlant``).
        reference_detuning:    Detuning used to build the nominal vector-sum
                     feedforward (``None`` uses ``cavities[0].detuning``).
        drive_limit_ratio:     Max RL residual as a fraction of the drive scale.
        obs_per_cavity:        Include every cavity voltage in the observation.
        tuner_action:          Add one slow per-cavity tuner actuator; the
                     action becomes ``[residual_I, residual_Q, tuner_1, ...]``
                     and the observation gains the per-cavity tuner states.
        tuner_tau:             Tuner time constant, s.
        tuner_detuning_limit:  Full-scale tuner detuning, rad/s.
        tuner_effort_weight:   Penalty weight of the mean squared tuner command.
    """

    cavities: list[CavityParams] = field(default_factory=list)
    reference_detuning: float | None = None
    drive_limit_ratio: float = 0.30
    obs_per_cavity: bool = True
    tuner_action: bool = False
    tuner_tau: float = 8.0e-5
    tuner_detuning_limit: float = 2.0 * np.pi * 3.0e3
    tuner_effort_weight: float = 0.5

    def validate(self) -> None:
        super().validate()
        if self.drive_limit_ratio <= 0:
            raise ValueError("drive_limit_ratio must be positive.")
        if self.tuner_tau <= 0 or self.tuner_detuning_limit <= 0:
            raise ValueError("tuner_tau and tuner_detuning_limit must be positive.")
        if self.tuner_effort_weight < 0:
            raise ValueError("tuner_effort_weight must be non-negative.")
        # Note: the cavity list itself is validated by ``VectorSumEnv`` /
        # ``VectorModulePlant`` because the default dataclass value must remain
        # constructible (defaults are evaluated at class-definition time).

    def __post_init__(self) -> None:
        self.validate()
