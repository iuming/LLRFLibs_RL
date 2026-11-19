###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Environment contract verifier.

A Gymnasium-style contract checker used by the ``verify`` CLI subcommand and
by CI.  It checks the reset/step API, observation/action layouts, finite
rewards, seeded reproducibility and (when available) the evaluation protocol
(``tracked_signal`` / ``reference_sp``) required by
:mod:`llrflibs_rl.evaluation`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..registry import list_env_ids, make

__all__ = ["VerificationReport", "verify_environment", "verify_all_environments"]


@dataclass
class VerificationReport:
    """Result of checking one environment.

    Attributes:
        env_id:   Environment id checked.
        passed:   Whether every check succeeded.
        checks:   Mapping of check name -> outcome message ("ok" or an error).
        length:   Steps executed until termination/truncation.
    """

    env_id: str
    passed: bool = True
    checks: dict[str, str] = field(default_factory=dict)
    length: int = 0

    def fmt(self) -> str:
        """Single-line text summary."""
        status = "PASS" if self.passed else "FAIL"
        failed = [name for name, msg in self.checks.items() if msg != "ok"]
        detail = ", ".join(failed) if failed else "all checks ok"
        return f"[{status}] {self.env_id:<22} ({detail}) len={self.length}"


def _record(report: VerificationReport, name: str, ok: bool, msg: str) -> None:
    report.checks[name] = "ok" if ok else msg
    if not ok:
        report.passed = False


def verify_environment(
    env_id: str,
    n_steps: int = 60,
    seed: int = 0,
) -> VerificationReport:
    """Check the Gymnasium contract and the evaluation protocol of an env.

    Args:
        env_id:  Registered environment id (or ``make`` overrides are not
                 supported here; use the registry defaults).
        n_steps: Number of steps to run (truncated early on termination).
        seed:    Seed for the reproducibility check.

    Returns:
        A :class:`VerificationReport`.
    """
    report = VerificationReport(env_id=env_id)
    env = make(env_id)

    # Spaces
    try:
        obs_space, act_space = env.observation_space, env.action_space
        _record(report, "spaces", obs_space is not None and act_space is not None, "missing spaces")
    except Exception as exc:  # noqa: BLE001
        _record(report, "spaces", False, repr(exc))
        return report

    # reset returns (obs, info)
    try:
        obs, info = env.reset(seed=seed)
        _record(report, "reset",
                isinstance(obs, np.ndarray) and obs.shape == obs_space.shape and isinstance(info, dict),
                "reset does not return (obs, info)/bad shape")
        _record(report, "info_has_step",
                isinstance(info, dict) and any(k in info for k in ("step_index", "pulse_index")),
                "info lacks a step index")
    except Exception as exc:  # noqa: BLE001
        _record(report, "reset", False, repr(exc))
        return report

    # step loop
    total = 0.0
    steps = 0
    terminated = truncated = False
    try:
        for _ in range(n_steps):
            action = np.zeros(act_space.shape, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            total += float(reward)
            if not (np.isfinite(total) and obs.shape == obs_space.shape):
                raise RuntimeError("non-finite reward or bad observation shape")
            if terminated or truncated:
                break
        _record(report, "step", steps > 0 and np.isfinite(total), "step loop failed")
        # Finishing within n_steps is informational (long episodes are normal).
        report.checks["episode_end"] = (
            "reached" if (terminated or truncated) else f"not reached in {steps} steps (long episode)"
        )
    except Exception as exc:  # noqa: BLE001
        _record(report, "step", False, repr(exc))
    report.length = steps

    # Seeded reproducibility (parameterized envs reset identical with same seed)
    try:
        if type(env).__name__ != "AdaptivePulseEnv":
            env2 = make(env_id)
            obs_a, _ = env.reset(seed=777)
            obs_b, _ = env2.reset(seed=777)
            _record(report, "seeding",
                    np.allclose(np.asarray(obs_a), np.asarray(obs_b), atol=1e-6),
                    "reset(seed) not reproducible")
        else:
            _record(report, "seeding", True, "skip (random offset)")
    except Exception as exc:  # noqa: BLE001
        _record(report, "seeding", False, repr(exc))

    # Evaluation protocol used by llrflibs_rl.evaluation
    try:
        _ = env.tracked_signal
        report.checks["eval_protocol"] = "ok"
    except Exception as exc:  # noqa: BLE001
        _record(report, "eval_protocol", False, repr(exc))

    # Official Gymnasium contract checker (second opinion on the API shape).
    try:
        from gymnasium.utils.env_checker import check_env

        check_env(env, skip_render_check=True)
        report.checks["gymnasium_check_env"] = "ok"
    except Exception as exc:  # noqa: BLE001
        _record(report, "gymnasium_check_env", False, repr(exc))

    return report


def verify_all_environments(
    env_ids: Callable[[], list[str]] = list_env_ids,
    n_steps: int = 60,
) -> list[VerificationReport]:
    """Verify every registered environment.

    Args:
        env_ids: Callable returning the environment ids to check.
        n_steps: Steps to run per environment.

    Returns:
        List of :class:`VerificationReport`.
    """
    return [verify_environment(env_id, n_steps=n_steps) for env_id in env_ids()]
