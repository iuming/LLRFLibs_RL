###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Per-environment agent presets.

These helpers map an environment id to a recommended training budget /
algorithm, which keeps the CLI and the example scripts consistent and avoids
boom-and-bust training budgets across tasks.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional

from ..configs.agent import AgentConfig
from ..registry import recommended_timesteps

__all__ = ["agent_config_for"]

#: Per-environment overrides on top of the defaults.  ``VectorSumTuner-v0``
#: is unstable under the default on-policy PPO on longer budgets; a stable
#: off-policy SAC setup (study: experiments/vector_sum_tuner_study.py) is used
#: instead.
_ENV_AGENT_OVERRIDES: Dict[str, Dict[str, object]] = {
    "VectorSumTuner-v0": {
        "algorithm": "sac",
        "train_freq": 64,
        "gradient_steps": 64,
        "batch_size": 256,
        "normalize_obs": False,
    },
    "FrequencyControl-v0": {
        "algorithm": "sac",
    },
}


def agent_config_for(
    env_id: str,
    timesteps: Optional[int] = None,
    algorithm: Optional[str] = None,
    seed: int = 0,
    quick: bool = False,
    verbose: int = 0,
) -> AgentConfig:
    """Return the recommended :class:`AgentConfig` for an environment.

    Per-environment presets can override the algorithm and training
    hyperparameters (e.g. SAC for ``VectorSumTuner-v0``).

    Args:
        env_id:     Environment id.
        timesteps:  Explicit training budget (overrides the recommendation).
        algorithm:  Algorithm name (``"ppo"`` / ``"sac"`` / ``"td3"``); the
                    per-environment preset wins when this is ``None``.
        seed:       Random seed.
        quick:      Use a small sanity-check budget.
        verbose:    SB3 verbosity.

    Returns:
        A validated :class:`AgentConfig`.
    """
    budget = int(timesteps) if timesteps is not None else recommended_timesteps(env_id, quick=quick)
    overrides = dict(_ENV_AGENT_OVERRIDES.get(env_id, {}))
    if algorithm is not None:
        overrides["algorithm"] = algorithm
    cfg = AgentConfig(algorithm="ppo", timesteps=budget, seed=seed, verbose=verbose)
    return replace(cfg, **overrides)
