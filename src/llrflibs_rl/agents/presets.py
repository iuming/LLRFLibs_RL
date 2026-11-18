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

from typing import Optional

from ..configs.agent import AgentConfig
from ..registry import recommended_timesteps

__all__ = ["agent_config_for"]


def agent_config_for(
    env_id: str,
    timesteps: Optional[int] = None,
    algorithm: str = "ppo",
    seed: int = 0,
    quick: bool = False,
    verbose: int = 0,
) -> AgentConfig:
    """Return the recommended :class:`AgentConfig` for an environment.

    Args:
        env_id:     Environment id.
        timesteps:  Explicit training budget (overrides the recommendation).
        algorithm:  Algorithm name (``"ppo"`` or ``"sac"``).
        seed:       Random seed.
        quick:      Use a small sanity-check budget.
        verbose:    SB3 verbosity.

    Returns:
        A validated :class:`AgentConfig`.
    """
    budget = int(timesteps) if timesteps is not None else recommended_timesteps(env_id, quick=quick)
    return AgentConfig(algorithm=algorithm, timesteps=budget, seed=seed, verbose=verbose)
