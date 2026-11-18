###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Seeding helpers.

Reproducibility is central to RL experiments; this module centralises the
numpy / random / torch-free seeding so that all training and evaluation runs
share the same policy.
"""

from __future__ import annotations

import random

import numpy as np

__all__ = ["set_global_seed"]


def set_global_seed(seed: int) -> None:
    """Seed the ``random`` and ``numpy`` global generators.

    Note:
        Torch-based algorithms (stable-baselines3) should additionally pass
        the seed through their own ``seed`` argument; this helper is the
        common baseline used by the training wrappers.

    Args:
        seed: Non-negative integer seed.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
