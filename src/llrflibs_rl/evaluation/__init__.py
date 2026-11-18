###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""Evaluation layer: physics-aware metrics and report generation."""

from .metrics import action_from_model, evaluate_episode, evaluate_over_seeds
from .reporting import write_markdown_table, write_results_json

__all__ = [
    "action_from_model",
    "evaluate_episode",
    "evaluate_over_seeds",
    "write_markdown_table",
    "write_results_json",
]
