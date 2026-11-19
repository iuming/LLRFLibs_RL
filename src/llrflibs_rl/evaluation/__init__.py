###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""Evaluation layer: physics-aware metrics and report generation."""

from .kpis import KPIS, kpi_report, kpi_report_from_wave
from .loop import cavity_plant_abcd, loop_margins, pi_controller_abcd
from .metrics import (
    action_from_model,
    cross_check_with_sb3,
    evaluate_episode,
    evaluate_over_seeds,
)
from .reporting import write_markdown_table, write_results_json
from .spectral import tracking_error_spectrum

__all__ = [
    "KPIS",
    "action_from_model",
    "cavity_plant_abcd",
    "cross_check_with_sb3",
    "evaluate_episode",
    "evaluate_over_seeds",
    "kpi_report",
    "kpi_report_from_wave",
    "loop_margins",
    "pi_controller_abcd",
    "tracking_error_spectrum",
    "write_markdown_table",
    "write_results_json",
]
