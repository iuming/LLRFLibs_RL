###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Lightweight structured logging.

These helpers produce formatted section banners and key/value tables that are
printed at the end of training / evaluation runs.  All messages are plain
English so that log files stay greppable.
"""

from __future__ import annotations

__all__ = ["print_header", "print_metrics"]


def print_header(title: str, width: int = 72) -> None:
    """Print a centered banner.

    Args:
        title: Banner text.
        width: Total banner width.
    """
    fill = max(0, width - len(title) - 2)
    left = fill // 2
    right = fill - left
    print("=" * width)
    print(" " * left + title + " " * right)
    print("=" * width)


def print_metrics(title: str, metrics: dict[str, float], verbose: bool = True) -> None:
    """Print a sorted key/value metric table.

    Args:
        title:   Table title.
        metrics: Mapping of metric name to value.
        verbose: Whether to actually print (set to ``False`` to compute only).
    """
    if not verbose:
        return
    print(f"\n--- {title} ---")
    if not metrics:
        print("  (no metrics)")
        return
    width = max(len(k) for k in metrics)
    for key, value in metrics.items():
        print(f"  {key:<{width}} : {value:,.4f}")
