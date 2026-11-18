###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Report generation.

Small helpers that turn evaluation result dictionaries into readable Markdown
tables and JSON files.  The example scripts use these to summarize a run
(baseline vs RL) right next to the artifact figures.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

__all__ = ["write_results_json", "write_markdown_table"]


def write_results_json(
    results: Dict[str, object],
    path: Path,
    extra: Dict[str, object] | None = None,
) -> Path:
    """Write a results dictionary as JSON.

    Args:
        results: Metric / result mapping.
        path:    Output path.
        extra:   Optional extra metadata appended as ``"meta"``.

    Returns:
        The written path.
    """
    payload: Dict[str, object] = dict(results)
    if extra:
        payload["meta"] = extra
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _format_value(value) -> str:
    if isinstance(value, float):
        if abs(value) >= 1e6 or (abs(value) > 0.0 and abs(value) < 1e-3):
            return f"{value:.2e}"
        return f"{value:.4f}"
    return str(value)


def write_markdown_table(
    rows: Sequence[Dict[str, object]],
    path: Path,
    columns: Sequence[str] | None = None,
    title: str = "Results",
) -> Path:
    """Write a list of result dicts as a Markdown table.

    Args:
        rows:    Each row is a mapping of column name to value.  Column order
                 is taken from the first row unless ``columns`` is given.
        path:    Output path.
        columns: Optional explicit column order.
        title:   Markdown heading.

    Returns:
        The written path.
    """
    if not rows:
        path.write_text(f"# {title}\n\n(no rows)\n", encoding="utf-8")
        return path
    cols = list(columns) if columns is not None else list(rows[0].keys())
    lines = [f"# {title}", "", "| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for row in rows:
        cells = [_format_value(row.get(c, "-")) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def timestamped_filename(base: str) -> str:
    """Return ``base_YYYYmmdd-HHMMSS``.

    Args:
        base: Base name without extension.

    Returns:
        Timestamped name (no extension).
    """
    return f"{base}_{time.strftime('%Y%m%d-%H%M%S')}"
