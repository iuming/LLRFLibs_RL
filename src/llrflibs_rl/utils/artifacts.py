###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Run-artifact management.

Every training / evaluation run writes a self-contained directory under
``runs/`` containing the environment config, the trained model, evaluation
metrics and figures.  This makes individual runs reproducible from a single
folder.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["RunArtifacts"]


def _default_runs_dir() -> Path:
    """Return the repository-level ``runs`` directory."""
    # <repo>/src/llrflibs_rl/utils/artifacts.py -> <repo>
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "runs"


def _sanitize_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


class RunArtifacts:
    """Collect and persist the outputs of one experiment run.

    Args:
        run_name:   Short identifier of the experiment (e.g. ``DualLoop-v0``).
        runs_dir:   Base directory for runs (default: ``<repo>/runs``).
    """

    def __init__(self, run_name: str, runs_dir: Path | None = None) -> None:
        base = Path(runs_dir) if runs_dir is not None else _default_runs_dir()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = base / f"{_sanitize_name(run_name)}-{stamp}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._figure_count = 0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_config(self, config: dict[str, Any], filename: str = "config.json") -> Path:
        """Write a JSON-serializable configuration next to the other artifacts.

        Args:
            config:   Configuration mapping.
            filename: Output file name.

        Returns:
            Path of the written file.
        """
        path = self.dir / filename
        path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def save_metrics(self, metrics: dict[str, float], filename: str = "metrics.json") -> Path:
        """Write evaluation metrics as JSON.

        Args:
            metrics:  Metric mapping.
            filename: Output file name.

        Returns:
            Path of the written file.
        """
        path = self.dir / filename
        path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def save_text(self, content: str, filename: str = "report.txt") -> Path:
        """Write a plain-text report.

        Args:
            content:  Report body.
            filename: Output file name.

        Returns:
            Path of the written file.
        """
        path = self.dir / filename
        path.write_text(content, encoding="utf-8")
        return path

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def save_figure(self, fig: plt.Figure, name: str) -> Path:
        """Save a matplotlib figure into the run directory.

        Args:
            fig:  Figure to save (closed afterwards).
            name: Base name (``.png`` added automatically).

        Returns:
            Path of the written file.
        """
        self._figure_count += 1
        path = self.dir / f"{name}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # Waveform utilities shared by the example figures
    # ------------------------------------------------------------------

    @staticmethod
    def time_us(env: Any) -> np.ndarray:
        """Return a microsecond time axis for an environment's episode.

        Args:
            env: Environment exposing ``config.Ts`` and ``episode_len``.

        Returns:
            Time axis in microseconds.
        """
        return np.arange(env.episode_len) * env.config.Ts * 1e6
