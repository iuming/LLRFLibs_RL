###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Trained-model zoo.

A small registry of trained SB3 models on disk:

* ``models/<env_id>_<tag>.zip`` — the model (SB3 ``*.save``/``*.load``);
* ``models/<env_id>_<tag>.json`` — metadata (algorithm, timesteps, seed,
  env description, timestamp).

The ``zoo`` CLI subcommand lists and inspects the zoo; the ``gallery``
subcommand renders comparison figures from the latest zoo models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from stable_baselines3 import PPO, SAC, TD3

from ..registry import get_spec

__all__ = ["ModelZoo", "default_zoo_dir"]

_ALGORITHM_CLASSES = {"ppo": PPO, "sac": SAC, "td3": TD3}


def default_zoo_dir() -> Path:
    """Default zoo directory (``<repository>/models``)."""
    root = Path(__file__).resolve().parents[3]
    return root / "models"


class ModelZoo:
    """A directory of trained models with metadata.

    Args:
        zoo_dir: Directory holding the models (defaults to the repo ``models/``).
    """

    def __init__(self, zoo_dir: Optional[Path] = None) -> None:
        self.zoo_dir = (zoo_dir or default_zoo_dir()).resolve()
        self.zoo_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _paths(self, env_id: str, tag: str) -> Path:
        return self.zoo_dir / f"{env_id}_{tag}.zip"

    def _meta_path(self, env_id: str, tag: str) -> Path:
        return self.zoo_dir / f"{env_id}_{tag}.json"

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, model, env_id: str, tag: Optional[str] = None, **extra) -> Path:
        """Save a trained model plus its metadata.

        Args:
            model: Trained SB3 model.
            env_id: Environment id it was trained on.
            tag:    Short label (default: ``"best"``).
            extra:  Extra metadata key/values to persist.

        Returns:
            The path of the saved model.
        """
        tag = tag or "best"
        path = self._paths(env_id, tag)
        model.save(str(path))
        meta = {
            "env_id": env_id,
            "tag": tag,
            "algorithm": type(model).__name__.lower(),
            "env_description": getattr(get_spec(env_id), "description", ""),
        }
        meta.update(extra)
        self._meta_path(env_id, tag).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path

    def load(self, env_id: str, tag: str = "best"):
        """Load a model by environment id and tag.

        Args:
            env_id: Environment id.
            tag:    Tag of the model.

        Returns:
            The loaded SB3 model.

        Raises:
            FileNotFoundError: If no model exists for that pair.
        """
        path = self._paths(env_id, tag)
        if not path.exists():
            raise FileNotFoundError(f"no zoo model {env_id}_{tag}")
        meta = self.metadata(env_id, tag)
        cls = _ALGORITHM_CLASSES.get(str(meta.get("algorithm", "ppo")).lower(), PPO)
        return cls.load(str(path))

    def metadata(self, env_id: str, tag: str = "best") -> Dict[str, object]:
        """Return the metadata of one saved model."""
        path = self._meta_path(env_id, tag)
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self) -> List[Dict[str, object]]:
        """List every saved model (sorted by modification time, newest first)."""
        rows = []
        for path in sorted(self.zoo_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True):
            meta_path = path.with_suffix(".json")
            meta: Dict[str, object] = {}
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["model_path"] = str(path)
            rows.append(meta)
        return rows

    def latest(self, env_id: str):
        """Load the most recent model of an environment (any tag).

        Args:
            env_id: Environment id.

        Returns:
            The newest model or ``None``.

        Raises:
            FileNotFoundError: If the environment id is not registered.
        """
        get_spec(env_id)  # validate id
        candidates = sorted(
            self.zoo_dir.glob(f"{env_id}_*.zip"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            return None
        tag = candidates[0].stem[len(env_id) + 1:]
        return self.load(env_id, tag)
