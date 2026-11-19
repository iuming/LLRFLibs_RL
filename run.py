###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Repository-root CLI bootstrap — runs the ``llrflibs-rl`` command line without
installing the package (no ``pip install -e .`` and no ``PYTHONPATH`` needed).

Usage (from the repository root)::

    python run.py list
    python run.py train --env CavityVoltage-v0 --render --render-freq 1024
    python run.py demo --env CavityVoltage-v0 --policy pi --show
    python run.py verify

The single-file example ``examples/render_demo.py`` follows the same pattern.
After ``pip install -e .`` you can also use ``python -m llrflibs_rl.scripts.run``
or the ``llrflibs-rl`` console script.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def main() -> None:
    from llrflibs_rl.scripts.run import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
