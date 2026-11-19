###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""Protection and sensing logic layered on top of the environments."""

from .quench import QuenchDetector, detect_quench_index

__all__ = ["QuenchDetector", "detect_quench_index"]
