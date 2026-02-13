#!/usr/bin/env python3
"""
Shared dataclasses used by all environment backends (VSIEnv, HabitatEnv).

Extracted into a separate module so that importing one env backend does
**not** drag in the heavy dependencies of the other (e.g. Open3D, cv2,
habitat-sim).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np


@dataclass
class Observation:
    """
    Everything the policy sees — shared across all env backends.
    """

    image_paths: List[str]  # paths accumulated so far (len = step+1)
    prompt_text: str  # full instruction text for the *current* step
    step: int  # 0-indexed current step
    is_final_step: bool  # True when step == max_steps
    question: str = ""
    choices: Any = None
    question_type: str = "unknown"
    is_numerical: bool = False
    # Camera metadata (useful for reward shaping / logging)
    cam_position: Optional[np.ndarray] = None
    bbox: Optional[Tuple[List[float], List[float]]] = None
