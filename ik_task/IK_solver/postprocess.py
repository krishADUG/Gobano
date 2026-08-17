from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def joint_limit(robot: Any, q: np.ndarray) -> str | None:
    """Dummy post-processor hook for joint-limit notes."""
    return None


def jump_limit(robot: Any, q: np.ndarray, start_position: Sequence[float]) -> str | None:
    """Dummy post-processor hook for jump-limit notes."""
    return None


def solver_stuck(error_history: list[float]) -> str | None:
    """Dummy post-processor hook for stalled solver notes."""
    return None


def outside_workspace(robot: Any, goal_pose: Any) -> str | None:
    """Dummy post-processor hook for workspace notes."""
    return None


def collect_notes(
    robot: Any,
    q: np.ndarray,
    start_position: Sequence[float],
    goal_pose: Any,
    error_history: list[float],
) -> list[str]:
    notes = [
        joint_limit(robot, q),
        jump_limit(robot, q, start_position),
        solver_stuck(error_history),
        outside_workspace(robot, goal_pose),
    ]
    return [note for note in notes if note]
