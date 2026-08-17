from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np

from .pinocchio_model import Pose3D

OUTSIDE_WORKSPACE_MARGIN = 0.05


@dataclass(frozen=True)
class Scenario:
    label: str
    current_joints: list[float]
    target_pose: Pose3D
    expected: str
    config_overrides: dict[str, Any] | None = None


def build_scenarios(robot) -> list[Scenario]:
    home = home_joints(robot)
    reachable_goal = _offset_joints(robot, home, [0.08, -0.04, 0.05, 0.03, -0.02, 0.02, 0.0])

    home_pose = robot.forward_kinematics(home)
    
    joint_limit_start, joint_limit_target = _joint_limit_blocked_case(robot)
    jump_limit_start, jump_limit_target = _jump_limit_blocked_case(robot)
    progress_start, progress_target = _solver_progress_limited_case(robot)

    return [
        Scenario(
            label="reachable from home",
            current_joints=home,
            target_pose=robot.forward_kinematics(reachable_goal),
            expected="success",
        ),
        Scenario(
            label="joint-limit blocked target",
            current_joints=joint_limit_start,
            target_pose=joint_limit_target,
            expected="joint-limit constrained result",
        ),
        Scenario(
            label="jump-limit blocked target",
            current_joints=jump_limit_start,
            target_pose=jump_limit_target,
            expected="jump-limit constrained result",
        ),
        Scenario(
            label="solver progress limited",
            current_joints=progress_start,
            target_pose=progress_target,
            expected="solver progress constrained result",
            config_overrides={
                "max_iterations": 50,
                "max_solution_jump": [10.0] * robot.dof,
            },
        ),
        Scenario(
            label="outside workspace",
            current_joints=home,
            target_pose=_outside_workspace_pose(robot, home_pose),
            expected="non_convergence / outside workspace",
        ),
    ]


def build_custom_scenario(robot) -> Scenario:
    custom_start = _offset_joints(robot,home_joints(robot),[0.25, -0.22, 0.18, -0.14, 0.12, -0.10, 0.08],)
    custom_goal = _offset_joints(robot,custom_start,[0.18, 0.12, -0.16, 0.10, -0.08, 0.06, -0.04],)
    return Scenario(
        label="custom override target",
        current_joints=custom_start,
        target_pose=robot.forward_kinematics(custom_goal),
        expected="yet to be decided",
        config_overrides={
            "max_iterations": 80,
            "max_step": 0.08,
            "damping": 0.05,
            "orientation_weight": 0.8,
            "max_solution_jump": [0.75] * robot.dof,
        },
    )


def scenario_constraints(base_config, scenario: Scenario):
    if not scenario.config_overrides:
        return base_config
    return replace(base_config, **scenario.config_overrides)


def home_joints(robot) -> list[float]:
    return robot.clamp_to_limits(robot.home_joint_position)


def _joint_limit_blocked_case(robot) -> tuple[list[float], Pose3D]:
    joint_index = _joint_limit_demo_index(robot)
    start = home_joints(robot)
    limit = robot.joint_limits[joint_index]
    start[joint_index] = limit.upper - min(0.03, 0.05 * (limit.upper - limit.lower))
    target_joints = list(start)
    target_joints[joint_index] = limit.upper + 0.12
    return robot.clamp_to_limits(start), robot.forward_kinematics(target_joints)


def _jump_limit_blocked_case(robot) -> tuple[list[float], Pose3D]:
    start = _offset_joints(robot, home_joints(robot), [-0.18, 0.14, -0.12, 0.10, -0.08, 0.06, -0.04])
    return start, robot.forward_kinematics(_jump_limit_blocked_goal(robot, start))


def _jump_limit_blocked_goal(robot, start: Sequence[float]) -> list[float]:
    goal = list(start)
    for joint_index, limit in enumerate(robot.joint_limits):
        direction = 1.0 if joint_index % 2 == 0 else -1.0
        desired = goal[joint_index] + direction * 0.85
        if desired > limit.upper - 0.05:
            desired = goal[joint_index] - 0.85
        if desired < limit.lower + 0.05:
            desired = goal[joint_index] + 0.85
        goal[joint_index] = desired
    return robot.clamp_to_limits(goal)


def _solver_progress_limited_case(robot) -> tuple[list[float], Pose3D]:
    examples = {
        "franka_panda": (
            [0.727, 0.667, 1.651, -0.235, -1.011, 3.477, -2.756],
            [0.062, 0.306, 2.299, -0.424, -1.648, 2.237, 0.987],
        ),
        "ur5": (
            [-0.181, -1.840, 1.473, -1.799, -0.581, -0.227],
            [-5.757, -4.949, 2.653, 5.996, -0.049, -0.103],
        ),
    }
    if robot.name in examples:
        start, goal = examples[robot.name]
        return robot.clamp_to_limits(start), robot.forward_kinematics(robot.clamp_to_limits(goal))

    start = _offset_joints(
        robot,
        home_joints(robot),
        [0.7 if index % 2 == 0 else -0.7 for index in range(robot.dof)],
    )
    goal = _offset_joints(
        robot,
        start,
        [-1.1 if index % 2 == 0 else 1.1 for index in range(robot.dof)],
    )
    return start, robot.forward_kinematics(goal)


def _outside_workspace_pose(robot, reference_pose: Pose3D) -> Pose3D:
    center = np.asarray(robot.workspace_center, dtype=float)
    reference = np.array([reference_pose.x, reference_pose.y, reference_pose.z], dtype=float)
    direction = reference - center
    if np.linalg.norm(direction) < 1e-9:
        direction = np.array([1.0, 0.0, 0.0])
    direction = direction / np.linalg.norm(direction)
    target = center + direction * (robot.workspace_radius + OUTSIDE_WORKSPACE_MARGIN)
    return Pose3D(
        x=float(target[0]),
        y=float(target[1]),
        z=float(target[2]),
        roll=reference_pose.roll,
        pitch=reference_pose.pitch,
        yaw=reference_pose.yaw,
    )


def _offset_joints(robot, joints: Sequence[float], offsets: Sequence[float]) -> list[float]:
    padded_offsets = list(offsets[: robot.dof])
    padded_offsets.extend([0.0] * (robot.dof - len(padded_offsets)))
    return robot.clamp_to_limits(
        [joint + offset for joint, offset in zip(joints, padded_offsets)]
    )


def _joint_limit_demo_index(robot) -> int:
    spans = [limit.upper - limit.lower for limit in robot.joint_limits]
    if robot.dof <= 1:
        return 0
    return min(range(1, robot.dof), key=lambda index: spans[index])
