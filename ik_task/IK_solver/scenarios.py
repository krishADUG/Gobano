from __future__ import annotations

from dataclasses import dataclass

from IK_solver.pinocchio_model import Pose3D, load_robot_model


@dataclass(frozen=True)
class IKScene:
    name: str
    robot_name: str
    start_position: list[float]
    goal_pose: Pose3D
    description: str


def build_scene(robot_name: str, scene_name: str) -> IKScene:
    robot = load_robot_model(robot_name)
    scenes = {
        "scene1_reachable": _reachable,
        "scene2_close_to_joint_limit": _close_to_joint_limit,
        "scene3_close_to_jump_limit": _close_to_jump_limit,
        "scene4_exhaust_solver": _exhaust_solver,
        "scene5_outside_workspace": _outside_workspace,
    }
    if scene_name not in scenes:
        available = ", ".join(scenes)
        raise ValueError(f"Unknown scene '{scene_name}'. Available scenes: {available}")
    start, goal_q, description = scenes[scene_name](robot)
    goal_pose = robot.forward_kinematics(goal_q)
    if scene_name == "scene5_outside_workspace":
        goal_pose = Pose3D(
            x=float(robot.workspace_center[0] + robot.workspace_radius + 0.25),
            y=float(robot.workspace_center[1]),
            z=float(robot.workspace_center[2]),
            roll=goal_pose.roll,
            pitch=goal_pose.pitch,
            yaw=goal_pose.yaw,
        )
    return IKScene(scene_name, robot_name, start, goal_pose, description)


def available_scenes() -> list[str]:
    return [
        "scene1_reachable",
        "scene2_close_to_joint_limit",
        "scene3_close_to_jump_limit",
        "scene4_exhaust_solver",
        "scene5_outside_workspace",
    ]


def _reachable(robot):
    start = list(robot.home_joint_position)
    goal = _add_offset(start, [0.03, -0.02, 0.02, 0.0, 0.02, -0.01, 0.01])
    return start, goal, "Home position with a small reachable offset."


def _close_to_joint_limit(robot):
    start = list(robot.home_joint_position)
    start[0] = robot.joint_limits[0].upper - 0.02
    goal = list(start)
    goal[0] = min(robot.joint_limits[0].upper, start[0] + 0.12)
    for index in range(1, robot.dof):
        goal[index] = robot.joint_limits[index].clamp(goal[index] + 0.02)
    return start, goal, "Reachable pose blocked by starting near a joint limit."


def _close_to_jump_limit(robot):
    start = list(robot.home_joint_position)
    goal = [
        limit.clamp(joint + 0.4)
        for joint, limit in zip(start, robot.joint_limits)
    ]
    return start, goal, "Reachable pose requiring larger jumps than configured."


def _exhaust_solver(robot):
    start = list(robot.home_joint_position)
    goal = [
        limit.clamp(joint + offset)
        for joint, offset, limit in zip(
            start,
            _offset(robot.dof, [0.45, -0.35, 0.3, -0.25, 0.2, -0.15, 0.1]),
            robot.joint_limits,
        )
    ]
    return start, goal, "Reachable target intended to stress the iteration limit."


def _outside_workspace(robot):
    start = list(robot.home_joint_position)
    return start, list(start), "Goal translated just outside the configured workspace."


def _offset(dof: int, values: list[float]) -> list[float]:
    offset = [0.0] * dof
    offset[: min(dof, len(values))] = values[:dof]
    return offset


def _add_offset(values: list[float], offset_values: list[float]) -> list[float]:
    return [value + offset for value, offset in zip(values, _offset(len(values), offset_values))]
