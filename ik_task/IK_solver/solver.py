from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite, sqrt
from pathlib import Path
from typing import Sequence

import numpy as np

from .pinocchio_model import ALIASES, ROBOTS_ROOT, Pose3D, RobotModel


class SolveStatus(str, Enum):
    SUCCESS = "success"
    APPROXIMATE = "approximate"
    INVALID_INPUT = "invalid_input"
    NON_CONVERGENCE = "non_convergence"


@dataclass(frozen=True)
class ConstraintConfig:
    position_tolerance: float
    orientation_tolerance: float
    approx_position_multiplier: float
    approx_orientation_multiplier: float
    max_iterations: int
    damping: float
    max_step: float
    orientation_weight: float
    motion_weight: float
    max_solution_jump: list[float]
    qp_solver: str
    improvement_epsilon: float
    stagnation_iterations: int


REQUIRED_CONFIG_KEYS = tuple(ConstraintConfig.__dataclass_fields__)


def load_constraint_config(
    robot_name: str | None = None,
    path: str | Path | None = None,
) -> ConstraintConfig:
    if path is not None:
        config_path = Path(path)
    elif robot_name is not None:
        normalized_name = ALIASES.get(robot_name, robot_name)
        config_path = ROBOTS_ROOT / normalized_name / "solver_config.yaml"
    else:
        raise ValueError("load_constraint_config requires a robot_name or explicit path.")

    if not config_path.exists():
        raise FileNotFoundError(f"Solver config not found: {config_path}")
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Solver config loading requires PyYAML.") from exc

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping.")

    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in data]
    if missing:
        raise ValueError(
            f"{config_path} is missing required solver setting(s): "
            + ", ".join(missing)
        )

    unknown = sorted(set(data) - set(REQUIRED_CONFIG_KEYS))
    if unknown:
        raise ValueError(
            f"{config_path} contains unknown solver setting(s): "
            + ", ".join(unknown)
        )

    values = {key: data[key] for key in REQUIRED_CONFIG_KEYS}
    return ConstraintConfig(**values)


@dataclass(frozen=True)
class IterationTrace:
    iteration: int
    joint_position: list[float]
    achieved_pose: Pose3D
    position_error: float
    orientation_error: float
    cost: float


@dataclass(frozen=True)
class SolveResult:
    status: SolveStatus
    joint_position: list[float]
    achieved_pose: Pose3D
    position_error: float
    orientation_error: float
    iterations: int
    message: str
    trace: list[IterationTrace]


def solve(
    robot_model: RobotModel,
    target_pose: Pose3D,
    current_joint_position: Sequence[float],
    constraints: ConstraintConfig,
    collect_trace: bool = False,
) -> SolveResult:
    config = constraints
    trace: list[IterationTrace] = []
    invalid_reason = _validate_inputs(robot_model, target_pose, current_joint_position, config)
    if invalid_reason:
        fallback = _fallback_pose(robot_model, current_joint_position)
        return SolveResult(
            status=SolveStatus.INVALID_INPUT,
            joint_position=list(current_joint_position),
            achieved_pose=fallback,
            position_error=float("inf"),
            orientation_error=float("inf"),
            iterations=0,
            message=invalid_reason,
            trace=trace,
        )

    q_ref = robot_model.clamp_to_limits(current_joint_position)
    workspace_reason = _workspace_violation(robot_model, target_pose)
    if workspace_reason:
        fallback = robot_model.forward_kinematics(q_ref)
        return SolveResult(
            status=SolveStatus.NON_CONVERGENCE,
            joint_position=q_ref,
            achieved_pose=fallback,
            position_error=_position_error(fallback, target_pose),
            orientation_error=robot_model.orientation_error_norm(q_ref, target_pose),
            iterations=0,
            message=workspace_reason,
            trace=trace,
        )

    q = list(q_ref)
    best_q = list(q)
    best_pose = robot_model.forward_kinematics(q)
    best_cost = _spatial_cost(robot_model, target_pose, q, q_ref, config)
    if collect_trace:
        _append_trace(trace, 0, robot_model, target_pose, q, q_ref, config)
    stagnant_count = 0
    max_stagnant_count = 0

    for iteration in range(1, config.max_iterations + 1):
        pose = robot_model.forward_kinematics(q)
        position_error = _position_error(pose, target_pose)
        orientation_error = robot_model.orientation_error_norm(q, target_pose)
        if (
            position_error <= config.position_tolerance
            and orientation_error <= config.orientation_tolerance
        ):
            return SolveResult(
                status=SolveStatus.SUCCESS,
                joint_position=q,
                achieved_pose=pose,
                position_error=position_error,
                orientation_error=orientation_error,
                iterations=iteration - 1,
                message="Target reached within tolerances.",
                trace=trace,
            )

        error = np.asarray(robot_model.pose_error(q, target_pose), dtype=float)
        jacobian = np.asarray(robot_model.jacobian(q), dtype=float)
        weights = np.array([1.0, 1.0, 1.0, *([config.orientation_weight] * 3)])
        weighted_jacobian = jacobian * weights[:, None]
        weighted_error = error * weights
        dq = _qp_step(weighted_jacobian, weighted_error, q, q_ref, robot_model, config)

        if float(np.max(np.abs(dq))) < 1e-12:
            stagnant_count += 1
            max_stagnant_count = max(max_stagnant_count, stagnant_count)
            if collect_trace:
                _append_trace(trace, iteration, robot_model, target_pose, q, q_ref, config)
            continue

        q = robot_model.integrate(q, dq)
        if collect_trace:
            _append_trace(trace, iteration, robot_model, target_pose, q, q_ref, config)
        pose = robot_model.forward_kinematics(q)
        cost = _spatial_cost(robot_model, target_pose, q, q_ref, config)
        if cost + config.improvement_epsilon < best_cost:
            best_cost = cost
            best_q = list(q)
            best_pose = pose
            stagnant_count = 0
        else:
            stagnant_count += 1
            max_stagnant_count = max(max_stagnant_count, stagnant_count)

    position_error = _position_error(best_pose, target_pose)
    orientation_error = robot_model.orientation_error_norm(best_q, target_pose)
    return SolveResult(
        status=_final_status(position_error, orientation_error, config),
        joint_position=best_q,
        achieved_pose=best_pose,
        position_error=position_error,
        orientation_error=orientation_error,
        iterations=config.max_iterations,
        message=_failure_message(
            robot_model,
            best_q,
            q_ref,
            config,
            position_error,
            orientation_error,
            max_stagnant_count,
        ),
        trace=trace,
    )


def _append_trace(
    trace: list[IterationTrace],
    iteration: int,
    robot_model: RobotModel,
    target_pose: Pose3D,
    q: Sequence[float],
    q_ref: Sequence[float],
    config: ConstraintConfig,
) -> None:
    pose = robot_model.forward_kinematics(q)
    trace.append(
        IterationTrace(
            iteration=iteration,
            joint_position=list(q),
            achieved_pose=pose,
            position_error=_position_error(pose, target_pose),
            orientation_error=robot_model.orientation_error_norm(q, target_pose),
            cost=_spatial_cost(robot_model, target_pose, q, q_ref, config),
        )
    )


def _validate_inputs(
    robot_model: RobotModel,
    target_pose: Pose3D,
    current_joint_position: Sequence[float],
    config: ConstraintConfig,
) -> str | None:
    if not isinstance(target_pose, Pose3D):
        return "target_pose must be a Pose3D."
    if len(current_joint_position) != robot_model.dof:
        return f"Expected {robot_model.dof} joints, got {len(current_joint_position)}."
    required_methods = ("pose_error", "orientation_error_norm", "integrate")
    missing = [name for name in required_methods if not hasattr(robot_model, name)]
    if missing:
        return "IK requires a Pinocchio-style robot model with " + ", ".join(missing) + "."
    values = [
        target_pose.x,
        target_pose.y,
        target_pose.z,
        target_pose.roll,
        target_pose.pitch,
        target_pose.yaw,
        *current_joint_position,
    ]
    if not all(isfinite(value) for value in values):
        return "Inputs must be finite numbers."
    if config.max_iterations <= 0:
        return "max_iterations must be positive."
    if config.position_tolerance < 0 or config.orientation_tolerance < 0:
        return "Tolerances must be non-negative."
    if config.approx_position_multiplier < 1.0 or config.approx_orientation_multiplier < 1.0:
        return "Approximate tolerance multipliers must be at least 1.0."
    if config.damping <= 0 or config.max_step <= 0:
        return "damping and max_step must be positive."
    if len(config.max_solution_jump) != robot_model.dof:
        return (
            f"max_solution_jump must contain {robot_model.dof} values, "
            f"got {len(config.max_solution_jump)}."
        )
    if not all(isfinite(value) and value > 0.0 for value in config.max_solution_jump):
        return "max_solution_jump values must be finite positive numbers."
    return None


def _final_status(
    position_error: float,
    orientation_error: float,
    config: ConstraintConfig,
) -> SolveStatus:
    if (
        position_error <= config.approx_position_multiplier * config.position_tolerance
        and orientation_error
        <= config.approx_orientation_multiplier * config.orientation_tolerance
    ):
        return SolveStatus.APPROXIMATE
    return SolveStatus.NON_CONVERGENCE


def _failure_message(
    robot_model: RobotModel,
    joints: Sequence[float],
    reference_joints: Sequence[float],
    config: ConstraintConfig,
    position_error: float,
    orientation_error: float,
    max_stagnant_count: int,
) -> str:
    return (
        _joint_limit_reason(robot_model, joints)
        or _jump_limit_reason(joints, reference_joints, config)
        or _stagnation_reason(config, max_stagnant_count, position_error, orientation_error)
        or (
            f"Solver exhausted max_iterations={config.max_iterations} while remaining "
            "outside tolerance. Final "
            f"position error {position_error:.5f} m and orientation error "
            f"{orientation_error:.5f} rad."
        )
    )


def _stagnation_reason(
    config: ConstraintConfig,
    max_stagnant_count: int,
    position_error: float,
    orientation_error: float,
) -> str | None:
    if max_stagnant_count < config.stagnation_iterations:
        return None
    return (
        "Solver stagnated: the best cost did not improve by at least "
        f"improvement_epsilon={config.improvement_epsilon:g} for "
        f"{max_stagnant_count} consecutive iteration(s). Reached "
        f"max_iterations={config.max_iterations} with position error "
        f"{position_error:.5f} m and orientation error {orientation_error:.5f} rad."
    )


def _joint_limit_reason(robot_model: RobotModel, joints: Sequence[float]) -> str | None:
    active_limits = []
    tolerance = 1e-4
    for index, (joint, limit) in enumerate(zip(joints, robot_model.joint_limits), start=1):
        if abs(joint - limit.lower) <= tolerance:
            active_limits.append(f"J{index} lower_limit")
        elif abs(joint - limit.upper) <= tolerance:
            active_limits.append(f"J{index} upper_limit")
    if not active_limits:
        return None
    return (
        "Joint limits prevented convergence. "
        f"Active bounded joint(s): {', '.join(active_limits)}."
    )


def _jump_limit_reason(
    joints: Sequence[float],
    reference_joints: Sequence[float],
    config: ConstraintConfig,
) -> str | None:
    active_limit: tuple[int, float, float] | None = None
    tolerance = 1e-4
    for index, (joint, reference, jump_limit) in enumerate(
        zip(joints, reference_joints, config.max_solution_jump),
        start=1,
    ):
        jump = abs(joint - reference)
        if jump < jump_limit - tolerance:
            continue
        if active_limit is None or jump > active_limit[1]:
            active_limit = (index, jump, jump_limit)
    if active_limit is None:
        return None
    index, jump, jump_limit = active_limit
    return (
        "Jump limits prevented convergence. "
        f"Best bounded solution is at the J{index} jump limit "
        f"({jump:.4f} rad of {jump_limit:.4f} rad allowed)."
    )


def _workspace_violation(robot_model: RobotModel, target_pose: Pose3D) -> str | None:
    center = getattr(robot_model, "workspace_center", None)
    radius = getattr(robot_model, "workspace_radius", None)
    if center is None or radius is None:
        return None
    distance = sqrt(
        (target_pose.x - center[0]) ** 2
        + (target_pose.y - center[1]) ** 2
        + (target_pose.z - center[2]) ** 2
    )
    outside_by = distance - float(radius)
    if outside_by <= 0.0:
        return None
    return (
        "Point outside the workspace. Target is outside workspace"
        f" by {outside_by:.4f} m."
    )


def _fallback_pose(robot_model: RobotModel, joints: Sequence[float]) -> Pose3D:
    try:
        if len(joints) == robot_model.dof:
            return robot_model.forward_kinematics(robot_model.clamp_to_limits(joints))
    except Exception:
        pass
    return Pose3D(0.0, 0.0, 0.0)


def _position_error(pose: Pose3D, target_pose: Pose3D) -> float:
    return sqrt(
        (target_pose.x - pose.x) ** 2
        + (target_pose.y - pose.y) ** 2
        + (target_pose.z - pose.z) ** 2
    )


def _spatial_cost(
    robot_model: RobotModel,
    target_pose: Pose3D,
    q: Sequence[float],
    q_ref: Sequence[float],
    config: ConstraintConfig,
) -> float:
    pose = robot_model.forward_kinematics(q)
    position_error = _position_error(pose, target_pose)
    orientation_error = robot_model.orientation_error_norm(q, target_pose)
    motion_cost = sum((joint - ref) ** 2 for joint, ref in zip(q, q_ref))
    return (
        position_error * position_error
        + (config.orientation_weight * orientation_error) ** 2
        + config.motion_weight * motion_cost
    )


def _qp_step(
    jacobian,
    error_vector,
    q: Sequence[float],
    q_ref: Sequence[float],
    robot_model: RobotModel,
    config: ConstraintConfig,
):
    dof = len(q)
    regularization = (config.damping * config.damping) + config.motion_weight
    hessian = jacobian.T @ jacobian + regularization * np.eye(dof)
    gradient = -jacobian.T @ error_vector + config.motion_weight * (
        np.asarray(q, dtype=float) - np.asarray(q_ref, dtype=float)
    )
    lower = []
    upper = []
    for joint, reference, limit, jump_limit in zip(
        q,
        q_ref,
        robot_model.joint_limits,
        config.max_solution_jump,
    ):
        lower.append(
            max(
                -config.max_step,
                limit.lower - joint,
                reference - jump_limit - joint,
            )
        )
        upper.append(
            min(
                config.max_step,
                limit.upper - joint,
                reference + jump_limit - joint,
            )
        )

    lower_bounds = np.asarray(lower, dtype=float)
    upper_bounds = np.asarray(upper, dtype=float)
    try:
        from qpsolvers import solve_qp

        try:
            from scipy import sparse

            qp_hessian = sparse.csc_matrix(0.5 * (hessian + hessian.T))
        except ImportError:
            qp_hessian = 0.5 * (hessian + hessian.T)

        step = solve_qp(
            P=qp_hessian,
            q=gradient,
            lb=lower_bounds,
            ub=upper_bounds,
            solver=config.qp_solver,
        )
        if step is not None and np.all(np.isfinite(step)):
            return np.asarray(step, dtype=float)
    except Exception:
        pass

    try:
        step = -np.linalg.solve(hessian, gradient)
    except np.linalg.LinAlgError:
        step = -np.linalg.pinv(hessian) @ gradient
    return np.clip(step, lower_bounds, upper_bounds)
