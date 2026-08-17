from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

try:
    import numpy as np
    import pinocchio as pin
except ImportError as exc:  # pragma: no cover - depends on optional robotics stack.
    raise ImportError(
        "PinocchioRobotModel requires the robotics dependencies. Install them with "
        "`pip install -r requirements.txt`."
    ) from exc

ROBOTS_ROOT = Path(__file__).resolve().parent.parent / "robots"
ALIASES = {"panda": "franka_panda"}


@dataclass(frozen=True)
class Pose3D:
    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


@dataclass(frozen=True)
class JointLimit:
    lower: float
    upper: float

    def clamp(self, value: float) -> float:
        return min(max(value, self.lower), self.upper)


class RobotModel(Protocol):
    @property
    def dof(self) -> int:
        ...

    @property
    def joint_limits(self) -> Sequence[JointLimit]:
        ...

    @property
    def workspace_center(self) -> Sequence[float]:
        ...

    @property
    def workspace_radius(self) -> float:
        ...

    def forward_kinematics(self, joints: Sequence[float]) -> Pose3D:
        ...

    def jacobian(self, joints: Sequence[float]):
        ...

    def pose_error(self, joints: Sequence[float], target_pose: Pose3D):
        ...

    def orientation_error_norm(self, joints: Sequence[float], target_pose: Pose3D) -> float:
        ...

    def integrate(self, joints: Sequence[float], delta: Sequence[float]) -> list[float]:
        ...

    def clamp_to_limits(self, joints: Sequence[float]) -> list[float]:
        ...


class PinocchioRobotModel:
    """URDF-backed robot adapter using Pinocchio for FK and frame Jacobians."""

    def __init__(
        self,
        urdf_path: str | Path,
        end_effector_frame: str,
        name: str | None = None,
        metadata: dict | None = None,
    ):
        self.urdf_path = Path(urdf_path)
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")

        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.data = self.model.createData()
        self.end_effector_frame = end_effector_frame
        metadata = metadata or {}
        self.frame_id = self.model.getFrameId(end_effector_frame)
        if self.frame_id >= len(self.model.frames):
            raise ValueError(f"Frame {end_effector_frame!r} not found in {self.urdf_path}")

        self.name = name or self.model.name or self.urdf_path.stem
        self._joint_limits = [
            JointLimit(float(lower), float(upper))
            for lower, upper in zip(self.model.lowerPositionLimit, self.model.upperPositionLimit)
        ]
        self.home_joint_position = self._metadata_joint_vector(metadata, "home_joint_position")
        workspace = metadata.get("workspace", {})
        self.workspace_center = self._metadata_vector(workspace.get("center", [0.0, 0.0, 0.0]), 3)
        self.workspace_radius = float(workspace.get("radius", 1.0))

    @property
    def dof(self) -> int:
        return self.model.nq

    @property
    def joint_limits(self) -> Sequence[JointLimit]:
        return self._joint_limits

    def forward_kinematics(self, joints: Sequence[float]) -> Pose3D:
        placement = self._frame_placement(joints)
        roll, pitch, yaw = pin.rpy.matrixToRpy(placement.rotation)
        return Pose3D(
            x=float(placement.translation[0]),
            y=float(placement.translation[1]),
            z=float(placement.translation[2]),
            roll=float(roll),
            pitch=float(pitch),
            yaw=float(yaw),
        )

    def jacobian(self, joints: Sequence[float]):
        q = self._as_q(joints)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return pin.getFrameJacobian(
            self.model,
            self.data,
            self.frame_id,
            pin.ReferenceFrame.LOCAL,
        )

    def pose_error(self, joints: Sequence[float], target_pose: Pose3D):
        current = self._frame_placement(joints)
        target = self._se3_from_pose(target_pose)
        return pin.log(current.actInv(target)).vector

    def orientation_error_norm(self, joints: Sequence[float], target_pose: Pose3D) -> float:
        return float(np.linalg.norm(self.pose_error(joints, target_pose)[3:6]))

    def integrate(self, joints: Sequence[float], delta: Sequence[float]) -> list[float]:
        q = pin.integrate(self.model, self._as_q(joints), np.asarray(delta, dtype=float))
        return self.clamp_to_limits(q.tolist())

    def clamp_to_limits(self, joints: Sequence[float]) -> list[float]:
        self._validate_joints(joints)
        return [limit.clamp(float(joint)) for limit, joint in zip(self.joint_limits, joints)]

    def _frame_placement(self, joints: Sequence[float]):
        q = self._as_q(joints)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.frame_id].copy()

    def _as_q(self, joints: Sequence[float]):
        self._validate_joints(joints)
        return np.asarray(joints, dtype=float)

    def _validate_joints(self, joints: Sequence[float]) -> None:
        if len(joints) != self.dof:
            raise ValueError(f"expected {self.dof} joints, got {len(joints)}")

    def _metadata_joint_vector(self, metadata: dict, key: str) -> list[float]:
        values = metadata.get(key)
        if values is None:
            midpoint = [(limit.lower + limit.upper) * 0.5 for limit in self.joint_limits]
            return self.clamp_to_limits(midpoint)
        return self.clamp_to_limits(self._metadata_vector(values, self.dof))

    @staticmethod
    def _metadata_vector(values, expected_size: int) -> list[float]:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError("metadata vector must be a numeric sequence")
        if len(values) != expected_size:
            raise ValueError(f"metadata vector must have {expected_size} values")
        return [float(value) for value in values]

    @staticmethod
    def _se3_from_pose(pose: Pose3D):
        rotation = pin.rpy.rpyToMatrix(pose.roll, pose.pitch, pose.yaw)
        translation = np.array([pose.x, pose.y, pose.z], dtype=float)
        return pin.SE3(rotation, translation)


def load_robot_model(name: str) -> PinocchioRobotModel:
    robot_name = ALIASES.get(name, name)
    robot_dir = ROBOTS_ROOT / robot_name
    urdf_path = robot_dir / f"{robot_name}.urdf"
    dimensions_path = robot_dir / "dimensions.yaml"
    if not robot_dir.exists():
        choices = ", ".join(available_robot_names())
        raise ValueError(f"Unknown robot {name!r}. Choose one of: {choices}.")
    if not urdf_path.exists():
        raise FileNotFoundError(f"Expected URDF at {urdf_path}")
    metadata = _load_dimensions_metadata(dimensions_path)
    end_effector = metadata.get("end_effector_frame")
    if not end_effector:
        raise ValueError(f"{dimensions_path} must define end_effector_frame.")

    return PinocchioRobotModel(
        urdf_path,
        end_effector_frame=end_effector,
        name=robot_name,
        metadata=metadata,
    )


def available_robot_names() -> list[str]:
    if not ROBOTS_ROOT.exists():
        return []
    return sorted(
        path.name
        for path in ROBOTS_ROOT.iterdir()
        if path.is_dir() and (path / f"{path.name}.urdf").exists()
    )


def _load_dimensions_metadata(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Expected robot metadata at {path}")
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Robot metadata loading requires PyYAML.") from exc
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data
