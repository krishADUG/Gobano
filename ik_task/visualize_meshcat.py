from __future__ import annotations

import argparse
from io import BytesIO
import textwrap
from typing import Sequence

import numpy as np
import pinocchio as pin

from IK_solver import (
    ConstraintConfig,
    Scenario,
    SolveResult,
    available_robot_names,
    build_scenarios,
    build_custom_scenario,
    load_constraint_config,
    load_robot_model,
    scenario_constraints,
    solve,
)


COLORS = {
    "start": 0xF2B84B,
    "trace": 0x243746,
    "solution": 0x126A7A,
    "target": 0xC83F49,
    "achieved": 0xF2B84B,
    "ground": 0xE7EBEF,
    "workspace": 0x4F9BC7,
    "status_success": 0x126A7A,
    "status_approximate": 0xD68A1A,
    "status_non_convergence": 0xC83F49,
    "status_invalid_input": 0x7B3F99,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive 3D Pinocchio + Meshcat IK view.")
    robot_choices = ["all", *available_robot_names()]
    parser.add_argument(
        "--robot",
        choices=robot_choices,
        default="all",
        help="Robot to show.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Ask Meshcat to open the viewer URL in your browser.",
    )
    parser.add_argument(
        "--scene",
        "--scenario",
        choices=["standard", "custom"],
        default="standard",
        help="Scene set to show: standard five or custom only.",
    )
    parser.add_argument(
        "--workspace",
        choices=["wire", "hidden"],
        default="wire",
        help="Workspace sphere style. 'wire' is easiest to see through.",
    )
    parser.add_argument(
        "--trace",
        type=_parse_bool,
        default=False,
        help="Render every collected IK iteration as a translucent intermediate pose.",
    )
    parser.add_argument(
        "trace_setting",
        nargs="?",
        help="Optional shorthand: trace=true or trace=false.",
    )
    args = parser.parse_args()
    collect_trace = _trace_enabled(args.trace, args.trace_setting)

    try:
        import meshcat
        import meshcat.geometry as g
    except ImportError as exc:
        raise SystemExit("Install Meshcat first: python3 -m pip install meshcat") from exc

    vis = meshcat.Visualizer()
    vis.delete()
    if args.open:
        vis.open()
    else:
        print(f"Meshcat viewer: {vis.url()}")

    _draw_ground(vis, g)
    robot_names = available_robot_names() if args.robot == "all" else [args.robot]

    for robot_index, robot_name in enumerate(robot_names):
        robot = load_robot_model(robot_name)
        config = load_constraint_config(robot.name)
        scenarios = _meshcat_scenarios(robot, args.scenario)
        for scenario_index, scenario in enumerate(scenarios):
            result = solve(
                robot,
                scenario.target_pose,
                scenario.current_joints,
                scenario_constraints(config, scenario),
                collect_trace=collect_trace,
            )
            offset = _scene_offset(scenario_index, len(scenarios), robot_index)
            root = vis[f"{robot.name}/{scenario_index}_{_slug(scenario.label)}"]
            start_points = _chain_points(robot, scenario.current_joints, offset)
            solved_points = _chain_points(robot, result.joint_position, offset)
            target = np.array(
                [scenario.target_pose.x, scenario.target_pose.y, scenario.target_pose.z],
                dtype=float,
            ) + offset
            achieved = np.array(
                [result.achieved_pose.x, result.achieved_pose.y, result.achieved_pose.z],
                dtype=float,
            ) + offset

            _draw_workspace(root["workspace"], g, robot, offset, args.workspace)
            _draw_chain(root["start"], g, start_points, COLORS["start"], opacity=0.75)
            if collect_trace:
                _draw_trace(root["trace"], g, robot, result.trace, offset)
            _draw_chain(root["solution"], g, solved_points, COLORS["solution"], opacity=0.95)
            _draw_marker(root["target"], g, target, COLORS["target"], radius=0.045)
            _draw_marker(root["achieved"], g, achieved, COLORS["achieved"], radius=0.04)
            _draw_axes(root["tcp_axes"], g, achieved, scale=0.13)
            _draw_status_tag(root["status"], g, robot, result, offset)

            print(
                f"{robot.name} / {scenario.label}: "
                f"{result.status.value}, position error={result.position_error:.5f}, "
                f"orientation error={result.orientation_error:.5f}"
            )
            print(
                "  q: ["
                + ", ".join(f"{joint:+.3f}" for joint in result.joint_position)
                + "]"
            )
            if collect_trace:
                print(f"  trace states: {len(result.trace)}")

    print("Press Ctrl+C to stop the Meshcat server.")
    try:
        while True:
            input()
    except (KeyboardInterrupt, EOFError):
        pass


def _meshcat_scenarios(robot, scene_mode: str) -> list[Scenario]:
    if scene_mode == "custom":
        return [build_custom_scenario(robot)]
    if scene_mode == "custom":
        return [build_custom_scenario(robot)]
    raise ValueError(f"Unknown scene mode: {scene_mode}")


def _scene_offset(scenario_index: int, scenario_count: int, robot_index: int) -> np.ndarray:
    scene_spacing = 2.3
    robot_spacing = 1.8
    centered_index = scenario_index - (scenario_count - 1) / 2.0
    return np.array([centered_index * scene_spacing, -robot_index * robot_spacing, 0.0])


def _trace_enabled(flag_value: bool, trace_setting: str | None) -> bool:
    if trace_setting is None:
        return flag_value
    key, separator, value = trace_setting.partition("=")
    if key != "trace" or separator != "=":
        raise SystemExit("Use trace=true, trace=false, --trace true, or --trace false.")
    return _parse_bool(value)


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _chain_points(robot, joints: Sequence[float], offset: np.ndarray) -> list[np.ndarray]:
    q = np.asarray(joints, dtype=float)
    pin.forwardKinematics(robot.model, robot.data, q)
    pin.updateFramePlacements(robot.model, robot.data)

    points = [offset.copy()]
    for joint_id in range(1, robot.model.njoints):
        point = np.asarray(robot.data.oMi[joint_id].translation, dtype=float) + offset
        if points and np.linalg.norm(point - points[-1]) < 1e-5:
            continue
        points.append(point)
    tcp = np.asarray(robot.data.oMf[robot.frame_id].translation, dtype=float) + offset
    if not points or np.linalg.norm(tcp - points[-1]) >= 1e-5:
        points.append(tcp)
    return points


def _draw_chain(root, geometry, points: list[np.ndarray], color: int, opacity: float) -> None:
    material = geometry.MeshLambertMaterial(color=color, opacity=opacity)
    for index, point in enumerate(points):
        _draw_marker(root[f"joint_{index}"], geometry, point, color, radius=0.035, opacity=opacity)
    for index, (start, end) in enumerate(zip(points, points[1:])):
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            continue
        root[f"link_{index}"].set_object(geometry.Cylinder(length, 0.018), material)
        root[f"link_{index}"].set_transform(_cylinder_transform(start, end))


def _draw_trace(root, geometry, robot, trace, offset: np.ndarray) -> None:
    if not trace:
        return
    for trace_state in trace:
        points = _chain_points(robot, trace_state.joint_position, offset)
        pose_root = root[f"iteration_{trace_state.iteration:03d}"]
        _draw_chain(pose_root, geometry, points, COLORS["trace"], opacity=0.32)


def _draw_status_tag(root, geometry, robot, result: SolveResult, offset: np.ndarray) -> None:
    label_lines = [f"status: {result.status.value}"]
    if result.status.value in {"approximate", "non_convergence"}:
        label_lines.extend(textwrap.wrap(result.message, width=56))

    image_bytes, width_px, height_px = _label_image(label_lines, _status_color(result.status.value))
    material = geometry.MeshBasicMaterial(
        color=0xFFFFFF,
        map=geometry.ImageTexture(geometry.PngImage(image_bytes)),
        transparent=True,
        opacity=0.98,
    )
    width_m = 0.82
    height_m = width_m * height_px / width_px
    root.set_object(geometry.Box([width_m, 0.01, height_m]), material)

    workspace_center = np.asarray(robot.workspace_center, dtype=float) + offset
    tag_position = workspace_center + np.array(
        [0.0, -robot.workspace_radius - 0.16, robot.workspace_radius + 0.24]
    )
    root.set_transform(_translation(tag_position))


def _label_image(lines: list[str], status_color: int) -> tuple[bytes, int, int]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise SystemExit("Meshcat status tags require Pillow: python3 -m pip install Pillow") from exc

    font = _label_font()
    padding_x = 16
    padding_y = 12
    line_gap = 5
    line_boxes = [ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), line, font=font) for line in lines]
    line_widths = [right - left for left, top, right, bottom in line_boxes]
    line_heights = [bottom - top for left, top, right, bottom in line_boxes]
    width = max(260, max(line_widths, default=0) + padding_x * 2)
    height = padding_y * 2 + sum(line_heights) + line_gap * max(0, len(lines) - 1)

    image = Image.new("RGBA", (width, height), (255, 255, 255, 232))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (1, 1, width - 2, height - 2),
        radius=12,
        fill=(255, 255, 255, 232),
        outline=_rgb_tuple(status_color) + (255,),
        width=3,
    )
    draw.rectangle((0, 0, 10, height), fill=_rgb_tuple(status_color) + (255,))

    y = padding_y
    for index, line in enumerate(lines):
        color = _rgb_tuple(status_color) + (255,) if index == 0 else (34, 43, 50, 255)
        draw.text((padding_x + 8, y), line, font=font, fill=color)
        y += line_heights[index] + line_gap

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), width, height


def _draw_label_background(draw, width: int, height: int, status_color: int) -> None:
    bounds = (1, 1, width - 2, height - 2)
    fill = (255, 255, 255, 232)
    outline = _rgb_tuple(status_color) + (255,)
    if hasattr(draw, "rounded_rectangle"):
        draw.rounded_rectangle(bounds, radius=12, fill=fill, outline=outline, width=3)
    else:
        draw.rectangle(bounds, fill=fill, outline=outline, width=3)

        
def _label_font():
    from PIL import ImageFont

    for font_path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(font_path, 20)
        except OSError:
            pass
    return ImageFont.load_default()


def _status_color(status: str) -> int:
    return COLORS.get(f"status_{status}", COLORS["status_non_convergence"])


def _rgb_tuple(color: int) -> tuple[int, int, int]:
    return ((color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF)


def _draw_marker(node, geometry, position: np.ndarray, color: int, radius: float, opacity: float = 1.0) -> None:
    node.set_object(
        geometry.Sphere(radius),
        geometry.MeshLambertMaterial(color=color, opacity=opacity),
    )
    node.set_transform(_translation(position))


def _draw_axes(root, geometry, origin: np.ndarray, scale: float) -> None:
    axes = [
        ("x", np.array([scale, 0.0, 0.0]), 0xD64045),
        ("y", np.array([0.0, scale, 0.0]), 0x2E8B57),
        ("z", np.array([0.0, 0.0, scale]), 0x2E5EAA),
    ]
    for name, direction, color in axes:
        start = origin
        end = origin + direction
        root[name].set_object(
            geometry.Cylinder(float(np.linalg.norm(direction)), 0.008),
            geometry.MeshLambertMaterial(color=color),
        )
        root[name].set_transform(_cylinder_transform(start, end))


def _draw_workspace(root, geometry, robot, offset: np.ndarray, mode: str) -> None:
    if mode == "hidden":
        return

    center = np.asarray(robot.workspace_center, dtype=float) + offset
    if mode == "wire":
        material = geometry.MeshBasicMaterial(
            color=COLORS["workspace"],
            opacity=0.42,
            transparent=True,
            wireframe=True,
            wireframeLinewidth=1.0,
        )
    else:
        material = geometry.MeshLambertMaterial(
            color=COLORS["workspace"],
            opacity=0.018,
            transparent=True,
        )
    root.set_object(
        geometry.Sphere(robot.workspace_radius),
        material,
    )
    root.set_transform(_translation(center))


def _draw_ground(vis, geometry) -> None:
    vis["ground"].set_object(
        geometry.Box([7.2, 4.8, 0.01]),
        geometry.MeshLambertMaterial(color=COLORS["ground"], opacity=0.08, transparent=True),
    )
    vis["ground"].set_transform(_translation(np.array([0.0, -1.0, -0.01])))


def _cylinder_transform(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    midpoint = (start + end) * 0.5
    direction = end - start
    direction = direction / np.linalg.norm(direction)
    transform = np.eye(4)
    transform[:3, :3] = _rotation_from_y(direction)
    transform[:3, 3] = midpoint
    return transform


def _rotation_from_y(direction: np.ndarray) -> np.ndarray:
    y_axis = np.array([0.0, 1.0, 0.0])
    cross = np.cross(y_axis, direction)
    dot = float(np.dot(y_axis, direction))
    if np.linalg.norm(cross) < 1e-9:
        return np.eye(3) if dot > 0.0 else np.diag([1.0, -1.0, 1.0])
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    return np.eye(3) + skew + skew @ skew * (1.0 / (1.0 + dot))


def _translation(position: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = position
    return transform


def _slug(label: str) -> str:
    return "_".join(label.lower().split())


if __name__ == "__main__":
    main()
