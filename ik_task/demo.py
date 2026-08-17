from __future__ import annotations

import argparse

from IK_solver import (
    available_robot_names,
    build_scenarios,
    load_constraint_config,
    load_robot_model,
    scenario_constraints,
    solve,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IK solver scenarios for every robot.")
    parser.add_argument(
        "--trace",
        type=_parse_bool,
        default=False,
        help="Collect and report iteration traces: true or false.",
    )
    parser.add_argument(
        "trace_setting",
        nargs="?",
        help="Optional shorthand: trace=true or trace=false.",
    )
    args = parser.parse_args()
    collect_trace = _trace_enabled(args.trace, args.trace_setting)

    for robot_name in available_robot_names():
        robot = load_robot_model(robot_name)
        config = load_constraint_config(robot.name)
        print(f"\n{robot.name} ({robot.dof} DoF, frame={robot.end_effector_frame})")

        print(
            "  home q: ["
            + ", ".join(f"{joint:+.3f}" for joint in robot.home_joint_position)
            + "]"
        )
        print(f"  workspace radius: {robot.workspace_radius:.2f} m")

        for scenario in build_scenarios(robot):
            scenario_config = scenario_constraints(config, scenario)
            result = solve(
                robot,
                scenario.target_pose,
                scenario.current_joints,
                scenario_config,
                collect_trace=collect_trace,
            )
            joints = ", ".join(f"{joint:+.3f}" for joint in result.joint_position)
            print(f"\n  {scenario.label}")
            print(f"    expected: {scenario.expected}")
            print(f"    status: {result.status.value}")
            print(f"    joints: [{joints}]")
            print(
                "    target pose:   "
                f"x={scenario.target_pose.x:+.3f}, "
                f"y={scenario.target_pose.y:+.3f}, "
                f"z={scenario.target_pose.z:+.3f}, "
                f"rpy=[{scenario.target_pose.roll:+.3f}, "
                f"{scenario.target_pose.pitch:+.3f}, "
                f"{scenario.target_pose.yaw:+.3f}]"
            )
            print(
                "    achieved pose: "
                f"x={result.achieved_pose.x:+.3f}, "
                f"y={result.achieved_pose.y:+.3f}, "
                f"z={result.achieved_pose.z:+.3f}, "
                f"rpy=[{result.achieved_pose.roll:+.3f}, "
                f"{result.achieved_pose.pitch:+.3f}, "
                f"{result.achieved_pose.yaw:+.3f}]"
            )
            print(
                "    errors: "
                f"position={result.position_error:.5f}, "
                f"orientation={result.orientation_error:.5f}, "
                f"iterations={result.iterations}"
            )
            if collect_trace:
                print(f"    trace states: {len(result.trace)}")
            print(f"    note: {result.message}")


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


if __name__ == "__main__":
    main()
