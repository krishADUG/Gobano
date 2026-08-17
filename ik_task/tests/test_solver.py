from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from IK_solver import (
    ConstraintConfig,
    OUTSIDE_WORKSPACE_MARGIN,
    Pose3D,
    SolveStatus,
    available_robot_names,
    build_custom_scenario,
    build_scenarios,
    home_joints,
    load_constraint_config,
    load_robot_model,
    scenario_constraints,
    solve,
)


class SolverTests(unittest.TestCase):
    def test_robot_assets_are_discovered_from_robot_folder(self) -> None:
        self.assertEqual(available_robot_names(), ["franka_panda", "ur5"])
        root = Path(__file__).resolve().parent.parent / "robots"
        for robot_name in available_robot_names():
            self.assertTrue((root / robot_name / f"{robot_name}.urdf").exists())
            self.assertTrue((root / robot_name / "dimensions.yaml").exists())
            self.assertTrue((root / robot_name / "solver_config.yaml").exists())

    def test_solver_config_loads_from_robot_folder(self) -> None:
        root = Path(__file__).resolve().parent.parent / "robots"
        for robot_name in available_robot_names():
            config = load_constraint_config(robot_name)
            with (root / robot_name / "solver_config.yaml").open("r", encoding="utf-8") as file:
                data = yaml.safe_load(file)
            self.assertIsInstance(config, ConstraintConfig)
            self.assertEqual(config.max_iterations, data["max_iterations"])
            self.assertEqual(config.qp_solver, data["qp_solver"])
            self.assertAlmostEqual(config.position_tolerance, data["position_tolerance"])
            self.assertAlmostEqual(
                config.approx_position_multiplier,
                data["approx_position_multiplier"],
            )

    def test_solver_config_requires_all_fields(self) -> None:
        config_path = Path("/tmp/incomplete_solver_config.yaml")
        config_path.write_text("max_iterations: 10\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing required solver setting"):
            load_constraint_config(path=config_path)

    def test_urdf_robots_load_and_expose_pinocchio_quantities(self) -> None:
        for robot_name, dof in (("ur5", 6), ("franka_panda", 7)):
            robot = load_robot_model(robot_name)
            q = robot.clamp_to_limits([0.0] * robot.dof)
            pose = robot.forward_kinematics(q)
            jacobian = robot.jacobian(q)
            self.assertEqual(robot.dof, dof)
            self.assertIsInstance(pose, Pose3D)
            self.assertEqual(jacobian.shape, (6, dof))
            self.assertEqual(len(robot.home_joint_position), dof)
            self.assertGreater(robot.workspace_radius, 0.0)

    def test_reachable_fk_generated_targets_succeed(self) -> None:
        for robot_name in available_robot_names():
            robot = load_robot_model(robot_name)
            config = load_constraint_config(robot.name)
            self.assertEqual(len(build_scenarios(robot)), 5)
            scenario = next(
                scenario
                for scenario in build_scenarios(robot)
                if scenario.label == "reachable from home"
            )
            self.assertEqual(scenario.current_joints, home_joints(robot))
            result = solve(
                robot,
                scenario.target_pose,
                scenario.current_joints,
                scenario_constraints(config, scenario),
            )
            self.assertEqual(result.status, SolveStatus.SUCCESS, robot_name)
            self.assertLessEqual(result.position_error, config.position_tolerance)

    def test_solver_trace_is_optional(self) -> None:
        robot = load_robot_model("ur5")
        config = load_constraint_config(robot.name)
        scenario = next(
            scenario
            for scenario in build_scenarios(robot)
            if scenario.label == "reachable from home"
        )
        result_without_trace = solve(
            robot,
            scenario.target_pose,
            scenario.current_joints,
            scenario_constraints(config, scenario),
        )
        result_with_trace = solve(
            robot,
            scenario.target_pose,
            scenario.current_joints,
            scenario_constraints(config, scenario),
            collect_trace=True,
        )
        self.assertEqual(result_without_trace.trace, [])
        self.assertGreaterEqual(len(result_with_trace.trace), 1)
        self.assertEqual(result_with_trace.trace[0].iteration, 0)

    def test_custom_scenario_is_separate_from_standard_scenarios(self) -> None:
        for robot_name in available_robot_names():
            robot = load_robot_model(robot_name)
            config = load_constraint_config(robot.name)
            standard_labels = [scenario.label for scenario in build_scenarios(robot)]
            custom = build_custom_scenario(robot)
            self.assertNotIn(custom.label, standard_labels)
            self.assertIsNotNone(custom.config_overrides)

            result = solve(
                robot,
                custom.target_pose,
                custom.current_joints,
                scenario_constraints(config, custom),
            )
            self.assertEqual(result.status, SolveStatus.SUCCESS, robot_name)
            
    def test_failed_scenarios_report_distinct_reasons(self) -> None:
        expected_messages = {
            "joint-limit blocked target": "Joint limits prevented convergence",
            "jump-limit blocked target": "Jump limits prevented convergence",
            "solver progress limited": "Solver stagnated",
        }
        for robot_name in available_robot_names():
            robot = load_robot_model(robot_name)
            config = load_constraint_config(robot.name)
            scenarios = build_scenarios(robot)
            for label, message in expected_messages.items():
                scenario = next(scenario for scenario in scenarios if scenario.label == label)
                result = solve(
                    robot,
                    scenario.target_pose,
                    scenario.current_joints,
                    scenario_constraints(config, scenario),
                )
                self.assertIn(
                    result.status,
                    {SolveStatus.APPROXIMATE, SolveStatus.NON_CONVERGENCE},
                    (robot_name, label),
                )
                self.assertIn(message, result.message)
                if result.status == SolveStatus.APPROXIMATE:
                    self.assertLessEqual(
                        result.position_error,
                        config.approx_position_multiplier * config.position_tolerance,
                    )
                    self.assertLessEqual(
                        result.orientation_error,
                        config.approx_orientation_multiplier * config.orientation_tolerance,
                    )
                else:
                    self.assertTrue(
                        result.position_error
                        > config.approx_position_multiplier * config.position_tolerance
                        or result.orientation_error
                        > config.approx_orientation_multiplier * config.orientation_tolerance
                    )
                if label == "solver progress limited":
                    self.assertEqual(result.iterations, 50)
                if label == "joint-limit blocked target":
                    self.assertNotIn("J1 ", result.message)

    def test_outside_workspace_target_reports_non_convergence(self) -> None:
        for robot_name in available_robot_names():
            robot = load_robot_model(robot_name)
            config = load_constraint_config(robot.name)
            scenario = next(
                scenario
                for scenario in build_scenarios(robot)
                if scenario.label == "outside workspace"
            )
            center = robot.workspace_center
            distance = (
                (scenario.target_pose.x - center[0]) ** 2
                + (scenario.target_pose.y - center[1]) ** 2
                + (scenario.target_pose.z - center[2]) ** 2
            ) ** 0.5
            self.assertAlmostEqual(
                distance,
                robot.workspace_radius + OUTSIDE_WORKSPACE_MARGIN,
                places=8,
            )
            result = solve(
                robot,
                scenario.target_pose,
                scenario.current_joints,
                scenario_constraints(config, scenario),
            )
            self.assertEqual(result.status, SolveStatus.NON_CONVERGENCE)
            self.assertEqual(result.iterations, 0)
            self.assertIn("Point outside the workspace", result.message)
            for joint, limit in zip(result.joint_position, robot.joint_limits):
                self.assertGreaterEqual(joint, limit.lower)
                self.assertLessEqual(joint, limit.upper)

    def test_invalid_input_reports_status(self) -> None:
        robot = load_robot_model("ur5")
        config = load_constraint_config(robot.name)
        result = solve(robot, Pose3D(0.4, 0.2, 0.3), [0.0, 0.0], config)
        self.assertEqual(result.status, SolveStatus.INVALID_INPUT)


if __name__ == "__main__":
    unittest.main()
