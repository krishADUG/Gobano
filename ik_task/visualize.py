from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

from IK_solver import (
    ConstraintConfig,
    Scenario,
    available_robot_names,
    build_scenarios,
    load_constraint_config,
    load_robot_model,
    scenario_constraints,
    solve,
)


@dataclass(frozen=True)
class JointState:
    key: str
    label: str
    axis_label: str
    joints: list[float]
    status: str
    iterations: int
    position_error: float
    orientation_error: float
    position_tolerance: float
    orientation_tolerance: float
    approximate_position_tolerance: float
    approximate_orientation_tolerance: float


@dataclass(frozen=True)
class RobotPlot:
    robot_name: str
    joint_names: list[str]
    joint_limits: list[tuple[float, float]]
    states: list[JointState]


def main() -> None:
    output_path = Path("ik_visualization.html")
    plots = []
    for name in available_robot_names():
        robot = load_robot_model(name)
        config = load_constraint_config(robot.name)
        plots.append(_build_robot_plot(robot, config))
    output_path.write_text(_page("\n".join(_render_robot(plot) for plot in plots)), encoding="utf-8")
    print(f"Wrote {output_path.resolve()}")


def _build_robot_plot(robot, config: ConstraintConfig) -> RobotPlot:
    states = []
    for scenario in build_scenarios(robot):
        scenario_config = scenario_constraints(config, scenario)
        result = solve(
            robot,
            scenario.target_pose,
            scenario.current_joints,
            scenario_config,
        )
        states.append(
            JointState(
                _scenario_key(scenario),
                scenario.label,
                _scenario_axis_label(scenario),
                result.joint_position,
                result.status.value,
                result.iterations,
                result.position_error,
                result.orientation_error,
                scenario_config.position_tolerance,
                scenario_config.orientation_tolerance,
                scenario_config.approx_position_multiplier
                * scenario_config.position_tolerance,
                scenario_config.approx_orientation_multiplier
                * scenario_config.orientation_tolerance,
            )
        )
    return RobotPlot(
        robot_name=robot.name,
        joint_names=[str(robot.model.names[index]) for index in range(1, robot.model.njoints)],
        joint_limits=[(limit.lower, limit.upper) for limit in robot.joint_limits],
        states=states,
    )


def _scenario_key(scenario: Scenario) -> str:
    keys = {
        "reachable from home": "reachable",
        "joint-limit blocked target": "joint_limit",
        "jump-limit blocked target": "jump_limit",
        "solver progress limited": "solver_stuck",
        "outside workspace": "outside_workspace",
    }
    return keys[scenario.label]


def _scenario_axis_label(scenario: Scenario) -> str:
    labels = {
        "reachable from home": "reach",
        "joint-limit blocked target": "limit",
        "jump-limit blocked target": "jump",
        "solver progress limited": "stuck",
        "outside workspace": "outside",
    }
    return labels[scenario.label]


def _render_robot(plot: RobotPlot) -> str:
    rows = "\n".join(
        _render_joint_row(index, name, limits, plot.states)
        for index, (name, limits) in enumerate(zip(plot.joint_names, plot.joint_limits), start=1)
    )
    return f"""
    <section class="panel">
      <div class="panel-header">
        <h2>{escape(plot.robot_name)}</h2>
        <p class="meta">{len(plot.joint_names)} joints · values in radians</p>
      </div>
      {_render_legend(plot.states)}
      <div class="joint-plots">
        {rows}
      </div>
      {_render_error_section(plot.states)}
    </section>
    """


def _render_legend(states: list[JointState]) -> str:
    items = "\n".join(
        f'<span><i class="marker legend-marker {escape(state.key)}"></i>{escape(state.label)} · {state.iterations} iter</span>'
        for state in states
    )
    return f'<div class="legend">{items}</div>'


def _render_error_section(states: list[JointState]) -> str:
    return f"""
      <div class="subsection-title">Error vs tolerance</div>
      <div class="joint-plots">
        {_render_error_row("Position error", "m", states, "position_error", "position_tolerance")}
        {_render_error_row("Orientation error", "rad", states, "orientation_error", "orientation_tolerance")}
      </div>
    """


def _render_error_row(
    label: str,
    unit: str,
    states: list[JointState],
    error_attr: str,
    tolerance_attr: str,
) -> str:
    width = 760
    height = 170
    left = 58
    right = width - 30
    top = 18
    bottom = height - 42
    x_positions = _scenario_x_positions(left, right, len(states))
    error_values = [float(getattr(state, error_attr)) for state in states]
    tolerance_values = [float(getattr(state, tolerance_attr)) for state in states]
    approximate_attr = (
        "approximate_position_tolerance"
        if error_attr == "position_error"
        else "approximate_orientation_tolerance"
    )
    approximate_values = [float(getattr(state, approximate_attr)) for state in states]
    y_max = max([*error_values, *tolerance_values, *approximate_values, 1e-12]) * 1.15

    def sy(value: float) -> float:
        return bottom - value / y_max * (bottom - top)

    error_points = [
        (x_positions[state_index], sy(error), state, error)
        for state_index, (state, error) in enumerate(zip(states, error_values))
    ]
    tolerance_points = [
        (x_positions[state_index], sy(tolerance), state, tolerance)
        for state_index, (state, tolerance) in enumerate(zip(states, tolerance_values))
    ]
    approximate_points = [
        (x_positions[state_index], sy(tolerance), state, tolerance)
        for state_index, (state, tolerance) in enumerate(zip(states, approximate_values))
    ]
    error_polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y, _state, _value in error_points)
    tolerance_polyline = " ".join(
        f"{x:.2f},{y:.2f}" for x, y, _state, _value in tolerance_points
    )
    approximate_polyline = " ".join(
        f"{x:.2f},{y:.2f}" for x, y, _state, _value in approximate_points
    )
    markers = "\n".join(
        _error_marker(
            state,
            x,
            y,
            value,
            tolerance_values[index],
            approximate_values[index],
            unit,
        )
        for index, (x, y, state, value) in enumerate(error_points)
    )
    tolerance_markers = "\n".join(
        f'<circle class="tolerance-dot" cx="{x:.2f}" cy="{y:.2f}" r="3"><title>{escape(state.label)} success tolerance: {value:.5g} {escape(unit)}</title></circle>'
        for x, y, state, value in tolerance_points
    )
    approximate_markers = "\n".join(
        f'<circle class="approximate-dot" cx="{x:.2f}" cy="{y:.2f}" r="3"><title>{escape(state.label)} approximate threshold: {value:.5g} {escape(unit)}</title></circle>'
        for x, y, state, value in approximate_points
    )
    x_labels = "\n".join(
        f'<text class="scenario-label" x="{x_positions[state_index]:.2f}" y="{height - 14}" text-anchor="middle">{escape(state.axis_label)}</text>'
        for state_index, state in enumerate(states)
    )
    return f"""
      <div class="joint-row">
        <div class="joint-label">
          <strong>{escape(label)}</strong>
          <span>computed error / allowed tolerance ({escape(unit)})</span>
        </div>
        <svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(label)} tolerance plot">
          <rect class="valid-band" x="{left}" y="{top}" width="{right - left}" height="{bottom - top}" />
          <line class="axis-line" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" />
          <text class="limit-text" x="8" y="{top + 4}" text-anchor="start">{y_max:.2g}</text>
          <text class="limit-text" x="8" y="{bottom + 4}" text-anchor="start">0</text>
          <polyline class="approximate-line" points="{approximate_polyline}" />
          <polyline class="tolerance-line" points="{tolerance_polyline}" />
          <polyline class="scenario-line" points="{error_polyline}" />
          {approximate_markers}
          {tolerance_markers}
          {markers}
          {x_labels}
        </svg>
      </div>
    """


def _render_joint_row(
    index: int,
    joint_name: str,
    limits: tuple[float, float],
    states: list[JointState],
) -> str:
    lower, upper = limits
    width = 760
    height = 170
    left = 58
    right = width - 30
    top = 18
    bottom = height - 42
    x_positions = _scenario_x_positions(left, right, len(states))

    def sy(value: float) -> float:
        if upper <= lower:
            return (top + bottom) * 0.5
        return bottom - (value - lower) / (upper - lower) * (bottom - top)

    points = [
        (x_positions[state_index], sy(state.joints[index - 1]), state)
        for state_index, state in enumerate(states)
    ]
    polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y, _state in points)
    markers = "\n".join(
        _state_marker(state, x, y, state.joints[index - 1])
        for x, y, state in points
    )
    x_labels = "\n".join(
        f'<text class="scenario-label" x="{x_positions[state_index]:.2f}" y="{height - 14}" text-anchor="middle">{escape(state.axis_label)}</text>'
        for state_index, state in enumerate(states)
    )
    return f"""
      <div class="joint-row">
        <div class="joint-label">
          <strong>J{index}</strong>
          <span>{escape(joint_name)}</span>
        </div>
        <svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(joint_name)} joint limit plot">
          <rect class="valid-band" x="{left}" y="{top}" width="{right - left}" height="{bottom - top}" />
          <line class="limit-line upper" x1="{left}" y1="{top}" x2="{right}" y2="{top}" />
          <line class="limit-line lower" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" />
          <line class="axis-line" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" />
          <text class="limit-text" x="8" y="{top + 4}" text-anchor="start">{upper:+.2f}</text>
          <text class="limit-text" x="8" y="{bottom + 4}" text-anchor="start">{lower:+.2f}</text>
          <polyline class="scenario-line" points="{polyline}" />
          {markers}
          {x_labels}
        </svg>
      </div>
    """


def _scenario_x_positions(left: float, right: float, count: int) -> list[float]:
    if count == 1:
        return [(left + right) * 0.5]
    step = (right - left) / (count - 1)
    return [left + step * index for index in range(count)]


def _state_marker(state: JointState, x: float, y: float, value: float) -> str:
    return f"""
      <g class="state {escape(state.key)}">
        <circle class="state-dot" cx="{x:.2f}" cy="{y:.2f}" r="6">
          <title>{escape(state.label)}: {value:+.4f} rad ({escape(state.status)}, {state.iterations} iterations)</title>
        </circle>
      </g>
    """


def _error_marker(
    state: JointState,
    x: float,
    y: float,
    value: float,
    tolerance: float,
    approximate_tolerance: float,
    unit: str,
) -> str:
    return f"""
      <g class="state {escape(state.key)}">
        <circle class="state-dot" cx="{x:.2f}" cy="{y:.2f}" r="6">
          <title>{escape(state.label)}: error {value:.5g} {escape(unit)}, success tolerance {tolerance:.5g} {escape(unit)}, approximate threshold {approximate_tolerance:.5g} {escape(unit)} ({escape(state.status)}, {state.iterations} iterations)</title>
        </circle>
      </g>
    """


def _page(panels: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Joint Limit IK Visualization</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f5f7f8;
      color: #172026;
    }}
    body {{
      margin: 0;
      padding: 28px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 26px;
      font-weight: 760;
      letter-spacing: 0;
    }}
    .intro {{
      margin: 0 0 22px;
      max-width: 940px;
      color: #596773;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 18px;
      max-width: 1180px;
    }}
    .panel {{
      background: #ffffff;
      border: 1px solid #dbe1e7;
      border-radius: 8px;
      padding: 16px;
      box-shadow: 0 1px 2px rgba(16, 24, 32, 0.06);
    }}
    .panel-header {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 14px;
      margin-bottom: 12px;
    }}
    h2 {{
      margin: 0;
      font-size: 18px;
      letter-spacing: 0;
    }}
    .meta {{
      margin: 0;
      color: #5d6874;
      font-size: 13px;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 18px;
      margin: 0 0 14px;
      color: #44515d;
      font-size: 13px;
    }}
    .marker {{
      display: inline-block;
      width: 11px;
      height: 11px;
      margin-right: 6px;
      border-radius: 50%;
      vertical-align: -1px;
    }}
    .joint-plots {{
      display: grid;
      gap: 10px;
    }}
    .subsection-title {{
      margin: 18px 0 10px;
      color: #33414d;
      font-size: 14px;
      font-weight: 720;
    }}
    .joint-row {{
      display: grid;
      grid-template-columns: minmax(170px, 230px) minmax(360px, 1fr);
      align-items: stretch;
      gap: 12px;
      padding: 8px 10px;
      border: 1px solid #e7ebef;
      border-radius: 6px;
      background: #fbfcfd;
    }}
    .joint-label {{
      display: grid;
      gap: 2px;
      min-width: 0;
    }}
    .joint-label strong {{
      color: #33414d;
      font-size: 13px;
    }}
    .joint-label span {{
      color: #65727e;
      font-size: 12px;
      overflow-wrap: anywhere;
    }}
    svg {{
      display: block;
      width: 100%;
      height: auto;
      overflow: visible;
    }}
    .valid-band {{
      fill: #edf3f6;
      stroke: #dce5eb;
      stroke-width: 1;
    }}
    .limit-line {{
      stroke: #81909c;
      stroke-width: 1.5;
      stroke-dasharray: 5 5;
    }}
    .axis-line {{
      stroke: #c5ced6;
      stroke-width: 1;
    }}
    .scenario-line {{
      fill: none;
      stroke: #8d98a3;
      stroke-width: 2;
      stroke-opacity: 0.58;
    }}
    .tolerance-line {{
      fill: none;
      stroke: #47525d;
      stroke-width: 1.5;
      stroke-dasharray: 6 5;
      stroke-opacity: 0.72;
    }}
    .approximate-line {{
      fill: none;
      stroke: #d08a00;
      stroke-width: 1.5;
      stroke-dasharray: 8 4;
      stroke-opacity: 0.72;
    }}
    .tolerance-dot {{
      fill: #ffffff;
      stroke: #47525d;
      stroke-width: 1.4;
    }}
    .approximate-dot {{
      fill: #ffffff;
      stroke: #d08a00;
      stroke-width: 1.4;
    }}
    .limit-text {{
      fill: #687581;
      font-size: 11px;
    }}
    .scenario-label {{
      fill: #596673;
      font-size: 10px;
    }}
    .state-dot {{
      stroke: #ffffff;
      stroke-width: 2;
    }}
    .reachable, .reachable .state-stem {{
      color: #126a7a;
      stroke: #126a7a;
      fill: #126a7a;
      background: #126a7a;
    }}
    .joint_limit, .joint_limit .state-stem {{
      color: #d08a00;
      stroke: #d08a00;
      fill: #d08a00;
      background: #d08a00;
    }}
    .jump_limit, .jump_limit .state-stem {{
      color: #9a5bce;
      stroke: #9a5bce;
      fill: #9a5bce;
      background: #9a5bce;
    }}
    .solver_stuck, .solver_stuck .state-stem {{
      color: #6d7781;
      stroke: #6d7781;
      fill: #6d7781;
      background: #6d7781;
    }}
    .outside_workspace, .outside_workspace .state-stem {{
      color: #c83f49;
      stroke: #c83f49;
      fill: #c83f49;
      background: #c83f49;
    }}
    @media (max-width: 760px) {{
      body {{
        padding: 18px;
      }}
      .joint-row {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <h1> IK Visualization</h1>
  <p class="intro">Each row shows one robot joint. The gray bar is the joint limit range, and colored markers show solved joint values.</p>
  <main class="grid">
    {panels}
  </main>
</body>
</html>
"""


if __name__ == "__main__":
    main()
