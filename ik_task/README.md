# URDF-Based Constrained Inverse Kinematics

This project solves constrained inverse kinematics for robots defined in the local `robots/` folder. Robot kinematics come from Pinocchio: URDF loading, forward kinematics, SE(3) pose error, configuration integration, and 6D frame Jacobians all come from the Pinocchio model.

## Setup

Requires Python 3.11+.

```bash
pip install -r requirements.txt
python3 demo.py
python3 demo.py trace=true # to get traces of all the interations

python3 visualize.py 
python3 visualize_meshcat.py --robot franka_panda --scenario all --trace true --open 
python3 visualize_meshcat.py --robot ur5 --scenario all --trace true --open
python3 -m unittest
```

`visualize.py` writes `ik_visualization.html`, which can be opened directly in a browser. `visualize_meshcat.py` starts an interactive Meshcat 3D viewer. Use `--trace true` or `trace=true` in `demo.py` to collect iteration traces. In Meshcat, `--trace true` draws translucent intermediate robot poses for every collected IK iteration.

## Robot Folder Contract

Each robot has:

```text
robots/<robot_name>/<robot_name>.urdf
robots/<robot_name>/dimensions.yaml
robots/<robot_name>/solver_config.yaml
```

The repository currently includes:

- `robots/ur5/ur5.urdf`
- `robots/franka_panda/franka_panda.urdf`

Solver settings are robot-specific and live beside each robot model:

```text
robots/<robot_name>/solver_config.yaml
```

That file controls tolerances, approximation multipliers, iteration limits, max joint step, damping, QP backend, and regularization weights. 
Every solver key is mandatory; missing or unknown keys raise an error at load time.

## API Shape

```python
from IK_solver import Pose3D, load_constraint_config, load_robot_model, solve

robot = load_robot_model("franka_panda")
constraints = load_constraint_config(robot.name)
q0 = robot.clamp_to_limits(home_pose)
target = Pose3D(x=0.5, y=0.1, z=0.7, roll=3.14, pitch=0.0, yaw=0.0)

result = solve(
    robot_model=robot,
    target_pose=target,
    current_joint_position=q0,
    constraints=constraints,
)
```

The result includes:

- `status`: `success`, `approximate`, `invalid_input`, or `non_convergence`.
- `joint_position`: the returned bounded joint configuration.
- `achieved_pose`: Pinocchio forward kinematics at the returned configuration.
- `position_error` and `orientation_error`.
- `message`: conditions unders which the solution was founds or not found .

Status is based on final Cartesian error:

- `success`: position and orientation errors are within `position_tolerance` and `orientation_tolerance`.
- `approximate`: errors are outside success tolerance, but still within `approx_*_multiplier * *_tolerance`.
- `non_convergence`: errors are outside the approximation band, or the target is outside the workspace.

The message explains the reason, such as joint limits, jump limits, solver stagnation, max-iteration exhaustion, or outside workspace.

Targets outside the configured `workspace` sphere are rejected before the QP loop and return `non_convergence` with a `Point outside the workspace` message.

## Solver Approach

Each IK iteration solves a box-constrained differential QP: as shown in the OCP.pdf

`max_solution_jump` is enforced as a total joint-space bound from the scenario start. If the Cartesian target is reachable only by exceeding that jump limit, the solver reports the bounded result and explains that jump limits prevented convergence. The final error decides whether that bounded result is `approximate` or `non_convergence`.

`qpsolvers` dispatches to OSQP by default. If the QP backend is unavailable, the code falls back to a clipped regularized least-squares step so callers still get a bounded result and clear status.

## Visualization

`demo.py` runs the same scenario set for every discovered robot:

- `reachable from home`: starts at the configured home pose and reaches a nearby FK target from offset joints.
- `joint-limit blocked target`: constrained by joint limits; final error decides approximate vs non-convergence.
- `jump-limit blocked target`: constrained by configured jump limits; final error decides approximate vs non-convergence.
- `solver progress limited`: constrained by solver progress; final error decides approximate vs non-convergence.
- `outside workspace`: non-convergence, because the point is outside the configured workspace.

`visualize.py` renders one XY chart per joint: x is the scenario category, y is joint value in radians, and the joint limits are shown as horizontal bounds. It also renders position-error and orientation-error charts against each scenario's active success tolerance and approximation threshold. `visualize_meshcat.py` renders an interactive 3D primitive skeleton from Pinocchio joint placements plus an optional workspace shell.



## Custom scene

in the scenario file, you can edit your own custome scenario in the lines 70-85, to define a custome start and goal point, with custom solver configs.

These can be visualized in meshcat using teh command 
python3 visualize_meshcat.py --scenario custom --trace true --open