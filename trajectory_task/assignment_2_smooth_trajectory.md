# Smooth Trajectory Generation for Physical-AI Commands

*Take-home engineering assignment*

> **Suggested timebox:** 3–4 hours  
> **Environment:** Simulation only; no physical robot is required  
> **AI assistance:** Allowed and encouraged. Please briefly describe how you used it.

## Context

A policy or higher-level control system may continuously provide either a single target position or a chunk of future target positions. These commands can arrive in succession, and consecutive chunks may overlap in time or repeat some of the same points. We want a low-level component that turns this evolving command stream into a smooth, executable trajectory.

The component should avoid abrupt motion and account for limits or regularization on velocity and acceleration. It should also behave predictably when new commands arrive while the system is already moving.

## Your task

Design and implement a proof-of-concept smooth trajectory generator. The component should expose an interface broadly similar to the following; the exact API is up to you:

```text
update(command)
sample(time) -> position, velocity, acceleration
```

A command may contain either one target or a sequence of timestamped or fixed-rate targets. The component should accept an ongoing succession of either type, including mixed sequences of individual points and chunks. A newly received chunk may partially overlap the currently active chunk.

## Core requirements

Your prototype should:

- Accept a continuing stream of commands, where each command may be a single target position or a multi-point trajectory chunk.
- Support arbitrary successions of individual points and chunks rather than treating each input as an isolated trajectory.
- Handle consecutive chunks that partially overlap in time or contain repeated points.
- Define and implement a clear policy for reconciling an overlapping chunk with the active trajectory, such as replacing, merging, or re-planning the overlapping portion.
- Generate continuous, time-sampled position commands.
- Avoid discontinuities when transitioning from the current motion to a new command.
- Apply explicit velocity and acceleration limits, or use a clearly explained regularization or smoothing objective.
- Report when an input cannot be followed exactly under the configured constraints.

The implementation may operate in one dimension, Cartesian space, or joint space. It should be structured so that the same logic can be applied to multiple dimensions.

> **Scope note:** Real-time guarantees, motor control, ROS integration, and hardware testing are not required. A compact numerical prototype with plots, logs, or automated checks is sufficient.

## Required demonstrations

Demonstrate at least:

1. Motion to a single target.
2. Execution of a multi-point path, such as a square.
3. A sharp or noisy input that requires smoothing.
4. Replacement of the active command while the system is already moving.
5. Two consecutive trajectory chunks with a partially overlapping time range, showing how the overlap is reconciled without introducing a motion discontinuity.

Plots, logged samples, or numerical checks are all acceptable. The demonstration should make continuity and constraint behavior easy to assess.

## Design discussion

In a short README or design note, explain:

- Your trajectory representation and interpolation method.
- How you preserve continuity between commands.
- Whether constraints are strict or treated as regularization.
- How you trade tracking accuracy against smoothness and delay.
- How the design would handle commands arriving at an irregular rate.
- What should happen when a new trajectory chunk overlaps or conflicts with the active trajectory, including which points are retained, replaced, or blended.
- How you would extend the system for jerk limits, closed-loop feedback, model-predictive control, or physical robot execution.
- Important safety considerations when accepting commands from an AI policy.

## Deliverables

- Runnable source code.
- A short README with setup and usage instructions.
- A few tests or demonstrations covering the scenarios above.
- A concise discussion of design choices, limitations, and next steps.
- A brief note describing any AI tools used and how you validated their output.

Production-grade code and a polished user interface are not expected. Prioritize a clear, working core and explicit engineering decisions.

## Evaluation criteria

| Area | What we look for |
| --- | --- |
| Working proof of concept | Correct trajectory generation and convincing demonstrations |
| Motion quality | Continuity, smoothness, and appropriate handling of limits |
| Online behavior | Predictable command replacement and treatment of infeasible inputs |
| System design | Clear API, reusable structure, and sound engineering tradeoffs |
| Clarity | Readable code, concise documentation, and reproducible execution |

Candidates are not penalized for consciously omitting features outside the timebox, provided the limitations are identified and the proposed next steps are well reasoned.
