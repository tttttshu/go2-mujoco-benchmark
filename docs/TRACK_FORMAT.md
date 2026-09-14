# Track configuration

TrackSpec v2 describes a deterministic course as an ordered YAML list. It works
identically on Windows and Ubuntu; no machine-specific paths or GPU settings are
part of the format.

## Difficulty levels

Set `difficulty_level` to an integer from 1 through 9. A segment may override
the course level:

```yaml
schema_version: 2
name: custom_course
seed: 42
width: 3.0
difficulty_level: 5
difficulty_profile: train
start_flat_length: 3.0
finish_flat_length: 3.0
segments:
  - type: flat
    length: 2.0
  - type: maze
    length: 7.0
    difficulty_level: 8
  - type: slope_up
    length: 2.0
    difficulty_level: 3
  - type: slope_down
    length: 2.0
  - type: stairs_up
    steps: 3
  - type: stairs_down
    steps: 3
```

The order in `segments` is the physical order of the course. Any supported type
may be selected, omitted, repeated, or rearranged. The compiler automatically
joins each segment at the previous segment's end position and height.

The command line can temporarily override the global level without editing YAML:

```bash
python -m go2_mujoco_benchmark.viewer \
  --track configs/tracks/mixed_maze.yaml --difficulty-level 7
```

TrackSpec v1 files with `global_difficulty: 0.0..1.0` remain readable and are
mapped to the nearest v2 level. New files should use schema v2.

## Maze terrain

The maze is a chainable, collision-enabled obstacle field. Its short internal
walls are free-standing and distributed across the road center and both sides;
they do not extend from or depend on the course boundary walls. Course boundary
walls remain independently controlled by `boundary_walls.enabled`. The terminal
barrier always has exactly one opening centered at `y = 0`, with a fixed width
of 1.2 m; that exit width cannot be overridden.

Without explicit values, levels 1–9 control all three maze dimensions:

| Parameter | Level 1 | Level 9 | Effect |
| --- | ---: | ---: | --- |
| `wall_thickness` | 0.04 m | 0.08 m | thicker is harder |
| `wall_length` | 0.30 m | 0.55 m | longer is harder, but remains free-standing |
| `wall_count` | 4 | 10 | more walls is harder |

Each dimension may instead be fixed explicitly for controlled experiments:

```yaml
- type: maze
  length: 5.0
  wall_thickness: 0.06
  wall_length: 0.45
  wall_count: 8
```

`wall_length` must be smaller than the track `width`, and the segment must be
long enough to place the requested number and thickness of walls. For backward
compatibility, `corridor_width` may be supplied instead; it derives a centered
wall length as `track width - 2 × corridor width`. The compiler rejects
impossible geometry rather than emitting overlapping walls.

## Supported segment fields

- `flat`: required `length`.
- `slope_up`, `slope_down`: required `length`; optional non-negative `slope`.
- `stairs_up`, `stairs_down`: required positive integer `steps`; optional
  `step_height` and `step_depth`.
- `maze`: required `length`; optional `wall_thickness`, `wall_length`,
  `corridor_width`, and positive integer `wall_count`.

All segment types accept an optional `difficulty_level`. Explicit geometric
values take precedence over difficulty-derived defaults.
