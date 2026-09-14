"""Deterministic primitive terrain generators with a common chaining contract."""

from __future__ import annotations

import math

from .geometry import BoxGeom, TerrainPatch
from .track import SegmentSpec, TerrainLimits, difficulty_to_normalized

FOUNDATION_DEPTH = 0.2
MAZE_EXIT_WIDTH = 1.2
MAZE_WALL_HEIGHT = 0.55


def _difficulty_level(segment: SegmentSpec, global_difficulty_level: int) -> int:
    return global_difficulty_level if segment.difficulty_level is None else segment.difficulty_level


def _difficulty(segment: SegmentSpec, global_difficulty_level: int) -> tuple[int, float]:
    level = _difficulty_level(segment, global_difficulty_level)
    return level, difficulty_to_normalized(level)


def build_flat(
    segment: SegmentSpec,
    index: int,
    start_x: float,
    start_z: float,
    width: float,
    global_difficulty_level: int,
) -> TerrainPatch:
    assert segment.length is not None
    difficulty_level, difficulty = _difficulty(segment, global_difficulty_level)
    geom = BoxGeom(
        name=f"terrain_{index:03d}_flat_000",
        center=(start_x + segment.length / 2.0, 0.0, start_z - FOUNDATION_DEPTH / 2.0),
        half_size=(segment.length / 2.0, width / 2.0, FOUNDATION_DEPTH / 2.0),
    )
    return TerrainPatch(
        index=index,
        kind=segment.kind,
        start_x=start_x,
        end_x=start_x + segment.length,
        start_z=start_z,
        end_z=start_z,
        difficulty=difficulty,
        difficulty_level=difficulty_level,
        out_of_distribution=False,
        parameters={"length": segment.length},
        geoms=(geom,),
    )


def build_slope(
    segment: SegmentSpec,
    index: int,
    start_x: float,
    start_z: float,
    width: float,
    global_difficulty_level: int,
    profile: str,
    limits: TerrainLimits,
) -> TerrainPatch:
    assert segment.length is not None
    difficulty_level, difficulty = _difficulty(segment, global_difficulty_level)
    grade = limits.resolve_slope(difficulty, profile) if segment.slope is None else segment.slope
    direction = 1.0 if segment.kind == "slope_up" else -1.0
    rise = direction * grade * segment.length
    end_z = start_z + rise
    pitch = -math.atan(direction * grade)
    surface_length = math.hypot(segment.length, rise)
    top_mid_x = start_x + segment.length / 2.0
    top_mid_z = (start_z + end_z) / 2.0
    center_x = top_mid_x - math.sin(pitch) * FOUNDATION_DEPTH / 2.0
    center_z = top_mid_z - math.cos(pitch) * FOUNDATION_DEPTH / 2.0
    geom = BoxGeom(
        name=f"terrain_{index:03d}_{segment.kind}_000",
        center=(center_x, 0.0, center_z),
        half_size=(surface_length / 2.0, width / 2.0, FOUNDATION_DEPTH / 2.0),
        quaternion_wxyz=(math.cos(pitch / 2.0), 0.0, math.sin(pitch / 2.0), 0.0),
    )
    train_low, train_high = limits.slope_train_range
    return TerrainPatch(
        index=index,
        kind=segment.kind,
        start_x=start_x,
        end_x=start_x + segment.length,
        start_z=start_z,
        end_z=end_z,
        difficulty=difficulty,
        difficulty_level=difficulty_level,
        out_of_distribution=not train_low <= grade <= train_high,
        parameters={"length": segment.length, "slope": grade, "angle_rad": abs(pitch)},
        geoms=(geom,),
    )


def build_stairs(
    segment: SegmentSpec,
    index: int,
    start_x: float,
    start_z: float,
    width: float,
    global_difficulty_level: int,
    profile: str,
    limits: TerrainLimits,
) -> TerrainPatch:
    assert segment.steps is not None
    difficulty_level, difficulty = _difficulty(segment, global_difficulty_level)
    step_height = (
        limits.resolve_step_height(difficulty, profile) if segment.step_height is None else segment.step_height
    )
    direction = 1.0 if segment.kind == "stairs_up" else -1.0
    end_z = start_z + direction * segment.steps * step_height
    foundation_z = min(start_z, end_z) - FOUNDATION_DEPTH
    geoms = []
    for step_index in range(segment.steps):
        top_z = start_z + direction * (step_index + 1) * step_height
        height = top_z - foundation_z
        geoms.append(
            BoxGeom(
                name=f"terrain_{index:03d}_{segment.kind}_{step_index:03d}",
                center=(
                    start_x + (step_index + 0.5) * segment.step_depth,
                    0.0,
                    foundation_z + height / 2.0,
                ),
                half_size=(segment.step_depth / 2.0, width / 2.0, height / 2.0),
            )
        )
    train_low, train_high = limits.step_height_train_range
    return TerrainPatch(
        index=index,
        kind=segment.kind,
        start_x=start_x,
        end_x=start_x + segment.steps * segment.step_depth,
        start_z=start_z,
        end_z=end_z,
        difficulty=difficulty,
        difficulty_level=difficulty_level,
        out_of_distribution=not train_low <= step_height <= train_high,
        parameters={"steps": segment.steps, "step_height": step_height, "step_depth": segment.step_depth},
        geoms=tuple(geoms),
    )


def build_maze(
    segment: SegmentSpec,
    index: int,
    start_x: float,
    start_z: float,
    width: float,
    global_difficulty_level: int,
    seed: int,
) -> TerrainPatch:
    """Build a deterministic slalom maze that remains chainable with other terrain."""

    assert segment.length is not None
    difficulty_level, difficulty = _difficulty(segment, global_difficulty_level)
    wall_thickness = segment.wall_thickness or (0.06 + 0.12 * difficulty)
    corridor_width = segment.corridor_width or (1.8 - 0.9 * difficulty)
    wall_count = segment.wall_count or (2 + round(6 * difficulty))

    if width <= MAZE_EXIT_WIDTH:
        raise ValueError(f"Maze track width must exceed the fixed {MAZE_EXIT_WIDTH:.1f} m exit width")
    if corridor_width >= width:
        raise ValueError("Maze corridor_width must be smaller than the track width")
    wall_spacing = segment.length / (wall_count + 1)
    if wall_thickness >= wall_spacing:
        raise ValueError("Maze walls are too thick or numerous for the segment length")

    floor = BoxGeom(
        name=f"terrain_{index:03d}_maze_000",
        center=(start_x + segment.length / 2.0, 0.0, start_z - FOUNDATION_DEPTH / 2.0),
        half_size=(segment.length / 2.0, width / 2.0, FOUNDATION_DEPTH / 2.0),
    )
    obstacles = [
        BoxGeom(
            name=f"obstacle_{index:03d}_maze_side_left",
            center=(
                start_x + segment.length / 2.0,
                width / 2.0 + wall_thickness / 2.0,
                start_z + MAZE_WALL_HEIGHT / 2.0,
            ),
            half_size=(segment.length / 2.0, wall_thickness / 2.0, MAZE_WALL_HEIGHT / 2.0),
        ),
        BoxGeom(
            name=f"obstacle_{index:03d}_maze_side_right",
            center=(
                start_x + segment.length / 2.0,
                -width / 2.0 - wall_thickness / 2.0,
                start_z + MAZE_WALL_HEIGHT / 2.0,
            ),
            half_size=(segment.length / 2.0, wall_thickness / 2.0, MAZE_WALL_HEIGHT / 2.0),
        ),
    ]

    blocked_width = width - corridor_width
    first_gap_on_left = (seed + index) % 2 == 0
    for wall_index in range(wall_count):
        gap_on_left = first_gap_on_left if wall_index % 2 == 0 else not first_gap_on_left
        center_y = -corridor_width / 2.0 if gap_on_left else corridor_width / 2.0
        obstacles.append(
            BoxGeom(
                name=f"obstacle_{index:03d}_maze_inner_{wall_index:03d}",
                center=(
                    start_x + (wall_index + 1) * wall_spacing,
                    center_y,
                    start_z + MAZE_WALL_HEIGHT / 2.0,
                ),
                half_size=(wall_thickness / 2.0, blocked_width / 2.0, MAZE_WALL_HEIGHT / 2.0),
            )
        )

    # The terminal wall always leaves one exact 1.2 m opening centered at y=0.
    end_piece_width = (width - MAZE_EXIT_WIDTH) / 2.0
    end_piece_center = (width + MAZE_EXIT_WIDTH) / 4.0
    for side_name, side in (("left", 1.0), ("right", -1.0)):
        obstacles.append(
            BoxGeom(
                name=f"obstacle_{index:03d}_maze_exit_{side_name}",
                center=(
                    start_x + segment.length - wall_thickness / 2.0,
                    side * end_piece_center,
                    start_z + MAZE_WALL_HEIGHT / 2.0,
                ),
                half_size=(wall_thickness / 2.0, end_piece_width / 2.0, MAZE_WALL_HEIGHT / 2.0),
            )
        )

    return TerrainPatch(
        index=index,
        kind=segment.kind,
        start_x=start_x,
        end_x=start_x + segment.length,
        start_z=start_z,
        end_z=start_z,
        difficulty=difficulty,
        difficulty_level=difficulty_level,
        out_of_distribution=False,
        parameters={
            "length": segment.length,
            "wall_thickness": wall_thickness,
            "corridor_width": corridor_width,
            "wall_count": wall_count,
            "exit_width": MAZE_EXIT_WIDTH,
            "wall_height": MAZE_WALL_HEIGHT,
        },
        geoms=(floor,),
        obstacle_geoms=tuple(obstacles),
    )


def build_segment(
    segment: SegmentSpec,
    index: int,
    start_x: float,
    start_z: float,
    width: float,
    global_difficulty_level: int,
    profile: str,
    limits: TerrainLimits,
    seed: int = 0,
) -> TerrainPatch:
    if segment.kind == "flat":
        return build_flat(segment, index, start_x, start_z, width, global_difficulty_level)
    if segment.kind in {"slope_up", "slope_down"}:
        return build_slope(segment, index, start_x, start_z, width, global_difficulty_level, profile, limits)
    if segment.kind in {"stairs_up", "stairs_down"}:
        return build_stairs(segment, index, start_x, start_z, width, global_difficulty_level, profile, limits)
    return build_maze(segment, index, start_x, start_z, width, global_difficulty_level, seed)
