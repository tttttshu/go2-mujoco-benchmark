"""Deterministic primitive terrain generators with a common chaining contract."""

from __future__ import annotations

import math

from .geometry import BoxGeom, TerrainPatch
from .track import SegmentSpec, TerrainLimits

FOUNDATION_DEPTH = 0.2


def _difficulty(segment: SegmentSpec, global_difficulty: float) -> float:
    return global_difficulty if segment.difficulty is None else segment.difficulty


def build_flat(
    segment: SegmentSpec,
    index: int,
    start_x: float,
    start_z: float,
    width: float,
    global_difficulty: float,
) -> TerrainPatch:
    assert segment.length is not None
    difficulty = _difficulty(segment, global_difficulty)
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
    global_difficulty: float,
    profile: str,
    limits: TerrainLimits,
) -> TerrainPatch:
    assert segment.length is not None
    difficulty = _difficulty(segment, global_difficulty)
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
    global_difficulty: float,
    profile: str,
    limits: TerrainLimits,
) -> TerrainPatch:
    assert segment.steps is not None
    difficulty = _difficulty(segment, global_difficulty)
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
        out_of_distribution=not train_low <= step_height <= train_high,
        parameters={"steps": segment.steps, "step_height": step_height, "step_depth": segment.step_depth},
        geoms=tuple(geoms),
    )


def build_segment(
    segment: SegmentSpec,
    index: int,
    start_x: float,
    start_z: float,
    width: float,
    global_difficulty: float,
    profile: str,
    limits: TerrainLimits,
) -> TerrainPatch:
    if segment.kind == "flat":
        return build_flat(segment, index, start_x, start_z, width, global_difficulty)
    if segment.kind in {"slope_up", "slope_down"}:
        return build_slope(segment, index, start_x, start_z, width, global_difficulty, profile, limits)
    return build_stairs(segment, index, start_x, start_z, width, global_difficulty, profile, limits)
