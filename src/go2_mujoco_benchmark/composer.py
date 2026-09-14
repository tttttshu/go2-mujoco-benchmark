"""Compose validated terrain segments into one continuous forward track."""

from __future__ import annotations

from .generators import build_segment
from .geometry import BoxGeom, CompiledTrack
from .track import SegmentSpec, TerrainLimits, TrackSpec


class TrackComposer:
    def __init__(self, limits: TerrainLimits | None = None) -> None:
        self.limits = TerrainLimits() if limits is None else limits

    def compile(self, spec: TrackSpec) -> CompiledTrack:
        patches = []
        cursor_x = 0.0
        cursor_z = 0.0
        segments = (
            SegmentSpec(kind="flat", length=spec.start_flat_length),
            *spec.segments,
            SegmentSpec(kind="flat", length=spec.finish_flat_length),
        )
        for index, segment in enumerate(segments):
            patch = build_segment(
                segment=segment,
                index=index,
                start_x=cursor_x,
                start_z=cursor_z,
                width=spec.width,
                global_difficulty_level=spec.difficulty_level,
                profile=spec.difficulty_profile,
                limits=self.limits,
                seed=spec.seed,
            )
            if abs(patch.start_x - cursor_x) > 1.0e-9 or abs(patch.start_z - cursor_z) > 1.0e-9:
                raise RuntimeError(f"Terrain generator broke the chaining contract at patch {index}")
            patches.append(patch)
            cursor_x = patch.end_x
            cursor_z = patch.end_z
        boundary_geoms = []
        if spec.boundary_walls.enabled:
            wall = spec.boundary_walls
            for patch in patches:
                for geom_index, geom in enumerate(patch.geoms):
                    w, x, y, z = geom.quaternion_wxyz
                    local_up = (
                        2.0 * (x * z + w * y),
                        2.0 * (y * z - w * x),
                        1.0 - 2.0 * (x * x + y * y),
                    )
                    lift = geom.half_size[2] + wall.height / 2.0
                    for side_name, side in (("left", 1.0), ("right", -1.0)):
                        boundary_geoms.append(
                            BoxGeom(
                                name=f"boundary_{patch.index:03d}_{geom_index:03d}_{side_name}",
                                center=(
                                    geom.center[0] + local_up[0] * lift,
                                    side * (spec.width / 2.0 + wall.thickness / 2.0),
                                    geom.center[2] + local_up[2] * lift,
                                ),
                                half_size=(geom.half_size[0], wall.thickness / 2.0, wall.height / 2.0),
                                quaternion_wxyz=geom.quaternion_wxyz,
                            )
                        )
        return CompiledTrack(
            name=spec.name,
            seed=spec.seed,
            width=spec.width,
            difficulty_level=spec.difficulty_level,
            total_length=cursor_x,
            start_z=0.0,
            end_z=cursor_z,
            patches=tuple(patches),
            boundary_geoms=tuple(boundary_geoms),
        )
