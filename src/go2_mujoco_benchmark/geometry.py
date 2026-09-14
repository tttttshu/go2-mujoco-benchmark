"""Simulator-neutral geometry emitted by terrain generators."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BoxGeom:
    name: str
    center: tuple[float, float, float]
    half_size: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class TerrainPatch:
    index: int
    kind: str
    start_x: float
    end_x: float
    start_z: float
    end_z: float
    difficulty: float
    difficulty_level: int
    out_of_distribution: bool
    parameters: dict[str, float | int | str]
    geoms: tuple[BoxGeom, ...]
    obstacle_geoms: tuple[BoxGeom, ...] = ()


@dataclass(frozen=True)
class CompiledTrack:
    name: str
    seed: int
    width: float
    difficulty_level: int
    total_length: float
    start_z: float
    end_z: float
    patches: tuple[TerrainPatch, ...]
    boundary_geoms: tuple[BoxGeom, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
