"""Validated user-facing specifications for composable terrain tracks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

TERRAIN_TYPES = frozenset({"flat", "slope_up", "slope_down", "stairs_up", "stairs_down"})
DIFFICULTY_PROFILES = frozenset({"train", "stress"})


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number") from exc
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"{field} must be finite")
    return result


def _positive_float(value: Any, field: str) -> float:
    result = _finite_float(value, field)
    if result <= 0.0:
        raise ValueError(f"{field} must be positive")
    return result


def _difficulty(value: Any, field: str) -> float:
    result = _finite_float(value, field)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must lie in [0, 1]")
    return result


@dataclass(frozen=True)
class BoundaryWallSpec:
    enabled: bool = False
    height: float = 0.45
    thickness: float = 0.08

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> BoundaryWallSpec:
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise ValueError("boundary_walls must be a mapping")
        allowed = {"enabled", "height", "thickness"}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"Unknown boundary wall fields: {sorted(unknown)}")
        enabled = payload.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError("boundary_walls.enabled must be boolean")
        return cls(
            enabled=enabled,
            height=_positive_float(payload.get("height", 0.45), "boundary_walls.height"),
            thickness=_positive_float(payload.get("thickness", 0.08), "boundary_walls.thickness"),
        )


@dataclass(frozen=True)
class SegmentSpec:
    kind: str
    length: float | None = None
    steps: int | None = None
    difficulty: float | None = None
    slope: float | None = None
    step_height: float | None = None
    step_depth: float = 0.3

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> SegmentSpec:
        allowed = {"type", "length", "steps", "difficulty", "slope", "step_height", "step_depth"}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"Unknown segment fields: {sorted(unknown)}")
        kind = str(payload.get("type", ""))
        if kind not in TERRAIN_TYPES:
            raise ValueError(f"Unsupported terrain type: {kind!r}")

        length = _positive_float(payload["length"], "segment.length") if "length" in payload else None
        difficulty = _difficulty(payload["difficulty"], "segment.difficulty") if "difficulty" in payload else None
        slope = _finite_float(payload["slope"], "segment.slope") if "slope" in payload else None
        step_height = (
            _positive_float(payload["step_height"], "segment.step_height") if "step_height" in payload else None
        )
        step_depth = _positive_float(payload.get("step_depth", 0.3), "segment.step_depth")

        raw_steps = payload.get("steps")
        if raw_steps is not None and (isinstance(raw_steps, bool) or not isinstance(raw_steps, int) or raw_steps < 1):
            raise ValueError("segment.steps must be a positive integer")
        steps = raw_steps

        if kind in {"flat", "slope_up", "slope_down"} and length is None:
            raise ValueError(f"{kind} requires segment.length")
        if kind in {"stairs_up", "stairs_down"} and steps is None:
            raise ValueError(f"{kind} requires segment.steps")
        if slope is not None and kind not in {"slope_up", "slope_down"}:
            raise ValueError("segment.slope is valid only for slope terrain")
        if slope is not None and slope < 0.0:
            raise ValueError("segment.slope must be non-negative")
        if (step_height is not None or "step_depth" in payload) and kind not in {"stairs_up", "stairs_down"}:
            raise ValueError("step_height and step_depth are valid only for stair terrain")

        return cls(
            kind=kind,
            length=length,
            steps=steps,
            difficulty=difficulty,
            slope=slope,
            step_height=step_height,
            step_depth=step_depth,
        )


@dataclass(frozen=True)
class TrackSpec:
    name: str
    seed: int
    width: float
    global_difficulty: float
    difficulty_profile: str
    start_flat_length: float
    finish_flat_length: float
    segments: tuple[SegmentSpec, ...]
    boundary_walls: BoundaryWallSpec = BoundaryWallSpec()
    schema_version: int = 1

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> TrackSpec:
        allowed = {
            "schema_version",
            "name",
            "seed",
            "width",
            "global_difficulty",
            "difficulty_profile",
            "start_flat_length",
            "finish_flat_length",
            "segments",
            "boundary_walls",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"Unknown track fields: {sorted(unknown)}")
        schema_version = payload.get("schema_version", 1)
        if schema_version != 1:
            raise ValueError(f"Unsupported TrackSpec schema: {schema_version!r}")
        name = str(payload.get("name", "")).strip()
        if not name:
            raise ValueError("Track name must not be empty")
        seed = payload.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("Track seed must be an integer")
        profile = str(payload.get("difficulty_profile", "train"))
        if profile not in DIFFICULTY_PROFILES:
            raise ValueError(f"Unsupported difficulty profile: {profile!r}")
        segment_payloads = payload.get("segments")
        if not isinstance(segment_payloads, list) or not segment_payloads:
            raise ValueError("Track segments must be a non-empty list")
        if not all(isinstance(item, Mapping) for item in segment_payloads):
            raise ValueError("Every track segment must be a mapping")
        return cls(
            name=name,
            seed=seed,
            width=_positive_float(payload.get("width", 2.0), "track.width"),
            global_difficulty=_difficulty(payload.get("global_difficulty", 0.0), "track.global_difficulty"),
            difficulty_profile=profile,
            start_flat_length=_positive_float(payload.get("start_flat_length", 2.0), "start_flat_length"),
            finish_flat_length=_positive_float(payload.get("finish_flat_length", 2.0), "finish_flat_length"),
            segments=tuple(SegmentSpec.from_mapping(item) for item in segment_payloads),
            boundary_walls=BoundaryWallSpec.from_mapping(payload.get("boundary_walls")),
            schema_version=schema_version,
        )

    @classmethod
    def load(cls, path: str | Path) -> TrackSpec:
        path = Path(path)
        with path.open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
        if not isinstance(payload, Mapping):
            raise ValueError("Track file must contain a mapping")
        return cls.from_mapping(payload)


@dataclass(frozen=True)
class TerrainLimits:
    slope_train_range: tuple[float, float] = (0.0, 0.35)
    step_height_train_range: tuple[float, float] = (0.05, 0.15)
    slope_stress_max: float = 0.50
    step_height_stress_max: float = 0.23

    @classmethod
    def from_deploy_json(cls, path: str | Path) -> TerrainLimits:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        terrain = payload.get("training_terrain")
        if not isinstance(terrain, Mapping):
            raise ValueError("Deployment bundle does not contain training_terrain")
        slope_range = tuple(float(value) for value in terrain["slope_range"])
        step_range = tuple(float(value) for value in terrain["step_height_range"])
        if len(slope_range) != 2 or len(step_range) != 2:
            raise ValueError("Training terrain ranges must contain two values")
        return cls(slope_train_range=slope_range, step_height_train_range=step_range)

    def resolve_slope(self, difficulty: float, profile: str) -> float:
        if profile == "train":
            low, high = self.slope_train_range
        else:
            low, high = self.slope_train_range[1], self.slope_stress_max
        return low + difficulty * (high - low)

    def resolve_step_height(self, difficulty: float, profile: str) -> float:
        if profile == "train":
            low, high = self.step_height_train_range
        else:
            low, high = self.step_height_train_range[1], self.step_height_stress_max
        return low + difficulty * (high - low)
