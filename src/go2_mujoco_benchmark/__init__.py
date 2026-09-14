"""Composable MuJoCo terrain benchmark for Unitree Go2."""

from .composer import TrackComposer
from .geometry import BoxGeom, CompiledTrack, TerrainPatch
from .mujoco_scene import compile_mujoco_track
from .track import BoundaryWallSpec, SegmentSpec, TerrainLimits, TrackSpec

__all__ = [
    "BoxGeom",
    "BoundaryWallSpec",
    "CompiledTrack",
    "SegmentSpec",
    "TerrainLimits",
    "TerrainPatch",
    "TrackComposer",
    "TrackSpec",
    "compile_mujoco_track",
]

__version__ = "0.2.0"
