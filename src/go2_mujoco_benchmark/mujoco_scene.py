"""Compile a generated track into an existing MuJoCo robot scene."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from types import ModuleType

from .geometry import CompiledTrack

TERRAIN_COLORS = {
    "flat": (0.32, 0.42, 0.32, 1.0),
    "slope_up": (0.34, 0.42, 0.55, 1.0),
    "slope_down": (0.34, 0.42, 0.55, 1.0),
    "stairs_up": (0.52, 0.40, 0.28, 1.0),
    "stairs_down": (0.52, 0.40, 0.28, 1.0),
}
BOUNDARY_COLOR = (0.20, 0.24, 0.30, 1.0)


def _mujoco_readable_scene_path(
    scene_path: Path,
    *,
    force_ascii_cache: bool = False,
    cache_root: Path | None = None,
) -> Path:
    """Mirror models out of non-ASCII Windows paths before C-level XML loading."""
    try:
        str(scene_path).encode("ascii")
        ascii_path = True
    except UnicodeEncodeError:
        ascii_path = False
    if not force_ascii_cache and (os.name != "nt" or ascii_path):
        return scene_path

    source_dir = scene_path.parent
    if cache_root is None:
        local_data = os.environ.get("LOCALAPPDATA")
        cache_root = Path(local_data) if local_data else Path(tempfile.gettempdir())
        cache_root = cache_root / "go2_mujoco_benchmark" / "model_cache"
    fingerprint = hashlib.sha256(str(source_dir).encode("utf-8")).hexdigest()[:12]
    destination = cache_root / f"model-{fingerprint}"
    marker = destination / ".copy_complete"
    cached_scene = destination / scene_path.name
    if not marker.is_file() or not cached_scene.is_file():
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_dir, destination, dirs_exist_ok=True)
        marker.write_text(str(source_dir), encoding="utf-8")
    return cached_scene


def _import_mujoco() -> ModuleType:
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("Install the 'runtime' extra to compile MuJoCo scenes") from exc
    return mujoco


def compile_mujoco_track(
    robot_scene: str | Path,
    track: CompiledTrack,
    *,
    timestep: float = 0.005,
    floor_name: str | None = "floor",
    imu_site: str = "imu",
    gyro_sensor: str = "imu_gyro",
    quaternion_sensor: str = "imu_quat",
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001),
    mujoco_module: ModuleType | None = None,
):
    """Load ``robot_scene``, replace its floor, add the track, and compile it.

    MuJoCo's native ``MjSpec`` keeps mesh assets resolved relative to the source
    scene, so the generated course does not need to copy or rewrite robot assets.
    """

    if timestep <= 0.0:
        raise ValueError("timestep must be positive")
    if len(friction) != 3 or any(value < 0.0 for value in friction):
        raise ValueError("friction must contain three non-negative coefficients")

    mujoco = _import_mujoco() if mujoco_module is None else mujoco_module
    scene_path = Path(robot_scene).expanduser().resolve()
    if not scene_path.is_file():
        raise FileNotFoundError(f"MuJoCo robot scene not found: {scene_path}")

    readable_scene_path = _mujoco_readable_scene_path(scene_path)
    spec = mujoco.MjSpec.from_file(str(readable_scene_path))
    if floor_name:
        floor = spec.geom(floor_name)
        if floor is not None:
            spec.delete(floor)
    spec.modelname = f"go2 benchmark: {track.name}"
    spec.option.timestep = timestep

    if spec.site(imu_site) is None:
        raise KeyError(f"MuJoCo robot scene has no IMU site named {imu_site!r}")
    if spec.sensor(gyro_sensor) is None:
        spec.add_sensor(
            name=gyro_sensor,
            type=mujoco.mjtSensor.mjSENS_GYRO,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname=imu_site,
        )
    if spec.sensor(quaternion_sensor) is None:
        spec.add_sensor(
            name=quaternion_sensor,
            type=mujoco.mjtSensor.mjSENS_FRAMEQUAT,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname=imu_site,
        )

    for patch in track.patches:
        color = TERRAIN_COLORS[patch.kind]
        for geom in patch.geoms:
            spec.worldbody.add_geom(
                name=geom.name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=list(geom.center),
                size=list(geom.half_size),
                quat=list(geom.quaternion_wxyz),
                rgba=list(color),
                friction=list(friction),
                contype=1,
                conaffinity=1,
                condim=4,
                margin=0.001,
            )

    for geom in track.boundary_geoms:
        spec.worldbody.add_geom(
            name=geom.name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=list(geom.center),
            size=list(geom.half_size),
            quat=list(geom.quaternion_wxyz),
            rgba=list(BOUNDARY_COLOR),
            friction=list(friction),
            contype=1,
            conaffinity=1,
            condim=4,
            margin=0.001,
        )

    model = spec.compile()
    expected_names = {geom.name for patch in track.patches for geom in patch.geoms}
    compiled_names = {
        model.geom(index).name
        for index in range(model.ngeom)
        if model.geom(index).name and model.geom(index).name.startswith("terrain_")
    }
    if compiled_names != expected_names:
        missing = sorted(expected_names - compiled_names)
        extra = sorted(compiled_names - expected_names)
        raise RuntimeError(f"Compiled terrain mismatch: missing={missing}, extra={extra}")
    expected_boundary_names = {geom.name for geom in track.boundary_geoms}
    compiled_boundary_names = {
        model.geom(index).name
        for index in range(model.ngeom)
        if model.geom(index).name and model.geom(index).name.startswith("boundary_")
    }
    if compiled_boundary_names != expected_boundary_names:
        missing = sorted(expected_boundary_names - compiled_boundary_names)
        extra = sorted(compiled_boundary_names - expected_boundary_names)
        raise RuntimeError(f"Compiled boundary mismatch: missing={missing}, extra={extra}")
    expected_sensor_dims = {gyro_sensor: 3, quaternion_sensor: 4}
    for sensor_name, expected_dim in expected_sensor_dims.items():
        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if sensor_id < 0 or int(model.sensor_dim[sensor_id]) != expected_dim:
            raise RuntimeError(f"Compiled IMU sensor mismatch: {sensor_name}")
    return model
