from __future__ import annotations

from pathlib import Path

import pytest

from go2_mujoco_benchmark import TrackComposer, TrackSpec, compile_mujoco_track
from go2_mujoco_benchmark.mujoco_scene import _mujoco_readable_scene_path

mujoco = pytest.importorskip("mujoco")


def test_generated_track_replaces_floor_and_compiles(tmp_path: Path):
    scene = tmp_path / "scene.xml"
    scene.write_text(
        """<mujoco model="fixture">
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05"/>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <geom type="sphere" size="0.05"/>
      <site name="imu"/>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    spec = TrackSpec.from_mapping(
        {
            "name": "compile_fixture",
            "start_flat_length": 1.0,
            "finish_flat_length": 1.0,
            "boundary_walls": {"enabled": True, "height": 0.4, "thickness": 0.08},
            "segments": [{"type": "slope_up", "length": 1.0, "slope": 0.1}],
        }
    )
    track = TrackComposer().compile(spec)

    model = compile_mujoco_track(scene, track, timestep=0.002)

    assert model.opt.timestep == pytest.approx(0.002)
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor") == -1
    gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
    quat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat")
    assert model.sensor_dim[gyro_id] == 3
    assert model.sensor_dim[quat_id] == 4
    for patch in track.patches:
        for geom in patch.geoms:
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom.name) >= 0
    assert track.boundary_geoms
    for geom in track.boundary_geoms:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom.name) >= 0


def test_model_directory_can_be_materialized_in_ascii_cache(tmp_path: Path):
    source = tmp_path / "模型" / "scene.xml"
    source.parent.mkdir()
    source.write_text("<mujoco/>", encoding="utf-8")
    (source.parent / "mesh.obj").write_text("mesh", encoding="utf-8")

    cached = _mujoco_readable_scene_path(
        source.resolve(),
        force_ascii_cache=True,
        cache_root=tmp_path / "ascii_cache",
    )

    assert cached.read_text(encoding="utf-8") == "<mujoco/>"
    assert (cached.parent / "mesh.obj").read_text(encoding="utf-8") == "mesh"
    str(cached).encode("ascii")


@pytest.mark.parametrize("timestep", [0.0, -0.001])
def test_invalid_timestep_is_rejected(tmp_path: Path, timestep: float):
    track = TrackComposer().compile(
        TrackSpec.from_mapping(
            {"name": "x", "segments": [{"type": "flat", "length": 1.0}]}
        )
    )
    with pytest.raises(ValueError, match="positive"):
        compile_mujoco_track(tmp_path / "missing.xml", track, timestep=timestep)
