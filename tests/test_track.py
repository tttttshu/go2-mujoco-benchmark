from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from go2_mujoco_benchmark import TerrainLimits, TrackComposer, TrackSpec

ROOT = Path(__file__).parents[1]


def test_mixed_track_is_continuous_and_returns_to_zero_height():
    spec = TrackSpec.load(ROOT / "configs" / "tracks" / "mixed.yaml")
    track = TrackComposer().compile(spec)

    assert track.total_length == pytest.approx(16.0)
    assert track.end_z == pytest.approx(0.0)
    assert len(track.patches) == len(spec.segments) + 2
    assert len({geom.name for patch in track.patches for geom in patch.geoms}) == sum(
        len(patch.geoms) for patch in track.patches
    )
    assert track.boundary_geoms == ()
    for left, right in zip(track.patches, track.patches[1:]):
        assert left.end_x == pytest.approx(right.start_x)
        assert left.end_z == pytest.approx(right.start_z)


def test_slope_box_top_surface_matches_requested_endpoints():
    spec = TrackSpec.from_mapping(
        {
            "name": "slope",
            "global_difficulty": 1.0,
            "segments": [{"type": "slope_up", "length": 4.0}],
        }
    )
    patch = TrackComposer().compile(spec).patches[1]
    geom = patch.geoms[0]
    w, _, y, _ = geom.quaternion_wxyz
    pitch = 2.0 * math.atan2(y, w)
    local_half_length = geom.half_size[0]
    local_half_height = geom.half_size[2]

    def top_endpoint(local_x: float) -> tuple[float, float]:
        world_x = geom.center[0] + math.cos(pitch) * local_x + math.sin(pitch) * local_half_height
        world_z = geom.center[2] - math.sin(pitch) * local_x + math.cos(pitch) * local_half_height
        return world_x, world_z

    assert top_endpoint(-local_half_length) == pytest.approx((patch.start_x, patch.start_z))
    assert top_endpoint(local_half_length) == pytest.approx((patch.end_x, patch.end_z))


def test_stair_boxes_share_a_solid_foundation_and_have_expected_tops():
    spec = TrackSpec.from_mapping(
        {
            "name": "stairs",
            "global_difficulty": 1.0,
            "segments": [{"type": "stairs_up", "steps": 3}],
        }
    )
    patch = TrackComposer().compile(spec).patches[1]
    bottoms = [geom.center[2] - geom.half_size[2] for geom in patch.geoms]
    tops = [geom.center[2] + geom.half_size[2] for geom in patch.geoms]

    assert bottoms == pytest.approx([bottoms[0]] * 3)
    assert tops == pytest.approx([0.15, 0.30, 0.45])
    assert patch.end_z == pytest.approx(0.45)


def test_stress_profile_and_explicit_override_are_marked_ood():
    stress = TrackSpec.from_mapping(
        {
            "name": "stress",
            "difficulty_profile": "stress",
            "global_difficulty": 0.5,
            "segments": [
                {"type": "slope_up", "length": 1.0},
                {"type": "stairs_up", "steps": 1, "step_height": 0.22},
            ],
        }
    )
    track = TrackComposer().compile(stress)

    assert track.patches[1].parameters["slope"] == pytest.approx(0.425)
    assert track.patches[1].out_of_distribution
    assert track.patches[2].parameters["step_height"] == pytest.approx(0.22)
    assert track.patches[2].out_of_distribution


def test_training_limits_can_be_loaded_from_deployment_bundle(tmp_path: Path):
    deploy = tmp_path / "deploy.json"
    deploy.write_text(
        json.dumps({"training_terrain": {"slope_range": [0.0, 0.4], "step_height_range": [0.04, 0.16]}}),
        encoding="utf-8",
    )

    limits = TerrainLimits.from_deploy_json(deploy)

    assert limits.resolve_slope(0.5, "train") == pytest.approx(0.2)
    assert limits.resolve_step_height(0.5, "train") == pytest.approx(0.1)


def test_boundary_walls_follow_both_track_edges():
    spec = TrackSpec.load(ROOT / "configs" / "tracks" / "flat_20m.yaml")
    track = TrackComposer().compile(spec)

    terrain_geom_count = sum(len(patch.geoms) for patch in track.patches)
    assert len(track.boundary_geoms) == 2 * terrain_geom_count
    for wall in track.boundary_geoms:
        assert abs(wall.center[1]) - wall.half_size[1] == pytest.approx(track.width / 2.0)
        assert wall.half_size[2] == pytest.approx(0.225)


def test_maze_difficulty_controls_geometry_and_exit_is_exactly_1_2m():
    def compile_level(level: int):
        return TrackComposer().compile(
            TrackSpec.from_mapping(
                {
                    "schema_version": 2,
                    "name": f"maze_{level}",
                    "width": 3.0,
                    "difficulty_level": level,
                    "boundary_walls": {"enabled": True},
                    "segments": [{"type": "maze", "length": 5.0}],
                }
            )
        )

    easy = compile_level(1)
    hard = compile_level(9)
    easy_maze = easy.patches[1]
    hard_maze = hard.patches[1]

    assert hard_maze.parameters["wall_thickness"] > easy_maze.parameters["wall_thickness"]
    assert hard_maze.parameters["wall_length"] > easy_maze.parameters["wall_length"]
    assert hard_maze.parameters["corridor_width"] < easy_maze.parameters["corridor_width"]
    assert hard_maze.parameters["wall_count"] > easy_maze.parameters["wall_count"]
    assert hard_maze.parameters["exit_width"] == pytest.approx(1.2)
    exit_walls = [geom for geom in hard_maze.obstacle_geoms if "maze_exit" in geom.name]
    assert len(exit_walls) == 2
    inner_edges = sorted(
        geom.center[1] - geom.half_size[1] if geom.center[1] > 0 else geom.center[1] + geom.half_size[1]
        for geom in exit_walls
    )
    assert inner_edges == pytest.approx([-0.6, 0.6])
    internal_walls = [geom for geom in hard_maze.obstacle_geoms if "maze_inner" in geom.name]
    assert len(internal_walls) == hard_maze.parameters["wall_count"]
    assert len(hard_maze.obstacle_geoms) == hard_maze.parameters["wall_count"] + 2
    assert all(abs(geom.center[1]) + geom.half_size[1] < hard.width / 2.0 for geom in internal_walls)
    assert any(geom.name.startswith("boundary_001") for geom in hard.boundary_geoms)


def test_maze_overrides_and_arbitrary_segment_order_are_preserved():
    spec = TrackSpec.from_mapping(
        {
            "schema_version": 2,
            "name": "custom_maze",
            "width": 3.2,
            "difficulty_level": 4,
            "segments": [
                {"type": "maze", "length": 5.0, "wall_thickness": 0.11, "wall_length": 0.65, "wall_count": 9},
                {"type": "flat", "length": 1.0},
                {"type": "maze", "length": 6.0, "difficulty_level": 8},
            ],
        }
    )
    track = TrackComposer().compile(spec)

    assert [patch.kind for patch in track.patches] == ["flat", "maze", "flat", "maze", "flat"]
    assert track.patches[1].parameters["wall_thickness"] == pytest.approx(0.11)
    assert track.patches[1].parameters["wall_length"] == pytest.approx(0.65)
    assert track.patches[1].parameters["wall_count"] == 9
    assert track.patches[3].difficulty_level == 8
    for left, right in zip(track.patches, track.patches[1:]):
        assert (left.end_x, left.end_z) == pytest.approx((right.start_x, right.start_z))


def test_v1_normalized_difficulty_is_backward_compatible():
    spec = TrackSpec.from_mapping(
        {"schema_version": 1, "name": "legacy", "global_difficulty": 0.5, "segments": [{"type": "flat", "length": 1.0}]}
    )
    assert spec.difficulty_level == 5
    assert spec.global_difficulty == pytest.approx(0.5)


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"name": "missing", "segments": []}, "non-empty"),
        ({"name": "bad", "segments": [{"type": "slope_up"}]}, "length"),
        ({"name": "bad", "segments": [{"type": "stairs_up", "steps": 0}]}, "positive integer"),
        ({"name": "bad", "segments": [{"type": "flat", "length": 1.0, "slope": 0.2}]}, "slope"),
        (
            {
                "schema_version": 2,
                "name": "bad",
                "difficulty_level": 10,
                "segments": [{"type": "flat", "length": 1.0}],
            },
            r"\[1, 9\]",
        ),
        (
            {
                "schema_version": 2,
                "name": "bad",
                "difficulty_level": 2.5,
                "segments": [{"type": "flat", "length": 1.0}],
            },
            "integer",
        ),
        (
            {
                "schema_version": 2,
                "name": "bad",
                "segments": [{"type": "maze", "length": 2.0, "wall_count": 0}],
            },
            "positive integer",
        ),
    ],
)
def test_invalid_track_specs_are_rejected(payload, message):
    with pytest.raises(ValueError, match=message):
        TrackSpec.from_mapping(payload)
