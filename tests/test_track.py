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


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"name": "missing", "segments": []}, "non-empty"),
        ({"name": "bad", "segments": [{"type": "slope_up"}]}, "length"),
        ({"name": "bad", "segments": [{"type": "stairs_up", "steps": 0}]}, "positive integer"),
        ({"name": "bad", "segments": [{"type": "flat", "length": 1.0, "slope": 0.2}]}, "slope"),
    ],
)
def test_invalid_track_specs_are_rejected(payload, message):
    with pytest.raises(ValueError, match=message):
        TrackSpec.from_mapping(payload)
