from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from go2_mujoco_benchmark.deployment import DeploymentConfig, ObservationHistory, build_proprio


def _v1_payload() -> dict:
    return {
        "schema_version": 1,
        "route": "him",
        "observation": {"history_steps": 6},
        "control": {
            "dt": 0.02,
            "action_scale": [0.25] * 12,
            "command_ranges": [[-1.0, 1.0], [-0.5, 0.5], [-1.0, 1.0]],
        },
        "robot": {
            "joint_names": [f"j{i}" for i in range(12)],
            "motor_sdk_indices": list(range(12)),
            "default_joint_pos": [0.0] * 12,
            "stiffness": [25.0] * 12,
            "damping": [0.5] * 12,
            "joint_pos_limits": [[-1.0, 1.0]] * 12,
            "effort_limits": [23.5] * 12,
            "velocity_limits": [30.0] * 12,
        },
    }


def test_history_is_oldest_to_newest() -> None:
    history = ObservationHistory(3, frame_dim=2)
    np.testing.assert_array_equal(history.reset(np.asarray([1, 2])), [[1, 2, 1, 2, 1, 2]])
    np.testing.assert_array_equal(history.append(np.asarray([3, 4])), [[1, 2, 1, 2, 3, 4]])


def test_upright_proprio_contract() -> None:
    frame = build_proprio(
        np.asarray([1, 2, 3]),
        np.asarray([1, 0, 0, 0]),
        np.asarray([0.4, 0, 0]),
        np.ones(12),
        np.zeros(12),
        np.zeros(12),
        np.ones(12),
    )
    assert frame.shape == (45,)
    np.testing.assert_array_equal(frame[3:6], [0, 0, -1])
    np.testing.assert_array_equal(frame[9:21], np.zeros(12))


def test_v1_bundle_remains_supported(tmp_path: Path) -> None:
    path = tmp_path / "deploy.json"
    path.write_text(json.dumps(_v1_payload()), encoding="utf-8")
    config = DeploymentConfig.load(path)
    assert config.schema_version == 1
    assert config.mujoco_joint_names == tuple(f"j{i}_joint" for i in range(12))
    np.testing.assert_allclose(config.validate_command([0.4, 0.0, 0.0]), [0.4, 0.0, 0.0])
    np.testing.assert_array_equal(config.action_to_joint_target(np.full(12, 10.0)), np.ones(12))
