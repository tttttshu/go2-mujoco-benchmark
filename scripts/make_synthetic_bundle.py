"""Create a clearly-labelled zero-action ONNX bundle for interface testing only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_policy(path: Path, input_width: int) -> None:
    observation = helper.make_tensor_value_info("observation", TensorProto.FLOAT, [None, input_width])
    action = helper.make_tensor_value_info("action", TensorProto.FLOAT, [None, 12])
    weights = numpy_helper.from_array(np.zeros((input_width, 12), dtype=np.float32), name="zero_weights")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["observation", "zero_weights"], ["action"])],
        "synthetic_zero_action_policy",
        [observation],
        [action],
        [weights],
    )
    model = helper.make_model(graph, producer_name="go2_mujoco_benchmark", opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, path)


def deploy_payload(route: str) -> dict:
    history_steps = {"competition": 1, "him": 6}[route]
    joint_names = [
        f"{leg}_{joint}"
        for leg in ("FR", "FL", "RR", "RL")
        for joint in ("hip", "thigh", "calf")
    ]
    front_limits = [[-1.0472, 1.0472], [-1.5708, 3.4907], [-2.7227, -0.83776]]
    rear_limits = [[-1.0472, 1.0472], [-0.5236, 4.5379], [-2.7227, -0.83776]]
    return {
        "schema_version": 2,
        "route": route,
        "policy_file": "policy.onnx",
        "policy": {
            "file": "policy.onnx",
            "dtype": "float32",
            "input_dim": 45 * history_steps,
            "output_dim": 12,
        },
        "observation": {
            "frame_dim": 45,
            "history_steps": history_steps,
            "history_order": "oldest_to_newest",
            "terms": [
                ["base_ang_vel", 3],
                ["projected_gravity", 3],
                ["velocity_command", 3],
                ["joint_pos_rel", 12],
                ["joint_vel_rel", 12],
                ["previous_action", 12],
            ],
            "quaternion_order": "wxyz",
        },
        "control": {
            "mode": "joint_position_pd",
            "dt": 0.02,
            "physics_dt": 0.005,
            "action_clip": 100.0,
            "action_scale": [0.25] * 12,
            "command_ranges": [[-1.0, 1.0], [-0.5, 0.5], [-1.0, 1.0]],
        },
        "robot": {
            "name": "Unitree Go2 synthetic interface fixture",
            "joint_names": joint_names,
            "usd_joint_names": [f"{name}_joint" for name in joint_names],
            "mujoco_joint_names": [f"{name}_joint" for name in joint_names],
            "mujoco_actuator_names": joint_names,
            "motor_sdk_indices": list(range(12)),
            "sensors": {"gyro": "imu_gyro", "quaternion": "imu_quat"},
            "default_joint_pos": [value for _ in range(4) for value in (0.0, 0.9, -1.8)],
            "stiffness": [25.0] * 12,
            "damping": [0.5] * 12,
            "joint_pos_limits": front_limits * 2 + rear_limits * 2,
            "effort_limits": [value for _ in range(4) for value in (23.7, 23.7, 45.43)],
            "velocity_limits": [30.0] * 12,
        },
        "training_terrain": {
            "difficulty_range": [0.0, 1.0],
            "slope_range": [0.0, 0.35],
            "step_height_range": [0.05, 0.15],
            "step_width": 0.3,
            "terrain_types": ["flat", "slope_up", "slope_down", "stairs_up", "stairs_down"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route", choices=("competition", "him"), default="competition")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    deploy = deploy_payload(args.route)
    policy_path = args.output / "policy.onnx"
    deploy_path = args.output / "deploy.json"
    make_policy(policy_path, deploy["policy"]["input_dim"])
    deploy_path.write_text(json.dumps(deploy, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "schema_version": 1,
        "synthetic": True,
        "purpose": "interface_validation_only_not_a_trained_policy",
        "artifacts": {
            "policy.onnx": {"sha256": _sha256(policy_path)},
            "deploy.json": {"sha256": _sha256(deploy_path)},
        },
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"SYNTHETIC_BUNDLE_OK route={args.route} output={args.output}")


if __name__ == "__main__":
    main()
