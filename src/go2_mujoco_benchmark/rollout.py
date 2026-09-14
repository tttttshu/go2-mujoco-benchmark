"""Run a deployment bundle through the shared Go2 MuJoCo runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .composer import TrackComposer
from .mujoco_scene import _import_mujoco, compile_mujoco_track
from .paths import default_output, default_policy_dir, default_robot_scene, default_track
from .track import TrackSpec


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deployment_api():
    from .deployment import DeploymentConfig, MujocoRuntime

    return DeploymentConfig, MujocoRuntime


def _roll_pitch(quaternion_wxyz: np.ndarray) -> tuple[float, float]:
    w, x, y, z = (float(value) for value in quaternion_wxyz)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_term = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    return roll, math.asin(pitch_term)


def _body_linear_velocity(quaternion_wxyz: np.ndarray, world_velocity: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(value) for value in quaternion_wxyz)
    rotation = np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )
    return rotation.T @ np.asarray(world_velocity, dtype=np.float64)


def run_policy_rollout(
    policy_dir: str | Path,
    robot_scene: str | Path,
    track_spec: TrackSpec,
    *,
    command: tuple[float, float, float],
    policy_steps: int,
    fall_height: float = 0.12,
    max_tilt: float = 1.0,
) -> dict[str, Any]:
    """Exercise ONNX inference, observation history, PD control, and physics."""

    if policy_steps < 1:
        raise ValueError("policy_steps must be positive")
    DeploymentConfig, MujocoRuntime = _deployment_api()
    mujoco = _import_mujoco()
    policy_dir = Path(policy_dir).expanduser().resolve()
    config = DeploymentConfig.load(policy_dir / "deploy.json")
    track = TrackComposer().compile(track_spec)
    model = compile_mujoco_track(
        robot_scene,
        track,
        timestep=config.physics_dt or 0.005,
        gyro_sensor=config.imu_gyro_sensor,
        quaternion_sensor=config.imu_quaternion_sensor,
        mujoco_module=mujoco,
    )
    runtime = MujocoRuntime.from_model(policy_dir, model, mujoco_module=mujoco)
    runtime.set_command(command)
    start_patch = track.patches[0]
    start_x = start_patch.start_x + min(
        0.75,
        (start_patch.end_x - start_patch.start_x) / 2.0,
    )
    spawn = runtime.place_base_above_surface(
        (start_x, 0.0),
        surface_z=start_patch.start_z,
    )

    gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, config.imu_gyro_sensor)
    gyro_address = int(model.sensor_adr[gyro_id])
    terrain_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom.name)
        for patch in track.patches
        for geom in patch.geoms
    }
    initial_position = np.asarray(runtime.data.qpos[:3]).copy()
    velocity_errors = []
    contact_steps = 0
    min_base_z = float(runtime.data.qpos[2])
    max_tilt_seen = 0.0
    finite = True
    fell = False
    completed_steps = 0

    for _ in range(policy_steps):
        runtime.step_control()
        completed_steps += 1
        quaternion = np.asarray(runtime.data.qpos[3:7])
        roll, pitch = _roll_pitch(quaternion)
        tilt = max(abs(roll), abs(pitch))
        max_tilt_seen = max(max_tilt_seen, tilt)
        min_base_z = min(min_base_z, float(runtime.data.qpos[2]))

        body_velocity = _body_linear_velocity(quaternion, runtime.data.qvel[:3])
        gyro_z = float(runtime.data.sensordata[gyro_address + 2])
        actual_command = np.asarray((body_velocity[0], body_velocity[1], gyro_z))
        velocity_errors.append(actual_command - np.asarray(command))

        for contact_index in range(runtime.data.ncon):
            contact = runtime.data.contact[contact_index]
            if (int(contact.geom1) in terrain_ids) != (int(contact.geom2) in terrain_ids):
                contact_steps += 1
                break

        finite = bool(
            np.isfinite(runtime.data.qpos).all()
            and np.isfinite(runtime.data.qvel).all()
            and np.isfinite(runtime.data.sensordata).all()
        )
        fell = float(runtime.data.qpos[2]) < fall_height or tilt > max_tilt
        if not finite or fell:
            break

    final_position = np.asarray(runtime.data.qpos[:3]).copy()
    final_roll, final_pitch = _roll_pitch(np.asarray(runtime.data.qpos[3:7]))
    errors = np.asarray(velocity_errors)
    tracking_rmse = np.sqrt(np.mean(errors * errors, axis=0)) if len(errors) else np.full(3, np.nan)
    checks = {
        "finite_state_and_sensors": finite,
        "all_policy_steps_completed": completed_steps == policy_steps,
        "terrain_contact_observed": contact_steps > 0,
        "did_not_fall": not fell,
        "remained_within_track": (
            0.0 <= float(final_position[0]) <= track.total_length
            and abs(float(final_position[1])) <= track.width / 2.0
        ),
    }
    providers = list(runtime.policy.session.get_providers())
    return {
        "status": "POLICY_INTERFACE_OK" if all(checks.values()) else "POLICY_INTERFACE_FAILED",
        "checks": checks,
        "route": config.route,
        "track": {
            "name": track.name,
            "difficulty_level": track.difficulty_level,
            "total_length": track.total_length,
            "width": track.width,
        },
        "history_steps": config.history_steps,
        "observation_width": config.frame_dim * config.history_steps,
        "action_width": config.policy_output_dim,
        "onnx_providers": providers,
        "requested_policy_steps": policy_steps,
        "completed_policy_steps": completed_steps,
        "physics_steps": completed_steps * runtime.sim_steps_per_control,
        "sim_time": float(runtime.data.time),
        "command": list(command),
        "spawn": spawn,
        "velocity_tracking_rmse": tracking_rmse.tolist(),
        "initial_position": initial_position.tolist(),
        "final_position": final_position.tolist(),
        "forward_progress": float(final_position[0] - initial_position[0]),
        "course_completed": float(final_position[0]) >= track.total_length - 0.5,
        "contact_policy_steps": contact_steps,
        "min_base_z": min_base_z,
        "final_roll_rad": final_roll,
        "final_pitch_rad": final_pitch,
        "max_tilt_rad": max_tilt_seen,
        "max_abs_torque": runtime.max_torque,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, default=default_policy_dir())
    parser.add_argument("--robot-scene", type=Path, default=default_robot_scene())
    parser.add_argument("--track", type=Path, default=default_track())
    parser.add_argument("--difficulty-level", type=int, choices=range(1, 10), default=None)
    parser.add_argument("--output", type=Path, default=default_output())
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--label", default="unspecified")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy_dir = args.policy_dir.resolve()
    track_spec = TrackSpec.load(args.track)
    if args.difficulty_level is not None:
        track_spec = replace(track_spec, difficulty_level=args.difficulty_level, schema_version=2)
    report = run_policy_rollout(
        policy_dir,
        args.robot_scene,
        track_spec,
        command=(args.vx, args.vy, args.wz),
        policy_steps=args.steps,
    )
    bundle_provenance_path = policy_dir / "provenance.json"
    bundle_provenance = (
        json.loads(bundle_provenance_path.read_text(encoding="utf-8"))
        if bundle_provenance_path.is_file()
        else {}
    )
    report["evaluation_scope"] = (
        "interface_only" if bundle_provenance.get("synthetic") is True else "policy_candidate"
    )
    report["provenance"] = {
        "label": args.label,
        "bundle_synthetic": bundle_provenance.get("synthetic"),
        "policy_sha256": _sha256(policy_dir / "policy.onnx"),
        "deploy_sha256": _sha256(policy_dir / "deploy.json"),
        "track_sha256": _sha256(args.track),
        "robot_scene_sha256": _sha256(args.robot_scene),
        "python_version": platform.python_version(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"{report['status']} policy_steps={report['completed_policy_steps']} "
        f"progress={report['forward_progress']:.3f} output={args.output}"
    )
    if report["status"] != "POLICY_INTERFACE_OK":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
