"""Run a deterministic Go2 drop-and-hold physics smoke test on a generated track."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np

from .composer import TrackComposer
from .mujoco_scene import _import_mujoco, compile_mujoco_track
from .paths import default_robot_scene, default_track
from .track import TrackSpec

GO2_JOINT_NAMES = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)
GO2_ACTUATOR_NAMES = tuple(name.removesuffix("_joint") for name in GO2_JOINT_NAMES)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object_id(mujoco, model, object_type, name: str) -> int:
    object_id = int(mujoco.mj_name2id(model, object_type, name))
    if object_id < 0:
        raise KeyError(f"Required Go2 model object not found: {name}")
    return object_id


def _roll_pitch(quaternion_wxyz: np.ndarray) -> tuple[float, float]:
    w, x, y, z = (float(value) for value in quaternion_wxyz)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_term = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    return roll, math.asin(pitch_term)


def run_go2_physics_smoke(
    model,
    track,
    *,
    duration: float = 3.0,
    drop_height: float = 0.08,
    stiffness: float = 25.0,
    damping: float = 0.5,
    mujoco_module=None,
) -> dict[str, Any]:
    """Drop the Go2 from 8 cm while holding its home joint pose with PD."""

    if duration <= 0.0:
        raise ValueError("duration must be positive")
    if drop_height < 0.0 or stiffness < 0.0 or damping < 0.0:
        raise ValueError("drop_height, stiffness and damping must be non-negative")
    if not track.patches or track.patches[0].kind != "flat":
        raise ValueError("physics smoke test requires a starting flat patch")

    mujoco = _import_mujoco() if mujoco_module is None else mujoco_module
    data = mujoco.MjData(model)
    key_id = _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_KEY, "home")
    joint_ids = [
        _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in GO2_JOINT_NAMES
    ]
    actuator_ids = [
        _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in GO2_ACTUATOR_NAMES
    ]
    qpos_addresses = np.asarray([model.jnt_qposadr[index] for index in joint_ids], dtype=np.int32)
    qvel_addresses = np.asarray([model.jnt_dofadr[index] for index in joint_ids], dtype=np.int32)

    mujoco.mj_resetDataKeyframe(model, data, key_id)
    target = np.asarray(data.qpos[qpos_addresses]).copy()
    home_base_z = float(data.qpos[2])
    start = track.patches[0]
    data.qpos[0] = start.start_x + min(0.75, (start.end_x - start.start_x) / 2.0)
    data.qpos[1] = 0.0
    data.qpos[2] = home_base_z + drop_height
    mujoco.mj_forward(model, data)

    terrain_ids = {
        _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, geom.name)
        for patch in track.patches
        for geom in patch.geoms
    }
    steps = max(1, round(duration / model.opt.timestep))
    finite = True
    contact_steps = 0
    terrain_contact_events = 0
    min_base_z = float(data.qpos[2])
    max_abs_joint_velocity = 0.0
    max_abs_torque = 0.0

    for _ in range(steps):
        torque = stiffness * (target - data.qpos[qpos_addresses]) - damping * data.qvel[qvel_addresses]
        limits = model.actuator_ctrlrange[actuator_ids]
        torque = np.clip(torque, limits[:, 0], limits[:, 1])
        data.ctrl[actuator_ids] = torque
        max_abs_torque = max(max_abs_torque, float(np.max(np.abs(torque))))
        mujoco.mj_step(model, data)

        step_contacts = 0
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            if (int(contact.geom1) in terrain_ids) != (int(contact.geom2) in terrain_ids):
                step_contacts += 1
        if step_contacts:
            contact_steps += 1
            terrain_contact_events += step_contacts

        min_base_z = min(min_base_z, float(data.qpos[2]))
        max_abs_joint_velocity = max(
            max_abs_joint_velocity,
            float(np.max(np.abs(data.qvel[qvel_addresses]))),
        )
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            finite = False
            break

    roll, pitch = _roll_pitch(np.asarray(data.qpos[3:7]))
    final_base_z = float(data.qpos[2])
    checks = {
        "finite_state": finite,
        "terrain_contacts_observed": contact_steps > 0,
        "did_not_fall_through": min_base_z > -0.05,
        "base_height_recovered": abs(final_base_z - home_base_z) < 0.08,
        "upright_at_end": max(abs(roll), abs(pitch)) < 0.35,
        "bounded_joint_velocity": max_abs_joint_velocity < 80.0,
    }
    return {
        "status": "PHYSICS_SMOKE_OK" if all(checks.values()) else "PHYSICS_SMOKE_FAILED",
        "checks": checks,
        "duration": steps * float(model.opt.timestep),
        "physics_steps": steps,
        "terrain_geom_count": len(terrain_ids),
        "contact_steps": contact_steps,
        "terrain_contact_events": terrain_contact_events,
        "home_base_z": home_base_z,
        "final_base_z": final_base_z,
        "min_base_z": min_base_z,
        "final_roll_rad": roll,
        "final_pitch_rad": pitch,
        "max_abs_joint_velocity": max_abs_joint_velocity,
        "max_abs_torque": max_abs_torque,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-scene", type=Path, default=default_robot_scene())
    parser.add_argument("--track", type=Path, default=default_track())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--drop-height", type=float, default=0.08)
    parser.add_argument("--model-revision", default="unknown")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mujoco = _import_mujoco()
    spec = TrackSpec.load(args.track)
    track = TrackComposer().compile(spec)
    model = compile_mujoco_track(args.robot_scene, track, mujoco_module=mujoco)
    report = run_go2_physics_smoke(
        model,
        track,
        duration=args.duration,
        drop_height=args.drop_height,
        mujoco_module=mujoco,
    )
    report["provenance"] = {
        "track_file": str(args.track.resolve()),
        "track_sha256": _sha256(args.track),
        "robot_scene": str(args.robot_scene.resolve()),
        "robot_scene_sha256": _sha256(args.robot_scene),
        "model_revision": args.model_revision,
        "mujoco_version": mujoco.__version__,
        "python_version": platform.python_version(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"{report['status']} steps={report['physics_steps']} "
        f"contacts={report['terrain_contact_events']} output={args.output}"
    )
    if report["status"] != "PHYSICS_SMOKE_OK":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
