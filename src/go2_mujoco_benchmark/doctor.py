"""Check that a checkout can run the same MuJoCo policy stack on any OS."""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
from pathlib import Path

from .composer import TrackComposer
from .deployment import DeploymentConfig, MujocoRuntime
from .mujoco_scene import _import_mujoco, compile_mujoco_track
from .paths import default_policy_dir, default_robot_scene, default_track
from .track import TrackSpec


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "MISSING"


def run_doctor(
    *,
    policy_dir: Path,
    robot_scene: Path,
    track_path: Path,
    allow_missing_policy: bool = False,
) -> None:
    print(f"platform={platform.system()} {platform.release()} architecture={platform.machine()}")
    print(f"python={platform.python_version()}")
    for package in ("numpy", "PyYAML", "mujoco", "onnxruntime", "mjviser", "viser"):
        print(f"package.{package}={_version(package)}")

    missing = [path for path in (robot_scene, track_path) if not path.is_file()]
    policy_files = (policy_dir / "deploy.json", policy_dir / "policy.onnx")
    missing_policy = [path for path in policy_files if not path.is_file()]
    if missing or (missing_policy and not allow_missing_policy):
        formatted = "\n".join(f"  - {path}" for path in (*missing, *missing_policy))
        raise FileNotFoundError(f"Required runtime files are missing:\n{formatted}")

    mujoco = _import_mujoco()
    track = TrackComposer().compile(TrackSpec.load(track_path))
    if missing_policy:
        print("policy=SKIPPED (copy deploy.json and policy.onnx into policies/him_policy)")
        compile_mujoco_track(robot_scene, track, mujoco_module=mujoco)
        print("DOCTOR_OK scope=scene_only")
        return

    config = DeploymentConfig.load(policy_dir / "deploy.json")
    model = compile_mujoco_track(
        robot_scene,
        track,
        timestep=config.physics_dt or 0.005,
        gyro_sensor=config.imu_gyro_sensor,
        quaternion_sensor=config.imu_quaternion_sensor,
        mujoco_module=mujoco,
    )
    runtime = MujocoRuntime.from_model(policy_dir, model, mujoco_module=mujoco)
    runtime.set_command((0.0, 0.0, 0.0))
    start_patch = track.patches[0]
    runtime.place_base_above_surface(
        (start_patch.start_x + min(0.75, (start_patch.end_x - start_patch.start_x) / 2.0), 0.0),
        surface_z=start_patch.start_z,
    )
    runtime.step_control()
    print(
        f"DOCTOR_OK scope=full policy_route={config.route} "
        f"observation={config.frame_dim * config.history_steps} action={config.policy_output_dim}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, default=default_policy_dir())
    parser.add_argument("--robot-scene", type=Path, default=default_robot_scene())
    parser.add_argument("--track", type=Path, default=default_track())
    parser.add_argument("--allow-missing-policy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_doctor(
        policy_dir=args.policy_dir.resolve(),
        robot_scene=args.robot_scene.resolve(),
        track_path=args.track.resolve(),
        allow_missing_policy=args.allow_missing_policy,
    )


if __name__ == "__main__":
    main()
