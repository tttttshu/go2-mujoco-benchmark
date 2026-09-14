"""View a generated Go2 policy course with a native or browser-based viewer."""

from __future__ import annotations

import argparse
import math
import time
import webbrowser
from pathlib import Path

import numpy as np

from .composer import TrackComposer
from .mujoco_scene import _import_mujoco, compile_mujoco_track
from .paths import default_policy_dir, default_robot_scene, default_track
from .rollout import _deployment_api
from .track import TrackSpec


def _camera_offset(distance: float, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Return a world-frame orbit offset for the browser camera."""
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    horizontal = distance * math.cos(elevation)
    return np.asarray(
        [
            horizontal * math.cos(azimuth),
            horizontal * math.sin(azimuth),
            distance * math.sin(elevation),
        ],
        dtype=np.float64,
    )


def _outside_track(qpos: np.ndarray, track, *, margin: float = 0.25) -> bool:
    """Return whether the floating base has left the finite course volume."""
    base = np.asarray(qpos[:3], dtype=np.float64)
    if not np.all(np.isfinite(base)):
        return True
    min_surface_z = min(
        track.start_z,
        track.end_z,
        *(min(patch.start_z, patch.end_z) for patch in track.patches),
    )
    return bool(
        base[0] < -margin
        or base[0] > track.total_length + margin
        or abs(base[1]) > track.width / 2.0 + margin
        or base[2] < min_surface_z - 0.5
    )


class _BrowserCameraTracker:
    """Give every browser client a close initial view that follows the Go2 base."""

    def __init__(self, server, runtime, *, offset: np.ndarray, fov_deg: float) -> None:
        self.server = server
        self.runtime = runtime
        self.offset = np.asarray(offset, dtype=np.float64)
        self.fov = math.radians(fov_deg)
        self.enabled = True
        self._last_target: dict[int, np.ndarray] = {}
        server.on_client_connect(self._on_connect)
        server.on_client_disconnect(self._on_disconnect)

    def _target(self) -> np.ndarray:
        return np.asarray(self.runtime.data.qpos[:3], dtype=np.float64).copy()

    def _set_close_view(self, client) -> None:
        target = self._target()
        camera = client.camera
        camera.position = target + self.offset
        camera.look_at = target
        camera.up_direction = (0.0, 0.0, 1.0)
        camera.fov = self.fov
        camera.min_orbit_distance = 0.15
        camera.max_orbit_distance = 12.0
        self._last_target[id(client)] = target

    def _on_connect(self, client) -> None:
        self._set_close_view(client)

    def _on_disconnect(self, client) -> None:
        self._last_target.pop(id(client), None)

    def update(self) -> None:
        """Translate each camera with the robot while preserving user orbit/zoom."""
        target = self._target()
        for client in self.server.get_clients().values():
            key = id(client)
            previous = self._last_target.get(key)
            if previous is None:
                self._set_close_view(client)
                continue
            if self.enabled:
                delta = target - previous
                if np.any(np.abs(delta) > 1e-9):
                    # Viser's position setter translates look_at by the same offset,
                    # retaining any orbit or zoom adjustment made by the user.
                    client.camera.position = np.asarray(client.camera.position) + delta
            self._last_target[key] = target.copy()

    def reset_all(self) -> None:
        for client in self.server.get_clients().values():
            self._set_close_view(client)


def _create_runtime(args: argparse.Namespace):
    DeploymentConfig, MujocoRuntime = _deployment_api()
    mujoco = _import_mujoco()
    config = DeploymentConfig.load(args.policy_dir / "deploy.json")
    track = TrackComposer().compile(TrackSpec.load(args.track))
    model = compile_mujoco_track(
        args.robot_scene,
        track,
        timestep=config.physics_dt or 0.005,
        gyro_sensor=config.imu_gyro_sensor,
        quaternion_sensor=config.imu_quaternion_sensor,
        mujoco_module=mujoco,
    )
    runtime = MujocoRuntime.from_model(args.policy_dir, model, mujoco_module=mujoco)
    runtime.set_command((args.vx, args.vy, args.wz))

    def reset(model_arg=None, data_arg=None) -> None:
        runtime.reset(model_arg, data_arg)
        start_patch = track.patches[0]
        runtime.data.qpos[0] = start_patch.start_x + min(
            0.75,
            (start_patch.end_x - start_patch.start_x) / 2.0,
        )
        runtime.data.qpos[1] = 0.0
        mujoco.mj_forward(model, runtime.data)

    reset()
    return mujoco, runtime, config, reset, track


def _add_command_controls(server, runtime, config, camera_tracker=None) -> None:
    command = runtime.get_command()
    with server.gui.add_folder("Go2 Command"):
        sliders = []
        for label, index in (("vx (m/s)", 0), ("vy (m/s)", 1), ("wz (rad/s)", 2)):
            slider = server.gui.add_slider(
                label,
                min=float(config.command_ranges[index, 0]),
                max=float(config.command_ranges[index, 1]),
                step=0.01,
                initial_value=float(command[index]),
            )
            sliders.append(slider)

        def update(_) -> None:
            runtime.set_command(np.asarray([slider.value for slider in sliders], dtype=np.float32))

        for slider in sliders:
            slider.on_update(update)

        if camera_tracker is not None:
            follow = server.gui.add_checkbox("Follow robot", initial_value=True)

            def update_follow(_) -> None:
                camera_tracker.enabled = bool(follow.value)

            follow.on_update(update_follow)
            reset_camera = server.gui.add_button("Reset camera")
            reset_camera.on_click(lambda _: camera_tracker.reset_all())


def _run_mjviser(args: argparse.Namespace, runtime, config, reset, track) -> None:
    try:
        import viser
        from mjviser import Viewer
    except ImportError as exc:
        raise RuntimeError("Install mjviser in the project viewer environment") from exc

    server = viser.ViserServer(host=args.host, port=args.port)
    camera_tracker = _BrowserCameraTracker(
        server,
        runtime,
        offset=_camera_offset(args.camera_distance, args.camera_azimuth, args.camera_elevation),
        fov_deg=args.camera_fov,
    )
    camera_update_stride = max(1, round(0.05 / runtime.model.opt.timestep))
    physics_steps = 0

    def step_with_camera(model, data) -> None:
        nonlocal physics_steps
        runtime.step_physics(model, data)
        physics_steps += 1
        if args.auto_reset and _outside_track(runtime.data.qpos, track):
            reset(model, data)
            camera_tracker.reset_all()
            physics_steps = 0
            return
        if physics_steps % camera_update_stride == 0:
            camera_tracker.update()

    viewer = Viewer(
        runtime.model,
        runtime.data,
        step_fn=step_with_camera,
        reset_fn=reset,
        server=server,
    )
    _add_command_controls(server, runtime, config, camera_tracker)
    print(f"VIEWER_READY backend=mjviser url=http://{args.host}:{args.port}", flush=True)
    if args.open_browser:
        webbrowser.open(f"http://{args.host}:{args.port}")
    viewer.run()


def _run_native(args: argparse.Namespace, mujoco, runtime) -> None:
    import mujoco.viewer

    start_wall_time = time.perf_counter()
    with mujoco.viewer.launch_passive(runtime.model, runtime.data) as viewer:
        base_id = mujoco.mj_name2id(runtime.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = base_id
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -20.0
        print("VIEWER_READY backend=native", flush=True)
        while viewer.is_running():
            step_start = time.perf_counter()
            runtime.step_control()
            viewer.sync()
            if args.duration > 0.0 and time.perf_counter() - start_wall_time >= args.duration:
                break
            remaining = runtime.config.control_dt - (time.perf_counter() - step_start)
            if remaining > 0.0:
                time.sleep(remaining)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, default=default_policy_dir())
    parser.add_argument("--robot-scene", type=Path, default=default_robot_scene())
    parser.add_argument("--track", type=Path, default=default_track())
    parser.add_argument("--backend", choices=("mjviser", "native"), default="mjviser")
    parser.add_argument("--vx", type=float, default=0.4)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--camera-distance", type=float, default=2.2)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=20.0)
    parser.add_argument("--camera-fov", type=float, default=45.0)
    parser.add_argument(
        "--auto-reset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset after the robot falls or leaves the finite track (default: enabled)",
    )
    parser.add_argument("--duration", type=float, default=0.0, help="Native-viewer seconds; zero runs until closed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mujoco, runtime, config, reset, track = _create_runtime(args)
    if args.backend == "mjviser":
        _run_mjviser(args, runtime, config, reset, track)
    else:
        _run_native(args, mujoco, runtime)


if __name__ == "__main__":
    main()
