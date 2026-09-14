"""View a generated Go2 policy course with a native or browser-based viewer."""

from __future__ import annotations

import argparse
import math
import time
import webbrowser
from dataclasses import replace
from pathlib import Path

import numpy as np

from .composer import TrackComposer
from .mujoco_scene import DEPTH_CAMERA_NAME, _import_mujoco, compile_mujoco_track
from .paths import default_policy_dir, default_robot_scene, default_track
from .rollout import _body_linear_velocity, _deployment_api, _roll_pitch
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


def _outside_track_reason(qpos: np.ndarray, track, *, margin: float = 0.25) -> str | None:
    """Return a concise reset reason when the floating base leaves the course."""
    base = np.asarray(qpos[:3], dtype=np.float64)
    if not np.all(np.isfinite(base)):
        return "non-finite state"
    min_surface_z = min(
        track.start_z,
        track.end_z,
        *(min(patch.start_z, patch.end_z) for patch in track.patches),
    )
    if base[2] < min_surface_z - 0.5:
        return "fall below course"
    if base[0] < -margin:
        return "left course start"
    if base[0] > track.total_length + margin:
        return "reached course end"
    if abs(base[1]) > track.width / 2.0 + margin:
        return "left course side"
    return None


def _outside_track(qpos: np.ndarray, track, *, margin: float = 0.25) -> bool:
    """Return whether the floating base has left the finite course volume."""
    return _outside_track_reason(qpos, track, margin=margin) is not None


def _track_patch_at(track, x_position: float):
    for index, patch in enumerate(track.patches):
        if patch.start_x <= x_position < patch.end_x:
            return index, patch
    if x_position < track.patches[0].start_x:
        return 0, track.patches[0]
    return len(track.patches) - 1, track.patches[-1]


def _telemetry_snapshot(runtime, track) -> dict[str, object]:
    """Collect viewer telemetry without depending on GUI objects."""
    command = np.asarray(runtime.get_command(), dtype=np.float64)
    position = np.asarray(runtime.data.qpos[:3], dtype=np.float64)
    quaternion = np.asarray(runtime.data.qpos[3:7], dtype=np.float64)
    body_velocity = _body_linear_velocity(quaternion, runtime.data.qvel[:3])
    roll, pitch = _roll_pitch(quaternion)
    patch_index, patch = _track_patch_at(track, float(position[0]))
    progress = float(np.clip(position[0] / track.total_length, 0.0, 1.0))
    return {
        "command": command,
        "body_velocity": body_velocity,
        "yaw_velocity": float(runtime.data.qvel[5]),
        "position": position,
        "roll_deg": math.degrees(roll),
        "pitch_deg": math.degrees(pitch),
        "progress": progress,
        "patch_index": patch_index,
        "patch_kind": patch.kind,
        "sim_time": float(runtime.data.time),
    }


def _colorize_depth(depth: np.ndarray, *, near: float = 0.15, far: float = 4.0) -> np.ndarray:
    """Convert metric depth into a compact near-warm/far-cool RGB image."""
    if not 0.0 <= near < far:
        raise ValueError("Depth range must satisfy 0 <= near < far")
    values = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(values) & (values > 0.0)
    safe_values = np.where(valid, values, far)
    normalized = np.clip((safe_values - near) / (far - near), 0.0, 1.0)
    red = 255.0 * (1.0 - normalized)
    green = 255.0 * (1.0 - np.abs(2.0 * normalized - 1.0))
    blue = 255.0 * normalized
    image = np.stack((red, green, blue), axis=-1).astype(np.uint8)
    image[~valid] = 0
    return image


class _DepthCameraStream:
    def __init__(
        self,
        runtime,
        *,
        camera_name: str = DEPTH_CAMERA_NAME,
        width: int = 320,
        height: int = 180,
        near: float = 0.15,
        far: float = 4.0,
    ) -> None:
        self.runtime = runtime
        self.camera_name = camera_name
        self.near = near
        self.far = far
        self.renderer = runtime.mujoco.Renderer(runtime.model, height=height, width=width)
        self.renderer.enable_depth_rendering()
        self.image_handle = None

    def capture(self) -> np.ndarray:
        self.renderer.update_scene(self.runtime.data, camera=self.camera_name)
        return _colorize_depth(self.renderer.render(), near=self.near, far=self.far)

    def bind(self, image_handle) -> None:
        self.image_handle = image_handle

    def update(self) -> None:
        if self.image_handle is not None:
            self.image_handle.image = self.capture()

    def close(self) -> None:
        self.renderer.close()


class _StallDetector:
    def __init__(self, *, window_seconds: float = 2.5, minimum_progress: float = 0.08) -> None:
        self.window_seconds = window_seconds
        self.minimum_progress = minimum_progress
        self._time = 0.0
        self._x = 0.0
        self._direction = 0.0

    def update(self, snapshot: dict[str, object]) -> bool:
        command_x = float(np.asarray(snapshot["command"])[0])
        direction = float(np.sign(command_x)) if abs(command_x) >= 0.25 else 0.0
        now = float(snapshot["sim_time"])
        x_position = float(np.asarray(snapshot["position"])[0])
        if direction == 0.0 or direction != self._direction or now < self._time:
            self._time = now
            self._x = x_position
            self._direction = direction
            return False
        elapsed = now - self._time
        progress = direction * (x_position - self._x)
        if progress >= self.minimum_progress:
            self._time = now
            self._x = x_position
            return False
        return elapsed >= self.window_seconds


def _telemetry_markdown(
    snapshot: dict[str, object],
    *,
    reset_count: int,
    reset_reason: str,
    stalled: bool = False,
) -> str:
    command = np.asarray(snapshot["command"])
    velocity = np.asarray(snapshot["body_velocity"])
    position = np.asarray(snapshot["position"])
    return (
        (
            "**⚠ Motion appears stalled — use Reverse briefly or Reset & align.**\n\n"
            if stalled
            else "**● Motion active**\n\n"
        )
        + "| Motion | Target | Actual |\n"
        "|:--|--:|--:|\n"
        f"| Forward | {command[0]:+.2f} m/s | {velocity[0]:+.2f} m/s |\n"
        f"| Lateral | {command[1]:+.2f} m/s | {velocity[1]:+.2f} m/s |\n"
        f"| Yaw | {command[2]:+.2f} rad/s | {float(snapshot['yaw_velocity']):+.2f} rad/s |\n\n"
        f"**Terrain:** `{int(snapshot['patch_index']) + 1}` · `{snapshot['patch_kind']}`  \n"
        f"**Position:** x `{position[0]:.2f}` · y `{position[1]:+.2f}` · z `{position[2]:.2f}` m  \n"
        f"**Attitude:** roll `{float(snapshot['roll_deg']):+.1f}°` · "
        f"pitch `{float(snapshot['pitch_deg']):+.1f}°`  \n"
        f"**Simulation:** `{float(snapshot['sim_time']):.1f} s` · resets `{reset_count}`  \n"
        f"**Last reset:** `{reset_reason}`"
    )


class _ViewerDashboard:
    def __init__(self, runtime, track, progress_handle, markdown_handle) -> None:
        self.runtime = runtime
        self.track = track
        self.progress_handle = progress_handle
        self.markdown_handle = markdown_handle
        self.reset_count = 0
        self.reset_reason = "none"
        self.stall_detector = _StallDetector()

    def update(self) -> None:
        snapshot = _telemetry_snapshot(self.runtime, self.track)
        stalled = self.stall_detector.update(snapshot)
        self.progress_handle.value = float(snapshot["progress"])
        self.markdown_handle.content = _telemetry_markdown(
            snapshot,
            reset_count=self.reset_count,
            reset_reason=self.reset_reason,
            stalled=stalled,
        )

    def record_reset(self, reason: str) -> None:
        self.reset_count += 1
        self.reset_reason = reason
        self.update()


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

    def set_view(self, *, distance: float, azimuth_deg: float, elevation_deg: float, fov_deg: float) -> None:
        self.offset = _camera_offset(distance, azimuth_deg, elevation_deg)
        self.fov = math.radians(fov_deg)
        self.reset_all()


def _create_runtime(args: argparse.Namespace):
    DeploymentConfig, MujocoRuntime = _deployment_api()
    mujoco = _import_mujoco()
    config = DeploymentConfig.load(args.policy_dir / "deploy.json")
    track_spec = TrackSpec.load(args.track)
    if args.difficulty_level is not None:
        track_spec = replace(track_spec, difficulty_level=args.difficulty_level, schema_version=2)
    track = TrackComposer().compile(track_spec)
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
        start_x = start_patch.start_x + min(
            0.75,
            (start_patch.end_x - start_patch.start_x) / 2.0,
        )
        runtime.place_base_above_surface(
            (start_x, 0.0),
            surface_z=start_patch.start_z,
            foot_clearance=getattr(args, "spawn_foot_clearance", 0.015),
        )

    reset()
    return mujoco, runtime, config, reset, track


def _add_viewer_controls(
    server,
    runtime,
    config,
    track,
    camera_tracker=None,
    reset_callback=None,
    depth_stream=None,
) -> _ViewerDashboard:
    command = runtime.get_command()
    with server.gui.add_folder("Drive", order=-100.0):
        server.gui.add_markdown("Set a target or use a quick preset. **STOP** always zeros all commands.")
        emergency_stop = server.gui.add_button("STOP", color="red", hint="Immediately set vx, vy and wz to zero")
        reset_robot = server.gui.add_button(
            "Reset & align",
            color="blue",
            hint="Return to the course start with yaw aligned to the lane",
        )
        presets = server.gui.add_button_group(
            "Quick speed",
            ("Reverse 0.5", "Stop", "Forward 0.5", "Forward 1.0"),
        )
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

        def set_command(values: tuple[float, float, float]) -> None:
            clamped = np.clip(
                np.asarray(values, dtype=np.float32),
                config.command_ranges[:, 0],
                config.command_ranges[:, 1],
            )
            for slider, value in zip(sliders, clamped, strict=True):
                slider.value = float(value)
            runtime.set_command(clamped)

        emergency_stop.on_click(lambda _: set_command((0.0, 0.0, 0.0)))
        if reset_callback is not None:
            reset_robot.on_click(lambda _: reset_callback("manual"))
        preset_commands = {
            "Reverse 0.5": (-0.5, 0.0, 0.0),
            "Stop": (0.0, 0.0, 0.0),
            "Forward 0.5": (0.5, 0.0, 0.0),
            "Forward 1.0": (1.0, 0.0, 0.0),
        }
        presets.on_click(lambda _: set_command(preset_commands[presets.value]))

    with server.gui.add_folder("Live telemetry", order=-90.0):
        progress = server.gui.add_progress_bar(0.0, color="blue")
        telemetry = server.gui.add_markdown("Starting simulation…")

    dashboard = _ViewerDashboard(runtime, track, progress, telemetry)
    dashboard.update()

    with server.gui.add_folder("Camera", order=-80.0):
        if camera_tracker is not None:
            follow = server.gui.add_checkbox("Follow robot", initial_value=True)
            camera_preset = server.gui.add_dropdown(
                "View",
                ("Close chase", "Side", "Course overview"),
                initial_value="Close chase",
            )

            def update_follow(_) -> None:
                camera_tracker.enabled = bool(follow.value)

            follow.on_update(update_follow)
            camera_views = {
                "Close chase": (1.25, 135.0, 18.0, 38.0),
                "Side": (3.0, 90.0, 12.0, 48.0),
                "Course overview": (7.5, 150.0, 38.0, 55.0),
            }

            def update_camera(_) -> None:
                distance, azimuth, elevation, fov = camera_views[camera_preset.value]
                camera_tracker.set_view(
                    distance=distance,
                    azimuth_deg=azimuth,
                    elevation_deg=elevation,
                    fov_deg=fov,
                )

            camera_preset.on_update(update_camera)
            reset_camera = server.gui.add_button("Reset camera")
            reset_camera.on_click(lambda _: camera_tracker.reset_all())

    if depth_stream is not None:
        with server.gui.add_folder("Forward depth camera", order=-75.0):
            server.gui.add_markdown("Near objects are warm; far objects are cool. Range: 0.15–4.0 m.")
            depth_image = server.gui.add_image(
                depth_stream.capture(),
                label="320 × 180 · 5 FPS",
                format="jpeg",
                jpeg_quality=80,
            )
            depth_stream.bind(depth_image)

    with server.gui.add_folder("Course", order=-70.0, expand_by_default=False):
        patch_summary = " → ".join(patch.kind for patch in track.patches)
        server.gui.add_markdown(
            f"**{track.name}**  \n"
            f"Difficulty `{track.difficulty_level}/9` · length `{track.total_length:.1f} m` · "
            f"width `{track.width:.1f} m` · "
            f"segments `{len(track.patches)}`  \n\n"
            f"`{patch_summary}`"
        )
    return dashboard


def _run_mjviser(args: argparse.Namespace, runtime, config, reset, track) -> None:
    try:
        import viser
        from mjviser import Viewer
    except ImportError as exc:
        raise RuntimeError("Install mjviser in the project viewer environment") from exc

    server = viser.ViserServer(host=args.host, port=args.port, label="Go2 MuJoCo Benchmark")
    server.gui.configure_theme(
        dark_mode=True,
        show_logo=False,
        show_share_button=False,
        brand_color=(37, 99, 235),
    )
    server.gui.main_panel.dock_right()
    server.gui.main_panel.set_width(360.0)
    dashboard_ref: list[_ViewerDashboard | None] = [None]
    camera_ref: list[_BrowserCameraTracker | None] = [None]

    def reset_and_align(reason: str, *, count: bool = True) -> None:
        reset()
        if dashboard_ref[0] is not None:
            if count:
                dashboard_ref[0].record_reset(reason)
            else:
                dashboard_ref[0].reset_reason = reason
                dashboard_ref[0].update()
        if camera_ref[0] is not None:
            camera_ref[0].reset_all()

    first_client_connected = False

    def reset_for_first_client(_) -> None:
        nonlocal first_client_connected
        if not first_client_connected:
            first_client_connected = True
            reset_and_align("browser connected", count=False)

    server.on_client_connect(reset_for_first_client)
    camera_tracker = _BrowserCameraTracker(
        server,
        runtime,
        offset=_camera_offset(args.camera_distance, args.camera_azimuth, args.camera_elevation),
        fov_deg=args.camera_fov,
    )
    camera_ref[0] = camera_tracker
    depth_stream = None
    if args.depth_camera:
        try:
            depth_stream = _DepthCameraStream(runtime)
        except Exception as exc:
            print(f"DEPTH_CAMERA_DISABLED reason={exc}", flush=True)
    camera_update_stride = max(1, round(0.05 / runtime.model.opt.timestep))
    depth_update_stride = max(1, round(0.2 / runtime.model.opt.timestep))
    physics_steps = 0
    dashboard = _add_viewer_controls(
        server,
        runtime,
        config,
        track,
        camera_tracker,
        reset_callback=reset_and_align,
        depth_stream=depth_stream,
    )
    dashboard_ref[0] = dashboard

    def step_with_camera(model, data) -> None:
        nonlocal physics_steps
        runtime.step_physics(model, data)
        physics_steps += 1
        outside_reason = _outside_track_reason(runtime.data.qpos, track)
        if args.auto_reset and outside_reason is not None:
            reset(model, data)
            dashboard.record_reset(outside_reason)
            camera_tracker.reset_all()
            physics_steps = 0
            return
        if physics_steps % camera_update_stride == 0:
            camera_tracker.update()
            dashboard.update()
        if depth_stream is not None and physics_steps % depth_update_stride == 0:
            depth_stream.update()

    def reset_from_viewer(model, data) -> None:
        reset_and_align("manual")

    viewer = Viewer(
        runtime.model,
        runtime.data,
        step_fn=step_with_camera,
        reset_fn=reset_from_viewer,
        server=server,
    )
    print(f"VIEWER_READY backend=mjviser url=http://{args.host}:{args.port}", flush=True)
    if args.open_browser:
        webbrowser.open(f"http://{args.host}:{args.port}")
    try:
        viewer.run()
    finally:
        if depth_stream is not None:
            depth_stream.close()


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
    parser.add_argument("--difficulty-level", type=int, choices=range(1, 10), default=None)
    parser.add_argument("--backend", choices=("mjviser", "native"), default="mjviser")
    parser.add_argument("--vx", type=float, default=0.4)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--camera-distance", type=float, default=1.25)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=18.0)
    parser.add_argument("--camera-fov", type=float, default=38.0)
    parser.add_argument(
        "--depth-camera",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the robot-mounted real-time depth camera in the side panel",
    )
    parser.add_argument(
        "--spawn-foot-clearance",
        type=float,
        default=0.015,
        help="Initial clearance between the lowest foot collision sphere and course surface",
    )
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
