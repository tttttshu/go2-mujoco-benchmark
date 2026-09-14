from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from go2_mujoco_benchmark.viewer import (
    _BrowserCameraTracker,
    _StallDetector,
    _camera_offset,
    _colorize_depth,
    _outside_track,
    _outside_track_reason,
    _telemetry_markdown,
    _telemetry_snapshot,
)


class _FakeCamera:
    def __init__(self) -> None:
        self._position = np.zeros(3)
        self.look_at = np.zeros(3)
        self.up_direction = np.asarray([0.0, 0.0, 1.0])
        self.fov = 0.0
        self.min_orbit_distance = 0.01
        self.max_orbit_distance = 1e4

    @property
    def position(self) -> np.ndarray:
        return self._position

    @position.setter
    def position(self, value) -> None:
        value = np.asarray(value, dtype=np.float64)
        delta = value - self._position
        self._position = value
        self.look_at = np.asarray(self.look_at) + delta


class _FakeServer:
    def __init__(self) -> None:
        self.clients = {}
        self.connect_callback = None
        self.disconnect_callback = None

    def on_client_connect(self, callback):
        self.connect_callback = callback

    def on_client_disconnect(self, callback):
        self.disconnect_callback = callback

    def get_clients(self):
        return self.clients.copy()


def test_camera_offset_has_requested_distance() -> None:
    offset = _camera_offset(2.2, 135.0, 20.0)
    assert np.linalg.norm(offset) == pytest.approx(2.2)
    assert offset[2] > 0.0


def test_browser_camera_starts_close_and_follows_base() -> None:
    server = _FakeServer()
    runtime = SimpleNamespace(data=SimpleNamespace(qpos=np.asarray([0.75, 0.0, 0.35])))
    tracker = _BrowserCameraTracker(
        server,
        runtime,
        offset=np.asarray([-1.5, 1.5, 0.75]),
        fov_deg=45.0,
    )
    client = SimpleNamespace(camera=_FakeCamera())
    server.clients[1] = client
    server.connect_callback(client)

    initial_offset = client.camera.position - client.camera.look_at
    runtime.data.qpos[:3] += np.asarray([0.4, -0.1, 0.0])
    tracker.update()

    np.testing.assert_allclose(client.camera.position - client.camera.look_at, initial_offset)
    np.testing.assert_allclose(client.camera.look_at, runtime.data.qpos[:3])


def test_outside_track_detects_lateral_exit_and_fall() -> None:
    patch = SimpleNamespace(start_z=0.0, end_z=0.0)
    track = SimpleNamespace(
        start_z=0.0,
        end_z=0.0,
        total_length=20.0,
        width=3.0,
        patches=(patch,),
    )
    assert not _outside_track(np.asarray([1.0, 0.0, 0.35]), track)
    assert _outside_track(np.asarray([1.0, 1.8, 0.35]), track)
    assert _outside_track(np.asarray([1.0, 0.0, -0.6]), track)
    assert _outside_track(np.asarray([np.nan, 0.0, 0.35]), track)
    assert _outside_track_reason(np.asarray([1.0, 1.8, 0.35]), track) == "left course side"
    assert _outside_track_reason(np.asarray([20.3, 0.0, 0.35]), track) == "reached course end"


def test_telemetry_reports_track_progress_and_motion() -> None:
    patch_a = SimpleNamespace(start_x=0.0, end_x=2.0, start_z=0.0, end_z=0.0, kind="flat")
    patch_b = SimpleNamespace(start_x=2.0, end_x=4.0, start_z=0.0, end_z=0.2, kind="slope_up")
    track = SimpleNamespace(
        start_z=0.0,
        end_z=0.2,
        total_length=4.0,
        width=3.0,
        patches=(patch_a, patch_b),
    )
    runtime = SimpleNamespace(
        data=SimpleNamespace(
            qpos=np.asarray([3.0, 0.1, 0.4, 1.0, 0.0, 0.0, 0.0]),
            qvel=np.asarray([0.8, 0.0, 0.0, 0.0, 0.0, 0.2]),
            time=2.5,
        ),
        get_command=lambda: np.asarray([1.0, 0.0, 0.0]),
    )

    snapshot = _telemetry_snapshot(runtime, track)
    assert snapshot["progress"] == pytest.approx(0.75)
    assert snapshot["patch_index"] == 1
    assert snapshot["patch_kind"] == "slope_up"
    np.testing.assert_allclose(snapshot["body_velocity"], [0.8, 0.0, 0.0])
    text = _telemetry_markdown(snapshot, reset_count=1, reset_reason="manual")
    assert "Forward" in text
    assert "slope_up" in text
    assert "resets `1`" in text


def test_depth_colorization_maps_near_to_warm_and_far_to_cool() -> None:
    image = _colorize_depth(np.asarray([[np.nan, 0.15, 2.075, 4.0]], dtype=np.float32))
    assert image.shape == (1, 4, 3)
    assert image.dtype == np.uint8
    np.testing.assert_array_equal(image[0, 0], [0, 0, 0])
    np.testing.assert_array_equal(image[0, 1], [255, 0, 0])
    np.testing.assert_array_equal(image[0, 3], [0, 0, 255])


def test_stall_detector_flags_commanded_motion_without_progress() -> None:
    detector = _StallDetector(window_seconds=2.5, minimum_progress=0.08)
    snapshot = {
        "command": np.asarray([1.0, 0.0, 0.0]),
        "position": np.asarray([0.75, 0.0, 0.35]),
        "sim_time": 0.0,
    }
    assert not detector.update(snapshot)
    snapshot["position"] = np.asarray([0.76, 0.0, 0.35])
    snapshot["sim_time"] = 3.0
    assert detector.update(snapshot)
    snapshot["position"] = np.asarray([0.90, 0.0, 0.35])
    assert not detector.update(snapshot)
