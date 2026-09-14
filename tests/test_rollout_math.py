from __future__ import annotations

import math

import numpy as np
import pytest

from go2_mujoco_benchmark.rollout import _body_linear_velocity, _roll_pitch


def test_identity_orientation_preserves_world_velocity():
    result = _body_linear_velocity(
        np.asarray([1.0, 0.0, 0.0, 0.0]),
        np.asarray([0.4, -0.2, 0.1]),
    )

    np.testing.assert_allclose(result, [0.4, -0.2, 0.1])


def test_body_velocity_uses_inverse_yaw_rotation():
    half_angle = math.pi / 4.0
    quaternion = np.asarray([math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)])

    result = _body_linear_velocity(quaternion, np.asarray([0.0, 1.0, 0.0]))

    np.testing.assert_allclose(result, [1.0, 0.0, 0.0], atol=1.0e-7)


def test_roll_pitch_recovers_pitch_angle():
    angle = -0.2
    quaternion = np.asarray([math.cos(angle / 2.0), 0.0, math.sin(angle / 2.0), 0.0])

    roll, pitch = _roll_pitch(quaternion)

    assert roll == pytest.approx(0.0)
    assert pitch == pytest.approx(angle)
