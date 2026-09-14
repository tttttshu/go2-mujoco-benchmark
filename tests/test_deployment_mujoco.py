from __future__ import annotations

import numpy as np
import pytest

from go2_mujoco_benchmark.deployment import DeploymentConfig, MujocoRuntime

mujoco = pytest.importorskip("mujoco")


class _ConstantPolicy:
    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def __call__(self, history: np.ndarray) -> np.ndarray:
        self.calls.append(history.copy())
        return np.full((1, 12), 0.5, dtype=np.float32)


def _model():
    bodies = "".join(
        f'<body name="link{i}" pos="0 0 0.05"><joint name="j{i}_joint" axis="0 1 0" '
        f'range="-2 2"/><geom type="capsule" size="0.01 0.025" mass="0.1"/></body>'
        for i in range(12)
    )
    actuators = "".join(f'<motor name="j{i}" joint="j{i}_joint"/>' for i in range(12))
    return mujoco.MjModel.from_xml_string(
        f"""<mujoco><option timestep="0.005"/><worldbody><body name="base" pos="0 0 0.5">
        <freejoint/><geom type="box" size="0.1 0.05 0.03" mass="1"/><site name="imu"/>{bodies}
        </body></worldbody><actuator>{actuators}</actuator><sensor><gyro name="imu_gyro" site="imu"/>
        <framequat name="imu_quat" objtype="site" objname="imu"/></sensor></mujoco>"""
    )


def _config() -> DeploymentConfig:
    return DeploymentConfig(
        route="him",
        control_dt=0.02,
        history_steps=6,
        joint_names=tuple(f"j{i}" for i in range(12)),
        motor_sdk_indices=tuple(range(12)),
        default_joint_pos=np.zeros(12, dtype=np.float32),
        stiffness=np.full(12, 25.0, dtype=np.float32),
        damping=np.full(12, 0.5, dtype=np.float32),
        action_scale=np.full(12, 0.25, dtype=np.float32),
        command_ranges=np.asarray([[-1, 1], [-0.5, 0.5], [-1, 1]], dtype=np.float32),
        joint_pos_limits=np.tile([-2, 2], (12, 1)).astype(np.float32),
        effort_limits=np.full(12, 2.0, dtype=np.float32),
        velocity_limits=np.full(12, 30.0, dtype=np.float32),
    )


def test_one_control_period_is_four_physics_steps() -> None:
    model = _model()
    data = mujoco.MjData(model)
    policy = _ConstantPolicy()
    runtime = MujocoRuntime(model, data, _config(), policy, mujoco)
    runtime.set_command((0.4, 0.0, 0.2))
    runtime.step_control()
    assert runtime.sim_steps_per_control == 4
    assert runtime.policy_steps == 1
    assert len(policy.calls) == 1
    assert data.time == pytest.approx(0.02)
    np.testing.assert_allclose(policy.calls[0][0, 6:9], [0.4, 0.0, 0.2])
    assert runtime.max_torque == pytest.approx(2.0)
