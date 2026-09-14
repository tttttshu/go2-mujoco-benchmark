"""Shared MuJoCo policy runtime for batch evaluation and viewing."""

from __future__ import annotations

import threading
from pathlib import Path
from types import ModuleType

import numpy as np

from .runtime import DeploymentConfig, ObservationHistory, OnnxPolicy, build_proprio


class MujocoRuntime:
    def __init__(self, model, data, config: DeploymentConfig, policy: OnnxPolicy, mujoco_module: ModuleType) -> None:
        config.validate()
        self.model = model
        self.data = data
        self.config = config
        self.policy = policy
        self.mujoco = mujoco_module
        self._lock = threading.RLock()
        self.qpos_addresses: list[int] = []
        self.qvel_addresses: list[int] = []
        self.actuator_ids: list[int] = []
        for policy_name, joint_name, actuator_name in zip(
            config.joint_names,
            config.mujoco_joint_names,
            config.mujoco_actuator_names,
            strict=True,
        ):
            joint_id = mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_JOINT, joint_name)
            actuator_id = mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            if joint_id < 0 or actuator_id < 0:
                raise KeyError(
                    "Go2 joint/actuator not found in MuJoCo model: "
                    f"policy={policy_name}, joint={joint_name}, actuator={actuator_name}"
                )
            self.qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
            self.qvel_addresses.append(int(model.jnt_dofadr[joint_id]))
            self.actuator_ids.append(int(actuator_id))

        self.sim_steps_per_control = round(config.control_dt / model.opt.timestep)
        if self.sim_steps_per_control < 1 or not np.isclose(
            self.sim_steps_per_control * model.opt.timestep,
            config.control_dt,
        ):
            raise ValueError("MuJoCo timestep must divide the exported control period")
        if config.physics_dt is not None and not np.isclose(model.opt.timestep, config.physics_dt):
            raise ValueError(
                f"MuJoCo timestep {model.opt.timestep} does not match exported physics_dt {config.physics_dt}"
            )
        self.command = config.validate_command(config.command_ranges.mean(axis=1))
        self.history = ObservationHistory(config.history_steps)
        self.previous_action = np.zeros(12, dtype=np.float32)
        self._pending_action = np.zeros(12, dtype=np.float32)
        self._target = config.default_joint_pos.copy()
        self._substep = 0
        self.policy_steps = 0
        self.max_torque = 0.0
        self.reset()

    @classmethod
    def from_paths(cls, policy_dir: str | Path, model_path: str | Path):
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError("Install the 'runtime' extra to run MuJoCo") from exc
        policy_dir = Path(policy_dir)
        config = DeploymentConfig.load(policy_dir / "deploy.json")
        policy = OnnxPolicy(
            policy_dir / config.policy_file,
            expected_input_dim=config.frame_dim * config.history_steps,
            expected_output_dim=config.policy_output_dim,
        )
        model = mujoco.MjModel.from_xml_path(str(Path(model_path).expanduser().resolve()))
        return cls(model, mujoco.MjData(model), config, policy, mujoco)

    @classmethod
    def from_model(cls, policy_dir: str | Path, model, data=None, mujoco_module: ModuleType | None = None):
        if mujoco_module is None:
            try:
                import mujoco as mujoco_module
            except ImportError as exc:
                raise RuntimeError("Install the 'runtime' extra to run MuJoCo") from exc
        policy_dir = Path(policy_dir)
        config = DeploymentConfig.load(policy_dir / "deploy.json")
        policy = OnnxPolicy(
            policy_dir / config.policy_file,
            expected_input_dim=config.frame_dim * config.history_steps,
            expected_output_dim=config.policy_output_dim,
        )
        data = mujoco_module.MjData(model) if data is None else data
        return cls(model, data, config, policy, mujoco_module)

    def set_command(self, command: np.ndarray | tuple[float, float, float]) -> np.ndarray:
        with self._lock:
            self.command = self.config.validate_command(command)
            return self.command.copy()

    def get_command(self) -> np.ndarray:
        with self._lock:
            return self.command.copy()

    def reset(self, model=None, data=None) -> None:
        with self._lock:
            model = self.model if model is None else model
            data = self.data if data is None else data
            if model.nkey:
                self.mujoco.mj_resetDataKeyframe(model, data, 0)
            else:
                self.mujoco.mj_resetData(model, data)
            data.qpos[self.qpos_addresses] = self.config.default_joint_pos
            self.mujoco.mj_forward(model, data)
            self.history = ObservationHistory(self.config.history_steps)
            self.previous_action = np.zeros(12, dtype=np.float32)
            self._pending_action = np.zeros(12, dtype=np.float32)
            self._target = self.config.default_joint_pos.copy()
            self._substep = 0
            self.policy_steps = 0
            self.max_torque = 0.0

    def _sensor(self, name: str) -> np.ndarray:
        sensor_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sensor_id < 0:
            raise KeyError(f"MuJoCo sensor not found: {name}")
        start = int(self.model.sensor_adr[sensor_id])
        size = int(self.model.sensor_dim[sensor_id])
        return np.asarray(self.data.sensordata[start : start + size]).copy()

    def _prepare_action(self) -> None:
        joint_pos = np.asarray(self.data.qpos[self.qpos_addresses], dtype=np.float32)
        joint_vel = np.asarray(self.data.qvel[self.qvel_addresses], dtype=np.float32)
        frame = build_proprio(
            self._sensor(self.config.imu_gyro_sensor),
            self._sensor(self.config.imu_quaternion_sensor),
            self.command,
            joint_pos,
            joint_vel,
            self.previous_action,
            self.config.default_joint_pos,
        )
        self._pending_action = self.config.clip_action(self.policy(self.history.append(frame))[0])
        self._target = self.config.action_to_joint_target(self._pending_action)
        self.policy_steps += 1

    def step_physics(self, model=None, data=None) -> None:
        with self._lock:
            if model is not None and model is not self.model:
                raise ValueError("MujocoRuntime received an unexpected model")
            if data is not None and data is not self.data:
                raise ValueError("MujocoRuntime received an unexpected data object")
            if self._substep == 0:
                self._prepare_action()
            torque = self.config.stiffness * (
                self._target - self.data.qpos[self.qpos_addresses]
            ) - self.config.damping * self.data.qvel[self.qvel_addresses]
            torque = np.clip(torque, -self.config.effort_limits, self.config.effort_limits)
            self.data.ctrl[self.actuator_ids] = torque
            self.max_torque = max(self.max_torque, float(np.abs(torque).max()))
            self.mujoco.mj_step(self.model, self.data)
            self._substep += 1
            if self._substep >= self.sim_steps_per_control:
                self.previous_action = self._pending_action.copy()
                self._substep = 0

    def step_control(self) -> None:
        for _ in range(self.sim_steps_per_control):
            self.step_physics()

    def run_headless(self, steps: int) -> dict[str, object]:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        for _ in range(steps):
            self.step_control()
        return self.result(steps)

    def result(self, steps: int | None = None) -> dict[str, object]:
        return {
            "status": "SIM2SIM_OK",
            "policy_steps": self.policy_steps if steps is None else steps,
            "sim_time": float(self.data.time),
            "base_position": np.asarray(self.data.qpos[:3]).tolist(),
            "max_abs_torque": self.max_torque,
        }
