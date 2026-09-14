"""NumPy/ONNX policy contract with no Isaac Lab dependency."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROPRIO_TERMS = (
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("velocity_command", 3),
    ("joint_pos_rel", 12),
    ("joint_vel_rel", 12),
    ("previous_action", 12),
)


def _validate_v2_payload(payload: dict[str, Any]) -> None:
    required = {
        "policy": {"file", "dtype", "input_dim", "output_dim"},
        "observation": {"frame_dim", "history_steps", "history_order", "terms", "quaternion_order"},
        "control": {"mode", "dt", "physics_dt", "action_clip", "action_scale", "command_ranges"},
        "robot": {"joint_names", "mujoco_joint_names", "mujoco_actuator_names", "motor_sdk_indices", "sensors"},
    }
    for section, keys in required.items():
        values = payload.get(section)
        if not isinstance(values, dict):
            raise ValueError(f"Deployment schema v2 is missing section {section!r}")
        missing = keys - values.keys()
        if missing:
            raise ValueError(f"Deployment schema v2 {section} is missing fields: {sorted(missing)}")
    policy = payload["policy"]
    observation = payload["observation"]
    control = payload["control"]
    if policy["dtype"] != "float32":
        raise ValueError(f"Unsupported policy dtype: {policy['dtype']!r}")
    if observation["history_order"] != "oldest_to_newest":
        raise ValueError(f"Unsupported observation history order: {observation['history_order']!r}")
    if tuple(tuple(term) for term in observation["terms"]) != PROPRIO_TERMS:
        raise ValueError("Deployment observation terms do not match the proprio45 contract")
    if observation["quaternion_order"] != "wxyz":
        raise ValueError(f"Unsupported quaternion order: {observation['quaternion_order']!r}")
    if control["mode"] != "joint_position_pd":
        raise ValueError(f"Unsupported control mode: {control['mode']!r}")


@dataclass(frozen=True)
class DeploymentConfig:
    route: str
    control_dt: float
    history_steps: int
    joint_names: tuple[str, ...]
    motor_sdk_indices: tuple[int, ...]
    default_joint_pos: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray
    action_scale: np.ndarray
    command_ranges: np.ndarray
    joint_pos_limits: np.ndarray
    effort_limits: np.ndarray
    velocity_limits: np.ndarray
    schema_version: int = 1
    frame_dim: int = 45
    policy_file: str = "policy.onnx"
    policy_input_dim: int | None = None
    policy_output_dim: int = 12
    physics_dt: float | None = None
    action_clip: float | None = None
    mujoco_joint_names: tuple[str, ...] = ()
    mujoco_actuator_names: tuple[str, ...] = ()
    imu_gyro_sensor: str = "imu_gyro"
    imu_quaternion_sensor: str = "imu_quat"

    def __post_init__(self) -> None:
        if not self.mujoco_joint_names:
            object.__setattr__(self, "mujoco_joint_names", tuple(f"{name}_joint" for name in self.joint_names))
        if not self.mujoco_actuator_names:
            object.__setattr__(self, "mujoco_actuator_names", tuple(self.joint_names))

    @classmethod
    def load(cls, path: str | Path) -> DeploymentConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        schema_version = payload.get("schema_version")
        if schema_version not in (1, 2):
            raise ValueError(f"Unsupported deployment schema: {schema_version}")
        if schema_version == 2:
            _validate_v2_payload(payload)
        observation = payload["observation"]
        control = payload["control"]
        robot = payload["robot"]
        policy = payload.get("policy", {})
        sensors = robot.get("sensors", {})
        joint_names = tuple(robot["joint_names"])
        config = cls(
            route=payload["route"],
            control_dt=float(control["dt"]),
            history_steps=int(observation["history_steps"]),
            joint_names=joint_names,
            motor_sdk_indices=tuple(robot["motor_sdk_indices"]),
            default_joint_pos=np.asarray(robot["default_joint_pos"], dtype=np.float32),
            stiffness=np.asarray(robot["stiffness"], dtype=np.float32),
            damping=np.asarray(robot["damping"], dtype=np.float32),
            action_scale=np.asarray(control["action_scale"], dtype=np.float32),
            command_ranges=np.asarray(control["command_ranges"], dtype=np.float32),
            joint_pos_limits=np.asarray(robot["joint_pos_limits"], dtype=np.float32),
            effort_limits=np.asarray(robot["effort_limits"], dtype=np.float32),
            velocity_limits=np.asarray(robot["velocity_limits"], dtype=np.float32),
            schema_version=int(schema_version),
            frame_dim=int(observation.get("frame_dim", 45)),
            policy_file=str(policy.get("file", payload.get("policy_file", "policy.onnx"))),
            policy_input_dim=int(policy["input_dim"]) if policy.get("input_dim") is not None else None,
            policy_output_dim=int(policy.get("output_dim", 12)),
            physics_dt=float(control["physics_dt"]) if control.get("physics_dt") is not None else None,
            action_clip=float(control["action_clip"]) if control.get("action_clip") is not None else None,
            mujoco_joint_names=tuple(robot.get("mujoco_joint_names", (f"{name}_joint" for name in joint_names))),
            mujoco_actuator_names=tuple(robot.get("mujoco_actuator_names", joint_names)),
            imu_gyro_sensor=str(sensors.get("gyro", "imu_gyro")),
            imu_quaternion_sensor=str(sensors.get("quaternion", "imu_quat")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.route not in {"competition", "him"}:
            raise ValueError(f"Unsupported deployment route: {self.route!r}")
        expected_history_steps = {"competition": 1, "him": 6}[self.route]
        if self.history_steps != expected_history_steps:
            raise ValueError(
                f"Route {self.route!r} requires {expected_history_steps} history steps, got {self.history_steps}"
            )
        if len(self.joint_names) != 12 or len(self.motor_sdk_indices) != 12:
            raise ValueError("Go2 deployment requires exactly 12 ordered joints")
        if len(self.mujoco_joint_names) != 12 or len(self.mujoco_actuator_names) != 12:
            raise ValueError("MuJoCo joint and actuator mappings must contain exactly 12 names")
        if len(set(self.mujoco_joint_names)) != 12 or len(set(self.mujoco_actuator_names)) != 12:
            raise ValueError("MuJoCo joint and actuator mappings must be unique")
        if self.history_steps < 1 or self.frame_dim != 45 or self.control_dt <= 0.0:
            raise ValueError("Invalid history length or control period")
        expected_input_dim = self.frame_dim * self.history_steps
        if self.policy_input_dim is not None and self.policy_input_dim != expected_input_dim:
            raise ValueError(
                f"Policy input_dim={self.policy_input_dim} does not match frame_dim*history_steps={expected_input_dim}"
            )
        if self.policy_output_dim != 12:
            raise ValueError(f"Go2 policy output must contain 12 actions, got {self.policy_output_dim}")
        if self.physics_dt is not None and self.physics_dt <= 0.0:
            raise ValueError("Physics timestep must be positive")
        if self.action_clip is not None and (not np.isfinite(self.action_clip) or self.action_clip <= 0.0):
            raise ValueError("Action clipping limit must be finite and positive")
        if not self.policy_file or Path(self.policy_file).name != self.policy_file:
            raise ValueError("Policy file must be a plain filename inside the deployment bundle")
        for name, value, shape in (
            ("default_joint_pos", self.default_joint_pos, (12,)),
            ("stiffness", self.stiffness, (12,)),
            ("damping", self.damping, (12,)),
            ("action_scale", self.action_scale, (12,)),
            ("command_ranges", self.command_ranges, (3, 2)),
            ("joint_pos_limits", self.joint_pos_limits, (12, 2)),
            ("effort_limits", self.effort_limits, (12,)),
            ("velocity_limits", self.velocity_limits, (12,)),
        ):
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid {name}: expected finite shape {shape}, got {value.shape}")

    def validate_command(self, command: np.ndarray) -> np.ndarray:
        command = np.asarray(command, dtype=np.float32).reshape(3)
        if np.any(command < self.command_ranges[:, 0]) or np.any(command > self.command_ranges[:, 1]):
            raise ValueError(f"Command {command.tolist()} is outside exported training ranges")
        return command

    def clip_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(12)
        if not np.isfinite(action).all():
            raise ValueError("Policy action contains non-finite values")
        if self.action_clip is not None:
            action = np.clip(action, -self.action_clip, self.action_clip)
        return action

    def action_to_joint_target(self, action: np.ndarray) -> np.ndarray:
        action = self.clip_action(action)
        target = self.default_joint_pos + self.action_scale * action
        return np.clip(target, self.joint_pos_limits[:, 0], self.joint_pos_limits[:, 1])


class ObservationHistory:
    """Oldest-to-newest observation history matching the export contract."""

    def __init__(self, steps: int, frame_dim: int = 45) -> None:
        self.steps = steps
        self.frame_dim = frame_dim
        self._frames = np.zeros((steps, frame_dim), dtype=np.float32)
        self._initialized = False

    def reset(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float32).reshape(self.frame_dim)
        self._frames[:] = frame
        self._initialized = True
        return self.value

    def append(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float32).reshape(self.frame_dim)
        if not self._initialized:
            return self.reset(frame)
        self._frames[:-1] = self._frames[1:]
        self._frames[-1] = frame
        return self.value

    @property
    def value(self) -> np.ndarray:
        return self._frames.reshape(1, -1)


class OnnxPolicy:
    def __init__(self, model_path: str | Path, expected_input_dim: int | None = None, expected_output_dim: int = 12):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Install the 'runtime' extra to run ONNX policies") from exc
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Deployment policy must have one observation input and one action output")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.input_shape = inputs[0].shape
        self.output_shape = outputs[0].shape
        self.expected_input_dim = expected_input_dim
        self.expected_output_dim = expected_output_dim
        if expected_input_dim is not None and isinstance(self.input_shape[-1], int):
            if self.input_shape[-1] != expected_input_dim:
                raise ValueError(f"ONNX input width {self.input_shape[-1]} does not match {expected_input_dim}")
        if isinstance(self.output_shape[-1], int) and self.output_shape[-1] != expected_output_dim:
            raise ValueError(f"ONNX output width {self.output_shape[-1]} does not match {expected_output_dim}")

    def __call__(self, history: np.ndarray) -> np.ndarray:
        history = np.asarray(history, dtype=np.float32)
        if history.ndim != 2:
            raise ValueError(f"Policy input must have shape (batch, features), got {history.shape}")
        if self.expected_input_dim is not None and history.shape[1] != self.expected_input_dim:
            raise ValueError(f"Policy input width {history.shape[1]} does not match {self.expected_input_dim}")
        output = np.asarray(
            self.session.run([self.output_name], {self.input_name: history})[0],
            dtype=np.float32,
        )
        if output.shape != (history.shape[0], self.expected_output_dim) or not np.isfinite(output).all():
            raise RuntimeError(f"Policy produced invalid action shape/value: {output.shape}")
        return output


def projected_gravity(quaternion_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm < 1.0e-8:
        raise ValueError("Invalid zero-norm IMU quaternion")
    w, x, y, z = q / norm
    rotation = np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )
    return (rotation.T @ np.asarray((0.0, 0.0, -1.0))).astype(np.float32)


def build_proprio(
    angular_velocity: np.ndarray,
    quaternion_wxyz: np.ndarray,
    command: np.ndarray,
    joint_position: np.ndarray,
    joint_velocity: np.ndarray,
    previous_action: np.ndarray,
    default_joint_position: np.ndarray,
) -> np.ndarray:
    frame = np.concatenate(
        (
            np.asarray(angular_velocity, dtype=np.float32).reshape(3),
            projected_gravity(quaternion_wxyz),
            np.asarray(command, dtype=np.float32).reshape(3),
            np.asarray(joint_position, dtype=np.float32).reshape(12)
            - np.asarray(default_joint_position, dtype=np.float32).reshape(12),
            np.asarray(joint_velocity, dtype=np.float32).reshape(12),
            np.asarray(previous_action, dtype=np.float32).reshape(12),
        )
    ).astype(np.float32)
    if frame.shape != (45,) or not np.isfinite(frame).all():
        raise ValueError("Non-finite or malformed proprioception frame")
    return frame
