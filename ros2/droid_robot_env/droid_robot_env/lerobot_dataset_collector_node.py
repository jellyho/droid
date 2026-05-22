import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Float64, Float64MultiArray, String
from std_srvs.srv import Trigger


def _stamp_to_sec(msg) -> float:
    return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9


def _image_to_array(msg: Image) -> np.ndarray:
    dtype = np.uint8
    channels = 1
    if msg.encoding in ("rgb8", "bgr8"):
        channels = 3
    elif msg.encoding in ("rgba8", "bgra8"):
        channels = 4
    elif msg.encoding in ("mono8", "8UC1"):
        channels = 1
    elif msg.encoding == "passthrough":
        channels = max(1, int(msg.step / max(1, msg.width)))
    else:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")

    array = np.frombuffer(msg.data, dtype=dtype)
    if channels == 1:
        return array.reshape((msg.height, msg.width))
    return array.reshape((msg.height, msg.width, channels))


def _image_to_rgb(array: np.ndarray, encoding: str) -> np.ndarray:
    if array.ndim == 2:
        return array
    if encoding == "bgr8":
        return array[..., ::-1]
    if encoding == "bgra8":
        return array[..., [2, 1, 0, 3]]
    return array


@dataclass
class LatestObservation:
    joint_positions: Optional[List[float]] = None
    joint_velocities: Optional[List[float]] = None
    cartesian_position: Optional[List[float]] = None
    gripper_position: Optional[float] = None
    state_json: Optional[dict] = None
    action_info_json: Optional[dict] = None
    observation_meta_json: Optional[dict] = None
    observation_json: Optional[dict] = None
    images: Dict[str, np.ndarray] = field(default_factory=dict)
    image_encodings: Dict[str, str] = field(default_factory=dict)
    image_timestamps: Dict[str, float] = field(default_factory=dict)


class LeRobotDatasetCollectorNode(Node):
    """Collects episodes from DROID ROS2 topics and Gello button triggers."""

    def __init__(self):
        super().__init__("lerobot_dataset_collector")

        self.declare_parameter("dataset_name", "droid_lerobot_dataset")
        self.declare_parameter("dataset_root", "lerobot_datasets")
        self.declare_parameter("lerobot_repo_id", "")
        self.declare_parameter("language_instruction", "")
        self.declare_parameter("operator", "")
        self.declare_parameter("robot", "franka")
        self.declare_parameter("fps", 10.0)
        self.declare_parameter("use_lerobot_native", True)
        self.declare_parameter("use_lerobot_videos", True)
        self.declare_parameter("hf_cache_dir", "")
        self.declare_parameter("reset_service", "/droid_ros/reset")

        self.declare_parameter("joint_states_topic", "/droid_ros/joint_states")
        self.declare_parameter("cartesian_topic", "/droid_ros/cartesian_position")
        self.declare_parameter("gripper_topic", "/droid_ros/gripper_position")
        self.declare_parameter("state_json_topic", "/droid_ros/state_json")
        self.declare_parameter("action_info_json_topic", "/droid_ros/action_info_json")
        self.declare_parameter("observation_meta_json_topic", "/droid_ros/observation_meta_json")
        self.declare_parameter("observation_json_topic", "/droid_ros/observation_json")
        self.declare_parameter("camera_topics", Parameter.Type.STRING_ARRAY)
        self.declare_parameter("gello_joint_command_topic", "/gello/franka_joint_command")

        self.declare_parameter("start_topic", "/gello/start_collection")
        self.declare_parameter("success_topic", "/gello/mark_success")
        self.declare_parameter("failure_topic", "/gello/mark_failure")
        self.declare_parameter("button_trigger_value", True)
        self.declare_parameter("min_episode_steps", 1)

        self.dataset_name = self.get_parameter("dataset_name").value
        self.dataset_root = Path(self.get_parameter("dataset_root").value).expanduser()
        self.dataset_dir = self.dataset_root / self.dataset_name
        self.lerobot_root = self.dataset_dir / "lerobot"

        self.latest = LatestObservation()
        self.latest_action: Optional[List[float]] = None
        self.latest_action_stamp: Optional[float] = None
        self.recording = False
        self.episode_steps = []
        self.episode_id = self._next_episode_id()
        self.last_button_values = {}
        self._subscriptions = []
        self.lerobot_dataset = None
        self.lerobot_feature_shapes = {}

        self._prepare_dataset_dir()

        self._subscriptions.append(
            self.create_subscription(
                JointState,
                self.get_parameter("joint_states_topic").value,
                self._on_joint_state,
                50,
            )
        )
        self._subscriptions.append(
            self.create_subscription(
                Float64MultiArray,
                self.get_parameter("cartesian_topic").value,
                self._on_cartesian_position,
                50,
            )
        )
        self._subscriptions.append(
            self.create_subscription(Float64, self.get_parameter("gripper_topic").value, self._on_gripper_position, 50)
        )
        self._subscriptions.append(
            self.create_subscription(String, self.get_parameter("state_json_topic").value, self._on_state_json, 50)
        )
        self._subscriptions.append(
            self.create_subscription(
                String,
                self.get_parameter("action_info_json_topic").value,
                self._on_action_info_json,
                50,
            )
        )
        self._subscriptions.append(
            self.create_subscription(
                String,
                self.get_parameter("observation_meta_json_topic").value,
                self._on_observation_meta_json,
                50,
            )
        )
        self._subscriptions.append(
            self.create_subscription(
                String,
                self.get_parameter("observation_json_topic").value,
                self._on_observation_json,
                50,
            )
        )
        self._subscriptions.append(
            self.create_subscription(
                Float64MultiArray,
                self.get_parameter("gello_joint_command_topic").value,
                self._on_gello_action,
                50,
            )
        )

        for topic in self._get_camera_topics():
            self._subscriptions.append(self.create_subscription(Image, topic, self._make_image_callback(topic), 10))

        self._subscriptions.append(
            self.create_subscription(Bool, self.get_parameter("start_topic").value, self._on_start_button, 10)
        )
        self._subscriptions.append(
            self.create_subscription(Bool, self.get_parameter("success_topic").value, self._on_success_button, 10)
        )
        self._subscriptions.append(
            self.create_subscription(Bool, self.get_parameter("failure_topic").value, self._on_failure_button, 10)
        )

        self.reset_client = self.create_client(Trigger, self.get_parameter("reset_service").value)

        fps = float(self.get_parameter("fps").value)
        self.timer = self.create_timer(1.0 / fps, self._record_tick)
        self.get_logger().info(f"LeRobot dataset collector ready: {self.dataset_dir}")

    def _prepare_dataset_dir(self):
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    def _get_camera_topics(self) -> List[str]:
        value = self.get_parameter("camera_topics").get_parameter_value()
        return list(value.string_array_value)

    def _repo_id(self) -> str:
        repo_id = self.get_parameter("lerobot_repo_id").value
        if repo_id:
            return repo_id
        return f"local/{self.dataset_name}"

    def _next_episode_id(self) -> int:
        return 0

    def _on_joint_state(self, msg: JointState):
        self.latest.joint_positions = list(msg.position)
        self.latest.joint_velocities = list(msg.velocity)

    def _on_cartesian_position(self, msg: Float64MultiArray):
        self.latest.cartesian_position = list(msg.data)

    def _on_gripper_position(self, msg: Float64):
        self.latest.gripper_position = float(msg.data)

    def _on_state_json(self, msg: String):
        try:
            self.latest.state_json = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning("Received invalid state_json; keeping previous state.")

    def _on_action_info_json(self, msg: String):
        try:
            self.latest.action_info_json = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning("Received invalid action_info_json; keeping previous action info.")

    def _on_observation_meta_json(self, msg: String):
        try:
            self.latest.observation_meta_json = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning("Received invalid observation_meta_json; keeping previous observation metadata.")

    def _on_observation_json(self, msg: String):
        try:
            self.latest.observation_json = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning("Received invalid observation_json; keeping previous full observation metadata.")

    def _on_gello_action(self, msg: Float64MultiArray):
        self.latest_action = list(msg.data)
        self.latest_action_stamp = self.get_clock().now().nanoseconds / 1e9

    def _make_image_callback(self, topic: str):
        camera_key = self._camera_key_from_topic(topic)

        def callback(msg: Image):
            try:
                self.latest.images[camera_key] = _image_to_array(msg).copy()
                self.latest.image_encodings[camera_key] = msg.encoding
                self.latest.image_timestamps[camera_key] = _stamp_to_sec(msg)
            except Exception as exc:
                self.get_logger().warning(f"Failed to decode image from {topic}: {exc}")

        return callback

    def _camera_key_from_topic(self, topic: str) -> str:
        parts = topic.strip("/").split("/")
        if "camera" in parts:
            index = parts.index("camera")
            if index + 1 < len(parts):
                return parts[index + 1]
        return topic.strip("/").replace("/", "_")

    def _on_start_button(self, msg: Bool):
        if self._is_rising_edge("start", msg.data):
            self._start_episode()

    def _on_success_button(self, msg: Bool):
        if self._is_rising_edge("success", msg.data):
            self._finish_episode(success=True)

    def _on_failure_button(self, msg: Bool):
        if self._is_rising_edge("failure", msg.data):
            self._finish_episode(success=False)

    def _is_rising_edge(self, name: str, value: bool) -> bool:
        trigger_value = bool(self.get_parameter("button_trigger_value").value)
        previous = self.last_button_values.get(name, not trigger_value)
        current = bool(value)
        self.last_button_values[name] = current
        return previous != trigger_value and current == trigger_value

    def _start_episode(self):
        if self.recording:
            self.get_logger().warning("Start trigger ignored; already recording.")
            return
        self.recording = True
        self.episode_steps = []
        self.get_logger().info(f"Started episode {self.episode_id}.")

    def _finish_episode(self, success: bool):
        if not self.recording:
            self.get_logger().warning("Finish trigger ignored; no active episode.")
            return

        min_steps = int(self.get_parameter("min_episode_steps").value)
        if len(self.episode_steps) < min_steps:
            self.get_logger().warning(
                f"Episode {self.episode_id} has {len(self.episode_steps)} steps; dropping because min_episode_steps={min_steps}."
            )
            self.recording = False
            self.episode_steps = []
            self._request_reset()
            return

        self.recording = False
        episode_id = self.episode_id
        self._save_episode(episode_id, success)
        self.episode_id += 1
        self.episode_steps = []
        self._request_reset()

    def _record_tick(self):
        if not self.recording:
            return
        if self.latest_action is None:
            self.get_logger().warning("Recording tick skipped; no Gello joint command received yet.")
            return
        if self.latest.joint_positions is None:
            self.get_logger().warning("Recording tick skipped; no robot joint state received yet.")
            return

        now_sec = self.get_clock().now().nanoseconds / 1e9
        step = {
            "timestamp": now_sec,
            "action": list(self.latest_action),
            "action_timestamp": self.latest_action_stamp,
            "joint_positions": list(self.latest.joint_positions),
            "joint_velocities": list(self.latest.joint_velocities or []),
            "cartesian_position": list(self.latest.cartesian_position or []),
            "gripper_position": self.latest.gripper_position,
            "state_json": self.latest.state_json or {},
            "action_info_json": self.latest.action_info_json or {},
            "observation_meta_json": self.latest.observation_meta_json or {},
            "observation_json": self.latest.observation_json or {},
            "images": {key: value.copy() for key, value in self.latest.images.items()},
            "image_encodings": dict(self.latest.image_encodings),
            "image_timestamps": dict(self.latest.image_timestamps),
        }
        self.episode_steps.append(step)

    def _save_episode(self, episode_id: int, success: bool):
        self._save_lerobot_episode(success)
        label = "success" if success else "failure"
        self.get_logger().info(f"Saved {label} LeRobot episode {episode_id} with {len(self.episode_steps)} steps.")

    def _save_lerobot_episode(self, success: bool):
        if not bool(self.get_parameter("use_lerobot_native").value):
            return

        rewards = np.zeros(len(self.episode_steps), dtype=np.float32)
        if success and len(rewards) > 0:
            rewards[-1] = 1.0
        dones = np.zeros(len(self.episode_steps), dtype=np.float32)
        if len(dones) > 0:
            dones[-1] = 1.0

        if self.lerobot_dataset is None:
            self.lerobot_dataset = self._create_lerobot_dataset()

        for step_idx, step in enumerate(self.episode_steps):
            frame = {
                "observation.state": self._state_vector(step),
                "action": np.asarray(step["action"], dtype=np.float32),
                "action.timestamp": np.asarray(
                    [np.nan if step["action_timestamp"] is None else float(step["action_timestamp"])],
                    dtype=np.float32,
                ),
                "action.info_json": self._json_string(step["action_info_json"]),
                "reward": np.asarray([rewards[step_idx]], dtype=np.float32),
                "done": np.asarray([dones[step_idx]], dtype=np.float32),
                "success": np.asarray([1.0 if success else 0.0], dtype=np.float32),
                "failure": np.asarray([0.0 if success else 1.0], dtype=np.float32),
                "task": self.get_parameter("language_instruction").value,
                "collection.timestamp": np.asarray([float(step["timestamp"])], dtype=np.float32),
                "observation.robot_state_json": self._json_string(step["state_json"]),
                "observation.full_json": self._json_string(step["observation_json"]),
                "observation.metadata_json": self._json_string(step["observation_meta_json"]),
                "observation.image_encodings_json": self._json_string(step["image_encodings"]),
                "observation.image_timestamps_json": self._json_string(step["image_timestamps"]),
            }
            for feature_key, shape in self.lerobot_feature_shapes.items():
                if feature_key in frame:
                    continue
                values = self._extra_numeric_values(step).get(feature_key)
                if values is None:
                    values = np.zeros(shape, dtype=np.float32)
                frame[feature_key] = values
            for camera_key, image in step["images"].items():
                frame[f"observation.images.{camera_key}"] = self._image_for_lerobot(
                    image,
                    step["image_encodings"].get(camera_key, "passthrough"),
                )
            self.lerobot_dataset.add_frame(frame)

        self.lerobot_dataset.save_episode()
        finalize = getattr(self.lerobot_dataset, "finalize", None)
        if finalize is not None:
            finalize()

    def _create_lerobot_dataset(self):
        hf_cache_dir = self.get_parameter("hf_cache_dir").value
        if hf_cache_dir:
            cache_path = Path(hf_cache_dir).expanduser()
            cache_path.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", str(cache_path))
            os.environ.setdefault("HF_DATASETS_CACHE", str(cache_path / "datasets"))

        try:
            from lerobot.datasets import LeRobotDataset
        except ImportError:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

        if self.lerobot_root.exists():
            dataset = LeRobotDataset(self._repo_id(), root=self.lerobot_root)
            self._set_lerobot_feature_shapes(dataset.meta.features)
            return dataset

        sample_step = self.episode_steps[0]
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": self._state_vector(sample_step).shape,
                "names": self._state_names(sample_step),
            },
            "action": {
                "dtype": "float32",
                "shape": np.asarray(sample_step["action"], dtype=np.float32).shape,
                "names": [f"joint_command_{idx}" for idx in range(len(sample_step["action"]))],
            },
            "action.timestamp": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["action_timestamp"],
            },
            "action.info_json": {
                "dtype": "string",
                "shape": (1,),
            },
            "collection.timestamp": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["timestamp"],
            },
            "reward": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["reward"],
            },
            "done": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["done"],
            },
            "success": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["success"],
            },
            "failure": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["failure"],
            },
            "observation.robot_state_json": {
                "dtype": "string",
                "shape": (1,),
            },
            "observation.full_json": {
                "dtype": "string",
                "shape": (1,),
            },
            "observation.metadata_json": {
                "dtype": "string",
                "shape": (1,),
            },
            "observation.image_encodings_json": {
                "dtype": "string",
                "shape": (1,),
            },
            "observation.image_timestamps_json": {
                "dtype": "string",
                "shape": (1,),
            },
        }

        features.update(self._extra_numeric_features(sample_step))

        for camera_key, image in sample_step["images"].items():
            image_array = self._image_for_lerobot(image, sample_step["image_encodings"].get(camera_key, "passthrough"))
            features[f"observation.images.{camera_key}"] = {
                "dtype": "video",
                "shape": image_array.shape,
            }

        self._set_lerobot_feature_shapes(features)
        return LeRobotDataset.create(
            repo_id=self._repo_id(),
            fps=int(round(float(self.get_parameter("fps").value))),
            features=features,
            robot_type=self.get_parameter("robot").value,
            root=self.lerobot_root,
            use_videos=bool(self.get_parameter("use_lerobot_videos").value),
        )

    def _state_vector(self, step) -> np.ndarray:
        values = []
        values.extend(step["joint_positions"])
        values.extend(step["joint_velocities"])
        values.extend(step["cartesian_position"])
        if step["gripper_position"] is not None:
            values.append(float(step["gripper_position"]))
        return np.asarray(values, dtype=np.float32)

    def _state_names(self, step) -> List[str]:
        names = []
        names.extend([f"joint_position_{idx}" for idx in range(len(step["joint_positions"]))])
        names.extend([f"joint_velocity_{idx}" for idx in range(len(step["joint_velocities"]))])
        names.extend([f"cartesian_position_{idx}" for idx in range(len(step["cartesian_position"]))])
        if step["gripper_position"] is not None:
            names.append("gripper_position")
        return names

    def _set_lerobot_feature_shapes(self, features):
        self.lerobot_feature_shapes = {}
        for key, spec in features.items():
            if key.startswith("observation.images."):
                continue
            if spec.get("dtype") == "float32" and "shape" in spec:
                self.lerobot_feature_shapes[key] = tuple(spec["shape"])

    def _json_string(self, data) -> str:
        return json.dumps(data, default=self._json_default)

    def _json_default(self, value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    def _extra_numeric_features(self, step):
        features = {}
        for key, value in self._extra_numeric_values(step).items():
            spec = {
                "dtype": "float32",
                "shape": value.shape,
            }
            if value.ndim == 1 and value.shape[0] == 1:
                spec["names"] = [key.split(".")[-1]]
            features[key] = spec
        return features

    def _extra_numeric_values(self, step):
        values = {}
        values.update(self._flatten_numeric_features("observation.robot_state", step["state_json"]))
        values.update(self._flatten_numeric_features("observation.camera_intrinsics", step["observation_meta_json"].get("camera_intrinsics", {})))
        values.update(self._flatten_numeric_features("observation.camera_extrinsics", step["observation_meta_json"].get("camera_extrinsics", {})))
        values.update(self._flatten_numeric_features("observation.camera_type", step["observation_meta_json"].get("camera_type", {})))
        values.update(self._flatten_numeric_features("observation.timestamp.robot_state", step["observation_meta_json"].get("timestamp", {}).get("robot_state", {})))
        values.update(self._flatten_numeric_features("observation.timestamp.cameras", step["observation_meta_json"].get("timestamp", {}).get("cameras", {})))
        values.update(self._flatten_numeric_features("observation.image_timestamps", step["image_timestamps"]))
        return values

    def _flatten_numeric_features(self, prefix, data):
        output = {}
        if not isinstance(data, dict):
            return output
        for key, value in data.items():
            feature_key = f"{prefix}.{self._sanitize_feature_name(key)}"
            if isinstance(value, dict):
                output.update(self._flatten_numeric_features(feature_key, value))
                continue
            array = self._as_float_array(value)
            if array is not None:
                output[feature_key] = array
        return output

    def _as_float_array(self, value):
        if isinstance(value, bool):
            return np.asarray([1.0 if value else 0.0], dtype=np.float32)
        if isinstance(value, (int, float, np.number)):
            return np.asarray([value], dtype=np.float32)
        try:
            array = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError):
            return None
        if array.dtype == np.dtype("O") or array.size == 0:
            return None
        return array

    def _sanitize_feature_name(self, value: str) -> str:
        return "".join(char if char.isalnum() or char == "_" else "_" for char in str(value))

    def _image_for_lerobot(self, image: np.ndarray, encoding: str) -> np.ndarray:
        image = _image_to_rgb(image, encoding)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        if image.shape[-1] == 4:
            image = image[..., :3]
        return np.ascontiguousarray(image, dtype=np.uint8)

    def _request_reset(self):
        if not self.reset_client.service_is_ready():
            self.get_logger().warning("Reset service is not ready; skipping robot reset request.")
            return
        future = self.reset_client.call_async(Trigger.Request())
        future.add_done_callback(self._on_reset_done)

    def _on_reset_done(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info("Robot reset requested after episode finish.")
            else:
                self.get_logger().warning(f"Robot reset request failed: {response.message}")
        except Exception as exc:
            self.get_logger().warning(f"Robot reset service call failed: {exc}")


def main(args=None):
    rclpy.init(args=args)
    node = LeRobotDatasetCollectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
