import json
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import rclpy
import cv2
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, Float64MultiArray


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "lerobot_dataset_collector.yaml"


def _with_repo_params_file(args):
    argv = list(sys.argv if args is None else args)
    config_path = _default_config_path()
    if not config_path.exists():
        raise FileNotFoundError(f"Default ROS2 config not found: {config_path}")

    print(f"[lerobot_dataset_collector_node] loading config: {config_path}", flush=True)
    if "--ros-args" in argv:
        return argv + ["--params-file", str(config_path)]
    return argv + ["--ros-args", "--params-file", str(config_path)]


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


def _compressed_image_to_array(msg: CompressedImage) -> np.ndarray:
    encoded = np.frombuffer(msg.data, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("cv2.imdecode failed")
    return image


def _image_to_rgb(array: np.ndarray, encoding: str) -> np.ndarray:
    if array.ndim == 2:
        return array
    if encoding == "bgr8":
        return array[..., ::-1]
    if encoding == "bgra8":
        return array[..., [2, 1, 0, 3]]
    return array


def _state_key_from_topic(topic: str) -> str:
    parts = topic.strip("/").split("/")
    return parts[-1]


@dataclass
class LatestObservation:
    states: Dict[str, List[float]] = field(default_factory=dict)
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

        self.declare_parameter("robot_state_topics", Parameter.Type.STRING_ARRAY)
        self.declare_parameter("camera_topics", Parameter.Type.STRING_ARRAY)
        self.declare_parameter("applied_action_topic", "/droid_ros/applied_action")

        self.declare_parameter("success_topic", "/gello/mark_success")
        self.declare_parameter("failure_topic", "/gello/mark_failure")
        self.declare_parameter("button_trigger_value", True)
        self.declare_parameter("min_episode_steps", 1)
        self.declare_parameter("action_toggle_topic", "/gello/switch/record")
        self.declare_parameter("home_topic", "/droid_ros/go_home")

        self.dataset_name = self.get_parameter("dataset_name").value
        self.dataset_root = Path(self.get_parameter("dataset_root").value).expanduser()
        self.dataset_dir = self.dataset_root / self.dataset_name
        self.lerobot_root = self.dataset_dir / "lerobot"

        self.latest = LatestObservation()
        self.latest_action: Optional[List[float]] = None
        self.latest_action_stamp: Optional[float] = None
        self.recording = False
        self.saving = False
        self.episode_steps = []
        self.episode_id = self._next_episode_id()
        self._subscriptions = []
        self.lerobot_dataset = None
        self._save_thread: Optional[threading.Thread] = None

        self._prepare_dataset_dir()

        self.state_keys: List[str] = []
        for topic in self._get_robot_state_topics():
            key = _state_key_from_topic(topic)
            self.state_keys.append(key)
            self._subscriptions.append(
                self.create_subscription(
                    Float64MultiArray, topic, self._make_state_callback(key), 50
                )
            )
            self.get_logger().info(f"Subscribing to state topic: {topic} -> key={key}")

        self._subscriptions.append(
            self.create_subscription(
                Float64MultiArray,
                self.get_parameter("applied_action_topic").value,
                self._on_applied_action,
                50,
            )
        )

        for topic in self._get_camera_topics():
            if topic.endswith("/compressed") or topic.endswith("image_compressed"):
                self._subscriptions.append(
                    self.create_subscription(CompressedImage, topic, self._make_compressed_image_callback(topic), 10)
                )
            else:
                self._subscriptions.append(self.create_subscription(Image, topic, self._make_image_callback(topic), 10))

        self._subscriptions.append(
            self.create_subscription(Bool, self.get_parameter("success_topic").value, self._on_success_button, 10)
        )
        self._subscriptions.append(
            self.create_subscription(Bool, self.get_parameter("failure_topic").value, self._on_failure_button, 10)
        )

        self.action_toggle_pub = self.create_publisher(
            Bool, self.get_parameter("action_toggle_topic").value, 10
        )
        self.home_pub = self.create_publisher(
            Bool, self.get_parameter("home_topic").value, 10
        )

        fps = float(self.get_parameter("fps").value)
        self.timer = self.create_timer(1.0 / fps, self._record_tick)
        if self.episode_id > 0:
            self.get_logger().info(
                f"Continuing dataset at {self.dataset_dir} from episode {self.episode_id} "
                f"({self.episode_id} existing episodes)."
            )
        else:
            self.get_logger().info(f"New dataset at {self.dataset_dir}.")
        self.get_logger().info(f"State keys: {self.state_keys}")

    def _prepare_dataset_dir(self):
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    def _get_robot_state_topics(self) -> List[str]:
        value = self.get_parameter("robot_state_topics").get_parameter_value()
        return list(value.string_array_value)

    def _get_camera_topics(self) -> List[str]:
        value = self.get_parameter("camera_topics").get_parameter_value()
        return list(value.string_array_value)

    def _repo_id(self) -> str:
        repo_id = self.get_parameter("lerobot_repo_id").value
        if repo_id:
            return repo_id
        return f"local/{self.dataset_name}"

    def _next_episode_id(self) -> int:
        info_path = self.lerobot_root / "meta" / "info.json"
        if not info_path.exists():
            return 0
        import json as _json
        info = _json.loads(info_path.read_text())
        return int(info.get("total_episodes", 0))

    # --- State topic callbacks ---

    def _make_state_callback(self, key: str):
        def callback(msg: Float64MultiArray):
            self.latest.states[key] = list(msg.data)
        return callback

    def _on_applied_action(self, msg: Float64MultiArray):
        self.latest_action = list(msg.data)
        self.latest_action_stamp = self.get_clock().now().nanoseconds / 1e9

    # --- Image callbacks ---

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

    def _make_compressed_image_callback(self, topic: str):
        camera_key = self._camera_key_from_topic(topic)

        def callback(msg: CompressedImage):
            try:
                self.latest.images[camera_key] = _compressed_image_to_array(msg).copy()
                self.latest.image_encodings[camera_key] = "bgr8"
                self.latest.image_timestamps[camera_key] = _stamp_to_sec(msg)
            except Exception as exc:
                self.get_logger().warning(f"Failed to decode compressed image from {topic}: {exc}")

        return callback

    def _camera_key_from_topic(self, topic: str) -> str:
        parts = topic.strip("/").split("/")
        if "camera" in parts:
            index = parts.index("camera")
            if index + 1 < len(parts):
                return parts[index + 1]
        return topic.strip("/").replace("/", "_")

    # --- Button callbacks ---

    def _on_success_button(self, msg: Bool):
        trigger_value = bool(self.get_parameter("button_trigger_value").value)
        if bool(msg.data) != trigger_value:
            return
        if not self.recording:
            self._start_episode()
        else:
            self._finish_episode(success=True)

    def _on_failure_button(self, msg: Bool):
        trigger_value = bool(self.get_parameter("button_trigger_value").value)
        if bool(msg.data) != trigger_value:
            return
        self._finish_episode(success=False)

    # --- Robot control ---

    def _send_action_toggle(self):
        msg = Bool()
        msg.data = True
        self.action_toggle_pub.publish(msg)

    def _send_home(self):
        msg = Bool()
        msg.data = True
        self.home_pub.publish(msg)

    # --- Episode management ---

    def _start_episode(self):
        if self.recording:
            self.get_logger().warning("Start trigger ignored; already recording.")
            return
        self._send_action_toggle()
        self.recording = True
        self.episode_steps = []
        self.get_logger().info(f"Recording episode {self.episode_id}... (action enabled)")

    def _finish_episode(self, success: bool):
        if not self.recording:
            self.get_logger().warning("Finish trigger ignored; no active episode.")
            return
        if self.saving:
            self.get_logger().warning("Finish trigger ignored; previous episode is still saving.")
            return

        min_steps = int(self.get_parameter("min_episode_steps").value)
        if len(self.episode_steps) < min_steps:
            self.get_logger().warning(
                f"Episode {self.episode_id} has {len(self.episode_steps)} steps; "
                f"dropping because min_episode_steps={min_steps}."
            )
            self.recording = False
            self.episode_steps = []
            return

        self.recording = False
        self._send_action_toggle()
        self._send_home()
        episode_id = self.episode_id
        steps = self.episode_steps
        self.episode_steps = []
        self.episode_id += 1
        label = "success" if success else "failure"
        self.get_logger().info(
            f"Episode {episode_id} finished ({label}, {len(steps)} steps). "
            f"Action disabled, robot going home. Saving..."
        )
        self.saving = True
        self._save_thread = threading.Thread(
            target=self._save_episode_background,
            args=(episode_id, steps, success),
            daemon=True,
        )
        self._save_thread.start()

    def _record_tick(self):
        if not self.recording:
            return
        if self.latest_action is None:
            self.get_logger().warning("Recording tick skipped; no action received yet.")
            return
        missing = [k for k in self.state_keys if k not in self.latest.states]
        if missing:
            self.get_logger().warning(f"Recording tick skipped; missing state topics: {missing}")
            return

        now_sec = self.get_clock().now().nanoseconds / 1e9
        step = {
            "timestamp": now_sec,
            "action": list(self.latest_action),
            "action_timestamp": self.latest_action_stamp,
            "states": {key: list(self.latest.states[key]) for key in self.state_keys},
            "images": {key: value.copy() for key, value in self.latest.images.items()},
            "image_encodings": dict(self.latest.image_encodings),
        }
        self.episode_steps.append(step)

    # --- Saving ---

    def _save_episode_background(self, episode_id: int, steps: list, success: bool):
        label = "success" if success else "failure"
        try:
            if not bool(self.get_parameter("use_lerobot_native").value):
                self.get_logger().info(f"Skipping save for {label} episode {episode_id} (use_lerobot_native=False).")
                return
            self._save_lerobot_episode(steps, success)
            self.get_logger().info(f"Saved {label} LeRobot episode {episode_id} ({len(steps)} steps).")
            print("[lerobot_dataset_collector] Ready for next episode. Press Enter to finalize and exit.", flush=True)
        except Exception as exc:
            self.get_logger().error(f"Failed to save episode {episode_id}: {exc}")
        finally:
            self.saving = False

    def _save_lerobot_episode(self, steps: list, success: bool):
        n = len(steps)
        rewards = np.zeros(n, dtype=np.float32)
        if success and n > 0:
            rewards[-1] = 1.0
        dones = np.zeros(n, dtype=np.float32)
        if n > 0:
            dones[-1] = 1.0

        if self.lerobot_dataset is None:
            self.lerobot_dataset = self._create_lerobot_dataset(steps)

        for step_idx, step in enumerate(steps):
            frame = {
                "observation.state": self._state_vector(step),
                "action": np.asarray(step["action"], dtype=np.float32),
                "reward": np.asarray([rewards[step_idx]], dtype=np.float32),
                "done": np.asarray([dones[step_idx]], dtype=np.float32),
                "success": np.asarray([1.0 if success else 0.0], dtype=np.float32),
                "task": self.get_parameter("language_instruction").value,
            }
            for key in self.state_keys:
                frame[f"observation.{key}"] = np.asarray(step["states"][key], dtype=np.float32)
            for camera_key, image in step["images"].items():
                frame[f"observation.images.{camera_key}"] = self._image_for_lerobot(
                    image,
                    step["image_encodings"].get(camera_key, "passthrough"),
                )
            self.lerobot_dataset.add_frame(frame)

        self.lerobot_dataset.save_episode()

    def _create_lerobot_dataset(self, steps: list):
        try:
            from lerobot.datasets import LeRobotDataset
        except ImportError:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

        if self.lerobot_root.exists():
            required = [
                self.lerobot_root / "meta" / "info.json",
                self.lerobot_root / "meta" / "tasks.parquet",
            ]
            if all(f.exists() for f in required):
                self.get_logger().info(f"Resuming existing dataset at {self.lerobot_root}")
                dataset = LeRobotDataset(self._repo_id(), root=self.lerobot_root)
                return dataset
            else:
                import shutil
                missing = [str(f.name) for f in required if not f.exists()]
                self.get_logger().warning(
                    f"Incomplete dataset at {self.lerobot_root} (missing: {missing}); removing and creating fresh."
                )
                shutil.rmtree(self.lerobot_root)

        sample_step = steps[0]
        state_vec = self._state_vector(sample_step)
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": state_vec.shape,
                "names": self._state_names(sample_step),
            },
            "action": {
                "dtype": "float32",
                "shape": np.asarray(sample_step["action"], dtype=np.float32).shape,
                "names": [f"action_{idx}" for idx in range(len(sample_step["action"]))],
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
        }

        for key in self.state_keys:
            values = np.asarray(sample_step["states"][key], dtype=np.float32)
            features[f"observation.{key}"] = {
                "dtype": "float32",
                "shape": values.shape,
                "names": [f"{key}_{idx}" for idx in range(values.shape[0])],
            }

        use_videos = bool(self.get_parameter("use_lerobot_videos").value)
        for camera_key, image in sample_step["images"].items():
            image_array = self._image_for_lerobot(image, sample_step["image_encodings"].get(camera_key, "passthrough"))
            features[f"observation.images.{camera_key}"] = {
                "dtype": "video" if use_videos else "image",
                "shape": image_array.shape,
            }

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
        for key in self.state_keys:
            values.extend(step["states"][key])
        return np.asarray(values, dtype=np.float32)

    def _state_names(self, step) -> List[str]:
        names = []
        for key in self.state_keys:
            n = len(step["states"][key])
            names.extend([f"{key}_{idx}" for idx in range(n)])
        return names

    def _image_for_lerobot(self, image: np.ndarray, encoding: str) -> np.ndarray:
        image = _image_to_rgb(image, encoding)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        if image.shape[-1] == 4:
            image = image[..., :3]
        return np.ascontiguousarray(image, dtype=np.uint8)


def main(args=None):
    args = _with_repo_params_file(args)
    rclpy.init(args=args)
    node = LeRobotDatasetCollectorNode()
    shutdown_event = threading.Event()

    def _wait_for_enter():
        print("\n[lerobot_dataset_collector] Press Enter to finalize dataset and exit.\n", flush=True)
        try:
            input()
        except EOFError:
            pass
        shutdown_event.set()

    stdin_thread = threading.Thread(target=_wait_for_enter, daemon=True)
    stdin_thread.start()

    try:
        while not shutdown_event.is_set():
            rclpy.spin_once(node, timeout_sec=0.5)
    finally:
        if node.recording:
            node.get_logger().warning(
                f"Shutting down while recording episode {node.episode_id}; "
                f"discarding {len(node.episode_steps)} unsaved steps."
            )
        if node._save_thread is not None and node._save_thread.is_alive():
            node.get_logger().info("Waiting for episode save to finish...")
            node._save_thread.join()
        dataset = node.lerobot_dataset
        repo_id = node._repo_id()
        if dataset is not None:
            finalize = getattr(dataset, "finalize", None)
            if finalize is not None:
                node.get_logger().info("Finalizing LeRobot dataset...")
                finalize()
                node.get_logger().info("Dataset finalized.")
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()

        if dataset is None and node.lerobot_root.exists():
            required = [
                node.lerobot_root / "meta" / "info.json",
                node.lerobot_root / "meta" / "tasks.parquet",
            ]
            if all(f.exists() for f in required):
                try:
                    from lerobot.datasets import LeRobotDataset
                except ImportError:
                    from lerobot.datasets.lerobot_dataset import LeRobotDataset
                dataset = LeRobotDataset(repo_id, root=node.lerobot_root)

        if dataset is not None:
            try:
                answer = input(f"\nUpload dataset '{repo_id}' to HuggingFace Hub? [y/N] ").strip().lower()
            except EOFError:
                answer = ""
            if answer in ("y", "yes"):
                print(f"Uploading to {repo_id}...", flush=True)
                dataset.push_to_hub()
                print(f"Upload complete: https://huggingface.co/datasets/{repo_id}", flush=True)
            else:
                print("Skipping upload.", flush=True)


if __name__ == "__main__":
    main()
