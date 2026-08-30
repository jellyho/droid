import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

print("[franka_robot_env_node] importing ROS/Python dependencies...", flush=True)
try:
    import numpy as np
    import rclpy
    from rcl_interfaces.msg import ParameterDescriptor
    from rclpy.node import Node
    import cv2
    from sensor_msgs.msg import CompressedImage, Image
    from std_msgs.msg import Bool, Float64MultiArray, String
    from std_srvs.srv import Trigger
except Exception:
    print("[franka_robot_env_node] failed while importing ROS/Python dependencies:", flush=True)
    print(traceback.format_exc(), flush=True)
    raise
print("[franka_robot_env_node] ROS/Python dependency imports complete.", flush=True)


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "droid_robot_env.yaml"


def _with_repo_params_file(args):
    argv = list(sys.argv if args is None else args)
    config_path = _default_config_path()
    if not config_path.exists():
        raise FileNotFoundError(f"Default ROS2 config not found: {config_path}")

    print(f"[franka_robot_env_node] loading config: {config_path}", flush=True)
    if "--ros-args" in argv:
        return argv + ["--params-file", str(config_path)]
    return argv + ["--ros-args", "--params-file", str(config_path)]


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _to_json(data: Any) -> str:
    return json.dumps(data, default=_json_default)


def _drop_alpha_channel(image: np.ndarray) -> np.ndarray:
    if image.ndim >= 3 and image.shape[-1] == 4:
        return image[..., :3]
    return image


def _image_msg(image: np.ndarray, frame_id: str, stamp) -> Image:
    image = _drop_alpha_channel(image)
    if image.ndim == 2:
        height, width = image.shape
        encoding = "mono8"
        step = width
    elif image.ndim == 3:
        height, width, channels = image.shape
        if channels == 4:
            encoding = "bgra8"
        elif channels == 3:
            encoding = "bgr8"
        else:
            encoding = "passthrough"
        step = width * channels * image.dtype.itemsize
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    contiguous = np.ascontiguousarray(image)
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(height)
    msg.width = int(width)
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = int(step)
    msg.data = contiguous.tobytes()
    return msg


def _compressed_image_msg(image: np.ndarray, frame_id: str, stamp, image_format: str, jpeg_quality: int) -> CompressedImage:
    image = _drop_alpha_channel(image)
    if image_format not in ("jpg", "jpeg", "png"):
        raise ValueError(f"Unsupported compressed image format: {image_format}")

    extension = ".jpg" if image_format in ("jpg", "jpeg") else ".png"
    params = []
    if extension == ".jpg":
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

    success, encoded = cv2.imencode(extension, np.ascontiguousarray(image), params)
    if not success:
        raise ValueError("cv2.imencode failed")

    msg = CompressedImage()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.format = "jpeg" if extension == ".jpg" else "png"
    msg.data = encoded.tobytes()
    return msg


class FrankaRobotEnvNode(Node):
    """Long-running ROS2 wrapper around DROID's RobotEnv."""

    def __init__(self):
        super().__init__("droid_ros")
        self._lock = threading.RLock()
        self._camera_publishers = {}
        self._numeric_publishers = {}
        self._string_publishers = {}
        self._camera_aliases = {}
        self._latest_camera_frames = {}
        self._latest_camera_lock = threading.Lock()
        self._stop_image_thread = threading.Event()
        self._image_thread = None
        self._skipped_camera_keys = set()
        self._hand_camera_id = None
        self._varied_camera_1_id = None
        self._varied_camera_2_id = None
        self._last_publish_warning_sec = 0.0
        self._last_timing_log_sec = 0.0
        self._last_publish_breakdown = {}
        self._published_first_observation = False
        self._logged_graph_after_publish = False
        self._logged_action_received_after_toggle = False
        self._logged_action_sent_after_toggle = False
        self._logged_disabled_action_warning = False
        self._last_action_ramp_warning_sec = 0.0
        self._action_enabled = False
        self._action_ramp_active = False
        self._action_ramp_start_sec = 0.0
        self._action_ramp_start_action = None
        self._latest_robot_state = {}
        self._action_space = None
        self._gripper_action_space = None

        self.declare_parameter("action_space", "cartesian_velocity")
        self.declare_parameter("gripper_action_space", "position")
        self.declare_parameter("control_hz", 10.0)
        self.declare_parameter("do_reset", True)
        self.declare_parameter("publish_timing_log_period_sec", 5.0)
        self.declare_parameter("publish_cameras", True)
        self.declare_parameter("async_image_publish", True)
        self.declare_parameter("camera_transport", "compressed")
        self.declare_parameter("compressed_image_format", "jpg")
        self.declare_parameter("compressed_jpeg_quality", 80)
        self.declare_parameter("publish_camera_metadata", False)
        self.declare_parameter("publish_json_topics", False)
        dynamic_param = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter("camera_width", 224, dynamic_param)
        self.declare_parameter("camera_height", 224, dynamic_param)
        self.declare_parameter("camera_resize_func", "cv2")
        self.declare_parameter("camera_side", "left")
        self.declare_parameter("enable_hand_camera", True)
        self.declare_parameter("enable_varied_camera", True)
        self.declare_parameter("topic_prefix", "/droid_ros")
        self.declare_parameter("action_enabled", False)
        self.declare_parameter("action_topic", "/droid_ros/action")
        self.declare_parameter("action_toggle_topic", "/droid_ros/toggle_action")
        self.declare_parameter("ramp_action_on_enable", True)
        self.declare_parameter("action_ramp_duration_sec", 1.0)
        self.declare_parameter("enable_home_subscriber", True)
        self.declare_parameter("home_topic", "/droid_ros/go_home")
        self.declare_parameter("home_trigger_value", True)
        self.declare_parameter("home_randomize", False)

        action_space = self.get_parameter("action_space").value
        gripper_action_space = self.get_parameter("gripper_action_space").value
        self._action_space = action_space
        self._gripper_action_space = gripper_action_space
        control_hz = float(self.get_parameter("control_hz").value)
        do_reset = bool(self.get_parameter("do_reset").value)

        camera_kwargs = self._camera_kwargs_from_params()
        self.get_logger().info(
            "Starting RobotEnv "
            f"(action_space={action_space}, gripper_action_space={gripper_action_space}, "
            f"control_hz={control_hz}, do_reset={do_reset})"
        )
        try:
            from droid.misc.parameters import hand_camera_id, varied_camera_1_id, varied_camera_2_id
            from droid.robot_env import RobotEnv
        except Exception:
            self.get_logger().fatal("Failed to import DROID RobotEnv dependencies:\n" + traceback.format_exc())
            raise

        self._hand_camera_id = hand_camera_id
        self._varied_camera_1_id = varied_camera_1_id
        self._varied_camera_2_id = varied_camera_2_id

        self.env = RobotEnv(
            action_space=action_space,
            gripper_action_space=gripper_action_space,
            camera_kwargs=camera_kwargs,
            do_reset=do_reset,
            control_hz=control_hz,
        )
        self._camera_aliases = self._build_camera_aliases()

        self._action_enabled = bool(self.get_parameter("action_enabled").value)
        self._action_ramp_active = self._action_enabled and bool(self.get_parameter("ramp_action_on_enable").value)
        action_topic = self.get_parameter("action_topic").value
        action_toggle_topic = self.get_parameter("action_toggle_topic").value
        self.action_sub = self.create_subscription(Float64MultiArray, action_topic, self._on_action, 10)
        self.action_toggle_sub = self.create_subscription(Bool, action_toggle_topic, self._on_action_toggle, 10)
        self.get_logger().info(
            f"Action subscriber on {action_topic}; "
            f"toggle topic={action_toggle_topic}; action_enabled={self._action_enabled}; "
            f"ramp_on_enable={bool(self.get_parameter('ramp_action_on_enable').value)}"
        )
        self.home_sub = None
        if bool(self.get_parameter("enable_home_subscriber").value):
            home_topic = self.get_parameter("home_topic").value
            self.home_sub = self.create_subscription(Bool, home_topic, self._on_home_request, 10)
            self.get_logger().info(
                f"Home subscriber enabled on {home_topic}; "
                f"trigger_value={bool(self.get_parameter('home_trigger_value').value)}"
            )
        self.reset_srv = self.create_service(Trigger, self._topic("reset"), self._on_reset)
        self.reconnect_srv = self.create_service(Trigger, self._topic("reconnect"), self._on_reconnect)
        self.observation_srv = self.create_service(Trigger, self._topic("get_observation"), self._on_get_observation)

        self.heartbeat_pub = self.create_publisher(String, self._topic("heartbeat"), 10)
        self.action_enabled_pub = self.create_publisher(Bool, self._topic("action_enabled"), 10)
        self.applied_action_pub = self.create_publisher(Float64MultiArray, self._topic("applied_action"), 10)
        self.joint_positions_pub = self.create_publisher(Float64MultiArray, self._topic("joint_positions"), 10)
        self.joint_velocities_pub = self.create_publisher(Float64MultiArray, self._topic("joint_velocities"), 10)
        self.cartesian_pub = self.create_publisher(Float64MultiArray, self._topic("cartesian_position"), 10)
        self.gripper_pub = self.create_publisher(Float64MultiArray, self._topic("gripper_position"), 10)
        self.state_json_pub = None
        self.action_info_pub = None
        self.observation_meta_pub = None
        self.observation_json_pub = None
        self.timestamp_json_pub = None
        self.camera_type_json_pub = None
        self.camera_intrinsics_json_pub = None
        self.camera_extrinsics_json_pub = None
        self.camera_alias_json_pub = None
        if bool(self.get_parameter("publish_json_topics").value):
            self.state_json_pub = self.create_publisher(String, self._topic("state_json"), 10)
            self.action_info_pub = self.create_publisher(String, self._topic("action_info_json"), 10)
            self.observation_meta_pub = self.create_publisher(String, self._topic("observation_meta_json"), 10)
            self.observation_json_pub = self.create_publisher(String, self._topic("observation_json"), 10)
            self.timestamp_json_pub = self.create_publisher(String, self._topic("timestamp_json"), 10)
            self.camera_type_json_pub = self.create_publisher(String, self._topic("camera_type_json"), 10)
            self.camera_intrinsics_json_pub = self.create_publisher(String, self._topic("camera_intrinsics_json"), 10)
            self.camera_extrinsics_json_pub = self.create_publisher(String, self._topic("camera_extrinsics_json"), 10)
            self.camera_alias_json_pub = self.create_publisher(String, self._topic("camera_alias_json"), 10)

        self.timer = self.create_timer(1.0 / control_hz, self._publish_observation)
        self.heartbeat_timer = self.create_timer(1.0, self._publish_heartbeat)
        if bool(self.get_parameter("async_image_publish").value):
            self._image_thread = threading.Thread(target=self._image_publish_loop, name="droid_ros_image_publisher", daemon=True)
            self._image_thread.start()
            self.get_logger().info("Async image publisher thread started.")
        self._log_ros_interfaces()
        self._log_ros_environment()
        self.get_logger().info("Franka RobotEnv ROS2 node is ready.")

    def _topic(self, suffix: str) -> str:
        prefix = str(self.get_parameter("topic_prefix").value).strip()
        if not prefix.startswith("/"):
            prefix = f"/{prefix}"
        prefix = prefix.rstrip("/")
        return f"{prefix}/{suffix.lstrip('/')}"

    def _publish_heartbeat(self):
        msg = String()
        msg.data = "ready"
        self.heartbeat_pub.publish(msg)
        self._publish_action_enabled()

    def _publish_action_enabled(self):
        msg = Bool()
        msg.data = bool(self._action_enabled)
        self.action_enabled_pub.publish(msg)

    def _log_ros_interfaces(self):
        published_topics = [
            self._topic("heartbeat"),
            self._topic("action_enabled"),
            self._topic("applied_action"),
            self._topic("joint_positions"),
            self._topic("joint_velocities"),
            self._topic("cartesian_position"),
            self._topic("gripper_position"),
        ]
        if bool(self.get_parameter("publish_json_topics").value):
            published_topics.extend(
                [
                    self._topic("state_json"),
                    self._topic("action_info_json"),
                    self._topic("observation_meta_json"),
                    self._topic("observation_json"),
                    self._topic("timestamp_json"),
                    self._topic("camera_type_json"),
                    self._topic("camera_intrinsics_json"),
                    self._topic("camera_extrinsics_json"),
                    self._topic("camera_alias_json"),
                ]
            )
        self.get_logger().info("Publishing base topics: " + ", ".join(published_topics))
        self.get_logger().info(
            "Services: "
            + ", ".join([self._topic("reset"), self._topic("reconnect"), self._topic("get_observation")])
        )

    def _log_ros_environment(self):
        env_keys = ("ROS_DOMAIN_ID", "ROS_LOCALHOST_ONLY", "RMW_IMPLEMENTATION", "ROS_DISCOVERY_SERVER")
        env_summary = ", ".join(f"{key}={os.environ.get(key, '<unset>')}" for key in env_keys)
        self.get_logger().info(f"ROS environment: {env_summary}")
        self.get_logger().info(f"Node name: {self.get_fully_qualified_name()}")

    def _log_graph_topics(self, label):
        topics = self.get_topic_names_and_types()
        droid_topics = [(name, types) for name, types in topics if name.startswith("/droid_ros")]
        self.get_logger().info(f"{label}: node sees {len(droid_topics)} /droid_ros topics: {_to_json(droid_topics)}")

    def _camera_kwargs_from_params(self):
        width = self._optional_positive_int("camera_width")
        height = self._optional_positive_int("camera_height")
        resize_func = self.get_parameter("camera_resize_func").value
        if width is None or height is None:
            width = 0
            height = 0
            resize_func = None
        camera_kwargs = {}
        common_kwargs = {
            "image": True,
            "concatenate_images": False,
            "resolution": (width, height),
            "resize_func": resize_func,
        }
        if bool(self.get_parameter("enable_hand_camera").value):
            camera_kwargs["hand_camera"] = dict(common_kwargs)
        if bool(self.get_parameter("enable_varied_camera").value):
            camera_kwargs["varied_camera"] = dict(common_kwargs)
        return camera_kwargs

    def _optional_positive_int(self, parameter_name):
        value = self.get_parameter(parameter_name).value
        if value is None:
            return None
        if isinstance(value, str) and value.lower() in ("", "none", "null", "original"):
            return None
        value = int(value)
        if value <= 0:
            return None
        return value

    def _build_camera_aliases(self):
        aliases = {}
        configured_cameras = (
            (self._hand_camera_id, "wrist"),
            (self._varied_camera_1_id, "side_view_1"),
            (self._varied_camera_2_id, "side_view_2"),
        )
        for serial, alias in configured_cameras:
            if serial and serial != "99999999":
                aliases[str(serial)] = alias

        connected_serials = {str(serial) for serial in getattr(self.env.camera_reader, "camera_dict", {}).keys()}
        for serial, alias in aliases.items():
            status = "connected" if serial in connected_serials else "configured"
            self.get_logger().info(f"Assigned camera alias: {serial} -> {alias} ({status})")
        for serial in sorted(connected_serials - set(aliases)):
            self.get_logger().warning(
                f"Connected camera {serial} is not listed in droid/misc/parameters.py; "
                "it will not be published."
            )
        self.get_logger().info(f"Camera aliases: {_to_json(aliases)}")
        return aliases

    def _camera_alias_for_serial(self, serial: str):
        return self._camera_aliases.get(str(serial))

    def _alias_camera_key(self, key: str):
        key = str(key)
        side = str(self.get_parameter("camera_side").value).lower()
        suffixes = ("_left_gripper_offset", "_right_gripper_offset", "_gripper_offset", "_left", "_right")
        for suffix in suffixes:
            if key.endswith(suffix):
                if not self._camera_suffix_enabled(suffix, side):
                    return None
                serial = key[: -len(suffix)]
                alias = self._camera_alias_for_serial(serial)
                return f"{alias}{suffix}" if alias is not None else None
        return self._camera_alias_for_serial(key)

    def _camera_suffix_enabled(self, suffix: str, side: str) -> bool:
        if side in ("both", "all", ""):
            return True
        if side not in ("left", "right"):
            self.get_logger().warning(f"Unknown camera_side={side}; defaulting to left.")
            side = "left"
        if suffix in (f"_{side}", f"_{side}_gripper_offset"):
            return True
        if suffix == "_gripper_offset":
            return True
        return False

    def _alias_timestamp_key(self, key: str):
        key = str(key)
        for serial, alias in self._camera_aliases.items():
            prefix = f"{serial}_"
            if key.startswith(prefix):
                return f"{alias}_{key[len(prefix):]}"
        return None

    def _alias_camera_dict(self, data):
        output = {}
        for key, value in data.items():
            alias = self._alias_camera_key(key)
            if alias is None:
                self._log_skipped_camera_key(key)
                continue
            output[alias] = value
        return output

    def _log_skipped_camera_key(self, key):
        key = str(key)
        if key in self._skipped_camera_keys:
            return
        self._skipped_camera_keys.add(key)
        self.get_logger().info(
            f"Skipping camera metadata for {key}; not configured in droid/misc/parameters.py"
        )

    def _alias_observation(self, obs):
        aliased = dict(obs)
        if "image" in obs:
            aliased["image"] = self._alias_camera_dict(obs["image"])
        if "camera_type" in obs:
            aliased["camera_type"] = self._alias_camera_dict(obs["camera_type"])
        if "camera_intrinsics" in obs:
            aliased["camera_intrinsics"] = self._alias_camera_dict(obs["camera_intrinsics"])
        if "camera_extrinsics" in obs:
            aliased["camera_extrinsics"] = self._alias_camera_dict(obs["camera_extrinsics"])
        if "timestamp" in obs:
            aliased_timestamps = {}
            for group, value in obs["timestamp"].items():
                if group == "cameras" and isinstance(value, dict):
                    camera_timestamps = {}
                    for key, item in value.items():
                        alias = self._alias_timestamp_key(key)
                        if alias is None:
                            self._log_skipped_camera_key(key)
                            continue
                        camera_timestamps[alias] = item
                    aliased_timestamps[group] = camera_timestamps
                else:
                    aliased_timestamps[group] = value
            aliased["timestamp"] = aliased_timestamps
        aliased["camera_aliases"] = dict(self._camera_aliases)
        return aliased

    def _on_action(self, msg: Float64MultiArray):
        action = np.asarray(msg.data, dtype=np.float64)
        self._log_action_received_once(action)
        if not self._action_enabled:
            if not self._logged_disabled_action_warning:
                self.get_logger().warning(
                    f"Ignoring action command because action control is disabled. "
                    f"len={action.shape[0]}, head={self._short_action(action)}"
                )
                self._logged_disabled_action_warning = True
            return

        with self._lock:
            expected = self.env.DoF
            if action.shape[0] != expected:
                self.get_logger().error(f"Expected action length {expected}, got {action.shape[0]}")
                return
            action = self._ramp_action_if_needed(action)
            if action is None:
                return
            self._log_action_sent_once(action)
            try:
                action_info = self.env.step(action)
            except Exception as exc:
                self.get_logger().exception(f"Failed to apply action: {exc}")
                return

        applied_msg = Float64MultiArray()
        applied_msg.data = [float(item) for item in action.reshape(-1)]
        self.applied_action_pub.publish(applied_msg)

        out = String()
        out.data = _to_json(action_info)
        if self.action_info_pub is not None:
            self.action_info_pub.publish(out)

    def _log_action_received_once(self, action):
        if self._logged_action_received_after_toggle:
            return
        self.get_logger().info(
            f"Action command received after toggle/start: enabled={self._action_enabled}, "
            f"ramp_active={self._action_ramp_active}, len={action.shape[0]}, "
            f"head={self._short_action(action)}"
        )
        self._logged_action_received_after_toggle = True

    def _log_action_sent_once(self, action):
        if self._logged_action_sent_after_toggle:
            return
        self.get_logger().info(
            f"Action command sent to RobotEnv after toggle/start: "
            f"ramp_active={self._action_ramp_active}, len={action.shape[0]}, "
            f"head={self._short_action(action)}"
        )
        self._logged_action_sent_after_toggle = True

    def _short_action(self, action, max_items=4):
        values = np.asarray(action, dtype=np.float64).reshape(-1)
        prefix = ", ".join(f"{item:.3f}" for item in values[:max_items])
        if values.shape[0] > max_items:
            prefix += ", ..."
        return f"[{prefix}]"

    def _on_action_toggle(self, _msg: Bool):
        enabled = not self._action_enabled
        self._action_enabled = enabled
        self._logged_action_received_after_toggle = False
        self._logged_action_sent_after_toggle = False
        self._logged_disabled_action_warning = False
        if enabled:
            self._start_action_ramp()
        else:
            self._action_ramp_active = False
            self._action_ramp_start_action = None
        state = "enabled" if enabled else "disabled"
        ramp_msg = " smoothing first commands." if self._action_ramp_active else ""
        self.get_logger().info(f"Action control toggled {state}.{ramp_msg}")
        self._publish_action_enabled()

    def _start_action_ramp(self):
        if not bool(self.get_parameter("ramp_action_on_enable").value):
            self._action_ramp_active = False
            self._action_ramp_start_action = None
            return
        self._action_ramp_start_action = self._current_action_anchor()
        self._action_ramp_start_sec = self.get_clock().now().nanoseconds / 1e9
        self._action_ramp_active = self._action_ramp_start_action is not None
        if not self._action_ramp_active:
            self.get_logger().warning("Action ramp requested, but no compatible robot state is available yet.")

    def _current_action_anchor(self):
        state = self._latest_robot_state
        if not state:
            return None

        if self._action_space == "joint_position":
            current = self._float_list(state.get("joint_positions"))
            if len(current) < 7:
                return None
            anchor = np.asarray(current[:7], dtype=np.float64)

        elif self._action_space == "cartesian_position":
            current = self._float_list(state.get("cartesian_position"))
            if len(current) < 6:
                return None
            anchor = np.asarray(current[:6], dtype=np.float64)

        elif self._action_space in ("joint_velocity", "cartesian_velocity"):
            dof_without_gripper = self.env.DoF - (1 if self._gripper_action_space is not None else 0)
            anchor = np.zeros(dof_without_gripper, dtype=np.float64)

        else:
            return None

        if self._gripper_action_space == "position":
            current_gripper = self._float_list(state.get("gripper_position"))
            if current_gripper:
                anchor = np.concatenate([anchor, np.asarray([current_gripper[0]], dtype=np.float64)])
        return anchor

    def _ramp_action_if_needed(self, target_action):
        if not self._action_ramp_active:
            return target_action

        if self._action_ramp_start_action is None:
            self._action_ramp_start_action = self._current_action_anchor()
            if self._action_ramp_start_action is None:
                now_sec = self.get_clock().now().nanoseconds / 1e9
                if now_sec - self._last_action_ramp_warning_sec > 2.0:
                    self.get_logger().warning("Ignoring action while waiting for robot state to start ramp.")
                    self._last_action_ramp_warning_sec = now_sec
                return None
            self._action_ramp_start_sec = self.get_clock().now().nanoseconds / 1e9

        if self._action_ramp_start_action.shape[0] != target_action.shape[0]:
            self.get_logger().warning(
                f"Action ramp anchor length {self._action_ramp_start_action.shape[0]} "
                f"does not match target length {target_action.shape[0]}; disabling ramp."
            )
            self._action_ramp_active = False
            return target_action

        duration = max(0.0, float(self.get_parameter("action_ramp_duration_sec").value))
        if duration <= 0:
            self._action_ramp_active = False
            return target_action

        now_sec = self.get_clock().now().nanoseconds / 1e9
        alpha = min(1.0, max(0.0, (now_sec - self._action_ramp_start_sec) / duration))
        smoothed = (1.0 - alpha) * self._action_ramp_start_action + alpha * target_action
        if alpha >= 1.0:
            self._action_ramp_active = False
            self._action_ramp_start_action = None
            self.get_logger().info("Action ramp complete; passing through Gello commands directly.")
        return smoothed

    def _on_home_request(self, msg: Bool):
        self.get_logger().info(f"Received home request: {msg.data}")
        trigger_value = bool(self.get_parameter("home_trigger_value").value)
        if bool(msg.data) != trigger_value:
            return
        try:
            self._reset_robot_to_home()
            self.get_logger().info("Robot moved to home position from Bool trigger.")
        except Exception as exc:
            self.get_logger().exception(f"Home reset failed: {exc}")

    def _on_reset(self, _request, response):
        try:
            self._reset_robot_to_home()
            response.success = True
            response.message = "Robot reset complete."
        except Exception as exc:
            self.get_logger().exception(f"Reset failed: {exc}")
            response.success = False
            response.message = str(exc)
        return response

    def _reset_robot_to_home(self):
        randomize = bool(self.get_parameter("home_randomize").value)
        with self._lock:
            self.env.reset(randomize=randomize)

    def _on_reconnect(self, _request, response):
        try:
            with self._lock:
                establish_connection = getattr(self.env._robot, "establish_connection", None)
                if establish_connection is None:
                    response.success = True
                    response.message = "Robot backend has no establish_connection method."
                else:
                    establish_connection()
                    response.success = True
                    response.message = "Robot backend reconnected."
        except Exception as exc:
            self.get_logger().exception(f"Reconnect failed: {exc}")
            response.success = False
            response.message = str(exc)
        return response

    def _on_get_observation(self, _request, response):
        try:
            with self._lock:
                obs = self.env.get_observation()
            obs = self._alias_observation(obs)
            self._publish_from_observation(obs)
            response.success = True
            response.message = _to_json(self._observation_metadata(obs))
        except Exception as exc:
            self.get_logger().exception(f"Observation read failed: {exc}")
            response.success = False
            response.message = str(exc)
        return response

    def _publish_observation(self):
        try:
            total_start = time.perf_counter()
            with self._lock:
                obs_start = time.perf_counter()
                obs = self.env.get_observation()
                obs_elapsed = time.perf_counter() - obs_start
            alias_start = time.perf_counter()
            obs = self._alias_observation(obs)
            alias_elapsed = time.perf_counter() - alias_start
            publish_start = time.perf_counter()
            self._publish_from_observation(obs)
            publish_elapsed = time.perf_counter() - publish_start
            total_elapsed = time.perf_counter() - total_start
            self._log_publish_timing(total_elapsed, obs_elapsed, alias_elapsed, publish_elapsed)
        except Exception as exc:
            now_sec = self.get_clock().now().nanoseconds / 1e9
            if now_sec - self._last_publish_warning_sec > 5.0:
                self.get_logger().warning(f"Skipping observation publish: {exc}")
                self._last_publish_warning_sec = now_sec

    def _log_publish_timing(self, total_elapsed, obs_elapsed, alias_elapsed, publish_elapsed):
        period = float(self.get_parameter("publish_timing_log_period_sec").value)
        if period <= 0:
            return
        now_sec = self.get_clock().now().nanoseconds / 1e9
        if now_sec - self._last_timing_log_sec < period:
            return
        self._last_timing_log_sec = now_sec
        effective_hz = 1.0 / total_elapsed if total_elapsed > 0 else float("inf")
        self.get_logger().info(
            "Observation loop timing: "
            f"total={total_elapsed * 1000.0:.1f} ms ({effective_hz:.2f} Hz max), "
            f"get_observation={obs_elapsed * 1000.0:.1f} ms, "
            f"alias={alias_elapsed * 1000.0:.1f} ms, "
            f"publish={publish_elapsed * 1000.0:.1f} ms, "
            f"breakdown={_to_json(self._last_publish_breakdown)}"
        )

    def _publish_from_observation(self, obs):
        breakdown = {}
        stamp = self.get_clock().now().to_msg()
        state = obs.get("robot_state", {})
        self._latest_robot_state = state

        state_start = time.perf_counter()
        joint_positions = self._float_list(state.get("joint_positions", []))
        joint_velocities = self._float_list(state.get("joint_velocities", []))
        cartesian = self._float_list(state.get("cartesian_position"))
        gripper = self._float_list(state.get("gripper_position"))
        gripper_val = [gripper[0]] if gripper else []

        if joint_positions:
            joint_msg = Float64MultiArray()
            joint_msg.data = joint_positions + gripper_val
            self.joint_positions_pub.publish(joint_msg)
        if joint_velocities:
            joint_vel_msg = Float64MultiArray()
            joint_vel_msg.data = joint_velocities + gripper_val
            self.joint_velocities_pub.publish(joint_vel_msg)
        if cartesian:
            cart_msg = Float64MultiArray()
            cart_msg.data = cartesian + gripper_val
            self.cartesian_pub.publish(cart_msg)
        if gripper:
            grip_msg = Float64MultiArray()
            grip_msg.data = gripper_val
            self.gripper_pub.publish(grip_msg)
        breakdown["main_state_ms"] = (time.perf_counter() - state_start) * 1000.0

        json_start = time.perf_counter()
        if bool(self.get_parameter("publish_json_topics").value):
            self._publish_json(self.state_json_pub, state)
            self._publish_json(self.observation_meta_pub, self._observation_metadata(obs))
            self._publish_json(self.observation_json_pub, self._observation_without_images(obs))
            self._publish_json(self.timestamp_json_pub, obs.get("timestamp", {}))
            self._publish_json(self.camera_type_json_pub, obs.get("camera_type", {}))
            self._publish_json(self.camera_intrinsics_json_pub, obs.get("camera_intrinsics", {}))
            self._publish_json(self.camera_extrinsics_json_pub, obs.get("camera_extrinsics", {}))
            self._publish_json(self.camera_alias_json_pub, obs.get("camera_aliases", {}))
        breakdown["json_ms"] = (time.perf_counter() - json_start) * 1000.0

        robot_dynamic_start = time.perf_counter()
        self._publish_dynamic_robot_state(state)
        breakdown["robot_dynamic_ms"] = (time.perf_counter() - robot_dynamic_start) * 1000.0

        metadata_start = time.perf_counter()
        if bool(self.get_parameter("publish_camera_metadata").value):
            self._publish_dynamic_camera_metadata("camera_intrinsics", obs.get("camera_intrinsics", {}))
            self._publish_dynamic_camera_metadata("camera_extrinsics", obs.get("camera_extrinsics", {}))
        breakdown["camera_metadata_ms"] = (time.perf_counter() - metadata_start) * 1000.0

        camera_start = time.perf_counter()
        if bool(self.get_parameter("publish_cameras").value):
            if bool(self.get_parameter("async_image_publish").value):
                self._store_latest_camera_frames(obs.get("image", {}), stamp)
            else:
                camera_transport = str(self.get_parameter("camera_transport").value).lower()
                for cam_id, image in obs.get("image", {}).items():
                    self._publish_camera_image(cam_id, image, stamp, camera_transport)
        breakdown["camera_images_ms"] = (time.perf_counter() - camera_start) * 1000.0

        depth_start = time.perf_counter()
        if bool(self.get_parameter("publish_cameras").value):
            for cam_id, depth_arr in obs.get("depth", {}).items():
                self._publish_depth_image(cam_id, depth_arr, stamp)
        breakdown["depth_ms"] = (time.perf_counter() - depth_start) * 1000.0

        self._last_publish_breakdown = {key: round(value, 1) for key, value in breakdown.items()}

        if not self._published_first_observation:
            self._published_first_observation = True
            self.get_logger().info(
                "Published first observation "
                f"(robot_state_keys={list(state.keys())}, cameras={list(obs.get('image', {}).keys())})"
            )
        if not self._logged_graph_after_publish:
            self._logged_graph_after_publish = True
            self._log_graph_topics("After first publish")

    def _store_latest_camera_frames(self, images, stamp):
        with self._latest_camera_lock:
            self._latest_camera_frames = {
                cam_id: (image, stamp)
                for cam_id, image in images.items()
            }

    def _image_publish_loop(self):
        target_period = 1.0 / max(1e-6, float(self.get_parameter("control_hz").value))
        while not self._stop_image_thread.is_set():
            loop_start = time.perf_counter()
            if bool(self.get_parameter("publish_cameras").value):
                with self._latest_camera_lock:
                    frames = dict(self._latest_camera_frames)
                camera_transport = str(self.get_parameter("camera_transport").value).lower()
                for cam_id, (image, stamp) in frames.items():
                    try:
                        self._publish_camera_image(cam_id, image, stamp, camera_transport)
                    except Exception as exc:
                        self.get_logger().warning(f"Failed to publish camera {cam_id}: {exc}")
            elapsed = time.perf_counter() - loop_start
            self._stop_image_thread.wait(max(0.0, target_period - elapsed))

    def _publish_camera_image(self, cam_id, image, stamp, camera_transport):
        if camera_transport == "raw":
            topic = self._topic(f"camera/{cam_id}/image_raw")
            publisher_key = (cam_id, camera_transport)
            pub = self._camera_publishers.get(publisher_key)
            if pub is None:
                pub = self.create_publisher(Image, topic, 5)
                self._camera_publishers[publisher_key] = pub
                self.get_logger().info(f"Publishing raw camera stream: {topic}")
            pub.publish(_image_msg(image, cam_id, stamp))
            return

        if camera_transport != "compressed":
            self.get_logger().warning(f"Unknown camera_transport={camera_transport}; using compressed.")

        topic = self._topic(f"camera/{cam_id}/image_raw/compressed")
        publisher_key = (cam_id, "compressed")
        pub = self._camera_publishers.get(publisher_key)
        if pub is None:
            pub = self.create_publisher(CompressedImage, topic, 5)
            self._camera_publishers[publisher_key] = pub
            self.get_logger().info(f"Publishing compressed camera stream: {topic}")
        image_format = str(self.get_parameter("compressed_image_format").value).lower()
        jpeg_quality = int(self.get_parameter("compressed_jpeg_quality").value)
        pub.publish(_compressed_image_msg(image, cam_id, stamp, image_format, jpeg_quality))

    def _publish_depth_image(self, cam_id, depth_arr, stamp):
        """Publish depth as a raw Image (32FC1, meters)."""
        topic = self._topic(f"camera/{cam_id}/depth/image_raw")
        publisher_key = (cam_id, "depth")
        pub = self._camera_publishers.get(publisher_key)
        if pub is None:
            pub = self.create_publisher(Image, topic, 5)
            self._camera_publishers[publisher_key] = pub
            self.get_logger().info(f"Publishing depth stream: {topic}")
        depth_f32 = np.ascontiguousarray(depth_arr, dtype=np.float32)
        if depth_f32.ndim == 3:
            depth_f32 = depth_f32[:, :, 0]
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = cam_id
        msg.height, msg.width = depth_f32.shape
        msg.encoding = "32FC1"
        msg.is_bigendian = False
        msg.step = msg.width * 4
        msg.data = depth_f32.tobytes()
        pub.publish(msg)

    def _publish_json(self, publisher, data):
        msg = String()
        msg.data = _to_json(data)
        publisher.publish(msg)

    def _publish_dynamic_robot_state(self, state):
        for key, value in state.items():
            topic = self._topic(f"robot_state/{self._sanitize_topic_part(key)}")
            self._publish_dynamic_value(topic, value)

    def _publish_dynamic_camera_metadata(self, group, data):
        for cam_key, value in data.items():
            topic = self._topic(f"{group}/{self._sanitize_topic_part(cam_key)}")
            self._publish_dynamic_value(topic, value)

    def _publish_dynamic_value(self, topic, value):
        numeric = self._numeric_list(value)
        if numeric is not None:
            pub = self._numeric_publishers.get(topic)
            if pub is None:
                pub = self.create_publisher(Float64MultiArray, topic, 10)
                self._numeric_publishers[topic] = pub
            msg = Float64MultiArray()
            msg.data = [float(item) for item in numeric]
            pub.publish(msg)
            return

        pub = self._string_publishers.get(topic)
        if pub is None:
            pub = self.create_publisher(String, topic, 10)
            self._string_publishers[topic] = pub
        msg = String()
        msg.data = _to_json(value)
        pub.publish(msg)

    def _numeric_list(self, value):
        if isinstance(value, bool):
            return [1.0 if value else 0.0]
        if isinstance(value, (int, float, np.number)):
            return [float(value)]
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if array.dtype == np.dtype("O"):
            return None
        if array.size == 0:
            return None
        return array.reshape(-1).tolist()

    def _float_list(self, value):
        if value is None:
            return []
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            return []
        if array.size == 0:
            return []
        return [float(item) for item in array.reshape(-1)]

    def _sanitize_topic_part(self, value: str) -> str:
        token = "".join(char if char.isalnum() or char == "_" else "_" for char in str(value))
        if token and token[0].isdigit():
            token = f"camera_{token}"
        return token

    def _observation_without_images(self, obs):
        output = dict(obs)
        output.pop("image", None)
        output["image_shapes"] = {cam_id: list(image.shape) for cam_id, image in obs.get("image", {}).items()}
        return output

    def _observation_metadata(self, obs):
        image_shapes = {}
        for cam_id, image in obs.get("image", {}).items():
            image_shapes[cam_id] = list(image.shape)

        return {
            "timestamp": obs.get("timestamp", {}),
            "camera_type": obs.get("camera_type", {}),
            "camera_intrinsics": obs.get("camera_intrinsics", {}),
            "camera_extrinsics": obs.get("camera_extrinsics", {}),
            "camera_aliases": obs.get("camera_aliases", {}),
            "image_shapes": image_shapes,
        }


def main(args=None):
    args = _with_repo_params_file(args)
    rclpy.init(args=args)
    node = None
    try:
        print('Starting Franka RobotEnv ROS2 node...', flush=True)
        node = FrankaRobotEnvNode()
        rclpy.spin(node)
    except Exception:
        if node is not None:
            node.get_logger().fatal(traceback.format_exc())
        else:
            print(traceback.format_exc(), flush=True)
        raise
    finally:
        if node is not None:
            node._stop_image_thread.set()
            if node._image_thread is not None:
                node._image_thread.join(timeout=2.0)
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
