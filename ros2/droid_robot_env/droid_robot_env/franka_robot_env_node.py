import json
import threading
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Float64, Float64MultiArray, String
from std_srvs.srv import Trigger

from droid.misc.parameters import hand_camera_id, varied_camera_1_id, varied_camera_2_id
from droid.robot_env import RobotEnv


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


def _image_msg(image: np.ndarray, frame_id: str, stamp) -> Image:
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


class FrankaRobotEnvNode(Node):
    """Long-running ROS2 wrapper around DROID's RobotEnv."""

    def __init__(self):
        super().__init__("droid_ros")
        self._lock = threading.RLock()
        self._camera_publishers = {}
        self._numeric_publishers = {}
        self._string_publishers = {}
        self._camera_aliases = {}
        self._last_publish_warning_sec = 0.0

        self.declare_parameter("action_space", "cartesian_velocity")
        self.declare_parameter("gripper_action_space", "position")
        self.declare_parameter("control_hz", 10.0)
        self.declare_parameter("do_reset", False)
        self.declare_parameter("publish_rate", 10.0)
        self.declare_parameter("publish_cameras", True)
        self.declare_parameter("camera_width", 224)
        self.declare_parameter("camera_height", 224)
        self.declare_parameter("camera_resize_func", "cv2")
        self.declare_parameter("enable_hand_camera", True)
        self.declare_parameter("enable_varied_camera", True)
        self.declare_parameter("enable_home_subscriber", True)
        self.declare_parameter("home_topic", "~/go_home")
        self.declare_parameter("home_trigger_value", True)
        self.declare_parameter("home_randomize", False)

        action_space = self.get_parameter("action_space").value
        gripper_action_space = self.get_parameter("gripper_action_space").value
        control_hz = float(self.get_parameter("control_hz").value)
        do_reset = bool(self.get_parameter("do_reset").value)

        camera_kwargs = self._camera_kwargs_from_params()
        self.get_logger().info(
            "Starting RobotEnv "
            f"(action_space={action_space}, gripper_action_space={gripper_action_space}, "
            f"control_hz={control_hz}, do_reset={do_reset})"
        )
        self.env = RobotEnv(
            action_space=action_space,
            gripper_action_space=gripper_action_space,
            camera_kwargs=camera_kwargs,
            do_reset=do_reset,
            control_hz=control_hz,
        )
        self._camera_aliases = self._build_camera_aliases()

        self.action_sub = self.create_subscription(Float64MultiArray, "~/action", self._on_action, 10)
        self.home_sub = None
        if bool(self.get_parameter("enable_home_subscriber").value):
            home_topic = self.get_parameter("home_topic").value
            self.home_sub = self.create_subscription(Bool, home_topic, self._on_home_request, 10)
            self.get_logger().info(
                f"Home subscriber enabled on {home_topic}; "
                f"trigger_value={bool(self.get_parameter('home_trigger_value').value)}"
            )
        self.reset_srv = self.create_service(Trigger, "~/reset", self._on_reset)
        self.reconnect_srv = self.create_service(Trigger, "~/reconnect", self._on_reconnect)
        self.observation_srv = self.create_service(Trigger, "~/get_observation", self._on_get_observation)

        self.joint_state_pub = self.create_publisher(JointState, "~/joint_states", 10)
        self.cartesian_pub = self.create_publisher(Float64MultiArray, "~/cartesian_position", 10)
        self.gripper_pub = self.create_publisher(Float64, "~/gripper_position", 10)
        self.state_json_pub = self.create_publisher(String, "~/state_json", 10)
        self.action_info_pub = self.create_publisher(String, "~/action_info_json", 10)
        self.observation_meta_pub = self.create_publisher(String, "~/observation_meta_json", 10)
        self.observation_json_pub = self.create_publisher(String, "~/observation_json", 10)
        self.timestamp_json_pub = self.create_publisher(String, "~/timestamp_json", 10)
        self.camera_type_json_pub = self.create_publisher(String, "~/camera_type_json", 10)
        self.camera_intrinsics_json_pub = self.create_publisher(String, "~/camera_intrinsics_json", 10)
        self.camera_extrinsics_json_pub = self.create_publisher(String, "~/camera_extrinsics_json", 10)
        self.camera_alias_json_pub = self.create_publisher(String, "~/camera_alias_json", 10)

        publish_rate = float(self.get_parameter("publish_rate").value)
        self.timer = self.create_timer(1.0 / publish_rate, self._publish_observation)
        self.get_logger().info("Franka RobotEnv ROS2 node is ready.")

    def _camera_kwargs_from_params(self):
        width = int(self.get_parameter("camera_width").value)
        height = int(self.get_parameter("camera_height").value)
        resize_func = self.get_parameter("camera_resize_func").value
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

    def _build_camera_aliases(self):
        aliases = {}
        serials = sorted(self.env.camera_reader.camera_dict.keys())
        side_count = 0
        if varied_camera_1_id in serials:
            side_count = 1
        if varied_camera_2_id in serials:
            side_count = 2
        for serial in serials:
            if serial == hand_camera_id:
                aliases[serial] = "wrist"
            elif serial == varied_camera_1_id:
                aliases[serial] = "side_view_1"
            elif serial == varied_camera_2_id:
                aliases[serial] = "side_view_2"
            else:
                side_count += 1
                aliases[serial] = f"side_view_{side_count}"
        self.get_logger().info(f"Camera aliases: {_to_json(aliases)}")
        return aliases

    def _alias_camera_key(self, key: str) -> str:
        for suffix in ("_left", "_right"):
            if key.endswith(suffix):
                serial = key[: -len(suffix)]
                alias = self._camera_aliases.get(serial, serial)
                return f"{alias}{suffix}"
        return self._camera_aliases.get(key, key)

    def _alias_timestamp_key(self, key: str) -> str:
        for serial, alias in self._camera_aliases.items():
            prefix = f"{serial}_"
            if key.startswith(prefix):
                return f"{alias}_{key[len(prefix):]}"
        return key

    def _alias_camera_dict(self, data):
        return {self._alias_camera_key(key): value for key, value in data.items()}

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
                    aliased_timestamps[group] = {self._alias_timestamp_key(key): item for key, item in value.items()}
                else:
                    aliased_timestamps[group] = value
            aliased["timestamp"] = aliased_timestamps
        aliased["camera_aliases"] = dict(self._camera_aliases)
        return aliased

    def _on_action(self, msg: Float64MultiArray):
        action = np.asarray(msg.data, dtype=np.float64)
        with self._lock:
            expected = self.env.DoF
            if action.shape[0] != expected:
                self.get_logger().error(f"Expected action length {expected}, got {action.shape[0]}")
                return
            try:
                action_info = self.env.step(action)
            except Exception as exc:
                self.get_logger().exception(f"Failed to apply action: {exc}")
                return

        out = String()
        out.data = _to_json(action_info)
        self.action_info_pub.publish(out)

    def _on_home_request(self, msg: Bool):
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
            with self._lock:
                obs = self.env.get_observation()
            obs = self._alias_observation(obs)
            self._publish_from_observation(obs)
        except Exception as exc:
            now_sec = self.get_clock().now().nanoseconds / 1e9
            if now_sec - self._last_publish_warning_sec > 5.0:
                self.get_logger().warning(f"Skipping observation publish: {exc}")
                self._last_publish_warning_sec = now_sec

    def _publish_from_observation(self, obs):
        stamp = self.get_clock().now().to_msg()
        state = obs.get("robot_state", {})

        joint_positions = state.get("joint_positions", [])
        joint_velocities = state.get("joint_velocities", [])
        if joint_positions:
            joint_msg = JointState()
            joint_msg.header.stamp = stamp
            joint_msg.name = [f"panda_joint{i + 1}" for i in range(len(joint_positions))]
            joint_msg.position = [float(x) for x in joint_positions]
            if joint_velocities:
                joint_msg.velocity = [float(x) for x in joint_velocities]
            self.joint_state_pub.publish(joint_msg)

        cartesian = state.get("cartesian_position")
        if cartesian is not None:
            cart_msg = Float64MultiArray()
            cart_msg.data = [float(x) for x in cartesian]
            self.cartesian_pub.publish(cart_msg)

        gripper = state.get("gripper_position")
        if gripper is not None:
            grip_msg = Float64()
            grip_msg.data = float(gripper)
            self.gripper_pub.publish(grip_msg)

        state_msg = String()
        state_msg.data = _to_json(state)
        self.state_json_pub.publish(state_msg)

        meta_msg = String()
        meta_msg.data = _to_json(self._observation_metadata(obs))
        self.observation_meta_pub.publish(meta_msg)

        observation_msg = String()
        observation_msg.data = _to_json(self._observation_without_images(obs))
        self.observation_json_pub.publish(observation_msg)

        self._publish_json(self.timestamp_json_pub, obs.get("timestamp", {}))
        self._publish_json(self.camera_type_json_pub, obs.get("camera_type", {}))
        self._publish_json(self.camera_intrinsics_json_pub, obs.get("camera_intrinsics", {}))
        self._publish_json(self.camera_extrinsics_json_pub, obs.get("camera_extrinsics", {}))
        self._publish_json(self.camera_alias_json_pub, obs.get("camera_aliases", {}))
        self._publish_dynamic_robot_state(state)
        self._publish_dynamic_camera_metadata("camera_intrinsics", obs.get("camera_intrinsics", {}))
        self._publish_dynamic_camera_metadata("camera_extrinsics", obs.get("camera_extrinsics", {}))

        if bool(self.get_parameter("publish_cameras").value):
            for cam_id, image in obs.get("image", {}).items():
                pub = self._camera_publishers.get(cam_id)
                if pub is None:
                    topic = f"~/camera/{cam_id}/image_raw"
                    pub = self.create_publisher(Image, topic, 5)
                    self._camera_publishers[cam_id] = pub
                    self.get_logger().info(f"Publishing camera stream: {topic}")
                pub.publish(_image_msg(image, cam_id, stamp))

    def _publish_json(self, publisher, data):
        msg = String()
        msg.data = _to_json(data)
        publisher.publish(msg)

    def _publish_dynamic_robot_state(self, state):
        for key, value in state.items():
            topic = f"~/robot_state/{self._sanitize_topic_part(key)}"
            self._publish_dynamic_value(topic, value)

    def _publish_dynamic_camera_metadata(self, group, data):
        for cam_key, value in data.items():
            topic = f"~/{group}/{self._sanitize_topic_part(cam_key)}"
            self._publish_dynamic_value(topic, value)

    def _publish_dynamic_value(self, topic, value):
        numeric = self._numeric_list(value)
        if numeric is not None:
            if len(numeric) == 1:
                pub = self._numeric_publishers.get((topic, "scalar"))
                if pub is None:
                    pub = self.create_publisher(Float64, topic, 10)
                    self._numeric_publishers[(topic, "scalar")] = pub
                msg = Float64()
                msg.data = float(numeric[0])
                pub.publish(msg)
            else:
                pub = self._numeric_publishers.get((topic, "array"))
                if pub is None:
                    pub = self.create_publisher(Float64MultiArray, topic, 10)
                    self._numeric_publishers[(topic, "array")] = pub
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

    def _sanitize_topic_part(self, value: str) -> str:
        return "".join(char if char.isalnum() or char == "_" else "_" for char in str(value))

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
    rclpy.init(args=args)
    node = FrankaRobotEnvNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
