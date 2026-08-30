"""ROS2 node that runs OpenPI policy inference locally on the DROID robot.

Loads an OpenPI policy (pi0, pi0.5, pi0_fast) directly and runs inference
on observations from FrankaRobotEnvNode — no websocket server needed.
"""

import sys
import threading
import time
import traceback
import json
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, Float64MultiArray, String


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "openpi_inference.yaml"


def _with_repo_params_file(args):
    argv = list(sys.argv if args is None else args)
    config_path = _default_config_path()
    if not config_path.exists():
        raise FileNotFoundError(f"Default ROS2 config not found: {config_path}")
    print(f"[openpi_inference_node] loading config: {config_path}", flush=True)
    if "--ros-args" in argv:
        idx = argv.index("--ros-args")
        return argv[:idx + 1] + ["--params-file", str(config_path)] + argv[idx + 1:]
    return [argv[0], "--ros-args", "--params-file", str(config_path)] + argv[1:]


class OpenPIInferenceNode(Node):
    """ROS2 node that loads an OpenPI policy locally and runs inference.

    Subscribes to:
      - /droid_ros/joint_positions (Float64MultiArray)
      - /droid_ros/gripper_position (Float64MultiArray)
      - /droid_ros/camera/<exterior_cam>/image_raw/compressed (CompressedImage)
      - /droid_ros/camera/<wrist_cam>/image_raw/compressed (CompressedImage)

    Publishes:
      - /droid_ros/action (Float64MultiArray)
    """

    def __init__(self):
        super().__init__("openpi_inference")
        self._lock = threading.Lock()

        # Declare parameters
        self.declare_parameter("policy_config", "pi05_droid")
        self.declare_parameter("checkpoint_dir", "")
        self.declare_parameter("pytorch_device", "cuda")
        self.declare_parameter("policy_mode", "remote")
        self.declare_parameter("remote_host", "127.0.0.1")
        self.declare_parameter("remote_port", 8000)
        self.declare_parameter("instruction", "")
        self.declare_parameter("exterior_camera_alias", "side_view_1_left")
        self.declare_parameter("wrist_camera_alias", "wrist_left")
        self.declare_parameter("open_loop_horizon", 8)
        self.declare_parameter("control_hz", 15.0)
        self.declare_parameter("max_timesteps", 600)
        self.declare_parameter("image_width", 224)
        self.declare_parameter("image_height", 224)
        self.declare_parameter("camera_transport", "compressed")
        self.declare_parameter("topic_prefix", "/droid_ros")
        self.declare_parameter("action_topic", "/droid_ros/action")
        self.declare_parameter("action_toggle_topic", "/droid_ros/action_toggle")
        self.declare_parameter("home_topic", "/droid_ros/go_home")
        self.declare_parameter("go_home_on_start", True)
        self.declare_parameter("go_home_between_rollouts", True)
        self.declare_parameter("go_home_wait_sec", 5.0)
        self.declare_parameter("confirm_home", True)
        self.declare_parameter("binarize_gripper", True)
        self.declare_parameter("gripper_threshold", 0.5)
        self.declare_parameter("eval_video_dir", "~/jellyho/openpi_eval_videos")
        self.declare_parameter("eval_video_fps", 10.0)
        self.declare_parameter("clip_actions", True)
        self.declare_parameter("auto_start", False)

        # Read parameters
        self._policy_config_name = self.get_parameter("policy_config").value
        self._checkpoint_dir = self.get_parameter("checkpoint_dir").value
        self._pytorch_device = self.get_parameter("pytorch_device").value
        self._policy_mode = self.get_parameter("policy_mode").value
        self._remote_host = self.get_parameter("remote_host").value
        self._remote_port = int(self.get_parameter("remote_port").value)
        self._instruction = self.get_parameter("instruction").value
        self._exterior_cam = self.get_parameter("exterior_camera_alias").value
        self._wrist_cam = self.get_parameter("wrist_camera_alias").value
        self._open_loop_horizon = self.get_parameter("open_loop_horizon").value
        self._control_hz = float(self.get_parameter("control_hz").value)
        self._max_timesteps = self.get_parameter("max_timesteps").value
        self._img_w = self.get_parameter("image_width").value
        self._img_h = self.get_parameter("image_height").value
        self._camera_transport = self.get_parameter("camera_transport").value
        self._topic_prefix = self.get_parameter("topic_prefix").value
        self._action_topic = self.get_parameter("action_topic").value
        self._action_toggle_topic = self.get_parameter("action_toggle_topic").value
        self._home_topic = self.get_parameter("home_topic").value
        self._go_home_on_start = bool(self.get_parameter("go_home_on_start").value)
        self._go_home_between_rollouts = bool(self.get_parameter("go_home_between_rollouts").value)
        self._go_home_wait_sec = float(self.get_parameter("go_home_wait_sec").value)
        self._confirm_home = bool(self.get_parameter("confirm_home").value)
        self._binarize_gripper = self.get_parameter("binarize_gripper").value
        self._gripper_threshold = self.get_parameter("gripper_threshold").value
        self._eval_video_dir = Path(self.get_parameter("eval_video_dir").value).expanduser()
        self._eval_video_fps = float(self.get_parameter("eval_video_fps").value)
        self._clip_actions = bool(self.get_parameter("clip_actions").value)

        # State
        self._joint_positions = None  # (7,)
        self._gripper_position = None  # (1,)
        self._exterior_image = None  # (H, W, 3) uint8
        self._wrist_image = None  # (H, W, 3) uint8
        self._running = False
        self._inference_thread = None
        self._stop_event = threading.Event()
        self._policy = None
        self._episode_count = 0
        self._current_step = 0
        self._episode_video_frames = []
        self._last_episode_steps = 0
        self._last_episode_interrupted = False
        self._last_episode_error = None

        # Publishers
        self.action_pub = self.create_publisher(
            Float64MultiArray, self._action_topic, 10
        )
        self.action_toggle_pub = self.create_publisher(
            Bool, self._action_toggle_topic, 10
        )
        self.home_pub = self.create_publisher(Bool, self._home_topic, 10)
        self.status_pub = self.create_publisher(
            String, self._topic("inference_status"), 10
        )

        # Subscribers — robot state
        self.joint_sub = self.create_subscription(
            Float64MultiArray,
            self._topic("joint_positions"),
            self._on_joint_positions,
            10,
        )
        self.gripper_sub = self.create_subscription(
            Float64MultiArray,
            self._topic("gripper_position"),
            self._on_gripper_position,
            10,
        )

        # Subscribers — cameras
        self._setup_camera_subscribers()

        # Subscriber — start/stop trigger
        self.start_sub = self.create_subscription(
            String,
            self._topic("inference_command"),
            self._on_command,
            10,
        )

        # Status heartbeat
        self.status_timer = self.create_timer(1.0, self._publish_status)

        self.get_logger().info(
            f"OpenPI Inference node initialized (local mode). "
            f"Policy mode: {self._policy_mode}, "
            f"Policy config: {self._policy_config_name}, "
            f"Checkpoint: {self._checkpoint_dir or 'default'}, "
            f"Device: {self._pytorch_device}, Remote: {self._remote_host}:{self._remote_port}, "
            f"Exterior cam: {self._exterior_cam}, Wrist cam: {self._wrist_cam}, "
            f"Control Hz: {self._control_hz}, Open-loop horizon: {self._open_loop_horizon}"
        )

        if self.get_parameter("auto_start").value:
            self.get_logger().info("auto_start enabled; loading policy for interactive evaluation...")
            self._load_policy()

    def _topic(self, suffix: str) -> str:
        prefix = self._topic_prefix.rstrip("/")
        if not prefix.startswith("/"):
            prefix = f"/{prefix}"
        return f"{prefix}/{suffix.lstrip('/')}"

    def _setup_camera_subscribers(self):
        transport = self._camera_transport.lower()
        if transport == "compressed":
            ext_topic = self._topic(f"camera/{self._exterior_cam}/image_raw/compressed")
            wrist_topic = self._topic(f"camera/{self._wrist_cam}/image_raw/compressed")
            self.ext_cam_sub = self.create_subscription(
                CompressedImage, ext_topic, self._on_exterior_image_compressed, 5
            )
            self.wrist_cam_sub = self.create_subscription(
                CompressedImage, wrist_topic, self._on_wrist_image_compressed, 5
            )
            self.get_logger().info(f"Subscribed to compressed cameras: {ext_topic}, {wrist_topic}")
        else:
            ext_topic = self._topic(f"camera/{self._exterior_cam}/image_raw")
            wrist_topic = self._topic(f"camera/{self._wrist_cam}/image_raw")
            self.ext_cam_sub = self.create_subscription(
                Image, ext_topic, self._on_exterior_image_raw, 5
            )
            self.wrist_cam_sub = self.create_subscription(
                Image, wrist_topic, self._on_wrist_image_raw, 5
            )
            self.get_logger().info(f"Subscribed to raw cameras: {ext_topic}, {wrist_topic}")

    # ── Camera callbacks ──────────────────────────────────────────────

    def _compressed_to_rgb(self, msg: CompressedImage) -> np.ndarray:
        import cv2
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _raw_to_rgb(self, msg: Image) -> np.ndarray:
        import cv2
        if msg.encoding in ("bgr8", "bgra8"):
            channels = 4 if msg.encoding == "bgra8" else 3
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)
            return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB if channels == 3 else cv2.COLOR_BGRA2RGB)
        elif msg.encoding in ("rgb8", "rgba8"):
            channels = 4 if msg.encoding == "rgba8" else 3
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)
            return arr[..., :3]
        else:
            channels = max(1, int(msg.step / max(1, msg.width)))
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)
            return arr[..., :3] if channels >= 3 else np.stack([arr[..., 0]] * 3, axis=-1)

    def _on_exterior_image_compressed(self, msg: CompressedImage):
        with self._lock:
            self._exterior_image = self._compressed_to_rgb(msg)

    def _on_wrist_image_compressed(self, msg: CompressedImage):
        with self._lock:
            self._wrist_image = self._compressed_to_rgb(msg)

    def _on_exterior_image_raw(self, msg: Image):
        with self._lock:
            self._exterior_image = self._raw_to_rgb(msg)

    def _on_wrist_image_raw(self, msg: Image):
        with self._lock:
            self._wrist_image = self._raw_to_rgb(msg)

    # ── Robot state callbacks ─────────────────────────────────────────

    def _on_joint_positions(self, msg: Float64MultiArray):
        data = np.asarray(msg.data, dtype=np.float64)
        with self._lock:
            self._joint_positions = data[:7]

    def _on_gripper_position(self, msg: Float64MultiArray):
        data = np.asarray(msg.data, dtype=np.float64)
        with self._lock:
            self._gripper_position = data[:1]

    # ── Command interface ─────────────────────────────────────────────

    def _on_command(self, msg: String):
        """Handle start/stop/instruction commands.

        Publish to /droid_ros/inference_command:
          - "start"  or "start:<instruction>" — begin inference loop
          - "stop"   — stop the current episode
          - "instruction:<text>" — update the language instruction
        """
        cmd = msg.data.strip()
        if cmd.startswith("start"):
            parts = cmd.split(":", 1)
            if len(parts) > 1 and parts[1].strip():
                self._instruction = parts[1].strip()
                self.get_logger().info(f"Instruction updated: {self._instruction}")
            self._start_inference()
        elif cmd == "stop":
            self._stop_inference()
        elif cmd.startswith("instruction:"):
            self._instruction = cmd.split(":", 1)[1].strip()
            self.get_logger().info(f"Instruction updated: {self._instruction}")
        else:
            self.get_logger().warning(f"Unknown command: {cmd}")

    # ── Local policy loading ──────────────────────────────────────────

    def _load_policy(self):
        if self._policy is not None:
            return
        if str(self._policy_mode).lower() in ("remote", "websocket", "server"):
            self._load_remote_policy()
            return
        if str(self._policy_mode).lower() not in ("local", "direct"):
            self.get_logger().error(f"Unknown policy_mode={self._policy_mode}; expected local or remote.")
            return

        try:
            from openpi.training import config as train_config
            from openpi.policies import policy_config
            from openpi.shared import download

            config_name = self._policy_config_name
            self.get_logger().info(f"Loading OpenPI config: {config_name}")
            config = train_config.get_config(config_name)

            checkpoint_dir = self._checkpoint_dir
            if not checkpoint_dir:
                default_gcs = f"gs://openpi-assets/checkpoints/{config_name}"
                self.get_logger().info(
                    f"No checkpoint_dir specified, downloading default: {default_gcs}"
                )
                checkpoint_dir = download.maybe_download(default_gcs)
            else:
                checkpoint_dir = str(Path(checkpoint_dir).expanduser())

            self.get_logger().info(
                f"Creating policy from checkpoint: {checkpoint_dir} "
                f"(device={self._pytorch_device})"
            )
            self._policy = policy_config.create_trained_policy(
                config,
                checkpoint_dir,
                pytorch_device=self._pytorch_device,
            )
            self.get_logger().info("Policy loaded successfully.")

            # Warmup inference
            self.get_logger().info("Running warmup inference...")
            dummy_obs = {
                "observation/exterior_image_1_left": np.zeros(
                    (self._img_h, self._img_w, 3), dtype=np.uint8
                ),
                "observation/wrist_image_left": np.zeros(
                    (self._img_h, self._img_w, 3), dtype=np.uint8
                ),
                "observation/joint_position": np.zeros(7, dtype=np.float64),
                "observation/gripper_position": np.zeros(1, dtype=np.float64),
                "prompt": "warmup",
            }
            result = self._policy.infer(dummy_obs)
            self.get_logger().info(
                f"Warmup done. Action shape: {result['actions'].shape}, "
                f"Inference time: {result.get('policy_timing', {}).get('infer_ms', '?')}ms"
            )
        except Exception as exc:
            self.get_logger().error(f"Failed to load policy: {exc}\n{traceback.format_exc()}")
            self._policy = None

    def _load_remote_policy(self):
        try:
            from openpi_client import websocket_client_policy

            self.get_logger().info(
                f"Connecting to OpenPI policy server at {self._remote_host}:{self._remote_port}..."
            )
            self._policy = websocket_client_policy.WebsocketClientPolicy(
                self._remote_host,
                self._remote_port,
            )
            metadata = self._policy.get_server_metadata()
            self.get_logger().info(f"Connected to OpenPI policy server. metadata={metadata}")
        except Exception as exc:
            self.get_logger().error(f"Failed to connect to remote policy: {exc}\n{traceback.format_exc()}")
            self._policy = None

    # ── Inference loop ────────────────────────────────────────────────

    def _start_inference(self):
        if self._running:
            self.get_logger().warning("Inference already running.")
            return
        if not self._instruction:
            self.get_logger().error(
                "No instruction set. Use 'start:<instruction>' or set instruction param."
            )
            return

        self._load_policy()
        if self._policy is None:
            self.get_logger().error("Cannot start: policy not loaded.")
            return

        self._stop_event.clear()
        self._running = True
        self._inference_thread = threading.Thread(
            target=self._inference_loop, name="openpi_inference", daemon=True
        )
        self._inference_thread.start()
        self.get_logger().info(f"Inference started. Instruction: '{self._instruction}'")

    def _stop_inference(self):
        if not self._running:
            return
        self.get_logger().info("Stopping inference...")
        self._stop_event.set()
        if self._inference_thread is not None:
            self._inference_thread.join(timeout=5.0)
        self._running = False
        self.get_logger().info("Inference stopped.")

    def _observation_ready(self) -> bool:
        return (
            self._joint_positions is not None
            and self._gripper_position is not None
            and self._exterior_image is not None
            and self._wrist_image is not None
        )

    def _get_current_observation(self) -> dict:
        from openpi_client import image_tools

        with self._lock:
            exterior_img = self._exterior_image.copy()
            wrist_img = self._wrist_image.copy()
            joint_pos = self._joint_positions.copy()
            gripper_pos = self._gripper_position.copy()

        exterior_img = image_tools.resize_with_pad(exterior_img, self._img_h, self._img_w)
        wrist_img = image_tools.resize_with_pad(wrist_img, self._img_h, self._img_w)

        return {
            "observation/exterior_image_1_left": exterior_img,
            "observation/wrist_image_left": wrist_img,
            "observation/joint_position": joint_pos,
            "observation/gripper_position": gripper_pos,
            "prompt": self._instruction,
        }

    def _process_action(self, raw_action: np.ndarray) -> np.ndarray:
        action = raw_action.copy()
        if self._binarize_gripper:
            gripper_val = 1.0 if action[-1] > self._gripper_threshold else 0.0
            action = np.concatenate([action[:-1], [gripper_val]])
        if self._clip_actions:
            action = np.clip(action, -1.0, 1.0)
        return action

    def _publish_action(self, action: np.ndarray):
        msg = Float64MultiArray()
        msg.data = [float(x) for x in action.reshape(-1)]
        self.action_pub.publish(msg)

    def request_home(self):
        msg = Bool()
        msg.data = True
        self.home_pub.publish(msg)
        self.get_logger().info(f"Published home request on {self._home_topic}")

    def _video_frame_from_observation(self, obs: dict) -> np.ndarray:
        exterior = obs["observation/exterior_image_1_left"]
        wrist = obs["observation/wrist_image_left"]
        height = min(exterior.shape[0], wrist.shape[0])
        if exterior.shape[0] != height or wrist.shape[0] != height:
            import cv2

            def resize_to_height(image):
                width = max(1, int(round(image.shape[1] * height / image.shape[0])))
                return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

            exterior = resize_to_height(exterior)
            wrist = resize_to_height(wrist)
        return np.concatenate([exterior[..., :3], wrist[..., :3]], axis=1)

    def _inference_loop(self):
        try:
            self._episode_count += 1
            self._current_step = 0
            self._last_episode_steps = 0
            self._last_episode_interrupted = False
            self._last_episode_error = None
            self._episode_video_frames = []

            self.get_logger().info("Waiting for observation data...")
            while not self._stop_event.is_set():
                if self._observation_ready():
                    break
                time.sleep(0.05)
            if self._stop_event.is_set():
                return

            self.get_logger().info("Observations ready. Running policy inference loop.")
            period = 1.0 / self._control_hz

            actions_executed = 0
            action_chunk = None

            for t_step in range(self._max_timesteps):
                if self._stop_event.is_set():
                    break

                step_start = time.time()
                self._current_step = t_step
                obs = self._get_current_observation()
                self._episode_video_frames.append(self._video_frame_from_observation(obs))

                if action_chunk is None or actions_executed >= self._open_loop_horizon:
                    try:
                        infer_start = time.time()
                        result = self._policy.infer(obs)
                        action_chunk = result["actions"]
                        infer_ms = (time.time() - infer_start) * 1000
                        actions_executed = 0
                        if t_step % 50 == 0:
                            self.get_logger().info(
                                f"Step {t_step}: inference {infer_ms:.1f}ms, "
                                f"action_chunk shape {action_chunk.shape}"
                            )
                    except Exception as exc:
                        self.get_logger().error(f"Policy inference failed: {exc}")
                        break

                raw_action = action_chunk[actions_executed]
                action = self._process_action(raw_action)
                self._publish_action(action)
                actions_executed += 1

                elapsed = time.time() - step_start
                sleep_time = period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            self._last_episode_interrupted = self._stop_event.is_set()
            self._last_episode_steps = self._current_step + 1
            self.get_logger().info(
                f"Episode {self._episode_count} finished after {self._last_episode_steps} steps."
            )
        except Exception as exc:
            self._last_episode_error = str(exc)
            self.get_logger().error(f"Inference loop error:\n{traceback.format_exc()}")
        finally:
            self._running = False

    def save_last_episode_video(self, success: bool, instruction: str) -> Path | None:
        frames = list(self._episode_video_frames)
        if not frames:
            self.get_logger().warning("No video frames recorded for the last episode.")
            return None

        import cv2

        self._eval_video_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        label = "success" if success else "failure"
        stem = f"episode_{self._episode_count:04d}_{label}_{timestamp}"
        video_path = self._eval_video_dir / f"{stem}.mp4"
        meta_path = self._eval_video_dir / f"{stem}.json"

        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(1e-6, self._eval_video_fps),
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {video_path}")

        try:
            for frame in frames:
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2BGR))
        finally:
            writer.release()

        metadata = {
            "episode": self._episode_count,
            "success": bool(success),
            "instruction": instruction,
            "steps": int(self._last_episode_steps),
            "max_timesteps": int(self._max_timesteps),
            "interrupted": bool(self._last_episode_interrupted),
            "error": self._last_episode_error,
            "policy_config": self._policy_config_name,
            "checkpoint_dir": self._checkpoint_dir,
            "video_path": str(video_path),
        }
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
        self.get_logger().info(f"Saved eval video: {video_path}")
        return video_path

    # ── Status ────────────────────────────────────────────────────────

    def _publish_status(self):
        obs_status = "ready" if self._observation_ready() else "waiting"
        policy_status = "loaded" if self._policy is not None else "not_loaded"
        msg = String()
        msg.data = (
            f"running={self._running}, obs={obs_status}, policy={policy_status}, "
            f"mode={self._policy_mode}, device={self._pytorch_device}, config={self._policy_config_name}, "
            f"episode={self._episode_count}, step={self._current_step}, "
            f"instruction='{self._instruction}'"
        )
        self.status_pub.publish(msg)


def main(args=None):
    args = _with_repo_params_file(args)
    rclpy.init(args=args)
    node = None
    shutdown_event = threading.Event()
    try:
        print("Starting OpenPI Inference ROS2 node (local mode)...", flush=True)
        node = OpenPIInferenceNode()

        def _interactive_eval_loop():
            _run_interactive_eval(node, shutdown_event)

        eval_thread = threading.Thread(target=_interactive_eval_loop, name="openpi_eval_cli", daemon=True)
        eval_thread.start()

        while rclpy.ok() and not shutdown_event.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    except Exception:
        if node is not None:
            node.get_logger().fatal(traceback.format_exc())
        else:
            print(traceback.format_exc(), flush=True)
        raise
    finally:
        shutdown_event.set()
        if node is not None:
            node._stop_event.set()
            if node._inference_thread is not None:
                node._inference_thread.join(timeout=2.0)
            node.destroy_node()
        rclpy.shutdown()


def _prompt_success() -> bool:
    while True:
        answer = input("Was this rollout successful? [y/n] ").strip().lower()
        if answer in ("y", "yes", "s", "success"):
            return True
        if answer in ("n", "no", "f", "failure", "fail"):
            return False
        print("Please answer y or n.", flush=True)


def _prompt_continue() -> bool:
    answer = input("Continue evaluation? [Y/n] ").strip().lower()
    return answer not in ("n", "no", "q", "quit", "finish", "done")


def _run_interactive_eval(node: OpenPIInferenceNode, shutdown_event: threading.Event):
    print("\n[openpi_eval] Loading policy before the first rollout...", flush=True)
    node._load_policy()
    if node._policy is None:
        print("[openpi_eval] Policy failed to load. See ROS logs above.", flush=True)
        shutdown_event.set()
        return

    _go_home_on_start_if_configured(node)
    previous_instruction = node._instruction.strip()
    rollout_count = 0
    while not shutdown_event.is_set():
        if previous_instruction:
            prompt = f"\nLanguage instruction [{previous_instruction}]: "
            entered = input(prompt).strip()
            instruction = entered or previous_instruction
        else:
            instruction = ""
            while not instruction:
                instruction = input("\nLanguage instruction: ").strip()

        previous_instruction = instruction
        node._instruction = instruction

        if rollout_count > 0:
            _go_home_between_rollouts_if_configured(node)
        input("Press Enter to start evaluation.")
        node._start_inference()
        if not node._running:
            print("[openpi_eval] Rollout did not start. Check the logs above.", flush=True)
            if not _prompt_continue():
                break
            continue

        _monitor_rollout(node)
        node._stop_inference()

        success = _prompt_success()
        try:
            video_path = node.save_last_episode_video(success, instruction)
            if video_path is not None:
                print(f"Saved video: {video_path}", flush=True)
        except Exception as exc:
            print(f"Failed to save video: {exc}", flush=True)

        if not _prompt_continue():
            break
        rollout_count += 1

    shutdown_event.set()


def _go_home_on_start_if_configured(node: OpenPIInferenceNode):
    if not node._go_home_on_start:
        return
    print("Sending robot home on eval client start.", flush=True)
    _request_home_until_confirmed(node)


def _go_home_between_rollouts_if_configured(node: OpenPIInferenceNode):
    if not node._go_home_between_rollouts:
        return
    answer = input("Send robot home before this rollout? [Y/n] ").strip().lower()
    if answer in ("n", "no", "skip"):
        return
    _request_home_until_confirmed(node)


def _request_home_until_confirmed(node: OpenPIInferenceNode):
    while True:
        node.request_home()
        wait_sec = max(0.0, float(node._go_home_wait_sec))
        if wait_sec > 0:
            print(f"Waiting {wait_sec:.1f}s for home motion...", flush=True)
            time.sleep(wait_sec)
        if not node._confirm_home:
            return
        answer = input("Is the robot home? [Y/n/retry] ").strip().lower()
        if answer in ("", "y", "yes"):
            return
        if answer in ("n", "no", "r", "retry", "again"):
            print("Retrying home request.", flush=True)
            continue
        print("Please answer y, n, or retry.", flush=True)


def _monitor_rollout(node: OpenPIInferenceNode):
    import select

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    print("Evaluation running. Press Enter to interrupt and finish this rollout.", flush=True)
    if tqdm is None:
        while node._running:
            readable, _, _ = select.select([sys.stdin], [], [], 0.2)
            if readable:
                sys.stdin.readline()
                node._stop_event.set()
                break
            print(f"\rstep {node._current_step + 1}/{node._max_timesteps}", end="", flush=True)
        print("", flush=True)
        return

    last_step = 0
    with tqdm(total=int(node._max_timesteps), desc="OpenPI eval", unit="step") as bar:
        while node._running:
            current = min(int(node._current_step) + 1, int(node._max_timesteps))
            if current > last_step:
                bar.update(current - last_step)
                last_step = current
            readable, _, _ = select.select([sys.stdin], [], [], 0.2)
            if readable:
                sys.stdin.readline()
                node._stop_event.set()
                break
        current = min(int(node._current_step) + 1, int(node._max_timesteps))
        if current > last_step:
            bar.update(current - last_step)


if __name__ == "__main__":
    main()
