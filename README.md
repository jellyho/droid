# DROID Extensions

This repository is based on the original [DROID robot platform](https://github.com/droid-dataset/droid) and adds ROS2 integration on top of the existing DROID robot environment.

### Why ROS2?

DROID's `RobotEnv` is expensive to initialize — it connects to the Franka robot controller, opens ZED camera streams, and calibrates camera parameters. Every time you create a new `RobotEnv` instance, this entire setup runs again, which takes several seconds and interrupts the robot. This makes it impractical for workflows that need repeated rollouts (e.g., policy evaluation loops).

Wrapping `RobotEnv` in a long-running ROS2 node solves this: the node initializes once and stays alive, publishing observations and accepting actions over ROS2 topics. Any other process can control the robot by subscribing and publishing — no need to import DROID or re-initialize the hardware.

This also **isolates the Python environment**. DROID has specific dependency requirements (protobuf, ZED SDK, etc.) that can conflict with policy training frameworks. With the ROS2 interface, the DROID node runs in its own conda environment while policy nodes, data collectors, or visualization tools run in separate environments with their own dependencies. They communicate purely through ROS2 messages, so there are no import-level conflicts.

---------
## Setup Guide

We assembled a step-by-step guide for setting up the DROID robot platform in DROID's [developer documentation](https://droid-dataset.github.io/droid).
This guide has been used to set up 18 DROID robot platforms over the course of the DROID dataset collection. Please refer to the steps in this guide for setting up your own robot. Specifically, you can follow these key steps:

1. [Hardware Assembly and Setup](https://droid-dataset.github.io/droid/docs/hardware-setup)
2. [Software Installation and Setup](https://droid-dataset.github.io/droid/docs/software-setup)
3. [Example Workflows to collect data or calibrate cameras](https://droid-dataset.github.io/droid/docs/example-workflows)

---------
## ROS2 Installation

### ROS2 Humble

Install ROS2 Humble on Ubuntu 22.04 following the [official docs](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html):

```bash
sudo apt update && sudo apt install -y software-properties-common curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install -y ros-humble-desktop
```

### Required ROS2 packages

Core message types and Python client:

```bash
sudo apt install -y \
  ros-humble-rclpy \
  ros-humble-std-msgs \
  ros-humble-std-srvs \
  ros-humble-sensor-msgs
```

For compressed image transport (used by default for camera topics):

```bash
sudo apt install -y \
  ros-humble-image-transport \
  ros-humble-compressed-image-transport
```

### Python dependencies

Source the ROS2 setup before running any node:

```bash
source /opt/ros/humble/setup.bash
```

The nodes run directly with Python (no colcon build required). Make sure `rclpy` is importable in your conda environment. If it's not, source the ROS2 setup first — it adds the system ROS2 Python packages to the path:

```bash
source /opt/ros/humble/setup.bash
conda activate droid
python -c "import rclpy; print('rclpy OK')"
```

Install the DROID package and LeRobot dependency:

```bash
conda activate droid
pip install -e .
```

---------
## ROS2 RobotEnv Node

This repo also includes direct-run ROS2 node scripts under `ros2/droid_robot_env` that keep a DROID `RobotEnv` alive as a long-running process. This is useful when repeatedly evaluating policies: start the node once, then control the Franka from other ROS2 nodes without recreating `RobotEnv` every rollout.

Run the Franka node directly from the repository root:

```bash
source /opt/ros/humble/setup.bash
conda activate droid
python ros2/droid_robot_env/droid_robot_env/franka_robot_env_node.py
```

Run the LeRobot collector the same way:

```bash
source /opt/ros/humble/setup.bash
conda activate droid
python ros2/droid_robot_env/droid_robot_env/lerobot_dataset_collector_node.py
```

The Franka node loads `ros2/droid_robot_env/config/droid_robot_env.yaml`. The LeRobot collector loads `ros2/droid_robot_env/config/lerobot_dataset_collector.yaml`. Edit the relevant file and rerun the Python file; no build or launch step is needed.

The node wraps `droid.robot_env.RobotEnv` and exposes standard ROS2 interfaces:

- Subscribe: configured `action_topic`, for example `/gello/joint_command` (`std_msgs/Float64MultiArray`)
- Subscribe: configured `action_toggle_topic`, for example `/gello/switch/record` (`std_msgs/Bool`) to toggle action execution on/off
- Subscribe: `/droid_ros/go_home` (`std_msgs/Bool`) to move the robot to the configured home/reset position
- Services: `/droid_ros/reset`, `/droid_ros/reconnect`, `/droid_ros/get_observation` (`std_srvs/Trigger`)
- Publish: `/droid_ros/action_enabled`, `/droid_ros/joint_positions`, `/droid_ros/joint_velocities`, `/droid_ros/cartesian_position`, `/droid_ros/gripper_position`
- Publish: `/droid_ros/applied_action`, the exact action sent to `RobotEnv.step()` after enable/ramp handling
- Publish: `/droid_ros/robot_state/<key>` for every DROID robot state key
- Optionally publish: `/droid_ros/camera_intrinsics/<camera>` and `/droid_ros/camera_extrinsics/<camera>` when `publish_camera_metadata` is true
- Publish: `/droid_ros/camera/<camera_alias>/image_raw` when `camera_transport: raw`, or `/droid_ros/camera/<camera_alias>/image_raw/compressed` when `camera_transport: compressed`

Camera topics use aliases from `droid/misc/parameters.py`: `hand_camera_id` becomes `wrist`, `varied_camera_1_id` becomes `side_view_1`, and `varied_camera_2_id` becomes `side_view_2`. Unknown calibration serials are skipped. Stereo ZED streams keep the configured side suffix, for example `wrist_left` when `camera_side: left`.

Common parameters:

- `action_space`: default `cartesian_velocity`
- `gripper_action_space`: default `position`
- `control_hz`: default `10.0`; controls both RobotEnv control and ROS publish frequency
- `do_reset`: default `True`
- `publish_cameras`: default `True`
- `publish_camera_metadata`: default `False`
- `camera_side`: default `left`; use `right` or `both` if needed
- `camera_width`, `camera_height`: set to `none` for original camera resolution, or positive integers to resize
- `action_enabled`: default `False`
- `action_toggle_topic`: Bool topic whose arrival toggles action execution; the Bool value itself is ignored
- `ramp_action_on_enable`: default `True`; smooths the first commands after enabling by interpolating from the current robot state to the incoming target
- `action_ramp_duration_sec`: default `1.0`
- `enable_home_subscriber`: default `True`
- `action_topic`: configured in `ros2/droid_robot_env/config/droid_robot_env.yaml`
- `home_topic`: default `/droid_ros/go_home`
- `home_trigger_value`: default `True`
- `home_randomize`: default `False`

Example action command:

```bash
ros2 topic pub --once /gello/joint_command std_msgs/msg/Float64MultiArray "{data: [0.0, -0.628, 0.0, -2.513, 0.0, 1.885, 0.0, 0.0]}"
```

Example home/reset trigger:

```bash
ros2 topic pub --once /droid_ros/go_home std_msgs/msg/Bool "{data: true}"
```

Or source the local aliases and use the short command:

```bash
source ros2/droid_robot_env/scripts/aliases.sh
droid_home
```

ROS2 Humble uses Python 3.10. If `rclpy` import fails inside a conda Python, source the ROS2 environment and run with the ROS-compatible Python environment instead.

### OpenPI Remote Evaluation

For OpenPI evaluation, keep ROS2/DROID on the `droid` Python 3.10 environment and run the OpenPI policy as a separate websocket server. This avoids importing the full OpenPI model stack inside the ROS2 process.

Install the lightweight OpenPI client into the `droid` environment:

```bash
conda activate droid
python -m pip install -e /home/rllab2/jellyho/openpi/packages/openpi-client
python -m pip install numpy==2.2.6
```

The second command restores the NumPy version expected by OpenCV/ZED packages in this DROID environment. The OpenPI client declares `numpy<2`, but the websocket client path works with NumPy 2.x.

Verify the ROS client dependencies:

```bash
conda activate droid
python -c "import rclpy, cv2; from openpi_client import websocket_client_policy, image_tools; print('OpenPI ROS client OK')"
```

Start the policy server in a separate terminal:

```bash
cd /home/rllab2/jellyho/droid
./scripts/openpi/run_policy_server.sh
```

The server script defaults to:

- `OPENPI_ROOT=/home/rllab2/jellyho/openpi`
- `POLICY_CONFIG=pi05_droid_finetune_pressing`
- `CHECKPOINT_DIR=/home/rllab2/jellyho/checkpoints/pi05_droid_finetune_pressing/pressing_run/19999`
- `PORT=8000`

Override any of these as environment variables:

```bash
PORT=8001 CHECKPOINT_DIR=/path/to/checkpoint ./scripts/openpi/run_policy_server.sh
```

Start the interactive ROS2 evaluation client after the Franka node and policy server are running:

```bash
cd /home/rllab2/jellyho/droid
./scripts/openpi/run_ros_eval_client.sh
```

The evaluation client loads `ros2/droid_robot_env/config/openpi_inference.yaml`. For server mode, keep:

```yaml
policy_mode: "remote"
remote_host: "127.0.0.1"
remote_port: 8000
go_home_on_start: true
go_home_between_rollouts: true
go_home_wait_sec: 5.0
confirm_home: true
```

The client will send the robot home through `/droid_ros/go_home` when it starts, ask whether the robot is home, and retry homing if you answer `n` or `retry`. It then asks for a language instruction, optionally sends the robot home again between later rollouts, waits for Enter to start, shows a timestep progress bar, lets Enter interrupt the rollout, asks success/failure, saves an evaluation video, and asks whether to continue. Press Enter at the next language prompt to reuse the previous instruction.

### LeRobot Dataset Collection Node

`droid_robot_env` also provides a dataset collection node that listens to the DROID ROS2 node and a Gello node. It subscribes to configurable robot state topics and camera streams, records actions, and uses Gello buttons to control episode recording.

#### Collection workflow

1. Start the Franka robot env node.
2. Start the LeRobot collector node.
3. Press the **success button** to start recording an episode.
4. When done:
   - Press **success** again to save the episode with `success=true` and `reward=1.0` on the last step.
   - Press **failure** to save the episode with `success=false` and all rewards at `0.0`.
5. Repeat steps 3-4 for more episodes.
6. Press **Enter** in the collector terminal to finalize the dataset and exit.

If you restart the collector with the same dataset config, it resumes from the existing dataset (appending new episodes).

#### Topics

The collector subscribes to topics configured in `ros2/droid_robot_env/config/lerobot_dataset_collector.yaml`:

- `robot_state_topics`: list of `Float64MultiArray` topics. Each becomes its own LeRobot feature (e.g., `observation.joint_positions`). `observation.state` is the concatenation of all in config order.
- `applied_action_topic`: `Float64MultiArray` topic for the action vector.
- `camera_topics`: list of `CompressedImage` or `Image` topics.
- `success_topic`: `Bool` topic that toggles recording start/stop (saves as success).
- `failure_topic`: `Bool` topic that stops recording and saves as failure.

#### LeRobot parameters

- `dataset_name`: name of the dataset directory
- `dataset_root`: parent directory for datasets
- `lerobot_repo_id`: optional HuggingFace-style dataset id, default `local/<dataset_name>`
- `language_instruction`: task description string stored with each frame
- `fps`: recording frequency (should match `control_hz` of the Franka node)
- `use_lerobot_videos`: `true` to encode camera streams as video, `false` for individual images
- `hf_cache_dir`: optional Hugging Face cache directory
- `min_episode_steps`: minimum steps required to save an episode (shorter episodes are dropped)
