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

Run the Franka node directly from the repository root. It takes a required **mode** argument that selects where actions come from:

```bash
source /opt/ros/humble/setup.bash
conda activate droid

# teleop: the arm follows the Gello leader arm
python ros2/droid_robot_env/droid_robot_env/franka_robot_env_node.py record

# policy rollout: the arm follows /droid_ros/action
python ros2/droid_robot_env/droid_robot_env/franka_robot_env_node.py deploy
```

See [Action modes](#action-modes) for what each mode sets and how to add your own.

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
- `control_hz`: default `10.0`, set to `15.0` in the shipped config; controls both RobotEnv control and ROS publish frequency. A mode may override it
- `do_reset`: default `True`
- `publish_cameras`: default `True`
- `publish_camera_metadata`: default `False`
- `camera_side`: default `left`; use `right` or `both` if needed
- `camera_width`, `camera_height`: set to `none` for original camera resolution, or positive integers to resize
- `action_enabled`: set by the selected mode, not at the top level; see [Action modes](#action-modes)
- `action_toggle_topic`: set by the selected mode. Action execution toggles on *any* Bool arriving on this topic; the Bool value itself is ignored
- `ramp_action_on_enable`: default `True`; smooths the first commands after enabling by interpolating from the current robot state to the incoming target
- `action_ramp_duration_sec`: default `1.0`
- `enable_home_subscriber`: default `True`
- `action_topic`: set by the selected mode; see [Action modes](#action-modes)
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

#### Short commands

`ros2/droid_robot_env/scripts/aliases.sh` defines shell helpers for the commands above. Source it once per shell, after ROS2 and the `droid` environment are active:

```bash
source /opt/ros/humble/setup.bash
conda activate droid
source ros2/droid_robot_env/scripts/aliases.sh
```

| Command | Runs |
| --- | --- |
| `franka_ros <mode>` | the Franka RobotEnv node in `record` or `deploy` mode |
| `droid_collect` | the LeRobot dataset collector |
| `droid_home` | publishes `true` to `/droid_ros/go_home` |

`franka_ros` and `droid_collect` forward any extra arguments to the node, so `franka_ros record --ros-args -p control_hz:=30.0` works. The script locates the repository from its own path, so it can be sourced from any directory.

ROS2 Humble uses Python 3.10. If `rclpy` import fails inside a conda Python, source the ROS2 environment and run with the ROS-compatible Python environment instead.

### Action modes

`franka_robot_env_node.py` requires a mode argument. Modes live in the `modes:` block of `ros2/droid_robot_env/config/droid_robot_env.yaml`, and each one overrides a few action parameters on top of the shared settings in the same file:

```yaml
droid_ros:
  ros__parameters:
    control_hz: 15.0          # shared by every mode
    action_space: joint_position
    # ...

    modes:
      record:
        action_enabled: false
        action_topic: /gello/joint_command
        action_toggle_topic: /gello/switch/record
      deploy:
        action_enabled: true
        action_topic: /droid_ros/action
        action_toggle_topic: /droid_ros/action_toggle
```

| Mode | `action_topic` | `action_toggle_topic` | `action_enabled` | Use |
| --- | --- | --- | --- | --- |
| `record` | `/gello/joint_command` | `/gello/switch/record` | `false` | Gello teleop and dataset collection |
| `deploy` | `/droid_ros/action` | `/droid_ros/action_toggle` | `true` | OpenPI policy rollout |

`record` starts with actions disabled, so the arm holds still until the Gello record button engages it. `deploy` starts enabled, so the inference node can drive the arm as soon as it connects.

A mode may only set `action_enabled`, `action_topic`, `action_toggle_topic` and `control_hz`. Everything else is shared and belongs at the top level of the file. To add a mode, add a block under `modes:` defining at least the three action keys.

The mode is resolved before `rclpy.init()`, so mistakes stop the node rather than half-wiring it:

- no mode, or an unknown mode, prints the modes defined in the config and exits `2`
- a mode setting a key outside the four allowed names exits `2` and names the key
- a mode missing any of the three required action keys exits `2` and names them

Mode overrides are applied before your own arguments, so an explicit override still wins:

```bash
franka_ros record --ros-args -p control_hz:=30.0
```

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

#### Recording with Gello

Start the Franka node in `record` mode. Nothing else in the config needs editing:

```bash
franka_ros record
```

That mode points the action subscriber at `/gello/joint_command` and the toggle at `/gello/switch/record`, and starts with actions disabled so the arm stays put until you press the Gello record button. See [Action modes](#action-modes) for the full definition.

Keep `control_hz` in `droid_robot_env.yaml` equal to `fps` in `lerobot_dataset_collector.yaml` — both are `15.0`. If they diverge, recorded frames no longer land at the dataset's declared frame rate.

#### Gello buttons

The Gello driver runs on a separate machine (`gello_ros2_franka/gello_driver`, configured in its `config/gello_params.yaml`). Its three physical buttons are Jetson GPIO pins published as `std_msgs/Bool`, one `true` pulse per press on the falling edge, with a 250 ms debounce:

| Button | Pin | Topic | Consumed by |
| --- | --- | --- | --- |
| record | 7 | `/gello/switch/record` | Franka node — toggles action execution on/off |
| success | 11 | `/gello/switch/success` | collector — starts recording, then saves as success |
| failure | 13 | `/gello/switch/failed` | collector — stops recording, saves as failure |

The record button goes straight to the Franka node, not through the collector. That is why the `record` mode sets `action_toggle_topic` to `/gello/switch/record` — in `deploy` mode the Gello record button does nothing, because the node is listening on `/droid_ros/action_toggle` instead.

The driver also publishes the leader arm pose as `std_msgs/Float64MultiArray` on `/gello/joint_command`, which is the message type the Franka node's action subscriber expects.

#### Collection workflow

1. Start the Franka robot env node.
2. Start the LeRobot collector node.
3. Press **record** on Gello to engage the arm. It ramps from its current pose to the leader arm pose over `action_ramp_duration_sec` (1 second), then follows continuously.
4. Press the **success button** to start recording an episode.
5. When done:
   - Press **success** again to save the episode with `success=true` and `reward=1.0` on the last step.
   - Press **failure** to save the episode with `success=false` and all rewards at `0.0`.
6. Press **record** again to disengage the arm, then home it if you want a fresh start pose:
   ```bash
   ros2 topic pub --once /droid_ros/go_home std_msgs/msg/Bool "{data: true}"
   ```
7. Repeat steps 3-6 for more episodes.
8. Press **Enter** in the collector terminal to finalize the dataset and exit.

Steps 3 and 6 are manual, and this is a change in behaviour. The collector previously engaged action execution when recording started, and disengaged plus homed the robot when an episode finished. It no longer does either, so engaging the arm and homing between episodes are yours to drive. The upside is that arm engagement is now independent of recording: you can move the arm into position, then start the episode.

If you restart the collector with the same dataset config, it resumes from the existing dataset (appending new episodes).

#### Topics

The collector subscribes to topics configured in `ros2/droid_robot_env/config/lerobot_dataset_collector.yaml`:

- `robot_state_topics`: list of `Float64MultiArray` topics. Each becomes its own LeRobot feature (e.g., `observation.joint_positions`). `observation.state` is the concatenation of all in config order.
- `applied_action_topic`: `Float64MultiArray` topic for the action vector.
- `camera_topics`: list of `CompressedImage` or `Image` topics.
- `success_topic`: `Bool` topic that toggles recording start/stop (saves as success).
- `failure_topic`: `Bool` topic that stops recording and saves as failure.

`action_toggle_topic` and `home_topic` are still declared in the collector config, and the collector still creates publishers for them, but nothing calls them any more. Both keys are currently inert — set them however you like and the collector will not drive the robot. Engage the arm and home it as described in the workflow above.

#### LeRobot parameters

- `dataset_name`: name of the dataset directory
- `dataset_root`: parent directory for datasets
- `lerobot_repo_id`: optional HuggingFace-style dataset id, default `local/<dataset_name>`
- `language_instruction`: task description string stored with each frame
- `fps`: recording frequency (should match `control_hz` of the Franka node)
- `use_lerobot_videos`: `true` to encode camera streams as video, `false` for individual images
- `hf_cache_dir`: optional Hugging Face cache directory
- `min_episode_steps`: minimum steps required to save an episode (shorter episodes are dropped)
