# DROID Extensions

This repository is based on the original [DROID robot platform](https://github.com/droid-dataset/droid) and adds extra utilities for local robot experiments. In addition to the upstream DROID data collection and robot environment code, this repo includes custom changes for Franka evaluation workflows, online policy experiments, and ROS2 integration.

The main added interface is a ROS2 `RobotEnv` node that keeps the DROID Franka environment alive as a long-running process, so other ROS2 nodes can control the robot without recreating `RobotEnv` for every evaluation.

---------
## Setup Guide

We assembled a step-by-step guide for setting up the DROID robot platform in our [developer documentation](https://droid-dataset.github.io/droid).
This guide has been used to set up 18 DROID robot platforms over the course of the DROID dataset collection. Please refer to the steps in this guide for setting up your own robot. Specifically, you can follow these key steps:

1. [Hardware Assembly and Setup](https://droid-dataset.github.io/droid/docs/hardware-setup)
2. [Software Installation and Setup](https://droid-dataset.github.io/droid/docs/software-setup)
3. [Example Workflows to collect data or calibrate cameras](https://droid-dataset.github.io/droid/docs/example-workflows)

If you encounter issues during setup, please raise them as issues in this github repo.

---------
## ROS2 RobotEnv Node

This repo also includes a small ROS2 package under `ros2/droid_robot_env` that keeps a DROID `RobotEnv` alive as a long-running node. This is useful when repeatedly evaluating policies: start the node once, then control the Franka from other ROS2 nodes without recreating `RobotEnv` every rollout.

Build it from the repository root:

```bash
source /opt/ros/humble/setup.bash
colcon build --base-paths ros2 --packages-select droid_robot_env
source install/setup.bash
```

Run the node:

```bash
ros2 run droid_robot_env franka_robot_env_node
```

The node wraps `droid.robot_env.RobotEnv` and exposes standard ROS2 interfaces:

- Subscribe: `/droid_ros/action` (`std_msgs/Float64MultiArray`)
- Subscribe: `/droid_ros/go_home` (`std_msgs/Bool`) to move the robot to the configured home/reset position
- Services: `/droid_ros/reset`, `/droid_ros/reconnect`, `/droid_ros/get_observation` (`std_srvs/Trigger`)
- Publish: `/droid_ros/joint_states`, `/droid_ros/cartesian_position`, `/droid_ros/gripper_position`, `/droid_ros/state_json`, `/droid_ros/action_info_json`, `/droid_ros/observation_meta_json`, `/droid_ros/observation_json`
- Publish: `/droid_ros/timestamp_json`, `/droid_ros/camera_type_json`, `/droid_ros/camera_intrinsics_json`, `/droid_ros/camera_extrinsics_json`, `/droid_ros/camera_alias_json`
- Publish: `/droid_ros/robot_state/<key>` for every DROID robot state key, plus `/droid_ros/camera_intrinsics/<camera>` and `/droid_ros/camera_extrinsics/<camera>` numeric topics when those values are numeric
- Publish: `/droid_ros/camera/<camera_alias>/image_raw`

Camera topics use aliases from `droid/misc/parameters.py`: the configured hand camera becomes `wrist`, and varied cameras become `side_view_1`, `side_view_2`, etc. Stereo ZED streams keep the side suffix, for example `wrist_left`, `wrist_right`, and `side_view_1_left`.

Common parameters:

- `action_space`: default `cartesian_velocity`
- `gripper_action_space`: default `position`
- `control_hz`: default `10.0`
- `publish_rate`: default `10.0`
- `do_reset`: default `False`
- `publish_cameras`: default `True`
- `camera_width`, `camera_height`: default `224`
- `enable_home_subscriber`: default `True`
- `home_topic`: default `~/go_home`
- `home_trigger_value`: default `True`
- `home_randomize`: default `False`

Example action command:

```bash
ros2 topic pub --once /droid_ros/action std_msgs/msg/Float64MultiArray "{data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}"
```

Example home/reset trigger:

```bash
ros2 topic pub --once /droid_ros/go_home std_msgs/msg/Bool "{data: true}"
```

ROS2 Humble uses Python 3.10. If `rclpy` import fails inside a conda Python, source the ROS2 environment and run with the ROS-compatible Python environment instead.

### LeRobot Dataset Collection Node

`droid_robot_env` also provides a dataset collection node that listens to the DROID ROS2 node and a Gello node. It caches the latest robot observations from DROID, records the latest Franka joint command from Gello as the action, and uses three Gello button topics to start an episode, mark success, or mark failure.

LeRobot is included in this repo's Python dependencies. On this machine, DROID is available in `/home/rllab2/miniconda3/envs/droid`, so installing DROID in that environment also installs the dataset dependency:

```bash
/home/rllab2/miniconda3/envs/droid/bin/python -m pip install -e .
```

```bash
ros2 run droid_robot_env lerobot_dataset_collector_node --ros-args \
  -p dataset_name:=my_task \
  -p dataset_root:=~/lerobot_datasets \
  -p language_instruction:="pick up the object" \
  -p hf_cache_dir:=~/lerobot_datasets/.hf_cache \
  -p camera_topics:="['/droid_ros/camera/wrist_left/image_raw','/droid_ros/camera/side_view_1_left/image_raw']"
```

Default input topics:

- DROID observations: `/droid_ros/joint_states`, `/droid_ros/cartesian_position`, `/droid_ros/gripper_position`, `/droid_ros/state_json`, `/droid_ros/action_info_json`, `/droid_ros/observation_meta_json`, `/droid_ros/observation_json`
- Gello action: `/gello/franka_joint_command` (`std_msgs/Float64MultiArray`)
- Gello buttons: `/gello/start_collection`, `/gello/mark_success`, `/gello/mark_failure` (`std_msgs/Bool`)

Useful LeRobot parameters:

- `lerobot_repo_id`: optional dataset id, default `local/<dataset_name>`
- `use_lerobot_native`: default `True`
- `use_lerobot_videos`: default `True`
- `hf_cache_dir`: optional Hugging Face cache directory for LeRobot metadata/cache files

Episode behavior:

- Start button rising edge begins recording.
- Success button rising edge saves the episode with `success=true`, sets the final reward to `1.0`, then calls `/droid_ros/reset`.
- Failure button rising edge saves the episode with `success=false`, leaves all rewards at `0.0`, then calls `/droid_ros/reset`.

The collector writes only a native LeRobot dataset under `<dataset_root>/<dataset_name>/lerobot`. Robot state, camera metadata, timestamps, action info, success/failure, reward, and full JSON snapshots are stored as LeRobot fields; camera streams are stored as LeRobot image/video fields.
