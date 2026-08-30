#!/usr/bin/env bash
#
# Short commands for the droid_robot_env ROS2 nodes.
#
#   source ros2/droid_robot_env/scripts/aliases.sh
#
# Assumes ROS2 and the `droid` Python environment are already active:
#   source /opt/ros/humble/setup.bash && conda activate droid

_droid_repo_root() {
  local src="${BASH_SOURCE[0]:-$0}"
  cd "$(dirname "$src")/../../.." && pwd
}

DROID_REPO_ROOT="${DROID_REPO_ROOT:-$(_droid_repo_root)}"
export DROID_REPO_ROOT

# Franka RobotEnv node. Takes the action mode: record | deploy
# See the "Action modes" section of the README.
franka_ros() {
  python "$DROID_REPO_ROOT/ros2/droid_robot_env/droid_robot_env/franka_robot_env_node.py" "$@"
}

# LeRobot dataset collector.
droid_collect() {
  python "$DROID_REPO_ROOT/ros2/droid_robot_env/droid_robot_env/lerobot_dataset_collector_node.py" "$@"
}

# Send the robot to its home/reset pose.
droid_home() {
  ros2 topic pub --once /droid_ros/go_home std_msgs/msg/Bool "{data: true}"
}
