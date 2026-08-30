#!/usr/bin/env bash
set -euo pipefail

DROID_ROOT="${DROID_ROOT:-/home/rllab2/jellyho/droid}"
OPENPI_ROOT="${OPENPI_ROOT:-/home/rllab2/jellyho/openpi}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
PYTHON_BIN="${PYTHON_BIN:-/home/rllab2/miniconda3/envs/droid/bin/python}"
EVAL_VIDEO_DIR="${EVAL_VIDEO_DIR:-/home/rllab2/jellyho/openpi_eval_videos/c00_s10}"

set +u
source "$ROS_SETUP"
set -u
cd "$DROID_ROOT"

export PYTHONPATH="$OPENPI_ROOT/packages/openpi-client/src:${PYTHONPATH:-}"

ROS_OVERRIDES=()
if [[ -n "$EVAL_VIDEO_DIR" ]]; then
  ROS_OVERRIDES+=(-p "eval_video_dir:=$EVAL_VIDEO_DIR")
fi

exec "$PYTHON_BIN" ros2/droid_robot_env/droid_robot_env/openpi_inference_node.py \
  ${ROS_OVERRIDES[@]+"--ros-args" "${ROS_OVERRIDES[@]}"}
