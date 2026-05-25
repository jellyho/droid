#!/usr/bin/env python3
import argparse
import json
import traceback
from typing import Any

import numpy as np


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _summarize_observation(obs):
    summary = {}
    for key, value in obs.items():
        if key == "image" and isinstance(value, dict):
            summary[key] = {
                cam_key: {
                    "shape": list(image.shape),
                    "dtype": str(image.dtype),
                    "min": float(np.min(image)) if image.size else None,
                    "max": float(np.max(image)) if image.size else None,
                }
                for cam_key, image in value.items()
            }
        elif key == "robot_state" and isinstance(value, dict):
            summary[key] = {
                state_key: (
                    {"shape": list(np.asarray(state_value).shape), "value": np.asarray(state_value).tolist()}
                    if np.asarray(state_value).size <= 16
                    else {"shape": list(np.asarray(state_value).shape)}
                )
                for state_key, state_value in value.items()
            }
        elif isinstance(value, dict):
            summary[key] = {
                "keys": list(value.keys()),
                "num_keys": len(value),
            }
        else:
            summary[key] = value
    return summary


def _camera_kwargs(args):
    if args.no_cameras:
        return {}

    width = args.camera_width
    height = args.camera_height
    resize_func = args.camera_resize_func
    if width is None or height is None:
        width = 0
        height = 0
        resize_func = None

    common_kwargs = {
        "image": True,
        "concatenate_images": False,
        "resolution": (width, height),
        "resize_func": resize_func,
    }

    camera_kwargs = {}
    if args.hand_camera:
        camera_kwargs["hand_camera"] = dict(common_kwargs)
    if args.varied_camera:
        camera_kwargs["varied_camera"] = dict(common_kwargs)
    return camera_kwargs


def main():
    parser = argparse.ArgumentParser(description="Smoke-test DROID RobotEnv without ROS2.")
    parser.add_argument("--action-space", default="cartesian_velocity")
    parser.add_argument("--gripper-action-space", default="position")
    parser.add_argument("--control-hz", type=float, default=10.0)
    parser.add_argument("--do-reset", action=argparse.BooleanOptionalAction, default=True, help="Call RobotEnv reset during construction.")
    parser.add_argument("--no-cameras", action="store_true", help="Disable camera creation/readout.")
    parser.add_argument("--hand-camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--varied-camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera-width", type=int, default=None, help="None means original camera width.")
    parser.add_argument("--camera-height", type=int, default=None, help="None means original camera height.")
    parser.add_argument("--camera-resize-func", default="cv2")
    args = parser.parse_args()

    try:
        print("[robot_env_smoke_test] importing DROID RobotEnv...", flush=True)
        from droid.misc.parameters import hand_camera_id, varied_camera_1_id, varied_camera_2_id
        from droid.robot_env import RobotEnv

        print(
            "[robot_env_smoke_test] configured camera ids: "
            f"wrist={hand_camera_id}, side_view_1={varied_camera_1_id}, side_view_2={varied_camera_2_id}",
            flush=True,
        )

        camera_kwargs = _camera_kwargs(args)
        print(f"[robot_env_smoke_test] camera_kwargs={camera_kwargs}", flush=True)
        print("[robot_env_smoke_test] creating RobotEnv...", flush=True)
        env = RobotEnv(
            action_space=args.action_space,
            gripper_action_space=args.gripper_action_space,
            camera_kwargs=camera_kwargs,
            do_reset=args.do_reset,
            control_hz=args.control_hz,
        )

        connected_cameras = sorted(getattr(env.camera_reader, "camera_dict", {}).keys())
        print(f"[robot_env_smoke_test] RobotEnv created. DoF={env.DoF}", flush=True)
        print(f"[robot_env_smoke_test] connected camera ids={connected_cameras}", flush=True)
        print("[robot_env_smoke_test] reading one observation...", flush=True)
        obs = env.get_observation()

        print("[robot_env_smoke_test] observation summary:", flush=True)
        print(json.dumps(_summarize_observation(obs), indent=2, default=_json_default), flush=True)
        print("[robot_env_smoke_test] ok", flush=True)
    except Exception:
        print("[robot_env_smoke_test] failed:", flush=True)
        print(traceback.format_exc(), flush=True)
        raise


if __name__ == "__main__":
    main()
