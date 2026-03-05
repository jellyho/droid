import os
import time
import shutil
from copy import deepcopy
from datetime import date

import cv2
import h5py

import droid.trajectory_utils.misc as tu
from droid.calibration.calibration_utils import check_calibration_info
from droid.misc.parameters import hand_camera_id, droid_version, robot_serial_number, robot_type

# Prepare Data Folder #
dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(dir_path, "../../data")


class DataCollecter:
    def __init__(self, env, controller, policy=None, save_data=True, save_traj_dir=None):
        self.env = env
        self.controller = controller
        self.policy = policy

        self.traj_running = False
        self.traj_saved = False
        self.obs_pointer = {}
        
        # --- NEW: Initialize Session & Counters ---
        self.traj_counter = 0  # Global counter for this session
        self.session_name = time.strftime("%H-%M-%S") + "_Session"
        self.today_date = str(date.today())
        # ------------------------------------------

        # Get Camera Info #
        self.cam_ids = list(env.camera_reader.camera_dict.keys())
        self.cam_ids.sort()

        _, full_cam_ids = self.get_camera_feed()
        self.num_cameras = len(full_cam_ids)
        self.full_cam_ids = full_cam_ids
        self.advanced_calibration = False

        # --- NEW: Create Static Directory Structure ---
        if save_traj_dir is None:
            save_traj_dir = data_dir
        
        # Structure: data/2026-01-14/15-30-00_Session/
        self.session_dir = os.path.join(save_traj_dir, self.today_date, self.session_name)
        
        self.success_dir = os.path.join(self.session_dir, "success")
        self.failure_dir = os.path.join(self.session_dir, "failure")
        # self.recordings_dir = os.path.join(self.session_dir, "recordings") # Optional: Keep videos separate

        self.save_data = save_data
        if self.save_data:
            if not os.path.isdir(self.success_dir):
                os.makedirs(self.success_dir)
            if not os.path.isdir(self.failure_dir):
                os.makedirs(self.failure_dir)
            # if not os.path.isdir(self.recordings_dir):
            #     os.makedirs(self.recordings_dir)
            print(f"[DataCollector] Saving to: {self.session_dir}")

    def reset_robot(self, randomize=False):
        self.env._robot.establish_connection()
        self.controller.reset_state()
        self.env.reset(randomize=False)

    def get_user_feedback(self):
        info = self.controller.get_info()
        return deepcopy(info)

    def enable_advanced_calibration(self):
        self.advanced_calibration = True
        self.env.camera_reader.enable_advanced_calibration()

    def disable_advanced_calibration(self):
        self.advanced_calibration = False
        self.env.camera_reader.disable_advanced_calibration()

    def set_calibration_mode(self, cam_id):
        self.env.camera_reader.set_calibration_mode(cam_id)

    def set_trajectory_mode(self):
        self.env.camera_reader.set_trajectory_mode()

    def collect_trajectory(self, info=None, practice=False, reset_robot=True):
        # 1. Define Names
        traj_name = f"trajectory_{self.traj_counter}"
        
        if info is None:
            info = {}
        info["time"] = time.asctime()
        info["robot_serial_number"] = "{0}-{1}".format(robot_type, robot_serial_number)
        info["version_number"] = droid_version
        info["traj_id"] = self.traj_counter

        # 2. Setup Paths
        if practice or (not self.save_data):
            save_filepath = None
            recording_folderpath = None
        else:
            if len(self.full_cam_ids) != 4: 
                # Note: Adjust camera count check as needed for your specific rig
                raise ValueError("WARNING: User is trying to collect data without all cameras running!")
            
            # Default to failure folder initially
            save_filepath = os.path.join(self.failure_dir, f"{traj_name}.h5")
            recording_folderpath = None
            # Save videos to a subfolder to keep main folders clean? 
            # Or inside failure/recordings? 
            # Here we put them in a central recordings folder to avoid moving folders.
            # recording_folderpath = os.path.join(self.recordings_dir, traj_name)
            # if not os.path.isdir(recording_folderpath):
                # os.makedirs(recording_folderpath)

        # 3. Collect Trajectory
        self.traj_running = True
        self.env._robot.establish_connection()
        
        controller_info = tu.collect_trajectory(
            self.env,
            controller=self.controller,
            metadata=info,
            policy=self.policy,
            obs_pointer=self.obs_pointer,
            reset_robot=reset_robot,
            recording_folderpath=recording_folderpath,
            save_filepath=save_filepath,
            wait_for_controller=True,
            # save_images=True
        )
        self.traj_running = False
        self.obs_pointer = {}

        # 4. Handle Sorting (Move File if Success)
        self.traj_saved = controller_info["success"] and (save_filepath is not None)
        
        if self.traj_saved:
            # Move .h5 file from failure -> success
            new_path = os.path.join(self.success_dir, f"{traj_name}.h5")
            try:
                os.rename(save_filepath, new_path)
                # We also track the current path in case we need to toggle it later
                self.current_h5_path = new_path 
            except OSError as e:
                print(f"Error moving file to success: {e}")
        else:
            self.current_h5_path = save_filepath

        # Increment counter only if we actually tried to save
        if self.save_data and not practice:
            self.traj_counter += 1

    def calibrate_camera(self, cam_id, reset_robot=True):
        self.traj_running = True
        self.env._robot.establish_connection()
        success = tu.calibrate_camera(
            self.env,
            cam_id,
            controller=self.controller,
            obs_pointer=self.obs_pointer,
            wait_for_controller=True,
            reset_robot=reset_robot,
        )
        self.traj_running = False
        self.obs_pointer = {}
        return success

    def check_calibration_info(self, remove_hand_camera=False):
        info_dict = check_calibration_info(self.full_cam_ids)
        if remove_hand_camera:
            info_dict["old"] = [cam_id for cam_id in info_dict["old"] if (hand_camera_id not in cam_id)]
        return info_dict

    def get_gui_imgs(self, obs):
        all_cam_ids = list(obs["image"].keys())
        all_cam_ids.sort()

        gui_images = []
        for cam_id in all_cam_ids:
            img = cv2.cvtColor(obs["image"][cam_id], cv2.COLOR_BGRA2RGB)
            gui_images.append(img)

        return gui_images, all_cam_ids

    def get_camera_feed(self):
        if self.traj_running:
            if "image" not in self.obs_pointer:
                raise ValueError
            obs = deepcopy(self.obs_pointer)
        else:
            obs = self.env.read_cameras()[0]
        gui_images, cam_ids = self.get_gui_imgs(obs)
        return gui_images, cam_ids

    def change_trajectory_status(self, success=False):
        """
        Manually toggle the last trajectory between success and failure folders.
        """
        if not self.save_data or not hasattr(self, 'current_h5_path'):
            return

        # Determine where the file IS and where it SHOULD be
        filename = os.path.basename(self.current_h5_path) # e.g., trajectory_5.h5
        
        if success:
            dest_folder = self.success_dir
        else:
            dest_folder = self.failure_dir
            
        new_path = os.path.join(dest_folder, filename)

        # Only move if it's not already there
        if self.current_h5_path != new_path:
            try:
                # 1. Update Internal Metadata
                with h5py.File(self.current_h5_path, "r+") as traj_file:
                    traj_file.attrs["success"] = success
                    traj_file.attrs["failure"] = not success
                
                # 2. Move File
                os.rename(self.current_h5_path, new_path)
                self.current_h5_path = new_path
                self.traj_saved = success
                print(f"Traj moved to: {new_path}")
            except Exception as e:
                print(f"Failed to change status: {e}")