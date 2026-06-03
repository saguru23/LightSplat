""" This module includes the Gaussian-SLAM class, which is responsible for controlling Mapper and Tracker
    It also decides when to start a new submap and when to update the estimated camera poses.
"""
import os
import pprint
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import roma
import threading
import queue
import time

from src.entities.arguments import OptimizationParams
from src.entities.datasets import get_dataset
from src.entities.gaussian_model import GaussianModel
from src.entities.mapper import Mapper
from src.entities.tracker import Tracker
from src.entities.lc import Loop_closure
from src.entities.logger import Logger
from src.utils.io_utils import save_dict_to_ckpt, save_dict_to_yaml
from src.utils.mapper_utils import exceeds_motion_thresholds 
from src.utils.utils import np2torch, setup_seed, torch2np
from src.utils.vis_utils import *  # noqa - needed for debugging
from src.utils.tracker_utils import compute_camera_opt_params

import rospy
from src.entities.visualizer import LightViewer
from src.tools.lgVO import lightglueVO
from src.tools.slam import lightglueSLAM
from src.tools.run_slam import PreloadDataset


class GaussianSLAM(object):

    def __init__(self, config: dict) -> None:

        self._setup_output_path(config)
        self.device = "cuda"
        self.config = config

        self.optimize_queue = queue.Queue()
        self.optimize_finished_queue = queue.Queue()
        self.ready_event = threading.Event()
        self.is_running = True  

        self.scene_name = config["data"]["scene_name"]
        self.dataset_name = config["dataset_name"]
        self.dataset = get_dataset(config["dataset_name"])({**config["data"], **config["cam"]})
        self.dataset = PreloadDataset(self.dataset)
        # self.dataset = get_dataset(config["dataset_name"])({**config["data"], **config["cam"], **config["ros"]})  # step-0

        # Mapping frame IDs, with the last frame appended.
        n_frames = len(self.dataset)
        frame_ids = list(range(n_frames))
        self.mapping_frame_ids = frame_ids[::config["mapping"]["map_every"]] + [n_frames - 1]

        # Per-frame pose estimates.
        self.estimated_c2ws = torch.empty(len(self.dataset), 4, 4)
        self.estimated_c2ws[0] = torch.from_numpy(self.dataset[0][3])
        self.exposures_ab = torch.zeros(len(self.dataset), 2)

        save_dict_to_yaml(config, "config.yaml", directory=self.output_path)

        self.submap_using_motion_heuristic = config["mapping"]["submap_using_motion_heuristic"]

        self.keyframes_info = {}
        self.opt = OptimizationParams(ArgumentParser(description="Training script parameters"))

        # Submap frame IDs, with the last frame appended.
        if self.submap_using_motion_heuristic:
            self.new_submap_frame_ids = [0]
        else:
            self.new_submap_frame_ids = frame_ids[::config["mapping"]["new_submap_every"]] + [n_frames - 1]
            self.new_submap_frame_ids.pop(0)

        # Initialize modules.
        self.logger = Logger(self.output_path, config["use_wandb"])
        self.mapper = Mapper(config["mapping"], self.dataset, self.logger)
        self.tracker = Tracker(config["tracking"], self.dataset, self.logger)
        self.enable_exposure = self.tracker.enable_exposure
        self.loop_closer = Loop_closure(config, self.dataset, self.logger)
        self.loop_closer.submap_path = self.output_path / "submaps"

        self.use_lgvo = config["light"]["used"]
        if self.use_lgvo:
            # self.lgvo_tracker = lightglueVO(config["light"], self.dataset)
            self.lgvo_tracker = lightglueSLAM(config["light"], self.dataset)
        
        print('Tracking config')
        pprint.PrettyPrinter().pprint(config["tracking"])
        print('Mapping config')
        pprint.PrettyPrinter().pprint(config["mapping"])
        print('Loop closure config')
        pprint.PrettyPrinter().pprint(config["lc"])
        

    def _setup_output_path(self, config: dict) -> None:
        """ Sets up the output path for saving results based on the provided configuration. If the output path is not
        specified in the configuration, it creates a new directory with a timestamp.
        Args:
            config: A dictionary containing the experiment configuration including data and output path information.
        """
        if "output_path" not in config["data"]:
            output_path = Path(config["data"]["output_path"])
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path = output_path / self.timestamp
        else:
            self.output_path = Path(config["data"]["output_path"])
        self.output_path.mkdir(exist_ok=True, parents=True)
        
        os.makedirs(self.output_path / "mapping_vis", exist_ok=True)
        os.makedirs(self.output_path / "tracking_vis", exist_ok=True)

    def should_start_new_submap(self, frame_id: int) -> bool:
        """ Determines whether a new submap should be started based on the motion heuristic or specific frame IDs.
        Args:
            frame_id: The ID of the current frame being processed.
        Returns:
            A boolean indicating whether to start a new submap.
        """
        if self.submap_using_motion_heuristic:# Motion threshold.
            if exceeds_motion_thresholds(
                self.estimated_c2ws[frame_id], self.estimated_c2ws[self.new_submap_frame_ids[-1]],
                    rot_thre=50, trans_thre=0.5):
                print(f"\nNew submap at {frame_id}")
                return True
        elif frame_id in self.new_submap_frame_ids:# Fixed frame interval.
            return True
        return False

    def save_current_submap(self, gaussian_model: GaussianModel):
        """Saving the current submap's checkpoint and resetting the Gaussian model

        Args:
            gaussian_model (GaussianModel): The current GaussianModel instance to capture and reset for the new submap.
        """
        
        gaussian_params = gaussian_model.capture_dict()
        submap_ckpt_name = str(self.submap_id).zfill(6)
        submap_ckpt = {
            "gaussian_params": gaussian_params,
            "submap_keyframes": sorted(list(self.keyframes_info.keys()))
        }
        save_dict_to_ckpt(
            submap_ckpt, f"{submap_ckpt_name}.ckpt", directory=self.output_path / "submaps")
    
    def start_new_submap(self, frame_id: int, gaussian_model: GaussianModel) -> None:
        """ Initializes a new submap.
        This function updates the submap count and optionally marks the current frame ID for new submap initiation.
        Args:
            frame_id: The ID of the current frame at which the new submap is started.
            gaussian_model: The current GaussianModel instance to capture and reset for the new submap.
        Returns:
            A new, reset GaussianModel instance for the new submap.
        """
        
        gaussian_model = GaussianModel(0)
        gaussian_model.training_setup(self.opt)
        self.mapper.keyframes = []
        self.keyframes_info = {}
        if self.submap_using_motion_heuristic:
            self.new_submap_frame_ids.append(frame_id)
        self.mapping_frame_ids.append(frame_id) if frame_id not in self.mapping_frame_ids else self.mapping_frame_ids
        self.submap_id += 1
        self.loop_closer.submap_id += 1
        return gaussian_model
    
    def rigid_transform_gaussians(self, gaussian_params, tsfm_matrix):
        '''
        Apply a rigid transformation to the Gaussian parameters.
        
        Args:
            gaussian_params (dict): Dictionary containing Gaussian parameters.
            tsfm_matrix (torch.Tensor): 4x4 rigid transformation matrix.
            
        Returns:
            dict: Updated Gaussian parameters after applying the transformation.
        '''
        # Transform Gaussian centers (xyz)
        tsfm_matrix = torch.from_numpy(tsfm_matrix).float()
        xyz = gaussian_params['xyz']
        pts_ones = torch.ones((xyz.shape[0], 1))
        pts_homo = torch.cat([xyz, pts_ones], dim=1)
        transformed_xyz = (tsfm_matrix @ pts_homo.T).T[:, :3]
        gaussian_params['xyz'] = transformed_xyz

        # Rotate covariance matrix (rotation)
        rotation = gaussian_params['rotation']
        cur_rot = roma.unitquat_to_rotmat(rotation)
        rot_mat = tsfm_matrix[:3, :3].unsqueeze(0)  # Adding batch dimension
        new_rot = rot_mat @ cur_rot
        new_quat = roma.rotmat_to_unitquat(new_rot)
        gaussian_params['rotation'] = new_quat.squeeze()

        return gaussian_params
    
    def update_keyframe_poses(self, lc_output, submaps_kf_ids, cur_frame_id):
        '''
        Update the keyframe poses using the correction from pgo, currently update the frame range that covered by the keyframes.
        
        '''
        def _get_submap_range(submap_id):
            start_id = self.new_submap_frame_ids[submap_id]
            if submap_id + 1 < len(self.new_submap_frame_ids):
                end_id = self.new_submap_frame_ids[submap_id + 1] - 1
            else:
                end_id = cur_frame_id
            return start_id, end_id
        
        for correction in lc_output:
            submap_id = correction['submap_id']
            correct_tsfm = correction['correct_tsfm']
            submap_kf_ids = submaps_kf_ids[submap_id]
            # min_id, max_id = min(submap_kf_ids), max(submap_kf_ids)
            min_id, max_id = _get_submap_range(submap_id)
            self.estimated_c2ws[min_id:max_id + 1] = torch.from_numpy(correct_tsfm).float() @ self.estimated_c2ws[min_id:max_id + 1]
        
        # # last tracked frame is based on last submap, update it as well
        # self.estimated_c2ws[cur_frame_id] = torch.from_numpy(lc_output[-1]['correct_tsfm']).float() @ self.estimated_c2ws[cur_frame_id]
        
        
    def apply_correction_to_submaps(self, correction_list):
        submaps_kf_ids= {}
        for correction in correction_list:
            submap_id = correction['submap_id']
            correct_tsfm = correction['correct_tsfm']

            submap_ckpt_name = str(submap_id).zfill(6) + ".ckpt"
            submap_ckpt = torch.load(self.output_path / "submaps" / submap_ckpt_name)
            submaps_kf_ids[submap_id] = submap_ckpt["submap_keyframes"]

            gaussian_params = submap_ckpt["gaussian_params"]
            updated_gaussian_params = self.rigid_transform_gaussians(
                gaussian_params, correct_tsfm)

            submap_ckpt["gaussian_params"] = updated_gaussian_params
            torch.save(submap_ckpt, self.output_path / "submaps" / submap_ckpt_name)
        return submaps_kf_ids
    
    def save_submap_from_result(self, result_data):
        """Saving the current submap's checkpoint and resetting the Gaussian model

        Args:
            gaussian_model (GaussianModel): The current GaussianModel instance to capture and reset for the new submap.
        """
        
        gaussian_params = result_data["gaussian_model"].capture_dict()
        submap_ckpt_name = str(result_data["submap_id"]).zfill(6)
        submap_ckpt = {
            "gaussian_params": gaussian_params,
            "submap_keyframes": sorted(list(result_data["keyframes_info"].keys()))
        }
        save_dict_to_ckpt(
            submap_ckpt, f"{submap_ckpt_name}.ckpt", directory=self.output_path / "submaps")
        print(f"\n[Thread] updated submap {submap_ckpt_name} ! ! !")
    
    def run(self) -> None:
        """ Starts the main program flow for Gaussian-SLAM, including tracking and mapping. """
        setup_seed(self.config["seed"])
        gaussian_model = GaussianModel(0)
        gaussian_model.training_setup(self.opt)
        self.submap_id = 0# Initial submap ID.
        viewer = LightViewer()

        total_compute_time = 0.0
        total_frames = 0

        try:
            for frame_id in range(len(self.dataset)):
                time0 = time.time()

                # Tracking
                estimated_c2w, _is_keyframe, optimized_poses_dict = self.lgvo_tracker.step(frame_id, torch2np(self.estimated_c2ws[torch.tensor([0, frame_id - 2, frame_id - 1])]))
                if self.enable_exposure:
                    exposure_ab = torch.nn.Parameter(torch.tensor(
                        [0.0], device="cuda")), torch.nn.Parameter(torch.tensor([0.0], device="cuda"))
                else :
                    exposure_ab = None
                self.estimated_c2ws[frame_id] = np2torch(estimated_c2w)
                if optimized_poses_dict:
                    for kf_id, corrected in optimized_poses_dict.items():
                        self.estimated_c2ws[kf_id] = np2torch(corrected)

                # optimize thread1
                if self.ready_event.is_set():
                    while not self.optimize_finished_queue.empty():
                        result_data = self.optimize_finished_queue.get()
                        self.save_submap_from_result(result_data)
                    self.ready_event.clear()
                    
                if frame_id > 2:
                    total_compute_time += (time.time() - time0)
                
                # Reinitialize gaussian model for new segment
                if self.should_start_new_submap(frame_id):
                    # first save current submap and its keyframe info
                    self.save_current_submap(gaussian_model)
                    
                    # update submap infomation for loop closer
                    self.loop_closer.update_submaps_info(self.keyframes_info)
                    
                    # apply loop closure
                    lc_output = self.loop_closer.loop_closure(self.estimated_c2ws)
                    
                    if len(lc_output) > 0:
                        submaps_kf_ids = self.apply_correction_to_submaps(lc_output)
                        self.update_keyframe_poses(lc_output, submaps_kf_ids, frame_id)
                    
                    save_dict_to_ckpt(self.estimated_c2ws[:frame_id + 1], "estimated_c2w.ckpt", directory=self.output_path)

                    # optimize thread2
                    keyframes_copy = list(self.mapper.keyframes)
                    keyframes_info_copy = self.keyframes_info.copy()
                    poses_copy = torch2np(self.estimated_c2ws.clone())
                    task = (
                        self.submap_id, 
                        frame_id, 
                        gaussian_model, 
                        keyframes_copy, 
                        keyframes_info_copy,
                        poses_copy)
                    self.optimize_queue.put(task)
                    
                    gaussian_model = self.start_new_submap(frame_id, gaussian_model)
                    if len(lc_output) > 0:
                        self.lgvo_tracker.restart(frame_id, torch2np(self.estimated_c2ws[frame_id]))

                time1 = time.time()

                # Mapping
                if _is_keyframe:
                    print("\nMapping frame", frame_id)
                    gaussian_model.training_setup(self.opt, exposure_ab) 
                    estimate_c2w = torch2np(self.estimated_c2ws[frame_id])
                    new_submap = not bool(self.keyframes_info)
                    opt_dict = self.mapper.map(
                        frame_id, estimate_c2w, gaussian_model, new_submap, exposure_ab)# Section 3.2 Gaussian alignment.

                    # Keyframes info update
                    self.keyframes_info[frame_id] = {
                        "keyframe_id": frame_id, 
                        "opt_dict": opt_dict,
                    }
                    if self.enable_exposure:
                        self.keyframes_info[frame_id]["exposure_a"] = exposure_ab[0].item()
                        self.keyframes_info[frame_id]["exposure_b"] = exposure_ab[1].item()

                if frame_id > 2:
                    total_compute_time += (time.time() - time1)
                    total_frames += 1

                # optimize thread3
                if (frame_id == len(self.dataset) - 1 or rospy.is_shutdown()):
                    self.optimize_queue.put(None)
                    while self.is_running: time.sleep(0.1)
                    if self.ready_event.is_set():
                        while not self.optimize_finished_queue.empty():
                            result_data = self.optimize_finished_queue.get()
                            self.save_submap_from_result(result_data)
                        self.ready_event.clear()
                
                # Run loop closure on the last frame.
                if (frame_id == len(self.dataset) - 1 or rospy.is_shutdown()) and self.config['lc']['final']:
                    print("\n Final loop closure ...")
                    self.loop_closer.update_submaps_info(self.keyframes_info)
                    lc_output = self.loop_closer.loop_closure(self.estimated_c2ws, final=True)
                    if len(lc_output) > 0:
                        submaps_kf_ids = self.apply_correction_to_submaps(lc_output)
                        self.update_keyframe_poses(lc_output, submaps_kf_ids, frame_id)
                if self.enable_exposure:
                    self.exposures_ab[frame_id] = torch.tensor([exposure_ab[0].item(), exposure_ab[1].item()])

                # Visualization.
                viewer.update(self.estimated_c2ws[:frame_id + 1], _is_keyframe)

                # ROS shutdown check.
                if rospy.is_shutdown():
                    self.dataset.close()
                    self.is_running = False
                    print("ROS shutdown requested, exiting loop...")
                    break

        except KeyboardInterrupt:
            self.is_running = False
            print("\n[GaussianSLAM] Ctrl+C detected, exiting gracefully...")

        finally:
            print(f"📊 System FPS: {total_frames / total_compute_time:.1f}. Please disable unnecessary vis.")
            save_dict_to_ckpt(self.estimated_c2ws[:frame_id + 1], "estimated_c2w.ckpt", directory=self.output_path)
            if self.enable_exposure:
                save_dict_to_ckpt(self.exposures_ab, "exposures_ab.ckpt", directory=self.output_path)
            viewer.close()

    def run_thread(self):
        opt_thread = threading.Thread(target=self.optimize_thread, name="opt_thread")
        opt_thread.start()
        
        try:
            self.run()
            opt_thread.join()
            
        except KeyboardInterrupt:
            self.is_running = False

    def optimize_thread(self):
        while True:
            try:
                task = self.optimize_queue.get(timeout=0.1) 
            except queue.Empty:
                if not self.is_running: break
                continue

            if task is None: 
                self.is_running = False
                break

            submap_id, frame_id, gaussian_model, keyframes, keyframes_info, poses_copy = task
            print(f"\n[Thread] Start refining Submap {submap_id} at frame {frame_id}.....................................")

            start_time = time.time()
            self.mapper.map_refine(frame_id, poses_copy, gaussian_model, keyframes)
            end_time = time.time() - start_time

            result_data = {
                "submap_id": submap_id,
                "frame_id": frame_id,
                "gaussian_model": gaussian_model,
                "keyframes_info": keyframes_info
            }
            self.optimize_finished_queue.put(result_data)
            self.ready_event.set()
            print(f"[Thread] Submap {submap_id} ready with time {end_time}.....................................\n")
