import os
import numpy as np
import cv2
import torch
import time

from src.entities.datasets import BaseDataset 
from src.tools.lgVO import lightglueVO 
from src.tools.map import KeyFrame, Mapper


class Tracker:
    def __init__(self, config: dict, dataset: BaseDataset, lgVO: lightglueVO, current_map: Mapper):
        
        self.device = "cuda"
        self.dataset = dataset
        self.config = config
        self.K = dataset.intrinsics

        self.state = "INITIALIZING" # "INITIALIZING", "TRACKING", "LOST"
        self.odometer = lgVO
        self.map = current_map
        
        # Current frame pose.
        self.current_c2w = np.eye(4)
        self.last_keyframe_c2w = np.eye(4)
        
        # Current frame data.
        self.current_frame_features = None # Contains 'keypoints' and 'descriptors'.
        self.current_frame_image = None
        self.current_frame_depth = None
        self.current_gt_pose = np.eye(4)
        
        self.min_keyframe_dist = config["min_keyframe_dist"] # Translation threshold.
        self.min_keyframe_angle = np.deg2rad(config["min_keyframe_angle"]) # Rotation threshold.
        self.num_tracked_mappoints = 0
        self.valid_matches_dict = {}

    # ------------------------------------------------------------------
    #  Tracker helpers.
    # ------------------------------------------------------------------

    def _load_current_frame_data(self, frame_id):
        _, img, depth, gt_pose = self.dataset[frame_id]
        self.current_frame_image = img
        self.current_frame_depth = depth
        self.current_gt_pose = gt_pose
        self.valid_matches_dict = {}
        self.num_tracked_mappoints = 0

    def initial_pose_estimation(self, frame_id, pose_history: np.ndarray) -> np.ndarray:
        self._catch_features()
        pose = pose_history[2, :] @ np.linalg.inv(pose_history[1, :]) @ pose_history[2, :]
        return pose

    def _catch_features(self):
        image_tensor = torch.from_numpy(self.current_frame_image).permute(2, 0, 1).float().to(self.odometer.device) / 255.0
        feats = self.odometer.extractor.extract(image_tensor)
        self.current_frame_features = {
            'keypoints': feats['keypoints'][0].cpu().numpy(),
            'descriptors': feats['descriptors'][0].cpu().numpy(),
            'scores': feats['keypoint_scores'][0].cpu().numpy()
        } 

    def tracking_threshold(self):
        T_rel = np.linalg.inv(self.last_keyframe_c2w) @ self.current_c2w
        dist = np.linalg.norm(T_rel[:3, 3])
        R, _ = cv2.Rodrigues(T_rel[:3, :3])
        angle = np.linalg.norm(R)
        
        MAX_THRESH_DIST = self.config["max_thresh_dist"]
        MAX_THRESH_ANGLE = np.deg2rad(self.config["max_thresh_angle"])
        
        if dist > MAX_THRESH_DIST or angle > MAX_THRESH_ANGLE:
            print(f"[Tracker] New keyframe (lost): Dist={dist:.2f}m, Angle={np.rad2deg(angle):.1f}deg")
            return False
        else:
            return True

    def needs_new_keyframe(self, success) -> bool:
        if not success:
            print("[Tracker] Skip keyframe while tracking is not reliable.")
            return False

        T_rel = np.linalg.inv(self.last_keyframe_c2w) @ self.current_c2w
        dist = np.linalg.norm(T_rel[:3, 3])
        R, _ = cv2.Rodrigues(T_rel[:3, :3])
        angle = np.linalg.norm(R)
        
        if dist > self.min_keyframe_dist or angle > self.min_keyframe_angle:
            print(f"[Tracker] New keyframe (motion): Dist={dist:.2f}m, Angle={np.rad2deg(angle):.1f}deg")
            return True
        elif success and self.num_tracked_mappoints < 150:
            print(f"[Tracker] New keyframe (sparse): num_tracked_mappoints={self.num_tracked_mappoints}")
            return True
        else:
            return False

    def create_new_keyframe(self, frame_id):
        new_kf = KeyFrame(
            frame_id=frame_id,
            pose_c2w=self.current_c2w,
            features=self.current_frame_features,
        )
        # Mark tracked map points.
        if self.valid_matches_dict:
            new_kf.mappoint_ids = self.valid_matches_dict.copy()
        self.map.insert_keyframe(new_kf)
        self.odometer.set_reference(
            frame_id,
            self.current_frame_image,
            self.current_frame_depth,
            self.current_c2w,
        )
        self.last_keyframe_c2w = self.current_c2w

    def lightglue_reloc(self, frame_id, pose_history: np.ndarray, relocalizing: bool = False) -> np.ndarray:
        pose, is_ok = self.odometer.update(
            frame_id,
            self.current_frame_image,
            self.current_frame_depth,
            pose_history,
            relocalizing=relocalizing,
        )
        return pose, is_ok

    # ------------------------------------------------------------------
    #  Tracker local tracking.
    # ------------------------------------------------------------------

    def track_local_map(self) -> bool:
        
        # Fetch the local map.
        local_map = self.map.get_local_map(self.current_c2w)
        if local_map is None:
            print("[Tracker] No local map.")
            self.num_tracked_mappoints = 0
            return False

        mp_positions = local_map['positions']
        mp_descriptors = local_map['descriptors']
        mp_ids = local_map['ids']

        if len(mp_positions) < 80:
            print(f"[Tracker] Too few local map points ({len(mp_positions)}), skipping.")
            self.num_tracked_mappoints = 0
            return False

        feat_positions   = self.current_frame_features['keypoints']
        feat_descriptors = self.current_frame_features['descriptors']

        # Match feature descriptors for 2D-3D tracking.
        curr_w2c = np.linalg.inv(self.current_c2w)
        rvec, _ = cv2.Rodrigues(curr_w2c[:3, :3])
        tvec = curr_w2c[:3, 3]
        predicted_pts, _ = cv2.projectPoints(mp_positions, rvec, tvec, self.K, distCoeffs=None)
        predicted_pts = predicted_pts.reshape(-1, 2)

        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(mp_descriptors, feat_descriptors, k=2)
        candidates_strict = []
        candidates_loose = []

        for knn_result in matches:
            if len(knn_result) == 2:
                m, n = knn_result
                if m.distance < 0.8 * n.distance:
                    dist = np.linalg.norm(feat_positions[m.trainIdx] - predicted_pts[m.queryIdx])
                    item = (mp_positions[m.queryIdx], feat_positions[m.trainIdx], (m.trainIdx, mp_ids[m.queryIdx]))
                    if dist < 100.0:
                        candidates_loose.append(item)
                        if dist < 50.0:
                            candidates_strict.append(item)

        if len(candidates_loose) > 10 * len(candidates_strict):
            final_selection = candidates_loose
        else:
            final_selection = candidates_strict
        
        if len(final_selection) < 80:
            print(f"[Tracker] Too few local PnP matches ({len(final_selection)}), skipping.")
            self.num_tracked_mappoints = 0
            return False

        localmap_points = np.array([x[0] for x in final_selection], dtype=np.float64)
        image_points    = np.array([x[1] for x in final_selection], dtype=np.float32)
        temp_matches_indices = [x[2] for x in final_selection]

        # Solve PnP from 2D-3D matches.
        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                localmap_points, 
                image_points, 
                self.K, 
                distCoeffs=None,
                iterationsCount=100,
                reprojectionError=4.0,
                flags=cv2.SOLVEPNP_EPNP 
            )

            if not success or inliers is None or len(inliers) < 6:
                print("[Tracker] PnP RANSAC failed.")
                self.num_tracked_mappoints = 0
                return False
            
            inliers_flat = inliers.flatten()
            for idx in inliers_flat:
                feat_idx, mp_id = temp_matches_indices[idx]
                self.valid_matches_dict[feat_idx] = mp_id

            localmap_points_inliers = localmap_points[inliers.flatten()]
            image_points_inliers = image_points[inliers.flatten()]

            success, rvec, tvec = cv2.solvePnP(
                localmap_points_inliers,
                image_points_inliers,
                self.K,
                distCoeffs=None,
                rvec=rvec, 
                tvec=tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                print("[Tracker] PnP Refinement (ITERATIVE) failed.")
                return False
            else:
                R, _ = cv2.Rodrigues(rvec)
                t = tvec.flatten()

            if not (np.isfinite(R).all() and np.isfinite(t).all()):
                print("[Tracker] PnP RANSAC returned NaN.")
                return False
            else:
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R
                T[:3, 3] = t
                self.current_c2w = np.linalg.inv(T)
                self.num_tracked_mappoints = len(inliers)
                return True

        except cv2.error as e:
            print(f"[Tracker] cv2.solvePnPRansac error: {e}")
            self.num_tracked_mappoints = 0
            return False

    # ------------------------------------------------------------------
    #  Tracker main function.
    #  
    #  Returns:
    #  - T_c2w: final estimated c2w pose.
    #  - _is_keyframe: whether this frame is a keyframe.
    # ------------------------------------------------------------------

    def track(self, frame_id, pose_history: np.ndarray):

        start_time = time.time()
        
        # 1. Load data.
        self._load_current_frame_data(frame_id)

        was_lost = self.state == "LOST"

        if frame_id in [0, 1]:
            self.state = "INITIALIZING"
            self._catch_features()
            self.current_c2w = self.current_gt_pose
            _is_keyframe = True
            self.create_new_keyframe(frame_id)
            return self.current_c2w, _is_keyframe
        else:
            self.state = "LOST" if was_lost else "TRACKING"
        
        # 2. Estimate the initial pose.
        self.current_c2w = self.initial_pose_estimation(frame_id, pose_history)

        # 3. Track the local map.
        success = self.track_local_map()
        if not (success and self.tracking_threshold()): 
            self.current_c2w, success = self.lightglue_reloc(
                frame_id,
                pose_history,
                relocalizing=was_lost or not success,
            )
            self.state = "TRACKING" if success else "LOST"
        
        # 4. Decide whether to add a keyframe.
        _is_keyframe = False
        if self.needs_new_keyframe(success):
            _is_keyframe = True
            self.create_new_keyframe(frame_id)

        end_time = time.time()
        print(f"[Tracker] Frame {frame_id} done in {(end_time-start_time)*1000:.1f} ms")
        print(f"[Tracker] State: {self.state}\n")
        
        return self.current_c2w, _is_keyframe
