import os
import numpy as np
import cv2
import torch
import time

from thirdparty.LightGlue.lightglue import LightGlue, SuperPoint, match_pair
from src.entities.datasets import BaseDataset
from src.tools.viz import draw_matches_paper


class lightglueVO():
    def __init__(self, config: dict, dataset: BaseDataset):
        self.device = "cuda"
        self.dataset = dataset
        self.K = dataset.intrinsics
        
        self.curframe_id = 0
        self.keyframe_id = 1
        self.i0 = None
        self.d0 = None

        # SuperPoint + LightGlue
        self.extractor = SuperPoint(max_num_keypoints=config["max_num_keypoints"]).eval().to(self.device)  # load the extractor
        self.matcher = LightGlue(features='superpoint', depth_confidence=0.95, width_confidence=0.99).eval().to(self.device)  # load the matcher        

    @staticmethod
    def _form_transf(R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    
    def _load_data(self, image, depth):
        self.i1 = image
        self.d1 = depth
        if self.i0 is None or self.d0 is None: 
            self.keyframe_id = self.curframe_id - 1
        _, self.i0, self.d0, _ = self.dataset[self.keyframe_id]

    def get_matches(self):
        i0 = torch.from_numpy(self.i0).permute(2, 0, 1).float().to(self.device) / 255.0
        i1 = torch.from_numpy(self.i1).permute(2, 0, 1).float().to(self.device) / 255.0
        
        f1, f2, matches01 = match_pair(extractor=self.extractor, matcher=self.matcher, image0=i0, image1=i1)
        
        matches = matches01['matches']
        valid_matches = matches[(matches[..., 0] > -1) & (matches[..., 1] > -1) & (matches[..., 0] < f1['keypoints'].shape[0]) & (matches[..., 1] < f2['keypoints'].shape[0])]
        q1 = np.float32(f1['keypoints'][valid_matches[..., 0]])
        q2 = np.float32(f2['keypoints'][valid_matches[..., 1]])

        return q1,q2
    
    # ------------------------------------------------------------------
    #  PnP/ICP pose estimation.
    # ------------------------------------------------------------------

    def get_pose(self, q1, q2):
        H, W = self.d0.shape[:2]
        fx, fy, cx, cy = self.dataset.fx, self.dataset.fy, self.dataset.cx, self.dataset.cy
        
        # --- 2D-3D: back-project with single-frame depth. ---
        u0 = np.clip(np.round(q1[:, 0]).astype(int), 0, W - 1)
        v0 = np.clip(np.round(q1[:, 1]).astype(int), 0, H - 1)
        z0 = self.d0[v0, u0].astype(np.float32)

        valid_3d = (z0 > 0.1) & (z0 < 20.0)
        if valid_3d.sum() < 50:
            print("Not enough valid 3D points.")
            return None, False
        else:
            print("valid_3d: ", valid_3d.sum())

        q1v = q1[valid_3d].astype(np.float32)
        q2v = q2[valid_3d].astype(np.float32)
        z0v = z0[valid_3d]

        # Back-project to camera coordinates.
        X1 = np.column_stack([
            (q1v[:, 0] - cx) / fx * z0v,
            (q1v[:, 1] - cy) / fy * z0v,
            z0v
        ]).astype(np.float64)

        # RANSAC + PnP 
        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                X1,                     # 3D object points
                q2v,                    # 2D image points
                self.K,                 # Camera matrix
                None,                   # distCoeffs
                iterationsCount=100,    # RANSAC iterations.
                reprojectionError=4.0,  # Reprojection threshold in pixels.
                flags=cv2.SOLVEPNP_EPNP # Use EPNP.
            )

            if not success or inliers is None or len(inliers) < 6:
                print("PnP RANSAC returned success=False.")
                return None, False

            success, rvec, tvec = cv2.solvePnP(
                X1[inliers.flatten()], 
                q2v[inliers.flatten()], 
                self.K, 
                None,
                rvec=rvec,    # Use the RANSAC estimate as initialization.
                tvec=tvec,
                useExtrinsicGuess=True, 
                flags=cv2.SOLVEPNP_ITERATIVE # Use iterative refinement.
            )
            
            if not success:
                print("PnP Refinement (ITERATIVE) failed.")
                return None, False
            else:
                R, _ = cv2.Rodrigues(rvec)
                t = tvec.flatten()

            if not (np.isfinite(R).all() and np.isfinite(t).all()):
                print("PnP RANSAC returned NaN.")
                return None, False
            else:
                return self._form_transf(R, t), True

        except cv2.error as e:
            print(f"cv2.solvePnPRansac error: {e}")
            return None, False
        
    # ------------------------------------------------------------------
    #  VO main function.
    # ------------------------------------------------------------------
    
    def update_keyframe(self, num_matches):
        MIN_MATCHES_TO_KEEP = 150
        if num_matches < MIN_MATCHES_TO_KEEP:
            return self.keyframe_id

        self.keyframe_id = self.curframe_id
        self.i0 = self.i1
        self.d0 = self.d1
        return self.keyframe_id
        
    def update(self, idx, image, depth, prev_c2ws: np.ndarray):
        self.curframe_id = idx
        print(f"\nTracking frame {idx}")
        
        self._load_data(image, depth)
        q1, q2 = self.get_matches()
        
        transf_rel = np.linalg.inv(prev_c2ws[2, :]) @ prev_c2ws[1, :]
        transf, is_ok = self.get_pose(q1, q2)
        if not is_ok: 
            transf = transf_rel
        else:
            dist = np.linalg.norm(transf[:3, 3])
            R, _ = cv2.Rodrigues(transf[:3, :3])
            angle = np.linalg.norm(R)
            
            MAX_DIST_THRESH = 0.5
            MAX_ANGLE_THRESH = np.deg2rad(15.0)
            
            if dist > MAX_DIST_THRESH or angle > MAX_ANGLE_THRESH:
                print(f"LightGlue (lost): Dist={dist:.2f}m, Angle={np.rad2deg(angle):.1f}deg")
                transf = transf_rel
                is_ok = False

        cur_pose = prev_c2ws[-1] @ np.linalg.inv(transf)
        self.update_keyframe(len(q1))
        
        return cur_pose, is_ok
    


class lightglue_Registration():
    def __init__(self, config: dict, dataset: BaseDataset):
        self.device = "cuda"
        self.dataset = dataset
        self.K = dataset.intrinsics

        # SuperPoint + LightGlue
        self.extractor = SuperPoint(max_num_keypoints=config["max_num_keypoints"]).eval().to(self.device)  # load the extractor
        self.matcher = LightGlue(features='superpoint', depth_confidence=0.95, width_confidence=0.99).eval().to(self.device)  # load the matcher        

    @staticmethod
    def _form_transf(R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_matches(self, image0, image1):
        i0 = torch.from_numpy(image0).permute(2, 0, 1).float().to(self.device) / 255.0
        i1 = torch.from_numpy(image1).permute(2, 0, 1).float().to(self.device) / 255.0
        
        f1, f2, matches01 = match_pair(extractor=self.extractor, matcher=self.matcher, image0=i0, image1=i1)
        
        matches = matches01['matches']
        valid_matches = matches[(matches[..., 0] > -1) & (matches[..., 1] > -1) & (matches[..., 0] < f1['keypoints'].shape[0]) & (matches[..., 1] < f2['keypoints'].shape[0])]
        q1 = np.float32(f1['keypoints'][valid_matches[..., 0]])
        q2 = np.float32(f2['keypoints'][valid_matches[..., 1]])

        return q1,q2
    
    def get_pose(self, q1, q2, depth0, depth1):
        H, W = depth0.shape[:2]
        fx, fy, cx, cy = self.dataset.fx, self.dataset.fy, self.dataset.cx, self.dataset.cy
        
        # --- 3D-3D: back-project with two-frame depth. ---
        u0 = np.clip(np.round(q1[:, 0]).astype(int), 0, W - 1)
        v0 = np.clip(np.round(q1[:, 1]).astype(int), 0, H - 1)
        z0 = depth0[v0, u0].astype(np.float32)

        u1 = np.clip(np.round(q2[:, 0]).astype(int), 0, W - 1)
        v1 = np.clip(np.round(q2[:, 1]).astype(int), 0, H - 1)
        z1 = depth1[v1, u1].astype(np.float32)

        valid_3d3d = (z0 > 0.1) & (z0 < 20.0) & (z1 > 0.1) & (z1 < 20.0)
        valid_num = int(valid_3d3d.sum())
        if valid_num < 100:
            # print("Not enough valid 3D points.")
            return None, False, {"valid_num": valid_num}
        # else:
        #     print("valid_3d3d: ", valid_num)

        q1v = q1[valid_3d3d].astype(np.float32)
        q2v = q2[valid_3d3d].astype(np.float32)
        z0v = z0[valid_3d3d]
        z1v = z1[valid_3d3d]

        # Back-project to camera coordinates.
        X1 = np.column_stack([
            (q1v[:, 0] - cx) / fx * z0v,
            (q1v[:, 1] - cy) / fy * z0v,
            z0v
        ]).astype(np.float64)

        X2 = np.column_stack([
            (q2v[:, 0] - cx) / fx * z1v,
            (q2v[:, 1] - cy) / fy * z1v,
            z1v
        ]).astype(np.float64)

        # Mean-center + SVD (Umeyama / Horn).
        mu1 = X1.mean(axis=0)
        mu2 = X2.mean(axis=0)
        X1c = X1 - mu1
        X2c = X2 - mu2
        W = X2c.T @ X1c  # 3x3
        
        try:
            U, _, Vt = np.linalg.svd(W)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                U[:, -1] *= -1
                R = U @ Vt
            t = mu2 - R @ mu1
            
        except np.linalg.LinAlgError as e:
            print(f"SVD failed: {e}.")
            return None, False, {"valid_num": valid_num}

        # MAD adaptive threshold.
        errs = np.linalg.norm(X2 - (X1 @ R.T + t), axis=1)
        med = np.median(errs)
        mad = 1.4826 * np.median(np.abs(errs - med)) + 1e-6
        mask = errs <= (med + 2.5 * mad)
        inlier_num = int(mask.sum())
        if inlier_num < 50:
            return None, False, {
                "valid_num": valid_num,
                "inlier_num": inlier_num,
                "inlier_ratio": float(inlier_num / max(valid_num, 1)),
                "rmse": float(np.sqrt(np.mean(errs[mask] ** 2))) if inlier_num > 0 else float("inf"),
            }

        X1m, X2m = X1[mask], X2[mask]
        mu1 = X1m.mean(axis=0)
        mu2 = X2m.mean(axis=0)
        X1c = X1m - mu1
        X2c = X2m - mu2
        W = X2c.T @ X1c
        
        try:
            U, _, Vt = np.linalg.svd(W)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                U[:, -1] *= -1
                R = U @ Vt
            t = mu2 - R @ mu1
            
        except np.linalg.LinAlgError as e:
            print(f"SVD failed: {e}.")
            return None, False, {
                "valid_num": valid_num,
                "inlier_num": inlier_num,
                "inlier_ratio": float(inlier_num / max(valid_num, 1)),
            }

        final_errs = np.linalg.norm(X2m - (X1m @ R.T + t), axis=1)
        stats = {
            "rmse": float(np.sqrt(np.mean(final_errs ** 2))),
            "median": float(np.median(final_errs)),
            "inlier_num": inlier_num,
            "valid_num": valid_num,
            "inlier_ratio": float(inlier_num / max(valid_num, 1)),
        }

        return self._form_transf(R, t), True, stats
        
    def update(self, frame_id0, frame_id1):
        _, image0, depth0, _ = self.dataset[frame_id0]
        _, image1, depth1, _ = self.dataset[frame_id1]
        
        q1, q2 = self.get_matches(image0, image1)
        transf, is_ok, stats = self.get_pose(q1, q2, depth0, depth1)

        # draw_matches_paper(image0, image1, q1, q2, save_path="match_result.png")
        
        return transf, is_ok, stats
    
