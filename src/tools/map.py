import numpy as np
import cv2
import time
import torch
import torch.optim as optim
from collections import deque, OrderedDict
from scipy.spatial import cKDTree

from src.entities.datasets import BaseDataset 


class KeyFrame:
    def __init__(self, frame_id: int, pose_c2w: np.ndarray, 
                 features: dict[str, np.ndarray]):
        self.frame_id = frame_id
        self.pose_c2w = pose_c2w
        self.features = features
        
        # MapPoint IDs observed by this keyframe.
        self.mappoint_ids: dict[int, int] = {}

    def add_mappoint_observation(self, feat_idx: int, mp_id: int):
        self.mappoint_ids[feat_idx] = mp_id


class MapPoint:
    _next_id = 0
    
    def __init__(self, position: np.ndarray, descriptor: np.ndarray, created_kf_id: int | None = None):
        self.id = MapPoint._next_id
        MapPoint._next_id += 1
        self.position = position.astype(np.float64)
        self.descriptor = descriptor.astype(np.float32)
        desc_norm = np.linalg.norm(self.descriptor)
        if desc_norm > 1e-12:
            self.descriptor /= desc_norm
        self.created_kf_id = created_kf_id
        self.last_observed_kf = created_kf_id
        
        # KeyFrame IDs that observe this point.
        self.observed_by_kfs: set[int] = set() 

    def add_observation(self, kf_id: int, descriptor: np.ndarray | None = None, position: np.ndarray | None = None):
        is_new_observation = kf_id not in self.observed_by_kfs
        obs_count = len(self.observed_by_kfs)

        if is_new_observation and position is not None and obs_count > 0:
            self.position = (self.position * obs_count + position.astype(np.float64)) / (obs_count + 1)

        if is_new_observation and descriptor is not None:
            desc = descriptor.astype(np.float32)
            desc_norm = np.linalg.norm(desc)
            if desc_norm > 1e-12:
                desc /= desc_norm
                self.descriptor = (self.descriptor * obs_count + desc) / (obs_count + 1)
                avg_norm = np.linalg.norm(self.descriptor)
                if avg_norm > 1e-12:
                    self.descriptor /= avg_norm

        self.observed_by_kfs.add(kf_id)
        if self.last_observed_kf is None or kf_id > self.last_observed_kf:
            self.last_observed_kf = kf_id
        
    def remove_observation(self, kf_id: int):
        if kf_id in self.observed_by_kfs:
            self.observed_by_kfs.remove(kf_id)


class Mapper:
    def __init__(self, config: dict, dataset: BaseDataset):
        
        self.dataset = dataset
        self.config = config
        self.K = dataset.intrinsics

        self.keyframe_buffer: deque[KeyFrame] = deque()
        self.keyframes: OrderedDict[int, KeyFrame] = OrderedDict()
        self.mappoints: dict[int, MapPoint] = {}

        self.sliding_window = config["sliding_window"]
        self.local_map_radius = 10.0

    # ------------------------------------------------------------------
    #  Mapper helpers.
    # ------------------------------------------------------------------

    def _delete_mappoints(self, mp_ids: set[int]):
        if not mp_ids:
            return
        for kf in self.keyframes.values():
            stale_feat_ids = [fid for fid, mid in kf.mappoint_ids.items() if mid in mp_ids]
            for fid in stale_feat_ids:
                del kf.mappoint_ids[fid]
        for mp_id in mp_ids:
            self.mappoints.pop(mp_id, None)

    def sliding_window_culling(self):
        
        while len(self.keyframes) > self.sliding_window:
            oldest_frame_id, oldest_kf = self.keyframes.popitem(last=False)
            points_to_delete = set()

            for mp_id in oldest_kf.mappoint_ids.values(): 
                if mp_id in self.mappoints:
                    mp = self.mappoints[mp_id]
                    mp.remove_observation(oldest_frame_id)
                    if len(mp.observed_by_kfs) == 0:  # Delete if no other frame observes it.
                        points_to_delete.add(mp_id)
            
            self._delete_mappoints(points_to_delete)

    def insert_keyframe(self, kf: KeyFrame | None):
        if kf is None: return
        self.keyframe_buffer.append(kf)

    def get_local_map(self, current_pose_c2w: np.ndarray) -> dict | None:
        if not self.mappoints:
            return None

        # Use all map points in the sliding window.
        all_mps = list(self.mappoints.values())
        res_positions = np.array([mp.position for mp in all_mps], dtype=np.float64)
        res_descriptors = np.array([mp.descriptor for mp in all_mps], dtype=np.float32)
        res_ids = np.array([mp.id for mp in all_mps], dtype=int)

        return {
            'positions': res_positions,
            'descriptors': res_descriptors,
            'ids': res_ids
        }

    def create_new_mappoints(self, kf: KeyFrame):
        # 1. Prepare data and back-project points.
        _, _, depth, _ = self.dataset[kf.frame_id]
        H, W = depth.shape
        pts, descs = kf.features['keypoints'], kf.features['descriptors']
        
        u, v = pts[:, 0], pts[:, 1]
        u_i, v_i = np.round(u).astype(int), np.round(v).astype(int)
        mask = (u_i >= 0) & (u_i < W) & (v_i >= 0) & (v_i < H)
        
        z = np.zeros_like(u)
        z[mask] = depth[v_i[mask], u_i[mask]]
        valid = mask & (z > 0.1) & np.isfinite(z)
        idxs = np.where(valid)[0] # Valid feature indices.
        
        if len(idxs) == 0: return

        fx, fy, cx, cy = self.dataset.fx, self.dataset.fy, self.dataset.cx, self.dataset.cy
        x = (u[valid] - cx) / fx * z[valid]
        y = (v[valid] - cy) / fy * z[valid]
        P_cam = np.stack((x, y, z[valid], np.ones_like(x)), axis=1)
        P_w = (kf.pose_c2w @ P_cam.T).T[:, :3]

        # 2. Maintain old observations and prepare the local map.
        is_linked = np.array([i in kf.mappoint_ids for i in idxs])
        for local_idx in np.flatnonzero(is_linked):
            feat_idx = idxs[local_idx]
            mp = self.mappoints.get(kf.mappoint_ids[feat_idx])
            if mp:
                mp.add_observation(kf.frame_id, descs[feat_idx], P_w[local_idx])
            else:
                del kf.mappoint_ids[feat_idx]

        local_map = self.get_local_map(kf.pose_c2w)
        unlinked = np.where(~is_linked)[0]
        matched = np.zeros(len(unlinked), dtype=bool)

        # 3. Batch matching.
        if len(unlinked) > 0 and local_map and len(local_map['positions']) > 0:
            P_q, D_q = P_w[unlinked], descs[idxs[unlinked]]
            map_pos, map_descs, map_ids = local_map['positions'], local_map['descriptors'], local_map['ids']
            
            # Batch radius search.
            neighbors = cKDTree(map_pos).query_ball_point(P_q, r=0.05)
            lens = [len(n) for n in neighbors]
            
            if sum(lens) > 0:
                q_flat = np.repeat(np.arange(len(P_q)), lens)
                m_flat = np.concatenate(neighbors).astype(int)
                
                # Compute and filter similarities.
                sims = np.einsum('ij,ij->i', D_q[q_flat], map_descs[m_flat])
                valid_s = sims > 0.85
                
                if np.any(valid_s):
                    vq, vm, vs = q_flat[valid_s], m_flat[valid_s], sims[valid_s]
                    # Keep the best match per query.
                    order = np.lexsort((-vs, vq))
                    vq_s, vm_s = vq[order], vm[order]
                    _, u_idx = np.unique(vq_s, return_index=True)
                    
                    orig_sub = idxs[unlinked]
                    for q_i, m_i in zip(vq_s[u_idx], vm_s[u_idx]):
                        mp = self.mappoints.get(map_ids[m_i])
                        if mp:
                            mp.add_observation(kf.frame_id, D_q[q_i], P_q[q_i])
                            kf.add_mappoint_observation(orig_sub[q_i], mp.id)
                            matched[q_i] = True

        # 4. Create new points.
        new_mpts = unlinked[~matched]
        count_new_mpts = len(new_mpts)
        if count_new_mpts > 0:
            P_new, D_new, orig_new = P_w[new_mpts], descs[idxs[new_mpts]], idxs[new_mpts]
            for i, oid in enumerate(orig_new):
                mp = MapPoint(P_new[i], D_new[i], created_kf_id=kf.frame_id)
                mp.add_observation(kf.frame_id, D_new[i], P_new[i])
                self.mappoints[mp.id] = mp
                kf.add_mappoint_observation(oid, mp.id)

        print(f"[Mapper] KeyFrame {kf.frame_id}: created {count_new_mpts} map points.")

    def recent_mappoints_culling(self, kf: KeyFrame):
        if not self.mappoints:
            return

        keyframe_ids = list(self.keyframes.keys())
        keyframe_order = {frame_id: i for i, frame_id in enumerate(keyframe_ids)}
        current_order = keyframe_order.get(kf.frame_id, len(keyframe_ids) - 1)
        points_to_delete = set()

        for mp_id, mp in list(self.mappoints.items()):
            if not np.isfinite(mp.position).all() or len(mp.observed_by_kfs) == 0:
                points_to_delete.add(mp_id)
                continue

            if mp.created_kf_id not in keyframe_order:
                continue

            age = current_order - keyframe_order[mp.created_kf_id]
            if age >= 2 and len(mp.observed_by_kfs) < 2:
                points_to_delete.add(mp_id)

        if points_to_delete:
            self._delete_mappoints(points_to_delete)
            print(f"[Mapper] Culled {len(points_to_delete)} weak map points.")

    def local_keyframes_culling(self, kf: KeyFrame):
        pass
    
    # ------------------------------------------------------------------
    #  Mapper local BA.
    # ------------------------------------------------------------------

    def needs_local_ba(self, kf: KeyFrame) -> bool:
        if len(self.keyframes) >= self.sliding_window:
            return True
        return False

    def local_bundle_adjustment(self):
        t0 = time.time()

        kfs = list(self.keyframes.values())
        mps = list(self.mappoints.values())
        if len(kfs) < 2 or not mps:
            return {}

        # 1. Build observation edges.
        mp_map = {mp.id: i for i, mp in enumerate(mps)}
        edges = [[i, mp_map[mid], *kf.features['keypoints'][fid]] 
                 for i, kf in enumerate(kfs) 
                 for fid, mid in kf.mappoint_ids.items() if mid in mp_map]
        if len(edges) < 20:
            print(f"[Mapper] Skip local BA: only {len(edges)} observation edges.")
            return {}

        device = "cpu" # Force CPU.
        T = torch.tensor(edges, dtype=torch.float64, device=device)
        cam_idx, pt_idx, uv_gt = T[:, 0].long(), T[:, 1].long(), T[:, 2:]
        K = torch.tensor([self.dataset.fx, self.dataset.fy, self.dataset.cx, self.dataset.cy], 
                         dtype=torch.float64, device=device)

        # 2. Initialize variables.
        se3_list = []
        for kf in kfs:
            w2c = np.linalg.inv(kf.pose_c2w)
            r, _ = cv2.Rodrigues(w2c[:3, :3])
            se3_list.append(np.hstack((r.flatten(), w2c[:3, 3])))
        
        se3_t = torch.tensor(np.array(se3_list), dtype=torch.float64, device=device)
        fixed, active = se3_t[0:1], se3_t[1:].clone().detach().requires_grad_(True)
        xyz = torch.tensor(np.array([mp.position for mp in mps]), 
                           dtype=torch.float64, device=device, requires_grad=True)

        # 3. Optimize.
        optimizer = optim.LBFGS([active, xyz], lr=1.0, max_iter=20, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            poses = torch.cat([fixed, active]) # (N_frames, 6)
            
            # Batch Rodrigues.
            r, t = poses[:, :3], poses[:, 3:]
            theta = torch.norm(r, dim=1, keepdim=True) + 1e-8
            k = r / theta
            kx, ky, kz = k[:,0], k[:,1], k[:,2]
            zero = torch.zeros_like(kx)
            K_x = torch.stack([zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], dim=1).reshape(-1, 3, 3)
            Rs = torch.eye(3, dtype=torch.float64, device=device) + \
                 torch.sin(theta).unsqueeze(2)*K_x + (1-torch.cos(theta).unsqueeze(2))*(K_x@K_x)
            
            # Broadcast and project: P = R @ X + t.
            P = (Rs[cam_idx] @ xyz[pt_idx].unsqueeze(2)).squeeze(2) + t[cam_idx]
            z = torch.clamp(P[:, 2:3], min=1e-5)
            uv_pred = P[:, :2] / z * K[:2] + K[2:]
            
            loss = torch.nn.functional.huber_loss(uv_pred, uv_gt, delta=2.0, reduction='sum')
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
        except Exception as e:
            print(f"[Mapper] Local BA failed: {e}")
            return {}

        if not torch.isfinite(active).all() or not torch.isfinite(xyz).all():
            print("[Mapper] Local BA produced non-finite values; skipping writeback.")
            return {}

        # 4. Write back results.
        optimized_poses = {}
        opt_poses = active.detach().numpy()
        for i, kf in enumerate(kfs[1:]):
            se3 = opt_poses[i]
            R, _ = cv2.Rodrigues(se3[:3])
            kf.pose_c2w = np.linalg.inv(np.vstack((np.hstack((R, se3[3:].reshape(3,1))), [0,0,0,1])))
            optimized_poses[kf.frame_id] = kf.pose_c2w
            
        opt_pts = xyz.detach().numpy()
        for i, mp in enumerate(mps):
            mp.position = opt_pts[i]

        print(f"[Mapper] Local BA done in {(time.time()-t0)*1000:.1f} ms")
        return optimized_poses

    # ------------------------------------------------------------------
    #  Mapper main function.
    # ------------------------------------------------------------------

    def map(self):
        
        optimized_poses_dict = {}

        while self.keyframe_buffer:
            start_time = time.time()
            current_kf = self.keyframe_buffer.popleft()
            
            # Fetch keyframe.
            if current_kf.frame_id in self.keyframes:
                print(f"[Mapper] KeyFrame {current_kf.frame_id} already processed, skipping.")
                continue
            else:
                self.keyframes[current_kf.frame_id] = current_kf
            
            # Add new map points.
            self.create_new_mappoints(current_kf)

            # Cull weak map points after the current observations are merged.
            self.recent_mappoints_culling(current_kf)
            
            # Local BA.
            if self.needs_local_ba(current_kf):
                optimized_poses = self.local_bundle_adjustment() 
                optimized_poses_dict.update(optimized_poses)
            
            # Cull redundant keyframes.
            self.local_keyframes_culling(current_kf) 

            # Sliding window.
            self.sliding_window_culling()

            end_time = time.time()
            print(f"[Mapper] Mapping frame {current_kf.frame_id} done in {(end_time-start_time)*1000:.1f} ms")

        return optimized_poses_dict
        
