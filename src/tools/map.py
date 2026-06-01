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
        
        # 此KF观测到的 MapPoint ID 字典 
        self.mappoint_ids: dict[int, int] = {}

    def add_mappoint_observation(self, feat_idx: int, mp_id: int):
        self.mappoint_ids[feat_idx] = mp_id


class MapPoint:
    _next_id = 0
    
    def __init__(self, position: np.ndarray, descriptor: np.ndarray):
        self.id = MapPoint._next_id
        MapPoint._next_id += 1
        self.position = position.astype(np.float64)
        self.descriptor = descriptor.astype(np.float32)
        
        # 观测到此点的 KeyFrame ID 集合
        self.observed_by_kfs: set[int] = set() 

    def add_observation(self, kf_id: int):
        self.observed_by_kfs.add(kf_id)
        
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
    #  Mapper 工具函数
    # ------------------------------------------------------------------

    def sliding_window_culling(self):
        
        while len(self.keyframes) > self.sliding_window:
            oldest_frame_id, oldest_kf = self.keyframes.popitem(last=False)
            points_to_delete = set()

            for mp_id in oldest_kf.mappoint_ids.values(): 
                if mp_id in self.mappoints:
                    mp = self.mappoints[mp_id]
                    mp.remove_observation(oldest_frame_id)
                    if len(mp.observed_by_kfs) == 0:  # 没有其他帧观测，标记为待删除
                        points_to_delete.add(mp_id)
            
            for mp_id in points_to_delete:
                del self.mappoints[mp_id]

    def insert_keyframe(self, kf: KeyFrame | None):
        if kf is None: return
        self.keyframe_buffer.append(kf)

    def get_local_map(self, current_pose_c2w: np.ndarray) -> dict | None:
        if not self.mappoints:
            return None

        # 直接提取滑窗内所有地图点
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
        # 1. 数据准备与反投影
        _, _, depth, _ = self.dataset[kf.frame_id]
        H, W = depth.shape
        pts, descs = kf.features['keypoints'], kf.features['descriptors']
        
        u, v = pts[:, 0], pts[:, 1]
        u_i, v_i = np.round(u).astype(int), np.round(v).astype(int)
        mask = (u_i >= 0) & (u_i < W) & (v_i >= 0) & (v_i < H)
        
        z = np.zeros_like(u)
        z[mask] = depth[v_i[mask], u_i[mask]]
        valid = mask & (z > 0.1) & np.isfinite(z)
        idxs = np.where(valid)[0] # 有效特征点索引
        
        if len(idxs) == 0: return

        fx, fy, cx, cy = self.dataset.fx, self.dataset.fy, self.dataset.cx, self.dataset.cy
        x = (u[valid] - cx) / fx * z[valid]
        y = (v[valid] - cy) / fy * z[valid]
        P_cam = np.stack((x, y, z[valid], np.ones_like(x)), axis=1)
        P_w = (kf.pose_c2w @ P_cam.T).T[:, :3]

        # 2. 维护旧观测 & 准备局部地图
        is_linked = np.array([i in kf.mappoint_ids for i in idxs])
        for i in idxs[is_linked]:
            mp = self.mappoints.get(kf.mappoint_ids[i])
            if mp: mp.add_observation(kf.frame_id)
            else: del kf.mappoint_ids[i]

        local_map = self.get_local_map(kf.pose_c2w)
        unlinked = np.where(~is_linked)[0]
        matched = np.zeros(len(unlinked), dtype=bool)

        # 3. 批量匹配 (Vectorized Matching)
        if len(unlinked) > 0 and local_map and len(local_map['positions']) > 0:
            P_q, D_q = P_w[unlinked], descs[idxs[unlinked]]
            map_pos, map_descs, map_ids = local_map['positions'], local_map['descriptors'], local_map['ids']
            
            # 批量半径搜索
            neighbors = cKDTree(map_pos).query_ball_point(P_q, r=0.05)
            lens = [len(n) for n in neighbors]
            
            if sum(lens) > 0:
                q_flat = np.repeat(np.arange(len(P_q)), lens)
                m_flat = np.concatenate(neighbors).astype(int)
                
                # 矩阵计算相似度 & 筛选
                sims = np.einsum('ij,ij->i', D_q[q_flat], map_descs[m_flat])
                valid_s = sims > 0.85
                
                if np.any(valid_s):
                    vq, vm, vs = q_flat[valid_s], m_flat[valid_s], sims[valid_s]
                    # 分组择优: 按sim降序排，unique取第一个
                    order = np.lexsort((-vs, vq))
                    vq_s, vm_s = vq[order], vm[order]
                    _, u_idx = np.unique(vq_s, return_index=True)
                    
                    orig_sub = idxs[unlinked]
                    for q_i, m_i in zip(vq_s[u_idx], vm_s[u_idx]):
                        mp = self.mappoints.get(map_ids[m_i])
                        if mp:
                            mp.add_observation(kf.frame_id)
                            kf.add_mappoint_observation(orig_sub[q_i], mp.id)
                            matched[q_i] = True

        # 4. 批量新建
        new_mpts = unlinked[~matched]
        count_new_mpts = len(new_mpts)
        if count_new_mpts > 0:
            P_new, D_new, orig_new = P_w[new_mpts], descs[idxs[new_mpts]], idxs[new_mpts]
            for i, oid in enumerate(orig_new):
                mp = MapPoint(P_new[i], D_new[i])
                mp.add_observation(kf.frame_id)
                self.mappoints[mp.id] = mp
                kf.add_mappoint_observation(oid, mp.id)

        print(f"[Mapper] KeyFrame {kf.frame_id}: 新建 {count_new_mpts} 个地图点.")

    def recent_mappoints_culling(self, kf: KeyFrame):
        pass

    def local_keyframes_culling(self, kf: KeyFrame):
        pass
    
    # ------------------------------------------------------------------
    #  Mapper 局部BA
    # ------------------------------------------------------------------

    def needs_local_ba(self, kf: KeyFrame) -> bool:
        if len(self.keyframes) >= self.sliding_window:
            return True
        return False

    def local_bundle_adjustment(self):
        t0 = time.time()

        kfs = list(self.keyframes.values())
        mps = list(self.mappoints.values())
        if len(kfs) < 2 or not mps: return {}

        # 1. 构建观测边 (Index Mapping)
        mp_map = {mp.id: i for i, mp in enumerate(mps)}
        edges = [[i, mp_map[mid], *kf.features['keypoints'][fid]] 
                 for i, kf in enumerate(kfs) 
                 for fid, mid in kf.mappoint_ids.items() if mid in mp_map]
        if not edges: return {}

        device = "cpu" # 强制 CPU
        T = torch.tensor(edges, dtype=torch.float64, device=device)
        cam_idx, pt_idx, uv_gt = T[:, 0].long(), T[:, 1].long(), T[:, 2:]
        K = torch.tensor([self.dataset.fx, self.dataset.fy, self.dataset.cx, self.dataset.cy], 
                         dtype=torch.float64, device=device)

        # 2. 初始化变量 (Vectorized)
        se3_list = []
        for kf in kfs:
            w2c = np.linalg.inv(kf.pose_c2w)
            r, _ = cv2.Rodrigues(w2c[:3, :3])
            se3_list.append(np.hstack((r.flatten(), w2c[:3, 3])))
        
        se3_t = torch.tensor(np.array(se3_list), dtype=torch.float64, device=device)
        fixed, active = se3_t[0:1], se3_t[1:].clone().detach().requires_grad_(True)
        xyz = torch.tensor(np.array([mp.position for mp in mps]), 
                           dtype=torch.float64, device=device, requires_grad=True)

        # 3. 优化 (Matrix Operation)
        optimizer = optim.LBFGS([active, xyz], lr=1.0, max_iter=20, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            poses = torch.cat([fixed, active]) # (N_frames, 6)
            
            # 批量 Rodrigues
            r, t = poses[:, :3], poses[:, 3:]
            theta = torch.norm(r, dim=1, keepdim=True) + 1e-8
            k = r / theta
            kx, ky, kz = k[:,0], k[:,1], k[:,2]
            zero = torch.zeros_like(kx)
            K_x = torch.stack([zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], dim=1).reshape(-1, 3, 3)
            Rs = torch.eye(3, dtype=torch.float64, device=device) + \
                 torch.sin(theta).unsqueeze(2)*K_x + (1-torch.cos(theta).unsqueeze(2))*(K_x@K_x)
            
            # 广播与投影 P = R @ X + t
            P = (Rs[cam_idx] @ xyz[pt_idx].unsqueeze(2)).squeeze(2) + t[cam_idx]
            z = torch.clamp(P[:, 2:3], min=1e-5)
            uv_pred = P[:, :2] / z * K[:2] + K[2:]
            
            loss = torch.nn.functional.huber_loss(uv_pred, uv_gt, delta=2.0, reduction='sum')
            loss.backward()
            return loss

        try: optimizer.step(closure)
        except: return {}

        # 4. 回写结果
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

        print(f"[Mapper] 局部 BA 完成: 耗时 {(time.time()-t0)*1000:.1f} ms")
        return optimized_poses

    # ------------------------------------------------------------------
    #  Mapper 主函数 
    # ------------------------------------------------------------------

    def map(self):
        
        optimized_poses_dict = {}

        while self.keyframe_buffer:
            start_time = time.time()
            current_kf = self.keyframe_buffer.popleft()
            
            # 提取关键帧
            if current_kf.frame_id in self.keyframes:
                print(f"[Mapper] 重复处理 {current_kf.frame_id}，跳过。")
                continue
            else:
                self.keyframes[current_kf.frame_id] = current_kf
            
            # 剔除冗余地图点
            self.recent_mappoints_culling(current_kf) 
            
            # 添加新地图点
            self.create_new_mappoints(current_kf)
            
            # 局部 BA
            if self.needs_local_ba(current_kf):
                optimized_poses = self.local_bundle_adjustment() 
                optimized_poses_dict.update(optimized_poses)
            
            # 剔除冗余关键帧
            self.local_keyframes_culling(current_kf) 

            # 滑动窗口
            self.sliding_window_culling()

            end_time = time.time()
            print(f"[Mapper] 建图帧 {current_kf.frame_id} 处理完毕，耗时: {(end_time-start_time)*1000:.1f} ms")

        return optimized_poses_dict
        