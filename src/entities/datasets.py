import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import json
import imageio
import trimesh

import threading
import time
from collections import deque
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
import rospy
import rostopic


class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_config: dict):
        self.dataset_path = Path(dataset_config["input_path"])
        self.frame_limit = dataset_config.get("frame_limit", -1)
        self.dataset_config = dataset_config
        self.height = dataset_config["H"]
        self.width = dataset_config["W"]
        self.fx = dataset_config["fx"]
        self.fy = dataset_config["fy"]
        self.cx = dataset_config["cx"]
        self.cy = dataset_config["cy"]

        self.depth_scale = dataset_config["depth_scale"]
        self.distortion = np.array(
            dataset_config['distortion']) if 'distortion' in dataset_config else None
        self.crop_edge = dataset_config['crop_edge'] if 'crop_edge' in dataset_config else 0
        if self.crop_edge:
            self.height -= 2 * self.crop_edge
            self.width -= 2 * self.crop_edge
            self.cx -= self.crop_edge
            self.cy -= self.crop_edge

        self.fovx = 2 * math.atan(self.width / (2 * self.fx))
        self.fovy = 2 * math.atan(self.height / (2 * self.fy))
        self.intrinsics = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

        self.color_paths = []
        self.depth_paths = []

    def __len__(self):
        return len(self.color_paths) if self.frame_limit < 0 else int(self.frame_limit)


class Replica(BaseDataset):

    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.color_paths = sorted(
            list((self.dataset_path / "results").glob("frame*.jpg")))
        self.depth_paths = sorted(
            list((self.dataset_path / "results").glob("depth*.png")))
        self.load_poses(self.dataset_path / "traj.txt")
        print(f"Loaded {len(self.color_paths)} frames")

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for line in lines:
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            self.poses.append(c2w.astype(np.float32))

    def __getitem__(self, index):
        color_data = cv2.imread(str(self.color_paths[index]))
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        depth_data = cv2.imread(
            str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        return index, color_data, depth_data, self.poses[index]


class TUM_RGBD(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.color_paths, self.depth_paths, self.poses = self.loadtum(
            self.dataset_path, frame_rate=32)

    def parse_list(self, filepath, skiprows=0):
        """ read list data """
        return np.loadtxt(filepath, delimiter=' ', dtype=np.unicode_, skiprows=skiprows)

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """ pair images, depths, and poses """
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt):
                    associations.append((i, j))
            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt) and (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))
        return associations

    def loadtum(self, datapath, frame_rate=-1):
        """ read video data in tum-rgbd format """
        if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
            pose_list = os.path.join(datapath, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(datapath, 'pose.txt')):
            pose_list = os.path.join(datapath, 'pose.txt')

        image_list = os.path.join(datapath, 'rgb.txt')
        depth_list = os.path.join(datapath, 'depth.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        images, poses, depths = [], [], []
        inv_pose = None
        for ix in indicies:
            (i, j, k) = associations[ix]
            images += [os.path.join(datapath, image_data[i, 1])]
            depths += [os.path.join(datapath, depth_data[j, 1])]
            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
            if inv_pose is None:
                inv_pose = np.linalg.inv(c2w)
                c2w = np.eye(4)
            else:
                c2w = inv_pose@c2w
            poses += [c2w.astype(np.float32)]

        return images, depths, poses

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose

    def __getitem__(self, index):
        color_data = cv2.imread(str(self.color_paths[index]))
        if self.distortion is not None:
            color_data = cv2.undistort(
                color_data, self.intrinsics, self.distortion)
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)

        depth_data = cv2.imread(
            str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        edge = self.crop_edge
        if edge > 0:
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
        # Interpolate depth values for splatting
        return index, color_data, depth_data, self.poses[index]


class ScanNet(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.color_paths = sorted(list(
            (self.dataset_path / "rgb").glob("*.png")), key=lambda x: int(os.path.basename(x)[-9:-4]))
        self.depth_paths = sorted(list(
            (self.dataset_path / "depth").glob("*.TIFF")), key=lambda x: int(os.path.basename(x)[-10:-5]))
        self.n_img = len(self.color_paths)
        self.load_poses(self.dataset_path / "gt_pose.txt")

    def load_poses(self, path):
        self.poses = []
        pose_data = np.loadtxt(path, delimiter=" ", dtype=np.unicode_, skiprows=1)
        pose_vecs = pose_data[:, 0:].astype(np.float64)
        for i in range(self.n_img):
            quat = pose_vecs[i][4:]
            trans = pose_vecs[i][1:4]
            T = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
            T[:3, 3] = trans
            pose = T
            self.poses.append(pose)

    def __getitem__(self, index):
        color_data = cv2.imread(str(self.color_paths[index]))
        if self.distortion is not None:
            color_data = cv2.undistort(
                color_data, self.intrinsics, self.distortion)
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = cv2.resize(color_data, (self.dataset_config["W"], self.dataset_config["H"]))

        depth_data = cv2.imread(
            str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        edge = self.crop_edge
        if edge > 0:
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
        # Interpolate depth values for splatting
        return index, color_data, depth_data, self.poses[index]


class ScanNetPP(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.use_train_split = dataset_config["use_train_split"]
        self.train_test_split = json.load(open(f"{self.dataset_path}/dslr/train_test_lists.json", "r"))
        if self.use_train_split:
            self.image_names = self.train_test_split["train"]
        else:
            self.image_names = self.train_test_split["test"]
        self.load_data()

    def load_data(self):
        self.poses = []
        cams_path = self.dataset_path / "dslr" / "nerfstudio" / "transforms_undistorted.json"
        cams_metadata = json.load(open(str(cams_path), "r"))
        frames_key = "frames" if self.use_train_split else "test_frames"
        frames_metadata = cams_metadata[frames_key]
        frame2idx = {frame["file_path"]: index for index, frame in enumerate(frames_metadata)}
        P = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]).astype(np.float32)
        for image_name in self.image_names:
            frame_metadata = frames_metadata[frame2idx[image_name]]
            # if self.ignore_bad and frame_metadata['is_bad']:
            #     continue
            color_path = str(self.dataset_path / "dslr" / "undistorted_images" / image_name)
            depth_path = str(self.dataset_path / "dslr" / "undistorted_depths" / image_name.replace('.JPG', '.png'))
            self.color_paths.append(color_path)
            self.depth_paths.append(depth_path)
            c2w = np.array(frame_metadata["transform_matrix"]).astype(np.float32)
            c2w = P @ c2w @ P.T
            self.poses.append(c2w)

    def __len__(self):
        if self.use_train_split:
            return len(self.image_names) if self.frame_limit < 0 else int(self.frame_limit)
        else:
            return len(self.image_names)

    def __getitem__(self, index):

        color_data = np.asarray(imageio.imread(self.color_paths[index]), dtype=float)
        color_data = cv2.resize(color_data, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        color_data = color_data.astype(np.uint8)

        depth_data = np.asarray(imageio.imread(self.depth_paths[index]), dtype=np.int64)
        depth_data = cv2.resize(depth_data.astype(float), (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        return index, color_data, depth_data, self.poses[index]
    

class RealsenseROS(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        # ---- ROS config ----
        self.frames_topic = dataset_config.get("frames_topic", "/frames")
        self.queue_size   = int(dataset_config.get("queue_size", 10))
        self.max_buffer   = int(dataset_config.get("max_buffer", 10))
        self.ros_virtual_length = int(dataset_config.get("ros_virtual_length", 3000))

        self._bridge = CvBridge()
        self._buffer = deque(maxlen=self.max_buffer)
        self._buffer_new = False
        self._lock   = threading.Lock()

        self._history_color = []
        self._history_depth = []
        self._pose_I = np.eye(4, dtype=np.float32)
        self.poses   = [self._pose_I for _ in range(self.ros_virtual_length)]

        rgb_dir = self.dataset_path / "rgb"
        depth_dir = self.dataset_path / "depth"
        self.color_paths = [str(rgb_dir / f"{i:06d}.png") for i in range(self.ros_virtual_length)]
        self.depth_paths = [str(depth_dir / f"{i:06d}.png") for i in range(self.ros_virtual_length)]

        # ---- ROS init ----
        try:
            if not rospy.core.is_initialized():
                rospy.init_node("RealsenseROS", anonymous=True, disable_signals=True)
        except Exception:
            pass

        cls, real_topic, _ = rostopic.get_topic_class(self.frames_topic, blocking=True)
        if cls is None:
            raise RuntimeError(f"Cannot resolve message type for {self.frames_topic}. Check that the topic exists.")
        self._frames_sub = rospy.Subscriber(real_topic, cls, self._frames_callback, queue_size=self.queue_size)

    # ------------------------- Helpers -------------------------
    def _safe_crop(self, img, edge: int):
        if img is None or edge <= 0:
            return img
        return img[edge:-edge, edge:-edge]

    def _quat_to_mat4(self, x, y, z, w) -> np.ndarray:
        n = np.sqrt(x*x + y*y + z*z + w*w) + 1e-12
        x, y, z, w = x/n, y/n, z/n, w/n
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        R = np.array([
            [1 - 2*(yy+zz), 2*(xy - wz),   2*(xz + wy)],
            [2*(xy + wz),   1 - 2*(xx+zz), 2*(yz - wx)],
            [2*(xz - wy),   2*(yz + wx),   1 - 2*(xx+yy)]
        ], dtype=np.float32)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        return T

    def _pose_to_mat4(self, p: Pose) -> np.ndarray:
        T = self._quat_to_mat4(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)
        T[:3, 3] = [p.position.x, p.position.y, p.position.z]
        return T

    def _frames_callback(self, msg):
        try:
            cv_bgr = self._bridge.imgmsg_to_cv2(msg.rgb, desired_encoding="bgr8")
            color  = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            rospy.logwarn(f"[RealsenseROS] RGB convert failed: {e}")
            return

        try:
            cv_depth = self._bridge.imgmsg_to_cv2(msg.depth, desired_encoding="passthrough")
            depth = cv_depth.astype(np.float32)
            if self.depth_scale not in (0.0, 1.0):
                depth /= float(self.depth_scale)
        except Exception as e:
            rospy.logwarn(f"[RealsenseROS] Depth convert failed: {e}")
            return

        try:
            pose = self._pose_to_mat4(msg.pose)
        except Exception as e:
            rospy.logwarn(f"[RealsenseROS] Pose convert failed: {e}")
            pose = self._pose_I

        with self._lock:
            self._buffer.append((color, depth, pose))
            self._buffer_new = True
            # index = len(self._history_color)
            # if index < self.ros_virtual_length:
            #     self._history_color.append(color)
            #     self._history_depth.append(depth)
            #     self.poses[index] = pose
            # else:
            #     rospy.logwarn_throttle(2.0, "[RealsenseROS] Max frame count reached; stop recording.")

    # ------------------------- Dataset API -------------------------
    def __len__(self):
        return self.ros_virtual_length

    def __getitem__(self, index):
        edge  = int(self.crop_edge)
        with self._lock:
            n_hist = len(self._history_color)
            if 0 <= index < n_hist:
                color = self._history_color[index]
                depth = self._history_depth[index]
                pose = self.poses[index]
                return index, self._safe_crop(color, edge), self._safe_crop(depth, edge), pose

        t0 = time.time()
        while True:
            with self._lock:
                if index != n_hist:
                    print("DANGEROUS INDEX!!!")
                    continue
                if self._buffer_new:
                    self._buffer_new = False
                    color, depth, pose = self._buffer[-1]
                    self._history_color.append(color)
                    self._history_depth.append(depth)
                    self.poses[index] = pose

                    if False: 
                        self._save_frame_sync(index, color, depth)
                    return index, self._safe_crop(color, edge), self._safe_crop(depth, edge), pose
                    
            if time.time() - t0 > 5.0:
                raise RuntimeError("RealsenseROS: wait for new frame timeout.")
            time.sleep(0.001)

    def _save_frame_sync(self, idx, color, depth):
        rgb_dir = self.dataset_path / "rgb"
        depth_dir = self.dataset_path / "depth"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)
        
        idx_str = f"{idx:06d}"
        try:
            cv2.imwrite(str(rgb_dir / f"{idx_str}.png"), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
            scale = float(self.depth_scale)
            depth_u16 = (np.clip(depth * scale, 0, np.iinfo(np.uint16).max)).astype(np.uint16)
            cv2.imwrite(str(depth_dir / f"{idx_str}.png"), depth_u16)
        except Exception as e:
            rospy.logwarn(f"[RealsenseROS] Sync save failed @ {idx}: {e}")

    def close(self):
        try:
            rospy.signal_shutdown("RealsenseROS closed")
        except Exception:
            pass


class Realsense2(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.ros_virtual_length = int(dataset_config.get("ros_virtual_length", 3884))
        self.color_paths = sorted(
            list((self.dataset_path / "rgb").glob("*.png")))
        self.depth_paths = sorted(
            list((self.dataset_path / "depth").glob("*.png")))
        # self.load_poses(self.dataset_path / "traj.txt")
        self._pose_I = np.eye(4, dtype=np.float32)
        self.poses   = [self._pose_I for _ in range(self.ros_virtual_length)]

    def load_poses(self, path):
        from scipy.spatial.transform import Rotation as R
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith('#'): continue
            parts = list(map(float, line.split()))
            if len(parts) != 8: continue 
            t = np.array(parts[1:4])
            q = np.array(parts[4:8])
            c2w = np.eye(4)
            c2w[:3, 3] = t
            c2w[:3, :3] = R.from_quat(q).as_matrix()
            self.poses.append(c2w.astype(np.float32))

    def __len__(self):
        return len(self.color_paths)

    def __getitem__(self, index):
        color = cv2.cvtColor(cv2.imread(self.color_paths[index]), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(self.depth_paths[index], cv2.IMREAD_UNCHANGED).astype(np.float32) / float(self.depth_scale)
        edge = self.crop_edge
        if edge > 0:
            color = color[edge:-edge, edge:-edge]
            depth = depth[edge:-edge, edge:-edge]
        return index, color, depth, self.poses[index]


class ActiveSplat(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.ros_virtual_length = int(dataset_config.get("ros_virtual_length", 1500))
        self.color_paths = sorted(
            list((self.dataset_path).glob("frame*.png")))
        self.depth_paths = sorted(
            list((self.dataset_path).glob("depth*.png")))
        self.load_poses(self.dataset_path / "traj.txt")
        print(f"Loaded {len(self.color_paths)} frames")

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for line in lines:
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            self.poses.append(c2w.astype(np.float32))

    def __len__(self):
        return self.ros_virtual_length

    def __getitem__(self, index):
        color = cv2.cvtColor(cv2.imread(self.color_paths[index]), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(self.depth_paths[index], cv2.IMREAD_UNCHANGED).astype(np.float32) / float(self.depth_scale)
        edge = self.crop_edge
        if edge > 0:
            color = color[edge:-edge, edge:-edge]
            depth = depth[edge:-edge, edge:-edge]
        return index, color, depth, self.poses[index]


def get_dataset(dataset_name: str):
    if dataset_name == "replica":
        return Replica
    elif dataset_name == "tum_rgbd":
        return TUM_RGBD
    elif dataset_name == "scan_net":
        return ScanNet
    elif dataset_name == "scannetpp":
        return ScanNetPP
    elif dataset_name == "realsense":
        return RealsenseROS
    elif dataset_name == "realsense2":
        return Realsense2
    elif dataset_name == "activesplat":
        return ActiveSplat
    raise NotImplementedError(f"Dataset {dataset_name} not implemented")
