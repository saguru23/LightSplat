import os
import time
import numpy as np
import torch

from src.entities.datasets import BaseDataset 
from src.tools.lgVO import lightglueVO 
from src.tools.map import Mapper
from src.tools.track import Tracker


class lightglueSLAM:
    def __init__(self, config: dict, dataset: BaseDataset):

        self.dataset = dataset
        self.config = config
        
        self.lgvo = lightglueVO(config, dataset)
        self.mapper = Mapper(config, dataset)
        self.tracker = Tracker(config, dataset, self.lgvo, self.mapper)
        
        print("lightglueSLAM Initializing ...")
        
    def step(self, frame_id, pose_history: np.ndarray):
        start_time = time.time()

        with torch.no_grad():
            estimated_c2w, _is_keyframe = self.tracker.track(frame_id, pose_history)
            optimized_poses_dict = self.mapper.map()
        
        end_time = time.time()
        print(f"总耗时 {(end_time-start_time)*1000:.1f} ms\n")

        return estimated_c2w, _is_keyframe, optimized_poses_dict
    
    def restart(self, frame_id: int, current_c2w):

        self.mapper.keyframes.clear()
        self.mapper.mappoints.clear()

        self.tracker._load_current_frame_data(frame_id)
        self.tracker._catch_features()
        self.tracker.current_c2w = current_c2w
        self.tracker.create_new_keyframe(frame_id)
        
        current_kf = self.mapper.keyframe_buffer.popleft()
        self.mapper.create_new_mappoints(current_kf)
    