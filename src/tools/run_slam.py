import os
import numpy as np
import torch
import time

from src.utils.io_utils import load_config
from src.entities.datasets import get_dataset, BaseDataset
from src.entities.visualizer import LoopSplatViewer
from src.tools.slam import lightglueSLAM

def PreloadDataset(dataset):

    OriginalClass = type(dataset)

    class PreloadDataset(OriginalClass):
        def __init__(self, dataset: BaseDataset):
            self.dataset = dataset
            self.cache = []
            
            print(f"[System] 正在预加载 {len(dataset)} 帧数据到内存 (RAM)...")
            for i in range(len(dataset)):
                data = dataset[i] 
                self.cache.append(data)

        def __getitem__(self, idx):
            return self.cache[idx]

        def __len__(self):
            return len(self.cache)
        
        # 允许访问原始 dataset 的其他属性
        def __getattr__(self, name):
            return getattr(self.dataset, name)
        
    return PreloadDataset(dataset)

def main():
    config_path = "configs/Realsense.yaml"
    config = load_config(config_path)
    dataset = get_dataset(config["dataset_name"])({**config["data"], **config["cam"]})
    # dataset = PreloadDataset(dataset)
    
    slam = lightglueSLAM(config["light"], dataset)
    viewer = LoopSplatViewer()
    
    estimated_c2ws = np.zeros((len(dataset), 4, 4), dtype=np.float64)
    is_keyframe = np.full(len(dataset), False, dtype=bool)

    for frame_id in range(len(dataset)):
        hist_idx = [0, max(0, frame_id - 2), max(0, frame_id - 1)]
        estimated_c2w, _is_keyframe, optimized_poses_dict = slam.step(frame_id, estimated_c2ws[hist_idx])

        estimated_c2ws[frame_id] = estimated_c2w
        is_keyframe[frame_id] = _is_keyframe

        if optimized_poses_dict:
            for kf_id, corrected in optimized_poses_dict.items():
                estimated_c2ws[kf_id] = corrected

        viewer.update(estimated_c2ws[:frame_id + 1], _is_keyframe)

    viewer.close()

if __name__ == "__main__":
    main()
