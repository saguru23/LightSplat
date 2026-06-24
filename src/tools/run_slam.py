import argparse
import numpy as np
import torch  # noqa: F401. Preload conda shared libs before Open3D/PIL imports.

from src.utils.io_utils import load_config
from src.entities.datasets import get_dataset, BaseDataset
from src.entities.visualizer import LightViewer
from src.tools.slam import lightglueSLAM


def _record_gt_pose(gt_c2ws, gt_valid, frame_id, pose):
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        return

    gt_c2ws[frame_id] = pose
    gt_valid[frame_id] = True


def _translation_error_stats(estimated_t, gt_t):
    error = np.linalg.norm(estimated_t - gt_t, axis=1)
    return {
        "compared_pose_pairs": int(error.shape[0]),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mean": float(np.mean(error)),
        "median": float(np.median(error)),
        "max": float(np.max(error)),
    }


def _align_positions_se3(estimated_t, gt_t):
    est_center = estimated_t.mean(axis=0)
    gt_center = gt_t.mean(axis=0)
    est_zero = estimated_t - est_center
    gt_zero = gt_t - gt_center

    u, _, vt = np.linalg.svd(est_zero.T @ gt_zero)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1
        rot = vt.T @ u.T

    trans = gt_center - rot @ est_center
    return (rot @ estimated_t.T).T + trans


def _compute_ate(estimated_c2ws, gt_c2ws, gt_valid):
    finite_est = np.isfinite(estimated_c2ws).all(axis=(1, 2))
    finite_gt = np.isfinite(gt_c2ws).all(axis=(1, 2))
    valid = gt_valid & finite_est & finite_gt

    if valid.sum() < 2:
        return None, None, int(valid.sum())

    estimated_t = estimated_c2ws[valid, :3, 3]
    gt_t = gt_c2ws[valid, :3, 3]
    ate = _translation_error_stats(estimated_t, gt_t)
    aligned_t = _align_positions_se3(estimated_t, gt_t)
    ate_aligned = _translation_error_stats(aligned_t, gt_t)
    return ate, ate_aligned, int(valid.sum())


def _print_ate(estimated_c2ws, gt_c2ws, gt_valid):
    ate, ate_aligned, valid_count = _compute_ate(estimated_c2ws, gt_c2ws, gt_valid)
    if ate is None:
        print(f"[Eval] ATE skipped: only {valid_count} valid GT poses.")
        return

    valid_gt_t = gt_c2ws[gt_valid, :3, 3]
    gt_motion = np.linalg.norm(valid_gt_t - valid_gt_t[0], axis=1).max()
    if gt_motion < 1e-6:
        print("[Eval] Warning: GT trajectory has near-zero motion; ATE may be invalid without real GT poses.")

    print(
        "[Eval] ATE RMSE: "
        f"{ate['rmse'] * 100:.2f} cm | "
        f"mean: {ate['mean'] * 100:.2f} cm | "
        f"median: {ate['median'] * 100:.2f} cm | "
        f"max: {ate['max'] * 100:.2f} cm | "
        f"poses: {ate['compared_pose_pairs']}"
    )
    print(
        "[Eval] ATE RMSE (SE3 aligned): "
        f"{ate_aligned['rmse'] * 100:.2f} cm"
    )


def PreloadDataset(dataset):

    OriginalClass = type(dataset)

    class PreloadDataset(OriginalClass):
        def __init__(self, dataset: BaseDataset):
            self.dataset = dataset
            self.cache = []

            print(f"[System] Preloading {len(dataset)} frames into RAM...")
            for i in range(len(dataset)):
                data = dataset[i]
                self.cache.append(data)

        def __getitem__(self, idx):
            return self.cache[idx]

        def __len__(self):
            return len(self.cache)

        # Forward other attributes to the original dataset.
        def __getattr__(self, name):
            return getattr(self.dataset, name)

    return PreloadDataset(dataset)


def get_args():
    parser = argparse.ArgumentParser(description="Run the LightGlue front-end only.")
    parser.add_argument("config_path", nargs="?", default="configs/realsense.yaml")
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--no-ate", action="store_true", help="Skip final ATE evaluation.")
    return parser.parse_args()


def main():
    args = get_args()
    config = load_config(args.config_path)
    dataset = get_dataset(config["dataset_name"])({**config["data"], **config["cam"]})
    if args.preload:
        dataset = PreloadDataset(dataset)

    slam = lightglueSLAM(config["light"], dataset)
    viewer = LightViewer()

    estimated_c2ws = np.zeros((len(dataset), 4, 4), dtype=np.float64)
    gt_c2ws = np.zeros((len(dataset), 4, 4), dtype=np.float64)
    gt_valid = np.full(len(dataset), False, dtype=bool)
    is_keyframe = np.full(len(dataset), False, dtype=bool)

    for frame_id in range(len(dataset)):
        hist_idx = [0, max(0, frame_id - 2), max(0, frame_id - 1)]
        estimated_c2w, _is_keyframe, optimized_poses_dict = slam.step(frame_id, estimated_c2ws[hist_idx])

        estimated_c2ws[frame_id] = estimated_c2w
        is_keyframe[frame_id] = _is_keyframe
        if not args.no_ate:
            _record_gt_pose(gt_c2ws, gt_valid, frame_id, slam.tracker.current_gt_pose)

        if optimized_poses_dict:
            for kf_id, corrected in optimized_poses_dict.items():
                estimated_c2ws[kf_id] = corrected

        viewer.update(estimated_c2ws[:frame_id + 1], _is_keyframe)

    viewer.close()
    if not args.no_ate:
        _print_ate(estimated_c2ws, gt_c2ws, gt_valid)


if __name__ == "__main__":
    main()
