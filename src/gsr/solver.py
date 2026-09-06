import torch, roma
import numpy as np
import copy
import time

from src.gsr.renderer import render
from src.gsr.loss import get_loss_tracking
from src.gsr.overlap import compute_overlap_gaussians
from src.utils.pose_utils import update_pose


class CustomPipeline:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False

def viewpoint_localizer(viewpoint, gaussians, base_lr: float=1e-3, fast_mode=False):
    """Localize a single viewpoint in a 3DGS

    Args:
        viewpoint (Camera): Camera instance
        gaussians (Gaussians): 3D Gaussians to locate the viewpoint
        base_lr (float, optional). Defaults to 1e-3.

    Returns:
        _type_: _description_
    """
    opt_params = []
    pipe = CustomPipeline()
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda", requires_grad=False)
    config = {
        'Training': {
            'monocular': False,
            "rgb_boundary_threshold": 0.01,
        }
    }

    init_T = viewpoint.get_T.detach()
    
    opt_params.append(
        {
            "params": [viewpoint.cam_rot_delta],
            "lr": 3*base_lr,
            "name": "rot_{}".format(viewpoint.uid),
        }
    )
    opt_params.append(
        {
            "params": [viewpoint.cam_trans_delta],
            "lr": base_lr,
            "name": "trans_{}".format(viewpoint.uid),
        }
    )
    opt_params.append(
        {
            "params": [viewpoint.exposure_a],
            "lr": 0.01,
            "name": "exposure_a_{}".format(viewpoint.uid),
        }
    )
    opt_params.append(
        {
            "params": [viewpoint.exposure_b],
            "lr": 0.01,
            "name": "exposure_b_{}".format(viewpoint.uid),
        }
    )
    optimizer = torch.optim.Adam(opt_params)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.98, patience=5, verbose=False)
    
    loss_log = []
    opt_iterations = 100
    if (fast_mode): 
        opt_iterations = 10
    for tracking_itr in range(opt_iterations):
        optimizer.zero_grad()
        render_pkg = render(
            viewpoint, gaussians, pipe, bg_color
        )
        image, depth, opacity = (
            render_pkg["render"],
            render_pkg["depth"],
            render_pkg["opacity"],
        )
        
        loss = get_loss_tracking(config, image, depth, opacity, viewpoint)
        loss.backward()
        loss_log.append(loss.item())
        
        with torch.no_grad():
            optimizer.step()
            scheduler.step(loss)
            converged = update_pose(viewpoint)
        
        if converged:
            break
    
    rel_tsfm = (init_T.inverse() @ viewpoint.get_T).inverse()
    loss_residual = loss.item()
    
    return converged, rel_tsfm, loss_residual, loss_log

def gaussian_registration_nromal(src_dict, tgt_dict, config: dict, visualize=False):
    """_summary_

    Args:
        src_dict (dict): dictionary of source gaussians and its keyframes
        tgt_dict (dict): dictionary of target gaussians and its keyframes
        base_lr (float, optional): the base learning rate for optimization. Defaults to 5e-3.

    Returns:
        dict: dictionary of registration result
    """
    
    # print("Pairwise registration ...")
    init_overlap = compute_overlap_gaussians(src_dict['gaussians'], tgt_dict['gaussians'], 0.1)
    if init_overlap< 0.2:
        print("Initial overlap between two submaps are too small, skipping ...")
        return {
            'successful': False,
            "pred_tsfm": torch.eye(4).cuda(),
            "gt_tsfm": torch.eye(4).cuda(),
            "overlap": init_overlap.item()
        }
    
    src_3dgs, src_view_list = copy.deepcopy(src_dict['gaussians']), copy.deepcopy(src_dict['cameras'])
    tgt_3dgs, tgt_view_list = copy.deepcopy(tgt_dict['gaussians']), copy.deepcopy(tgt_dict['cameras'])
    
    # compute gt tsfm
    src_keyframe= src_dict['cameras'][0].get_T.detach()
    src_gt= src_dict['cameras'][0].get_T_gt.detach()
    tgt_keyframe= tgt_dict['cameras'][0].get_T.detach()
    tgt_gt= tgt_dict['cameras'][0].get_T_gt.detach()
    delta_src = src_gt.inverse() @ src_keyframe
    delta_tgt = tgt_gt.inverse() @ tgt_keyframe
    gt_tsfm = delta_tgt.inverse() @ delta_src
    
    # similarity choosing
    device = "cuda" if torch.cuda.is_available() else "cpu"
    src_desc, tgt_desc = src_dict['kf_desc'], tgt_dict['kf_desc']
    
    score_cross = torch.einsum("id,jd->ij", src_desc.to(device), tgt_desc.to(device))
    score_best_src, _ = score_cross.topk(1)
    _, ii = score_best_src.view(-1).topk(2)
    
    score_best_tgt, _ = score_cross.T.topk(1)
    _, jj = score_best_tgt.view(-1).topk(2)
    
    src_view_list = [src_view_list[i.item()] for i in ii]
    tgt_view_list = [tgt_view_list[j.item()] for j in jj]
    
    pred_list, residual_list, converged_list, loss_log_list = [], [], [], []
    
    pipe = CustomPipeline()
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda", requires_grad=False)
    # per-cam
    for viewpoint in src_view_list:
        
        # use rendered image as target not the raw observation
        if config["use_render"]:
            render_pkg = render(viewpoint, src_3dgs, pipe, bg_color)
            viewpoint.load_rgb(render_pkg['render'].detach())
            viewpoint.depth = render_pkg['depth'].squeeze().detach().cpu().numpy()
        else:
            viewpoint.load_rgb()
        converged, pred_tsfm, residual, loss_log = viewpoint_localizer(viewpoint, tgt_3dgs, config["base_lr"])
        pred_list.append(pred_tsfm)
        residual_list.append(residual)
        converged_list.append(converged)
        loss_log_list.append(loss_log)
    
    for viewpoint in tgt_view_list:
        if config["use_render"]:
            render_pkg = render(viewpoint, tgt_3dgs, pipe, bg_color)
            viewpoint.load_rgb(render_pkg['render'].detach())
            viewpoint.depth = render_pkg['depth'].squeeze().detach().cpu().numpy()
        else:
            viewpoint.load_rgb()
        converged, pred_tsfm, residual, loss_log = viewpoint_localizer(viewpoint, src_3dgs, config["base_lr"])
        pred_list.append(pred_tsfm.inverse())
        residual_list.append(residual)
        converged_list.append(converged)
        loss_log_list.append(loss_log)
    
    
    pred_tsfms = torch.stack(pred_list)
    residuals = torch.Tensor(residual_list).cuda().float()
    # probability based on residuals
    prob = 1/residuals / (1/residuals).sum()
    
    M = torch.sum(prob[:, None, None] * pred_tsfms[:,:3,:3], dim=0)
    try:
        R_w = roma.special_procrustes(M)
    except Exception as e:
        print(f"Error in roma.special_procrustes: {e}")
        return {
            'successful': False,
            "pred_tsfm": torch.eye(4).cuda(),
            "gt_tsfm": torch.eye(4).cuda(),
            "overlap": init_overlap.item()
        }
    t_w = torch.sum(prob[:, None] * pred_tsfms[:,:3, 3], dim=0)
    
    best_tsfm = torch.eye(4).cuda().float()
    best_tsfm[:3,:3]  = R_w
    best_tsfm[:3, 3]  = t_w
    
    result_dict = {
        "gt_tsfm": gt_tsfm,
        "pred_tsfm": best_tsfm,
        "successful": True,
        "best_viewpoint": src_view_list[0].get_T
    }
    
    if visualize:
        import matplotlib
        import matplotlib.pyplot as plt
        from src.gsr.utils import visualize_registration
        matplotlib.use('TkAgg')
        plt.figure(figsize=(10, 6))
        for log in loss_log_list:
            plt.plot(log)

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curves of 10 Independent Optimizations')
        plt.legend([f'Run {i+1}' for i in range(len(loss_log_list))], loc='upper right')
        plt.grid(True)
        plt.show()
        
        visualize_registration(src_3dgs, tgt_3dgs, best_tsfm, gt_tsfm)
    
    del src_3dgs, src_view_list, tgt_3dgs, tgt_view_list
    return result_dict

def gaussian_registration_fast(src_dict, tgt_dict, config: dict, visualize=False, lg=None, fast_mode=False):
    """_summary_

    Args:
        src_dict (dict): dictionary of source gaussians and its keyframes
        tgt_dict (dict): dictionary of target gaussians and its keyframes
        base_lr (float, optional): the base learning rate for optimization. Defaults to 5e-3.

    Returns:
        dict: dictionary of registration result
    """
    
    # print("Pairwise registration ...")    
    init_overlap = compute_overlap_gaussians(src_dict['gaussians'], tgt_dict['gaussians'], 0.1)
    if init_overlap< 0.05:
        # print("Initial overlap between two submaps are too small, skipping ...")
        return {
            'successful': False,
            "pred_tsfm": torch.eye(4).cuda(),
            "gt_tsfm": torch.eye(4).cuda(),
            "overlap": init_overlap.item()
        }
    
    src_3dgs, full_src_view_list = copy.deepcopy(src_dict['gaussians']), copy.deepcopy(src_dict['cameras'])
    tgt_3dgs, full_tgt_view_list = copy.deepcopy(tgt_dict['gaussians']), copy.deepcopy(tgt_dict['cameras'])
    
    # compute gt tsfm
    src_keyframe= src_dict['cameras'][0].get_T.detach()
    src_gt= src_dict['cameras'][0].get_T_gt.detach()
    tgt_keyframe= tgt_dict['cameras'][0].get_T.detach()
    tgt_gt= tgt_dict['cameras'][0].get_T_gt.detach()
    delta_src = src_gt.inverse() @ src_keyframe
    delta_tgt = tgt_gt.inverse() @ tgt_keyframe
    gt_tsfm = delta_tgt.inverse() @ delta_src
    
    # similarity choosing
    device = "cuda" if torch.cuda.is_available() else "cpu"
    src_desc, tgt_desc = src_dict['kf_desc'], tgt_dict['kf_desc']
    
    score_cross = torch.einsum("id,jd->ij", src_desc.to(device), tgt_desc.to(device))
    score_best_src, _ = score_cross.topk(1)
    _, ii = score_best_src.view(-1).topk(2)
    
    score_best_tgt, _ = score_cross.T.topk(1)
    _, jj = score_best_tgt.view(-1).topk(2)
    
    src_view_list = [full_src_view_list[i.item()] for i in ii]
    tgt_view_list = [full_tgt_view_list[j.item()] for j in jj]
    
    pred_list, residual_list, converged_list, loss_log_list = [], [], [], []

    src_original_poses = {cam.uid: cam.get_T.detach().clone() for cam in src_view_list}
    tgt_original_poses = {cam.uid: cam.get_T.detach().clone() for cam in tgt_view_list}
    match_found = False
    candidate_list = []
    topk = min(4, score_cross.numel())
    _, flat_indices = score_cross.view(-1).topk(topk)
    src_indices = flat_indices // score_cross.shape[1]
    tgt_indices = flat_indices % score_cross.shape[1]
    for k in range(topk):
        s_idx = src_indices[k].item()
        t_idx = tgt_indices[k].item()
        s_cam_match = full_src_view_list[s_idx]
        t_cam_match = full_tgt_view_list[t_idx]

        T_rel, is_ok, stats = lg.update(s_cam_match.uid, t_cam_match.uid)
        if not is_ok:
            continue

        rmse = stats.get("rmse", float("inf"))
        if not np.isfinite(rmse):
            continue

        candidate_list.append({
            "score": rmse,
            "T_rel": T_rel,
            "s_cam": s_cam_match,
            "t_cam": t_cam_match,
            "stats": stats,
        })

        if rmse < 0.05 and stats.get("inlier_num", 0) > 100:
            break

    if len(candidate_list) > 0:
        best = min(candidate_list, key=lambda x: x["score"])
        T_rel = torch.from_numpy(best["T_rel"]).float().to(device)
        s_cam_match = best["s_cam"]
        t_cam_match = best["t_cam"]
        stats = best["stats"]

        # print(
        #     f"[LC] best LightGlue pair src={s_cam_match.uid}, tgt={t_cam_match.uid}, "
        #     f"rmse={stats.get('rmse', float('nan')):.4f}, "
        #     f"median={stats.get('median', float('nan')):.4f}, "
        #     f"inliers={stats.get('inlier_num', 0)}/{stats.get('valid_num', 0)}, "
        #     f"ratio={stats.get('inlier_ratio', 0.0):.2f}"
        # )

        T_tgt_old = t_cam_match.get_T.detach()
        T_src_old = s_cam_match.get_T.detach()
        # T_rel maps source-camera coordinates to target-camera coordinates.
        # Camera.get_T stores world-to-camera. The induced submap transform maps
        # source-world coordinates into target-world coordinates.
        source_to_target = T_tgt_old.inverse() @ T_rel @ T_src_old
        target_to_source = source_to_target.inverse()

        match_found = True

    if not match_found:
        return {
            'successful': False,
            "pred_tsfm": torch.eye(4).cuda(),
            "gt_tsfm": torch.eye(4).cuda(),
            "overlap": init_overlap.item()
        }
    
    start_time = time.time()
    pipe = CustomPipeline()
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda", requires_grad=False)
    # per-cam
    best_pred_tsfm = torch.eye(4).cuda().float()
    valid_flag = False
    for viewpoint in src_view_list:
        
        # use rendered image as target not the raw observation
        if config["use_render"]:
            render_pkg = render(viewpoint, src_3dgs, pipe, bg_color)
            viewpoint.load_rgb(render_pkg['render'].detach())
            viewpoint.depth = render_pkg['depth'].squeeze().detach().cpu().numpy()
        else:
            viewpoint.load_rgb()
        curr_T = viewpoint.get_T.detach()
        new_T = curr_T @ target_to_source
        viewpoint.update_RT(new_T[:3, :3], new_T[:3, 3])
        converged, pred_tsfm, residual, loss_log = viewpoint_localizer(viewpoint, tgt_3dgs, config["base_lr"], fast_mode)

        final_T = viewpoint.get_T.detach()
        init_T_backup = src_original_poses[viewpoint.uid]
        pred_tsfm = (init_T_backup.inverse() @ final_T).inverse()
        
        pred_list.append(pred_tsfm)
        residual_list.append(residual)
        converged_list.append(converged)
        loss_log_list.append(loss_log)

        if residual < 0.001:
            best_pred_tsfm = pred_tsfm
            valid_flag = True
            break

    if not valid_flag:        
        for viewpoint in tgt_view_list:
            if config["use_render"]:
                render_pkg = render(viewpoint, tgt_3dgs, pipe, bg_color)
                viewpoint.load_rgb(render_pkg['render'].detach())
                viewpoint.depth = render_pkg['depth'].squeeze().detach().cpu().numpy()
            else:
                viewpoint.load_rgb()
            curr_T = viewpoint.get_T.detach()
            new_T = curr_T @ source_to_target
            viewpoint.update_RT(new_T[:3, :3], new_T[:3, 3])
            converged, pred_tsfm, residual, loss_log = viewpoint_localizer(viewpoint, src_3dgs, config["base_lr"], fast_mode)

            final_T = viewpoint.get_T.detach()
            init_T_backup = tgt_original_poses[viewpoint.uid]
            pred_tsfm = (init_T_backup.inverse() @ final_T).inverse()

            pred_list.append(pred_tsfm.inverse())
            residual_list.append(residual)
            converged_list.append(converged)
            loss_log_list.append(loss_log)

            if residual < 0.001:
                best_pred_tsfm = pred_tsfm.inverse()
                valid_flag = True
                break

    # print("relocate once(s): ", time.time()-start_time)
    
    if valid_flag:
        best_tsfm = torch.eye(4).cuda().float()
        best_tsfm[:3, :3] = best_pred_tsfm[:3, :3]
        best_tsfm[:3, 3] = best_pred_tsfm[:3, 3]
        result_dict = {
            "gt_tsfm": gt_tsfm,
            "pred_tsfm": best_tsfm,
            "successful": True,
            "best_viewpoint": src_view_list[0].get_T
        }
    else:
        pred_tsfms = torch.stack(pred_list)
        residuals = torch.Tensor(residual_list).cuda().float()
        # probability based on residuals
        prob = 1/residuals / (1/residuals).sum()
        
        M = torch.sum(prob[:, None, None] * pred_tsfms[:,:3,:3], dim=0)
        try:
            R_w = roma.special_procrustes(M)
        except Exception as e:
            print(f"Error in roma.special_procrustes: {e}")
            return {
                'successful': False,
                "pred_tsfm": torch.eye(4).cuda(),
                "gt_tsfm": torch.eye(4).cuda(),
                "overlap": init_overlap.item()
            }
        t_w = torch.sum(prob[:, None] * pred_tsfms[:,:3, 3], dim=0)
        
        best_tsfm = torch.eye(4).cuda().float()
        best_tsfm[:3,:3]  = R_w
        best_tsfm[:3, 3]  = t_w
        
        result_dict = {
            "gt_tsfm": gt_tsfm,
            "pred_tsfm": best_tsfm,
            "successful": True,
            "best_viewpoint": src_view_list[0].get_T
        }
    
    if visualize:
        import matplotlib
        import matplotlib.pyplot as plt
        from src.gsr.utils import visualize_registration
        matplotlib.use('TkAgg')
        plt.figure(figsize=(10, 6))
        for log in loss_log_list:
            plt.plot(log)

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curves of 10 Independent Optimizations')
        plt.legend([f'Run {i+1}' for i in range(len(loss_log_list))], loc='upper right')
        plt.grid(True)
        plt.show()
        
        visualize_registration(src_3dgs, tgt_3dgs, best_tsfm, gt_tsfm)
    
    del src_3dgs, src_view_list, tgt_3dgs, tgt_view_list
    return result_dict

def gaussian_registration(src_dict, tgt_dict, config: dict, visualize=False, lg=None, mode="normal"):
    if mode == "normal":
        return gaussian_registration_nromal(src_dict, tgt_dict, config, visualize)
    elif mode == "fast":
        return gaussian_registration_fast(src_dict, tgt_dict, config, visualize, lg)
    elif mode == "fastest":
        return gaussian_registration_fast(src_dict, tgt_dict, config, visualize, lg, True)
    else:
        raise ValueError(f"Unknown registration mode: {mode}")
