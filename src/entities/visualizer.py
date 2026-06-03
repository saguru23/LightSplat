import open3d as o3d
import numpy as np
import torch
import threading, time


class LightViewer:
    def __init__(self, curr_axis_size=0.06, line_width=3.0, kf_size=0.03):
        self.pts = []
        self.full_c2ws = []   # Latest full trajectory poses.
        self.kf_indices = []  # Store keyframe indices, not absolute poses.
        self._dirty = False
        self._last_c2w = None
        
        self._curr_axis_size = curr_axis_size
        self._line_width = line_width
        self._kf_size = kf_size
        
        self._line_set = None
        self._kf_line_set = None
        self._curr_frame = None
        
        self._lock = threading.Lock()
        self._running, self._ready = True, threading.Event()
        threading.Thread(target=self._render_loop, daemon=True).start()
        self._ready.wait()

    def _point_visible_in_view(self, vis, p_world):
        vc = vis.get_view_control()
        cam = vc.convert_to_pinhole_camera_parameters()
        K = cam.intrinsic.intrinsic_matrix
        E = cam.extrinsic
        w, h = cam.intrinsic.width, cam.intrinsic.height
        pw = np.array([p_world[0], p_world[1], p_world[2], 1.0], np.float64)
        pc = (E @ pw)[:3]
        if pc[2] <= 0:  
            return False
        u = K[0,0]*pc[0]/pc[2] + K[0,2]
        v = K[1,1]*pc[1]/pc[2] + K[1,2]
        return (0 <= u < w) and (0 <= v < h)

    def _create_frustums(self, c2ws):
        points = []
        lines = []
        colors = []
        
        w, h, z = self._kf_size, self._kf_size * 0.75, self._kf_size
        base_pts = np.array([
            [0, 0, 0],         
            [-w, -h, z],       
            [ w, -h, z],       
            [ w,  h, z],       
            [-w,  h, z]        
        ])
        
        base_lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],
            [1, 2], [2, 3], [3, 4], [4, 1]
        ]
        
        for i, c2w in enumerate(c2ws):
            R = c2w[:3, :3]
            t = c2w[:3, 3]
            pts_world = (R @ base_pts.T).T + t
            
            offset = i * 5
            cur_lines = [[u + offset, v + offset] for u, v in base_lines]
            
            points.append(pts_world)
            lines.extend(cur_lines)
            colors.extend([[0.25, 0.55, 0.75]] * 8) 
            
        if not points:
            return np.zeros((0, 3)), np.zeros((0, 2), dtype=np.int32), np.zeros((0, 3))
            
        return np.vstack(points), np.array(lines, dtype=np.int32), np.array(colors)

    def _render_loop(self):
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        vis = o3d.visualization.Visualizer()
        # Use a smaller default window.
        vis.create_window(window_name="Trajectory & Keyframes", width=1100, height=700)
        opt = vis.get_render_option()
        
        opt.background_color = np.array([0.0, 0.0, 0.0]) 
        opt.line_width = self._line_width

        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1),
                         reset_bounding_box=False)
        self._ready.set()

        while self._running:
            pts = None; kf_c2ws_copy = None; c2w = None
            with self._lock:
                if self._dirty:
                    if self.pts:
                        pts = np.vstack(self.pts).astype(np.float64)
                    
                    # Fetch the latest keyframe poses.
                    if self.full_c2ws is not None and self.kf_indices:
                        # Guard against out-of-range indices.
                        valid_indices = [idx for idx in self.kf_indices if idx < len(self.full_c2ws)]
                        kf_c2ws_copy = [self.full_c2ws[idx] for idx in valid_indices]
                    
                    c2w = self._last_c2w
                    self._dirty = False

            if pts is not None:
                n = len(pts)
                if n >= 2:
                    if self._line_set is None:
                        self._line_set = o3d.geometry.LineSet()
                        vis.add_geometry(self._line_set, reset_bounding_box=False)
                    lines = np.column_stack((np.arange(n-1, dtype=np.int32),
                                             np.arange(1,   n, dtype=np.int32)))
                    self._line_set.points = o3d.utility.Vector3dVector(pts)
                    self._line_set.lines  = o3d.utility.Vector2iVector(lines)
                    self._line_set.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]] * (n-1)) 
                    vis.update_geometry(self._line_set)
                
                if kf_c2ws_copy is not None:
                    if self._kf_line_set is None:
                        self._kf_line_set = o3d.geometry.LineSet()
                        vis.add_geometry(self._kf_line_set, reset_bounding_box=False)
                    
                    f_pts, f_lines, f_colors = self._create_frustums(kf_c2ws_copy)
                    if len(f_pts) > 0:
                        self._kf_line_set.points = o3d.utility.Vector3dVector(f_pts)
                        self._kf_line_set.lines  = o3d.utility.Vector2iVector(f_lines)
                        self._kf_line_set.colors = o3d.utility.Vector3dVector(f_colors)
                        vis.update_geometry(self._kf_line_set)

                if c2w is not None:
                    if self._curr_frame is not None:
                        vis.remove_geometry(self._curr_frame, reset_bounding_box=False)
                    self._curr_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=self._curr_axis_size)
                    self._curr_frame.transform(c2w)
                    vis.add_geometry(self._curr_frame, reset_bounding_box=False)

                if n >= 1 and not self._point_visible_in_view(vis, pts[-1]):
                    vis.reset_view_point(True)

            vis.poll_events(); vis.update_renderer()
            time.sleep(0.01)

        vis.destroy_window()

    def update(self, c2ws, is_keyframe=False):
        c2ws = c2ws.detach().cpu().numpy() if hasattr(c2ws, "detach") else np.asarray(c2ws)
        with self._lock:
            self.full_c2ws = c2ws  # Store the full trajectory.
            self.pts = [p[:3, 3].reshape(1, 3) for p in c2ws]
            self._last_c2w = c2ws[-1]
            
            if is_keyframe:
                current_idx = len(c2ws) - 1
                # Avoid adding the same frame twice.
                if not self.kf_indices or self.kf_indices[-1] != current_idx:
                    self.kf_indices.append(current_idx)
                    
            self._dirty = True

    def close(self):
        self._running = False


class o3dImageViewer:
    def __init__(self, window_name="Rendered Image"):
        self.viz = o3d.visualization.Visualizer()
        self.viz.create_window(window_name, visible=True)
        self.img = None

    def update(self, img, mask=None):
        with torch.no_grad():
            if mask is not None:
                mask = mask.squeeze(0)
                img = img * mask.float()
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            if img.shape[0] in (1, 3):
                img = np.transpose(img, (1, 2, 0))
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)
            img = np.ascontiguousarray(img)

            if self.img is not None:
                self.viz.clear_geometries()
            self.img = o3d.geometry.Image(img)
            self.viz.add_geometry(self.img)
            self.viz.poll_events()
            self.viz.update_renderer()
            time.sleep(0.05)

    def close(self):
        self.viz.destroy_window()
