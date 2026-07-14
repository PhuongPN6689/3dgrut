import open3d as o3d
import numpy as np
import os
from typing import List, Dict, Any
from ..geometry.camera import Camera
from ..geometry.pose import qvec2rotmat
from ..mapping.filtering import (
    filter_by_track_length,
    filter_by_baseline_angle,
    filter_by_depth_and_front,
    filter_by_reprojection_error,
    filter_by_consistency,
    compute_max_baseline_angle,
    get_max_valid_distance
)

def get_camera_frustum_geometry(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    width: int,
    height: int,
    scale: float = 0.1,
    color: List[float] = [1.0, 0.0, 0.0]
) -> o3d.geometry.LineSet:
    """Create a LineSet representing the camera frustum and shooting direction."""
    R_c2w = R.T
    C = -R_c2w @ t
    
    K_inv = np.linalg.inv(K)
    corners_cam = [
        K_inv @ np.array([0, 0, 1.0]) * scale,
        K_inv @ np.array([width, 0, 1.0]) * scale,
        K_inv @ np.array([width, height, 1.0]) * scale,
        K_inv @ np.array([0, height, 1.0]) * scale
    ]
    corners_world = [R_c2w @ (pt_cam - t) for pt_cam in corners_cam]
    
    vertices = [C] + corners_world
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # Rays
        [1, 2], [2, 3], [3, 4], [4, 1]   # Image plane rectangle
    ]
    colors = [color for _ in range(len(lines))]
    
    # Add camera shooting direction indicator (Yellow ray)
    forward_cam = np.array([0, 0, 2.5]) * scale
    forward_world = R_c2w @ (forward_cam - t)
    vertices.append(forward_world)
    lines.append([0, 5]) 
    colors.append([1.0, 1.0, 0.0]) 
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

class Viewer3D:
    def __init__(
        self,
        points3d: List[Dict[str, Any]],
        images_poses: Dict[str, Dict[str, Any]],
        cameras: Dict[int, Camera],
        config: Dict[str, Any],
        cache_dir: str = None
    ):
        self.points3d_all = points3d  # Fallback
        self.images_poses = images_poses
        self.cameras = cameras
        self.config = config
        
        # Load raw unfiltered points if they exist in the cache
        if cache_dir:
            from ..io.feature_io import load_points3d_all
            all_pts = load_points3d_all(cache_dir)
            if all_pts is not None and len(all_pts) > 0:
                self.points3d_all = all_pts
                print(f"[DEBUG MODE] Loaded {len(self.points3d_all)} raw points for interactive tuning.")
                
        # Interactive filter state
        self.point_size = float(config["viewer"]["point_size"])
        self.show_cameras = True
        self.show_points = True
        self.color_mode = "true_color"
        self.filter_error = False
        
        # Filter thresholds
        self.min_track_len = int(config["filter"]["min_track_length"])
        self.min_baseline = float(config["filter"]["min_baseline_angle"])
        
        # Adaptive starting threshold in centimeters (90th percentile of errors, fallback to 10cm)
        all_errs_cm = [pt.get("avg_error_cm", 5.0) for pt in self.points3d_all]
        if len(all_errs_cm) > 0:
            self.reproj_thresh_cm = float(np.percentile(all_errs_cm, 90))
        else:
            self.reproj_thresh_cm = 10.0
        
        self.pcd = o3d.geometry.PointCloud()
        self.frustums = []
        self.trajectory = o3d.geometry.LineSet()
        self.build_geometries()

    def build_geometries(self) -> None:
        """Convert points3d and poses into Open3D geometries."""
        self.update_pcd_geometry()
        
        self.frustums = []
        centers = []
        for img_name, pose in self.images_poses.items():
            cam = self.cameras[pose["camera_id"]]
            K = cam.get_intrinsic_matrix()
            R = qvec2rotmat(pose["qvec"])
            t = pose["tvec"]
            
            frustum = get_camera_frustum_geometry(K, R, t, cam.width, cam.height, scale=0.15)
            self.frustums.append(frustum)
            centers.append((img_name, -R.T @ t))
            
        centers_sorted = sorted(centers, key=lambda x: x[0])
        traj_pts = [c[1] for c in centers_sorted]
        traj_lines = [[i, i+1] for i in range(len(traj_pts)-1)]
        traj_colors = [[0.0, 1.0, 0.0] for _ in range(len(traj_lines))]
        
        self.trajectory.points = o3d.utility.Vector3dVector(traj_pts)
        self.trajectory.lines = o3d.utility.Vector2iVector(traj_lines)
        self.trajectory.colors = o3d.utility.Vector3dVector(traj_colors)

    def update_pcd_geometry(self) -> None:
        """Filter points based on interactive parameters and update coordinates/colors."""
        filtered_pts = []
        max_dist = get_max_valid_distance(self.images_poses)
        
        for pt in self.points3d_all:
            xyz = pt["xyz"]
            track = pt["track"]
            track_len = len(track["image_names"])
            
            # Apply dynamic filters
            if self.filter_error:
                thresh_cm = float(self.config.get("triangulation", {}).get("reprojection_threshold_cm", 10.0))
                if pt.get("avg_error_cm", 999.0) > thresh_cm:
                    continue
            
            if not filter_by_track_length(track_len, self.min_track_len):
                continue
            max_angle = compute_max_baseline_angle(xyz, track["image_names"], self.images_poses)
            if not filter_by_baseline_angle(max_angle, self.min_baseline, track_len):
                continue
            if not filter_by_depth_and_front(xyz, track["image_names"], self.images_poses, max_dist):
                continue
                
            # Reprojection Error in centimeters
            pt_err_cm = pt.get("avg_error_cm")
            if pt_err_cm is None:
                from ..mapping.filtering import compute_point_metric_error
                err_m = compute_point_metric_error(xyz, track, self.images_poses, self.cameras)
                pt["avg_error_cm"] = err_m * 100.0
                pt_err_cm = pt["avg_error_cm"]
                
            if pt_err_cm > self.reproj_thresh_cm:
                continue
                
            # Consistency using config-level pixel threshold
            config_reproj_thresh = float(self.config.get("triangulation", {}).get("reprojection_threshold", 3.0))
            if not filter_by_consistency(xyz, track, self.images_poses, self.cameras, config_reproj_thresh, track_len):
                continue
                
            filtered_pts.append(pt)
            
        xyzs = np.array([pt["xyz"] for pt in filtered_pts]) if len(filtered_pts) > 0 else np.empty((0, 3))
        self.pcd.points = o3d.utility.Vector3dVector(xyzs)
        self.color_filtered_points(filtered_pts)

    def color_filtered_points(self, filtered_pts: list) -> None:
        """Apply colors to point cloud based on color mode."""
        n_points = len(filtered_pts)
        colors = np.zeros((n_points, 3))
        
        if self.color_mode == "true_color":
            for i, pt in enumerate(filtered_pts):
                colors[i] = pt["track"].get("color", [0.5, 0.5, 0.5])
        elif self.color_mode == "slate_blue":
            colors[:] = [0.2, 0.5, 0.8]
        elif self.color_mode == "track_length":
            lengths = np.array([pt["track"]["length"] for pt in filtered_pts], dtype=np.float32)
            max_len = max(lengths) if len(lengths) > 0 else 1
            min_len = min(lengths) if len(lengths) > 0 else 1
            for i, l in enumerate(lengths):
                val = (l - min_len) / max((max_len - min_len), 1e-5)
                colors[i] = [val, 0.2, 1.0 - val]
        elif self.color_mode == "error":
            errors = np.array([pt.get("avg_error_cm", 0.0) for pt in filtered_pts], dtype=np.float32)
            max_err = max(errors) if len(errors) > 0 else 1.0
            min_err = min(errors) if len(errors) > 0 else 0.0
            for i, e in enumerate(errors):
                val = (e - min_err) / max((max_err - min_err), 1e-5)
                colors[i] = [val, 1.0 - val, 0.0]
                
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

    def update_point_colors(self) -> None:
        """Compatibility wrapper."""
        self.update_pcd_geometry()

    # Keyboard Callback Methods
    def cb_toggle_cameras(self, vis):
        self.show_cameras = not self.show_cameras
        print(f"Show Cameras: {self.show_cameras}")
        for frustum in self.frustums:
            if self.show_cameras:
                vis.add_geometry(frustum, reset_bounding_box=False)
            else:
                vis.remove_geometry(frustum, reset_bounding_box=False)
        if self.show_cameras:
            vis.add_geometry(self.trajectory, reset_bounding_box=False)
        else:
            vis.remove_geometry(self.trajectory, reset_bounding_box=False)
        return False

    def cb_toggle_points(self, vis):
        self.show_points = not self.show_points
        print(f"Show Points: {self.show_points}")
        if self.show_points:
            vis.add_geometry(self.pcd, reset_bounding_box=False)
        else:
            vis.remove_geometry(self.pcd, reset_bounding_box=False)
        return False

    def cb_increase_point_size(self, vis):
        self.point_size = min(self.point_size + 1.0, 10.0)
        vis.get_render_option().point_size = self.point_size
        print(f"Point Size: {self.point_size}")
        return True

    def cb_decrease_point_size(self, vis):
        self.point_size = max(self.point_size - 1.0, 1.0)
        vis.get_render_option().point_size = self.point_size
        print(f"Point Size: {self.point_size}")
        return True

    def cb_color_original(self, vis):
        self.color_mode = "true_color"
        self.update_pcd_geometry()
        vis.update_geometry(self.pcd)
        print("Color Mode: True Color")
        return True

    def cb_color_slate_blue(self, vis):
        self.color_mode = "slate_blue"
        self.update_pcd_geometry()
        vis.update_geometry(self.pcd)
        print("Color Mode: Slate Blue")
        return True

    def cb_color_track_length(self, vis):
        self.color_mode = "track_length"
        self.update_pcd_geometry()
        vis.update_geometry(self.pcd)
        print("Color Mode: Track Length")
        return True

    def cb_color_error(self, vis):
        self.color_mode = "error"
        self.update_pcd_geometry()
        vis.update_geometry(self.pcd)
        print("Color Mode: Reprojection Error")
        return True

    def cb_toggle_filter_error(self, vis):
        self.filter_error = not self.filter_error
        thresh = float(self.config.get("triangulation", {}).get("reprojection_threshold", 3.0))
        print(f"Filter hard outliers (avg_error > {thresh}px): {self.filter_error}")
        self.update_pcd_geometry()
        vis.update_geometry(self.pcd)
        return True

    # Parameter Tuning Callback Methods
    def cb_increase_track_len(self, vis):
        self.min_track_len = min(self.min_track_len + 1, 10)
        print(f"[TUNING] Increase min_track_length: {self.min_track_len}")
        self.update_pcd_geometry()
        vis.update_geometry(self.pcd)
        return True

    def cb_decrease_track_len(self, vis):
        self.min_track_len = max(self.min_track_len - 1, 2)
        print(f"[TUNING] Decrease min_track_length: {self.min_track_len}")
        self.update_pcd_geometry()
        vis.update_geometry(self.pcd)
        return True

    def cb_increase_reproj_thresh(self, vis):
        self.reproj_thresh_cm = round(self.reproj_thresh_cm + 1.0, 1)
        print(f"[TUNING] Increase metric reprojection_threshold: {self.reproj_thresh_cm} cm")
        self.update_pcd_geometry()
        vis.update_geometry(self.pcd)
        return True

    def cb_decrease_reproj_thresh(self, vis):
        self.reproj_thresh_cm = max(round(self.reproj_thresh_cm - 1.0, 1), 0.5)
        print(f"[TUNING] Decrease metric reprojection_threshold: {self.reproj_thresh_cm} cm")
        self.update_pcd_geometry()
        vis.update_geometry(self.pcd)
        return True

    def cb_increase_baseline(self, vis):
        self.min_baseline = round(self.min_baseline + 0.5, 2)
        print(f"[TUNING] Increase min_baseline_angle: {self.min_baseline} deg")
        self.update_pcd_geometry()
        vis.update_geometry(self.pcd)
        return True

    def cb_decrease_baseline(self, vis):
        self.min_baseline = max(round(self.min_baseline - 0.5, 2), 0.0)
        print(f"[TUNING] Decrease min_baseline_angle: {self.min_baseline} deg")
        self.update_pcd_geometry()
        vis.update_geometry(self.pcd)
        return True

    def show(self) -> None:
        """Run the interactive Open3D visualization window."""
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="3D Point Map Debug Viewer", width=1280, height=720)
        
        if self.show_points:
            vis.add_geometry(self.pcd)
        if self.show_cameras:
            for frustum in self.frustums:
                vis.add_geometry(frustum)
            vis.add_geometry(self.trajectory)
            
        render_opt = vis.get_render_option()
        render_opt.point_size = self.point_size
        render_opt.background_color = np.array([0.05, 0.05, 0.05])
        
        # Register keys
        vis.register_key_callback(ord("C"), self.cb_toggle_cameras)
        vis.register_key_callback(ord("P"), self.cb_toggle_points)
        vis.register_key_callback(ord("]"), self.cb_increase_point_size)
        vis.register_key_callback(ord("["), self.cb_decrease_point_size)
        vis.register_key_callback(ord("O"), self.cb_color_original)
        vis.register_key_callback(ord("B"), self.cb_color_slate_blue)
        vis.register_key_callback(ord("T"), self.cb_color_track_length)
        vis.register_key_callback(ord("E"), self.cb_color_error)
        vis.register_key_callback(ord("F"), self.cb_toggle_filter_error)
        
        # Tuning parameters
        vis.register_key_callback(ord("G"), self.cb_increase_track_len)
        vis.register_key_callback(ord("H"), self.cb_decrease_track_len)
        vis.register_key_callback(ord("Y"), self.cb_decrease_reproj_thresh)
        vis.register_key_callback(ord("T"), self.cb_increase_reproj_thresh)
        vis.register_key_callback(ord("U"), self.cb_increase_baseline)
        vis.register_key_callback(ord("I"), self.cb_decrease_baseline)
        
        print("\n=== 3D Point Map Debug Viewer Controls ===")
        print("C: Toggle Camera Frustums & Trajectory")
        print("P: Toggle Point Cloud")
        print("]: Increase Point Size | [: Decrease Point Size")
        print("O: Color by True Color (Màu thật) | B: Color by Slate Blue")
        print("T: Color by Track Length | E: Color by Reprojection Error")
        print("F: Toggle Hard Outliers Filter")
        print("\n--- Tuning Filters (Real-time Parameters) ---")
        print(f"G: Increase min_track_length (Gom camera) | current: {self.min_track_len}")
        print(f"H: Decrease min_track_length (Gom camera) | current: {self.min_track_len}")
        print(f"T: Increase reprojection_threshold (Tăng sai số) | current: {self.reproj_thresh_cm:.1f} cm")
        print(f"Y: Decrease reprojection_threshold (Giảm sai số) | current: {self.reproj_thresh_cm:.1f} cm")
        print(f"U: Increase min_baseline_angle (Tăng góc mở) | current: {self.min_baseline} deg")
        print(f"I: Decrease min_baseline_angle (Giảm góc mở) | current: {self.min_baseline} deg")
        print("Q / ESC: Close viewer")
        print("==========================================\n")
        
        vis.run()
        vis.destroy_window()
