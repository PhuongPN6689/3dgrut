import numpy as np
from typing import Dict, List, Any
from ..geometry.camera import Camera
from ..geometry.pose import get_camera_center, qvec2rotmat
from ..geometry.projection import compute_reprojection_error

def compute_max_baseline_angle(
    pt_3d: np.ndarray,
    image_names: List[str],
    images_poses: Dict[str, Dict[str, Any]]
) -> float:
    """Compute the maximum angle (in degrees) between viewing rays from cameras to the point."""
    centers = []
    for img_name in image_names:
        pose = images_poses.get(img_name)
        if pose is not None:
            centers.append(get_camera_center(pose["qvec"], pose["tvec"]))
            
    if len(centers) < 2:
        return 0.0
        
    vectors = []
    for c in centers:
        v = pt_3d - c
        norm = np.linalg.norm(v)
        if norm > 1e-8:
            vectors.append(v / norm)
            
    if len(vectors) < 2:
        return 0.0
        
    max_angle = 0.0
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            dot = np.dot(vectors[i], vectors[j])
            dot = np.clip(dot, -1.0, 1.0)
            angle = float(np.degrees(np.arccos(dot)))
            if angle > max_angle:
                max_angle = angle
                
    return max_angle

def filter_by_track_length(track_len: int, min_len: int) -> bool:
    """Check if the point has sufficient camera observations."""
    return track_len >= min_len

def filter_by_baseline_angle(max_angle: float, min_angle: float, track_len: int) -> bool:
    """Check if the triangulation geometry is stable (not parallel)."""
    # 2-view points require larger baseline angle to be stable
    effective_min_baseline = min_angle
    if track_len == 2:
        effective_min_baseline = max(min_angle * 2.0, 3.0)
    return max_angle >= effective_min_baseline

def filter_by_depth_and_front(
    xyz: np.ndarray,
    image_names: List[str],
    images_poses: Dict[str, Dict[str, Any]],
    max_dist: float
) -> bool:
    """Ensure point is in front of observing cameras and within scene bounding box."""
    for img_name in image_names:
        pose = images_poses.get(img_name)
        if pose is None:
            continue
            
        cam_center = get_camera_center(pose["qvec"], pose["tvec"])
        dist = np.linalg.norm(xyz - cam_center)
        if dist > max_dist:
            return False
            
        R = qvec2rotmat(pose["qvec"])
        t = pose["tvec"]
        z_cam = (R @ xyz + t)[2]
        if z_cam <= 0.1: # At least 10cm in front of camera
            return False
            
    return True

def filter_by_reprojection_error(avg_error: float, max_error: float, track_len: int) -> bool:
    """Check if average reprojection error is below threshold."""
    # 2-view points require tighter error bounds
    effective_thresh = max_error
    if track_len == 2:
        effective_thresh = max_error * 0.8
    return avg_error <= effective_thresh

def filter_by_consistency(
    xyz: np.ndarray,
    track: Dict[str, Any],
    images_poses: Dict[str, Dict[str, Any]],
    cameras: Dict[int, Camera],
    max_error: float,
    track_len: int
) -> bool:
    """Ensure no individual camera observation has an abnormally large error."""
    effective_thresh = max_error
    if track_len == 2:
        effective_thresh = max_error * 0.8
        
    image_names = track["image_names"]
    pts_2d = track["pts_2d"]
    
    for i, img_name in enumerate(image_names):
        pose = images_poses.get(img_name)
        cam = cameras.get(pose["camera_id"]) if pose else None
        if pose is not None and cam is not None:
            K = cam.get_intrinsic_matrix()
            R = qvec2rotmat(pose["qvec"])
            t = pose["tvec"]
            pt_undist = cam.undistort_points(np.array([pts_2d[i]]))[0]
            
            err = compute_reprojection_error(xyz, pt_undist, K, R, t)
            if err > 1.8 * effective_thresh:
                return False
                
    return True

def get_max_valid_distance(images_poses: Dict[str, Dict[str, Any]]) -> float:
    """Calculate maximum valid scene distance based on camera trajectory size."""
    centers = []
    for img_name, pose in images_poses.items():
        centers.append(get_camera_center(pose["qvec"], pose["tvec"]))
        
    if len(centers) > 1:
        centers = np.array(centers)
        cam_min = np.min(centers, axis=0)
        cam_max = np.max(centers, axis=0)
        scene_diagonal = np.linalg.norm(cam_max - cam_min)
        return max(scene_diagonal * 4.0, 500.0)
    return 1000.0

def compute_point_metric_error(
    xyz: np.ndarray,
    track: Dict[str, Any],
    images_poses: Dict[str, Dict[str, Any]],
    cameras: Dict[int, Camera]
) -> float:
    """Compute the average 3D projection error of the point in meters."""
    image_names = track["image_names"]
    pts_2d = track["pts_2d"]
    metric_errors = []
    
    for i, img_name in enumerate(image_names):
        pose = images_poses.get(img_name)
        cam = cameras.get(pose["camera_id"]) if pose else None
        if pose is not None and cam is not None:
            # 1. Distance in meters
            cam_center = get_camera_center(pose["qvec"], pose["tvec"])
            dist = np.linalg.norm(xyz - cam_center)
            
            # 2. 2D projection error in pixels
            K = cam.get_intrinsic_matrix()
            R = qvec2rotmat(pose["qvec"])
            t = pose["tvec"]
            pt_undist = cam.undistort_points(np.array([pts_2d[i]]))[0]
            err_px = compute_reprojection_error(xyz, pt_undist, K, R, t)
            
            # 3. Convert to meters (err_m = dist * err_px / fx)
            f_px = K[0, 0]
            if f_px > 0:
                metric_errors.append(dist * (err_px / f_px))
                
    if len(metric_errors) > 0:
        return float(np.mean(metric_errors))
    return 0.0

def filter_points3d(
    points3d: List[Dict[str, Any]],
    images_poses: Dict[str, Dict[str, Any]],
    cameras: Dict[int, Camera],
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Filter 3D points using modular geometric filter functions."""
    reproj_thresh = float(config["triangulation"]["reprojection_threshold"])
    min_track_len = int(config["filter"]["min_track_length"])
    min_baseline = float(config["filter"]["min_baseline_angle"])
    
    max_dist = get_max_valid_distance(images_poses)
    filtered_points = []
    
    for pt in points3d:
        xyz = pt["xyz"]
        track = pt["track"]
        track_len = len(track["image_names"])
        
        # 1. Track Length
        if not filter_by_track_length(track_len, min_track_len):
            continue
            
        # 2. Baseline Angle
        max_angle = compute_max_baseline_angle(xyz, track["image_names"], images_poses)
        if not filter_by_baseline_angle(max_angle, min_baseline, track_len):
            continue
            
        # 3. Depth & Front Check
        if not filter_by_depth_and_front(xyz, track["image_names"], images_poses, max_dist):
            continue
            
        # 4. Reprojection Error
        if not filter_by_reprojection_error(pt["avg_error"], reproj_thresh, track_len):
            continue
            
        # 5. Individual Consistency
        if not filter_by_consistency(xyz, track, images_poses, cameras, reproj_thresh, track_len):
            continue
            
        # Compute and append metric error in centimeters
        err_m = compute_point_metric_error(xyz, track, images_poses, cameras)
        pt["avg_error_cm"] = err_m * 100.0
        
        filtered_points.append(pt)
        
    return filtered_points
