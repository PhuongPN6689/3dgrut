import numpy as np
from scipy.optimize import least_squares
from typing import Dict, List, Any, Tuple
from ..geometry.camera import Camera
from ..geometry.pose import qvec2rotmat

def reprojection_residuals(
    point_3d: np.ndarray,
    pts_2d: np.ndarray,
    Ks: List[np.ndarray],
    Rs: List[np.ndarray],
    ts: List[np.ndarray]
) -> np.ndarray:
    """Compute projection residuals (reprojection errors in x and y) for a 3D point."""
    residuals = []
    for i in range(len(pts_2d)):
        K = Ks[i]
        R = Rs[i]
        t = ts[i]
        pt_2d = pts_2d[i]
        
        # Project 3D point to camera
        X_c = R @ point_3d + t
        if X_c[2] <= 1e-5:
            # Point behind camera, return large residuals
            residuals.extend([1000.0, 1000.0])
            continue
            
        x_proj = K @ X_c
        pt_proj = x_proj[:2] / x_proj[2]
        
        # Residuals in x and y
        residuals.extend(pt_proj - pt_2d)
        
    return np.array(residuals)

def optimize_single_point(
    pt_3d_init: np.ndarray,
    pts_2d: np.ndarray,
    Ks: List[np.ndarray],
    Rs: List[np.ndarray],
    ts: List[np.ndarray],
    loss: str = "huber"
) -> Tuple[np.ndarray, float]:
    """Refine a single 3D point using scipy least_squares."""
    res = least_squares(
        reprojection_residuals,
        pt_3d_init,
        args=(pts_2d, Ks, Rs, ts),
        loss=loss,
        f_scale=2.0, # pixel scale for Huber threshold
        method="lm" if loss == "linear" else "trf"
    )
    
    # Calculate average error
    opt_pt_3d = res.x
    residuals = reprojection_residuals(opt_pt_3d, pts_2d, Ks, Rs, ts)
    errors = np.linalg.norm(residuals.reshape(-1, 2), axis=1)
    avg_error = float(np.mean(errors))
    
    return opt_pt_3d, avg_error

def run_bundle_adjustment(
    points3d: List[Dict[str, Any]],
    images_poses: Dict[str, Dict[str, Any]],
    cameras: Dict[int, Camera]
) -> List[Dict[str, Any]]:
    """Refine all 3D points globally using Bundle Adjustment."""
    optimized_points = []
    
    for pt in points3d:
        track = pt["track"]
        image_names = track["image_names"]
        pts_2d = track["pts_2d"]
        
        Ks, Rs, ts = [], [], []
        undist_pts_2d = []
        
        valid = True
        for i, img_name in enumerate(image_names):
            pose = images_poses.get(img_name)
            if pose is None:
                valid = False
                break
            cam = cameras.get(pose["camera_id"])
            if cam is None:
                valid = False
                break
                
            K = cam.get_intrinsic_matrix()
            R = qvec2rotmat(pose["qvec"])
            t = pose["tvec"]
            
            # Undistort point
            pt_undist = cam.undistort_points(np.array([pts_2d[i]]))[0]
            
            Ks.append(K)
            Rs.append(R)
            ts.append(t)
            undist_pts_2d.append(pt_undist)
            
        if not valid:
            continue
            
        undist_pts_2d = np.array(undist_pts_2d)
        
        opt_xyz, avg_error = optimize_single_point(pt["xyz"], undist_pts_2d, Ks, Rs, ts)
        
        optimized_points.append({
            "point3d_id": pt["point3d_id"],
            "xyz": opt_xyz,
            "avg_error": avg_error,
            "track": track
        })
        
    return optimized_points
