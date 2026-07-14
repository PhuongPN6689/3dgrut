import numpy as np
from typing import Dict, List, Tuple, Any
from ..geometry.camera import Camera
from ..geometry.projection import get_projection_matrix, triangulate_point_dlt, compute_reprojection_error
from ..geometry.pose import qvec2rotmat

def triangulate_track(
    track: Dict[str, Any],
    images_poses: Dict[str, Dict[str, Any]],
    cameras: Dict[int, Camera]
) -> Tuple[np.ndarray | None, float]:
    """Triangulate a single global track.
    
    Args:
        track: Dict containing 'image_names', 'pts_2d', 'kp_indices'.
        images_poses: Dict of image poses (qvec, tvec, camera_id).
        cameras: Dict of camera objects.
        
    Returns:
        Tuple of (pt_3d, avg_reprojection_error).
        If triangulation fails, returns (None, float('inf')).
    """
    image_names = track["image_names"]
    pts_2d = track["pts_2d"]
    
    num_views = len(image_names)
    if num_views < 2:
        return None, float('inf')
        
    undist_pts_2d = []
    proj_mats = []
    
    K_mats = []
    R_mats = []
    t_mats = []
    
    for i in range(num_views):
        img_name = image_names[i]
        pt = pts_2d[i]
        
        pose = images_poses.get(img_name)
        if pose is None:
            return None, float('inf')
            
        cam = cameras.get(pose["camera_id"])
        if cam is None:
            return None, float('inf')
            
        K = cam.get_intrinsic_matrix()
        R = qvec2rotmat(pose["qvec"])
        t = pose["tvec"]
        
        # Undistort the 2D point (keeping it in image space via P=K)
        pt_undist = cam.undistort_points(np.array([pt]))[0]
        
        P = get_projection_matrix(K, R, t)
        
        undist_pts_2d.append(pt_undist)
        proj_mats.append(P)
        
        K_mats.append(K)
        R_mats.append(R)
        t_mats.append(t)
        
    undist_pts_2d = np.array(undist_pts_2d)
    proj_mats = np.array(proj_mats)
    
    # Triangulate using DLT
    pt_3d = triangulate_point_dlt(undist_pts_2d, proj_mats)
    
    # Check if point is behind any camera (positive depth check)
    for R, t in zip(R_mats, t_mats):
        z_cam = (R @ pt_3d + t)[2]
        if z_cam <= 0:
            return None, float('inf')
            
    # Calculate reprojection errors
    errors = []
    for i in range(num_views):
        err = compute_reprojection_error(pt_3d, undist_pts_2d[i], K_mats[i], R_mats[i], t_mats[i])
        errors.append(err)
        
    avg_error = float(np.mean(errors))
    return pt_3d, avg_error

def triangulate_all_tracks(
    tracks: List[Dict[str, Any]],
    images_poses: Dict[str, Dict[str, Any]],
    cameras: Dict[int, Camera]
) -> List[Dict[str, Any]]:
    """Triangulate all global tracks in the dataset."""
    points3d = []
    
    for track_idx, track in enumerate(tracks):
        pt_3d, avg_error = triangulate_track(track, images_poses, cameras)
        if pt_3d is not None:
            points3d.append({
                "point3d_id": track_idx,
                "xyz": pt_3d,
                "avg_error": avg_error,
                "track": track
            })
            
    return points3d
