import numpy as np

def get_projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compute projection matrix P = K * [R | t]."""
    Rt = np.column_stack((R, t))
    return K @ Rt

def triangulate_point_dlt(pts_2d: np.ndarray, proj_mats: np.ndarray) -> np.ndarray:
    """Triangulate a 3D point from multiple 2D observations using DLT.
    
    Args:
        pts_2d: (N, 2) array of 2D coordinates.
        proj_mats: (N, 3, 4) array of projection matrices.
        
    Returns:
        (3,) array representing the 3D point coordinates.
    """
    num_views = len(pts_2d)
    A = np.zeros((2 * num_views, 4), dtype=np.float64)
    
    for i in range(num_views):
        u, v = pts_2d[i]
        P = proj_mats[i]
        A[2 * i] = u * P[2] - P[0]
        A[2 * i + 1] = v * P[2] - P[1]
        
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if np.abs(X[3]) > 1e-8:
        X = X[:3] / X[3]
    else:
        X = X[:3]
    return X

def compute_reprojection_error(
    pt_3d: np.ndarray,
    pt_2d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray
) -> float:
    """Compute reprojection error for a single 3D point in a camera view."""
    # Transform to camera coordinates
    X_c = R @ pt_3d + t
    if X_c[2] <= 1e-5:
        # Behind camera
        return float('inf')
        
    # Project to image coordinates
    x_proj = K @ X_c
    pt_2d_proj = x_proj[:2] / x_proj[2]
    
    # Euclidean distance
    error = np.linalg.norm(pt_2d - pt_2d_proj)
    return float(error)
