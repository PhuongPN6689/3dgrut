import numpy as np

def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y**2 - 2 * z**2,
         2 * x * y - 2 * w * z,
         2 * z * x + 2 * w * y],
        [2 * x * y + 2 * w * z,
         1 - 2 * x**2 - 2 * z**2,
         2 * y * z - 2 * w * x],
        [2 * z * x - 2 * w * y,
         2 * y * z + 2 * w * x,
         1 - 2 * x**2 - 2 * y**2]
    ])

def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]
    ]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def get_camera_center(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Compute camera center in world coordinates: C = -R^T * t."""
    R = qvec2rotmat(qvec)
    return -R.T @ tvec

def compute_position_distance(c1: np.ndarray, c2: np.ndarray) -> float:
    """Euclidean distance between two camera centers."""
    return float(np.linalg.norm(c1 - c2))

def compute_rotation_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """Rotation distance (angle in degrees) between two quaternions."""
    # Ensure normalized
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    # Cosine of angle is the dot product
    dot = np.abs(np.sum(q1 * q2))
    dot = np.clip(dot, 0.0, 1.0)
    angle_rad = 2.0 * np.arccos(dot)
    return float(np.degrees(angle_rad))

def compute_pose_distance(
    q1: np.ndarray, t1: np.ndarray,
    q2: np.ndarray, t2: np.ndarray,
    pos_weight: float = 1.0,
    rot_weight: float = 0.0
) -> float:
    """Compute distance between two camera poses."""
    c1 = get_camera_center(q1, t1)
    c2 = get_camera_center(q2, t2)
    pos_dist = compute_position_distance(c1, c2)
    rot_dist = compute_rotation_distance(q1, q2)
    return pos_weight * pos_dist + rot_weight * rot_dist
