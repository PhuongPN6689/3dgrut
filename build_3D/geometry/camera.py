import numpy as np

class Camera:
    def __init__(self, camera_id: int, model: str, width: int, height: int, params: np.ndarray):
        self.id = camera_id
        self.model = model
        self.width = width
        self.height = height
        self.params = np.array(params, dtype=np.float64)

    def get_intrinsic_matrix(self) -> np.ndarray:
        """Get 3x3 camera intrinsic matrix K."""
        K = np.eye(3, dtype=np.float64)
        if self.model == "PINHOLE":
            fx, fy, cx, cy = self.params
            K[0, 0] = fx
            K[1, 1] = fy
            K[0, 2] = cx
            K[1, 2] = cy
        elif self.model == "SIMPLE_PINHOLE":
            f, cx, cy = self.params
            K[0, 0] = f
            K[1, 1] = f
            K[0, 2] = cx
            K[1, 2] = cy
        elif self.model in ["SIMPLE_RADIAL", "RADIAL"]:
            f, cx, cy = self.params[0:3]
            K[0, 0] = f
            K[1, 1] = f
            K[0, 2] = cx
            K[1, 2] = cy
        else:
            # Fallback
            if len(self.params) >= 4:
                fx, fy, cx, cy = self.params[:4]
                K[0, 0] = fx
                K[1, 1] = fy
                K[0, 2] = cx
                K[1, 2] = cy
            elif len(self.params) >= 3:
                f, cx, cy = self.params[:3]
                K[0, 0] = f
                K[1, 1] = f
                K[0, 2] = cx
                K[1, 2] = cy
        return K

    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        """Undistort 2D points using camera parameters."""
        if self.model in ["PINHOLE", "SIMPLE_PINHOLE"] or len(self.params) <= 3:
            return points
        
        import cv2
        K = self.get_intrinsic_matrix()
        if self.model == "SIMPLE_RADIAL":
            k1 = self.params[3]
            dist_coeffs = np.array([k1, 0.0, 0.0, 0.0], dtype=np.float64)
        elif self.model == "RADIAL":
            k1, k2 = self.params[3:5]
            dist_coeffs = np.array([k1, k2, 0.0, 0.0], dtype=np.float64)
        else:
            dist_coeffs = np.zeros(4, dtype=np.float64)
            
        pts_reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        undistorted = cv2.undistortPoints(pts_reshaped, K, dist_coeffs, P=K)
        return undistorted.reshape(-1, 2).astype(np.float64)
