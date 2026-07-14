import torch
import cv2
import numpy as np
from typing import Dict, Any, Tuple
from lightglue import LightGlue

class LightGlueMatcher:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.matcher = LightGlue(features='superpoint').eval().to(self.device)

    def match_pair(
        self,
        features1: Dict[str, Any],
        features2: Dict[str, Any],
        ransac_threshold: float = 4.0
    ) -> Dict[str, Any]:
        """Match keypoints between two images using LightGlue and run geometric verification.
        
        Args:
            features1: Dict containing keypoints, descriptors, scores for image 1.
            features2: Dict containing keypoints, descriptors, scores for image 2.
            ransac_threshold: RANSAC threshold for Fundamental Matrix estimation.
            
        Returns:
            Dict containing:
            - matches: np.ndarray of shape (M, 2) (indices of matches)
            - confidence: np.ndarray of shape (M,) (match confidence scores)
            - inliers: np.ndarray of shape (M,) (boolean mask of RANSAC inliers)
        """
        kpts1 = features1["keypoints"]
        kpts2 = features2["keypoints"]
        
        if len(kpts1) == 0 or len(kpts2) == 0:
            return {
                "matches": np.empty((0, 2), dtype=np.int32),
                "confidence": np.empty((0,), dtype=np.float32),
                "inliers": np.empty((0,), dtype=bool)
            }
            
        # Convert features to PyTorch tensors
        feats1 = {
            "keypoints": torch.from_numpy(kpts1).unsqueeze(0).to(self.device).float(),
            "descriptors": torch.from_numpy(features1["descriptors"]).unsqueeze(0).to(self.device).float(),
        }
        feats2 = {
            "keypoints": torch.from_numpy(kpts2).unsqueeze(0).to(self.device).float(),
            "descriptors": torch.from_numpy(features2["descriptors"]).unsqueeze(0).to(self.device).float(),
        }
        
        # Add image sizes if they exist
        if "image_size" in features1:
            feats1["image_size"] = torch.from_numpy(features1["image_size"]).unsqueeze(0).to(self.device).float()
        if "image_size" in features2:
            feats2["image_size"] = torch.from_numpy(features2["image_size"]).unsqueeze(0).to(self.device).float()
            
        with torch.no_grad():
            match_result = self.matcher({"image0": feats1, "image1": feats2})
            
        matches = match_result["matches"][0].cpu().numpy()
        scores = match_result["scores"][0].cpu().numpy()
        
        # Run Geometric Verification using RANSAC with Fundamental Matrix
        inliers = np.zeros(len(matches), dtype=bool)
        if len(matches) >= 8:
            pts1 = kpts1[matches[:, 0]]
            pts2 = kpts2[matches[:, 1]]
            
            pts1 = np.ascontiguousarray(pts1, dtype=np.float32)
            pts2 = np.ascontiguousarray(pts2, dtype=np.float32)
            
            if pts1.ndim == 2 and pts1.shape[0] >= 8:
                try:
                    _, mask = cv2.findFundamentalMat(
                        pts1, pts2,
                        method=cv2.FM_RANSAC,
                        ransacReprojThreshold=ransac_threshold,
                        confidence=0.99,
                        maxIters=2000
                    )
                    
                    if mask is not None:
                        inliers = mask.ravel().astype(bool)
                except Exception as e:
                    print(f"[-] Warning: cv2.findFundamentalMat failed: {e}")
                
        # Cleanup tensors
        del feats1
        del feats2
        del match_result
        import gc
        gc.collect()
        
        return {
            "matches": matches,
            "confidence": scores,
            "inliers": inliers
        }
