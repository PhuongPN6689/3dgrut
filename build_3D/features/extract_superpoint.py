import torch
import numpy as np
from PIL import Image
from lightglue import SuperPoint

class SuperPointExtractor:
    def __init__(self, max_keypoints: int = 4096, detection_threshold: float = 0.005, device: str = "cpu"):
        self.device = torch.device(device)
        self.extractor = SuperPoint(
            max_num_keypoints=max_keypoints,
            detection_threshold=detection_threshold
        ).eval().to(self.device)

    def extract(self, image_path: str) -> dict:
        """Extract SuperPoint keypoints, descriptors, and scores for a given image.
        
        Args:
            image_path: Path to the image.
            
        Returns:
            Dict containing:
            - keypoints: np.ndarray of shape (N, 2)
            - descriptors: np.ndarray of shape (N, D)
            - scores: np.ndarray of shape (N,)
            - local_keypoint_ids: List of N strings ["kp0", "kp1", ...]
        """
        # Load image
        img_orig = Image.open(image_path)
        W_orig, H_orig = img_orig.size
        
        # Scale down to 640 to prevent out-of-memory on CPU
        max_dim = 640
        if max(W_orig, H_orig) > max_dim:
            scale = max_dim / max(W_orig, H_orig)
            W_new = int(round(W_orig * scale))
            H_new = int(round(H_orig * scale))
            img_resized = img_orig.resize((W_new, H_new), Image.Resampling.BILINEAR)
        else:
            scale = 1.0
            img_resized = img_orig
            
        img_gray = img_resized.convert('L')
        # Convert to tensor normalized to [0, 1]
        img_tensor = torch.from_numpy(np.array(img_gray)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(self.device) # (1, 1, H, W)
        
        import gc
        gc.collect()
        
        with torch.no_grad():
            feats = self.extractor({'image': img_tensor})
            
        # Get keypoints, descriptors, and scores
        kpts = feats['keypoints'][0].cpu().numpy()
        descriptors = feats['descriptors'][0].cpu().numpy()
        
        # Scale keypoints back to original resolution
        if scale != 1.0:
            kpts = kpts / scale
            
        scores_key = 'scores' if 'scores' in feats else ('keypoint_scores' if 'keypoint_scores' in feats else None)
        if scores_key is not None:
            scores = feats[scores_key][0].cpu().numpy()
        else:
            scores = np.ones(len(kpts), dtype=np.float32)
            
        local_keypoint_ids = [f"kp{i}" for i in range(len(kpts))]
        
        # Free PyTorch tensors and run GC
        del img_tensor
        del feats
        import gc
        gc.collect()
        
        # Extract RGB colors at keypoint locations from original image using memory-efficient getpixel
        img_rgb = img_orig.convert('RGB')
        colors_list = []
        for kp in kpts:
            px = int(round(kp[0]))
            py = int(round(kp[1]))
            px = max(0, min(px, W_orig - 1))
            py = max(0, min(py, H_orig - 1))
            r, g, b = img_rgb.getpixel((px, py))
            colors_list.append([r / 255.0, g / 255.0, b / 255.0])
            
        kpt_colors = np.array(colors_list, dtype=np.float32) if len(kpts) > 0 else np.empty((0, 3), dtype=np.float32)
        
        # Free images
        del img_orig
        del img_rgb
        if max_dim < max(W_orig, H_orig):
            del img_resized
        del img_gray
        gc.collect()
            
        return {
            "keypoints": kpts,
            "descriptors": descriptors,
            "scores": scores,
            "local_keypoint_ids": local_keypoint_ids,
            "colors": kpt_colors
        }
