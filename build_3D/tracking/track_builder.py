import numpy as np
from typing import Dict, List, Tuple, Any, Set
from ..matching.union_find import UnionFind

def build_global_tracks(
    graph_edges: List[Tuple[str, str]],
    matches_cache: Dict[Tuple[str, str], Dict[str, Any]],
    features_cache: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Build global tracks of keypoint observations across all images.
    
    Args:
        graph_edges: List of image pairs that were matched.
        matches_cache: Dict mapping (img1, img2) to match results (with 'matches' and 'inliers').
        features_cache: Dict mapping image name to feature dictionary (with 'keypoints', 'descriptors', 'scores').
        
    Returns:
        List of tracks, where each track is a dict:
        - image_names: List[str]
        - kp_indices: List[int]
        - pts_2d: np.ndarray of shape (N, 2)
        - descriptor: np.ndarray of shape (D,) (representative descriptor)
        - length: int
    """
    # 1. Identify all local keypoints: (image_name, kp_idx)
    # We will use string keys: "image_name/kp_idx"
    all_kps = []
    for img_name, feats in features_cache.items():
        num_kps = len(feats["keypoints"])
        for i in range(num_kps):
            all_kps.append(f"{img_name}/{i}")
            
    uf = UnionFind(all_kps)
    
    # 2. Union matching inliers
    for name1, name2 in graph_edges:
        # Find the correct cache key
        sorted_names = tuple(sorted([name1, name2]))
        match_data = matches_cache.get(sorted_names)
        if match_data is None:
            continue
            
        matches = match_data["matches"]
        inliers = match_data["inliers"]
        
        # Only union geometric inliers
        inlier_matches = matches[inliers]
        for idx1, idx2 in inlier_matches:
            kp_key1 = f"{name1}/{idx1}"
            kp_key2 = f"{name2}/{idx2}"
            uf.union(kp_key1, kp_key2)
            
    # 3. Extract components
    components = uf.get_components()
    
    global_tracks = []
    
    # 4. Refine components into tracks
    for comp in components:
        # Filter out trivial tracks (single observation)
        if len(comp) < 2:
            continue
            
        # Group observations by image
        img_obs = {}
        for kp_str in comp:
            img_name, kp_idx_str = kp_str.split("/")
            kp_idx = int(kp_idx_str)
            
            # If an image appears multiple times in the same track (conflict),
            # keep the one with the highest SuperPoint confidence score.
            score = features_cache[img_name]["scores"][kp_idx]
            
            if img_name not in img_obs or score > img_obs[img_name]["score"]:
                img_obs[img_name] = {
                    "kp_idx": kp_idx,
                    "score": score
                }
                
        # Skip if resolved track length is less than 2
        if len(img_obs) < 2:
            continue
            
        # Build track info
        image_names = list(img_obs.keys())
        kp_indices = [img_obs[img]["kp_idx"] for img in image_names]
        
        pts_2d_list = []
        descriptors_list = []
        colors_list = []
        
        for img, idx in zip(image_names, kp_indices):
            pts_2d_list.append(features_cache[img]["keypoints"][idx])
            descriptors_list.append(features_cache[img]["descriptors"][idx])
            if "colors" in features_cache[img]:
                colors_list.append(features_cache[img]["colors"][idx])
            else:
                colors_list.append(np.array([0.5, 0.5, 0.5], dtype=np.float32))
            
        pts_2d = np.array(pts_2d_list, dtype=np.float64)
        
        # Calculate representative descriptor (average and normalize)
        descriptors = np.array(descriptors_list, dtype=np.float64)
        rep_desc = np.mean(descriptors, axis=0)
        norm = np.linalg.norm(rep_desc)
        if norm > 1e-8:
            rep_desc = rep_desc / norm
            
        # Calculate representative color (average)
        rep_color = np.mean(colors_list, axis=0)
            
        global_tracks.append({
            "image_names": image_names,
            "kp_indices": kp_indices,
            "pts_2d": pts_2d,
            "descriptor": rep_desc,
            "length": len(image_names),
            "color": rep_color
        })
        
    return global_tracks
