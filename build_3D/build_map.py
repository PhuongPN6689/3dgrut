import os
import sys
import time
import argparse
from omegaconf import OmegaConf
import numpy as np
from tqdm import tqdm

# Add parent directory to path to enable imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Add local packages folder on D drive if it exists
custom_packages = os.path.join(parent_dir, ".pip_packages")
if os.path.exists(custom_packages):
    sys.path.insert(0, custom_packages)

from build_3D.io.pose_reader import read_poses_from_sparse
from build_3D.features.extract_superpoint import SuperPointExtractor
from build_3D.features.feature_cache import get_cached_features, cache_features
from build_3D.io.feature_io import (
    save_matches, load_matches,
    save_graph, load_graph,
    save_tracks, load_tracks,
    save_points3d, load_points3d,
    save_points3d_all, load_points3d_all
)
from build_3D.matching.graph_builder import (
    compute_distance_matrix, build_neighbor_graph,
    merge_disconnected_components, get_cached_distances, cache_distances
)
from build_3D.matching.lightglue_matcher import LightGlueMatcher
from build_3D.tracking.track_builder import build_global_tracks
from build_3D.mapping.triangulation import triangulate_all_tracks
from build_3D.mapping.bundle_adjustment import run_bundle_adjustment
from build_3D.mapping.filtering import filter_points3d
from build_3D.viewer.viewer3d import Viewer3D

def torch_cuda_check(device_preference: str) -> bool:
    """Helper to check if CUDA is available if requested."""
    import torch
    if device_preference == "cuda":
        cuda_avail = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_avail}")
        return cuda_avail
    return False

def step_read_camera_poses(sparse_dir: str, image_dir: str):
    """Step 0: Read camera poses and filter to existing images."""
    print("\n--- Step 0: Reading Camera Poses ---")
    cameras, all_poses = read_poses_from_sparse(sparse_dir)
    if not os.path.exists(image_dir):
        print(f"Image directory not found: {image_dir}")
        sys.exit(1)
        
    existing_images = set(os.listdir(image_dir))
    poses = {name: pose for name, pose in all_poses.items() if name in existing_images}
    image_names = sorted(list(poses.keys()))
    print(f"Found {len(cameras)} cameras and {len(image_names)} training images with poses.")
    return cameras, poses, image_names

def step_extract_features(image_names: list, image_dir: str, cache_dir: str, device: str, config: any, force_rebuild: bool = False):
    """Step 1 & 2: Extract features and cache them with colors."""
    print("\n--- Step 1 & 2: Extracting & Caching SuperPoint Features ---")
    start_time = time.time()
    extractor = None
    features_cache = {}
    total_keypoints = 0
    
    for img_name in tqdm(image_names, desc="Feature Extraction"):
        img_path = os.path.join(image_dir, img_name)
        cached_feats = None if force_rebuild else get_cached_features(cache_dir, img_name)
        
        if cached_feats is not None:
            # Migration check: if colors is not in cached features, extract on the fly to self-heal
            if "colors" not in cached_feats:
                from PIL import Image
                try:
                    img_orig = Image.open(img_path)
                    W_orig, H_orig = img_orig.size
                    img_rgb = img_orig.convert('RGB')
                    kpts = cached_feats["keypoints"]
                    colors_list = []
                    for kp in kpts:
                        px = int(round(kp[0]))
                        py = int(round(kp[1]))
                        px = max(0, min(px, W_orig - 1))
                        py = max(0, min(py, H_orig - 1))
                        r, g, b = img_rgb.getpixel((px, py))
                        colors_list.append([r / 255.0, g / 255.0, b / 255.0])
                    cached_feats["colors"] = np.array(colors_list, dtype=np.float32) if len(kpts) > 0 else np.empty((0, 3), dtype=np.float32)
                    cache_features(cache_dir, img_name, cached_feats)
                except Exception as e:
                    print(f"Error repairing colors cache for {img_name}: {e}")
            features_cache[img_name] = cached_feats
        else:
            if extractor is None:
                print(f"Initializing SuperPoint on {device}...")
                extractor = SuperPointExtractor(
                    max_keypoints=config.superpoint.max_keypoints,
                    detection_threshold=config.superpoint.detection_threshold,
                    device=device
                )
            feats = extractor.extract(img_path)
            from PIL import Image
            with Image.open(img_path) as img:
                feats["image_size"] = np.array(img.size, dtype=np.float32)
            cache_features(cache_dir, img_name, feats)
            features_cache[img_name] = feats
            
        total_keypoints += len(features_cache[img_name]["keypoints"])
        
    print(f"Feature extraction completed in {time.time() - start_time:.2f}s.")
    print(f"Total keypoints: {total_keypoints} (average {total_keypoints/len(image_names):.1f} per image).")
    return features_cache

def step_calculate_graph_and_matches(poses: dict, image_names: list, cache_dir: str, device: str, config: any, features_cache: dict, force_rebuild: bool):
    """Steps 3 to 7: Compute pose distances, neighbor graph, match features, and merge components."""
    # 3. Calculate camera pose distance matrix
    print("\n--- Step 3: Calculating Pose Distances ---")
    cached_distances = get_cached_distances(cache_dir)
    
    distances_valid = False
    if cached_distances is not None:
        dist_images = set()
        for k in cached_distances.keys():
            dist_images.add(k[0])
            dist_images.add(k[1])
        if set(image_names).issubset(dist_images) and len(dist_images) == len(image_names):
            distances_valid = True
            
    distances = cached_distances if distances_valid else compute_distance_matrix(
        poses, pos_weight=config.neighbor.position_weight, rot_weight=config.neighbor.rotation_weight
    )
    if not distances_valid:
        if cached_distances is not None:
            print("Cached distances are invalid or for a different dataset size. Re-calculating...")
        cache_distances(cache_dir, distances)
        
    # 4. Build initial neighbor graph
    print("\n--- Step 4: Building Neighbor Graph ---")
    cached_graph = load_graph(cache_dir)
    
    graph_valid = False
    if not force_rebuild and cached_graph is not None:
        graph_nodes = set()
        for u, v in cached_graph:
            graph_nodes.add(u)
            graph_nodes.add(v)
        if graph_nodes.issubset(set(image_names)) and len(graph_nodes) == len(image_names):
            graph_valid = True
            
    graph_edges = cached_graph if graph_valid else build_neighbor_graph(poses, distances, num_neighbors=config.neighbor.num_neighbors)
    if not graph_valid:
        if cached_graph is not None and not force_rebuild:
            print("Cached graph is invalid or for a different dataset size. Re-building...")
        elif force_rebuild:
            print("[*] Force rebuild specified. Re-building neighbor graph...")
        save_graph(cache_dir, graph_edges)
        
    # 5. Match with LightGlue & 6. Geometric Verification
    print("\n--- Step 5 & 6: Matching Features & Geometric Verification ---")
    matcher = None
    matches_cache = {}
    for name1, name2 in tqdm(graph_edges, desc="Pairwise Matching"):
        sorted_names = tuple(sorted([name1, name2]))
        cached_match = load_matches(cache_dir, sorted_names[0], sorted_names[1])
        if not force_rebuild and cached_match is not None:
            matches_cache[sorted_names] = cached_match
        else:
            if matcher is None:
                matcher = LightGlueMatcher(device=device)
            feats1 = features_cache[sorted_names[0]]
            feats2 = features_cache[sorted_names[1]]
            match_res = matcher.match_pair(feats1, feats2, ransac_threshold=config.matching.ransac_threshold)
            save_matches(cache_dir, sorted_names[0], sorted_names[1], match_res)
            matches_cache[sorted_names] = match_res
            
    # 7. Merge disconnected components (Extra Links)
    print("\n--- Step 7: Merging Disconnected Components ---")
    def match_fn(n1: str, n2: str) -> dict:
        nonlocal matcher
        sorted_pair = tuple(sorted([n1, n2]))
        cached = load_matches(cache_dir, sorted_pair[0], sorted_pair[1])
        if not force_rebuild and cached is not None:
            matches_cache[sorted_pair] = cached
            return cached
        if matcher is None:
            matcher = LightGlueMatcher(device=device)
        feats1 = features_cache[sorted_pair[0]]
        feats2 = features_cache[sorted_pair[1]]
        res = matcher.match_pair(feats1, feats2, ransac_threshold=config.matching.ransac_threshold)
        save_matches(cache_dir, sorted_pair[0], sorted_pair[1], res)
        matches_cache[sorted_pair] = res
        return res
        
    graph_edges, added_edges = merge_disconnected_components(
        graph_edges, image_names, distances, match_fn=match_fn,
        extra_links=config.graph.extra_links, min_matches=config.matching.min_matches
    )
    save_graph(cache_dir, graph_edges)
    return graph_edges, matches_cache

def step_build_tracks_and_triangulate(graph_edges: list, matches_cache: dict, features_cache: dict, poses: dict, cameras: dict, cache_dir: str, force_rebuild: bool):
    """Step 8 & 9: Build global tracks and triangulate 3D points."""
    # 8. Build Global Tracks
    print("\n--- Step 8: Building Global Tracks ---")
    cached_tracks = load_tracks(cache_dir)
    has_colors_in_tracks = cached_tracks is not None and len(cached_tracks) > 0 and "color" in cached_tracks[0]
    
    if not force_rebuild and has_colors_in_tracks:
        tracks = cached_tracks
        print(f"Loaded {len(tracks)} global tracks from cache.")
    else:
        if cached_tracks is not None and not has_colors_in_tracks:
            print("Cached tracks lack color information. Re-building tracks to extract colors...")
        tracks = build_global_tracks(graph_edges, matches_cache, features_cache)
        save_tracks(cache_dir, tracks)
        print(f"Constructed {len(tracks)} global tracks.")
        
    # 9. Multi-view Triangulation
    print("\n--- Step 9: Multi-view Triangulation ---")
    cached_points3d = load_points3d(cache_dir)
    has_colors_in_points3d = False
    if cached_points3d is not None and len(cached_points3d) > 0:
        first_pt = cached_points3d[0]
        if "color" in first_pt["track"]:
            all_colors = np.array([pt["track"]["color"] for pt in cached_points3d])
            if np.std(all_colors) > 1e-4 or not np.allclose(all_colors[0], [0.5, 0.5, 0.5]):
                has_colors_in_points3d = True
                
    if not force_rebuild and has_colors_in_points3d:
        points3d = cached_points3d
        print(f"Loaded {len(points3d)} triangulated points from cache.")
    else:
        if cached_points3d is not None and not has_colors_in_points3d:
            print("Cached triangulated points lack color. Re-triangulating...")
        points3d = triangulate_all_tracks(tracks, poses, cameras)
        save_points3d(cache_dir, points3d)
        print(f"Triangulated {len(points3d)} points from global tracks.")
        
    avg_reproj_err = np.mean([pt["avg_error"] for pt in points3d]) if len(points3d) > 0 else 0.0
    print(f"Average reprojection error before BA: {avg_reproj_err:.3f} pixels.")
    return tracks, points3d

def step_bundle_adjustment_and_filter(points3d: list, poses: dict, cameras: dict, cache_dir: str, config: any):
    """Step 10 & 11: Bundle Adjustment and Outlier Filtering."""
    # 10. Bundle Adjustment
    print("\n--- Step 10: Bundle Adjustment ---")
    start_time = time.time()
    points3d_optimized = run_bundle_adjustment(points3d, poses, cameras)
    print(f"Bundle Adjustment completed in {time.time() - start_time:.2f}s.")
    
    avg_reproj_err_opt = np.mean([pt["avg_error"] for pt in points3d_optimized]) if len(points3d_optimized) > 0 else 0.0
    print(f"Average reprojection error after BA: {avg_reproj_err_opt:.3f} pixels.")
    
    # Calculate metric errors in cm for all optimized points
    from build_3D.mapping.filtering import compute_point_metric_error
    for pt in points3d_optimized:
        err_m = compute_point_metric_error(pt["xyz"], pt["track"], poses, cameras)
        pt["avg_error_cm"] = err_m * 100.0
        
    # Save the raw/unfiltered optimized points for debug viewer mode
    save_points3d_all(cache_dir, points3d_optimized)
    
    # 11. Filtering Outliers
    print("\n--- Step 11: Filtering Outliers ---")
    points3d_filtered = filter_points3d(points3d_optimized, poses, cameras, config)
    num_filtered = len(points3d_optimized) - len(points3d_filtered)
    print(f"Filtered out {num_filtered} outlier points. Final reconstruction has {len(points3d_filtered)} points.")
    
    # Save final filtered 3D points
    save_points3d(cache_dir, points3d_filtered)
    return points3d_optimized, points3d_filtered

def main():
    parser = argparse.ArgumentParser(description="3D Point Map Reconstruction Pipeline using SuperPoint + LightGlue")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset path")
    parser.add_argument("--no_viewer", action="store_true", help="Disable the 3D viewer at the end")
    parser.add_argument(
        "--force",
        type=str,
        nargs="?",
        const="all",
        default=None,
        choices=["graph", "track", "all"],
        help="Force re-running reconstruction steps. Options: 'graph' (rebuild neighbor graph & match), 'track' (rebuild tracks & triangulate), 'all' (force rebuild everything)."
    )
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config if args.config else os.path.join(script_dir, "config.yaml")
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)
        
    config = OmegaConf.load(config_path)
    if args.dataset:
        config.dataset.path = args.dataset
        
    print("Loaded configuration successfully.")
    dataset_path = config.dataset.path
    dataset_path_name = os.path.basename(os.path.normpath(dataset_path))
    print(f"Dataset path: {dataset_path_name} ({dataset_path})")
    device = config.dataset.device
    if not torch_cuda_check(device):
        device = "cpu"
        
    image_dir = os.path.join(dataset_path, "train", "images")
    sparse_dir = os.path.join(dataset_path, "train", "sparse", "0")
    cache_dir = os.path.join(script_dir, "cache", dataset_path_name)
    final_points3d_path = os.path.join(cache_dir, "points3d", "points3d.pkl")
    
    cameras, poses, image_names = step_read_camera_poses(sparse_dir, image_dir)
    
    # Resolve force level
    force_feature = False
    force_graph = False
    force_track = False
    if args.force == "all":
        force_feature = True
        force_graph = True
        force_track = True
    elif args.force == "graph":
        force_feature = True
        force_graph = True
        force_track = True
    elif args.force == "track":
        force_track = True
        
    # Check if final cached points3d exist and --force is not specified
    if not force_track and os.path.exists(final_points3d_path):
        points3d_filtered = load_points3d(cache_dir)
        has_real_colors = False
        if points3d_filtered is not None and len(points3d_filtered) > 0:
            first_pt = points3d_filtered[0]
            if "color" in first_pt["track"]:
                all_colors = np.array([pt["track"]["color"] for pt in points3d_filtered])
                if np.std(all_colors) > 1e-4 or not np.allclose(all_colors[0], [0.5, 0.5, 0.5]):
                    has_real_colors = True
        
        if has_real_colors:
            points3d_optimized = load_points3d_all(cache_dir)
            print("\n==================================================")
            print("Found final cached 3D point map with true colors. Loading directly...")
            print("==================================================")
            if not args.no_viewer:
                print("\n--- Step 13: Launching 3D Viewer ---")
                viewer = Viewer3D(points3d_filtered, poses, cameras, config, cache_dir)
                viewer.show()
            return
        else:
            print("\nFound cached 3D points, but they lack true color information. Re-running mapping to extract colors...")
            
    # Executing the full pipeline
    features_cache = step_extract_features(image_names, image_dir, cache_dir, device, config, force_feature)
    graph_edges, matches_cache = step_calculate_graph_and_matches(poses, image_names, cache_dir, device, config, features_cache, force_graph)
    tracks, points3d = step_build_tracks_and_triangulate(graph_edges, matches_cache, features_cache, poses, cameras, cache_dir, force_track)
    points3d_opt, points3d_filtered = step_bundle_adjustment_and_filter(points3d, poses, cameras, cache_dir, config)
    
    # 12. Done
    print("\n--- Step 12: Pipeline Done ---")
    print(f"- Number of 3D points: {len(points3d_filtered)}")
    
    if not args.no_viewer:
        print("\n--- Step 13: Launching 3D Viewer ---")
        viewer = Viewer3D(points3d_filtered, poses, cameras, config, cache_dir)
        viewer.show()

if __name__ == "__main__":
    main()
