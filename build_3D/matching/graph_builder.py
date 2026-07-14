import os
import pickle
import numpy as np
from typing import Dict, List, Tuple, Any, Callable
from ..geometry.pose import compute_pose_distance
from .union_find import UnionFind

def compute_distance_matrix(
    images_poses: Dict[str, Dict[str, Any]],
    pos_weight: float = 1.0,
    rot_weight: float = 0.0
) -> Dict[Tuple[str, str], float]:
    """Compute and return pairwise distances between all camera poses."""
    distances = {}
    image_names = list(images_poses.keys())
    n = len(image_names)
    
    for i in range(n):
        name1 = image_names[i]
        pose1 = images_poses[name1]
        for j in range(i + 1, n):
            name2 = image_names[j]
            pose2 = images_poses[name2]
            
            dist = compute_pose_distance(
                pose1["qvec"], pose1["tvec"],
                pose2["qvec"], pose2["tvec"],
                pos_weight, rot_weight
            )
            distances[(name1, name2)] = dist
            distances[(name2, name1)] = dist
            
    return distances

def build_neighbor_graph(
    images_poses: Dict[str, Dict[str, Any]],
    distances: Dict[Tuple[str, str], float],
    num_neighbors: int = 5
) -> List[Tuple[str, str]]:
    """Build neighbor graph based on pose distances.
    
    For each image, sort all other images by distance. Add edges to the closest
    num_neighbors images to build a well-connected graph.
    
    Returns:
        List of edges as tuples (image1, image2).
    """
    image_names = list(images_poses.keys())
    edges = set()
    
    for name in image_names:
        # Sort other images by distance
        others = [other for other in image_names if other != name]
        others_sorted = sorted(others, key=lambda other: distances[(name, other)])
        
        # Add edges to the K closest neighbors
        added = 0
        for neighbor in others_sorted:
            if added >= num_neighbors:
                break
            edge = tuple(sorted([name, neighbor]))
            edges.add(edge)
            added += 1
                
    return list(edges)

def merge_disconnected_components(
    edges: List[Tuple[str, str]],
    image_names: List[str],
    distances: Dict[Tuple[str, str], float],
    match_fn: Callable[[str, str], Dict[str, Any]],
    extra_links: int = 2,
    min_matches: int = 30
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Find disconnected components and add extra links between them to merge them.
    
    Args:
        edges: Existing edges in the graph.
        image_names: All image names.
        distances: Pairwise pose distances.
        match_fn: A function that takes (img1, img2) and returns a dict with 'matches' and 'inliers'.
        extra_links: Maximum number of links to add.
        min_matches: Minimum number of inliers required to add a link.
        
    Returns:
        Tuple of (new_edges, added_edges).
    """
    uf = UnionFind(image_names)
    for u, v in edges:
        uf.union(u, v)
        
    components = uf.get_components()
    print(f"Initial connected components: {len(components)}")
    if len(components) <= 1:
        # Already fully connected
        return edges, []
        
    # Generate all candidate edges between different components
    candidates = []
    for i in range(len(components)):
        comp_i = components[i]
        for j in range(i + 1, len(components)):
            comp_j = components[j]
            for u in comp_i:
                for v in comp_j:
                    dist = distances.get((u, v), float('inf'))
                    candidates.append((u, v, dist))
                    
    # Sort candidates by distance in ascending order
    candidates.sort(key=lambda x: x[2])
    
    added_edges = []
    links_added = 0
    
    for u, v, dist in candidates:
        if links_added >= extra_links:
            break
            
        # Check if they are still in different components
        if uf.find(u) != uf.find(v):
            print(f"Connecting components: Attempting match between {u} and {v} (distance: {dist:.4f})...")
            # Run matching
            match_res = match_fn(u, v)
            inliers = match_res.get("inliers", np.array([], dtype=bool))
            num_inliers = np.sum(inliers)
            
            if num_inliers >= min_matches:
                print(f"-> Match successful! {num_inliers} inliers. Adding edge {u} - {v}.")
                edge = tuple(sorted([u, v]))
                edges.append(edge)
                added_edges.append(edge)
                uf.union(u, v)
                links_added += 1
                
                # Check if we are down to 1 component
                new_comps = uf.get_components()
                print(f"Remaining components: {len(new_comps)}")
                if len(new_comps) <= 1:
                    print("Graph is now fully connected!")
                    break
            else:
                print(f"-> Match failed: Only {num_inliers} inliers (min required: {min_matches}).")
                
    return edges, added_edges

def get_cached_distances(cache_dir: str) -> Dict[Tuple[str, str], float] | None:
    """Load distance matrix from cache if exists."""
    dist_path = os.path.join(cache_dir, "graph", "distances.pkl")
    if os.path.exists(dist_path):
        with open(dist_path, "rb") as f:
            return pickle.load(f)
    return None

def cache_distances(cache_dir: str, distances: Dict[Tuple[str, str], float]) -> None:
    """Save distance matrix to cache."""
    dist_path = os.path.join(cache_dir, "graph", "distances.pkl")
    os.makedirs(os.path.dirname(dist_path), exist_ok=True)
    with open(dist_path, "wb") as f:
        pickle.dump(distances, f)
