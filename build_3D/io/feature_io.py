import os
import pickle
from typing import Dict, Any, Optional

def get_safe_pkl_filename(name: str) -> str:
    """Replace path separators with underscores to make image filenames safe as cache filenames."""
    return name.replace("/", "_").replace("\\", "_") + ".pkl"

def save_features(cache_dir: str, image_name: str, data: Dict[str, Any]) -> None:
    """Save feature extraction results for an image."""
    features_dir = os.path.join(cache_dir, "features")
    os.makedirs(features_dir, exist_ok=True)
    file_path = os.path.join(features_dir, get_safe_pkl_filename(image_name))
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

def load_features(cache_dir: str, image_name: str) -> Optional[Dict[str, Any]]:
    """Load feature extraction results for an image."""
    features_dir = os.path.join(cache_dir, "features")
    file_path = os.path.join(features_dir, get_safe_pkl_filename(image_name))
    if not os.path.exists(file_path):
        return None
    with open(file_path, "rb") as f:
        return pickle.load(f)

def save_matches(cache_dir: str, name1: str, name2: str, data: Dict[str, Any]) -> None:
    """Save match results for an image pair."""
    matches_dir = os.path.join(cache_dir, "matches")
    os.makedirs(matches_dir, exist_ok=True)
    # Sort names to ensure same cache key for A-B and B-A
    sorted_names = sorted([name1, name2])
    pair_name = f"{sorted_names[0]}__AND__{sorted_names[1]}"
    file_path = os.path.join(matches_dir, get_safe_pkl_filename(pair_name))
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

def load_matches(cache_dir: str, name1: str, name2: str) -> Optional[Dict[str, Any]]:
    """Load match results for an image pair."""
    matches_dir = os.path.join(cache_dir, "matches")
    sorted_names = sorted([name1, name2])
    pair_name = f"{sorted_names[0]}__AND__{sorted_names[1]}"
    file_path = os.path.join(matches_dir, get_safe_pkl_filename(pair_name))
    if not os.path.exists(file_path):
        return None
    with open(file_path, "rb") as f:
        return pickle.load(f)

def save_graph(cache_dir: str, graph: Any) -> None:
    """Save neighbor graph structure."""
    os.makedirs(cache_dir, exist_ok=True)
    file_path = os.path.join(cache_dir, "graph", "graph.pkl")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(graph, f)

def load_graph(cache_dir: str) -> Optional[Any]:
    """Load neighbor graph structure."""
    file_path = os.path.join(cache_dir, "graph", "graph.pkl")
    if not os.path.exists(file_path):
        return None
    with open(file_path, "rb") as f:
        return pickle.load(f)

def save_tracks(cache_dir: str, tracks: Any) -> None:
    """Save reconstructed global tracks."""
    os.makedirs(cache_dir, exist_ok=True)
    file_path = os.path.join(cache_dir, "tracks", "tracks.pkl")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(tracks, f)

def load_tracks(cache_dir: str) -> Optional[Any]:
    """Load reconstructed global tracks."""
    file_path = os.path.join(cache_dir, "tracks", "tracks.pkl")
    if not os.path.exists(file_path):
        return None
    with open(file_path, "rb") as f:
        return pickle.load(f)

def save_points3d(cache_dir: str, points3d: Any) -> None:
    """Save 3D point cloud results."""
    os.makedirs(cache_dir, exist_ok=True)
    file_path = os.path.join(cache_dir, "points3d", "points3d.pkl")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(points3d, f)

def load_points3d(cache_dir: str) -> Optional[Any]:
    """Load 3D point cloud results."""
    file_path = os.path.join(cache_dir, "points3d", "points3d.pkl")
    if not os.path.exists(file_path):
        return None
    with open(file_path, "rb") as f:
        return pickle.load(f)

def save_points3d_all(cache_dir: str, points3d: Any) -> None:
    """Save all unfiltered 3D point cloud results."""
    os.makedirs(cache_dir, exist_ok=True)
    file_path = os.path.join(cache_dir, "points3d", "points3d_all.pkl")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(points3d, f)

def load_points3d_all(cache_dir: str) -> Optional[Any]:
    """Load all unfiltered 3D point cloud results."""
    file_path = os.path.join(cache_dir, "points3d", "points3d_all.pkl")
    if not os.path.exists(file_path):
        return None
    with open(file_path, "rb") as f:
        return pickle.load(f)
