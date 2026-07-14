from typing import Dict, Any, Optional
from ..io.feature_io import save_features, load_features

def get_cached_features(cache_dir: str, image_name: str) -> Optional[Dict[str, Any]]:
    """Retrieve features from cache if they exist."""
    return load_features(cache_dir, image_name)

def cache_features(cache_dir: str, image_name: str, data: Dict[str, Any]) -> None:
    """Save extracted features to cache."""
    save_features(cache_dir, image_name, data)
