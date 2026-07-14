import os
import struct
import numpy as np
from typing import Dict, Tuple, Any
from ..geometry.camera import Camera

# COLMAP camera models
CAMERA_MODEL_NAMES = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE"
}

CAMERA_MODEL_NUM_PARAMS = {
    0: 3, # f, cx, cy
    1: 4, # fx, fy, cx, cy
    2: 4, # f, cx, cy, k
    3: 5, # f, cx, cy, k1, k2
    4: 8,
    5: 8,
    6: 12,
    7: 5,
    8: 4,
    9: 5,
    10: 12
}

def read_next_bytes(fid, num_bytes: int, format_char_sequence: str, endian_character: str = "<") -> Tuple[Any, ...]:
    """Read and unpack the next bytes from a binary file."""
    data = fid.read(num_bytes)
    if len(data) < num_bytes:
        raise EOFError(f"Expected to read {num_bytes} bytes, but got only {len(data)} bytes.")
    return struct.unpack(endian_character + format_char_sequence, data)

def read_cameras_binary(path: str) -> Dict[int, Camera]:
    """Read camera intrinsics from COLMAP cameras.bin file."""
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            
            model_name = CAMERA_MODEL_NAMES.get(model_id, "PINHOLE")
            num_params = CAMERA_MODEL_NUM_PARAMS.get(model_id, 4)
            
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[camera_id] = Camera(
                camera_id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params, dtype=np.float64)
            )
    return cameras

def read_images_binary(path: str) -> Dict[str, Dict[str, Any]]:
    """Read camera extrinsics (poses) from COLMAP images.bin file.
    
    Returns:
        Dict mapping image_name to a dictionary containing:
        - id: int (image ID)
        - qvec: np.ndarray (qw, qx, qy, qz)
        - tvec: np.ndarray (tx, ty, tz)
        - camera_id: int
    """
    images = {}
    with open(path, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5], dtype=np.float64)
            tvec = np.array(binary_image_properties[5:8], dtype=np.float64)
            camera_id = binary_image_properties[8]
            
            # Read null-terminated ASCII image name
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
                
            # Skip 2D point observations stored in images.bin
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            fid.seek(24 * num_points2D, os.SEEK_CUR) # Each point2D has double, double, int64 (8 + 8 + 8 = 24 bytes)
            
            images[image_name] = {
                "id": image_id,
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id
            }
    return images

def read_poses_from_sparse(sparse_dir: str) -> Tuple[Dict[int, Camera], Dict[str, Dict[str, Any]]]:
    """Read both camera intrinsics and poses from a COLMAP sparse directory."""
    cameras_path = os.path.join(sparse_dir, "cameras.bin")
    images_path = os.path.join(sparse_dir, "images.bin")
    
    if not os.path.exists(cameras_path) or not os.path.exists(images_path):
        raise FileNotFoundError(f"COLMAP bin files not found in {sparse_dir}")
        
    cameras = read_cameras_binary(cameras_path)
    images = read_images_binary(images_path)
    
    return cameras, images
