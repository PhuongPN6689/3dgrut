import os
import sys
import argparse
import glob
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F

import shutil

# Try importing pycolmap
try:
    import pycolmap
except ImportError:
    print("[-] Error: pycolmap is not installed. Please run 'pip install pycolmap'.")
    sys.exit(1)

# Try importing transformers
try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
except ImportError:
    print("[-] Error: transformers is not installed. Please run 'pip install transformers'.")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Densify COLMAP sparse points using Depth Anything V2")
    parser.add_argument("--scene_path", type=str, required=True, help="Path to the scene directory (containing train/)")
    parser.add_argument("--model_path", type=str, default="depth-anything/Depth-Anything-V2-Small-hf", help="Hugging Face model path or local model directory")
    parser.add_argument("--stride", type=int, default=8, help="Stride for pixel downsampling when back-projecting (default: 8)")
    parser.add_argument("--min_depth", type=float, default=0.1, help="Minimum depth clip value (default: 0.1)")
    parser.add_argument("--max_depth", type=float, default=100.0, help="Maximum depth clip value (default: 100.0)")
    parser.add_argument("--max_images", type=int, default=-1, help="Maximum number of images to process for testing (-1 for all)")
    return parser.parse_args()

def extract_camera_intrinsics(camera):
    params = camera.params
    if camera.model_name == "PINHOLE":
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    elif camera.model_name == "SIMPLE_PINHOLE":
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    elif camera.model_name in ["OPENCV", "RADIAL", "SIMPLE_RADIAL"]:
        fx, fy = params[0], params[1]
        cx, cy = params[2], params[3]
    else:
        # Fallback
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    return fx, fy, cx, cy

def get_image_pose(image):
    if hasattr(image, "cam_from_world"):
        pose = image.cam_from_world()
        R = pose.rotation.matrix()
        t = pose.translation
    else:
        from pycolmap import qvec_to_rotmat
        R = qvec_to_rotmat(image.qvec)
        t = image.tvec
    return R, t

def main():
    args = parse_args()
    
    sparse_dir = os.path.join(args.scene_path, "train", "sparse", "0")
    images_dir = os.path.join(args.scene_path, "train", "images")
    
    if not os.path.exists(sparse_dir):
        print(f"[-] Error: Sparse COLMAP directory not found at {sparse_dir}")
        sys.exit(1)
        
    bin_path = os.path.join(sparse_dir, "points3D.bin")
    bak_bin = os.path.join(sparse_dir, "points3D.bin.bak")
    
    # Temporarily restore backup binary if points3D.bin is missing
    restored_bin = False
    if not os.path.exists(bin_path) and os.path.exists(bak_bin):
        shutil.copy2(bak_bin, bin_path)
        print(f"[+] Temporarily restored binary points3D.bin from points3D.bin.bak for pycolmap loading.")
        restored_bin = True
        
    print(f"[+] Loading COLMAP reconstruction from {sparse_dir}...")
    try:
        reconstruction = pycolmap.Reconstruction(sparse_dir)
    finally:
        # Clean up temporary binary points3D.bin immediately after loading if we restored it
        if restored_bin and os.path.exists(bin_path):
            os.remove(bin_path)
            print("[*] Cleaned up temporary binary points3D.bin after loading.")
    
    print(f"[+] Found {len(reconstruction.cameras)} cameras, {len(reconstruction.images)} images, and {len(reconstruction.points3D)} 3D points.")
    
    # Initialize Depth Anything V2 model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Initializing Depth Anything V2 model '{args.model_path}' on {device}...")
    try:
        image_processor = AutoImageProcessor.from_pretrained(args.model_path)
        model = AutoModelForDepthEstimation.from_pretrained(args.model_path).to(device)
        model.eval()
    except Exception as e:
        print(f"[-] Failed to load model: {e}")
        print("[*] Checking if local fallback or huggingface download is blocked. Please ensure online mode is enabled on Kaggle or model path is correct.")
        sys.exit(1)
        
    # Read existing 3D points
    original_points = []
    original_colors = []
    original_errors = []
    
    for pt3d_id, pt3d in reconstruction.points3D.items():
        original_points.append(pt3d.xyz)
        original_colors.append(pt3d.color)
        original_errors.append(pt3d.error)
        
    print(f"[+] Read {len(original_points)} original COLMAP points.")
    
    new_points = []
    new_colors = []
    new_errors = []
    
    processed = 0
    # Process each image in the reconstruction
    for img_idx, (image_id, colmap_image) in enumerate(reconstruction.images.items()):
        if args.max_images > 0 and processed >= args.max_images:
            print(f"[+] Reached max_images limit ({args.max_images}). Stopping densification...")
            break
            
        img_name = colmap_image.name
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            print(f"[WARNING] Image file not found: {img_path}")
            continue
            
        print(f"[*] [{img_idx+1}/{len(reconstruction.images)}] Aligning depth for image: {img_name}")
        processed += 1
        
        # Load image
        img_pil = Image.open(img_path).convert("RGB")
        W, H = img_pil.size
        
        # Get pose and intrinsics
        R, t = get_image_pose(colmap_image)
        camera = reconstruction.cameras[colmap_image.camera_id]
        fx, fy, cx, cy = extract_camera_intrinsics(camera)
        
        # 1. Run monocular depth estimation
        inputs = image_processor(images=img_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth
            
        # Interpolate to full image size
        prediction = F.interpolate(
            predicted_depth.unsqueeze(1),
            size=(H, W),
            mode="bicubic",
            align_corners=False
        ).squeeze()
        D_pred = prediction.cpu().numpy() # [H, W] relative depth/disparity map
        
        # 2. Gather matching COLMAP 3D points for this image
        pts_colmap_depth = []
        pts_pred_val = []
        
        for point2D in colmap_image.points2D:
            if point2D.has_point3D():
                pt3d = reconstruction.points3D[point2D.point3D_id]
                P_world = pt3d.xyz
                
                # Transform to camera space
                P_cam = R @ P_world + t
                depth = P_cam[2] # Z is depth
                
                u, v = int(point2D.xy[0]), int(point2D.xy[1])
                if 0 <= u < W and 0 <= v < H:
                    pts_colmap_depth.append(depth)
                    pts_pred_val.append(D_pred[v, u])
                    
        # 3. Perform Scale-Shift Alignment
        if len(pts_colmap_depth) < 10:
            print(f"[WARNING] Not enough keypoints ({len(pts_colmap_depth)}) on {img_name} for depth alignment. Skipping...")
            continue
            
        D_col = np.array(pts_colmap_depth)
        D_p = np.array(pts_pred_val)
        
        # Determine if model output correlates better directly or inversely
        corr_direct = np.corrcoef(D_col, D_p)[0, 1]
        D_inv = 1.0 / np.clip(D_p, 1e-5, None)
        corr_inverse = np.corrcoef(D_col, D_inv)[0, 1]
        
        # Fallback handle if correlation contains NaNs
        if np.isnan(corr_direct): corr_direct = 0.0
        if np.isnan(corr_inverse): corr_inverse = 0.0
        
        if abs(corr_inverse) > abs(corr_direct):
            D_ref = D_inv
            D_ref_full = 1.0 / np.clip(D_pred, 1e-5, None)
            use_inverse = True
        else:
            D_ref = D_p
            D_ref_full = D_pred
            use_inverse = False
            
        # Fit scale-shift using robust RANSAC (or least-squares fallback)
        try:
            from sklearn.linear_model import RANSACRegressor
            ransac = RANSACRegressor(min_samples=min(10, len(D_ref) // 2))
            ransac.fit(D_ref.reshape(-1, 1), D_col)
            a = ransac.estimator_.coef_[0]
            b = ransac.estimator_.intercept_
        except Exception:
            # Fallback to standard polyfit
            a, b = np.polyfit(D_ref, D_col, 1)
            
        print(f"    - Alignment (inverse={use_inverse}): scale={a:.4f}, shift={b:.4f} (corr: direct={corr_direct:.3f}, inverse={corr_inverse:.3f})")
        
        # Calculate absolute depth map
        D_abs = a * D_ref_full + b
        D_abs = np.clip(D_abs, args.min_depth, args.max_depth)
        
        # 4. Backproject dense points (using stride)
        img_np = np.array(img_pil) / 255.0 # [H, W, 3]
        
        # Create meshgrid of coordinates
        us = np.arange(0, W, args.stride)
        vs = np.arange(0, H, args.stride)
        u_mesh, v_mesh = np.meshgrid(us, vs)
        
        # Filter bounds
        u_flat = u_mesh.flatten()
        v_flat = v_mesh.flatten()
        
        depths = D_abs[v_flat, u_flat]
        colors = img_np[v_flat, u_flat]
        
        # Compute camera space coordinates
        X_cam = (u_flat - cx) * depths / fx
        Y_cam = (v_flat - cy) * depths / fy
        Z_cam = depths
        
        P_cam_all = np.stack([X_cam, Y_cam, Z_cam], axis=1) # [N, 3]
        
        # Transform back to world space: P_world = R^T @ (P_cam - t)
        R_inv = R.T
        P_world_all = (P_cam_all - t) @ R_inv.T # Transpose multiplication for batch
        
        # Convert colors to 0-255 scale
        colors_rgb = (colors * 255.0).astype(np.uint8)
        
        for i in range(len(P_world_all)):
            new_points.append(P_world_all[i])
            new_colors.append(colors_rgb[i])
            new_errors.append(1.0) # Error set to 1.0 for dense points
            
    print(f"[+] Backprojected {len(new_points)} new dense points from Depth maps.")
    
    # Merge original SIFT points with new dense points
    final_points = original_points + new_points
    final_colors = original_colors + new_colors
    final_errors = original_errors + new_errors
    
    print(f"[+] Total merged point cloud has {len(final_points)} points.")
    
    # Write to points3D.txt
    txt_path = os.path.join(args.scene_path, "train", "sparse", "0", "points3D.txt")
    print(f"[+] Overwriting merged point cloud to: {txt_path}")
    with open(txt_path, "w") as f:
        f.write("# 3D point list with merged Depth Anything V2 dense points\n")
        f.write(f"# Number of points: {len(final_points)}\n")
        for i in range(len(final_points)):
            pt = final_points[i]
            col = final_colors[i]
            err = final_errors[i]
            # Format: POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK_LIST... (track list is left empty)
            f.write(f"{i+1} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {int(col[0])} {int(col[1])} {int(col[2])} {err:.4f}\n")
            
    # Also delete binary points3D.bin so COLMAP reader prioritizes points3D.txt
    bin_path = os.path.join(args.scene_path, "train", "sparse", "0", "points3D.bin")
    bak_bin = os.path.join(args.scene_path, "train", "sparse", "0", "points3D.bin.bak")
    if os.path.exists(bin_path):
        if not os.path.exists(bak_bin):
            os.rename(bin_path, bak_bin)
            print(f"[+] Backed up binary points3D.bin to points3D.bin.bak")
        else:
            os.remove(bin_path)
            print("[*] Removed points3D.bin to prioritize the new points3D.txt")
            
    print("[+] Depth Densification finished successfully!")

if __name__ == "__main__":
    main()
