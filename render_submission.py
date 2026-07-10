import os
import sys
import math
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.datasets.protocols import Batch
from threedgrut.utils.render import apply_background, apply_feature_decoder, apply_post_processing
from threedgrut.utils.logger import logger
import ncore.sensors
from ncore.data import OpenCVPinholeCameraModelParameters, ShutterType

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[1] * qvec[3] + 2 * qvec[0] * qvec[2],
         0.0],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
         0.0],
        [2 * qvec[1] * qvec[3] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2,
         0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

def render_poses(checkpoint_path, test_poses_path, output_images_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for rendering")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    conf = checkpoint["config"]
    
    model = MixtureOfGaussians(conf).to(device)
    model.init_from_checkpoint(checkpoint, setup_optimizer=False)
    model.build_acc()
    model.eval()
    post_processing = None
    method = conf.post_processing.method
    if "post_processing" in checkpoint and method == "linear-to-srgb":
        from threedgrut.utils.post_processing_linear_to_srgb import LinearToSrgbPostProcessing
        post_processing = LinearToSrgbPostProcessing()
        post_processing.load_state_dict(checkpoint["post_processing"]["module"])
        post_processing = post_processing.to(device)
    elif "post_processing" in checkpoint and method == "ppisp":
        from ppisp import PPISP, PPISPConfig
        use_controller = conf.post_processing.get("use_controller", True)
        n_distillation_steps = conf.post_processing.get("n_distillation_steps", 5000)
        if use_controller and n_distillation_steps > 0:
            main_training_steps = conf.n_iterations - n_distillation_steps
            controller_activation_ratio = main_training_steps / conf.n_iterations
            controller_distillation = True
        elif use_controller:
            controller_activation_ratio = 0.8
            controller_distillation = False
        else:
            controller_activation_ratio = 0.0
            controller_distillation = False
        ppisp_config = PPISPConfig(
            use_controller=use_controller,
            controller_distillation=controller_distillation,
            controller_activation_ratio=controller_activation_ratio,
        )
        post_processing = PPISP.from_state_dict(checkpoint["post_processing"]["module"], config=ppisp_config)
        post_processing = post_processing.to(device)
    feature_decoder = None
    if "feature_decoder" in checkpoint:
        from threedgrut.model.feature_decoder import FeatureDecoder
        from threedgrut.model.features import Features
        if model.feature_type == Features.Type.NHT:
            dec = conf.model.nht_decoder
            feature_decoder = FeatureDecoder(
                ray_feature_dim=model.ray_feature_dim,
                hidden_dim=dec.hidden_dim,
                num_layers=getattr(dec, "num_layers", 4),
                dir_encoding=getattr(dec, "dir_encoding", "SphericalHarmonics"),
                dir_encoding_degree=getattr(dec, "dir_encoding_degree", 3),
                sh_scale=getattr(dec, "sh_scale", 1.0),
                output_activation=getattr(dec, "output_activation", "Sigmoid"),
                ema_decay=getattr(dec, "ema_decay", 0.0),
                ema_start_step=getattr(dec, "ema_start_step", 0),
                unpremultiply_alpha=getattr(dec, "unpremultiply_alpha", False),
            ).to(device)
            feature_decoder.load_state_dict(checkpoint["feature_decoder"]["module"])
            ema_state = checkpoint["feature_decoder"].get("ema")
            if ema_state is not None:
                feature_decoder.load_ema_state_dict(ema_state)
                feature_decoder.apply_ema_shadow()
            feature_decoder.eval()
    df = pd.read_csv(test_poses_path)
    os.makedirs(output_images_dir, exist_ok=True)
    k = 0.0
    try:
        scene_dir = os.path.dirname(os.path.dirname(test_poses_path))
        cameras_bin_path = os.path.join(scene_dir, "train", "sparse", "0", "cameras.bin")
        if os.path.exists(cameras_bin_path):
            from threedgrut.datasets.utils import read_colmap_intrinsics_binary
            intrinsics = read_colmap_intrinsics_binary(cameras_bin_path)
            for cam_id in intrinsics:
                intr = intrinsics[cam_id]
                if intr.model == "SIMPLE_RADIAL" and len(intr.params) >= 4:
                    k = float(intr.params[3])
                    print(f"Detected SIMPLE_RADIAL camera from cameras.bin, k = {k}")
                    break
    except Exception as e:
        print(f"Could not read camera distortion params: {e}")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_name = row['image_name']
        qw, qx, qy, qz = row['qw'], row['qx'], row['qy'], row['qz']
        tx, ty, tz = row['tx'], row['ty'], row['tz']
        fx, fy = row['fx'], row['fy']
        cx, cy = row['cx'], row['cy']
        width, height = int(row['width']), int(row['height'])
        R_4x4 = qvec2rotmat([qw, qx, qy, qz])
        W2C = np.eye(4, dtype=np.float32)
        W2C[:3, :3] = R_4x4[:3, :3]
        W2C[:3, 3] = [tx, ty, tz]
        C2W = np.linalg.inv(W2C)
        T_to_world = torch.tensor(C2W, dtype=torch.float32, device=device).unsqueeze(0)
        params = OpenCVPinholeCameraModelParameters(
            resolution=np.array([width, height], dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([cx, cy], dtype=np.float32),
            focal_length=np.array([fx, fy], dtype=np.float32),
            radial_coeffs=np.array([k, 0, 0, 0, 0, 0], dtype=np.float32),
            tangential_coeffs=np.zeros((2,), dtype=np.float32),
            thin_prism_coeffs=np.zeros((4,), dtype=np.float32),
        )
        camera_model = ncore.sensors.CameraModel.from_parameters(params, device="cpu", dtype=torch.float32)
        u = np.tile(np.arange(width), height)
        v = np.arange(height).repeat(width)
        int_pixel_coords = torch.tensor(np.stack([u, v], axis=1), dtype=torch.int32)
        image_points = camera_model.pixels_to_image_points(int_pixel_coords)
        rays_d_cam = camera_model.image_points_to_camera_rays(image_points)
        rays_o_cam = torch.zeros_like(rays_d_cam)
        rays_ori = rays_o_cam.to(torch.float32).reshape(1, height, width, 3).to(device)
        rays_dir = rays_d_cam.to(torch.float32).reshape(1, height, width, 3).to(device)
        sample = {
            "rays_ori": rays_ori,
            "rays_dir": rays_dir,
            "T_to_world": T_to_world,
            "intrinsics_OpenCVPinholeCameraModelParameters": params.to_dict(),
            "camera_idx": 0,
            "frame_idx": 0,
        }
        batch = Batch(**sample)
        with torch.no_grad():
            outputs = model(batch)
            if feature_decoder is not None:
                outputs = apply_feature_decoder(
                    feature_decoder,
                    outputs,
                    batch,
                    training=False,
                    center_ray_encoding=bool(getattr(conf.model.nht_decoder, "center_ray_encoding", False)),
                )
            outputs = apply_background(model.background, outputs, batch, training=False)
            if post_processing is not None:
                outputs = apply_post_processing(post_processing, outputs, batch, training=False)
            pred_features = outputs["pred_features"]
            pred_img = pred_features.squeeze(0).permute(2, 0, 1)
            torchvision.utils.save_image(pred_img, os.path.join(output_images_dir, image_name))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-m", required=True, type=str)
    parser.add_argument("--test_poses", "-p", required=True, type=str)
    parser.add_argument("--output_dir", "-o", required=True, type=str)
    args = parser.parse_args()
    render_poses(args.checkpoint, args.test_poses, args.output_dir)
