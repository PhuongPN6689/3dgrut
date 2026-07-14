import os
import sys
import glob
import time
import zipfile
import subprocess
import argparse
import shutil

class MockProcess:
    """Simulates a running training process for dry-runs."""
    def __init__(self, cmd, scene, duration=3.0):
        self.cmd = cmd
        self.scene = scene
        self.duration = duration
        self.start_time = time.time()
        
    def poll(self):
        # Returns 0 (success) after self.duration seconds have passed
        if time.time() - self.start_time >= self.duration:
            return 0
        return None

def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End Multi-GPU 3DGS Pipeline Runner")
    
    # Modes
    parser.add_argument("--train_mode", type=str, default="True", choices=["True", "False", "true", "false"], help="Train and render, or just render from checkpoints")
    parser.add_argument("--config_name", type=str, default="apps/colmap_3dgut_mcmc.yaml", help="Config YAML file path")
    parser.add_argument("--iterations", type=int, default=30000, help="Number of iterations for training")
    parser.add_argument("--max_gaussians", type=int, default=3000000, help="Maximum number of Gaussians allowed")
    parser.add_argument("--dry_run", type=str, default="False", choices=["True", "False", "true", "false"], help="Simulate execution without running GPU processes")
    
    # Custom strategy options during training
    parser.add_argument("--enable_custom_a_train", type=str, default="False", choices=["True", "False", "true", "false"])
    parser.add_argument("--enable_custom_a_test", type=str, default="False", choices=["True", "False", "true", "false"])
    parser.add_argument("--enable_custom_b", type=str, default="False", choices=["True", "False", "true", "false"])
    
    # build_3D (SuperPoint + LightGlue) options
    parser.add_argument("--enable_build_3d", type=str, default="True", choices=["True", "False", "true", "false"], help="Enable build_3D (SuperPoint+LightGlue) to densify Colmap")
    parser.add_argument("--build_3d_max_keypoints", type=int, default=4096)
    parser.add_argument("--build_3d_detection_threshold", type=float, default=0.005)
    parser.add_argument("--build_3d_position_weight", type=float, default=1.0)
    parser.add_argument("--build_3d_rotation_weight", type=float, default=0.0)
    parser.add_argument("--build_3d_num_neighbors", type=int, default=4)
    parser.add_argument("--build_3d_extra_links", type=int, default=1)
    parser.add_argument("--build_3d_min_matches", type=int, default=50)
    parser.add_argument("--build_3d_ransac_threshold", type=float, default=1.5)
    parser.add_argument("--build_3d_reproj_threshold", type=float, default=1.5)
    parser.add_argument("--build_3d_min_track_length", type=int, default=3)
    parser.add_argument("--build_3d_min_baseline_angle", type=float, default=2.0)
    parser.add_argument("--build_3d_max_points", type=int, default=100000, help="Maximum points merged in points3D.txt to avoid OOM")
    
    # Paths
    parser.add_argument("--data_dir", type=str, default="/kaggle/input/datasets/phuongpn2/vai-nvs-data-phase-1/phase1/private_set1", help="Path to input dataset folder containing scenes")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/output", help="Output directory for checkpoints")
    parser.add_argument("--submission_dir", type=str, default="/kaggle/working/submission", help="Directory where submission images are generated")
    parser.add_argument("--zip_path", type=str, default="/kaggle/working/submission_round1.zip", help="Path where the final zip will be stored")
    
    # Execution
    parser.add_argument("--gpus", type=str, default="0,1", help="Comma-separated list of GPU IDs to use, e.g., '0,1'")
    parser.add_argument("--scenes", type=str, default="HNI0131,HNI0265", help="Comma-separated list of scene names to process, or 'auto' to auto-detect")
    
    args = parser.parse_args()
    
    # Convert string boolean types to actual booleans
    def to_bool(val):
        return val.lower() in ["true", "1"]
        
    args.train_mode = to_bool(args.train_mode)
    args.enable_custom_a_train = to_bool(args.enable_custom_a_train)
    args.enable_custom_a_test = to_bool(args.enable_custom_a_test)
    args.enable_custom_b = to_bool(args.enable_custom_b)
    args.enable_build_3d = to_bool(args.enable_build_3d)
    args.dry_run = to_bool(args.dry_run)
    
    return args

def setup_cuda_environment():
    # Keep the CUDA dynamic linker paths matched in notebook
    cuda_lib = "/usr/local/nvidia/lib64"
    if os.path.exists(cuda_lib):
        os.environ["LD_LIBRARY_PATH"] = (
            cuda_lib
            + ":/usr/local/cuda/lib64"
            + ":/usr/local/cuda/lib64/stubs:"
            + os.environ.get("LD_LIBRARY_PATH", "")
        )
        os.environ["LIBRARY_PATH"] = (
            cuda_lib
            + ":/usr/local/cuda/lib64"
            + ":/usr/local/cuda/lib64/stubs:"
            + os.environ.get("LIBRARY_PATH", "")
        )

def main():
    args = parse_args()
    setup_cuda_environment()
    
    print("=== Configuration ===")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
        
    # Auto-detect data_dir if default does not exist (or use public_set if local)
    if not os.path.exists(args.data_dir):
        # Local search fallback for testing
        local_public = "data_phase1/public_set"
        if os.path.exists(local_public):
            args.data_dir = local_public
            print(f"[+] Local fallback detected data_dir at: {args.data_dir}")
        else:
            found = glob.glob("/kaggle/input/**/private_set1", recursive=True)
            if found:
                args.data_dir = found[0]
                print(f"[+] Automatically detected data_dir at: {args.data_dir}")
            else:
                print(f"[-] Error: Dataset directory not found at {args.data_dir}")
                sys.exit(1)
            
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.submission_dir, exist_ok=True)
    
    # Parse scenes
    if args.scenes.lower() == "auto":
        scenes = sorted([
            d for d in os.listdir(args.data_dir)
            if os.path.isdir(os.path.join(args.data_dir, d))
        ])
    else:
        scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
        
    print(f"[+] Scenes to process: {scenes}")
    
    # Verify tiny-cuda-nn if NHT is used
    try:
        import tinycudann
        has_tcnn = True
    except ImportError:
        has_tcnn = False
        
    config_name = args.config_name
    if args.train_mode and (not has_tcnn) and ("nht" in config_name.lower()):
        print("[WARNING] tiny-cuda-nn is not installed. Disabling NHT config, fallback to non-NHT config.")
        config_name = "apps/colmap_3dgut_mcmc.yaml"
        
    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    
    if args.train_mode:
        print("\n========== TRAIN + RENDER PIPELINE ==========\n")
        active_processes = {}
        scenes_queue = list(scenes)
        
        while scenes_queue or active_processes:
            # 1. Check active processes for completion
            for gpu_id in list(active_processes.keys()):
                proc, scene_name = active_processes[gpu_id]
                ret = proc.poll()
                
                if ret is not None:
                    print(f"\n>>> Train completed for {scene_name} (GPU {gpu_id}) with return code={ret}")
                    del active_processes[gpu_id]
                    
                    # For dry-runs, write a dummy checkpoint to keep downstream working
                    if args.dry_run:
                        dummy_ckpt_dir = os.path.join(args.output_dir, scene_name)
                        os.makedirs(dummy_ckpt_dir, exist_ok=True)
                        scene_ckpt = os.path.join(dummy_ckpt_dir, "ckpt_last.pt")
                        with open(scene_ckpt, "w") as f:
                            f.write("dummy checkpoint")
                    else:
                        scene_ckpt = os.path.join(args.output_dir, scene_name, "ckpt_last.pt")
                        if not os.path.exists(scene_ckpt):
                            ckpts = glob.glob(os.path.join(args.output_dir, scene_name, "**", "ckpt_last.pt"), recursive=True)
                            if ckpts:
                                scene_ckpt = ckpts[0]
                            
                    if not os.path.exists(scene_ckpt):
                        print(f"[-] Error: Could not find checkpoint for {scene_name} at {scene_ckpt}")
                        continue
                        
                    # Trigger render
                    scene_test = os.path.join(args.data_dir, scene_name, "test", "test_poses.csv")
                    out_dir = os.path.join(args.submission_dir, scene_name)
                    
                    print(f"[*] Rendering novel views for {scene_name} using GPU {gpu_id}...")
                    render_cmd = (
                        f"CUDA_VISIBLE_DEVICES={gpu_id} "
                        f"python render_submission.py "
                        f"-m {scene_ckpt} "
                        f"-p {scene_test} "
                        f"-o {out_dir}"
                    )
                    
                    if args.dry_run:
                        print(f"[DRY-RUN] Would run: {render_cmd}")
                        os.makedirs(out_dir, exist_ok=True)
                        with open(os.path.join(out_dir, "0001.png"), "w") as f:
                            f.write("dummy image")
                    else:
                        subprocess.run(render_cmd, shell=True, env=os.environ.copy())
            
            # 2. Launch new scene on available GPU
            for gpu_id in gpus:
                if gpu_id in active_processes:
                    continue
                if not scenes_queue:
                    continue
                    
                scene = scenes_queue.pop(0)
                orig_scene_dir = os.path.join(args.data_dir, scene)
                
                if args.enable_build_3d:
                    print(f"\n[+] Processing build_3D reconstruction for {scene}...")
                    # Working copy directory so we can write/modify points3D.txt
                    new_scene_dir = os.path.join("/tmp/data" if args.dry_run else "/kaggle/working/data", scene)
                    os.makedirs(new_scene_dir, exist_ok=True)
                    
                    # Symlink images
                    new_train_images = os.path.join(new_scene_dir, "train", "images")
                    if not os.path.lexists(new_train_images):
                        os.makedirs(os.path.dirname(new_train_images), exist_ok=True)
                        os.symlink(os.path.join(orig_scene_dir, "train", "images"), new_train_images, target_is_directory=True)
                        
                    # Symlink test poses
                    new_test_dir = os.path.join(new_scene_dir, "test")
                    orig_test_dir = os.path.join(orig_scene_dir, "test")
                    if os.path.exists(orig_test_dir) and not os.path.lexists(new_test_dir):
                        os.symlink(orig_test_dir, new_test_dir, target_is_directory=True)
                        
                    # Copy sparse folder (which contains cameras.bin, images.bin, points3D.bin)
                    new_train_sparse = os.path.join(new_scene_dir, "train", "sparse")
                    if os.path.exists(new_train_sparse):
                        shutil.rmtree(new_train_sparse)
                    shutil.copytree(os.path.join(orig_scene_dir, "train", "sparse"), new_train_sparse)
                    
                    # Run build_map.py
                    print(f"[*] Running build_3D build_map.py for {scene}...")
                    build_map_cmd = (
                        f"python build_3D/build_map.py --dataset {new_scene_dir} --no_viewer "
                        f"superpoint.max_keypoints={args.build_3d_max_keypoints} "
                        f"superpoint.detection_threshold={args.build_3d_detection_threshold} "
                        f"neighbor.position_weight={args.build_3d_position_weight} "
                        f"neighbor.rotation_weight={args.build_3d_rotation_weight} "
                        f"neighbor.num_neighbors={args.build_3d_num_neighbors} "
                        f"graph.extra_links={args.build_3d_extra_links} "
                        f"matching.min_matches={args.build_3d_min_matches} "
                        f"matching.ransac_threshold={args.build_3d_ransac_threshold} "
                        f"triangulation.reprojection_threshold={args.build_3d_reproj_threshold} "
                        f"filter.min_track_length={args.build_3d_min_track_length} "
                        f"filter.min_baseline_angle={args.build_3d_min_baseline_angle}"
                    )
                    
                    if args.dry_run:
                        print(f"[DRY-RUN] Would run: {build_map_cmd}")
                    else:
                        subprocess.run(build_map_cmd, shell=True, env=os.environ.copy())
                    
                    # Merge SuperPoint+LightGlue points into COLMAP sparse model
                    print(f"[*] Merging SuperPoint+LightGlue points into sparse model for {scene}...")
                    merge_cmd = f"python build_3D/merge_to_colmap.py --scene_path {new_scene_dir} --mode merge --max_points {args.build_3d_max_points}"
                    
                    if args.dry_run:
                        print(f"[DRY-RUN] Would run: {merge_cmd}")
                        # In dry_run, write a dummy points3D.txt to make sure downstream path is happy
                        with open(os.path.join(new_train_sparse, "0", "points3D.txt"), "w") as f:
                            f.write("# Dummy merged points")
                    else:
                        subprocess.run(merge_cmd, shell=True, env=os.environ.copy())
                    
                    # Set train path to the modified copy
                    train_path = os.path.join(new_scene_dir, "train")
                else:
                    train_path = os.path.join(orig_scene_dir, "train")
                    print(f"[+] build_3D is disabled. Training directly from original: {train_path}")
                    
                # Build training command
                cmd = (
                    f"CUDA_VISIBLE_DEVICES={gpu_id} "
                    f"python train.py "
                    f"--config-name {config_name} "
                    f"path={train_path} "
                    f"out_dir={args.output_dir} "
                    f"experiment_name={scene} "
                    f"n_iterations={args.iterations} "
                    f"strategy.add.max_n_gaussians={args.max_gaussians} "
                    f"+strategy.enable_custom_a_train={str(args.enable_custom_a_train).lower()} "
                    f"+strategy.enable_custom_a_test={str(args.enable_custom_a_test).lower()} "
                    f"+strategy.enable_custom_b={str(args.enable_custom_b).lower()}"
                )
                
                print(f"\n[+] Launching train for {scene} on GPU {gpu_id}...")
                print(f"Command: {cmd}")
                
                if args.dry_run:
                    proc = MockProcess(cmd, scene, duration=2.0)
                else:
                    proc = subprocess.Popen(cmd, shell=True, env=os.environ.copy())
                    
                active_processes[gpu_id] = (proc, scene)
                
            time.sleep(1)
            
    else:
        print("\n========== RENDER ONLY MODE ==========\n")
        for scene_name in scenes:
            scene_ckpt = os.path.join(args.output_dir, scene_name, "ckpt_last.pt")
            if not os.path.exists(scene_ckpt):
                ckpts = glob.glob(os.path.join(args.output_dir, scene_name, "**", "ckpt_last.pt"), recursive=True)
                if ckpts:
                    scene_ckpt = ckpts[0]
                    
            if not os.path.exists(scene_ckpt):
                print(f"[SKIP] {scene_name}: checkpoint not found, cannot render.")
                continue
                
            scene_test = os.path.join(args.data_dir, scene_name, "test", "test_poses.csv")
            out_dir = os.path.join(args.submission_dir, scene_name)
            
            print(f"\n[*] Rendering {scene_name} from checkpoint {scene_ckpt}...")
            cmd = (
                f"python render_submission.py "
                f"-m {scene_ckpt} "
                f"-p {scene_test} "
                f"-o {out_dir}"
            )
            
            if args.dry_run:
                print(f"[DRY-RUN] Would run: {cmd}")
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, "0001.png"), "w") as f:
                    f.write("dummy image")
            else:
                subprocess.run(cmd, shell=True, env=os.environ.copy())
            
    # Zip output files into submission zip
    if os.path.exists(args.submission_dir):
        print(f"\n[*] Zipping submission directory {args.submission_dir} into {args.zip_path}...")
        with zipfile.ZipFile(args.zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(args.submission_dir):
                for f in files:
                    fp = os.path.join(root, f)
                    z.write(fp, os.path.relpath(fp, args.submission_dir))
        print(f"[+] Zip created successfully at: {args.zip_path}")
    else:
        print("[-] Error: Submission directory not found. No zip created.")

if __name__ == "__main__":
    main()
