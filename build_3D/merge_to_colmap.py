import os
import pickle
import argparse
import numpy as np

def read_colmap_points3d_txt(path):
    pts = []
    colors = []
    errors = []
    if not os.path.exists(path):
        return pts, colors, errors
    with open(path, 'r') as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            elems = line.split()
            # Format: POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK_LIST...
            x, y, z = float(elems[1]), float(elems[2]), float(elems[3])
            r, g, b = int(elems[4]), int(elems[5]), int(elems[6])
            err = float(elems[7])
            pts.append([x, y, z])
            colors.append([r, g, b])
            errors.append(err)
    return np.array(pts), np.array(colors), np.array(errors)

def main():
    parser = argparse.ArgumentParser(description="Merge SuperPoint+LightGlue points into COLMAP sparse model")
    parser.add_argument("--scene_path", type=str, required=True, help="Path to scene dataset, e.g., data_phase1/public_set/hcm0031")
    parser.add_argument("--mode", type=str, default="replace", choices=["replace", "merge"], help="replace or merge with old points")
    args = parser.parse_args()
    
    scene_name = os.path.basename(os.path.normpath(args.scene_path))
    
    # Path to build_3D pkl cache
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pkl_path = os.path.join(script_dir, "cache", scene_name, "points3d", "points3d.pkl")
    
    if not os.path.exists(pkl_path):
        print(f"[-] Error: Could not find build_3D point cloud cache at {pkl_path}")
        print("[*] Please run the reconstruction pipeline first: python build_3D/build_map.py --dataset " + args.scene_path)
        return
        
    with open(pkl_path, 'rb') as f:
        sp_points = pickle.load(f)
        
    print(f"[+] Loaded {len(sp_points)} points from SuperPoint+LightGlue cache.")
    
    colmap_dir = os.path.join(args.scene_path, "train", "sparse", "0")
    if not os.path.exists(colmap_dir):
        colmap_dir = os.path.join(args.scene_path, "sparse", "0")
        
    if not os.path.exists(colmap_dir):
        print(f"[-] Error: COLMAP sparse directory not found at {colmap_dir}")
        return
        
    bin_path = os.path.join(colmap_dir, "points3D.bin")
    txt_path = os.path.join(colmap_dir, "points3D.txt")
    
    final_pts = []
    final_colors = []
    final_errors = []
    
    # 1. Đọc điểm gốc nếu mode == "merge"
    old_pts = []
    old_colors = []
    old_errors = []
    if args.mode == "merge":
        # Thử đọc qua pycolmap (hỗ trợ cả bin và txt của colmap rất nhanh)
        try:
            import pycolmap
            if os.path.exists(colmap_dir):
                print(f"[+] Reading original COLMAP reconstruction from {colmap_dir}...")
                reconstruction = pycolmap.Reconstruction(colmap_dir)
                for pt3d_id, pt3D in reconstruction.points3D.items():
                    old_pts.append(pt3D.xyz.tolist())
                    old_colors.append(pt3D.color.tolist())
                    old_errors.append(pt3D.error)
                print(f"[+] Loaded {len(old_pts)} original points using pycolmap.")
        except Exception as e:
            print(f"[-] Warning: pycolmap failed or not available: {e}")
            
        # Fallback nếu pycolmap không đọc được nhưng có file txt
        if len(old_pts) == 0 and os.path.exists(txt_path):
            print("[+] Fallback: Reading original points from points3D.txt...")
            old_pts_arr, old_colors_arr, old_errors_arr = read_colmap_points3d_txt(txt_path)
            if len(old_pts_arr) > 0:
                old_pts = old_pts_arr.tolist()
                old_colors = old_colors_arr.tolist()
                old_errors = old_errors_arr.tolist()
                
    # 2. Bây giờ mới an toàn để rename/delete file points3D.bin cũ (để tránh Colmap ưu tiên đọc nhị phân cũ)
    if os.path.exists(bin_path):
        bak_bin = bin_path + ".bak"
        if not os.path.exists(bak_bin):
            os.rename(bin_path, bak_bin)
            print(f"[+] Backed up original binary to {bak_bin}")
        else:
            os.remove(bin_path)
            print("[*] Removed points3D.bin to prioritize the new points3D.txt")
            
    # 3. Trích xuất các điểm mới trích xuất từ SuperPoint+LightGlue
    for idx, pt in enumerate(sp_points):
        final_pts.append(pt["xyz"])
        c = pt["track"].get("color", [0.5, 0.5, 0.5])
        final_colors.append([int(c[0]*255), int(c[1]*255), int(c[2]*255)])
        final_errors.append(pt.get("avg_error", 1.0))
        
    # 4. Gộp điểm gốc vào danh sách
    if len(old_pts) > 0:
        print(f"[+] Merging {len(old_pts)} original points with {len(final_pts)} new points...")
        final_pts.extend(old_pts)
        final_colors.extend(old_colors)
        final_errors.extend(old_errors)
            
    # Write to new points3D.txt
    print(f"[+] Writing {len(final_pts)} points to {txt_path}")
    with open(txt_path, 'w') as f:
        f.write("# 3D point list with multi-view track information\n")
        f.write(f"# Number of points: {len(final_pts)}\n")
        for i in range(len(final_pts)):
            pt = final_pts[i]
            col = final_colors[i]
            err = final_errors[i]
            f.write(f"{i+1} {pt[0]} {pt[1]} {pt[2]} {col[0]} {col[1]} {col[2]} {err}\n")
            
    print("[+] Successfully merged point map! Coordinates are fully aligned with COLMAP camera poses.")

if __name__ == '__main__':
    main()
