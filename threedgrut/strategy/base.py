# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from threedgrut.model.model import MixtureOfGaussians
from threedgrut.utils.logger import logger


class BaseStrategy:
    def __init__(self, config, model: MixtureOfGaussians) -> None:
        self.conf = config
        self.model = model
        self._suspended = False

        # Load test poses for custom plane interpolation (Cách A)
        self.test_cameras = []
        try:
            import pandas as pd
            parent_path = self.conf.path
            if parent_path.rstrip("/").endswith("train"):
                parent_path = os.path.dirname(parent_path.rstrip("/"))
            csv_path = os.path.join(parent_path, "test", "test_poses.csv")
            
            # Fallback check
            if not os.path.exists(csv_path):
                csv_path = os.path.join(self.conf.path, "test", "test_poses.csv")
            if not os.path.exists(csv_path):
                csv_path = os.path.join(self.conf.path, "test_poses.csv")
                
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                for idx, row in df.iterrows():
                    qw, qx, qy, qz = row['qw'], row['qx'], row['qy'], row['qz']
                    tx, ty, tz = row['tx'], row['ty'], row['tz']
                    fx, fy = row['fx'], row['fy']
                    cx, cy = row['cx'], row['cy']
                    width, height = int(row['width']), int(row['height'])
                    
                    R = self._qvec2rotmat([qw, qx, qy, qz])
                    W2C = np.eye(4, dtype=np.float32)
                    W2C[:3, :3] = R[:3, :3]
                    W2C[:3, 3] = [tx, ty, tz]
                    
                    self.test_cameras.append({
                        "W2C": torch.tensor(W2C, dtype=torch.float32, device=self.model.device),
                        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                        "width": width, "height": height
                    })
                logger.info(f"✨ [BaseStrategy] Loaded {len(self.test_cameras)} test cameras for custom plane interpolation (Cách A).")
            else:
                logger.warning(f"⚠️ [BaseStrategy] test_poses.csv not found at {csv_path}. Custom test-pose densification is disabled.")
        except Exception as e:
            logger.warning(f"⚠️ [BaseStrategy] Failed to load test poses: {e}")

    def suspend(self) -> None:
        """Suspend the strategy, causing all training callbacks to no-op.

        Used during PPISP controller distillation to prevent densification, pruning, and other
        parameter mutations while Gaussian parameters are frozen.
        """
        self._suspended = True

    def init_densification_buffer(self, checkpoint: Optional[dict] = None):
        """Callback function to initialize the densification buffers."""
        pass

    def pre_backward(self, step: int, scene_extent: float, train_dataset, batch=None, writer=None) -> bool:
        """Callback function to be executed before the `loss.backward()` call."""
        if self._suspended:
            return False
        return self._pre_backward(step, scene_extent, train_dataset, batch, writer)

    def _pre_backward(self, step: int, scene_extent: float, train_dataset, batch=None, writer=None) -> bool:
        return False

    def post_backward(self, step: int, scene_extent: float, train_dataset, batch=None, writer=None) -> bool:
        """Callback function to be executed after the `loss.backward()` call."""
        if self._suspended:
            return False
        return self._post_backward(step, scene_extent, train_dataset, batch, writer)

    def _post_backward(self, step: int, scene_extent: float, train_dataset, batch=None, writer=None) -> bool:
        return False

    def post_optimizer_step(self, step: int, scene_extent: float, train_dataset, batch=None, writer=None) -> bool:
        """Callback function to be executed after the optimizer step."""
        self.current_step = step
        if self._suspended:
            return False
        return self._post_optimizer_step(step, scene_extent, train_dataset, batch, writer)

    def _post_optimizer_step(self, step: int, scene_extent: float, train_dataset, batch=None, writer=None) -> bool:
        return False

    def update_gradient_buffer(self, sensor_position: torch.Tensor) -> None:
        """Callback function to update the gradient buffer."""
        pass

    def get_strategy_parameters(self) -> dict:
        """Callback function to get the strategy parameters."""
        return {}

    @torch.no_grad()
    def _update_param_with_optimizer(
        self,
        update_param_fn: Callable[[str, torch.Tensor], torch.Tensor] | None,
        update_optimizer_fn: Callable[[str, torch.Tensor], torch.Tensor] | None,
        names: Union[list[str], None] = None,
    ) -> None:
        """Update the parameters and the state in the optimizers using the provided lambda functions.

        Args:
            update_param_fn: A function that takes the name of the parameter and the parameter itself,
                and returns the new parameter.
            optimizer_fn: A function that takes the key of the optimizer state and the state value,
                and returns the new state value.
            names: A list of key names to update. If None, update all. Default: None.
        """
        for i, param_group in enumerate(self.model.optimizer.param_groups):
            name = param_group["name"]
            if (names is None) or (name in names):
                p = param_group["params"][0]
                p_state = self.model.optimizer.state[p]
                del self.model.optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        v = p_state[key]
                        if update_optimizer_fn is not None:
                            p_state[key] = update_optimizer_fn(key, v)
                if update_param_fn is not None:
                    p_new = update_param_fn(name, p)
                    self.model.optimizer.param_groups[i]["params"] = [p_new]
                    self.model.optimizer.state[p_new] = p_state
                    setattr(self.model, name, p_new)

    def _qvec2rotmat(self, qvec):
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

    def _find_sparse_rays_from_camera(self, camera, points_3d, grid_res=32, rays_per_cam=50, target_density=80, top_percentage=0.3):
        W2C = camera["W2C"]
        R = W2C[:3, :3]
        T = W2C[:3, 3]
        fx, fy = camera["fx"], camera["fy"]
        cx, cy = camera["cx"], camera["cy"]
        W, H = camera["width"], camera["height"]
        
        pts_cam = torch.matmul(points_3d, R.T) + T
        z = pts_cam[:, 2]
        
        valid_mask = z > 0.1
        if valid_mask.sum() == 0:
            return torch.empty((0, 3), device=points_3d.device), torch.empty((0, 3), device=points_3d.device)
            
        pts_cam_valid = pts_cam[valid_mask]
        z_valid = z[valid_mask]
        
        u = (fx * pts_cam_valid[:, 0] / z_valid + cx)
        v = (fy * pts_cam_valid[:, 1] / z_valid + cy)
        
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if in_bounds.sum() == 0:
            return torch.empty((0, 3), device=points_3d.device), torch.empty((0, 3), device=points_3d.device)
            
        u_in = u[in_bounds]
        v_in = v[in_bounds]
        
        grid_w = W / grid_res
        grid_h = H / grid_res
        
        grid_x = (u_in / grid_w).long().clamp(0, grid_res - 1)
        grid_y = (v_in / grid_h).long().clamp(0, grid_res - 1)
        
        grid_counts = torch.zeros((grid_res, grid_res), dtype=torch.int32, device=points_3d.device)
        grid_indices = grid_y * grid_res + grid_x
        grid_counts.put_(grid_indices, torch.ones_like(grid_indices, dtype=torch.int32), accumulate=True)
        
        # 1. Xác định ô hợp lệ: Bản thân >= target_density hoặc lân cận >= target_density
        has_high_density = (grid_counts >= target_density).float().unsqueeze(0).unsqueeze(0)
        neighbor_high_density = F.max_pool2d(has_high_density, kernel_size=3, stride=1, padding=1).squeeze() > 0
        
        # Lấy các ô hợp lệ
        valid_candidates = torch.nonzero(neighbor_high_density)
        if valid_candidates.shape[0] == 0:
            # Fallback nếu không có ô nào đạt target_density, ta lấy các ô có điểm
            valid_candidates = torch.nonzero(grid_counts > 0)
            
        if valid_candidates.shape[0] == 0:
            return torch.empty((0, 3), device=points_3d.device), torch.empty((0, 3), device=points_3d.device)
            
        # Lấy số lượng điểm tại các ô hợp lệ
        counts_at_candidates = grid_counts[valid_candidates[:, 0], valid_candidates[:, 1]]
        
        # 2. Lấy top 30% thưa nhất trong các ô hợp lệ
        sorted_indices = torch.argsort(counts_at_candidates)
        valid_candidates = valid_candidates[sorted_indices]
        counts_at_candidates = counts_at_candidates[sorted_indices]
        
        num_candidates = valid_candidates.shape[0]
        cutoff_idx = max(1, int(num_candidates * top_percentage))
        top_candidates = valid_candidates[:cutoff_idx]
        
        # 3. Chọn ra tối đa rays_per_cam (50 ô) từ top candidates
        selected_candidates = top_candidates[:rays_per_cam]
        
        u_centers = (selected_candidates[:, 1].float() + 0.5) * grid_w
        v_centers = (selected_candidates[:, 0].float() + 0.5) * grid_h
        
        d_cam = torch.stack([
            (u_centers - cx) / fx,
            (v_centers - cy) / fy,
            torch.ones_like(u_centers)
        ], dim=-1)
        d_cam = F.normalize(d_cam, p=2, dim=-1)
        
        R_T = R.T
        o_world = -torch.matmul(T, R)
        rays_o = o_world.unsqueeze(0).repeat(d_cam.shape[0], 1)
        rays_d = torch.matmul(d_cam, R_T.T)
        
        return rays_o, rays_d

    def _plane_fitting_and_ray_intersection(self, rays_o, rays_d, points_3d, k_neighbors=15, max_dist=0.3):
        if rays_o.shape[0] == 0:
            return torch.empty((0, 3), device=points_3d.device)
            
        M = rays_o.shape[0]
        new_points = []
        
        for i in range(M):
            o = rays_o[i]
            d = rays_d[i]
            
            v = points_3d - o
            cross_prod = torch.cross(v, d.expand_as(v), dim=-1)
            dists = torch.norm(cross_prod, dim=-1)
            
            proj = torch.sum(v * d, dim=-1)
            valid_mask = (dists < max_dist) & (proj > 0)
            
            if valid_mask.sum() < k_neighbors:
                continue
                
            valid_points = points_3d[valid_mask]
            valid_dists = dists[valid_mask]
            
            _, indices = torch.topk(valid_dists, k=k_neighbors, largest=False)
            pts_neighborhood = valid_points[indices]
            
            centroid = torch.mean(pts_neighborhood, dim=0)
            pts_centered = pts_neighborhood - centroid
            
            try:
                _, _, V = torch.linalg.svd(pts_centered)
                normal = V[-1, :]
                
                denom = torch.dot(normal, d)
                if torch.abs(denom) < 1e-6:
                    continue
                    
                t = torch.dot(normal, centroid - o) / denom
                if t < 0:
                    continue
                    
                p_new = o + t * d
                new_points.append(p_new)
            except Exception:
                continue
                
        if len(new_points) > 0:
            return torch.stack(new_points, dim=0)
        return torch.empty((0, 3), device=points_3d.device)

    def _edge_guided_3d_interpolation(self, points_3d, gpu_batch, edge_threshold=0.1, n_interpolate=4, target_points=100):
        if gpu_batch.rgb_gt is None:
            return torch.empty((0, 3), device=points_3d.device)
            
        rgb_gt = gpu_batch.rgb_gt
        B, H, W, C = rgb_gt.shape
        images = rgb_gt.permute(0, 3, 1, 2)
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=points_3d.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=points_3d.device).view(1, 1, 3, 3)
        
        gray_imgs = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
        gray_imgs = gray_imgs.unsqueeze(1)
        
        grad_x = F.conv2d(gray_imgs, sobel_x, padding=1)
        grad_y = F.conv2d(gray_imgs, sobel_y, padding=1)
        edges = torch.sqrt(grad_x**2 + grad_y**2).squeeze(1)
        
        if gpu_batch.intrinsics is not None:
            fx, fy, cx, cy = gpu_batch.intrinsics
        else:
            params = gpu_batch.intrinsics_OpenCVPinholeCameraModelParameters
            if params is not None:
                fx, fy = params["focal_length"]
                cx, cy = params["principal_point"]
            else:
                return torch.empty((0, 3), device=points_3d.device)
                
        new_points = []
        
        step = getattr(self, "current_step", 0)
        # Luân phiên kích thước lưới và góc tọa độ theo step
        if (step // 1000) % 2 == 0:
            grid_res = 16
            x_min, y_min = 0.0, 0.0
        else:
            grid_res = 15
            W_grid_temp = W / 15.0
            H_grid_temp = H / 15.0
            x_min, y_min = -W_grid_temp / 2.0, -H_grid_temp / 2.0
            
        W_grid = W / float(grid_res)
        H_grid = H / float(grid_res)
        
        for i in range(B):
            edge_map = edges[i]
            c2w = gpu_batch.T_to_world[i]
            w2c = torch.inverse(c2w)
            
            R = w2c[:3, :3]
            T = w2c[:3, 3]
            
            pts_cam = torch.matmul(points_3d, R.T) + T
            z = pts_cam[:, 2]
            valid_z = z > 0.1
            
            u = (fx * pts_cam[:, 0] / z + cx)
            v = (fy * pts_cam[:, 1] / z + cy)
            
            valid_uv = (u >= 0) & (u < W) & (v >= 0) & (v < H) & valid_z
            if valid_uv.sum() < 5:
                continue
                
            edge_values = torch.zeros(points_3d.shape[0], device=points_3d.device)
            edge_values[valid_uv] = edge_map[v[valid_uv].long(), u[valid_uv].long()]
            
            edge_pts_mask = edge_values > edge_threshold
            if edge_pts_mask.sum() < 5:
                continue
                
            edge_points_3d = points_3d[edge_pts_mask]
            edge_uv = torch.stack([u[edge_pts_mask], v[edge_pts_mask]], dim=-1)
            
            # GIỮ LẠI FALLBACK DỰ PHÒNG MAX 4000: giới hạn số lượng điểm tránh cdist khổng lồ
            if edge_points_3d.shape[0] > 4000:
                perm = torch.randperm(edge_points_3d.shape[0], device=points_3d.device)[:4000]
                edge_points_3d = edge_points_3d[perm]
                edge_uv = edge_uv[perm]
                
            # Chia điểm biên vào các ô lưới
            u_shifted = edge_uv[:, 0] - x_min
            v_shifted = edge_uv[:, 1] - y_min
            
            grid_x = (u_shifted / W_grid).long().clamp(0, grid_res - 1)
            grid_y = (v_shifted / H_grid).long().clamp(0, grid_res - 1)
            
            cell_indices = grid_y * grid_res + grid_x
            
            candidates = []
            candidate_lengths = []
            
            unique_cells = torch.unique(cell_indices)
            unique_cells = unique_cells[torch.randperm(unique_cells.shape[0])]
            
            for cell_id in unique_cells:
                if len(candidates) >= 500:
                    break
                    
                cell_mask = cell_indices == cell_id
                if cell_mask.sum() < 5:
                    continue
                    
                cell_pts = edge_points_3d[cell_mask]
                cell_uv_pts = edge_uv[cell_mask]
                
                # Tính cdist trong nội bộ ô lưới (cực kỳ nhỏ và an toàn)
                dists_cell = torch.cdist(cell_pts, cell_pts)
                
                num_cell_pts = cell_pts.shape[0]
                sample_pts = torch.randperm(num_cell_pts)[:min(15, num_cell_pts)]
                
                for idx in sample_pts:
                    if len(candidates) >= 500:
                        break
                        
                    _, knn_idx = torch.topk(dists_cell[idx], k=5, largest=False)
                    pts_5 = cell_pts[knn_idx]
                    uv_5 = cell_uv_pts[knn_idx]
                    
                    # Kiểm tra khoảng cách lân cận hợp lý
                    valid_dist = True
                    for j in range(4):
                        d_j = dists_cell[knn_idx[j], knn_idx[j+1]]
                        if not (0.02 < d_j < 0.5):
                            valid_dist = False
                            break
                            
                    if valid_dist:
                        # 1. Sắp xếp 5 điểm theo Nearest Neighbor (chống xoắn Bezier bậc 4)
                        sorted_idx = [0]
                        remaining = list(range(1, 5))
                        current = 0
                        for _ in range(4):
                            min_d = float('inf')
                            next_idx = -1
                            for r in remaining:
                                d_r = torch.norm(pts_5[current] - pts_5[r])
                                if d_r < min_d:
                                    min_d = d_r
                                    next_idx = r
                            sorted_idx.append(next_idx)
                            remaining.remove(next_idx)
                            current = next_idx
                            
                        pts_5_sorted = pts_5[sorted_idx]
                        uv_5_sorted = uv_5[sorted_idx]
                        
                        # 2. Tính chiều dài đường đi 2D nối tiếp của bộ 5 điểm
                        path_len_2d = 0.0
                        for j in range(4):
                            path_len_2d += torch.norm(uv_5_sorted[j+1] - uv_5_sorted[j])
                            
                        candidates.append(pts_5_sorted)
                        candidate_lengths.append(path_len_2d)
                        
            if len(candidates) == 0:
                continue
                
            # Lọc lấy top 25 bộ thưa nhất (Lớn nhất)
            candidate_lengths_tensor = torch.stack(candidate_lengths) if isinstance(candidate_lengths[0], torch.Tensor) else torch.tensor(candidate_lengths, device=points_3d.device)
            num_to_select = min(25, len(candidates))
            _, top_indices = torch.topk(candidate_lengths_tensor, k=num_to_select, largest=True)
            
            # Nội suy Bezier bậc 4 cho top 25 bộ thưa nhất
            for idx in top_indices:
                pts_5 = candidates[idx]
                P1, P2, P3, P4, P5 = pts_5[0], pts_5[1], pts_5[2], pts_5[3], pts_5[4]
                
                # Bổ sung đúng 4 điểm tại các vị trí t = 0.125, 0.375, 0.625, 0.875
                t_vals = torch.tensor([0.125, 0.375, 0.625, 0.875], device=points_3d.device)
                for t in t_vals:
                    pt_new = (1 - t)**4 * P1 + 4 * t * (1 - t)**3 * P2 + 6 * t**2 * (1 - t)**2 * P3 + 4 * t**3 * (1 - t) * P4 + t**4 * P5
                    new_points.append(pt_new)
                        
        if len(new_points) > 0:
            return torch.stack(new_points, dim=0)[:target_points]
        return torch.empty((0, 3), device=points_3d.device)

    def _find_neighbor_train_cameras(self, test_camera, train_dataset, k_neighbors=1):
        c2w_test = torch.inverse(test_camera["W2C"]).cpu().numpy()
        pos_test = c2w_test[:3, 3]
        dir_test = c2w_test[:3, 2]
        
        train_poses = train_dataset.get_poses()
        N_train = train_poses.shape[0]
        
        dists = []
        for idx in range(N_train):
            c2w_train = train_poses[idx]
            pos_train = c2w_train[:3, 3]
            dir_train = c2w_train[:3, 2]
            
            pos_dist = np.linalg.norm(pos_test - pos_train)
            rot_dist = 1.0 - np.dot(dir_test, dir_train)
            
            total_dist = pos_dist + 0.5 * rot_dist
            dists.append((total_dist, idx))
            
        dists.sort(key=lambda x: x[0])
        neighbors = [idx for _, idx in dists[:k_neighbors]]
        return neighbors

    def _project_point_to_camera_color(self, point_3d, camera_data, rgb_img):
        W2C = camera_data["W2C"]
        R = W2C[:3, :3]
        T = W2C[:3, 3]
        fx, fy = camera_data["fx"], camera_data["fy"]
        cx, cy = camera_data["cx"], camera_data["cy"]
        W, H = camera_data["width"], camera_data["height"]
        
        pt_cam = torch.matmul(R, point_3d) + T
        z = pt_cam[2]
        if z <= 0.1:
            return None
            
        u = int(fx * pt_cam[0] / z + cx)
        v = int(fy * pt_cam[1] / z + cy)
        
        if 0 <= u < W and 0 <= v < H:
            color = rgb_img[v, u]
            return color
        return None

    def custom_bts_densification(self, train_dataset, batch):
        points_3d = self.model.get_positions()
        new_pts_list = []
        new_colors_list = []
        
        enable_custom_a_train = getattr(self.conf.strategy, "enable_custom_a_train", True)
        enable_custom_a_test = getattr(self.conf.strategy, "enable_custom_a_test", True)
        enable_custom_b = getattr(self.conf.strategy, "enable_custom_b", True)
        
        # 1. CÁCH A: Phóng tia từ camera train hiện tại - bổ sung 50 điểm thưa nhất hợp lệ
        if enable_custom_a_train and batch.T_to_world is not None:
            c2w = batch.T_to_world[0]
            w2c = torch.inverse(c2w)
            
            if batch.intrinsics is not None:
                fx, fy, cx, cy = batch.intrinsics
            else:
                params = batch.intrinsics_OpenCVPinholeCameraModelParameters
                if params is not None:
                    fx, fy = params["focal_length"]
                    cx, cy = params["principal_point"]
                else:
                    fx, fy, cx, cy = None, None, None, None
            
            if fx is not None:
                width, height = batch.rgb_gt.shape[2] if batch.rgb_gt is not None else 1000, batch.rgb_gt.shape[1] if batch.rgb_gt is not None else 1000
                train_camera = {
                    "W2C": w2c, "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                    "width": width, "height": height
                }
                
                # grid_res=32, target_density=80, top_percentage=0.3, rays_per_cam=50
                rays_o_train, rays_d_train = self._find_sparse_rays_from_camera(train_camera, points_3d, grid_res=32, rays_per_cam=50, target_density=80, top_percentage=0.3)
                new_pts_a_train = self._plane_fitting_and_ray_intersection(rays_o_train, rays_d_train, points_3d, k_neighbors=8, max_dist=0.6)
                if new_pts_a_train.shape[0] > 0:
                    new_pts_list.append(new_pts_a_train)
                    train_rgb = batch.rgb_gt[0] if batch.rgb_gt is not None else None
                    train_colors = []
                    for pt in new_pts_a_train:
                        color = self._project_point_to_camera_color(pt, train_camera, train_rgb) if train_rgb is not None else None
                        if color is None:
                            color = torch.tensor([0.5, 0.5, 0.5], device=points_3d.device)
                        train_colors.append(color)
                    new_colors_list.append(torch.stack(train_colors, dim=0))
                    logger.info(f"✨ [Cách A - Train] Added {new_pts_a_train.shape[0]} custom plane points.")
        
        # 2. CÁCH A: Phóng tia từ toàn bộ các camera test - tổng cộng bổ sung đúng 300 điểm
        if enable_custom_a_test and len(self.test_cameras) > 0 and train_dataset is not None:
            test_rays_o_list = []
            test_rays_d_list = []
            test_cameras_used = []
            
            # Chia đều 300 tia cho toàn bộ các camera test để chăm sóc đồng đều mọi góc nhìn test
            rays_per_test_cam = max(1, 300 // len(self.test_cameras))
            
            for cam in self.test_cameras:
                rays_o_test, rays_d_test = self._find_sparse_rays_from_camera(cam, points_3d, grid_res=32, rays_per_cam=rays_per_test_cam, target_density=80, top_percentage=0.3)
                if rays_o_test.shape[0] > 0:
                    test_rays_o_list.append(rays_o_test)
                    test_rays_d_list.append(rays_d_test)
                    test_cameras_used.extend([cam] * rays_o_test.shape[0])
                    
            if len(test_rays_o_list) > 0:
                all_test_rays_o = torch.cat(test_rays_o_list, dim=0)
                all_test_rays_d = torch.cat(test_rays_d_list, dim=0)
                
                new_pts_a_test = self._plane_fitting_and_ray_intersection(all_test_rays_o, all_test_rays_d, points_3d, k_neighbors=8, max_dist=0.6)
                if new_pts_a_test.shape[0] > 0:
                    new_pts_list.append(new_pts_a_test)
                    
                    test_colors = []
                    for idx, pt in enumerate(new_pts_a_test):
                        cam_test = test_cameras_used[idx]
                        neighbor_idxs = self._find_neighbor_train_cameras(cam_test, train_dataset, k_neighbors=1)
                        if len(neighbor_idxs) > 0:
                            neighbor_idx = neighbor_idxs[0]
                            try:
                                train_data = train_dataset[neighbor_idx]
                                image_data = train_data["data"][0]
                                rgb_img = image_data.float().to(self.model.device) / 255.0
                                
                                c2w_neighbor = train_data["pose"][0]
                                w2c_neighbor = torch.inverse(c2w_neighbor)
                                fx_n, fy_n, cx_n, cy_n = train_data["intr"]
                                cam_neighbor = {
                                    "W2C": w2c_neighbor, "fx": fx_n, "fy": fy_n, "cx": cx_n, "cy": cy_n,
                                    "width": rgb_img.shape[1], "height": rgb_img.shape[0]
                                }
                                
                                color = self._project_point_to_camera_color(pt, cam_neighbor, rgb_img)
                            except Exception:
                                color = None
                        else:
                            color = None
                            
                        if color is None:
                            color = torch.tensor([0.5, 0.5, 0.5], device=points_3d.device)
                        test_colors.append(color)
                        
                    new_colors_list.append(torch.stack(test_colors, dim=0))
                    logger.info(f"✨ [Cách A - Test] Added {new_pts_a_test.shape[0]} custom plane points from Test viewpoints.")
        
        # 3. CÁCH B: Bổ sung góc/cạnh dọc theo biên dạng 3D - bổ sung đúng 100 điểm
        if enable_custom_b:
            new_pts_b = self._edge_guided_3d_interpolation(points_3d, batch, edge_threshold=0.1, n_interpolate=3, target_points=100)
            if new_pts_b.shape[0] > 0:
                new_pts_list.append(new_pts_b)
                train_rgb = batch.rgb_gt[0] if batch.rgb_gt is not None else None
                b_colors = []
                for pt in new_pts_b:
                    color = self._project_point_to_camera_color(pt, train_camera, train_rgb) if (train_rgb is not None and 'train_camera' in locals()) else None
                    if color is None:
                        color = torch.tensor([0.5, 0.5, 0.5], device=points_3d.device)
                    b_colors.append(color)
                new_colors_list.append(torch.stack(b_colors, dim=0))
                logger.info(f"✨ [Cách B] Added {new_pts_b.shape[0]} custom edge-guided points.")
            
        if len(new_pts_list) > 0:
            added_points = torch.cat(new_pts_list, dim=0)
            added_colors = torch.cat(new_colors_list, dim=0)
            if added_points.shape[0] > 0:
                self.add_custom_points(added_points, added_colors)

    def add_custom_points(self, new_positions, new_colors=None):
        num_new = new_positions.shape[0]
        
        def safe_mean(tensor, default_val=0.0):
            m = torch.nanmean(tensor, dim=0, keepdim=True)
            return torch.where(torch.isnan(m) | torch.isinf(m), torch.tensor(default_val, device=tensor.device, dtype=tensor.dtype), m)
            
        mean_scale = safe_mean(self.model.scale, -2.0).repeat(num_new, 1)
        mean_rotation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.model.device).repeat(num_new, 1)
        mean_density = safe_mean(self.model.density, 0.1).repeat(num_new, 1)
        
        if self.model.feature_type.name == "SH":
            if new_colors is not None:
                new_albedo = (new_colors - 0.5) / 0.28209479177387814
                mean_albedo = safe_mean(self.model.features_albedo, 0.0).repeat(num_new, 1)
                valid_color_mask = ~torch.isnan(new_colors).any(dim=-1)
                mean_albedo[valid_color_mask] = new_albedo[valid_color_mask]
            else:
                mean_albedo = safe_mean(self.model.features_albedo, 0.0).repeat(num_new, 1)
            mean_specular = safe_mean(self.model.features_specular, 0.0).repeat(num_new, 1)
        else:
            if new_colors is not None:
                mean_features = safe_mean(self.model.features, 0.0).repeat(num_new, 1)
                valid_color_mask = ~torch.isnan(new_colors).any(dim=-1)
                padded_colors = torch.zeros((num_new, mean_features.shape[1]), device=self.model.device)
                padded_colors[:, :3] = new_colors
                mean_features[valid_color_mask] = padded_colors[valid_color_mask]
            else:
                mean_features = safe_mean(self.model.features, 0.0).repeat(num_new, 1)

        def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            if name == "positions":
                p_new = torch.cat([param, new_positions])
            elif name == "scale":
                p_new = torch.cat([param, mean_scale])
            elif name == "rotation":
                p_new = torch.cat([param, mean_rotation])
            elif name == "density":
                p_new = torch.cat([param, mean_density])
            elif name == "features_albedo":
                p_new = torch.cat([param, mean_albedo])
            elif name == "features_specular":
                p_new = torch.cat([param, mean_specular])
            elif name == "features":
                p_new = torch.cat([param, mean_features])
            else:
                p_new = param
            return torch.nn.Parameter(p_new, requires_grad=param.requires_grad)

        def update_optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            v_new = torch.zeros((num_new, *v.shape[1:]), device=v.device)
            return torch.cat([v, v_new])

        self._update_param_with_optimizer(update_param_fn, update_optimizer_fn)
        if hasattr(self, "reset_densification_buffers"):
            self.reset_densification_buffers()
