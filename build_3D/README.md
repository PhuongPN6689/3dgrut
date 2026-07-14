# 3D Point Map Reconstruction Pipeline (SuperPoint + LightGlue)

Module này thực hiện dựng bản đồ 3D Point Map từ tập dữ liệu training của camera sử dụng mô hình học sâu **SuperPoint** để phát hiện các đặc trưng và **LightGlue** để ghép cặp đặc trưng, kết hợp với phép tối ưu hóa Bundle Adjustment phi tuyến.

## Cấu trúc thư mục

```text
build_3D/
    ├── README.md               : Tài liệu hướng dẫn sử dụng này
    ├── config.yaml             : Cấu hình tham số cho toàn bộ pipeline
    ├── build_map.py            : File chạy chính điều phối 13 bước trong pipeline
    ├── features/
    │   ├── extract_superpoint.py : Trích xuất đặc trưng SuperPoint & Màu sắc điểm ảnh
    │   └── feature_cache.py    : Đọc/ghi cache đặc trưng cục bộ
    ├── matching/
    │   ├── lightglue_matcher.py  : Khớp đặc trưng bằng LightGlue + RANSAC lọc nhiễu
    │   ├── graph_builder.py    : Tính toán khoảng cách pose & xây dựng neighbor graph
    │   └── union_find.py       : Cấu trúc Union-Find quản lý connected components
    ├── tracking/
    │   └── track_builder.py    : Gom cụm các điểm tương đồng thành Global Tracks
    ├── mapping/
    │   ├── triangulation.py    : Phép Triangulate đa góc chiếu DLT
    │   ├── bundle_adjustment.py: Tối ưu Least-Squares (Huber) tinh chỉnh tọa độ 3D
    │   └── filtering.py        : Lọc nhiễu theo baseline angle, track length, error
    ├── geometry/
    │   ├── camera.py           : Mô hình Camera & khử méo ảnh (undistortion)
    │   ├── pose.py             : Tính toán chuyển đổi pose, góc quay & khoảng cách pose
    │   └── projection.py       : Triangulation DLT & tính sai số reprojection
    ├── io/
    │   ├── pose_reader.py      : Đọc file cameras.bin & images.bin từ COLMAP
    │   └── feature_io.py       : Đọc/ghi cache trung gian (.pkl)
    └── viewer/
        └── viewer3d.py         : Hiển thị 3D Point Cloud & Camera trajectory (Open3D)
```

---

## Cài đặt thư viện

Yêu cầu Python 3.9+. Cài đặt các thư viện cần thiết bằng lệnh sau:

```bash
pip install torch torchvision kornia scipy omegaconf pycolmap
pip install git+https://github.com/cvg/LightGlue.git
```

*Lưu ý khi hết bộ nhớ ổ cứng C:* Nếu ổ C bị đầy, bạn có thể cài đặt thư viện vào ổ D (ví dụ thư mục dự án):
```bash
pip install --target d:\Projects\vtit\ai_race\DroneSplat\.pip_packages open3d
```

---

## Cấu hình tham số (`config.yaml`)

Tất cả các tham số có thể chỉnh sửa trực tiếp trong file `build_3D/config.yaml`:
- **dataset**: Đường dẫn tới scene dữ liệu (mặc định: `data/public_set/hcm0031`) và cấu hình thiết bị chạy (`cuda` hoặc `cpu`).
- **superpoint**: Giới hạn số keypoint tối đa (`max_keypoints: 4096`).
- **neighbor**: Trọng số khoảng cách camera theo vị trí và góc nhìn.
- **graph**: Số liên kết bổ sung giữa các cụm rời rạc (`extra_links: 2`).
- **matching**: Ngưỡng RANSAC lọc nhiễu khớp ảnh.
- **triangulation & filter**: Ngưỡng lọc reprojection error và baseline angle để đảm bảo độ tin cậy của điểm 3D.

---

## Hướng dẫn chạy Pipeline & Cơ chế Tối ưu hóa Cache

Để chạy toàn bộ pipeline dựng bản đồ 3D và hiển thị viewer:

```bash
python build_3D/build_map.py
```

### Cơ chế Tối ưu hóa Cache nâng cao:
1. **Kiểm tra cache cuối cùng:** Khi chạy lại chương trình, nếu file kết quả điểm 3D đã tối ưu (`cache/points3d/points3d.pkl`) đã tồn tại, chương trình sẽ **tự động bỏ qua** tất cả các bước tính toán nặng (trích xuất, matching, triangulation, BA, filtering) và tải trực tiếp dữ liệu để mở Viewer. Lần tải lại sẽ hoàn tất **tức thì (dưới 1 giây)**.
2. **Resume giữa chừng:** Nếu quá trình chạy lần đầu bị gián đoạn, các cache thành phần (`cache/features/`, `cache/matches/`, v.v.) vẫn được giữ nguyên để lần chạy tiếp theo tiếp tục thực hiện các bước còn thiếu.

### Các tham số tùy chọn:
- `--force`: Bỏ qua cơ chế kiểm tra cache cuối cùng, ép buộc chạy lại toàn bộ pipeline từ đầu (ví dụ khi bạn thay đổi cấu hình thuật toán).
- `--dataset <path>`: Thay đổi đường dẫn scene cần dựng bản đồ (ví dụ: `--dataset data/public_set/hcm0034`).
- `--config <path>`: Sử dụng file cấu hình yaml khác.
- `--no_viewer`: Chạy dựng bản đồ và lưu kết quả mà không mở cửa sổ hiển thị 3D (phù hợp khi chạy trên server không màn hình).

---

## Các phím điều khiển trong 3D Viewer (Open3D)

Cửa sổ hiển thị Open3D hỗ trợ tương tác trực quan bằng bàn phím:
*   `C`: Bật/tắt hiển thị camera frustum, quỹ đạo bay (trajectory), và **hướng chụp của camera** (hiển thị dưới dạng tia laser màu vàng chỉ hướng nhìn chính diện của camera).
*   `P`: Bật/tắt hiển thị đám mây điểm (point cloud).
*   `[` / `]`: Giảm / tăng kích thước điểm 3D (point size).
*   `O`: Tô màu các điểm 3D theo **Màu sắc thật của ảnh (True Color)** (Mặc định).
*   `B`: Tô màu mặc định (Slate Blue).
*   `T`: Tô màu điểm 3D theo **độ dài của Track** (Đỏ là track dài/nhiều camera quan sát, Xanh là track ngắn).
*   `E`: Tô màu điểm 3D theo **sai số reprojection** (Xanh lá là sai số thấp/chuẩn xác cao, Đỏ là sai số cao).
*   `Q` hoặc `ESC`: Đóng cửa sổ hiển thị.
