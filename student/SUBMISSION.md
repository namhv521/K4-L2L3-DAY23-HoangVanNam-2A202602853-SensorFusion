# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

## Thông tin học viên

- Họ tên: Hoàng Văn Nam
- MSSV: 2A202602853
- Email: namhv.nitc@gmail.com (theo cấu hình Git của workspace)
- Link repo (fork): https://github.com/namhv521/K4-L2L3-DAY23-HoangVanNam-2A202602853-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): lấy hash 40 ký tự sau commit cuối; URL bài nộp VLearn dùng `/tree/<hash>` để khóa đúng phiên bản. Không nhúng hash của chính commit chứa báo cáo vào file vì làm thay đổi hash đó.

## Tóm tắt kết quả

- `fusion_mode`: **compare**; `frames`: **[0, 198]** (199 frame/mode); `seed`: **0**.
- `segment`: `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`.
- `detection.precision`: **0.9700934579439252**; `detection.recall`: **0.7004048582995951**.
- `detection.tp/fp/fn`: **519 / 16 / 222** (giống nhau giữa hai mode, cùng detector có sẵn).

| Chỉ số tracking | LiDAR | Fused (LiDAR + camera) |
|---|---:|---:|
| `rmse` | 0.15032268781360134 | 0.1358667883353908 |
| `matches` | 502 | 502 |
| `sum_sq_err` | 11.343649056695735 | 9.26681165463209 |
| `ghost_track_frames` | 0 | 0 |
| `missed_gt_frames` | 239 | 239 |
| `mean_confirmed_tracks` | 2.522613065326633 | 2.522613065326633 |
| `precision_track` (tính từ matches/ghost) | 1.0 | 1.0 |
| `coverage` (matches/detection.tp) | 0.9672447013487476 | 0.9672447013487476 |

Số trên đọc trực tiếp từ [`metrics.json`](artifacts/metrics.json), không chỉnh sửa artifacts. Đã đối chiếu toàn bộ với [`grade_run.log`](artifacts/grade_run.log): **398 record**, mỗi `(mode, frame)` đúng một lần; hai log riêng có **199 record/mode**. Mỗi record thỏa `matches + ghosts == confirmed`, `matches + misses == valid_gt`; tổng detection, matches, sum_sq_err, ghost/miss và mean_confirmed_tracks khớp hoàn toàn với metrics. Hai file metrics riêng cũng khớp các log riêng và phần tracking tương ứng trong metrics compare.

**Diễn giải hai mode.** Camera update giảm RMSE 3D từ 0.150323 xuống 0.135867 m, giảm 9.62%; `rmse_fused - rmse_lidar = -0.014455899 m`. Số cặp ghép đều 502, ghost đều 0, miss đều 239, mean confirmed đều 2.522613. Trên segment này, giảm RMSE đi kèm số matches/ghost/miss không thay đổi. Tuy nhiên, metrics tổng không chứng minh hai mode giữ nguyên cùng danh tính track ở mỗi frame. Camera không thay đổi trực tiếp score/init/delete, và không thể tạo track mới cho xe detector LiDAR bỏ sót.

Cả hai RMSE đều <= 0.45 m; precision_track=1.000000 >= 0.75; coverage=0.967245 >= 0.70; chênh lệch RMSE <= 0.05 m. Các điều kiện chất lượng tự động của rubric đạt trên lần chạy này; điểm thủ công và vấn đáp vẫn do giảng viên đánh giá.

Ví dụ record cuối frame 198 (đối chiếu trực tiếp log):

- `lidar`, frame 198: confirmed=3, matches=3, ghosts=0, valid_gt=9, misses=6, sum_sq_err=0.06297798579477262.
- `fused`, frame 198: confirmed=3, matches=3, ghosts=0, valid_gt=9, misses=6, sum_sq_err=0.07530478180761385.

`rmse = sqrt(sum_sq_err/matches)` tính trên vị trí **3D** của confirmed tracks được ghép một-một với GT trong cổng **XY 2 m**; cần đọc cùng matches/ghost/miss. Camera trong lab dùng tâm hộp 2D **ground-truth FRONT có nhiễu theo seed**, không chạy camera detector; kết quả là thí nghiệm fusion với đo mô phỏng từ nhãn, không chứng minh chất lượng một hệ perception độc lập với GT.


## Tái lập và tự kiểm tra

Bài làm giữ nguyên Part A–D, platform và toàn bộ test gốc. Hoàn thiện 17 hàm Part E–H trong `workspace/`. Không huấn luyện lại detector, không thay đổi tham số tracking và không sửa tay artifacts.

Trên Windows, chạy từ root repo bằng môi trường `.venv`:

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = 'platform;student;platform/third_party/waymo_reader'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
./.venv/Scripts/python.exe -m pytest student/tests -q
./.venv/Scripts/python.exe -m fusion_lab.scripts.run_lab --config student/config/paths.yaml --fusion compare --seed 0
./.venv/Scripts/python.exe tools/check_submission.py
```

Kết quả test gốc: **128 passed**, không còn failed/xfailed. Các warning chủ yếu đến từ `np.matrix` mà API lab sử dụng và ca thử tọa độ không hữu hạn. `PYTHONUTF8=1` xử lý lỗi encoding Windows ở fixture ghi tiếng Việt; không sửa test để né lỗi. Môi trường dùng Python 3.12, NumPy 2.3.5, SciPy 1.17.1, PyTorch 2.8.0+cu128 (chạy CPU), easydict 1.13 và protobuf 6.33.6. Hai dependency cuối được cài riêng vào `.venv`, không thay thế package của Python chung.

Cấu hình local `student/config/paths.yaml`: `frame_start: 0`, `frame_end: 198`, segment mặc định, `waymo_dir: ../data/Waymo`, `weights_dir: ../data/weights`. File cấu hình, dataset và weights không đưa lên Git.

## Giải thích kỹ thuật — sáu câu Parts E–H

### 1. Khác biệt đo LiDAR 3D và camera 2D trong EKF (`z`, `R`)?

State là `x = [px, py, pz, vx, vy, vz]^T`, kích thước 6×1. LiDAR đo vị trí `z = [xs, ys, zs]^T` bằng mét, `R` 3×3 với phương sai từng trục `0.1² = 0.01 m²`. Mô hình đo LiDAR tuyến tính sau phép đổi hệ tọa độ; `H` 3×6, các cột vận tốc bằng 0.

Camera đo pixel `z = [u, v]^T`, `R` 2×2 bằng `diag(5², 5²) = diag(25, 25)` pixel² theo tham số lab. Đổi vị trí vehicle sang sensor bằng `p_s = R_sv p + t_sv`, rồi chiếu `u = c_i - f_i*y_s/x_s`, `v = c_j - f_j*z_s/x_s`. Phép chia theo độ sâu làm mô hình phi tuyến; platform tính Jacobian `H` 2×6 theo chain rule. Chỉ chiếu khi tọa độ hữu hạn và `x_s > 1e-6`.

Cả hai dùng `gamma = z - h(x)`, `S = H P H^T + R`, `K = P H^T S^-1` và correction EKF. Code dùng `np.linalg.solve` thay vì tạo nghịch đảo trực tiếp. Dẫn chứng: [`kalman.ekf_update`](workspace/kalman.py), [`camera_measurement_prediction`, `build_camera_measurement`](workspace/camera_fusion.py), [`Sensor.get_H`, `Measurement`](../platform/fusion_lab/tracking/sensors.py). Nhiễu pixel được sinh từ nhãn trong runtime có độ lệch chuẩn 0.5 pixel, khác sigma=5 pixel dùng để xây `R`; đây là hai thiết lập riêng của platform.

### 2. Vì sao cần gating Mahalanobis trước khi gán?

Mahalanobis bình phương `d² = gamma^T S^-1 gamma` đo sai khác đã chuẩn hóa bởi bất định của track và cảm biến, phù hợp hơn khoảng cách Euclidean khi các trục có phương sai khác nhau. Cổng `chi2.ppf(0.995, dim_meas)` dùng 3 bậc tự do cho LiDAR, 2 cho camera. Cặp không hợp lệ hoặc vượt cổng nhận cost `inf`; greedy chỉ chọn cặp hữu hạn nhỏ nhất, rồi loại đúng hàng/cột để mỗi track và measurement được dùng tối đa một lần. Greedy không bảo đảm tối ưu toàn cục nhưng đúng thuật toán đề yêu cầu.

Kiểm tra FOV trước Mahalanobis để không chiếu track sau camera hoặc có độ sâu 0. Khi ma trận rỗng/toàn `inf`, trả `nan` và giữ nguyên danh sách chưa ghép để lifecycle xử lý. Dẫn chứng: [`mahalanobis_distance`, `chi2_gate`, `association_cost_matrix`, `pick_next_pair`](workspace/association.py); test gốc `test_out_of_fov_gating_precedes_projection` và `test_greedy_assignment_keeps_remaining_objects` trong [`test_tracking_regressions.py`](tests/test_tracking_regressions.py).

### 3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log.

Đây là **track-then-fuse**. Detector tạo hộp xe từng frame; tracker duy trì ID, vị trí và vận tốc qua thời gian. Mỗi mode có một danh sách track: mỗi frame predict đúng một lần, association/update LiDAR và lifecycle, sau đó mode fused association/update thêm camera trên cùng trạng thái. Không predict lần nữa giữa hai cảm biến và không gộp raw point cloud với ảnh trước detector.

Thứ tự thực thi kiểm tra được trong [`run_lab.run`](../platform/fusion_lab/scripts/run_lab.py): `KF.predict` → `associate_and_update(..., lidar_sensor)` → `associate_and_update(..., camera_sensor)`. [`grade_run.log`](artifacts/grade_run.log) ghi `mode` và `frame` cho hai mode, một record/mode/frame; log chấm chỉ ghi counts/error sau xử lý, **không ghi riêng từng bước predict/update** nên bản thân log không chứng minh được thứ tự nội bộ. Đối chiếu code runtime với log, không suy diễn từ record cuối frame.

### 4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?

Sai extrinsic/intrinsic gây lệch `h(x)` và Jacobian, làm residual pixel có thiên lệch có hệ thống thay vì dao động quanh 0; độ lệch có thể phụ thuộc khoảng cách hoặc vị trí trong ảnh. Mahalanobis tăng, cổng từ chối nhiều cặp camera hoặc gán sai; update sai có thể kéo track lệch và làm RMSE fused tăng so với LiDAR. Covariance `R` đang mô tả nhiễu đo ngẫu nhiên, không tự sửa một lỗi calibration có bias.

Cách kiểm tra là ghi thêm residual trước update, xem mean/phân bố theo pixel/depth, kiểm tra transform và reprojection, rồi so sánh trên cùng frame và seed. [`innovation`](workspace/kalman.py) và [`camera_measurement_prediction`](workspace/camera_fusion.py) là nơi cần kiểm tra. Log chuẩn hiện không có residual hoặc thử nghiệm drift, nên báo cáo **không khẳng định đã đo calibration drift**.

### 5. Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng?

Khi `meas_list=[]`, không thể suy ra loại cảm biến từ measurement đầu tiên. Sensor tường minh cho phép kết thúc đúng lifecycle của lượt LiDAR: track chưa ghép và trong FOV phải nhận miss, rồi xét xóa dù detector không có đo. Lượt camera rỗng không được giảm score, xóa hoặc tạo track; thiếu đo camera không có nghĩa xe biến mất.

[`associate_and_update`](workspace/association.py) luôn gọi `manager.manage_tracks(tracks, measurements, sensor)`. [`TrackManager.handle_updated_track`, `manage_tracks`](../platform/fusion_lab/tracking/manager.py) chỉ dùng lượt LiDAR để tăng/giảm score, xét deletion và tạo track từ đo chưa ghép. Camera hit chỉ correction EKF; camera miss cũng không đổi lifecycle. Test gốc: `test_empty_lidar_frame_scores_then_deletes_exhausted_track` và `test_camera_pass_never_deletes_or_spawns` trong [`test_tracking_regressions.py`](tests/test_tracking_regressions.py).

### 6. Điều kiện xác nhận, giữ confirmed sau miss và xóa track?

Khởi tạo từ đo LiDAR: đổi `z` về vehicle frame, đặt vận tốc bằng 0; covariance vị trí `R_vs R_meas R_vs^T`, covariance vận tốc `diag(50², 50², 5²)`. State `initialized`, score `1/6`. Mỗi LiDAR hit cộng `1/6`, chặn trên tại 1; miss trong FOV trừ `1/6`. Hit chưa đủ xác nhận chuyển sang `tentative`; **score > 0.8** chuyển `confirmed` (dấu `>` là nghiêm ngặt). Track confirmed giữ state sau miss, miễn chưa bị xóa.

Xóa nếu **bất kỳ** điều kiện nào đúng: `Pxx > 9` hoặc `Pyy > 9` m²; confirmed có score **< 0.6**; initialized/tentative có score **<= 0**. Tại đúng ngưỡng score=0.6 hoặc variance=9 chưa xóa. Track ngoài LiDAR FOV không bị trừ score vì miss, nhưng lượt LiDAR vẫn xét giới hạn covariance. Camera không gọi quyết định tồn tại. Dẫn chứng: [`init_track_state_from_meas`, `update_track_score`, `should_delete_track`](workspace/track_management.py), [`tracking_params.py`](../platform/fusion_lab/tracking_params.py), test `test_deletion_thresholds`.

## Bonus

Không. Không nhận điểm bonus khi chưa có file và bằng chứng đúng yêu cầu.

## Khai báo sử dụng AI

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): OpenAI Codex/ChatGPT, có agent review code theo workflow Superpowers.
- Dùng cho phần nào (hàm, câu hỏi, debug): Đọc đề và hợp đồng platform; hoàn thiện hàm E–H; chẩn đoán dependency/encoding môi trường Windows; tổng hợp metrics/log 
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): Codex đã chạy 128 test gốc pass,kiểm tra thêm 32 tổ hợp matrix/ndarray cho EKF trong bước review, đối chiếu công thức với tham số lab. Kết quả Waymo và kiểm tra log được cập nhật sau khi lượt compare hoàn tất

## Checklist nộp

- [x] Part E–H đã implement; 128 test gốc passed, không còn failed/xfailed.
- [x] Giữ nguyên Part A–D, platform và test gốc.
- [x] Chạy chấm `--fusion compare --seed 0`, frames 0–198 hoàn tất sau sửa code cuối; chạy lại ngày 10/10/2026, exit code 0.
- [x] Commit đủ sáu artifacts nguyên bản do runtime sinh; lần chạy lại có nội dung khớp artifacts đã commit sau chuẩn hóa xuống dòng của Git trên Windows. Không chỉnh sửa tay artifacts.
- [x] Đã điền thông tin, sáu câu giải thích và khai báo AI.
- [x] Không commit dữ liệu, weights, `paths.yaml`, `.env`, API key hay file nén.
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`.
- [x] Push fork cá nhân, kiểm tra commit từ xa bằng `git ls-remote origin refs/heads/main`.
- [x] Nộp URL fork và hash 40 ký tự trên VLearn, mở lại để xác nhận.

Lần cập nhật VLearn ngày 10/10/2026 bị chuyển về trang đăng nhập với `reason=session-expired` khi xác nhận lưu URL commit. Chưa xác minh được bài nộp mới đã lưu; cần đăng nhập lại và nộp URL `/tree/<hash>` của commit cuối đã push. Bài nộp cũ có mã `BN-261010-40253`, nhưng trạng thái cũ không chứng minh hash mới đã được lưu.
