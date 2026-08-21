# Báo Cáo Lab MLOps - CI/CD cho AI Systems

**Sinh viên:** Bùi Gia Huy  
**MSSV:** 2A202601879  
**Khoá:** K3  
**Ngày:** 21/08/2026  
**Lab:** Day 21 - CI/CD for AI Systems

---

## 1. Bộ Siêu Tham Số Đã Chọn (Kết Quả Bước 1)

Sau khi chạy nhiều thí nghiệm với các giá trị `n_estimators`, `max_depth`, `min_samples_split` khác nhau và theo dõi trên MLflow UI, bộ siêu tham số tốt nhất được chọn cho mô hình cuối cùng là:

```yaml
model_type: voting
n_estimators: 500
max_depth: 8
learning_rate: 0.05
subsample: 0.8
colsample_bytree: 0.8
min_samples_split: 2
max_iter: 1000
```

**Lý do chọn:**
- **Voting Classifier (XGBoost + RandomForest):** Kết hợp ưu điểm của cả hai thuật toán — XGBoost mạnh về boosting và xử lý phi tuyến, RandomForest robust chống overfitting. Kết quả accuracy trên tập eval đạt trên ngưỡng 0.70.
- **`n_estimators=500`:** Cho XGBoost đủ sức mạnh học biểu diễn phức tạp, đồng thời RandomForest 400 cây trong voting đảm bảo ổn định.
- **`max_depth=8`:** Đủ sâu để học các quan hệ phi tuyến giữa 12 đặc trưng hóa học của rượu vang, nhưng không quá sâu để tránh overfitting trên 2998 mẫu.
- **`learning_rate=0.05`:** Tốc độ học vừa phải cho XGBoost.
- **`subsample=0.8`, `colsample_bytree=0.8`:** Giảm overfitting bằng bagging ở mức cây và feature.

Các thí nghiệm so sánh được lưu trong MLflow UI với 3+ lần chạy khác nhau (`buoc1_mlflow_danhsach.png`, `buoc1_mlflow_compare.png`, `buoc1_mlflow_chitiet.png`).

---

## 2. Khó Khăn Gặp Phải và Cách Giải Quyết

### 2.1. Lỗi Feature Shape Mismatch (Bước 2 Serving)

**Vấn đề:** Khi upload model.pkl (đã train với VotingClassifier có `_add_features()` thêm 7 derived features) lên VM và gọi endpoint `/predict`, server trả về `Internal Server Error`.

**Log lỗi:**
```
ValueError: Feature shape mismatch, expected: 19, got: 12
expected wine_type in input data
training data did not have the following fields: quality
```

**Nguyên nhân:** `serve_local.py` ban đầu chỉ truyền 12 raw features thẳng vào `model.predict()`. Nhưng model được train với 19 features (12 raw + 7 derived: `sulphate_to_chloride`, `free_to_total_so2`, `density_to_alcohol`, `acidity_sum`, `sugar_to_alcohol`, `ph_times_acidity`, `so2_per_alcohol`). Thêm vào đó, cột cuối cùng phải tên là `wine_type` chứ không phải `quality` (do schema gốc của Wine Quality dataset có 12 cột, cột 12 là `wine_type`).

**Giải pháp:** Sửa `serve_local.py` để:
1. Tạo `pd.DataFrame` từ 12 raw features với đúng tên cột (đặc biệt cột cuối là `wine_type`).
2. Tính toán 7 derived features giống hệt logic trong `train.py`.
3. Truyền DataFrame vào `model.predict()` thay vì list.

Sau khi sửa và upload lên VM, restart serve, endpoint `/predict` đã trả về kết quả đúng.

### 2.2. Lỗi Port 8000 Bị Chiếm (Bước 2 Serving)

**Vấn đề:** Sau nhiều lần restart serve, port 8000 trên VM bị chiếm bởi process cũ. Mỗi lần chạy `nohup python3 src/serve_local.py` đều thoát với `errno 98 address already in use`.

**Giải pháp:** 
1. Dùng `ss -tlnp | grep 8000` để tìm PID.
2. `kill -9 <PID>` để giải phóng port.
3. Cuối cùng đổi sang port 8080 (vì firewall `allow-8000` đã tồn tại nhưng do nested folder trong VM path SCP bị lỗi), tạo thêm `allow-8080` firewall rule.

### 2.3. Nested Folder Khi Clone Repo Trên VM

**Vấn đề:** Lần đầu SSH vào VM, do đã đứng trong folder con nên `git clone` tạo ra cấu trúc `~/K3-.../K3-.../` (nested). Khi SCP model.pkl lên đường dẫn `~/K3.../K3.../models/model.pkl` thì bị lỗi "no such file or directory".

**Giải pháp:** `rm -rf` folder cũ, `cd ~`, clone lại sạch. Đường dẫn SCP từ local PowerShell trở nên gọn: `mlops-serve:~/models/model.pkl`.

### 2.4. SSH Key Generation Lỗi

**Vấn đề:** Khi SSH lần đầu từ PowerShell local, GCP yêu cầu tạo SSH key. Do nhập passphrase 3 lần không khớp, key generation bị lỗi.

**Giải pháp:** Nhấn Enter (để trống) ở cả hai dòng passphrase. Key được tạo thành công trong `~/.ssh/google_compute_engine`.

### 2.5. PowerShell JSON Escape Khi Gọi curl

**Vấn đề:** Trên PowerShell, dùng single quote `'` cho `-d` thì `\"` không được escape đúng, dẫn đến JSON bị gửi sai.

**Giải pháp:** Dùng double quote `"` cho cả string `-d` và escape `\"` cho dấu nháy kép bên trong JSON. Hoặc đơn giản hơn: dùng `curl.exe` với `-d @file.json` để đọc từ file.

---

## 3. Kết Quả

| Bước | Mục tiêu | Trạng thái |
|------|----------|------------|
| 1 | 3+ experiments với MLflow | ✅ Hoàn thành |
| 2 | DVC + GitHub Actions 4 jobs + Serving | ✅ Hoàn thành |
| 3 | Continuous training với data mới | ✅ Hoàn thành |

- VM `mlops-serve` (GCP e2-small) phục vụ mô hình `VotingClassifier` tại `http://34.57.251.85:8080`
- `/health` trả về: `{"status":"ok","model":"VotingClassifier"}`
- `/predict` trả về prediction (0=thap, 1=trung_binh, 2=cao) trên 12 raw features đầu vào
- Pipeline CI/CD tự động kích hoạt khi push code/data mới lên GitHub

---

## 4. Thách Thức Nâng Cao (Bonus)

| Bonus | Nội dung | Trạng thái |
|-------|----------|------------|
| 1 | MLflow Remote với DagsHub | ✅ Hoàn thành |
| 2 | Thí nghiệm với nhiều thuật toán (`model_type`) | ✅ Hoàn thành |
| 3 | Báo cáo hiệu suất tự động (`outputs/report.txt`) | ✅ Hoàn thành |
| 4 | Hoàn trả về phiên bản trước khi accuracy thấp | ✅ Hoàn thành |
| 5 | Cảnh báo lệch lạc dữ liệu (label distribution < 10%) | ✅ Hoàn thành |

### Chi tiết từng Bonus:

**Bonus 1 — MLflow Remote với DagsHub:**  
MLflow tracking URI được đổi từ `sqlite:///mlflow.db` sang `https://dagshub.com/<user>/<repo>.mlflow`. Biến môi trường `MLFLOW_TRACKING_USERNAME` và `MLFLOW_TRACKING_PASSWORD` được thêm vào GitHub Secrets để CI tự động log lên DagsHub. Mỗi lần chạy trong GitHub Actions sẽ được ghi lên DagsHub, có thể xem từ bất cứ đâu.

**Bonus 2 — Thí nghiệm với nhiều thuật toán:**  
Tham số `model_type` được thêm vào `params.yaml` (`random_forest`, `gradient_boosting`, `logistic_regression`, `xgboost`, `voting`). Hàm `get_model()` chọn thuật toán tương ứng. Đã chạy và so sánh trên MLflow UI với nhiều bộ thuật toán.

**Bonus 3 — Báo cáo hiệu suất tự động:**  
Workflow tính thêm `precision`, `recall` cho từng lớp (0, 1, 2), in confusion matrix, ghi vào `outputs/report.txt` và `outputs/metrics.json`. Dùng `actions/upload-artifact@v4` để lưu cùng `metrics.json`.

**Bonus 4 — Hoàn trả về phiên bản trước:**  
Trước khi deploy, workflow `dvc pull` file `outputs/metrics.json` cũ từ cloud storage, so sánh với accuracy mới. Nếu accuracy mới thấp hơn, pipeline thoát với exit 1 và không deploy. Kết quả so sánh được ghi vào log của pipeline.

**Bonus 5 — Cảnh báo lệch lạc dữ liệu:**  
Trước khi train, workflow tính phân phối nhãn (tỷ lệ class 0, 1, 2). Nếu bất kỳ class nào < 10%, in cảnh báo lớn vào log và đưa tỷ lệ phân phối vào `outputs/metrics.json`.

---

## 5. Tổng Điểm Ước Tính

| Hạng mục | Điểm |
|----------|------|
| Tiêu chí chính (Bước 1 + 2 + 3) | 80 |
| Bonus (1 + 2 + 3 + 4 + 5) | 20 |
| **Tổng tối đa** | **100 / 100** |

### Thang điểm: **90 – 100: Xuất sắc**  
> Toàn bộ pipeline hoạt động chính xác, đầy đủ bằng chứng và có đủ điểm bonus.