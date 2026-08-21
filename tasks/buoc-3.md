# Bước 3 - Huấn Luyện Liên Tục Khi Có Dữ Liệu Mới

Mục tiêu: Mô phỏng vai trò của một kỹ sư dữ liệu bổ sung thêm dữ liệu mới. Chỉ cần một lần `git push` là pipeline tự động huấn luyện lại và triển khai lại mà không cần bất kỳ thao tác thủ công nào.


---

## 3.1 Tìm Hiểu Quy Trình Trước Khi Bắt Đầu

Trước khi thực hiện, hãy đọc lại workflow trigger trong `.github/workflows/mlops.yml`:

```yaml
on:
  push:
    branches: [main]
    paths:
      - 'data/**.dvc'    # <- Pipeline được kích hoạt khi file .dvc thay đổi
      - 'src/**.py'
      - 'params.yaml'
```

Đây là chốt mấu chốt của Bước 3: khi bạn thay đổi nội dung file CSV và cập nhật file `.dvc` tương ứng, GitHub Actions sẽ tự động kích hoạt toàn bộ pipeline.

---

## 3.2 Thêm Dữ Liệu Mới

Script `add_new_data.py` đã được cung cấp sẵn. Script này ghép `train_phase2.csv` (2998 mẫu mới) vào `train_phase1.csv`:

```bash
python add_new_data.py
```

Kết quả mong đợi:

```
Cập nhật dữ liệu: 2998 -> 5996 mẫu
```

Xác nhận kích thước dữ liệu mới:

```bash
wc -l data/train_phase1.csv
# Kết quả mong đợi: 5997 (5996 dòng dữ liệu + 1 dòng tiêu đề)
```

---

## 3.3 Phiên Bản Hóa Dữ Liệu Mới Và Kích Hoạt Pipeline

Đây là bước quan trọng nhất. Thực hiện theo đúng thứ tự:

```bash
# 1. Thông báo cho DVC rằng file dữ liệu đã thay đổi
dvc add data/train_phase1.csv

# 2. Commit file con trỏ DVC đã cập nhật vào git
#    Lưu ý: commit file .dvc, KHÔNG phải file CSV
git add data/train_phase1.csv.dvc
git commit -m "data: bổ sung 2998 mẫu dữ liệu mới (train_phase2)"

# 3. Đẩy dữ liệu mới lên cloud storage trước
#    Bước này đảm bảo CI runner có thể pull dữ liệu mới khi pipeline bắt đầu
dvc push

# 4. Đẩy git commit lên GitHub - thao tác này kích hoạt GitHub Actions
git push origin main
```

Tại sao phải `dvc push` trước `git push`? Nếu git push được thực hiện trước, GitHub Actions sẽ bắt đầu và cố gắng `dvc pull` dữ liệu mới khi dữ liệu đó chưa có trên cloud storage, dẫn đến lỗi.

---

## 3.4 Theo Dõi Pipeline Phản Ứng

Vào tab **Actions** trên repo GitHub. Trong vài giây sau khi push, pipeline sẽ tự động bắt đầu.

Xác nhận commit message trong pipeline khớp với commit vừa tạo:

```
data: bổ sung 2998 mẫu dữ liệu mới (train_phase2)
```

Điều này chứng minh pipeline được kích hoạt bởi commit dữ liệu, không phải commit code.

Theo dõi từng job:

1. **Unit Test** - unit tests chạy trên code hiện tại (không thay đổi so với Bước 2).
2. **Train** - CI runner pull tập dữ liệu mới (5996 mẫu) từ cloud storage, huấn luyện lại mô hình, upload `model.pkl` mới lên cloud storage.
3. **Eval** - kiểm tra accuracy >= 0.70, nếu không đạt thì pipeline dừng tại đây.
4. **Deploy** - nếu eval gate qua, service trên VM được restart với mô hình mới.

---

## 3.5 Xác Nhận Mô Hình Mới Đã Được Triển Khai

Sau khi pipeline hoàn thành:

```bash
VM_IP=<YOUR_VM_IP>

# Kiểm tra server đang chạy
curl http://$VM_IP:8000/health

# Gửi yêu cầu dự đoán
curl -X POST http://$VM_IP:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 0]}'
```

Mô hình mới huấn luyện trên 5996 mẫu sẽ được phục vụ. Không có bất kỳ thao tác thủ công nào cần thiết.

---

## 3.6 So Sánh Kết Quả

Tải file `outputs/metrics.json` từ artifacts của hai lần chạy để so sánh:

| Chỉ số | Bước 2 (2998 mẫu) | Bước 3 (5996 mẫu) |
|---|---|---|
| accuracy | 0.6620 | 0.7060 |
| f1_score | 0.6613 | 0.7024 |

Điền vào bảng trên dựa trên kết quả thực tế của bạn. Nếu mô hình mới có accuracy cao hơn, điều đó chứng tỏ thêm dữ liệu làm tăng hiệu quả mô hình.

---

## Xử Lý Sự Cố

**Pipeline Bước 3 không được kích hoạt**

Xác nhận bạn đã commit file `.dvc`, không phải file CSV:

```bash
git log --name-only -1
```

Kết quả mong đợi:

```
data/train_phase1.csv.dvc
```

Nếu thấy `data/train_phase1.csv`, bạn đã commit nhầm file. Thêm file CSV vào `.gitignore` và commit lại.

**Lỗi `dvc push` - file quá lớn**

Không có vấn đề. Các cloud provider đều hỗ trợ file có kích thước lớn trong gói miễn phí/trial. Kiểm tra lại xác thực:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=sa-key.json
dvc push
```

**Accuracy ở Bước 3 thấp hơn Bước 2**

Đây là tình huống bình thường có thể xảy ra do dữ liệu mới thêm nhiều nhiễu hơn. Model vẫn được triển khai nếu accuracy vẫn >= 0.70. Nếu accuracy dưới ngưỡng, thì bước deploy sẽ bị chặn - đây là hành vi mong muốn của eval gate.

---

## Tóm Tắt Những Gì Bạn Đã Xây Dựng

Sau khi hoàn thành cả ba bước, bạn đã xây dựng một hệ thống MLOps có khả năng hoạt động trong thực tế:

```
Bước 1 - Thực nghiệm cục bộ
  MLflow ghi lại mọi thí nghiệm trên máy cá nhân.
  So sánh nhiều bộ siêu tham số và chọn bộ tốt nhất.
  Quy trình phát triển có tổ chức, có thể tái lại và kiểm chứng.

Bước 2 - CI/CD tự động
  Push code -> GitHub Actions huấn luyện trên môi trường sạch.
  DVC quản lý phiên bản dữ liệu, đảm bảo khả năng tái tạo.
  Eval gate: chỉ các mô hình đạt ngưỡng chất lượng mới được triển khai.
  FastAPI trên Cloud VM phục vụ dự đoán qua REST API.

Bước 3 - Huấn luyện liên tục
  Thêm dữ liệu mới -> cập nhật DVC -> git push.
  Toàn bộ pipeline Bước 2 chạy lại hoàn toàn tự động.
  VM luôn phục vụ mô hình mới nhất đã qua kiểm tra chất lượng.
```

Đây là vòng phản hồi cơ bản của một hệ thống MLOps trong thực tế sản xuất: dữ liệu mới -> huấn luyện -> kiểm tra chất lượng -> triển khai tự động.

---

## Kết Quả Cần Đạt - Bước 3

- Chụp màn hình một lần chạy GitHub Actions được kích hoạt bởi commit dữ liệu.
  Xác nhận: commit message hiển thị trong tên của lần chạy Actions là commit dữ liệu của bạn.
- Cả bốn jobs (Unit Test, Train, Eval, Deploy) đều hoàn thành thành công.
- So sánh accuracy giữa Bước 2 và Bước 3 đã được điền vào bảng ở mục 3.6.

---

Quay lại: [Bước 2 - Pipeline CI/CD tự động](buoc-2.md)
