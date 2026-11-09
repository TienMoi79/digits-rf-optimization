# Hướng dẫn tự chạy tái lập cho checkpoint tuần 10

Phiên ngày 09/11/2026 do Codex thực hiện trong repo. Để tự trình bày với cô,
sinh viên cần tự chạy và đọc kết quả; việc trợ lý chạy thành công không xác
nhận sinh viên đã làm bước đó.

## 1 Chạy lại pipeline đã khóa

Mở terminal ở thư mục gốc dự án sau khi setup môi trường, chọn tên đầu ra mới:

```powershell
.\.venv\Scripts\python.exe -X utf8 -u run_experiment.py --config configs/experiment.json --output results/reproduction-student-01
```

Không dùng lại thư mục đã có kết quả. Không sửa config, seed, epsilon hoặc
budget trong lần tái lập này. Chương trình có 180 search trials; thời gian
phụ thuộc máy và tải nền. Nếu bị gián đoạn, giữ kết quả dang dở và chạy vào
thư mục mới; pipeline không hỗ trợ resume từ trial cuối.

## 2 Xem kết quả và kiểm tra

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.validation --output results/reproduction-student-01
```

Đọc `run_manifest.json` (completed, config, versions, runtime),
`selection.json` (ngưỡng/quyết định trước test), `final_metrics.csv`,
`trials.json` và `figures/` trong thư mục vừa chạy.

Đối chiếu toàn bộ trial và predictions với bộ gốc, chỉ in kết quả ra terminal:

```powershell
.\.venv\Scripts\python.exe -X utf8 -c "import json; from pathlib import Path; from src.week10 import compare_runs; print(json.dumps(compare_runs(Path('results'), Path('results/reproduction-student-01'))[0], indent=2))"
```

Lệnh so sánh sẽ báo sai khác ở config, split, tham số, fold score, model size,
quyết định hoặc predictions. Runtime/latency được giữ như số đo, không ép
khớp với phiên gốc. Nếu có sai khác, giữ log và gửi lại để phân tích; không
sửa số liệu cho giống bộ gốc.

`results/reproduction*` là thư mục tự chạy đã được gitignore để tránh upload
nhầm nhiều bộ kết quả lớn. Nếu muốn đưa lần chạy của sinh viên lên GitHub,
cần chọn bộ thực sự cần lưu và cập nhật nhật ký trước khi add/push. Phiên
tuần 10 do trợ lý chạy được giữ trong `results/week10_2026-11-09/reproduction/`
để làm bằng chứng thực nghiệm của checkpoint hiện tại.

## 3 Câu hỏi cần tự giải thích

1. Vì sao train có thể đạt 100% mà validation/test thấp hơn?
2. Vì sao tái lập cùng test không phải đánh giá độc lập trên dữ liệu mới?
3. Seed nào điều khiển split, CV, RF và tìm kiếm siêu tham số?
4. Khi CV score giống nhau nhưng runtime khác, có thể kết luận gì?
5. Vì sao selected vẫn là default dưới quy tắc epsilon đã định trước?
6. Trong 10 mẫu selected sai, Random sửa đúng bao nhiêu và sai thêm bao nhiêu?

Khi đã tự chạy, sinh viên chuyển tên thư mục kết quả và output lệnh kiểm tra
để cập nhật đúng trạng thái nhật ký. Không tự đánh dấu giảng viên duyệt hoặc
buổi bảo vệ đã hoàn thành chỉ dựa vào việc có file kết quả.
