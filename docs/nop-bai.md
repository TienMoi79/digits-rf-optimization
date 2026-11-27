# Mở và nộp đồ án

Báo cáo cập nhật ngày **27/11/2026**; rà soát kỹ thuật ngày 26/11. Theo thông tin sinh viên, hạn nộp
là **27/11/2026**. Việc nộp LMS, tự trình bày và cô duyệt do sinh viên xác nhận.

## Nộp và tìm tài liệu

- Repo public: [TienMoi79/digits-rf-optimization](https://github.com/TienMoi79/digits-rf-optimization).
  Lịch sử commit và các checkpoint được giữ nguyên.
- Nếu cần nộp file ZIP, dùng bản `BAN_NOP_DIGITS_RF_2026-11-27_GITHUB_PUBLIC.zip` đặt cạnh
  folder đồ án trên Desktop. ZIP được tạo từ commit đã chốt, có file SHA256
  đi kèm và không chứa `.venv`, `.git`, cache hay file tạm.
- Word theo mẫu sinh viên cung cấp: [bản có link GitHub Public 27/11](../report/Bao_cao_Do_an_Toi_uu_Random_Forest_Le_Quyet_Tien_2026-11-27_GitHub_Public.docx).
  Hai Word 03/11 và 09/11 giữ nguyên ở thư mục `report/` để đối chiếu.
- Checkpoint, phản hồi và các đợt kiểm chứng: [mục lục báo cáo](../report/README.md).

## Mở web nhanh, không cần cài thư viện ML

1. Giải nén ZIP; mở thư mục `digits-rf-optimization`.
2. Cài Python 3.12 nếu máy chưa có Python.
3. Nhấp đúp `web/start_web.bat`; giữ cửa sổ terminal đang chạy.
4. Trình duyệt mở `http://127.0.0.1:8765`. Ctrl+C trong terminal để dừng.

Hoặc mở terminal tại gốc và chạy:

```powershell
py -3 -X utf8 web/serve.py
```

Web đọc bảng/log và mô hình đã xuất. Bản online trên Sites đang riêng tư;
giảng viên có thể dùng bản local mà không cần tài khoản ChatGPT của sinh viên.

## Demo phần tối ưu và kiểm chứng bằng Python

Cài môi trường đầy đủ lần đầu:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Notebook
.\.venv\Scripts\python.exe -X utf8 demo_optimization.py replay --seed 11 --checkpoint 15
.\.venv\Scripts\python.exe -X utf8 demo.py --index 123
.\.venv\Scripts\python.exe -m src.validation
```

Replay đọc prefix log đã lưu, không huấn luyện mới. Xem [kịch bản demo TPE](tpe-va-demo-toi-uu.md).
Hai notebook đọc kết quả mặc định; không cần chạy lại 180/450 trials để mở đồ án.

## Tạo lại ZIP từ commit sạch

```powershell
python scripts/package_submission.py --output ../BAN_NOP_DIGITS_RF_MOI.zip
```

Script từ chối ghi đè file có sẵn, từ chối đóng gói khi chưa commit và kiểm
tra từng file ZIP với nội dung nguồn. ZIP phải đặt ngoài folder repo.
Không đổi số liệu, model hoặc làm lại lịch sử Git để chuẩn bị nộp.
