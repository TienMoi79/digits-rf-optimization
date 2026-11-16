# Digits Lab — giao diện web

Giao diện đầu tiên để sinh viên xem và góp ý. Đọc số liệu thực nghiệm đã khóa;
không huấn luyện lại hoặc chọn model bằng test.

## Mở trên máy

Nhấp đúp **`start_web.bat`** trong thư mục này. Hoặc từ gốc đồ án:

```powershell
.\.venv\Scripts\python.exe -X utf8 web/serve.py
```

Mở [http://127.0.0.1:8765](http://127.0.0.1:8765). Giữ terminal đang chạy;
Ctrl+C để dừng. Có thể chọn cổng khác bằng `--port 8766`. Không cần cài thêm
thư viện, Node.js hay tài khoản để chạy bản trên máy.

Bản Sites: [Digits Lab](https://digits-rf-lab-tienmoi79.flowypheasant16.chatgpt.site).
Site giữ quyền truy cập riêng của chủ sở hữu; bản trên máy vẫn chạy độc lập.

## Các màn hình

1. **Tổng quan:** mục tiêu tối ưu, 180 trials, CV/test/kích thước và quy tắc epsilon.
2. **So sánh mô hình:** RF mặc định/Random/TPE/selected, train–validation–test,
   tải CSV và đủ 10 trường hợp selected sai từ predictions đã lưu.
3. **Quá trình tìm kiếm:** chọn seed 11/22/33, thanh trial và phát lại log;
   chỉ dùng prefix đã hoàn thành, không chạy HPO.
4. **Train & Validation:** đúng hai đường accuracy hoặc loss, đủ 100 số cây;
   loss có thể phóng vùng 20–100 cây. Không làm mượt.
5. **Thử nhận dạng:** 30 mẫu thuộc development hoặc vẽ trên lưới 8×8,
   dự đoán bằng 100 cây thật và xem ba điểm số cao nhất.

## Dữ liệu và kiểm chứng

- [scripts/export_web_data.py](../scripts/export_web_data.py) xuất dữ liệu từ logs,
  bảng và `results/models/selected.joblib`; không fit model.
- [dist/forest.js](dist/forest.js) duyệt đúng cây đã xuất, ép input float32 như
  sklearn và lấy trung bình xác suất. Browser không cần Python backend.
- Số liệu nằm ở `dist/data/`; fixtures kiểm tra ở `tests/`, không được web server
  cung cấp vì server chỉ phục vụ `dist/`.
- 1.471 fixtures = 1.437 mẫu development + 34 mẫu tổng hợp, không inference trên test.
- Kết quả tay vẽ có thể sai do khác phân phối Digits; điểm RF chưa hiệu chuẩn.
  Không coi chọn mẫu development rồi dự đoán đúng là test accuracy mới.

Kiểm chứng file đã xuất, không ghi lại dữ liệu:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/export_web_data.py --check
node web/tests/check-model.mjs
```

Node.js 18+ chỉ cần cho lệnh kiểm tra JavaScript, không cần để mở web.
Nếu cần dựng lại dữ liệu sau khi cập nhật code export, chạy
`python scripts/export_web_data.py`, kiểm tra và commit; không sửa JSON bằng tay.

## Cấu trúc

```text
dist/               HTML/CSS/JS và dữ liệu static, dùng cho local và Sites
tests/              Đối chiếu số liệu, hash và inference JS/Python
.openai/hosting.json Danh tính Site, không chứa token
serve.py            Server static chỉ bind loopback
start_web.bat       Mở bản local trên Windows
```

Token publish không lưu trong repo. Source xuất bản Sites được đồng bộ ở
checkout riêng, không thay remote hoặc lịch sử GitHub của đồ án.
