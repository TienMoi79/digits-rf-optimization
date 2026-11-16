# Giao diện web Digits Lab — bản đầu để góp ý

- Ngày thực hiện: 16/11/2026.
- Yêu cầu sinh viên: làm giao diện web để xem và góp ý.
- [Bản web trên Sites](https://digits-rf-lab-tienmoi79.flowypheasant16.chatgpt.site).
- [Cách mở trên máy](../../web/README.md): nhấp đúp `web/start_web.bat` hoặc
  chạy `python web/serve.py`, mở `http://127.0.0.1:8765`.

## Đã thực hiện

| Màn hình | Nội dung |
|---|---|
| Tổng quan | Chỉ số selected, 180 trials, mean best-so-far Random/TPE, epsilon và mô hình nhỏ |
| So sánh mô hình | Train/validation/test, RF mặc định/Random/TPE/selected, CSV và đủ 10 mẫu sai |
| Quá trình tìm kiếm | Chọn seed 11/22/33, thanh 1–30 trials, phát/tạm dừng lịch sử và các trial vừa hoàn thành |
| Train & Validation | Hai đường accuracy hoặc loss, không làm mượt; tùy chọn zoom loss 20–100 cây |
| Thử nhận dạng | Chọn 30 mẫu development hoặc vẽ trực tiếp trên lưới 8×8, nhận dạng và xem ba điểm số cao nhất |

Thiết kế dashboard tiếng Việt, menu bên trái trên máy tính và menu thu gọn
trên màn hình nhỏ. Dùng HTML/CSS/JavaScript thuần; không cần thư viện mới để
mở bản local. Không thêm chức năng train lại từ giao diện.

## Model và nguồn số liệu

`scripts/export_web_data.py` đọc logs, CSV và `results/models/selected.joblib`,
xuất 100 cây cùng dữ liệu sang `web/dist/data/`. `web/dist/forest.js` duyệt cây,
ép input float32 như sklearn rồi lấy trung bình xác suất; không dùng nhãn mẫu
để trả kết quả dự đoán. File model nguồn và 143 artifact gốc giữ nguyên hash.

Mẫu thử chỉ lấy từ development. 10 ảnh lỗi test chỉ hiển thị predictions đã
lưu, không được dùng làm gallery dự đoán mới. Điểm số RF chưa hiệu chuẩn;
chữ tự vẽ có thể sai vì khác phân phối Digits. Trục learning curve là số cây,
không đổi thành epoch. Train 100% vẫn được giải thích là in-sample.

## Kiểm tra đã chạy

| Kiểm tra | Kết quả thực tế |
|---|---|
| `scripts/export_web_data.py --check` | Khớp dữ liệu nguồn, hash gốc, mẫu development và fixtures sklearn |
| `node web/tests/check-model.mjs` | 1.471 đầu vào JS/Python khớp, sai số xác suất tối đa bằng 0; input sai bị từ chối |
| Thành phần fixtures | 1.437 development + 34 tổng hợp; không inference trên features test |
| `node --check web/dist/app.js` | Đạt |
| Unit tests Python | 25 tests đạt |
| Kiểm chứng kết quả gốc / mở rộng | 401 / 3.278 checks đạt |
| Report facts/evaluation và tuần 10 | Đạt |
| Local HTTP | Trang chính, JSON, model và file CSV được phục vụ thành công |
| Trình duyệt | Đã thao tác menu, seed, trial slider, phát lại, accuracy/loss/zoom, chọn mẫu, đổi mẫu, vẽ, xóa và nhận dạng |
| Responsive | Kiểm tra DOM ở 1.366×900 và 390×844; trang không tràn ngang; menu nhỏ hoạt động |

Ví dụ UI: mẫu development nhãn 8 được model dự đoán 8; xóa ảnh rồi nhấn nhận
dạng có thông báo yêu cầu đầu vào. Nét vẽ thử được model trả dự đoán thực,
không áp đặt nhãn đúng. Kiểm tra console local không ghi nhận error/warning.

IAB không hoàn tất cơ chế chờ sự kiện download trong lần thử; nút tải sau
đó được chuyển sang liên kết file CSV static và file trả HTTP 200. Chưa ghi
nhận xác nhận file đã được trình duyệt của sinh viên lưu vào Downloads.

## Xuất bản và quyền xem

Xuất bản bằng skill Sites, trạng thái native **succeeded**. Site ID:
`appgprj_6afaf73b0f8881919e42d59f4c7fe9cf`; source commit của bản Sites:
`b964cf255f8e18c1ed9abe23060ec11eb04e4ddd`. Checkout xuất bản riêng, không thay
remote GitHub của đồ án. Token chỉ ở bộ nhớ phiên, không ghi vào file/repo.

Giữ quyền xem riêng của chủ sở hữu. Trình duyệt kiểm tra online hiện tới cổng
đăng nhập ChatGPT; sinh viên cần dùng tài khoản chủ sở hữu để xem. Không đổi
Site thành public. Bản local không yêu cầu đăng nhập và đã được kiểm thử UI.

Công cụ tar trên Windows gặp lỗi đường dẫn tiếng Việt khi đóng gói; đã giữ
nguyên source và chuyển vị trí archive tạm sang đường dẫn ASCII, sau đó xuất
bản thành công. Không đổi ứng dụng để xử lý lỗi đóng gói này.

## Phần cần sinh viên góp ý

Góp ý màu sắc, bố cục, độ dễ đọc, trình tự demo và các phần cần cô xem. Đây
là bản giao diện đầu, không thay xác nhận sinh viên hiểu bài, cô duyệt checkpoint,
tạo release cuối hoặc nộp bài. Model, hyperparameter, epsilon và kết quả
thực nghiệm vẫn như trước.
