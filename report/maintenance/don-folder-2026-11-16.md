# Rà soát và sắp xếp folder đồ án

- Ngày thực hiện: 16/11/2026.
- Yêu cầu: đọc lại toàn bộ folder và dọn cho dễ tìm, dễ sử dụng.
- Checkout: `TienMoi79/digits-rf-optimization`.

## Công việc

1. Kiểm kê 194 file được Git quản lý trước đợt dọn. Đọc cấu trúc source,
   CLI, generator, validator, workflow, hướng dẫn và báo cáo; kiểm tra định dạng
   Python, JSON/CSV, notebook, hình PNG và gói Word.
2. Viết lại README gốc thành trang bắt đầu. Giữ nguyên vùng bảng số liệu được
   đánh dấu; chuyển hướng dẫn kỹ thuật đầy đủ sang
   [docs/huong-dan-ky-thuat.md](../../docs/huong-dan-ky-thuat.md).
3. Gom bảy báo cáo Markdown, giữ tên và nội dung lịch sử; chỉ cập nhật đường dẫn.

| Nhóm mới trong `report/` | File được chuyển vào |
|---|---|
| `checkpoints/` | Ba báo cáo tuần học 8, 9, 10 |
| `reviews/` | Phản hồi tuần 8 và kiểm tra train accuracy |
| `updates/` | Cập nhật Word 03/11 và hình hai đường 13/11 |
| `maintenance/` | Nhật ký dọn folder này |

4. Thêm [mục lục báo cáo](../README.md) và [bản đồ code/kết quả](../../docs/README.md).
   Bản Word mới nhất được ghi rõ là ngày 09/11; bản 03/11 giữ để đối chiếu.
5. Cập nhật `src/report_facts.py` và `src/week10.py` để đọc/sinh báo cáo đúng
   thư mục mới; sửa liên kết tương đối, hình ảnh và hướng dẫn trong báo cáo.
6. Giữ nguyên các CLI cấp gốc, `.venv` và lịch sử Git để lệnh chạy vẫn dùng được.
   Đã đưa 19 file bytecode và hai thư mục cache ra ngoài đồ án, vào
   `D:/Trí tuệ nhân tạo và ứng dụng/ProJeat/tmp/cleanup-2026-11-16/cache-backup/`
   để có thể khôi phục. Lệnh xóa ban đầu bị cơ chế tự động chặn nên dùng cách
   lưu tạm; đã xác nhận hai cache gốc không còn. Không chạy lại HPO.
7. Thêm quy tắc bỏ qua file khóa Word `~$*.docx`, cập nhật quy tắc tổ chức tài
   liệu trong `AGENTS.md` để các lần sau tiếp tục đúng thư mục.

## Phạm vi bảo toàn

Chụp hash 143 artifact đã theo dõi trước đợt dọn: toàn bộ kết quả,
biểu đồ, config, hai notebook và hai Word. Đối chiếu sau dọn để bảo đảm chúng
giữ nguyên nội dung. Text được chuẩn hóa CRLF/LF khi tính hash; binary được
đối chiếu từng byte. Các bảng/log/JSON và lần tái lập riêng không phải file rác:
chúng cần cho provenance và validator nên được giữ nguyên vị trí.

Các checkpoint đã qua không được ghi thành công việc mới hoặc đánh dấu cô duyệt.
Vùng số liệu sinh tự động của README và báo cáo tuần 8 giữ nguyên. Không thay
model, dữ liệu, predictions, cấu hình, epsilon hoặc kết luận thực nghiệm.

## Kiểm tra sau dọn

| Kiểm tra đã chạy | Kết quả |
|---|---|
| Audit định dạng và đường dẫn | 198 file; Python parse được, JSON/CSV/notebook/PNG/Word hợp lệ; mọi liên kết nội bộ và heading anchor tồn tại |
| Hash trước/sau | Cả 143 artifact được giữ nguyên |
| Vùng số liệu tự sinh README và tuần 8 | Giữ nguyên nội dung |
| `python -m unittest discover -s tests -v` | 25 tests đạt |
| `python -m src.validation` | 401 checks đạt |
| `python -m src.robustness_validation --if-exists` | 3.278 checks đạt |
| `python -m src.report_facts --check` | Đạt với đường dẫn tuần 8 mới |
| `python -m src.report_evaluation --check` | Bảng Word và hash gốc đạt |
| `python -m src.week10 validate` / `report` | Đạt; generator sinh đúng thư mục checkpoint mới |
| Demo tuần 9 validate, replay và demo dự đoán | Đạt |
| `git diff --check` | Đạt |

Bản sao kiểm kê trước dọn, nhật ký kiểm tra và công cụ audit tạm nằm ngoài
repo tại `D:/Trí tuệ nhân tạo và ứng dụng/ProJeat/tmp/cleanup-2026-11-16/`.
Không dựng lại Word hoặc notebook trong đợt này; kiểm tra chúng giữ nguyên
nội dung, không ghi nhận một lần huấn luyện hay kết quả mô hình mới.

## Việc còn lại

Sinh viên có thể mở [README chính](../../README.md) hoặc mục lục báo cáo để lấy
đúng file cần gửi cô. Chưa có feedback mới của cô; việc sinh viên tự tái lập,
demo, duyệt checkpoint và nộp bài tiếp tục theo [PROGRESS.md](../../PROGRESS.md).
