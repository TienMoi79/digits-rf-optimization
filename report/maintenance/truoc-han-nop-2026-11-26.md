# Rà soát folder và chuẩn bị bản nộp

- Thực hiện ngày **26/11/2026**; hạn nộp theo cập nhật của sinh viên là **27/11/2026**.
- Repo: [TienMoi79/digits-rf-optimization](https://github.com/TienMoi79/digits-rf-optimization).
- Phạm vi: checkout Digits/Random Forest trong `Bài Tập Lớn cô Anh` và repo tương ứng.

## Kết quả rà soát và thay đổi

1. Đã fetch GitHub, kiểm tra nhánh `main` và trạng thái đồng bộ. Trước đợt này
   có 215 file theo dõi; không có thay đổi chưa commit hoặc file lạ cần xóa
   khỏi GitHub. Môi trường `.venv` đã được bỏ qua đúng quy định.
2. Giữ cách tổ chức báo cáo theo checkpoint/reviews/updates/maintenance.
   Không xáo trộn source, dữ liệu hay các bộ log sát hạn nộp. Các bản chạy
   tái lập, hai notebook, báo cáo cũ và model là bằng chứng cần giữ.
3. Thêm [hướng dẫn mở và nộp bài](../../docs/nop-bai.md), đưa đường dẫn lên
   đầu README để mở được ngay web, demo và đúng bản Word mới nhất.
4. Sửa launcher `web/start_web.bat`: ưu tiên venv có sẵn, nếu chưa có thì
   dùng `py -3` hoặc `python`. Web chỉ cần thư viện chuẩn, không bắt máy mới
   phải cài môi trường ML. Không thay giao diện/model trên Site đang host.
5. Thêm `scripts/package_submission.py`: tạo ZIP từ commit sạch, đối chiếu
   từng file, kiểm tra CRC và ghi SHA256. Refuse ghi đè file cũ, đóng gói
   trạng thái chưa commit hoặc đặt ZIP trong repo. CI chạy kiểm tra ZIP.
6. Bổ sung ignore cho ZIP, checksum và thư mục QA/bản nộp cục bộ. Các file
   này được giữ ngoài checkout; không đưa môi trường/cache vào GitHub.

## Kiểm chứng thực tế tại máy

| Hạng mục | Kết quả |
|---|---|
| `pip check` | Không có dependency hỏng |
| Python unit tests | 27 tests đạt, gồm kiểm tra xuống dòng Windows và chặn source chưa commit |
| Kết quả gốc | 401 checks đạt |
| Nghiên cứu mở rộng | 3.278 checks đạt |
| Report facts/evaluation | Đạt |
| Tái lập và lỗi tuần 10 | Đạt |
| Web exports | Khớp logs/model/fixtures nguồn |
| JavaScript inference | 1.471 fixtures JS/Python khớp; sai số xác suất tối đa 0 |
| Demo validate/replay/predict | Đạt |
| Web bản copy sạch | HTTP thành công cho HTML/CSS/JS/model/JSON/CSV bằng Python hệ thống `-I -S`, không có venv hay thư viện ML |
| Bảo toàn artifact | Đối chiếu hash trước/sau cho 149 file kết quả/hình/config/notebook/Word và bằng chứng web |

Trong CI Windows, validator ghi hai `validation.json` bằng CRLF khiến
`git status` báo modified dù `git diff` không có thay đổi nội dung. Đã tái
hiện trên clone sạch với `core.autocrlf=true`, chuyển kiểm tra sang content
diff của Git và danh sách file chưa theo dõi. Việc so sánh từng file trong
ZIP vẫn bắt buộc; thêm tests cho thay đổi thật, đã stage và file chưa commit.
Bản clone Windows sau kiểm chứng đã đóng gói đạt, giữ nguyên số liệu.

ZIP cuối được tạo **sau commit** ở ngoài folder đồ án. Script kiểm chứng
thành công mới ghi checksum; SHA256 và commit chính xác nằm trong file
`BAN_NOP_DIGITS_RF_2026-11-26.zip.sha256.txt` đi kèm. Kiểm tra trên Windows/Linux
và bước package của bản GitHub cuối có log riêng trong Actions.

Nhật ký kiểm tra và bản copy QA nằm ngoài repo tại
`C:/Users/lequy/.codex/tmp/digits-final-20261126/`. Không tạo lại Word, hình
hoặc notebook; không chạy lại HPO chỉ để dọn cấu trúc.

## Giới hạn và trạng thái bàn giao

Thí nghiệm và model giữ nguyên; đây là rà soát kỹ thuật và chuẩn bị file nộp,
không phải xác nhận đã đạt điểm, cô đã duyệt hay sinh viên đã tự bảo vệ.
Site online vẫn riêng tư, nên giảng viên có thể dùng web local không cần
đăng nhập ChatGPT. Không thay audience hoặc công bố một release mới trong
đợt dọn này. Sinh viên tự nộp link/file đúng nơi và xác nhận đã nộp.
