# Báo cáo Word hoàn chỉnh theo mẫu ngày 27/11/2026

## Yêu cầu và phạm vi

Sinh viên yêu cầu viết báo cáo theo tài liệu Word tham khảo, dùng nội dung
đồ án Digits/Random Forest của mình. Tác giả **Lê Quyết Tiến**, MSSV
**2474802010386**; thông tin trường/khoa/học phần theo mẫu đã cung cấp.
Mẫu chỉ hướng dẫn cấu trúc và trình bày, không cung cấp số liệu cho đồ án này.

Bản mới: [Báo cáo đồ án hoàn chỉnh](../Bao_cao_Do_an_Toi_uu_Random_Forest_Le_Quyet_Tien_2026-11-27.docx).
Hai Word 03/11 và 09/11 giữ nguyên đường dẫn và hash.

## Nội dung đã hoàn thành

- 34 trang: bìa Văn Lang, nhận xét giảng viên để trống, thông tin sinh viên,
  lời mở đầu, mục lục, danh mục hình/bảng/từ viết tắt, bốn chương, tham khảo
  đánh số và phụ lục nguồn kiểm chứng.
- Bài toán tối ưu bốn biến, miền 560 cấu hình, ngân sách, CV Macro-F1,
  Random Search, TPE, quy tắc epsilon và sáu công thức Word có thể chỉnh sửa.
- Train in-sample, validation ba folds, test 360 mẫu, so sánh baseline,
  cấu hình đại diện, dung lượng và độ nhạy epsilon.
- Tách rõ 180 trials gốc, 450 trials CV-only, kiểm chứng đường cong,
  tái lập tuần 10, phân tích lỗi, runtime/latency và giao diện web.
- 11 hình, 22 bảng nội dung; thêm hai bảng biểu mẫu đầu tài liệu.
  Hai hình mới chỉ minh họa mẫu development và pipeline, không sinh metric mới.
- Bỏ toàn bộ tên/MSSV, số liệu, nội dung, hình và liên kết của đồ án tham khảo;
  không đưa tài liệu gốc của người khác vào repository.

## Kết luận khoa học giữ đúng

Selected trùng RF mặc định, CV Macro-F1 0,97062, test accuracy 97,22%
(350/360). Mô hình nhỏ hơn 40,28% so với đại diện Random tốt nhất,
nhưng không nhỏ hơn hoặc cải thiện so với chính baseline mặc định.
Train 100% là kết quả trên dữ liệu đã fit, không phải bằng chứng tổng quát hóa.
Không dùng test để chọn hyperparameter/epsilon hoặc triển khai lại mô hình.

## Kiểm tra thực tế

| Kiểm tra | Kết quả |
|---|---|
| Render Word và xem mọi trang | Đã xem đủ 34 PNG, sửa các đoạn tràn lẻ trang |
| Số trang trong ba mục lục | 75 tham chiếu khớp bookmarks/trang Word |
| Bảng số liệu trong DOCX | 215 ô trọng yếu khớp CSV/JSON gốc |
| Section, lề, kích thước và số trang | Ba section A4 theo mẫu, Roman ở đầu/Arabic từ Chương I |
| Danh tính và nội dung tham khảo | Không còn tên/MSSV/nội dung đồ án của người khác trong package |
| Artifact cũ | 144 file được đối chiếu, không đổi hash |
| `python -m src.report_facts --check` | Đạt |
| `python -m src.report_evaluation --check` | Đạt; chỉ kiểm tra bảng đã lưu |

Xem [manifest nguồn](final-report-sources-2026-11-27.json) và
[kết quả kiểm tra](final-report-validation-2026-11-27.json).
Không chạy HPO hoặc suy luận test mới cho đợt viết báo cáo này.

Script dựng: `scripts/build_final_report.py`; template đã bỏ nội dung tham khảo
ở `scripts/templates/`. Microsoft Word cập nhật các trường mục lục, chuẩn hóa
ID styles/cache và lưu bản native. Thư viện render đóng gói không tìm thấy
LibreOffice; dùng Word COM qua PowerShell 7 và Poppler để kiểm tra. Các PDF/PNG
QA nằm ngoài repo, không phải deliverable bổ sung. Windows PowerShell 5.1
gặp treo COM xuất PDF, nên helper kiểm tra yêu cầu PowerShell 7.

## Phần cần sinh viên và giảng viên xác nhận

Sinh viên đọc báo cáo, tự mở/chạy bài và giải thích kết quả khi được hỏi.
Nhận xét, điểm, chữ ký của giảng viên để trống. Chưa ghi nhận duyệt báo cáo,
xác nhận sinh viên tự tái lập hoặc nộp LMS; sẽ sửa khi có phản hồi thực tế.
Bản ZIP ngày 27/11 được tạo từ commit sạch bằng tiện ích đóng gói, kèm SHA256.
