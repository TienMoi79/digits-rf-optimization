# Mẫu Word đã loại bỏ nội dung tham khảo

`van_lang_report_base.docx` giữ khung trang bìa Văn Lang, logo, styles,
header/footer, ba section và hai bảng biểu mẫu từ tài liệu sinh viên cung cấp.
Đã bỏ toàn bộ nội dung, hình, dữ liệu, liên kết và danh tính của đồ án tham khảo.
Thông tin tác giả được đổi sang Lê Quyết Tiến, MSSV 2474802010386.
Không đưa tài liệu gốc của người khác vào repository.

Báo cáo mới được dựng bởi `../build_final_report.py` từ CSV/JSON đã lưu.
Cần runtime tài liệu có python-docx, Pillow, lxml và font Arial/Times New Roman
trên Windows. Môi trường ML không bắt buộc cài các thư viện soạn Word.

Sau khi dựng ra thư mục tạm ngoài repo, dùng `../finalize_report_word.ps1`
với `-InputPath` là bản dựng và `-OutputPath` là bản báo cáo cần giao.
Microsoft Word cập nhật ba mục lục/số trang và chuẩn hóa cache, công thức,
ID styles theo ngôn ngữ giao diện. Giữ bản lưu native Word để tương thích.
Tên styles, font, lề, header/footer và cách trình bày vẫn theo mẫu.
Hai tiện ích Word yêu cầu PowerShell 7; gọi trực tiếp bằng `pwsh`,
không gọi qua Windows PowerShell 5.1 vì COM xuất PDF bị treo trong phiên này.

`../inspect_report_word.ps1` mở bản copy chỉ đọc trong thư mục QA ngoài repo,
xuất PDF kiểm tra và JSON trang của các bookmarks; không lưu vào báo cáo.
PDF/PNG chỉ phục vụ kiểm tra, không phải bản nộp được yêu cầu.
Trước khi bàn giao cần xem mọi trang và kiểm tra lại mục lục/hình/bảng.

Không chạy lại HPO, suy luận test hoặc sửa các logs để dựng báo cáo.
Các bản Word cũ được giữ nguyên để đối chiếu.
