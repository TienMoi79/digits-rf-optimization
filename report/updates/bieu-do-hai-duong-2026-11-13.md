# Biểu đồ train và validation chỉ có hai đường

- Ngày thực hiện: 13/11/2026.
- Theo yêu cầu sinh viên: mỗi hình chỉ có hai đường như ảnh mẫu.
- Tạo hình accuracy và log loss từ toàn bộ 600 dòng đã lưu ở
  `results/curve_audit/all_tree_metrics_per_fold.csv`, lấy trung bình ba folds
  tại từng số cây 1–100. Không làm mượt, bỏ mốc hay chạy huấn luyện mới.
- Trục ngang là số cây Random Forest; không đổi thành epoch hoặc thêm
  checkpoint early stopping vì thí nghiệm không có cơ chế đó.
- Script: `scripts/plot_train_validation.py`.
- Hình: `figures/train_validation_two_lines/accuracy.png` và `log_loss.png`.
- Kiểm chứng: đủ 600 dòng, mỗi fold có đủ 100 mốc; các giá trị trung bình tại
  checkpoint cũ khớp trong sai số 1e-12; đúng hai đường mỗi hình; nguồn không đổi.
  Bằng chứng: `results/train_validation_two_lines/checks.json` và `mean_metrics.csv`.
- Chỉ đổi cách trình bày; model, dữ liệu và các kết quả tuần 10 giữ nguyên.
- Chưa có xác nhận cô duyệt hình hoặc sinh viên tự chạy; có thể gửi hình để cô góp ý.
