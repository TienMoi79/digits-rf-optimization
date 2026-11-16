# Bổ sung báo cáo Word train validation test và so sánh baseline

- Ngày thực hiện: 03/11/2026.
- Yêu cầu mới: một file Word có kết quả train, validation, test và so sánh baseline.
- File: `Bao_cao_Train_Validation_Test_Baselines_2026-11-03.docx`.
- Trạng thái: đã hoàn thành bảng/nội dung, render tám trang và kiểm tra trực quan từng trang; chưa có cô duyệt.

## Nội dung và phạm vi số liệu

Báo cáo gồm tám phần: tóm tắt và dữ liệu; formulation/protocol; train;
validation; test; so sánh baseline; hiệu quả HPO/ổn định; quyết định và tái lập.
RF mặc định là baseline mô hình, Random Search là mốc so sánh thuật toán;
RF Random và RF TPE là hai cấu hình forest tìm được. Selected trùng default.
Không bổ sung một classifier mới sau khi xem test và coi đó là so sánh ban đầu.

| Phần | Cách tính | Bằng chứng |
|---|---|---|
| Train | Refit cấu hình cố định trên toàn bộ 1.437 development rồi chấm chính tập đó | `comparison.csv`, classification reports |
| Validation | Ba folds train 958 / validation 479, mean ± sample SD | `validation_per_fold.csv`, đối chiếu fold F1 gốc |
| Test | Accuracy/precision/recall/F1 tính từ nhãn dự đoán đã lưu của 360 mẫu | `results/test_predictions.csv`, đối chiếu final_metrics |
| So sánh | Các đại diện gốc và model size; mean best CV qua seeds được tách riêng | `results/selection.json`, search_summary, các hình gốc |

Các bảng bổ sung ở `results/report_evaluation_2026-11-03/`. Có ba cấu hình
phân biệt, tổng 12 fits development (refit toàn bộ + ba CV mỗi cấu hình).
Không chạy lại HPO hoặc thay model, không thêm inference test khi tính bảng.
Train của tất cả đại diện đạt 100%, có giải thích in-sample và khoảng cách
với validation/test; không gọi đó là hoàn hảo hoặc bằng chứng không overfit.
Test log loss không được bổ sung vì predictions gốc không lưu xác suất của
tất cả đại diện. Chỉ số log loss có cho train/CV.

## Code và kiểm chứng

- `src/report_evaluation.py`: tính số liệu, đối chiếu fold F1/test metrics và hash artifact cũ.
- `scripts/build_train_validation_test_report.py`: dựng DOCX bằng python-docx từ CSV/JSON, dùng runtime tài liệu của Codex.
- Train của selected refit khớp dự đoán train của model bàn giao.
- Fold Macro-F1 tái tạo khớp từng fold gốc trong dung sai 1e-12.
- Test accuracy/Macro-F1 tái tạo từ predictions khớp `final_metrics.csv`.
- Hash xác nhận 75 artifact cũ không đổi.

Kỹ năng Documents yêu cầu render rồi xem ảnh từng trang. Renderer LibreOffice
đóng gói báo thiếu `soffice.exe` trên Windows; dùng Microsoft Word mở read-only,
export PDF tạm và Poppler render PNG để kiểm tra. PDF/PNG nằm ngoài repo ở
thư mục QA tạm, không phải sản phẩm bàn giao. Chỉ gửi file Word được yêu cầu.

Kiểm tra cuối: 21 unit tests đạt; 401 checks kết quả gốc đạt; 3.278 checks
phần mở rộng đạt; report facts tuần 8 và validator demo tuần 9 đạt. Các bảng
bổ sung khớp mean/SD của fold, classification report và hashes gốc qua
`python -m src.report_evaluation --check` (đã chạy đạt, không retrain).
Lệnh mới cũng được đưa vào CI và tiện ích cập nhật GitHub.

Đây là bổ sung theo yêu cầu mới của cô, chưa có xác nhận cô duyệt báo cáo
hoặc sinh viên đã tự trình bày. Các checkpoint cũ không bị đổi ngày hay trạng thái.
