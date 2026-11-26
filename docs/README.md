# Bản đồ đồ án và tài liệu

Từ [trang bắt đầu](../README.md), chọn phần cần đọc:

- [Hướng dẫn mở và nộp bài](nop-bai.md): link GitHub, ZIP sạch, web và demo.

- [Hướng dẫn kỹ thuật đầy đủ](huong-dan-ky-thuat.md): formulation, protocol,
  kết quả, mở rộng, kiểm chứng dữ liệu/biểu đồ, lệnh chạy và giới hạn.
- [TPE và kịch bản demo tối ưu](tpe-va-demo-toi-uu.md).
- [Tự chạy tái lập tuần 10](tu-chay-tai-lap-tuan-10.md).
- [Mục lục Word, checkpoint và phản hồi](../report/README.md).
- [Giao diện web và hướng dẫn mở trên máy](../web/README.md).

## Code nằm ở đâu?

| Nhóm | File chính |
|---|---|
| Dữ liệu, seed, config, Random Forest | [src/common.py](../src/common.py), [configs](../configs) |
| Baseline, Random/TPE và chọn nghiệm | [src/experiment.py](../src/experiment.py), [run_experiment.py](../run_experiment.py) |
| Phân tích thí nghiệm gốc | [src/analysis.py](../src/analysis.py) |
| Nghiên cứu 5 seeds/ablation | [src/robustness.py](../src/robustness.py), [run_robustness.py](../run_robustness.py) |
| Demo dự đoán / quá trình tối ưu | [demo.py](../demo.py), [demo_optimization.py](../demo_optimization.py) |
| Bảng train/validation/test | [src/report_evaluation.py](../src/report_evaluation.py) |
| Phân tích tái lập và lỗi tuần 10 | [src/week10.py](../src/week10.py) |
| Dựng Word, notebook, biểu đồ | [scripts](../scripts) |
| Kiểm tra và CI | [tests](../tests), [workflow](../.github/workflows/ci.yml) |
| Web dashboard và suy luận trong trình duyệt | [web](../web), [export dữ liệu](../scripts/export_web_data.py) |

## Vì sao có nhiều thư mục kết quả?

Mỗi thư mục là một bộ bằng chứng khác nhau, không phải bản sao để xóa tùy ý:

| Vị trí | Mục đích |
|---|---|
| [results/](../results) — các file ngay cấp này | Thí nghiệm gốc: 180 trials + baseline, chọn nghiệm, predictions test và bảng cuối |
| [results/models/selected.joblib](../results/models/selected.joblib) | Model dùng cho demo và bàn giao |
| [results/robustness/](../results/robustness) | 450 trials, 5 seeds, chỉ development; so sánh budget và startup TPE |
| [results/week9_demo/](../results/week9_demo) | Lần demo tối ưu nhỏ có log riêng |
| [results/week10_2026-11-09/](../results/week10_2026-11-09) | Đối chiếu, lỗi và latency; `reproduction/` là lần chạy lại 180 trials |
| [results/report_evaluation_2026-11-03/](../results/report_evaluation_2026-11-03) | Số liệu bổ sung để dựng bảng Word |
| [results/train_accuracy_audit_2026-11-03/](../results/train_accuracy_audit_2026-11-03) | Kiểm tra độc lập train accuracy, từng pixel/nhãn UCI |
| [results/tree_diagnostics/](../results/tree_diagnostics) | Số đo train/validation tại các checkpoint số cây ban đầu |
| [results/curve_audit/](../results/curve_audit) | Kiểm chứng và số đo chi tiết tại đủ 100 số cây |
| [results/train_validation_two_lines/](../results/train_validation_two_lines) | Mean 3 folds dùng cho hình chỉ có hai đường |

Các JSON/CSV có thể biểu diễn cùng dữ liệu để vừa đọc bằng code vừa mở bảng;
không xóa một định dạng nếu chưa sửa generator và validator. Giữ nguyên vị trí
các artifact đã ghi hash, để các lần kiểm chứng tiếp tục đối chiếu được.

## Xem hình nào?

| Cần xem | Đường dẫn |
|---|---|
| Hai đường train/validation | [Accuracy](../figures/train_validation_two_lines/accuracy.png), [loss](../figures/train_validation_two_lines/log_loss.png) |
| Từng fold và vùng loss phóng to | [figures/curve_audit/](../figures/curve_audit) |
| Search history và đánh đổi kích thước | [figures/](../figures) — các hình ngay cấp này |
| Budget và ablation 5 seeds | [figures/robustness/](../figures/robustness) |
| 10 mẫu sai, confusion matrix, latency | [figures/week10_2026-11-09/](../figures/week10_2026-11-09) |

`data/` chứa [nguồn và tiền xử lý](../data/README.md); dataset gốc nằm trong
scikit-learn nên không cần thêm một bản CSV mới. `.venv/` là thư viện cục bộ,
`.git/` giữ lịch sử. Cache Python có thể tạo lại và không được commit.
