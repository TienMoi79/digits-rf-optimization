# Báo cáo rà soát và tổ chức checkpoint tuần học 8

- **Ngày cập nhật:** 27/10/2026.
- **Môn học:** Nhập môn tối ưu hóa.
- **Đề tài:** Tối ưu siêu tham số Random Forest cho nhận dạng chữ số viết tay.
- **Repo:** [TienMoi79/digits-rf-optimization](https://github.com/TienMoi79/digits-rf-optimization).
- **Mốc hiện tại:** tuần học 8; hạn nộp theo thông tin sinh viên là tuần học 11.
- **Phần hoàn thành trong đợt này:** rà soát yêu cầu, bổ sung sơ đồ pipeline,
  lập kế hoạch checkpoint còn lại và kiểm chứng bộ kết quả đã có.
- **Trạng thái giảng viên phản hồi/duyệt:** chưa có phản hồi được chuyển lại.

## 1 Phạm vi gửi cô kiểm tra

Báo cáo này ghi nhận công việc mới của đợt rà soát tuần học 8 và bằng chứng
đã tồn tại trước đó. Dự án đã có mã nguồn và các lần chạy thực nghiệm trước
ngày lập báo cáo. Phần vừa hoàn thành là tổ chức lại checkpoint và kiểm tra
hiện trạng; chưa phải xác nhận đã hoàn thành đủ năm checkpoint của đề.

Kính nhờ cô xem tính phù hợp của bài toán tối ưu, pipeline, cách lưu bằng
chứng và kế hoạch hoàn thiện đến tuần học 11. Phần giải thích TPE, demo tối ưu,
tái lập do sinh viên tự chạy và chuẩn bị bảo vệ vẫn còn công việc cần làm.

Theo cập nhật của sinh viên, phạm vi nộp không có báo cáo Word/PDF hoặc slide.
File Markdown này được tạo theo yêu cầu của sinh viên để trao đổi tiến độ và
nhận góp ý sau từng phần, không thay thế việc demo/bảo vệ trực tiếp.

## 2 Công việc mới hoàn thành ngày 27 tháng 10

| Nội dung | Thay đổi thực hiện | Bằng chứng |
|---|---|---|
| Đối chiếu yêu cầu | Phân biệt sản phẩm đã có, thay đổi mới và checkpoint chưa được duyệt | `PROGRESS.md` |
| Sơ đồ pipeline | Thể hiện development/test, CV, tìm kiếm, kiểm tra miền, chọn theo epsilon và đánh giá sau khi khóa cấu hình | Mục Sơ đồ pipeline trong `README.md` |
| Kế hoạch đến hạn nộp | Lập công việc và bằng chứng cần có cho tuần học 8–11 | `PROGRESS.md` |
| Quy trình nhận góp ý | Sau từng phần có báo cáo `.md`; khi nhận feedback sẽ ghi cách sửa và kiểm tra sau sửa | `AGENTS.md`, `PROGRESS.md`, thư mục `report/` |
| Kiểm chứng hiện trạng | Chạy unit tests và hai bộ kiểm chứng kết quả đã lưu | Kết quả lệnh tại mục 5 |

Trong đợt này không cài lại thuật toán, chạy lại toàn bộ HPO, đổi dữ liệu/nhãn,
điều chỉnh epsilon hoặc đánh giá thêm mô hình mới trên test. Các số liệu thực
nghiệm ở phần sau được ghi rõ là kết quả đã có trước đợt rà soát.

## 3 Bài toán và protocol hiện tại

Dữ liệu là Digits do scikit-learn phân phối, gồm 1.797 ảnh chữ số 8×8,
64 đặc trưng và 10 lớp. Nguồn, phiên bản và tiền xử lý nằm trong
[`data/README.md`](../data/README.md). Dữ liệu được giữ nguyên các vector pixel
do loader cung cấp, không có bước tiền xử lý được học từ toàn bộ dataset.

Biến quyết định:

```text
x = (n_estimators, max_depth, min_samples_leaf, max_features)

n_estimators       in {50, 75, 100, 125, 150, 175, 200}
max_depth          in {5, 10, 15, 20, None}
min_samples_leaf   in {1, 2, 3, 4}
max_features       in {sqrt, log2, half, all}

maximize f(x) = mean(Macro-F1 over 3 development CV folds)
budget = 30 completed trials per method and search seed
```

Miền tìm kiếm có 560 cấu hình; không tuyên bố đã duyệt hết miền hoặc tìm được
tối ưu toàn cục. Các cấu hình được kiểm tra thuộc miền khả thi. Sau tìm kiếm,
quy tắc chọn cấu hình triển khai là:

```text
minimize serialized_model_bytes(x)
subject to x being an observed feasible candidate
           CV_F1(x) >= best_observed_CV_F1 - 0.005
```

Baseline là Random Forest với tham số học mặc định, cố định RF seed 123 và
`n_jobs=1`. Random Search và Optuna TPE dùng cùng miền tìm kiếm, metric, folds
và ngân sách. Cấu hình thực nghiệm nằm trong `configs/experiment.json`.

Development có 1.437 mẫu, test có 360 mẫu; stratified split seed 42. Ba CV
folds chỉ nằm trong development, CV seed 42. `selection.json` được lưu trước
đánh giá test trong pipeline gốc. Phần mở rộng chỉ dùng development.

Kết quả test gốc đã được xem và công bố. Đợt rà soát hiện tại không coi tập
test này là một đánh giá mới chưa từng quan sát, và không dùng kết quả đã
biết để thay đổi cấu hình, epsilon hoặc kết luận phương pháp thắng.

## 4 Bằng chứng kỹ thuật đã có trước đợt rà soát

| Hạng mục | Artifact kiểm tra trong repo |
|---|---|
| Chia dữ liệu và dựng RF | `src/common.py`, `results/split_indices.json` |
| Baseline, Random Search, TPE, chọn cấu hình | `src/experiment.py`, `results/baseline.json`, `results/selection.json` |
| Thí nghiệm gốc 180 search trials | `results/trials.json`, `results/trials.csv`, `results/optuna_*.csv` |
| Phân tích trade-off và sensitivity epsilon | `src/analysis.py`, `results/epsilon_sensitivity.csv`, `figures/` |
| Phần mở rộng 450 trials trên 5 search seeds | `src/robustness.py`, `results/robustness/`, `notebooks/robustness_study.ipynb` |
| Kiểm tra accuracy và log loss | `src/curve_audit.py`, `results/curve_audit/`, `figures/curve_audit/` |
| Tái lập và demo dự đoán | Các file requirements, `setup.ps1`, `run_experiment.py`, `run_robustness.py`, `demo.py` |

Model được chọn trong thí nghiệm gốc là RF mặc định vì đáp ứng quy tắc epsilon
và có kích thước nhỏ nhất trong nhóm ứng viên đạt ngưỡng. CV Macro-F1 khoảng
0,97062; test accuracy 0,97222. Đây là kết quả đã lưu của lần chạy gốc, không
phải kết quả huấn luyện mới ngày 27/10.

Đường accuracy/loss theo số cây là chẩn đoán development, không phải đường
theo epoch. Bản kiểm tra chi tiết lưu từng fold, từng số cây; giữ nguyên dao
động thực tế. Mean/SD qua search seeds không thay cho bất định do đổi dataset,
split hoặc RF seed. Thời gian chạy có outlier đã được giữ trong bảng kết quả.

## 5 Kết quả kiểm tra trong đợt này

Các lệnh đã chạy ngày 27/10 bằng Python trong `.venv` với tùy chọn `-X utf8`:

| Lệnh | Kết quả thực tế |
|---|---|
| `python -m unittest discover -s tests -v` | 14 tests đạt |
| `python -m src.validation` | 401 kiểm tra kết quả gốc đạt |
| `python -m src.robustness_validation` | 3.278 kiểm tra phần mở rộng đạt; bộ kết quả gốc được bảo toàn |
| `git diff --check` | Không phát hiện lỗi whitespace ở thay đổi được theo dõi |

Các kiểm chứng bao gồm sự tách biệt của split/folds, ngân sách, tính khả thi,
quy tắc chọn cấu hình, tính lại metric từ predictions và tính nhất quán giữa
logs/bảng kết quả. Không chạy lại 180 hoặc 450 trials trong đợt này. Các checks
không thay thế khả năng sinh viên tự chạy và giải thích, cũng không bảo đảm
đã loại trừ mọi lỗi có thể có.

## 6 Hiện trạng lịch sử Git và checkpoint

Trước lần cập nhật báo cáo này, repo có 7 commit trên hai ngày:

- **13/10/2026:** `f7ea277`, `73271cf` — bản dự án ban đầu và chuẩn hóa repo.
- **24/10/2026:** `2ab1a4c`, `4c1b160`, `b41a801`, `c4cecec`, `1e18603` —
  phần mở rộng, sửa lỗi, kết quả và kiểm tra đường cong.

Commit đầu chứa nhiều nội dung cùng lúc. Các commit này thể hiện các thay đổi
thực tế nhưng chưa đủ để tự khẳng định có checkpoint đã được duyệt trong
năm tuần riêng biệt. Nhật ký hiện tại không gán lại các commit đó thành năm
tuần hoặc xác nhận ngược rằng sinh viên đã bảo vệ từng mốc.

Báo cáo này và sơ đồ/kế hoạch đi kèm sẽ được lưu trong một commit mới đúng
ngày thực hiện. Việc chấp nhận bằng chứng trước đó cho các checkpoint đã qua
cần được giảng viên xác nhận.

## 7 Kế hoạch phần còn lại

| Tuần học | Nội dung cần làm | Trạng thái |
|---|---|---|
| 8 | Rà soát yêu cầu, bổ sung pipeline và kế hoạch; sinh viên tự kiểm tra formulation/baseline | Phần tài liệu và kiểm chứng đã làm; phần sinh viên tự giải thích chưa xác nhận |
| 9 | Làm rõ đầu vào/đầu ra, nguyên lý và lý do chọn TPE; demo quá trình tìm kiếm và chọn theo epsilon | Dự kiến |
| 10 | Sinh viên chạy tái lập vào thư mục mới; đối chiếu số liệu; phân tích dự đoán sai và giới hạn | Dự kiến |
| 11 | Kiểm tra cuối, chốt bản nộp, tag/release v1.0, demo/bảo vệ và nộp link | Dự kiến |

Các tuần học không được gán ngày dương lịch khi chưa có lịch cụ thể của lớp.
Đây là kế hoạch hoàn thiện từ hiện tại đến hạn nộp, không phải tuyên bố các
giai đoạn tương lai đã thực hiện hoặc giảng viên đã đồng ý lịch bổ sung.

## 8 Các điểm kính nhờ cô góp ý

1. **Mô hình hóa:** mục tiêu tối đa Macro-F1 CV và quy tắc chọn model nhỏ theo
   epsilon có thể hiện đúng trọng tâm môn tối ưu hóa không?
2. **Protocol:** cách dùng cùng split/folds/budget, khóa lựa chọn trước test và
   giữ phần mở rộng chỉ trên development có điểm nào cần chỉnh sửa không?
3. **Tiến độ:** các commit/logs đã có và kế hoạch tuần 8–11 có thể được dùng
   làm bằng chứng bổ sung như thế nào cho các checkpoint đã qua?
4. **Ưu tiên sửa:** cô muốn phần giải thích TPE, demo tối ưu hoặc phân tích lỗi
   được bổ sung ở mức nào trước lần kiểm tra tiếp theo?

## 9 Theo dõi phản hồi và sửa đổi

Hiện chưa có phản hồi của cô được sinh viên chuyển lại cho đợt này. Khi nhận
phản hồi, ghi từng ý cùng cách xử lý, artifact thay đổi và kiểm tra sau sửa.
Không đánh dấu đã sửa hoặc đã được duyệt trước khi thực hiện/xác nhận.

| Mã phản hồi | Nội dung cô góp ý | Cách sửa và artifact | Kiểm tra sau sửa | Trạng thái |
|---|---|---|---|---|
| Chưa có | Chờ phản hồi thực tế | Chưa áp dụng | Chưa có | Chờ sinh viên chuyển lại |

## 10 Nguồn đối chiếu và công cụ hỗ trợ

- Đề bài `DE_BAI_TAP_LON_CA_NHAN_5_TUAN_GITHUB_PORTFOLIO.pdf` do sinh viên
  cung cấp, đặc biệt yêu cầu tối ưu hóa, checkpoint và GitHub portfolio.
- [README và pipeline](../README.md), [nhật ký tiến độ](../PROGRESS.md),
  [lịch sử Git công khai](https://github.com/TienMoi79/digits-rf-optimization/commits/main/).
- [Digits loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
  và [Optuna TPE](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)
  là nguồn tài liệu của thư viện; không thay thế code/logs thực nghiệm trong repo.

Codex hỗ trợ rà soát code, tổ chức pipeline/kế hoạch, chạy kiểm chứng và soạn
báo cáo tiến độ. Sinh viên cần tự kiểm tra, giải thích và demo; trạng thái này
chưa được xác nhận trong đợt báo cáo hiện tại.
