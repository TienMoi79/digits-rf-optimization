# Báo cáo rà soát và tổ chức checkpoint tuần học 8

- **Ngày cập nhật:** 27/10/2026.
- **Môn học:** Nhập môn tối ưu hóa.
- **Đề tài:** Tối ưu siêu tham số Random Forest cho nhận dạng chữ số viết tay bằng Optuna: cân bằng chất lượng dự đoán và kích thước mô hình.
- **Repo:** [TienMoi79/digits-rf-optimization](https://github.com/TienMoi79/digits-rf-optimization).
- **Mốc hiện tại:** tuần học 8; hạn nộp theo thông tin sinh viên là tuần học 11.
- **Phần hoàn thành trong đợt này:** rà soát yêu cầu, bổ sung sơ đồ pipeline, lập kế hoạch checkpoint còn lại và kiểm chứng bộ kết quả đã có.
- **Trạng thái phản hồi:** đã nhận bản biên tập do sinh viên chuyển ngày 27/10 và ghi nhận là cô gửi; đã đối chiếu với logs và sửa các điểm chưa khớp. Chưa có xác nhận duyệt checkpoint.
- **Chi tiết xử lý:** [Đối chiếu bản biên tập và sửa báo cáo](feedback-tuan-8-2026-10-27.md).

## 1 Phạm vi gửi cô kiểm tra

Báo cáo này ghi nhận công việc mới của đợt rà soát tuần học 8 và các bằng chứng kỹ thuật đã tồn tại trước đó. Dự án đã có mã nguồn và dữ liệu thực nghiệm trước ngày lập báo cáo. Phần vừa hoàn thành là tổ chức lại hệ thống checkpoint, cập nhật sơ đồ pipeline và kiểm tra tính toàn vẹn; chưa phải xác nhận đã hoàn thành đủ năm checkpoint của đề.

Kính nhờ cô xem xét tính phù hợp của bài toán mô hình hóa tối ưu, tính công bằng của pipeline thực nghiệm, cách lưu trữ bằng chứng và kế hoạch hoàn thiện từ tuần học 8 đến tuần học 11. Các nội dung giải thích sâu toán học về TPE, kịch bản demo tối ưu và tự chạy tái lập độc lập sẽ được hoàn thiện ở các tuần tiếp theo.

Theo cập nhật của sinh viên, phạm vi nộp của đề tài tập trung vào mã nguồn, notebook, artifact và demo (không nộp báo cáo Word/PDF hay slide trình chiếu). Báo cáo Markdown này được tạo định kỳ sau mỗi mốc nhằm giúp sinh viên báo cáo tiến độ và tiếp thu phản hồi từ cô.

## 2 Công việc mới hoàn thành ngày 27 tháng 10

| Nội dung | Thay đổi thực hiện | Bằng chứng |
|---|---|---|
| Đối chiếu yêu cầu | Phân định rạch ròi sản phẩm đã có, thay đổi mới và các checkpoint chưa được duyệt | `PROGRESS.md` |
| Sơ đồ pipeline | Chuẩn hóa sơ đồ phân tách Development/Test, CV, tìm kiếm, kiểm tra miền, chọn theo epsilon và refit | Mục Sơ đồ pipeline trong `README.md` |
| Kế hoạch đến hạn nộp | Lập kế hoạch hành động và bằng chứng kiểm tra cụ thể cho tuần học 8–11 | `PROGRESS.md` |
| Quy trình nhận góp ý | Sau từng phần có báo cáo `.md`; khi nhận feedback sẽ ghi nhận cách sửa và lệnh kiểm tra | `AGENTS.md`, `PROGRESS.md`, thư mục `report/` |
| Kiểm chứng hiện trạng | Chạy 100% unit tests và hai bộ kiểm chứng tính toàn vẹn của kết quả đã lưu | Kết quả lệnh tại mục 5 |

Trong đợt này không cài lại thuật toán, không chạy lại toàn bộ HPO, không thay đổi nhãn/dữ liệu và không dùng tập test để điều chỉnh tham số. Các kết quả thực nghiệm được dẫn chứng minh bạch từ lịch sử chạy trước đợt rà soát.

Sau khi nhận bản biên tập, bổ sung `src/report_facts.py` để sinh và kiểm tra
phần số liệu bên dưới trực tiếp từ logs/model. Đây là công cụ kiểm chứng báo
cáo, không phải thay đổi thuật toán tối ưu hoặc chạy thêm thực nghiệm.

## 3 Bài toán và formulation tối ưu hóa

Dữ liệu thực nghiệm là tập **Digits** từ scikit-learn (phái sinh từ benchmark UCI ML Repository), gồm 1.797 ảnh chữ số 8×8, 64 đặc trưng pixel và 10 lớp gần cân bằng, không bằng nhau tuyệt đối. Chi tiết provenance được lưu tại [`data/README.md`](../data/README.md).

### 3.1 Giai đoạn 1: Tìm kiếm cấu hình cực đại chất lượng dự đoán

Vector biến quyết định gồm 4 siêu tham số cấu trúc của Random Forest:
$$x = (n\_estimators, max\_depth, min\_samples\_leaf, max\_features) \in \Omega$$

Miền tìm kiếm rời rạc $\Omega$ gồm $7 \times 5 \times 4 \times 4 = 560$ cấu hình khả thi:
* $n\_estimators \in \{50, 75, 100, 125, 150, 175, 200\}$
* $max\_depth \in \{5, 10, 15, 20, \text{None}\}$
* $min\_samples\_leaf \in \{1, 2, 3, 4\}$
* $max\_features \in \{\text{'sqrt'}, \text{'log2'}, 0.5, 1.0\}$

Bài toán tối ưu hóa tìm kiếm:
$$\max_{x \in \Omega} f(x) = \frac{1}{3} \sum_{k=1}^3 \text{Macro-F1}_k(x) \quad \text{(trên 3 folds Stratified CV thuộc tập Development)}$$
* **Ngân sách đánh giá (Budget):** Cố định đúng 30 trials hoàn thành cho mỗi thuật toán (Random Search và Optuna TPE) trên mỗi search seed.
* Baseline là Random Forest với tham số mặc định (`n_estimators=100`, `max_depth=None`, `min_samples_leaf=1`, `max_features='sqrt'`), cố định RF seed 123 và `n_jobs=1`.

### 3.2 Giai đoạn 2: Bài toán cân bằng đa mục tiêu qua $\epsilon$-Constraint

Sau khi hoàn thành ngân sách tìm kiếm, bài toán chuyển sang bài toán ra quyết định thực tế: Giảm thiểu kích thước mô hình triển khai (đo bằng dung lượng file nhị phân tuần tự hóa `serialized_bytes`) trong khi vẫn đảm bảo chất lượng gần với nghiệm tốt nhất tìm được:

$$\min_{x \in \text{Observed}} \text{Size}(x)$$
$$\text{subject to } f(x) \ge f^*_{\text{observed}} - \epsilon$$

* Ngưỡng dung sai hiệu năng được ấn định trước là $\epsilon = 0.005$, tức 0,5 điểm phần trăm Macro-F1 tuyệt đối; không phải giảm 0,5% tương đối. Quy tắc áp dụng cho các ứng viên khả thi đã quan sát, bao gồm baseline, không phải toàn bộ miền chưa thử.

### 3.3 Protocol chống rò rỉ dữ liệu (No Test Leakage)
Toàn bộ tập dữ liệu được phân chia ngay từ đầu (Stratified Split, seed 42) thành:
* **Development Set (1.437 mẫu - 80%):** Dùng toàn bộ cho 3-fold CV trong quá trình tối ưu và refit mô hình đại diện.
* **Test Set (360 mẫu - 20%):** Không được truyền vào objective hoặc dùng để chọn cấu hình. Pipeline gốc lưu lựa chọn trong `selection.json`, sau đó đánh giá bốn đại diện định trước: default, Random tốt nhất theo CV, TPE tốt nhất theo CV và selected.

Loader đọc dataset đầy đủ từ đầu để tạo stratified split; phần tìm kiếm chỉ
nhận các lát cắt development. Việc lưu `selection.json` ghi nhận quyết định
trước đánh giá, không phải cơ chế khóa file chống chỉnh sửa. Bộ kiểm chứng
còn tái tạo dự đoán từ model đã lưu trên cùng test để đối chiếu, vì vậy không
khẳng định test chỉ được đọc hoặc dự đoán đúng một lần trong toàn bộ dự án.
Kết quả test gốc đã được xem và công bố; các lần kiểm tra sau đó không phải
đánh giá độc lập mới và không dùng để đổi cấu hình hoặc epsilon.

## 4 Bằng chứng kỹ thuật và phân tích nghiệm

| Hạng mục | Artifact kiểm chứng trong repo |
|---|---|
| Phân chia dữ liệu và cấu hình RF | `src/common.py`, `results/split_indices.json` |
| Baseline, Random Search, TPE, nghiệm lựa chọn | `src/experiment.py`, `results/baseline.json`, `results/selection.json` |
| Thí nghiệm gốc 180 search trials | `results/trials.json`, `results/trials.csv`, `results/optuna_*.csv` |
| Phân tích trade-off và độ nhạy epsilon | `src/analysis.py`, `results/epsilon_sensitivity.csv`, `figures/` |
| Nghiên cứu mở rộng 450 trials (5 seeds) | `src/robustness.py`, `results/robustness/`, `notebooks/robustness_study.ipynb` |
| Kiểm tra đường cong chẩn đoán | `src/curve_audit.py`, `results/curve_audit/`, `figures/curve_audit/` |
| Công cụ demo và tái lập | `setup.ps1`, `run_experiment.py`, `run_robustness.py`, `demo.py` |

### Phân tích ý nghĩa việc chọn nghiệm cuối

<!-- EXPERIMENT_FACTS_START -->

Các số dưới đây được sinh từ `selection.json`, `trials.json`,
`final_metrics.csv`, `search_summary.csv` và kích thước file model đã lưu.
Đây là kết quả của thí nghiệm gốc 180 trials, không trộn với nghiên cứu 450 trials.

| Đại diện | CV Macro-F1 | Model MB | Test accuracy |
|---|---:|---:|---:|
| default | 0.97061552 | 4.975993 | 97.22222% |
| random | 0.97484838 | 8.332873 | 97.50000% |
| tpe | 0.97414233 | 8.730601 | 97.22222% |
| selected | 0.97061552 | 4.975993 | 97.22222% |

MB dùng đơn vị thập phân: 1 MB = 1.000.000 byte, joblib không nén.

- Điểm cao nhất đã quan sát là **0.97484838**, thuộc **random**, seed **22**, trial **21** (trial đánh số từ 0).
- Mean của best CV qua ba search seeds: Random **0.97340**, TPE **0.97221**. Mean qua seeds khác với điểm tốt nhất của một cấu hình.
- Epsilon cố định **0.005**; ngưỡng chấp nhận **0.96984838**.
- Selected đạt CV **0.97061552**; khoảng cách đến điểm cao nhất **0.00423286**, tương đương **0.42329 điểm phần trăm**.
- File selected có **4,975,993 byte**, tức **4.975993 MB**.
- So với đại diện Random tốt nhất, selected nhỏ hơn **40.28%**. So với baseline mặc định, mức giảm là **0.00%**; selected chính là cấu hình baseline.

Test accuracy của selected là **97.22222%**, của đại diện Random là **97.50000%**, chênh **0.27778 điểm phần trăm**. Bảng test chỉ dùng để báo cáo sau khi khóa lựa chọn, không dùng chọn cấu hình.

Quy tắc epsilon chấp nhận một khoảng giảm CV để ưu tiên model nhỏ. Kết quả chỉ
xác định model nhỏ nhất trong các ứng viên đã quan sát đáp ứng ngưỡng; không
chứng minh toàn bộ miền đã bão hòa, tìm được tối ưu toàn cục, hoặc không có
đánh đổi chất lượng. Chưa có cơ sở suy rộng chi phí suy luận sang môi trường triển khai khác.

<!-- EXPERIMENT_FACTS_END -->

## 5 Kết quả kiểm tra trong đợt này

Lần rà soát đầu ngày 27/10 đạt 14 tests. Sau khi đối chiếu bản biên tập,
thêm hai kiểm tra báo cáo và chạy lại trên `.venv` (Python 3.12, `-X utf8`):

| Lệnh thực thi | Kết quả thực tế |
|---|---|
| `python -m unittest discover -s tests -v` | **16/16 tests PASSED** |
| `python -m src.validation` | **401/401 checks PASSED** (kiểm chứng toàn bộ bộ kết quả gốc) |
| `python -m src.robustness_validation` | **3.278/3.278 checks PASSED** (bảo toàn tính toàn vẹn nghiên cứu mở rộng) |
| `python -m src.report_facts --check` | **PASSED**, vùng số liệu khớp logs và kích thước model |
| `git diff --check` | **Không phát hiện lỗi whitespace** |

Các kiểm chứng bao gồm: tính độc lập của tập test, kiểm tra giới hạn ngân sách 30 trials, tính khả thi của biến quyết định, kiểm tra logic $\epsilon$-constraint và tính nhất quán giữa logs JSON và bảng tổng hợp. Không chạy lại 180 hoặc 450 trials trong đợt này.

## 6 Hiện trạng lịch sử Git và cam kết checkpoint

Trước đợt rà soát tuần 8, repository có 7 commits được tạo trên 2 ngày làm việc:
* **13/10/2026:** `f7ea277`, `73271cf` — Khởi tạo pipeline tối ưu Random Forest trên Digits và chuẩn hóa repo.
* **24/10/2026:** `2ab1a4c`, `4c1b160`, `b41a801`, `c4cecec`, `1e18603` — Mở rộng nghiên cứu 5 seeds, xử lý khóa file CI, bổ sung notebook phân tích ngân sách và audit đường cong.

Các mốc trên là bằng chứng kỹ thuật đã làm nhưng tập trung trong 2 ngày,
không tự chứng minh năm checkpoint ở năm tuần học khác nhau. Chưa có xác nhận
sinh viên đã tự giải thích/bảo vệ các mốc này. Quy trình báo cáo tuần 8–11
ghi nhận công việc và phản hồi thực tế; việc chấp thuận tiến độ cần cô xác nhận.

Sau đợt đầu còn có commit `39c4a00` về kế hoạch/pipeline và `36861ba` về bản
biên tập báo cáo ngày 27/10. Lần sửa hiện tại giữ nguyên lịch sử này, đối chiếu
lại số liệu trong bản biên tập và không gán chúng thành checkpoint tuần khác.

## 7 Kế hoạch hoàn thiện tuần học 8 đến 11

| Tuần học | Trọng tâm công việc | Sản phẩm / Bằng chứng kiểm tra | Trạng thái |
|---|---|---|---|
| **Tuần 8 (Hiện tại)** | Rà soát yêu cầu rubric; bổ sung sơ đồ pipeline; kiểm chứng toàn vẹn số liệu; lập kế hoạch checkpoint | README có pipeline; `PROGRESS.md`; báo cáo tuần 8; checks đạt | **Đã hoàn thành phần rà soát và kiểm chứng; chưa xác nhận sinh viên tự chạy/giải thích hoặc cô duyệt** |
| **Tuần 9** | Làm rõ cơ chế thuật toán Optuna TPE (phân bố $l(x)$ và $g(x)$, Expected Improvement); xây dựng kịch bản demo trực tiếp quá trình tối ưu trong 5–7 phút | Cập nhật README/Notebook phần nguyên lý TPE; kịch bản demo tìm kiếm và chọn nghiệm $\epsilon$ | Dự kiến |
| **Tuần 10** | Sinh viên tự chạy tái lập thực nghiệm vào thư mục độc lập mới; phân tích trường hợp phân loại sai (confusion matrix/failure cases); đo độ ổn định runtime | Log chạy mới độc lập; phân tích chi tiết lỗi nhận dạng chữ số; commit thực tế | Dự kiến |
| **Tuần 11 (Nộp bài)** | Kiểm tra tổng duyệt; chốt model và cấu hình; gắn Git Tag `v1.0`; sinh viên luyện tập bảo vệ và demo cá nhân | Git Release `v1.0`; đường dẫn repo public; hoàn tất bảo vệ | Dự kiến |

## 8 Các điểm kính nhờ cô góp ý

1. **Về bài toán tối ưu:** Cách thiết lập bài toán tối ưu 2 giai đoạn (Tối đa hóa Macro-F1 CV bằng TPE, sau đó tối thiểu hóa kích thước mô hình theo ràng buộc $\epsilon$-constraint với $\epsilon = 0.005$) đã đáp ứng đúng trọng tâm môn học chưa ạ?
2. **Về tính công bằng thực nghiệm:** Việc cố định ngân sách 30 trials, cô lập hoàn toàn tập test và mở rộng 5 seeds trên tập Development để đo độ ổn định có điểm nào cần điều chỉnh không ạ?
3. **Về bằng chứng checkpoint:** Các bằng chứng commit/log hiện có trong repo kết hợp với kế hoạch tuần 8–11 này có được cô chấp thuận để ghi nhận cho các mốc tiến độ không ạ?
4. **Về ưu tiên tuần 9:** Cô muốn sinh viên đào sâu thêm vào lý thuyết xác suất của TPE hay tập trung xây dựng giao diện/script demo trực quan quá trình tối ưu hơn ạ?

## 9 Bảng theo dõi phản hồi và sửa đổi

Đã nhận bản biên tập do sinh viên chuyển ngày 27/10 và ghi nhận là cô gửi.
Bản này vẫn có câu hỏi xin góp ý và bảng chờ phản hồi, chưa có xác nhận duyệt
checkpoint hoặc chỉ đạo riêng ngoài nội dung được biên tập. Bảng dưới ghi
những điểm đã tiếp thu/đối chiếu, không gán các phát hiện kiểm tra thành lời cô.

| Mã phản hồi | Nội dung cô góp ý | Cách sửa và Artifact tương ứng | Kết quả kiểm tra sau sửa | Trạng thái |
|---|---|---|---|---|
| FB-01 | Bản biên tập làm rõ bài toán hai giai đoạn và epsilon | Giữ cấu trúc; làm rõ tập ứng viên và đơn vị điểm phần trăm | Đối chiếu formulation với `choose_candidates` | Đã cập nhật nội dung |
| FB-02 | Bản biên tập ghi TPE 0.9734, gap 0.0028 và model 1.05 MB | Sửa bằng logs/model thật; thêm `src/report_facts.py` | Lệnh và kết quả sau sửa tại báo cáo xử lý riêng | Đã sửa số liệu; chưa có cô xác nhận |
| FB-03 | Các câu về test đọc một lần, bão hòa và không đánh đổi accuracy | Mô tả đúng bốn đại diện, kiểm chứng lại và giới hạn kết luận | Đối chiếu source, predictions và bảng kết quả | Đã sửa diễn giải |
| FB-04 | Bản biên tập khẳng định sinh viên đã tự kiểm chứng toàn bộ | Ghi đúng phần trợ lý đã kiểm tra, phần sinh viên chưa xác nhận | Cần sinh viên tự chạy và giải thích | Còn chờ xác nhận của sinh viên |
| FB-05 | Kế hoạch tuần 9 đào sâu TPE và demo 5–7 phút | Giữ kế hoạch, chưa đánh dấu triển khai hoặc duyệt | Sẽ kiểm tra khi thực hiện | Dự kiến |

## 10 Nguồn tài liệu tham khảo và công cụ hỗ trợ

- Đề bài và Hướng dẫn phạm vi đề BTL Cá nhân 5 tuần - Môn Nhập môn Tối ưu hóa (Khoa CNTT - ĐH Văn Lang).
- [README và pipeline](../README.md), [nhật ký tiến độ](../PROGRESS.md), [lịch sử Git công khai](https://github.com/TienMoi79/digits-rf-optimization/commits/main/).
- Scikit-learn documentation: [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) và [Digits Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).
- Optuna Documentation: [Tree-structured Parzen Estimator (TPESampler)](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html).
- Bergstra et al. (2011), *"Algorithms for Hyper-Parameter Optimization"*, NeurIPS 2011.
- Công cụ AI hỗ trợ: Codex hỗ trợ viết/rà soát code, thiết lập và chạy thực nghiệm, kiểm chứng logs/model, tổ chức pipeline và soạn/sửa báo cáo tiến độ. Các kiểm chứng tự động do trợ lý chạy không thay thế việc sinh viên tự chạy, hiểu và giải thích; phần sinh viên tự kiểm chứng toàn bộ chưa được xác nhận.
