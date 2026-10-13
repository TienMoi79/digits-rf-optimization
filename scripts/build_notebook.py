"""Create a walkthrough notebook. Run from the project root after an experiment."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md("""# Tối ưu Random Forest cho nhận dạng chữ số viết tay

Notebook trình bày bài toán, kiểm tra dữ liệu, đọc lịch sử tìm kiếm và demo mô hình.
Kết quả hiển thị lấy từ CSV/JSON do mã nguồn sinh ra. Notebook không tự chạy lại
180 trials khi bạn mở hoặc Run All.

**Câu hỏi:** với cùng 30 trials mỗi lần chạy, TPE và Random Search tìm được
cấu hình có Macro-F1 validation ra sao? Trong những cấu hình gần điểm tốt nhất,
cấu hình nào có kích thước nhỏ nhất?
"""),
    code("""from pathlib import Path
import sys, json
ROOT = Path.cwd().resolve()
if not (ROOT / 'src').exists():
    ROOT = ROOT.parent
if not (ROOT / 'src').exists():
    raise RuntimeError('Open this notebook inside the project folder.')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image
from src.common import load_config, make_data, make_model
OUTPUT = ROOT / 'results'
config = load_config(ROOT / 'configs/experiment.json')
config"""),
    md("""## Mô hình hóa bài toán

Biến quyết định `x = (n_estimators, max_depth, min_samples_leaf, max_features)`.
Miền khả thi lấy từ `configs/experiment.json`.

Hàm mục tiêu chính: `max mean(Macro-F1 trên 3 folds development)`.
Macro-F1 là trung bình F1 của 10 lớp, mỗi lớp có trọng số bằng nhau.
Ngân sách: 30 trials cho mỗi phương pháp và mỗi search seed, không pruning.

Quyết định triển khai: tối thiểu kích thước mô hình trong tập ứng viên có
`CV Macro-F1 >= CV tốt nhất - 0.005`. Đây là quy tắc epsilon định trước trên
**development**, không sử dụng điểm test. Epsilon 0.005 là 0.5 điểm phần trăm
F1 tuyệt đối, không phải kiểm định tương đương thống kê.
"""),
    code("""digits, development, test, folds = make_data(config)
print('Dataset:', digits.data.shape, 'development:', len(development), 'test:', len(test))
print('Test overlap with development:', len(set(development) & set(test)))
fig, axes = plt.subplots(2, 5, figsize=(9, 3.5))
for digit, ax in enumerate(axes.flat):
    idx = development[np.flatnonzero(digits.target[development] == digit)[0]]
    ax.imshow(digits.images[idx], cmap='gray_r', vmin=0, vmax=16)
    ax.set_title(f'Digit {digit}')
    ax.axis('off')
plt.tight_layout()
plt.show()"""),
    md("""## Baseline và thuật toán tìm kiếm

Baseline là Random Forest với tham số học mặc định, chỉ cố định `random_state`
và `n_jobs=1`. Random Search lấy mẫu độc lập trong miền tìm kiếm. TPE dùng 10
trials đầu để khởi tạo ngẫu nhiên, sau đó dùng lịch sử đánh giá để mô hình hóa
nhóm cấu hình tốt và nhóm còn lại, ưu tiên vùng có triển vọng cho trial tiếp theo.

Các phương pháp dùng cùng split, folds, metric, miền tìm kiếm và ngân sách.
RF seed giữ nguyên 123. Ba search seed 11, 22, 33 đánh giá độ biến thiên của
quá trình tìm kiếm, không đo toàn bộ bất định do cách chia dữ liệu.
"""),
    code("""baseline = json.loads((OUTPUT / 'baseline.json').read_text(encoding='utf-8'))
print('Default parameters:', baseline['params'])
print('CV fold scores:', baseline['fold_scores'])
print('Default CV Macro-F1:', baseline['cv_f1'])
trials = pd.read_csv(OUTPUT / 'trials.csv')
display(trials.groupby(['method','seed']).agg(trials=('trial','count'), best_cv_f1=('cv_f1','max')))
display(trials.head())"""),
    md("""## Chạy lại thực nghiệm

Toàn bộ pipeline nằm trong `run_experiment.py`. Lệnh:

```bash
python run_experiment.py --output results/reproduction-01
```

Mỗi lần chạy dùng một thư mục kết quả mới để giữ nguyên số liệu đã có.
Nếu thay config hoặc budget, kết quả đó thuộc một thực nghiệm khác. Chạy tất
cả cells dưới đây chỉ đọc và kiểm tra bộ kết quả đã lưu.
"""),
    code("""# Chỉ đặt True khi muốn chạy một thực nghiệm MỚI đầy đủ.
RUN_NEW_EXPERIMENT = False
if RUN_NEW_EXPERIMENT:
    from src.experiment import run_experiment
    from src.analysis import analyze
    from src.validation import validate_results
    new_output = ROOT / 'results' / 'reproduction-notebook'
    run_experiment(config, new_output)
    analyze(new_output)
    validate_results(new_output)"""),
    md("## Lịch sử tìm kiếm và độ ổn định\nDải màu thể hiện ±1 độ lệch chuẩn qua ba search seed. Đây không phải khoảng tin cậy."),
    code("""display(pd.read_csv(OUTPUT / 'search_summary.csv'))
display(pd.read_csv(OUTPUT / 'best_per_seed.csv'))
display(Image(filename=str(ROOT / 'figures/search_history.png')))"""),
    md("""## Trade off chất lượng và kích thước

Đồ thị sử dụng **điểm CV**, đo kích thước joblib không nén theo MB thập phân.
Đường Pareto chỉ là biên của các cấu hình đã thử, không chứng minh tối ưu toàn cục.
Thời gian dự đoán đo trên batch development cố định 256 ảnh, 3 lần warmup và
15 lần lặp. `ms_per_sample` là thời gian batch chia số mẫu, không phải latency
của một request riêng lẻ. Thời gian có thể thay đổi theo máy và tải hệ thống.
"""),
    code("""display(Image(filename=str(ROOT / 'figures/quality_size_tradeoff.png')))
decision = json.loads((OUTPUT / 'selection.json').read_text(encoding='utf-8'))
print('Best CV score:', decision['best_cv_f1'])
print('Minimum acceptable CV score:', decision['minimum_acceptable_cv_f1'])
print('Chosen parameters:', decision['selected_params'])
display(pd.read_csv(OUTPUT / 'epsilon_sensitivity.csv'))
display(Image(filename=str(ROOT / 'figures/epsilon_sensitivity.png')))"""),
    md("""## Đánh giá test sau khi chốt cấu hình

Mặc định, cấu hình tốt nhất của Random Search, cấu hình tốt nhất của TPE và
cấu hình nhỏ được chọn là bốn đại diện đã định trước. Mã nguồn chốt các đại
diện bằng CV rồi mới đánh giá test. Không đổi cấu hình dựa trên bảng test.
Tất cả mô hình huấn luyện trên development 80%, không refit lên test.
"""),
    code("""display(pd.read_csv(OUTPUT / 'final_metrics.csv'))
display(Image(filename=str(ROOT / 'figures/confusion_matrix.png')))
display(pd.read_json(OUTPUT / 'classification_selected.json').T)
key_results = json.loads((OUTPUT / 'key_results.json').read_text(encoding='utf-8'))
key_results"""),
    md("## Demo mô hình đã lưu\nĐây là demo trên mẫu có sẵn trong dataset. Khả năng xử lý ảnh điện thoại hoặc chữ viết tay ngoài dataset chưa được kiểm chứng."),
    code("""import joblib
model = joblib.load(OUTPUT / 'models/selected.joblib')
example = test[0]
predicted = int(model.predict(digits.data[[example]])[0])
plt.figure(figsize=(3,3))
plt.imshow(digits.images[example], cmap='gray_r', vmin=0, vmax=16)
plt.title(f'True {digits.target[example]}, prediction {predicted}')
plt.axis('off')
plt.show()
print('Parameters:', {k:model.get_params()[k] for k in decision['selected_params']})"""),
    md("""## Kiểm chứng và giới hạn

Kiểm chứng kiểm tra split, ngân sách, miền khả thi, quy tắc chọn mô hình,
khả năng tái tạo dự đoán từ model và tính lại metric từ CSV.
Chỉ có một dataset, một holdout và ba search seed. Chênh lệch F1 nhỏ có thể
do mẫu hữu hạn. Folds có liên quan vì cùng development set. Ba search seed
không đủ để kết luận một thuật toán luôn tốt hơn. Bộ dữ liệu vốn là subset
test của UCI do sklearn đóng gói, nhưng trong nghiên cứu này được chia lại
độc lập và tên gọi đó không làm thay đổi protocol đang dùng.
"""),
    code("""from src.validation import validate_results
checks = validate_results(OUTPUT)
print('Number of passed checks:', len(checks))
manifest = json.loads((OUTPUT / 'run_manifest.json').read_text(encoding='utf-8'))
print('Actual runtime in seconds:', round(manifest['runtime_seconds'], 2))
print('Library versions:', manifest['versions'])"""),
    md("""## Nguồn và công cụ hỗ trợ

- Dataset: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
- UCI: https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits
- Random Forest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- Optuna: https://optuna.readthedocs.io/en/stable/reference/samplers/index.html

Codex hỗ trợ tổ chức mã nguồn, viết notebook và kiểm tra pipeline. Kết quả
đến từ thực nghiệm Python đã chạy. Người thực hiện cần tự chạy lại và giải
thích được biến quyết định, objective, protocol, trade off và quyết định cuối.
"""),
]
nb.metadata = {"kernelspec": {"display_name": "Python (Digits RF)", "language": "python", "name": "digits-rf"},
               "language_info": {"name": "python", "version": "3.12.5"}}
(ROOT / 'notebooks').mkdir(exist_ok=True)
nbf.write(nb, ROOT / 'notebooks/optimization_walkthrough.ipynb')
print('Created notebooks/optimization_walkthrough.ipynb')
