"""Build the walkthrough for the five-seed development-only extension."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb = nbf.v4.new_notebook()
nb.cells = [
    md("""# So sánh ngân sách tìm kiếm và ảnh hưởng khởi tạo TPE

Phần mở rộng nghiên cứu Random Search, TPE startup 5 và TPE startup 10 trên
5 search seeds. Mỗi phương pháp dùng 30 trials mỗi seed, tổng 450 trials.
Notebook đọc logs đã lưu, không tự chạy lại HPO.

Các thí nghiệm này chỉ sử dụng development. Kết quả test, lựa chọn mô hình
và model của thí nghiệm ban đầu được giữ nguyên. Không dùng bảng test cũ để
chọn cấu hình, epsilon hoặc phương pháp cho phần mở rộng.
"""),
    code("""from pathlib import Path
import sys, json
import pandas as pd
from IPython.display import display, Image
ROOT = Path.cwd().resolve()
if not (ROOT / 'src').exists():
    ROOT = ROOT.parent
if not (ROOT / 'src').exists():
    raise RuntimeError('Open the notebook inside the project folder.')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / 'results/robustness'
FIGURES = ROOT / 'figures/robustness'
if not (OUTPUT / 'budget_summary.csv').exists():
    raise RuntimeError('Run python run_robustness.py first, using a fresh output folder if needed.')
manifest = json.loads((OUTPUT / 'run_manifest.json').read_text(encoding='utf-8'))
assert manifest['status'] == 'completed'
assert manifest['test_evaluations'] == 0
print('Search seeds:', manifest['plan']['search_seeds'])
print('Completed trials:', manifest['completed_trials'])
print('Additional test evaluations:', manifest['test_evaluations'])
print('Actual runtime seconds:', round(manifest['runtime_seconds'], 2))
display(pd.Series(manifest['versions'], name='Installed version'))"""),
    md("""## Thiết kế thực nghiệm có kiểm soát

Giữ nguyên development 1.437 mẫu, ba folds đã lưu, miền 560 cấu hình,
Macro-F1 và RF seed 123. Chỉ số startup của TPE là yếu tố thay đổi trong
ablation. Random Search làm mốc so sánh. TPE startup 5 dùng 5 trials đầu lấy
mẫu ngẫu nhiên, TPE startup 10 dùng 10 trials đầu lấy mẫu ngẫu nhiên.

Tất cả studies chạy mới trong cùng phiên, tuần tự, n_jobs=1, không pruning.
Thứ tự ba phương pháp xoay vòng theo seed. Không ghép thời gian từ lần chạy
ngày trước. Phần mở rộng chỉ thực hiện CV, không refit để đo model size/latency,
nên thời gian này không so trực tiếp với tổng thời gian pipeline ban đầu.

Mean và SD mô tả biến thiên qua search seeds trên cùng dữ liệu và folds.
Chúng không phải confidence interval hoặc bằng chứng tổng quát cho dataset khác.
"""),
    code("""display(pd.DataFrame(manifest['schedule']))
print('Original protocol fingerprint:', manifest['base_config_sha256'])
print('Study plan fingerprint:', manifest['plan_sha256'])
trials = pd.read_csv(OUTPUT / 'trials.csv')
display(trials.groupby(['method','seed']).agg(trials=('trial','count'), best_cv_f1=('cv_f1','max')))"""),
    md("""## Chất lượng theo ngân sách số trials

Mỗi checkpoint chỉ xét các trials đã hoàn tất trong prefix 10, 20 hoặc 30.
Không lấy cấu hình tốt của trial 30 để báo kết quả cho ngân sách 10. Phần
startup cũng tính trong 30 trials. Đường thể hiện điểm tốt nhất đã quan sát,
dải màu là một độ lệch chuẩn qua năm search seeds.
"""),
    code("""budget = pd.read_csv(OUTPUT / 'budget_summary.csv')
display(budget)
display(Image(filename=str(FIGURES / 'search_budget.png')))
last = budget[budget.budget == manifest['plan']['trials_per_method']].sort_values('best_cv_f1_mean', ascending=False)
print('Highest mean development CV score at the full budget:', last.iloc[0]['method'])
print('This does not change the previously published deployment model.')"""),
    md("""## Chất lượng theo ngân sách thời gian

Elapsed search time bao gồm gợi ý tham số, CV, bookkeeping và overhead ghi log
trước đó. Đồng hồ bắt đầu riêng cho từng study. Các phương pháp dùng cùng miền
thời gian: bắt đầu khi mọi run đã hoàn tất ít nhất một trial và kết thúc trước
khi bất kỳ run nào hết ngân sách 30 trials.

Ở mỗi thời điểm, chỉ dùng score của trials đã hoàn tất. Không nội suy điểm,
không điền ngược score trước trial đầu và không kéo dài đường sau khi study kết
thúc. Thời gian vẫn chịu ảnh hưởng của CPU và tải nền trên máy. Bảng bên dưới
giữ lại cả những run chậm. Xem mean, SD và median cùng logs từng seed trước khi
kết luận về tốc độ. Logs không đủ để quy toàn bộ biến động thời gian cho thuật toán.
"""),
    code("""time_window = json.loads((OUTPUT / 'time_comparison.json').read_text(encoding='utf-8'))
display(time_window)
display(Image(filename=str(FIGURES / 'time_budget.png')))
timed = pd.read_csv(OUTPUT / 'time_budget_per_seed.csv')
if not timed.empty:
    display(timed.groupby('method').tail(2))
cost = pd.read_csv(OUTPUT / 'budget_per_seed.csv')
display(cost[cost.budget == manifest['plan']['trials_per_method']][['method','seed','elapsed_seconds','cv_seconds']])"""),
    md("""## Ablation số trials khởi tạo TPE

Ghép TPE startup 5 và startup 10 theo cùng search seed. Delta dương nghĩa là
startup 5 có best CV Macro-F1 cao hơn ở ngân sách 30, delta âm nghĩa là startup
10 cao hơn. Cùng seed không đồng nghĩa hai phương pháp thử toàn bộ cùng cấu
hình sau giai đoạn khởi tạo. Đây là so sánh mô tả, không tuyên bố có ý nghĩa
thống kê từ chỉ năm cặp.
"""),
    code("""pairs = pd.read_csv(OUTPUT / 'startup_ablation_pairs.csv')
ablation = json.loads((OUTPUT / 'startup_ablation_summary.json').read_text(encoding='utf-8'))
display(pairs)
display(ablation)
print('Mean paired F1 difference, percentage points:', round(100*ablation['mean_delta_cv_f1'], 4))
display(Image(filename=str(FIGURES / 'startup_ablation.png')))"""),
    md("""## Kiểm chứng kết quả và bảo toàn thí nghiệm gốc

Các kiểm tra đối chiếu log JSON/CSV, mean và SD, ngân sách mỗi seed, cặp
ablation và các score trong đường thời gian. SHA-256 xác nhận các logs,
predictions, bảng test, selection và model gốc không thay đổi. Văn bản chuẩn
hóa CRLF thành LF để việc checkout Git trên Windows/Linux không tạo báo lỗi giả.
Model nhị phân được kiểm tra chính xác từng byte.
"""),
    code("""from src.robustness_validation import validate_robustness
checks = validate_robustness(OUTPUT)
print('Passed integrity checks:', len(checks))
print('Preserved original artifacts:', len(manifest['frozen_artifacts']))"""),
    md("""## Chạy lại và giới hạn kết luận

```bash
python run_robustness.py --output results/robustness-reproduction-01
```

Một lần chạy đầy đủ có 450 trials x 3 folds = 1.350 lần fit CV, cộng baseline
3 folds. Dùng thư mục mới, không ghi đè các kết quả đã lưu. Thư mục mặc định
`results/robustness` là bộ kết quả công bố; các lần chạy lại tự đặt figures cạnh
logs của lần chạy đó. Pipeline lưu JSON sau mỗi trial nhưng không hỗ trợ resume.

Kết luận chỉ áp dụng cho dataset, folds, phiên bản thư viện, miền tìm kiếm và
budget hiện tại. Chưa kiểm tra độ nhạy với cách chia dữ liệu hoặc RF seed.
Không dùng thí nghiệm development bổ sung để tuyên bố có kết quả test mới.

Nguồn: https://optuna.readthedocs.io/en/stable/reference/samplers/index.html
và https://docs.github.com/en/actions/tutorials/build-and-test-code/python.
Codex hỗ trợ code và kiểm chứng. Số liệu hiển thị sinh từ logs thực nghiệm thật.
"""),
]
nb.metadata = {"kernelspec":{"display_name":"Python (Digits RF)","language":"python","name":"digits-rf"},
               "language_info":{"name":"python","version":"3.12.5"}}
nbf.validate(nb)
nbf.write(nb, ROOT / 'notebooks/robustness_study.ipynb')
print('Created notebooks/robustness_study.ipynb')
