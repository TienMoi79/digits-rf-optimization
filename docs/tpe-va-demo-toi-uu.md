# TPE và kịch bản demo tối ưu

Tài liệu chuẩn bị cho checkpoint tuần học 9, thực hiện ngày 01/11/2026.
Mục tiêu là giải thích thuật toán thực sự dùng trong repo và demo được chuỗi
đề xuất cấu hình → đánh giá CV → ghi lịch sử → chọn theo epsilon.

## 1 Đầu vào và đầu ra của từng phần

| Thành phần | Đầu vào | Đầu ra |
|---|---|---|
| Sampler Random/TPE | Miền siêu tham số, seed; TPE còn dùng lịch sử các trial trong study | Một cấu hình `x` cho trial tiếp theo |
| Objective của project | `x`, development X/y, ba folds và RF seed | Một số: mean CV Macro-F1; kèm log các fold, runtime và model bytes |
| Study | Sampler, objective, hướng maximize, số trials | Lịch sử cấu hình và CV score; best CV đã quan sát |
| Quy tắc epsilon | Các ứng viên đã quan sát, CV score, model bytes, epsilon 0.005 | Ứng viên nhỏ nhất đủ ngưỡng; không dùng test score |

Random Forest học các cây; TPE chọn **siêu tham số của forest**. TPE không
chỉnh từng trọng số/pixel và không trực tiếp tối ưu log loss. Hàm mục tiêu
của study là CV Macro-F1; kích thước dùng ở bước quyết định sau tìm kiếm.

## 2 TPE xây dựng đề xuất như thế nào

Trong thí nghiệm gốc, TPE có 10 trial khởi tạo ngẫu nhiên trong ngân sách 30.
Sau đó, nó dùng các điểm đã quan sát để phân biệt vùng có chất lượng tốt và
phần còn lại. Cách trình bày kinh điển dùng loss cần tối thiểu hóa; đặt
`y = 1 - CV_MacroF1` để liên hệ với bài toán maximize của repo:

```text
l(x) = p(x | y < y_star)       nhóm loss tốt
g(x) = p(x | y >= y_star)      nhóm còn lại
gamma = P(y < y_star)
p(x) = gamma*l(x) + (1-gamma)*g(x)

EI(x) = E[max(y_star - y, 0) | x]
      = C*l(x) / [gamma*l(x) + (1-gamma)*g(x)]
      = C / [gamma + (1-gamma)*g(x)/l(x)]

C = integral over y<y_star of (y_star-y)*p(y) dy
```

Với ngưỡng và mô hình mật độ cố định, `C` không phụ thuộc `x`; tăng tỷ số
`l(x)/g(x)` làm tăng EI trong mô hình TPE này. `y_star` là ngưỡng phân nhóm,
không bắt buộc bằng loss tốt nhất. Trong code, Optuna dùng `direction="maximize"`
trực tiếp với F1, không có bước đổi loss thủ công.

Optuna lấy một tập ứng viên từ mô hình tốt rồi chấm acquisition bằng tỷ số
mật độ (trên thang log trong implementation). Biến số và biến categorical
có cách ước lượng phù hợp; không diễn giải mọi categorical thành Gaussian.
TPE là thuật toán xác suất nên trial mới vẫn có thể kém hơn trial trước.
Đường **best so far** dùng cumulative maximum nên không giảm theo định nghĩa;
đó không phải điểm từng trial đã được làm mượt.

Nguồn: [Bergstra và cộng sự 2011, mục 4](https://papers.nips.cc/paper_files/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf)
và [Optuna TPESampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html).

## 3 Tham số và lý do chọn trong project

| Tham số | Thí nghiệm gốc | Live demo nhỏ |
|---|---|---|
| Budget | 30/method/seed | 12/method, một seed |
| TPE startup | 10 | 5, còn 7 trial theo TPE |
| Search seeds | 11,22,33 | 11 |
| RF seed / folds | 123 / ba folds cố định | Giữ nguyên |
| `n_ei_candidates` | Không truyền; mặc định Optuna 5.0.0 là 24 | Ghi tường minh 24 |
| `multivariate` | Không truyền; Optuna 5.0.0 tự chọn theo số objective | Ghi tường minh True |
| Profile repeats | 15 | 3 để demo gọn; không trộn runtime với thí nghiệm gốc |

Đã kiểm tra phiên bản cài trong `.venv` là Optuna 5.0.0. Với phiên bản này,
`multivariate=None` tự chọn True cho single-objective. Vì vậy không mô tả
thí nghiệm gốc là đã cố định `multivariate=False`. Demo ghi rõ cấu hình để
tránh phụ thuộc thay đổi mặc định khi người đọc dùng phiên bản khác.
Xem [mô tả multivariate trong Optuna](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html).

Miền của project trộn số nguyên và lựa chọn categorical, CV tốn nhiều hơn
việc gợi ý một cấu hình. Đây là lý do thực dụng để thử TPE; Random Search
là đối chứng đơn giản cùng ngân sách. Lựa chọn này không bảo đảm TPE thắng.
Các kết quả gốc/mở rộng đã cho thấy thứ hạng có thể đổi theo seeds/startup.
Tên “tree-structured” của TPE không có nghĩa TPE là cây trong Random Forest.
Miền bốn biến hiện tại cũng không có nhánh siêu tham số điều kiện.

## 4 Hai chế độ demo

### Replay từ log cũ

```powershell
.\.venv\Scripts\python.exe -X utf8 demo_optimization.py replay --seed 11 --checkpoint 5
.\.venv\Scripts\python.exe -X utf8 demo_optimization.py replay --seed 11 --checkpoint 15
.\.venv\Scripts\python.exe -X utf8 demo_optimization.py replay --seed 11 --checkpoint 30
```

Replay không fit, không ghi file, không nạp model hoặc bảng test. Checkpoint
5 chỉ xét baseline và năm trial đầu **của mỗi phương pháp ở seed 11**; không
đưa trial tương lai vào lựa chọn. Đây là quyết định minh họa trên một prefix,
có thể khác lựa chọn gốc vốn gộp các search seeds. Bảng hiển thị số trial từ 1;
JSON trial vẫn đánh số từ 0.

### Chạy một tìm kiếm nhỏ trực tiếp

```powershell
.\.venv\Scripts\python.exe -X utf8 demo_optimization.py live --trials 12 --startup 5 --seed 11 --output results/week9_demo/run-moi
```

Luôn dùng tên thư mục mới; script từ chối ghi đè và chỉ cho ghi dưới
`results/week9_demo/`. Live demo dùng lại development/folds gốc, fit baseline
và 12 trials/phương pháp. Mỗi trial chạy ba CV folds và refit development
để đo model bytes. Có tổng 24 search trials, không tính baseline.

Outputs: `trials.json`, `demo_selection.json`, `search_history.png`,
`run_manifest.json` với phiên bản, thời điểm, cấu hình, hashes bảo toàn artifact.
Trong manifest, `config` kế thừa cấu hình RF/CV gốc và đổi số profiling repeats;
ngân sách/startup/seed thực thi demo được ghi riêng ở `trials_per_method`,
`startup_trials`, `seed`, không lấy các giá trị search gốc bên trong `config`.
Chỉ lưu quyết định của demo, không thay `results/selection.json` hoặc model
bàn giao và không đánh giá held-out. Một seed và ngân sách nhỏ không dùng
để kết luận ưu thế thuật toán, độ ổn định hoặc tốc độ. Runtime phụ thuộc máy;
không cam kết mọi lần đều vừa đúng thời lượng thuyết trình.

Kiểm tra lại log demo mà không retrain:

```powershell
.\.venv\Scripts\python.exe -X utf8 demo_optimization.py validate --output results/week9_demo/run-2026-11-01
```

## 5 Kịch bản trình bày 5 đến 7 phút

| Thời gian dự kiến | Nội dung và thao tác |
|---|---|
| 0–1 phút | Nêu x, objective CV, ngân sách, test giữ riêng; phân biệt RF và TPE |
| 1–2 phút | Giải thích startup, nhóm tốt/còn lại, tỷ số l/g và ý nghĩa EI |
| 2–3 phút | Chạy replay checkpoint 5 rồi 15: đọc trial score và best so far, chỉ rõ startup |
| 3–5 phút | Nếu thời gian/máy cho phép chạy live vào thư mục mới; nếu không dùng logs live có sẵn và nói rõ đang xem kết quả đã chạy |
| 5–6 phút | Chỉ ngưỡng best CV trừ epsilon và model nhỏ nhất đủ ngưỡng; giải thích có đánh đổi |
| 6–7 phút | Nêu giới hạn một seed, CV selection bias, không tuning test; trả lời câu hỏi |

## 6 Câu hỏi tự kiểm tra

1. Vì sao model có CV cao nhất chưa chắc được chọn cho triển khai?
2. Vì sao best-so-far không giảm trong khi điểm trial có thể giảm?
3. Với startup=10, checkpoint=5 đã quan sát hành vi TPE thích nghi chưa?
4. Vì sao cần dùng cùng folds/budget và nhiều seeds để so sánh?
5. Nếu thay epsilon sau khi xem test, kết luận nào sẽ không còn hợp lệ?
6. Replay một seed khác gì lựa chọn gốc gộp ba seeds?

Sinh viên cần tự thực hiện và giải thích. Code chạy được không tự xác nhận
đã bảo vệ checkpoint; tình trạng cô duyệt được theo dõi trong báo cáo riêng.
