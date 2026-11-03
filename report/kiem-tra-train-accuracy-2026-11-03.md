# Kiểm tra độc lập kết quả train accuracy 100 phần trăm

- Ngày kiểm tra: 03/11/2026.
- Lý do: sinh viên chuyển thắc mắc của cô về các giá trị train 100% trong báo cáo Word.
- Kết luận kiểm tra: các cấu hình cố định đều dự đoán đúng 1.437/1.437 mẫu development sau khi fit trên chính tập đó. Đây là in-sample accuracy, không phải độ chính xác trên mẫu chưa được model học.

## 1 Đã kiểm tra gì

Script `scripts/audit_train_accuracy.py` không dùng helper dựng RF hoặc chia
dữ liệu của project. Nó gọi trực tiếp `load_digits`, `train_test_split`,
`StratifiedKFold` và `RandomForestClassifier` từ scikit-learn, dùng tham số
gốc đã lưu và cố định seed.

1. Tải UCI ZIP, đọc `optdigits.tes`, đối chiếu **từng pixel và nhãn** với dữ liệu sklearn.
2. Tái tạo split/folds và đối chiếu exact indices đã công bố; train/validation/test không trùng indices.
3. Đặt các bản sao development X/y thành read-only, fit ba cấu hình phân biệt rồi dự đoán development.
4. Đếm trực tiếp `count_nonzero(prediction == y)`, kiểm tra bằng `accuracy_score`; prediction là mảng riêng và khớp argmax của xác suất model, không gán nhãn thật làm prediction.
5. Fit lại ba CV folds cho mỗi cấu hình, chỉ chấm mẫu ngoài phần fit của fold; F1 từng fold phải khớp log gốc.
6. Đối chiếu dữ liệu/nhãn trước và sau fit, hash artifact trước/sau. Không dự đoán mới trên features test.

## 2 Kết quả đếm thực tế

| Đại diện | Train đúng/tổng | Train sai | Train accuracy | CV OOF đúng/tổng | CV OOF accuracy |
|---|---:|---:|---:|---:|---:|
| RF mặc định | 1.437/1.437 | 0 | 100% | 1.395/1.437 | 97,07724% |
| RF Random | 1.437/1.437 | 0 | 100% | 1.401/1.437 | 97,49478% |
| RF TPE | 1.437/1.437 | 0 | 100% | 1.400/1.437 | 97,42519% |
| Selected | 1.437/1.437 | 0 | 100% | 1.395/1.437 | 97,07724% |

Selected trùng cấu hình RF mặc định, không phải một mô hình độc lập thứ tư.
Các folds có cùng 479 mẫu validation nên accuracy OOF bằng mean accuracy
của ba folds; F1 OOF nói chung khác mean F1, không thay objective bằng F1 OOF.

Test accuracy gốc của selected vẫn là **350/360 = 97,22222%**, lấy từ
`results/test_predictions.csv`. Các lần audit không coi test này là đánh giá
độc lập mới và không dùng để chỉnh tham số hoặc epsilon.

## 3 Vì sao train có thể đạt 100 phần trăm

Ở đây train accuracy được tính theo:

```python
model.fit(X_train, y_train)
prediction = model.predict(X_train)
train_accuracy = (prediction == y_train).mean()
```

Model được phép dùng `y_train` khi fit; đo lại trên chính X_train đánh giá
mức khớp mẫu đã học. Forest có các cây đủ linh hoạt có thể khớp hết nhãn
train. Baseline trong repo dùng `max_depth=None`, `min_samples_leaf=1`.
Tài liệu scikit-learn mô tả cấu hình mặc định tạo các cây phát triển đầy đủ,
không pruning; cấu hình tuned khác cũng có thể khớp toàn bộ tập mẫu hữu hạn.

Nguồn: [RandomForestClassifier, tham số max_depth và phần Notes](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

Vì vậy, train 100% không thể hiện mô hình đạt khả năng tổng quát hóa hoàn
hảo. Khoảng cách train–validation cần được báo cáo và theo dõi; chỉ một
train score cũng không đủ để định lượng mức overfitting hoặc loại trừ mọi
sai lệch khác. Việc “train không bao giờ được 100%” không phải quy tắc đúng
cho mọi thuật toán và mọi dataset. Cần thống nhất với cô cách dùng thuật ngữ
train in-sample, validation chưa tham gia fit của fold và test giữ riêng.

## 4 Bằng chứng để tự kiểm tra

- `results/train_accuracy_audit_2026-11-03/counts.csv`: số đếm chưa làm tròn.
- `results/train_accuracy_audit_2026-11-03/per_sample_predictions.csv`: global index, nhãn thật, train prediction và validation OOF prediction cho từng mẫu/đại diện.
- `results/train_accuracy_audit_2026-11-03/audit.json`: đối chiếu UCI, split, dữ liệu không đổi và các kết quả kiểm tra.
- `scripts/audit_train_accuracy.py`: code kiểm tra độc lập để đọc lại hoặc chạy ở thư mục đầu ra mới.

Lệnh đã thực hiện:

```powershell
.\.venv\Scripts\python.exe -X utf8 -u scripts/audit_train_accuracy.py
```

Script từ chối ghi đè thư mục audit đã tồn tại. Nếu cần chạy lại, dùng tên
đầu ra mới trong script để giữ bằng chứng cũ. Không đổi dữ liệu, nhãn hoặc
siêu tham số để ép train accuracy xuống dưới 100%. Báo cáo Word và model gốc
được giữ nguyên trong lần audit này; chưa có xác nhận cô chấp nhận giải thích.
