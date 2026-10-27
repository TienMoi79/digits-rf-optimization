# Báo cáo tiếp nhận bản biên tập và đối chiếu số liệu tuần học 8

- **Ngày xử lý:** 27/10/2026.
- **Nguồn tiếp nhận:** văn bản sinh viên chuyển trong cuộc trò chuyện, ghi nhận là cô gửi nhận xét.
- **Bản được cập nhật:** [Báo cáo checkpoint tuần 8](checkpoint-tuan-8-2026-10-27.md).
- **Trạng thái:** đã rà soát nội dung và sửa báo cáo theo bằng chứng; chưa có xác nhận cô duyệt checkpoint.

## 1 Nội dung đã nhận và phạm vi xử lý

Văn bản nhận được là một bản biên tập lại báo cáo: diễn đạt đề tài rõ hơn,
tách formulation thành hai giai đoạn, bổ sung epsilon-constraint và dự kiến
đào sâu TPE/demo ở tuần 9. Bản này vẫn có các câu hỏi xin góp ý và mục chờ
phản hồi; không có kết luận riêng xác nhận checkpoint đã đạt hoặc kế hoạch
tuần 8–11 đã được chấp thuận.

Cấu trúc hai giai đoạn được giữ. Các giá trị, mô tả protocol và kết luận
được đối chiếu lại với source/logs. Những điểm sai khác trong bản biên tập
không được dùng để thay đổi dữ liệu hoặc kết quả thí nghiệm cho khớp báo cáo.

Bản báo cáo trước sửa đã có trong Git tại
[`36861ba`](https://github.com/TienMoi79/digits-rf-optimization/blob/36861ba/report/checkpoint-tuan-8-2026-10-27.md).
Lịch sử đó được giữ nguyên. Bảng dưới ghi rõ phần tiếp thu từ bản nhận được
và phần phát hiện khi kiểm tra; không trình bày các phát hiện này như lời cô.

## 2 Bảng đối chiếu và xử lý

| Mã | Nội dung trong bản nhận được | Xử lý theo bằng chứng | Trạng thái |
|---|---|---|---|
| FB-01 | Tách bài toán tối đa CV và chọn model nhỏ qua epsilon-constraint | Giữ cấu trúc; ghi rõ tập ứng viên khả thi đã quan sát, có cả baseline; epsilon là 0,5 điểm phần trăm tuyệt đối | Đã cập nhật |
| FB-02 | Ghi điểm cao nhất qua TPE khoảng 0.9734 | Điểm cao nhất thí nghiệm gốc là 0.97484838 của Random, seed 22, trial 21; 0.97340 là mean best CV qua ba seeds của Random | Đã sửa |
| FB-03 | Ghi selected khoảng 1.05 MB | File model không nén là 4.975.993 byte, tương đương 4.975993 MB thập phân | Đã sửa |
| FB-04 | Ghi chênh lệch CV đến đỉnh khoảng 0.0028 | Chênh lệch đúng là 0.00423286; ngưỡng CV đúng là 0.96984838 | Đã sửa |
| FB-05 | Kết luận không gian đã bão hòa và không đánh đổi accuracy | Bỏ kết luận bão hòa; chỉ kết luận trên ứng viên quan sát. Selected có test accuracy 97.22222%, đại diện Random 97.50000%; không tuyên bố chất lượng bằng nhau | Đã sửa |
| FB-06 | Ghi test bị cô lập hoàn toàn và chỉ nạp một lần cho một model cuối | Nêu loader đọc dataset đầy đủ để split; objective chỉ nhận development; pipeline đánh giá bốn đại diện định trước sau khi lưu quyết định. Validation tái tạo dự đoán từ model cũ | Đã sửa mô tả |
| FB-07 | Ghi 10 lớp cân bằng | Đổi thành gần cân bằng, không bằng nhau tuyệt đối; không thay dữ liệu | Đã sửa cách diễn đạt |
| FB-08 | Ghi sinh viên đã tự kiểm chứng toàn bộ code/thực nghiệm | Nêu đúng phần Codex hỗ trợ chạy và kiểm chứng; việc sinh viên tự chạy/giải thích chưa được xác nhận | Chờ sinh viên tự thực hiện/xác nhận |
| FB-09 | Nêu kế hoạch TPE sâu hơn và demo 5–7 phút ở tuần 9 | Giữ là kế hoạch; không đánh dấu đã triển khai hoặc cô đã đồng ý ưu tiên | Dự kiến |
| FB-10 | Bản vẫn đặt câu hỏi về việc ghi nhận checkpoint | Giữ trạng thái chưa có xác nhận duyệt; không đánh dấu đủ năm tuần từ các commit cũ | Chờ cô xác nhận |

Các số ở FB-02 đến FB-05 thuộc thí nghiệm gốc **180 trials**, không lấy mean
của nghiên cứu mở rộng 450 trials để thay vào quy tắc chọn model gốc.
Selected trùng baseline nên mức giảm kích thước so với baseline là 0%; mức
40,28% là so với đại diện Random có CV cao nhất, không phải so với baseline.

## 3 Bằng chứng sử dụng

- `results/trials.json`: trial tốt nhất và các ứng viên đã quan sát.
- `results/selection.json`: epsilon, ngưỡng, cấu hình và đại diện đã khóa.
- `results/search_summary.csv`: mean của best CV qua search seeds theo phương pháp.
- `results/final_metrics.csv`: CV, test metrics và kích thước các đại diện.
- `results/models/selected.joblib`: số byte thực của model đã lưu.
- `src/experiment.py`: objective chỉ dùng development, thứ tự ghi quyết định và đánh giá bốn đại diện.
- `src/validation.py`: tính lại metric từ predictions và tái tạo dự đoán model cũ để đối chiếu.

Việc kiểm chứng lại model cũ trên test không phải một lần chọn cấu hình mới,
nhưng có nghĩa không được nói test chỉ được đọc đúng một lần trong toàn bộ
vòng đời dự án. Kết quả test đã biết tiếp tục được giữ nguyên, không dùng
để tuning hay đổi epsilon.

## 4 Công cụ bổ sung để tránh chép sai số liệu

Thêm `src/report_facts.py` đọc các artifact đã lưu và sinh vùng số liệu giữa
hai marker `EXPERIMENT_FACTS_START` / `EXPERIMENT_FACTS_END` trong báo cáo tuần 8.
Công cụ đối chiếu quyết định epsilon, kích thước file thực và số liệu bảng cuối.
Nó không fit model, không đọc features test, không ghi lại logs thực nghiệm.

```powershell
# Chỉ đối chiếu vùng số liệu
.\.venv\Scripts\python.exe -X utf8 -m src.report_facts --check

# Đồng bộ lại đúng vùng số liệu từ logs, giữ các mục nhận xét khác
.\.venv\Scripts\python.exe -X utf8 -m src.report_facts --write
```

Thêm hai kiểm tra hồi quy: từ chối kích thước bị sửa thành 1.05 MB; sửa lại
vùng ghi sai phương pháp thắng mà giữ nguyên nhận xét bên ngoài vùng đó.
Lệnh kiểm tra báo cáo cũng được đưa vào tiện ích cập nhật GitHub và CI.
Phạm vi tự động chỉ là vùng số liệu được sinh, không phải xác nhận mọi câu
diễn giải hoặc trạng thái cô duyệt trong toàn bộ báo cáo.

## 5 Kiểm tra sau sửa

| Lệnh | Kết quả thực tế |
|---|---|
| `python -m unittest discover -s tests -v` | 16 tests đạt, gồm hai kiểm tra số liệu báo cáo mới |
| `python -m src.validation` | 401 kiểm tra kết quả gốc đạt |
| `python -m src.robustness_validation` | 3.278 kiểm tra đạt; kết quả gốc được bảo toàn |
| `python -m src.report_facts --check` | Vùng số liệu báo cáo khớp logs và kích thước model |
| `git diff --check` | Không phát hiện lỗi whitespace trong thay đổi được theo dõi |

Không chạy lại toàn bộ 180/450 trials trong lần xử lý bản biên tập này.
Không có dữ liệu, nhãn, trial metrics, model hoặc predictions mới được tạo
để thay bộ kết quả cũ. Các kết quả kiểm tra trên là do trợ lý chạy trong repo;
không dùng chúng để xác nhận sinh viên đã tự thực hiện hoặc cô đã duyệt.

## 6 Phần còn chờ

- Cô xác nhận cách ghi nhận bằng chứng checkpoint đã qua và kế hoạch đến tuần 11.
- Sinh viên tự chạy và giải thích được formulation, objective, baseline, epsilon và ý nghĩa bảng số liệu.
- Triển khai nội dung tuần 9 về TPE/demo sau mốc rà soát hiện tại; chưa được tính là hoàn thành.
- Nếu có nhận xét riêng ngoài bản biên tập, cần nhận đúng nội dung để bổ sung bảng xử lý; không tự tạo yêu cầu thay cô.
