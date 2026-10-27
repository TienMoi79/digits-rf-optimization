# Theo dõi checkpoint và kế hoạch hoàn thiện

## Mốc thời gian và phạm vi

- Cập nhật kế hoạch ngày **27/10/2026** theo thông tin sinh viên: hiện là
  **tuần học 8**, hạn nộp **tuần học 11**. Chưa có ngày/giờ nộp cụ thể.
- Số tuần học 8–11 không phải số thứ tự 1–5 trong các giai đoạn của đề.
  Kế hoạch này ánh xạ công việc vào thời gian còn lại, không đổi lịch học.
- Dự án đã có code và kết quả trước đợt rà soát này. Không coi bản đang có là
  một dự án bắt đầu từ số không hoặc tạo lại lịch sử phát triển của quá khứ.
- Theo cập nhật của sinh viên, cô không yêu cầu báo cáo hoặc slide. README,
  notebook, mã nguồn, kết quả và demo vẫn thuộc phạm vi bàn giao. Theo yêu cầu
  mới ngày 27/10, sau mỗi phần hoàn thành sẽ có báo cáo tiến độ **`.md`** trong
  `report/` để sinh viên gửi cô lấy feedback. Không tạo Word/PDF hoặc slide.
- Nhật ký Git chứng minh thay đổi đã commit; không chứng minh sinh viên đã
  bảo vệ checkpoint hoặc được cô chấp nhận. Trạng thái đó cần xác nhận riêng.

## Bằng chứng có sẵn trước ngày lập kế hoạch

| Ngày commit | Commit | Nội dung có thể kiểm tra trong Git |
|---|---|---|
| 13/10/2026 | `f7ea277` | Bản ban đầu có pipeline Digits/Random Forest, thí nghiệm và kết quả |
| 13/10/2026 | `73271cf` | Chuẩn hóa cấu trúc repo và tiện ích cập nhật GitHub |
| 24/10/2026 | `2ab1a4c` | Thêm runner nghiên cứu 5 seeds, ablation và CI |
| 24/10/2026 | `4c1b160` | Sửa lỗi khóa file tạm thời, cô lập môi trường CI |
| 24/10/2026 | `b41a801` | Công bố kết quả phần mở rộng và notebook phân tích |
| 24/10/2026 | `c4cecec` | Bổ sung đường accuracy/log loss theo số cây |
| 24/10/2026 | `1e18603` | Đối chiếu UCI và fit độc lập để kiểm tra đường cong |

Các mốc trên là bằng chứng công việc thật trên hai ngày, **không được đánh
số lại thành năm checkpoint đã diễn ra ở năm tuần khác nhau**. Bản ban đầu
chứa nhiều nội dung cùng lúc. Việc cô có chấp nhận các bằng chứng này cho
checkpoint đã qua hay không hiện chưa được xác nhận.

## Đối chiếu năm giai đoạn của đề

| Giai đoạn trong đề | Bằng chứng đang có | Việc cần hoàn thiện/kiểm tra trực tiếp |
|---|---|---|
| 1: Bài toán và formulation | README: Problem, Optimization Formulation, Dataset; repo public | Sơ đồ pipeline và kế hoạch được bổ sung ngày 27/10; sinh viên giải thích được biến, mục tiêu, miền và trade-off |
| 2: Preprocessing, baseline, tái lập | `src/common.py`, `results/baseline.json`, notebook gốc, requirements đã khóa | Sinh viên tự chạy môi trường/baseline và đối chiếu đầu ra; lịch sử theo tuần chưa được xác nhận |
| 3: Thuật toán, logging, kiểm tra nghiệm | `src/experiment.py`, `trials.json/csv`, search history | Giải thích rõ đầu vào/đầu ra, lý do chọn TPE; demo chuỗi tối ưu thay vì chỉ dự đoán chữ số |
| 4: Thực nghiệm và phân tích | 180 trials gốc; 450 trials mở rộng; sensitivity, runtime, ablation, bảng/hình | Chạy tái lập có kiểm soát, phân tích trường hợp sai và kiểm tra kết luận/giới hạn |
| 5: Chốt nghiệm và bàn giao | `selection.json`, model đã lưu, demo dự đoán, README | Chốt bản nộp, tag/release v1.0 và sinh viên tự demo/bảo vệ; báo cáo/slide đã được miễn theo cập nhật |

Một artifact đã tồn tại không đồng nghĩa checkpoint cá nhân đã được thông qua.
File formulation riêng 1–2 trang không được tạo trong đợt này; phần mô hình hóa
đang nằm trong README/notebook theo phạm vi code, không báo cáo.

## Kế hoạch từ tuần học 8 đến tuần học 11

| Tuần học | Công việc sẽ thực hiện | Bằng chứng phải có sau khi làm | Trạng thái hiện tại |
|---|---|---|---|
| 8, hiện tại | Rà soát yêu cầu và bằng chứng cũ; bổ sung pipeline; giữ cố định protocol; sinh viên kiểm tra lại formulation và baseline | README có pipeline; nhật ký này; kết quả kiểm chứng; commit đúng ngày; phần sinh viên tự giải thích | Đã bổ sung pipeline và kế hoạch; phần tự chạy/giải thích của sinh viên chưa xác nhận |
| 9 | Bổ sung giải thích TPE: đầu vào/đầu ra, nguyên lý, tham số, lý do phù hợp; xây demo quá trình tìm kiếm và quy tắc epsilon | Source/demo, ví dụ chạy, notebook/README cập nhật, kiểm tra phù hợp, commit thật | Dự kiến, chưa thực hiện trong đợt này |
| 10 | Sinh viên chạy tái lập vào thư mục mới; đối chiếu kết quả; phân tích lỗi và hạn chế; rà soát log/bảng/hình | Log của lần chạy thực tế, đối chiếu metric; ví dụ dự đoán sai; mô tả dao động runtime; commit thật | Dự kiến, chưa thực hiện trong đợt này |
| 11, hạn nộp | Chạy kiểm tra cuối; chốt code, cấu hình, model và demo; tạo tag/release v1.0 cho commit nộp; sinh viên tập bảo vệ và nộp link | Các checks đạt; repo public; tag/release đúng bản; demo phần tối ưu; xác nhận nộp của sinh viên | Dự kiến, chưa thực hiện trong đợt này |

Chưa gán ngày dương lịch cho tuần 9–11 vì chưa có lịch cụ thể của lớp.
Kế hoạch không thay thế việc xin cô xác nhận cách bổ sung các checkpoint đã qua.
Không có lịch tự động chạy hoặc cam kết rằng các công việc tương lai đã hoàn thành.

## Nhật ký bổ sung ngày 27/10/2026

Công việc thực hiện trong lần cập nhật này:

1. Đọc lại quy tắc của repo và đối chiếu các artifact/checkpoint đang có.
2. Bổ sung sơ đồ pipeline vào README, có tách development/test và nhánh mở rộng
   chỉ dùng development; giữ nguyên các quyết định và số liệu đã công bố.
3. Ghi lịch sử có thật và kế hoạch tuần học 8–11 trong file này.
4. Bổ sung quy tắc cập nhật checkpoint cho các lần làm việc tiếp theo.
5. Tạo `report/checkpoint-tuan-8-2026-10-27.md` để sinh viên gửi cô kiểm tra
   phần vừa làm và phản hồi các điểm còn thiếu.

Đã chạy các lệnh kiểm tra sau ngày 27/10 trước khi commit (dùng Python của
`.venv` với `-X utf8`):

| Lệnh | Kết quả thực tế |
|---|---|
| `python -m unittest discover -s tests -v` | 14 tests đạt |
| `python -m src.validation` | 401 kiểm tra kết quả gốc đạt |
| `python -m src.robustness_validation` | 3.278 kiểm tra nghiên cứu mở rộng đạt; kết quả đã công bố được bảo toàn |
| `git diff --check` | Không phát hiện lỗi whitespace trong các thay đổi được theo dõi |

Đây là lần kiểm chứng bộ kết quả hiện có, không phải lần chạy mới toàn bộ HPO.

Không chạy lại HPO, thay nhãn/dữ liệu, điều chỉnh epsilon, hoặc đánh giá thêm
mô hình mới trên test trong lần bổ sung kế hoạch này. Các bảng/hình cũ là
bằng chứng từ lần chạy cũ, không ghi thành kết quả mới ngày 27/10.

## Cách cập nhật nhật ký từ lần sau

- Chỉ đánh dấu hoàn thành sau khi công việc và kiểm tra tương ứng thực sự xong.
- Mỗi lần cập nhật ghi ngày, việc mới làm, lệnh kiểm tra và artifact liên quan.
- Commit/push từng thay đổi có ý nghĩa, không tạo commit rỗng để đủ số mốc.
- Không thay ngày commit hoặc viết lại lịch sử đã public để giả lập tiến độ.
- Trạng thái sinh viên tự chạy, giải thích, demo, cô duyệt và nộp LMS cần do
  sinh viên xác nhận; trợ lý không tự đánh dấu thay.
- Sau mỗi phần hoàn thành, tạo/cập nhật báo cáo `.md` trong `report/`, ghi
  rõ việc mới làm, bằng chứng, lệnh/kết quả kiểm tra, phần còn thiếu và các
  điểm cần cô xem. Không tạo Word/PDF. Giữ báo cáo mốc cũ khi chuyển sang mốc mới.
- Khi sinh viên chuyển feedback của cô, ghi nội dung phản hồi thực nhận,
  cách sửa, artifact thay đổi và kiểm tra sau sửa; không tự tạo ý kiến hoặc
  đánh dấu giảng viên đã duyệt khi chưa có xác nhận.
- Trong tuần 11 mới chốt tag/release bản nộp sau khi kiểm tra, không gắn tag
  hoàn thành ngay khi các việc còn lại vẫn đang ở trạng thái dự kiến.

## Tiếp nhận bản biên tập và sửa báo cáo ngày 27/10/2026

Sinh viên chuyển một bản biên tập và ghi nhận là cô gửi nhận xét. Bản này
vẫn có phần chờ phản hồi, chưa có xác nhận duyệt checkpoint. Đã đối chiếu với
code/logs và ghi rõ từng điểm trong
[báo cáo xử lý](report/feedback-tuan-8-2026-10-27.md).

Công việc mới:

1. Giữ cấu trúc formulation hai giai đoạn; làm rõ tập ứng viên quan sát và
   đơn vị epsilon. Sửa mô tả test theo bốn đại diện và hoạt động kiểm chứng.
2. Sửa số liệu trong bản biên tập theo logs: model 4.975993 MB, best CV
   0.97484838 của Random, gap CV 0.00423286; không thay số liệu thực nghiệm.
3. Bỏ kết luận bão hòa/không đánh đổi chất lượng; ghi rõ phần sinh viên tự
   kiểm chứng và việc cô duyệt vẫn chưa được xác nhận.
4. Thêm `src/report_facts.py` để sinh/kiểm tra vùng số liệu trong báo cáo từ
   logs/model; thêm hai kiểm tra hồi quy, đưa lệnh check vào CI và helper push.
5. Cập nhật báo cáo Markdown và nhật ký phản hồi. Giữ lịch sử trước sửa ở
   commit `36861ba`; không sửa hoặc bỏ commit đã public.

Phần TPE sâu hơn/demo và các công việc tuần 9–11 tiếp tục là dự kiến. Lần
xử lý này không được ghi thành một checkpoint tuần học mới hoặc duyệt của cô.

Kiểm tra sau sửa: 16 unit tests đạt (thay cho 14 ở lần rà soát đầu), 401 kiểm
tra kết quả gốc đạt, 3.278 kiểm tra nghiên cứu mở rộng đạt; lệnh
`python -m src.report_facts --check` xác nhận vùng số liệu khớp logs/model.
Không chạy lại HPO hoặc điều chỉnh kết quả thực nghiệm trong đợt này.
