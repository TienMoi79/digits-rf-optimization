"""Build the requested Word report from verified machine-readable results.

Run with the bundled document runtime (python-docx), not the ML venv.
"""
from pathlib import Path
import csv
import json

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/report_evaluation_2026-11-03"
DEST = ROOT / "report/Bao_cao_Train_Validation_Test_Baselines_2026-11-03.docx"


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


records = {r["label"]: r for r in read_csv(DATA / "comparison.csv")}
folds = read_csv(DATA / "validation_per_fold.csv")
counts = read_csv(DATA / "class_counts.csv")
classes = json.loads((DATA / "classification_reports.json").read_text(encoding="utf-8"))
selection = json.loads((ROOT / "results/selection.json").read_text(encoding="utf-8"))
run = json.loads((ROOT / "results/run_manifest.json").read_text(encoding="utf-8"))
manifest = json.loads((DATA / "run_manifest.json").read_text(encoding="utf-8"))
names = {"default": "RF mặc định", "random": "RF Random", "tpe": "RF TPE", "selected": "Selected"}
order = list(names)
doc = Document()
section = doc.sections[0]
section.page_width, section.page_height = Cm(21), Cm(29.7)
section.top_margin, section.bottom_margin = Cm(1.8), Cm(1.8)
section.left_margin, section.right_margin = Cm(1.9), Cm(1.9)
for name in ["Normal", "Title", "Subtitle", "Heading 1", "Heading 2", "Caption"]:
    style = doc.styles[name]
    style.font.name = "Arial"
    style.font.color.rgb = RGBColor(0, 0, 0)
    style.font.size = Pt(11 if name == "Normal" else 16 if name == "Heading 1" else 12)
    style.paragraph_format.space_after = Pt(6)
doc.styles["Normal"].paragraph_format.line_spacing = 1.12
doc.styles["Title"].font.size = Pt(22)
doc.styles["Title"].font.bold = True
doc.styles["Heading 1"].paragraph_format.space_before = Pt(0)
doc.styles["Heading 2"].font.bold = True
doc.styles["Caption"].font.size = Pt(9)
doc.styles["Caption"].font.italic = True
# The default Office template carries blue rules in Title/Subtitle styles.
for style in doc.styles:
    for border in list(style.element.iter(qn("w:pBdr"))):
        border.getparent().remove(border)
footer = section.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
footer.add_run("Trang ").font.size = Pt(9)
field = OxmlElement("w:fldSimple")
field.set(qn("w:instr"), "PAGE")
footer._p.append(field)
doc.core_properties.title = "Báo cáo train validation test và so sánh các mốc baseline"
doc.core_properties.subject = "Tối ưu siêu tham số Random Forest trên Digits"
doc.core_properties.author = ""


def p(text, bold=False):
    result = doc.add_paragraph()
    result.add_run(text).bold = bold
    return result


def heading(text, level=1):
    doc.add_heading(text, level=level)


def page(title):
    doc.add_page_break()
    heading(title)


def table(headers, rows, widths=None):
    tab = doc.add_table(rows=1, cols=len(headers))
    tab.autofit = False
    widths = widths or [17.2 / len(headers)] * len(headers)
    borders = OxmlElement("w:tblBorders")
    for edge in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        element = OxmlElement("w:" + edge)
        for k, v in {"val": "single", "sz": "4", "color": "D9D9D9"}.items():
            element.set(qn("w:" + k), v)
        borders.append(element)
    tab._tbl.tblPr.append(borders)
    for i, label in enumerate(headers):
        tab.rows[0].cells[i].text = str(label)
    repeat = OxmlElement("w:tblHeader")
    tab.rows[0]._tr.get_or_add_trPr().append(repeat)
    for values in rows:
        for cell, value in zip(tab.add_row().cells, values):
            cell.text = str(value)
    for ri, row in enumerate(tab.rows):
        cant_split = OxmlElement("w:cantSplit")
        row._tr.get_or_add_trPr().append(cant_split)
        for ci, cell in enumerate(row.cells):
            cell.width = Cm(widths[ci])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            shading = OxmlElement("w:shd")
            shading.set(qn("w:fill"), "23465C" if ri == 0 else "F1F5F7" if ri % 2 == 0 else "FFFFFF")
            cell._tc.get_or_add_tcPr().append(shading)
            margins = OxmlElement("w:tcMar")
            for side in ["top", "bottom", "left", "right"]:
                e = OxmlElement("w:" + side)
                e.set(qn("w:w"), "85")
                e.set(qn("w:type"), "dxa")
                margins.append(e)
            cell._tc.get_or_add_tcPr().append(margins)
            for par in cell.paragraphs:
                par.paragraph_format.space_after = Pt(0)
                par.paragraph_format.line_spacing = 1.04
                par.alignment = WD_ALIGN_PARAGRAPH.LEFT if ci == 0 else WD_ALIGN_PARAGRAPH.CENTER
                for r in par.runs:
                    r.font.name = "Arial"
                    r.font.size = Pt(9.5)
                    r.bold = ri == 0
                    r.font.color.rgb = RGBColor(255, 255, 255) if ri == 0 else RGBColor(0, 0, 0)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return tab


def num(label, metric, digits=5, percent=False):
    v = float(records[label][metric])
    return f"{v*100:.2f}%" if percent else f"{v:.{digits}f}"


def cv(label, metric):
    return f"{float(records[label]['validation_'+metric+'_mean']):.5f} ± {float(records[label]['validation_'+metric+'_sd']):.5f}"


def image(path, caption, width=16.6):
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.add_run().add_picture(str(ROOT / path), width=Cm(width))
    doc.add_paragraph(caption, style="Caption")


doc.add_paragraph("Báo cáo train validation test\nvà so sánh các mốc baseline", style="Title")
doc.add_paragraph("Tối ưu siêu tham số Random Forest cho nhận dạng chữ số viết tay", style="Subtitle")
p("Môn Nhập môn tối ưu hóa • Cập nhật 03/11/2026")
p("Báo cáo trình bày kết quả học trên train, bảng validation bằng 3-fold CV và đánh giá test của các cấu hình đã chốt trong thí nghiệm gốc. RF mặc định được chọn theo ngưỡng epsilon để ưu tiên kích thước nhỏ; cấu hình có CV cao nhất thuộc Random Search.")
heading("1 Tóm tắt kết quả", 2)
table(["Cấu hình", "Train accuracy", "Validation F1", "Test accuracy"],
      [[names[k], num(k, "train_accuracy", percent=True), num(k, "validation_f1_macro_mean"), num(k, "test_accuracy", percent=True)] for k in order], [4.3,4.3,4.3,4.3])
p("Selected trùng cấu hình RF mặc định nên các dòng tương ứng không phải hai mô hình độc lập. Validation F1 là trung bình ba folds; train accuracy đo trên dữ liệu đã fit và thường lạc quan hơn validation/test.")
heading("Dữ liệu và vai trò các tập", 2)
table(["Phần dữ liệu", "Số mẫu", "Vai trò"], [
    ["Toàn bộ Digits", "1.797", "10 lớp chữ số 0–9, ảnh 8×8"],
    ["Development dùng để refit train", "1.437", "Chia CV và học model đại diện"],
    ["Mỗi fold train / validation", "958 / 479", "Ba folds chỉ trong development"],
    ["Test giữ riêng", "360", "Báo cáo sau khi chốt cấu hình"]], [6.2,3.1,7.9])
p("Repo công khai: https://github.com/TienMoi79/digits-rf-optimization", bold=True)

page("2 Bài toán và thiết kế thực nghiệm")
p("Mục tiêu chính là tìm siêu tham số tối đa hóa Macro-F1 validation trong ngân sách hữu hạn, sau đó chọn cấu hình nhỏ trong nhóm gần điểm tốt nhất đã quan sát. Đây là quyết định theo chất lượng và dung lượng lưu trữ; không chứng minh tối ưu toàn cục.")
table(["Biến quyết định", "Miền tìm kiếm"], [
    ["n_estimators", "50, 75, 100, 125, 150, 175, 200"],
    ["max_depth", "5, 10, 15, 20, None"],
    ["min_samples_leaf", "1, 2, 3, 4"],
    ["max_features", "sqrt, log2, 0.5, 1.0"]], [5.2,12])
p("Miền rời rạc có 560 cấu hình. Hàm mục tiêu và quy tắc ra quyết định:")
p("max f(x) = mean Macro-F1 trên 3 folds development", bold=True)
p("min Size(x), với f(x) ≥ best CV đã quan sát − 0,005", bold=True)
p("Epsilon 0,005 là 0,5 điểm phần trăm F1 tuyệt đối. Size là byte stream joblib không nén; 1 MB = 1.000.000 byte. Chỉ xét baseline và các ứng viên đã thử, không dùng điểm test để điều chỉnh epsilon.")
heading("Các mốc so sánh và cấu hình đại diện", 2)
table(["Đại diện", "Số cây", "Độ sâu", "Min leaf", "Max features"], [
    [names[k], selection["candidates"][k]["params"]["n_estimators"], str(selection["candidates"][k]["params"]["max_depth"]),
     selection["candidates"][k]["params"]["min_samples_leaf"], selection["candidates"][k]["params"]["max_features"]] for k in order], [4.8,2.7,3,3,3.7])
p("RF mặc định là baseline mô hình. Random Search là mốc so sánh thuật toán tìm kiếm; RF Random và RF TPE là các forest được chọn bởi hai cách tìm kiếm, không phải hai loại classifier khác nhau. TPE dùng lịch sử trial để đề xuất vùng hứa hẹn, sau 10 trial khởi tạo ngẫu nhiên.")
p("Protocol gốc: stratified split seed 42; StratifiedKFold 3 folds seed 42; RF seed 123, n_jobs=1. Mỗi phương pháp 30 trials cho từng search seed 11,22,33, tổng 180 trials. Bốn đại diện được chốt bằng CV trước khi đánh giá test.")

page("3 Báo cáo tập train")
p("Các model đại diện được refit trên toàn bộ 1.437 mẫu development với tham số đã khóa. Bảng dưới là in-sample: dự đoán trên chính các mẫu đã học. Đợt bổ sung ngày 03/11 chỉ tính thêm số liệu báo cáo, không chạy HPO hoặc chọn lại mô hình.")
table(["Đại diện", "Accuracy", "Precision macro", "Recall macro", "F1 macro", "Log loss"],
      [[names[k]] + [num(k, "train_"+m) for m in ["accuracy","precision_macro","recall_macro","f1_macro","log_loss"]] for k in order], [4.2,2.4,2.7,2.7,2.6,2.6])
p("Accuracy là tỷ lệ nhãn đúng. Precision/recall/F1 macro là trung bình không trọng số giữa 10 lớp. Log loss đánh giá xác suất dự đoán, càng thấp càng tốt; đây không phải objective mà TPE dùng để chọn cấu hình trong study.")
p("Train gần hoặc bằng 100% phản ánh khả năng khớp mẫu đã học. Nó không chứng minh mô hình dự đoán hoàn hảo trên dữ liệu mới; cần đọc cùng validation và test. Selected cho phép cây phát triển không giới hạn độ sâu, min_samples_leaf=1.")
heading("Phân bố nhãn của development và test", 2)
table(["Nhãn", "Development", "Test", "Tổng"], [[r["class"],r["development"],r["test"],r["total"]] for r in counts], [4.3]*4)
p("Các lớp gần cân bằng nhưng không bằng nhau tuyệt đối. Preprocessing chỉ dùng vector 64 pixel do load_digits cung cấp, giá trị 0–16. Không chuẩn hóa, augment hoặc đổi nhãn. Mỗi mẫu development làm validation đúng một lần trong ba folds.")

page("4 Bảng validation bằng ba fold CV")
p("Mỗi cấu hình được fit trên 958 mẫu của từng fold và đánh giá trên 479 mẫu còn lại. Bảng trình bày mean ± sample SD qua ba folds, không phải confidence interval. Metric chính là Macro-F1; các chỉ số phụ bổ sung để đối chiếu cùng cách chia.")
table(["Đại diện", "Accuracy mean ± SD", "Macro F1 mean ± SD", "Log loss mean ± SD"],
      [[names[k],cv(k,"accuracy"),cv(k,"f1_macro"),cv(k,"log_loss")] for k in order], [3.4,4.6,4.6,4.6])
heading("Macro F1 theo từng fold", 2)
table(["Đại diện", "Fold 1", "Fold 2", "Fold 3"],
      [[names[k]] + [f"{float(r['f1_macro']):.8f}" for r in folds if r["label"]==k] for k in order], [4.9,4.1,4.1,4.1])
heading("Precision và recall validation", 2)
table(["Đại diện", "Precision macro mean ± SD", "Recall macro mean ± SD"],
      [[names[k],cv(k,"precision_macro"),cv(k,"recall_macro")] for k in order], [4.2,6.5,6.5])
p("Các Macro-F1 của từng fold được tính lại và đối chiếu với fold_scores đã lưu của thí nghiệm gốc, dung sai 1e−12. Điểm cao nhất trong nhóm đại diện là RF Random. CV cũng đã được dùng trong tìm kiếm siêu tham số, vì vậy không xem bảng này là một ước lượng độc lập sau tuning.")
p("Báo cáo classification_reports.json còn lưu per-class từ out-of-fold predictions. Chỉ số từ toàn bộ OOF không nhất thiết bằng trung bình metric qua folds; bảng chính ở đây dùng mean từng fold để khớp objective gốc.")

page("5 Báo cáo tập test giữ riêng")
p("Test gồm 360 mẫu. Các metric dưới đây được tính từ test_predictions.csv đã công bố, không chạy thêm inference trên features test trong đợt bổ sung. Lựa chọn cấu hình vẫn giữ nguyên như selection.json gốc; không dùng thứ hạng test để đổi phương pháp hoặc epsilon.")
table(["Đại diện", "Accuracy", "Precision macro", "Recall macro", "F1 macro", "Đúng Sai"],
      [[names[k]] + [num(k,"test_"+m) for m in ["accuracy","precision_macro","recall_macro","f1_macro"]]
       + [f"{records[k]['test_correct']} / {records[k]['test_errors']}"] for k in order], [4.2,2.4,2.7,2.7,2.6,2.6])
p("Selected đạt 350/360 mẫu đúng, tương đương 97,22%; RF Random đạt 351/360, tương đương 97,50%. Chênh lệch một mẫu chưa chứng minh ý nghĩa thống kê hoặc sự vượt trội tổng quát. Log loss test không được báo cáo vì bảng dự đoán gốc không lưu xác suất cho tất cả đại diện.")
heading("Classification report của selected trên test", 2)
table(["Lớp", "Precision", "Recall", "F1", "Support"],
      [[c] + [f"{classes['selected']['test'][str(c)][m]:.4f}" for m in ["precision","recall","f1-score"]]
       + [int(classes["selected"]["test"][str(c)]["support"])] for c in range(10)], [3.2,3.5,3.5,3.5,3.5])
p("Support là số mẫu thật của mỗi lớp trong test. Precision thấp gợi ý có nhãn khác bị dự đoán nhầm vào lớp đó; recall thấp gợi ý mẫu của lớp bị bỏ sót. Xem ma trận nhầm lẫn đã lưu trong figures/confusion_matrix.png để xác định hướng nhầm cụ thể.")

page("6 So sánh baseline và các cấu hình tối ưu")
table(["Đại diện", "Train F1", "Val F1", "Test F1", "Size MB"],
      [[names[k],num(k,"train_f1_macro"),num(k,"validation_f1_macro_mean"),num(k,"test_f1_macro"),num(k,"model_mb",3)] for k in order], [4.8,3.1,3.1,3.1,3.1])
p("RF mặc định là mốc học không tuning; Random Search là đối chứng tìm kiếm cùng ngân sách với TPE. Cả hai phương pháp tối ưu đều tìm được đại diện CV tốt hơn default trong lần chạy gốc, nhưng cần forest lớn hơn. So sánh các đại diện tốt nhất khác với so sánh mean hiệu quả thuật toán qua seeds ở trang 7.")
image("figures/quality_size_tradeoff.png", "Hình 1  Trade-off CV Macro-F1 và kích thước của các ứng viên gốc đã quan sát")
p("Selected giữ 4,975993 MB, nhỏ hơn 40,28% so với RF Random 8,332873 MB, với mức giảm CV F1 khoảng 0,00423. So với RF mặc định, selected không giảm kích thước vì trùng cấu hình. Không khẳng định không có đánh đổi: test accuracy selected thấp hơn Random khoảng 0,278 điểm phần trăm trong mẫu test hiện tại.")

page("7 Hiệu quả tìm kiếm và độ ổn định")
summary = read_csv(ROOT / "results/search_summary.csv")
table(["Phương pháp", "Mean best CV F1", "SD qua seeds", "CV search mean giây"],
      [[r["method"],f"{float(r['best_cv_f1_mean']):.5f}",f"{float(r['best_cv_f1_std']):.5f}",f"{float(r['cv_seconds_mean']):.2f}"] for r in summary], [4,4.4,4.4,4.4])
p("Ba search seeds gốc là 11,22,33, mỗi method/seed 30 trials. Mean best CV của Random cao hơn TPE trong nhóm này. SD chỉ mô tả độ biến thiên do search seed trên cùng dữ liệu/folds/RF seed; không phản ánh mọi nguồn bất định.")
image("figures/search_history.png", "Hình 2  Best so far mean và SD qua ba search seeds của thí nghiệm gốc", width=15.5)
p("Đường best-so-far không giảm theo định nghĩa cumulative maximum, không phải số liệu từng trial đã được làm mượt. Tổng runtime pipeline gốc khoảng " + f"{run['runtime_seconds']:.1f}" + " giây. Thời gian CV search trong bảng không bao gồm mọi refit/profiling hoặc toàn bộ pipeline.")
heading("Nghiên cứu mở rộng đã có", 2)
p("Phần mở rộng giữ nguyên development/folds và chạy 450 trials với năm search seeds. Ở budget 30, mean best CV: Random 0,97231; TPE startup 10 là 0,97282; startup 5 là 0,97442. Startup 5 hơn startup 10 ở ba seeds và hòa hai seeds. Kết quả mô tả không chứng minh ưu thế trên mọi dataset; runtime có outlier đã được giữ lại. Phần này không đánh giá test mới hoặc thay model đã chọn.")

page("8 Quyết định kiểm chứng và khả năng tái lập")
best = selection["best_cv_f1"]
threshold = selection["minimum_acceptable_cv_f1"]
p(f"Best CV gốc là {best:.8f}; epsilon 0,005 cho ngưỡng {threshold:.8f}. RF mặc định đạt {num('selected','validation_f1_macro_mean',8)}, nằm trong ngưỡng và có kích thước nhỏ nhất trong các ứng viên đủ điều kiện. Do đó giữ cấu hình default làm selected; quy tắc được khóa trước đánh giá test.")
heading("Nguồn số liệu và kiểm chứng", 2)
p(f"Các bảng train/validation bổ sung ngày 03/11 sử dụng {manifest['fresh_development_fits']} lần fit development của {manifest['distinct_configs']} cấu hình phân biệt; không chạy lại HPO. Fold F1 được đối chiếu log cũ, test accuracy/F1 được tính lại từ predictions, dự đoán train của selected khớp model bàn giao. Hash xác nhận {len(manifest['protected_sha256'])} artifact cũ giữ nguyên.")
p("Nguồn máy đọc: results/report_evaluation_2026-11-03/comparison.csv, validation_per_fold.csv, class_counts.csv, classification_reports.json và run_manifest.json. Code bổ sung: src/report_evaluation.py; script dựng Word: scripts/build_train_validation_test_report.py.")
heading("Lệnh tái kiểm tra", 2)
for command in ["python -m unittest discover -s tests -v", "python -m src.validation", "python -m src.robustness_validation", "python -m src.report_facts --check"]:
    par=p(command)
    par.runs[0].font.name="Consolas"
    par.runs[0].font.size=Pt(9)
p("Các lệnh kiểm chứng kết quả gốc có thể tái tạo dự đoán từ model cũ trên test; đó là kiểm tra tính nhất quán, không phải đánh giá độc lập mới. Báo cáo không dùng việc kiểm tra lại để tuning.")
heading("Giới hạn và hướng tiếp theo", 2)
p("Train là in-sample; CV đã tham gia chọn cấu hình; chỉ có một dataset nhỏ và một holdout. Các folds có liên quan và SD không phải confidence interval. Cần thận trọng với chênh lệch nhỏ, runtime theo tải máy, xác suất chưa calibration và ảnh ngoài phân phối Digits. Model giữ nguyên train development 80%; chưa kiểm chứng ảnh điện thoại. Sinh viên cần tự chạy và giải thích trong bảo vệ cá nhân.")
heading("Tài liệu và công cụ hỗ trợ", 2)
for text in [
    "Scikit-learn load_digits: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html",
    "Scikit-learn classification metrics: https://scikit-learn.org/stable/modules/model_evaluation.html",
    "Optuna TPESampler: https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html",
    "UCI Optical Recognition of Handwritten Digits: DOI 10.24432/C50P49.",
    "Codex hỗ trợ code, chạy kiểm chứng và soạn báo cáo. Kết quả sinh từ code/logs thực tế; người nộp chịu trách nhiệm tự kiểm tra và giải thích."]:
    par=p(text)
    for r in par.runs:
        r.font.size=Pt(8.5)

DEST.parent.mkdir(exist_ok=True)
doc.save(DEST)
print(DEST)
