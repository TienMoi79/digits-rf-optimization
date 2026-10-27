"""Generate/check the week-8 report's numeric evidence from frozen logs.

This command does not fit a model, read held-out features or change results.
Only --write replaces the marked evidence section in the Markdown report.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report/checkpoint-tuan-8-2026-10-27.md"
START = "<!-- EXPERIMENT_FACTS_START -->"
END = "<!-- EXPERIMENT_FACTS_END -->"


def render_facts(base):
    base = Path(base)
    decision = json.loads((base / "selection.json").read_text(encoding="utf-8"))
    trials = json.loads((base / "trials.json").read_text(encoding="utf-8"))
    with (base / "final_metrics.csv").open(encoding="utf-8", newline="") as handle:
        final = {row["label"]: row for row in csv.DictReader(handle)}
    with (base / "search_summary.csv").open(encoding="utf-8", newline="") as handle:
        summary = {row["method"]: row for row in csv.DictReader(handle)}
    candidates = decision["candidates"]
    for method in ["random", "tpe"]:
        runs = {}
        for row in trials:
            if row["method"] == method:
                runs.setdefault(row["seed"], []).append(row["cv_f1"])
        mean_best = fmean(max(scores) for scores in runs.values())
        if not math.isclose(mean_best, float(summary[method]["best_cv_f1_mean"]), rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Mean of per-seed best scores disagrees with trials: {method}")
    chosen = candidates["selected"]
    best = max(trials, key=lambda row: row["cv_f1"])
    best_score = best["cv_f1"]
    threshold = best_score - decision["epsilon"]
    eligible = [row for row in trials if row["feasible"] and row["cv_f1"] >= threshold]
    if (not math.isclose(best_score, decision["best_cv_f1"], rel_tol=0, abs_tol=1e-12)
            or not math.isclose(threshold, decision["minimum_acceptable_cv_f1"], rel_tol=0, abs_tol=1e-12)
            or chosen not in eligible
            or chosen["model_bytes"] != min(row["model_bytes"] for row in eligible)):
        raise ValueError("Locked decision does not reproduce the CV epsilon rule.")
    saved_bytes = (base / "models/selected.joblib").stat().st_size
    if saved_bytes != chosen["model_bytes"]:
        raise ValueError("Selected model bytes differ from the recorded size.")
    for label, candidate in candidates.items():
        row = final[label]
        if (not math.isclose(float(row["cv_f1"]), candidate["cv_f1"], rel_tol=0, abs_tol=1e-12)
                or not math.isclose(float(row["model_mb"]), candidate["model_bytes"] / 1_000_000,
                                    rel_tol=0, abs_tol=1e-12)):
            raise ValueError(f"Final table disagrees with locked candidate: {label}")
    gap = best_score - chosen["cv_f1"]
    reduction_random = (1 - saved_bytes / candidates["random"]["model_bytes"]) * 100
    reduction_default = (1 - saved_bytes / candidates["default"]["model_bytes"]) * 100
    selected_accuracy = float(final["selected"]["test_accuracy"])
    random_accuracy = float(final["random"]["test_accuracy"])
    lines = [
        "Các số dưới đây được sinh từ `selection.json`, `trials.json`,",
        "`final_metrics.csv`, `search_summary.csv` và kích thước file model đã lưu.",
        "Đây là kết quả của thí nghiệm gốc 180 trials, không trộn với nghiên cứu 450 trials.",
        "", "| Đại diện | CV Macro-F1 | Model MB | Test accuracy |",
        "|---|---:|---:|---:|",
    ]
    for label in ["default", "random", "tpe", "selected"]:
        row = final[label]
        lines.append(f"| {label} | {float(row['cv_f1']):.8f} | {float(row['model_mb']):.6f} | "
                     f"{float(row['test_accuracy']) * 100:.5f}% |")
    lines += [
        "", "MB dùng đơn vị thập phân: 1 MB = 1.000.000 byte, joblib không nén.", "",
        f"- Điểm cao nhất đã quan sát là **{best_score:.8f}**, thuộc **{best['method']}**, "
        f"seed **{best['seed']}**, trial **{best['trial']}** (trial đánh số từ 0).",
        f"- Mean của best CV qua ba search seeds: Random **{float(summary['random']['best_cv_f1_mean']):.5f}**, "
        f"TPE **{float(summary['tpe']['best_cv_f1_mean']):.5f}**. Mean qua seeds khác với điểm tốt nhất của một cấu hình.",
        f"- Epsilon cố định **{decision['epsilon']:.3f}**; ngưỡng chấp nhận **{threshold:.8f}**.",
        f"- Selected đạt CV **{chosen['cv_f1']:.8f}**; khoảng cách đến điểm cao nhất "
        f"**{gap:.8f}**, tương đương **{gap * 100:.5f} điểm phần trăm**.",
        f"- File selected có **{saved_bytes:,} byte**, tức **{saved_bytes / 1_000_000:.6f} MB**.",
        f"- So với đại diện Random tốt nhất, selected nhỏ hơn **{reduction_random:.2f}%**. "
        f"So với baseline mặc định, mức giảm là **{reduction_default:.2f}%**; selected chính là cấu hình baseline.",
        "", f"Test accuracy của selected là **{selected_accuracy * 100:.5f}%**, "
        f"của đại diện Random là **{random_accuracy * 100:.5f}%**, chênh "
        f"**{(random_accuracy - selected_accuracy) * 100:.5f} điểm phần trăm**. "
        "Bảng test chỉ dùng để báo cáo sau khi khóa lựa chọn, không dùng chọn cấu hình.",
        "", "Quy tắc epsilon chấp nhận một khoảng giảm CV để ưu tiên model nhỏ. Kết quả chỉ",
        "xác định model nhỏ nhất trong các ứng viên đã quan sát đáp ứng ngưỡng; không",
        "chứng minh toàn bộ miền đã bão hòa, tìm được tối ưu toàn cục, hoặc không có",
        "đánh đổi chất lượng. Chưa có cơ sở suy rộng chi phí suy luận sang môi trường triển khai khác.",
    ]
    return "\n".join(lines)


def checked_report_text(report, base, *, write=False):
    report = Path(report)
    text = report.read_text(encoding="utf-8")
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError("Report needs exactly one numeric evidence marker pair.")
    before, tail = text.split(START)
    body, after = tail.split(END)
    expected = render_facts(base)
    if write:
        report.write_text(before + START + "\n\n" + expected + "\n\n" + END + after,
                          encoding="utf-8")
    elif body.strip() != expected:
        raise ValueError("Report numeric evidence is stale or modified; run python -m src.report_facts --write.")
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--base", type=Path, default=ROOT / "results")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Verify the marked section (default).")
    mode.add_argument("--write", action="store_true", help="Refresh only the marked section.")
    args = parser.parse_args()
    checked_report_text(args.report, args.base, write=args.write)
    print("Report facts updated from saved logs." if args.write else "PASS: report numeric evidence matches saved logs and model size.")


if __name__ == "__main__":
    main()
