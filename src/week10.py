"""Week-10 error analysis and frozen-protocol reproduction comparison.

No tuning. Failure analysis uses existing test labels/predictions and images.
The new experiment is a replay of the known holdout, not an independent test.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix

from .common import ROOT, fingerprint, write_json
from .robustness import canonical_sha256

BASE = ROOT / "results"
OUT = BASE / "week10_2026-11-09"
REPRO = OUT / "reproduction"
FIGS = ROOT / "figures/week10_2026-11-09"


def capture_originals():
    """Snapshot original results/figures/configs/report versions, not this run."""
    paths = [p for folder in [BASE, ROOT / "figures", ROOT / "configs"]
             for p in folder.rglob("*") if p.is_file()
             and not p.is_relative_to(OUT) and not p.is_relative_to(FIGS)]
    paths += list((ROOT / "report").glob("*.docx"))
    return {p.relative_to(ROOT).as_posix(): canonical_sha256(p) for p in sorted(paths)}


def verify_originals(snapshot):
    if any(not (ROOT / p).is_file() or canonical_sha256(ROOT / p) != sha for p, sha in snapshot.items()):
        raise AssertionError("An original result/config/figure/report changed.")


def failure_records(predictions):
    """Return actual selected errors; never sample only convenient cases."""
    if not {"dataset_index", "y_true", "default", "random", "tpe", "selected"} <= set(predictions):
        raise ValueError("Missing published prediction columns.")
    if predictions.dataset_index.duplicated().any():
        raise ValueError("Duplicate dataset index in held-out predictions.")
    frame = predictions[predictions.selected != predictions.y_true].copy()
    for name in ["default", "random", "tpe"]:
        frame[f"{name}_correct"] = frame[name] == frame.y_true
    frame = frame.rename(columns={"selected": "selected_prediction"})
    pairs = frame.groupby(["y_true", "selected_prediction"]).size().reset_index(name="count")
    return frame.reset_index(drop=True), pairs.sort_values(["count", "y_true", "selected_prediction"], ascending=[False, True, True]).reset_index(drop=True)


def analyze_errors():
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    proof = OUT / "original_artifacts.json"
    if not proof.exists():
        write_json(proof, {"captured_at_utc": datetime.now(timezone.utc).isoformat(),
                           "scope": "before week10 analysis; reproduction may already be running",
                           "sha256": capture_originals()})
    else:
        verify_originals(json.loads(proof.read_text(encoding="utf-8"))["sha256"])
    predictions = pd.read_csv(BASE / "test_predictions.csv")
    split = json.loads((BASE / "split_indices.json").read_text(encoding="utf-8"))
    if predictions.dataset_index.tolist() != split["test"]:
        raise AssertionError("Failure analysis must use the frozen test split.")
    digits = load_digits()
    np.testing.assert_array_equal(digits.target[predictions.dataset_index], predictions.y_true)
    errors, pairs = failure_records(predictions)
    errors.to_csv(OUT / "selected_errors.csv", index=False)
    pairs.to_csv(OUT / "confusion_pairs.csv", index=False)
    matrix = confusion_matrix(predictions.y_true, predictions.selected, labels=np.arange(10))
    np.testing.assert_array_equal(matrix, np.loadtxt(BASE / "confusion_matrix.csv", delimiter=",", dtype=int))
    class_rows = [{"class": c, "support": int(matrix[c].sum()), "correct": int(matrix[c, c]),
                   "errors": int(matrix[c].sum() - matrix[c, c]), "recall": float(matrix[c, c] / matrix[c].sum())}
                  for c in range(10)]
    pd.DataFrame(class_rows).to_csv(OUT / "errors_by_class.csv", index=False)
    print("Selected error pairs:", pairs.to_dict(orient="records"), flush=True)

    plt.rcParams.update({"font.size": 10, "figure.dpi": 160})
    ncols = 5
    nrows = max(1, (len(errors) + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.6 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, row in zip(axes.flat, errors.itertuples()):
        ax.imshow(digits.images[row.dataset_index], cmap="gray_r", vmin=0, vmax=16, interpolation="nearest")
        ax.set_title(f"Index {row.dataset_index}\nTrue {row.y_true} / Pred {row.selected_prediction}", fontsize=15)
    fig.suptitle("All selected-model errors on the original test set; no new test inference", y=.99, fontsize=15)
    fig.tight_layout(h_pad=4, w_pad=1.4, rect=[0, 0, 1, .92])
    fig.savefig(FIGS / "selected_test_errors.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5.3))
    im = ax.imshow(matrix, cmap="Blues")
    for a in range(10):
        for b in range(10):
            ax.text(b, a, str(matrix[a, b]), ha="center", va="center", fontsize=8,
                    color="white" if matrix[a, b] > matrix.max()/2 else "black")
    ax.set(xticks=range(10), yticks=range(10), xlabel="Predicted digit", ylabel="True digit", title="Frozen selected model: original test confusion matrix")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(FIGS / "confusion_matrix.png")
    plt.close(fig)

    comparison = pd.read_csv(BASE / "report_evaluation_2026-11-03/comparison.csv")
    gaps = comparison[["label", "train_accuracy", "validation_accuracy_mean", "test_accuracy"]].copy()
    gaps["train_minus_validation_pp"] = (gaps.train_accuracy - gaps.validation_accuracy_mean) * 100
    gaps["train_minus_test_pp"] = (gaps.train_accuracy - gaps.test_accuracy) * 100
    gaps.to_csv(OUT / "generalization_gaps.csv", index=False)
    # Compare the errors of the two fixed representatives without selecting a new model.
    random_error = predictions.random != predictions.y_true
    selected_error = predictions.selected != predictions.y_true
    error_summary = {"test_samples": len(predictions), "selected_errors": len(errors),
                     "random_errors": int(random_error.sum()),
                     "selected_errors_random_corrects": int((selected_error & ~random_error).sum()),
                     "random_errors_selected_corrects": int((~selected_error & random_error).sum()),
                     "errors_shared_by_selected_and_random": int((selected_error & random_error).sum()),
                     "test_source": "original saved predictions; no model fit or new inference in error analysis",
                     "limitation": "These are observed mistakes on one known holdout; visual interpretation is not a proven causal explanation."}
    write_json(OUT / "error_summary.json", error_summary)
    verify_originals(json.loads(proof.read_text(encoding="utf-8"))["sha256"])
    return error_summary


def compare_runs(base, reproduction):
    """Compare every trial's deterministic fields; preserve measured runtimes."""
    base, reproduction = Path(base), Path(reproduction)
    read = lambda root, name: json.loads((root / name).read_text(encoding="utf-8"))
    original, rerun = read(base, "run_manifest.json"), read(reproduction, "run_manifest.json")
    if original["status"] != "completed" or rerun["status"] != "completed":
        raise ValueError("Both experiment runs must complete before comparison.")
    if (original["config"] != rerun["config"]
            or fingerprint(original["config"]) != original["config_sha256"]
            or fingerprint(rerun["config"]) != rerun["config_sha256"]):
        raise ValueError("Reproduction configuration differs from the original protocol.")
    if read(base, "split_indices.json") != read(reproduction, "split_indices.json"):
        raise ValueError("Reproduction split differs from original.")
    a, b = read(base, "trials.json"), read(reproduction, "trials.json")
    key = lambda row: (row["method"], row["seed"], row["trial"])
    am, bm = {key(r): r for r in a}, {key(r): r for r in b}
    if len(am) != len(a) or len(bm) != len(b) or am.keys() != bm.keys():
        raise ValueError("Trial identity sets differ or contain duplicate records.")
    deltas = []
    for identity, old in am.items():
        new = bm[identity]
        if old["params"] != new["params"]:
            raise ValueError(f"Reproduction trial configuration differs: {identity}")
        if old["model_bytes"] != new["model_bytes"] or old["total_nodes"] != new["total_nodes"]:
            raise ValueError(f"Reproduction model structure differs: {identity}")
        if old["feasible"] != new["feasible"]:
            raise ValueError("Feasibility status changed.")
        diff = abs(old["cv_f1"] - new["cv_f1"])
        np.testing.assert_allclose(old["fold_scores"], new["fold_scores"], rtol=0, atol=1e-12)
        if diff > 1e-12:
            raise ValueError(f"Reproduction trial CV differs: {identity}")
        deltas.append(diff)
    decision_a, decision_b = read(base, "selection.json"), read(reproduction, "selection.json")
    for field in ["rule", "epsilon", "best_cv_f1", "minimum_acceptable_cv_f1", "selected_params"]:
        if decision_a[field] != decision_b[field]:
            raise ValueError(f"Reproduction decision differs: {field}")
    for name in decision_a["candidates"]:
        if key(decision_a["candidates"][name]) != key(decision_b["candidates"][name]):
            raise ValueError("Candidate source changed.")
    pred_a, pred_b = pd.read_csv(base / "test_predictions.csv"), pd.read_csv(reproduction / "test_predictions.csv")
    pd.testing.assert_frame_equal(pred_a, pred_b)
    metrics_a, metrics_b = pd.read_csv(base / "final_metrics.csv"), pd.read_csv(reproduction / "final_metrics.csv")
    metric_rows = []
    for row in metrics_a.itertuples():
        new = metrics_b[metrics_b.label == row.label].iloc[0]
        for metric in ["cv_f1", "test_f1", "test_accuracy", "model_mb"]:
            value = float(getattr(row, metric))
            difference = float(new[metric]) - value
            if abs(difference) > 1e-12:
                raise ValueError("Representative metrics do not reproduce.")
            metric_rows.append({"label": row.label, "metric": metric, "original": value,
                                "reproduction": float(new[metric]), "delta": difference})
    latency_rows = []
    for label in decision_a["candidates"]:
        old, new = decision_a["candidates"][label], decision_b["candidates"][label]
        latency_rows.append({"label": label, "original_batch_median_ms": old["batch_predict_ms_median"],
                             "reproduction_batch_median_ms": new["batch_predict_ms_median"],
                             "original_batch_p95_ms": old["batch_predict_ms_p95"],
                             "reproduction_batch_p95_ms": new["batch_predict_ms_p95"]})
    per_seed = []
    for method in ["random", "tpe"]:
        for seed in original["config"]["search_seeds"]:
            oa = [r for r in a if r["method"] == method and r["seed"] == seed]
            ob = [r for r in b if r["method"] == method and r["seed"] == seed]
            old_time = sum(r["cv_seconds"] for r in oa)
            new_time = sum(r["cv_seconds"] for r in ob)
            per_seed.append({"method": method, "seed": seed, "trials": len(ob),
                             "original_cv_search_seconds": old_time,
                             "reproduction_cv_search_seconds": new_time,
                             "ratio_new_over_original": new_time / old_time,
                             "original_best_cv_f1": max(r["cv_f1"] for r in oa),
                             "reproduction_best_cv_f1": max(r["cv_f1"] for r in ob)})
    model_sha_equal = canonical_sha256(base / "models/selected.joblib") == canonical_sha256(reproduction / "models/selected.joblib")
    summary = {"config_matches": True, "split_matches": True, "trial_records_compared": len(a),
               "search_trials_reproduced": len(a)-1, "max_absolute_trial_cv_delta": max(deltas),
               "all_trial_params_and_fold_scores_match": True, "decision_matches": True,
               "all_test_predictions_match": True, "representative_metrics_match": True,
               "selected_model_binary_sha256_matches": model_sha_equal,
               "original_runtime_seconds": original["runtime_seconds"],
               "reproduction_runtime_seconds": rerun["runtime_seconds"],
               "runtime_ratio_new_over_original": rerun["runtime_seconds"] / original["runtime_seconds"],
               "original_versions": original["versions"], "reproduction_versions": rerun["versions"],
               "versions_match": original["versions"] == rerun["versions"],
               "heldout_interpretation": "Predictions were recomputed on the same already known split after replaying the fixed decision protocol; no independent new test evidence.",
               "executor": "Codex executed this run; student-run reproduction is not confirmed",
               "timing_interpretation": "One original and one rerun session; descriptive runtime differences, not a controlled speed benchmark or stability estimate."}
    return summary, metric_rows, latency_rows, per_seed


def compare():
    summary, metrics, latency, per_seed = compare_runs(BASE, REPRO)
    pd.DataFrame(metrics).to_csv(OUT / "metric_comparison.csv", index=False)
    pd.DataFrame(latency).to_csv(OUT / "latency_comparison.csv", index=False)
    pd.DataFrame(per_seed).to_csv(OUT / "runtime_by_method_seed.csv", index=False)
    write_json(OUT / "reproduction_comparison.json", summary)
    proof = json.loads((OUT / "original_artifacts.json").read_text(encoding="utf-8"))["sha256"]
    verify_originals(proof)
    print(json.dumps({key: value for key, value in summary.items() if key != "batch_global_indices"}, indent=2), flush=True)
    return summary


def benchmark():
    """Small latency stability measurement after the full HPO has finished."""
    import joblib
    completed = json.loads((REPRO / "run_manifest.json").read_text(encoding="utf-8"))
    if completed["status"] != "completed":
        raise ValueError("Finish HPO first, so this benchmark does not contend with that training process.")
    path = OUT / "latency_measurements.csv"
    if path.exists():
        raise FileExistsError("Preserve the recorded benchmark; do not replace it with a more favorable run.")
    proof = json.loads((OUT / "original_artifacts.json").read_text(encoding="utf-8"))["sha256"]
    verify_originals(proof)
    split = json.loads((BASE / "split_indices.json").read_text(encoding="utf-8"))
    ids = split["development"][:256]
    assert not set(ids) & set(split["test"])
    batch = load_digits().data[ids]
    model = joblib.load(BASE / "models/selected.joblib")
    assert model.n_jobs == 1
    reference = model.predict(batch)
    started = datetime.now(timezone.utc).isoformat()
    rows = []
    for run in range(1, 6):
        for _ in range(3):
            model.predict(batch)
        for repeat in range(1, 16):
            start = time.perf_counter()
            prediction = model.predict(batch)
            elapsed_ms = (time.perf_counter() - start) * 1000
            np.testing.assert_array_equal(prediction, reference)
            rows.append({"round": run, "repeat": repeat, "batch_samples": len(batch), "elapsed_ms": elapsed_ms})
    frame = pd.DataFrame(rows)
    rounds = frame.groupby("round").elapsed_ms.agg(median_ms="median", mean_ms="mean", sd_ms="std").reset_index()
    rounds["p95_ms"] = frame.groupby("round").elapsed_ms.quantile(.95).to_numpy()
    summary = {"started_at_utc": started, "ended_at_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "saved selected model; development-only inference; same local session",
               "rounds": 5, "warmups_per_round": 3, "repeats_per_round": 15, "total_measurements": 75,
               "batch_samples": len(batch), "batch_global_indices": ids, "n_jobs": 1,
               "all_predictions_match": True, "test_feature_evaluations": 0,
               "overall_median_ms": float(frame.elapsed_ms.median()),
               "overall_p95_ms": float(frame.elapsed_ms.quantile(.95)),
               "round_median_mean_ms": float(rounds.median_ms.mean()),
               "round_median_sd_ms": float(rounds.median_ms.std(ddof=1)),
               "round_median_cv_percent": float(rounds.median_ms.std(ddof=1)/rounds.median_ms.mean()*100),
               "limitation": "No training or serialization is timed; batch latency is not single-request latency. Five quick rounds on one host/session do not establish cross-session/platform stability."}
    frame.to_csv(path, index=False)
    rounds.to_csv(OUT / "latency_rounds.csv", index=False)
    write_json(OUT / "latency_benchmark.json", summary)
    fig, ax = plt.subplots(figsize=(7.8, 4.4), dpi=150)
    for run, values in frame.groupby("round"):
        ax.plot(values.repeat, values.elapsed_ms, ".-", lw=.8, label=f"Round {run}")
    ax.set(xlabel="Timed repeat within round", ylabel="Prediction batch latency (ms)",
           title="Selected model on 256 fixed development samples; raw measurements")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "latency_stability.png")
    plt.close(fig)
    verify_originals(proof)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def validate():
    verify_originals(json.loads((OUT / "original_artifacts.json").read_text(encoding="utf-8"))["sha256"])
    predictions = pd.read_csv(BASE / "test_predictions.csv")
    errors, pairs = failure_records(predictions)
    pd.testing.assert_frame_equal(errors, pd.read_csv(OUT / "selected_errors.csv"))
    pd.testing.assert_frame_equal(pairs, pd.read_csv(OUT / "confusion_pairs.csv"))
    summary, rows, latency, runtimes = compare_runs(BASE, REPRO)
    stored = json.loads((OUT / "reproduction_comparison.json").read_text(encoding="utf-8"))
    if summary != stored:
        raise AssertionError("Stored reproduction summary differs from recomputed evidence.")
    for name, expected in [("metric_comparison.csv", rows), ("latency_comparison.csv", latency),
                           ("runtime_by_method_seed.csv", runtimes)]:
        pd.testing.assert_frame_equal(pd.DataFrame(expected), pd.read_csv(OUT / name), check_exact=False, atol=1e-12, rtol=1e-12)
    matrix = confusion_matrix(predictions.y_true, predictions.selected, labels=np.arange(10))
    by_class = pd.read_csv(OUT / "errors_by_class.csv")
    np.testing.assert_array_equal(by_class.support, matrix.sum(axis=1))
    np.testing.assert_array_equal(by_class.errors, matrix.sum(axis=1) - matrix.diagonal())
    np.testing.assert_allclose(by_class.recall, matrix.diagonal()/matrix.sum(axis=1), rtol=0, atol=1e-12)
    info = json.loads((OUT / "error_summary.json").read_text(encoding="utf-8"))
    se, re = predictions.selected != predictions.y_true, predictions.random != predictions.y_true
    assert info["selected_errors"] == int(se.sum())
    assert info["random_errors"] == int(re.sum())
    assert info["selected_errors_random_corrects"] == int((se & ~re).sum())
    assert info["random_errors_selected_corrects"] == int((~se & re).sum())
    assert info["errors_shared_by_selected_and_random"] == int((se & re).sum())
    gap = pd.read_csv(OUT / "generalization_gaps.csv")
    np.testing.assert_allclose(gap.train_minus_validation_pp, (gap.train_accuracy-gap.validation_accuracy_mean)*100, rtol=0, atol=1e-12)
    np.testing.assert_allclose(gap.train_minus_test_pp, (gap.train_accuracy-gap.test_accuracy)*100, rtol=0, atol=1e-12)
    for filename in ["selected_test_errors.png", "confusion_matrix.png"]:
        assert (FIGS / filename).is_file()
    measurement = pd.read_csv(OUT / "latency_measurements.csv")
    benchmark_info = json.loads((OUT / "latency_benchmark.json").read_text(encoding="utf-8"))
    assert len(measurement) == 75 and (measurement.batch_samples == 256).all()
    assert (measurement.elapsed_ms > 0).all()
    assert sorted(measurement["round"].unique().tolist()) == [1, 2, 3, 4, 5]
    assert benchmark_info["test_feature_evaluations"] == 0 and benchmark_info["all_predictions_match"]
    split = json.loads((BASE / "split_indices.json").read_text(encoding="utf-8"))
    assert benchmark_info["batch_global_indices"] == split["development"][:256]
    for run, data in measurement.groupby("round"):
        assert data.repeat.tolist() == list(range(1, 16))
    np.testing.assert_allclose(measurement.elapsed_ms.median(), benchmark_info["overall_median_ms"], rtol=1e-12)
    np.testing.assert_allclose(measurement.elapsed_ms.quantile(.95), benchmark_info["overall_p95_ms"], rtol=1e-12)
    rounds = measurement.groupby("round").elapsed_ms.agg(median_ms="median", mean_ms="mean", sd_ms="std").reset_index()
    rounds["p95_ms"] = measurement.groupby("round").elapsed_ms.quantile(.95).to_numpy()
    pd.testing.assert_frame_equal(rounds, pd.read_csv(OUT / "latency_rounds.csv"), check_exact=False, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(rounds.median_ms.mean(), benchmark_info["round_median_mean_ms"], rtol=1e-12)
    np.testing.assert_allclose(rounds.median_ms.std(ddof=1), benchmark_info["round_median_sd_ms"], rtol=1e-12)
    assert (FIGS / "latency_stability.png").is_file()
    print("PASS: week10 reproduction, all selected errors and original artifact preservation verified.")


def write_report():
    validate()
    summary = json.loads((OUT / "reproduction_comparison.json").read_text(encoding="utf-8"))
    errors = pd.read_csv(OUT / "selected_errors.csv")
    info = json.loads((OUT / "error_summary.json").read_text(encoding="utf-8"))
    gaps = pd.read_csv(OUT / "generalization_gaps.csv")
    timing = pd.read_csv(OUT / "runtime_by_method_seed.csv")
    pair = pd.read_csv(OUT / "confusion_pairs.csv")
    proof = json.loads((OUT / "original_artifacts.json").read_text(encoding="utf-8"))["sha256"]
    bench = json.loads((OUT / "latency_benchmark.json").read_text(encoding="utf-8"))
    lines = ["# Báo cáo checkpoint tuần học 10", "",
             "- **Ngày thực hiện:** 09/11/2026.",
             "- **Nội dung:** tái lập protocol đã khóa, đối chiếu kết quả và phân tích lỗi/giới hạn.",
             "- **Người chạy phiên này:** Codex; chưa có xác nhận sinh viên đã tự chạy hoặc cô duyệt.",
             "- **Báo cáo Word cập nhật:** `Bao_cao_Train_Validation_Test_Baselines_2026-11-09.docx`.", "",
             "## 1 Việc mới thực hiện", "",
             "1. Chạy lại 180 search trials vào `results/week10_2026-11-09/reproduction/`, cùng config/seed/folds và quy tắc epsilon.",
             "2. Đối chiếu toàn bộ trials, đại diện, model và predictions với bộ gốc; giữ nguyên các số đo runtime thay vì ép khớp.",
             "3. Lưu đầy đủ 10 mẫu test selected sai, ma trận nhầm lẫn, cặp nhầm và khoảng cách train–validation–test.",
             "4. Bổ sung validator, tests và hướng dẫn sinh viên tự chạy; cập nhật Word bằng bản ngày 09/11, giữ bản 03/11.", "",
             "Không điều chỉnh tham số, dữ liệu, epsilon hoặc chọn lại model theo các lỗi test. "
             "Test được tính lại trong lần tái lập là cùng holdout đã biết, không phải đánh giá độc lập mới.", "",
             "## 2 Tái lập kết quả", "",
             "Lệnh thực sự đã chạy:", "", "```powershell",
             ".\\.venv\\Scripts\\python.exe -X utf8 -u run_experiment.py --config configs/experiment.json --output results/week10_2026-11-09/reproduction",
             "```", "",
             f"Đối chiếu **{summary['trial_records_compared']} bản ghi**: một baseline và **{summary['search_trials_reproduced']} search trials**. "
             f"Tham số, fold scores, tính khả thi và model size của từng trial khớp; sai số CV tối đa **{summary['max_absolute_trial_cv_delta']:.3g}**.", "",
             f"Config, split, quyết định và toàn bộ test predictions khớp. Binary selected model khớp hash: **{summary['selected_model_binary_sha256_matches']}**. "
             f"Phiên bản thư viện khớp: **{summary['versions_match']}**. Hash xác nhận **{len(proof)} artifact cũ** không đổi.", "",
             "Các file mới giữ riêng; model bàn giao và selection gốc không bị ghi đè.", "",
             "## 3 Runtime và giới hạn phép đo", "",
             f"Runtime pipeline gốc: **{summary['original_runtime_seconds']:.2f} giây**; tái lập: "
             f"**{summary['reproduction_runtime_seconds']:.2f} giây**; tỷ số mới/cũ **{summary['runtime_ratio_new_over_original']:.2f}**.", "",
             "| Phương pháp | Seed | CV search gốc giây | CV search tái lập giây |",
             "|---|---:|---:|---:|"]
    for row in timing.itertuples():
        lines.append(f"| {row.method} | {row.seed} | {row.original_cv_search_seconds:.2f} | {row.reproduction_cv_search_seconds:.2f} |")
    lines += ["", "Runtime toàn pipeline bao gồm CV, refit, profiling/logging; bảng trên chỉ cộng phần CV. "
              "Latency median/p95 theo đại diện nằm trong `latency_comparison.csv`. Hai phiên khác ngày và tải nền, "
              "chỉ có một lần chạy lại nên chưa đủ ước lượng độ ổn định runtime hoặc kết luận tốc độ thuật toán.", "",
              f"Sau khi HPO kết thúc, đo riêng latency selected trên batch cố định 256 mẫu development: 5 rounds, "
              f"3 warmups/round và 15 phép đo/round (75 số đo). Median chung **{bench['overall_median_ms']:.4f} ms/batch**, "
              f"p95 **{bench['overall_p95_ms']:.4f} ms/batch**; mean ± SD của năm median **{bench['round_median_mean_ms']:.4f} ± {bench['round_median_sd_ms']:.4f} ms**. "
              "Toàn bộ số đo được giữ, không loại outlier. Đây không phải single-request latency hay bằng chứng ổn định trên nhiều phiên/máy.", "",
              "![Latency từng lần đo](../figures/week10_2026-11-09/latency_stability.png)", "",
              "## 4 Toàn bộ lỗi selected trên test gốc", "",
              "| Global index | Nhãn thật | Selected dự đoán | Random dự đoán | Random đúng |",
              "|---:|---:|---:|---:|---|"]
    for row in errors.itertuples():
        lines.append(f"| {row.dataset_index} | {row.y_true} | {row.selected_prediction} | {row.random} | {bool(row.random_correct)} |")
    lines += ["", "![Toàn bộ 10 mẫu selected sai](../figures/week10_2026-11-09/selected_test_errors.png)", "",
              "| Nhãn thật | Nhãn dự đoán | Số lần |", "|---:|---:|---:|"]
    for row in pair.itertuples():
        lines.append(f"| {row.y_true} | {row.selected_prediction} | {row.count} |")
    lines += ["", f"Random sửa đúng **{info['selected_errors_random_corrects']}** lỗi selected nhưng sai thêm "
              f"**{info['random_errors_selected_corrects']}** mẫu selected đúng; có **{info['errors_shared_by_selected_and_random']}** lỗi chung. "
              "Các ảnh 8×8 cho thấy thông tin nét hữu hạn; nhận xét trực quan không phải chứng minh nguyên nhân sai.", "",
              "![Confusion matrix test gốc](../figures/week10_2026-11-09/confusion_matrix.png)", "",
              "## 5 Khoảng cách train validation test", "",
              "| Đại diện | Train accuracy | Validation accuracy | Test accuracy | Train trừ val điểm phần trăm |",
              "|---|---:|---:|---:|---:|"]
    for row in gaps.itertuples():
        lines.append(f"| {row.label} | {row.train_accuracy*100:.2f}% | {row.validation_accuracy_mean*100:.2f}% | {row.test_accuracy*100:.2f}% | {row.train_minus_validation_pp:.3f} |")
    lines += ["", "Train là in-sample trên development đã fit. Validation là mean qua ba folds; CV đã tham gia tuning. "
              "Train 100% không chứng minh tổng quát hóa hoàn hảo. Test nhỏ và đã biết; không dùng các cặp nhầm để tuning lại.", "",
              "## 6 Kiểm chứng và bằng chứng", "", "```powershell",
              ".\\.venv\\Scripts\\python.exe -X utf8 -m src.validation --output results/week10_2026-11-09/reproduction",
              ".\\.venv\\Scripts\\python.exe -X utf8 -m src.week10 validate",
              ".\\.venv\\Scripts\\python.exe -X utf8 -m unittest discover -s tests -v", "```", "",
              "Pipeline tái lập đã chạy bộ kiểm chứng kết quả gốc (401 checks). Validator tuần 10 đối chiếu từng trial, "
              "cấu hình, fold scores, decisions, predictions, bảng phân tích lỗi và hashes. Tests mới phát hiện trial "
              "bị sửa/duplicate index, kiểm tra chiều cặp nhầm và cho phép runtime thực thay đổi.", "",
              "Nguồn: thư mục `results/week10_2026-11-09/`, `src/week10.py`, `tests/test_week10.py`. "
              "Không chạy lại phần mở rộng 450 trials chỉ để bổ sung tài liệu tuần 10.", "",
              "## 7 Phần sinh viên và giảng viên cần xác nhận", "",
              "Sinh viên làm theo [hướng dẫn tự chạy](../docs/tu-chay-tai-lap-tuan-10.md), đọc log và giải thích kết quả. "
              "Phiên do trợ lý chạy không tự xác nhận sinh viên đã tự tái lập. Chưa có feedback hoặc duyệt checkpoint tuần 10 từ cô.", "",
              "Tuần 11 còn kiểm tra cuối, chốt bản nộp/tag v1.0, tập demo/bảo vệ và nộp link. "
              "Chưa tạo release cuối hoặc đánh dấu nộp bài trong checkpoint này.", ""]
    verification = OUT / "verification.json"
    if verification.exists():
        verified = json.loads(verification.read_text(encoding="utf-8"))
        lines += ["## 8 Xác minh sau khi hoàn thiện", "",
                  f"Đã chạy **{verified['unit_tests_passed']} unit tests** đạt; bộ gốc **401 checks**, "
                  "phần mở rộng **3.278 checks**, report evaluation, demo tuần 9 và validator tuần 10 đạt. "
                  f"Bản Word mới có **{verified['word_pages']} trang**, đã render và kiểm tra từng trang.", "",
                  "Renderer LibreOffice thiếu soffice.exe trên Windows; dùng Microsoft Word mở read-only, "
                  "export PDF tạm và Poppler render PNG để kiểm tra. PDF/PNG QA ở ngoài repo, không phải file bàn giao.", "",
                  "Kết quả xác minh nằm trong `verification.json`; việc kiểm chứng của trợ lý không thay xác nhận "
                  "sinh viên tự chạy hoặc giảng viên duyệt.", ""]
    (ROOT / "report/checkpoint-tuan-10-2026-11-09.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote week10 checkpoint report from verified evidence.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["errors", "compare", "benchmark", "validate", "report"])
    mode = parser.parse_args().mode
    {"errors": analyze_errors, "compare": compare, "benchmark": benchmark, "validate": validate, "report": write_report}[mode]()
