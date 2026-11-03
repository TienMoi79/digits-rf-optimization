"""Supplement train/CV tables for fixed original representatives.

Refits only development data. Test metrics use published predictions, never
new test feature predictions. Original artifacts are checked for preservation.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss, precision_score, recall_score

from .common import ROOT, make_model, write_json
from .optimization_demo import protected_hashes
from .robustness import canonical_sha256, load_development

OUT = ROOT / "results/report_evaluation_2026-11-03"


def metrics(y, prediction, probability=None):
    result = {"accuracy": float(accuracy_score(y, prediction)),
              "precision_macro": float(precision_score(y, prediction, average="macro", zero_division=0)),
              "recall_macro": float(recall_score(y, prediction, average="macro", zero_division=0)),
              "f1_macro": float(f1_score(y, prediction, average="macro", zero_division=0))}
    if probability is not None:
        result["log_loss"] = float(log_loss(y, probability, labels=np.arange(10)))
    return result


def generate():
    if OUT.exists():
        raise FileExistsError("Report evaluation already exists; preserve it or choose a new dated folder in source.")
    before = protected_hashes()
    x, y, folds, cfg, data_sha = load_development(ROOT / "results")
    decision = json.loads((ROOT / "results/selection.json").read_text(encoding="utf-8"))
    saved = pd.read_csv(ROOT / "results/test_predictions.csv")
    final = pd.read_csv(ROOT / "results/final_metrics.csv").set_index("label")
    split = json.loads((ROOT / "results/split_indices.json").read_text(encoding="utf-8"))
    assert saved.dataset_index.tolist() == split["test"]
    cache, records, cv_rows, classes = {}, [], [], {}
    for label, candidate in decision["candidates"].items():
        key = json.dumps(candidate["params"], sort_keys=True)
        if key not in cache:
            model = make_model(candidate["params"], cfg).fit(x, y)
            train_pred = model.predict(x)
            train = metrics(y, train_pred, model.predict_proba(x))
            oof = np.full(len(y), -1, dtype=int)
            cv = []
            for fold, (a, b) in enumerate(folds):
                m = make_model(candidate["params"], cfg).fit(x[a], y[a])
                pred = m.predict(x[b])
                oof[b] = pred
                score = metrics(y[b], pred, m.predict_proba(x[b]))
                assert np.isclose(score["f1_macro"], candidate["fold_scores"][fold], rtol=0, atol=1e-12)
                cv.append({"fold": fold + 1, "n_train": len(a), "n_validation": len(b), **score})
            assert (oof >= 0).all()
            cv_mean = {f"{name}_mean": float(np.mean([r[name] for r in cv])) for name in train}
            cv_sd = {f"{name}_sd": float(np.std([r[name] for r in cv], ddof=1)) for name in train}
            assert np.isclose(cv_mean["f1_macro_mean"], candidate["cv_f1"], rtol=0, atol=1e-12)
            per_class = {"train": classification_report(y, train_pred, output_dict=True, zero_division=0),
                         "validation_oof": classification_report(y, oof, output_dict=True, zero_division=0)}
            cache[key] = (train, cv, cv_mean | cv_sd, per_class)
            if candidate["params"] == decision["selected_params"]:
                original_model = joblib.load(ROOT / "results/models/selected.joblib")
                np.testing.assert_array_equal(train_pred, original_model.predict(x))
        train, cv, cv_summary, per_class = cache[key]
        test = metrics(saved.y_true, saved[label])
        assert np.isclose(test["accuracy"], final.loc[label, "test_accuracy"], rtol=0, atol=1e-12)
        assert np.isclose(test["f1_macro"], final.loc[label, "test_f1"], rtol=0, atol=1e-12)
        records.append({"label": label, "model_mb": candidate["model_bytes"] / 1e6,
                        **{f"train_{k}": v for k, v in train.items()},
                        **{f"validation_{k}": v for k, v in cv_summary.items()},
                        **{f"test_{k}": v for k, v in test.items()},
                        "test_correct": int((saved.y_true == saved[label]).sum()),
                        "test_errors": int((saved.y_true != saved[label]).sum())})
        cv_rows.extend({"label": label, **row} for row in cv)
        classes[label] = {**per_class, "test": classification_report(saved.y_true, saved[label], output_dict=True, zero_division=0)}
        print(f"{label}: train={train['accuracy']:.5f}, validation={cv_summary['accuracy_mean']:.5f}, test={test['accuracy']:.5f}", flush=True)
    assert all(canonical_sha256(ROOT / p) == digest for p, digest in before.items())
    counts = [{"class": c, "development": int((y == c).sum()), "test": int((saved.y_true == c).sum()),
               "total": int((y == c).sum() + (saved.y_true == c).sum())} for c in range(10)]
    OUT.mkdir(parents=True)
    pd.DataFrame(records).to_csv(OUT / "comparison.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(OUT / "validation_per_fold.csv", index=False)
    pd.DataFrame(counts).to_csv(OUT / "class_counts.csv", index=False)
    write_json(OUT / "classification_reports.json", classes)
    write_json(OUT / "run_manifest.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "report tables for fixed original representatives, not model selection",
        "train_definition": "in-sample on all 1437 development samples after refit",
        "validation_definition": "mean and sample SD of 3 fixed folds; OOF per-class report is separate",
        "test_source": "original test_predictions.csv, no fresh test feature inference",
        "new_test_feature_evaluations": 0, "hyperparameter_search_trials": 0,
        "distinct_configs": len(cache), "fresh_development_fits": len(cache) * 4,
        "development_samples": len(y), "test_samples": len(saved), "development_sha256": data_sha,
        "fold_sizes": [{"train": len(a), "validation": len(b)} for a, b in folds],
        "original_artifacts_preserved": True, "protected_sha256": before})
    print(f"PASS: fixed CV scores/test metrics reproduced; {len(before)} original artifacts preserved.")


def validate():
    """Check saved report tables against fold/classification and frozen evidence."""
    table = pd.read_csv(OUT / "comparison.csv").set_index("label")
    folds = pd.read_csv(OUT / "validation_per_fold.csv")
    reports = json.loads((OUT / "classification_reports.json").read_text(encoding="utf-8"))
    manifest = json.loads((OUT / "run_manifest.json").read_text(encoding="utf-8"))
    final = pd.read_csv(ROOT / "results/final_metrics.csv").set_index("label")
    decision = json.loads((ROOT / "results/selection.json").read_text(encoding="utf-8"))
    assert set(table.index) == {"default", "random", "tpe", "selected"}
    for label in table.index:
        cv = folds[folds.label == label].sort_values("fold")
        assert cv.fold.tolist() == [1, 2, 3]
        np.testing.assert_allclose(cv.f1_macro, decision["candidates"][label]["fold_scores"], rtol=0, atol=1e-12)
        for metric in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "log_loss"]:
            assert np.isclose(cv[metric].mean(), table.loc[label, "validation_" + metric + "_mean"], atol=1e-12, rtol=0)
            assert np.isclose(cv[metric].std(ddof=1), table.loc[label, "validation_" + metric + "_sd"], atol=1e-12, rtol=0)
        for split in ["train", "test"]:
            assert np.isclose(table.loc[label, split + "_f1_macro"], reports[label][split]["macro avg"]["f1-score"], atol=1e-12, rtol=0)
            assert np.isclose(table.loc[label, split + "_accuracy"], reports[label][split]["accuracy"], atol=1e-12, rtol=0)
        assert table.loc[label, "test_correct"] + table.loc[label, "test_errors"] == 360
        assert np.isclose(table.loc[label, "test_accuracy"], final.loc[label, "test_accuracy"], atol=1e-12, rtol=0)
        assert np.isclose(table.loc[label, "test_f1_macro"], final.loc[label, "test_f1"], atol=1e-12, rtol=0)
        assert np.isclose(table.loc[label, "model_mb"], final.loc[label, "model_mb"], atol=1e-12, rtol=0)
    counts = pd.read_csv(OUT / "class_counts.csv")
    assert counts.development.sum() == 1437 and counts.test.sum() == 360 and counts.total.sum() == 1797
    assert manifest["new_test_feature_evaluations"] == manifest["hyperparameter_search_trials"] == 0
    assert manifest["original_artifacts_preserved"]
    assert all(canonical_sha256(ROOT / p) == digest for p, digest in manifest["protected_sha256"].items())
    print("PASS: report train/CV/test tables match saved evidence and original artifact hashes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate saved tables without retraining.")
    if parser.parse_args().check:
        validate()
    else:
        generate()
