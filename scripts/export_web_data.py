"""Export frozen experiment evidence and the fitted RF for a buildless web demo.

No fit, no search and no held-out feature inference. Development-only fixtures
verify that browser tree traversal reproduces sklearn's fitted probabilities.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_digits

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def read_csv(name):
    with (ROOT / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")


def digest(path):
    data = path.read_bytes()
    if path.suffix in {".json", ".csv", ".md", ".ipynb"}:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def export():
    protected = [p for folder in ["results", "figures", "configs", "notebooks"]
                 for p in (ROOT / folder).rglob("*") if p.is_file()]
    protected += list((ROOT / "report").glob("*.docx"))
    before = {p: digest(p) for p in protected}
    model = joblib.load(ROOT / "results/models/selected.joblib")
    forest = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        values = []
        for node in range(tree.node_count):
            if tree.children_left[node] != -1:
                values.append(None)
                continue
            probabilities = tree.value[node, 0] / tree.value[node, 0].sum()
            values.append(int(np.argmax(probabilities)) if np.count_nonzero(probabilities) == 1 else probabilities.tolist())
        forest.append({"left": tree.children_left.tolist(), "right": tree.children_right.tolist(),
                       "feature": tree.feature.tolist(), "threshold": tree.threshold.tolist(), "value": values})
    model_data = {"classes": model.classes_.tolist(), "features": int(model.n_features_in_), "trees": forest,
                  "source_sha256": before[ROOT / "results/models/selected.joblib"]}
    write(WEB / "dist/data/forest.json", model_data)
    split = read_json("results/split_indices.json")
    digits = load_digits()
    development = np.array(split["development"])
    sample_indices = [int(i) for label in range(10) for i in development[digits.target[development] == label][:3]]
    samples = [{"index": i, "label": int(digits.target[i]), "pixels": digits.data[i].tolist()} for i in sample_indices]
    metrics = [{k: (v if k == "label" else float(v)) for k, v in row.items()}
               for row in read_csv("results/report_evaluation_2026-11-03/comparison.csv")]
    curves = [{k: (v if k == "split" else float(v)) for k, v in row.items()}
              for row in read_csv("results/train_validation_two_lines/mean_metrics.csv")]
    trials = [{k: row[k] for k in ["method", "seed", "trial", "params", "cv_f1", "model_bytes", "cv_seconds"]}
              for row in read_json("results/trials.json")]
    selection = read_json("results/selection.json")
    manifest = read_json("results/run_manifest.json")
    saved_errors = read_csv("results/week10_2026-11-09/selected_errors.csv")
    errors = [{"index": int(r["dataset_index"]), "actual": int(r["y_true"]),
               "predicted": int(r["selected_prediction"]), "pixels": digits.data[int(r["dataset_index"])].tolist()}
              for r in saved_errors]
    data = {"schema_version": 1, "dataset": {"name": "Digits", "samples": 1797, "features": 64,
             "classes": 10, "development": len(development), "test": len(split["test"]), "folds": 3},
            "config": manifest["config"], "runtime_seconds": manifest["runtime_seconds"],
            "metrics": metrics, "selection": {k: selection[k] for k in ["epsilon", "best_cv_f1", "minimum_acceptable_cv_f1", "selected_params"]},
            "trials": trials, "curves": curves, "samples": samples, "errors": errors,
            "confusion": np.loadtxt(ROOT / "results/confusion_matrix.csv", delimiter=",", dtype=int).tolist(),
            "reproduction": read_json("results/week10_2026-11-09/reproduction_comparison.json"),
            "provenance": {"model": "results/models/selected.joblib", "metrics": "results/report_evaluation_2026-11-03/comparison.csv",
                           "trials": "results/trials.json", "curves": "results/train_validation_two_lines/mean_metrics.csv",
                           "note": "Frozen original protocol. No training or model selection in this website."}}
    assert len(trials) == 181 and len(curves) == 200 and len(samples) == 30 and len(errors) == 10
    assert set(sample_indices).issubset(set(split["development"]))
    write(WEB / "dist/data/dashboard.json", data)
    columns = ["label", "train_accuracy", "validation_accuracy_mean", "validation_f1_macro_mean", "test_accuracy", "model_mb"]
    with (WEB / "dist/data/comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics)
    # Float32 matters for exact sklearn threshold behavior on fractional inputs.
    rng = np.random.default_rng(20261116)
    fixtures_x = np.vstack([digits.data[development], np.zeros((1, 64)), np.full((1, 64), 16),
                            rng.uniform(0, 16, (32, 64))]).astype(np.float32)
    expected = model.predict_proba(fixtures_x)
    write(WEB / "tests/model-fixtures.json", {"inputs": fixtures_x.tolist(), "probabilities": expected.tolist(),
          "predictions": model.predict(fixtures_x).tolist(), "development_samples": len(development),
          "synthetic_samples": len(fixtures_x) - len(development), "test_evaluations": 0})
    assert all(digest(p) == sha for p, sha in before.items())
    write(WEB / "tests/export-checks.json", {"original_artifacts_preserved": len(before), "search_trials": len(trials) - 1,
          "curve_rows": len(curves), "gallery_samples": len(samples), "model_trees": len(forest),
          "test_feature_inference": 0, "hash_policy": "Git text normalizes CRLF to LF; binaries use exact bytes.",
          "source_sha256": {p.relative_to(ROOT).as_posix(): before[p] for p in before},
          "export_sha256": {p.relative_to(WEB).as_posix(): digest(p) for p in [WEB / "dist/data/dashboard.json", WEB / "dist/data/forest.json", WEB / "dist/data/comparison.csv", WEB / "tests/model-fixtures.json"]}})
    print(f"PASS: exported {len(forest)} trees and frozen evidence; preserved {len(before)} artifacts; no training.")


def check():
    proof = json.loads((WEB / "tests/export-checks.json").read_text(encoding="utf-8"))
    assert all(digest(ROOT / p) == sha for p, sha in proof["source_sha256"].items())
    assert all(digest(WEB / p) == sha for p, sha in proof["export_sha256"].items())
    data = json.loads((WEB / "dist/data/dashboard.json").read_text(encoding="utf-8"))
    expected_metrics = [{k: (v if k == "label" else float(v)) for k, v in row.items()}
                        for row in read_csv("results/report_evaluation_2026-11-03/comparison.csv")]
    assert data["metrics"] == expected_metrics
    expected_trials = [{k: r[k] for k in ["method", "seed", "trial", "params", "cv_f1", "model_bytes", "cv_seconds"]}
                       for r in read_json("results/trials.json")]
    assert data["trials"] == expected_trials
    split = read_json("results/split_indices.json")
    digits = load_digits()
    assert len(data["samples"]) == 30
    for sample in data["samples"]:
        assert sample["index"] in split["development"]
        assert sample["pixels"] == digits.data[sample["index"]].tolist()
        assert sample["label"] == int(digits.target[sample["index"]])
    fixtures = json.loads((WEB / "tests/model-fixtures.json").read_text(encoding="utf-8"))
    inputs = np.asarray(fixtures["inputs"], dtype=np.float32)
    np.testing.assert_array_equal(inputs[:1437], digits.data[split["development"]])
    model = joblib.load(ROOT / "results/models/selected.joblib")
    np.testing.assert_allclose(model.predict_proba(inputs), fixtures["probabilities"], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(model.predict(inputs), fixtures["predictions"])
    assert fixtures["test_evaluations"] == proof["test_feature_inference"] == 0
    print(f"PASS: web exports match frozen sources; {len(inputs)} development/synthetic fixtures verified against sklearn.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify committed exports without writing files.")
    check() if parser.parse_args().check else export()
