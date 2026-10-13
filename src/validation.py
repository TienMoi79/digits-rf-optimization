"""Independent checks of split isolation, budgets, decision and test metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .common import ROOT, check_feasible, fingerprint, write_json
from .analysis import pareto_mask


def validate_results(output):
    output = Path(output)
    read = lambda name: json.loads((output / name).read_text(encoding="utf-8"))
    manifest, split, decision, rows = [read(p) for p in
                                      ["run_manifest.json", "split_indices.json", "selection.json", "trials.json"]]
    cfg = manifest["config"]
    checks = []

    def require(condition, message):
        if not condition:
            raise AssertionError(message)
        checks.append(message)

    require(manifest["status"] == "completed", "Experiment completed")
    require(fingerprint(cfg) == manifest["config_sha256"], "Config fingerprint matches")
    data = load_digits()
    require(hashlib.sha256(data.data.tobytes()+data.target.tobytes()).hexdigest() == split["dataset_sha256"], "Dataset checksum matches")
    development, test = set(split["development"]), set(split["test"])
    require(len(development) == len(split["development"]) and len(test) == len(split["test"]), "No duplicate split indices")
    require(not development & test and development | test == set(range(len(data.target))), "Development and test are disjoint and exhaustive")
    validation_ids = []
    require(len(split["cv_folds_global"]) == cfg["cv_folds"], "Correct number of CV folds")
    for fold in split["cv_folds_global"]:
        train, val = set(fold["train"]), set(fold["validation"])
        require(not train & val and train | val == development and not (train | val) & test,
                "CV fold contains development data only with disjoint training and validation")
        validation_ids.extend(fold["validation"])
        require(set(data.target[list(train)]) == set(range(10)) and set(data.target[list(val)]) == set(range(10)),
                "Every CV fold contains all ten classes")
    require(len(validation_ids) == len(set(validation_ids)) == len(development), "Each development sample validates exactly once")
    searches = [r for r in rows if r["method"] != "default"]
    require(len(searches) == 2*len(cfg["search_seeds"])*cfg["trials_per_method"], "Both search methods have equal total budgets")
    for method in ["random", "tpe"]:
        for seed in cfg["search_seeds"]:
            run = [r for r in searches if r["method"] == method and r["seed"] == seed]
            require(sorted(r["trial"] for r in run) == list(range(cfg["trials_per_method"])), "Every method and seed has its complete trial budget")
    for r in rows:
        require(check_feasible(r["params"], cfg) and r["feasible"], "Recorded configuration meets search-space constraints")
        require(len(r["fold_scores"]) == cfg["cv_folds"] and np.isclose(np.mean(r["fold_scores"]), r["cv_f1"], atol=1e-12), "CV objective equals saved fold mean")
    threshold = max(r["cv_f1"] for r in rows) - cfg["epsilon"]
    eligible = [r for r in rows if r["cv_f1"] >= threshold and r["feasible"]]
    chosen = decision["candidates"]["selected"]
    require(chosen["cv_f1"] >= threshold and chosen["model_bytes"] == min(r["model_bytes"] for r in eligible), "Selected model is smallest within the CV epsilon threshold")
    require(chosen["params"] == decision["selected_params"], "Saved deployment parameters match locked decision")
    for name, candidate in decision["candidates"].items():
        require(candidate in rows, "Final candidate is traceable to persisted CV log")
        if name in ["random", "tpe"]:
            require(candidate["cv_f1"] == max(r["cv_f1"] for r in rows if r["method"] == name), "Method representative is chosen by CV score")
    predictions = pd.read_csv(output / "test_predictions.csv")
    require(predictions.dataset_index.tolist() == split["test"], "Test prediction indices match held-out split")
    y = data.target[split["test"]]
    require(np.array_equal(y, predictions.y_true.to_numpy()), "Saved test targets match original dataset")
    final = pd.read_csv(output / "final_metrics.csv")
    require(set(final.label) == {"default", "random", "tpe", "selected"}, "All predetermined model representatives are evaluated")
    for r in final.itertuples():
        pred = predictions[r.label].to_numpy()
        require(np.isclose(f1_score(y, pred, average="macro"), r.test_f1, atol=1e-12)
                and np.isclose(accuracy_score(y, pred), r.test_accuracy, atol=1e-12), "Final metrics recompute from saved predictions")
    model = joblib.load(output / "models/selected.joblib")
    require(np.array_equal(model.predict(data.data[split["test"]]), predictions.selected.to_numpy()), "Saved model reproduces selected test predictions")
    require((output / "models/selected.joblib").stat().st_size == chosen["model_bytes"], "Saved model file size matches profiling")
    require(np.array_equal(np.loadtxt(output / "confusion_matrix.csv", delimiter=",", dtype=int),
                           confusion_matrix(y, predictions.selected)), "Confusion matrix recomputes from predictions")
    pareto = pd.read_csv(output / "pareto_candidates.csv")
    require(np.array_equal(pareto.pareto_non_dominated.to_numpy(), pareto_mask(pareto.cv_f1, pareto.model_bytes)), "Observed Pareto flags recompute")
    write_json(output / "validation.json", {"passed": True, "check_count": len(checks), "checks": checks})
    print(f"PASS: {len(checks)} result integrity checks", flush=True)
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    validate_results(parser.parse_args().output)
