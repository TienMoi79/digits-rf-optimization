"""Baseline, fair HPO, an epsilon decision, and a final held-out evaluation."""
from __future__ import annotations

import io
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score

from .common import (DEFAULT_PARAMS, check_feasible, fingerprint, make_data,
                     make_model, write_json)

optuna.logging.set_verbosity(optuna.logging.WARNING)
METHODS = ["random", "tpe"]


def profile(model, x, repeats):
    """Use an uncompressed joblib byte stream and warmed full-batch predict.

    ms/sample is batch time divided by batch size, not single-request latency.
    Measurements cover CPU inference without file IO or GUI overhead.
    """
    stream = io.BytesIO()
    joblib.dump(model, stream, compress=0)
    for _ in range(3):
        model.predict(x)
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(x)
        durations.append((time.perf_counter() - start) * 1000)
    return {"model_bytes": stream.tell(),
            "total_nodes": sum(tree.tree_.node_count for tree in model.estimators_),
            "batch_predict_ms_median": float(np.median(durations)),
            "batch_predict_ms_p95": float(np.percentile(durations, 95)),
            "batch_samples": len(x),
            "ms_per_sample": float(np.median(durations) / len(x))}


def evaluate_cv(params, x, y, folds, config):
    start = time.perf_counter()
    scores = cross_val_score(make_model(params, config), x, y, cv=folds,
                             scoring="f1_macro", n_jobs=1, error_score="raise")
    cv_seconds = time.perf_counter() - start
    model = make_model(params, config)
    start = time.perf_counter()
    model.fit(x, y)
    fit_seconds = time.perf_counter() - start
    # A fixed DEVELOPMENT batch is used for profiling all candidates.
    metrics = profile(model, x[:256], config["latency_repeats"])
    return {"cv_f1": float(scores.mean()), "cv_fold_std": float(scores.std(ddof=1)),
            "fold_scores": scores.tolist(), "cv_seconds": cv_seconds,
            "refit_seconds": fit_seconds, "feasible": check_feasible(params, config),
            **metrics}


def sample_params(trial, config):
    s = config["search_space"]
    return {
        "n_estimators": trial.suggest_int("n_estimators", **s["n_estimators"]),
        "max_depth": trial.suggest_categorical("max_depth", s["max_depth"]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", **s["min_samples_leaf"]),
        "max_features": trial.suggest_categorical("max_features", s["max_features"]),
    }


def choose_candidates(rows, epsilon):
    """This function accepts CV metrics and size only; it never sees test data."""
    ordered = sorted(rows, key=lambda r: (-r["cv_f1"], r["model_bytes"],
                                          r["method"], r["seed"], r["trial"]))
    best_score = ordered[0]["cv_f1"]
    eligible = [r for r in ordered if r["cv_f1"] >= best_score - epsilon and r["feasible"]]
    if not eligible:
        raise ValueError("No feasible candidate meets the epsilon criterion.")
    compact = min(eligible, key=lambda r: (r["model_bytes"], -r["cv_f1"],
                                          r["method"], r["seed"], r["trial"]))
    chosen = {"default": next(r for r in rows if r["method"] == "default")}
    for method in METHODS:
        chosen[method] = next(r for r in ordered if r["method"] == method)
    chosen["selected"] = compact
    return chosen, best_score


def run_experiment(config, output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}. Use a NEW --output folder to preserve existing results.")
    output.mkdir(parents=True, exist_ok=True)
    for name in ["models", "figures"]:
        (output / name).mkdir()
    manifest = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running", "config": config, "config_sha256": fingerprint(config),
        "python": sys.version, "platform": platform.platform(),
        "cpu": platform.processor(), "logical_cpu_count": os.cpu_count(),
        "versions": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "optuna", "matplotlib", "joblib"]},
        "protocol": "Fixed stratified test holdout; fixed stratified development CV; RF seed fixed; search seeds vary; no pruning; serial execution."
    }
    write_json(output / "run_manifest.json", manifest)
    digits, dev, test, folds = make_data(config)
    write_json(output / "split_indices.json", {
        "development": dev.tolist(), "test": test.tolist(),
        "cv_folds_global": [{"train": dev[a].tolist(), "validation": dev[b].tolist()} for a, b in folds],
        "dataset_shape": list(digits.data.shape),
        "dataset_sha256": __import__("hashlib").sha256(digits.data.tobytes() + digits.target.tobytes()).hexdigest()
    })
    x, y = digits.data[dev], digits.target[dev]
    rows = []
    start_run = time.perf_counter()
    baseline = {"method": "default", "seed": config["model_seed"], "trial": -1,
                "params": DEFAULT_PARAMS, **evaluate_cv(DEFAULT_PARAMS, x, y, folds, config)}
    rows.append(baseline)
    write_json(output / "baseline.json", baseline)
    write_json(output / "trials.json", rows)
    print(f"Default CV Macro-F1: {baseline['cv_f1']:.5f}", flush=True)
    for method in METHODS:
        for seed in config["search_seeds"]:
            sampler = (optuna.samplers.RandomSampler(seed=seed) if method == "random"
                       else optuna.samplers.TPESampler(seed=seed, n_startup_trials=config["tpe_startup_trials"]))
            study = optuna.create_study(direction="maximize", sampler=sampler,
                                        study_name=f"{method}_{seed}")

            def objective(trial):
                started = datetime.now(timezone.utc).isoformat()
                params = sample_params(trial, config)
                metrics = evaluate_cv(params, x, y, folds, config)
                if not metrics["feasible"]:
                    raise ValueError("Search produced an infeasible configuration.")
                row = {"method": method, "seed": seed, "trial": trial.number,
                       "started_at_utc": started, "params": params, **metrics}
                rows.append(row)
                for key, value in metrics.items():
                    trial.set_user_attr(key, value)
                # Persist after EVERY completed trial, even if the process is interrupted.
                write_json(output / "trials.json", rows)
                if (trial.number + 1) % 5 == 0:
                    best = max(r["cv_f1"] for r in rows if r["method"] == method and r["seed"] == seed)
                    print(f"{method:6s} seed={seed}: {trial.number+1:2d}/{config['trials_per_method']}, best={best:.5f}", flush=True)
                return metrics["cv_f1"]

            study.optimize(objective, n_trials=config["trials_per_method"])
            study.trials_dataframe().to_csv(output / f"optuna_{method}_{seed}.csv", index=False)
    flat = [{k: v for k, v in r.items() if k not in ["params", "fold_scores"]} | r["params"]
            | {"fold_scores": json.dumps(r["fold_scores"])} for r in rows]
    pd.DataFrame(flat).to_csv(output / "trials.csv", index=False)
    chosen, best_score = choose_candidates(rows, config["epsilon"])
    decision = {"rule": "Smallest feasible model with development CV Macro-F1 >= best observed CV Macro-F1 - epsilon.",
                "epsilon": config["epsilon"], "best_cv_f1": best_score,
                "minimum_acceptable_cv_f1": best_score - config["epsilon"],
                "candidates": chosen, "selected_params": chosen["selected"]["params"],
                "locked_at_utc": datetime.now(timezone.utc).isoformat()}
    # Lock the decision on disk BEFORE reading test feature/target values.
    write_json(output / "selection.json", decision)
    x_test, y_test = digits.data[test], digits.target[test]
    test_rows = []
    predictions = pd.DataFrame({"dataset_index": test, "y_true": y_test})
    for label, candidate in chosen.items():
        model = make_model(candidate["params"], config).fit(x, y)
        pred = model.predict(x_test)
        test_rows.append({"label": label, "cv_f1": candidate["cv_f1"],
                          "test_f1": float(f1_score(y_test, pred, average="macro")),
                          "test_accuracy": float(accuracy_score(y_test, pred)),
                          "source_method": candidate["method"], "source_seed": candidate["seed"],
                          "source_trial": candidate["trial"],
                          "model_mb": candidate["model_bytes"] / 1_000_000,
                          "total_nodes": candidate["total_nodes"],
                          "development_batch_ms": candidate["batch_predict_ms_median"]})
        predictions[label] = pred
        write_json(output / f"classification_{label}.json", classification_report(y_test, pred, output_dict=True, zero_division=0))
        if label == "selected":
            joblib.dump(model, output / "models/selected.joblib", compress=0)
            np.savetxt(output / "confusion_matrix.csv", confusion_matrix(y_test, pred), fmt="%d", delimiter=",")
    predictions.to_csv(output / "test_predictions.csv", index=False)
    pd.DataFrame(test_rows).to_csv(output / "final_metrics.csv", index=False)
    manifest["status"] = "completed"
    manifest["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["runtime_seconds"] = time.perf_counter() - start_run
    manifest["completed_search_trials"] = len(rows) - 1
    write_json(output / "run_manifest.json", manifest)
    print(pd.DataFrame(test_rows).to_string(index=False), flush=True)
    return output
