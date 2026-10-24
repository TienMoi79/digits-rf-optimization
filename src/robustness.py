"""Development-only replication and a controlled TPE startup ablation.

No held-out predictions, test metrics, deployment selection or model files are
produced here. The original experiment remains a separate frozen result.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score

from .common import ROOT, check_feasible, fingerprint, make_model, write_json
from .experiment import sample_params

VERSIONS = ["numpy", "pandas", "scikit-learn", "optuna", "matplotlib", "joblib"]


def canonical_sha256(path):
    """Ignore Git line-ending conversion for text; hash binary files verbatim."""
    path = Path(path)
    data = path.read_bytes()
    if path.suffix in {".json", ".csv", ".md"}:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def frozen_artifacts(base):
    base = Path(base)
    # validation.json is a diagnostic rewritten by validation, not a result.
    files = [p for p in base.iterdir() if p.is_file() and p.name != "validation.json"]
    files += list((base / "models").glob("*.joblib"))
    return {p.relative_to(base).as_posix(): canonical_sha256(p) for p in sorted(files)}


def verify_frozen(base, expected):
    actual = frozen_artifacts(base)
    if actual != expected:
        changed = sorted(k for k in actual.keys() | expected.keys() if actual.get(k) != expected.get(k))
        raise AssertionError("Original published artifacts changed: " + ", ".join(changed))


def load_plan(path=None):
    plan = json.loads(Path(path or ROOT / "configs/robustness.json").read_text(encoding="utf-8"))
    seeds = plan["search_seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Search seeds must be distinct nonnegative integers.")
    budget = plan["trials_per_method"]
    if type(budget) is not int or budget < 2:
        raise ValueError("The search budget must be an integer >= 2.")
    startups = plan["tpe_startup_trials"]
    if len(startups) != 2 or len(set(startups)) != 2 or any(type(s) is not int or not 1 <= s < budget for s in startups):
        raise ValueError("Exactly two distinct startup counts below the trial budget are required.")
    checkpoints = plan["checkpoints"]
    if (not checkpoints or checkpoints != sorted(set(checkpoints))
            or any(type(b) is not int or not 1 <= b <= budget for b in checkpoints)
            or checkpoints[-1] != budget):
        raise ValueError("Checkpoints must be ordered, unique and end at the full trial budget.")
    if type(plan["time_grid_points"]) is not int or plan["time_grid_points"] < 2:
        raise ValueError("At least two time-grid points are required.")
    return plan


def method_specs(plan):
    return [{"method": "random", "startup": None}] + [
        {"method": f"tpe_startup_{n}", "startup": n} for n in sorted(plan["tpe_startup_trials"])]


def make_schedule(plan):
    """Rotate order across seeds rather than always running a method last."""
    methods = method_specs(plan)
    schedule = []
    for index, seed in enumerate(plan["search_seeds"]):
        offset = index % len(methods)
        for spec in methods[offset:] + methods[:offset]:
            schedule.append({"seed": seed, **spec})
    return schedule


def load_development(base):
    """Return only development X/y and saved folds in development coordinates."""
    base = Path(base)
    manifest = json.loads((base / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed":
        raise ValueError("The original experiment must be completed first.")
    cfg = manifest["config"]
    if fingerprint(cfg) != manifest["config_sha256"]:
        raise ValueError("Original config fingerprint is invalid.")
    split = json.loads((base / "split_indices.json").read_text(encoding="utf-8"))
    dev, heldout = split["development"], set(split["test"])
    if len(dev) != len(set(dev)) or set(dev) & heldout:
        raise ValueError("Development and held-out indices must be disjoint.")
    position = {global_id: i for i, global_id in enumerate(dev)}
    folds, validations = [], []
    for fold in split["cv_folds_global"]:
        train, val = set(fold["train"]), set(fold["validation"])
        if train & val or train | val != set(dev) or (train | val) & heldout:
            raise ValueError("CV folds must be confined to development data.")
        folds.append((np.array([position[i] for i in fold["train"]]),
                      np.array([position[i] for i in fold["validation"]])))
        validations.extend(fold["validation"])
    if sorted(validations) != sorted(dev) or len(folds) != cfg["cv_folds"]:
        raise ValueError("Invalid development CV coverage.")
    digits = load_digits()
    # The loader distributes the entire dataset, but ONLY development slices
    # are passed to the objective. No held-out X/y slices or metrics are used.
    x, y = digits.data[dev], digits.target[dev]
    checksum = hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()
    return x, y, folds, cfg, checksum


def score_development(params, x, y, folds, config):
    start = time.perf_counter()
    scores = cross_val_score(make_model(params, config), x, y, cv=folds,
                             scoring="f1_macro", n_jobs=1, error_score="raise")
    return {"cv_f1": float(scores.mean()), "cv_fold_std": float(scores.std(ddof=1)),
            "fold_scores": scores.tolist(), "cv_seconds": time.perf_counter() - start}


def run_robustness(plan, output, base=None):
    base = Path(base or ROOT / "results").resolve()
    output = Path(output).resolve()
    if output == base or output in base.parents or output.is_relative_to(base / "models"):
        raise ValueError("Output must not replace original results or models.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty output folder; existing results are preserved.")
    output.mkdir(parents=True, exist_ok=True)
    x, y, folds, cfg, data_sha = load_development(base)
    frozen = frozen_artifacts(base)
    manifest = {
        "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "development_only", "test_evaluations": 0,
        "plan": plan, "plan_sha256": fingerprint(plan),
        "base_config": cfg, "base_config_sha256": fingerprint(cfg),
        "development_sha256": data_sha, "development_samples": len(y),
        "schedule": make_schedule(plan), "frozen_artifacts": frozen,
        "frozen_hash_policy": "UTF-8 text normalizes CRLF to LF; model binary hashes exact bytes.",
        "python": sys.version, "platform": platform.platform(),
        "cpu": platform.processor(), "logical_cpu_count": os.cpu_count(),
        "versions": {p: version(p) for p in VERSIONS},
        "timing": "Fresh serial CV-only studies, n_jobs=1, no refit/profiling. Elapsed time recorded on completed-trial callback, including sampler and earlier logging overhead.",
    }
    write_json(output / "run_manifest.json", manifest)
    all_rows = []
    run_start = time.perf_counter()
    baseline = json.loads((base / "baseline.json").read_text(encoding="utf-8"))
    baseline_cv = score_development(baseline["params"], x, y, folds, cfg)
    if not np.allclose(baseline_cv["fold_scores"], baseline["fold_scores"], atol=1e-12, rtol=0):
        raise AssertionError("Development baseline no longer reproduces the frozen experiment.")
    write_json(output / "baseline_cv.json", baseline_cv)
    try:
        for order, spec in enumerate(manifest["schedule"]):
            method, seed, startup = spec["method"], spec["seed"], spec["startup"]
            sampler = (optuna.samplers.RandomSampler(seed=seed) if startup is None else
                       optuna.samplers.TPESampler(seed=seed, n_startup_trials=startup))
            started = time.perf_counter()
            study = optuna.create_study(direction="maximize", sampler=sampler,
                                        study_name=f"{method}_{seed}")

            def objective(trial):
                params = sample_params(trial, cfg)
                if not check_feasible(params, cfg):
                    raise ValueError("An infeasible configuration was sampled.")
                metrics = score_development(params, x, y, folds, cfg)
                for key, value in metrics.items():
                    trial.set_user_attr(key, value)
                return metrics["cv_f1"]

            def completed(study, trial):
                row = {"method": method, "seed": seed, "startup_trials": startup,
                       "schedule_order": order, "trial": trial.number, "params": trial.params,
                       "elapsed_seconds": time.perf_counter() - started,
                       "trial_wall_seconds": trial.duration.total_seconds(),
                       "started_at_local": trial.datetime_start.astimezone().isoformat(),
                       **trial.user_attrs}
                all_rows.append(row)
                write_json(output / "trials.json", all_rows)
                if (trial.number + 1) % 5 == 0 or trial.number + 1 == plan["trials_per_method"]:
                    print(f"{method} seed={seed}: {trial.number+1}/{plan['trials_per_method']}, best={study.best_value:.5f}", flush=True)

            study.optimize(objective, n_trials=plan["trials_per_method"], callbacks=[completed])
            study.trials_dataframe().to_csv(output / f"optuna_{method}_{seed}.csv", index=False)
            verify_frozen(base, frozen)
        flat = [{k: v for k, v in r.items() if k not in {"params", "fold_scores"}}
                | r["params"] | {"fold_scores": json.dumps(r["fold_scores"])} for r in all_rows]
        pd.DataFrame(flat).to_csv(output / "trials.csv", index=False)
        manifest.update(status="completed", completed_trials=len(all_rows),
                        runtime_seconds=time.perf_counter() - run_start,
                        ended_at_utc=datetime.now(timezone.utc).isoformat())
        verify_frozen(base, frozen)
        write_json(output / "run_manifest.json", manifest)
    except BaseException as error:
        manifest.update(status="failed", completed_trials=len(all_rows),
                        failure_type=type(error).__name__)
        write_json(output / "run_manifest.json", manifest)
        raise
    return output
