"""Configuration, reproducible splitting, and model construction.

The search only receives development data. The held-out test set is accessed
after candidates and the deployment configuration have been locked.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
FEATURES = {"sqrt": "sqrt", "log2": "log2", "half": 0.5, "all": 1.0}
DEFAULT_PARAMS = {"n_estimators": 100, "max_depth": None,
                  "min_samples_leaf": 1, "max_features": "sqrt"}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    # Cloud sync and antivirus can briefly lock a destination on Windows.
    # Preserve the previous complete log and retry the atomic replacement.
    for attempt in range(8):
        try:
            temporary.write_text(payload, encoding="utf-8")
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.025 * 2**attempt)


def load_config(path=None):
    config = json.loads(Path(path or ROOT / "configs/experiment.json").read_text(encoding="utf-8"))
    if config["dataset"] != "sklearn_digits":
        raise ValueError("Only sklearn_digits is supported.")
    for key in ["cv_folds", "trials_per_method", "latency_repeats", "tpe_startup_trials"]:
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if config["cv_folds"] < 2 or not 0 < config["test_size"] < 1:
        raise ValueError("Invalid CV folds or test fraction.")
    if not 0 <= config["epsilon"] < 1 or config["n_jobs"] != 1:
        raise ValueError("epsilon must be in [0,1); n_jobs must be 1 for controlled timing.")
    seeds = config["search_seeds"]
    if not seeds or len(seeds) != len(set(seeds)) or any(not isinstance(s, int) for s in seeds):
        raise ValueError("search_seeds must contain distinct integers.")
    space = config["search_space"]
    est = space["n_estimators"]
    leaf = space["min_samples_leaf"]
    if not 1 <= est["low"] <= est["high"] or est["step"] < 1 or (est["high"]-est["low"]) % est["step"]:
        raise ValueError("Invalid n_estimators range.")
    if not 1 <= leaf["low"] <= leaf["high"]:
        raise ValueError("Invalid min_samples_leaf range.")
    if not space["max_depth"] or any(d is not None and (not isinstance(d, int) or d < 1) for d in space["max_depth"]):
        raise ValueError("Invalid max_depth choices.")
    if not space["max_features"] or any(f not in FEATURES for f in space["max_features"]):
        raise ValueError("Invalid max_features choices.")
    return config


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def make_data(config):
    digits = load_digits()
    ids = np.arange(len(digits.target))
    development, test = train_test_split(ids, test_size=config["test_size"],
                                        stratify=digits.target, random_state=config["split_seed"])
    x, y = digits.data[development], digits.target[development]
    cv = StratifiedKFold(n_splits=config["cv_folds"], shuffle=True,
                        random_state=config["cv_seed"])
    folds = list(cv.split(x, y))
    return digits, development, test, folds


def make_model(params, config):
    values = dict(params)
    values["max_features"] = FEATURES[values["max_features"]]
    return RandomForestClassifier(**values, random_state=config["model_seed"],
                                  n_jobs=config["n_jobs"])


def check_feasible(params, config):
    space = config["search_space"]
    est = space["n_estimators"]
    leaf = space["min_samples_leaf"]
    return (est["low"] <= params["n_estimators"] <= est["high"]
            and (params["n_estimators"] - est["low"]) % est["step"] == 0
            and params["max_depth"] in space["max_depth"]
            and leaf["low"] <= params["min_samples_leaf"] <= leaf["high"]
            and params["max_features"] in space["max_features"])
