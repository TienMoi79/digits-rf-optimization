"""Check new development studies and the preservation of published artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ROOT, check_feasible, fingerprint, write_json
from .robustness import load_development, method_specs, verify_frozen


def validate_robustness(output=None, base=None):
    output, base = Path(output or ROOT / "results/robustness"), Path(base or ROOT / "results")
    read = lambda name: json.loads((output / name).read_text(encoding="utf-8"))
    manifest, rows = read("run_manifest.json"), read("trials.json")
    plan, cfg = manifest["plan"], manifest["base_config"]
    checks = []

    def require(condition, message):
        if not condition:
            raise AssertionError(message)
        checks.append(message)

    require(manifest["status"] == "completed", "Development study is completed")
    require(manifest["scope"] == "development_only" and manifest["test_evaluations"] == 0,
            "Additional study has zero held-out evaluations")
    require(fingerprint(plan) == manifest["plan_sha256"], "Study plan fingerprint matches")
    require(fingerprint(cfg) == manifest["base_config_sha256"], "Original modeling protocol is unchanged")
    verify_frozen(base, manifest["frozen_artifacts"])
    checks.append("Original published logs, test results, selection and model hashes are preserved")
    x, y, folds, base_config, checksum = load_development(base)
    require(base_config == cfg and checksum == manifest["development_sha256"]
            and len(y) == manifest["development_samples"], "Development data and fixed CV folds match the original protocol")
    baseline = read("baseline_cv.json")
    original_baseline = json.loads((base / "baseline.json").read_text(encoding="utf-8"))
    require(np.allclose(baseline["fold_scores"], original_baseline["fold_scores"], rtol=0, atol=1e-12),
            "Fresh baseline reproduces the original development fold scores")
    methods = [r["method"] for r in method_specs(plan)]
    expected_runs = {(m, seed) for m in methods for seed in plan["search_seeds"]}
    require({(r["method"],r["seed"]) for r in rows} == expected_runs, "Every method uses exactly the same search seeds")
    require(len(rows) == len(expected_runs)*plan["trials_per_method"] == manifest["completed_trials"], "Equal complete trial budgets")
    runs = {}
    for key in expected_runs:
        run = sorted([r for r in rows if (r["method"], r["seed"]) == key], key=lambda r:r["trial"])
        require([r["trial"] for r in run] == list(range(plan["trials_per_method"])), "Each study contains every trial exactly once")
        require(all(a["elapsed_seconds"] < b["elapsed_seconds"] for a,b in zip(run,run[1:])), "Completion times are strictly increasing within a study")
        require(run[0]["elapsed_seconds"] > 0, "Completion times are positive")
        runs[key] = run
    for r in rows:
        require(check_feasible(r["params"],cfg), "Sampled configuration is feasible")
        require(len(r["fold_scores"]) == len(folds) and np.isfinite(r["fold_scores"]).all()
                and all(0 <= s <= 1 for s in r["fold_scores"]), "Every saved fold score is finite and in range")
        require(np.isclose(np.mean(r["fold_scores"]),r["cv_f1"],rtol=0,atol=1e-12)
                and np.isclose(np.std(r["fold_scores"],ddof=1),r["cv_fold_std"],rtol=0,atol=1e-12),
                "CV mean and fold SD recompute")
        require(not any("test" in k or "prediction" in k for k in r), "Trial records contain development metrics only")
        spec = next(s for s in manifest["schedule"] if (s["method"],s["seed"]) == (r["method"],r["seed"]))
        require(r["startup_trials"] == spec["startup"], "TPE startup treatment matches the declared schedule")
    csv = pd.read_csv(output / "trials.csv")
    require(len(csv) == len(rows) and np.allclose(csv.cv_f1,[r["cv_f1"] for r in rows],rtol=0,atol=1e-12), "JSON and CSV trial logs agree")
    budgets = pd.read_csv(output / "budget_per_seed.csv")
    require(len(budgets) == len(expected_runs)*len(plan["checkpoints"]), "All trial-budget checkpoints are present")
    for r in budgets.itertuples():
        prefix = runs[(r.method,r.seed)][:r.budget]
        require(np.isclose(r.best_cv_f1,max(t["cv_f1"] for t in prefix),rtol=0,atol=1e-12)
                and np.isclose(r.elapsed_seconds,prefix[-1]["elapsed_seconds"],rtol=0,atol=1e-9),
                "Budget result uses only trials completed within that budget")
    summary = pd.read_csv(output / "budget_summary.csv")
    require(len(summary) == len(methods)*len(plan["checkpoints"]), "Complete method and budget summaries")
    for r in summary.itertuples():
        subset = budgets[(budgets.method==r.method)&(budgets.budget==r.budget)]
        scores = subset.best_cv_f1
        require(r.seeds == len(plan["search_seeds"]) and np.isclose(r.best_cv_f1_mean,scores.mean(),rtol=0,atol=1e-12)
                and np.isclose(r.best_cv_f1_std,scores.std(ddof=1),rtol=0,atol=1e-12), "Mean and sample SD recompute across search seeds")
        require(np.isclose(r.elapsed_seconds_mean,subset.elapsed_seconds.mean(),rtol=0,atol=1e-9)
                and np.isclose(r.elapsed_seconds_std,subset.elapsed_seconds.std(ddof=1),rtol=0,atol=1e-9)
                and np.isclose(r.elapsed_seconds_median,subset.elapsed_seconds.median(),rtol=0,atol=1e-9),
                "Elapsed mean, SD and median recompute across search seeds")
    pairs = pd.read_csv(output / "startup_ablation_pairs.csv")
    low, high = sorted(plan["tpe_startup_trials"])
    require(set(pairs.seed) == set(plan["search_seeds"]) and len(pairs) == len(plan["search_seeds"]), "Ablation uses complete matched-seed pairs")
    for r in pairs.to_dict(orient="records"):
        expected = max(t["cv_f1"] for t in runs[(f"tpe_startup_{low}",r["seed"])]) - max(t["cv_f1"] for t in runs[(f"tpe_startup_{high}",r["seed"])])
        require(np.isclose(r["delta_low_minus_high"],expected,rtol=0,atol=1e-12), "Paired startup difference recomputes from raw trials")
    ablation = read("startup_ablation_summary.json")
    require(np.isclose(ablation["mean_delta_cv_f1"],pairs.delta_low_minus_high.mean(),rtol=0,atol=1e-12), "Ablation mean difference recomputes")
    window = read("time_comparison.json")
    timed = pd.read_csv(output / "time_budget_per_seed.csv")
    if window["available"]:
        require(len(timed) == len(expected_runs)*plan["time_grid_points"], "Every study contributes to every common time point")
        for r in timed.itertuples():
            run = runs[(r.method,r.seed)]
            # CSV float conversion can move an endpoint by a few ulps.
            completed = [t for t in run if t["elapsed_seconds"] <= r.elapsed_seconds + 1e-9]
            require(bool(completed) and r.elapsed_seconds <= run[-1]["elapsed_seconds"]+1e-9
                    and np.isclose(r.best_cv_f1,max(t["cv_f1"] for t in completed),rtol=0,atol=1e-12)
                    and r.completed_trials == len(completed), "Equal-time score uses completed trials with no extrapolation")
    else:
        require(timed.empty, "No equal-time scores are invented when the common window is empty")
    verify_frozen(base,manifest["frozen_artifacts"])
    write_json(output / "validation.json", {"passed":True,"check_count":len(checks),"checks":checks})
    print(f"PASS: {len(checks)} development-study checks; published results preserved.",flush=True)
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"results/robustness")
    parser.add_argument("--if-exists",action="store_true")
    args = parser.parse_args()
    if args.if_exists and not (args.output/"run_manifest.json").exists():
        print("No completed robustness study committed yet; smoke pipeline tests still apply.")
    else:
        validate_robustness(args.output)
