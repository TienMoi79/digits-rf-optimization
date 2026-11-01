"""Replay published HPO or run a small development-only optimization demo."""
from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import optuna

from .common import ROOT, DEFAULT_PARAMS, check_feasible, fingerprint, write_json
from .experiment import choose_candidates, evaluate_cv, sample_params
from .robustness import canonical_sha256, load_development

BASE = ROOT / "results"
LIVE_ROOT = BASE / "week9_demo"


def protected_hashes():
    paths = [p for folder in [BASE, ROOT / "figures", ROOT / "configs"]
             for p in folder.rglob("*") if p.is_file() and not p.is_relative_to(LIVE_ROOT)]
    return {p.relative_to(ROOT).as_posix(): canonical_sha256(p) for p in sorted(paths)}


def summarize(rows, epsilon):
    """Select within the rows provided; replay never supplies future trials."""
    chosen, best = choose_candidates(rows, epsilon)
    selected = chosen["selected"]
    return {"best_cv_f1": best, "epsilon": epsilon, "threshold": best - epsilon,
            "selected": selected,
            "eligible_rows": sum(r["feasible"] and r["cv_f1"] >= best - epsilon for r in rows),
            "representatives": {name: row for name, row in chosen.items() if name != "selected"}}


def replay(seed=11, checkpoint=30):
    """Read only original CV logs. No model/test features or final test table."""
    manifest = json.loads((BASE / "run_manifest.json").read_text(encoding="utf-8"))
    cfg = manifest["config"]
    if seed not in cfg["search_seeds"] or not 1 <= checkpoint <= cfg["trials_per_method"]:
        raise ValueError("Use an original search seed and a checkpoint from 1 to 30.")
    rows = json.loads((BASE / "trials.json").read_text(encoding="utf-8"))
    prefix = [r for r in rows if r["method"] == "default" or
              (r["seed"] == seed and r["method"] in {"random", "tpe"} and r["trial"] < checkpoint)]
    for method in ["random", "tpe"]:
        ids = sorted(r["trial"] for r in prefix if r["method"] == method)
        if ids != list(range(checkpoint)):
            raise ValueError(f"Missing/duplicate replay trial: {method}")
    return prefix, {"mode": "replay", "scope": "development_only", "seed": seed,
                    "trials_per_method": checkpoint, "startup_trials": cfg["tpe_startup_trials"],
                    "selection_scope": "baseline plus the selected seed's prefix, not the full published pool",
                    "test_evaluations": 0, **summarize(prefix, cfg["epsilon"])}


def print_walkthrough(rows, summary):
    print(f"\nMODE: {summary['mode']} | DEVELOPMENT ONLY | test evaluations: 0")
    print("Decision variables: trees, depth, min_samples_leaf, max_features (560 configurations)")
    print("Objective: maximize mean 3-fold CV Macro-F1; epsilon decision: minimize model bytes")
    print(f"Seed: {summary['seed']} | equal budget: {summary['trials_per_method']} trials/method")
    guided = max(0, summary['trials_per_method'] - summary['startup_trials'])
    print(f"TPE startup: {summary['startup_trials']} completed trials; guided trials in this view: {guided}")
    print("No test score participates in this demo or selection.\n")
    for method in ["random", "tpe"]:
        best = float("-inf")
        print(f"{method.upper()}  trial  phase     CV F1     best so far   model MB")
        for row in sorted((r for r in rows if r["method"] == method), key=lambda r: r["trial"]):
            best = max(best, row["cv_f1"])
            phase = "random" if method == "random" or row["trial"] < summary["startup_trials"] else "TPE"
            print(f"          {row['trial'] + 1:3d}  {phase:7s}  {row['cv_f1']:.6f}  {best:.6f}      {row['model_bytes']/1e6:.4f}")
    print(f"\nBest observed CV: {summary['best_cv_f1']:.8f}")
    print(f"Acceptance threshold = best - {summary['epsilon']:.3f} = {summary['threshold']:.8f}")
    selected = summary["selected"]
    print(f"Selected from this demo's pool: {selected['method']}, CV={selected['cv_f1']:.8f}, "
          f"size={selected['model_bytes']/1e6:.6f} MB")
    print("Parameters:", json.dumps(selected["params"]))
    print("This is a demonstration decision. The published selected model is unchanged.")


def plot_history(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=150)
    for method, color in [("random", "#176B87"), ("tpe", "#D65A31")]:
        run = sorted((r for r in rows if r["method"] == method), key=lambda r: r["trial"])
        steps = range(1, len(run) + 1)
        scores = [r["cv_f1"] for r in run]
        ax.scatter(steps, scores, s=20, color=color, alpha=.45, label=f"{method}: individual trial")
        ax.step(steps, np.maximum.accumulate(scores), where="post", color=color, label=f"{method}: best so far")
    baseline = next(r for r in rows if r["method"] == "default")
    ax.axhline(baseline["cv_f1"], ls="--", color="#555555", label="Default RF")
    ax.set(xlabel="Completed trials per method", ylabel="Development CV Macro-F1",
           title="Small live demo on one search seed; not the main experiment")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "search_history.png")
    plt.close(fig)


def run_live(output, trials=12, startup=5, seed=11):
    if type(trials) is not int or type(startup) is not int or not 1 <= startup < trials <= 30:
        raise ValueError("Require 1 <= startup < trials <= 30, so guided TPE is actually exercised.")
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a nonnegative integer.")
    output = Path(output).resolve()
    if output == LIVE_ROOT.resolve() or not output.is_relative_to(LIVE_ROOT.resolve()):
        raise ValueError("Live output must be a NEW subfolder under results/week9_demo/.")
    before = protected_hashes()
    x, y, folds, cfg, data_sha = load_development(BASE)
    cfg = copy.deepcopy(cfg)
    cfg["latency_repeats"] = 3
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    now = datetime.now(timezone.utc).isoformat()
    manifest = {"status": "running", "started_at_utc": now, "scope": "development_only",
                "test_evaluations": 0, "purpose": "small teaching demo, not the main experiment",
                "seed": seed, "trials_per_method": trials, "startup_trials": startup,
                "development_sha256": data_sha, "development_samples": len(y),
                "config": cfg, "config_sha256": fingerprint(cfg),
                "python": sys.version, "platform": platform.platform(),
                "versions": {p: version(p) for p in ["numpy", "scikit-learn", "optuna", "joblib"]},
                "sampler_settings": {"n_ei_candidates": 24, "multivariate": True,
                                     "group": False, "constant_liar": True},
                "timing_caveat": "serial Random then TPE, one seed, 3 profiling repeats; not a speed benchmark",
                "protected_artifacts_sha256": before}
    write_json(output / "run_manifest.json", manifest)
    rows = []
    start = time.perf_counter()
    try:
        rows.append({"method": "default", "seed": cfg["model_seed"], "trial": -1,
                     "params": dict(DEFAULT_PARAMS), **evaluate_cv(DEFAULT_PARAMS, x, y, folds, cfg)})
        write_json(output / "trials.json", rows)
        for method in ["random", "tpe"]:
            sampler = (optuna.samplers.RandomSampler(seed=seed) if method == "random" else
                       optuna.samplers.TPESampler(seed=seed, n_startup_trials=startup,
                                                 n_ei_candidates=24, multivariate=True,
                                                 group=False, constant_liar=True))
            study = optuna.create_study(direction="maximize", sampler=sampler)

            def objective(trial):
                params = sample_params(trial, cfg)
                if not check_feasible(params, cfg):
                    raise ValueError("Infeasible demo parameters.")
                metrics = evaluate_cv(params, x, y, folds, cfg)
                if not metrics["feasible"]:
                    raise ValueError("Infeasible recorded demo configuration.")
                rows.append({"method": method, "seed": seed, "trial": trial.number,
                             "phase": "random" if method == "random" or trial.number < startup else "TPE",
                             "params": params, **metrics})
                write_json(output / "trials.json", rows)
                print(f"{method} {trial.number+1}/{trials}: CV={metrics['cv_f1']:.6f}", flush=True)
                return metrics["cv_f1"]

            study.optimize(objective, n_trials=trials)
        summary = {"mode": "live", "scope": "development_only", "seed": seed,
                   "trials_per_method": trials, "startup_trials": startup, "test_evaluations": 0,
                   **summarize(rows, cfg["epsilon"])}
        write_json(output / "demo_selection.json", summary)
        plot_history(rows, output)
        if before != protected_hashes():
            raise AssertionError("A protected published artifact changed during the demo.")
        manifest.update(status="completed", ended_at_utc=datetime.now(timezone.utc).isoformat(),
                        runtime_seconds=time.perf_counter() - start, completed_search_trials=len(rows) - 1,
                        original_artifacts_preserved=True)
        write_json(output / "run_manifest.json", manifest)
        return rows, summary
    except BaseException as error:
        manifest.update(status="failed", error_type=type(error).__name__,
                        completed_search_trials=len(rows) - 1)
        write_json(output / "run_manifest.json", manifest)
        raise


def validate_live(output):
    """Check persisted demo evidence without refitting or held-out prediction."""
    output = Path(output)
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    rows = json.loads((output / "trials.json").read_text(encoding="utf-8"))
    saved = json.loads((output / "demo_selection.json").read_text(encoding="utf-8"))
    cfg = manifest["config"]
    if (manifest["status"] != "completed" or manifest["scope"] != "development_only"
            or manifest["test_evaluations"] != 0 or fingerprint(cfg) != manifest["config_sha256"]):
        raise ValueError("Invalid completed development-only demo manifest.")
    budget, startup = manifest["trials_per_method"], manifest["startup_trials"]
    if not 1 <= startup < budget <= 30:
        raise ValueError("Invalid demo budget/startup.")
    if len(rows) != 1 + 2 * budget or manifest["completed_search_trials"] != 2 * budget:
        raise ValueError("Demo has an incomplete or unequal budget.")
    baseline = [r for r in rows if r["method"] == "default"]
    if len(baseline) != 1 or baseline[0]["params"] != DEFAULT_PARAMS:
        raise ValueError("Demo baseline must retain default learning parameters.")
    for method in ["random", "tpe"]:
        run = [r for r in rows if r["method"] == method]
        if sorted(r["trial"] for r in run) != list(range(budget)):
            raise ValueError("Missing/duplicate demo trials.")
        for row in run:
            expected = "random" if method == "random" or row["trial"] < startup else "TPE"
            if row["seed"] != manifest["seed"] or row["phase"] != expected:
                raise ValueError("Demo seed/phase disagrees with manifest.")
    for row in rows:
        if (not row["feasible"] or not check_feasible(row["params"], cfg)
                or len(row["fold_scores"]) != cfg["cv_folds"]
                or not math.isclose(sum(row["fold_scores"]) / cfg["cv_folds"], row["cv_f1"], rel_tol=0, abs_tol=1e-12)
                or any(not 0 <= score <= 1 for score in row["fold_scores"])
                or row["model_bytes"] <= 0):
            raise ValueError("Invalid demo configuration or CV score.")
    actual = summarize(rows, cfg["epsilon"])
    if any(saved[key] != value for key, value in actual.items()):
        raise ValueError("Demo epsilon decision does not reproduce from logs.")
    for key in ["scope", "seed", "trials_per_method", "startup_trials", "test_evaluations"]:
        if saved[key] != manifest[key]:
            raise ValueError("Demo selection metadata disagrees with manifest.")
    if any(not (ROOT / path).is_file() or canonical_sha256(ROOT / path) != checksum
           for path, checksum in manifest["protected_artifacts_sha256"].items()):
        raise ValueError("Original evidence no longer matches its recorded protected hash.")
    if not manifest["original_artifacts_preserved"] or not (output / "search_history.png").is_file():
        raise ValueError("Demo preservation/plot evidence is missing.")
    print(f"PASS: demo has {2*budget} search trials, feasible configs, reproducible epsilon decision, "
          f"zero held-out evaluations and {len(manifest['protected_artifacts_sha256'])} preserved artifacts.")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    saved = sub.add_parser("replay", help="Read existing CV logs without fitting or writing.")
    saved.add_argument("--seed", type=int, default=11)
    saved.add_argument("--checkpoint", type=int, default=30)
    live = sub.add_parser("live", help="New small CV search; no held-out evaluation.")
    live.add_argument("--output", type=Path, required=True)
    live.add_argument("--trials", type=int, default=12)
    live.add_argument("--startup", type=int, default=5)
    live.add_argument("--seed", type=int, default=11)
    verify = sub.add_parser("validate", help="Validate a saved live demo without retraining.")
    verify.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.mode == "validate":
            validate_live(args.output)
            return
        rows, summary = (replay(args.seed, args.checkpoint) if args.mode == "replay" else
                         run_live(args.output, args.trials, args.startup, args.seed))
        print_walkthrough(rows, summary)
    except (ValueError, FileExistsError) as error:
        parser.exit(2, f"Demo error: {error}\n")


if __name__ == "__main__":
    main()
