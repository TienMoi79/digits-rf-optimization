"""Budget curves and paired descriptive analysis using development logs only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import ROOT, write_json
from .robustness import method_specs

COLORS = ["#176B87", "#00865A", "#D65A31"]


def budget_records(rows, checkpoints):
    records = []
    for method, seed in sorted({(r["method"], r["seed"]) for r in rows}):
        run = sorted([r for r in rows if r["method"] == method and r["seed"] == seed],
                     key=lambda r: r["trial"])
        if [r["trial"] for r in run] != list(range(len(run))):
            raise ValueError("Trials must be contiguous from zero.")
        for budget in checkpoints:
            prefix = run[:budget]
            if len(prefix) != budget:
                raise ValueError("A requested checkpoint exceeds the available trials.")
            best = max(prefix, key=lambda r: r["cv_f1"])
            records.append({"method": method, "seed": seed, "budget": budget,
                            "best_cv_f1": best["cv_f1"], "best_trial": best["trial"],
                            "elapsed_seconds": prefix[-1]["elapsed_seconds"],
                            "cv_seconds": sum(r["cv_seconds"] for r in prefix),
                            "unique_configs": len({json.dumps(r["params"], sort_keys=True) for r in prefix})})
    return pd.DataFrame(records)


def aggregate_budgets(records):
    return records.groupby(["method", "budget"], as_index=False).agg(
        best_cv_f1_mean=("best_cv_f1", "mean"), best_cv_f1_std=("best_cv_f1", "std"),
        elapsed_seconds_mean=("elapsed_seconds", "mean"), elapsed_seconds_std=("elapsed_seconds", "std"),
        elapsed_seconds_median=("elapsed_seconds", "median"),
        cv_seconds_mean=("cv_seconds", "mean"), unique_configs_mean=("unique_configs", "mean"),
        seeds=("seed", "count"))


def common_time_curves(rows, points):
    runs = {}
    for r in rows:
        runs.setdefault((r["method"], r["seed"]), []).append(r)
    runs = {k: sorted(run, key=lambda r: r["trial"]) for k, run in runs.items()}
    start = max(run[0]["elapsed_seconds"] for run in runs.values())
    stop = min(run[-1]["elapsed_seconds"] for run in runs.values())
    if stop <= start:
        raise ValueError("No shared time window with a completed trial in every run.")
    grid = np.linspace(start, stop, points)
    records = []
    for (method, seed), run in sorted(runs.items()):
        elapsed = np.array([r["elapsed_seconds"] for r in run])
        if (np.diff(elapsed) <= 0).any():
            raise ValueError("Elapsed completion times must increase strictly.")
        best = np.maximum.accumulate([r["cv_f1"] for r in run])
        indices = np.searchsorted(elapsed, grid, side="right") - 1
        for time_value, index in zip(grid, indices):
            if index < 0:
                raise ValueError("Cannot backfill a score before a trial completes.")
            records.append({"method": method, "seed": seed,
                            "elapsed_seconds": float(time_value),
                            "best_cv_f1": float(best[index]), "completed_trials": int(index + 1)})
    return pd.DataFrame(records)


def paired_ablation(records, startups, budget):
    low, high = sorted(startups)
    names = [f"tpe_startup_{low}", f"tpe_startup_{high}"]
    pairs = records[(records.budget == budget) & records.method.isin(names)]
    pivot = pairs.pivot(index="seed", columns="method", values="best_cv_f1")
    if set(pivot.columns) != set(names) or pivot.isna().any().any():
        raise ValueError("Ablation needs complete matched-seed pairs.")
    pivot = pivot.reset_index()
    pivot["delta_low_minus_high"] = pivot[names[0]] - pivot[names[1]]
    pivot["winner"] = np.where(pivot.delta_low_minus_high > 1e-12, names[0],
                              np.where(pivot.delta_low_minus_high < -1e-12, names[1], "tie"))
    return pivot


def analyze_robustness(output, figures=None):
    output = Path(output)
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed":
        raise ValueError("Analyze a completed study only.")
    plan = manifest["plan"]
    rows = json.loads((output / "trials.json").read_text(encoding="utf-8"))
    figures = Path(figures or (ROOT / "figures/robustness" if output.resolve() == (ROOT / "results/robustness").resolve() else output / "figures"))
    figures.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((output / "baseline_cv.json").read_text(encoding="utf-8"))["cv_f1"]
    per_seed = budget_records(rows, plan["checkpoints"])
    summary = aggregate_budgets(per_seed)
    per_seed.to_csv(output / "budget_per_seed.csv", index=False)
    summary.to_csv(output / "budget_summary.csv", index=False)
    pairs = paired_ablation(per_seed, plan["tpe_startup_trials"], plan["trials_per_method"])
    pairs.to_csv(output / "startup_ablation_pairs.csv", index=False)
    low, high = sorted(plan["tpe_startup_trials"])
    delta = pairs.delta_low_minus_high
    ablation = {"comparison": f"TPE startup {low} minus startup {high}",
                "paired_seeds": len(pairs), "mean_delta_cv_f1": float(delta.mean()),
                "std_delta_cv_f1": float(delta.std(ddof=1)) if len(pairs) > 1 else 0.,
                "low_startup_wins": int((delta > 1e-12).sum()),
                "high_startup_wins": int((delta < -1e-12).sum()),
                "ties": int((delta.abs() <= 1e-12).sum()),
                "interpretation": "Paired descriptive differences over search seeds, not a significance test or dataset-level confidence interval."}
    write_json(output / "startup_ablation_summary.json", ablation)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 140})
    methods = [m["method"] for m in method_specs(plan)]
    fig, ax = plt.subplots(figsize=(8.8, 5))
    for method, color in zip(methods, COLORS):
        curves = [np.maximum.accumulate([r["cv_f1"] for r in sorted(rows, key=lambda r: r["trial"])
                                        if r["method"] == method and r["seed"] == seed])
                  for seed in plan["search_seeds"]]
        curves = np.stack(curves)
        mean = curves.mean(axis=0)
        std = curves.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean)
        axis = np.arange(1, len(mean)+1)
        ax.plot(axis, mean, lw=2, color=color, label=method)
        ax.fill_between(axis, mean-std, mean+std, color=color, alpha=.12)
    ax.axhline(baseline, color="gray", ls="--", label="Frozen default RF CV score")
    ax.set(xlabel="Completed trials per run", ylabel="Best development CV Macro-F1",
           title=f"Fresh comparison: mean and SD over {len(plan['search_seeds'])} search seeds")
    ax.grid(alpha=.18)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(figures / "search_budget.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.8, 5))
    try:
        timed = common_time_curves(rows, plan["time_grid_points"])
    except ValueError as error:
        if "No shared time window" not in str(error):
            raise
        timed = pd.DataFrame(columns=["method", "seed", "elapsed_seconds", "best_cv_f1", "completed_trials"])
        window = {"available": False, "reason": str(error)}
        ax.text(.5, .5, "No common time window for these short runs", transform=ax.transAxes, ha="center")
    else:
        window = {"available": True, "start_seconds": float(timed.elapsed_seconds.min()),
                  "stop_seconds": float(timed.elapsed_seconds.max()), "grid_points": plan["time_grid_points"],
                  "policy": "Common overlap only; no interpolation, backfill, or extrapolation of trial scores."}
        for method, color in zip(methods, COLORS):
            values = timed[timed.method == method].groupby("elapsed_seconds").best_cv_f1.agg(["mean", "std"]).fillna(0)
            axis = values.index.to_numpy()
            ax.step(axis, values["mean"], where="post", color=color, lw=2, label=method)
            ax.fill_between(axis, values["mean"]-values["std"], values["mean"]+values["std"], step="post", color=color, alpha=.12)
        ax.axhline(baseline, color="gray", ls="--", label="Frozen default RF CV score")
        ax.legend(fontsize=9)
    timed.to_csv(output / "time_budget_per_seed.csv", index=False)
    write_json(output / "time_comparison.json", window)
    ax.set(xlabel="Elapsed search seconds in the shared window", ylabel="Best development CV Macro-F1",
           title="Equal-time comparison of fresh CV-only runs")
    ax.grid(alpha=.18)
    fig.tight_layout()
    fig.savefig(figures / "time_budget.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    positions = np.arange(len(pairs))
    ax.bar(positions, delta, color=np.where(delta >= 0, COLORS[1], COLORS[2]))
    ax.axhline(0, color="gray", lw=1)
    ax.set(xticks=positions, xticklabels=pairs.seed.astype(str), xlabel="Matched search seed",
           ylabel=f"CV F1 gain: startup {low} minus {high}", title="TPE startup ablation at the same trial budget")
    ax.grid(axis="y", alpha=.18)
    fig.tight_layout()
    fig.savefig(figures / "startup_ablation.png")
    plt.close(fig)
    lines = ["| Method | Trials/run | Seeds | Best CV F1 mean ± SD | Elapsed mean ± SD (s) | Elapsed median (s) |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in summary.itertuples():
        lines.append(f"| {r.method} | {r.budget} | {r.seeds} | {r.best_cv_f1_mean:.5f} ± {r.best_cv_f1_std:.5f} | {r.elapsed_seconds_mean:.2f} ± {r.elapsed_seconds_std:.2f} | {r.elapsed_seconds_median:.2f} |")
    (output / "metrics_table.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("Development-only budget and ablation analysis saved.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/robustness")
    analyze_robustness(parser.parse_args().output)
