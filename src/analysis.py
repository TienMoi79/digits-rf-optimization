"""Generate figures and numerical summaries from persisted measurements."""
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
from .experiment import choose_candidates

COLORS = {"random": "#176B87", "tpe": "#D65A31", "default": "#555555"}


def pareto_mask(scores, sizes):
    """True if no point is at least as good on both axes and better on one."""
    scores, sizes = np.asarray(scores), np.asarray(sizes)
    mask = np.ones(len(scores), dtype=bool)
    for i in range(len(scores)):
        dominates = ((scores >= scores[i]) & (sizes <= sizes[i])
                     & ((scores > scores[i]) | (sizes < sizes[i])))
        mask[i] = not dominates.any()
    return mask


def analyze(output):
    output = Path(output)
    rows = json.loads((output / "trials.json").read_text(encoding="utf-8"))
    cfg = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))["config"]
    df = pd.read_csv(output / "trials.csv")
    search = df[df.method != "default"].copy()
    final = pd.read_csv(output / "final_metrics.csv")
    figs = output / "figures"
    figs.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 140})
    best_runs = []
    for (method, seed), run in search.groupby(["method", "seed"]):
        run = run.sort_values("trial")
        idx = run.cv_f1.idxmax()
        best_runs.append({"method": method, "seed": int(seed),
                          "best_cv_f1": float(run.loc[idx, "cv_f1"]),
                          "search_cv_seconds": float(run.cv_seconds.sum()),
                          "refit_seconds": float(run.refit_seconds.sum()),
                          "trials": len(run)})
    per_run = pd.DataFrame(best_runs)
    per_run.to_csv(output / "best_per_seed.csv", index=False)
    summary = per_run.groupby("method").agg(
        best_cv_f1_mean=("best_cv_f1", "mean"), best_cv_f1_std=("best_cv_f1", "std"),
        cv_seconds_mean=("search_cv_seconds", "mean"), cv_seconds_std=("search_cv_seconds", "std"),
        runs=("seed", "count"))
    summary.to_csv(output / "search_summary.csv")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for method in ["random", "tpe"]:
        curves = [run.sort_values("trial").cv_f1.cummax().to_numpy()
                  for _, run in search[search.method == method].groupby("seed")]
        values = np.stack(curves)
        mean, std = values.mean(axis=0), values.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros(values.shape[1])
        trials = np.arange(1, len(mean) + 1)
        ax.plot(trials, mean, label=method.upper(), color=COLORS[method], lw=2)
        ax.fill_between(trials, mean-std, mean+std, alpha=.14, color=COLORS[method])
    ax.axhline(df[df.method == "default"].cv_f1.iloc[0], color=COLORS["default"], ls="--", label="Default RF")
    ax.set(xlabel="Completed trials per run", ylabel="Best development CV Macro-F1",
           title=f"Search history: mean and SD over {len(cfg['search_seeds'])} search seeds")
    ax.legend(loc="lower right")
    ax.grid(alpha=.18)
    fig.tight_layout()
    fig.savefig(figs / "search_history.png")
    plt.close(fig)
    mask = pareto_mask(df.cv_f1, df.model_bytes)
    df["pareto_non_dominated"] = mask
    df.to_csv(output / "pareto_candidates.csv", index=False)
    selected = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    point = selected["candidates"]["selected"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for method in ["random", "tpe", "default"]:
        subset = df[df.method == method]
        ax.scatter(subset.model_bytes/1e6, subset.cv_f1, s=28 if method != "default" else 90,
                   alpha=.55 if method != "default" else 1, label=method.upper(), color=COLORS[method])
    frontier = df[mask].sort_values("model_bytes").drop_duplicates(["model_bytes", "cv_f1"])
    ax.plot(frontier.model_bytes/1e6, frontier.cv_f1, color="#333333", lw=1, label="Observed Pareto frontier")
    ax.scatter([point["model_bytes"]/1e6], [point["cv_f1"]], s=180,
               marker="*", color="#00865A", edgecolor="black", label="Selected by CV epsilon rule", zorder=6)
    ax.axhline(selected["minimum_acceptable_cv_f1"], ls=":", color="#00865A", label="CV acceptance threshold")
    ax.set(xlabel="Uncompressed model size (decimal MB)", ylabel="Development CV Macro-F1",
           title="Quality and model size on development data")
    ax.grid(alpha=.18)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(figs / "quality_size_tradeoff.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    matrix = np.loadtxt(output / "confusion_matrix.csv", delimiter=",", dtype=int)
    image = ax.imshow(matrix, cmap="Blues")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(matrix[i,j]), ha="center", va="center", fontsize=8,
                    color="white" if matrix[i,j] > matrix.max()/2 else "black")
    ax.set(xticks=range(10), yticks=range(10), xlabel="Predicted digit", ylabel="True digit",
           title="Selected model: held-out test confusion matrix")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(figs / "confusion_matrix.png")
    plt.close(fig)
    sensitivity = []
    for epsilon in sorted(set([0., .0025, cfg["epsilon"], .01, .02])):
        candidates, best = choose_candidates(rows, epsilon)
        p = candidates["selected"]
        sensitivity.append({"epsilon": epsilon, "selected_cv_f1": p["cv_f1"],
                            "cv_gap": best-p["cv_f1"], "model_mb": p["model_bytes"]/1e6,
                            "method": p["method"], "seed": p["seed"], "trial": p["trial"]})
    sens = pd.DataFrame(sensitivity)
    sens.to_csv(output / "epsilon_sensitivity.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sens.epsilon, sens.model_mb, "o-", color="#00865A")
    ax.set(xlabel="Allowed development CV Macro-F1 drop (epsilon)", ylabel="Selected model size (decimal MB)",
           title="Sensitivity of the size decision; no test data used")
    ax.grid(alpha=.18)
    fig.tight_layout()
    fig.savefig(figs / "epsilon_sensitivity.png")
    plt.close(fig)
    default = final[final.label == "default"].iloc[0]
    compact = final[final.label == "selected"].iloc[0]
    comparison = {"selected_test_f1": float(compact.test_f1),
                  "selected_test_accuracy": float(compact.test_accuracy),
                  "selected_model_mb": float(compact.model_mb),
                  "size_reduction_vs_default_percent": float((1-compact.model_mb/default.model_mb)*100),
                  "cv_gap_to_best": float(selected["best_cv_f1"]-compact.cv_f1),
                  "search_seed_summary": summary.reset_index().fillna(0).to_dict(orient="records")}
    write_json(output / "key_results.json", comparison)
    # A compact code-generated table can be embedded into the README without typing numbers.
    lines = ["| Model | CV Macro-F1 | Test Macro-F1 | Test accuracy | Size MB |", "|---|---:|---:|---:|---:|"]
    for r in final.itertuples():
        lines.append(f"| {r.label} | {r.cv_f1:.5f} | {r.test_f1:.5f} | {r.test_accuracy:.5f} | {r.model_mb:.3f} |")
    (output / "metrics_table.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("Figures and numerical summaries saved.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    analyze(parser.parse_args().output)
