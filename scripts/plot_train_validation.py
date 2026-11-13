"""Draw exactly two curves from saved, audited development-fold scores."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "results/curve_audit/all_tree_metrics_per_fold.csv"
    raw_bytes = source.read_bytes()
    raw = pd.read_csv(source)
    assert set(raw.split) == {"train", "validation"}
    assert not raw.duplicated(["split", "fold", "trees"]).any()
    assert len(raw) == 600
    for _, group in raw.groupby(["split", "fold"]):
        assert sorted(group.trees.tolist()) == list(range(1, 101))
    summary = raw.groupby(["split", "trees"], as_index=False)[["accuracy", "log_loss"]].mean()
    old = pd.read_csv(ROOT / "results/tree_diagnostics/metrics_summary.csv")
    overlap = summary.merge(old, on=["split", "trees"])
    for metric in ["accuracy", "log_loss"]:
        np.testing.assert_allclose(overlap[metric], overlap[f"{metric}_mean"], rtol=0, atol=1e-12)
    destination = ROOT / "figures/train_validation_two_lines"
    output = ROOT / "results/train_validation_two_lines"
    destination.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "mean_metrics.csv", index=False)
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 180})
    for metric, ylabel in [("accuracy", "Accuracy"), ("log_loss", "Cross-entropy (log loss)")]:
        fig, ax = plt.subplots(figsize=(9, 5.4))
        for split, color, label in [("train", "#21614E", "Train"),
                                    ("validation", "#246387", "Validation")]:
            values = summary[summary.split == split].sort_values("trees")
            ax.plot(values.trees, values[metric], color=color, lw=2,
                    marker="o", markersize=3, markevery=5, label=label)
        assert len(ax.lines) == 2
        ax.set(xlabel="Number of trees", ylabel=ylabel, xlim=(1, 100),
               title=f"Random Forest: train and validation {metric.replace('_', ' ')}")
        if metric == "accuracy":
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_ylim(.65, 1.015)
        else:
            ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
        ax.legend(loc="best")
        fig.text(.5, .025, "Mean of 3 fixed CV folds | All 100 tree counts | No smoothing",
                 ha="center", fontsize=10)
        fig.tight_layout(rect=(0, .055, 1, 1))
        fig.savefig(destination / f"{metric}.png")
        plt.close(fig)
    assert source.read_bytes() == raw_bytes
    checks = {"source": source.relative_to(ROOT).as_posix(),
              "source_sha256_lf": hashlib.sha256(raw_bytes.replace(b"\r\n", b"\n")).hexdigest(),
              "source_rows": len(raw), "mean_rows": len(summary),
              "lines_per_figure": 2, "tree_counts": "all integers 1 through 100",
              "averaging": "arithmetic mean over the three original CV folds",
              "smoothing": False, "training_or_test_evaluations": 0,
              "saved_checkpoint_means_match": True, "source_unchanged": True}
    (output / "checks.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print("PASS: exactly two lines per image; 600 source rows and original checkpoint means verified.")


if __name__ == "__main__":
    main()
