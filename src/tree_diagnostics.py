"""Accuracy and log-loss curves for prefixes of the selected Random Forest.

Fit the fixed selected configuration in the original development folds, then
average the first k trees. This is not a neural-network epoch/loss history.
The held-out test set and published deployment model remain untouched.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss

from .common import ROOT, make_model, write_json
from .robustness import frozen_artifacts, load_development, verify_frozen


def build_curves():
    base = ROOT / "results"
    frozen = frozen_artifacts(base)
    selection = json.loads((base / "selection.json").read_text(encoding="utf-8"))
    params = selection["selected_params"]
    x, y, folds, config, checksum = load_development(base)
    records, last_f1 = [], []
    for fold, (train, validation) in enumerate(folds):
        model = make_model(params, config).fit(x[train], y[train])
        inputs = {"train": train, "validation": validation}
        sums = {name: np.zeros((len(ids),len(model.classes_))) for name,ids in inputs.items()}
        for count, tree in enumerate(model.estimators_, start=1):
            for name, ids in inputs.items():
                sums[name] += tree.predict_proba(x[ids])
            if count != 1 and count % 5 != 0 and count != len(model.estimators_):
                continue
            for name, ids in inputs.items():
                probabilities = sums[name] / count
                predictions = model.classes_[probabilities.argmax(axis=1)]
                records.append({"fold":fold,"trees":count,"split":name,"samples":len(ids),
                                "accuracy":float(accuracy_score(y[ids],predictions)),
                                "log_loss":float(log_loss(y[ids],probabilities,labels=model.classes_)),
                                "macro_f1":float(f1_score(y[ids],predictions,average="macro"))})
        for name, ids in inputs.items():
            np.testing.assert_allclose(sums[name]/len(model.estimators_),model.predict_proba(x[ids]),rtol=0,atol=1e-12)
        last_f1.append(records[-1]["macro_f1"])
    np.testing.assert_allclose(last_f1,selection["candidates"]["selected"]["fold_scores"],rtol=0,atol=1e-12)
    detail = pd.DataFrame(records)
    summary = detail.groupby(["split","trees"],as_index=False).agg(
        accuracy_mean=("accuracy","mean"),accuracy_std=("accuracy","std"),
        log_loss_mean=("log_loss","mean"),log_loss_std=("log_loss","std"))
    output = base / "tree_diagnostics"
    figures = ROOT / "figures/tree_diagnostics"
    output.mkdir(exist_ok=True)
    figures.mkdir(parents=True,exist_ok=True)
    detail.to_csv(output/"metrics_per_fold.csv",index=False)
    summary.to_csv(output/"metrics_summary.csv",index=False)
    plt.rcParams.update({"font.size":12,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150})
    colors={"train":"#176B87","validation":"#D65A31"}
    for metric, title, ylabel in [
        ("accuracy","Random Forest accuracy","Accuracy"),
        ("log_loss","Random Forest log loss","Log loss (lower is better)")]:
        fig, ax = plt.subplots(figsize=(8.5,4.8))
        for name in ["train","validation"]:
            values=summary[summary.split==name]
            axis=values.trees.to_numpy()
            mean=values[f"{metric}_mean"].to_numpy()
            std=values[f"{metric}_std"].to_numpy()
            lower,upper=mean-std,mean+std
            if metric=="accuracy": lower,upper=np.clip(lower,0,1),np.clip(upper,0,1)
            ax.plot(axis,mean,color=colors[name],lw=2,label="Training" if name=="train" else "Validation (3-fold CV)")
            ax.fill_between(axis,lower,upper,color=colors[name],alpha=.12)
        ax.set(xlabel="Number of trees averaged in the forest",ylabel=ylabel,
               title=title+"\nFixed selected configuration on development data")
        if metric=="accuracy":
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_ylim(max(0,float((summary.accuracy_mean-summary.accuracy_std).min())-.025),1.01)
        else:
            ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(figures/f"{metric}.png")
        plt.close(fig)
    verify_frozen(base,frozen)
    final=summary[summary.trees==params["n_estimators"]].set_index("split")
    write_json(output/"run_manifest.json",{
        "scope":"development_only","test_evaluations":0,"params":params,
        "development_sha256":checksum,"folds":len(folds),
        "curve_axis":"Number of trees, not epochs or optimization trials",
        "loss_definition":"Mean negative log likelihood evaluated by sklearn.metrics.log_loss; not the Gini split objective.",
        "band":"One sample SD across the three fixed folds, not a confidence interval.",
        "validation_accuracy_at_full_forest":float(final.loc['validation','accuracy_mean']),
        "validation_log_loss_at_full_forest":float(final.loc['validation','log_loss_mean']),
        "checks":{"prefix_probabilities_match_full_forest":True,"final_cv_f1_matches_original":True,"published_artifacts_preserved":True}})
    print(final[["accuracy_mean","log_loss_mean"]].to_string())
    print("Accuracy and loss curves saved; all consistency checks passed.")


if __name__ == "__main__":
    build_curves()
