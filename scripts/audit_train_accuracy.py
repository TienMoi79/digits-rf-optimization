"""Independently reproduce train counts without project model/split helpers.

Downloads official UCI data for equality checks. Fits fixed published configs
on development only. Exports actual labels/predictions, not rounded scores.
"""
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/train_accuracy_audit_2026-11-03"
URL = "https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip"


def run():
    if OUT.exists():
        raise FileExistsError("Keep the previous audit; use a new output path for a further audit.")
    # Snapshot only already published evidence; no original artifact is written.
    protected = [p for directory in [ROOT / "results", ROOT / "figures", ROOT / "configs"]
                 for p in directory.rglob("*") if p.is_file()]
    protected += list((ROOT / "report").glob("*.docx"))
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    read = lambda name: json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))
    cfg = read("run_manifest.json")["config"]
    saved_split = read("split_indices.json")
    decision = read("selection.json")
    digits = load_digits()
    with urllib.request.urlopen(URL, timeout=60) as handle:
        archive_bytes = handle.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        name = next(n for n in archive.namelist() if Path(n).name == "optdigits.tes")
        official_bytes = archive.read(name)
    official = np.loadtxt(io.BytesIO(official_bytes), delimiter=",", dtype=np.int64)
    np.testing.assert_array_equal(official[:, :64], digits.data)
    np.testing.assert_array_equal(official[:, 64], digits.target)
    dataset_sha_before = hashlib.sha256(digits.data.tobytes() + digits.target.tobytes()).hexdigest()
    ids = np.arange(len(digits.target))
    dev, test = train_test_split(ids, test_size=cfg["test_size"], stratify=digits.target,
                                 random_state=cfg["split_seed"])
    np.testing.assert_array_equal(dev, saved_split["development"])
    np.testing.assert_array_equal(test, saved_split["test"])
    assert not set(dev) & set(test)
    # Array equality across boundaries would indicate duplicate feature rows.
    overlap = set(map(tuple, digits.data[dev])) & set(map(tuple, digits.data[test]))
    x, y = digits.data[dev].copy(), digits.target[dev].copy()
    x.flags.writeable = False
    y.flags.writeable = False
    cv = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=cfg["cv_seed"]).split(x, y))
    rows, prediction_rows, cache = [], [], {}
    for label, candidate in decision["candidates"].items():
        key = json.dumps(candidate["params"], sort_keys=True)
        if key not in cache:
            params = dict(candidate["params"])
            params["max_features"] = {"sqrt": "sqrt", "log2": "log2", "half": .5, "all": 1.}[params["max_features"]]
            model = RandomForestClassifier(**params, random_state=cfg["model_seed"], n_jobs=1).fit(x, y)
            predicted = model.predict(x)
            probabilities = model.predict_proba(x)
            assert not np.shares_memory(predicted, y)
            np.testing.assert_array_equal(predicted, model.classes_[probabilities.argmax(axis=1)])
            correct = int(np.count_nonzero(predicted == y))
            assert correct / len(y) == accuracy_score(y, predicted)
            oof = np.full(len(y), -1, dtype=int)
            for fold, (a, b) in enumerate(cv):
                np.testing.assert_array_equal(dev[a], saved_split["cv_folds_global"][fold]["train"])
                np.testing.assert_array_equal(dev[b], saved_split["cv_folds_global"][fold]["validation"])
                m = RandomForestClassifier(**params, random_state=cfg["model_seed"], n_jobs=1).fit(x[a], y[a])
                oof[b] = m.predict(x[b])
                assert np.isclose(f1_score(y[b], oof[b], average="macro"), candidate["fold_scores"][fold], atol=1e-12, rtol=0)
            cache[key] = (correct, predicted, oof)
        correct, predicted, oof = cache[key]
        rows.append({"label": label, "train_correct": correct, "train_total": len(y),
                     "train_errors": len(y) - correct, "train_accuracy_exact": correct / len(y),
                     "oof_validation_correct": int(np.count_nonzero(oof == y)),
                     "oof_validation_total": len(y), "oof_validation_accuracy": float(np.mean(oof == y))})
        prediction_rows.extend({"label": label, "dataset_index": int(index), "y_true": int(truth),
                                "train_prediction": int(pred), "validation_oof_prediction": int(val)}
                               for index, truth, pred, val in zip(dev, y, predicted, oof))
        print(rows[-1], flush=True)
    np.testing.assert_array_equal(x, official[dev, :64])
    np.testing.assert_array_equal(y, official[dev, 64])
    assert dataset_sha_before == hashlib.sha256(digits.data.tobytes() + digits.target.tobytes()).hexdigest()
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in before.items())
    OUT.mkdir(parents=True)
    for filename, data in [("counts.csv", rows), ("per_sample_predictions.csv", prediction_rows)]:
        with (OUT / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    findings = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "passed",
                "official_source": URL, "official_member": name,
                "official_member_sha256": hashlib.sha256(official_bytes).hexdigest(),
                "all_original_pixels_and_labels_equal_official_source": True,
                "train_labels_and_features_unchanged_after_fit": True,
                "prediction_array_is_separate_from_true_labels": True,
                "saved_split_and_folds_reproduced_independently": True,
                "development_test_index_overlap": 0,
                "identical_feature_rows_across_development_test": len(overlap),
                "distinct_configs_refitted": len(cache), "forest_fits": len(cache) * 4,
                "new_test_predictions": 0, "original_artifact_count_preserved": len(before),
                "dataset_sha256": dataset_sha_before, "counts": rows,
                "interpretation": "Train is resubstitution on fitted data. A perfect train score does not imply perfect generalization or rule out every form of error/bias."}
    (OUT / "audit.json").write_text(json.dumps(findings, indent=2, ensure_ascii=False), encoding="utf-8")
    print("PASS: UCI pixels/labels and all fixed model train counts independently reproduced.", flush=True)


if __name__ == "__main__":
    run()
