"""Ensure failure analysis includes all errors and cannot silently deduplicate."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import pandas as pd

from src.week10 import BASE, compare_runs, failure_records


class Week10Tests(unittest.TestCase):
    def test_error_pair_direction_and_comparator_flags(self):
        frame = pd.DataFrame([
            dict(dataset_index=1, y_true=8, default=1, random=8, tpe=1, selected=1),
            dict(dataset_index=2, y_true=8, default=1, random=1, tpe=1, selected=1),
            dict(dataset_index=3, y_true=1, default=1, random=8, tpe=1, selected=1),
        ])
        errors, pairs = failure_records(frame)
        self.assertEqual(errors.dataset_index.tolist(), [1, 2])
        self.assertEqual(errors.random_correct.tolist(), [True, False])
        self.assertEqual(pairs.to_dict(orient="records"), [dict(y_true=8, selected_prediction=1, count=2)])

    def test_duplicate_test_indices_are_rejected(self):
        row = dict(dataset_index=1, y_true=8, default=1, random=8, tpe=1, selected=1)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            failure_records(pd.DataFrame([row, row]))

    def copy_comparison_fixture(self, dest):
        dest.mkdir()
        (dest / "models").mkdir()
        for name in ["run_manifest.json", "split_indices.json", "trials.json", "selection.json",
                     "test_predictions.csv", "final_metrics.csv", "models/selected.joblib"]:
            shutil.copyfile(BASE / name, dest / name)

    def test_runtime_change_is_reported_without_altering_reproducibility(self):
        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder) / "rerun"
            self.copy_comparison_fixture(dest)
            path = dest / "run_manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["runtime_seconds"] *= 1.5
            path.write_text(json.dumps(manifest), encoding="utf-8")
            result, _, _, _ = compare_runs(BASE, dest)
            self.assertTrue(result["all_trial_params_and_fold_scores_match"])
            self.assertAlmostEqual(result["runtime_ratio_new_over_original"], 1.5)
            self.assertTrue(result["all_test_predictions_match"])

    def test_changed_trial_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder) / "rerun"
            self.copy_comparison_fixture(dest)
            path = dest / "trials.json"
            rows = json.loads(path.read_text(encoding="utf-8"))
            rows[1]["params"]["n_estimators"] += 25
            path.write_text(json.dumps(rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "trial configuration differs"):
                compare_runs(BASE, dest)


if __name__ == "__main__":
    unittest.main()
