"""Guard demo prefix decisions, test isolation and preservation of results."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from src.optimization_demo import BASE, replay, run_live, summarize, validate_live


class OptimizationDemoTests(unittest.TestCase):
    def test_replay_selection_never_includes_future_trials(self):
        rows, summary = replay(seed=11, checkpoint=5)
        self.assertEqual(len(rows), 11)
        self.assertTrue(all(r["trial"] < 5 for r in rows))
        self.assertEqual(summary["test_evaluations"], 0)
        self.assertEqual(summary["best_cv_f1"], max(r["cv_f1"] for r in rows))
        self.assertIn(summary["selected"], rows)
        self.assertGreaterEqual(summary["selected"]["cv_f1"], summary["threshold"])

    def test_demo_rejects_published_output_and_a_budget_without_guided_trials(self):
        with self.assertRaisesRegex(ValueError, "NEW subfolder"):
            run_live(BASE / "robustness", trials=2, startup=1)
        with self.assertRaisesRegex(ValueError, "startup < trials"):
            run_live(BASE / "week9_demo" / "never-created", trials=5, startup=5)

    def test_live_objective_only_receives_development_and_equal_budgets(self):
        calls = []

        def development_score(params, x, y, folds, cfg):
            self.assertEqual(len(x), 1437)
            self.assertEqual(len(y), 1437)
            self.assertEqual(len(folds), 3)
            self.assertTrue(all(max(train.max(), val.max()) < 1437 for train, val in folds))
            calls.append(params)
            return {"cv_f1": .95, "model_bytes": params["n_estimators"] * 100,
                    "fold_scores": [.95] * 3, "feasible": True}

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch("src.optimization_demo.LIVE_ROOT", root), \
                 patch("src.optimization_demo.evaluate_cv", side_effect=development_score), \
                 patch("src.optimization_demo.plot_history"):
                rows, summary = run_live(root / "new-run", trials=2, startup=1)
                self.assertEqual(len(calls), 5)  # baseline plus two trials per method
                self.assertEqual(sum(r["method"] == "random" for r in rows), 2)
                self.assertEqual(sum(r["method"] == "tpe" for r in rows), 2)
                self.assertEqual([r["phase"] for r in rows if r["method"] == "tpe"], ["random", "TPE"])
                self.assertEqual(summary["test_evaluations"], 0)
                with self.assertRaises(FileExistsError):
                    run_live(root / "new-run", trials=2, startup=1)

    def test_changed_test_scores_cannot_change_demo_selection(self):
        rows, original = replay(seed=11, checkpoint=3)
        altered = [{**r, "test_accuracy": 0 if r == original["selected"] else 1} for r in rows]
        result = summarize(altered, original["epsilon"])
        self.assertEqual(result["selected"]["params"], original["selected"]["params"])

    def test_saved_demo_validation_rejects_a_missing_trial(self):
        source = BASE / "week9_demo/run-2026-11-01"
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory)
            for name in ["run_manifest.json", "demo_selection.json", "trials.json"]:
                (dest / name).write_bytes((source / name).read_bytes())
            rows = json.loads((dest / "trials.json").read_text(encoding="utf-8"))
            (dest / "trials.json").write_text(json.dumps(rows[:-1]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete or unequal budget"):
                validate_live(dest)


if __name__ == "__main__":
    unittest.main()
