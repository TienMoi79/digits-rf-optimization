"""Adversarial checks for prefix budgets, pairing, time and frozen results."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.common import ROOT
from src.robustness import (canonical_sha256, frozen_artifacts, load_development,
                            load_plan, make_schedule, score_development, verify_frozen)
from src.robustness_analysis import budget_records, common_time_curves, paired_ablation


def sample_row(method, seed, trial, score, elapsed):
    return dict(method=method,seed=seed,trial=trial,cv_f1=score,elapsed_seconds=elapsed,
                cv_seconds=.1,params={"n_estimators":50+25*trial})


class RobustnessTests(unittest.TestCase):
    def test_checkpoints_cannot_see_a_future_good_trial(self):
        rows = [sample_row("random",11,0,.8,1),sample_row("random",11,1,.7,2),sample_row("random",11,2,.99,3)]
        frame = budget_records(rows,[1,2,3])
        np.testing.assert_allclose(frame.best_cv_f1,[.8,.8,.99])
        self.assertEqual(frame.best_trial.tolist(),[0,0,2])
        with self.assertRaises(ValueError): budget_records(rows,[4])

    def test_equal_time_uses_last_completed_trial_without_future_scores(self):
        rows = [sample_row("random",11,0,.5,1),sample_row("random",11,1,.99,5),
                sample_row("tpe",11,0,.7,2),sample_row("tpe",11,1,.8,4)]
        frame = common_time_curves(rows,3)
        random = frame[frame.method=="random"]
        np.testing.assert_allclose(random.elapsed_seconds,[2,3,4])
        np.testing.assert_allclose(random.best_cv_f1,[.5,.5,.5])
        self.assertTrue((random.completed_trials==1).all())

    def test_missing_shared_time_window_is_not_filled_with_invented_scores(self):
        rows = [sample_row("a",11,0,.8,1),sample_row("a",11,1,.9,2),
                sample_row("b",11,0,.7,3),sample_row("b",11,1,.8,4)]
        with self.assertRaisesRegex(ValueError,"No shared time window"):
            common_time_curves(rows,3)

    def test_ablation_pairs_seeds_not_sorted_scores(self):
        rows = pd.DataFrame([
            dict(method="tpe_startup_5",seed=11,budget=30,best_cv_f1=.9),
            dict(method="tpe_startup_5",seed=22,budget=30,best_cv_f1=.8),
            dict(method="tpe_startup_10",seed=22,budget=30,best_cv_f1=.85),
            dict(method="tpe_startup_10",seed=11,budget=30,best_cv_f1=.87)])
        pairs = paired_ablation(rows,[5,10],30).set_index("seed")
        self.assertAlmostEqual(pairs.loc[11,"delta_low_minus_high"],.03)
        self.assertAlmostEqual(pairs.loc[22,"delta_low_minus_high"],-.05)
        with self.assertRaises(ValueError): paired_ablation(rows.iloc[:-1],[5,10],30)

    def test_frozen_results_reject_metric_or_model_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)
            (base/"models").mkdir()
            (base/"final_metrics.csv").write_text("metric\n0.9\n",encoding="utf-8")
            (base/"models/selected.joblib").write_bytes(b"original-model")
            expected=frozen_artifacts(base)
            (base/"validation.json").write_text('{}')
            verify_frozen(base,expected)
            (base/"final_metrics.csv").write_text("metric\n0.99\n",encoding="utf-8")
            with self.assertRaisesRegex(AssertionError,"artifacts changed"): verify_frozen(base,expected)

    def test_git_line_endings_do_not_invalidate_text_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            a,b=Path(temp)/"a.csv",Path(temp)/"b.csv"
            a.write_bytes(b"a,b\r\n1,2\r\n")
            b.write_bytes(b"a,b\n1,2\n")
            self.assertEqual(canonical_sha256(a),canonical_sha256(b))

    def test_development_loader_and_objective_do_not_receive_holdout_rows(self):
        x,y,folds,cfg,_=load_development(ROOT/"results")
        split=json.loads((ROOT/"results/split_indices.json").read_text())
        self.assertEqual(len(y),len(split["development"]))
        self.assertEqual(sorted(np.concatenate([b for a,b in folds]).tolist()),list(range(len(y))))
        def spy(model,features,target,**kwargs):
            np.testing.assert_array_equal(features,x)
            np.testing.assert_array_equal(target,y)
            self.assertEqual(len(target),1437)
            return np.array([.9,.91,.92])
        with patch("src.robustness.cross_val_score",side_effect=spy):
            metrics=score_development({"n_estimators":100,"max_depth":None,"min_samples_leaf":1,"max_features":"sqrt"},x,y,folds,cfg)
        self.assertAlmostEqual(metrics["cv_f1"],.91)
        self.assertFalse(any("test" in k for k in metrics))

    def test_plan_rotates_method_order_and_rejects_invalid_startup(self):
        plan=load_plan()
        schedule=make_schedule(plan)
        self.assertEqual(len(schedule),3*len(plan["search_seeds"]))
        self.assertEqual(len({r["method"] for r in schedule[::3]}),3)
        bad=copy.deepcopy(plan)
        bad["tpe_startup_trials"]=[5,30]
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"plan.json"
            p.write_text(json.dumps(bad))
            with self.assertRaises(ValueError): load_plan(p)


if __name__ == "__main__":
    unittest.main()
