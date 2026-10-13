"""Protocol tests: prevent leakage and accidental selection on test metrics."""
import copy
import unittest

import numpy as np

from src.analysis import pareto_mask
from src.common import load_config, make_data, make_model, DEFAULT_PARAMS
from src.experiment import choose_candidates


class ProtocolTests(unittest.TestCase):
    def test_holdout_never_appears_in_cv(self):
        cfg = load_config()
        _, dev, test, folds = make_data(cfg)
        self.assertFalse(set(dev) & set(test))
        for train, validation in folds:
            self.assertFalse(set(dev[train]) & set(dev[validation]))
            self.assertFalse((set(dev[train]) | set(dev[validation])) & set(test))
        self.assertEqual(sorted(np.concatenate([v for _,v in folds]).tolist()), list(range(len(dev))))

    def test_epsilon_selection_ignores_test_metrics(self):
        rows = [
            dict(method="default", seed=123, trial=-1, cv_f1=.97, model_bytes=500, feasible=True),
            dict(method="random", seed=11, trial=0, cv_f1=.981, model_bytes=900, feasible=True),
            dict(method="tpe", seed=11, trial=0, cv_f1=.978, model_bytes=300, feasible=True),
            dict(method="tpe", seed=11, trial=1, cv_f1=.94, model_bytes=100, feasible=True),
        ]
        first, _ = choose_candidates(rows, .005)
        modified = copy.deepcopy(rows)
        for row in modified:
            row["test_f1"] = 1.0 if row["model_bytes"] == 100 else 0.0
        second, _ = choose_candidates(modified, .005)
        self.assertEqual(first["selected"]["model_bytes"], 300)
        self.assertEqual(second["selected"]["model_bytes"], 300)
        strict, _ = choose_candidates(rows, 0)
        self.assertEqual(strict["selected"]["model_bytes"], 900)

    def test_infeasible_small_model_is_rejected(self):
        rows = [dict(method=m, seed=11, trial=0, cv_f1=.98, model_bytes=s, feasible=f)
                for m,s,f in [("default",500,True),("random",300,True),("tpe",10,False)]]
        chosen, _ = choose_candidates(rows, .005)
        self.assertEqual(chosen["selected"]["method"], "random")

    def test_pareto_dominance_handles_equal_points(self):
        np.testing.assert_array_equal(pareto_mask([.9,.9,.8,.95,.9], [10,20,5,30,10]),
                                      [True,False,True,True,True])

    def test_rf_baseline_keeps_default_learning_parameters(self):
        from sklearn.ensemble import RandomForestClassifier
        baseline = make_model(DEFAULT_PARAMS, load_config()).get_params()
        defaults = RandomForestClassifier().get_params()
        for name in ["n_estimators", "max_depth", "min_samples_leaf", "max_features", "criterion", "bootstrap"]:
            self.assertEqual(baseline[name], defaults[name])


if __name__ == "__main__":
    unittest.main()
