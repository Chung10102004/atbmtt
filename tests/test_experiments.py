import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class ExperimentTests(unittest.TestCase):
    def test_baseline_does_not_require_augmentation_libraries(self):
        import augment
        self.assertIn("random", augment.GENERATORS)

    def test_random_oversampling_only_repeats_training_positives(self):
        from augment import GENERATORS
        x = pd.DataFrame({"value": [2., 9.]})
        generated = GENERATORS["random"](x, 20, 0)
        self.assertEqual(len(generated), 20)
        self.assertEqual(set(generated.value), {2., 9.})

    def test_short_llm_cache_is_rejected(self):
        from augment import gen_llm
        with patch("llm_gen.load_cached", return_value=pd.DataFrame({"value": [1.]})):
            with self.assertRaises(ValueError):
                gen_llm(pd.DataFrame({"value": [1., 2.]}), 2, 0)

    def test_llm_variants_are_available_as_separate_experiments(self):
        from augment import GENERATORS
        self.assertIn("llm_basic", GENERATORS)
        self.assertIn("llm_aml", GENERATORS)

    def test_alert_budget_metrics_rank_by_score(self):
        from model import alert_budget_metrics
        scores = np.array([.8, .9, .1, .7])
        labels = np.array([0, 1, 1, 1])
        result = alert_budget_metrics(labels, scores, k=2)
        self.assertEqual(result["precision_at_2"], .5)
        self.assertAlmostEqual(result["recall_at_2"], 1 / 3)

    def test_generated_counts_and_null_history_are_consistent(self):
        from augment import _postprocess
        x = pd.DataFrame({"src_prev_out": [0., 4.], "src_prev_in": [0., 2.],
                          "src_prev_n": [0., 6.], "src_same_day_n": [0., 3.],
                          "src_same_day_dir_n": [0., 2.], "src_prev_out_amt_mean": [np.nan, 4.],
                          "src_prev_in_amt_mean": [np.nan, 5.], "src_secs_since_prev": [np.nan, 60.]})
        syn = x.iloc[[1]].copy()
        syn.loc[:, "src_prev_out"] = 1.7
        syn.loc[:, "src_prev_in"] = 0.
        syn.loc[:, "src_prev_n"] = 5.
        result = _postprocess(syn, x)
        self.assertEqual(result.src_prev_out.iloc[0], 2.)
        self.assertEqual(result.src_prev_n.iloc[0], 2.)
        self.assertLessEqual(result.src_same_day_n.iloc[0], 2.)
        self.assertTrue(np.isnan(result.src_prev_in_amt_mean.iloc[0]))

    def test_same_day_opposite_direction_cannot_exceed_lifetime_count(self):
        from augment import _postprocess
        from data import build_features
        from test_data import transactions
        x = build_features(transactions()).drop(columns=["timestamp", "is_laundering"])
        syn = x.iloc[[1]].copy()
        syn.loc[:, "src_same_day_dir_n"] = 0
        result = _postprocess(syn, x)
        self.assertLessEqual(float(result.src_same_day_n.iloc[0] - result.src_same_day_dir_n.iloc[0]),
                             float(result.src_prev_in.iloc[0]))

    def test_self_transfer_keeps_one_shared_history_snapshot(self):
        from augment import _postprocess
        from data import build_features
        from test_data import transactions
        x = build_features(transactions()).drop(columns=["timestamp", "is_laundering"])
        syn = x.iloc[[1]].copy()
        syn.loc[:, "self_txn"] = 1
        result = _postprocess(syn, x)
        self.assertEqual(result.src_prev_n.iloc[0], result.dst_prev_n.iloc[0])
        self.assertEqual(result.src_secs_since_prev.iloc[0], result.dst_secs_since_prev.iloc[0])
        self.assertEqual(result.src_same_day_dir_n.iloc[0] + result.dst_same_day_dir_n.iloc[0], result.src_same_day_n.iloc[0])


if __name__ == "__main__":
    unittest.main()
