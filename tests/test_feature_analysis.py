import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


class FeatureAnalysisTests(unittest.TestCase):
    def test_fixed_budget_recall_is_not_recall_at_fpr(self):
        from analyze_feature_study import alert_columns
        self.assertEqual(alert_columns(["recall_at_fpr1","precision_at_1000","recall_at_1000"]),
                         ("precision_at_1000","recall_at_1000",1000))

    def test_independent_metrics_catches_score_and_label_corruption(self):
        from analyze_feature_study import checked_metrics
        y = np.array([0,1,0,1])
        p = np.array([.1,.9,.2,.8])
        metrics = checked_metrics(y,p,.5,y)
        self.assertEqual(metrics["pr_auc"],1.)
        self.assertEqual(metrics["f1"],1.)
        self.assertEqual(metrics["precision_at_4"],.5)
        with self.assertRaises(ValueError):
            checked_metrics(y[::-1],p,.5,y)
        with self.assertRaises(ValueError):
            checked_metrics(y,np.array([.1,np.nan,.2,.8]),.5,y)
        with self.assertRaises(ValueError):
            checked_metrics(y,np.array([.1,1.1,.2,.8]),.5,y)


if __name__ == "__main__":
    unittest.main()
