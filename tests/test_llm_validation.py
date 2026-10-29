import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import build_features
from test_data import transactions


class ValidationTests(unittest.TestCase):
    def rows(self):
        return build_features(transactions()).drop(columns=["timestamp", "is_laundering"])

    def test_valid_actual_feature_rows_are_accepted(self):
        from llm_validate import valid_mask
        self.assertTrue(valid_mask(self.rows()).all())

    def test_fractional_count_is_rejected(self):
        from llm_validate import valid_mask
        x = self.rows().iloc[[1]].copy()
        x.loc[:, "src_prev_out"] = 1.5
        self.assertFalse(valid_mask(x).any())

    def test_impossible_shared_minute_is_rejected(self):
        from llm_validate import valid_mask
        x = self.rows().iloc[[1]].copy()
        x.loc[:, "src_secs_since_prev"] = 86400
        self.assertFalse(valid_mask(x).any())

    def test_null_history_and_directional_counts_are_checked(self):
        from llm_validate import valid_mask
        x = self.rows().iloc[[1]].copy()
        x.loc[:, "src_prev_out_amt_mean"] = np.nan
        self.assertFalse(valid_mask(x).any())


if __name__ == "__main__":
    unittest.main()
