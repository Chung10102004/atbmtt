import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data


def transactions():
    return pd.DataFrame({
        "Timestamp": pd.to_datetime(["2022-09-01 12:00", "2022-09-01 12:01", "2022-09-01 12:02"]),
        "From Bank": [1, 1, 1], "To Bank": [1, 1, 1],
        "Account": ["A", "A", "B"], "Account.1": ["A", "B", "C"],
        "Amount Paid": [100., 100., 50.], "Amount Received": [100., 200., 50.],
        "Payment Format": ["ACH"] * 3,
        "Payment Currency": ["US Dollar"] * 3,
        "Receiving Currency": ["US Dollar", "Euro", "US Dollar"],
        "Is Laundering": [0, 1, 0],
    })


class FeatureTests(unittest.TestCase):
    def test_timestamp_seconds_independent_of_datetime_storage_unit(self):
        for unit in ["us", "ns"]:
            df = transactions()
            df["Timestamp"] = df["Timestamp"].astype(f"datetime64[{unit}]")
            self.assertEqual(int(data.build_features(df).timestamp.iloc[0]), 1662033600)

    def test_self_transfer_has_no_history_from_its_own_other_event(self):
        f = data.build_features(transactions())
        self.assertEqual(f.loc[0, "dst_prev_n"], 0)
        self.assertEqual(f.loc[0, "dst_same_day_n"], 0)
        self.assertTrue(np.isnan(f.loc[0, "dst_secs_since_prev"]))
        self.assertEqual(f.loc[1, "src_prev_n"], 2)
        self.assertEqual(f.loc[1, "src_secs_since_prev"], 60)

    def test_receiver_history_uses_received_amount(self):
        f = data.build_features(transactions())
        self.assertAlmostEqual(float(f.loc[2, "src_prev_in_amt_mean"]), float(np.log1p(200)), places=5)

    def test_currency_codes_share_one_codebook(self):
        f = data.build_features(transactions())
        self.assertEqual(f.loc[0, "pay_currency"], f.loc[0, "recv_currency"])

    def test_timestamp_ties_do_not_cross_split_boundaries(self):
        f = pd.DataFrame({"timestamp": [1, 2, 3, 4, 5, 6, 6, 7, 7, 8],
                          "is_laundering": [0] * 10, "id": range(10)})
        parts, _ = data.temporal_split(f)
        self.assertEqual(parts["train"][0].id.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(parts["val"][0].id.tolist(), [5, 6])
        self.assertEqual(parts["test"][0].id.tolist(), [7, 8, 9])

    def test_unsorted_features_are_rejected(self):
        f = pd.DataFrame({"timestamp": [2, 1, 3, 4, 5], "is_laundering": [0] * 5})
        with self.assertRaises(ValueError):
            data.temporal_split(f)

    def test_stale_cache_is_rejected(self):
        # Existing file without matching cache metadata must never be accepted.
        with patch.object(data, "FEATURES_PARQUET", Path(__file__)):
            with self.assertRaises(ValueError):
                data.load_features()


if __name__ == "__main__":
    unittest.main()
