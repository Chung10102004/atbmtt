import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from feature_study import build_study_features, feature_groups


def fixture():
    # A receives USD then EUR, pays B, and makes two simultaneous payments.
    return pd.DataFrame({
        "Timestamp": pd.to_datetime(["2022-09-01 00:00", "2022-09-01 00:30",
                                     "2022-09-01 01:00", "2022-09-01 01:00",
                                     "2022-09-01 01:01", "2022-09-02 01:00"]),
        "From Bank": [1]*6, "To Bank": [1]*6,
        "Account": ["X", "Y", "A", "A", "B", "A"],
        "Account.1": ["A", "A", "B", "A", "A", "B"],
        "Amount Paid": [100., 1000., 40., 20., 10., 5.],
        "Amount Received": [90., 900., 40., 20., 9., 5.],
        "Payment Currency": ["US Dollar", "Euro", "US Dollar", "US Dollar", "US Dollar", "US Dollar"],
        "Receiving Currency": ["US Dollar", "Euro", "US Dollar", "US Dollar", "US Dollar", "US Dollar"],
        "Payment Format": ["ACH"]*6, "Is Laundering": [0, 0, 1, 0, 1, 0],
    })


class StudyFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = fixture()
        cls.f = build_study_features(cls.df)

    def test_strict_ties_and_self_transfer_snapshot(self):
        f = self.f
        self.assertEqual(f.loc[2, "src_prev_n"], 2)
        self.assertEqual(f.loc[3, "src_prev_n"], 2)
        self.assertEqual(f.loc[3, "dst_prev_n"], 2)
        self.assertEqual(f.loc[4, "dst_prev_n"], 5)
        self.assertEqual(f.loc[2, "src_prev_out"], 0)

    def test_window_left_boundary_inclusive_right_exclusive(self):
        f = self.f
        self.assertEqual(f.loc[2, "src_in_n_1h"], 2)
        self.assertEqual(f.loc[4, "dst_in_n_1h"], 2)  # Y and self; X expired
        self.assertEqual(f.loc[5, "src_out_n_24h"], 2)  # exactly 24h included
        self.assertEqual(f.loc[5, "src_in_n_1h"], 0)

    def test_money_is_same_currency_and_uses_received_amount(self):
        f = self.f
        self.assertAlmostEqual(f.loc[2, "src_log_in_sum_1h"], np.log1p(90), places=5)
        self.assertAlmostEqual(f.loc[2, "src_log_amt_vs_in_mean"], np.log1p(40)-np.log1p(90), places=5)
        self.assertTrue(np.isnan(f.loc[2, "src_log_amt_vs_out_mean"]))

    def test_high_dynamic_range_does_not_leave_phantom_money(self):
        from feature_study import _money_scan
        actual = _money_scan(np.zeros(4,dtype=np.int64), np.array([0,1,3601,3602],np.int64),
                             np.ones(4,np.int8), np.array([1e12,.01,0.,0.]))
        self.assertAlmostEqual(actual[2,0], np.log1p(.01), places=8)
        self.assertEqual(actual[3,0], 0.)

    def test_partner_counts_reverse_pair_and_shares(self):
        f = self.f
        self.assertEqual(f.loc[2, "src_unique_in_1h"], 2)
        self.assertEqual(f.loc[4, "reverse_pair_prev_n"], 1)
        self.assertEqual(f.loc[4, "pair_prev_n"], 0)
        self.assertEqual(f.loc[5, "pair_n_24h"], 1)
        self.assertEqual(f.loc[5, "src_partner_share_24h"], .5)
        self.assertEqual(f.loc[5, "pair_age_seconds"], 86400)

    def test_permuting_same_timestamp_does_not_change_features(self):
        order = [0, 1, 3, 2, 4, 5]
        permuted = build_study_features(self.df.iloc[order].reset_index(drop=True))
        np.testing.assert_allclose(self.f.to_numpy(dtype=float)[order], permuted.to_numpy(dtype=float), equal_nan=True)

    def test_future_and_labels_do_not_affect_features(self):
        prefix = build_study_features(self.df.iloc[:4].copy())
        np.testing.assert_allclose(self.f.iloc[:4].to_numpy(dtype=float), prefix.to_numpy(dtype=float), equal_nan=True)
        changed = self.df.copy()
        changed["Is Laundering"] = 1-changed["Is Laundering"]
        compare = build_study_features(changed).drop(columns="is_laundering")
        np.testing.assert_allclose(self.f.drop(columns="is_laundering"), compare, equal_nan=True)

    def test_group_ablation_excludes_ids_labels_and_timestamps(self):
        groups = feature_groups(self.f)
        self.assertEqual(set(groups), {"F0", "F1", "F2", "F3", "F4"})
        self.assertEqual(len(groups["F0"]), 11)
        self.assertEqual(len(groups["F1"]), 28)
        self.assertEqual(set(groups["F2"]) & set(groups["F3"]), set(groups["F1"]))
        self.assertEqual(set(groups["F2"]) | set(groups["F3"]), set(groups["F4"]))
        for columns in groups.values():
            self.assertFalse({"timestamp", "is_laundering", "Account", "Account.1"} & set(columns))

    def test_unsorted_time_is_rejected(self):
        with self.assertRaises(ValueError):
            build_study_features(self.df.iloc[::-1])

    def test_randomized_features_against_independent_brute_force(self):
        rng = np.random.default_rng(47)
        n = 65
        df = pd.concat([fixture().iloc[[0]]]*n, ignore_index=True)
        df["Account"] = rng.choice(["A", "B", "C", "D"], n)
        df["Account.1"] = rng.choice(["A", "B", "C", "D"], n)
        df["From Bank"] = rng.integers(1, 3, n)
        df["To Bank"] = rng.integers(1, 3, n)
        df["Timestamp"] = pd.to_datetime("2022-09-01")+pd.to_timedelta(np.sort(rng.choice(np.arange(0, 200000, 600), n)), unit="s")
        for col in ["Payment Currency", "Receiving Currency"]:
            df[col] = rng.choice(["Euro", "US Dollar"], n)
        df["Amount Paid"] = rng.integers(0, 200, n).astype(float)
        df["Amount Received"] = rng.integers(0, 200, n).astype(float)
        actual = build_study_features(df)
        times = df.Timestamp.to_numpy(dtype="datetime64[s]").astype(np.int64)
        src = list(zip(df["From Bank"], df["Account"]))
        dst = list(zip(df["To Bank"], df["Account.1"]))
        # Independently enumerate historical events, without scanner/group state.
        for i in range(n):
            past = [j for j in range(n) if times[j] < times[i]]
            pair = [j for j in past if src[j] == src[i] and dst[j] == dst[i]]
            reverse = [j for j in past if src[j] == dst[i] and dst[j] == src[i]]
            self.assertEqual(actual.loc[i, "pair_prev_n"], len(pair))
            self.assertEqual(actual.loc[i, "reverse_pair_prev_n"], len(reverse))
            if pair:
                self.assertEqual(actual.loc[i, "pair_age_seconds"], times[i]-times[pair[0]])
                self.assertEqual(actual.loc[i, "pair_seconds_since_previous"], times[i]-times[pair[-1]])
            for role, acct, ccy, current in [
                    ("src", src[i], df.loc[i,"Payment Currency"], df.loc[i,"Amount Paid"]),
                    ("dst", dst[i], df.loc[i,"Receiving Currency"], df.loc[i,"Amount Received"])]:
                incoming = [j for j in past if dst[j] == acct]
                outgoing = [j for j in past if src[j] == acct]
                self.assertEqual(actual.loc[i, role+"_prev_n"], len(incoming)+len(outgoing))
                for direction, rows, amount_col, ccy_col, partner in [
                        ("in", incoming, "Amount Received", "Receiving Currency", src),
                        ("out", outgoing, "Amount Paid", "Payment Currency", dst)]:
                    self.assertEqual(actual.loc[i, f"{role}_prev_{direction}"], len(rows))
                    mean = np.log1p(df.loc[rows,amount_col]).mean() if rows else np.nan
                    np.testing.assert_allclose(actual.loc[i,f"{role}_prev_{direction}_amt_mean"], mean, atol=1e-5)
                    currency_rows = [j for j in rows if df.loc[j,ccy_col] == ccy]
                    dev = np.log1p(current)-np.log1p(df.loc[currency_rows,amount_col]).mean() if currency_rows else np.nan
                    np.testing.assert_allclose(actual.loc[i,f"{role}_log_amt_vs_{direction}_mean"], dev, atol=1e-5)
                    for seconds, label in [(3600,"1h"),(86400,"24h")]:
                        recent = [j for j in rows if times[j] >= times[i]-seconds]
                        self.assertEqual(actual.loc[i,f"{role}_{direction}_n_{label}"], len(recent))
                        self.assertEqual(actual.loc[i,f"{role}_unique_{direction}_{label}"], len({partner[j] for j in recent}))
                        money = sum(df.loc[j,amount_col] for j in recent if df.loc[j,ccy_col] == ccy)
                        np.testing.assert_allclose(actual.loc[i,f"{role}_log_{direction}_sum_{label}"], np.log1p(money), atol=1e-5)



if __name__ == "__main__":
    unittest.main()
