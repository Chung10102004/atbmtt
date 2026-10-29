"""Necessary row-level invariants, not verification of graph laundering typologies."""
import numpy as np
import pandas as pd

from data import CAT_COLS, CURRENCIES, PAY_FORMATS


def valid_mask(frame):
    x = frame.apply(pd.to_numeric, errors="coerce")
    valid = pd.Series(True, index=x.index)
    optional = [c for c in x if c.endswith("_amt_mean") or c.endswith("_secs_since_prev")]
    required = [c for c in x if c not in optional]
    valid &= x[required].notna().all(axis=1)
    valid &= pd.Series(~np.isinf(x.to_numpy(dtype=float)).any(axis=1), index=x.index)
    domains = {"hour": range(24), "weekday": range(7), "pay_format": range(len(PAY_FORMATS)),
               "pay_currency": range(len(CURRENCIES)), "recv_currency": range(len(CURRENCIES))}
    for c in CAT_COLS:
        domains.setdefault(c, [0, 1])
    for c, values in domains.items():
        valid &= x[c].isin(values)
    counts = [c for c in x if c == "pair_prev_n" or c.endswith(("_prev_n", "_prev_out", "_prev_in", "_same_day_n", "_same_day_dir_n"))]
    for c in counts:
        valid &= (x[c] >= 0) & (x[c] == np.floor(x[c]))
    valid &= (x.cross_currency == (x.pay_currency != x.recv_currency).astype(int))
    same_currency = x.pay_currency == x.recv_currency
    valid &= ~same_currency | np.isclose(x.log_amt_paid, x.log_amt_recv, atol=.015, rtol=0)
    valid &= (x.log_amt_paid >= 0) & (x.log_amt_recv >= 0)
    # Intersect possible current minutes for the two account histories.
    minute = x.hour.to_numpy()[:, None] * 60 + np.arange(60)[None, :]
    possible = np.ones((len(x), 60), dtype=bool)
    for side, direction, opposite in (("src", "out", "in"), ("dst", "in", "out")):
        total, out, incoming = (x[f"{side}_prev_{c}"] for c in ("n", "out", "in"))
        day, day_dir = x[f"{side}_same_day_n"], x[f"{side}_same_day_dir_n"]
        valid &= (total == out + incoming) & (day <= total) & (day_dir <= day)
        valid &= (day_dir <= x[f"{side}_prev_{direction}"])
        valid &= (day - day_dir <= x[f"{side}_prev_{opposite}"])
        for d in ("out", "in"):
            mean = x[f"{side}_prev_{d}_amt_mean"]
            valid &= ((x[f"{side}_prev_{d}"] == 0) == mean.isna()) & (mean.isna() | (mean >= 0))
        gap = x[f"{side}_secs_since_prev"]
        valid &= ((total == 0) == gap.isna()) & (gap.isna() | ((gap >= 0) & (gap % 60 == 0)))
        valid &= gap.notna() | (day == 0)
        in_today = gap.to_numpy()[:, None] <= minute * 60
        possible &= gap.isna().to_numpy()[:, None] | (in_today == (day.to_numpy()[:, None] > 0))
    valid &= possible.any(axis=1)
    valid &= x.pair_prev_n <= np.minimum(x.src_prev_out, x.dst_prev_in)
    same_account = x.self_txn == 1
    valid &= ~same_account | (x.same_bank == 1)
    for suffix in ("prev_n", "prev_out", "prev_in", "prev_out_amt_mean", "prev_in_amt_mean", "secs_since_prev", "same_day_n"):
        equal = np.isclose(x[f"src_{suffix}"], x[f"dst_{suffix}"], atol=.015, rtol=0, equal_nan=True)
        valid &= ~same_account | equal
    valid &= ~same_account | (x.src_same_day_dir_n + x.dst_same_day_dir_n == x.src_same_day_n)
    return valid


def training_fingerprint(frame):
    import hashlib
    # Quantize explicitly so a CSV round trip across pandas versions is stable.
    array = frame.to_numpy(dtype=np.float32).round(4)
    array = np.nan_to_num(array, nan=-1).astype("<f4")
    digest = hashlib.sha256("\n".join(frame.columns).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()
