"""Load IBM AML (HI-Small), build features, and split train/val/test by time."""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = Path(os.environ.get("AML_RAW_CSV", ROOT / "data/raw/HI-Small_Trans.csv"))
FEATURES_PARQUET = ROOT / "data/processed/features.parquet"

LABEL = "is_laundering"
# Categorical features (integer codes) - augmentation must not interpolate these
CAT_COLS = ["pay_format", "pay_currency", "recv_currency",
            "same_bank", "self_txn", "cross_currency", "round_amt"]
SPLIT = (0.6, 0.2, 0.2)  # train / val / test in time order


def load_raw(path=RAW_CSV):
    df = pd.read_csv(path, dtype={"Account": str, "Account.1": str})
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M")
    return df.sort_values("Timestamp", kind="stable").reset_index(drop=True)


def _account_history_features(src, dst, t, log_amt):
    """Account history features computed causally (only transactions *before* the current one).

    Each transaction produces 2 events: an 'out' for the sender and an 'in' for the receiver.
    """
    n = len(src)
    tx = np.arange(n)
    ev = pd.DataFrame({
        "acct": np.concatenate([src, dst]),
        "tx": np.concatenate([tx, tx]),
        "is_out": np.concatenate([np.ones(n, np.int8), np.zeros(n, np.int8)]),
        "t": np.concatenate([t, t]),
        "amt": np.concatenate([log_amt, log_amt]),
    })
    ev = ev.iloc[np.lexsort((ev["tx"], ev["t"], ev["acct"]))].reset_index(drop=True)
    ev["day"] = ev["t"] // 86400

    g = ev.groupby("acct", sort=False)
    ev["prev_n"] = g.cumcount()
    ev["prev_out"] = g["is_out"].cumsum() - ev["is_out"]
    ev["prev_in"] = ev["prev_n"] - ev["prev_out"]
    out_amt = ev["amt"] * ev["is_out"]
    in_amt = ev["amt"] * (1 - ev["is_out"])
    ev["prev_out_amt_mean"] = ((out_amt.groupby(ev["acct"]).cumsum() - out_amt)
                               / ev["prev_out"].replace(0, np.nan))
    ev["prev_in_amt_mean"] = ((in_amt.groupby(ev["acct"]).cumsum() - in_amt)
                              / ev["prev_in"].replace(0, np.nan))
    ev["secs_since_prev"] = g["t"].diff()
    ev["same_day_n"] = ev.groupby(["acct", "day"], sort=False).cumcount()
    ev["same_day_dir_n"] = ev.groupby(["acct", "day", "is_out"], sort=False).cumcount()

    cols = ["prev_n", "prev_out", "prev_in", "prev_out_amt_mean", "prev_in_amt_mean",
            "secs_since_prev", "same_day_n", "same_day_dir_n"]
    feats = {}
    for role, mask in (("src", ev["is_out"] == 1), ("dst", ev["is_out"] == 0)):
        sub = ev.loc[mask]
        for c in cols:
            arr = np.full(n, np.nan, dtype=np.float32)
            arr[sub["tx"].to_numpy()] = sub[c].to_numpy(dtype=np.float32)
            feats[f"{role}_{c}"] = arr
    return pd.DataFrame(feats)


def build_features(df):
    src_s = df["From Bank"].astype(str) + "_" + df["Account"]
    dst_s = df["To Bank"].astype(str) + "_" + df["Account.1"]
    codes, _ = pd.factorize(pd.concat([src_s, dst_s], ignore_index=True))
    n = len(df)
    src, dst = codes[:n], codes[n:]
    t = df["Timestamp"].astype("int64").to_numpy() // 10**9
    log_paid = np.log1p(df["Amount Paid"].to_numpy())

    f = pd.DataFrame({
        "log_amt_paid": log_paid.astype(np.float32),
        "log_amt_recv": np.log1p(df["Amount Received"]).astype(np.float32),
        "hour": df["Timestamp"].dt.hour.astype(np.int8),
        "weekday": df["Timestamp"].dt.weekday.astype(np.int8),
        "pay_format": pd.factorize(df["Payment Format"], sort=True)[0].astype(np.int8),
        "pay_currency": pd.factorize(df["Payment Currency"], sort=True)[0].astype(np.int8),
        "recv_currency": pd.factorize(df["Receiving Currency"], sort=True)[0].astype(np.int8),
        "same_bank": (df["From Bank"] == df["To Bank"]).astype(np.int8),
        "self_txn": (src == dst).astype(np.int8),
        "cross_currency": (df["Payment Currency"] != df["Receiving Currency"]).astype(np.int8),
        "round_amt": np.isclose(np.mod(df["Amount Paid"], 100), 0).astype(np.int8),
    })
    # Number of prior transactions on the same (sender -> receiver) pair
    f["pair_prev_n"] = (pd.Series(src).groupby([src, dst], sort=False).cumcount()
                        .to_numpy(dtype=np.float32))
    f = pd.concat([f, _account_history_features(src, dst, t, log_paid)], axis=1)
    f["timestamp"] = t
    f[LABEL] = df["Is Laundering"].astype(np.int8).to_numpy()
    return f


def load_features(rebuild=False):
    if FEATURES_PARQUET.exists() and not rebuild:
        return pd.read_parquet(FEATURES_PARQUET)
    f = build_features(load_raw())
    FEATURES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    f.to_parquet(FEATURES_PARQUET, index=False)
    return f


def temporal_split(f):
    """Rows are already sorted by time -> cut by position (60/20/20)."""
    n = len(f)
    a, b = int(n * SPLIT[0]), int(n * (SPLIT[0] + SPLIT[1]))
    feat_cols = [c for c in f.columns if c not in (LABEL, "timestamp")]
    parts = {}
    for name, sl in (("train", slice(0, a)), ("val", slice(a, b)), ("test", slice(b, n))):
        part = f.iloc[sl]
        parts[name] = (part[feat_cols].reset_index(drop=True),
                       part[LABEL].to_numpy())
    return parts, feat_cols
