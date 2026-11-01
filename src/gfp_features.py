"""Graph pattern features from IBM's Graph Feature Preprocessor (Snap ML), computed in time order.

snapml only ships wheels for Python <= 3.12, so on Kaggle (Python 3.13) this file runs as a CLI in a
separate 3.12 environment and only needs numpy/pandas/snapml. It writes an .npz aligned with the
timestamp-sorted raw file (same stable sort as run_feature_study.py).

Time policy: edges are inserted minute by minute. GraphFeaturePreprocessor.transform inserts a batch
before computing its features, so a transaction sees every earlier minute plus the other transactions
of its own minute (order-invariant), never a later minute. Calling fit_transform on the whole file
would let every edge see future edges and is therefore not used.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

BINS = [2, 3]
# Settings follow the GFP paper for the AML datasets: 6h window for scatter-gather, 1 day otherwise.
GFP_PARAMS = {
    "time_window": 86400,
    "vertex_stats": True, "vertex_stats_cols": [4], "vertex_stats_feats": [0, 1, 2, 3, 4, 8, 9, 10],
    "fan": True, "fan_tw": 86400, "fan_bins": BINS,
    "degree": True, "degree_tw": 86400, "degree_bins": BINS,
    "scatter-gather": True, "scatter-gather_tw": 21600, "scatter-gather_bins": BINS,
    "temp-cycle": True, "temp-cycle_tw": 86400, "temp-cycle_bins": BINS,
    "lc-cycle": True, "lc-cycle_tw": 86400, "lc-cycle_len": 10, "lc-cycle_bins": BINS,
}
N_PATTERN = 14  # 7 pattern histograms x len(BINS) columns; the remaining columns are vertex statistics


def load_sorted_raw(path):
    raw = pd.read_csv(path, dtype={"Account": str, "Account.1": str})
    raw["Timestamp"] = pd.to_datetime(raw["Timestamp"], format="%Y/%m/%d %H:%M")
    raw["csv_row"] = np.arange(len(raw))
    return raw.sort_values("Timestamp", kind="stable").reset_index(drop=True)


def edge_list(raw):
    """[edge id, source, target, timestamp, log1p(amount paid)] as float64, as GFP expects."""
    n = len(raw)
    accounts = pd.concat([raw["From Bank"].astype(str) + "_" + raw["Account"],
                          raw["To Bank"].astype(str) + "_" + raw["Account.1"]], ignore_index=True)
    account = pd.factorize(accounts)[0]
    t = raw["Timestamp"].to_numpy(dtype="datetime64[s]").astype(np.int64)
    return np.c_[np.arange(n), account[:n], account[n:], t, np.log1p(raw["Amount Paid"].to_numpy(float))
                 ].astype(np.float64)


def stream_gfp(X, params=GFP_PARAMS, threads=None):
    from snapml import GraphFeaturePreprocessor

    gfp = GraphFeaturePreprocessor()
    gfp.set_params({**params, "num_threads": threads or os.cpu_count()})
    t = X[:, 3]
    if np.any(t[1:] < t[:-1]):
        raise ValueError("Edges must be sorted by timestamp")
    bounds = np.r_[0, np.flatnonzero(np.diff(t)) + 1, len(X)]
    out = np.empty((len(X), 0), np.float32)
    for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        batch = gfp.fit_transform(X[a:b]) if k == 0 else gfp.transform(X[a:b])
        if k == 0:
            out = np.empty((len(X), batch.shape[1] - X.shape[1]), np.float32)
        out[a:b] = batch[:, X.shape[1]:]
    return out


def column_names(n_cols):
    return ([f"gfp_pat_{i:02d}" for i in range(N_PATTERN)] +
            [f"gfp_vs_{i:02d}" for i in range(n_cols - N_PATTERN)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    started = time.perf_counter()
    raw = load_sorted_raw(args.raw)
    X = edge_list(raw)
    tick = time.perf_counter()
    F = stream_gfp(X)
    gfp_seconds = time.perf_counter() - tick
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, features=F, csv_row=raw["csv_row"].to_numpy(), timestamp=X[:, 3].astype(np.int64))
    meta = {"columns": column_names(F.shape[1]), "params": GFP_PARAMS, "rows": len(F),
            "gfp_seconds": gfp_seconds, "total_seconds": time.perf_counter() - started,
            "time_policy": "minute batches: earlier minutes + same minute visible, later minutes never"}
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "columns"}), flush=True)


if __name__ == "__main__":
    main()
