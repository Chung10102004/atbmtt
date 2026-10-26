"""Run the experiments. The test set (real data) is fixed; only the train set changes.

    E0: original train
    E1: original train + SMOTE
    E2: original train + CTGAN
    E3: original train + LLM (ChatGPT API, pre-generated with generate_llm_data.py)

Examples:
    python run_experiments.py --exps E0 E1 E2 --seeds 0 1 2 --ratio 1.0
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from augment import GENERATORS, synth_quality  # noqa: E402
from data import load_features, temporal_split  # noqa: E402
from model import evaluate, train  # noqa: E402

EXPERIMENTS = {"E0": None, "E1": "smote", "E2": "ctgan", "E3": "llm"}
RESULTS = Path(__file__).parent / "results"


def run_one(exp, seed, ratio, parts):
    X_tr, y_tr = parts["train"]
    X_val, y_val = parts["val"]
    X_te, y_te = parts["test"]
    row = {"exp": exp, "method": EXPERIMENTS[exp] or "none", "seed": seed,
           "ratio": ratio if EXPERIMENTS[exp] else 0.0, "n_synth": 0}

    t0 = time.time()
    if EXPERIMENTS[exp]:
        X_pos = X_tr[y_tr == 1].reset_index(drop=True)
        n = int(round(ratio * len(X_pos)))
        X_syn = GENERATORS[EXPERIMENTS[exp]](X_pos, n, seed)
        row.update(n_synth=len(X_syn), **synth_quality(X_pos, X_syn, seed))
        X_tr = pd.concat([X_tr, X_syn], ignore_index=True)
        y_tr = np.r_[y_tr, np.ones(len(X_syn), dtype=y_tr.dtype)]
    row["gen_secs"] = time.time() - t0

    t0 = time.time()
    model = train(X_tr, y_tr, X_val, y_val, seed)
    row["train_secs"] = time.time() - t0
    row.update(evaluate(model, X_val, y_val, X_te, y_te))
    return row


def summarize(df):
    metrics = ["pr_auc", "roc_auc", "f1", "precision", "recall", "recall_at_fpr1",
               "ks_mean", "disc_auc"]
    metrics = [m for m in metrics if m in df]
    g = df.groupby(["exp", "method", "ratio"])[metrics]
    mean, std = g.mean(), g.std().fillna(0)
    out = mean.copy().astype(object)
    for m in metrics:
        out[m] = [("-" if np.isnan(a) else f"{a:.4f} ± {s:.4f}")
                  for a, s in zip(mean[m], std[m])]
    out["runs"] = g.size()
    return out.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", default=["E0", "E1", "E2"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--ratio", type=float, default=1.0,
                    help="number of synthetic samples = ratio * number of real positives in train")
    args = ap.parse_args()

    parts, _ = temporal_split(load_features())
    for k, (X, y) in parts.items():
        print(f"{k}: {len(y):,} rows, {int(y.sum()):,} laundering ({y.mean():.3%})")

    RESULTS.mkdir(exist_ok=True)
    csv = RESULTS / "results.csv"
    for exp in args.exps:
        for seed in args.seeds:
            row = run_one(exp, seed, args.ratio, parts)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in row.items()}))
            # Rewrite the whole file because E0 lacks the ks_mean/disc_auc columns -> appending would misalign columns
            df = pd.DataFrame([row])
            if csv.exists():
                df = pd.concat([pd.read_csv(csv), df], ignore_index=True)
            df.to_csv(csv, index=False)

    summary = summarize(pd.read_csv(csv))
    (RESULTS / "summary.md").write_text(summary.to_markdown(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
