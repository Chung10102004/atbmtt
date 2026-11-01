"""Does adding IBM GFP multi-hop graph pattern features to the 63 behavioural features help?

GFP can only be computed in minute batches (a transaction also sees its own minute, H(t) = {t_j <= t}),
while the feature study is strict (H(t) = {t_j < t}). "F4i" recomputes the 63 features with the same
inclusive policy so that GFP is compared under one time policy.

Configurations (same XGBoost, same split, seeds 0/1/2):
    F4       63 strict-time features (feature study baseline)
    F4i      the same 63 features, inclusive same-minute policy
    F0+GFP   11 transaction features + all GFP features (close to the GFP paper setup), inclusive
    F4i+PAT  F4i + GFP pattern histograms only (fan, degree, scatter-gather, cycles)
    F4i+GFP  F4i + all GFP features (patterns + vertex statistics)
    F4+PAT / F4+GFP  strict F4 + GFP (mixed policies, reference only)
Besides AP/F1/P@1000, recall is reported per laundering typology (HI-Small_Patterns.txt, analysis only).
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from data import RAW_CSV, temporal_split  # noqa: E402
from feature_study import build_study_features, feature_groups  # noqa: E402
from gfp_features import load_sorted_raw  # noqa: E402
from model import XGB_PARAMS, evaluate, train  # noqa: E402
from typology import row_typology  # noqa: E402


def typology_recall(y, p, typ, threshold, k=1000):
    top = np.zeros(len(p), bool)
    top[np.argsort(-p, kind="stable")[:min(k, len(p))]] = True
    rows = []
    for name in sorted(set(typ[y == 1])):
        m = (typ == name) & (y == 1)
        rows.append({"typology": name, "positives": int(m.sum()),
                     "recall_at_threshold": float((p[m] >= threshold).mean()),
                     f"recall_in_top{k}": float(top[m].mean())})
    return rows


def summarize(df, metrics):
    g = df.groupby("exp", sort=False)[metrics]
    mean, std = g.mean(), g.std().fillna(0)
    out = pd.DataFrame({m: [f"{a:.4f} ± {s:.4f}" for a, s in zip(mean[m], std[m])] for m in metrics},
                       index=mean.index)
    return out.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=RAW_CSV)
    ap.add_argument("--patterns", type=Path, required=True)
    ap.add_argument("--gfp", type=Path, required=True, help=".npz written by src/gfp_features.py")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "results/gfp_study")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--exps", nargs="+", default=["F4", "F4i", "F0+GFP", "F4i+PAT", "F4i+GFP", "F4+GFP"])
    ap.add_argument("--smoke", action="store_true", help="tiny tree budget for a local smoke test only")
    args = ap.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        XGB_PARAMS.update(n_estimators=15, max_depth=3, early_stopping_rounds=4, n_jobs=2)

    started = time.perf_counter()
    raw = load_sorted_raw(args.raw)
    typ = row_typology(raw, args.patterns)
    gz = np.load(args.gfp)
    meta = json.loads(args.gfp.with_suffix(".json").read_text())
    if not np.array_equal(gz["csv_row"], raw["csv_row"].to_numpy()):
        raise ValueError("GFP features are not aligned with the sorted raw file")
    raw = raw.drop(columns="csv_row")
    f = build_study_features(raw)
    base = feature_groups(f)  # before adding columns: feature_groups puts every unknown column into "activity"
    behaviour = [c for c in base["F4"] if c not in base["F0"]]
    incl = build_study_features(raw, inclusive=True)[behaviour].add_prefix("i_")
    del raw
    gfp_cols = meta["columns"]
    f = pd.concat([f, incl, pd.DataFrame(gz["features"], columns=gfp_cols)], axis=1)
    del gz, incl
    gc.collect()
    feature_seconds = time.perf_counter() - started

    pat_cols =[c for c in gfp_cols if c.startswith("gfp_pat_")]
    f4i = base["F0"] + ["i_" + c for c in behaviour]
    groups = {"F4": base["F4"], "F4i": f4i, "F0+GFP": base["F0"] + gfp_cols,
              "F4i+PAT": f4i + pat_cols, "F4i+GFP": f4i + gfp_cols,
              "F4+PAT": base["F4"] + pat_cols, "F4+GFP": base["F4"] + gfp_cols}
    groups = {k: groups[k] for k in args.exps}
    parts, _ = temporal_split(f)
    n_tr, n_val = len(parts["train"][1]), len(parts["val"][1])
    typ_val, typ_test = typ[n_tr:n_tr + n_val], typ[n_tr + n_val:]
    del f
    gc.collect()
    (out / "manifest.json").write_text(json.dumps({
        "groups": {k: len(v) for k, v in groups.items()}, "gfp": {k: v for k, v in meta.items() if k != "columns"},
        "splits": {k: {"rows": len(y), "positives": int(y.sum())} for k, (_, y) in parts.items()},
        "seeds": args.seeds, "xgb_params": XGB_PARAMS, "smoke": args.smoke,
        "feature_seconds": feature_seconds}, indent=2, default=str))
    print(f"Features ready in {feature_seconds:.0f}s: { {k: len(v) for k, v in groups.items()} }", flush=True)

    rows, typ_rows = [], []
    (xtr, ytr), (xv, yv), (xt, yt) = parts["train"], parts["val"], parts["test"]
    for exp, cols in groups.items():
        a, b, c = xtr[cols], xv[cols], xt[cols]
        for seed in args.seeds:
            tick = time.perf_counter()
            model = train(a, ytr, b, yv, seed)
            row = {"exp": exp, "seed": seed, "n_features": len(cols), "train_seconds": time.perf_counter() - tick}
            row.update(evaluate(model, b, yv, c, yt))
            pv, pt = model.predict_proba(b)[:, 1], model.predict_proba(c)[:, 1]
            row["val_ap"] = float(average_precision_score(yv, pv))
            for split, y, p, t in (("val", yv, pv, typ_val), ("test", yt, pt, typ_test)):
                for r in typology_recall(y, p, t, row["threshold"]):
                    typ_rows.append({"exp": exp, "seed": seed, "split": split, **r})
            gain = model.get_booster().get_score(importance_type="total_gain")
            pd.DataFrame({"feature": cols, "total_gain": [float(gain.get(x, 0.)) for x in cols]}).to_csv(
                out / f"importance_{exp.replace('+', '_')}_seed{seed}.csv", index=False)
            rows.append(row)
            pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
            pd.DataFrame(typ_rows).to_csv(out / "typology_recall.csv", index=False)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
            del model
            gc.collect()

    res = pd.DataFrame(rows)
    summary = summarize(res, ["val_ap", "pr_auc", "f1", "precision_at_1000", "recall_at_1000"])
    t = pd.DataFrame(typ_rows)
    typ_summary = (t.groupby(["split", "typology", "exp"], sort=False)["recall_at_threshold"].mean()
                   .unstack("exp").reindex(columns=list(groups)).round(3))
    text = ("## Overall (test AP stored as pr_auc; val_ap = validation AP)\n\n" + summary.to_markdown(index=False) +
            "\n\n## Recall at the validation threshold, per typology (mean of seeds)\n\n" + typ_summary.to_markdown())
    (out / "summary.md").write_text(text)
    print(text, flush=True)
    print(f"Done in {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
