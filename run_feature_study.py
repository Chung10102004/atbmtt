"""Run the predeclared strict-timestamp feature ablation, without augmentation."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/"src"))
from data import RAW_CSV, temporal_split
from feature_study import VERSION, build_study_features, feature_groups
from model import XGB_PARAMS, evaluate, train


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=RAW_CSV)
    ap.add_argument("--output-dir", type=Path, default=ROOT/"results/feature_study_20261029")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0,1,2])
    ap.add_argument("--smoke", action="store_true", help="Small tree budget for a synthetic local smoke, never the main results")
    args = ap.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output/"manifest.json").exists():
        raise ValueError("Output already contains a study; choose a new directory")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Duplicate seeds")
    if args.smoke:
        XGB_PARAMS.update(n_estimators=15, max_depth=3, early_stopping_rounds=4, n_jobs=2)
    started = time.perf_counter()
    raw = pd.read_csv(args.raw, dtype={"Account": str, "Account.1": str})
    raw["Timestamp"] = pd.to_datetime(raw["Timestamp"], format="%Y/%m/%d %H:%M")
    raw = raw.sort_values("Timestamp", kind="stable").reset_index(drop=True)
    print(f"Loaded {len(raw):,} transactions", flush=True)
    f = build_study_features(raw)
    del raw
    gc.collect()
    elapsed = time.perf_counter()-started
    groups = feature_groups(f)
    parts, columns = temporal_split(f)
    start, split_stats, arrays = 0, {}, {}
    for name, (x,y) in parts.items():
        stop = start+len(y)
        times = f.timestamp.iloc[start:stop].to_numpy()
        split_stats[name] = {"rows":len(y), "positives":int(y.sum()),
                             "first_timestamp":int(times[0]), "last_timestamp":int(times[-1])}
        if name != "train":
            arrays[f"{name}_y"] = y
            arrays[f"{name}_timestamp"] = times
            arrays[f"{name}_row_id"] = np.arange(start,stop,dtype=np.int64)
        start = stop
    feature_memory = int(f.memory_usage(deep=True).sum())
    del f
    gc.collect()
    sources = ["src/data.py", "src/model.py", "src/feature_study.py", "run_feature_study.py"]
    versions = {}
    for name in ["numpy", "pandas", "scikit-learn", "xgboost", "numba"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed; Python feature scanner"
    manifest = {"feature_version":VERSION, "raw_sha256":sha256(args.raw),
                "sources":{p:sha256(ROOT/p) for p in sources}, "versions":versions,
                "groups":groups, "splits":split_stats, "seeds":args.seeds,
                "xgb_params":XGB_PARAMS, "smoke":args.smoke,
                "feature_build_seconds_including_io_jit":elapsed, "feature_memory_bytes":feature_memory,
                "history_policy":"strict timestamp: all events with time < current time; windows [t-W,t)",
                "test_reuse":"HI-Small test was already examined in prior augmentation study; exploratory evaluation",
                "new_money_policy":"account/currency-specific; no cross-currency sums"}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    np.savez_compressed(output/"split_arrays.npz", **arrays)
    pd.DataFrame([{"feature":c,"family":("raw" if c in groups["F0"] else
        "history" if c in groups["F1"] else "activity" if c in groups["F2"] else "relationships")}
        for c in columns]).to_csv(output/"feature_catalog.csv",index=False)
    print(f"Features: { {k:len(v) for k,v in groups.items()} }; built in {elapsed:.1f}s",flush=True)
    print(f"Splits: {split_stats}",flush=True)
    rows = []
    for exp, names in groups.items():
        xtr, ytr = parts["train"]
        xv, yv = parts["val"]
        xt, yt = parts["test"]
        a,b,c = xtr[names],xv[names],xt[names]
        for seed in args.seeds:
            print(f"Training {exp} seed{seed}: {len(names)} features",flush=True)
            tick = time.perf_counter()
            model = train(a,ytr,b,yv,seed)
            row = {"exp":exp,"seed":seed,"n_features":len(names),"train_seconds":time.perf_counter()-tick}
            row.update(evaluate(model,b,yv,c,yt,prediction_path=output/f"{exp}_seed{seed}_test.npz"))
            pv = model.predict_proba(b)[:,1]
            row["val_ap"] = float(average_precision_score(yv,pv))
            np.savez_compressed(output/f"{exp}_seed{seed}_val.npz",y=yv,p=pv,threshold=row["threshold"])
            gain = model.get_booster().get_score(importance_type="total_gain")
            pd.DataFrame([{"feature":name,"total_gain":float(gain.get(name,0.))} for name in names]).to_csv(
                output/f"{exp}_seed{seed}_importance.csv",index=False)
            rows.append(row)
            pd.DataFrame(rows).to_csv(output/"results.csv",index=False)
            print(json.dumps(row),flush=True)
            del model, pv
            gc.collect()
        del a,b,c
    (output/"complete.json").write_text(json.dumps({"complete":True,"runs":len(rows),
        "elapsed_seconds":time.perf_counter()-started},indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
