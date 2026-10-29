"""Rebuild and run AML experiments with isolated, versioned results."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from augment import GENERATORS, synth_quality
from data import FEATURE_VERSION, RAW_CSV, load_features, temporal_split
from model import XGB_PARAMS, evaluate, train

EXPERIMENTS = {"E0": None, "ER": "random", "E1": "smote", "E2": "ctgan", "E3": "llm"}
EXPERIMENTS.update({"E3-basic": "llm_basic", "E3-AML": "llm_aml"})


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def llm_cache_identity(directory, experiments, seeds):
    identities, models = {}, set()
    for exp, variant in (("E3-basic", "basic"), ("E3-AML", "aml")):
        if exp not in experiments:
            continue
        for seed in seeds:
            csv = directory / variant / f"llm_seed{seed}.csv"
            meta_path = csv.with_name(csv.stem + "_meta.json")
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if (metadata.get("seed") != seed or metadata.get("variant") != variant
                    or metadata.get("prompt_version") != "paired-basic-aml-v1" or not metadata.get("model")):
                raise ValueError("LLM cache has wrong seed, variant, prompt version or missing model")
            models.add(metadata["model"])
            identities[f"{variant}/seed{seed}"] = {"csv_sha256": sha256_file(csv), "metadata_sha256": sha256_file(meta_path)}
    if len(models) > 1:
        raise ValueError("Paired LLM caches must use the same model")
    return identities


def run_one(exp, seed, ratio, parts, output_dir):
    X_tr, y_tr = parts["train"]
    X_val, y_val = parts["val"]
    X_te, y_te = parts["test"]
    method = EXPERIMENTS[exp]
    row = {"exp": exp, "method": method or "none", "seed": seed,
           "ratio": ratio if method else 0., "n_synth": 0, "feature_version": FEATURE_VERSION}
    t0 = time.time()
    if method:
        X_pos = X_tr[y_tr == 1].reset_index(drop=True)
        n = int(round(ratio * len(X_pos)))
        X_syn = GENERATORS[method](X_pos, n, seed)
        if len(X_syn) != n:
            raise ValueError(f"{exp}: expected {n} synthetic rows, got {len(X_syn)}")
        row.update(n_synth=n, repaired_rows=X_syn.attrs.get("repaired_rows", 0),
                   **synth_quality(X_pos, X_syn, seed))
        X_syn.to_parquet(output_dir / f"{exp}_seed{seed}_synthetic.parquet", index=False)
        X_tr = pd.concat([X_tr, X_syn], ignore_index=True)
        y_tr = np.r_[y_tr, np.ones(n, dtype=y_tr.dtype)]
    row["gen_secs"] = time.time() - t0
    t0 = time.time()
    model = train(X_tr, y_tr, X_val, y_val, seed)
    row["train_secs"] = time.time() - t0
    row.update(evaluate(model, X_val, y_val, X_te, y_te,
                        prediction_path=output_dir / f"{exp}_seed{seed}_predictions.npz"))
    del model
    return row


def summarize(df):
    metrics = [m for m in ["pr_auc", "roc_auc", "f1", "precision", "recall", "recall_at_fpr1",
                           "precision_at_1000", "recall_at_1000", "ks_mean", "disc_auc"] if m in df]
    grouped = df.groupby(["exp", "method", "ratio"])[metrics]
    mean, std = grouped.mean(), grouped.std().fillna(0)
    out = mean.copy().astype(object)
    for m in metrics:
        out[m] = ["-" if np.isnan(a) else f"{a:.4f} +/- {s:.4f}" for a, s in zip(mean[m], std[m])]
    out["runs"] = grouped.size()
    return out.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", choices=list(EXPERIMENTS), default=["E0", "ER", "E1", "E2"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--ratio", type=float, default=.5)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "results/rerun_20261029")
    ap.add_argument("--rebuild-features", action="store_true")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--llm-dir", type=Path, default=ROOT / "data/llm/rerun_20261029")
    args = ap.parse_args()
    os.environ["AML_LLM_DIR"] = str(args.llm_dir.resolve())
    cache_inputs = llm_cache_identity(args.llm_dir.resolve(), args.exps, args.seeds)
    if not np.isfinite(args.ratio) or args.ratio <= 0:
        ap.error("--ratio must be positive and finite")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not args.resume:
        raise ValueError("Output already has a manifest; choose a new directory or use --resume")
    print("Building/loading versioned features...", flush=True)
    f = load_features(rebuild=args.rebuild_features)
    parts, columns = temporal_split(f)
    split_stats, start = {}, 0
    for name, (x, y) in parts.items():
        stop = start + len(y)
        split_stats[name] = {"rows": len(y), "positives": int(y.sum()),
                             "first_timestamp": int(f.timestamp.iloc[start]),
                             "last_timestamp": int(f.timestamp.iloc[stop - 1])}
        print(f"{name}: {len(y):,} rows, {int(y.sum()):,} positives", flush=True)
        start = stop
    del f
    gc.collect()
    sources = [*sorted((ROOT / "src").glob("*.py")), Path(__file__).resolve()]
    versions = {}
    for name in ["numpy", "pandas", "scikit-learn", "xgboost", "imbalanced-learn", "ctgan", "torch", "pyarrow"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    manifest = {"feature_version": FEATURE_VERSION, "columns": columns, "splits": split_stats,
                "raw_sha256": sha256_file(RAW_CSV), "ratio": args.ratio,
                "seeds": args.seeds, "experiments": args.exps, "xgboost_params": XGB_PARAMS,
                "source_sha256": {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sources},
                "python": platform.python_version(), "packages": versions, "llm_cache_inputs": cache_inputs,
                "primary_metric": "average_precision_score (AP), stored as pr_auc for compatibility",
                "alert_budget": 1000, "split_policy": "timestamp-disjoint; stable transaction order within each timestamp"}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError("Cannot resume: data, source, environment or parameters changed")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    X, y = parts["train"]
    positive_path = output / "train_positives.csv"
    X[y == 1].to_csv(positive_path, index=False)
    (output / "train_positives_meta.json").write_text(json.dumps({
        "feature_version": FEATURE_VERSION, "sha256": sha256_file(positive_path),
        "rows": int(y.sum()), "columns": columns, "splits": split_stats,
    }, indent=2), encoding="utf-8")
    if args.prepare_only:
        print("Prepared training positives and manifest; no detector trained.", flush=True)
        return
    csv = output / "results.csv"
    df = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
    for exp in args.exps:
        for seed in args.seeds:
            if not df.empty and ((df.exp == exp) & (df.seed == seed)).any():
                print(f"Already recorded: {exp} seed {seed}", flush=True)
                continue
            print(f"Running {exp} seed {seed}...", flush=True)
            row = run_one(exp, seed, args.ratio, parts, output)
            print(json.dumps(row), flush=True)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_csv(csv, index=False)
            summary = summarize(df)
            summary.to_csv(output / "summary.csv", index=False)
            try:
                rendered = summary.to_markdown(index=False)
            except ImportError:
                rendered = summary.to_string(index=False)
            (output / "summary.md").write_text(rendered, encoding="utf-8")
            gc.collect()
    print(summarize(df).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
