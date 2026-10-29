"""Generate synthetic laundering transactions (positive class) to add to the train set.

Every method shares the signature: gen(X_pos, n, seed) -> DataFrame of n synthetic rows
with the same columns as X_pos. E3 (LLM) only needs to add a function to GENERATORS.
"""
import numpy as np
import pandas as pd
import os
import json
import hashlib
from pathlib import Path
from functools import partial
from scipy.stats import ks_2samp
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from data import CAT_COLS, FEATURE_VERSION, ROOT


def _postprocess(X_syn, X_pos):
    """Clip to the value range of the real data and round categorical/count columns."""
    X_syn = X_syn[X_pos.columns].copy()
    for c in X_pos.columns:
        lo, hi = X_pos[c].min(), X_pos[c].max()
        X_syn[c] = X_syn[c].clip(lo, hi)
        if pd.api.types.is_integer_dtype(X_pos[c]):
            X_syn[c] = X_syn[c].round().astype(X_pos[c].dtype)
        else:
            X_syn[c] = X_syn[c].astype(X_pos[c].dtype)
    before = X_syn.copy()
    counts = [c for c in X_syn if c == "pair_prev_n" or c.endswith(("_prev_n", "_prev_out", "_prev_in", "_same_day_n", "_same_day_dir_n"))]
    for c in counts:
        X_syn[c] = X_syn[c].clip(lower=0).round()
    for side, direction in (("src", "out"), ("dst", "in")):
        if f"{side}_prev_out" not in X_syn:
            continue
        X_syn[f"{side}_prev_n"] = X_syn[f"{side}_prev_out"] + X_syn[f"{side}_prev_in"]
        X_syn[f"{side}_same_day_n"] = X_syn[[f"{side}_same_day_n", f"{side}_prev_n"]].min(axis=1)
        X_syn[f"{side}_same_day_dir_n"] = X_syn[[f"{side}_same_day_dir_n", f"{side}_same_day_n", f"{side}_prev_{direction}"]].min(axis=1)
        opposite = "in" if direction == "out" else "out"
        lower = (X_syn[f"{side}_same_day_n"] - X_syn[f"{side}_prev_{opposite}"]).clip(lower=0)
        X_syn[f"{side}_same_day_dir_n"] = np.maximum(X_syn[f"{side}_same_day_dir_n"], lower)
        for d in ("out", "in"):
            c = f"{side}_prev_{d}_amt_mean"
            X_syn.loc[X_syn[f"{side}_prev_{d}"] == 0, c] = np.nan
            has_history = X_syn[f"{side}_prev_{d}"] > 0
            X_syn.loc[has_history, c] = X_syn.loc[has_history, c].fillna(X_pos[c].median())
        gap = f"{side}_secs_since_prev"
        X_syn.loc[X_syn[f"{side}_prev_n"] == 0, gap] = np.nan
        has_history = X_syn[f"{side}_prev_n"] > 0
        X_syn.loc[has_history, gap] = (X_syn.loc[has_history, gap].fillna(X_pos[gap].median()) / 60).round() * 60
    if "self_txn" in X_syn:
        same_account = X_syn.self_txn == 1
        shared = ["prev_n", "prev_out", "prev_in", "prev_out_amt_mean", "prev_in_amt_mean", "secs_since_prev", "same_day_n"]
        for suffix in shared:
            X_syn.loc[same_account, f"dst_{suffix}"] = X_syn.loc[same_account, f"src_{suffix}"].to_numpy()
        X_syn.loc[same_account, "dst_same_day_dir_n"] = (X_syn.loc[same_account, "src_same_day_n"] - X_syn.loc[same_account, "src_same_day_dir_n"])
        X_syn.loc[same_account, "same_bank"] = 1
    if "pair_prev_n" in X_syn:
        X_syn["pair_prev_n"] = X_syn[["pair_prev_n", "src_prev_out", "dst_prev_in"]].min(axis=1)
    if "cross_currency" in X_syn:
        X_syn["cross_currency"] = (X_syn.pay_currency != X_syn.recv_currency).astype(X_pos.cross_currency.dtype)
        same = X_syn.pay_currency == X_syn.recv_currency
        X_syn.loc[same, "log_amt_recv"] = X_syn.loc[same, "log_amt_paid"]
    changed = ~np.isclose(before.to_numpy(dtype=float), X_syn.to_numpy(dtype=float), equal_nan=True)
    X_syn.attrs["repaired_rows"] = int(changed.any(axis=1).sum())
    return X_syn.reset_index(drop=True)


def gen_smote(X_pos, n, seed):
    """SMOTENC: interpolate between nearby positive samples; categorical columns use the neighbours' mode.

    SMOTE only uses the positive class, so we pass in a few dummy negative rows
    instead of all ~3M rows to save memory.
    """
    from imblearn.over_sampling import SMOTENC
    from sklearn.preprocessing import StandardScaler

    original = X_pos
    numeric = [c for c in X_pos.columns if c not in CAT_COLS]
    missing = [c for c in numeric if X_pos[c].isna().any()]
    X_pos = X_pos.copy()
    for c in missing:
        X_pos[c + "__missing"] = X_pos[c].isna().astype(np.int8)
        X_pos[c] = X_pos[c].fillna(X_pos[c].median())
    scaler = StandardScaler()
    X_pos[numeric] = scaler.fit_transform(X_pos[numeric])
    dummy_neg = X_pos.iloc[:10]
    X = pd.concat([X_pos, dummy_neg], ignore_index=True)
    y = np.r_[np.ones(len(X_pos), int), np.zeros(len(dummy_neg), int)]
    categories = [*CAT_COLS, *[c + "__missing" for c in missing]]
    sm = SMOTENC(categorical_features=[X.columns.get_loc(c) for c in categories],
                 sampling_strategy={1: len(X_pos) + n}, k_neighbors=5, random_state=seed)
    X_res, _ = sm.fit_resample(X, y)
    X_syn = X_res.iloc[len(X):].copy()
    X_syn[numeric] = scaler.inverse_transform(X_syn[numeric])
    for c in missing:
        X_syn.loc[X_syn[c + "__missing"] == 1, c] = np.nan
    return _postprocess(X_syn, original)


def gen_ctgan(X_pos, n, seed, epochs=300):
    """CTGAN (conditional GAN for tabular data), trained only on the positive class."""
    import torch
    from ctgan import CTGAN

    torch.manual_seed(seed)
    np.random.seed(seed)
    import inspect
    gpu_arg = "enable_gpu" if "enable_gpu" in inspect.signature(CTGAN).parameters else "cuda"
    model = CTGAN(epochs=epochs, batch_size=500, verbose=False,
                  **{gpu_arg: torch.cuda.is_available()})
    model.set_random_state(seed)
    model.fit(X_pos.fillna(-1), discrete_columns=CAT_COLS)  # -1 = "no history"
    X_syn = model.sample(n)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    nan_cols = X_pos.columns[X_pos.isna().any()]
    X_syn[nan_cols] = X_syn[nan_cols].mask(X_syn[nan_cols] < -0.5)  # restore NaN
    return _postprocess(X_syn, X_pos)


def gen_llm(X_pos, n, seed):
    """LLM (ChatGPT API): read the file pre-generated by generate_llm_data.py (no API call here)."""
    from llm_gen import load_cached

    X_syn = load_cached(seed)[X_pos.columns]
    if len(X_syn) < n:
        raise ValueError(f"LLM cache has {len(X_syn)} rows; {n} required. Generate the full cache first.")
    X_syn = X_syn.sample(n, random_state=seed)
    return _postprocess(X_syn, X_pos)


def gen_random(X_pos, n, seed):
    return X_pos.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)


def gen_llm_variant(X_pos, n, seed, variant):
    from llm_validate import training_fingerprint, valid_mask
    base = Path(os.environ.get("AML_LLM_DIR", ROOT / "data/llm/rerun_20261029"))
    csv = base / variant / f"llm_seed{seed}.csv"
    meta = json.loads(csv.with_name(csv.stem + "_meta.json").read_text(encoding="utf-8"))
    if (meta.get("feature_version") != FEATURE_VERSION or meta.get("variant") != variant
            or meta.get("seed") != seed or meta.get("prompt_version") != "paired-basic-aml-v1" or not meta.get("model")
            or meta.get("train_fingerprint") != training_fingerprint(X_pos)):
        raise ValueError("LLM cache does not match the feature version, prompt variant or training set")
    if meta.get("cache_sha256") != hashlib.sha256(csv.read_bytes()).hexdigest():
        raise ValueError("LLM cache hash differs from its metadata")
    x = pd.read_csv(csv)[X_pos.columns]
    if not meta.get("complete") or len(x) != meta.get("rows") or len(x) < n:
        raise ValueError(f"Incomplete LLM cache: {len(x)} rows; {n} required")
    if not valid_mask(x).all() or x.duplicated().any():
        raise ValueError("LLM cache contains invalid or duplicate rows")
    return _postprocess(x.sample(n=n, random_state=seed), X_pos)


GENERATORS = {"random": gen_random, "smote": gen_smote, "ctgan": gen_ctgan, "llm": gen_llm}
GENERATORS.update(llm_basic=partial(gen_llm_variant, variant="basic"),
                  llm_aml=partial(gen_llm_variant, variant="aml"))


def synth_quality(X_pos, X_syn, seed=0):
    """Quality of synthetic data relative to the real positive samples.

    - ks_mean: mean KS statistic per column (0 = identical distributions).
    - disc_auc: AUC of a real-vs-synthetic classifier (0.5 = indistinguishable).
    """
    ks = [ks_2samp(X_pos[c].dropna(), X_syn[c].dropna()).statistic
          for c in X_pos.columns if X_syn[c].notna().any()]
    m = min(len(X_pos), len(X_syn))
    X = pd.concat([X_pos.sample(m, random_state=seed), X_syn.sample(m, random_state=seed)])
    y = np.r_[np.zeros(m), np.ones(m)]
    clf = XGBClassifier(n_estimators=100, max_depth=4, n_jobs=8, random_state=seed)
    auc = cross_val_score(clf, X, y, cv=5, scoring="roc_auc").mean()
    return {"ks_mean": float(np.mean(ks)), "disc_auc": float(auc)}
