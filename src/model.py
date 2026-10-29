"""Train XGBoost and compute metrics. Hyperparameters are fixed across all experiments."""
import os

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from xgboost import XGBClassifier

XGB_PARAMS = dict(n_estimators=1000, learning_rate=0.05, max_depth=8, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=1, tree_method="hist",
                  eval_metric="aucpr", early_stopping_rounds=100, n_jobs=-1,
                  device=os.environ.get("AML_DEVICE", "cpu"))  # "cuda" when running on a Kaggle GPU


def train(X_tr, y_tr, X_val, y_val, seed):
    model = XGBClassifier(**XGB_PARAMS, random_state=seed)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[np.argmax(f1[:-1])])


def recall_at_fpr(y, p, fpr_target=0.01):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.interp(fpr_target, fpr, tpr))


def alert_budget_metrics(y, p, k=1000):
    k = min(k, len(y))
    if k < 1:
        raise ValueError("Alert budget and evaluation set must be nonempty")
    top = np.argsort(-np.asarray(p), kind="stable")[:k]
    tp = int(np.asarray(y)[top].sum())
    return {f"precision_at_{k}": tp / k,
            f"recall_at_{k}": tp / max(1, int(np.asarray(y).sum()))}


def evaluate(model, X_val, y_val, X_test, y_test, prediction_path=None):
    """The threshold is chosen on val (max F1) and then applied to test - never tuned on test."""
    thr = best_f1_threshold(y_val, model.predict_proba(X_val)[:, 1])
    p = model.predict_proba(X_test)[:, 1]
    if prediction_path is not None:
        np.savez_compressed(prediction_path, y=y_test, p=p, threshold=thr)
    pred = (p >= thr).astype(int)
    return {
        "pr_auc": average_precision_score(y_test, p),
        "roc_auc": roc_auc_score(y_test, p),
        "f1": f1_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
        "recall_at_fpr1": recall_at_fpr(y_test, p),
        "threshold": thr,
        "best_iter": model.best_iteration,
        **alert_budget_metrics(y_test, p),
    }
