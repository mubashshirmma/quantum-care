"""Binary classification metrics. y_score is any monotonic confidence for class 1."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def compute_metrics(y_true, y_pred, y_score=None) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),          # sensitivity
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if y_score is not None else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_test": int(len(y_true)),
    }
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in out.items()}


METRIC_COLUMNS = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]


def format_comparison_table(results: dict[str, dict], extra: dict[str, dict] | None = None) -> str:
    """results: {model_name: metrics_dict}. extra: {model_name: {'train_time_s': ..}} optional."""
    name_w = max(len("Model"), *(len(n) for n in results))
    header = f"{'Model':<{name_w}}  " + "  ".join(f"{c:>11}" for c in METRIC_COLUMNS)
    if extra:
        header += f"  {'train_s':>8}"
    lines = [header, "-" * len(header)]
    for name, m in results.items():
        row = f"{name:<{name_w}}  " + "  ".join(
            f"{(m[c] if m[c] is not None else float('nan')):>11.4f}" for c in METRIC_COLUMNS)
        if extra:
            row += f"  {extra.get(name, {}).get('train_time_s', float('nan')):>8.2f}"
        lines.append(row)
    return "\n".join(lines)


def format_confusion(m: dict) -> str:
    cm = m["confusion_matrix"]
    return (f"                 pred 0   pred 1\n"
            f"  actual 0 (neg) {cm['tn']:>6}   {cm['fp']:>6}\n"
            f"  actual 1 (pos) {cm['fn']:>6}   {cm['tp']:>6}")
