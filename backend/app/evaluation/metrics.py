"""Métricas de clasificación implementadas en NumPy (sin sklearn.metrics)."""
from __future__ import annotations

import numpy as np


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
        "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
        "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
        "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
    }


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AUC vía estadístico U de Mann-Whitney con rangos promedio para empates."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos, n_neg = int(y_true.sum()), int((1 - y_true).sum())
    if n_pos == 0 or n_neg == 0 or not np.all(np.isfinite(scores)):
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    i = 0
    while i < scores.size:
        j = i
        while j + 1 < scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    u = ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def roc_curve(y_true: np.ndarray, scores: np.ndarray, n_points: int = 101) -> dict[str, list[float]]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=0.5)
    thresholds = np.linspace(1.0, 0.0, n_points)
    P, N = max(y_true.sum(), 1), max((1 - y_true).sum(), 1)
    tpr, fpr = [], []
    for t in thresholds:
        pred = scores >= t
        tpr.append(float(np.sum(pred & (y_true == 1)) / P))
        fpr.append(float(np.sum(pred & (y_true == 0)) / N))
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds.tolist()}


def classification_report(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict:
    proba = np.asarray(proba, dtype=np.float64)
    finite = bool(np.all(np.isfinite(proba)))
    y_pred = (np.nan_to_num(proba, nan=0.0) >= threshold).astype(int)
    c = confusion(y_true, y_pred)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "confusion": c,
        "accuracy": _safe_div(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "sensitivity": recall,
        "specificity": _safe_div(tn, tn + fp),
        "f1": _safe_div(2 * precision * recall, precision + recall),
        "auc": roc_auc(y_true, proba) if finite else float("nan"),
        "threshold": threshold,
    }
