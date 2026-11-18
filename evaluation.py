"""Evaluation helpers for the semantic router."""
from __future__ import annotations

import numpy as np
import pandas as pd

from router import FALLBACK


def predict(S: np.ndarray, names: list[str], threshold: float):
    S = np.asarray(S)
    best_idx = S.argmax(axis=1)
    best = S.max(axis=1)
    preds = [names[i] if b >= threshold else FALLBACK for i, b in zip(best_idx, best)]
    return preds, best


def accuracy(S, y_true, names, threshold) -> float:
    preds, _ = predict(S, names, threshold)
    return float(np.mean(np.array(preds) == np.array(y_true)))


def sweep(S, y_true, names, thresholds=None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.round(np.arange(0.0, 0.91, 0.01), 2)
    return pd.DataFrame(
        {
            "threshold": thresholds,
            "accuracy": [accuracy(S, y_true, names, t) for t in thresholds],
        }
    )


def best_threshold(sw: pd.DataFrame) -> tuple[float, float]:
    """Median threshold of the plateau that achieves max accuracy."""
    top = sw[sw["accuracy"] == sw["accuracy"].max()]
    return float(top["threshold"].median()), float(sw["accuracy"].max())


def evaluate(S, queries, y_true, names, threshold) -> dict:
    y_pred, best = predict(S, names, threshold)
    labels = list(names) + [FALLBACK]

    rows = []
    for lab in labels:
        tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred))
        fp = sum(t != lab and p == lab for t, p in zip(y_true, y_pred))
        fn = sum(t == lab and p != lab for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append(
            {
                "route": lab,
                "precision": round(prec, 3),
                "recall": round(rec, 3),
                "f1": round(f1, 3),
                "support": tp + fn,
            }
        )

    confusion = pd.crosstab(
        pd.Series(y_true, name="actual"), pd.Series(y_pred, name="predicted")
    ).reindex(index=labels, columns=labels, fill_value=0)

    errors = pd.DataFrame(
        [
            {"query": q, "expected": t, "predicted": p, "top score": round(float(b), 3)}
            for q, t, p, b in zip(queries, y_true, y_pred, best)
            if t != p
        ],
        columns=["query", "expected", "predicted", "top score"],
    )

    return {
        "accuracy": float(np.mean(np.array(y_pred) == np.array(y_true))),
        "report": pd.DataFrame(rows),
        "confusion": confusion,
        "errors": errors,
    }