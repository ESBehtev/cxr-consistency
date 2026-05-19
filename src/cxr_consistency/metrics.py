import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def compute_binary_metrics(labels, probs, threshold: float = 0.5) -> dict:
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)

    preds = (probs >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "threshold": float(threshold),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))
    except Exception:
        metrics["roc_auc"] = 0.0

    try:
        metrics["pr_auc"] = float(average_precision_score(labels, probs))
    except Exception:
        metrics["pr_auc"] = 0.0

    tn, fp, fn, tp = confusion_matrix(
        labels,
        preds,
        labels=[0, 1],
    ).ravel()

    metrics["tn"] = int(tn)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    metrics["tp"] = int(tp)

    return metrics


def find_best_threshold(
    labels,
    probs,
    thresholds=None,
    metric_name: str = "f1",
) -> dict:
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)

    best_metrics = None
    best_score = -1.0

    for threshold in thresholds:
        metrics = compute_binary_metrics(
            labels=labels,
            probs=probs,
            threshold=float(threshold),
        )

        score = metrics[metric_name]

        if score > best_score:
            best_score = score
            best_metrics = metrics

    return best_metrics


def compute_metrics_with_best_threshold(labels, probs) -> dict:
    default_metrics = compute_binary_metrics(
        labels=labels,
        probs=probs,
        threshold=0.5,
    )

    best_metrics = find_best_threshold(
        labels=labels,
        probs=probs,
        metric_name="f1",
    )

    result = {}

    for key, value in default_metrics.items():
        result[key] = value

    for key, value in best_metrics.items():
        result[f"best_{key}"] = value

    return result