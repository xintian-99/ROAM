"""Evaluation metrics: binary (AUC, ...) and multiclass (QWK, ...)."""

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)


def compute_binary_metrics(labels, probs, threshold=0.5):
    """Compute metrics for binary classification.

    Args:
        labels: (N,) array of ground truth {0, 1}.
        probs: (N,) array of predicted probabilities for class 1.
        threshold: Decision threshold.

    Returns:
        dict of metric_name -> Python float (safe for JSON serialization).
    """
    preds = (np.array(probs) >= threshold).astype(int)
    labels = np.array(labels)

    metrics = {}
    try:
        metrics["auc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        metrics["auc"] = 0.0

    metrics["balanced_accuracy"] = float(balanced_accuracy_score(labels, preds))
    metrics["f1"] = float(f1_score(labels, preds, zero_division=0))

    # Sensitivity (recall for class 1) and specificity (recall for class 0)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return metrics


def compute_multiclass_metrics(labels, probs_matrix, n_classes):
    """Compute metrics for multi-class classification.

    Args:
        labels: list/array of int ground truth (0..n_classes-1).
        probs_matrix: list of arrays, each shape (n_classes,) softmax probs.
        n_classes: number of classes.

    Returns:
        dict of metric_name -> Python float.
    """
    labels = np.array(labels, dtype=int)
    probs_matrix = np.array(probs_matrix)  # (N, n_classes)
    preds = np.argmax(probs_matrix, axis=1)

    metrics = {}
    metrics["qwk"] = float(cohen_kappa_score(labels, preds, weights="quadratic",
                                             labels=list(range(n_classes))))
    metrics["macro_f1"] = float(f1_score(labels, preds, average="macro", zero_division=0))
    metrics["accuracy"] = float((labels == preds).mean())

    # Per-class accuracy
    per_class = {}
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            per_class[f"class_{c}_acc"] = float((preds[mask] == c).mean())
        else:
            per_class[f"class_{c}_acc"] = 0.0
    metrics["per_class_accuracy"] = per_class

    try:
        metrics["auc_ovr"] = float(
            roc_auc_score(labels, probs_matrix, multi_class="ovr", average="macro")
        )
    except ValueError:
        metrics["auc_ovr"] = 0.0

    return metrics
