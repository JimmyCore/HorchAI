"""Evaluation metrics and reporting for keystroke classifiers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


@dataclass
class EvalResult:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    per_class: dict  # label -> {precision, recall, f1, support, errors}
    confusion: np.ndarray
    labels: list[str]


def evaluate(y_true: list[str], y_pred: list[str], labels: list[str]) -> EvalResult:
    """Compute accuracy, macro precision/recall/F1, confusion matrix, and
    per-class error counts.

    Args:
        y_true: ground-truth key labels.
        y_pred: predicted key labels.
        labels: full ordered label set (defines row/column order of the
            confusion matrix, and ensures classes with 0 support still show
            up as rows).
    """
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class = {}
    for i, label in enumerate(labels):
        errors = int(support[i] - cm[i, i]) if support[i] > 0 else 0
        per_class[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "errors": errors,
        }

    return EvalResult(
        accuracy=float(acc),
        precision_macro=float(np.mean(precision)),
        recall_macro=float(np.mean(recall)),
        f1_macro=float(np.mean(f1)),
        per_class=per_class,
        confusion=cm,
        labels=list(labels),
    )


def print_report(result: EvalResult) -> None:
    print(f"Accuracy:  {result.accuracy:.3f}")
    print(f"Precision (macro): {result.precision_macro:.3f}")
    print(f"Recall    (macro): {result.recall_macro:.3f}")
    print(f"F1        (macro): {result.f1_macro:.3f}")
    print()
    print(f"{'key':<6}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}{'errors':>10}")
    for label, m in result.per_class.items():
        print(
            f"{label:<6}{m['precision']:>10.3f}{m['recall']:>10.3f}{m['f1']:>10.3f}"
            f"{m['support']:>10d}{m['errors']:>10d}"
        )


def plot_confusion_matrix(result: EvalResult, ax=None):
    """Plot a normalized confusion matrix (rows sum to 1)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    cm = result.confusion.astype("float64")
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(result.labels)))
    ax.set_yticks(range(len(result.labels)))
    ax.set_xticklabels(result.labels)
    ax.set_yticklabels(result.labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (accuracy={result.accuracy:.2f})")

    for i in range(len(result.labels)):
        for j in range(len(result.labels)):
            count = int(result.confusion[i, j])
            if count > 0:
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, str(count), ha="center", va="center", color=color, fontsize=9)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax
