"""Explicit ten-class metrics; absent predicted classes still count in macro-F1."""
import numpy as np


def classification_metrics(truth, probabilities):
    truth = np.asarray(truth, dtype=np.int64)
    probabilities = np.asarray(probabilities)
    if probabilities.shape != (len(truth), 10) or not np.isfinite(probabilities).all():
        raise ValueError("Expected finite (N,10) probabilities.")
    if not np.isin(truth, np.arange(10)).all() or not len(truth):
        raise ValueError("Invalid evaluation labels.")
    predicted = probabilities.argmax(axis=1)
    confusion = np.bincount(truth * 10 + predicted, minlength=100).reshape(10, 10)
    tp = np.diag(confusion).astype(float)
    support = confusion.sum(axis=1)
    precision = np.divide(tp, confusion.sum(axis=0), out=np.zeros(10), where=confusion.sum(axis=0) > 0)
    recall = np.divide(tp, support, out=np.zeros(10), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(10), where=precision + recall > 0)
    return {
        "accuracy": float((truth == predicted).mean()), "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
        "per_class": [{"class_id": i, "precision": float(precision[i]), "recall": float(recall[i]),
                       "f1": float(f1[i]), "support": int(support[i])} for i in range(10)],
        "confusion_matrix": confusion.tolist(),
    }


def normalized_aulc(budgets, scores):
    x, y = np.asarray(budgets, dtype=float), np.asarray(scores, dtype=float)
    if len(x) < 2:
        return float(y[0])
    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) * 0.5) / (x[-1] - x[0]))
