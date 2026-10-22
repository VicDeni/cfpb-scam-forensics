"""Metrics for comparing the baseline and LLM prompt variants."""

from collections import Counter

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from .labels import LABELS
from .prompts import PARSE_ERROR

CLASSES = list(LABELS)


def summarize(y_true, y_pred, input_tokens=None, output_tokens=None, cost_usd=None) -> dict:
    """Accuracy, macro-F1, parse-error rate, and token/cost per 1,000 complaints.

    PARSE_ERROR predictions count as wrong, never dropped, so a prompt can't
    look better by refusing hard cases.
    """
    n = len(y_true)
    result = {
        "n": n,
        "accuracy": accuracy_score(y_true, y_pred),
        # Macro-F1 over the classes actually present, so a partial run isn't
        # penalized for classes it never saw.
        "macro_f1": f1_score(y_true, y_pred, labels=sorted(set(y_true)), average="macro",
                             zero_division=0),
        "parse_error_rate": sum(p == PARSE_ERROR for p in y_pred) / n if n else 0.0,
    }
    if input_tokens is not None:
        result["avg_input_tokens"] = sum(input_tokens) / n
        result["avg_output_tokens"] = sum(output_tokens) / n
    if cost_usd is not None:
        result["cost_per_1k_usd"] = 1000 * sum(cost_usd) / n
    return result


def per_class(y_true, y_pred) -> list:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, zero_division=0
    )
    return [
        {"label": c, "precision": p, "recall": r, "f1": f, "support": int(s)}
        for c, p, r, f, s in zip(CLASSES, precision, recall, f1, support)
    ]


def top_confusions(y_true, y_pred, k: int = 10) -> list:
    """Most common (true, predicted) mistakes, for error analysis."""
    pairs = Counter((t, p) for t, p in zip(y_true, y_pred) if t != p)
    return [{"true_label": t, "predicted": p, "count": c} for (t, p), c in pairs.most_common(k)]
