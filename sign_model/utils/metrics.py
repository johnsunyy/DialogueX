"""
sign_model/utils/metrics.py
----------------------------------------------------------------------
Evaluation helpers for the sign language recognition model.

Functions:
  - top_k_accuracy      : compute top-k accuracy
  - per_class_report    : precision/recall/F1 per class, pass/warn/fail
  - confusion_pairs     : find the N most confused class pairs
  - measure_inference   : time a single forward pass in ms

Usage:
  from utils.metrics import top_k_accuracy, per_class_report
----------------------------------------------------------------------
"""

import time
import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from typing import List, Dict, Tuple


def top_k_accuracy(y_true: np.ndarray, y_probs: np.ndarray, k: int = 3) -> float:
    """
    Compute top-k accuracy.

    Parameters
    ----------
    y_true  : 1-D array of true class indices
    y_probs : 2-D array of shape (N, num_classes) — raw logits or probabilities
    k       : number of top predictions to consider

    Returns
    -------
    float in [0, 1]
    """
    top_k_preds = np.argsort(y_probs, axis=1)[:, -k:]
    correct = sum(
        y_true[i] in top_k_preds[i] for i in range(len(y_true))
    )
    return correct / len(y_true)


def per_class_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    pass_threshold: float = 0.70,
    warn_threshold: float = 0.50,
) -> Dict[str, Dict]:
    """
    Compute per-class precision, recall, and F1, then assign pass/warn/fail.

    Returns
    -------
    dict keyed by class_name:
        {precision, recall, f1, support, status}
    """
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    result = {}
    for cls in class_names:
        if cls not in report:
            continue
        f1 = report[cls]["f1-score"]
        if f1 >= pass_threshold:
            status = "PASS [OK]"
        elif f1 >= warn_threshold:
            status = "WARN [!!]"
        else:
            status = "FAIL [XX]"
        result[cls] = {
            "precision": round(report[cls]["precision"], 4),
            "recall": round(report[cls]["recall"], 4),
            "f1": round(f1, 4),
            "support": int(report[cls]["support"]),
            "status": status,
        }
    return result


def confusion_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    n: int = 10,
) -> List[Tuple[str, str, int]]:
    """
    Find the N most confused class PAIRS (off-diagonal elements in confusion matrix).

    Returns
    -------
    list of (true_class, pred_class, count) sorted descending by count
    """
    cm = confusion_matrix(y_true, y_pred)
    np.fill_diagonal(cm, 0)  # zero out diagonal (correct predictions)

    # Get top-n off-diagonal indices
    flat = cm.flatten()
    top_indices = np.argsort(flat)[::-1][:n]

    pairs = []
    for idx in top_indices:
        row, col = divmod(idx, len(class_names))
        count = cm[row, col]
        if count == 0:
            break
        pairs.append((class_names[row], class_names[col], int(count)))

    return pairs


def measure_inference_ms(
    model: torch.nn.Module,
    sample: np.ndarray,
    device: torch.device,
    n_repeats: int = 50,
) -> float:
    """
    Measure average inference time for a single sample.

    Parameters
    ----------
    model    : trained PyTorch model
    sample   : numpy array shape (target_frames, feature_dim)
    device   : torch device
    n_repeats: number of forward passes to average over

    Returns
    -------
    float — average inference time in milliseconds
    """
    model.eval()
    # Add batch dimension and send to device
    x = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)

    # Warm up
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)

    # Timed runs
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_repeats):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) / n_repeats * 1000.0  # ms


def print_confusion_pairs(pairs: List[Tuple[str, str, int]]) -> None:
    """Pretty-print the top confused pairs."""
    print("\n── Top Most Confused Class Pairs ─────────────────────────────")
    print(f"  {'True Class':<20}  {'Predicted As':<20}  {'Count':>7}")
    print("  " + "-" * 52)
    for true_cls, pred_cls, count in pairs:
        print(f"  {true_cls:<20}  {pred_cls:<20}  {count:>7}")
    print()


def print_per_class_report(metrics: Dict[str, Dict]) -> None:
    """Pretty-print per-class pass/warn/fail table."""
    print("\n── Per-Class Metrics ─────────────────────────────────────────")
    print(f"  {'Class':<20}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Sup':>5}  Status")
    print("  " + "-" * 72)
    for cls, m in sorted(metrics.items(), key=lambda x: x[1]["f1"], reverse=True):
        print(
            f"  {cls:<20}  {m['precision']:>6.3f}  {m['recall']:>6.3f}  "
            f"{m['f1']:>6.3f}  {m['support']:>5}  {m['status']}"
        )
    print()
