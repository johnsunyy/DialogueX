"""
evaluate.py — Phase 7: Model Evaluation on Test Set

Loads checkpoints/best_model.pt and evaluates on test split.
Reports: Accuracy, Precision, Recall, F1, Top-3 Accuracy, Per-class accuracy.
Saves: confusion_matrix_test.png, classification_report_test.txt

Run: python evaluate.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

from model.sign_model import SignModel
from train import SignDataset, top3_accuracy, evaluate, CFG, SPLITS_DIR, CKPT_DIR
WLASL_JSON = MODULE_DIR.parent / "WLASL_v0.3.json"
NSLT_JSON  = MODULE_DIR.parent / "nslt_100.json"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("  NSLT-300 Model Evaluation — Test Set")
    print("=" * 60)
    print(f"  Device: {device}")

    # ── Load model ────────────────────────────────────────────
    ckpt_path = CKPT_DIR / "best_model.pt"
    if not ckpt_path.exists():
        print("  ERROR: No checkpoint found. Run train.py first.")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt.get("cfg", CFG)

    model = SignModel(
        num_classes = cfg["num_classes"],
        seq_len     = cfg["seq_len"],
        feat_dim    = cfg["feat_dim"],
        lstm_hidden = cfg["lstm_hidden"],
        dropout1    = cfg["dropout1"],
        dropout2    = cfg["dropout2"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Loaded checkpoint from epoch {ckpt['epoch']} (val_f1={ckpt['val_f1']:.4f})")

    # ── Load metadata ─────────────────────────────────────────
    with open(WLASL_JSON, "r") as f:
        wlasl = json.load(f)
    with open(NSLT_JSON, "r") as f:
        nslt = json.load(f)
    idx2gloss  = {i: entry["gloss"] for i, entry in enumerate(wlasl)}
    label_names = [idx2gloss.get(i, str(i)) for i in range(cfg["num_classes"])]

    # ── Test Dataset ──────────────────────────────────────────
    test_csv = SPLITS_DIR / "test.csv"
    if not test_csv.exists():
        print("  ERROR: test.csv not found. Run prepare_splits.py first.")
        sys.exit(1)

    test_ds = SignDataset(test_csv, nslt, idx2gloss, augment=False)
    test_loader = DataLoader(
        test_ds, batch_size=cfg.get("batch_size", 32) * 2,
        shuffle=False, num_workers=4, pin_memory=True,
    )
    print(f"  Test samples: {len(test_ds)}")

    # ── Evaluate ──────────────────────────────────────────────
    metrics = evaluate(model, test_loader, device)
    preds, targets = metrics["preds"], metrics["targets"]

    # ── Global metrics ────────────────────────────────────────
    print("\n  ── Global Metrics ──────────────────────────────────")
    print(f"  Accuracy   : {metrics['acc']:.4f}  ({metrics['acc']*100:.2f}%)")
    print(f"  Precision  : {metrics['precision']:.4f}")
    print(f"  Recall     : {metrics['recall']:.4f}")
    print(f"  F1 (macro) : {metrics['f1']:.4f}")
    print(f"  Top-3 Acc  : {metrics['top3']:.4f}  ({metrics['top3']*100:.2f}%)")

    # ── Per-class accuracy ────────────────────────────────────
    print("\n  ── Per-Class Accuracy ──────────────────────────────")
    from collections import defaultdict
    per_class_correct = defaultdict(int)
    per_class_total   = defaultdict(int)
    for t, p in zip(targets, preds):
        per_class_total[t]   += 1
        per_class_correct[t] += int(t == p)

    rows = []
    for cls_idx in sorted(per_class_total):
        total   = per_class_total[cls_idx]
        correct = per_class_correct[cls_idx]
        acc_cls = correct / total if total > 0 else 0.0
        rows.append((cls_idx, idx2gloss.get(cls_idx, str(cls_idx)), correct, total, acc_cls))

    rows.sort(key=lambda r: r[4])  # sort by accuracy ascending
    print(f"  {'Rank':>4}  {'Idx':>3}  {'Word':<22}  {'Correct':>7}  {'Total':>5}  {'Acc':>6}")
    print("  " + "-" * 58)
    # show bottom 10
    for rank, (idx, word, correct, total, acc_cls) in enumerate(rows[:10], 1):
        print(f"  {rank:4d}  {idx:3d}  {word:<22}  {correct:7d}  {total:5d}  {acc_cls:6.2%}")
    print("  ...")
    # show top 10
    for rank, (idx, word, correct, total, acc_cls) in enumerate(rows[-10:], len(rows)-9):
        print(f"  {rank:4d}  {idx:3d}  {word:<22}  {correct:7d}  {total:5d}  {acc_cls:6.2%}")

    # ── Confusion matrix ──────────────────────────────────────
    cm = confusion_matrix(targets, preds, labels=list(range(cfg["num_classes"])))
    fig, ax = plt.subplots(figsize=(24, 22))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix — Test Set", fontsize=14)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    cm_path = CKPT_DIR / "confusion_matrix_test.png"
    plt.savefig(str(cm_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Confusion matrix → {cm_path.name}")

    # ── Classification report ─────────────────────────────────
    report = classification_report(targets, preds, target_names=label_names, zero_division=0)
    rpt_path = CKPT_DIR / "classification_report_test.txt"
    with open(rpt_path, "w") as f:
        f.write(report)
    print(f"  Classification report → {rpt_path.name}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
