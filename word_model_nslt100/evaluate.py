"""
evaluate.py — Phase 7: Test Set Evaluation

Run: python evaluate.py
"""

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

from model.sign_model import SignModel
from train import SignDataset, evaluate, top3_accuracy, CFG, SPLITS_DIR, CKPT_DIR

WLASL_JSON = MODULE_DIR.parent / "WLASL_v0.3.json"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("  NSLT-100 Evaluation — Test Set")
    print(f"  Device: {device}")
    print("=" * 60)

    ckpt_path = CKPT_DIR / "best_model.pt"
    if not ckpt_path.exists():
        print("ERROR: Run train.py first."); sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt.get("cfg", CFG)

    model = SignModel(
        num_classes  = cfg["num_classes"],
        seq_len      = cfg["seq_len"],
        feat_dim     = cfg["feat_dim"],
        lstm1_hidden = cfg["lstm1_hidden"],
        lstm2_hidden = cfg["lstm2_hidden"],
        dropout1     = cfg["dropout1"],
        dropout2     = cfg["dropout2"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Loaded checkpoint: epoch {ckpt['epoch']} | val_f1={ckpt['val_f1']:.4f}")

    with open(WLASL_JSON) as f:
        wlasl = json.load(f)
    idx2gloss   = {i: e["gloss"] for i, e in enumerate(wlasl)}
    label_names = [idx2gloss.get(i, str(i)) for i in range(cfg["num_classes"])]

    test_csv = SPLITS_DIR / "test.csv"
    if not test_csv.exists():
        print("ERROR: test.csv not found."); sys.exit(1)

    test_ds     = SignDataset(test_csv, augment=False)
    test_loader = DataLoader(test_ds, cfg.get("batch_size",32)*2, shuffle=False,
                             num_workers=4, pin_memory=True)
    print(f"  Test samples: {len(test_ds)}")

    m = evaluate(model, test_loader, device)
    preds, targets = m["preds"], m["targets"]

    print("\n  ── Global Metrics ─────────────────────────────────")
    print(f"  Accuracy   : {m['acc']:.4f}  ({m['acc']*100:.2f}%)")
    print(f"  Precision  : {m['prec']:.4f}")
    print(f"  Recall     : {m['rec']:.4f}")
    print(f"  F1 (macro) : {m['f1']:.4f}")
    print(f"  Top-3 Acc  : {m['top3']:.4f}  ({m['top3']*100:.2f}%)")

    # Per-class accuracy
    per_c_correct = defaultdict(int)
    per_c_total   = defaultdict(int)
    for t, p in zip(targets, preds):
        per_c_total[t]   += 1
        per_c_correct[t] += int(t == p)

    rows = [(idx, idx2gloss.get(idx, str(idx)),
             per_c_correct[idx], per_c_total[idx],
             per_c_correct[idx]/per_c_total[idx] if per_c_total[idx] else 0)
            for idx in sorted(per_c_total)]
    rows.sort(key=lambda r: r[4])

    print("\n  ── Per-Class Accuracy (worst 10 / best 10) ────────")
    print(f"  {'Idx':>3}  {'Word':<20}  {'Correct':>7}  {'Total':>5}  {'Acc':>6}")
    print("  " + "-" * 48)
    for idx, word, correct, total, acc in rows[:10]:
        print(f"  {idx:3d}  {word:<20}  {correct:7d}  {total:5d}  {acc:6.2%}")
    print("  ...")
    for idx, word, correct, total, acc in rows[-10:]:
        print(f"  {idx:3d}  {word:<20}  {correct:7d}  {total:5d}  {acc:6.2%}")

    # Save artifacts
    cm = confusion_matrix(targets, preds, labels=list(range(cfg["num_classes"])))
    fig, ax = plt.subplots(figsize=(18, 16))
    ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_title("Confusion Matrix — Test Set")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    cm_path = CKPT_DIR / "confusion_matrix_test.png"
    plt.savefig(str(cm_path), dpi=150); plt.close()
    print(f"\n  → {cm_path.name}")

    report = classification_report(targets, preds, target_names=label_names, zero_division=0)
    rpt_path = CKPT_DIR / "classification_report_test.txt"
    with open(rpt_path, "w") as f:
        f.write(report)
    print(f"  → {rpt_path.name}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
