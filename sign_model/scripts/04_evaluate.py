"""
==============================================================================
sign_model/scripts/04_evaluate.py
==============================================================================
WHAT    : Loads the final saved model, runs evaluation on the held-out test
          set, and produces per-class metrics, confusion matrix, and a
          pass/warn/fail summary.

INPUT   : sign_model/data/processed/dataset.npz
          sign_model/models/final/model.pt
          sign_model/models/final/model_config.json

OUTPUT  : sign_model/models/final/per_class_metrics.json
          sign_model/models/final/confusion_matrix.npy
          Console report (accuracy, top-3, F1, confused pairs)

RUN     : cd sign_model && python scripts/04_evaluate.py
==============================================================================
"""

import sys
import os

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODULE_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, MODULE_ROOT)

import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from utils.dataset import SignLandmarkDataset, load_processed_dataset
from utils.metrics import (
    top_k_accuracy,
    per_class_report,
    confusion_pairs,
    measure_inference_ms,
    print_confusion_pairs,
    print_per_class_report,
)

# ──────────────────────────────────────────────────────────────────────────────
#  Paths
# ──────────────────────────────────────────────────────────────────────────────

FINAL_DIR       = os.path.join(MODULE_ROOT, "models", "final")
MODEL_PATH      = os.path.join(FINAL_DIR, "model.pt")
MODEL_CFG_PATH  = os.path.join(FINAL_DIR, "model_config.json")
NPZ_PATH        = os.path.join(MODULE_ROOT, "data", "processed", "dataset.npz")
METRICS_OUT     = os.path.join(FINAL_DIR, "per_class_metrics.json")
CONFMAT_OUT     = os.path.join(FINAL_DIR, "confusion_matrix.npy")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────────────────────
#  Rebuild model from config
# ──────────────────────────────────────────────────────────────────────────────

def build_model(model_cfg: dict) -> nn.Module:
    """
    Re-instantiate the correct model architecture using saved config.
    The classes are defined inline here to keep the script self-contained.
    """
    import math

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=512, dropout=0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))
        def forward(self, x):
            return self.dropout(x + self.pe[:, :x.size(1)])

    class LSTMClassifier(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, dropout=0.3):
            super().__init__()
            self.input_proj = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(dropout))
            self.lstm = nn.LSTM(128, 256, num_layers=num_layers, dropout=dropout if num_layers>1 else 0.0, batch_first=True)
            self.classifier = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, num_classes))
        def forward(self, x):
            x = self.input_proj(x)
            _, (h, _) = self.lstm(x)
            return self.classifier(h[-1])

    class TransformerClassifier(nn.Module):
        def __init__(self, input_dim, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.2, num_classes=70):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, d_model)
            self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)
            enc_layer       = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
            self.encoder    = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            self.classifier = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_classes))
        def forward(self, x):
            x = self.input_proj(x)
            x = self.pos_enc(x)
            x = self.encoder(x).mean(dim=1)
            return self.classifier(x)

    mtype = model_cfg["model_type"]
    num_c = model_cfg["num_classes"]
    idim  = model_cfg["input_dim"]
    hdim  = model_cfg["hidden_dim"]
    nlyr  = model_cfg["num_layers"]
    drop  = model_cfg["dropout"]

    if mtype == "lstm":
        return LSTMClassifier(idim, hdim, num_c, num_layers=nlyr, dropout=drop)
    elif mtype == "transformer":
        return TransformerClassifier(idim, hdim, 4, nlyr, 256, 0.2, num_c)
    else:
        raise ValueError(f"Unknown model_type: {mtype}")


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Sign Model — Step 04: Evaluation")
    print("=" * 70)

    # Load model config
    try:
        with open(MODEL_CFG_PATH) as f:
            model_cfg = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] model_config.json not found: {MODEL_CFG_PATH}")
        print("  → Run: python scripts/03_train.py first")
        sys.exit(1)

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, class_names = \
        load_processed_dataset(NPZ_PATH)

    print(f"  Test set: {len(X_test)} samples, {len(class_names)} classes")

    # Build and load model
    model = build_model(model_cfg).to(DEVICE)
    try:
        state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
    except FileNotFoundError:
        print(f"[ERROR] Model weights not found: {MODEL_PATH}")
        sys.exit(1)
    model.eval()

    # Dataloader
    test_ds     = SignLandmarkDataset(X_test, y_test, class_names, augment=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    # Inference — collect logits and predictions
    all_logits, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            logits  = model(X_batch)
            preds   = logits.argmax(dim=1).cpu().numpy()
            all_logits.append(logits.cpu().numpy())
            all_preds.extend(preds.tolist())
            all_labels.extend(y_batch.numpy().tolist())

    y_true      = np.array(all_labels)
    y_pred      = np.array(all_preds)
    y_probs     = np.concatenate(all_logits, axis=0)

    # ── Metrics ──────────────────────────────────────────────────────────
    acc1   = accuracy_score(y_true, y_pred)
    top3   = top_k_accuracy(y_true, y_probs, k=3)
    f1_mac = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_wgt = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n── Overall Metrics ─────────────────────────────────────────────")
    print(f"  Top-1 Accuracy  : {acc1*100:.2f}%")
    print(f"  Top-3 Accuracy  : {top3*100:.2f}%")
    print(f"  Macro F1        : {f1_mac:.4f}")
    print(f"  Weighted F1     : {f1_wgt:.4f}")

    # ── Per-class report ─────────────────────────────────────────────────
    class_metrics = per_class_report(y_true, y_pred, class_names)
    print_per_class_report(class_metrics)

    # Save
    try:
        with open(METRICS_OUT, "w") as f:
            json.dump(class_metrics, f, indent=2)
        print(f"  Per-class metrics → {METRICS_OUT}")
    except Exception as e:
        print(f"  [WARN] Could not save per_class_metrics.json: {e}")

    # ── Confusion matrix ─────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    try:
        np.save(CONFMAT_OUT, cm)
        print(f"  Confusion matrix  → {CONFMAT_OUT}")
    except Exception as e:
        print(f"  [WARN] Could not save confusion_matrix.npy: {e}")

    # ── Top confused pairs ───────────────────────────────────────────────
    pairs = confusion_pairs(y_true, y_pred, class_names, n=10)
    print_confusion_pairs(pairs)

    # ── Inference time ───────────────────────────────────────────────────
    infer_ms = measure_inference_ms(model, X_test[0], DEVICE, n_repeats=100)
    print(f"  Avg inference time: {infer_ms:.2f} ms/sample")

    # ── Pass/Warn/Fail summary ───────────────────────────────────────────
    passed = sum(1 for m in class_metrics.values() if "PASS" in m["status"])
    warned = sum(1 for m in class_metrics.values() if "WARN" in m["status"])
    failed = sum(1 for m in class_metrics.values() if "FAIL" in m["status"])

    print(f"\n── Pass/Warn/Fail Summary ──────────────────────────────────────")
    print(f"  PASS ✅ : {passed} classes  (F1 ≥ 0.70)")
    print(f"  WARN ⚠️  : {warned} classes  (F1 0.50–0.70)")
    print(f"  FAIL ❌ : {failed} classes  (F1 < 0.50)")
    print(f"\n  Evaluation complete ✅")


if __name__ == "__main__":
    main()
