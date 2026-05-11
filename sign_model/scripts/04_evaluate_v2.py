"""
==============================================================================
sign_model/scripts/04_evaluate_v2.py
==============================================================================
WHAT    : Evaluates the V2 model on the held-out test set.
          Prints full V1 vs V2 per-class comparison table.
          V1 files are NEVER touched.

INPUT   : sign_model/data/processed/dataset_v2.npz
          sign_model/models/v2/final/model.pt
          sign_model/models/v2/final/model_config.json
          sign_model/models/final/per_class_metrics.json  (V1, read-only)

OUTPUT  : sign_model/models/v2/final/per_class_metrics.json
          sign_model/models/v2/final/confusion_matrix.npy

RUN     : cd sign_model && python scripts/04_evaluate_v2.py
          cd sign_model && python scripts/04_evaluate_v2.py --config configs/config_v2.yaml
==============================================================================
"""

import sys
import os
import argparse

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODULE_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, MODULE_ROOT)

import json
import math
import time
import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

from utils.dataset import SignLandmarkDataset, load_processed_dataset
from utils.metrics import top_k_accuracy, per_class_report, confusion_pairs, measure_inference_ms

# ------------------------------------------------------------------------------
#  CLI
# ------------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="V2 Evaluation")
    p.add_argument(
        "--config",
        default=os.path.join(MODULE_ROOT, "configs", "config_v2.yaml"),
        help="Path to config YAML (default: configs/config_v2.yaml)",
    )
    return p.parse_args()


# ------------------------------------------------------------------------------
#  Model rebuild (mirrors 03_train_v2.py architecture)
# ------------------------------------------------------------------------------

def build_model(model_cfg: dict) -> nn.Module:

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=512, dropout=0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            pe  = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))
        def forward(self, x):
            return self.dropout(x + self.pe[:, :x.size(1)])

    class TransformerClassifier(nn.Module):
        def __init__(self, input_dim, d_model=128, nhead=4, num_layers=3,
                     dim_feedforward=256, dropout=0.45, num_classes=69):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, d_model)
            self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)
            enc_layer       = nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout, batch_first=True
            )
            self.encoder    = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            self.classifier = nn.Sequential(
                nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(64, num_classes),
            )
        def forward(self, x):
            x = self.input_proj(x)
            x = self.pos_enc(x)
            return self.classifier(self.encoder(x).mean(dim=1))

    return TransformerClassifier(
        input_dim=model_cfg["input_dim"],
        d_model=model_cfg["hidden_dim"],
        nhead=4,
        num_layers=model_cfg["num_layers"],
        dim_feedforward=256,
        dropout=model_cfg["dropout"],
        num_classes=model_cfg["num_classes"],
    )


# ------------------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 70)
    print("  Sign Model V2 — Step 04: Evaluation")
    print("=" * 70)
    print(f"  Config: {args.config}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    PROCESSED_DIR   = os.path.join(MODULE_ROOT, cfg["paths"]["processed_dir"])
    NPZ_PATH        = os.path.join(PROCESSED_DIR, cfg["paths"]["processed_filename"])
    V2_FINAL_DIR    = os.path.join(MODULE_ROOT, cfg["paths"]["final_model_dir"])
    MODEL_PATH      = os.path.join(V2_FINAL_DIR, "model.pt")
    MODEL_CFG_PATH  = os.path.join(V2_FINAL_DIR, "model_config.json")
    METRICS_OUT     = os.path.join(V2_FINAL_DIR, "per_class_metrics.json")
    CONFMAT_OUT     = os.path.join(V2_FINAL_DIR, "confusion_matrix.npy")

    # V1 reference files (read-only)
    V1_METRICS_PATH = os.path.join(MODULE_ROOT, "models", "final", "per_class_metrics.json")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model config
    if not os.path.isfile(MODEL_CFG_PATH):
        print(f"[ERROR] model_config.json not found: {MODEL_CFG_PATH}")
        print("  → Run: python scripts/03_train_v2.py first")
        sys.exit(1)

    with open(MODEL_CFG_PATH) as f:
        model_cfg = json.load(f)

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, class_names = \
        load_processed_dataset(NPZ_PATH)
    print(f"  Test set: {len(X_test)} samples, {len(class_names)} classes")

    # Build and load model
    model = build_model(model_cfg).to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"  V2 model loaded from {MODEL_PATH}")

    # Inference
    test_ds     = SignLandmarkDataset(X_test, y_test, class_names, augment=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    all_logits, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            logits  = model(X_batch)
            preds   = logits.argmax(dim=1).cpu().numpy()
            all_logits.append(logits.cpu().numpy())
            all_preds.extend(preds.tolist())
            all_labels.extend(y_batch.numpy().tolist())

    y_true  = np.array(all_labels)
    y_pred  = np.array(all_preds)
    y_probs = np.concatenate(all_logits, axis=0)

    # -- Overall metrics -------------------------------------------------------
    acc1   = accuracy_score(y_true, y_pred)
    top3   = top_k_accuracy(y_true, y_probs, k=3)
    f1_mac = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_wgt = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec   = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec    = recall_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n-- Overall V2 Metrics ----------------------------------------------")
    print(f"  Top-1 Accuracy : {acc1*100:.2f}%")
    print(f"  Top-3 Accuracy : {top3*100:.2f}%")
    print(f"  Macro F1       : {f1_mac:.4f}")
    print(f"  Weighted F1    : {f1_wgt:.4f}")
    print(f"  Macro Precision: {prec:.4f}")
    print(f"  Macro Recall   : {rec:.4f}")

    # -- Per-class metrics -----------------------------------------------------
    class_metrics = per_class_report(y_true, y_pred, class_names)

    # Save
    os.makedirs(V2_FINAL_DIR, exist_ok=True)
    with open(METRICS_OUT, "w") as f:
        json.dump(class_metrics, f, indent=2)
    print(f"  Per-class metrics → {METRICS_OUT}")

    # -- Confusion matrix ------------------------------------------------------
    cm = confusion_matrix(y_true, y_pred)
    np.save(CONFMAT_OUT, cm)
    print(f"  Confusion matrix  → {CONFMAT_OUT}")

    # -- Top confused pairs ----------------------------------------------------
    try:
        pairs = confusion_pairs(y_true, y_pred, class_names, n=10)
        print(f"\n-- Top 10 Most Confused Pairs (V2) ---------------------------------")
        for true_cls, pred_cls, count in pairs:
            print(f"  {true_cls:<20} → {pred_cls:<20} | {count}")
    except Exception:
        pass

    # -- Inference time --------------------------------------------------------
    infer_ms = measure_inference_ms(model, X_test[0], DEVICE, n_repeats=100)
    print(f"\n  Avg inference time: {infer_ms:.3f} ms/sample")

    # -- Pass / Warn / Fail summary --------------------------------------------
    passed = sum(1 for m in class_metrics.values() if "PASS" in m["status"])
    warned = sum(1 for m in class_metrics.values() if "WARN" in m["status"])
    failed = sum(1 for m in class_metrics.values() if "FAIL" in m["status"])
    print(f"\n-- V2 Pass/Warn/Fail Summary ---------------------------------------")
    print(f"  PASS ✅ : {passed} classes  (F1 ≥ 0.70)")
    print(f"  WARN ⚠️  : {warned} classes  (F1 0.50–0.70)")
    print(f"  FAIL ❌ : {failed} classes  (F1 < 0.50)")

    # -- V1 vs V2 Per-Class Comparison -----------------------------------------
    print(f"\n{'='*70}")
    print("  V1 vs V2 — Full Per-Class F1 Comparison")
    print(f"{'='*70}")

    v1_metrics = {}
    if os.path.isfile(V1_METRICS_PATH):
        with open(V1_METRICS_PATH) as f:
            v1_metrics = json.load(f)
    else:
        print(f"  [WARN] V1 per_class_metrics.json not found at {V1_METRICS_PATH}")

    improved_count  = 0
    degraded_count  = 0
    same_count      = 0
    new_pass_count  = 0   # was FAIL/WARN in v1, now PASS in v2

    print(f"\n  {'Class':<15} {'V1 F1':>7} {'V2 F1':>7} {'Change':>8}  {'V1 Status':<12} {'V2 Status'}")
    print("  " + "-" * 72)

    for cls in class_names:
        v2_m    = class_metrics.get(cls, {})
        v2_f1   = v2_m.get("f1", 0.0)
        v2_stat = v2_m.get("status", "?")

        if cls in v1_metrics:
            v1_f1   = v1_metrics[cls]["f1"]
            v1_stat = v1_metrics[cls]["status"]
            delta   = v2_f1 - v1_f1
            delta_str = f"{delta:+.4f}"

            if delta > 0.01:
                improved_count += 1
                marker = "↑"
            elif delta < -0.01:
                degraded_count += 1
                marker = "↓"
            else:
                same_count += 1
                marker = "="

            was_not_pass = "PASS" not in v1_stat
            now_pass     = "PASS" in v2_stat
            if was_not_pass and now_pass:
                new_pass_count += 1
                marker += "🆕"
        else:
            v1_f1     = 0.0
            v1_stat   = "N/A"
            delta_str = "N/A"
            marker    = ""

        print(f"  {cls:<15} {v1_f1:>7.4f} {v2_f1:>7.4f} {delta_str:>8}  {v1_stat:<12} {v2_stat}  {marker}")

    # -- Overall summary -------------------------------------------------------
    print(f"\n{'='*70}")
    print("  OVERALL SUMMARY")
    print(f"{'='*70}")

    v1_acc  = 0.5819
    v1_f1   = 0.5992
    v1_pass = 23
    v1_warn = 27
    v1_fail = 19
    v1_wf1  = 0.5708

    if v1_metrics:
        v1_f1s  = [v1_metrics[c]["f1"] for c in class_names if c in v1_metrics]
        v1_f1   = sum(v1_f1s) / len(v1_f1s) if v1_f1s else 0.0
        v1_pass = sum(1 for c in class_names if c in v1_metrics and v1_metrics[c]["f1"] >= 0.70)
        v1_warn = sum(1 for c in class_names if c in v1_metrics and 0.50 <= v1_metrics[c]["f1"] < 0.70)
        v1_fail = sum(1 for c in class_names if c in v1_metrics and v1_metrics[c]["f1"] < 0.50)

    print(f"\n  {'Metric':<28} {'V1':>10} {'V2':>10}  Change")
    print("  " + "-" * 60)
    print(f"  {'Test Accuracy':<28} {v1_acc*100:>9.2f}% {acc1*100:>9.2f}%  {(acc1-v1_acc)*100:>+.2f}pp")
    print(f"  {'Macro F1':<28} {v1_f1:>10.4f} {f1_mac:>10.4f}  {f1_mac-v1_f1:>+.4f}")
    print(f"  {'Weighted F1':<28} {v1_wf1:>10.4f} {f1_wgt:>10.4f}  {f1_wgt-v1_wf1:>+.4f}")
    print(f"  {'PASS classes (F1≥0.70)':<28} {v1_pass:>10} {passed:>10}  {passed-v1_pass:>+d}")
    print(f"  {'WARN classes':<28} {v1_warn:>10} {warned:>10}  {warned-v1_warn:>+d}")
    print(f"  {'FAIL classes':<28} {v1_fail:>10} {failed:>10}  {failed-v1_fail:>+d}")
    print(f"\n  Classes improved  (ΔF1 > +0.01): {improved_count}")
    print(f"  Classes degraded  (ΔF1 < -0.01): {degraded_count}")
    print(f"  Classes unchanged             : {same_count}")
    print(f"  New PASS classes (was FAIL/WARN, now PASS): {new_pass_count}")
    print(f"\n  V2 Evaluation complete ✅")


if __name__ == "__main__":
    main()
