"""
==============================================================================
sign_model/scripts/03_train_v2.py
==============================================================================
WHAT    : V2 Training — trains only the Transformer with improved settings:
          - Label smoothing (0.1)
          - Early stopping on val_loss (patience=20)
          - Higher dropout (0.45), lower LR (0.0008), higher WD (0.0002)
          - 100 epochs max
          - Reads from dataset_v2.npz
          - Saves to models/v2/

          V1 files are NEVER touched.

INPUT   : sign_model/data/processed/dataset_v2.npz
          sign_model/label_map.json
          sign_model/configs/config_v2.yaml

OUTPUT  : sign_model/models/v2/checkpoints/best_transformer_v2.pt
          sign_model/models/v2/final/model.pt
          sign_model/models/v2/final/model_config.json
          sign_model/models/v2/final/label_map.json
          sign_model/models/v2/transformer_v2_history.json
          sign_model/models/v2/v1_vs_v2_comparison.json

RUN     : cd sign_model && python scripts/03_train_v2.py
          cd sign_model && python scripts/03_train_v2.py --config configs/config_v2.yaml
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
import shutil
import time
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from utils.dataset import SignLandmarkDataset, load_processed_dataset

# ------------------------------------------------------------------------------
#  CLI
# ------------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="V2 Training — Transformer only")
    p.add_argument(
        "--config",
        default=os.path.join(MODULE_ROOT, "configs", "config_v2.yaml"),
        help="Path to config YAML (default: configs/config_v2.yaml)",
    )
    return p.parse_args()


# ------------------------------------------------------------------------------
#  Model Architecture (same as v1 Transformer, dropout from config)
# ------------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerClassifier(nn.Module):
    """
    Input:  (B, T, input_dim)
    Output: (B, num_classes)

    Architecture:
      Linear(input_dim → d_model) + PositionalEncoding
      TransformerEncoder(d_model, nhead=4, layers=num_layers, ff=256, dropout)
      Global avg pool → (B, d_model)
      Linear(d_model → 64) + ReLU + Dropout(0.3) + Linear(64 → num_classes)
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.45,
        num_classes: int = 69,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder    = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x).mean(dim=1)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------------------------------------------------------------
#  Training Loop
# ------------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, amp_scaler, use_amp, device, grad_clip):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    nan_batches = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(X_batch)
                loss   = criterion(logits, y_batch)
            if not torch.isfinite(loss):
                nan_batches += 1
                amp_scaler.update()
                continue
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            if not torch.isfinite(loss):
                nan_batches += 1
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(dim=1) == y_batch).sum().item()
        total      += len(y_batch)

    if nan_batches > 0:
        print(f"  [WARN] {nan_batches} batches skipped due to NaN loss")
    if total == 0:
        return float('nan'), 0.0
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        logits  = model(X_batch)
        loss    = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(dim=1) == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


def measure_inference_ms(model, sample_np, device, n_repeats=100):
    model.eval()
    x = torch.tensor(sample_np[None], dtype=torch.float32).to(device)
    with torch.no_grad():
        for _ in range(10):   # warm-up
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_repeats):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_repeats * 1000


# ------------------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 70)
    print("  Sign Model V2 — Step 03: Training (Transformer only)")
    print("=" * 70)
    print(f"  Config: {args.config}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Paths
    PROCESSED_DIR  = os.path.join(MODULE_ROOT, cfg["paths"]["processed_dir"])
    NPZ_FILENAME   = cfg["paths"]["processed_filename"]
    NPZ_PATH       = os.path.join(PROCESSED_DIR, NPZ_FILENAME)
    LABEL_MAP_PATH = os.path.join(MODULE_ROOT, cfg["paths"]["label_map"])
    CKPT_DIR       = os.path.join(MODULE_ROOT, cfg["paths"]["checkpoints_dir"])
    FINAL_DIR      = os.path.join(MODULE_ROOT, cfg["paths"]["final_model_dir"])
    V2_MODELS_DIR  = os.path.join(MODULE_ROOT, "models", "v2")

    V1_HISTORY_PATH = os.path.join(MODULE_ROOT, "models", "transformer_history.json")

    # Hyperparams
    SEED         = int(cfg["dataset"]["random_seed"])
    BATCH_SIZE   = int(cfg["training"]["batch_size"])
    EPOCHS       = int(cfg["training"]["epochs"])
    LR           = float(cfg["training"]["learning_rate"])
    WEIGHT_DECAY = float(cfg["training"]["weight_decay"])
    PATIENCE     = int(cfg["training"]["early_stopping_patience"])
    ES_METRIC    = cfg["training"].get("early_stopping_metric", "val_loss")
    LABEL_SMOOTH = float(cfg["training"].get("label_smoothing", 0.1))
    GRAD_CLIP    = float(cfg["training"].get("gradient_clip", 1.0))
    DROPOUT      = float(cfg["model"]["dropout"])
    HIDDEN_DIM   = int(cfg["model"]["hidden_dim"])
    NUM_LAYERS   = int(cfg["model"]["num_layers"])

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Safety check
    v1_npz = os.path.join(PROCESSED_DIR, "dataset.npz")
    if NPZ_PATH == v1_npz:
        print("[ERROR] NPZ path points to dataset.npz — refusing to use V1 data as V2.")
        sys.exit(1)

    # Load data
    print(f"\n  Loading {NPZ_FILENAME} ...")
    X_train, y_train, X_val, y_val, X_test, y_test, class_names = \
        load_processed_dataset(NPZ_PATH)
    num_classes = len(class_names)
    input_dim   = X_train.shape[2]

    print(f"  Classes: {num_classes}  |  Input dim: {input_dim}")
    print(f"  Train: {len(X_train)}  |  Val: {len(X_val)}  |  Test: {len(X_test)}")
    print(f"  Device: {DEVICE}")
    print(f"  Dropout: {DROPOUT}  |  LR: {LR}  |  Label smoothing: {LABEL_SMOOTH}")
    print(f"  Early stopping on: {ES_METRIC}  |  Patience: {PATIENCE}")

    # NaN check on loaded data
    nan_train = np.isnan(X_train).sum()
    nan_val   = np.isnan(X_val).sum()
    if nan_train > 0 or nan_val > 0:
        print(f"  [WARN] NaN values found: train={nan_train}, val={nan_val} — replacing with 0")
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_val   = np.nan_to_num(X_val,   nan=0.0)
    else:
        print(f"  Data check: no NaN values found -- OK")

    # Datasets
    train_ds = SignLandmarkDataset(X_train, y_train, class_names, augment=False)
    val_ds   = SignLandmarkDataset(X_val,   y_val,   class_names, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    # Class weights
    all_weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=y_train)
    cw_tensor   = torch.tensor(all_weights, dtype=torch.float32).to(DEVICE)

    # Build model
    model = TransformerClassifier(
        input_dim=input_dim,
        d_model=HIDDEN_DIM,
        nhead=4,
        num_layers=NUM_LAYERS,
        dim_feedforward=256,
        dropout=DROPOUT,
        num_classes=num_classes,
    ).to(DEVICE)

    params = count_parameters(model)
    print(f"\n  Model params: {params:,}")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(weight=cw_tensor, label_smoothing=LABEL_SMOOTH)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    use_amp    = DEVICE.type == "cuda"
    amp_scaler = torch.amp.GradScaler() if use_amp else None

    # Training loop
    os.makedirs(CKPT_DIR, exist_ok=True)
    best_ckpt = os.path.join(CKPT_DIR, "best_transformer_v2.pt")

    best_val_acc  = 0.0
    best_val_loss = float("inf")
    best_epoch    = 0
    no_improve    = 0
    history       = []

    print(f"\n  {'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>9} | {'Val Acc':>8} | {'LR':>8}")
    print("  " + "-" * 68)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, amp_scaler, use_amp, DEVICE, GRAD_CLIP
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 5),
            "train_acc":  round(train_acc,  5),
            "val_loss":   round(val_loss,   5),
            "val_acc":    round(val_acc,    5),
        })

        print(
            f"  {epoch:>5} | {train_loss:>10.4f} | {train_acc*100:>8.2f}% | "
            f"{val_loss:>9.4f} | {val_acc*100:>7.2f}% | {current_lr:.2e} | {elapsed:.1f}s"
        )

        # Early stopping on val_loss (V2) or val_acc (V1 behaviour)
        if ES_METRIC == "val_loss":
            improved = val_loss < best_val_loss
        else:
            improved = val_acc > best_val_acc

        if improved:
            best_val_acc  = val_acc
            best_val_loss = val_loss
            best_epoch    = epoch
            no_improve    = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs on {ES_METRIC})")
            break

    print(f"\n  Best -> epoch={best_epoch}  val_acc={best_val_acc*100:.2f}%  val_loss={best_val_loss:.5f}")

    # Save training history
    os.makedirs(V2_MODELS_DIR, exist_ok=True)
    hist_path = os.path.join(V2_MODELS_DIR, "transformer_v2_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Training history saved: {hist_path}")

    # Load best checkpoint (fallback: save current state if no checkpoint was created)
    if not os.path.isfile(best_ckpt):
        print("  [WARN] No best checkpoint found (all losses were NaN). Saving current state as fallback.")
        os.makedirs(CKPT_DIR, exist_ok=True)
        torch.save(model.state_dict(), best_ckpt)
    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE, weights_only=True))
    infer_ms = measure_inference_ms(model, X_val[0], DEVICE)
    print(f"  Avg inference time: {infer_ms:.3f} ms/sample")

    # Save final model
    os.makedirs(FINAL_DIR, exist_ok=True)
    final_model_path = os.path.join(FINAL_DIR, "model.pt")
    torch.save(model.state_dict(), final_model_path)
    model_size_mb = os.path.getsize(final_model_path) / (1024 * 1024)
    print(f"  Final model → {final_model_path}  ({model_size_mb:.2f} MB)")

    # Save model_config.json
    model_config = {
        "model_type":    "transformer",
        "version":       "v2",
        "input_dim":     int(input_dim),
        "num_classes":   int(num_classes),
        "class_names":   class_names,
        "hidden_dim":    HIDDEN_DIM,
        "num_layers":    NUM_LAYERS,
        "dropout":       DROPOUT,
        "val_acc":       round(best_val_acc,  5),
        "val_loss":      round(best_val_loss, 5),
        "best_epoch":    best_epoch,
        "params":        params,
        "infer_ms":      round(infer_ms, 3),
        "label_smoothing": LABEL_SMOOTH,
        "train_size":    int(len(X_train)),
    }
    cfg_out = os.path.join(FINAL_DIR, "model_config.json")
    with open(cfg_out, "w") as f:
        json.dump(model_config, f, indent=2)
    print(f"  model_config.json → {cfg_out}")

    # Copy label_map
    shutil.copy(LABEL_MAP_PATH, os.path.join(FINAL_DIR, "label_map.json"))
    print(f"  label_map.json → {FINAL_DIR}")

    # -- V1 vs V2 Comparison --------------------------------------------------
    print("\n" + "=" * 70)
    print("  ===== V1 vs V2 TRAINING COMPARISON =====")
    print("=" * 70)

    v1_best_val_acc  = 0.59091
    v1_best_val_loss = 3.45702
    v1_best_epoch    = 35
    v1_train_size    = 8013

    if os.path.isfile(V1_HISTORY_PATH):
        with open(V1_HISTORY_PATH) as f:
            v1_history = json.load(f)
        v1_best = max(v1_history, key=lambda e: e["val_acc"])
        v1_best_val_acc  = v1_best["val_acc"]
        v1_best_epoch    = v1_best["epoch"]
        v1_best_vl = min(v1_history, key=lambda e: e["val_loss"])
        v1_best_val_loss = v1_best_vl["val_loss"]

    print(f"  {'Metric':<25} {'V1':>12} {'V2':>12}  {'Change':>10}")
    print("  " + "-" * 65)
    print(f"  {'Val Accuracy':<25} {v1_best_val_acc*100:>11.2f}% {best_val_acc*100:>11.2f}%  {(best_val_acc - v1_best_val_acc)*100:>+9.2f}pp")
    print(f"  {'Val Loss':<25} {v1_best_val_loss:>12.5f} {best_val_loss:>12.5f}  {(best_val_loss - v1_best_val_loss):>+10.5f}")
    print(f"  {'Best Epoch':<25} {v1_best_epoch:>12} {best_epoch:>12}")
    print(f"  {'Training Samples':<25} {v1_train_size:>12,} {len(X_train):>12,}  {(len(X_train) - v1_train_size):>+10,}")
    print(f"  {'Dropout':<25} {'0.30':>12} {DROPOUT:>12.2f}")
    print(f"  {'Label Smoothing':<25} {'0.00':>12} {LABEL_SMOOTH:>12.2f}")
    print(f"  {'Early Stop Metric':<25} {'val_acc':>12} {ES_METRIC:>12}")
    print(f"  {'Inference (ms)':<25} {'1.022':>12} {infer_ms:>12.3f}")

    # Save comparison JSON
    comparison = {
        "v1": {
            "val_acc":       v1_best_val_acc,
            "val_loss":      v1_best_val_loss,
            "best_epoch":    v1_best_epoch,
            "train_size":    v1_train_size,
            "dropout":       0.30,
            "label_smoothing": 0.00,
            "early_stop_metric": "val_acc",
            "infer_ms":      1.022,
        },
        "v2": {
            "val_acc":       round(best_val_acc,  5),
            "val_loss":      round(best_val_loss, 5),
            "best_epoch":    best_epoch,
            "train_size":    int(len(X_train)),
            "dropout":       DROPOUT,
            "label_smoothing": LABEL_SMOOTH,
            "early_stop_metric": ES_METRIC,
            "infer_ms":      round(infer_ms, 3),
        },
        "delta": {
            "val_acc_pp":    round((best_val_acc - v1_best_val_acc) * 100, 3),
            "val_loss":      round(best_val_loss - v1_best_val_loss, 5),
            "train_size":    int(len(X_train)) - v1_train_size,
        },
    }
    comp_out = os.path.join(V2_MODELS_DIR, "v1_vs_v2_comparison.json")
    with open(comp_out, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  Comparison saved → {comp_out}")
    print("\n  V2 Training complete ✅")


if __name__ == "__main__":
    main()
