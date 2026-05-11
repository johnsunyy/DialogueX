"""
==============================================================================
sign_model/scripts/03_train.py
==============================================================================
WHAT    : Trains TWO models (LSTM + Transformer), compares them, and saves
          the best one as the final model.

INPUT   : sign_model/data/processed/dataset.npz
          sign_model/label_map.json
          sign_model/configs/config.yaml

OUTPUT  : sign_model/models/checkpoints/best_lstm.pt
          sign_model/models/checkpoints/best_transformer.pt
          sign_model/models/{lstm,transformer}_history.json
          sign_model/models/final/model.pt
          sign_model/models/final/model_config.json
          sign_model/models/final/label_map.json  (copy)

RUN     : cd sign_model && python scripts/03_train.py
==============================================================================
"""

import sys
import os

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODULE_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, MODULE_ROOT)

import json
import math
import time
import shutil
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from utils.dataset import SignLandmarkDataset, load_processed_dataset

# ──────────────────────────────────────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────────────────────────────────────

CONFIG_PATH = os.path.join(MODULE_ROOT, "configs", "config.yaml")
try:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
except FileNotFoundError:
    print(f"[ERROR] Config not found: {CONFIG_PATH}")
    sys.exit(1)

LABEL_MAP_PATH = os.path.join(MODULE_ROOT, cfg["paths"]["label_map"])
NPZ_PATH       = os.path.join(MODULE_ROOT, cfg["paths"]["processed_dir"], "dataset.npz")
CKPT_DIR       = os.path.join(MODULE_ROOT, cfg["paths"]["checkpoints_dir"])
FINAL_DIR      = os.path.join(MODULE_ROOT, cfg["paths"]["final_model_dir"])
MODELS_DIR     = os.path.join(MODULE_ROOT, "models")

SEED           = cfg["dataset"]["random_seed"]
BATCH_SIZE     = cfg["training"]["batch_size"]
EPOCHS         = cfg["training"]["epochs"]
LR             = cfg["training"]["learning_rate"]
WEIGHT_DECAY   = cfg["training"]["weight_decay"]
PATIENCE       = cfg["training"]["early_stopping_patience"]

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────────────────────
#  Model Definitions
# ──────────────────────────────────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    """
    Input: (batch, T, input_dim)
    Architecture:
      Linear(input_dim → 128) + ReLU + Dropout(0.3)
      LSTM(128, hidden=256, layers=2, dropout=0.3, batch_first)
      take last hidden state → (batch, 256)
      Linear(256→128) + ReLU + Dropout(0.3)
      Linear(128 → num_classes)
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):                       # x: (B, T, F)
        x = self.input_proj(x)                  # (B, T, 128)
        _, (h_n, _) = self.lstm(x)              # h_n: (num_layers, B, 256)
        h = h_n[-1]                             # last layer: (B, 256)
        return self.classifier(h)               # (B, num_classes)


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
        pe = pe.unsqueeze(0)           # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):              # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """
    Input: (batch, T, input_dim)
    Architecture:
      Linear(input_dim → 128) + PositionalEncoding
      TransformerEncoder(d_model=128, nhead=4, layers=3, ff=256, dropout=0.2)
      Global avg pool over T → (batch, 128)
      Linear(128→64) + ReLU + Dropout(0.3)
      Linear(64 → num_classes)
    """
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 3, dim_feedforward: int = 256,
                 dropout: float = 0.2, num_classes: int = 70):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):                        # (B, T, F)
        x = self.input_proj(x)                   # (B, T, 128)
        x = self.pos_enc(x)
        x = self.encoder(x)                      # (B, T, 128)
        x = x.mean(dim=1)                        # global avg pool → (B, 128)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────────────────────
#  Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler_amp, use_amp):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(X_batch)
                loss   = criterion(logits, y_batch)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds       = logits.argmax(dim=1)
        correct    += (preds == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        logits  = model(X_batch)
        loss    = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        preds       = logits.argmax(dim=1)
        correct    += (preds == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


def run_training(model, model_name, train_loader, val_loader, class_weights_tensor):
    """Full training loop with early stopping. Returns best val_acc and history."""

    os.makedirs(CKPT_DIR, exist_ok=True)
    best_ckpt = os.path.join(CKPT_DIR, f"best_{model_name}.pt")

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(DEVICE))
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    use_amp   = DEVICE.type == "cuda"
    amp_scaler = torch.amp.GradScaler() if use_amp else None

    best_val_acc  = 0.0
    best_val_loss = float("inf")
    no_improve    = 0
    history       = []

    print(f"\n  Training {model_name} on {DEVICE} | params={count_parameters(model):,}")
    print(f"  {'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>9} | {'Val Acc':>8}")
    print("  " + "-" * 58)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, amp_scaler, use_amp)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion)
        scheduler.step()
        elapsed = time.time() - t0

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_acc":  round(train_acc,  5),
            "val_loss":   round(val_loss,   5),
            "val_acc":    round(val_acc,    5),
        })

        print(
            f"  {epoch:>5} | {train_loss:>10.4f} | {train_acc*100:>8.2f}% | "
            f"{val_loss:>9.4f} | {val_acc*100:>7.2f}% | {elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_val_loss = val_loss
            no_improve    = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    # Save history
    hist_path = os.path.join(MODELS_DIR, f"{model_name}_history.json")
    try:
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"  [WARN] Could not save history: {e}")

    print(f"\n  Best {model_name} → val_acc={best_val_acc*100:.2f}%  val_loss={best_val_loss:.4f}")
    return best_val_acc, best_val_loss, best_ckpt


# ──────────────────────────────────────────────────────────────────────────────
#  Inference time measurement
# ──────────────────────────────────────────────────────────────────────────────

def measure_inference_ms(model, sample_np):
    model.eval()
    x = torch.tensor(sample_np[None], dtype=torch.float32).to(DEVICE)
    # warm-up
    with torch.no_grad():
        for _ in range(10):
            model(x)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            model(x)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / 100 * 1000


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Sign Model — Step 03: Training")
    print("=" * 70)

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, class_names = \
        load_processed_dataset(NPZ_PATH)
    num_classes = len(class_names)
    input_dim   = X_train.shape[2]
    print(f"  Classes: {num_classes}  |  Input dim: {input_dim}")
    print(f"  Train: {len(X_train)}  |  Val: {len(X_val)}  |  Test: {len(X_test)}")
    print(f"  Device: {DEVICE}")

    # Datasets + loaders
    train_ds = SignLandmarkDataset(X_train, y_train, class_names, augment=True)
    val_ds   = SignLandmarkDataset(X_val,   y_val,   class_names, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=(DEVICE.type=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(DEVICE.type=="cuda"))

    # Class weights
    all_weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=y_train)
    cw_tensor   = torch.tensor(all_weights, dtype=torch.float32)

    results = {}   # model_name → {val_acc, val_loss, ckpt_path, params, infer_ms}
    sample  = X_val[0]  # for inference timing

    # ── Model A: LSTM ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Model A: LSTM Classifier")
    print("=" * 70)
    lstm_model = LSTMClassifier(
        input_dim=input_dim,
        hidden_dim=cfg["model"]["hidden_dim"],
        num_classes=num_classes,
        num_layers=2,
        dropout=cfg["model"]["dropout"],
    ).to(DEVICE)

    val_acc_lstm, val_loss_lstm, ckpt_lstm = run_training(
        lstm_model, "lstm", train_loader, val_loader, cw_tensor
    )
    # Reload best weights for inference timing
    lstm_model.load_state_dict(torch.load(ckpt_lstm, map_location=DEVICE, weights_only=True))
    infer_lstm = measure_inference_ms(lstm_model, sample)
    results["lstm"] = {
        "val_acc":   val_acc_lstm,
        "val_loss":  val_loss_lstm,
        "ckpt_path": ckpt_lstm,
        "params":    count_parameters(lstm_model),
        "infer_ms":  round(infer_lstm, 3),
    }

    # ── Model B: Transformer ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Model B: Transformer Encoder Classifier")
    print("=" * 70)
    tf_model = TransformerClassifier(
        input_dim=input_dim,
        d_model=cfg["model"]["hidden_dim"],
        nhead=4,
        num_layers=cfg["model"]["num_layers"],
        dim_feedforward=256,
        dropout=0.2,
        num_classes=num_classes,
    ).to(DEVICE)

    val_acc_tf, val_loss_tf, ckpt_tf = run_training(
        tf_model, "transformer", train_loader, val_loader, cw_tensor
    )
    tf_model.load_state_dict(torch.load(ckpt_tf, map_location=DEVICE, weights_only=True))
    infer_tf = measure_inference_ms(tf_model, sample)
    results["transformer"] = {
        "val_acc":   val_acc_tf,
        "val_loss":  val_loss_tf,
        "ckpt_path": ckpt_tf,
        "params":    count_parameters(tf_model),
        "infer_ms":  round(infer_tf, 3),
    }

    # ── Comparison Table ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Model Comparison")
    print("=" * 70)
    print(f"  {'Model':<15}  {'Val Acc':>8}  {'Val Loss':>9}  {'Params':>10}  {'Infer ms':>9}")
    print("  " + "-" * 60)
    for name, r in results.items():
        print(
            f"  {name:<15}  {r['val_acc']*100:>7.2f}%  {r['val_loss']:>9.4f}"
            f"  {r['params']:>10,}  {r['infer_ms']:>8.2f}"
        )

    # ── Select Winner ─────────────────────────────────────────────────────
    winner_name = max(results, key=lambda n: results[n]["val_acc"])
    winner      = results[winner_name]
    winner_model = lstm_model if winner_name == "lstm" else tf_model

    print(f"\n  🏆 Winner: {winner_name.upper()} (val_acc={winner['val_acc']*100:.2f}%)")

    # Save final model
    os.makedirs(FINAL_DIR, exist_ok=True)
    final_model_path = os.path.join(FINAL_DIR, "model.pt")
    try:
        torch.save(winner_model.state_dict(), final_model_path)
        print(f"  Final model weights → {final_model_path}")
    except Exception as e:
        print(f"  [ERROR] Could not save final model: {e}")
        sys.exit(1)

    # model_config.json
    model_config = {
        "model_type":   winner_name,
        "input_dim":    int(input_dim),
        "num_classes":  int(num_classes),
        "class_names":  class_names,
        "hidden_dim":   cfg["model"]["hidden_dim"],
        "num_layers":   cfg["model"]["num_layers"] if winner_name == "transformer" else 2,
        "dropout":      cfg["model"]["dropout"],
        "val_acc":      round(winner["val_acc"], 5),
        "val_loss":     round(winner["val_loss"], 5),
        "params":       winner["params"],
        "infer_ms":     winner["infer_ms"],
    }
    cfg_out = os.path.join(FINAL_DIR, "model_config.json")
    try:
        with open(cfg_out, "w") as f:
            json.dump(model_config, f, indent=2)
        print(f"  model_config.json → {cfg_out}")
    except Exception as e:
        print(f"  [WARN] Could not save model_config.json: {e}")

    # Copy label_map to final/
    try:
        shutil.copy(LABEL_MAP_PATH, os.path.join(FINAL_DIR, "label_map.json"))
        print(f"  label_map.json → {FINAL_DIR}")
    except Exception as e:
        print(f"  [WARN] Could not copy label_map: {e}")

    print("\n  Training complete ✅")


if __name__ == "__main__":
    main()
