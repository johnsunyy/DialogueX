"""
training/train.py

Full training pipeline for ASL landmark MLP.

- Loads processed/X_train.npy, X_val.npy, y_train.npy, y_val.npy
- Trains with CrossEntropyLoss + Adam/AdamW
- Evaluates macro F1 on validation set after every epoch
- Saves best checkpoint (by val macro F1) to best_model.pt
- Early stopping with configurable patience

Usage:
    python -m training.train [options]
    python -m training.train --epochs 100 --batch_size 512 --lr 1e-3 --patience 10
"""

import os
import sys
import time
import argparse
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score
from training.model import build_baseline_model, ASLClassifier


# ─────────────────────────── helpers ─────────────────────────────

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROCESSED_DIR = os.path.join(ROOT_DIR, 'data', 'processed')
LOGS_DIR = os.path.join(ROOT_DIR, 'logs')


def get_device():
    """Return GPU if available, else CPU."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def load_splits(processed_dir: str):
    """Load train and val splits as torch tensors."""
    def _load(name):
        path = os.path.join(processed_dir, f'{name}.npy')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}. Run `python -m preprocessing.split_dataset` first."
            )
        return np.load(path)

    X_train = _load('X_train')
    X_val   = _load('X_val')
    y_train = _load('y_train')
    y_val   = _load('y_val')
    return X_train, X_val, y_train, y_val


def make_loaders(X_train, y_train, X_val, y_val, batch_size: int, device: torch.device):
    """Create DataLoader objects from numpy arrays."""
    def _to_tensors(X, y):
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        return TensorDataset(X_t, y_t)

    train_ds = _to_tensors(X_train, y_train)
    val_ds   = _to_tensors(X_val,   y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  pin_memory=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    return train_loader, val_loader


# ─────────────────────────── core ─────────────────────────────────

def evaluate(model, loader, device, criterion):
    """Run one pass over loader; return loss, accuracy, macro F1."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)

            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, acc, macro_f1


def train_model(
    model: ASLClassifier,
    X_train, y_train,
    X_val,   y_val,
    *,
    epochs: int = 150,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    optimizer_name: str = 'adam',
    save_path: str = None,
    log_path: str = None,
    verbose: bool = True,
):
    """
    Train model and return best val macro F1.
    Returns: best_val_f1 (float)
    """
    device = get_device()
    if verbose:
        print(f"\n  Device     : {device}")
        if device.type == 'cuda':
            print(f"  GPU        : {torch.cuda.get_device_name(0)}")
        print(f"  Epochs     : {epochs}")
        print(f"  Batch size : {batch_size}")
        print(f"  LR         : {lr}")
        print(f"  Patience   : {patience}")
        print(f"  Optimizer  : {optimizer_name}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_name.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=False
    )

    train_loader, val_loader = make_loaders(X_train, y_train, X_val, y_val, batch_size, device)

    best_val_f1 = -1.0
    best_epoch  = 0
    no_improve  = 0
    history     = []

    os.makedirs(os.path.dirname(save_path), exist_ok=True) if save_path else None
    os.makedirs(LOGS_DIR, exist_ok=True)

    if verbose:
        print(f"\n  {'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>8} | {'Val Acc':>7} | {'Val F1':>7} | {'LR':>8}")
        print(f"  {'-'*58}")

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # ── Training ──
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(y_batch)

        train_loss = epoch_loss / len(train_loader.dataset)

        # ── Validation ──
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device, criterion)
        scheduler.step(val_f1)

        current_lr = optimizer.param_groups[0]['lr']
        row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'lr': current_lr,
        }
        history.append(row)

        if verbose:
            print(f"  {epoch:>5} | {train_loss:>10.4f} | {val_loss:>8.4f} | {val_acc:>7.4f} | {val_f1:>7.4f} | {current_lr:>8.6f}")

        # ── Best model checkpoint ──
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            no_improve  = 0
            if save_path:
                # Extract hidden_layers as (units, dropout) tuples from model
                hl = []
                for layer in model.network:
                    if isinstance(layer, nn.Linear):
                        hl.append({'linear_out': layer.out_features})
                    elif isinstance(layer, nn.Dropout):
                        hl[-1]['dropout'] = layer.p

                # Build list of (units, dropout) — skip the final output layer
                hidden_layers_cfg = [
                    (entry['linear_out'], entry.get('dropout', 0.0))
                    for entry in hl[:-1]  # exclude output layer entry
                ]

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_f1': val_f1,
                    'val_acc': val_acc,
                    'config': {
                        'input_dim': 63,
                        'num_classes': 26,
                        'hidden_layers': hidden_layers_cfg,
                    }
                }, save_path)
        else:
            no_improve += 1

        # ── Early stopping ──
        if no_improve >= patience:
            if verbose:
                print(f"\n  [Early Stop] No improvement for {patience} epochs. Best epoch: {best_epoch}, Val F1: {best_val_f1:.4f}")
            break

    elapsed = time.time() - t0

    # Save log
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w') as f:
            json.dump(history, f, indent=2)

    if verbose:
        print(f"\n  Training complete in {elapsed:.1f}s")
        print(f"  Best val F1 : {best_val_f1:.4f}  (epoch {best_epoch})")
        if save_path:
            print(f"  Saved model : {save_path}")

    return best_val_f1


# ─────────────────────────── CLI ──────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train ASL Landmark MLP')
    parser.add_argument('--processed_dir', default=PROCESSED_DIR)
    parser.add_argument('--save_path',     default=os.path.join(ROOT_DIR, 'best_model.pt'))
    parser.add_argument('--log_path',      default=os.path.join(LOGS_DIR, 'training_log.json'))
    parser.add_argument('--epochs',        type=int,   default=150)
    parser.add_argument('--batch_size',    type=int,   default=512)
    parser.add_argument('--lr',            type=float, default=1e-3)
    parser.add_argument('--weight_decay',  type=float, default=1e-4)
    parser.add_argument('--patience',      type=int,   default=10)
    parser.add_argument('--optimizer',     choices=['adam', 'adamw'], default='adam')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  ASL Landmark MLP — Training")
    print(f"{'='*60}")

    processed_dir = os.path.abspath(args.processed_dir)
    X_train, X_val, y_train, y_val = load_splits(processed_dir)
    print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}")

    model = build_baseline_model(input_dim=63, num_classes=26)
    from training.model import count_parameters
    print(f"  Parameters: {count_parameters(model):,}")

    train_model(
        model,
        X_train, y_train,
        X_val,   y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        optimizer_name=args.optimizer,
        save_path=os.path.abspath(args.save_path),
        log_path=os.path.abspath(args.log_path),
        verbose=True,
    )


if __name__ == '__main__':
    main()
