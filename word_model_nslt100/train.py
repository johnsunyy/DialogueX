"""
train.py — Phase 6: Training Pipeline for NSLT-100

Run: python train.py
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODULE_DIR   = Path(__file__).resolve().parent
SPLITS_DIR   = MODULE_DIR / "data" / "splits"
CKPT_DIR     = MODULE_DIR / "checkpoints"
WLASL_JSON   = MODULE_DIR.parent / "WLASL_v0.3.json"

sys.path.insert(0, str(MODULE_DIR))
from model.sign_model import SignModel, LabelSmoothingCrossEntropy

# ─── Config ──────────────────────────────────────────────────────────────────
CFG = dict(
    batch_size       = 32,
    epochs           = 100,
    lr               = 5e-4,
    weight_decay     = 1e-4,
    label_smoothing  = 0.1,
    grad_clip        = 1.0,
    patience         = 12,
    lr_patience      = 5,
    lr_factor        = 0.5,
    num_workers      = 4,
    pin_memory       = True,
    seq_len          = 40,
    feat_dim         = 324,
    num_classes      = 100,
    lstm1_hidden     = 256,
    lstm2_hidden     = 128,
    dropout1         = 0.3,
    dropout2         = 0.4,
    seed             = 42,
)


# ─── Dataset ─────────────────────────────────────────────────────────────────

class SignDataset(Dataset):
    """Loads pre-extracted .npy feature files listed in CSV."""

    def __init__(self, csv_path: Path, augment: bool = False):
        import csv as _csv
        self.samples = []
        self.augment = augment
        with open(csv_path) as f:
            for row in _csv.DictReader(f):
                self.samples.append((Path(row["feature_path"]), int(row["label_index"])))

    def __len__(self):
        return len(self.samples)

    def _augment(self, seq: np.ndarray) -> np.ndarray:
        # Gaussian noise
        seq = seq + np.random.normal(0, 0.005, seq.shape).astype(np.float32)
        # Temporal shift ±2 frames
        s = np.random.randint(-2, 3)
        if s:
            seq = np.roll(seq, s, axis=0)
        # Scale jitter
        seq = seq * np.random.uniform(0.95, 1.05)
        return seq

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        if path.exists():
            seq = np.load(str(path)).astype(np.float32)
        else:
            seq = np.zeros((CFG["seq_len"], CFG["feat_dim"]), dtype=np.float32)
        if self.augment:
            seq = self._augment(seq)
        return torch.from_numpy(seq), torch.tensor(label, dtype=torch.long)


# ─── Utilities ───────────────────────────────────────────────────────────────

def compute_class_weights(csv_path, num_classes):
    import csv as _csv
    counts = Counter()
    with open(csv_path) as f:
        for row in _csv.DictReader(f):
            counts[int(row["label_index"])] += 1
    total = sum(counts.values())
    w = torch.zeros(num_classes)
    for cls, cnt in counts.items():
        w[cls] = total / (num_classes * cnt)
    return w


def top3_accuracy(logits, targets):
    _, top3 = logits.topk(3, dim=1)
    return top3.eq(targets.unsqueeze(1).expand_as(top3)).any(1).float().mean().item()


# ─── Train / Eval loops ───────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, scaler, device, clip):
    model.train()
    loss_sum = correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=scaler.is_enabled()):
            logits = model(x)
            loss   = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer); scaler.update()
        loss_sum += loss.item() * x.size(0)
        correct  += (logits.argmax(1) == y).sum().item()
        total    += x.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_targets = [], []
    for x, y in loader:
        all_logits.append(model(x.to(device, non_blocking=True)).cpu())
        all_targets.append(y)
    logits  = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    preds   = logits.argmax(1).numpy()
    tgts    = targets.numpy()
    return dict(
        acc  = accuracy_score(tgts, preds),
        f1   = f1_score(tgts, preds, average="macro", zero_division=0),
        prec = precision_score(tgts, preds, average="macro", zero_division=0),
        rec  = recall_score(tgts, preds, average="macro", zero_division=0),
        top3 = top3_accuracy(logits, targets),
        preds=preds, targets=tgts, logits=logits,
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print("=" * 60)

    train_csv = SPLITS_DIR / "train.csv"
    val_csv   = SPLITS_DIR / "val.csv"
    if not train_csv.exists():
        print("ERROR: Run prepare_splits.py first."); sys.exit(1)

    train_ds = SignDataset(train_csv, augment=True)
    val_ds   = SignDataset(val_csv,   augment=False)
    print(f"  Train : {len(train_ds)} | Val : {len(val_ds)}")

    train_loader = DataLoader(train_ds, CFG["batch_size"], shuffle=True,
                              num_workers=CFG["num_workers"], pin_memory=CFG["pin_memory"],
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   CFG["batch_size"]*2, shuffle=False,
                              num_workers=CFG["num_workers"], pin_memory=CFG["pin_memory"])

    model = SignModel(
        num_classes  = CFG["num_classes"],
        seq_len      = CFG["seq_len"],
        feat_dim     = CFG["feat_dim"],
        lstm1_hidden = CFG["lstm1_hidden"],
        lstm2_hidden = CFG["lstm2_hidden"],
        dropout1     = CFG["dropout1"],
        dropout2     = CFG["dropout2"],
    ).to(device)
    print(f"  Params : {model.count_parameters():,}")

    # Print architecture summary
    print("\n  Architecture:")
    print(f"    BatchNorm1d({CFG['feat_dim']})")
    print(f"    BiLSTM({CFG['lstm1_hidden']}) → Dropout({CFG['dropout1']})")
    print(f"    BiLSTM({CFG['lstm2_hidden']}) → TemporalAttention")
    print(f"    Linear(256→128) + ReLU → Dropout({CFG['dropout2']})")
    print(f"    Linear(128→{CFG['num_classes']})\n")

    class_weights = compute_class_weights(train_csv, CFG["num_classes"]).to(device)
    criterion     = LabelSmoothingCrossEntropy(CFG["label_smoothing"], class_weights)
    optimizer     = torch.optim.AdamW(model.parameters(), lr=CFG["lr"],
                                      weight_decay=CFG["weight_decay"])
    scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode="max", factor=CFG["lr_factor"],
                        patience=CFG["lr_patience"])
    use_amp   = device.type == "cuda"
    scaler    = GradScaler("cuda", enabled=use_amp)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_f1 = patience_cnt = 0
    history = []

    print(f"  Training — {CFG['epochs']} epochs | AMP={use_amp}")
    print("-" * 60)

    for epoch in range(1, CFG["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion,
                                      scaler, device, CFG["grad_clip"])
        vm = evaluate(model, val_loader, device)
        scheduler.step(vm["f1"])

        row = dict(epoch=epoch, tr_loss=round(tr_loss,4), tr_acc=round(tr_acc,4),
                   val_acc=round(vm["acc"],4), val_f1=round(vm["f1"],4),
                   val_pre=round(vm["prec"],4), val_rec=round(vm["rec"],4),
                   val_top3=round(vm["top3"],4),
                   lr=round(optimizer.param_groups[0]["lr"], 7))
        history.append(row)

        print(f"  Ep {epoch:3d}/{CFG['epochs']}  "
              f"loss={tr_loss:.4f}  tr={tr_acc:.3f}  "
              f"val={vm['acc']:.3f}  f1={vm['f1']:.3f}  "
              f"top3={vm['top3']:.3f}  "
              f"lr={optimizer.param_groups[0]['lr']:.6f}  [{time.time()-t0:.0f}s]")

        if vm["f1"] > best_f1:
            best_f1 = vm["f1"]; patience_cnt = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "val_f1": best_f1, "val_acc": vm["acc"], "cfg": CFG},
                       CKPT_DIR / "best_model.pt")
            print(f"    ✓ Best F1={best_f1:.4f} saved")
        else:
            patience_cnt += 1
            if patience_cnt >= CFG["patience"]:
                print(f"\n  Early stop at epoch {epoch}"); break

    # ── Post-training artifacts ───────────────────────────────
    with open(MODULE_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(WLASL_JSON) as f:
        wlasl = json.load(f)
    idx2gloss   = {i: e["gloss"] for i, e in enumerate(wlasl)}
    label_names = [idx2gloss.get(i, str(i)) for i in range(CFG["num_classes"])]

    ckpt = torch.load(CKPT_DIR / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    vm = evaluate(model, val_loader, device)

    cm = confusion_matrix(vm["targets"], vm["preds"], labels=list(range(CFG["num_classes"])))
    fig, ax = plt.subplots(figsize=(20, 18))
    ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_title("Confusion Matrix — Val Set"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout(); plt.savefig(str(CKPT_DIR / "confusion_matrix.png"), dpi=150); plt.close()

    report = classification_report(vm["targets"], vm["preds"],
                                   target_names=label_names, zero_division=0)
    with open(CKPT_DIR / "classification_report.txt", "w") as f:
        f.write(report)

    print("\n" + "=" * 60)
    print(f"  Best Val F1       : {best_f1:.4f}")
    print(f"  Best Val Accuracy : {ckpt['val_acc']:.4f}")
    print(f"  Checkpoint        : checkpoints/best_model.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
