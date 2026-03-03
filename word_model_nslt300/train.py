"""
train.py — Phase 6: Training Pipeline

Full training loop with:
  - GPU acceleration (RTX 3050 / AMP)
  - AdamW optimizer with ReduceLROnPlateau
  - Label smoothing + class weighting
  - Gradient clipping
  - Early stopping (patience=12)
  - Saves: checkpoints/best_model.pt, training_history.json,
           confusion_matrix.png, classification_report.txt

Run: python train.py
"""

import json
import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODULE_DIR    = Path(__file__).resolve().parent
SPLITS_DIR    = MODULE_DIR / "data" / "splits"
FEATURES_DIR  = MODULE_DIR / "data" / "features"
META_DIR      = MODULE_DIR / "data" / "metadata"
CKPT_DIR      = MODULE_DIR / "checkpoints"
NSLT_JSON     = MODULE_DIR.parent / "nslt_100.json"
WLASL_JSON    = MODULE_DIR.parent / "WLASL_v0.3.json"

sys.path.insert(0, str(MODULE_DIR))
from model.sign_model import SignModel, LabelSmoothingCrossEntropy

# ─── Hyper-parameters ────────────────────────────────────────────────────────
CFG = dict(
    batch_size      = 32,
    epochs          = 100,
    lr              = 5e-4,
    weight_decay    = 1e-4,
    label_smoothing = 0.1,
    grad_clip       = 1.0,
    patience        = 12,           # early stopping
    lr_patience     = 5,            # ReduceLROnPlateau
    lr_factor       = 0.5,
    num_workers     = 4,
    pin_memory      = True,
    seq_len         = 40,
    feat_dim        = 162,
    num_classes     = 100,
    lstm_hidden     = 256,
    dropout1        = 0.3,
    dropout2        = 0.4,
    seed            = 42,
)


# ─── Dataset ─────────────────────────────────────────────────────────────────

class SignDataset(Dataset):
    """
    Loads pre-extracted .npy features from data/features/ directory.
    Falls back to on-the-fly extraction if .npy is not found (slower).
    """

    def __init__(self, csv_path: Path, nslt: dict, idx2gloss: dict,
                 augment: bool = False):
        import csv
        self.samples  = []   # [(npy_path_or_None, video_path, label_idx)]
        self.augment  = augment
        self.idx2gloss = idx2gloss
        self.seq_len  = CFG["seq_len"]
        self.feat_dim = CFG["feat_dim"]

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_path  = Path(row["video_path"])
                label_idx   = int(row["label_index"])
                video_id    = video_path.stem
                gloss       = idx2gloss.get(label_idx, f"class_{label_idx}")
                safe_gloss  = "".join(c if c.isalnum() or c in "-_" else "_" for c in gloss)
                npy_path    = FEATURES_DIR / safe_gloss / f"{video_id}.npy"
                self.samples.append((npy_path, video_path, label_idx))

    def __len__(self):
        return len(self.samples)

    def _load(self, npy_path, video_path, label_idx):
        if npy_path.exists():
            seq = np.load(str(npy_path)).astype(np.float32)  # (30, 162)
        else:
            # Fallback: run MediaPipe on-the-fly (slower but robust)
            seq = self._extract_on_the_fly(video_path, label_idx)
        return seq

    def _extract_on_the_fly(self, video_path, label_idx):
        """Single-video extraction for missing .npy files."""
        try:
            import cv2
            import mediapipe as mp
            from preprocessing.extract_landmarks import (
                sample_frames, temporal_resample, normalize_frame,
                _lm_to_array, POSE_INDICES
            )
            cap = cv2.VideoCapture(str(video_path))
            seq = np.zeros((self.seq_len, self.feat_dim), dtype=np.float32)
            if not cap.isOpened():
                return seq
            total = int(cap.get(cv2.CAP_PROP_PROP_FRAME_COUNT))  # noqa
            frames = []
            while True:
                ret, fr = cap.read()
                if not ret:
                    break
                frames.append(fr)
            cap.release()
            if len(frames) >= self.seq_len:
                frames = temporal_resample(frames, self.seq_len)
            mp_h = mp.solutions.holistic
            with mp_h.Holistic(static_image_mode=False, model_complexity=1,
                               min_detection_confidence=0.5) as hol:
                for i, frame in enumerate(frames[:self.seq_len]):
                    import cv2
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = hol.process(rgb)
                    rh = _lm_to_array(res.right_hand_landmarks) if res.right_hand_landmarks \
                        else np.zeros((21, 3), dtype=np.float32)
                    lh = _lm_to_array(res.left_hand_landmarks) if res.left_hand_landmarks \
                        else np.zeros((21, 3), dtype=np.float32)
                    pu = _lm_to_array(res.pose_landmarks, indices=POSE_INDICES) \
                        if res.pose_landmarks else np.zeros((len(POSE_INDICES), 3), dtype=np.float32)
                    seq[i] = normalize_frame(rh, lh, pu)
            return seq
        except Exception:
            return np.zeros((self.seq_len, self.feat_dim), dtype=np.float32)

    def _augment(self, seq: np.ndarray) -> np.ndarray:
        """Light augmentation on landmark sequences."""
        # 1. Gaussian noise
        seq = seq + np.random.normal(0, 0.01, seq.shape).astype(np.float32)
        # 2. Temporal shift (roll by ±2 frames)
        shift = np.random.randint(-2, 3)
        if shift != 0:
            seq = np.roll(seq, shift, axis=0)
        # 3. Random scale (0.9–1.1)
        scale = np.random.uniform(0.9, 1.1)
        seq   = seq * scale
        return seq

    def __getitem__(self, idx):
        npy_path, video_path, label_idx = self.samples[idx]
        seq = self._load(npy_path, video_path, label_idx)
        if self.augment:
            seq = self._augment(seq)
        x = torch.from_numpy(seq)           # (30, 162)
        y = torch.tensor(label_idx, dtype=torch.long)
        return x, y


# ─── Training utilities ───────────────────────────────────────────────────────

def compute_class_weights(csv_path: Path, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights."""
    import csv
    counts = Counter()
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            counts[int(row["label_index"])] += 1
    total  = sum(counts.values())
    weights = torch.zeros(num_classes)
    for cls, cnt in counts.items():
        weights[cls] = total / (num_classes * cnt)
    return weights


def top3_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    _, top3 = logits.topk(3, dim=1)
    correct = top3.eq(targets.unsqueeze(1).expand_as(top3)).any(dim=1)
    return correct.float().mean().item()


def save_confusion_matrix(cm, label_names, path: Path):
    """Save a compact confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(24, 22))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix — Test Set", fontsize=14)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved → {path.name}")


# ─── Training loop ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, clip):
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=scaler.is_enabled()):
            logits = model(x)
            loss   = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss    += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_n       += x.size(0)

    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_targets = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        all_logits.append(logits.cpu())
        all_targets.append(y)

    logits  = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    preds   = logits.argmax(1).numpy()
    tgts    = targets.numpy()

    acc   = accuracy_score(tgts, preds)
    f1    = f1_score(tgts, preds, average="macro", zero_division=0)
    prec  = precision_score(tgts, preds, average="macro", zero_division=0)
    rec   = recall_score(tgts, preds, average="macro", zero_division=0)
    top3  = top3_accuracy(logits, targets)

    return dict(acc=acc, f1=f1, precision=prec, recall=rec, top3=top3,
                preds=preds, targets=tgts, logits=logits)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM   : {total_mem:.1f} GB")
    print("=" * 60)

    # ── Load metadata ─────────────────────────────────────────
    with open(WLASL_JSON, "r") as f:
        wlasl = json.load(f)
    idx2gloss = {i: entry["gloss"] for i, entry in enumerate(wlasl)}

    with open(NSLT_JSON, "r") as f:
        nslt = json.load(f)

    # ── Datasets & Dataloaders ────────────────────────────────
    train_csv = SPLITS_DIR / "train.csv"
    val_csv   = SPLITS_DIR / "val.csv"

    if not train_csv.exists():
        print("  ERROR: Run prepare_splits.py first!")
        sys.exit(1)

    train_ds = SignDataset(train_csv, nslt, idx2gloss, augment=True)
    val_ds   = SignDataset(val_csv,   nslt, idx2gloss, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=CFG["batch_size"], shuffle=True,
        num_workers=CFG["num_workers"], pin_memory=CFG["pin_memory"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG["batch_size"] * 2, shuffle=False,
        num_workers=CFG["num_workers"], pin_memory=CFG["pin_memory"],
    )

    print(f"  Train samples : {len(train_ds)}")
    print(f"  Val samples   : {len(val_ds)}")

    # ── Model ─────────────────────────────────────────────────
    model = SignModel(
        num_classes = CFG["num_classes"],
        seq_len     = CFG["seq_len"],
        feat_dim    = CFG["feat_dim"],
        lstm_hidden = CFG["lstm_hidden"],
        dropout1    = CFG["dropout1"],
        dropout2    = CFG["dropout2"],
    ).to(device)
    print(f"  Trainable params : {model.count_parameters():,}")

    # ── Class weights ─────────────────────────────────────────
    class_weights = compute_class_weights(train_csv, CFG["num_classes"])
    class_weights = class_weights.to(device)

    # ── Loss, optimizer, scheduler ────────────────────────────
    criterion  = LabelSmoothingCrossEntropy(
        smoothing=CFG["label_smoothing"],
        weight=class_weights,
    )
    optimizer  = torch.optim.AdamW(
        model.parameters(),
        lr=CFG["lr"],
        weight_decay=CFG["weight_decay"],
    )
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=CFG["lr_factor"],
        patience=CFG["lr_patience"], verbose=True,
    )
    use_amp    = device.type == "cuda"
    scaler     = GradScaler(enabled=use_amp)

    # ── Training ──────────────────────────────────────────────
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_f1      = 0.0
    patience_cnt = 0
    history      = []

    print(f"\n  Starting training — {CFG['epochs']} epochs, AMP={use_amp}")
    print("-" * 60)

    for epoch in range(1, CFG["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device,
            clip=CFG["grad_clip"],
        )
        val_metrics = evaluate(model, val_loader, device)
        elapsed     = time.time() - t0

        scheduler.step(val_metrics["f1"])

        row = dict(
            epoch   = epoch,
            tr_loss = round(tr_loss, 4),
            tr_acc  = round(tr_acc,  4),
            val_acc = round(val_metrics["acc"],  4),
            val_f1  = round(val_metrics["f1"],   4),
            val_pre = round(val_metrics["precision"], 4),
            val_rec = round(val_metrics["recall"],    4),
            val_top3= round(val_metrics["top3"], 4),
            lr      = round(optimizer.param_groups[0]["lr"], 7),
        )
        history.append(row)

        print(
            f"  Ep {epoch:3d}/{CFG['epochs']}  "
            f"loss={tr_loss:.4f}  tr_acc={tr_acc:.3f}  "
            f"val_acc={val_metrics['acc']:.3f}  val_f1={val_metrics['f1']:.3f}  "
            f"top3={val_metrics['top3']:.3f}  "
            f"lr={optimizer.param_groups[0]['lr']:.6f}  "
            f"[{elapsed:.0f}s]"
        )

        # ── Checkpoint ────────────────────────────────────────
        if val_metrics["f1"] > best_f1:
            best_f1      = val_metrics["f1"]
            patience_cnt = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "val_f1":      best_f1,
                "val_acc":     val_metrics["acc"],
                "cfg":         CFG,
            }, CKPT_DIR / "best_model.pt")
            print(f"    ✓ New best F1={best_f1:.4f} — checkpoint saved")
        else:
            patience_cnt += 1
            if patience_cnt >= CFG["patience"]:
                print(f"\n  Early stopping at epoch {epoch}  (no improvement for {CFG['patience']} epochs)")
                break

    # ── Post-training artefacts ───────────────────────────────
    with open(MODULE_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Confusion matrix (val set with best model)
    ckpt = torch.load(CKPT_DIR / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    val_metrics = evaluate(model, val_loader, device)
    label_names = [idx2gloss.get(i, str(i)) for i in range(CFG["num_classes"])]
    cm = confusion_matrix(val_metrics["targets"], val_metrics["preds"],
                          labels=list(range(CFG["num_classes"])))
    save_confusion_matrix(cm, label_names, CKPT_DIR / "confusion_matrix.png")

    report = classification_report(
        val_metrics["targets"], val_metrics["preds"],
        target_names=label_names, zero_division=0,
    )
    with open(CKPT_DIR / "classification_report.txt", "w") as f:
        f.write(report)

    print("\n" + "=" * 60)
    print(f"  Best Val F1       : {best_f1:.4f}")
    print(f"  Best Val Accuracy : {ckpt['val_acc']:.4f}")
    print(f"  Checkpoint        : checkpoints/best_model.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
