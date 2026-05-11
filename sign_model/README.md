# Sign Language Recognition — `sign_model/`

A **fully self-contained** training pipeline for word-level sign language recognition using MediaPipe Holistic landmarks + PyTorch (LSTM or Transformer).

> ⚡ All scripts are run from inside the `sign_model/` directory.  
> 🔒 Nothing in this folder reads from or writes to any other project code.

---

## ⚙️ Requirements

```bash
cd sign_model
pip install -r requirements.txt
```

| Package | Purpose |
|---|---|
| `mediapipe>=0.10.0` | Landmark extraction |
| `opencv-python>=4.8.0` | Video reading / webcam |
| `torch>=2.0.0` | Model training & inference |
| `numpy>=1.24.0` | Array operations |
| `scikit-learn>=1.3.0` | Scaler, split, metrics |
| `pyyaml>=6.0` | Config loading |
| `tqdm>=4.65.0` | Progress bars |

---

## 🚀 Full Pipeline

### Step 1 — Extract Landmarks

```bash
python scripts/01_extract_landmarks.py
```

**What it does:**
- Scans `../dataset_50_words/` (relative to `sign_model/`)
- Applies class filters: excludes classes with < 30 samples; caps classes at 80 samples; removes `KNIGHT3`
- Runs MediaPipe Holistic on every video frame
- Saves per-video `.npy` files of shape `(T, 225)` to `data/raw_landmarks/{CLASS}/`
- Generates `label_map.json`

**Expected output:**
```
── Class Filtering Summary ──────────────────────
  AGAIN                   42      42  INCLUDED
  AND                     40      40  INCLUDED
  ...
  WATER                  120      80  CAPPED
  ...
  SAY                      8       0  EXCLUDED (< 30 samples)

  Total included classes : 70
  Total videos to process: ~3900

Extraction Complete
  Total extracted : 3900
  Total failed    : 0
```

**Approximate runtime:** 20–60 min (CPU) depending on hardware  
**Output size:** ~50–200 MB of `.npy` files

---

### Step 2 — Preprocess

```bash
python scripts/02_preprocess.py
```

**What it does:**
- Loads all `.npy` files and normalizes each to 64 frames (center-crop or zero-pad)
- Performs stratified 70/15/15 train/val/test split
- Augments training set (×2 copies/sample): horizontal flip, temporal jitter, speed variation, Gaussian noise
- Fits `StandardScaler` on training data only (saved to `models/scaler.pkl`)
- Saves everything as `data/processed/dataset.npz`

**Expected output:**
```
  Loaded 3900 sequences, shape: (3900, 64, 225)
  Train: 2730  |  Val: 585  |  Test: 585
  Train size after augmentation: 8190
  Scaler saved → models/scaler.pkl
  Saved → data/processed/dataset.npz
```

**Approximate runtime:** 5–15 min

---

### Step 3 — Train

```bash
python scripts/03_train.py
```

**What it does:**
- Trains **Model A (LSTM)** and **Model B (Transformer)** for up to 80 epochs each
- Uses AdamW optimizer, CosineAnnealingLR scheduler, CrossEntropyLoss with class weights
- Early stopping (patience=15) on validation loss
- Saves best checkpoints to `models/checkpoints/`
- Prints a comparison table and saves the **winner** to `models/final/model.pt`

**Expected output:**
```
  Model Comparison
  ──────────────────────────────────────────────────────
  Model           Val Acc    Val Loss      Params  Infer ms
  lstm            87.30%      0.4321     512,000     2.10
  transformer     89.10%      0.3890     420,000     3.40
  🏆 Winner: TRANSFORMER (val_acc=89.10%)
```

**Approximate runtime:** 30 min–3 hrs depending on GPU/CPU

> 💡 **No GPU?** Training will fall back to CPU. Expect 2–4× longer training time. Reduce `epochs` in `configs/config.yaml` if needed.

---

### Step 4 — Evaluate

```bash
python scripts/04_evaluate.py
```

**What it does:**
- Loads `models/final/model.pt` and runs on the held-out test set
- Reports: Top-1, Top-3, Macro F1, Weighted F1
- Per-class pass/warn/fail (PASS ≥ 0.70 F1, WARN 0.50–0.70, FAIL < 0.50)
- Saves `per_class_metrics.json` and `confusion_matrix.npy`
- Prints the 10 most confused class pairs

**Expected output:**
```
  Top-1 Accuracy  : 88.50%
  Top-3 Accuracy  : 97.20%
  Macro F1        : 0.8791
  Weighted F1     : 0.8834

  Per-Class Metrics:
  AGAIN     Prec=0.91  Rec=0.88  F1=0.89  PASS ✅
  ...
```

---

### Step 5 — Live Test (Webcam)

```bash
python scripts/05_test_live.py
```

**What it does:**
- Opens webcam at index 0
- Runs MediaPipe Holistic on every frame
- Maintains a rolling 64-frame landmark buffer
- Runs inference every 8 new frames
- Displays top-3 predictions with confidence bars
- Hides prediction if confidence < 0.65
- Shows FPS and a "Collecting..." indicator for the first 64 frames
- Press `Q` to quit

---

## 📁 Output Files Reference

| File | Description |
|---|---|
| `label_map.json` | Class name → integer index mapping |
| `data/raw_landmarks/{CLASS}/{name}.npy` | Raw extracted landmarks, shape `(T, 225)` |
| `data/processed/dataset.npz` | All splits as numpy arrays |
| `data/splits/{train,val,test}_indices.json` | File-level split records |
| `models/scaler.pkl` | Fitted StandardScaler |
| `models/checkpoints/best_lstm.pt` | Best LSTM checkpoint |
| `models/checkpoints/best_transformer.pt` | Best Transformer checkpoint |
| `models/final/model.pt` | Winner model weights |
| `models/final/model_config.json` | Model architecture config |
| `models/final/per_class_metrics.json` | Per-class precision/recall/F1 |
| `models/final/confusion_matrix.npy` | Full confusion matrix |

---

## 🔧 Configuration (`configs/config.yaml`)

Edit this file to change any hyperparameters:

```yaml
dataset:
  min_samples: 30     # exclude classes below this threshold
  max_samples: 80     # cap classes at this count
  exclude_classes: ["KNIGHT3"]

preprocessing:
  target_frames: 64   # fixed sequence length

training:
  epochs: 80
  batch_size: 32
  learning_rate: 0.001
  early_stopping_patience: 15
```

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---|---|
| `No module named 'mediapipe'` | `pip install mediapipe>=0.10.0` |
| `No GPU detected` | Normal — scripts use CPU automatically. Reduce `epochs` for faster testing. |
| `RuntimeError: CUDA out of memory` | Reduce `batch_size` in `config.yaml` to 16 or 8 |
| `MediaPipe hand not detected` | Script fills missing hand with zeros — no action needed |
| `label_map.json not found` | Run `python scripts/01_extract_landmarks.py` first |
| `dataset.npz not found` | Run `python scripts/02_preprocess.py` first |
| `model.pt not found` | Run `python scripts/03_train.py` first |
| Webcam won't open | Check camera index — try `cv2.VideoCapture(1)` in `05_test_live.py` |
| Very low accuracy | Check dataset quality; augment or collect more data for flagged classes |

---

## 🗂️ Dataset Info

- Source: `../dataset_50_words/` (read-only, never modified)
- 75 total classes, 4,054 `.mp4` videos, 2.063 GB
- After filtering: ~70 classes, ~3,900 videos
- Video format: 640×480, ~30 fps, avg 73 frames (~2.5 sec)

---

*Self-contained module — does not import from or modify any other project files.*

---

## V2 Training Run (Improved Augmentation)

The V2 run uses **advanced augmentation** (mixup, cutmix, wrist transforms, body scaling)
and **improved training settings** (label smoothing, val_loss early stopping, higher dropout).
V1 model files are **never touched** — all V2 outputs go to `models/v2/`.

### What's New in V2

| Feature | V1 | V2 |
|---|---|---|
| Augmentations per sample | 2 | **6** |
| Augmentation techniques | 4 basic | **9 (+ wrist rotation, wrist scale, body scale, mixup, cutmix)** |
| Training samples | 8,013 | **~26,000+** |
| Dropout | 0.30 | **0.45** |
| Label smoothing | None | **0.1** |
| Early stopping metric | val_acc | **val_loss** |
| Learning rate | 0.001 | **0.0008** |
| Max epochs | 80 | **100** |

### V2 Pipeline (landmarks already extracted — skip step 1)

**Step 2 — Preprocess with V2 augmentation:**
```bash
python scripts/02_preprocess_v2.py --config configs/config_v2.yaml
```
- Reuses existing `data/raw_landmarks/` — no re-extraction
- Reuses existing `data/splits/` — same train/val/test indices for fair comparison
- Saves to `data/processed/dataset_v2.npz` (does NOT touch `dataset.npz`)
- Applies 6 augmented copies per training sample with all 9 techniques

**Step 3 — Train V2 model:**
```bash
python scripts/03_train_v2.py --config configs/config_v2.yaml
```
- Trains Transformer only (winner from V1)
- Prints live V1 vs V2 comparison table at the end

**Step 4 — Evaluate V2 model:**
```bash
python scripts/04_evaluate_v2.py --config configs/config_v2.yaml
```
- Prints full per-class V1 vs V2 F1 comparison for all 69 classes
- Shows which classes improved, degraded, or newly reached PASS status

**Step 5 — Live test (choose version):**
```bash
python scripts/05_test_live.py --version v2   # default
python scripts/05_test_live.py --version v1   # use original model
```
- The webcam overlay displays the active model version in the corner

### V2 Output Locations

| File | Path |
|---|---|
| Model weights | `sign_model/models/v2/final/model.pt` |
| Model config | `sign_model/models/v2/final/model_config.json` |
| Training history | `sign_model/models/v2/transformer_v2_history.json` |
| Evaluation metrics | `sign_model/models/v2/final/per_class_metrics.json` |
| Confusion matrix | `sign_model/models/v2/final/confusion_matrix.npy` |
| V1 vs V2 report | `sign_model/models/v2/v1_vs_v2_comparison.json` |
| V2 scaler | `sign_model/models/scaler_v2.pkl` |
| Best checkpoint | `sign_model/models/v2/checkpoints/best_transformer_v2.pt` |

### V1 Model Preserved (untouched)

| File | Path |
|---|---|
| V1 model weights | `sign_model/models/final/model.pt` ✅ |
| V1 metrics | `sign_model/models/final/per_class_metrics.json` ✅ |
| V1 config | `sign_model/configs/config.yaml` ✅ |
| V1 augment backup | `sign_model/utils/augment_v1_backup.py` ✅ |

---

*Self-contained module — does not import from or modify any other project files.*
