# ASL Alphabet Landmark Classifier

A standalone, production-grade **American Sign Language (ASL) A–Z classifier** using **MediaPipe** hand landmark extraction and a **PyTorch MLP** model.

- 🚀 **No CNN / raw images** — landmark-based only (63 features per frame)
- ⚡ **GPU training**, **CPU-efficient inference** (< 5 ms per frame)
- 📊 **Target ≥98% accuracy** with per-class precision/recall
- 📡 **Real-time webcam demo** with stability smoothing
- 🔬 **Modular & unit-tested**

---

## Project Structure

```
sign_alphabet_model/
├── data/processed/           ← extracted + split .npy files
├── preprocessing/
│   ├── extract_landmarks.py  ← MediaPipe extraction
│   ├── normalize.py          ← wrist-centered, scale-invariant normalization
│   └── split_dataset.py      ← stratified 80/10/10 split
├── training/
│   ├── model.py              ← flexible MLP (baseline + Optuna variants)
│   ├── train.py              ← training loop, early stopping, logging
│   └── tuner.py              ← Optuna hyperparameter search
├── evaluation/
│   ├── metrics.py            ← accuracy, F1, per-class report
│   └── confusion_matrix.py   ← heatmap plot
├── inference/
│   ├── predictor.py          ← loads model, predict(landmarks) → (label, confidence)
│   ├── smoothing.py          ← rolling buffer, vote threshold, confidence gating
│   └── webcam_test.py        ← live webcam loop
├── tests/
│   ├── test_preprocessing.py
│   └── test_model_output.py
├── best_model.pt             ← saved best checkpoint
├── label_encoder.pkl
├── requirements.txt
└── README.md
```

---

## Installation

```bash
cd sign_alphabet_model
pip install -r requirements.txt
```

> Requires Python 3.9+. For GPU training, install PyTorch with CUDA:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

---

## Step 1 — Extract Landmarks

Processes every image in `ASL_Alphabet_Dataset/asl_alphabet_train/{A..Z}/`:

```bash
# Full extraction (~220K images, 20–40 minutes)
python -m preprocessing.extract_landmarks

# Quick smoke test (100 images per class)
python -m preprocessing.extract_landmarks --limit 100
```

**Output:**
- `data/processed/X.npy` — shape `(N, 63)` raw landmark coordinates
- `data/processed/y.npy` — shape `(N,)` integer class labels
- `label_encoder.pkl` — maps integer → letter

**Prints:** total extracted, total failed, samples per class.

---

## Step 2 — Split Dataset

Applies normalization and creates stratified train/val/test splits:

```bash
python -m preprocessing.split_dataset
```

**Normalization per sample:**
1. Subtract wrist landmark (index 0) → position invariant
2. Divide by max pairwise distance → scale invariant
3. Flatten to 63-dim vector

**Output:** `X_train.npy`, `X_val.npy`, `X_test.npy`, `y_train.npy`, `y_val.npy`, `y_test.npy`

Split: **80% train / 10% val / 10% test** (stratified)

---

## Step 3 — Train Baseline Model

```bash
python -m training.train
```

| Option          | Default | Description                          |
|-----------------|---------|--------------------------------------|
| `--epochs`      | 150     | Max training epochs                  |
| `--batch_size`  | 512     | Training batch size                  |
| `--lr`          | 1e-3    | Initial learning rate                |
| `--patience`    | 10      | Early stopping patience (val F1)     |
| `--optimizer`   | adam    | `adam` or `adamw`                    |
| `--save_path`   | best_model.pt | Checkpoint save path           |

**Baseline architecture:**
```
Input(63) → Linear(256) → BN → ReLU → Dropout(0.4)
          → Linear(128) → BN → ReLU → Dropout(0.3)
          → Linear(64)  → ReLU
          → Linear(26)   [logits]
```

Training log saved to `logs/training_log.json`.

---

## Step 4 — Hyperparameter Tuning (Optuna)

```bash
python -m training.tuner
```

| Option          | Default | Description                    |
|-----------------|---------|--------------------------------|
| `--trials`      | 30      | Number of Optuna trials        |
| `--epochs`      | 100     | Max epochs per trial           |

**Search space:**
- Layers: 2–5
- Hidden units (per layer): 64–512
- Dropout rate (per layer): 0.2–0.5
- Learning rate: 1e-4 – 1e-2 (log scale)
- Optimizer: Adam / AdamW

After search, the best model is retrained on combined train+val and saved to `best_model.pt`.

---

## Step 5 — Evaluate on Test Set

```bash
python -m evaluation.metrics
```

**Outputs:**
- Accuracy, Macro F1, Macro Precision, Macro Recall
- Full per-class precision / recall / F1 table
- `evaluation/confusion_matrix.png` (row-normalised heatmap)

Target: **≥ 98% accuracy**

---

## Step 6 — Live Webcam Demo

```bash
python inference/webcam_test.py
```

| Option            | Default | Description             |
|-------------------|---------|-------------------------|
| `--camera_id`     | 0       | OpenCV camera index     |
| `--model_path`    | best_model.pt | Override model path |

**Prediction pipeline:**
```
Webcam → MediaPipe → Normalize → MLP → Rolling Buffer (10)
  → Output letter only if ≥7/10 agree AND avg confidence ≥0.85
```

**Overlay colours:**
- 🟢 Green — confirmed stable letter
- 🟡 Amber — candidate (not yet stable)
- ⬜ Grey — no hand detected / low confidence

Press **ESC** or **Q** to quit.

---

## Step 7 — Run Unit Tests

```bash
cd sign_alphabet_model
python -m pytest tests/ -v
```

**Tests cover:**
- `test_preprocessing.py` — normalization correctness (wrist zeroed, scale ≈ 1.0, dtype, translation/scale invariance)
- `test_model_output.py` — model output shape `(batch, 26)`, custom architectures, smoother thresholds

---

## Full Pipeline (Quick Reference)

```bash
# 1. Extract (full run)
python -m preprocessing.extract_landmarks

# 2. Split
python -m preprocessing.split_dataset

# 3. Train baseline
python -m training.train

# 4. Tune (optional, ~1-3hrs on RTX 3050)
python -m training.tuner --trials 30

# 5. Evaluate
python -m evaluation.metrics

# 6. Webcam
python inference/webcam_test.py
```

---

## Performance Notes

| Stage         | Hardware       | Expected Time       |
|---------------|---------------|---------------------|
| Extraction    | CPU (i5-12500H)| 20–40 min (full)    |
| Training      | RTX 3050       | ~5–15 min (baseline)|
| Tuning (30 trials) | RTX 3050  | ~1–3 hrs            |
| Inference     | CPU            | < 5 ms / frame      |

---

## Notes

- Only letters **A–Z** (26 classes) are used. `del`, `nothing`, `space` are excluded.
- MediaPipe discards images with 0 or >1 hand detected.
- Model uses `best_model.pt` — saved by **validation macro F1**, not just accuracy.
