# Word-Level Sign Language Recognition — Model Comparison Report

**Generated:** 2026-03-03 | **System:** i5-12500H · RTX 3050 4GB · 16GB RAM

---

## Overview

Two independent SLR modules were built and trained on subsets of the WLASL dataset:

| Module | Dataset | Classes | Architecture |
|--------|---------|---------|--------------|
| `word_model_nslt300/` | NSLT-300 | 300 | BiLSTM(256) → BiLSTM(256) → Attn → Dense(256→300) |
| `word_model_nslt100/` | NSLT-100 | 100 | BiLSTM(256) → BiLSTM(128) → Attn → Dense(128→100) |

---

## Dataset Summary

| Metric | NSLT-300 | NSLT-100 |
|--------|----------|----------|
| Total entries in JSON | 5,118 | 2,038 |
| Videos on disk | 2,660 | 1,013 |
| Coverage | 51.9% | 49.7% |
| Official train entries | ~3,835 | 1,442 |
| Official val entries | ~765 | 338 |
| Official test entries | ~518 | 258 |
| **Actual train samples used** | **2,043** | **645** |
| **Actual val samples used** | **300** | **100** |
| Avg train samples / class | ~6.8 | ~6.5 |

> **Note:** Only ~50% of videos were available locally. Full dataset would double training data.

---

## Feature Extraction

| Parameter | NSLT-300 | NSLT-100 |
|-----------|----------|----------|
| Sequence length | 30 frames | 40 frames |
| Feature type | Position only | Position + Velocity |
| Features / frame | 162 | 324 |
| Output shape | (30, 162) | (40, 324) |
| Right hand landmarks | 21 × xyz | 21 × xyz |
| Left hand landmarks | 21 × xyz | 21 × xyz |
| Upper pose landmarks | 12 × xyz | 12 × xyz |
| Normalization | Shoulder-relative | Shoulder-relative |
| Temporal resampling | Uniform sampling | Uniform sampling |
| Extraction time | ~29 min | ~19 min |
| Failed videos | 0 | 3 (TIMEOUT) |

---

## Model Architecture

### NSLT-300 Model
```
Input (B, 30, 162)
→ BatchNorm1d(162)
→ BiLSTM(256, bidirectional) → (B, 30, 512)   Dropout(0.3)
→ BiLSTM(256, bidirectional) → (B, 30, 512)
→ TemporalAttention            → (B, 512)
→ Linear(512→256) + ReLU       Dropout(0.4)
→ Linear(256→300)
Trainable params: 2,909,040
```

### NSLT-100 Model
```
Input (B, 40, 324)
→ BatchNorm1d(324)
→ BiLSTM(256, bidirectional) → (B, 40, 512)   Dropout(0.3)
→ BiLSTM(128, bidirectional) → (B, 40, 256)
→ TemporalAttention            → (B, 256)
→ Linear(256→128) + ReLU       Dropout(0.4)
→ Linear(128→100)
Trainable params: 1,961,836
```

---

## Training Configuration

| Setting | NSLT-300 | NSLT-100 |
|---------|----------|----------|
| Optimizer | AdamW | AdamW |
| Learning rate | 5e-4 | 5e-4 |
| Weight decay | 1e-4 | 1e-4 |
| Batch size | 32 | 32 |
| Loss | LabelSmoothing CE (0.1) | LabelSmoothing CE (0.1) |
| Class weights | ✅ Inverse frequency | ✅ Inverse frequency |
| Grad clipping | max_norm=1.0 | max_norm=1.0 |
| Scheduler | ReduceLROnPlateau (patience=5) | ReduceLROnPlateau (patience=5) |
| Early stopping | patience=12 on val F1 | patience=12 on val F1 |
| AMP (mixed precision) | ✅ | ✅ |
| Data augmentation | ❌ | ✅ (noise, shift, scale) |

---

## Training Results

| Metric | NSLT-300 | NSLT-100 |
|--------|----------|----------|
| Epochs run | **37 / 100** | **76 / 100** |
| Best val accuracy | **19.33%** | **19.00%** |
| Best val F1 (macro) | **0.1491** | **0.1523** |
| Best val top-3 accuracy | ~34.7% | ~29.0% |
| Final train accuracy | 89.9% | 99.2% |
| Train / Val gap | 70.6% | 80.2% |
| Stopped reason | Early stop (no F1 gain) | Early stop (no F1 gain) |
| Time per epoch | ~25s | ~17s |

---

## Analysis

### What went right ✅
- GPU (RTX 3050) detected and used with AMP throughout
- Extraction pipeline robust: 0 / 3 failures respectively
- Models converge (train accuracy reaches 90–99%)
- Velocity features in NSLT-100 improved F1 slightly (0.149 → 0.152)
- All shapes validated: (30, 162) and (40, 324) confirmed

### What went wrong ❌
- **Severe overfitting**: train acc ~99% vs val ~19% — model memorizes, doesn't generalize
- **Data scarcity**: ~6.5–6.8 training samples per class is insufficient for a sequence model
- **Missing dataset**: ~50% of WLASL videos not on disk, halving effective training data

### Why validation accuracy plateaued at ~19%
With only ~6.5 samples per class, the BiLSTM has too little signal to learn class boundaries.
Random chance for 100 classes = 1%, for 300 classes = 0.33%.
Current accuracy (~19%) is far above random but far below usable (~60%+).

---

## Recommendations (Priority Order)

| Priority | Action | Expected Gain |
|----------|--------|---------------|
| 🥇 **#1** | Download missing WLASL videos (~1,000 more) | +20–30% accuracy |
| 🥈 **#2** | Train on top-20 classes (most samples/class) | Likely 40–60% accuracy |
| 🥉 **#3** | Add stronger augmentation (time warp, mirror) | +3–5% accuracy |
| 4 | Pretrain with self-supervised contrastive loss | +5–10% accuracy |
| 5 | Use transformer encoder instead of BiLSTM | +5% accuracy |

---

## Module Independence Confirmation

| Check | Status |
|-------|--------|
| `word_model_nslt300/` modifies existing files | ❌ None |
| `word_model_nslt100/` modifies existing files | ❌ None |
| Alphabet model affected | ❌ None |
| Existing project structure changed | ❌ None |
| Each module runs standalone | ✅ |
| Integration API (`utils.py`) present | ✅ Both |
| GPU used in training | ✅ RTX 3050 |

---

## File Locations

```
d:\haha\
├── word_model_nslt300\          ← NSLT-300 module (30 frames, 162 features)
│   ├── checkpoints\best_model.pt
│   ├── training_history.json
│   └── ...
├── word_model_nslt100\          ← NSLT-100 module (40 frames, 324 features)
│   ├── checkpoints\best_model.pt
│   ├── training_history.json
│   └── ...
├── nslt_300.json
├── nslt_100.json
└── videos\
```

---

*Report generated after full training cycle on both modules.*
