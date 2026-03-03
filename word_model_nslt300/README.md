# word_model_nslt300 — Word-Level Sign Language Recognition

Independent WLASL / NSLT-300 module. **Zero dependency on the parent project.**

---

## Quick Start

```bash
cd d:\haha\word_model_nslt300

# 1. Analyse dataset (run once)
python dataset_analysis.py

# 2. Generate CSV splits (run once)
python preprocessing/prepare_splits.py

# 3. Extract landmarks (run once — takes ~1–3 hrs for 2660 videos)
python preprocessing/extract_landmarks.py

# 4. Train
python train.py

# 5. Evaluate on test set
python evaluate.py

# 6. Live webcam inference
python live_inference.py
```

---

## Dataset Summary

| Split | Videos | Notes |
|-------|--------|-------|
| Train | 2,043  | 90% of train+val pool, stratified |
| Val   | 300    | 10% of train+val pool, stratified |
| Test  | 317    | Official NSLT-300 test (untouched) |

- **300 classes** (WLASL words)
- **2,660 videos** available locally (of 5,118 total)

---

## Architecture

```
Input  (B, 30, 162)
 → BatchNorm1d(162)
 → BiLSTM(256)  → Dropout(0.3)
 → BiLSTM(256)
 → Temporal Attention
 → Linear(512→256) + ReLU → Dropout(0.4)
 → Linear(256→300)
```
**2,909,040 trainable parameters**

Feature vector per frame: `right_hand(63) + left_hand(63) + upper_pose(36) = 162`

---

## Integration API

```python
from word_model_nslt300.utils import load_model, predict_sequence
import numpy as np

model, label_map = load_model()

# sequence: numpy (30, 162) or torch tensor
result = predict_sequence(np.zeros((30, 162)), model, label_map)
# {"word": "book", "confidence": 0.91, "top3": [("book", 0.91), ...]}
```

---

## File Map

```
word_model_nslt300/
├── dataset_analysis.py          # Phase 1: dataset summary
├── train.py                     # Phase 6: training loop
├── evaluate.py                  # Phase 7: test evaluation
├── live_inference.py            # Phase 8: webcam demo
├── utils.py                     # Phase 9: integration API
├── preprocessing/
│   ├── extract_landmarks.py     # Phase 3: MediaPipe extraction
│   └── prepare_splits.py        # Phase 4: CSV splits
├── model/
│   └── sign_model.py            # Phase 5: BiLSTM + Attention
├── data/
│   ├── features/                # .npy files (30×162)
│   ├── splits/                  # train/val/test.csv
│   └── metadata/                # label_map.json
└── checkpoints/                 # best_model.pt
```

---

## Notes

- Requires: `torch`, `mediapipe`, `opencv-python`, `scikit-learn`, `matplotlib`
- GPU: RTX 3050 (AMP enabled automatically)
- Confidence threshold for live inference: **0.75**
- Early stopping patience: **12 epochs**
