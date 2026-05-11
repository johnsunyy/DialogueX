"""
==============================================================================
sign_model/scripts/05_test_live.py
==============================================================================
WHAT    : Real-time sign language recognition using webcam + the trained model.
          Applies MediaPipe Holistic to each frame, maintains a rolling 64-frame
          landmark buffer, runs inference every 8 new frames, and overlays
          top-3 predictions with confidence scores on the video feed.

INPUT   : Webcam (cv2.VideoCapture(0))
          V1: sign_model/models/final/model.pt  +  models/scaler.pkl
          V2: sign_model/models/v2/final/model.pt  +  models/scaler_v2.pkl

RUN     : cd sign_model && python scripts/05_test_live.py             # defaults to v2
          cd sign_model && python scripts/05_test_live.py --version v2
          cd sign_model && python scripts/05_test_live.py --version v1
          Press Q to quit.
=============================================================================="""

import sys
import os
import argparse

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODULE_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, MODULE_ROOT)

# ──────────────────────────────────────────────────────────────────────────────
#  CLI — parse --version before anything else
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Sign Language Live Test")
    p.add_argument(
        "--version",
        choices=["v1", "v2"],
        default="v2",
        help="Model version to load: v1 (original) or v2 (improved). Default: v2",
    )
    return p.parse_args()

_ARGS = parse_args()
_VERSION = _ARGS.version

import json
import math
import pickle
import time
import collections
import numpy as np
import cv2
import mediapipe as mp
import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────────────────────
#  Paths — resolved from --version argument
# ──────────────────────────────────────────────────────────────────────────────

if _VERSION == "v2":
    FINAL_DIR      = os.path.join(MODULE_ROOT, "models", "v2", "final")
    SCALER_PATH    = os.path.join(MODULE_ROOT, "models", "scaler_v2.pkl")
else:  # v1
    FINAL_DIR      = os.path.join(MODULE_ROOT, "models", "final")
    SCALER_PATH    = os.path.join(MODULE_ROOT, "models", "scaler.pkl")

MODEL_PATH     = os.path.join(FINAL_DIR, "model.pt")
MODEL_CFG_PATH = os.path.join(FINAL_DIR, "model_config.json")

TARGET_FRAMES  = 64
FEATURE_DIM    = 225
INFER_EVERY    = 8          # run inference every N new frames
CONF_THRESHOLD = 0.65       # only show prediction if max confidence ≥ this

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────────────────────
#  Model - inline definition (no cross-folder imports needed at runtime)
# ──────────────────────────────────────────────────────────────────────────────

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class _LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(128, 256, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0., batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(256, 128), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(128, num_classes))
    def forward(self, x):
        x = self.input_proj(x)
        _, (h, _) = self.lstm(x)
        return self.classifier(h[-1])


class _TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=3,
                 dim_feedforward=256, dropout=0.2, num_classes=70):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = _PositionalEncoding(d_model, dropout=dropout)
        enc_layer       = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                     dropout, batch_first=True)
        self.encoder    = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(),
                                        nn.Dropout(0.3), nn.Linear(64, num_classes))
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        return self.classifier(self.encoder(x).mean(dim=1))


def build_model(cfg):
    mtype = cfg["model_type"]
    if mtype == "lstm":
        return _LSTMClassifier(cfg["input_dim"], cfg["hidden_dim"],
                               cfg["num_classes"], num_layers=cfg["num_layers"],
                               dropout=cfg["dropout"])
    elif mtype == "transformer":
        return _TransformerClassifier(cfg["input_dim"], cfg["hidden_dim"], 4,
                                      cfg["num_layers"], 256, 0.2, cfg["num_classes"])
    raise ValueError(f"Unknown model_type: {mtype}")


# ──────────────────────────────────────────────────────────────────────────────
#  Landmark extraction (single frame)
# ──────────────────────────────────────────────────────────────────────────────

def extract_frame_landmarks(results) -> np.ndarray:
    """Extract 225-d landmark vector from a MediaPipe Holistic result."""
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z]
                        for lm in results.left_hand_landmarks.landmark], dtype=np.float32).flatten()
    else:
        lh = np.zeros(63, dtype=np.float32)

    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z]
                        for lm in results.right_hand_landmarks.landmark], dtype=np.float32).flatten()
    else:
        rh = np.zeros(63, dtype=np.float32)

    if results.pose_landmarks:
        ps = np.array([[lm.x, lm.y, lm.z]
                        for lm in results.pose_landmarks.landmark], dtype=np.float32).flatten()
    else:
        ps = np.zeros(99, dtype=np.float32)

    vec = np.concatenate([lh, rh, ps])
    vec[0::3] = np.clip(vec[0::3], 0., 1.)
    vec[1::3] = np.clip(vec[1::3], 0., 1.)
    return vec


# ──────────────────────────────────────────────────────────────────────────────
#  Overlay helpers
# ──────────────────────────────────────────────────────────────────────────────

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.65
THICKNESS  = 2

def put_text(img, text, pos, color=(255, 255, 255), scale=FONT_SCALE, thickness=THICKNESS):
    cv2.putText(img, text, pos, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, FONT, scale, color,     thickness,     cv2.LINE_AA)


def draw_predictions(frame, top3, conf_threshold):
    """Render top-3 predictions on the frame."""
    h, w = frame.shape[:2]
    panel_h = 130
    panel   = np.zeros((panel_h, w, 3), dtype=np.uint8)
    panel[:] = (20, 20, 20)

    for rank, (cls_name, conf) in enumerate(top3):
        y_pos = 30 + rank * 38
        color = (0, 220, 100) if rank == 0 and conf >= conf_threshold else (180, 180, 180)
        label = f"#{rank+1}  {cls_name:<20}  {conf*100:.1f}%"
        put_text(panel, label, (20, y_pos), color, scale=0.7)
        # Confidence bar
        bar_x   = int(conf * (w - 250))
        bar_col = (0, 200, 80) if rank == 0 and conf >= conf_threshold else (80, 80, 200)
        cv2.rectangle(panel, (230, y_pos - 16), (230 + bar_x, y_pos - 4), bar_col, -1)

    return np.vstack([panel, frame])


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print(f"  Sign Model — Step 05: Live Testing  [Model: {_VERSION.upper()}]")
    print("=" * 70)
    print(f"  Version  : {_VERSION.upper()}")
    print(f"  Model    : {MODEL_PATH}")
    print(f"  Scaler   : {SCALER_PATH}")

    # Load model config
    try:
        with open(MODEL_CFG_PATH) as f:
            model_cfg = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] model_config.json not found: {MODEL_CFG_PATH}")
        print("  → Run: python scripts/03_train.py first")
        sys.exit(1)

    class_names = model_cfg["class_names"]

    # Build & load model
    model = build_model(model_cfg).to(DEVICE)
    try:
        state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
    except FileNotFoundError:
        print(f"[ERROR] model.pt not found: {MODEL_PATH}")
        sys.exit(1)
    model.eval()
    print(f"  Model loaded ({model_cfg['model_type'].upper()} {_VERSION.upper()}) on {DEVICE}")

    # Load scaler
    try:
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        print(f"  Scaler loaded from {SCALER_PATH}")
    except FileNotFoundError:
        print(f"[WARN] Scaler not found at {SCALER_PATH} — raw (unscaled) features will be used")
        scaler = None

    # Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam (index 0)")
        sys.exit(1)
    print("  Webcam opened. Press Q to quit.\n")

    # Rolling buffer
    buffer       = collections.deque(maxlen=TARGET_FRAMES)
    frames_since = 0          # frames since last inference
    last_top3    = []         # last inference result
    collecting   = True

    # FPS tracking
    fps_times = collections.deque(maxlen=30)

    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=0,         # 0 = faster for live use
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame capture failed. Retrying...")
                continue

            t_frame_start = time.perf_counter()

            # MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True

            # Extract + buffer
            vec = extract_frame_landmarks(results)
            buffer.append(vec)
            frames_since += 1

            # Check collection status
            collecting = len(buffer) < TARGET_FRAMES

            # Run inference every INFER_EVERY frames (when buffer is full)
            if not collecting and frames_since >= INFER_EVERY:
                frames_since = 0
                seq = np.stack(list(buffer), axis=0)   # (64, 225)

                # Scale
                if scaler is not None:
                    seq_2d = seq.reshape(-1, FEATURE_DIM)
                    seq_2d = scaler.transform(seq_2d)
                    seq    = seq_2d.reshape(TARGET_FRAMES, FEATURE_DIM)

                x_t = torch.tensor(seq[None], dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    logits = model(x_t)[0]          # (num_classes,)
                probs  = torch.softmax(logits, dim=-1).cpu().numpy()

                # Top-3
                top3_idx = np.argsort(probs)[::-1][:3]
                last_top3 = [(class_names[i], float(probs[i])) for i in top3_idx]

            # ── Draw overlay ──────────────────────────────────────────────
            display = frame.copy()

            if collecting:
                ratio = len(buffer) / TARGET_FRAMES
                put_text(display, f"Collecting frames... {int(ratio*100)}%",
                         (20, 40), color=(0, 200, 255), scale=0.8)
                cv2.rectangle(display, (20, 55), (20 + int(ratio * 300), 70), (0, 200, 100), -1)
            else:
                # Render top-3 panel
                conf_threshold = CONF_THRESHOLD
                if last_top3:
                    top_name, top_conf = last_top3[0]
                    if top_conf >= conf_threshold:
                        put_text(display, f"▶ {top_name}", (20, 45),
                                 color=(0, 255, 100), scale=1.1, thickness=2)
                    else:
                        put_text(display, "Uncertain...", (20, 45),
                                 color=(100, 100, 255), scale=0.85)
                    display = draw_predictions(display, last_top3, conf_threshold)

            # FPS + version badge
            fps_times.append(time.perf_counter())
            if len(fps_times) >= 2:
                fps = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0])
                put_text(display, f"FPS: {fps:.1f}", (display.shape[1] - 130, 30),
                         color=(200, 200, 0), scale=0.65)

            # Model version badge (top-right corner)
            badge_color = (0, 180, 255) if _VERSION == "v2" else (180, 100, 0)
            put_text(display, f"Model: {_VERSION.upper()}",
                     (display.shape[1] - 150, 58), color=badge_color, scale=0.6)

            cv2.imshow(f"Sign Language — Live Test [{_VERSION.upper()}]  [Q to quit]", display)

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("\n  Live test ended.")


if __name__ == "__main__":
    main()
