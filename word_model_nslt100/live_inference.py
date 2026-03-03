"""
live_inference.py — Phase 8: Real-Time Webcam Inference (NSLT-100)

Intelligent motion segmentation + position/velocity feature extraction.

Segmentation:
  - Velocity magnitude per frame (L2 norm of landmark displacement)
  - 5-frame moving average smoothing
  - Motion threshold: segment is "active" if smoothed_vel > MOTION_THRESHOLD
  - Word boundary: active≥MIN_FRAMES AND (quiet≥DEAD_FRAMES OR buffer hits MAX_FRAMES)

On word boundary:
  - Build 40-frame position+velocity feature sequence
  - Run model
  - If top1_confidence > 0.75 → append word to sentence

Keys: Q=quit  C=clear sentence  S=toggle skeleton

Run: python live_inference.py
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn.functional as F

MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

from utils import load_model, predict_sequence
from preprocessing.extract_landmarks import (
    POSE_INDICES, normalize_landmarks, _lm_to_array,
    SEQ_LEN, N_FEATURES,
)

# ─── Config ──────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.75
MIN_FRAMES           = 12
MAX_FRAMES           = 60
DEAD_FRAMES          = 8
SMOOTH_WINDOW        = 5
MOTION_THRESHOLD     = 0.02

# ─── Display ─────────────────────────────────────────────────────────────────
WIN_NAME   = "NSLT-100 Live Inference"
FONT       = cv2.FONT_HERSHEY_SIMPLEX
C_GREEN    = (0, 220, 80)
C_RED      = (0, 60, 220)
C_GOLD     = (0, 200, 255)
C_WHITE    = (255, 255, 255)
C_GRAY     = (160, 160, 160)


# ─── Feature helpers ─────────────────────────────────────────────────────────

def extract_frame_position(results):
    """Extract normalized (162,) position vector from MediaPipe result."""
    rh = _lm_to_array(results.right_hand_landmarks) \
        if results.right_hand_landmarks else np.zeros((21, 3), dtype=np.float32)
    lh = _lm_to_array(results.left_hand_landmarks) \
        if results.left_hand_landmarks else np.zeros((21, 3), dtype=np.float32)
    pu = _lm_to_array(results.pose_landmarks, indices=POSE_INDICES) \
        if results.pose_landmarks else np.zeros((len(POSE_INDICES), 3), dtype=np.float32)
    return normalize_landmarks(rh, lh, pu)  # (162,)


def moving_average(arr, w):
    return float(np.mean(arr[-w:])) if arr else 0.0


def build_sequence_from_buffer(position_buffer):
    """
    Given a list of (162,) position vectors, build (40, 324) position+velocity.
    """
    N = len(position_buffer)
    pos_arr = np.array(position_buffer, dtype=np.float32)  # (N, 162)

    # Temporal resample
    if N >= SEQ_LEN:
        idx = np.linspace(0, N - 1, SEQ_LEN, dtype=int)
        pos = pos_arr[idx]
    else:
        pad = np.zeros((SEQ_LEN - N, 162), dtype=np.float32)
        pos = np.vstack([pos_arr, pad])

    # Velocity
    vel = np.zeros_like(pos)
    vel[1:] = pos[1:] - pos[:-1]

    return np.concatenate([pos, vel], axis=1)  # (40, 324)


# ─── UI drawing ──────────────────────────────────────────────────────────────

def draw_ui(frame, state):
    H, W = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, 85), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, "NSLT-100 Sign Language Recognition",
                (10, 26), FONT, 0.65, C_GOLD, 2)

    # Status dot
    sc = C_GREEN if state["capturing"] else C_GRAY
    cv2.circle(frame, (W - 30, 20), 10, sc, -1)
    cv2.putText(frame, "CAPTURING" if state["capturing"] else "WAITING",
                (W - 135, 25), FONT, 0.42, sc, 1)

    # Buffer bar
    bw = int((state["buf_len"] / MAX_FRAMES) * (W - 40))
    bc = C_RED if state["buf_len"] > MAX_FRAMES * 0.8 else C_GREEN
    cv2.rectangle(frame, (20, 45), (20 + bw, 60), bc, -1)
    cv2.rectangle(frame, (20, 45), (W - 20, 60), C_GRAY, 1)
    cv2.putText(frame, f"Buffer {state['buf_len']}/{MAX_FRAMES}",
                (25, 57), FONT, 0.35, C_WHITE, 1)

    # Velocity bar
    vn = min(state["velocity"] / (MOTION_THRESHOLD * 5), 1.0)
    cv2.rectangle(frame, (20, 65), (20 + int(vn * (W - 40)), 76), (220, 100, 30), -1)
    cv2.rectangle(frame, (20, 65), (W - 20, 76), C_GRAY, 1)
    cv2.putText(frame, f"Motion {state['velocity']:.3f}",
                (25, 75), FONT, 0.35, C_WHITE, 1)

    # Predictions
    if state["top3"]:
        y = 108
        cv2.putText(frame, "Prediction:", (10, y), FONT, 0.48, C_GOLD, 1)
        for i, (word, conf) in enumerate(state["top3"]):
            col  = C_GREEN if i == 0 else C_GRAY
            size = 0.52 if i == 0 else 0.40
            lbl  = f"{'>>>' if i==0 else '   '} {word:20s} {conf*100:5.1f}%"
            cv2.putText(frame, lbl, (10, y + 20 + i * 22), FONT, size, col,
                        2 if i == 0 else 1)

    # Sentence
    sent = " ".join(state["sentence"])
    y_s  = H - 75
    cv2.rectangle(frame, (0, y_s - 20), (W, H), (20, 20, 20), -1)
    cv2.putText(frame, "Sentence:", (10, y_s), FONT, 0.44, C_GOLD, 1)
    cv2.putText(frame, sent[-70:] if len(sent) > 70 else sent,
                (10, y_s + 22), FONT, 0.55, C_WHITE, 1)
    cv2.putText(frame, "Q=Quit  C=Clear  S=Skeleton",
                (W - 225, H - 10), FONT, 0.35, C_GRAY, 1)
    return frame


def draw_skeleton(frame, results):
    mp_drawing = mp.solutions.drawing_utils
    mp_styles  = mp.solutions.drawing_styles
    mp_h       = mp.solutions.holistic
    for hand_lm in [results.right_hand_landmarks, results.left_hand_landmarks]:
        if hand_lm:
            mp_drawing.draw_landmarks(frame, hand_lm, mp_h.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style())
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_h.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec((200, 200, 200), 1, 2),
            mp_drawing.DrawingSpec((150, 150, 150), 1))


# ─── Main loop ───────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("  NSLT-100 Live Sign Language Inference")
    print("=" * 60)

    print("  Loading model...")
    try:
        model, label_map = load_model()
        device = next(model.parameters()).device
        print(f"  Model loaded on {device}")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}\n  Run train.py first."); sys.exit(1)

    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ERROR: Cannot open webcam"); sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    state = {"capturing": False, "buf_len": 0, "velocity": 0.0,
             "top3": [], "sentence": []}

    pos_buffer   = []     # (162,) vectors
    vel_hist     = []     # smoothed velocity history
    prev_pos     = None
    quiet_cnt    = 0
    show_skeleton= True

    print("  Ready. Sign to start.")
    print("-" * 60)

    with mp_holistic.Holistic(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True

            if show_skeleton:
                draw_skeleton(frame, results)

            # Position + velocity this frame
            curr_pos = extract_frame_position(results)
            vel      = float(np.linalg.norm(curr_pos - prev_pos)) if prev_pos is not None else 0.0
            vel_hist.append(vel)
            smooth_vel = moving_average(vel_hist, SMOOTH_WINDOW)
            prev_pos   = curr_pos.copy()

            is_active  = smooth_vel > MOTION_THRESHOLD

            # Segmentation
            if is_active:
                pos_buffer.append(curr_pos)
                quiet_cnt = 0
                state["capturing"] = True
            elif state["capturing"]:
                quiet_cnt += 1
                pos_buffer.append(curr_pos)   # include trailing quiet frames

            state["buf_len"]  = len(pos_buffer)
            state["velocity"] = smooth_vel

            # Trigger
            trigger = (
                state["capturing"] and
                len(pos_buffer) >= MIN_FRAMES and
                (quiet_cnt >= DEAD_FRAMES or len(pos_buffer) >= MAX_FRAMES)
            )

            if trigger:
                seq    = build_sequence_from_buffer(pos_buffer)
                result = predict_sequence(seq, model, label_map, device=device)
                state["top3"] = result["top3"]
                word, conf    = result["word"], result["confidence"]
                print(f"  → '{word}'  {conf*100:.1f}%  "
                      f"top3={[(w,f'{c*100:.0f}%') for w,c in result['top3']]}")
                if conf >= CONFIDENCE_THRESHOLD:
                    state["sentence"].append(word)
                pos_buffer.clear()
                quiet_cnt          = 0
                state["capturing"] = False

            frame = draw_ui(frame, state)
            cv2.imshow(WIN_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("c"):
                state["sentence"].clear(); print("  Cleared.")
            elif key == ord("s"):
                show_skeleton = not show_skeleton

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n  Final sentence: {' '.join(state['sentence'])}")


if __name__ == "__main__":
    run()
