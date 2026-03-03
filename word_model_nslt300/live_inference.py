"""
live_inference.py — Phase 8: Real-Time Webcam Inference

Intelligent motion-based word segmentation + live prediction.

Segmentation algorithm:
  - Compute per-frame landmark velocity (L2 norm of displacement)
  - Apply N-frame moving average smoothing
  - Motion above threshold → frame is "active"
  - Word boundary triggered when:
      (a) active frames ≥ MIN_FRAMES AND motion drops below threshold for DEAD_FRAMES
      (b) buffer reaches MAX_FRAMES

  On word boundary:
    - Sample 30 frames from buffer
    - Run model
    - If top1_confidence > CONFIDENCE_THRESHOLD: append to sentence

Run: python live_inference.py
Keys: Q = quit, C = clear sentence, S = show/hide skeleton
"""

import sys
import time
import collections
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
    _lm_to_array, normalize_frame, temporal_resample, POSE_INDICES,
)

# ─── Configuration ────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.75     # minimum top-1 confidence to accept
MIN_FRAMES           = 10       # minimum active frames before triggering
MAX_FRAMES           = 60       # hard cap on buffer size
DEAD_FRAMES          = 8        # quiet frames to trigger word boundary
SMOOTH_WINDOW        = 5        # moving average window for velocity
MOTION_THRESHOLD     = 0.02     # normalised velocity threshold
SEQ_LEN              = 30
N_FEATURES           = 162

# ─── Display config ───────────────────────────────────────────────────────────
WIN_NAME    = "NSLT-300 Live Inference"
FONT        = cv2.FONT_HERSHEY_SIMPLEX
COLOR_GREEN = (0, 220, 80)
COLOR_RED   = (0, 50, 220)
COLOR_BLUE  = (220, 100, 30)
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY  = (180, 180, 180)
COLOR_GOLD  = (0, 200, 255)


def moving_average(data, window):
    if len(data) < window:
        return np.mean(data) if data else 0.0
    return np.mean(data[-window:])


def extract_frame_features(results):
    """Extract and normalize landmarks from a MediaPipe Holistic result."""
    right_hand = _lm_to_array(results.right_hand_landmarks) \
        if results.right_hand_landmarks else np.zeros((21, 3), dtype=np.float32)
    left_hand  = _lm_to_array(results.left_hand_landmarks) \
        if results.left_hand_landmarks else np.zeros((21, 3), dtype=np.float32)
    pose_upper = _lm_to_array(results.pose_landmarks, indices=POSE_INDICES) \
        if results.pose_landmarks else np.zeros((len(POSE_INDICES), 3), dtype=np.float32)
    return normalize_frame(right_hand, left_hand, pose_upper)   # (162,)


def frame_velocity(prev_feat, curr_feat):
    """L2 norm of feature displacement between consecutive frames."""
    if prev_feat is None:
        return 0.0
    return float(np.linalg.norm(curr_feat - prev_feat))


def prepare_sequence(buffer):
    """Sample SEQ_LEN frames from buffer → (1, 30, 162) tensor."""
    arr = np.array(buffer, dtype=np.float32)  # (N, 162)
    if len(arr) >= SEQ_LEN:
        indices = np.linspace(0, len(arr) - 1, SEQ_LEN, dtype=int)
        arr     = arr[indices]
    else:
        pad = np.zeros((SEQ_LEN - len(arr), N_FEATURES), dtype=np.float32)
        arr = np.vstack([arr, pad])
    return torch.from_numpy(arr).unsqueeze(0)  # (1, 30, 162)


def draw_ui(frame, state):
    """Overlay all UI elements onto the frame in-place."""
    H, W = frame.shape[:2]
    overlay = frame.copy()

    # ── Semi-transparent top bar ───────────────────────────────
    cv2.rectangle(overlay, (0, 0), (W, 80), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # ── Title ─────────────────────────────────────────────────
    cv2.putText(frame, "NSLT-300 Sign Language Recognition",
                (10, 25), FONT, 0.65, COLOR_GOLD, 2)

    # ── Status dot ────────────────────────────────────────────
    status_color = COLOR_GREEN if state["capturing"] else COLOR_GRAY
    cv2.circle(frame, (W - 30, 20), 10, status_color, -1)
    status_text = "CAPTURING" if state["capturing"] else "WAITING"
    cv2.putText(frame, status_text, (W - 130, 25), FONT, 0.45, status_color, 1)

    # ── Frame count bar ──────────────────────────────────────
    buf_len  = state["buf_len"]
    bar_w    = int((buf_len / MAX_FRAMES) * (W - 40))
    bar_color = COLOR_RED if buf_len > MAX_FRAMES * 0.8 else COLOR_GREEN
    cv2.rectangle(frame, (20, 45), (20 + bar_w, 60), bar_color, -1)
    cv2.rectangle(frame, (20, 45), (W - 20, 60), COLOR_GRAY, 1)
    cv2.putText(frame, f"Buffer: {buf_len}/{MAX_FRAMES}",
                (25, 57), FONT, 0.35, COLOR_WHITE, 1)

    # ── Velocity bar ──────────────────────────────────────────
    vel_norm = min(state["velocity"] / (MOTION_THRESHOLD * 5), 1.0)
    vel_w    = int(vel_norm * (W - 40))
    cv2.rectangle(frame, (20, 65), (20 + vel_w, 76), COLOR_BLUE, -1)
    cv2.rectangle(frame, (20, 65), (W - 20, 76), COLOR_GRAY, 1)
    cv2.putText(frame, f"Motion: {state['velocity']:.3f}",
                (25, 75), FONT, 0.35, COLOR_WHITE, 1)

    # ── Top predictions ───────────────────────────────────────
    if state["top3"]:
        y = 110
        cv2.putText(frame, "Predictions:", (10, y), FONT, 0.5, COLOR_GOLD, 1)
        for i, (word, conf) in enumerate(state["top3"]):
            bar_len = int(conf * 200)
            col     = COLOR_GREEN if i == 0 else COLOR_GRAY
            label   = f"{'>>>' if i == 0 else '   '} {word:20s}  {conf*100:5.1f}%"
            cv2.putText(frame, label, (10, y + 20 + i * 22),
                        FONT, 0.5 if i == 0 else 0.4, col, 1 if i > 0 else 2)

    # ── Sentence buffer ───────────────────────────────────────
    sentence = " ".join(state["sentence"])
    # wrap at ~50 chars
    max_chars = 50
    lines = []
    while len(sentence) > max_chars:
        split = sentence[:max_chars].rfind(" ")
        split = split if split > 0 else max_chars
        lines.append(sentence[:split])
        sentence = sentence[split:].strip()
    lines.append(sentence)

    y_sent = H - 80
    cv2.rectangle(frame, (0, y_sent - 18), (W, H), (20, 20, 20), -1)
    cv2.putText(frame, "Sentence:", (10, y_sent), FONT, 0.45, COLOR_GOLD, 1)
    for i, line in enumerate(lines[-2:]):  # show last 2 lines
        cv2.putText(frame, line, (10, y_sent + 18 + i * 20),
                    FONT, 0.55, COLOR_WHITE, 1)

    # ── Controls hint ─────────────────────────────────────────
    cv2.putText(frame, "Q=Quit  C=Clear  S=Skeleton",
                (W - 220, H - 10), FONT, 0.35, COLOR_GRAY, 1)

    return frame


def draw_landmarks(frame, results, draw_pose=True, draw_hands=True):
    """Optionally draw MediaPipe skeleton on frame."""
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic

    if draw_hands:
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )
    if draw_pose and results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(200, 200, 200), thickness=1, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(150, 150, 150), thickness=1),
        )


def run():
    print("=" * 60)
    print("  NSLT-300 Live Sign Language Inference")
    print("=" * 60)

    # ── Load model ────────────────────────────────────────────
    print("  Loading model...")
    try:
        model, label_map = load_model()
        device = next(model.parameters()).device
        print(f"  Model loaded on {device}")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        print("  Please run train.py first to generate checkpoints/best_model.pt")
        sys.exit(1)

    # ── MediaPipe ─────────────────────────────────────────────
    mp_holistic = mp.solutions.holistic

    # ── State ─────────────────────────────────────────────────
    state = {
        "capturing":  False,
        "buf_len":    0,
        "velocity":   0.0,
        "top3":       [],
        "sentence":   [],
    }

    landmark_buffer  = []       # list of (162,) feature vectors
    velocity_hist    = []       # for moving average
    prev_feat        = None
    quiet_frame_cnt  = 0        # consecutive below-threshold frames
    show_skeleton    = True

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ERROR: Cannot open webcam")
        sys.exit(1)

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("  Webcam open. Press Q to quit, C to clear, S to toggle skeleton.")
    print("  Performing recognition — sign a word to start.")
    print("-" * 60)

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("  Webcam read failed — exiting.")
                break

            frame = cv2.flip(frame, 1)   # mirror view
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True

            # ── Draw skeleton ─────────────────────────────────
            if show_skeleton:
                draw_landmarks(frame, results)

            # ── Feature extraction ────────────────────────────
            curr_feat = extract_frame_features(results)
            vel       = frame_velocity(prev_feat, curr_feat)
            velocity_hist.append(vel)
            smoothed_vel = moving_average(velocity_hist, SMOOTH_WINDOW)
            prev_feat    = curr_feat.copy()

            is_active = smoothed_vel > MOTION_THRESHOLD

            # ── Segmentation logic ────────────────────────────
            if is_active:
                landmark_buffer.append(curr_feat)
                quiet_frame_cnt = 0
                state["capturing"] = True
            else:
                if state["capturing"]:
                    quiet_frame_cnt += 1
                    landmark_buffer.append(curr_feat)  # include trailing quiet frames

            state["buf_len"]  = len(landmark_buffer)
            state["velocity"] = smoothed_vel

            # ── Check for word boundary ───────────────────────
            trigger = False
            if state["capturing"] and len(landmark_buffer) >= MIN_FRAMES:
                if quiet_frame_cnt >= DEAD_FRAMES:
                    trigger = True
                elif len(landmark_buffer) >= MAX_FRAMES:
                    trigger = True

            if trigger:
                # ── Inference ─────────────────────────────────
                seq_tensor = prepare_sequence(landmark_buffer)
                result     = predict_sequence(seq_tensor, model, label_map, device=device)
                top3       = result["top3"]
                word       = result["word"]
                conf       = result["confidence"]

                state["top3"] = top3
                print(f"  Predicted: '{word}' ({conf*100:.1f}%)  top3={[(w,f'{c*100:.0f}%') for w,c in top3]}")

                if conf >= CONFIDENCE_THRESHOLD:
                    state["sentence"].append(word)
                    print(f"  Accepted → sentence: {' '.join(state['sentence'])}")

                # Reset buffer
                landmark_buffer.clear()
                quiet_frame_cnt     = 0
                state["capturing"]  = False

            # ── Draw UI ───────────────────────────────────────
            frame = draw_ui(frame, state)
            cv2.imshow(WIN_NAME, frame)

            # ── Key handling ──────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("c"):
                state["sentence"].clear()
                print("  Sentence cleared.")
            elif key == ord("s"):
                show_skeleton = not show_skeleton

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n  Final sentence: {' '.join(state['sentence'])}")
    print("  Goodbye.")


if __name__ == "__main__":
    run()
