"""
inference/webcam_test.py

Live ASL letter prediction from webcam using MediaPipe + trained MLP.

Pipeline:
  Webcam → MediaPipe → normalize → ASLPredictor → PredictionSmoother → overlay on frame

Controls:
  ESC or 'q' — quit

Usage:
    python inference/webcam_test.py [--model_path PATH] [--camera_id N]
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
import mediapipe as mp

# Allow running as a script from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.normalize import normalize_sample
from inference.predictor import ASLPredictor
from inference.smoothing import PredictionSmoother

ROOT_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BEST_MODEL = os.path.join(ROOT_DIR, 'best_model.pt')
LE_PATH    = os.path.join(ROOT_DIR, 'label_encoder.pkl')

# ─── colour palette ───────────────────────────────────────────────
COLOUR_STABLE    = (0, 230, 0)     # Green — stable confirmed letter
COLOUR_UNSTABLE  = (0, 200, 255)   # Amber — candidate letter
COLOUR_NONE      = (80, 80, 80)    # Grey  — no hand or low confidence
COLOUR_SKELETON  = (255, 255, 255) # White skeleton
COLOUR_BG_BOX    = (20, 20, 20)    # Dark overlay


def draw_landmarks(frame, hand_landmarks):
    """Draw hand skeleton using MediaPipe's drawing utilities."""
    mp_drawing      = mp.solutions.drawing_utils
    mp_drawing_sty  = mp.solutions.drawing_styles
    mp_hands        = mp.solutions.hands
    mp_drawing.draw_landmarks(
        frame,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_sty.get_default_hand_landmarks_style(),
        mp_drawing_sty.get_default_hand_connections_style(),
    )


def overlay_prediction(frame, stable_letter, candidate_letter, confidence, fps):
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    bar_h = 90
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), COLOUR_BG_BOX, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

    if stable_letter:
        # Big confirmed letter
        cv2.putText(frame, stable_letter, (20, 75),
                    cv2.FONT_HERSHEY_DUPLEX, 2.5, COLOUR_STABLE, 3, cv2.LINE_AA)
        cv2.putText(frame, f"CONFIRMED  conf={confidence:.2f}", (110, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOUR_STABLE, 1, cv2.LINE_AA)
    elif candidate_letter:
        cv2.putText(frame, candidate_letter, (20, 75),
                    cv2.FONT_HERSHEY_DUPLEX, 2.5, COLOUR_UNSTABLE, 2, cv2.LINE_AA)
        cv2.putText(frame, f"candidate  conf={confidence:.2f}", (110, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOUR_UNSTABLE, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "—", (20, 70),
                    cv2.FONT_HERSHEY_DUPLEX, 2.0, COLOUR_NONE, 2, cv2.LINE_AA)
        cv2.putText(frame, "No hand / low confidence", (80, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOUR_NONE, 1, cv2.LINE_AA)

    # Info footer
    cv2.putText(frame, "ASL Alphabet | ESC/Q to quit", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)


def run_webcam(camera_id: int = 0, model_path: str = None, encoder_path: str = None):
    model_path   = model_path   or BEST_MODEL
    encoder_path = encoder_path or LE_PATH

    print(f"\n  ASL Webcam Inference")
    print(f"  Model  : {model_path}")
    print(f"  Camera : {camera_id}")
    print(f"  Press ESC or Q to quit.\n")

    predictor = ASLPredictor(model_path=model_path, label_encoder_path=encoder_path, device='cpu')
    smoother  = PredictionSmoother(buffer_size=10, agreement_threshold=7, confidence_threshold=0.85)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}. Try --camera_id 1 or 2.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_time    = time.time()
    stable_out   = None
    candidate    = None
    cand_conf    = 0.0

    print("  Webcam started. Show your hand sign!")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("  [WARN] Frame capture failed, skipping.")
            continue

        frame = cv2.flip(frame, 1)  # Mirror effect
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        stable_out = None
        candidate  = None
        cand_conf  = 0.0

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
            hand_lm = results.multi_hand_landmarks[0]
            draw_landmarks(frame, hand_lm)

            # Extract raw coords
            coords = []
            for lm in hand_lm.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            raw = np.array(coords, dtype=np.float32)

            # Normalize
            norm = normalize_sample(raw)

            # Predict
            label, confidence = predictor.predict(norm)
            candidate = label
            cand_conf = confidence

            # Smooth
            stable_out = smoother.update(label, confidence)
        else:
            smoother.reset()

        # FPS
        now  = time.time()
        fps  = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        overlay_prediction(frame, stable_out, candidate, cand_conf, fps)
        cv2.imshow("ASL Alphabet — Live Inference", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):  # ESC or Q
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("  Webcam closed.")


def main():
    parser = argparse.ArgumentParser(description='Live ASL webcam inference')
    parser.add_argument('--camera_id',    type=int, default=0,    help='OpenCV camera index')
    parser.add_argument('--model_path',   default=None,            help='Path to best_model.pt')
    parser.add_argument('--encoder_path', default=None,            help='Path to label_encoder.pkl')
    args = parser.parse_args()

    run_webcam(args.camera_id, args.model_path, args.encoder_path)


if __name__ == '__main__':
    main()
