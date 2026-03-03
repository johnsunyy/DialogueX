"""
preprocessing/extract_landmarks.py

Extracts MediaPipe hand landmarks from every A-Z image in the dataset.
Saves processed/X.npy (shape [N, 63]) and processed/y.npy (shape [N,]).
Also saves label_encoder.pkl mapping int -> letter.

Usage:
    python -m preprocessing.extract_landmarks [--data_dir PATH] [--out_dir PATH] [--limit N]
"""

import os
import sys
import pickle
import argparse
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# Only A-Z letters
VALID_CLASSES = [chr(c) for c in range(ord('A'), ord('Z') + 1)]


def extract_landmarks_from_image(image_bgr, hands_detector):
    """
    Given a BGR image and an initialised MediaPipe Hands detector,
    return a (63,) numpy array of (x, y, z) landmarks or None on failure.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None  # No hand detected

    if len(results.multi_hand_landmarks) != 1:
        return None  # Multiple hands — discard

    hand = results.multi_hand_landmarks[0]
    coords = []
    for lm in hand.landmark:
        coords.extend([lm.x, lm.y, lm.z])

    return np.array(coords, dtype=np.float32)  # (63,)


def run_extraction(data_dir: str, out_dir: str, limit: int = None):
    """
    Walk through data_dir/{A..Z}/ and extract landmarks from every image.
    """
    print(f"\n{'='*60}")
    print(f"  MediaPipe ASL Landmark Extraction")
    print(f"{'='*60}")
    print(f"  Dataset : {data_dir}")
    print(f"  Output  : {out_dir}")
    print(f"  Limit   : {limit if limit else 'None (all images)'}")
    print(f"{'='*60}\n")

    os.makedirs(out_dir, exist_ok=True)

    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )

    X_list = []
    y_list = []
    label_encoder = LabelEncoder()
    label_encoder.fit(VALID_CLASSES)

    total_processed = 0
    total_failed = 0
    per_class_counts = {}

    for letter in VALID_CLASSES:
        class_dir = os.path.join(data_dir, letter)
        if not os.path.isdir(class_dir):
            print(f"  [WARN] Missing class directory: {class_dir}")
            continue

        image_files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]

        if limit:
            image_files = image_files[:limit]

        label_int = label_encoder.transform([letter])[0]
        class_success = 0
        class_fail = 0

        for fname in tqdm(image_files, desc=f"  {letter}", leave=False):
            fpath = os.path.join(class_dir, fname)
            img = cv2.imread(fpath)
            if img is None:
                class_fail += 1
                continue

            landmarks = extract_landmarks_from_image(img, hands_detector)
            if landmarks is None:
                class_fail += 1
                total_failed += 1
                continue

            X_list.append(landmarks)
            y_list.append(label_int)
            class_success += 1
            total_processed += 1

        per_class_counts[letter] = class_success
        print(f"  {letter}: {class_success:>5} extracted | {class_fail:>4} failed")

    hands_detector.close()

    # Save
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    x_path = os.path.join(out_dir, 'X.npy')
    y_path = os.path.join(out_dir, 'y.npy')
    np.save(x_path, X)
    np.save(y_path, y)

    # Save label encoder alongside data
    le_path = os.path.join(os.path.dirname(os.path.dirname(out_dir)), 'label_encoder.pkl')
    with open(le_path, 'wb') as f:
        pickle.dump(label_encoder, f)

    print(f"\n{'='*60}")
    print(f"  EXTRACTION COMPLETE")
    print(f"  Total extracted : {total_processed:,}")
    print(f"  Total failed    : {total_failed:,}")
    print(f"  X shape         : {X.shape}")
    print(f"  y shape         : {y.shape}")
    print(f"  Saved X.npy     : {x_path}")
    print(f"  Saved y.npy     : {y_path}")
    print(f"  Saved encoder   : {le_path}")
    print(f"{'='*60}\n")

    return X, y, label_encoder


def main():
    parser = argparse.ArgumentParser(description='Extract MediaPipe hand landmarks from ASL dataset')
    parser.add_argument(
        '--data_dir',
        default=os.path.join(os.path.dirname(__file__), '..', '..', 'ASL_Alphabet_Dataset', 'asl_alphabet_train'),
        help='Path to asl_alphabet_train directory'
    )
    parser.add_argument(
        '--out_dir',
        default=os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'),
        help='Output directory for X.npy and y.npy'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit images per class (useful for quick smoke tests)'
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    out_dir = os.path.abspath(args.out_dir)
    run_extraction(data_dir, out_dir, args.limit)


if __name__ == '__main__':
    main()
