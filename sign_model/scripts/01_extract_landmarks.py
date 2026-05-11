"""
==============================================================================
sign_model/scripts/01_extract_landmarks.py
==============================================================================
WHAT    : Reads raw .mp4 videos from dataset_50_words/, applies class
          filtering (min/max samples, exclude list), extracts MediaPipe
          Holistic landmarks frame-by-frame, and saves them as .npy files.

INPUT   : ../dataset_50_words/{CLASS}/{video}.mp4

OUTPUT  : sign_model/data/raw_landmarks/{CLASS}/{video}.npy
          sign_model/label_map.json
          Each .npy shape: (num_frames, 225)
          225 = (21 left-hand + 21 right-hand + 33 pose) × 3 (x,y,z)

RUN     : cd sign_model && python scripts/01_extract_landmarks.py
==============================================================================
"""

import sys
import os

# ── Ensure we can import from sign_model/utils regardless of working dir ──
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))   # sign_model/scripts/
MODULE_ROOT = os.path.dirname(SCRIPT_DIR)                   # sign_model/
sys.path.insert(0, MODULE_ROOT)

import json
import random
import numpy as np
import cv2
import mediapipe as mp
import yaml
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
#  Config loading
# ──────────────────────────────────────────────────────────────────────────────

CONFIG_PATH = os.path.join(MODULE_ROOT, "configs", "config.yaml")

try:
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
except FileNotFoundError:
    print(f"[ERROR] Config file not found: {CONFIG_PATH}")
    sys.exit(1)

# Paths
DATASET_DIR   = os.path.normpath(os.path.join(MODULE_ROOT, cfg["dataset"]["source_dir"]))
LANDMARKS_DIR = os.path.join(MODULE_ROOT, cfg["paths"]["landmarks_dir"])
LABEL_MAP_OUT = os.path.join(MODULE_ROOT, cfg["paths"]["label_map"])

# Filtering params
MIN_SAMPLES     = cfg["dataset"]["min_samples"]
MAX_SAMPLES     = cfg["dataset"]["max_samples"]
EXCLUDE_CLASSES = set(cfg["dataset"]["exclude_classes"])
SEED            = cfg["dataset"]["random_seed"]

random.seed(SEED)
np.random.seed(SEED)


# ──────────────────────────────────────────────────────────────────────────────
#  Class Filtering
# ──────────────────────────────────────────────────────────────────────────────

def apply_class_filter(dataset_dir: str) -> dict:
    """
    Scan dataset_dir, apply min/max/exclude rules, return:
    {class_name: [selected_video_paths]}
    Also prints a summary table.
    """
    if not os.path.isdir(dataset_dir):
        print(f"[ERROR] Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    all_classes = sorted(os.listdir(dataset_dir))
    results = {}

    print("\n── Class Filtering Summary ─────────────────────────────────────────")
    print(f"  {'Class':<20}  {'Original':>8}  {'Final':>6}  Status")
    print("  " + "-" * 55)

    for cls in all_classes:
        cls_path = os.path.join(dataset_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        all_videos = [
            os.path.join(cls_path, f)
            for f in sorted(os.listdir(cls_path))
            if f.lower().endswith(".mp4")
        ]
        orig_count = len(all_videos)

        # Exclusion rule
        if cls in EXCLUDE_CLASSES:
            print(f"  {cls:<20}  {orig_count:>8}  {'0':>6}  EXCLUDED (blacklist)")
            continue

        # Min-samples rule
        if orig_count < MIN_SAMPLES:
            print(f"  {cls:<20}  {orig_count:>8}  {'0':>6}  EXCLUDED (< {MIN_SAMPLES} samples)")
            continue

        # Max-samples cap
        if orig_count > MAX_SAMPLES:
            rng = random.Random(SEED)
            selected = rng.sample(all_videos, MAX_SAMPLES)
            selected.sort()
            print(f"  {cls:<20}  {orig_count:>8}  {MAX_SAMPLES:>6}  CAPPED (→ {MAX_SAMPLES})")
        else:
            selected = all_videos
            print(f"  {cls:<20}  {orig_count:>8}  {orig_count:>6}  INCLUDED")

        results[cls] = selected

    print(f"\n  Total included classes : {len(results)}")
    print(f"  Total videos to process: {sum(len(v) for v in results.values())}\n")
    return results


# ──────────────────────────────────────────────────────────────────────────────
#  Landmark Extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_landmarks_from_video(
    video_path: str,
    holistic,
    frame_width: int = 640,
    frame_height: int = 480,
) -> np.ndarray:
    """
    Extract MediaPipe Holistic landmarks from every frame of a video.

    Returns
    -------
    np.ndarray of shape (num_frames, 225)
    225 = (21 left + 21 right + 33 pose) × 3 (x,y,z)
    Missing landmarks are filled with zeros.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frame_vectors = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        # Convert BGR → RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = holistic.process(rgb_frame)
        rgb_frame.flags.writeable = True

        # ── Left Hand: 21 × 3 = 63 ──
        if results.left_hand_landmarks:
            left_hand = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
                dtype=np.float32,
            ).flatten()  # (63,)
        else:
            left_hand = np.zeros(63, dtype=np.float32)

        # ── Right Hand: 21 × 3 = 63 ──
        if results.right_hand_landmarks:
            right_hand = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
                dtype=np.float32,
            ).flatten()
        else:
            right_hand = np.zeros(63, dtype=np.float32)

        # ── Pose: 33 × 3 = 99 ──
        if results.pose_landmarks:
            pose = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark],
                dtype=np.float32,
            ).flatten()  # (99,)
        else:
            pose = np.zeros(99, dtype=np.float32)

        # Concatenate: 63 + 63 + 99 = 225
        frame_vec = np.concatenate([left_hand, right_hand, pose])  # (225,)

        # Normalize x values by frame width, y by frame height
        # Feature layout per landmark: (x, y, z) repeating
        # x indices: 0, 3, 6, ...; y indices: 1, 4, 7, ...
        frame_vec[0::3] = np.clip(frame_vec[0::3], 0.0, 1.0)   # x already [0,1] from MediaPipe
        frame_vec[1::3] = np.clip(frame_vec[1::3], 0.0, 1.0)   # y already [0,1] from MediaPipe
        # z is relative depth — keep as-is

        frame_vectors.append(frame_vec)

    cap.release()

    if not frame_vectors:
        raise ValueError(f"No frames extracted from: {video_path}")

    return np.stack(frame_vectors, axis=0)  # (T, 225)


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Sign Model — Step 01: Landmark Extraction")
    print("=" * 70)
    print(f"  Dataset source : {DATASET_DIR}")
    print(f"  Output dir     : {LANDMARKS_DIR}")
    print(f"  Min samples    : {MIN_SAMPLES}")
    print(f"  Max samples    : {MAX_SAMPLES}")
    print(f"  Excluded       : {EXCLUDE_CLASSES}")

    # Apply class filter
    class_videos = apply_class_filter(DATASET_DIR)
    if not class_videos:
        print("[ERROR] No classes passed the filter. Exiting.")
        sys.exit(1)

    # Build sorted class list → label map
    class_names = sorted(class_videos.keys())
    label_map = {cls: idx for idx, cls in enumerate(class_names)}

    # Save label_map.json
    os.makedirs(os.path.dirname(LABEL_MAP_OUT), exist_ok=True)
    try:
        with open(LABEL_MAP_OUT, "w") as f:
            json.dump({"label_map": label_map, "class_names": class_names}, f, indent=2)
        print(f"\n  Saved label map → {LABEL_MAP_OUT}")
    except Exception as e:
        print(f"[ERROR] Could not save label_map.json: {e}")
        sys.exit(1)

    # Initialise MediaPipe Holistic
    mp_holistic = mp.solutions.holistic

    # Extraction stats
    total_extracted = 0
    total_failed    = 0
    class_frame_avgs = {}

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        for cls in class_names:
            video_paths = class_videos[cls]
            out_dir     = os.path.join(LANDMARKS_DIR, cls)
            os.makedirs(out_dir, exist_ok=True)

            frame_counts = []
            failed       = []

            for vpath in tqdm(video_paths, desc=f"  {cls:<20}", unit="video", leave=True):
                video_name = os.path.splitext(os.path.basename(vpath))[0]
                out_path   = os.path.join(out_dir, video_name + ".npy")

                try:
                    landmarks = extract_landmarks_from_video(vpath, holistic)
                    np.save(out_path, landmarks)
                    frame_counts.append(landmarks.shape[0])
                    total_extracted += 1
                except Exception as exc:
                    failed.append((vpath, str(exc)))
                    total_failed += 1

            avg_frames = np.mean(frame_counts) if frame_counts else 0
            class_frame_avgs[cls] = round(avg_frames, 1)

            if failed:
                print(f"\n  [WARN] {cls}: {len(failed)} failed extraction(s):")
                for fp, err in failed:
                    print(f"    {os.path.basename(fp)}: {err}")

    # Summary
    print("\n" + "=" * 70)
    print("  Extraction Complete")
    print("=" * 70)
    print(f"  Total extracted : {total_extracted}")
    print(f"  Total failed    : {total_failed}")
    print(f"\n  Avg frames per class:")
    for cls, avg in class_frame_avgs.items():
        print(f"    {cls:<20} {avg:>7.1f} frames")
    print()


if __name__ == "__main__":
    main()
