"""
extract_landmarks.py — Phase 3: Feature Extraction Pipeline

Extracts MediaPipe Holistic landmarks from all videos in the NSLT-300 dataset.
Output: data/features/{WORD_LABEL}/{video_id}.npy  — shape (30, 162)

Feature vector (162 per frame):
  - Right hand : 21 landmarks × 3 (x,y,z) = 63
  - Left hand  : 21 landmarks × 3 (x,y,z) = 63
  - Pose (upper body, 12 keypoints) × 3    = 36
  Total = 162

Normalization:
  - Translate so shoulder midpoint is origin
  - Scale by shoulder width (L11-L12 distance)
  - Depth (z) normalized independently

Usage:
  python preprocessing/extract_landmarks.py            # extract all
  python preprocessing/extract_landmarks.py --limit 5  # smoke test
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mediapipe_lib
import numpy as np

# ─── Configuration ───────────────────────────────────────────────────────────
MODULE_DIR   = Path(__file__).resolve().parent.parent     # word_model_nslt300/
ROOT         = MODULE_DIR.parent                          # d:\haha
NSLT_JSON    = ROOT / "nslt_300.json"
WLASL_JSON   = ROOT / "WLASL_v0.3.json"
VIDEOS_DIR   = ROOT / "videos"
FEATURES_DIR = MODULE_DIR / "data" / "features"

SEQ_LEN      = 40
N_FEATURES   = 162

# 12 upper-body pose landmark indices (MediaPipe Pose):
# 11=L_shoulder, 12=R_shoulder, 13=L_elbow, 14=R_elbow
# 15=L_wrist, 16=R_wrist, 23=L_hip, 24=R_hip
# 0=nose, 1=L_eye_inner, 2=L_eye, 4=R_eye_inner, 5=R_eye, 7=L_ear, 8=R_ear
POSE_INDICES = [0, 1, 2, 4, 5, 7, 8, 11, 12, 13, 14, 15, 16]  # 13 × 3 = 39? use 12:
POSE_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 0, 7, 8, 5]   # exactly 12 = 36

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("extract")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _lm_to_array(landmarks, indices=None):
    """Convert landmark list to numpy (N, 3). If indices given, select subset."""
    if landmarks is None:
        if indices is not None:
            return np.zeros((len(indices), 3), dtype=np.float32)
        return None
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)
    if indices is not None:
        pts = pts[indices]
    return pts


def normalize_frame(right_hand, left_hand, pose_upper):
    """
    Normalize landmarks relative to shoulder midpoint and shoulder width.
    Args:
        right_hand  : (21, 3) or zeros
        left_hand   : (21, 3) or zeros
        pose_upper  : (12, 3)
    Returns:
        flat vector of shape (162,)
    """
    # Shoulders are indices 0,1 in POSE_INDICES (11=L, 12=R → index 0,1)
    l_shoulder = pose_upper[0]   # landmark 11
    r_shoulder = pose_upper[1]   # landmark 12

    midpoint = (l_shoulder + r_shoulder) / 2.0
    width    = np.linalg.norm(r_shoulder[:2] - l_shoulder[:2]) + 1e-6
    depth_scale = width  # use same scale for z

    def _norm(pts):
        pts = pts - midpoint
        pts[:, :2] /= width
        pts[:, 2]  /= depth_scale
        return pts

    rh = _norm(right_hand.copy())
    lh = _norm(left_hand.copy())
    pu = _norm(pose_upper.copy())

    return np.concatenate([rh.flatten(), lh.flatten(), pu.flatten()])  # (162,)


def sample_frames(cap, action):
    """
    Extract relevant frames from video using action[1] (start) and action[2] (end).
    For videos with action=[cls, 1, N]: extract entire video frames.
    Returns list of BGR frames.
    """
    start_frame = max(0, action[1] - 1)
    end_frame   = action[2] - 1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame    = min(end_frame, total_frames - 1)

    if start_frame > end_frame:
        start_frame = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(end_frame - start_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames


def temporal_resample(frames, target_len=SEQ_LEN):
    """Uniform temporal sampling to target_len frames."""
    n = len(frames)
    if n == 0:
        return []
    indices = np.linspace(0, n - 1, target_len, dtype=int)
    return [frames[i] for i in indices]


def extract_video(args):
    """
    Worker function. args = (video_id, video_path, action, label_dir)
    Returns (video_id, success, error_message)
    """
    video_id, video_path, action, label_dir = args
    out_path = label_dir / f"{video_id}.npy"

    if out_path.exists():
        return (video_id, True, "cached")

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return (video_id, False, "cannot open video")

        frames = sample_frames(cap, action)
        cap.release()

        if len(frames) == 0:
            return (video_id, False, "no frames extracted")

        # Temporal resampling → exactly SEQ_LEN frames
        if len(frames) >= SEQ_LEN:
            frames = temporal_resample(frames, SEQ_LEN)
        # else: will zero-pad after landmark extraction

        # ── MediaPipe inference ───────────────────────────────
        mp_holistic = mediapipe_lib.solutions.holistic
        sequence    = np.zeros((SEQ_LEN, N_FEATURES), dtype=np.float32)

        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:
            for i, frame in enumerate(frames):
                if i >= SEQ_LEN:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = holistic.process(rgb)

                right_hand = _lm_to_array(res.right_hand_landmarks) \
                    if res.right_hand_landmarks else np.zeros((21, 3), dtype=np.float32)
                left_hand  = _lm_to_array(res.left_hand_landmarks) \
                    if res.left_hand_landmarks else np.zeros((21, 3), dtype=np.float32)
                pose_upper = _lm_to_array(res.pose_landmarks, indices=POSE_INDICES) \
                    if res.pose_landmarks else np.zeros((len(POSE_INDICES), 3), dtype=np.float32)

                feat = normalize_frame(right_hand, left_hand, pose_upper)
                sequence[i] = feat

        label_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), sequence)
        return (video_id, True, "ok")

    except Exception as e:
        return (video_id, False, str(e))


# ─── Main extraction pipeline ────────────────────────────────────────────────

def build_task_list(limit=None):
    """Build list of (video_id, video_path, action, label_dir) for all videos."""
    with open(NSLT_JSON, "r") as f:
        nslt = json.load(f)
    with open(WLASL_JSON, "r") as f:
        wlasl = json.load(f)

    idx2gloss = {i: entry["gloss"] for i, entry in enumerate(wlasl)}

    tasks = []
    for vid_id, info in nslt.items():
        video_path = VIDEOS_DIR / f"{vid_id}.mp4"
        if not video_path.exists():
            continue

        cls_idx   = info["action"][0]
        action    = info["action"]
        gloss     = idx2gloss.get(cls_idx, f"class_{cls_idx}")
        # Sanitize gloss for directory name
        safe_gloss = "".join(c if c.isalnum() or c in "-_" else "_" for c in gloss)
        label_dir  = FEATURES_DIR / safe_gloss

        tasks.append((vid_id, video_path, action, label_dir))
        if limit and len(tasks) >= limit:
            break

    return tasks


def run_extraction(limit=None, num_workers=None, timeout=90):
    """
    Extract landmarks using multiprocessing.Pool with full worker isolation.
    - maxtasksperchild=1  : each worker process handles exactly 1 video then exits
                            → a crashed/hung worker NEVER affects other videos
    - apply_async + .get(timeout) : per-video hard timeout
    - Already-saved .npy files are skipped automatically (cached)
    """
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    tasks = build_task_list(limit=limit)
    total = len(tasks)

    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    log.info(f"Starting extraction: {total} videos, {num_workers} workers, timeout={timeout}s/video")
    t0      = time.time()
    done    = 0
    success = 0
    cached  = 0
    failed  = []

    # maxtasksperchild=1 → every worker is a fresh process; a hang/crash is fully contained
    with mp.Pool(processes=num_workers, maxtasksperchild=1) as pool:
        async_results = []
        for task in tasks:
            ar = pool.apply_async(extract_video, args=(task,))
            async_results.append((task[0], ar))   # (video_id, AsyncResult)

        for vid_id, ar in async_results:
            done += 1
            try:
                _, ok, msg = ar.get(timeout=timeout)
                if ok:
                    success += 1
                    if msg == "cached":
                        cached += 1
                else:
                    failed.append((vid_id, msg))
            except mp.TimeoutError:
                failed.append((vid_id, "TIMEOUT"))
                log.warning(f"  TIMEOUT: {vid_id} exceeded {timeout}s — skipped")
            except Exception as e:
                failed.append((vid_id, str(e)[:80]))

            if done % 50 == 0 or done == total:
                elapsed = time.time() - t0
                rate    = done / elapsed if elapsed > 0 else 0
                eta     = (total - done) / rate if rate > 0 else 0
                log.info(
                    f"  [{done:5d}/{total}] "
                    f"ok={success} cached={cached} fail={len(failed)} "
                    f"| {rate:.1f} vid/s | ETA {eta/60:.1f} min"
                )

    elapsed = time.time() - t0
    log.info(f"\nExtraction complete in {elapsed/60:.1f} min")
    log.info(f"  Success : {success}")
    log.info(f"  Cached  : {cached}")
    log.info(f"  Failed  : {len(failed)}")

    if failed:
        log.warning(f"Failed/timed-out videos ({len(failed)}):")
        for vid_id, err in failed[:20]:
            log.warning(f"  {vid_id}: {err}")

    # Quick shape validation on a sample
    sample_files = list(FEATURES_DIR.rglob("*.npy"))[:5]
    for fp in sample_files:
        arr = np.load(str(fp))
        assert arr.shape == (SEQ_LEN, N_FEATURES), \
            f"Shape mismatch {arr.shape} in {fp}"
        log.info(f"  Shape check ✓ {fp.name}: {arr.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract MediaPipe landmarks from WLASL videos")
    parser.add_argument("--limit",   type=int, default=None, help="Limit number of videos (smoke test)")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    args = parser.parse_args()

    run_extraction(limit=args.limit, num_workers=args.workers)
