"""
extract_landmarks.py — Phase 3: Feature Extraction (324 features/frame)

Feature vector per frame = 324:
  - Position  : right_hand(63) + left_hand(63) + upper_pose(36) = 162
  - Velocity  : frame_t - frame_(t-1) for same 162 features = 162
  Total = 324

Sequence: 40 frames (uniform sample if longer, zero-pad if shorter)
Output  : data/features/{WORD}/{video_id}.npy  shape=(40, 324)

Isolation:
  - maxtasksperchild=1  → each worker is a fresh process
  - apply_async + 90s timeout → hung videos skipped automatically
  - failed_videos.txt → all skipped video IDs logged

Run:
  python preprocessing/extract_landmarks.py
  python preprocessing/extract_landmarks.py --limit 5   # smoke test
"""

import argparse
import json
import logging
import multiprocessing as mp
import time
from pathlib import Path

import cv2
import mediapipe as mediapipe_lib
import numpy as np

# ─── Configuration ───────────────────────────────────────────────────────────
MODULE_DIR   = Path(__file__).resolve().parent.parent
ROOT         = MODULE_DIR.parent
NSLT_JSON    = ROOT / "nslt_100.json"
WLASL_JSON   = ROOT / "WLASL_v0.3.json"
VIDEOS_DIR   = ROOT / "videos"
FEATURES_DIR = MODULE_DIR / "data" / "features"
FAILED_LOG   = MODULE_DIR / "data" / "failed_videos.txt"

SEQ_LEN      = 40
N_POS        = 162   # position features
N_FEATURES   = 324   # position + velocity

# 12 upper-body pose landmark indices
POSE_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 0, 7, 8, 5]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("extract")


# ─── Landmark helpers ─────────────────────────────────────────────────────────

def _lm_to_array(landmarks, indices=None):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)
    return pts[indices] if indices is not None else pts


def normalize_landmarks(right_hand, left_hand, pose_upper):
    """
    Normalize all landmarks relative to shoulder midpoint / shoulder width.
    Returns flat (162,) position vector.
    """
    l_shoulder = pose_upper[0]   # POSE_INDICES[0] = landmark 11
    r_shoulder = pose_upper[1]   # POSE_INDICES[1] = landmark 12

    center  = (l_shoulder + r_shoulder) / 2.0
    scale   = np.linalg.norm(r_shoulder[:2] - l_shoulder[:2]) + 1e-6
    z_scale = scale

    def _norm(pts):
        p = pts - center
        p[:, :2] /= scale
        p[:, 2]  /= z_scale
        return p

    rh = _norm(right_hand.copy())
    lh = _norm(left_hand.copy())
    pu = _norm(pose_upper.copy())
    return np.concatenate([rh.flatten(), lh.flatten(), pu.flatten()])  # (162,)


def extract_position_sequence(frames):
    """Run MediaPipe on frames, return position sequence (N, 162)."""
    mp_h = mediapipe_lib.solutions.holistic
    positions = []
    with mp_h.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)

            rh = _lm_to_array(res.right_hand_landmarks) \
                if res.right_hand_landmarks else np.zeros((21, 3), dtype=np.float32)
            lh = _lm_to_array(res.left_hand_landmarks) \
                if res.left_hand_landmarks else np.zeros((21, 3), dtype=np.float32)
            pu = _lm_to_array(res.pose_landmarks, indices=POSE_INDICES) \
                if res.pose_landmarks else np.zeros((len(POSE_INDICES), 3), dtype=np.float32)

            positions.append(normalize_landmarks(rh, lh, pu))

    return np.array(positions, dtype=np.float32)  # (N, 162)


def build_sequence(positions):
    """
    Temporal resample to SEQ_LEN then add velocity features.
    Returns (SEQ_LEN, 324).
    """
    N = len(positions)
    if N == 0:
        return np.zeros((SEQ_LEN, N_FEATURES), dtype=np.float32)

    # ── Temporal resampling ───────────────────────────────
    if N >= SEQ_LEN:
        idx = np.linspace(0, N - 1, SEQ_LEN, dtype=int)
        pos = positions[idx]                          # (40, 162)
    else:
        pad = np.zeros((SEQ_LEN - N, N_POS), dtype=np.float32)
        pos = np.vstack([positions, pad])              # (40, 162)

    # ── Velocity features ─────────────────────────────────
    vel = np.zeros_like(pos)                           # (40, 162)
    vel[1:] = pos[1:] - pos[:-1]                      # vel[0] = 0 (no prev frame)

    return np.concatenate([pos, vel], axis=1)          # (40, 324)


def load_frames(video_path, action):
    """Load frames from video using action start/end."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_f = max(0, action[1] - 1)
    end_f   = min(action[2] - 1, total - 1)
    if start_f > end_f:
        start_f = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    frames = []
    for _ in range(end_f - start_f + 1):
        ret, fr = cap.read()
        if not ret:
            break
        frames.append(fr)
    cap.release()
    return frames


# ─── Worker ───────────────────────────────────────────────────────────────────

def extract_video(args):
    """
    Worker: extract one video → .npy file.
    args = (video_id, video_path, action, out_path)
    Returns (video_id, success, message)
    """
    video_id, video_path, action, out_path = args
    out_path = Path(out_path)

    if out_path.exists():
        return (video_id, True, "cached")

    try:
        frames = load_frames(video_path, action)
        if not frames:
            return (video_id, False, "no frames")

        positions = extract_position_sequence(frames)
        sequence  = build_sequence(positions)

        assert sequence.shape == (SEQ_LEN, N_FEATURES), \
            f"Shape error: {sequence.shape}"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), sequence)
        return (video_id, True, "ok")

    except Exception as e:
        return (video_id, False, str(e)[:120])


# ─── Task list ────────────────────────────────────────────────────────────────

def build_task_list(limit=None):
    with open(NSLT_JSON) as f:
        nslt = json.load(f)
    with open(WLASL_JSON) as f:
        wlasl = json.load(f)
    idx2gloss = {i: e["gloss"] for i, e in enumerate(wlasl)}

    tasks = []
    for vid_id, info in nslt.items():
        video_path = VIDEOS_DIR / f"{vid_id}.mp4"
        if not video_path.exists():
            continue

        cls_idx  = info["action"][0]
        action   = info["action"]
        gloss    = idx2gloss.get(cls_idx, f"class_{cls_idx}")
        safe     = "".join(c if c.isalnum() or c in "-_" else "_" for c in gloss)
        out_path = FEATURES_DIR / safe / f"{vid_id}.npy"

        tasks.append((vid_id, video_path, action, out_path))
        if limit and len(tasks) >= limit:
            break

    return tasks


# ─── Extraction loop ──────────────────────────────────────────────────────────

def run_extraction(limit=None, num_workers=None, timeout=90):
    """
    Isolated extraction: maxtasksperchild=1 + apply_async timeout.
    A hung or crashed video CANNOT affect others.
    """
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    tasks = build_task_list(limit=limit)
    total = len(tasks)

    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    log.info(f"Extraction: {total} videos | {num_workers} workers | timeout={timeout}s/video")
    t0 = time.time()
    done = success = cached = 0
    failed = []

    with mp.Pool(processes=num_workers, maxtasksperchild=1) as pool:
        async_results = [
            (task[0], pool.apply_async(extract_video, args=(task,)))
            for task in tasks
        ]

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
                log.warning(f"  TIMEOUT: {vid_id} — skipped")
            except Exception as e:
                failed.append((vid_id, str(e)[:80]))

            if done % 50 == 0 or done == total:
                elapsed = time.time() - t0
                rate    = done / elapsed if elapsed > 0 else 0
                eta     = (total - done) / rate if rate > 0 else 0
                log.info(
                    f"  [{done:5d}/{total}] ok={success} cached={cached} "
                    f"fail={len(failed)} | {rate:.1f}/s | ETA {eta/60:.1f}min"
                )

    elapsed = time.time() - t0
    log.info(f"\nDone in {elapsed/60:.1f}min  success={success}  cached={cached}  failed={len(failed)}")

    # Log failed videos
    if failed:
        FAILED_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(FAILED_LOG, "w") as f:
            for vid_id, err in failed:
                f.write(f"{vid_id}\t{err}\n")
        log.warning(f"  Failed log: {FAILED_LOG}")
        for vid_id, err in failed[:10]:
            log.warning(f"    {vid_id}: {err}")

    # Shape validation
    samples = list(FEATURES_DIR.rglob("*.npy"))[:5]
    for fp in samples:
        arr = np.load(str(fp))
        assert arr.shape == (SEQ_LEN, N_FEATURES), f"Bad shape {arr.shape}: {fp}"
        log.info(f"  Shape ✓ {fp.name}: {arr.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",   type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()
    run_extraction(limit=args.limit, num_workers=args.workers, timeout=args.timeout)
