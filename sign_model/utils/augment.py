"""
sign_model/utils/augment.py
----------------------------------------------------------------------
V2 augmentation library for sign language landmark sequences.

Landmark coordinate layout (225 features per frame):
  [0:63]    Left hand  — 21 keypoints × (x,y,z), indices 0,3,6,...60 = x
  [63:126]  Right hand — 21 keypoints × (x,y,z), indices 63,66,...123 = x
  [126:225] Pose body  — 33 keypoints × (x,y,z), indices 126,129,...222 = x

Wrist anchors:
  Left wrist  → indices 0 (x), 1 (y), 2 (z)
  Right wrist → indices 63 (x), 64 (y), 65 (z)

All functions:
  Input:  np.ndarray  shape (64, 225)  float32
  Output: np.ndarray  shape (64, 225)  float32

Original v1 augmentations are preserved in utils/augment_v1_backup.py
----------------------------------------------------------------------
"""

import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────

LH_START, LH_END = 0, 63       # Left hand block
RH_START, RH_END = 63, 126     # Right hand block
POSE_START, POSE_END = 126, 225  # Pose block

LH_WRIST_X, LH_WRIST_Y = 0, 1
RH_WRIST_X, RH_WRIST_Y = 63, 64

TARGET_FRAMES = 64


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Horizontal Flip
# ═══════════════════════════════════════════════════════════════════════════════

def horizontal_flip(seq: np.ndarray) -> np.ndarray:
    """
    Mirror all x-coordinates (x → 1 - x) and swap left/right hand blocks.
    x values sit at every 3rd index starting from 0 within each block.
    Swapping hand blocks corrects for the change in handedness after mirroring.
    """
    out = seq.copy()
    # Flip all x values across the full feature vector
    x_indices = np.arange(0, out.shape[1], 3)  # 0, 3, 6, ..., 222
    out[:, x_indices] = 1.0 - out[:, x_indices]
    # Swap left-hand and right-hand blocks (handedness correction)
    lh = out[:, LH_START:LH_END].copy()
    rh = out[:, RH_START:RH_END].copy()
    out[:, LH_START:LH_END] = rh
    out[:, RH_START:RH_END] = lh
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Temporal Jitter
# ═══════════════════════════════════════════════════════════════════════════════

def temporal_jitter(
    seq: np.ndarray,
    max_jitter: int = 8,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Randomly shift the center-crop window by ±max_jitter frames.
    If the sequence is already at target length, shift by inserting padding.
    Always outputs exactly TARGET_FRAMES frames.
    """
    if rng is None:
        rng = np.random.default_rng()

    T = seq.shape[0]
    shift = int(rng.integers(-max_jitter, max_jitter + 1))

    if shift == 0:
        return seq.copy()

    if shift > 0:
        # Shift right: drop first `shift` frames, pad end with zeros
        out = np.zeros_like(seq)
        valid = max(0, T - shift)
        out[:valid] = seq[shift:shift + valid]
    else:
        # Shift left: drop last `|shift|` frames, pad start with zeros
        shift = -shift
        out = np.zeros_like(seq)
        valid = max(0, T - shift)
        out[shift:shift + valid] = seq[:valid]

    return out.astype(seq.dtype)


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Speed Warp
# ═══════════════════════════════════════════════════════════════════════════════

def speed_warp(seq: np.ndarray, rate: float = 1.0) -> np.ndarray:
    """
    Resample sequence at `rate` speed using linear interpolation.
    rate < 1.0 → slow down (fewer source frames used → stretch to fill target)
    rate > 1.0 → speed up (more source frames compressed into target)
    Always outputs exactly TARGET_FRAMES frames.
    """
    T = seq.shape[0]
    F = seq.shape[1]

    # Number of source frames to sample at this rate
    src_len = max(2, int(round(T * rate)))
    # Source sample positions (evenly spaced across original T frames)
    src_positions = np.linspace(0, T - 1, src_len)

    # Interpolate each feature dimension independently
    orig_idx = np.arange(T, dtype=float)
    resampled = np.stack(
        [np.interp(src_positions, orig_idx, seq[:, f]) for f in range(F)],
        axis=1,
    )  # shape: (src_len, F)

    # Now resize back to TARGET_FRAMES
    out_positions = np.linspace(0, src_len - 1, TARGET_FRAMES)
    src_idx = np.arange(src_len, dtype=float)
    out = np.stack(
        [np.interp(out_positions, src_idx, resampled[:, f]) for f in range(F)],
        axis=1,
    )  # shape: (TARGET_FRAMES, F)

    return out.astype(seq.dtype)


# ═══════════════════════════════════════════════════════════════════════════════
#  4. Gaussian Noise
# ═══════════════════════════════════════════════════════════════════════════════

def gaussian_noise(
    seq: np.ndarray,
    std: float = 0.008,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Add zero-mean Gaussian noise to all coordinates.
    Result is clipped to [0, 1] to keep coordinates valid.
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, std, size=seq.shape).astype(seq.dtype)
    return np.clip(seq + noise, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  5. Wrist-Anchored Rotation
# ═══════════════════════════════════════════════════════════════════════════════

def wrist_anchored_rotation(
    seq: np.ndarray,
    max_deg: float = 20.0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    For each hand independently:
    - Extract wrist (x, y) as rotation anchor
    - Apply random 2D rotation in [-max_deg, +max_deg] degrees
    - Leave z-coordinates unchanged
    - Only rotate hands that were detected (non-zero wrist)
    """
    if rng is None:
        rng = np.random.default_rng()

    out = seq.copy()

    for (start, end, wx_idx, wy_idx) in [
        (LH_START, LH_END, LH_WRIST_X, LH_WRIST_Y),
        (RH_START, RH_END, RH_WRIST_X, RH_WRIST_Y),
    ]:
        # Skip if hand not detected (wrist is zero across all frames)
        wrist_x = out[:, wx_idx]
        wrist_y = out[:, wy_idx]
        if np.all(wrist_x == 0) and np.all(wrist_y == 0):
            continue

        angle = rng.uniform(-max_deg, max_deg)
        rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)

        # x/y indices within this hand block
        local_x_idx = np.arange(start, end, 3)   # absolute indices: start, start+3, ...
        local_y_idx = local_x_idx + 1

        wx = out[:, wx_idx:wx_idx + 1]   # (T, 1) broadcast-friendly
        wy = out[:, wy_idx:wy_idx + 1]

        dx = out[:, local_x_idx] - wx   # (T, 21)
        dy = out[:, local_y_idx] - wy

        out[:, local_x_idx] = wx + cos_a * dx - sin_a * dy
        out[:, local_y_idx] = wy + sin_a * dx + cos_a * dy

    # Clip to valid range
    out[:, :RH_END] = np.clip(out[:, :RH_END], 0.0, 1.0)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  6. Wrist-Anchored Scale
# ═══════════════════════════════════════════════════════════════════════════════

def wrist_anchored_scale(
    seq: np.ndarray,
    scale_range: tuple = (0.85, 1.15),
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    For each hand independently:
    - Extract wrist (x, y) as scale anchor
    - Apply a random uniform scale factor
    - Scale all keypoints relative to wrist: new = wrist + (old - wrist) * scale
    - Only scale detected hands
    - Clip to [0, 1]
    """
    if rng is None:
        rng = np.random.default_rng()

    out = seq.copy()

    for (start, end, wx_idx, wy_idx) in [
        (LH_START, LH_END, LH_WRIST_X, LH_WRIST_Y),
        (RH_START, RH_END, RH_WRIST_X, RH_WRIST_Y),
    ]:
        wrist_x = out[:, wx_idx]
        wrist_y = out[:, wy_idx]
        if np.all(wrist_x == 0) and np.all(wrist_y == 0):
            continue

        scale = float(rng.uniform(scale_range[0], scale_range[1]))

        local_x_idx = np.arange(start, end, 3)
        local_y_idx = local_x_idx + 1

        wx = out[:, wx_idx:wx_idx + 1]
        wy = out[:, wy_idx:wy_idx + 1]

        out[:, local_x_idx] = wx + (out[:, local_x_idx] - wx) * scale
        out[:, local_y_idx] = wy + (out[:, local_y_idx] - wy) * scale

    out[:, :RH_END] = np.clip(out[:, :RH_END], 0.0, 1.0)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  7. Body Scale Transform
# ═══════════════════════════════════════════════════════════════════════════════

def body_scale_transform(
    seq: np.ndarray,
    x_range: tuple = (0.88, 1.12),
    y_range: tuple = (0.88, 1.12),
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Scale pose landmarks to simulate different signer body proportions.
    - Find pose center (mean x, mean y of non-zero pose keypoints)
    - Apply independent x_scale and y_scale
    - Clip to [0, 1]
    """
    if rng is None:
        rng = np.random.default_rng()

    out = seq.copy()

    pose_block = out[:, POSE_START:POSE_END]  # (T, 99)
    pose_x_idx = np.arange(POSE_START, POSE_END, 3)   # absolute
    pose_y_idx = pose_x_idx + 1

    # Compute pose center from non-zero keypoints
    px = out[:, pose_x_idx]   # (T, 33)
    py = out[:, pose_y_idx]

    nonzero_mask = (px != 0) | (py != 0)   # (T, 33)
    if not nonzero_mask.any():
        return out   # No pose detected; skip

    # Per-frame center — suppress warning for frames where all keypoints are zero
    cx = np.where(nonzero_mask, px, np.nan)
    cy = np.where(nonzero_mask, py, np.nan)
    with np.errstate(all='ignore'):
        center_x = np.nanmean(cx, axis=1, keepdims=True)   # (T, 1)
        center_y = np.nanmean(cy, axis=1, keepdims=True)
    # Frames with no pose get center = 0.5 (image midpoint) — no scaling effect
    center_x = np.where(np.isnan(center_x), 0.5, center_x)
    center_y = np.where(np.isnan(center_y), 0.5, center_y)

    x_scale = float(rng.uniform(x_range[0], x_range[1]))
    y_scale = float(rng.uniform(y_range[0], y_range[1]))

    out[:, pose_x_idx] = center_x + (px - center_x) * x_scale
    out[:, pose_y_idx] = center_y + (py - center_y) * y_scale

    out[:, POSE_START:POSE_END] = np.clip(out[:, POSE_START:POSE_END], 0.0, 1.0)
    # Final NaN guard
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  8. Mixup
# ═══════════════════════════════════════════════════════════════════════════════

def mixup(
    seq1: np.ndarray,
    seq2: np.ndarray,
    alpha: float = 0.3,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Linear interpolation between two same-class sequences.
    lambda ~ Beta(alpha, alpha); output = lambda * seq1 + (1 - lambda) * seq2
    Both sequences must be shape (64, 225).
    """
    if rng is None:
        rng = np.random.default_rng()

    lam = float(rng.beta(alpha, alpha))
    # Keep lambda closer to 1 so seq1 dominates (preserve original label)
    lam = max(lam, 1.0 - lam)
    return (lam * seq1 + (1.0 - lam) * seq2).astype(seq1.dtype)


# ═══════════════════════════════════════════════════════════════════════════════
#  9. Temporal CutMix
# ═══════════════════════════════════════════════════════════════════════════════

def temporal_cutmix(
    seq1: np.ndarray,
    seq2: np.ndarray,
    cut_frames: int = 32,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Replace a contiguous segment of seq1 with the same segment from seq2.
    - Cut point is chosen randomly within [16, 48]
    - Returns: seq1[:cut] + seq2[cut:cut+cut_frames] + seq1[cut+cut_frames:]
    - If cut + cut_frames > 64, segment is trimmed to fit.
    Both sequences must be shape (64, 225).
    """
    if rng is None:
        rng = np.random.default_rng()

    T = seq1.shape[0]
    max_start = max(1, T - cut_frames)
    cut_start = int(rng.integers(max(1, 16), min(48, max_start) + 1))
    cut_end = min(cut_start + cut_frames, T)

    out = seq1.copy()
    out[cut_start:cut_end] = seq2[cut_start:cut_end]
    return out.astype(seq1.dtype)


# ═══════════════════════════════════════════════════════════════════════════════
#  10. Compose Random — main entry point
# ═══════════════════════════════════════════════════════════════════════════════

_BASIC_AUG_POOL = [
    "horizontal_flip",
    "temporal_jitter",
    "speed_warp_slow1",    # rate 0.75
    "speed_warp_slow2",    # rate 0.85
    "speed_warp_fast1",    # rate 1.15
    "speed_warp_fast2",    # rate 1.25
    "gaussian_noise",
]


def compose_random(
    seq: np.ndarray,
    config: dict,
    all_class_sequences: list = None,
    seed: int = None,
) -> np.ndarray:
    """
    Apply a random combination of augmentations based on config.

    Parameters
    ----------
    seq                 : np.ndarray (64, 225) — the sequence to augment
    config              : augmentation sub-dict from config_v2.yaml
    all_class_sequences : list of np.ndarray — ALL sequences of the same class
                          (required for mixup / cutmix)
    seed                : int — random seed for reproducibility

    Returns
    -------
    np.ndarray (64, 225) — augmented sequence
    """
    rng = np.random.default_rng(seed)
    out = seq.copy()

    noise_std          = float(config.get("noise_std", 0.008))
    jitter_frames      = int(config.get("temporal_jitter_frames", 8))
    speed_rates        = list(config.get("speed_rates", [0.75, 0.85, 1.15, 1.25]))
    wrist_rot_deg      = float(config.get("wrist_rotation_max_deg", 20))
    wrist_scale_range  = tuple(config.get("wrist_scale_range", [0.85, 1.15]))
    body_x_range       = tuple(config.get("body_scale_x_range", [0.88, 1.12]))
    body_y_range       = tuple(config.get("body_scale_y_range", [0.88, 1.12]))
    mixup_alpha        = float(config.get("mixup_alpha", 0.3))
    cutmix_frames      = int(config.get("cutmix_frames", 32))
    enable_mixup       = bool(config.get("enable_mixup", True))
    enable_cutmix      = bool(config.get("enable_cutmix", True))
    enable_wrist       = bool(config.get("enable_wrist_transform", True))
    enable_body        = bool(config.get("enable_body_scale", True))

    # Rate map for speed_warp names
    rate_map = {
        "speed_warp_slow1": speed_rates[0] if len(speed_rates) > 0 else 0.75,
        "speed_warp_slow2": speed_rates[1] if len(speed_rates) > 1 else 0.85,
        "speed_warp_fast1": speed_rates[2] if len(speed_rates) > 2 else 1.15,
        "speed_warp_fast2": speed_rates[3] if len(speed_rates) > 3 else 1.25,
    }

    # ── Step 1: Apply 2–3 basic augmentations ────────────────────────────────
    n_basic = int(rng.integers(2, 4))   # 2 or 3
    chosen_basic = rng.choice(_BASIC_AUG_POOL, size=min(n_basic, len(_BASIC_AUG_POOL)),
                               replace=False)

    for aug in chosen_basic:
        if aug == "horizontal_flip":
            out = horizontal_flip(out)
        elif aug == "temporal_jitter":
            out = temporal_jitter(out, max_jitter=jitter_frames, rng=rng)
        elif aug in ("speed_warp_slow1", "speed_warp_slow2",
                     "speed_warp_fast1", "speed_warp_fast2"):
            out = speed_warp(out, rate=rate_map[aug])
        elif aug == "gaussian_noise":
            out = gaussian_noise(out, std=noise_std, rng=rng)

    # ── Step 2: Spatial transforms ────────────────────────────────────────────
    if enable_wrist:
        if rng.random() < 0.60:
            out = wrist_anchored_rotation(out, max_deg=wrist_rot_deg, rng=rng)
        if rng.random() < 0.60:
            out = wrist_anchored_scale(out, scale_range=wrist_scale_range, rng=rng)

    if enable_body:
        if rng.random() < 0.50:
            out = body_scale_transform(out, x_range=body_x_range,
                                       y_range=body_y_range, rng=rng)

    # ── Step 3: Cross-sample augmentations ───────────────────────────────────
    if all_class_sequences is not None and len(all_class_sequences) >= 2:
        # Pick a different sample from the same class
        other_idx = int(rng.integers(0, len(all_class_sequences)))
        other_seq = all_class_sequences[other_idx]

        if enable_mixup and rng.random() < 0.30:
            out = mixup(out, other_seq, alpha=mixup_alpha, rng=rng)

        if enable_cutmix and rng.random() < 0.20:
            out = temporal_cutmix(out, other_seq, cut_frames=cutmix_frames, rng=rng)

    # Final safety: eliminate any NaN/Inf that slipped through any augmentation
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    return out.astype(np.float32)


# ── V1 compatibility shim ──────────────────────────────────────────────────────

def apply_random_augmentations(
    sequence: np.ndarray,
    n: int = 2,
    target_frames: int = 64,
    noise_std: float = 0.01,
    temporal_jitter_frames: int = 5,
    speed_rates: list = None,
    seed: int = None,
) -> np.ndarray:
    """
    V1-compatible wrapper preserved for backward compatibility.
    Calls the V1 augmentations only (no spatial/mixup transforms).
    """
    if speed_rates is None:
        speed_rates = [0.85, 1.15]

    rng = np.random.default_rng(seed)
    out = sequence.copy()

    pool = ["horizontal_flip", "temporal_jitter", "speed_slow", "speed_fast", "gaussian_noise"]
    chosen = rng.choice(pool, size=min(n, len(pool)), replace=False)

    for aug in chosen:
        if aug == "horizontal_flip":
            out = horizontal_flip(out)
        elif aug == "temporal_jitter":
            out = temporal_jitter(out, max_jitter=temporal_jitter_frames, rng=rng)
        elif aug == "speed_slow":
            out = speed_warp(out, rate=speed_rates[0])
        elif aug == "speed_fast":
            out = speed_warp(out, rate=speed_rates[1] if len(speed_rates) > 1 else 1.15)
        elif aug == "gaussian_noise":
            out = gaussian_noise(out, std=noise_std, rng=rng)

    return out
