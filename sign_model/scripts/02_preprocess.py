"""
==============================================================================
sign_model/scripts/02_preprocess.py
==============================================================================
WHAT    : Reads raw .npy landmark files, normalises sequences to 64 frames,
          applies StandardScaler, performs stratified train/val/test split,
          augments training data, and saves everything as dataset.npz.

INPUT   : sign_model/data/raw_landmarks/{CLASS}/*.npy
          sign_model/label_map.json

OUTPUT  : sign_model/data/processed/dataset.npz
          sign_model/data/splits/{train,val,test}_indices.json
          sign_model/models/scaler.pkl

RUN     : cd sign_model && python scripts/02_preprocess.py
==============================================================================
"""

import sys
import os

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODULE_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, MODULE_ROOT)

import json
import pickle
import numpy as np
import yaml
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.augment import apply_random_augmentations

# ──────────────────────────────────────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────────────────────────────────────

CONFIG_PATH = os.path.join(MODULE_ROOT, "configs", "config.yaml")

try:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
except FileNotFoundError:
    print(f"[ERROR] Config not found: {CONFIG_PATH}")
    sys.exit(1)

LANDMARKS_DIR = os.path.join(MODULE_ROOT, cfg["paths"]["landmarks_dir"])
PROCESSED_DIR = os.path.join(MODULE_ROOT, cfg["paths"]["processed_dir"])
SPLITS_DIR    = os.path.join(MODULE_ROOT, cfg["paths"]["splits_dir"])
LABEL_MAP_PATH = os.path.join(MODULE_ROOT, cfg["paths"]["label_map"])
SCALER_PATH   = os.path.join(MODULE_ROOT, "models", "scaler.pkl")

TARGET_FRAMES = cfg["preprocessing"]["target_frames"]
FEATURE_DIM   = cfg["preprocessing"]["feature_dim"]
TRAIN_SPLIT   = cfg["preprocessing"]["train_split"]
VAL_SPLIT     = cfg["preprocessing"]["val_split"]
SEED          = cfg["dataset"]["random_seed"]

AUG_ENABLED   = cfg["augmentation"]["enabled"]
AUG_PER_SAMPLE = cfg["augmentation"]["augmentations_per_sample"]
NOISE_STD     = cfg["augmentation"]["noise_std"]
SPEED_RATES   = cfg["augmentation"]["speed_rates"]
JITTER_FRAMES = cfg["augmentation"]["temporal_jitter_frames"]

np.random.seed(SEED)

# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def normalize_sequence(seq: np.ndarray, target: int = 64) -> np.ndarray:
    """Center-crop if too long, zero-pad if too short. Output shape: (target, F)."""
    T = len(seq)
    if T > target:
        start = (T - target) // 2
        return seq[start: start + target]
    elif T < target:
        pad = np.zeros((target - T, seq.shape[1]), dtype=seq.dtype)
        return np.concatenate([seq, pad], axis=0)
    return seq


# ──────────────────────────────────────────────────────────────────────────────
#  Load raw landmarks
# ──────────────────────────────────────────────────────────────────────────────

def load_all_landmarks(landmarks_dir: str, label_map: dict, class_names: list):
    """
    Walk landmarks_dir/{class}/*.npy, load arrays, normalize to TARGET_FRAMES.

    Returns
    -------
    X : np.ndarray (N, TARGET_FRAMES, FEATURE_DIM)
    y : np.ndarray (N,)
    file_records : list of {filepath, label_index, class_name}
    """
    X, y, file_records = [], [], []

    for cls in tqdm(class_names, desc="  Loading classes", unit="class"):
        cls_dir = os.path.join(landmarks_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"  [WARN] Missing class dir: {cls_dir} — skipping")
            continue

        npy_files = sorted(
            f for f in os.listdir(cls_dir) if f.endswith(".npy")
        )
        if not npy_files:
            print(f"  [WARN] No .npy files in {cls_dir} — skipping")
            continue

        label_idx = label_map[cls]

        for nf in npy_files:
            fpath = os.path.join(cls_dir, nf)
            try:
                raw = np.load(fpath).astype(np.float32)  # (T, 225)
                seq = normalize_sequence(raw, TARGET_FRAMES)
                X.append(seq)
                y.append(label_idx)
                file_records.append({
                    "filepath": os.path.relpath(fpath, MODULE_ROOT),
                    "label_index": label_idx,
                    "class_name": cls,
                })
            except Exception as exc:
                print(f"  [WARN] Failed to load {fpath}: {exc}")

    if not X:
        print("[ERROR] No sequences loaded. Did you run 01_extract_landmarks.py?")
        sys.exit(1)

    return np.stack(X), np.array(y, dtype=np.int64), file_records


# ──────────────────────────────────────────────────────────────────────────────
#  Train/Val/Test Split
# ──────────────────────────────────────────────────────────────────────────────

def stratified_split(X, y, file_records, train_frac, val_frac, seed):
    """
    Stratified split preserving class ratios.
    Returns indices for train, val, test.
    """
    indices = np.arange(len(X))
    test_frac = 1.0 - train_frac - val_frac

    # First split: train vs (val + test)
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        indices, y,
        test_size=(val_frac + test_frac),
        stratify=y,
        random_state=seed,
    )

    # Split temp into val and test
    val_of_temp = val_frac / (val_frac + test_frac)
    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=1.0 - val_of_temp,
        stratify=y_temp,
        random_state=seed,
    )

    return idx_train, idx_val, idx_test


# ──────────────────────────────────────────────────────────────────────────────
#  Augmentation on training set
# ──────────────────────────────────────────────────────────────────────────────

def augment_training_set(X_train, y_train, aug_per_sample, seed):
    """
    Create `aug_per_sample` augmented copies of every training sample.
    Returns concatenated (original + augmented) arrays.
    """
    aug_X, aug_y = [], []

    for i in tqdm(range(len(X_train)), desc="  Augmenting train", unit="sample"):
        seq   = X_train[i]
        label = y_train[i]
        for k in range(aug_per_sample):
            aug_seq = apply_random_augmentations(
                seq,
                n=2,
                target_frames=TARGET_FRAMES,
                noise_std=NOISE_STD,
                temporal_jitter_frames=JITTER_FRAMES,
                speed_rates=SPEED_RATES,
                seed=seed + i * 100 + k,  # unique seed per sample+aug
            )
            # Ensure shape is (TARGET_FRAMES, FEATURE_DIM)
            aug_seq = normalize_sequence(aug_seq, TARGET_FRAMES)
            aug_X.append(aug_seq.astype(np.float32))
            aug_y.append(label)

    aug_X = np.stack(aug_X)
    aug_y = np.array(aug_y, dtype=np.int64)

    X_out = np.concatenate([X_train, aug_X], axis=0)
    y_out = np.concatenate([y_train, aug_y], axis=0)
    return X_out, y_out


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Sign Model — Step 02: Preprocessing")
    print("=" * 70)

    # Load label map
    try:
        with open(LABEL_MAP_PATH) as f:
            lm_data = json.load(f)
        label_map   = lm_data["label_map"]
        class_names = lm_data["class_names"]
    except FileNotFoundError:
        print(f"[ERROR] label_map.json not found at {LABEL_MAP_PATH}")
        print("  → Run: python scripts/01_extract_landmarks.py first")
        sys.exit(1)

    print(f"  Classes      : {len(class_names)}")
    print(f"  Target frames: {TARGET_FRAMES}")
    print(f"  Feature dim  : {FEATURE_DIM}")

    # Load all landmarks
    print("\n── Loading raw landmarks ──────────────────────────────────────────")
    X, y, file_records = load_all_landmarks(LANDMARKS_DIR, label_map, class_names)
    print(f"  Loaded {len(X)} sequences, shape: {X.shape}")

    # Stratified split
    print("\n── Splitting dataset ──────────────────────────────────────────────")
    idx_train, idx_val, idx_test = stratified_split(X, y, file_records, TRAIN_SPLIT, VAL_SPLIT, SEED)

    X_train_raw = X[idx_train];  y_train_raw = y[idx_train]
    X_val       = X[idx_val];    y_val       = y[idx_val]
    X_test      = X[idx_test];   y_test      = y[idx_test]

    print(f"  Train: {len(X_train_raw):>5}  |  Val: {len(X_val):>5}  |  Test: {len(X_test):>5}")

    # Save split index files
    os.makedirs(SPLITS_DIR, exist_ok=True)
    splits = {
        "train": idx_train.tolist(),
        "val":   idx_val.tolist(),
        "test":  idx_test.tolist(),
    }
    for split_name, indices in splits.items():
        out_path = os.path.join(SPLITS_DIR, f"{split_name}_indices.json")
        records  = [file_records[i] for i in indices]
        try:
            with open(out_path, "w") as f:
                json.dump(records, f, indent=2)
        except Exception as e:
            print(f"[WARN] Could not save {out_path}: {e}")

    print(f"  Split index files saved to {SPLITS_DIR}")

    # Augmentation on training set
    if AUG_ENABLED:
        print(f"\n── Augmenting training set (×{AUG_PER_SAMPLE} copies/sample) ────────────────")
        X_train, y_train = augment_training_set(X_train_raw, y_train_raw, AUG_PER_SAMPLE, SEED)
        print(f"  Train size after augmentation: {len(X_train)}")
    else:
        X_train, y_train = X_train_raw, y_train_raw

    # StandardScaler — fit on training data only
    print("\n── Fitting StandardScaler on training set ────────────────────────")
    N_tr, T_tr, F_tr = X_train.shape
    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, F_tr)
    scaler.fit(X_train_2d)

    # Transform all splits
    X_train = scaler.transform(X_train.reshape(-1, F_tr)).reshape(N_tr, T_tr, F_tr).astype(np.float32)
    X_val   = scaler.transform(X_val.reshape(-1, F_tr)).reshape(len(X_val), T_tr, F_tr).astype(np.float32)
    X_test  = scaler.transform(X_test.reshape(-1, F_tr)).reshape(len(X_test), T_tr, F_tr).astype(np.float32)

    # Save scaler
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    try:
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)
        print(f"  Scaler saved → {SCALER_PATH}")
    except Exception as e:
        print(f"[WARN] Could not save scaler: {e}")

    # Save compressed dataset
    print("\n── Saving processed dataset ──────────────────────────────────────")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_npz = os.path.join(PROCESSED_DIR, "dataset.npz")
    try:
        np.savez_compressed(
            out_npz,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            class_names=np.array(class_names),
        )
        print(f"  Saved → {out_npz}")
    except Exception as e:
        print(f"[ERROR] Could not save dataset.npz: {e}")
        sys.exit(1)

    # Final statistics
    print("\n── Final Dataset Statistics ──────────────────────────────────────")
    print(f"  X_train : {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_val   : {X_val.shape}  y_val  : {y_val.shape}")
    print(f"  X_test  : {X_test.shape}  y_test : {y_test.shape}")

    from collections import Counter
    print("\n  Class distribution in train split (first 10):")
    dist = Counter(y_train.tolist())
    for idx in sorted(dist)[:10]:
        print(f"    [{idx:>3}] {class_names[idx]:<20}: {dist[idx]:>4}")

    print("\n  Preprocessing complete ✅")


if __name__ == "__main__":
    main()
