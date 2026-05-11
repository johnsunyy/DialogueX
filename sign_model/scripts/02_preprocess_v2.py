"""
==============================================================================
sign_model/scripts/02_preprocess_v2.py
==============================================================================
WHAT    : V2 preprocessing — reuses existing raw_landmarks/ and split indices,
          applies 6 augmented copies per training sample using the upgraded
          augment.py (wrist rotation, body scale, mixup, cutmix, etc.).
          Saves output to data/processed/dataset_v2.npz.

          IMPORTANT: Does NOT touch dataset.npz, models/final/, or any v1 file.

INPUT   : sign_model/data/raw_landmarks/{CLASS}/*.npy   (already extracted)
          sign_model/data/splits/{train,val,test}_indices.json  (reused as-is)
          sign_model/label_map.json
          sign_model/configs/config_v2.yaml  (or --config argument)

OUTPUT  : sign_model/data/processed/dataset_v2.npz
          sign_model/models/scaler_v2.pkl

RUN     : cd sign_model && python scripts/02_preprocess_v2.py
          cd sign_model && python scripts/02_preprocess_v2.py --config configs/config_v2.yaml
==============================================================================
"""

import sys
import os
import argparse

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODULE_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, MODULE_ROOT)

import json
import pickle
import numpy as np
import yaml
from collections import Counter
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from utils.augment import compose_random

# ------------------------------------------------------------------------------
#  CLI
# ------------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="V2 Preprocessing")
    p.add_argument(
        "--config",
        default=os.path.join(MODULE_ROOT, "configs", "config_v2.yaml"),
        help="Path to config YAML (default: configs/config_v2.yaml)",
    )
    return p.parse_args()


# ------------------------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------------------------

def normalize_sequence(seq: np.ndarray, target: int = 64) -> np.ndarray:
    """Center-crop if too long, zero-pad if too short. Output: (target, F)."""
    T = len(seq)
    if T > target:
        start = (T - target) // 2
        return seq[start: start + target]
    elif T < target:
        pad = np.zeros((target - T, seq.shape[1]), dtype=seq.dtype)
        return np.concatenate([seq, pad], axis=0)
    return seq.copy()


def load_all_landmarks(landmarks_dir, label_map, class_names, target_frames):
    """Load and normalize all .npy landmark files -> (N, T, F), labels, file_records."""
    X, y, file_records = [], [], []

    for cls in tqdm(class_names, desc="  Loading classes", unit="class"):
        cls_dir = os.path.join(landmarks_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"  [WARN] Missing class dir: {cls_dir} — skipping")
            continue

        npy_files = sorted(f for f in os.listdir(cls_dir) if f.endswith(".npy"))
        if not npy_files:
            print(f"  [WARN] No .npy files in {cls_dir} — skipping")
            continue

        label_idx = label_map[cls]
        for nf in npy_files:
            fpath = os.path.join(cls_dir, nf)
            try:
                raw = np.load(fpath).astype(np.float32)
                seq = normalize_sequence(raw, target_frames)
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
        print("[ERROR] No sequences loaded.")
        sys.exit(1)

    return np.stack(X), np.array(y, dtype=np.int64), file_records


def load_split_indices(splits_dir):
    """
    Reload existing train/val/test split index files.
    Returns (idx_train, idx_val, idx_test) as numpy int arrays.
    Each JSON is a list of {filepath, label_index, class_name} records.
    We reconstruct integer indices by matching filepath order in file_records.
    """
    def load_records(path):
        with open(path) as f:
            return json.load(f)

    train_records = load_records(os.path.join(splits_dir, "train_indices.json"))
    val_records   = load_records(os.path.join(splits_dir, "val_indices.json"))
    test_records  = load_records(os.path.join(splits_dir, "test_indices.json"))
    return train_records, val_records, test_records


def records_to_indices(split_records, file_records):
    """
    Map split file records back to integer indices in the loaded array.
    Matches on 'filepath' field.
    """
    filepath_to_idx = {r["filepath"]: i for i, r in enumerate(file_records)}
    indices = []
    for r in split_records:
        fp = r["filepath"]
        if fp in filepath_to_idx:
            indices.append(filepath_to_idx[fp])
        else:
            print(f"  [WARN] Split record not found in loaded data: {fp}")
    return np.array(indices, dtype=np.int64)


# ------------------------------------------------------------------------------
#  V2 Augmentation
# ------------------------------------------------------------------------------

def augment_training_set_v2(X_train, y_train, class_names, aug_cfg, aug_per_sample, seed):
    """
    Create `aug_per_sample` augmented copies of every training sample.
    For mixup/cutmix, passes all same-class sequences as context.
    Returns (X_aug, y_aug) — augmented copies ONLY (not including originals).
    """
    num_classes = len(class_names)

    # Group indices by class for mixup/cutmix
    class_to_indices = {i: [] for i in range(num_classes)}
    for idx, lbl in enumerate(y_train):
        class_to_indices[int(lbl)].append(idx)

    aug_X, aug_y = [], []
    tech_counts = Counter()

    print(f"  Augmenting {len(X_train)} training samples x {aug_per_sample} copies...")
    for i in tqdm(range(len(X_train)), desc="  Augmenting", unit="sample"):
        seq   = X_train[i]
        label = int(y_train[i])

        # Collect all same-class sequences (excluding self)
        same_class_seqs = [X_train[j] for j in class_to_indices[label] if j != i]

        for k in range(aug_per_sample):
            unique_seed = seed + i * 1000 + k * 7
            aug_seq = compose_random(
                seq,
                config=aug_cfg,
                all_class_sequences=same_class_seqs if same_class_seqs else None,
                seed=unique_seed,
            )
            aug_seq = normalize_sequence(aug_seq, aug_cfg.get("target_frames", 64))
            aug_X.append(aug_seq.astype(np.float32))
            aug_y.append(label)

    aug_X = np.stack(aug_X)
    aug_y = np.array(aug_y, dtype=np.int64)
    return aug_X, aug_y


# ------------------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 70)
    print("  Sign Model V2 — Step 02: Preprocessing")
    print("=" * 70)
    print(f"  Config: {args.config}")

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    LANDMARKS_DIR  = os.path.join(MODULE_ROOT, cfg["paths"]["landmarks_dir"])
    SPLITS_DIR     = os.path.join(MODULE_ROOT, cfg["paths"]["splits_dir"])
    PROCESSED_DIR  = os.path.join(MODULE_ROOT, cfg["paths"]["processed_dir"])
    LABEL_MAP_PATH = os.path.join(MODULE_ROOT, cfg["paths"]["label_map"])
    OUT_NPZ        = os.path.join(PROCESSED_DIR, cfg["paths"]["processed_filename"])
    SCALER_V2_PATH = os.path.join(MODULE_ROOT, "models", "scaler_v2.pkl")

    TARGET_FRAMES     = int(cfg["preprocessing"]["target_frames"])
    AUG_PER_SAMPLE    = int(cfg["augmentation"]["augmentations_per_sample"])
    SEED              = int(cfg["dataset"]["random_seed"])
    AUG_ENABLED       = bool(cfg["augmentation"]["enabled"])

    # Safety check: never overwrite dataset.npz
    v1_npz = os.path.join(PROCESSED_DIR, "dataset.npz")
    if OUT_NPZ == v1_npz:
        print("[ERROR] Output path matches dataset.npz — refusing to overwrite V1 data.")
        sys.exit(1)

    np.random.seed(SEED)

    # Load label map
    with open(LABEL_MAP_PATH) as f:
        lm_data = json.load(f)
    label_map   = lm_data["label_map"]
    class_names = lm_data["class_names"]
    print(f"  Classes: {len(class_names)}  |  Target frames: {TARGET_FRAMES}")

    # Load all raw landmarks
    print("\n-- Loading raw landmarks ------------------------------------------")
    X, y, file_records = load_all_landmarks(LANDMARKS_DIR, label_map, class_names, TARGET_FRAMES)
    print(f"  Loaded {len(X)} sequences  shape={X.shape}")

    # Reload existing split indices for fair v1 vs v2 comparison
    print("\n-- Reloading existing split indices -------------------------------")
    train_recs, val_recs, test_recs = load_split_indices(SPLITS_DIR)
    idx_train = records_to_indices(train_recs, file_records)
    idx_val   = records_to_indices(val_recs,   file_records)
    idx_test  = records_to_indices(test_recs,  file_records)
    print(f"  Train raw: {len(idx_train)}  |  Val: {len(idx_val)}  |  Test: {len(idx_test)}")

    X_train_raw = X[idx_train];  y_train_raw = y[idx_train]
    X_val       = X[idx_val];    y_val       = y[idx_val]
    X_test      = X[idx_test];   y_test      = y[idx_test]

    # V2 Augmentation
    if AUG_ENABLED:
        aug_cfg = dict(cfg["augmentation"])
        aug_cfg["target_frames"] = TARGET_FRAMES

        print(f"\n-- V2 Augmentation (x{AUG_PER_SAMPLE} copies/sample) ----------------------")
        print(f"  Techniques enabled:")
        print(f"    Basic:            horizontal_flip, temporal_jitter, speed_warp, gaussian_noise")
        print(f"    Wrist rotation:   {aug_cfg.get('enable_wrist_transform', True)} (max {aug_cfg.get('wrist_rotation_max_deg',20)}°)")
        print(f"    Wrist scale:      {aug_cfg.get('enable_wrist_transform', True)} (range {aug_cfg.get('wrist_scale_range',[0.85,1.15])})")
        print(f"    Body scale:       {aug_cfg.get('enable_body_scale', True)} (x={aug_cfg.get('body_scale_x_range')}, y={aug_cfg.get('body_scale_y_range')})")
        print(f"    Mixup:            {aug_cfg.get('enable_mixup', True)} (alpha={aug_cfg.get('mixup_alpha',0.3)}, p=0.30)")
        print(f"    Temporal cutmix:  {aug_cfg.get('enable_cutmix', True)} (frames={aug_cfg.get('cutmix_frames',32)}, p=0.20)")

        aug_X, aug_y = augment_training_set_v2(
            X_train_raw, y_train_raw, class_names, aug_cfg, AUG_PER_SAMPLE, SEED
        )
        X_train = np.concatenate([X_train_raw, aug_X], axis=0)
        y_train = np.concatenate([y_train_raw, aug_y], axis=0)

        v1_train_size = 8013  # known from v1 run
        print(f"\n  V1 training size : {v1_train_size}")
        print(f"  V2 training size : {len(X_train)}  ({len(X_train)/v1_train_size:.2f}x more samples)")
        print(f"  Augmented copies : {len(aug_X)}")
    else:
        X_train, y_train = X_train_raw, y_train_raw
        print("  Augmentation disabled.")

    # Fit StandardScaler on augmented training data
    print("\n-- Fitting V2 StandardScaler -------------------------------------")
    N_tr, T_tr, F_tr = X_train.shape
    scaler_v2 = StandardScaler()
    scaler_v2.fit(X_train.reshape(-1, F_tr))

    X_train = scaler_v2.transform(X_train.reshape(-1, F_tr)).reshape(N_tr, T_tr, F_tr).astype(np.float32)
    X_val   = scaler_v2.transform(X_val.reshape(-1, F_tr)).reshape(len(X_val), T_tr, F_tr).astype(np.float32)
    X_test  = scaler_v2.transform(X_test.reshape(-1, F_tr)).reshape(len(X_test), T_tr, F_tr).astype(np.float32)

    os.makedirs(os.path.dirname(SCALER_V2_PATH), exist_ok=True)
    with open(SCALER_V2_PATH, "wb") as f:
        pickle.dump(scaler_v2, f)
    print(f"  V2 Scaler saved -> {SCALER_V2_PATH}")

    # Save dataset_v2.npz
    print("\n-- Saving dataset_v2.npz -----------------------------------------")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        class_names=np.array(class_names),
    )
    size_mb = os.path.getsize(OUT_NPZ) / (1024 * 1024)
    print(f"  Saved -> {OUT_NPZ}  ({size_mb:.1f} MB)")

    # Final stats
    print("\n-- Final Dataset Statistics --------------------------------------")
    print(f"  X_train : {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_val   : {X_val.shape}  y_val  : {y_val.shape}")
    print(f"  X_test  : {X_test.shape}  y_test : {y_test.shape}")

    train_dist = Counter(y_train.tolist())
    print(f"\n  Train class distribution (first 10 of {len(class_names)}):")
    for idx in sorted(train_dist)[:10]:
        print(f"    [{idx:>3}] {class_names[idx]:<20}: {train_dist[idx]:>5}")

    print("\n  V2 Preprocessing complete ✅")


if __name__ == "__main__":
    main()
