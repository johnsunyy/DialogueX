"""
preprocessing/split_dataset.py

Loads X.npy and y.npy from processed/, applies normalization,
then performs a stratified 80/10/10 split.

Saves:
    processed/X_train.npy, X_val.npy, X_test.npy
    processed/y_train.npy, y_val.npy, y_test.npy

Usage:
    python -m preprocessing.split_dataset [--processed_dir PATH]
"""

import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing.normalize import normalize_dataset


def run_split(processed_dir: str):
    print(f"\n{'='*60}")
    print(f"  Stratified Dataset Split  (80 / 10 / 10)")
    print(f"{'='*60}")

    x_path = os.path.join(processed_dir, 'X.npy')
    y_path = os.path.join(processed_dir, 'y.npy')

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(
            f"X.npy / y.npy not found in {processed_dir}\n"
            "Run `python -m preprocessing.extract_landmarks` first."
        )

    print(f"  Loading raw landmarks from {processed_dir} ...")
    X = np.load(x_path)
    y = np.load(y_path)
    print(f"  Loaded X: {X.shape}, y: {y.shape}")

    # Normalize
    print(f"  Normalising landmarks...")
    X_norm = normalize_dataset(X, verbose=True)

    # Split: 80 / 10 / 10
    # First split off 20% as temp (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_norm, y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )
    # Split the 20% into 10% val and 10% test (50/50 of the 20%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42
    )

    print(f"\n  Split results:")
    print(f"  Train : X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"  Val   : X_val={X_val.shape},   y_val={y_val.shape}")
    print(f"  Test  : X_test={X_test.shape},  y_test={y_test.shape}")

    # Verify class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\n  Samples per class in train: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")

    # Save splits
    splits = {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
    }
    for name, arr in splits.items():
        path = os.path.join(processed_dir, f'{name}.npy')
        np.save(path, arr)
        print(f"  Saved: {path}")

    print(f"\n  Done.\n{'='*60}\n")
    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    parser = argparse.ArgumentParser(description='Stratified split of processed ASL landmark data')
    parser.add_argument(
        '--processed_dir',
        default=os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'),
        help='Directory containing X.npy and y.npy'
    )
    args = parser.parse_args()
    processed_dir = os.path.abspath(args.processed_dir)
    run_split(processed_dir)


if __name__ == '__main__':
    main()
