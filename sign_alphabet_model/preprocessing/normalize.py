"""
preprocessing/normalize.py

Wrist-centered, scale-invariant normalization for MediaPipe hand landmarks.

Each sample is a (63,) flat array of (x, y, z) for 21 landmarks.
Normalization steps:
  1. Reshape to (21, 3).
  2. Subtract wrist landmark (index 0) from all landmarks.
  3. Compute max pairwise Euclidean distance between any two of the 21 landmarks.
  4. Divide all coordinates by that max distance.
  5. Flatten back to (63,).

Result: position-invariant, scale-invariant, camera-distance-invariant features.
"""

import numpy as np
from itertools import combinations


def normalize_sample(landmarks_63: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Normalize a single 63-dimensional landmark vector.

    Args:
        landmarks_63: np.ndarray of shape (63,) — raw (x, y, z) coords.
        epsilon: small constant to prevent division by zero.

    Returns:
        np.ndarray of shape (63,) — normalized landmarks.
    """
    pts = landmarks_63.reshape(21, 3).astype(np.float64)

    # Step 1: Wrist-center (subtract landmark 0)
    wrist = pts[0].copy()
    pts -= wrist  # landmark 0 is now (0, 0, 0)

    # Step 2: Compute max pairwise Euclidean distance
    # Use vectorised approach for speed
    # pts shape: (21, 3)
    # Compute all pairwise squared distances via broadcasting
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (21, 21, 3)
    sq_dists = np.sum(diff ** 2, axis=-1)                   # (21, 21)
    max_dist = np.sqrt(sq_dists.max())

    # Step 3: Scale normalisation
    pts /= (max_dist + epsilon)

    return pts.flatten().astype(np.float32)


def normalize_dataset(X: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    Apply normalize_sample to every row in X.

    Args:
        X: np.ndarray of shape (N, 63) — raw landmark samples.
        verbose: print progress.

    Returns:
        np.ndarray of shape (N, 63) — normalised samples.
    """
    N = X.shape[0]
    X_norm = np.empty_like(X, dtype=np.float32)

    for i in range(N):
        X_norm[i] = normalize_sample(X[i])

        if verbose and (i + 1) % 10_000 == 0:
            print(f"  Normalised {i+1:,}/{N:,} samples...")

    if verbose:
        print(f"  Normalisation complete. Shape: {X_norm.shape}")

    return X_norm
