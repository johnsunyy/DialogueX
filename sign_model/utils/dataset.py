"""
sign_model/utils/dataset.py
----------------------------------------------------------------------
PyTorch Dataset class for sign language landmark sequences.

The dataset loads pre-processed .npz arrays from disk. Supports
training (with augmentation) and eval modes.

Usage:
  from utils.dataset import SignLandmarkDataset
  ds = SignLandmarkDataset(X, y, class_names, augment=False)
----------------------------------------------------------------------
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional


class SignLandmarkDataset(Dataset):
    """
    Dataset wrapping preprocessed numpy landmark arrays.

    Parameters
    ----------
    X           : np.ndarray of shape (N, T, F) — sequences
    y           : np.ndarray of shape (N,)      — integer class labels
    class_names : list of class name strings
    augment     : bool — if True, apply light online augmentation
    noise_std   : float — std for online Gaussian noise (if augment=True)
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_names: List[str],
        augment: bool = False,
        noise_std: float = 0.005,
    ):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        seq = self.X[idx].copy()

        if self.augment:
            # Very light online augmentation: small Gaussian noise only
            # (heavy augmentation was done offline in preprocessing)
            rng = np.random.default_rng()
            if rng.random() < 0.5:
                seq += rng.normal(0.0, self.noise_std, size=seq.shape).astype(np.float32)

        x_tensor = torch.from_numpy(seq)        # (T, F)
        y_tensor = torch.tensor(self.y[idx], dtype=torch.long)
        return x_tensor, y_tensor

    @property
    def class_weights(self) -> torch.Tensor:
        """Compute balanced class weights from label distribution."""
        from sklearn.utils.class_weight import compute_class_weight
        weights = compute_class_weight(
            "balanced",
            classes=np.arange(self.num_classes),
            y=self.y,
        )
        return torch.tensor(weights, dtype=torch.float32)


def load_processed_dataset(npz_path: str):
    """
    Load sign_model/data/processed/dataset.npz and return split arrays.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test, class_names
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Processed dataset not found at '{npz_path}'.\n"
            "Run: python scripts/02_preprocess.py first."
        )

    X_train     = data["X_train"]
    y_train     = data["y_train"]
    X_val       = data["X_val"]
    y_val       = data["y_val"]
    X_test      = data["X_test"]
    y_test      = data["y_test"]
    class_names = list(data["class_names"])

    return X_train, y_train, X_val, y_val, X_test, y_test, class_names
