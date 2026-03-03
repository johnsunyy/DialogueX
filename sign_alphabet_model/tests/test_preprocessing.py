"""
tests/test_preprocessing.py

Unit tests for preprocessing pipeline.
Tests:
  - normalize_sample: wrist is (0,0,0) after normalization
  - normalize_sample: scale — max pairwise distance ≈ 1.0
  - normalize_sample: output shape is (63,)
  - normalize_dataset: processes a batch correctly
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.normalize import normalize_sample, normalize_dataset


def make_random_landmarks(seed=0) -> np.ndarray:
    """Create a random (63,) landmark vector."""
    rng = np.random.RandomState(seed)
    lm = rng.rand(21, 3).astype(np.float32) * 100  # realistic pixel scale
    return lm.flatten()


# ──────────────────────────────────────────────────────────────────

class TestNormalizeSample:

    def test_output_shape(self):
        """Output must be exactly (63,)."""
        raw = make_random_landmarks()
        out = normalize_sample(raw)
        assert out.shape == (63,), f"Expected (63,) got {out.shape}"

    def test_wrist_centered(self):
        """After normalization, wrist (landmark 0) must be (0, 0, 0)."""
        raw = make_random_landmarks(seed=42)
        out = normalize_sample(raw)
        wrist = out[:3]
        np.testing.assert_allclose(
            wrist, [0.0, 0.0, 0.0],
            atol=1e-5,
            err_msg=f"Wrist not zeroed: {wrist}"
        )

    def test_scale_invariant(self):
        """Max pairwise distance should be ≈ 1.0 after normalization."""
        raw = make_random_landmarks(seed=7)
        out = normalize_sample(raw)
        pts = out.reshape(21, 3)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff ** 2, axis=-1)
        max_dist = np.sqrt(sq_dists.max())
        assert abs(max_dist - 1.0) < 0.01, f"Max pairwise dist expected ≈ 1.0, got {max_dist}"

    def test_output_dtype(self):
        """Output should be float32."""
        raw = make_random_landmarks()
        out = normalize_sample(raw)
        assert out.dtype == np.float32, f"Expected float32 got {out.dtype}"

    def test_scale_invariance_of_inputs(self):
        """Scaling raw input 10x should produce the same normalized output."""
        raw = make_random_landmarks(seed=3)
        out1 = normalize_sample(raw)
        out2 = normalize_sample(raw * 10.0)
        np.testing.assert_allclose(out1, out2, atol=1e-4)

    def test_translation_invariance(self):
        """Translating raw input should produce the same normalized output."""
        raw = make_random_landmarks(seed=5)
        shift = np.tile([50.0, 30.0, 10.0], 21)
        out1 = normalize_sample(raw)
        out2 = normalize_sample(raw + shift)
        np.testing.assert_allclose(out1, out2, atol=1e-4)


class TestNormalizeDataset:

    def test_batch_shape(self):
        """normalize_dataset should preserve (N, 63) shape."""
        X = np.stack([make_random_landmarks(seed=i) for i in range(10)])
        X_norm = normalize_dataset(X)
        assert X_norm.shape == (10, 63), f"Expected (10, 63) got {X_norm.shape}"

    def test_each_row_wrist_zeroed(self):
        """Every row's wrist should be (0,0,0)."""
        X = np.stack([make_random_landmarks(seed=i) for i in range(5)])
        X_norm = normalize_dataset(X)
        for i in range(5):
            wrist = X_norm[i, :3]
            np.testing.assert_allclose(wrist, [0, 0, 0], atol=1e-5, err_msg=f"Row {i} wrist not zeroed")
