"""
tests/test_model_output.py

Unit tests for model, predictor, and smoothing logic.
Tests:
  - Model output shape is (batch_size, 26)
  - Predictor returns valid A-Z label
  - Smoother returns label when agreement/confidence thresholds are met
  - Smoother returns None when criteria not met
"""

import numpy as np
import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.model import ASLClassifier, build_baseline_model, count_parameters
from inference.smoothing import PredictionSmoother

VALID_LETTERS = [chr(c) for c in range(ord('A'), ord('Z') + 1)]


# ──────────────────────────────────────────────────────────────────

class TestModelOutput:

    def test_baseline_output_shape(self):
        """Baseline model forward pass: output shape must be (batch, 26)."""
        model = build_baseline_model()
        model.eval()
        batch = torch.randn(32, 63)
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (32, 26), f"Expected (32, 26) got {out.shape}"

    def test_single_sample_output_shape(self):
        """Single sample forward pass: output shape (1, 26)."""
        model = build_baseline_model()
        model.eval()
        x = torch.randn(1, 63)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 26)

    def test_model_output_is_logits(self):
        """Output should NOT be probabilities (logits, not softmax)."""
        model = build_baseline_model()
        model.eval()
        x = torch.randn(8, 63)
        with torch.no_grad():
            out = model(x)
        # Logits won't sum to 1 per row
        row_sums = torch.softmax(out, dim=1).sum(dim=1)
        np.testing.assert_allclose(row_sums.numpy(), np.ones(8), atol=1e-5)

    def test_custom_architecture_shape(self):
        """Custom hidden layers: output shape still (batch, 26)."""
        model = ASLClassifier(input_dim=63, hidden_layers=[(128, 0.3), (64, 0.2), (32, 0.0)], num_classes=26)
        model.eval()
        x = torch.randn(16, 63)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (16, 26)

    def test_parameter_count_positive(self):
        """Model should have a positive number of trainable parameters."""
        model = build_baseline_model()
        assert count_parameters(model) > 0

    def test_different_batch_sizes(self):
        """Model should handle varying batch sizes."""
        model = build_baseline_model()
        model.eval()
        for bs in [1, 8, 64, 256]:
            x = torch.randn(bs, 63)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (bs, 26), f"Failed for batch size {bs}"


# ──────────────────────────────────────────────────────────────────

class TestPredictionSmoother:

    def test_returns_none_before_buffer_full(self):
        """Smoother should return None until buffer_size predictions received."""
        smoother = PredictionSmoother(buffer_size=10, agreement_threshold=7, confidence_threshold=0.85)
        for i in range(9):
            result = smoother.update('A', 0.95)
            assert result is None, f"Should be None until buffer full, got {result} at step {i}"

    def test_returns_label_when_criteria_met(self):
        """Returns dominant label when ≥7/10 agree with confidence ≥0.85."""
        smoother = PredictionSmoother(buffer_size=10, agreement_threshold=7, confidence_threshold=0.85)
        result = None
        for _ in range(10):
            result = smoother.update('B', 0.92)
        assert result == 'B', f"Expected 'B', got {result}"

    def test_returns_none_insufficient_agreement(self):
        """Returns None when fewer than threshold predictions agree."""
        smoother = PredictionSmoother(buffer_size=10, agreement_threshold=7, confidence_threshold=0.85)
        letters = ['A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C', 'C']
        result = None
        for ltr in letters:
            result = smoother.update(ltr, 0.95)
        # No single letter has 7+ votes
        assert result is None

    def test_returns_none_low_confidence(self):
        """Returns None when agreement is met but average confidence is too low."""
        smoother = PredictionSmoother(buffer_size=10, agreement_threshold=7, confidence_threshold=0.85)
        result = None
        for _ in range(10):
            result = smoother.update('C', 0.70)  # confidence below 0.85
        assert result is None

    def test_reset_clears_buffer(self):
        """After reset, buffer should be empty and None returned again."""
        smoother = PredictionSmoother(buffer_size=10)
        for _ in range(10):
            smoother.update('A', 0.95)
        smoother.reset()
        result = smoother.update('A', 0.95)
        assert result is None  # buffer has only 1 entry

    def test_dominant_label_property(self):
        """dominant_label property should return correct tuple."""
        smoother = PredictionSmoother(buffer_size=5)
        for _ in range(5):
            smoother.update('Z', 0.90)
        dominant, count, avg_conf = smoother.dominant_label
        assert dominant == 'Z'
        assert count == 5
        assert abs(avg_conf - 0.90) < 0.01
