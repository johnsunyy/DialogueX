"""
inference/predictor.py

ASLPredictor: loads a trained model + label encoder and predicts
from a normalised 63-dimensional landmark vector.
"""

import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BEST_MODEL = os.path.join(ROOT_DIR, 'best_model.pt')
LE_PATH    = os.path.join(ROOT_DIR, 'label_encoder.pkl')


class ASLPredictor:
    """
    Loads trained ASL model and label encoder.
    Provides single-sample and batch prediction.
    """

    def __init__(
        self,
        model_path: str = None,
        label_encoder_path: str = None,
        device: str = 'cpu',
    ):
        model_path         = model_path or BEST_MODEL
        label_encoder_path = label_encoder_path or LE_PATH

        # Load label encoder
        with open(label_encoder_path, 'rb') as f:
            self.le = pickle.load(f)

        # Load model — reconstruct architecture from checkpoint config
        self.device = torch.device(device)
        ckpt = torch.load(model_path, map_location=self.device)

        from training.model import ASLClassifier
        cfg = ckpt.get('config', {})

        # Prefer full hidden_layers tuples; fall back to sizes-only for old checkpoints
        if 'hidden_layers' in cfg:
            hidden_layers = [tuple(h) for h in cfg['hidden_layers']]
        else:
            sizes = cfg.get('hidden_layer_sizes', [256, 128, 64])
            hidden_layers = [(sz, 0.0) for sz in sizes]

        self.model = ASLClassifier(
            input_dim=cfg.get('input_dim', 63),
            hidden_layers=hidden_layers,
            num_classes=cfg.get('num_classes', 26),
        )
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, landmarks_63: np.ndarray):
        """
        Predict the ASL letter from a normalised63-dim landmark vector.

        Args:
            landmarks_63: np.ndarray of shape (63,) — already normalised.

        Returns:
            (label: str, confidence: float)
        """
        x = torch.tensor(landmarks_63, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            proba  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        idx        = int(proba.argmax())
        confidence = float(proba[idx])
        label      = self.le.inverse_transform([idx])[0]
        return label, confidence

    def predict_batch(self, X: np.ndarray):
        """
        Predict labels for a batch.

        Args:
            X: np.ndarray of shape (N, 63) — already normalised.

        Returns:
            labels: list of str
            confidences: np.ndarray of shape (N,)
        """
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            proba  = F.softmax(logits, dim=1).cpu().numpy()

        idxs        = proba.argmax(axis=1)
        confidences = proba[np.arange(len(idxs)), idxs]
        labels      = list(self.le.inverse_transform(idxs))
        return labels, confidences
