"""
sign_model/predictor.py
=======================
Self-contained predictor for the word-level Transformer sign language model.
Includes all required architecture definitions so this file can be imported
anywhere without depending on the sign_model.scripts training pipeline.

Usage:
    from sign_model.predictor import WordSignPredictor
    predictor = WordSignPredictor()               # loads v1 model + scaler
    label, confidence = predictor.predict(frames) # frames: np.ndarray (T, 225)
"""

import os
import json
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
#  Paths
# ──────────────────────────────────────────────────────────────────────────────

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_V2_FINAL_DIR = os.path.join(_MODULE_DIR, "models", "v2", "final")

DEFAULT_MODEL_PATH  = os.path.join(_V2_FINAL_DIR, "model.pt")
DEFAULT_CONFIG_PATH = os.path.join(_V2_FINAL_DIR, "model_config.json")
DEFAULT_SCALER_PATH = os.path.join(_MODULE_DIR, "models", "scaler_v2.pkl")

TARGET_FRAMES = 64   # sequence length the model was trained on
FEATURE_DIM   = 225  # 63 left-hand + 63 right-hand + 99 pose

# ──────────────────────────────────────────────────────────────────────────────
#  Architecture  (mirrors sign_model/scripts/03_train.py exactly)
# ──────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)   # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):      # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """
    Input:  (batch, T, input_dim)
    Output: (batch, num_classes)  — raw logits
    """
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 3, dim_feedforward: int = 256,
                 dropout: float = 0.2, num_classes: int = 69):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder    = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):         # (B, T, F)
        x = self.input_proj(x)    # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)       # (B, T, d_model)
        x = x.mean(dim=1)         # global avg-pool -> (B, d_model)
        return self.classifier(x)


class LSTMClassifier(nn.Module):
    """Fallback LSTM in case the saved winner is an LSTM model."""
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=128, hidden_size=256,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.input_proj(x)
        _, (h_n, _) = self.lstm(x)
        return self.classifier(h_n[-1])


# ──────────────────────────────────────────────────────────────────────────────
#  Predictor
# ──────────────────────────────────────────────────────────────────────────────

class WordSignPredictor:
    """
    Loads the trained word-level Transformer model (+ StandardScaler) and
    provides a simple predict() interface compatible with SignTranslationHandler.

    Parameters
    ----------
    model_path   : path to model.pt
    config_path  : path to model_config.json
    scaler_path  : path to scaler.pkl  (REQUIRED — model was trained with scaling)
    device       : 'cpu' or 'cuda'
    """

    def __init__(
        self,
        model_path:  str = DEFAULT_MODEL_PATH,
        config_path: str = DEFAULT_CONFIG_PATH,
        scaler_path: str = DEFAULT_SCALER_PATH,
        device:      str = "cpu",
    ):
        self.device = torch.device(device)

        # ── Load config ──────────────────────────────────────────────────────
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"[WordSignPredictor] model_config.json not found: {config_path}"
            )
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        self.class_names: list = cfg["class_names"]
        num_classes = cfg["num_classes"]
        input_dim   = cfg.get("input_dim", FEATURE_DIM)
        hidden_dim  = cfg.get("hidden_dim", 128)
        num_layers  = cfg.get("num_layers", 3)
        dropout     = cfg.get("dropout", 0.2)
        model_type  = cfg.get("model_type", "transformer")

        # ── Load StandardScaler (critical — model was trained with scaled input) ──
        self.scaler = None
        if scaler_path and os.path.isfile(scaler_path):
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            print(f"[WordSignPredictor] Scaler loaded from {scaler_path}")
        else:
            print(
                f"[WordSignPredictor] WARNING: scaler.pkl not found at {scaler_path}. "
                "Predictions will be unreliable (raw, unscaled features)."
            )

        # ── Build architecture ───────────────────────────────────────────────
        if model_type == "transformer":
            self.model = TransformerClassifier(
                input_dim=input_dim, d_model=hidden_dim, nhead=4,
                num_layers=num_layers, dim_feedforward=256,
                dropout=dropout, num_classes=num_classes,
            )
        else:
            self.model = LSTMClassifier(
                input_dim=input_dim, hidden_dim=hidden_dim,
                num_classes=num_classes, num_layers=num_layers, dropout=dropout,
            )

        # ── Load weights ─────────────────────────────────────────────────────
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"[WordSignPredictor] model.pt not found: {model_path}"
            )
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        print(
            f"[WordSignPredictor] Loaded {model_type.upper()} model "
            f"({num_classes} classes) from {model_path}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, frames: np.ndarray):
        """
        Run word-level inference on a variable-length sequence of landmark frames.

        Parameters
        ----------
        frames : np.ndarray, shape (T, 225)
            Raw 225-float landmark vectors, one per frame.
            T can be any positive integer; the sequence will be zero-padded or
            center-cropped to exactly TARGET_FRAMES (64).

        Returns
        -------
        label      : str   — predicted word (uppercase)
        confidence : float — softmax confidence [0, 1]
        """
        seq = self._pad_or_crop(frames)   # (64, 225)
        seq = self._scale(seq)            # apply StandardScaler (critical!)

        x = torch.tensor(seq[None], dtype=torch.float32).to(self.device)  # (1, 64, 225)
        with torch.no_grad():
            logits = self.model(x)[0]    # (num_classes,)
            probs  = F.softmax(logits, dim=-1).cpu().numpy()

        idx        = int(probs.argmax())
        confidence = float(probs[idx])
        label      = self.class_names[idx]
        return label, confidence

    def top_k(self, frames: np.ndarray, k: int = 3):
        """Return the top-k (label, confidence) pairs."""
        seq = self._pad_or_crop(frames)
        seq = self._scale(seq)
        x   = torch.tensor(seq[None], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(x)[0]
            probs  = F.softmax(logits, dim=-1).cpu().numpy()
        top_idx = np.argsort(probs)[::-1][:k]
        return [(self.class_names[i], float(probs[i])) for i in top_idx]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _scale(self, seq: np.ndarray) -> np.ndarray:
        """Apply the fitted StandardScaler to a (T, 225) sequence."""
        if self.scaler is None:
            return seq
        T = seq.shape[0]
        scaled = self.scaler.transform(seq.reshape(-1, FEATURE_DIM))
        return scaled.reshape(T, FEATURE_DIM)

    @staticmethod
    def _pad_or_crop(frames: np.ndarray) -> np.ndarray:
        """
        Adjust a (T, 225) array to exactly (TARGET_FRAMES, 225).
          - If T < TARGET_FRAMES : zero-pad at the end.
          - If T > TARGET_FRAMES : center-crop.
          - If T == TARGET_FRAMES: return as-is.
        """
        frames = np.asarray(frames, dtype=np.float32)
        T = frames.shape[0]
        if T == TARGET_FRAMES:
            return frames
        if T < TARGET_FRAMES:
            pad = np.zeros((TARGET_FRAMES - T, FEATURE_DIM), dtype=np.float32)
            return np.concatenate([frames, pad], axis=0)
        start = (T - TARGET_FRAMES) // 2
        return frames[start: start + TARGET_FRAMES]
