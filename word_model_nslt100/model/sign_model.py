"""
sign_model.py — Phase 5: BiLSTM + Temporal Attention for NSLT-100

Input : (B, 40, 324)
  → BatchNorm1d(324)
  → BiLSTM(256 hidden, bidirectional)   → (B, 40, 512)
  → Dropout(0.3)
  → BiLSTM(128 hidden, bidirectional)   → (B, 40, 256)
  → TemporalAttention                   → (B, 256)
  → Linear(256 → 128) + ReLU
  → Dropout(0.4)
  → Linear(128 → 100)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Additive attention over T timesteps. Input (B,T,H) → (B,H)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v    = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        energy  = torch.tanh(self.attn(x))          # (B, T, H)
        scores  = self.v(energy).squeeze(-1)         # (B, T)
        weights = F.softmax(scores, dim=-1)          # (B, T)
        return (weights.unsqueeze(-1) * x).sum(dim=1)  # (B, H)


class SignModel(nn.Module):
    """
    Word-level SLR: dual BiLSTM + Temporal Attention.

    Args:
        num_classes : 100 for NSLT-100
        seq_len     : 40
        feat_dim    : 324 (position + velocity)
        lstm1_hidden: 256
        lstm2_hidden: 128
        dropout1    : 0.3
        dropout2    : 0.4
    """

    def __init__(
        self,
        num_classes : int   = 100,
        seq_len     : int   = 40,
        feat_dim    : int   = 324,
        lstm1_hidden: int   = 256,
        lstm2_hidden: int   = 128,
        dropout1    : float = 0.3,
        dropout2    : float = 0.4,
    ):
        super().__init__()
        self.num_classes  = num_classes
        self.feat_dim     = feat_dim
        self.seq_len      = seq_len
        lstm1_out = lstm1_hidden * 2   # bidirectional → 512
        lstm2_out = lstm2_hidden * 2   # bidirectional → 256

        self.bn_input  = nn.BatchNorm1d(feat_dim)

        self.lstm1 = nn.LSTM(feat_dim,    lstm1_hidden, bidirectional=True,  batch_first=True)
        self.drop1 = nn.Dropout(dropout1)
        self.lstm2 = nn.LSTM(lstm1_out,   lstm2_hidden, bidirectional=True,  batch_first=True)

        self.attention = TemporalAttention(lstm2_out)

        self.fc1   = nn.Linear(lstm2_out, 128)
        self.drop2 = nn.Dropout(dropout2)
        self.fc2   = nn.Linear(128, num_classes)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.zero_()
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, feat_dim) → logits (B, num_classes)"""
        B, T, feat_d = x.shape

        # BatchNorm over feature dimension
        xb = self.bn_input(x.reshape(B * T, feat_d)).reshape(B, T, feat_d)

        out, _ = self.lstm1(xb)        # (B, T, 512)
        out     = self.drop1(out)
        out, _ = self.lstm2(out)       # (B, T, 256)

        ctx = self.attention(out)      # (B, 256)

        h   = F.relu(self.fc1(ctx))
        h   = self.drop2(h)
        return self.fc2(h)             # (B, 100)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LabelSmoothingCrossEntropy(nn.Module):
    """CrossEntropy with label smoothing + optional inverse-frequency class weights."""

    def __init__(self, smoothing: float = 0.1, weight: torch.Tensor = None):
        super().__init__()
        self.smoothing = smoothing
        self.weight    = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        C = logits.size(-1)
        with torch.no_grad():
            sl = torch.full_like(log_probs, self.smoothing / (C - 1))
            sl.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        loss = -(sl * log_probs)
        if self.weight is not None:
            loss = loss * self.weight.to(logits.device).unsqueeze(0)
        return loss.sum(dim=-1).mean()


if __name__ == "__main__":
    m = SignModel(num_classes=100)
    x = torch.randn(4, 40, 324)
    y = m(x)
    assert y.shape == (4, 100), f"Bad shape: {y.shape}"
    print(f"Shape OK : {y.shape}")
    print(f"Params   : {m.count_parameters():,}")
    lf = LabelSmoothingCrossEntropy(0.1)
    t  = torch.randint(0, 100, (4,))
    print(f"Loss     : {lf(y, t).item():.4f}")
