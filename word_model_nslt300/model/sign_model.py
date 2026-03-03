"""
sign_model.py — Phase 5: BiLSTM + Temporal Attention Model

Architecture:
  Input (B, 30, 162)
  → BatchNorm1d(162)
  → BiLSTM(256, bidirectional=True)    output (B, 30, 512)
  → Dropout(0.3)
  → BiLSTM(256, bidirectional=True)    output (B, 30, 512)
  → TemporalAttention                  output (B, 512)
  → Linear(512 → 256) + ReLU
  → Dropout(0.4)
  → Linear(256 → num_classes)
  → (softmax at inference only)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """
    Additive (Bahdanau-style) temporal attention over sequence steps.
    Input : (B, T, H)
    Output: (B, H) — context vector
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn  = nn.Linear(hidden_dim, hidden_dim)
        self.v     = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, H)
        energy  = torch.tanh(self.attn(x))        # (B, T, H)
        scores  = self.v(energy).squeeze(-1)       # (B, T)
        weights = F.softmax(scores, dim=-1)        # (B, T)
        context = (weights.unsqueeze(-1) * x).sum(dim=1)  # (B, H)
        return context


class SignModel(nn.Module):
    """
    Word-level SLR model with dual BiLSTM + Temporal Attention.

    Args:
        num_classes : number of output word classes (300 for NSLT-300)
        seq_len     : sequence length (default 30)
        feat_dim    : feature dimension per frame (default 162)
        lstm_hidden : hidden units per direction (default 256 → 512 bidirectional)
        dropout1    : dropout after first BiLSTM (default 0.3)
        dropout2    : dropout before output layer (default 0.4)
    """

    def __init__(
        self,
        num_classes: int = 300,
        seq_len: int = 30,
        feat_dim: int = 162,
        lstm_hidden: int = 256,
        dropout1: float = 0.3,
        dropout2: float = 0.4,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim    = feat_dim
        self.seq_len     = seq_len
        lstm_out         = lstm_hidden * 2  # bidirectional

        # ── Feature normalization ──────────────────────────────
        self.bn_input    = nn.BatchNorm1d(feat_dim)

        # ── Temporal encoding ──────────────────────────────────
        self.lstm1 = nn.LSTM(
            input_size=feat_dim,
            hidden_size=lstm_hidden,
            bidirectional=True,
            batch_first=True,
        )
        self.drop1  = nn.Dropout(dropout1)

        self.lstm2 = nn.LSTM(
            input_size=lstm_out,
            hidden_size=lstm_hidden,
            bidirectional=True,
            batch_first=True,
        )

        # ── Temporal Attention ────────────────────────────────
        self.attention = TemporalAttention(lstm_out)

        # ── Classifier ────────────────────────────────────────
        self.fc1   = nn.Linear(lstm_out, 256)
        self.drop2 = nn.Dropout(dropout2)
        self.fc2   = nn.Linear(256, num_classes)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, feat_dim)
        Returns:
            logits : (B, num_classes)
        """
        B, T, feat_d = x.shape  # use feat_d to avoid shadowing F (nn.functional)

        # BatchNorm over feature dim — reshape to (B*T, feat_d)
        x_bn = x.reshape(B * T, feat_d)
        x_bn = self.bn_input(x_bn)
        x    = x_bn.reshape(B, T, feat_d)

        # BiLSTM 1
        out, _ = self.lstm1(x)      # (B, T, 512)
        out     = self.drop1(out)

        # BiLSTM 2
        out, _ = self.lstm2(out)    # (B, T, 512)

        # Temporal Attention → context vector
        ctx = self.attention(out)   # (B, 512)

        # Classifier
        h   = F.relu(self.fc1(ctx))
        h   = self.drop2(h)
        out = self.fc2(h)           # (B, num_classes)

        return out

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities. For inference only."""
        return F.softmax(self.forward(x), dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    Equivalent to nn.CrossEntropyLoss with label_smoothing (PyTorch ≥1.10),
    but also accepts pre-computed class weights.
    """

    def __init__(self, smoothing: float = 0.1, weight: torch.Tensor = None):
        super().__init__()
        self.smoothing = smoothing
        self.weight    = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)   # (B, C)
        n_classes = logits.size(-1)

        with torch.no_grad():
            smooth_labels = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
            smooth_labels.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = -(smooth_labels * log_probs)          # (B, C)

        if self.weight is not None:
            w = self.weight.to(logits.device)
            loss = loss * w.unsqueeze(0)

        return loss.sum(dim=-1).mean()


if __name__ == "__main__":
    # Quick sanity check
    model = SignModel(num_classes=300)
    x     = torch.randn(4, 30, 162)
    y     = model(x)
    assert y.shape == (4, 300), f"Expected (4,300), got {y.shape}"
    print(f"Model OK — output shape: {y.shape}")
    print(f"Trainable parameters: {model.count_parameters():,}")

    loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
    tgt     = torch.randint(0, 300, (4,))
    loss    = loss_fn(y, tgt)
    print(f"Loss: {loss.item():.4f}")
