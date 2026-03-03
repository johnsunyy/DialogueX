"""
training/model.py

Flexible ASL MLP classifier.
Baseline: Input(63) → [256, BN, ReLU, Drop(0.4)] → [128, BN, ReLU, Drop(0.3)] → [64, ReLU] → 26

Also supports dynamic architecture for Optuna tuning via build_model().
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class ASLClassifier(nn.Module):
    """
    Fully-connected MLP for ASL landmark classification.

    Args:
        input_dim    : Number of input features (default 63 = 21 landmarks × 3).
        hidden_layers: List of (units, dropout_rate) tuples, one per hidden layer.
                       dropout_rate=0.0 means no dropout layer is added.
        num_classes  : Number of output classes (default 26 for A-Z).
        use_bn       : Whether to use BatchNorm1d after each linear layer.
    """

    def __init__(
        self,
        input_dim: int = 63,
        hidden_layers: Optional[List[Tuple[int, float]]] = None,
        num_classes: int = 26,
        use_bn: bool = True,
    ):
        super().__init__()

        if hidden_layers is None:
            # Baseline architecture from spec
            hidden_layers = [(256, 0.4), (128, 0.3), (64, 0.0)]

        layers = []
        in_dim = input_dim

        for i, (units, drop_rate) in enumerate(hidden_layers):
            layers.append(nn.Linear(in_dim, units))
            if use_bn:
                layers.append(nn.BatchNorm1d(units))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=drop_rate))  # Always include; p=0.0 is a no-op
            in_dim = units

        # Output layer
        layers.append(nn.Linear(in_dim, num_classes))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """He (Kaiming) initialization for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def build_baseline_model(input_dim: int = 63, num_classes: int = 26) -> ASLClassifier:
    """Build the spec-compliant baseline MLP."""
    return ASLClassifier(
        input_dim=input_dim,
        hidden_layers=[(256, 0.4), (128, 0.3), (64, 0.0)],
        num_classes=num_classes,
        use_bn=True,
    )


def build_model_from_trial(trial, input_dim: int = 63, num_classes: int = 26) -> ASLClassifier:
    """
    Build a model parameterised by an Optuna trial object.
    Called inside tuner.py.
    """
    n_layers = trial.suggest_int('n_layers', 2, 5)
    hidden_layers = []
    for i in range(n_layers):
        units = trial.suggest_int(f'units_l{i}', 64, 512, step=64)
        dropout = trial.suggest_float(f'dropout_l{i}', 0.2, 0.5)
        hidden_layers.append((units, dropout))

    return ASLClassifier(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        num_classes=num_classes,
        use_bn=True,
    )


def count_parameters(model: nn.Module) -> int:
    """Return total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = build_baseline_model()
    print(model)
    print(f"\nTrainable parameters: {count_parameters(model):,}")
    dummy = torch.randn(8, 63)
    out = model(dummy)
    print(f"Forward pass output shape: {out.shape}")  # (8, 26)
