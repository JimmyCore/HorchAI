"""Models for keystroke classification.

Phase 3 baseline: a small CNN over mel spectrograms. Kept intentionally
small since the dataset (Phase 2) is small too — a large model would just
overfit. A Vision Transformer variant is planned for Phase 4 once this
baseline has a measured accuracy/confusion matrix to compare against.
"""

from __future__ import annotations

import torch
from torch import nn


def set_seed(seed: int = 42) -> None:
    """Seed torch (+ CUDA if available) for reproducible training runs."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SimpleCNN(nn.Module):
    """Small CNN classifier for (1, n_mels, n_frames) mel spectrograms.

    Uses adaptive average pooling before the classifier head so it accepts
    any spectrogram size without shape bookkeeping — useful since window
    length / hop settings may change between experiments.
    """

    def __init__(self, n_classes: int, n_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
