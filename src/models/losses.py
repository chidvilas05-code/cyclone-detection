"""
Custom Loss Functions for Meteorological Tropical Cyclone Estimation.
Includes:
  1. ClassBalancedFocalLoss: Focuses on hard boundary cases and compensates for class imbalance (e.g. Category 5 Super Cyclones).
  2. OrdinalDistanceLoss: Penalizes classification errors proportionally to numerical category rank distance |k - y|.
  3. FocalOrdinalLoss: Joint combination of focal classification loss and ordinal ranking penalty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalOrdinalLoss(nn.Module):
    """
    Combines:
      1. Focal Loss: (1 - p_t)^gamma * CE(p, y) to focus on hard boundary transition samples (e.g., Severe Cyclonic Storms).
      2. Class-Balanced Alpha Weights: Upweights rare classes (e.g., Category 5 Super Cyclones).
      3. Ordinal Distance Penalty: lambda_ord * sum_k(|k - y| * p_k) to heavily penalize multi-tier misclassifications.
    """
    def __init__(
        self,
        num_classes: int = 5,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        ordinal_weight: float = 0.15,
        label_smoothing: float = 0.01
    ):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.ordinal_weight = ordinal_weight
        self.label_smoothing = label_smoothing

        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

        # Precompute tier rank matrix for ordinal distance: |i - j|
        rank_matrix = torch.zeros((num_classes, num_classes), dtype=torch.float32)
        for i in range(num_classes):
            for j in range(num_classes):
                rank_matrix[i, j] = abs(float(i - j))
        self.register_buffer("rank_matrix", rank_matrix)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, num_classes)
        targets: (B,) class indices in [0, num_classes - 1]
        """
        # 1. Softmax probabilities
        probs = F.softmax(logits, dim=-1)

        # 2. Focal Loss computation
        log_probs = F.log_softmax(logits, dim=-1)

        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(min=1e-6, max=1.0)
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = torch.pow(1.0 - target_probs, self.gamma)

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_weight = focal_weight * alpha_t

        focal_loss = -focal_weight * target_log_probs

        if self.label_smoothing > 0.0:
            smooth_loss = -log_probs.mean(dim=-1)
            focal_loss = (1.0 - self.label_smoothing) * focal_loss + self.label_smoothing * smooth_loss

        if self.rank_matrix.device != targets.device:
            self.rank_matrix = self.rank_matrix.to(targets.device)
        if self.alpha is not None and self.alpha.device != targets.device:
            self.alpha = self.alpha.to(targets.device)

        # 3. Ordinal Distance Penalty: sum_k(|k - y| * p_k)
        target_distances = self.rank_matrix[targets]  # (B, num_classes)
        ordinal_penalty = (probs * target_distances).sum(dim=-1)  # (B,)

        total_loss = focal_loss.mean() + self.ordinal_weight * ordinal_penalty.mean()
        return total_loss
