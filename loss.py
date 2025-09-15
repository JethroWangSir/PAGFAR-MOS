import torch
import numpy as np
from scipy.stats import norm
import torch.nn.functional as F
import torch.nn as nn


class LearnableAdaptiveWeighting(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)   # pointwise, pairwise, listwise
        )
        self.logits = nn.Parameter(torch.zeros(3))  # residual learnable

    def forward(self, labels, preds):
        labels = labels.view(-1)
        preds = preds.view(-1)

        # === Batch-level features ===
        # 1. Entropy
        label_probs = F.softmax(labels, dim=0)
        label_entropy = -torch.sum(label_probs * torch.log(label_probs + 1e-10))

        pred_probs = F.softmax(preds, dim=0)
        pred_entropy = -torch.sum(pred_probs * torch.log(pred_probs + 1e-10))

        # 2. Standard deviation
        label_std = torch.std(labels)
        pred_std = torch.std(preds)

        # 3. Entropy mismatch
        mismatch = torch.abs(label_entropy - pred_entropy)

        # Combine features
        batch_input = torch.stack([
            label_entropy,
            pred_entropy,
            label_std,
            pred_std,
            mismatch
        ], dim=-1)

        logits = self.mlp(batch_input) + self.logits
        weights = F.softmax(logits, dim=-1)

        return weights


def pairwise_ranking_loss(preds, labels, preference_factor=7.0, margin_scale=0.2):
    preds = preds.squeeze()
    labels = labels.squeeze()
    B = preds.size(0)
    
    if labels.max() != labels.min():
        # === 1. Quality-aware normalization and weighting ===
        norm_labels = (labels - labels.min()) / (labels.max() - labels.min())
    else:
        return torch.tensor(0.0, device=preds.device)

    # Quality-aware weight matrix
    quality_level_matrix = torch.max(norm_labels.view(B, 1), norm_labels.view(1, B))
    weight_matrix = 1.0 + (preference_factor - 1.0) * quality_level_matrix

    # === 2. Pairwise difference and margin computation ===
    pred_diff_matrix = preds.view(B, 1) - preds.view(1, B)
    label_diff_matrix = labels.view(B, 1) - labels.view(1, B)
    margin_matrix = torch.abs(label_diff_matrix) * margin_scale

    # Only consider pairs with different labels
    mask = label_diff_matrix != 0
    sign_matrix = torch.sign(label_diff_matrix)

    # === 3. Weighted pairwise hinge loss ===
    loss_matrix = weight_matrix * torch.relu(-sign_matrix * pred_diff_matrix + margin_matrix)
    loss = torch.sum(loss_matrix[mask]) / torch.sum(mask)
    
    return loss


def listwise_ranking_loss(preds, labels, preference_factor=0.5, temperature=2.0, margin_scale=0.3):
    if preds.shape != labels.shape:
        raise ValueError("Shapes of preds and labels must match.")
    if preds.dim() > 2 or (preds.dim() == 2 and preds.shape[1] != 1):
        raise ValueError("Inputs must be (L, 1) or (L,) for a single list.")

    # Squeeze to (L,)
    preds = preds.squeeze(dim=-1)
    labels = labels.squeeze(dim=-1)

    # === 1. Quality-aware weighting ===
    min_y, max_y = labels.min(), labels.max()
    if (max_y - min_y).item() < 1e-8:  # Avoid division by zero if all labels same
        quality_weights = torch.ones_like(labels)
    else:
        quality_weights = 1.0 + preference_factor * (labels - min_y) / (max_y - min_y)

    # Ground-truth probability distribution with temperature scaling
    true_probs = F.softmax(temperature * labels, dim=0)

    # === 2. Margin-aware scaling of predictions ===
    label_mean = labels.mean()
    label_std = labels.std(unbiased=False) + 1e-8
    margin_scale = 1.0 + margin_scale * torch.abs(labels - label_mean) / label_std
    preds_scaled = preds * margin_scale

    # Predicted probability distribution
    pred_probs = F.softmax(preds_scaled, dim=0)

    # === 3. Weighted cross-entropy ===
    loss = -torch.sum(quality_weights * true_probs * torch.log(pred_probs + 1e-10))

    return loss
    