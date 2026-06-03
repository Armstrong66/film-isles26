"""
loss.py
-------
Loss functions for ISLES26 segmentation.

Components:
  1. Soft Dice loss          — handles class imbalance
  2. Cross-entropy loss      — pixel-level calibration
  3. Boundary focal loss     — upweights lesion boundary voxels
  4. Small-lesion weighting  — per-scan weight boost for small lesions
  5. Deep supervision wrapper — combines losses across all decoder scales

All losses operate on logits of shape (B, 2, H, W, D) and
binary masks of shape (B, 1, H, W, D).

Exported:
  ISLES26Loss(cfg) — full combined loss
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

log = logging.getLogger(__name__)


# ── Component losses ──────────────────────────────────────────────────────────

def soft_dice_loss(
    probs:  torch.Tensor,   # (B, H, W, D) — class-1 probabilities
    target: torch.Tensor,   # (B, H, W, D) — binary float mask
    smooth: float = 1e-5,
) -> torch.Tensor:
    """
    Soft Dice loss averaged over batch.
    smooth prevents division by zero on empty masks.
    """
    intersection = (probs * target).sum(dim=(1, 2, 3))
    union        = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice         = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def weighted_cross_entropy(
    logits:       torch.Tensor,   # (B, 2, H, W, D)
    target:       torch.Tensor,   # (B, H, W, D) long
    scan_weights: torch.Tensor,   # (B,) per-scan weight
) -> torch.Tensor:
    """
    Cross-entropy with per-scan weighting.
    scan_weights are applied at the sample level before averaging.
    """
    B = logits.shape[0]
    losses = torch.stack([
        F.cross_entropy(logits[i:i+1], target[i:i+1])
        for i in range(B)
    ])
    return (losses * scan_weights).mean()


def boundary_focal_loss(
    probs:  torch.Tensor,   # (B, H, W, D) — class-1 probabilities
    target: torch.Tensor,   # (B, H, W, D) — binary float mask
    gamma:  float = 2.0,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """
    Focal loss computed on lesion boundary voxels only.

    Boundary is approximated as the XOR between the mask and its
    max-pooled dilation — a lightweight boundary extraction that
    avoids importing scipy or computing distance transforms.

    Upweights uncertain boundary voxels; standard focal elsewhere.
    """
    # Dilate mask by 1 voxel using max pooling
    mask_dilated = F.max_pool3d(
        target.unsqueeze(1).float(), kernel_size=3, stride=1, padding=1
    ).squeeze(1)
    boundary = (mask_dilated - target).clamp(0, 1)   # ring around lesion

    # Focal weight: (1 - p_t)^gamma
    p_t         = probs * target + (1 - probs) * (1 - target)
    focal_weight = (1 - p_t).pow(gamma)

    bce = -(
        target * torch.log(probs + smooth) +
        (1 - target) * torch.log(1 - probs + smooth)
    )

    # Boundary-upweighted focal loss
    loss = focal_weight * bce
    # Apply boundary mask: boundary voxels get 2× weight
    loss = loss * (1 + boundary)

    return loss.mean()


def compute_scan_weights(
    masks:          torch.Tensor,   # (B, 1, H, W, D)
    voxel_vol_mm3:  float = 1.0,
    threshold_ml:   float = 1.0,
    small_weight:   float = 2.0,
) -> torch.Tensor:
    """
    Per-scan weight: scans with small lesions (< threshold_ml) get
    upweighted to prevent the model from ignoring small lesions during training.
    """
    lesion_voxels = masks.squeeze(1).sum(dim=(1, 2, 3)).float()
    lesion_ml     = lesion_voxels * voxel_vol_mm3 / 1000.0

    weights = torch.where(
        lesion_ml < threshold_ml,
        torch.full_like(lesion_ml, small_weight),
        torch.ones_like(lesion_ml),
    )
    return weights


# ── Combined loss ─────────────────────────────────────────────────────────────

class ISLES26Loss(nn.Module):
    """
    Combined loss = Dice + CE + Boundary focal, with:
      - Per-scan small-lesion upweighting
      - Deep supervision: losses computed at all scales, weighted coarse→fine

    Args:
        cfg: full pipeline config (reads cfg.loss and cfg.training.patch_size)
    """

    # Deep supervision scale weights (finest → coarsest)
    DS_WEIGHTS = [1.0, 0.5, 0.25, 0.125]

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        loss_cfg = cfg.loss
        self.dice_w       = loss_cfg.dice_weight
        self.ce_w         = loss_cfg.ce_weight
        self.boundary_w   = loss_cfg.boundary_weight
        self.small_thresh = loss_cfg.small_lesion_threshold_ml
        self.small_weight = loss_cfg.small_lesion_weight

        log.info(
            f"ISLES26Loss | dice={self.dice_w} ce={self.ce_w} "
            f"boundary={self.boundary_w} "
            f"small_lesion_threshold={self.small_thresh}mL "
            f"small_lesion_weight={self.small_weight}"
        )

    def _compute_scale_loss(
        self,
        logits:       torch.Tensor,   # (B, 2, H, W, D)
        target:       torch.Tensor,   # (B, 1, H, W, D) float
        scan_weights: torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        probs       = torch.softmax(logits, dim=1)[:, 1]   # (B, H, W, D)
        target_3d   = target.squeeze(1)                    # (B, H, W, D)
        target_long = target_3d.long()

        dice_l     = soft_dice_loss(probs, target_3d)
        ce_l       = weighted_cross_entropy(logits, target_long, scan_weights)
        boundary_l = boundary_focal_loss(probs, target_3d)

        return self.dice_w * dice_l + self.ce_w * ce_l + self.boundary_w * boundary_l

    def forward(
        self,
        multi_scale_logits: list[torch.Tensor],   # finest → coarsest
        target:             torch.Tensor,          # (B, 1, H, W, D) binary float
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            total_loss: scalar
            loss_dict:  {"total", "scale_0", ..., "dice", "ce", "boundary"}
        """
        scan_weights = compute_scan_weights(
            target,
            threshold_ml = self.small_thresh,
            small_weight = self.small_weight,
        ).to(target.device)

        total       = torch.tensor(0.0, device=target.device, requires_grad=True)
        loss_dict   = {}
        ds_weights  = self.DS_WEIGHTS[:len(multi_scale_logits)]
        # Normalise weights so they sum to 1
        ds_weight_sum = sum(ds_weights)

        for scale_idx, (logits, ds_w) in enumerate(zip(multi_scale_logits, ds_weights)):
            # Downsample target to match this scale's spatial resolution
            if logits.shape[2:] != target.shape[2:]:
                t_scaled = F.interpolate(
                    target.float(), size=logits.shape[2:],
                    mode="nearest"
                )
            else:
                t_scaled = target.float()

            scale_loss = self._compute_scale_loss(logits, t_scaled, scan_weights)
            weighted   = (ds_w / ds_weight_sum) * scale_loss
            total      = total + weighted
            loss_dict[f"scale_{scale_idx}"] = scale_loss.detach().item()

        loss_dict["total"] = total.detach().item()
        return total, loss_dict