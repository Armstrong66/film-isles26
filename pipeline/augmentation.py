"""
augmentation.py
---------------
MONAI-based augmentation pipeline for ISLES26.

Two augmentation modes driven by CHRONICITY_DERIVED:
  - acute/subacute : simulate diffusion-like contrast via mild blur + contrast reduction
  - chronic        : simulate cavity/encephalomalacia via lesion-neighbourhood
                     intensity perturbation
  - unknown        : standard augmentation only (no phase-specific transform)

All transforms operate on torch tensors with shape (1, H, W, D).
Image and mask receive the same spatial transforms; only image gets intensity transforms.

Exported:
  get_train_transforms(chronicity: str) -> Compose
  get_val_transforms()                  -> Compose
"""

from __future__ import annotations

import numpy as np
import torch
from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotate90d,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandAdjustContrastd,
    RandZoomd,
    EnsureTyped,
    ToTensord,
)
from monai.transforms import MapTransform
from monai.config import KeysCollection


# ── Custom chronicity-specific transforms ─────────────────────────────────────

class AcuteContrastReduction(MapTransform):
    """
    Acute/subacute phase augmentation.
    Simulates reduced T1 conspicuity of early infarcts by:
      - Mild Gaussian blur (softens lesion boundaries)
      - Random contrast reduction toward the mean
    Applied to image only; mask unchanged.
    """

    def __init__(
        self,
        keys:     KeysCollection,
        prob:     float = 0.4,
        blur_sigma_range: tuple = (0.3, 0.8),
    ) -> None:
        super().__init__(keys)
        self.prob              = prob
        self.blur_sigma_range  = blur_sigma_range
        self._blur = RandGaussianSmoothd(
            keys=keys, prob=1.0,
            sigma_x=blur_sigma_range,
            sigma_y=blur_sigma_range,
            sigma_z=blur_sigma_range,
        )
        self._contrast = RandAdjustContrastd(
            keys=keys, prob=1.0, gamma=(1.2, 1.8)
        )

    def __call__(self, data: dict) -> dict:
        if np.random.random() < self.prob:
            data = self._blur(data)
        if np.random.random() < self.prob:
            data = self._contrast(data)
        return data


class ChronicCavityPerturbation(MapTransform):
    """
    Chronic phase augmentation.
    Simulates T1 hypointensity of chronic cavities by inverting intensity
    within the lesion neighbourhood (± dilation), making the model robust
    to the dark cavity signal characteristic of chronic infarcts.
    Applied to image only; mask unchanged.
    """

    def __init__(
        self,
        image_key: str  = "image",
        mask_key:  str  = "mask",
        prob:      float = 0.35,
        inversion_strength: float = 0.6,
    ) -> None:
        super().__init__([image_key])
        self.image_key          = image_key
        self.mask_key           = mask_key
        self.prob               = prob
        self.inversion_strength = inversion_strength

    def __call__(self, data: dict) -> dict:
        if np.random.random() > self.prob:
            return data

        img  = data[self.image_key]   # (1, H, W, D) tensor
        mask = data[self.mask_key]    # (1, H, W, D) tensor

        if mask.sum() == 0:
            return data  # no lesion to perturb

        # Invert lesion-region intensity toward its negative (z-scored space)
        lesion_region = mask.bool()
        img_perturbed = img.clone()
        img_perturbed[lesion_region] = (
            img[lesion_region] * (1 - self.inversion_strength)
            - self.inversion_strength * img[lesion_region].abs()
        )
        data[self.image_key] = img_perturbed
        return data


# ── Standard spatial and intensity transforms ─────────────────────────────────

def _spatial_transforms(prob: float = 0.5) -> list:
    return [
        RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "mask"], prob=0.3, max_k=3, spatial_axes=(0, 1)),
        RandAffined(
            keys=["image", "mask"],
            prob=prob,
            rotate_range=(np.pi / 18,) * 3,   # ±10 degrees
            scale_range=(0.1, 0.1, 0.1),
            translate_range=(10, 10, 10),
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
        ),
        RandZoomd(
            keys=["image", "mask"],
            prob=0.3,
            min_zoom=0.85,
            max_zoom=1.15,
            mode=("trilinear", "nearest"),
        ),
    ]


def _intensity_transforms() -> list:
    return [
        RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.1),
        RandScaleIntensityd(keys=["image"], factors=0.15, prob=0.3),
        RandShiftIntensityd(keys=["image"], offsets=0.15, prob=0.3),
        RandGaussianSmoothd(
            keys=["image"], prob=0.2,
            sigma_x=(0.25, 0.75),
            sigma_y=(0.25, 0.75),
            sigma_z=(0.25, 0.75),
        ),
    ]


# ── Public API ────────────────────────────────────────────────────────────────

def get_train_transforms(chronicity: str) -> Compose:
    """
    Returns a MONAI Compose pipeline tailored to the chronicity class.
    chronicity: one of 'acute', 'subacute', 'chronic', 'unknown'
    """
    transforms = [EnsureTyped(keys=["image", "mask"], track_meta=False)]

    # Spatial transforms (same for all classes)
    transforms.extend(_spatial_transforms(prob=0.5))

    # Standard intensity transforms (all classes)
    transforms.extend(_intensity_transforms())

    # Phase-specific transforms
    if chronicity in ("acute", "subacute"):
        transforms.append(
            AcuteContrastReduction(keys=["image"], prob=0.4)
        )
    elif chronicity == "chronic":
        transforms.append(
            ChronicCavityPerturbation(image_key="image", mask_key="mask", prob=0.35)
        )
    # unknown: no phase-specific augmentation

    transforms.append(ToTensord(keys=["image", "mask"]))
    return Compose(transforms)


def get_val_transforms() -> Compose:
    """Validation: no augmentation, just type enforcement."""
    return Compose([
        EnsureTyped(keys=["image", "mask"], track_meta=False),
        ToTensord(keys=["image", "mask"]),
    ])