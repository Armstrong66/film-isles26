"""
dataset.py
----------
PyTorch Dataset for ISLES26. Shared by Track A (FiLM) and Track C (LLM).

Each __getitem__ returns a dict:
    {
        "image":      FloatTensor (1, H, W, D),   preprocessed T1w
        "mask":       LongTensor  (1, H, W, D),   binary lesion mask
        "metadata":   dict,                        raw metadata fields
        "meta_vec":   FloatTensor (4,),            [days_norm, acute, subacute, chronic]
                                                   (Track A conditioning input)
        "meta_text":  str,                         natural language summary
                                                   (Track C conditioning input)
        "uid":        str,                         scan identifier
        "chronicity": str,                         derived chronicity label
    }

The Dataset does not know which track is active — it always returns both
meta_vec and meta_text. The model's conditioning module picks what it needs.

Usage:
    from dataset import ISLES26Dataset, build_dataloaders
    train_dl, val_dl = build_dataloaders(cfg, fold=0)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig

from augmentation import get_train_transforms, get_val_transforms

log = logging.getLogger(__name__)


# ── Metadata encoding ──────────────────────────────────────────────────────────

CHRONICITY_TO_IDX = {"acute": 0, "subacute": 1, "chronic": 2, "unknown": 3}
DAYS_MAX          = 10000.0   # clip + log-normalise; EDA showed max ~10000


def encode_metadata_vector(
    days_post_stroke: Optional[float],
    chronicity_raw:   Optional[float],  # 1.0 = confirmed chronic, NaN = not available
    chronicity_derived: str,            # derived label from DAYS_POST_STROKE
) -> torch.Tensor:
    """
    Encode metadata as a fixed-length float vector for Track A (FiLM gate).

    Layout (dim=5):
        [0] days_norm      — log1p(days) / log1p(DAYS_MAX), in [0,1]; 0.5 if unknown
        [1] is_acute       — 1.0 if chronicity_derived == 'acute'
        [2] is_subacute    — 1.0 if chronicity_derived == 'subacute'
        [3] is_chronic     — 1.0 if chronicity_derived == 'chronic'
        [4] confirmed_chronic — 1.0 if chronicity_raw == 1.0 (confirmed 180+ days)
                                0.0 otherwise (unknown, not chronic, or NaN)

    The chronicity_derived from DAYS_POST_STROKE is the primary signal.
    The confirmed_chronic flag is added only when the organizer-provided
    chronicity == 1.0 (confirmed chronic, 180+ days post-stroke).

    Args:
        days_post_stroke: float or None (from DAYS_POST_STROKE column)
        chronicity_raw: 1.0 if confirmed chronic ( organizer-provided), NaN otherwise
        chronicity_derived: derived label ('acute', 'subacute', 'chronic', 'unknown')

    Returns:
        FloatTensor of shape (5,)
    """
    if days_post_stroke is None or np.isnan(float(days_post_stroke)):
        days_norm = 0.5   # midpoint sentinel for unknown
    else:
        days_clipped = min(float(days_post_stroke), DAYS_MAX)
        days_norm    = np.log1p(days_clipped) / np.log1p(DAYS_MAX)

    # One-hot from chronicity_derived (our primary signal)
    chron = chronicity_derived if chronicity_derived in CHRONICITY_TO_IDX else "unknown"
    one_hot = [
        1.0 if chron == "acute"    else 0.0,
        1.0 if chron == "subacute" else 0.0,
        1.0 if chron == "chronic"  else 0.0,
    ]

    # confirmed_chronic: only 1.0 when organizer says chronicity == 1.0
    confirmed_chronic = 1.0 if chronicity_raw == 1.0 else 0.0

    return torch.tensor([days_norm] + one_hot + [confirmed_chronic], dtype=torch.float32)


def encode_metadata_text(
    uid:              str,
    days_post_stroke: Optional[float],
    chronicity:       str,
    site:             str,
) -> str:
    """
    Encode metadata as a natural language string for Track C (LLM conditioning).
    Kept concise — the LLM needs enough context, not a paragraph.
    """
    days_str = (
        f"{int(days_post_stroke)} days" if (
            days_post_stroke is not None and not np.isnan(float(days_post_stroke))
        ) else "unknown duration"
    )
    phase_str = chronicity if chronicity != "unknown" else "unknown phase"
    return (
        f"Stroke MRI scan from site {site}. "
        f"Time since stroke onset: {days_str}. "
        f"Disease phase: {phase_str}. "
        f"Task: segment the ischemic lesion."
    )


# ── Dataset ───────────────────────────────────────────────────────────────────

class ISLES26Dataset(Dataset):
    """
    Loads preprocessed ATLAS scans for training or validation.

    Args:
        records:    list of dicts with keys uid, img_path, mask_path + metadata
        transform:  MONAI Compose pipeline (train or val)
        is_train:   controls whether augmentation fires
    """

    def __init__(
        self,
        records:  list[dict],
        transform,
        is_train: bool = True,
    ) -> None:
        self.records  = records
        self.transform = transform
        self.is_train  = is_train

        assert len(records) > 0, "Empty record list passed to ISLES26Dataset."
        log.info(f"Dataset initialised: {len(records)} scans | train={is_train}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        uid = rec["uid"]

        # ── Load NIfTI ────────────────────────────────────────────────────────
        img_data  = nib.load(rec["img_path"]).get_fdata(dtype=np.float32)
        mask_data = nib.load(rec["mask_path"]).get_fdata(dtype=np.float32)

        # Add channel dim: (H,W,D) → (1,H,W,D)
        img_t  = torch.from_numpy(img_data[np.newaxis])
        mask_t = torch.from_numpy(mask_data[np.newaxis])

        # ── Augmentation ──────────────────────────────────────────────────────
        aug_input = {"image": img_t, "mask": mask_t}
        aug_out   = self.transform(aug_input)
        img_t     = aug_out["image"].float()
        mask_t    = aug_out["mask"].long()

        # ── Metadata encoding ───────────────────────────────────────────────────
        chronicity_derived = rec.get("chronicity", "unknown")
        days               = rec.get("days_post_stroke", None)
        site               = rec.get("site", "unknown")
        chronicity_raw     = rec.get("chronicity_raw", None)  # 1.0 or NaN from organizer

        meta_vec  = encode_metadata_vector(days, chronicity_raw, chronicity_derived)
        meta_text = encode_metadata_text(uid, days, chronicity_derived, site)

        return {
            "image":      img_t,
            "mask":       mask_t,
            "meta_vec":   meta_vec,
            "meta_text":  meta_text,
            "uid":        uid,
            "chronicity": chronicity_derived,
            "metadata": {
                "days_post_stroke": days,
                "site":             site,
                "chronicity_raw":   chronicity_raw,
            },
        }


# ── Record builder ────────────────────────────────────────────────────────────

def build_records(
    uids:         list[str],
    df_meta:      pd.DataFrame,
    proc_img_dir: Path,
    proc_mask_dir: Path,
) -> list[dict]:
    """
    Build a list of per-scan dicts for the Dataset.
    Skips UIDs whose preprocessed files are missing (logs warning).
    """
    uid_to_meta = df_meta.set_index("UID").to_dict(orient="index")
    records = []
    skipped = 0

    for uid in uids:
        img_path  = proc_img_dir  / f"{uid}_T1w.nii.gz"
        mask_path = proc_mask_dir / f"{uid}_mask.nii.gz"

        if not img_path.exists() or not mask_path.exists():
            log.warning(f"Missing preprocessed files for {uid} — skipping.")
            skipped += 1
            continue

        meta = uid_to_meta.get(uid, {})
        records.append({
            "uid":              uid,
            "img_path":         str(img_path),
            "mask_path":        str(mask_path),
            "chronicity":       meta.get("CHRONICITY_DERIVED", "unknown"),
            "days_post_stroke": meta.get("DAYS_POST_STROKE", None),
            "site":             meta.get("SITE", "unknown"),
            "chronicity_raw":   meta.get("CHRONICITY", None),  # 1.0 or NaN
        })

    if skipped:
        log.warning(f"Skipped {skipped} UIDs with missing preprocessed files.")

    assert len(records) > 0, "No valid records found. Run preprocessing.py first."
    return records


# ── DataLoader factory ────────────────────────────────────────────────────────

def build_dataloaders(
    cfg:       DictConfig,
    fold:      int,
    splits:    list[dict],
    df_meta:   pd.DataFrame,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders for a given CV fold.

    Returns:
        (train_dataloader, val_dataloader)
    """
    assert 0 <= fold < len(splits), f"Fold {fold} out of range (0-{len(splits)-1})."

    fold_data     = splits[fold]
    train_uids    = fold_data["train_uids"]
    val_uids      = fold_data["val_uids"]

    proc_img_dir  = Path(cfg.data.processed_dir) / "images"
    proc_mask_dir = Path(cfg.data.processed_dir) / "masks"

    train_records = build_records(train_uids, df_meta, proc_img_dir, proc_mask_dir)
    val_records   = build_records(val_uids,   df_meta, proc_img_dir, proc_mask_dir)

    log.info(f"Fold {fold} — train={len(train_records)}  val={len(val_records)}")

    # Per-sample chronicity-specific augmentation for training
    # Dataset is constructed per-chronicity class and concatenated
    from torch.utils.data import ConcatDataset

    chron_classes = ["acute", "subacute", "chronic", "unknown"]
    train_datasets = []
    for chron in chron_classes:
        subset = [r for r in train_records if r["chronicity"] == chron]
        if subset:
            train_datasets.append(
                ISLES26Dataset(subset, get_train_transforms(chron, cfg.training.patch_size), is_train=True)
            )

    train_dataset = ConcatDataset(train_datasets)
    val_dataset   = ISLES26Dataset(val_records, get_val_transforms(cfg.training.patch_size), is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size  = cfg.training.batch_size,
        shuffle     = True,
        num_workers = cfg.preprocessing.num_workers,
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = 1,        # always 1 for validation
        shuffle     = False,
        num_workers = cfg.preprocessing.num_workers,
        pin_memory  = True,
    )

    return train_loader, val_loader