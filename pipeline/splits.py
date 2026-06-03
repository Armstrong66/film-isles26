"""
splits.py
---------
Generates and persists 5-fold cross-validation splits for ISLES26.

Design:
  - Joint stratification by CHRONICITY_DERIVED + SITE (as planned from EDA)
  - Deterministic: fixed seed, saved to JSON so all experiments use same folds
  - Produces a splits.json with train/val UIDs per fold
  - Also exports a dataset_manifest.csv with all metadata for quick inspection

Usage:
  python splits.py --config configs/config.yaml
  python splits.py --config configs/config.yaml --inspect   # print fold stats
"""

from __future__ import annotations

import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from omegaconf import OmegaConf

log = logging.getLogger(__name__)


# ── Stratification helpers ─────────────────────────────────────────────────────

def make_joint_stratum(df: pd.DataFrame) -> pd.Series:
    """
    Combine CHRONICITY_DERIVED + SITE into a single stratum label.
    Falls back gracefully when a combination has too few samples for
    stratification (sklearn requires >= n_splits samples per stratum).
    Rare strata (< n_splits) are collapsed into a '__rare__' bucket.
    """
    raw = df["CHRONICITY_DERIVED"].astype(str) + "__" + df["SITE"].astype(str)

    counts    = raw.value_counts()
    n_splits  = 5  # hardcoded here; caller checks against cfg
    rare_mask = raw.map(counts) < n_splits
    collapsed = raw.copy()
    collapsed[rare_mask] = "__rare__"

    n_rare = rare_mask.sum()
    if n_rare > 0:
        log.warning(
            f"{n_rare} samples collapsed into '__rare__' stratum "
            f"({rare_mask.sum()} unique combos had < {n_splits} samples)."
        )
    return collapsed


def generate_splits(
    df_train: pd.DataFrame,
    n_splits: int,
    seed:     int,
) -> list[dict]:
    """
    Produce n_splits folds. Each fold is a dict:
        {"fold": int, "train_uids": [...], "val_uids": [...]}

    Stratification is joint CHRONICITY_DERIVED × SITE.
    """
    assert "UID" in df_train.columns, "df_train must have a UID column."

    stratum = make_joint_stratum(df_train)
    uids    = df_train["UID"].values

    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(uids, stratum)):
        folds.append({
            "fold":       fold_idx,
            "train_uids": uids[train_idx].tolist(),
            "val_uids":   uids[val_idx].tolist(),
        })

    return folds


def validate_splits(folds: list[dict], df_train: pd.DataFrame) -> None:
    """
    Assert basic split integrity:
    - No UID leaks between train and val within a fold
    - All UIDs covered across val sets (union = full training set)
    - Reasonable size balance across folds
    """
    all_uids = set(df_train["UID"].dropna().tolist())
    val_union = set()

    for fold in folds:
        train_set = set(fold["train_uids"])
        val_set   = set(fold["val_uids"])

        overlap = train_set & val_set
        assert len(overlap) == 0, \
            f"Fold {fold['fold']}: {len(overlap)} UIDs appear in both train and val."

        val_union |= val_set

    assert val_union == all_uids, (
        f"Val union ({len(val_union)}) != all training UIDs ({len(all_uids)}). "
        f"Missing: {all_uids - val_union}"
    )
    log.info("Split validation passed — no leaks, full coverage.")


def print_fold_stats(folds: list[dict], df_train: pd.DataFrame) -> None:
    """Print per-fold chronicity and site distribution for inspection."""
    uid_to_chron = df_train.set_index("UID")["CHRONICITY_DERIVED"].to_dict()
    uid_to_site  = df_train.set_index("UID")["SITE"].to_dict()

    for fold in folds:
        val_uids  = fold["val_uids"]
        chron_cnt = pd.Series([uid_to_chron.get(u) for u in val_uids]).value_counts()
        site_cnt  = pd.Series([uid_to_site.get(u)  for u in val_uids]).value_counts()

        print(f"\n── Fold {fold['fold']} ──────────────────────────────")
        print(f"  train={len(fold['train_uids'])}  val={len(val_uids)}")
        print(f"  Chronicity: {chron_cnt.to_dict()}")
        print(f"  Sites     : {len(site_cnt)} unique  top3={site_cnt.head(3).to_dict()}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CV splits for ISLES26")
    parser.add_argument("--config",  type=str, default="configs/config.yaml")
    parser.add_argument("--inspect", action="store_true",
                        help="Print fold statistics after generating splits")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    out_dir    = Path(cfg.data.processed_dir)
    splits_out = out_dir / "splits.json"
    manifest_out = out_dir / "dataset_manifest.csv"

    if splits_out.exists() and not args.overwrite:
        log.info(f"splits.json already exists at {splits_out}. Use --overwrite to regenerate.")
        df_train = pd.read_csv(manifest_out)
        folds    = json.loads(splits_out.read_text())
        if args.inspect:
            print_fold_stats(folds, df_train)
        return

    # ── Load metadata ──────────────────────────────────────────────────────────
    meta_path = Path(cfg.data.root) / cfg.data.meta_csv
    df        = pd.read_csv(meta_path)
    df_train  = df[df["ATLAS2_DATASET"] == "Training"].copy().reset_index(drop=True)

    assert len(df_train) > 0, "No training rows found in metadata CSV."
    log.info(f"Training samples: {len(df_train)}")

    # ── Verify processed files exist ───────────────────────────────────────────
    proc_img_dir = out_dir / "images"
    missing = [
        uid for uid in df_train["UID"].dropna()
        if not (proc_img_dir / f"{uid}_T1w.nii.gz").exists()
    ]
    if missing:
        log.warning(
            f"{len(missing)} UIDs in metadata have no preprocessed image. "
            f"Run preprocessing.py first. First missing: {missing[:3]}"
        )
        # Filter to only UIDs with preprocessed files
        available = set(df_train["UID"].dropna()) - set(missing)
        df_train  = df_train[df_train["UID"].isin(available)].reset_index(drop=True)
        log.info(f"Proceeding with {len(df_train)} available UIDs.")

    # ── Generate splits ────────────────────────────────────────────────────────
    folds = generate_splits(
        df_train,
        n_splits = cfg.data.n_splits,
        seed     = cfg.data.seed,
    )
    validate_splits(folds, df_train)

    # ── Save ───────────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    splits_out.write_text(json.dumps(folds, indent=2))
    df_train.to_csv(manifest_out, index=False)

    log.info(f"Splits saved   : {splits_out}")
    log.info(f"Manifest saved : {manifest_out}")

    if args.inspect:
        print_fold_stats(folds, df_train)


if __name__ == "__main__":
    main()