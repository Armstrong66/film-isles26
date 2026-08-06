"""
preprocessing.py
----------------
Converts raw ATLAS NIfTI images and masks into clean, normalised volumes
ready for training. Applies:
  1. Reorientation → canonical RAS
  2. Intensity clipping at configurable percentiles
  3. Per-scan foreground z-score normalisation
  4. Saves outputs as float32 NIfTI + a per-scan stats JSON sidecar

Design:
  - Stateless functions: each takes paths/arrays, returns results
  - PreprocessingPipeline class: orchestrates, logs, skips existing files
  - No data leakage: stats computed per-scan from foreground only
  - Parallel workers via ProcessPoolExecutor

Usage:
  python preprocessing.py --config configs/config.yaml [--workers 4] [--overwrite]
"""

from __future__ import annotations

import json
import logging
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import nibabel as nib
import nibabel.orientations as nio
from omegaconf import OmegaConf
from tqdm import tqdm

log = logging.getLogger(__name__)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class ScanStats:
    """Per-scan preprocessing statistics saved as JSON sidecar."""
    uid:             str
    original_orientation: str
    reoriented:      bool
    original_shape:  tuple
    final_shape:     tuple
    clip_low:        float
    clip_high:       float
    fg_mean:         float
    fg_std:          float
    fg_voxel_count:  int
    lesion_voxels:   int


# ── Stateless transform functions ──────────────────────────────────────────────

def reorient_to_ras(img: nib.Nifti1Image) -> tuple[nib.Nifti1Image, bool]:
    """
    Reorient a NIfTI image to RAS+ canonical orientation.
    Returns (reoriented_image, was_changed).
    """
    current_ornt = nio.axcodes2ornt(nio.aff2axcodes(img.affine))
    target_ornt  = nio.axcodes2ornt(("R", "A", "S"))

    if np.array_equal(current_ornt, target_ornt):
        return img, False

    transform  = nio.ornt_transform(current_ornt, target_ornt)
    reoriented = img.as_reoriented(transform)
    return reoriented, True


def reorient_mask_to_ras(
    mask: nib.Nifti1Image,
    ref_img: nib.Nifti1Image
) -> nib.Nifti1Image:
    """
    Reorient mask to match ref_img's orientation.
    Uses nearest-neighbour semantics (integer labels preserved).
    """
    current_ornt = nio.axcodes2ornt(nio.aff2axcodes(mask.affine))
    target_ornt  = nio.axcodes2ornt(nio.aff2axcodes(ref_img.affine))

    if np.array_equal(current_ornt, target_ornt):
        return mask

    transform = nio.ornt_transform(current_ornt, target_ornt)
    return mask.as_reoriented(transform)


def clip_and_normalise(
    data: np.ndarray,
    clip_low_pct:  float = 0.5,
    clip_high_pct: float = 99.5,
) -> tuple[np.ndarray, float, float, float, float]:
    """
    1. Identify foreground voxels (> 0).
    2. Clip at clip_low_pct / clip_high_pct of foreground distribution.
    3. Z-score normalise using foreground mean and std.

    Returns:
        normalised_data, clip_low, clip_high, fg_mean, fg_std
    """
    fg_mask = data > 0
    fg      = data[fg_mask]

    assert len(fg) > 0, "No foreground voxels found — check skull-stripping."

    clip_low  = float(np.percentile(fg, clip_low_pct))
    clip_high = float(np.percentile(fg, clip_high_pct))

    clipped        = data.copy()
    clipped[fg_mask] = np.clip(fg, clip_low, clip_high)

    # Recompute fg stats on clipped data for z-score
    fg_clipped = clipped[fg_mask]
    fg_mean    = float(fg_clipped.mean())
    fg_std     = float(fg_clipped.std())

    assert fg_std > 0, f"Zero std in foreground — degenerate scan."

    normalised               = clipped.copy()
    normalised[fg_mask]      = (fg_clipped - fg_mean) / fg_std
    normalised[~fg_mask]     = 0.0        # background stays zero

    return normalised.astype(np.float32), clip_low, clip_high, fg_mean, fg_std


def binarise_mask(data: np.ndarray) -> np.ndarray:
    """Convert mask to binary uint8, handling float masks."""
    return (data > 0.5).astype(np.uint8)


# ── Per-scan processing ────────────────────────────────────────────────────────

def process_one_scan(
    uid:          str,
    img_path:     Path,
    mask_path:    Path,
    out_img_path: Path,
    out_mask_path: Path,
    out_stats_path: Path,
    cfg_preproc,
    overwrite:    bool = False,
) -> Optional[ScanStats]:
    """
    Full preprocessing for one (image, mask) pair.
    Returns ScanStats on success, None if skipped.
    Raises on any error so the caller can log it.
    """
    if not overwrite and out_img_path.exists() and out_mask_path.exists():
        return None  # already processed

    # ── Load ──────────────────────────────────────────────────────────────────
    img_nib  = nib.load(str(img_path))
    mask_nib = nib.load(str(mask_path))

    original_ornt  = "".join(nio.aff2axcodes(img_nib.affine))
    original_shape = img_nib.shape

    # ── Reorient ──────────────────────────────────────────────────────────────
    img_ras, was_reoriented = reorient_to_ras(img_nib)
    mask_ras = reorient_mask_to_ras(mask_nib, img_ras)

    # ── Normalise image ───────────────────────────────────────────────────────
    img_data = img_ras.get_fdata(dtype=np.float32)

    norm_data, clip_low, clip_high, fg_mean, fg_std = clip_and_normalise(
        img_data,
        clip_low_pct  = cfg_preproc.clip_percentiles[0],
        clip_high_pct = cfg_preproc.clip_percentiles[1],
    )

    # ── Binarise mask ─────────────────────────────────────────────────────────
    mask_data   = mask_ras.get_fdata(dtype=np.float32)
    mask_binary = binarise_mask(mask_data)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    out_mask_path.parent.mkdir(parents=True, exist_ok=True)

    nib.save(
        nib.Nifti1Image(norm_data,    img_ras.affine, img_ras.header),
        str(out_img_path)
    )
    nib.save(
        nib.Nifti1Image(mask_binary,  mask_ras.affine, mask_ras.header),
        str(out_mask_path)
    )

    # ── Stats sidecar ─────────────────────────────────────────────────────────
    stats = ScanStats(
        uid                  = uid,
        original_orientation = original_ornt,
        reoriented           = was_reoriented,
        original_shape       = tuple(original_shape),
        final_shape          = tuple(norm_data.shape),
        clip_low             = clip_low,
        clip_high            = clip_high,
        fg_mean              = fg_mean,
        fg_std               = fg_std,
        fg_voxel_count       = int((img_data > 0).sum()),
        lesion_voxels        = int(mask_binary.sum()),
    )
    with open(out_stats_path, "w") as f:
        json.dump(asdict(stats), f, indent=2)

    return stats


# ── Multiprocessing worker (module-level for pickling) ────────────────────────

_WORKER_CFG = None  # set once per worker process

def _worker_init(cfg_preproc_dict: dict) -> None:
    global _WORKER_CFG
    _WORKER_CFG = OmegaConf.create(cfg_preproc_dict)


def _worker_fn(args: tuple) -> tuple[str, Optional[ScanStats], Optional[str]]:
    uid, img_path, mask_path, out_img, out_mask, out_stats, overwrite = args
    try:
        stats = process_one_scan(
            uid, Path(img_path), Path(mask_path),
            Path(out_img), Path(out_mask), Path(out_stats),
            _WORKER_CFG, overwrite
        )
        return uid, stats, None
    except Exception as e:
        return uid, None, str(e)


# ── Pipeline orchestrator ─────────────────────────────────────────────────────

class PreprocessingPipeline:
    """
    Orchestrates preprocessing across all training scans.
    Skips already-processed files unless overwrite=True.
    """

    def __init__(self, cfg) -> None:
        self.cfg       = cfg
        self.data_root = Path(cfg.data.root)
        self.img_dir   = self.data_root / cfg.data.images_dir
        self.mask_dir  = self.data_root / cfg.data.masks_dir
        self.meta_csv  = self.data_root / cfg.data.meta_csv
        self.out_dir   = Path(cfg.data.processed_dir)

        self.out_img_dir   = self.out_dir / "images"
        self.out_mask_dir  = self.out_dir / "masks"
        self.out_stats_dir = self.out_dir / "stats"

        # Use /kaggle/temp if available (larger space), fall back to working
        if str(self.out_dir).startswith("/kaggle/temp"):
            self.out_dir.mkdir(parents=True, exist_ok=True)
            # Also symlink to /kaggle/working for easy access
            working_link = Path("/kaggle/working/processed")
            if not working_link.exists():
                try:
                    working_link.symlink_to(self.out_dir, target_is_directory=True)
                except (OSError, NotImplementedError):
                    pass  # Symlinks may not be available on all systems

    def _build_job_list(self, training_uids: set, overwrite: bool) -> list[tuple]:
        jobs = []
        for img_path in sorted(self.img_dir.glob("*.nii.gz")):
            uid = img_path.name.replace("_T1w.nii.gz", "")
            if uid not in training_uids:
                continue  # skip test-set scans

            mask_path  = self.mask_dir  / f"{uid}_rater1.nii.gz"
            out_img    = self.out_img_dir   / f"{uid}_T1w.nii.gz"
            out_mask   = self.out_mask_dir  / f"{uid}_mask.nii.gz"
            out_stats  = self.out_stats_dir / f"{uid}.json"

            if not mask_path.exists():
                log.warning(f"Mask missing for {uid} — skipping.")
                continue

            jobs.append((
                uid, str(img_path), str(mask_path),
                str(out_img), str(out_mask), str(out_stats),
                overwrite
            ))
        return jobs

    def run(self, training_uids: set, overwrite: bool = False,
            num_workers: int = 4) -> dict:
        """
        Run preprocessing on all training scans.
        Returns a summary dict of counts and any errors.
        """
        import pandas as pd
        jobs = self._build_job_list(training_uids, overwrite)
        log.info(f"Preprocessing: {len(jobs)} scans | workers={num_workers} | overwrite={overwrite}")

        results   = {"processed": 0, "skipped": 0, "errors": []}
        all_stats = []

        cfg_dict = OmegaConf.to_container(self.cfg.preprocessing, resolve=True)

        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_worker_init,
            initargs=(cfg_dict,)
        ) as executor:
            futures = {executor.submit(_worker_fn, job): job[0] for job in jobs}
            for future in tqdm(as_completed(futures), total=len(jobs),
                               desc="Preprocessing"):
                uid, stats, error = future.result()
                if error:
                    log.error(f"{uid}: {error}")
                    results["errors"].append({"uid": uid, "error": error})
                elif stats is None:
                    results["skipped"] += 1
                else:
                    results["processed"] += 1
                    all_stats.append(asdict(stats))

        # ── Save aggregate stats ───────────────────────────────────────────────
        summary_path = self.out_dir / "preprocessing_summary.json"
        with open(summary_path, "w") as f:
            json.dump({
                "processed":       results["processed"],
                "skipped":         results["skipped"],
                "error_count":     len(results["errors"]),
                "errors":          results["errors"],
                "reoriented_count": sum(s["reoriented"] for s in all_stats),
                "scan_stats":      all_stats,
            }, f, indent=2)

        log.info(
            f"Done — processed={results['processed']} "
            f"skipped={results['skipped']} "
            f"errors={len(results['errors'])}"
        )
        log.info(f"Summary saved: {summary_path}")
        return results


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ISLES26 Preprocessing Pipeline")
    parser.add_argument("--config",    type=str, default="configs/config.yaml")
    parser.add_argument("--workers",   type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    logging.basicConfig(
        level=getattr(logging, cfg.logging.level),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    import pandas as pd
    df_meta = pd.read_csv(Path(cfg.data.root) / cfg.data.meta_csv)
    training_uids = set(
        df_meta[df_meta["ATLAS2_DATASET"] == "Training"]["UID"].dropna()
    )
    log.info(f"Training UIDs loaded: {len(training_uids)}")

    n_workers = args.workers or cfg.preprocessing.num_workers
    pipeline  = PreprocessingPipeline(cfg)
    pipeline.run(training_uids, overwrite=args.overwrite, num_workers=n_workers)


if __name__ == "__main__":
    main() 