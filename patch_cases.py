"""
patch_cases.py
--------------
Patch two corrected cases (sub-r032s056, sub-r032s027) in the processed dataset.

The ISLES26 organizers identified and fixed two cases:
- sub-r032s056: New T1w image (original was over-skull-stripped)
- sub-r032s027: File error (fixes corrupted/missing files)

Usage:
    python patch_cases.py --input <input_dir> --output <output_dir> --patch_dir <patch_dir>

Arguments:
    --input:   Path to the processed dataset directory (contains images/ and masks/)
    --output:  Path to output the patched dataset
    --patch_dir: Path to directory containing the corrected files from SWITCHdrive
                 Expected files: sub-r032s056_T1w.nii.gz, sub-r032s056_mask.nii.gz,
                                 sub-r032s027_T1w.nii.gz, sub-r032s027_mask.nii.gz

Example:
    python patch_cases.py \
        --input /kaggle/temp/processed \
        --output /kaggle/temp/processed_patched \
        --patch_dir /kaggle/input/patched-cases

The script:
1. Copies the entire processed dataset to output
2. Replaces the corrected case files from patch_dir
3. Logs which files were patched
"""

import argparse
import logging
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

log = logging.getLogger(__name__)


def verify_nifti(path: Path) -> bool:
    """Verify a NIfTI file can be loaded and has valid data."""
    try:
        img = nib.load(path)
        data = img.get_fdata()
        return data is not None and data.size > 0
    except Exception as e:
        log.error(f"Failed to load {path}: {e}")
        return False


def patch_case(
    uid: str,
    input_dir: Path,
    output_dir: Path,
    patch_dir: Path,
) -> bool:
    """
    Patch a single case by copying from patch_dir to output_dir.

    Args:
        uid:        Case identifier (e.g., "sub-r032s056")
        input_dir:  Source processed dataset directory
        output_dir: Output directory for patched dataset
        patch_dir:  Directory containing corrected files

    Returns:
        True if patching succeeded, False otherwise
    """
    log.info(f"Patching case: {uid}")

    img_patch = patch_dir / f"{uid}_T1w.nii.gz"
    mask_patch = patch_dir / f"{uid}_mask.nii.gz"

    if not img_patch.exists():
        log.error(f"Missing patch file: {img_patch}")
        return False
    if not mask_patch.exists():
        log.error(f"Missing patch file: {mask_patch}")
        return False

    # Verify patch files are valid
    if not verify_nifti(img_patch):
        log.error(f"Invalid patch image: {img_patch}")
        return False
    if not verify_nifti(mask_patch):
        log.error(f"Invalid patch mask: {mask_patch}")
        return False

    # Create output directories
    output_img_dir = output_dir / "images"
    output_mask_dir = output_dir / "masks"
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    # Copy patched files to output
    output_img = output_img_dir / f"{uid}_T1w.nii.gz"
    output_mask = output_mask_dir / f"{uid}_mask.nii.gz"

    shutil.copy2(img_patch, output_img)
    shutil.copy2(mask_patch, output_mask)

    log.info(f"Patched: {uid}")
    return True


def copy_remaining_cases(
    uid_to_patch: set[str],
    input_dir: Path,
    output_dir: Path,
) -> int:
    """
    Copy all non-patched cases from input to output.

    Returns:
        Number of cases copied
    """
    input_img_dir = input_dir / "images"
    input_mask_dir = input_dir / "masks"

    output_img_dir = output_dir / "images"
    output_mask_dir = output_dir / "masks"

    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for img_path in input_img_dir.glob("*.nii.gz"):
        uid = img_path.stem.replace("_T1w", "")

        if uid in uid_to_patch:
            continue  # Skip cases we're patching

        mask_path = input_mask_dir / f"{uid}_mask.nii.gz"
        if mask_path.exists():
            shutil.copy2(img_path, output_img_dir / img_path.name)
            shutil.copy2(mask_path, output_mask_dir / mask_path.name)
            copied += 1

    return copied


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch corrected cases in the processed dataset"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the processed dataset directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output the patched dataset",
    )
    parser.add_argument(
        "--patch_dir",
        type=str,
        required=True,
        help="Path to directory containing corrected files",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be done without copying files",
    )
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    patch_dir = Path(args.patch_dir)

    if not input_dir.exists():
        log.error(f"Input directory not found: {input_dir}")
        return
    if not patch_dir.exists():
        log.error(f"Patch directory not found: {patch_dir}")
        return

    if args.dry_run:
        log.info("=== DRY RUN MODE ===")

    # Cases to patch
    cases_to_patch = ["sub-r032s056", "sub-r032s027"]

    # Verify patch files exist
    missing = []
    for case in cases_to_patch:
        if not (patch_dir / f"{case}_T1w.nii.gz").exists():
            missing.append(f"{case}_T1w.nii.gz")
        if not (patch_dir / f"{case}_mask.nii.gz").exists():
            missing.append(f"{case}_mask.nii.gz")

    if missing:
        log.error(f"Missing patch files: {', '.join(missing)}")
        log.error("Download patched cases from: https://drive.switch.ch/index.php/s/XXR7O5dNFjoCrpo")
        return

    if args.dry_run:
        log.info(f"Would patch {len(cases_to_patch)} cases:")
        for case in cases_to_patch:
            log.info(f"  - {case}")
        return

    # Patch the cases
    patched = 0
    for case in cases_to_patch:
        if patch_case(case, input_dir, output_dir, patch_dir):
            patched += 1
        else:
            log.error(f"Failed to patch {case}")

    # Copy remaining cases
    log.info("Copying remaining cases...")
    remaining = copy_remaining_cases(set(cases_to_patch), input_dir, output_dir)

    log.info("=" * 50)
    log.info(f"Patch complete!")
    log.info(f"  - Patched: {patched} cases")
    log.info(f"  - Copied:  {remaining} cases")
    log.info(f"  - Total:   {patched + remaining} cases")
    log.info(f"  - Output:  {output_dir}")


if __name__ == "__main__":
    main()
