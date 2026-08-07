#!/usr/bin/env python3
"""
flatten_images.py
-----------------
Flattens the ATLAS v3.0 nested image structure into a flat directory.

ATLAS v3.0 structure:
    root/
      SITE/
        sub-SITExxx/
          ses-1/
            anat/
              sub-SITExxx_ses-1_space-orig_desc-brain_T1w.nii.gz
              sub-SITExxx_ses-1_space-orig_label-lesion_desc-T1lesion_mask.nii.gz

Flattened structure (for preprocessing):
    root/images/
      sub-SITExxx_ses-1_T1w.nii.gz
    root/masks/
      sub-SITExxx_ses-1_rater1.nii.gz

Usage:
    python flatten_images.py --root /data/derrick/isles26/raw/ATLAS3_Training_Raw
"""

import argparse
from pathlib import Path
import shutil
import re


def flatten_images(root: Path, images_dir: str = "images", masks_dir: str = "masks") -> None:
    """
    Flatten ATLAS v3.0 nested structure into flat images/ and masks/ directories.
    """
    img_dest = root / images_dir
    mask_dest = root / masks_dir
    img_dest.mkdir(parents=True, exist_ok=True)
    mask_dest.mkdir(parents=True, exist_ok=True)

    count = 0
    for nii_file in root.rglob("*_T1w.nii.gz"):
        # Extract path components
        # Example: SOOP/sub-soop1650/ses-1/anat/sub-soop1650_ses-1_space-orig_desc-brain_T1w.nii.gz
        parts = nii_file.relative_to(root).parts
        if len(parts) < 4:
            continue

        site = parts[0]
        subject_part = parts[1]  # sub-soop1650
        session_part = parts[2]  # ses-1

        # Build UID: sub-SITExxx_ses-1
        uid = f"{subject_part}_{session_part}"

        # Create new filename
        new_name = f"{uid}_T1w.nii.gz"
        dest_path = img_dest / new_name

        # Copy file
        shutil.copy2(nii_file, dest_path)
        count += 1

        # Also look for corresponding mask
        mask_name = f"{uid}_space-orig_label-lesion_desc-T1lesion_mask.nii.gz"
        mask_src = nii_file.parent / mask_name
        if mask_src.exists():
            mask_dest_name = f"{uid}_rater1.nii.gz"
            shutil.copy2(mask_src, mask_dest / mask_dest_name)

    print(f"Flattened {count} images to {img_dest}")
    print(f"Flattened masks to {mask_dest}")


def main():
    parser = argparse.ArgumentParser(
        description="Flatten ATLAS v3.0 nested image structure"
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root path to ATLAS3_Training_Raw directory"
    )
    parser.add_argument(
        "--images",
        type=str,
        default="images",
        help="Output subdirectory for images (default: images)"
    )
    parser.add_argument(
        "--masks",
        type=str,
        default="masks",
        help="Output subdirectory for masks (default: masks)"
    )

    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    flatten_images(root, args.images, args.masks)


if __name__ == "__main__":
    main()
