#!/usr/bin/env python3
"""
generate_metadata_csv.py
------------------------
Aggregates per-session metadata CSVs into a master metadata.csv file.

This script is needed because the ATLAS v3.0 raw data stores metadata per-session
(e.g., R001/sub-r001s001/ses-1/anat/sub-r001s001_ses-1_metadata.csv),
but the preprocessing pipeline expects a single metadata/metadata.csv.

Usage:
    python generate_metadata_csv.py --root /data/derrick/isles26/raw/ATLAS3_Training_Raw --output /data/derrick/isles26/raw/ATLAS3_Training_Raw/metadata/metadata.csv
"""

import argparse
from pathlib import Path
import pandas as pd


def derive_chronicity(days):
    """Derive chronicity from DAYS_POST_STROKE."""
    if pd.isna(days):
        return "unknown"
    if days <= 7:
        return "acute"
    if days <= 90:
        return "subacute"
    return "chronic"


def find_raw_sessions(root: Path) -> set:
    """
    Find all session UIDs from raw image files (mirrors Kaggle's file-glob approach).
    Returns set of UIDs like 'SOOP__sub-soop1650__ses-1'
    """
    import re

    sessions = set()

    for nii_file in root.rglob("*_T1w.nii.gz"):
        # Extract site, subject, session from path
        # Path example: SOOP/sub-soop1650/ses-1/anat/sub-soop1650_ses-1_space-orig_desc-brain_T1w.nii.gz
        parts = nii_file.relative_to(root).parts
        if len(parts) < 3:
            continue

        site = parts[0]  # SOOP, R001, etc.

        # Extract subject and session from path components
        # subject_part could be: sub-soop1650
        # session_part could be: ses-1
        subject_part = parts[1] if len(parts) > 1 else None
        session_part = parts[2] if len(parts) > 2 else None

        if not subject_part or not session_part:
            continue

        # Extract subject and session from sub-XXX/ses-XXX naming
        subject_match = re.search(r"sub-(.+)", subject_part)
        session_match = re.search(r"ses-(\d+)", session_part)

        if subject_match and session_match:
            subject = subject_match.group(1)
            session_num = session_match.group(1)
            # Use full subject string from path to match metadata
            uid = f"{site}__{subject_part}__{session_part}"
            sessions.add(uid)

    return sessions


def aggregate_metadata(root: Path, output: Path, expected_total: int = None) -> None:
    """
    Aggregate per-session metadata CSVs into a master metadata.csv.

    Expected directory structure:
        root/
          R001/
            sub-r001s001/
              ses-1/
                anat/
                  sub-r001s001_ses-1_metadata.csv
                  sub-r001s001_ses-1_space-orig_desc-brain_T1w.nii.gz
                  sub-r001s001_ses-1_space-orig_label-lesion_desc-T1lesion_mask.nii.gz
          R002/
          ...

    Output:
        metadata.csv with columns: UID, DAYS_POST_STROKE, CHRONICITY, CHRONICITY_DERIVED, SITE, ATLAS2_DATASET
    """
    meta_records = []
    raw_sessions = find_raw_sessions(root)

    # Find all per-session metadata CSVs
    for csv_file in sorted(root.rglob("*_metadata.csv")):
        try:
            df = pd.read_csv(csv_file)
            if df.empty:
                continue

            # Get subject info from path
            # Path: .../R001/sub-r001s001/ses-1/anat/sub-r001s001_ses-1_metadata.csv
            parts = csv_file.relative_to(root).parts
            if len(parts) < 3:
                continue

            site = parts[0]  # R001, SOOP, etc.
            subject_part = parts[1]  # sub-r001s001, sub-soop1650, etc.
            session_part = parts[2]  # ses-1, ses-2, etc.

            # Extract SESSION_ID from filename (matches the naming in raw files)
            session_id = csv_file.stem  # sub-r001s001_ses-1, sub-soop1650_ses-1, etc.

            # Build UID using same format as find_raw_sessions
            uid = f"{site}__{subject_part}__{session_part}"

            # Get metadata values
            days = df["DAYS_POST_STROKE"].iloc[0] if "DAYS_POST_STROKE" in df.columns else None
            chronicity = df["CHRONICITY"].iloc[0] if "CHRONICITY" in df.columns else None

            # Derive chronicity from days if not provided
            if pd.isna(days):
                chronicity_derived = "unknown"
            else:
                chronicity_derived = derive_chronicity(days)

            # Determine ATLAS2_DATASET (Training vs Testing)
            # Assume Training for most, adjust if needed
            atlas2_dataset = "Training"

            meta_records.append({
                "UID": uid,
                "SESSION_ID": session_id,
                "DAYS_POST_STROKE": days,
                "CHRONICITY": chronicity,
                "CHRONICITY_DERIVED": chronicity_derived,
                "SITE": site,
                "ATLAS2_DATASET": atlas2_dataset,
            })

        except Exception as e:
            print(f"  WARN: Failed to process {csv_file}: {e}")

    if not meta_records:
        raise ValueError("No metadata records found. Check directory structure.")

    # Create DataFrame
    df_meta = pd.DataFrame(meta_records)

    # Ensure output directory exists
    output.parent.mkdir(parents=True, exist_ok=True)

    # Save
    df_meta.to_csv(output, index=False)
    print(f"Saved metadata.csv: {len(df_meta)} records to {output}")

    # Print summary
    print(f"\n--- Summary ---")
    print(f"Total sessions: {len(df_meta)}")
    print(f"DAYS_POST_STROKE known: {df_meta['DAYS_POST_STROKE'].notna().sum()}")
    print(f"CHRONICITY_DERIVED distribution:")
    print(df_meta["CHRONICITY_DERIVED"].value_counts())
    print(f"\nSites:")
    print(df_meta["SITE"].value_counts().head(10))

    # Validation against raw sessions (informational only)
    meta_sessions = set(df_meta["UID"])
    matched = raw_sessions & meta_sessions

    print(f"\n--- Validation ---")
    print(f"Raw sessions from image files: {len(raw_sessions)}")
    print(f"Metadata sessions: {len(meta_sessions)}")
    print(f"Sessions matched: {len(matched)}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-session metadata CSVs into master metadata.csv"
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root path to ATLAS3_Training_Raw directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="metadata/metadata.csv",
        help="Output path for metadata.csv"
    )

    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output)

    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    print(f"Aggregating metadata from: {root}")
    aggregate_metadata(root, output)


if __name__ == "__main__":
    main()
