# Metadata CSV Generation for RTX

## Quick Start

On your RTX workstation, run:

```bash
cd /home/derrick/projects/film-isles26

python generate_metadata_csv.py \
    --root /data/derrick/isles26/raw/ATLAS3_Training_Raw \
    --output /data/derrick/isles26/raw/ATLAS3_Training_Raw/metadata/metadata.csv
```

This will scan all per-session `*metadata.csv` files and aggregate them into `metadata/metadata.csv`.

## How It Works

The script:
1. Scans `ATLAS3_Training_Raw/` recursively for `*_metadata.csv` files
2. Extracts per-session metadata (DAYS_POST_STROKE, CHRONICITY, etc.)
3. Derives CHRONICITY_DERIVED from DAYS_POST_STROKE
4. Outputs a master `metadata/metadata.csv` with columns:
   - `UID`: Unique session ID (format: `R001__sub-r001s001__ses-1`)
   - `SESSION_ID`: Session identifier (format: `sub-r001s001_ses-1`)
   - `DAYS_POST_STROKE`: Float or NaN
   - `CHRONICITY`: 1.0 (chronic) or NaN
   - `CHRONICITY_DERIVED`: "acute" | "subacute" | "chronic" | "unknown"
   - `SITE`: Site code (e.g., "R001", "SOOP")
   - `ATLAS2_DATASET`: "Training"

## Expected Directory Structure

```
ATLAS3_Training_Raw/
├── R001/
│   └── sub-r001s001/
│       └── ses-1/
│           └── anat/
│               ├── sub-r001s001_ses-1_metadata.csv     ← These files
│               ├── sub-r001s001_ses-1_space-orig_desc-brain_T1w.nii.gz
│               └── sub-r001s001_ses-1_space-orig_label-lesion_desc-T1lesion_mask.nii.gz
└── ...
```

## Verification

After generation, verify the metadata.csv exists and has the expected content:

```bash
ls -la /data/derrick/isles26/raw/ATLAS3_Training_Raw/metadata/
head -5 /data/derrick/isles26/raw/ATLAS3_Training_Raw/metadata/metadata.csv
wc -l /data/derrick/isles26/raw/ATLAS3_Training_Raw/metadata/metadata.csv
```

Expected: ~653 training sessions (plus any testing sessions).

## If No Per-Session Metadata Files Exist

If the `*_metadata.csv` files are not present in your data:

1. **Check the raw data structure** - The ingestion notebook should have created them
2. **Contact the data provider** - They may need to provide per-session metadata
3. **Manually create metadata.csv** - If all else fails, you can create a minimal CSV with:
   - UID, DAYS_POST_STROKE (can be NaN), CHRONICITY_DERIVED, SITE, ATLAS2_DATASET

## After Generation

Run preprocessing:

```bash
python pipeline/preprocessing.py \
    --config configs/config_rtx.yaml \
    --workers 8
```