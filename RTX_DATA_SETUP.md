# RTX Data Setup Guide

## Directory Structure

On your RTX workstation, the data should be organized as:

```
/data/derrick/isles26/raw/ATLAS3_Training_Raw/
├── images/              # T1w NIfTI files (sub-xxx_ses-1_space-orig_desc-brain_T1w.nii.gz)
├── masks/               # Lesion masks (sub-xxx_ses-1_space-orig_label-lesion_desc-T1lesion_mask.nii.gz)
├── metadata/
│   └── metadata.csv     # CRITICAL: Per-scan metadata
└── inventory.json       # Data inventory (optional, used for validation)
```

## Required Files

### 1. metadata.csv

**CRITICAL** - This file must exist at `metadata/metadata.csv` with columns:
- `UID` - Unique subject ID (e.g., "sub-r001s001")
- `DAYS_POST_STROKE` - Float or NaN
- `CHRONICITY` - 1.0 (chronic) or NaN (not chronic), or NaN (unknown)
- `CHRONICITY_DERIVED` - "acute" | "subacute" | "chronic" | "unknown"
- `SITE` - Site code (e.g., "R001", "SOOP")
- `ATLAS2_DATASET` - "Training" | "SOOP" | "New"

**How to generate:**
Run the ingestion notebook (`notebooks/ingest-atlas.ipynb`) on your RTX workstation. It will create this file.

### 2. Directory Structure

The ingestion process expects:
```
RAW_DATA_ROOT/                  # Your raw ATLAS data
├── R001/
│   └── sub-r001s001_ses-1/
│       └── anat/
│           ├── sub-r001s001_ses-1_space-orig_desc-brain_T1w.nii.gz
│           └── sub-r001s001_ses-1_space-orig_label-lesion_desc-T1lesion_mask.nii.gz
├── R002/
└── ...
```

## Running Preprocessing on RTX

From the project root (`/home/derrick/projects/film-isles26`):

```bash
# 1. Copy or symlink your ATLAS data to the expected location
# The ingestion notebook should do this automatically

# 2. Run preprocessing
python pipeline/preprocessing.py \
    --config configs/config_rtx.yaml \
    --workers 8

# 3. Generate splits
python pipeline/splits.py \
    --config configs/config_rtx.yaml \
    --inspect
```

## Troubleshooting

### Missing metadata.csv
**Error:** `FileNotFoundError: ... metadata/metadata.csv`

**Solution:** Run the ingestion notebook first to generate the metadata:
```bash
# On RTX workstation
cd /home/derrick/projects/film-isles26
# Edit notebooks/ingest-atlas.ipynb if needed to set RAW_DATA_ROOT
# Then run it (or the relevant cells)
```

### Path not found
Verify the paths in `configs/config_rtx.yaml` match your actual data location:
```yaml
data:
  root: "/data/derrick/isles26/raw/ATLAS3_Training_Raw"
```

### Check if data exists
```bash
# On RTX workstation
ls -la /data/derrick/isles26/raw/ATLAS3_Training_Raw/
ls -la /data/derrick/isles26/raw/ATLAS3_Training_Raw/metadata/
```

## Summary

The `eda_summary.json` you have is **aggregate statistics only** - not sufficient for preprocessing. You need:

1. **Per-scan metadata** in `metadata/metadata.csv` format
2. **Proper directory structure** with images and masks organized by site
3. Run the ingestion notebook to generate/copy the metadata
