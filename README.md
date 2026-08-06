# ISLES 2026 — Automated Ischemic Stroke Lesion Segmentation

[![ISLES26 Challenge](https://img.shields.io/badge/Challenge-ISLES26-blue)](https://isles26.grand-challenge.org/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-orange)](https://pytorch.org/)

Automated ischemic stroke lesion segmentation in native-space T1-weighted MRI using
the ATLAS v3.0 dataset (N=1453). Submission to ISLES 2026 Grand Challenge @ MICCAI 2026.

---

## Overview

This project implements a 3D U-Net with pluggable metadata conditioning for stroke
lesion segmentation. Two interchangeable tracks:

| Track | Method | Speed | Quality |
|-------|--------|-------|---------|
| **A** | FiLM (Feature-wise Linear Modulation) | Fast | Strong baseline |
| **C** | LLM Conditioning (Qwen2.5-1.5B) | Slower | Potentially higher quality |

---

## Quick Start

### 1. Install Dependencies

```bash
# PyTorch (CUDA 12.1)
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

# Other dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

The dataset must be preprocessed before training:

```bash
# Preprocess raw NIfTI files
python pipeline/preprocessing.py \
    --config configs/config.yaml \
    --workers 4

# Generate 5-fold CV splits
python pipeline/splits.py \
    --config configs/config.yaml \
    --inspect
```

### 3. Training

```bash
# Train a single fold (Track A)
python pipeline/train.py \
    --config configs/config.yaml \
    --fold 0 \
    --track A

# Train all 5 folds
python pipeline/train.py \
    --config configs/config.yaml \
    --fold all \
    --track A
```

### 4. Evaluation

```bash
# Evaluate with Test-Time Augmentation
python pipeline/evaluate.py \
    --config configs/config.yaml \
    --fold all \
    --tta
```

---

## Repository Structure

```
film-isles26/
├── configs/              # Configuration files
│   ├── config.yaml       # Master config (all modules read from here)
│   ├── track_A.yaml      # Track A override (FiLM conditioning)
│   └── track_C.yaml      # Track C override (LLM conditioning)
├── pipeline/
│   ├── preprocessing.py  # Reorientation, clipping, z-score normalization
│   ├── splits.py         # 5-fold CV, stratified by CHRONICITY_DERIVED × SITE
│   ├── augmentation.py   # MONAI transforms + phase-specific augmentation
│   ├── dataset.py        # PyTorch Dataset, metadata encoding, DataLoader
│   ├── conditioning.py   # FiLMConditioner (A) + LLMConditioner (C)
│   ├── model.py          # 3D U-Net + FiLM injection + deep supervision
│   ├── loss.py           # Dice + CE + boundary focal loss
│   ├── train.py          # Poly LR, AdamW, mixed precision, early stopping
│   ├── evaluate.py       # All 5 official metrics
│   ├── visualize.py      # Training curves, overlays, track comparison
│   └── tests/
│       └── test_pipeline.py  # Unit tests (~60 tests)
├── utils/
│   ├── __init__.py
│   └── eval_utils.py     # ISLES26 official evaluation metrics
├── notebooks/
│   ├── ingest_atlas.ipynb          # Data ingestion from Kaggle
│   ├── eda-atlas.ipynb             # Exploratory data analysis
│   └── smoke_tests.ipynb           # Kaggle-compatible pytest runner
├── README.md
├── CLAUDE.md             # Project context for AI assistant
├── requirements.txt
└── pytest.ini
```

---

## Dataset

**ATLAS v3.0** (ISLES26 training set):
- **N = 1453** sessions from 33 sites (R001–R052 + SOOP)
- Native-space skull-stripped T1w MRI
- Single lesion mask per session
- Metadata: DAYS_POST_STROKE, CHRONICITY (1 or NaN), CHRONICITY_DERIVED, SITE

**Key characteristics:**
- Orientations: RAS (60%), LAS (40%) — reorientation mandatory
- Spacing: isotropic ~1mm³ — no resampling needed
- Inter-site intensity CV: 1.453 — per-scan z-score required
- Lesion sizes: small <1mL (27%), medium 1-10mL (37%), large >10mL (37%)

---

## Metadata Encoding

### Track A (FiLM)

5-dimensional metadata vector:
```
[days_norm, is_acute, is_subacute, is_chronic, confirmed_chronic]
```

- `days_norm`: log1p(days) / log1p(10000) in [0, 1]
- `is_acute/subacute/chronic`: One-hot encoding from CHRONICITY_DERIVED
- `confirmed_chronic`: 1.0 if CHRONICITY == 1.0 (organizer-provided)

### Track C (LLM)

Natural language string:
```
"Stroke MRI scan from site R001. Time since stroke: 45 days. Phase: chronic. Task: segment the ischemic lesion."
```

---

## Official Evaluation Metrics (5)

1. **Dice Score** — global binary DSC
2. **Absolute Volume Difference** — mL
3. **Absolute Lesion Count Difference** — instance count |GT − Pred|
4. **Lesion-wise F1** — recognition quality (panoptica, threshold=0.25)
5. **PR-AUC** — requires soft probability map (not binary mask)

---

## Important Notes

### Disk Space Requirements

- Raw data: ~8GB (compressed) / ~16GB (extracted)
- Preprocessed: ~10-12GB
- Training: Additional ~5GB for checkpoints

**Kaggle users**: Use `/kaggle/temp` for extraction to avoid 20GB working directory limit.

### Docker Submission Constraints

- **Time limit**: 10 minutes per scan on T4 GPU
- **Memory**: 32GB RAM
- **Native space only**: No registration to MNI allowed in final output
- **Track A only**: Track C adds ~2-3 min LLM loading overhead

Track C (LLM) is a time risk — benchmark Track A first.

---

## Docker Submission

### Overview

This project is designed for Grand Challenge Docker submission. The pipeline:
1. Loads a single T1w NIfTI input
2. Reorients to RAS, clips, and normalizes
3. Runs ensemble prediction across 5 fold models
4. Applies thresholding and connected component removal
5. Reorients output back to input orientation

### Prerequisites

- Docker installed (version 20.10+ recommended)
- Trained model checkpoints (`fold_0_best.pth` through `fold_4_best.pth`)
- Test NIfTI scan (can be any ATLAS training scan)

### Quick Start

#### 1. Build the Docker Image

```bash
# From the project root (where Dockerfile is located)
docker build -t isles26-submission .
```

#### 2. Prepare Checkpoints

The Docker image expects checkpoints in `/opt/algorithm/checkpoints/`:
```bash
mkdir -p checkpoints
cp path/to/fold_0_best.pth checkpoints/
cp path/to/fold_1_best.pth checkpoints/
cp path/to/fold_2_best.pth checkpoints/
cp path/to/fold_3_best.pth checkpoints/
cp path/to/fold_4_best.pth checkpoints/
```

#### 3. Build with Checkpoints

**Option A: Copy checkpoints into image (recommended for final submission)**
```bash
# Edit Dockerfile line 46 to: COPY checkpoints/ /opt/algorithm/checkpoints/
docker build -t isles26-submission .
```

**Option B: Mount checkpoints at runtime (for testing)**
```bash
docker build -t isles26-submission .
docker run --gpus all \
  -v /path/to/checkpoints:/opt/algorithm/checkpoints \
  isles26-submission
```

#### 4. Run Inference

```bash
# Test with a local scan
docker run --gpus all \
  -v /path/to/test_input:/input \
  -v /path/to/test_output:/output \
  isles26-submission
```

#### 5. Verify Output Geometry

```bash
python -c "
import nibabel as nib
import numpy as np

inp = nib.load('/path/to/test_input/image.nii.gz')
out = nib.load('/path/to/test_output/mask.nii.gz')

assert inp.shape == out.shape, f'Shape mismatch: {inp.shape} vs {out.shape}'
assert np.allclose(inp.affine, out.affine), 'Affine mismatch'

data = out.get_fdata()
assert set(np.unique(data)).issubset({0.0, 1.0}), 'Non-binary values found'

print('Geometry check passed!')
"
```

### Local Testing (RTX Workstation)

Before building Docker, test locally on your RTX workstation:

```bash
# Install dependencies
pip install nibabel monai omegaconf scipy scikit-learn

# Run local test (assumes checkpoints exist)
python test_docker_locally.py \
    --input /path/to/test_scan.nii.gz \
    --output /path/to/test_output.nii.gz \
    --checkpoint /path/to/checkpoint/fold_0_best.pth \
    --no-tta
```

**Expected timing on RTX 3090:**
- Model loading: ~10-15s
- Preprocessing: ~5-10s
- Inference (no TTA): ~3-5s
- Total: < 30s per scan (well under 10-min limit)

### Dockerfile Customization

Key files in the Docker image:
- `/opt/algorithm/entrypoint.py` — Main inference script
- `/opt/algorithm/pipeline/` — Pipeline modules
- `/opt/algorithm/utils/` — Evaluation utilities
- `/opt/algorithm/checkpoints/` — 5 fold models

To add benchmark logging to entrypoint, uncomment the timing sections.

### Grand Challenge Submission Checklist

- [ ] Output mask geometry matches input exactly (shape + affine)
- [ ] Container runs in < 10 min on a single scan on T4
- [ ] No internet access required inside container (all weights bundled)
- [ ] Empty mask output works correctly for healthy scans
- [ ] Test on both RAS and LAS input orientations
- [ ] Test on small and large lesion cases
- [ ] Submit to preliminary phase first (2 debug scans)

---

## Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Backbone | Custom 3D U-Net | Full control over conditioning injection |
| Conditioning | Decoder bottleneck | Full receptive field before modulation |
| Normalization | Per-scan foreground z-score | Inter-site CV=1.453 |
| Augmentation | MONAI + phase-specific | Acute: blur; Chronic: cavity inversion |
| Loss | Dice + CE + boundary focal | Class imbalance + small lesion upweighting |
| Training | 5-fold CV, poly LR, AdamW | Maximize use of 1453 scans |

---

## Troubleshooting

### Common Issues

1. **Memory error during preprocessing**
   - Use `/kaggle/temp` instead of `/kaggle/working`
   - Reduce `num_workers` in preprocessing config

2. **CUDA out of memory**
   - Reduce `training.batch_size`
   - Enable `training.mixed_precision`

3. **Missing CHRONICITY column**
   - Re-run preprocessing with latest metadata
   - Check `metadata/metadata.csv` exists

---

## Project Timeline

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | In progress | Verification, smoke tests, patching corrected cases |
| Phase 2 | Pending | Baseline training (Track A) |
| Phase 3 | Pending | Track C experiment |
| Phase 4 | Pending | Track B pseudo-labelling |
| Phase 5 | Pending | Docker submission |
| Phase 6 | Pending | Paper writing |

See `CLAUDE.md#roadmap` for details.

---

## Author

**Joseph Derrick**  
GHAiC-K Lab, Kumasi, Ghana

---

## License

MIT License
