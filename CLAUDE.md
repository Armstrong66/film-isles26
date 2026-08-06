# CLAUDE.md — ISLES26 Project Context

## Project overview
ISLES 2026 grand challenge submission: automated ischemic stroke lesion segmentation
in native-space T1-weighted MRI. Hosted on Grand Challenge, evaluated at MICCAI 2026.
Primary author: JD (GHAiC-K Lab, KATH, Kumasi, Ghana).

## Core scientific contribution
A 3D U-Net with a pluggable **metadata conditioning gate** that explicitly encodes
clinical metadata (DAYS_POST_STROKE, CHRONICITY_DERIVED) to modulate decoder features
per disease phase. Two interchangeable conditioning tracks:
- **Track A** — FiLM (Feature-wise Linear Modulation): lightweight MLP encodes a
  5-dim metadata vector into (γ, β) scale-shift pairs injected at the bottleneck.
- **Track C** — LLM conditioning: frozen Qwen2.5-1.5B encodes a natural language
  metadata string; mean-pooled hidden state projected to (γ, β). Only projection head trained.

Track switching: one config flag (`conditioning.track: A` or `C`). All other
modules (preprocessing, dataset, loss, evaluation) are track-agnostic.

## Dataset
- **ATLAS v3.0** (ISLES26 training set), N=1453 sessions
  - ATLAS v2.0 Train+Val (N=955), SOOP (N=169), new cases (N=329)
  - 33 sites: R001–R052 + SOOP
  - Native-space skull-stripped T1w MRI, 1 lesion mask per session
  - Metadata per session: DAYS_POST_STROKE (float, nullable), CHRONICITY (1 or NaN),
    CHRONICITY_DERIVED (derived: acute/subacute/chronic/unknown), SITE, ATLAS2_DATASET
- **Key EDA findings:**
  - Orientations: RAS (60%), LAS (40%) — reorientation mandatory
  - Spacing: isotropic ~1mm³ — no resampling needed
  - Inter-site intensity CV: 1.453 — per-scan z-score normalisation mandatory
  - Lesion sizes: small <1mL (27%), medium 1-10mL (37%), large >10mL (37%)
  - No empty masks in training set; test set may contain healthy scans (expect empty output)
- **Two corrected cases** (sub-r032s056 new T1w, sub-r032s027 file fix):
  patch files at https://drive.switch.ch/index.php/s/XXR7O5dNFjoCrpo

## Repository structure
```
isles26/
├── configs/
│   ├── config.yaml          # master config — all modules read from here
│   ├── track_A.yaml         # override: conditioning.track = A
│   └── track_C.yaml         # override: conditioning.track = C
├── pipeline/
│   ├── preprocessing.py     # reorient LAS→RAS, clip p0.5/p99.5, per-scan z-score
│   ├── splits.py            # 5-fold CV, joint stratified by CHRONICITY_DERIVED × SITE
│   ├── augmentation.py      # MONAI transforms + chronicity-specific augmentation
│   ├── dataset.py           # PyTorch Dataset, metadata encoding, DataLoader factory
│   ├── conditioning.py      # FiLMConditioner (A) + LLMConditioner (C) + factory
│   ├── model.py             # 3D U-Net + FiLM injection + deep supervision (4 scales)
│   ├── loss.py              # Dice + CE + boundary focal + small-lesion upweighting
│   ├── train.py             # poly LR, AdamW, mixed precision, early stopping, CV
│   ├── evaluate.py          # all 5 official metrics via utils/eval_utils
│   ├── visualize.py         # training curves, CV summary, overlays, track comparison
│   └── tests/
│       └── test_pipeline.py # 10 test classes, ~60 unit + smoke tests
├── utils/
│   ├── __init__.py
│   └── eval_utils.py        # organizer-provided — DO NOT MODIFY
├── notebooks/
│   ├── ingest_atlas.ipynb   # one-time data ingestion from NITRC
│   ├── ingest_atlas_fix.ipynb
│   ├── eda_atlas.ipynb
│   └── smoke_tests.ipynb    # Kaggle-compatible pytest runner
├── requirements.txt
├── pytest.ini
└── README.md
```

## Key design decisions
| Decision | Choice | Reason |
|---|---|---|
| Backbone | Custom 3D U-Net (nnU-Net-style) | Full control over conditioning injection |
| Conditioning injection point | Decoder bottleneck | Full receptive field seen before modulation |
| Normalisation | Per-scan foreground z-score | Inter-site CV=1.453 makes global normalisation harmful |
| Resampling | None | EDA confirmed isotropic 1mm³ across dataset |
| Augmentation | MONAI + custom phase-specific | Acute: blur; Chronic: cavity intensity inversion |
| Loss | Dice + CE + boundary focal | Handles class imbalance + small lesion upweighting |
| Training | 5-fold CV, poly LR, AdamW | Maximises use of 1453 labelled scans |
| Deep supervision | 4 scales, weights [1,0.5,0.25,0.125] | nnU-Net default, stable gradients |
| Metadata encoding (Track A) | 5-dim vector: [days_norm, is_acute, is_subacute, is_chronic, confirmed_chronic] | Continuous + categorical combined |
| Metadata encoding (Track C) | Natural language string → LLM embedding | Leverages pretraining clinical priors |

## Official evaluation metrics (5)
From `utils/eval_utils.py` — do not reimplement:
1. **Dice** — global binary DSC
2. **Absolute Volume Difference** — mL
3. **Absolute Lesion Count Difference** — instance count |GT − Pred|
4. **Lesion-wise F1** — recognition quality (panoptica, threshold=0.25)
5. **PR-AUC** — requires soft probability map (not binary mask)

## Critical constraints
- **Native space only**: final predictions must match exact dimensions/spacing/orientation
  of input. Do not submit registered outputs.
- **Docker: T4 GPU, 32GB RAM, 10-minute hard kill** — time Track C inference before committing.
- **Test set may include healthy scans** — model must output all-zero mask when no lesion.
- **Submission**: Dockerized algorithm on Grand Challenge platform.

## Compute environments
- **Development**: Kaggle (2× T4 GPU, 30GB RAM) — primary development environment
- **Training**: RTX workstation (remote, MobaXterm SSH) — full CV training
- **Data**: Kaggle Dataset (`josephderrick/ingest-atlas`, atlas_raw/) mounted at
  `/kaggle/input/notebooks/josephderrick/ingest-atlas/atlas_raw/`

## Running the pipeline
```bash
# 1. Preprocess (run once)
python pipeline/preprocessing.py --config configs/config.yaml --workers 4

# 2. Generate splits (run once)
python pipeline/splits.py --config configs/config.yaml --inspect

# 3. Train (Track A, fold 0)
python pipeline/train.py --config configs/config.yaml --fold 0 --track A

# 4. Train all folds
python pipeline/train.py --config configs/config.yaml --fold all --track A

# 5. Evaluate with TTA
python pipeline/evaluate.py --config configs/config.yaml --fold all --tta

# 6. Switch to Track C
python pipeline/train.py --config configs/config.yaml --fold 0 --track C

# 7. Run tests
python -m pytest pipeline/tests/test_pipeline.py -v
```

## Next steps (in order)
See `## Roadmap` below.

## Roadmap

### Phase 1 — Verification (current)
- [x] Fix CHRONICITY encoding: metadata_dim 4 → 5 in config.yaml and dataset.py
- [ ] Run smoke tests on Kaggle (smoke_tests.ipynb) — all tests must pass
- [ ] Patch sub-r032s056 and sub-r032s027 in atlas_raw using `patch_cases.py`
- [ ] Re-run ingestion for v3.0 (1453 sessions), verify inventory assertions pass
- [ ] Run preprocessing on full 1453 sessions
- [ ] Generate and inspect 5-fold splits (--inspect flag)

### Phase 2 — Baseline training (Track A)
- [ ] Train fold 0 (Track A) on Kaggle — verify loss decreases, Dice > 0.3 by epoch 50
- [ ] Train all 5 folds on RTX workstation
- [ ] Evaluate with official metrics, generate stratified report
- [ ] Report CV Dice ± std for SWITCH+ abstract (deadline: August 7)

### Phase 3 — Track C experiment
- [ ] Benchmark Track A inference time on T4 (must be < 8 min per scan to leave Docker margin)
- [ ] Train Track C fold 0 — compare conditioning embedding visualisations
- [ ] Ablation: Track A vs Track C vs no-conditioning baseline
- [ ] Plot track_comparison figure (visualize.py)

### Phase 4 — Track B (uncertainty pseudo-labelling)
- [ ] Generate pseudo-labels on unlabelled test scans using Track A ensemble
- [ ] Filter by MC Dropout uncertainty, chronicity-stratified threshold
- [ ] Retrain with combined labelled + filtered pseudo-labelled set
- [ ] Compare pseudo-label vs no-pseudo-label metrics

### Phase 5 — Docker submission
- [ ] Clone organizer Docker template: github.com/ezequieldlrosa/isles26-docker-template
- [ ] Integrate inference pipeline into Docker entrypoint
- [ ] Benchmark full pipeline runtime on T4 (< 10 min hard limit)
- [ ] Submit to preliminary Docker evaluation phase (2 debug scans)
- [ ] Fix any Docker issues, submit final Docker

### Phase 6 — Paper
- [ ] Write SWITCH+ long abstract (4-6 pages, Springer LNCS)
- [ ] Include: CV Dice table, ablation table (A vs C vs baseline), stratified results
- [ ] Submit via OpenReview challenge track

## Code style
- Modular, pluggable — no spaghetti mixing of tracks
- Every non-trivial function has a docstring with Args/Returns
- Assertions at critical transformation boundaries (not just try/except)
- Minimal but informative logging (log.info for stage boundaries, log.debug for per-scan)
- Config-driven — no hardcoded paths or hyperparameters in scripts
- Type hints on all public function signatures


## Docker submission guidelines

**Status: IMPLEMENTED** ✓

Files created:
- [`entrypoint.py`](entrypoint.py) — Docker entry point script
- [`Dockerfile`](Dockerfile) — Docker build configuration
- [`test_docker_locally.py`](test_docker_locally.py) — Local testing script
- See [`README.md#docker-submission`](README.md#docker-submission) for usage

### Hard constraints
- GPU  : T4 (16GB VRAM)
- RAM  : 32GB
- Time : 10-minute hard kill for entire container
- Input : single raw native-space T1w NIfTI (skull-stripped)
- Output: binary lesion mask NIfTI matching input dimensions/spacing/orientation exactly

### Inference pipeline inside Docker
Entry point must:
1. Read input path from environment variable or fixed path per template
2. Load T1w NIfTI → reorient to RAS → clip/normalise (per-scan z-score)
3. Load all 5 fold checkpoints from /opt/algorithm/checkpoints/
4. Run forward pass on each fold model → average softmax probabilities
5. Apply 0.5 threshold → remove components < 10 voxels
6. Reorient output mask back to original input orientation
7. Save binary mask NIfTI at the expected output path

### Track selection for Docker
Use Track A only for submission — Track C LLM loading adds ~2-3 min
overhead that risks the 10-minute kill. Benchmark Track A inference
on a T4 before finalising. Target: < 7 min total including model loading.

### Model loading optimisation
- Load all 5 checkpoints once at container startup (not per scan)
- Use torch.load(..., map_location='cuda') with map_location to avoid CPU spike
- Set model.eval() and torch.no_grad() globally
- Use mixed precision (torch.cuda.amp.autocast) at inference

### Output requirements
- Same spatial dimensions as input (no resampling of output)
- Same affine matrix as input (copy directly from input NIfTI header)
- dtype: uint8, values: 0 or 1 only
- File name and path: follow organizer template exactly

### Build and test locally
# Build
docker build -t isles26-submission .

# Test with a single training scan
docker run --gpus all \
  -v /path/to/test_input:/input \
  -v /path/to/test_output:/output \
  isles26-submission

# Verify output matches input geometry
python -c "
import nibabel as nib
inp = nib.load('/path/to/test_input/image.nii.gz')
out = nib.load('/path/to/test_output/mask.nii.gz')
assert inp.shape == out.shape
assert (inp.affine == out.affine).all()
print('Geometry check passed')
"

### Dockerfile key elements
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
# Copy only what inference needs — not the full repo
COPY pipeline/model.py pipeline/conditioning.py pipeline/preprocessing.py /opt/algorithm/
COPY utils/ /opt/algorithm/utils/
COPY configs/config.yaml /opt/algorithm/
COPY checkpoints/ /opt/algorithm/checkpoints/   # 5 fold best.pth files
COPY entrypoint.py /opt/algorithm/
RUN pip install nibabel monai omegaconf panoptica --no-cache-dir
ENTRYPOINT ["python", "/opt/algorithm/entrypoint.py"]

### entrypoint.py responsibilities
- Hardcode input/output paths per organizer template
- No argparse — Docker entry is not interactive
- Catch all exceptions, log to stderr, exit 1 on failure
- Log timing per stage so you can profile the 10-min budget

### Submission checklist
- [ ] Output mask geometry matches input exactly (shape + affine)
- [ ] Container runs in < 10 min on a single scan on T4
- [ ] No internet access required inside container (all weights bundled)
- [ ] Empty mask output works correctly for healthy scans
- [ ] Test on both RAS and LAS input orientations
- [ ] Test on small and large lesion cases
- [ ] Submit to preliminary phase first (2 debug scans) before final submission


## RTX Data Paths

### Directory layout (RTX workstation)
Data lives OUTSIDE the project root — do not move it.

/data/derrick/isles26/
├── raw/
│   ├── ATLAS3_Training_Raw/          # full dataset (1453 sessions, site folders R001-R071 + SOOP)
│   └── isles26_train/
│       ├── raw/                      # ← USE THIS for the 2 corrected cases + debug samples
│       │   ├── sub-r032s056_ses-1_space-orig_desc-brain_T1w.nii.gz      # corrected T1w
│       │   ├── sub-r032s027_ses-1_space-orig_label-lesion_...mask.nii.gz # corrected mask
│       │   ├── sub-r032s013_ses-1_space-orig_label-lesion_...mask.nii.gz
│       │   └── sub-r032s018_ses-1_space-orig_label-lesion_...mask.nii.gz
│       └── preprocessed/             # ← NEVER USE — MNI space, violates challenge rules

/home/derrick/projects/film-isles26/ # project root (code only)
├── configs/config.yaml
├── pipeline/
├── utils/
└── ...

### Rules
- ALWAYS use /data/derrick/isles26/raw/ATLAS3_Training_Raw/ as DATA_ROOT
- NEVER use isles26_train/preprocessed/ — it is MNI-registered space
- isles26_train/raw/ is ONLY for patching the 2 corrected cases before preprocessing

### config.yaml — update these paths for RTX

**Status: IMPLEMENTED** ✓

Created `configs/config_rtx.yaml` with RTX paths pre-configured:
- `data.root`: "/data/derrick/isles26/raw/ATLAS3_Training_Raw"
- `data.processed_dir`: "/data/derrick/isles26/processed"
- `logging.log_dir`: "/data/derrick/isles26/logs"

For Kaggle: use `configs/config.yaml` (default paths already set)

### Patch corrected cases before running preprocessing.py
Run this once from the project root:

python - <<'EOF'
import shutil
from pathlib import Path

FIX_SRC  = Path("/data/derrick/isles26/raw/isles26_train/raw")
DATA_ROOT = Path("/data/derrick/isles26/raw/ATLAS3_Training_Raw")

fixes = [
    # Corrected T1w (over-skull-stripped)
    ("sub-r032s056_ses-1_space-orig_desc-brain_T1w.nii.gz",
     "R032/sub-r032s056/ses-1/anat"),
    # Corrected mask (file error)
    ("sub-r032s027_ses-1_space-orig_label-lesion_desc-T1lesion_mask.nii.gz",
     "R032/sub-r032s027/ses-1/anat"),
]

for filename, rel_dest in fixes:
    src  = FIX_SRC / filename
    dest = DATA_ROOT / rel_dest / filename
    assert src.exists(), f"Fix file not found: {src}"
    shutil.copy2(src, dest)
    print(f"Patched: {dest}")
EOF

### Run order on RTX (from project root)
cd /home/derrick/projects/film-isles26

python pipeline/preprocessing.py --config configs/config.yaml --workers 8
python pipeline/splits.py        --config configs/config.yaml --inspect
python pipeline/train.py         --config configs/config.yaml --fold 0 --track A
python pipeline/train.py         --config configs/config.yaml --fold all --track A
python pipeline/evaluate.py      --config configs/config.yaml --fold all --tta