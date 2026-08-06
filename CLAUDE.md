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
- [ ] Fix CHRONICITY encoding: metadata_dim 4 → 5 in config.yaml and dataset.py
- [ ] Run smoke tests on Kaggle (smoke_tests.ipynb) — all 60 tests must pass
- [ ] Patch sub-r032s056 and sub-r032s027 in atlas_raw
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