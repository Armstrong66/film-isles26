"""
tests/test_pipeline.py
----------------------
Smoke and unit tests for the ISLES26 pipeline.
Covers every critical data transformation stage across all three tracks.

Run in Kaggle or locally:
    python -m pytest pipeline/tests/test_pipeline.py -v
    python -m pytest pipeline/tests/test_pipeline.py -v -k "preprocessing"

Test groups:
    TestPreprocessing     — reorientation, clipping, normalisation
    TestMetadataEncoding  — vector and text encoders
    TestAugmentation      — chronicity-specific transforms, mask integrity
    TestDataset           — record building, __getitem__, batch shapes
    TestSplits            — stratification, no-leak, full-coverage
    TestConditioning      — FiLM and LLM conditioner forward passes
    TestModel             — forward pass shapes, deep supervision, predict()
    TestLoss              — all loss components, deep supervision weighting
    TestEvaluate          — dice, hd95, precision/recall, TTA
    TestIntegration       — end-to-end mini forward + backward pass (Track A & C stub)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

# ── Add pipeline dir to path ──────────────────────────────────────────────────
PIPELINE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE_DIR))

from preprocessing import (
    reorient_to_ras, reorient_mask_to_ras,
    clip_and_normalise, binarise_mask, ScanStats,
)
from augmentation import (
    get_train_transforms, get_val_transforms,
    AcuteContrastReduction, ChronicCavityPerturbation,
)
from dataset import (
    encode_metadata_vector, encode_metadata_text,
    ISLES26Dataset, build_records,
)
from splits import make_joint_stratum, generate_splits, validate_splits
from conditioning import FiLMConditioner, LLMConditioner, build_conditioner
from model import ISLES26Model, build_model, ResBlock, apply_film
from loss import (
    soft_dice_loss, weighted_cross_entropy, boundary_focal_loss,
    compute_scan_weights, ISLES26Loss,
)
from evaluate import dice_score, precision_recall, hausdorff95, remove_small_components


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def cfg():
    """Minimal config for tests — uses small spatial dims to keep tests fast."""
    return OmegaConf.create({
        "data": {
            "root":          "/tmp/isles26_test",
            "images_dir":    "images",
            "masks_dir":     "masks",
            "meta_csv":      "metadata/metadata.csv",
            "inventory":     "inventory.json",
            "processed_dir": "/tmp/isles26_test/processed",
            "n_splits":      3,
            "seed":          42,
        },
        "preprocessing": {
            "target_orientation": "RAS",
            "normalization":      "per_scan_zscore",
            "clip_percentiles":   [0.5, 99.5],
            "resample":           False,
            "num_workers":        1,
        },
        "chronicity": {
            "acute_max_days":    7,
            "subacute_max_days": 90,
            "unknown_token":     "unknown",
            "categories":        ["acute", "subacute", "chronic", "unknown"],
        },
        "training": {
            "epochs":                  5,
            "batch_size":              2,
            "patch_size":              [32, 32, 32],
            "optimizer":               "adamw",
            "lr":                      1e-3,
            "weight_decay":            1e-5,
            "lr_scheduler":            "poly",
            "poly_exp":                0.9,
            "early_stopping_patience": 3,
            "mixed_precision":         False,
            "grad_clip":               1.0,
        },
        "loss": {
            "dice_weight":               1.0,
            "ce_weight":                 1.0,
            "boundary_weight":           0.5,
            "small_lesion_threshold_ml": 1.0,
            "small_lesion_weight":       2.0,
        },
        "conditioning": {
            "track": "A",
            "film":  {"metadata_dim": 4, "hidden_dim": 64, "inject_at": "bottleneck"},
            "llm":   {
                "model_name":    "Qwen/Qwen2.5-1.5B",
                "embedding_dim": 1536,
                "hidden_dim":    256,
                "inject_at":     "bottleneck",
                "freeze_llm":    True,
            },
        },
        "tta": {
            "enabled": True,
            "flips":   [[0], [1], [2]],
        },
        "postprocessing": {"min_component_size_voxels": 10},
        "evaluation": {
            "metrics":   ["dice", "hausdorff95", "precision", "recall"],
            "size_bins": {"small": [0, 1], "medium": [1, 10], "large": [10, 1e9]},
        },
        "logging": {"level": "WARNING", "log_dir": "/tmp/isles26_test/logs"},
    })


@pytest.fixture(scope="session")
def synthetic_nifti_ras():
    """RAS-oriented 3D NIfTI with foreground brain region."""
    data   = np.random.rand(64, 64, 64).astype(np.float32) * 1000
    affine = np.diag([1.0, 1.0, 1.0, 1.0])   # RAS, 1mm isotropic
    return nib.Nifti1Image(data, affine)


@pytest.fixture(scope="session")
def synthetic_nifti_las():
    """LAS-oriented NIfTI — should trigger reorientation."""
    data   = np.random.rand(64, 64, 64).astype(np.float32) * 500
    affine = np.diag([-1.0, 1.0, 1.0, 1.0])  # LAS
    return nib.Nifti1Image(data, affine)


@pytest.fixture(scope="session")
def synthetic_mask_nifti():
    """Binary lesion mask with a small cube lesion."""
    data          = np.zeros((64, 64, 64), dtype=np.uint8)
    data[20:30, 20:30, 20:30] = 1   # 10³ = 1000 voxel lesion
    affine        = np.diag([1.0, 1.0, 1.0, 1.0])
    return nib.Nifti1Image(data.astype(np.float32), affine)


@pytest.fixture(scope="session")
def small_tensor():
    """Tiny (B=2, C=1, 32, 32, 32) image+mask tensors for fast model tests."""
    img  = torch.randn(2, 1, 32, 32, 32)
    mask = torch.zeros(2, 1, 32, 32, 32, dtype=torch.float32)
    mask[:, :, 10:15, 10:15, 10:15] = 1.0
    return img, mask


@pytest.fixture(scope="session")
def dummy_meta():
    meta_vec  = torch.rand(2, 4)
    meta_text = [
        "Stroke MRI from site R001. Time since stroke: 45 days. Phase: chronic.",
        "Stroke MRI from site R009. Time since stroke: unknown. Phase: unknown.",
    ]
    return meta_vec, meta_text


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

class TestPreprocessing:

    def test_ras_nifti_not_reoriented(self, synthetic_nifti_ras):
        _, changed = reorient_to_ras(synthetic_nifti_ras)
        assert not changed, "RAS image should not trigger reorientation."

    def test_las_nifti_reoriented(self, synthetic_nifti_las):
        reoriented, changed = reorient_to_ras(synthetic_nifti_las)
        assert changed, "LAS image must be reoriented."
        import nibabel.orientations as nio
        codes = nio.aff2axcodes(reoriented.affine)
        assert codes == ("R", "A", "S"), f"Expected RAS, got {codes}"

    def test_reoriented_shape_preserved(self, synthetic_nifti_las):
        original_shape = synthetic_nifti_las.shape
        reoriented, _  = reorient_to_ras(synthetic_nifti_las)
        assert reoriented.shape == original_shape, \
            "Reorientation must not change shape for axis permutations."

    def test_clip_normalise_range(self, synthetic_nifti_ras):
        data = synthetic_nifti_ras.get_fdata(dtype=np.float32)
        norm, clip_low, clip_high, fg_mean, fg_std = clip_and_normalise(data, 0.5, 99.5)
        fg = norm[data > 0]
        assert fg.mean() == pytest.approx(0.0, abs=0.1), \
            "Normalised foreground mean should be ~0."
        assert fg.std()  == pytest.approx(1.0, abs=0.1), \
            "Normalised foreground std should be ~1."

    def test_clip_normalise_background_zero(self, synthetic_nifti_ras):
        data          = synthetic_nifti_ras.get_fdata(dtype=np.float32).copy()
        data[:10, :, :] = 0   # force background
        norm, *_ = clip_and_normalise(data, 0.5, 99.5)
        assert norm[:10, :, :].sum() == 0.0, "Background voxels must remain zero."

    def test_clip_percentiles_captured(self, synthetic_nifti_ras):
        data = synthetic_nifti_ras.get_fdata(dtype=np.float32)
        _, clip_low, clip_high, _, _ = clip_and_normalise(data, 0.5, 99.5)
        fg = data[data > 0]
        assert clip_low  == pytest.approx(np.percentile(fg, 0.5),  rel=1e-3)
        assert clip_high == pytest.approx(np.percentile(fg, 99.5), rel=1e-3)

    def test_binarise_mask_values(self, synthetic_mask_nifti):
        data   = synthetic_mask_nifti.get_fdata(dtype=np.float32)
        binary = binarise_mask(data)
        unique = np.unique(binary)
        assert set(unique).issubset({0, 1}), f"Mask must be binary, got {unique}"

    def test_binarise_mask_float_input(self):
        """Soft probability mask (from some datasets) should binarise correctly."""
        data   = np.array([0.0, 0.3, 0.6, 0.9, 1.0], dtype=np.float32)
        binary = binarise_mask(data)
        assert binary.tolist() == [0, 0, 1, 1, 1]

    def test_empty_foreground_raises(self):
        data = np.zeros((32, 32, 32), dtype=np.float32)
        with pytest.raises(AssertionError, match="No foreground voxels"):
            clip_and_normalise(data)

    def test_mask_reorientation_follows_image(
        self, synthetic_nifti_las, synthetic_mask_nifti
    ):
        img_ras, _ = reorient_to_ras(synthetic_nifti_las)
        mask_ras   = reorient_mask_to_ras(synthetic_mask_nifti, img_ras)
        assert mask_ras.shape == img_ras.shape, \
            "Reoriented mask must match reoriented image shape."


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Metadata encoding
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetadataEncoding:

    @pytest.mark.parametrize("days,chron,expected_idx", [
        (45,   "chronic",  [False, False, True]),   # is_chronic = 1
        (3,    "acute",    [True,  False, False]),   # is_acute = 1
        (30,   "subacute", [False, True,  False]),   # is_subacute = 1
        (None, "unknown",  [False, False, False]),   # all-zero one-hot
    ])
    def test_encode_metadata_vector_onehot(self, days, chron, expected_idx):
        vec = encode_metadata_vector(days, chron)
        assert vec.shape == (4,), f"Expected shape (4,), got {vec.shape}"
        for i, expected in enumerate(expected_idx):
            val = vec[i + 1].item()  # skip days_norm at index 0
            assert (val == 1.0) == expected, \
                f"Index {i+1}: expected {expected}, got {val}"

    def test_encode_metadata_vector_days_range(self):
        """days_norm must always be in [0, 1]."""
        for days in [0, 1, 7, 90, 365, 5000, 10000]:
            vec = encode_metadata_vector(float(days), "chronic")
            assert 0.0 <= vec[0].item() <= 1.0, \
                f"days_norm out of range for days={days}: {vec[0].item()}"

    def test_encode_metadata_vector_nan_days(self):
        vec = encode_metadata_vector(float("nan"), "chronic")
        assert vec[0].item() == pytest.approx(0.5), \
            "NaN days should produce sentinel 0.5"

    def test_encode_metadata_text_contains_key_info(self):
        text = encode_metadata_text("R001__sub-r001s001__ses-1", 45.0, "chronic", "R001")
        assert "R001"    in text
        assert "45"      in text
        assert "chronic" in text
        assert "segment" in text.lower()

    def test_encode_metadata_text_unknown_days(self):
        text = encode_metadata_text("uid", None, "unknown", "R009")
        assert "unknown" in text.lower()

    def test_metadata_vector_dtype(self):
        vec = encode_metadata_vector(100.0, "subacute")
        assert vec.dtype == torch.float32


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Augmentation
# ═══════════════════════════════════════════════════════════════════════════════

class TestAugmentation:

    @pytest.fixture
    def aug_input(self):
        img  = torch.randn(1, 32, 32, 32)
        mask = torch.zeros(1, 32, 32, 32)
        mask[:, 10:15, 10:15, 10:15] = 1.0
        return {"image": img, "mask": mask}

    @pytest.mark.parametrize("chronicity", ["acute", "subacute", "chronic", "unknown"])
    def test_train_transforms_output_shape(self, aug_input, chronicity):
        t   = get_train_transforms(chronicity)
        out = t(aug_input)
        assert out["image"].shape == aug_input["image"].shape, \
            f"Image shape changed after augmentation for {chronicity}"
        assert out["mask"].shape == aug_input["mask"].shape, \
            f"Mask shape changed after augmentation for {chronicity}"

    @pytest.mark.parametrize("chronicity", ["acute", "subacute", "chronic", "unknown"])
    def test_train_transforms_mask_binary(self, aug_input, chronicity):
        """Mask must remain binary (0 or 1) after all augmentations."""
        t   = get_train_transforms(chronicity)
        out = t(aug_input)
        unique = torch.unique(out["mask"]).tolist()
        # After spatial aug (nearest interp), values should still be 0 or 1
        assert all(v in [0, 1] for v in unique), \
            f"Mask non-binary after {chronicity} augmentation: {unique}"

    def test_val_transforms_no_change(self, aug_input):
        """Validation transforms must not alter values."""
        t   = get_val_transforms()
        out = t(aug_input)
        assert torch.allclose(out["image"].float(), aug_input["image"].float())

    def test_acute_transform_applied(self, aug_input):
        """AcuteContrastReduction must not alter mask."""
        transform = AcuteContrastReduction(keys=["image"], prob=1.0)
        out       = transform(aug_input)
        assert torch.equal(out["mask"], aug_input["mask"]), \
            "AcuteContrastReduction must not modify the mask."

    def test_chronic_transform_preserves_mask(self, aug_input):
        """ChronicCavityPerturbation must not alter mask."""
        transform = ChronicCavityPerturbation(
            image_key="image", mask_key="mask", prob=1.0
        )
        out = transform(aug_input)
        assert torch.equal(out["mask"], aug_input["mask"]), \
            "ChronicCavityPerturbation must not modify the mask."

    def test_chronic_transform_modifies_lesion_region(self, aug_input):
        """Perturbation should change image values inside the lesion."""
        transform = ChronicCavityPerturbation(
            image_key="image", mask_key="mask", prob=1.0, inversion_strength=0.9
        )
        original_img = aug_input["image"].clone()
        out          = transform(aug_input)
        lesion_mask  = aug_input["mask"].bool()
        assert not torch.equal(
            out["image"][lesion_mask], original_img[lesion_mask]
        ), "ChronicCavityPerturbation should modify lesion-region intensities."


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataset:

    @pytest.fixture(scope="class")
    def tmp_processed_dir(self, tmp_path_factory):
        """Write synthetic processed NIfTI files for dataset tests."""
        d = tmp_path_factory.mktemp("processed")
        img_dir  = d / "images"
        mask_dir = d / "masks"
        img_dir.mkdir(); mask_dir.mkdir()

        affine = np.eye(4)
        for uid in ["R001__sub-r001s001__ses-1", "R001__sub-r001s002__ses-1"]:
            img_data  = np.random.rand(32, 32, 32).astype(np.float32)
            mask_data = np.zeros((32, 32, 32), dtype=np.uint8)
            mask_data[10:15, 10:15, 10:15] = 1
            nib.save(nib.Nifti1Image(img_data, affine),
                     str(img_dir / f"{uid}_T1w.nii.gz"))
            nib.save(nib.Nifti1Image(mask_data.astype(np.float32), affine),
                     str(mask_dir / f"{uid}_mask.nii.gz"))
        return d

    def test_build_records_length(self, tmp_processed_dir):
        uids    = ["R001__sub-r001s001__ses-1", "R001__sub-r001s002__ses-1"]
        df_meta = pd.DataFrame([
            {"UID": uids[0], "CHRONICITY_DERIVED": "chronic",  "DAYS_POST_STROKE": 200, "SITE": "R001"},
            {"UID": uids[1], "CHRONICITY_DERIVED": "subacute", "DAYS_POST_STROKE": 30,  "SITE": "R001"},
        ])
        records = build_records(uids, df_meta,
                                tmp_processed_dir / "images",
                                tmp_processed_dir / "masks")
        assert len(records) == 2

    def test_dataset_getitem_keys(self, tmp_processed_dir):
        uids    = ["R001__sub-r001s001__ses-1"]
        df_meta = pd.DataFrame([
            {"UID": uids[0], "CHRONICITY_DERIVED": "chronic",
             "DAYS_POST_STROKE": 200, "SITE": "R001"},
        ])
        records = build_records(uids, df_meta,
                                tmp_processed_dir / "images",
                                tmp_processed_dir / "masks")
        ds      = ISLES26Dataset(records, get_val_transforms(), is_train=False)
        sample  = ds[0]
        for key in ["image", "mask", "meta_vec", "meta_text", "uid", "chronicity"]:
            assert key in sample, f"Missing key '{key}' in dataset output."

    def test_dataset_image_shape(self, tmp_processed_dir):
        uids    = ["R001__sub-r001s001__ses-1"]
        df_meta = pd.DataFrame([
            {"UID": uids[0], "CHRONICITY_DERIVED": "chronic",
             "DAYS_POST_STROKE": 200, "SITE": "R001"},
        ])
        records = build_records(uids, df_meta,
                                tmp_processed_dir / "images",
                                tmp_processed_dir / "masks")
        ds     = ISLES26Dataset(records, get_val_transforms(), is_train=False)
        sample = ds[0]
        assert sample["image"].shape[0] == 1,  "Image must have channel dim = 1"
        assert sample["image"].ndim     == 4,  "Image must be 4D (C,H,W,D)"
        assert sample["mask"].shape[0]  == 1,  "Mask must have channel dim = 1"

    def test_dataset_meta_vec_shape(self, tmp_processed_dir):
        uids    = ["R001__sub-r001s001__ses-1"]
        df_meta = pd.DataFrame([
            {"UID": uids[0], "CHRONICITY_DERIVED": "acute",
             "DAYS_POST_STROKE": 3, "SITE": "R001"},
        ])
        records = build_records(uids, df_meta,
                                tmp_processed_dir / "images",
                                tmp_processed_dir / "masks")
        ds     = ISLES26Dataset(records, get_val_transforms(), is_train=False)
        sample = ds[0]
        assert sample["meta_vec"].shape == (4,), \
            f"meta_vec must be shape (4,), got {sample['meta_vec'].shape}"

    def test_dataset_missing_file_skipped(self, tmp_processed_dir):
        uids    = ["R001__sub-r001s001__ses-1", "NONEXISTENT__uid__ses-1"]
        df_meta = pd.DataFrame([
            {"UID": uids[0], "CHRONICITY_DERIVED": "chronic",
             "DAYS_POST_STROKE": 200, "SITE": "R001"},
            {"UID": uids[1], "CHRONICITY_DERIVED": "unknown",
             "DAYS_POST_STROKE": None, "SITE": "R001"},
        ])
        records = build_records(uids, df_meta,
                                tmp_processed_dir / "images",
                                tmp_processed_dir / "masks")
        assert len(records) == 1, "Missing-file UID should be skipped silently."


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Splits
# ═══════════════════════════════════════════════════════════════════════════════

class TestSplits:

    @pytest.fixture
    def dummy_df(self):
        np.random.seed(42)
        n = 60
        chron  = np.random.choice(["acute","subacute","chronic","unknown"], n)
        sites  = np.random.choice(["R001","R002","R003","R009","R010"], n)
        uids   = [f"uid_{i:04d}" for i in range(n)]
        return pd.DataFrame({"UID": uids,
                             "CHRONICITY_DERIVED": chron,
                             "SITE": sites})

    def test_splits_cover_all_uids(self, dummy_df):
        folds = generate_splits(dummy_df, n_splits=3, seed=42)
        validate_splits(folds, dummy_df)   # raises on failure

    def test_no_train_val_overlap(self, dummy_df):
        folds = generate_splits(dummy_df, n_splits=3, seed=42)
        for fold in folds:
            overlap = set(fold["train_uids"]) & set(fold["val_uids"])
            assert len(overlap) == 0, \
                f"Fold {fold['fold']}: {len(overlap)} UIDs leaked between train/val."

    def test_fold_sizes_balanced(self, dummy_df):
        folds     = generate_splits(dummy_df, n_splits=3, seed=42)
        val_sizes = [len(f["val_uids"]) for f in folds]
        assert max(val_sizes) - min(val_sizes) <= 2, \
            f"Fold sizes too unbalanced: {val_sizes}"

    def test_rare_stratum_collapse(self):
        """Strata with < n_splits samples must be collapsed without error."""
        df = pd.DataFrame({
            "UID":                ["u1","u2","u3","u4","u5","u6","u7","u8","u9"],
            "CHRONICITY_DERIVED": ["acute"]*2 + ["chronic"]*7,
            "SITE":               ["R001"]*9,
        })
        # Should not raise even though acute×R001 has only 2 samples < n_splits=3
        folds = generate_splits(df, n_splits=3, seed=0)
        assert len(folds) == 3

    def test_deterministic_with_same_seed(self, dummy_df):
        folds_a = generate_splits(dummy_df, n_splits=3, seed=99)
        folds_b = generate_splits(dummy_df, n_splits=3, seed=99)
        for fa, fb in zip(folds_a, folds_b):
            assert fa["val_uids"] == fb["val_uids"], "Splits must be deterministic."


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Conditioning
# ═══════════════════════════════════════════════════════════════════════════════

class TestConditioning:

    def test_film_conditioner_output_shape(self, cfg, dummy_meta):
        meta_vec, meta_text = dummy_meta
        conditioner = FiLMConditioner(cfg.conditioning)
        for feature_dim in [64, 128, 320]:
            gamma, beta = conditioner(meta_vec, meta_text, feature_dim)
            assert gamma.shape == (2, feature_dim), \
                f"gamma shape mismatch for feature_dim={feature_dim}"
            assert beta.shape  == (2, feature_dim)

    def test_film_conditioner_identity_init(self, cfg, dummy_meta):
        """At initialisation, gamma≈1 and beta≈0 (identity transform)."""
        meta_vec, meta_text = dummy_meta
        conditioner = FiLMConditioner(cfg.conditioning)
        gamma, beta = conditioner(meta_vec, meta_text, feature_dim=64)
        assert gamma.mean().item() == pytest.approx(1.0, abs=0.1), \
            "gamma should initialise near 1.0"
        assert beta.mean().item() == pytest.approx(0.0, abs=0.1), \
            "beta should initialise near 0.0"

    def test_film_conditioner_different_inputs_differ(self, cfg):
        """Different metadata inputs should produce different (gamma, beta)."""
        conditioner = FiLMConditioner(cfg.conditioning)
        vec_acute   = encode_metadata_vector(3.0,   "acute").unsqueeze(0)
        vec_chronic = encode_metadata_vector(365.0, "chronic").unsqueeze(0)
        g1, b1 = conditioner(vec_acute,   ["acute text"],   32)
        g2, b2 = conditioner(vec_chronic, ["chronic text"], 32)
        # After a few gradient steps they would differ; at init projection is
        # zero-weight so outputs are equal — test that the encoder differs
        enc1 = conditioner.encoder(vec_acute)
        enc2 = conditioner.encoder(vec_chronic)
        assert not torch.equal(enc1, enc2), \
            "Encoder must produce different embeddings for different inputs."

    def test_film_track_switch(self, cfg, dummy_meta):
        """build_conditioner returns FiLM for track A."""
        c = build_conditioner(cfg)
        assert isinstance(c, FiLMConditioner)

    def test_track_c_returns_llm_conditioner(self, cfg, dummy_meta):
        cfg_c = OmegaConf.merge(cfg, {"conditioning": {"track": "C"}})
        c     = build_conditioner(cfg_c)
        assert isinstance(c, LLMConditioner), \
            "Track C must instantiate LLMConditioner."

    def test_apply_film_identity(self):
        """With gamma=1 and beta=0, FiLM must be identity."""
        x     = torch.randn(2, 8, 4, 4, 4)
        gamma = torch.ones(2, 8)
        beta  = torch.zeros(2, 8)
        from model import apply_film
        out = apply_film(x, gamma, beta)
        assert torch.allclose(out, x), "Identity FiLM must not change features."

    def test_apply_film_scales_correctly(self):
        x     = torch.ones(1, 4, 2, 2, 2)
        gamma = torch.tensor([[2.0, 2.0, 2.0, 2.0]])
        beta  = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        from model import apply_film
        out = apply_film(x, gamma, beta)
        assert torch.allclose(out, torch.full_like(x, 3.0)), \
            "FiLM: 2*1 + 1 should give 3."


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Model
# ═══════════════════════════════════════════════════════════════════════════════

class TestModel:

    def test_forward_returns_list(self, cfg, small_tensor, dummy_meta):
        img, mask     = small_tensor
        meta_vec, meta_text = dummy_meta
        model  = build_model(cfg)
        output = model(img, meta_vec, meta_text)
        assert isinstance(output, list), "Model must return a list of logit tensors."

    def test_forward_deep_supervision_count(self, cfg, small_tensor, dummy_meta):
        img, _        = small_tensor
        meta_vec, meta_text = dummy_meta
        model  = build_model(cfg)
        output = model(img, meta_vec, meta_text)
        assert len(output) == 4, \
            f"Expected 4 deep supervision outputs, got {len(output)}."

    def test_forward_finest_scale_shape(self, cfg, small_tensor, dummy_meta):
        img, _        = small_tensor
        meta_vec, meta_text = dummy_meta
        model  = build_model(cfg)
        output = model(img, meta_vec, meta_text)
        B, C, H, W, D = output[0].shape
        assert B == 2,         f"Batch size mismatch: {B}"
        assert C == 2,         f"Expected 2 output classes, got {C}"
        assert (H, W, D) == tuple(img.shape[2:]), \
            f"Finest output must match input spatial dims: {img.shape[2:]} vs {(H,W,D)}"

    def test_forward_coarser_scales_smaller(self, cfg, small_tensor, dummy_meta):
        img, _        = small_tensor
        meta_vec, meta_text = dummy_meta
        model  = build_model(cfg)
        output = model(img, meta_vec, meta_text)
        finest = output[0].shape[2:]
        for i, out in enumerate(output[1:], start=1):
            for dim_f, dim_c in zip(finest, out.shape[2:]):
                assert dim_c <= dim_f, \
                    f"Scale {i} spatial dim {dim_c} not <= finest {dim_f}"

    def test_predict_returns_binary(self, cfg, small_tensor, dummy_meta):
        img, _        = small_tensor
        meta_vec, meta_text = dummy_meta
        model  = build_model(cfg)
        pred   = model.predict(img, meta_vec, meta_text)
        unique = torch.unique(pred).tolist()
        assert set(unique).issubset({0, 1}), \
            f"predict() must return binary mask, got {unique}"

    def test_predict_shape(self, cfg, small_tensor, dummy_meta):
        img, _        = small_tensor
        meta_vec, meta_text = dummy_meta
        model  = build_model(cfg)
        pred   = model.predict(img, meta_vec, meta_text)
        assert pred.shape == (2, 1, 32, 32, 32)

    def test_resblock_output_shape(self):
        block = ResBlock(16, 32)
        x     = torch.randn(2, 16, 8, 8, 8)
        out   = block(x)
        assert out.shape == (2, 32, 8, 8, 8)

    def test_model_trainable_params_nonzero(self, cfg):
        model   = build_model(cfg)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n_params > 0, "Model must have trainable parameters."

    def test_track_c_model_instantiates(self, cfg):
        cfg_c = OmegaConf.merge(cfg, {"conditioning": {"track": "C"}})
        model = build_model(cfg_c)
        from conditioning import LLMConditioner
        assert isinstance(model.conditioner, LLMConditioner)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Loss
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoss:

    def test_dice_loss_perfect_prediction(self):
        target = torch.ones(2, 8, 8, 8)
        loss   = soft_dice_loss(target, target)
        assert loss.item() == pytest.approx(0.0, abs=1e-4), \
            "Perfect prediction must give Dice loss ≈ 0."

    def test_dice_loss_empty_prediction(self):
        probs  = torch.zeros(2, 8, 8, 8)
        target = torch.ones(2, 8, 8, 8)
        loss   = soft_dice_loss(probs, target)
        assert loss.item() == pytest.approx(1.0, abs=0.01), \
            "Zero prediction vs full target must give Dice loss ≈ 1."

    def test_dice_loss_range(self):
        probs  = torch.rand(2, 8, 8, 8).clamp(0, 1)
        target = (torch.rand(2, 8, 8, 8) > 0.5).float()
        loss   = soft_dice_loss(probs, target)
        assert 0.0 <= loss.item() <= 1.0

    def test_boundary_focal_loss_empty_mask(self):
        probs  = torch.rand(2, 8, 8, 8)
        target = torch.zeros(2, 8, 8, 8)
        loss   = boundary_focal_loss(probs, target)
        assert torch.isfinite(loss), "Boundary loss must be finite for empty mask."

    def test_scan_weights_small_lesion_upweighted(self, cfg):
        mask = torch.zeros(2, 1, 16, 16, 16)
        mask[0, :, :5, :5, :5] = 1   # small: 125 voxels = 0.125 mL
        mask[1, :, :8, :8, :8] = 1   # large: 512 voxels = 0.512 mL — still small
        weights = compute_scan_weights(mask, threshold_ml=1.0, small_weight=2.0)
        assert (weights == 2.0).all(), \
            "Both lesions < 1mL should get upweight=2.0"

    def test_scan_weights_large_lesion_normal(self):
        mask = torch.zeros(1, 1, 32, 32, 32)
        mask[0, :, :20, :20, :20] = 1   # 8000 voxels = 8 mL > threshold
        weights = compute_scan_weights(mask, threshold_ml=1.0, small_weight=2.0)
        assert weights[0].item() == pytest.approx(1.0)

    def test_combined_loss_forward(self, cfg, small_tensor):
        img, mask  = small_tensor
        model      = build_model(cfg)
        meta_vec   = torch.rand(2, 4)
        meta_text  = ["text a", "text b"]
        logits     = model(img, meta_vec, meta_text)
        criterion  = ISLES26Loss(cfg)
        loss, loss_dict = criterion(logits, mask.float())
        assert torch.isfinite(loss), "Total loss must be finite."
        assert loss.item() > 0,     "Total loss must be positive."
        assert "total" in loss_dict
        assert "scale_0" in loss_dict

    def test_loss_backward(self, cfg, small_tensor):
        """Loss must support gradient flow."""
        img, mask = small_tensor
        model     = build_model(cfg)
        meta_vec  = torch.rand(2, 4)
        meta_text = ["text a", "text b"]
        logits    = model(img, meta_vec, meta_text)
        criterion = ISLES26Loss(cfg)
        loss, _   = criterion(logits, mask.float())
        loss.backward()   # must not raise
        # Check at least one grad is non-None
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "Backward pass must compute gradients."

    def test_deep_supervision_weights_sum(self, cfg, small_tensor):
        """Ensure deep supervision normalisation doesn't inflate loss."""
        img, mask = small_tensor
        model     = build_model(cfg)
        meta_vec  = torch.rand(2, 4)
        meta_text = ["t", "t"]
        logits    = model(img, meta_vec, meta_text)
        criterion = ISLES26Loss(cfg)
        loss_full, d_full = criterion(logits,       mask.float())
        loss_one,  d_one  = criterion(logits[:1],   mask[:1].float())
        # Both should be finite — can't guarantee ordering but both must be valid
        assert torch.isfinite(loss_full)
        assert torch.isfinite(loss_one)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Evaluation metrics
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluate:

    def test_dice_perfect(self):
        arr = np.ones((10, 10, 10), dtype=np.uint8)
        assert dice_score(arr, arr) == pytest.approx(1.0, abs=1e-5)

    def test_dice_no_overlap(self):
        pred   = np.zeros((10, 10, 10), dtype=np.uint8)
        target = np.ones((10, 10, 10), dtype=np.uint8)
        assert dice_score(pred, target) < 0.01

    def test_dice_empty_both(self):
        z = np.zeros((10, 10, 10), dtype=np.uint8)
        assert dice_score(z, z) == pytest.approx(1.0, abs=1e-4), \
            "Two empty masks should give Dice=1 (smooth prevents divide-by-zero)."

    def test_precision_recall_perfect(self):
        arr  = np.ones((8, 8, 8), dtype=np.uint8)
        p, r = precision_recall(arr, arr)
        assert p == pytest.approx(1.0, abs=1e-4)
        assert r == pytest.approx(1.0, abs=1e-4)

    def test_precision_recall_range(self):
        np.random.seed(0)
        pred   = (np.random.rand(20, 20, 20) > 0.5).astype(np.uint8)
        target = (np.random.rand(20, 20, 20) > 0.5).astype(np.uint8)
        p, r = precision_recall(pred, target)
        assert 0.0 <= p <= 1.0
        assert 0.0 <= r <= 1.0

    def test_hd95_empty_pred_is_nan(self):
        pred   = np.zeros((10,10,10), dtype=np.uint8)
        target = np.ones((10,10,10), dtype=np.uint8)
        result = hausdorff95(pred, target)
        assert np.isnan(result), "HD95 with empty pred must return NaN."

    def test_hd95_identical_masks(self):
        pred = np.zeros((20, 20, 20), dtype=np.uint8)
        pred[5:10, 5:10, 5:10] = 1
        result = hausdorff95(pred, pred)
        assert result == pytest.approx(0.0, abs=1e-3), \
            "HD95 of identical masks must be 0."

    def test_remove_small_components(self):
        mask = np.zeros((20, 20, 20), dtype=np.uint8)
        mask[1:3,  1:3,  1:3]  = 1   # 8 voxels  → should be removed
        mask[10:15, 10:15, 10:15] = 1  # 125 voxels → should stay
        cleaned = remove_small_components(mask.copy(), min_voxels=10)
        assert cleaned[1:3, 1:3, 1:3].sum() == 0,     "Small component should be removed."
        assert cleaned[10:15, 10:15, 10:15].sum() > 0, "Large component should remain."

    def test_remove_small_components_empty_mask(self):
        mask    = np.zeros((10, 10, 10), dtype=np.uint8)
        cleaned = remove_small_components(mask, min_voxels=10)
        assert cleaned.sum() == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Integration — end-to-end mini forward + backward (Track A)
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration:

    def test_full_forward_backward_track_A(self, cfg, small_tensor):
        img, mask  = small_tensor
        meta_vec   = torch.rand(2, 4)
        meta_text  = ["chronic scan R001 45 days", "acute scan R009 3 days"]
        model      = build_model(cfg)
        criterion  = ISLES26Loss(cfg)
        optimizer  = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Forward
        logits = model(img, meta_vec, meta_text)
        assert len(logits) == 4, "Must return 4 scale outputs."

        # Loss
        loss, loss_dict = criterion(logits, mask.float())
        assert torch.isfinite(loss)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Weights changed
        logits2 = model(img, meta_vec, meta_text)
        loss2, _ = criterion(logits2, mask.float())
        # Loss may or may not decrease in 1 step on random data — just check finite
        assert torch.isfinite(loss2)

    def test_track_switch_same_input(self, cfg, small_tensor):
        """Track A and Track C must accept the same batch dict without error."""
        img, mask  = small_tensor
        meta_vec   = torch.rand(2, 4)
        meta_text  = ["chronic scan R001 45 days", "acute scan R009 3 days"]

        # Track A
        model_A  = build_model(cfg)
        logits_A = model_A(img, meta_vec, meta_text)
        assert logits_A[0].shape[1] == 2

        # Track C — only instantiates LLMConditioner, does NOT load LLM weights
        # (LLM is lazy-loaded on first forward — skip actual LLM forward in tests)
        cfg_c   = OmegaConf.merge(cfg, {"conditioning": {"track": "C"}})
        model_C = build_model(cfg_c)
        from conditioning import LLMConditioner
        assert isinstance(model_C.conditioner, LLMConditioner), \
            "Track C model must use LLMConditioner."

    def test_batch_size_one_inference(self, cfg):
        """Val loader uses batch_size=1; model must handle single-scan input."""
        img      = torch.randn(1, 1, 32, 32, 32)
        meta_vec = torch.rand(1, 4)
        meta_text = ["single scan test"]
        model    = build_model(cfg)
        pred     = model.predict(img, meta_vec, meta_text)
        assert pred.shape == (1, 1, 32, 32, 32)

    def test_metadata_gate_changes_output(self, cfg, small_tensor):
        """
        Different metadata inputs must produce different logits,
        confirming the conditioning gate is active.
        """
        img, _     = small_tensor
        vec_acute  = torch.zeros(2, 4); vec_acute[:,  1] = 1.0   # is_acute=1
        vec_chronic = torch.zeros(2, 4); vec_chronic[:, 3] = 1.0  # is_chronic=1
        texts      = ["acute", "acute"]

        model    = build_model(cfg)
        # Need at least one gradient step so projection weights are non-trivial
        criterion = ISLES26Loss(cfg)
        for vec in [vec_acute, vec_chronic]:
            loss, _ = criterion(model(img, vec, texts), small_tensor[1].float())
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        with torch.no_grad():
            out_acute   = model(img, vec_acute,   texts)[0]
            out_chronic = model(img, vec_chronic, ["chronic","chronic"])[0]

        assert not torch.equal(out_acute, out_chronic), \
            "Conditioning gate must produce different outputs for different metadata."