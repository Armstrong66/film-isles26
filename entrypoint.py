"""
entrypoint.py
-------------
Docker entrypoint for ISLES26 submission.

Inference pipeline (per organizer template):
  1. Read input path from environment variable or fixed path
  2. Load T1w NIfTI → reorient to RAS → clip/normalise (per-scan z-score)
  3. Load all 5 fold checkpoints from /opt/algorithm/checkpoints/
  4. Run forward pass on each fold model → average softmax probabilities
  5. Apply 0.5 threshold → remove components < 10 voxels
  6. Reorient output mask back to original input orientation
  7. Save binary mask NIfTI at expected output path

Track A only for submission (Track C adds ~2-3 min overhead).

Benchmark target: < 7 min total including model loading.
"""

from __future__ import annotations

import os
import sys
import time
import logging
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Import our pipeline modules (copied into container) ────────────────────────
from pipeline.preprocessing import reorient_to_ras, clip_and_normalise


# ── Time tracking helper ────────────────────────────────────────────────────────
class Timer:
    """Simple timing helper for profiling the 10-min budget."""

    def __init__(self, name: str):
        self.name = name
        self.start = time.time()
        log.info(f"[TIMING] {self.name} started")

    def elapsed(self) -> float:
        return time.time() - self.start

    def checkpoint(self, msg: str = "") -> float:
        e = self.elapsed()
        log.info(f"[TIMING] {self.name} {msg}elapsed: {e:.2f}s")
        return e


# ── Load model helper ──────────────────────────────────────────────────────────

def load_fold_model(cfg, fold: int, device: torch.device) -> torch.nn.Module:
    """Load a single fold model from checkpoint."""
    from pipeline.model import build_model

    checkpoint_path = Path("/opt/algorithm/checkpoints") / f"fold_{fold}_best.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    log.info(f"Loading fold {fold} checkpoint from {checkpoint_path}")

    model = build_model(cfg)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    return model


def load_all_folds(cfg, device: torch.device) -> list[torch.nn.Module]:
    """Load all 5 fold models at startup (not per-scan)."""
    timer = Timer("load_all_folds")
    models = []
    for fold in range(5):
        models.append(load_fold_model(cfg, fold, device))
    timer.checkpoint("(all 5 folds loaded) ")
    return models


# ── Preprocessing for inference ───────────────────────────────────────────────

def preprocess_inference(img_path: str) -> tuple[torch.Tensor, nib.Nifti1Image, dict]:
    """
    Preprocess single scan for inference.
    Returns (normalized_tensor, original_nib, metadata_dict)
    """
    timer = Timer("preprocess")

    # Load
    img_nib = nib.load(img_path)
    original_affine = img_nib.affine.copy()
    original_ornt = "".join(nib.aff2axcodes(img_nib.affine))

    # Reorient to RAS
    img_ras, was_reoriented = reorient_to_ras(img_nib)
    timer.checkpoint("reorient")

    # Get data
    img_data = img_ras.get_fdata(dtype=np.float32)

    # Clip and normalise
    fg_mask = img_data > 0
    assert fg_mask.sum() > 0, "No foreground voxels found"

    norm_data, clip_low, clip_high, fg_mean, fg_std = clip_and_normalise(img_data)
    timer.checkpoint("normalise")

    # Create metadata vector (Track A format: 5-dim)
    # Default values for inference (will be overridden by user config)
    days_norm = 0.5  # mid-range assumed
    is_acute = 0.0
    is_subacute = 1.0  # assume subacute if unknown
    is_chronic = 0.0
    confirmed_chronic = 0.0

    meta_vec = torch.tensor([days_norm, is_acute, is_subacute, is_chronic, confirmed_chronic])
    meta_text = ["Acute phase stroke patient with recent onset symptoms."]

    timer.checkpoint("meta_vec")

    metadata = {
        "original_affine": original_affine,
        "original_ornt": original_ornt,
        "was_reoriented": was_reoriented,
        "shape": img_nib.shape,
        "spacing": img_nib.header.get_zooms()[:3],
    }

    # Convert to tensor (B, C, H, W, D)
    tensor = torch.from_numpy(norm_data).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, D)
    tensor = tensor.to(device=torch.device("cuda") if torch.cuda.is_available() else "cpu")

    return tensor, img_nib, {
        "meta_vec": meta_vec.unsqueeze(0).to(tensor.device),
        "meta_text": meta_text,
        **metadata
    }


# ── Inference with TTA ─────────────────────────────────────────────────────────

def predict_with_tta(
    models: list[torch.nn.Module],
    x: torch.Tensor,
    meta_vec: torch.Tensor,
    meta_text: list[str],
    flip_axes: list[list[int]] = None
) -> torch.Tensor:
    """
    Run inference with test-time augmentation (flips).
    Average softmax probabilities across augmented versions.
    """
    if flip_axes is None:
        flip_axes = [[], [0], [1], [2]]  # no flip + single axis flips

    B, C, H, W, D = x.shape
    device = x.device

    # Accumulate probabilities
    prob_sum = torch.zeros(B, 2, H, W, D, device=device)

    for axes in flip_axes:
        # Apply flip
        x_aug = x
        meta_vec_aug = meta_vec

        if axes:
            x_aug = torch.flip(x_aug, dims=axes)
            # Flip metadata dimensions that encode spatial information
            # days_norm is [0], is_chronic is [3], confirmed_chronic is [4]
            # For simplicity, we don't flip metadata (clinical info is orientation-agnostic)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                logits_list = model(x_aug, meta_vec_aug, meta_text)
                logits = logits_list[0]  # finest scale only for inference

        # Average probabilities
        probs = torch.softmax(logits, dim=1)

        # Reverse flip if applied
        if axes:
            probs = torch.flip(probs, dims=axes)

        prob_sum += probs

    # Average
    avg_probs = prob_sum / len(flip_axes)

    return avg_probs


def predict_single_scan(
    models: list[torch.nn.Module],
    x: torch.Tensor,
    meta_vec: torch.Tensor,
    meta_text: list[str],
) -> np.ndarray:
    """
    Run ensemble prediction across all 5 folds.
    Returns binary mask (H, W, D) uint8.
    """
    timer = Timer("ensemble_inference")

    B, C, H, W, D = x.shape
    device = x.device

    # Accumulate probabilities across folds
    prob_sum = torch.zeros(B, 2, H, W, D, device=device)

    for fold, model in enumerate(models):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                logits_list = model(x, meta_vec, meta_text)
                logits = logits_list[0]  # finest scale

        probs = torch.softmax(logits, dim=1)
        prob_sum += probs
        timer.checkpoint(f"fold {fold}")

    # Average
    avg_probs = prob_sum / len(models)

    # Threshold to get binary mask
    binary = (avg_probs[:, 1:2] > 0.5).squeeze(1).cpu().numpy().astype(np.uint8)

    # Remove small components (< 10 voxels)
    binary = remove_small_components(binary, min_size=10)

    timer.checkpoint("finalization")

    return binary[0]  # (H, W, D)


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove connected components smaller than min_size voxels."""
    from scipy import ndimage

    # Label connected components
    labeled, num = ndimage.label(mask)

    if num == 0:
        return mask

    # Count voxels per component
    sizes = np.bincount(labeled.ravel())

    # Create mask of large components
    output = np.zeros_like(mask)
    for i in range(1, num + 1):
        if sizes[i] >= min_size:
            output[labeled == i] = 1

    return output


# ── Reorient back to original ──────────────────────────────────────────────────

def reorient_back_to_original(
    mask: np.ndarray,
    original_nib: nib.Nifti1Image,
    metadata: dict
) -> nib.Nifti1Image:
    """Reorient mask back to original input orientation."""
    from pipeline.preprocessing import reorient_mask_to_ras

    current_ornt = nib.aff2axcodes(original_nib.affine)

    # If original was not RAS, we need to reorient
    if metadata.get("was_reoriented", False):
        # Create temporary image with current mask
        temp_nib = nib.Nifti1Image(mask, original_nib.affine, original_nib.header)

        # Reorient to match original
        current_ornt_arr = nib.orientations.axcodes2ornt(current_ornt)
        target_ornt_arr = nib.orientations.axcodes2ornt(("R", "A", "S"))

        # Transform back
        transform = nib.orientations.ornt_transform(target_ornt_arr, current_ornt_arr)
        reoriented = temp_nib.as_reoriented(transform)

        return reoriented

    return nib.Nifti1Image(mask, original_nib.affine, original_nib.header)


# ── Main entrypoint ────────────────────────────────────────────────────────────

def main():
    """Docker entrypoint - runs inference on input and writes output."""
    start = time.time()
    global_timer = Timer("docker_inference")

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("=" * 60)
    log.info("ISLES26 Docker Inference")
    log.info("=" * 60)

    # Determine paths (per organizer template)
    input_path = os.environ.get("ISLES26_INPUT_PATH", "/input/image.nii.gz")
    output_path = os.environ.get("ISLES26_OUTPUT_PATH", "/output/mask.nii.gz")

    log.info(f"Input:  {input_path}")
    log.info(f"Output: {output_path}")

    # Check input exists
    if not os.path.exists(input_path):
        log.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load config
    from omegaconf import OmegaConf
    cfg_path = "/opt/algorithm/configs/config.yaml"
    if not os.path.exists(cfg_path):
        # Fallback to bundled config
        cfg_path = "/opt/algorithm/config.yaml"
    cfg = OmegaConf.load(cfg_path)

    log.info(f"Config loaded: {cfg_path}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Load all fold models (once at startup)
    models = load_all_folds(cfg, device)

    # Preprocess input
    x, original_nib, metadata = preprocess_inference(input_path)
    log.info(f"Input shape: {x.shape}")

    # Run prediction
    mask = predict_single_scan(
        models,
        x,
        metadata["meta_vec"],
        metadata["meta_text"],
    )
    log.info(f"Predicted mask shape: {mask.shape}, non-zero voxels: {mask.sum()}")

    # Reorient back if needed
    output_nib = reorient_back_to_original(mask, original_nib, metadata)

    # Save output
    nib.save(output_nib, output_path)
    log.info(f"Output saved: {output_path}")

    # Verify output geometry matches input
    assert output_nib.shape == original_nib.shape, "Output shape mismatch!"
    assert np.allclose(output_nib.affine, original_nib.affine), "Output affine mismatch!"

    # Check output dtype is uint8 with binary values
    output_data = output_nib.get_fdata()
    unique_vals = np.unique(output_data)
    assert set(unique_vals).issubset({0.0, 1.0}), f"Non-binary values found: {unique_vals}"

    total_time = global_timer.elapsed()
    log.info(f"Total inference time: {total_time:.2f}s")

    # Log summary
    log.info("=" * 60)
    log.info("Inference complete!")
    log.info(f"  Input:  {input_path}")
    log.info(f"  Output: {output_path}")
    log.info(f"  Mask size: {mask.sum()} voxels ({mask.sum() * np.prod(metadata['spacing']) / 1000:.2f} mL)")
    log.info(f"  Time: {total_time:.2f}s")
    log.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
