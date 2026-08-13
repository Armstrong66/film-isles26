#!/usr/bin/env python3
"""
test_docker_locally.py
----------------------
Local testing script for Docker submission before Kaggle upload.

This script simulates the Docker environment on a local RTX workstation
to verify the submission pipeline works before final submission.

Usage:
    python test_docker_locally.py \
        --input /path/to/test_scan.nii.gz \
        --output /path/to/output_mask.nii.gz \
        [--checkpoint /path/to/checkpoint.pth] \
        [--no-tta] \
        [--verbose]

Prerequisites:
    - Docker installed and running
    - PyTorch and dependencies installed
    - Test NIfTI scan available
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from torch.amp import autocast

log = __import__("logging"). getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def load_model(checkpoint_path: str, device: torch.device):
    """Load the trained model from checkpoint."""
    from pipeline.model import build_model
    from omegaconf import OmegaConf

    config_path = Path(__file__).parent / "configs" / "config.yaml"
    cfg = OmegaConf.load(str(config_path))

    model = build_model(cfg)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    return model, cfg


def preprocess_scan(img_path: str) -> tuple[torch.Tensor, dict]:
    """Preprocess a single scan for inference."""
    from pipeline.preprocessing import reorient_to_ras, clip_and_normalise

    # Load
    img_nib = nib.load(img_path)
    original_affine = img_nib.affine.copy()

    # Reorient to RAS
    img_ras, _ = reorient_to_ras(img_nib)

    # Get data and normalise
    img_data = img_ras.get_fdata(dtype=np.float32)
    fg_mask = img_data > 0

    assert fg_mask.sum() > 0, "No foreground voxels found"

    norm_data, _, _, fg_mean, fg_std = clip_and_normalise(img_data)

    # Create metadata vector (Track A format)
    meta_vec = torch.tensor([0.5, 0.0, 1.0, 0.0, 0.0])  # defaults
    meta_text = ["Acute phase stroke patient."]

    # Convert to tensor (B, C, H, W, D)
    tensor = torch.from_numpy(norm_data).unsqueeze(0).unsqueeze(0)
    tensor = tensor.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    metadata = {
        "original_affine": original_affine,
        "shape": img_nib.shape,
        "spacing": img_nib.header.get_zooms()[:3],
    }

    return tensor, {"meta_vec": meta_vec.unsqueeze(0).to(tensor.device), "meta_text": meta_text, **metadata}


def predict(model, x: torch.Tensor, meta_vec: torch.Tensor, meta_text: list, use_tta: bool = True) -> np.ndarray:
    """Run prediction with optional TTA."""
    device = x.device

    with torch.no_grad():
        with autocast(device_type=device.type, enabled=True):
            logits_list = model(x, meta_vec, meta_text)
            logits = logits_list[0]  # finest scale

    probs = torch.softmax(logits, dim=1)

    # Test-time augmentation (flips)
    if use_tta:
        flip_axes = [[], [0], [1], [2]]
        B, C, H, W, D = x.shape
        prob_sum = torch.zeros(B, 2, H, W, D, device=device)

        for axes in flip_axes:
            x_aug = torch.flip(x, dims=axes) if axes else x
            with torch.no_grad():
                with autocast(device_type=device.type, enabled=True):
                    logits_aug = model(x_aug, meta_vec, meta_text)[0]
            probs_aug = torch.softmax(logits_aug, dim=1)
            prob_sum += torch.flip(probs_aug, dims=axes) if axes else probs_aug

        avg_probs = prob_sum / len(flip_axes)
    else:
        avg_probs = probs

    # Threshold
    binary = (avg_probs[:, 1:2] > 0.5).squeeze(1).cpu().numpy().astype(np.uint8)

    # Remove small components
    binary = remove_small_components(binary[0], min_size=10)

    return binary


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove connected components smaller than min_size."""
    from scipy import ndimage

    labeled, num = ndimage.label(mask)
    if num == 0:
        return mask

    sizes = np.bincount(labeled.ravel())
    output = np.zeros_like(mask)
    for i in range(1, num + 1):
        if sizes[i] >= min_size:
            output[labeled == i] = 1

    return output


def save_output(mask: np.ndarray, metadata: dict, output_path: str):
    """Save output mask with original geometry."""
    output_nib = nib.Nifti1Image(mask.astype(np.uint8), metadata["original_affine"])
    nib.save(output_nib, output_path)
    return output_nib


def verify_output(input_path: str, output_path: str):
    """Verify output matches input geometry."""
    inp = nib.load(input_path)
    out = nib.load(output_path)

    assert inp.shape == out.shape, f"Shape mismatch: {inp.shape} vs {out.shape}"
    assert np.allclose(inp.affine, out.affine), "Affine mismatch"

    data = out.get_fdata()
    unique = np.unique(data)
    assert set(unique).issubset({0.0, 1.0}), f"Non-binary values: {unique}"

    return True


def main():
    parser = argparse.ArgumentParser(description="Test Docker submission locally")
    parser.add_argument("--input", "-i", required=True, help="Input NIfTI path")
    parser.add_argument("--output", "-o", required=True, help="Output mask path")
    parser.add_argument("--checkpoint", "-c", help="Checkpoint path (default: look in checkpoints/)")
    parser.add_argument("--no-tta", action="store_true", help="Disable test-time augmentation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    # Setup
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_dir = Path(__file__).parent / "checkpoints"
        checkpoints = list(checkpoint_dir.glob("fold_*_best.pth"))
        if checkpoints:
            checkpoint_path = str(checkpoints[0])
            logger.info(f"Using checkpoint: {checkpoint_path}")
        else:
            logger.error("No checkpoint found. Train a model first with pipeline/train.py")
            sys.exit(1)

    # Load model
    logger.info("Loading model...")
    model, cfg = load_model(checkpoint_path, device)
    logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Preprocess
    logger.info(f"Loading input: {args.input}")
    x, metadata = preprocess_scan(args.input)
    logger.info(f"Input shape: {x.shape}")

    # Predict
    logger.info("Running inference...")
    start = time.time()
    mask = predict(model, x, metadata["meta_vec"], metadata["meta_text"], use_tta=not args.no_tta)
    elapsed = time.time() - start
    logger.info(f"Inference time: {elapsed:.2f}s")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    logger.info(f"Saving output: {args.output}")
    output_nib = save_output(mask, metadata, args.output)

    # Verify
    logger.info("Verifying output...")
    verify_output(args.input, args.output)

    # Summary
    volume_ml = mask.sum() * np.prod(metadata["spacing"]) / 1000
    logger.info("=" * 60)
    logger.info("SUCCESS!")
    logger.info(f"  Input:  {args.input}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  Mask: {mask.sum()} voxels ({volume_ml:.2f} mL)")
    logger.info(f"  Time: {elapsed:.2f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
