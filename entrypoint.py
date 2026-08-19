"""
entrypoint.py
-------------
Docker entrypoint for ISLES26 submission (FastAPI Grand Challenge API).

INTERPRETABILITY IMPORTS ARE EXPLICITLY EXCLUDED FROM THIS FILE.
See pipeline/interpretability.py for post-hoc analysis (local only).

This implements the Grand Challenge algorithm API:
- GET /health  — Health check endpoint
- POST /invoke — Run inference on input image

Input format (from Grand Challenge):
  - /input/images/t1-brain-mri/*.nii.gz
  - /input/stroke-metadata.json (optional)

Output format (to Grand Challenge):
  - /output/images/stroke-lesion-segmentation/output.mha
  - /output/images/lesion-probability-map/output.mha (optional)

Track A only for submission (Track C adds ~2-3 min overhead).
Benchmark target: < 7 min total including model loading.
"""

from __future__ import annotations

import os
import sys
import time
import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import nibabel as nib
import torch
from torch.amp import autocast
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Import our pipeline modules (copied into container)
from pipeline.preprocessing import reorient_to_ras, clip_and_normalise

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="ISLES26 Algorithm", description="Automated stroke lesion segmentation")

# Global model state (loaded once at startup)
MODELS: Optional[list[torch.nn.Module]] = None
CFG: Optional[dict] = None
DEVICE: torch.device = torch.device("cpu")


# ── Time tracking helper ───────────────────────────────────────────────────────
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


# ── Model loading helper ───────────────────────────────────────────────────────

def load_fold_model(cfg: dict, fold: int, device: torch.device) -> torch.nn.Module:
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


def load_all_folds(cfg: dict, device: torch.device) -> list[torch.nn.Module]:
    """Load all available fold models at startup (supports 1 to 5 folds or single checkpoint)."""
    timer = Timer("load_all_folds")
    models = []
    ckpt_dir = Path("/opt/algorithm/checkpoints")

    for fold in range(5):
        ckpt_path = ckpt_dir / f"fold_{fold}_best.pth"
        if ckpt_path.exists():
            models.append(load_fold_model(cfg, fold, device))

    if len(models) == 0:
        # Fallback: look for any .pth checkpoint in checkpoints/
        any_ckpts = list(ckpt_dir.glob("*.pth"))
        if any_ckpts:
            log.info(f"Loading checkpoint {any_ckpts[0]}")
            from pipeline.model import build_model
            model = build_model(cfg)
            model.load_state_dict(torch.load(any_ckpts[0], map_location=device, weights_only=True))
            model.to(device)
            model.eval()
            models.append(model)
        else:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    log.info(f"Loaded {len(models)} model checkpoints for ensemble inference.")
    timer.checkpoint(f"({len(models)} models loaded)")
    return models


# ── Preprocessing for inference ────────────────────────────────────────────────

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
    tensor = tensor.to(device=DEVICE)

    return tensor, img_nib, {
        "meta_vec": meta_vec.unsqueeze(0).to(tensor.device),
        "meta_text": meta_text,
        **metadata
    }


# ── Inference functions ────────────────────────────────────────────────────────

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

    # Accumulate probabilities across folds
    prob_sum = torch.zeros(B, 2, H, W, D, device=DEVICE)

    for fold, model in enumerate(models):
        with torch.no_grad():
            with autocast(device_type=DEVICE.type, enabled=True):
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


def reorient_back_to_original(
    mask: np.ndarray,
    original_nib: nib.Nifti1Image,
    original_ornt: str
) -> nib.Nifti1Image:
    """Reorient mask back to original input orientation."""
    current_ornt = nib.aff2axcodes(original_nib.affine)

    # If original was not RAS, we need to reorient back
    if original_ornt != "RAS":
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


def write_nifti_as_mha(nib_img: nib.Nifti1Image, output_path: Path) -> None:
    """Convert NIfTI to MHA format for Grand Challenge output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get array data
    array = nib_img.get_fdata()

    # Create SimpleITK image
    import SimpleITK as sitk
    sitk_img = sitk.GetImageFromArray(array)
    sitk_img.SetSpacing(nib_img.header.get_zooms()[:3])
    sitk_img.SetOrigin(nib_img.affine[:3, 3].tolist())
    sitk_img.SetDirection(nib_img.affine[:3, :3].flatten().tolist())

    # Save as MHA
    output_path = output_path.with_suffix(".mha")
    sitk.WriteImage(sitk_img, str(output_path), useCompression=True)
    log.info(f"Output saved: {output_path}")


# ── Input/Output helpers for Grand Challenge format ────────────────────────────

def find_input_file() -> Path:
    """Find input T1w image in Grand Challenge format."""
    input_dir = Path("/input/images/t1-brain-mri")
    if input_dir.exists():
        # Find nii.gz files
        nii_files = list(input_dir.glob("*.nii.gz")) + list(input_dir.glob("*.nii"))
        if nii_files:
            return nii_files[0]

    # Fallback: check /input directly
    fallback_dir = Path("/input")
    fallback_files = list(fallback_dir.glob("*.nii.gz")) + list(fallback_dir.glob("*.nii"))
    if fallback_files:
        return fallback_files[0]

    raise FileNotFoundError("No input image found in /input/images/t1-brain-mri/")


def get_output_path() -> Path:
    """Get output path for Grand Challenge format."""
    return Path("/output/images/stroke-lesion-segmentation/output.mha")


def get_metadata() -> Optional[dict]:
    """Load optional metadata from Grand Challenge."""
    metadata_path = Path("/input/stroke-metadata.json")
    if metadata_path.exists():
        import json
        with open(metadata_path) as f:
            return json.load(f)
    return None


# ── API endpoints ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint - returns 200 if models are loaded."""
    if MODELS is None:
        return JSONResponse(status_code=503, content={"status": "not_loaded"})
    return Response(status_code=200)


@app.post("/invoke")
async def invoke():
    """Run inference on input image and write output."""
    global MODELS, CFG, DEVICE

    timer = Timer("invoke")

    # Check models are loaded
    if MODELS is None:
        log.error("Models not loaded!")
        return JSONResponse(status_code=503, content={"status": "model_not_loaded"})

    log.info("=" * 60)
    log.info("ISLES26 Docker Inference /invoke")
    log.info("=" * 60)

    try:
        # Find input file
        input_path = find_input_file()
        log.info(f"Input: {input_path}")

        # Load config (supports both Docker and local dev paths)
        cfg_path = Path("/opt/algorithm/configs/config.yaml")
        if not cfg_path.exists():
            cfg_path = Path("/opt/algorithm/config.yaml")
        if not cfg_path.exists():
            # Local dev fallback
            cfg_path = Path(__file__).parent / "configs" / "config.yaml"

        if CFG is None:
            from omegaconf import OmegaConf
            CFG = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
            log.info(f"Config loaded: {cfg_path}")

        # Preprocess
        x, original_nib, metadata = preprocess_inference(str(input_path))
        log.info(f"Input shape: {x.shape}")

        # Run prediction
        mask = predict_single_scan(
            MODELS,
            x,
            metadata["meta_vec"],
            metadata["meta_text"],
        )
        log.info(f"Predicted mask shape: {mask.shape}, non-zero voxels: {mask.sum()}")

        # Reorient back if needed
        output_nib = reorient_back_to_original(mask, original_nib, metadata["original_ornt"])

        # Write output in MHA format
        output_path = get_output_path()
        write_nifti_as_mha(output_nib, output_path)

        # Verify output geometry matches input
        assert output_nib.shape == original_nib.shape, "Output shape mismatch!"
        assert np.allclose(output_nib.affine, original_nib.affine), "Output affine mismatch!"

        # Check output dtype is uint8 with binary values
        output_data = output_nib.get_fdata()
        unique_vals = np.unique(output_data)
        assert set(unique_vals).issubset({0.0, 1.0}), f"Non-binary values found: {unique_vals}"

        total_time = timer.elapsed()
        log.info(f"Total inference time: {total_time:.2f}s")

        # Log summary
        log.info("=" * 60)
        log.info("Inference complete!")
        log.info(f"  Input:  {input_path}")
        log.info(f"  Output: {output_path}")
        log.info(f"  Mask size: {mask.sum()} voxels ({mask.sum() * np.prod(metadata['spacing']) / 1000:.2f} mL)")
        log.info(f"  Time: {total_time:.2f}s")
        log.info("=" * 60)

        return Response(status_code=201, content={"status": "success"})

    except Exception as e:
        log.error(f"Inference failed: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


# ── Lifespan: load models once at startup ─────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Load models once at container startup."""
    global MODELS, CFG, DEVICE

    log.info("Loading models at startup...")

    # Set device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {DEVICE}")

    # Load config (supports both local dev and Docker paths)
    cfg_path = Path("/opt/algorithm/configs/config.yaml")
    if not cfg_path.exists():
        cfg_path = Path("/opt/algorithm/config.yaml")
    if not cfg_path.exists():
        # Local dev fallback
        cfg_path = Path(__file__).parent / "configs" / "config.yaml"

    from omegaconf import OmegaConf
    CFG = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    log.info(f"Config loaded: {cfg_path}")

    # Load all fold models
    MODELS = load_all_folds(CFG, DEVICE)

    log.info("Models loaded successfully!")


# ── CLI mode (for local testing) ───────────────────────────────────────────────

def main_cli():
    """CLI mode for local testing (not used by Docker)."""
    start = time.time()
    global_timer = Timer("docker_inference")

    log.info("=" * 60)
    log.info("ISLES26 Docker Inference (CLI mode)")
    log.info("=" * 60)

    # Find input
    input_path = find_input_file()
    output_path = get_output_path()

    log.info(f"Input:  {input_path}")
    log.info(f"Output: {output_path}")

    # Load config (supports both local dev and Docker paths)
    cfg_path = Path("/opt/algorithm/configs/config.yaml")
    if not cfg_path.exists():
        cfg_path = Path("/opt/algorithm/config.yaml")
    if not cfg_path.exists():
        # Local dev fallback
        cfg_path = Path(__file__).parent / "configs" / "config.yaml"

    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    log.info(f"Config loaded: {cfg_path}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Load all fold models
    models = load_all_folds(cfg, device)

    # Preprocess input
    x, original_nib, metadata = preprocess_inference(str(input_path))
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
    output_nib = reorient_back_to_original(mask, original_nib, metadata["original_ornt"])

    # Write output
    write_nifti_as_mha(output_nib, output_path)

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
    main_cli()
