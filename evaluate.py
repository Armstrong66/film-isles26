"""
evaluate.py
-----------
Evaluation pipeline for ISLES26 using all five official metrics:
  1. Dice Score                     (global binary DSC)
  2. Absolute Volume Difference     (mL)
  3. Absolute Lesion Count Diff     (instance count |GT - Pred|)
  4. Lesion-wise F1 Score           (recognition quality via panoptica)
  5. PR-AUC                         (requires soft probability map)

Metrics 1-4 use the binary mask; metric 5 uses the raw probability output.
Directly wraps the official eval_utils functions provided by the organizers.

Usage:
  python evaluate.py --config configs/config.yaml --fold 0
  python evaluate.py --config configs/config.yaml --fold all --tta
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import nibabel as nib
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

from dataset import build_dataloaders
from model import build_model, ISLES26Model
from train import load_checkpoint

# Official organizer metrics
from utils.eval_utils import (
    compute_pr_auc,
    compute_absolute_volume_difference,
    compute_dice_f1_instance_difference,
)

log = logging.getLogger(__name__)


# ── TTA ───────────────────────────────────────────────────────────────────────

def predict_with_tta(
    model:     ISLES26Model,
    image:     torch.Tensor,       # (1, 1, H, W, D)
    meta_vec:  torch.Tensor,
    meta_text: list[str],
    flip_axes: list[list[int]],
    device:    torch.device,
) -> torch.Tensor:
    """Returns mean-pooled soft probability map (1, 1, H, W, D) over TTA flips."""
    model.eval()
    probs_sum = None

    configs = [[]] + flip_axes   # original + each flip variant
    for axes in configs:
        img_in = torch.flip(image, dims=[d + 2 for d in axes]) if axes else image
        with torch.no_grad():
            logits = model(img_in, meta_vec, meta_text)[0]
            p = torch.softmax(logits, dim=1)[:, 1:2]
        if axes:
            p = torch.flip(p, dims=[d + 2 for d in axes])
        probs_sum = p if probs_sum is None else probs_sum + p

    return probs_sum / len(configs)


# ── Post-processing ───────────────────────────────────────────────────────────

def remove_small_components(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    if mask.sum() == 0:
        return mask
    try:
        from scipy.ndimage import label
        labelled, n = label(mask)
        for i in range(1, n + 1):
            if (labelled == i).sum() < min_voxels:
                mask[labelled == i] = 0
    except ImportError:
        warnings.warn("scipy unavailable — skipping component filtering.")
    return mask


# ── Per-scan evaluation ───────────────────────────────────────────────────────

def evaluate_scan(
    model:    ISLES26Model,
    batch:    dict,
    cfg:      DictConfig,
    device:   torch.device,
    use_tta:  bool = False,
) -> dict:
    """
    Run inference on one scan and compute all five official metrics.
    Returns a flat dict of results for this scan.
    """
    image      = batch["image"].to(device)           # (1,1,H,W,D)
    mask_t     = batch["mask"]                        # (1,1,H,W,D)
    meta_vec   = batch["meta_vec"].to(device)
    meta_text  = batch["meta_text"]
    uid        = batch["uid"][0]
    chronicity = batch["chronicity"][0]

    # Ground truth as numpy uint8 (H,W,D)
    gt = mask_t.squeeze().cpu().numpy().astype(np.uint8)

    # ── Inference ─────────────────────────────────────────────────────────────
    if use_tta and cfg.tta.enabled:
        prob_map = predict_with_tta(
            model, image, meta_vec, meta_text, cfg.tta.flips, device
        )
    else:
        model.eval()
        with torch.no_grad():
            logits   = model(image, meta_vec, meta_text)[0]
            prob_map = torch.softmax(logits, dim=1)[:, 1:2]

    # Soft map (H,W,D) float32 — needed for PR-AUC
    soft_map = prob_map.squeeze().cpu().numpy().astype(np.float32)

    # Binary mask (H,W,D) uint8 — needed for all other metrics
    binary_pred = (soft_map > 0.5).astype(np.uint8)
    binary_pred = remove_small_components(
        binary_pred, cfg.postprocessing.min_component_size_voxels
    )

    # Voxel volume in mL (confirmed isotropic 1mm³ from EDA; read from header if available)
    voxel_vol_ml = np.array(1.0 / 1000.0)

    # ── Official metrics ───────────────────────────────────────────────────────

    # 1-3. Dice, lesion F1, instance count diff (panoptica)
    f1, lesion_count_diff, dice = compute_dice_f1_instance_difference(
        ground_truth = gt,
        prediction   = binary_pred,
        empty_value  = 1.0,     # reward correct empty predictions
    )

    # 4. Absolute volume difference (mL)
    abs_vol_diff = compute_absolute_volume_difference(
        im1        = gt,
        im2        = binary_pred,
        voxel_size = voxel_vol_ml,
    )

    # 5. PR-AUC (soft map)
    pr_auc = compute_pr_auc(
        ground_truth    = gt,
        prediction_map  = soft_map,
        empty_value     = 1.0,
    )

    # Lesion size category for stratified reporting
    gt_vol_ml = float(gt.sum()) / 1000.0
    size_bins = cfg.evaluation.size_bins
    size_cat  = (
        "small"  if gt_vol_ml < size_bins.small[1]  else
        "medium" if gt_vol_ml < size_bins.medium[1] else
        "large"
    )

    return {
        "uid":               uid,
        "chronicity":        chronicity,
        "size_cat":          size_cat,
        "gt_vol_ml":         gt_vol_ml,
        "pred_vol_ml":       float(binary_pred.sum()) / 1000.0,
        # Official metrics
        "dice":              float(dice),
        "abs_vol_diff_ml":   float(abs_vol_diff),
        "abs_lesion_count_diff": int(lesion_count_diff),
        "lesion_f1":         float(f1),
        "pr_auc":            float(pr_auc),
    }


# ── Aggregate metrics ─────────────────────────────────────────────────────────

OFFICIAL_METRICS = ["dice", "abs_vol_diff_ml", "abs_lesion_count_diff",
                    "lesion_f1", "pr_auc"]


def aggregate_metrics(records: list[dict], cfg: DictConfig) -> dict:
    df = pd.DataFrame(records)

    def summarise(subset: pd.DataFrame, label: str) -> dict:
        out = {"n": len(subset)}
        for m in OFFICIAL_METRICS:
            vals = subset[m].replace([np.inf, -np.inf], np.nan).dropna()
            out[f"{m}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            out[f"{m}_std"]  = float(vals.std())  if len(vals) else float("nan")
        return {label: out}

    result = {}
    result.update(summarise(df, "overall"))

    for chron in ["acute", "subacute", "chronic", "unknown"]:
        sub = df[df["chronicity"] == chron]
        if len(sub):
            result.update(summarise(sub, f"chronicity_{chron}"))

    for cat in ["small", "medium", "large"]:
        sub = df[df["size_cat"] == cat]
        if len(sub):
            result.update(summarise(sub, f"size_{cat}"))

    return result


def print_summary(agg: dict) -> None:
    """Print a clean summary table of the official metrics."""
    overall = agg.get("overall", {})
    print("\n" + "=" * 55)
    print("  ISLES26 EVALUATION SUMMARY (official metrics)")
    print("=" * 55)
    labels = {
        "dice":                  "Dice",
        "abs_vol_diff_ml":       "Abs Vol Diff (mL)",
        "abs_lesion_count_diff": "Abs Lesion Count Diff",
        "lesion_f1":             "Lesion-wise F1",
        "pr_auc":                "PR-AUC",
    }
    for key, label in labels.items():
        mean = overall.get(f"{key}_mean", float("nan"))
        std  = overall.get(f"{key}_std",  float("nan"))
        print(f"  {label:<28s}: {mean:.4f} ± {std:.4f}")
    print(f"  {'N scans':<28s}: {overall.get('n', '?')}")
    print("=" * 55)


# ── Per-fold evaluation ───────────────────────────────────────────────────────

def evaluate_fold(
    cfg:     DictConfig,
    fold:    int,
    splits:  list[dict],
    df_meta: pd.DataFrame,
    device:  torch.device,
    use_tta: bool = False,
) -> dict:
    log.info(f"Evaluating fold {fold} | track={cfg.conditioning.track} | tta={use_tta}")

    _, val_dl = build_dataloaders(cfg, fold, splits, df_meta)
    model     = build_model(cfg).to(device)

    ckpt_dir  = Path(cfg.logging.log_dir) / f"track_{cfg.conditioning.track}" / f"fold_{fold}"
    best_ckpt = ckpt_dir / "best.pth"
    assert best_ckpt.exists(), \
        f"No checkpoint at {best_ckpt}. Train fold {fold} first."
    load_checkpoint(best_ckpt, model)

    scan_records = []
    for batch in tqdm(val_dl, desc=f"Fold {fold} eval"):
        try:
            rec = evaluate_scan(model, batch, cfg, device, use_tta)
            scan_records.append(rec)
            log.debug(
                f"  {rec['uid']} | dice={rec['dice']:.4f} "
                f"pr_auc={rec['pr_auc']:.4f} f1={rec['lesion_f1']:.4f}"
            )
        except Exception as e:
            log.error(f"Failed scan {batch['uid'][0]}: {e}")

    assert scan_records, f"No scans evaluated for fold {fold}."

    agg = aggregate_metrics(scan_records, cfg)
    print_summary(agg)

    # Save outputs
    pd.DataFrame(scan_records).to_csv(ckpt_dir / "eval_per_scan.csv",    index=False)
    with open(ckpt_dir / "eval_aggregate.json", "w") as f:
        json.dump(agg, f, indent=2)

    log.info(f"Results saved to {ckpt_dir}")
    return {"fold": fold, "aggregate": agg, "scan_records": scan_records}


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ISLES26 Evaluation")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--fold",   type=str, default="0")
    parser.add_argument("--track",  type=str, default=None)
    parser.add_argument("--tta",    action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.track:
        cfg = OmegaConf.merge(cfg, {"conditioning": {"track": args.track.upper()}})

    Path(cfg.logging.log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level   = getattr(logging, cfg.logging.level),
        format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt = "%H:%M:%S",
        handlers = [
            logging.StreamHandler(),
            logging.FileHandler(
                Path(cfg.logging.log_dir) /
                f"eval_track{cfg.conditioning.track}.log"
            ),
        ],
    )

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    splits  = json.loads(
        (Path(cfg.data.processed_dir) / "splits.json").read_text()
    )
    df_meta = pd.read_csv(
        Path(cfg.data.processed_dir) / "dataset_manifest.csv"
    )

    folds_to_run = (
        list(range(cfg.data.n_splits)) if args.fold == "all"
        else [int(args.fold)]
    )

    all_results = []
    for fold in folds_to_run:
        result = evaluate_fold(cfg, fold, splits, df_meta, device, use_tta=args.tta)
        all_results.append(result)

    # CV-level summary across folds
    if len(all_results) > 1:
        print("\n" + "=" * 55)
        print("  CV SUMMARY ACROSS ALL FOLDS")
        print("=" * 55)
        for m in OFFICIAL_METRICS:
            vals = [r["aggregate"]["overall"][f"{m}_mean"] for r in all_results]
            print(f"  {m:<28s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
        print("=" * 55)

    summary_path = (
        Path(cfg.logging.log_dir) /
        f"cv_eval_track{cfg.conditioning.track}.json"
    )
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"CV evaluation saved: {summary_path}")


if __name__ == "__main__":
    main()
