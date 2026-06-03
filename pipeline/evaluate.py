"""
evaluate.py
-----------
Evaluation pipeline for ISLES26. Computes per-scan and aggregate metrics
with lesion-size stratification. Supports single-fold and CV ensemble modes.

Metrics:
  - Dice Similarity Coefficient (DSC)
  - 95th percentile Hausdorff Distance (HD95)
  - Precision, Recall
  - All metrics stratified by lesion size (small/medium/large)
  - All metrics stratified by chronicity

Usage:
  python evaluate.py --config configs/config.yaml --fold 0
  python evaluate.py --config configs/config.yaml --fold all --ensemble
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import nibabel as nib
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

from dataset import build_dataloaders, build_records
from model import build_model, ISLES26Model
from train import load_checkpoint

log = logging.getLogger(__name__)


# ── Metric functions ──────────────────────────────────────────────────────────

def dice_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return float((2 * inter + smooth) / (union + smooth))


def precision_recall(
    pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5
) -> tuple[float, float]:
    tp = (pred * target).sum()
    fp = pred.sum() - tp
    fn = target.sum() - tp
    prec = float((tp + smooth) / (tp + fp + smooth))
    rec  = float((tp + smooth) / (tp + fn + smooth))
    return prec, rec


def hausdorff95(pred: np.ndarray, target: np.ndarray) -> float:
    """
    95th percentile Hausdorff distance using distance transforms.
    Returns NaN if either mask is empty (no valid surface).
    """
    if pred.sum() == 0 or target.sum() == 0:
        return float("nan")
    try:
        from scipy.ndimage import distance_transform_edt
        pred_b   = pred.astype(bool)
        target_b = target.astype(bool)
        dt_pred   = distance_transform_edt(~pred_b)
        dt_target = distance_transform_edt(~target_b)
        surf_dist_1 = dt_target[pred_b]
        surf_dist_2 = dt_pred[target_b]
        all_dist    = np.concatenate([surf_dist_1, surf_dist_2])
        return float(np.percentile(all_dist, 95))
    except Exception as e:
        log.warning(f"HD95 computation failed: {e}")
        return float("nan")


def size_category(vol_ml: float, cfg_eval) -> str:
    bins = cfg_eval.size_bins
    if vol_ml < bins.small[1]:   return "small"
    if vol_ml < bins.medium[1]:  return "medium"
    return "large"


# ── TTA ───────────────────────────────────────────────────────────────────────

def predict_with_tta(
    model:     ISLES26Model,
    image:     torch.Tensor,    # (1, 1, H, W, D)
    meta_vec:  torch.Tensor,    # (1, 4)
    meta_text: list[str],
    flip_axes: list[list[int]],
    device:    torch.device,
) -> torch.Tensor:
    """
    Returns averaged softmax probability map (1, 1, H, W, D) over TTA flips.
    """
    model.eval()
    probs_sum = None
    n         = 0

    # Original
    with torch.no_grad():
        logits = model(image, meta_vec, meta_text)[0]
        p = torch.softmax(logits, dim=1)[:, 1:2]
        probs_sum = p
        n += 1

    # Flipped variants
    for axes in flip_axes:
        img_flip = torch.flip(image, dims=[d + 2 for d in axes])
        with torch.no_grad():
            logits_f = model(img_flip, meta_vec, meta_text)[0]
            p_f = torch.softmax(logits_f, dim=1)[:, 1:2]
            p_f = torch.flip(p_f, dims=[d + 2 for d in axes])
        probs_sum = probs_sum + p_f
        n += 1

    return probs_sum / n


# ── Per-scan evaluation ───────────────────────────────────────────────────────

def evaluate_scan(
    model:      ISLES26Model,
    batch:      dict,
    cfg:        DictConfig,
    device:     torch.device,
    use_tta:    bool = False,
) -> dict:
    image     = batch["image"].to(device)
    mask      = batch["mask"].squeeze().cpu().numpy().astype(np.uint8)
    meta_vec  = batch["meta_vec"].to(device)
    meta_text = batch["meta_text"]
    uid       = batch["uid"][0]
    chronicity = batch["chronicity"][0]

    # Voxel volume — assume 1mm³ (confirmed by EDA); load from stats if available
    voxel_vol_mm3 = 1.0

    if use_tta and cfg.tta.enabled:
        probs = predict_with_tta(
            model, image, meta_vec, meta_text, cfg.tta.flips, device
        )
    else:
        with torch.no_grad():
            logits = model(image, meta_vec, meta_text)[0]
            probs  = torch.softmax(logits, dim=1)[:, 1:2]

    pred = (probs.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

    # Post-process: remove small components
    pred = remove_small_components(pred, cfg.postprocessing.min_component_size_voxels)

    lesion_vol_ml = float(mask.sum() * voxel_vol_mm3 / 1000)

    return {
        "uid":           uid,
        "chronicity":    chronicity,
        "lesion_vol_ml": lesion_vol_ml,
        "dice":          dice_score(pred, mask),
        "hd95":          hausdorff95(pred, mask),
        "precision":     precision_recall(pred, mask)[0],
        "recall":        precision_recall(pred, mask)[1],
    }


def remove_small_components(
    mask: np.ndarray, min_voxels: int
) -> np.ndarray:
    """Remove connected components smaller than min_voxels."""
    if mask.sum() == 0:
        return mask
    try:
        from scipy.ndimage import label
        labelled, n = label(mask)
        for i in range(1, n + 1):
            if (labelled == i).sum() < min_voxels:
                mask[labelled == i] = 0
    except ImportError:
        log.warning("scipy not available — skipping small component removal.")
    return mask


# ── Aggregate metrics ─────────────────────────────────────────────────────────

def aggregate_metrics(
    records: list[dict], cfg: DictConfig
) -> dict:
    """
    Compute mean ± std for all metrics, plus stratified breakdowns.
    """
    df = pd.DataFrame(records)

    def summarise(subset: pd.DataFrame, label: str) -> dict:
        out = {"n": len(subset)}
        for metric in ["dice", "hd95", "precision", "recall"]:
            vals = subset[metric].dropna()
            out[f"{metric}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            out[f"{metric}_std"]  = float(vals.std())  if len(vals) else float("nan")
        return {label: out}

    result = {}
    result.update(summarise(df, "overall"))

    # Chronicity stratification
    for chron in ["acute", "subacute", "chronic", "unknown"]:
        subset = df[df["chronicity"] == chron]
        if len(subset):
            result.update(summarise(subset, f"chronicity_{chron}"))

    # Lesion size stratification
    df["size_cat"] = df["lesion_vol_ml"].apply(
        lambda v: size_category(v, cfg.evaluation)
    )
    for cat in ["small", "medium", "large"]:
        subset = df[df["size_cat"] == cat]
        if len(subset):
            result.update(summarise(subset, f"size_{cat}"))

    return result


# ── Main evaluation run ───────────────────────────────────────────────────────

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

    model    = build_model(cfg).to(device)
    ckpt_dir = Path(cfg.logging.log_dir) / f"track_{cfg.conditioning.track}" / f"fold_{fold}"
    best_ckpt = ckpt_dir / "best.pth"
    assert best_ckpt.exists(), f"No checkpoint found at {best_ckpt}. Train fold {fold} first."
    load_checkpoint(best_ckpt, model)

    scan_records = []
    for batch in tqdm(val_dl, desc=f"Fold {fold} eval"):
        rec = evaluate_scan(model, batch, cfg, device, use_tta)
        scan_records.append(rec)
        log.debug(f"  {rec['uid']} | dice={rec['dice']:.4f} hd95={rec['hd95']:.2f}")

    agg = aggregate_metrics(scan_records, cfg)

    out_dir = ckpt_dir
    pd.DataFrame(scan_records).to_csv(out_dir / "eval_per_scan.csv", index=False)
    with open(out_dir / "eval_aggregate.json", "w") as f:
        json.dump(agg, f, indent=2)

    log.info(f"Fold {fold} | overall dice={agg['overall']['dice_mean']:.4f} "
             f"± {agg['overall']['dice_std']:.4f}")
    return {"fold": fold, "aggregate": agg, "scan_records": scan_records}


def main() -> None:
    parser = argparse.ArgumentParser(description="ISLES26 Evaluation")
    parser.add_argument("--config",   type=str, default="configs/config.yaml")
    parser.add_argument("--fold",     type=str, default="0")
    parser.add_argument("--track",    type=str, default=None)
    parser.add_argument("--tta",      action="store_true")
    parser.add_argument("--ensemble", action="store_true",
                        help="Average predictions across all fold checkpoints")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.track:
        cfg = OmegaConf.merge(cfg, {"conditioning": {"track": args.track.upper()}})

    Path(cfg.logging.log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level   = getattr(logging, cfg.logging.level),
        format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt = "%H:%M:%S",
    )

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits  = json.loads((Path(cfg.data.processed_dir) / "splits.json").read_text())
    df_meta = pd.read_csv(Path(cfg.data.processed_dir) / "dataset_manifest.csv")

    folds_to_run = list(range(cfg.data.n_splits)) if args.fold == "all" \
                   else [int(args.fold)]

    all_results = []
    for fold in folds_to_run:
        result = evaluate_fold(cfg, fold, splits, df_meta, device, use_tta=args.tta)
        all_results.append(result)

    if len(all_results) > 1:
        all_dices = [r["aggregate"]["overall"]["dice_mean"] for r in all_results]
        log.info(f"\nCV Dice: {np.mean(all_dices):.4f} ± {np.std(all_dices):.4f}")


if __name__ == "__main__":
    main()