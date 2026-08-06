"""
eval_utils.py
-------------
Evaluator utilities for ISLES26 official metrics.

Note: These functions were originally provided by the ISLES26 organizers.
This file implements the interface expected by evaluate.py.

The official ISLES26 metrics:
1. Dice Score (DSC) - global binary overlap
2. Absolute Volume Difference (mL)
3. Absolute Lesion Count Difference - instance count |GT - Pred|
4. Lesion-wise F1 Score - panoptica threshold=0.25
5. PR-AUC - requires soft probability output
"""

import warnings
from typing import Optional

import numpy as np

# Try to import panoptica for lesion-wise metrics
try:
    import panoptica
    PANOPTICA_AVAILABLE = True
except ImportError:
    PANOPTICA_AVAILABLE = False
    warnings.warn(
        "panoptica not installed. Install with: pip install panoptica\n"
        "Lesion-wise F1 will return NaN without panoptica."
    )


# ── Dice Score ─────────────────────────────────────────────────────────────────

def compute_dice_score(
    ground_truth: np.ndarray,
    prediction:   np.ndarray,
    smooth: float = 1e-5,
) -> float:
    """
    Compute Dice Score (DSC) for binary segmentation.

    Args:
        ground_truth: Binary mask (H, W, D)
        prediction:   Binary mask (H, W, D)
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Dice score in [0, 1]
    """
    intersection = np.sum(ground_truth * prediction)
    union = np.sum(ground_truth) + np.sum(prediction)

    if union == 0:
        return 1.0  # Both empty = perfect match

    return float((2.0 * intersection + smooth) / (union + smooth))


# ── Volume Difference ──────────────────────────────────────────────────────────

def compute_absolute_volume_difference(
    im1:        np.ndarray,
    im2:        np.ndarray,
    voxel_size: float,
) -> float:
    """
    Compute absolute volume difference in mL.

    Args:
        im1:        Binary mask (H, W, D)
        im2:        Binary mask (H, W, D)
        voxel_size: Volume of one voxel in mL (e.g., 1.0/1000.0 for 1mm³)

    Returns:
        Absolute volume difference in mL
    """
    vol1 = float(np.sum(im1)) * voxel_size
    vol2 = float(np.sum(im2)) * voxel_size
    return abs(vol1 - vol2)


# ── Dice, Lesion F1, Instance Difference (combined for compatibility) ─────────

def compute_dice_f1_instance_difference(
    ground_truth: np.ndarray,
    prediction:   np.ndarray,
    empty_value:  float = 1.0,
    voxel_size:   float = 1.0 / 1000.0,
) -> tuple[float, int, float]:
    """
    Compute Dice Score, lesion-wise F1, and instance count difference.

    This function wraps multiple metrics for compatibility with the expected
    interface from the original ISLES26 evaluate.py.

    Args:
        ground_truth: Binary mask (H, W, D)
        prediction:   Binary mask (H, W, D)
        empty_value:  Value to return for empty masks (default 1.0)
        voxel_size:   Volume of one voxel in mL (default 1mm³)

    Returns:
        Tuple of (lesion_f1, lesion_count_diff, dice_score)
    """
    # Dice score
    dice = compute_dice_score(ground_truth, prediction)

    # Instance count difference (connected components)
    gt_labels, gt_n = _label_instances(ground_truth)
    pred_labels, pred_n = _label_instances(prediction)
    instance_diff = abs(gt_n - pred_n)

    # Lesion-wise F1 requires panoptica
    if PANOPTICA_AVAILABLE:
        # panoptica expects float masks in [0, 1]
        gt_float = ground_truth.astype(np.float32)
        pred_float = prediction.astype(np.float32)

        try:
            p_result = panoptica.evaluate(
                prediction=pred_float,
                ground_truth=gt_float,
                voxel_size=voxel_size,
                threshold=0.25,
            )
            lesion_f1 = float(p_result.f1_score)
        except Exception:
            # Fallback if panoptica fails
            lesion_f1 = float("nan")
    else:
        # No panoptica available
        lesion_f1 = float("nan")

    return lesion_f1, instance_diff, dice


def _label_instances(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Label connected components in a binary mask.

    Returns:
        Tuple of (labelled array, number of instances)
    """
    try:
        from scipy.ndimage import label
        labelled, n = label(mask)
        return labelled, int(n)
    except ImportError:
        warnings.warn("scipy unavailable for instance counting.")
        # Fallback: count non-zero voxels as single instance
        n = 1 if mask.sum() > 0 else 0
        return mask.astype(np.int32), n


# ── PR-AUC ─────────────────────────────────────────────────────────────────────

def compute_pr_auc(
    ground_truth:    np.ndarray,
    prediction_map:  np.ndarray,
    empty_value:     float = 1.0,
) -> float:
    """
    Compute Precision-Recall Area Under the Curve (PR-AUC).

    Uses soft probability map (not thresholded).

    Args:
        ground_truth:   Binary mask (H, W, D)
        prediction_map: Soft probability map (H, W, D), values in [0, 1]
        empty_value:    Value to return for empty ground truth (default 1.0)

    Returns:
        PR-AUC score in [0, 1]
    """
    from sklearn.metrics import average_precision_score

    gt_flat = ground_truth.ravel()
    pred_flat = prediction_map.ravel()

    # Handle empty ground truth
    if gt_flat.sum() == 0:
        return empty_value

    # Compute PR-AUC using sklearn
    try:
        pr_auc = average_precision_score(gt_flat, pred_flat)
        return float(pr_auc)
    except ValueError:
        # Edge case: all predictions are the same
        return float("nan")


# ── Convenience wrappers for evaluate.py compatibility ─────────────────────────

def dice_score(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    """Alias for compute_dice_score."""
    return compute_dice_score(ground_truth, prediction)


def precision_recall(
    ground_truth: np.ndarray,
    prediction:   np.ndarray,
    threshold:    float = 0.5,
) -> tuple[float, float]:
    """
    Compute precision and recall for binary masks.

    Args:
        ground_truth: Binary mask
        prediction:   Binary mask
        threshold:    Threshold for prediction (unused, kept for compatibility)

    Returns:
        Tuple of (precision, recall)
    """
    tp = np.sum((ground_truth == 1) & (prediction == 1))
    fp = np.sum((ground_truth == 0) & (prediction == 1))
    fn = np.sum((ground_truth == 1) & (prediction == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return float(precision), float(recall)


def hausdorff95(
    ground_truth: np.ndarray,
    prediction:   np.ndarray,
) -> float:
    """
    Compute 95th percentile Hausdorff distance.

    Returns NaN if either mask is empty.

    Args:
        ground_truth: Binary mask
        prediction:   Binary mask

    Returns:
        HD95 distance in voxel units
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        warnings.warn("scipy unavailable for HD95 computation.")
        return float("nan")

    if np.sum(ground_truth) == 0 or np.sum(prediction) == 0:
        return float("nan")

    # Distance from pred to gt and gt to pred
    dt_pred = distance_transform_edt(prediction == 0)
    dt_gt = distance_transform_edt(ground_truth == 0)

    # Distances at boundary voxels
    pred_boundary = (prediction > 0) & (dt_pred == 0)
    gt_boundary = (ground_truth > 0) & (dt_gt == 0)

    dist_pred_to_gt = dt_gt[pred_boundary] if pred_boundary.any() else [0]
    dist_gt_to_pred = dt_gt[gt_boundary] if gt_boundary.any() else [0]

    # 95th percentile
    hd95 = np.percentile(dist_pred_to_gt + dist_gt_to_pred, 95)
    return float(hd95)
