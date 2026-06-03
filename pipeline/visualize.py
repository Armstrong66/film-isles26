"""
visualize.py
------------
Plotting utilities for ISLES26. All functions save figures to disk
and optionally display them inline (Kaggle/Jupyter compatible).

Functions:
  plot_training_curves(history_path, out_dir)
  plot_cv_summary(cv_summary_path, out_dir)
  plot_metric_by_stratum(eval_csv_path, out_dir)
  plot_track_comparison(results_A, results_C, out_dir)
  plot_scan_overlay(img_path, mask_path, pred_path, out_path)
  plot_prediction_grid(scan_records, n_samples, out_dir)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import nibabel as nib

log = logging.getLogger(__name__)

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE = {
    "acute":    "#e07b54",
    "subacute": "#f5c242",
    "chronic":  "#4a90d9",
    "unknown":  "#aaaaaa",
    "train":    "#4a90d9",
    "val":      "#e07b54",
    "track_A":  "#6db88f",
    "track_C":  "#9b59b6",
    "small":    "#e07b54",
    "medium":   "#f5c242",
    "large":    "#4a90d9",
}

def _save(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {path}")


# ── Training curves ───────────────────────────────────────────────────────────

def plot_training_curves(
    history_path: Path,
    out_dir:      Path,
    fold:         int = 0,
) -> None:
    """Loss and Dice curves for a single fold."""
    with open(history_path) as f:
        history = json.load(f)

    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, label in zip(
        axes,
        [("train_loss", "val_loss"), ("train_dice", "val_dice")],
        ["Loss", "Dice"],
    ):
        ax.plot(df["epoch"], df[metric[0]], color=PALETTE["train"],
                label="Train", linewidth=1.5)
        ax.plot(df["epoch"], df[metric[1]], color=PALETTE["val"],
                label="Val", linewidth=1.5, linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(f"{label} — Fold {fold}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Mark best val epoch
        if "val_dice" in df.columns:
            best_ep = df["val_dice"].idxmax()
            ax.axvline(df.loc[best_ep, "epoch"], color="grey",
                       linestyle=":", linewidth=1, label=f"best ep={best_ep}")

    plt.tight_layout()
    _save(fig, out_dir / f"fold{fold}_training_curves.png")


def plot_cv_summary(
    cv_summary_path: Path,
    out_dir:         Path,
    track:           str = "A",
) -> None:
    """Bar chart of per-fold best val Dice for a full CV run."""
    with open(cv_summary_path) as f:
        results = json.load(f)

    folds = [r["fold"] for r in results]
    dices = [r["best_val_dice"] for r in results]
    mean  = np.mean(dices)
    std   = np.std(dices)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([f"Fold {f}" for f in folds], dices,
                  color=PALETTE[f"track_{track}"], edgecolor="white", width=0.5)
    ax.axhline(mean, color="#333333", linestyle="--", linewidth=1.5,
               label=f"Mean={mean:.4f} ± {std:.4f}")

    for bar, d in zip(bars, dices):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{d:.4f}", ha="center", fontsize=9)

    ax.set_ylim(max(0, min(dices) - 0.05), min(1.0, max(dices) + 0.05))
    ax.set_ylabel("Best Val Dice")
    ax.set_title(f"5-Fold CV Summary — Track {track}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, out_dir / f"cv_summary_track{track}.png")


# ── Metric stratification ─────────────────────────────────────────────────────

def plot_metric_by_stratum(
    eval_csv_path: Path,
    out_dir:       Path,
    fold:          int = 0,
    track:         str = "A",
) -> None:
    """
    Two panels: Dice by chronicity class, Dice by lesion size category.
    """
    df = pd.read_csv(eval_csv_path)

    def _boxplot(ax, groups: dict, title: str, palette: dict) -> None:
        data   = [v.dropna().values for v in groups.values()]
        labels = list(groups.keys())
        colors = [palette.get(k, "#888888") for k in labels]

        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        medianprops=dict(color="white", linewidth=2))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        # Overlay individual points
        for i, (vals, color) in enumerate(zip(data, colors), start=1):
            jitter = np.random.normal(0, 0.07, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       color=color, alpha=0.4, s=12, zorder=3)

        ax.set_ylabel("Dice")
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", alpha=0.3)

        # Annotate medians
        for i, vals in enumerate(data, start=1):
            if len(vals):
                ax.text(i, np.median(vals) + 0.03, f"{np.median(vals):.3f}",
                        ha="center", fontsize=8, color="black")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Chronicity
    chron_groups = {
        c: df[df["chronicity"] == c]["dice"]
        for c in ["acute", "subacute", "chronic", "unknown"]
        if (df["chronicity"] == c).any()
    }
    _boxplot(axes[0], chron_groups, f"Dice by Chronicity — Fold {fold} Track {track}",
             PALETTE)

    # Size category
    if "lesion_vol_ml" in df.columns:
        df["size_cat"] = pd.cut(
            df["lesion_vol_ml"],
            bins  = [0, 1, 10, 1e9],
            labels = ["small", "medium", "large"],
        )
        size_groups = {
            cat: df[df["size_cat"] == cat]["dice"]
            for cat in ["small", "medium", "large"]
            if (df["size_cat"] == cat).any()
        }
        _boxplot(axes[1], size_groups,
                 f"Dice by Lesion Size — Fold {fold} Track {track}", PALETTE)

    plt.tight_layout()
    _save(fig, out_dir / f"fold{fold}_track{track}_dice_by_stratum.png")


def plot_track_comparison(
    eval_csv_A: Path,
    eval_csv_C: Path,
    out_dir:    Path,
    fold:       int = 0,
) -> None:
    """
    Side-by-side Dice comparison of Track A vs Track C,
    broken down by chronicity and size.
    """
    df_A = pd.read_csv(eval_csv_A).assign(track="A")
    df_C = pd.read_csv(eval_csv_C).assign(track="C")
    df   = pd.concat([df_A, df_C], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, col, title in [
        (axes[0], "chronicity", "Dice by Chronicity"),
        (axes[1], "size_cat",   "Dice by Lesion Size"),
    ]:
        if col == "size_cat" and col not in df.columns:
            df["size_cat"] = pd.cut(
                df["lesion_vol_ml"], bins=[0,1,10,1e9],
                labels=["small","medium","large"]
            )

        categories = df[col].dropna().unique()
        x          = np.arange(len(categories))
        width      = 0.35

        for offset, track_label, color in [
            (-width/2, "A", PALETTE["track_A"]),
            ( width/2, "C", PALETTE["track_C"]),
        ]:
            means = [
                df[(df["track"] == track_label) & (df[col] == cat)]["dice"].mean()
                for cat in categories
            ]
            stds = [
                df[(df["track"] == track_label) & (df[col] == cat)]["dice"].std()
                for cat in categories
            ]
            bars = ax.bar(x + offset, means, width, label=f"Track {track_label}",
                          color=color, alpha=0.8, yerr=stds, capsize=4)

        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel("Mean Dice")
        ax.set_title(f"{title} — Fold {fold}")
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Track A (FiLM) vs Track C (LLM) Comparison", fontsize=13)
    plt.tight_layout()
    _save(fig, out_dir / f"fold{fold}_track_comparison.png")


# ── Scan overlay visualisation ────────────────────────────────────────────────

def plot_scan_overlay(
    img_path:  Path,
    mask_path: Path,
    out_path:  Path,
    pred_path: Optional[Path] = None,
    uid:       str = "",
    meta:      str = "",
) -> None:
    """
    3-plane (axial/coronal/sagittal) overlay of image + GT mask + optional prediction.
    Finds the slice with maximum lesion area for each plane.
    """
    img_data  = nib.load(str(img_path)).get_fdata(dtype=np.float32)
    mask_data = nib.load(str(mask_path)).get_fdata(dtype=np.float32)
    pred_data = nib.load(str(pred_path)).get_fdata(dtype=np.float32) \
                if pred_path and pred_path.exists() else None

    fg    = img_data[img_data > 0]
    vmin  = float(np.percentile(fg, 1)) if len(fg) else 0
    vmax  = float(np.percentile(fg, 99)) if len(fg) else 1

    n_cols  = 3
    n_rows  = 2 if pred_data is not None else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))

    if n_rows == 1:
        axes = axes[np.newaxis, :]   # ensure 2D indexing

    plane_specs = [
        ("Axial",    2, lambda d, i: d[:, :, i],     lambda m: m.sum(axis=(0,1))),
        ("Coronal",  1, lambda d, i: d[:, i, :],     lambda m: m.sum(axis=(0,2))),
        ("Sagittal", 0, lambda d, i: d[i, :, :],     lambda m: m.sum(axis=(1,2))),
    ]

    for col, (plane_name, axis, slicer, lesion_proj) in enumerate(plane_specs):
        # Find slice with largest lesion area
        proj     = lesion_proj(mask_data)
        best_idx = int(np.argmax(proj)) if proj.max() > 0 \
                   else img_data.shape[axis] // 2

        img_sl  = slicer(img_data,  best_idx).T
        mask_sl = slicer(mask_data, best_idx).T

        for row, (data_sl, row_label) in enumerate(
            [(mask_sl, "GT")] + ([(slicer(pred_data, best_idx).T, "Pred")]
                                  if pred_data is not None else [])
        ):
            ax = axes[row, col]
            ax.imshow(img_sl, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
            if data_sl.max() > 0:
                ax.imshow(
                    np.ma.masked_where(data_sl == 0, data_sl),
                    cmap="Reds", alpha=0.55, origin="lower",
                )
            ax.set_title(f"{plane_name} [{row_label}]", fontsize=9)
            ax.axis("off")

    title = f"{uid}  {meta}" if uid else meta
    fig.suptitle(title, fontsize=10, y=1.01)
    plt.tight_layout()
    _save(fig, out_path)


def plot_prediction_grid(
    eval_csv_path: Path,
    out_dir:       Path,
    proc_img_dir:  Path,
    proc_mask_dir: Path,
    n_per_class:   int = 2,
    fold:          int = 0,
    track:         str = "A",
) -> None:
    """
    Grid of scan overlays: best and worst Dice per chronicity class.
    """
    df = pd.read_csv(eval_csv_path)

    for chron in ["acute", "subacute", "chronic", "unknown"]:
        subset = df[df["chronicity"] == chron].sort_values("dice")
        if len(subset) == 0:
            continue

        # Worst and best
        picks = pd.concat([subset.head(n_per_class), subset.tail(n_per_class)])

        for _, row in picks.iterrows():
            uid      = row["uid"]
            img_path  = proc_img_dir  / f"{uid}_T1w.nii.gz"
            mask_path = proc_mask_dir / f"{uid}_mask.nii.gz"
            if not img_path.exists():
                continue

            label = "best" if row["dice"] == subset["dice"].max() else "worst"
            meta  = (f"chron={chron} dice={row['dice']:.3f} "
                     f"vol={row.get('lesion_vol_ml', 0):.1f}mL")
            out_path = out_dir / f"overlay_fold{fold}_{chron}_{label}_{uid}.png"
            plot_scan_overlay(img_path, mask_path, out_path,
                              uid=uid, meta=meta)