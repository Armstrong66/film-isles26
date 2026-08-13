"""
interpretability.py
-------------------
Research-grade visualization and interpretability for ISLES26.
Companion to visualize.py — call after evaluate.py completes.

All functions are stateless and safe to run post-training.
Nothing here modifies model weights or affects training stability.

Usage:
  python scripts/interpretability.py \
      --config configs/config.yaml \
      --fold 0 \
      --track A \
      [--no-gradcam]   # skip GradCAM if time-constrained
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

log = logging.getLogger(__name__)

# ── Publication colour palette (LNCS-safe, greyscale-distinguishable) ─────────
CHRON_PALETTE = {
    "acute":    "#E07B54",
    "subacute": "#F5C242",
    "chronic":  "#4A90D9",
    "unknown":  "#AAAAAA",
}
TRACK_PALETTE = {"A": "#2E8B57", "C": "#6A0DAD"}
LNCS_FULL_WIDTH_MM = 170
LNCS_HALF_WIDTH_MM = 83
DPI = 600

def mm_to_inches(mm: float) -> float:
    return mm / 25.4

def save_fig(fig: plt.Figure, path: Path, width_mm: float = LNCS_FULL_WIDTH_MM) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = mm_to_inches(width_mm)
    h = fig.get_size_inches()[1]
    fig.set_size_inches(w, h)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"Saved: {path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# A. FiLM Gate Analysis
# ═══════════════════════════════════════════════════════════════════════════════

class FiLMStatsHook:
    """
    Lightweight read-only hook for collecting γ/β statistics per forward pass.
    Attaches to model._film_hook — zero computational overhead beyond stats.
    """
    def __init__(self):
        self.records: list[dict] = []

    def __call__(
        self,
        gamma:      torch.Tensor,   # (B, C)
        beta:       torch.Tensor,   # (B, C)
        meta_text:  list[str],      # B strings (used to extract chronicity)
        chronicity_batch: list[str],
    ) -> None:
        for i, chron in enumerate(chronicity_batch):
            self.records.append({
                "chronicity":  chron,
                "gamma_mean":  gamma[i].mean().item(),
                "gamma_std":   gamma[i].std().item(),
                "beta_mean":   beta[i].mean().item(),
                "beta_std":    beta[i].std().item(),
                "gamma_vec":   gamma[i].detach().cpu().float().numpy(),
                "beta_vec":    beta[i].detach().cpu().float().numpy(),
            })

    def to_dataframe(self) -> pd.DataFrame:
        scalar_cols = ["chronicity", "gamma_mean", "gamma_std", "beta_mean", "beta_std"]
        return pd.DataFrame([{k: r[k] for k in scalar_cols} for r in self.records])

    def gamma_matrix(self) -> tuple[np.ndarray, list[str]]:
        """Returns (N, C) matrix of gamma vectors and list of chronicity labels."""
        vecs   = np.stack([r["gamma_vec"] for r in self.records])
        labels = [r["chronicity"] for r in self.records]
        return vecs, labels

    def beta_matrix(self) -> tuple[np.ndarray, list[str]]:
        vecs   = np.stack([r["beta_vec"] for r in self.records])
        labels = [r["chronicity"] for r in self.records]
        return vecs, labels


def collect_film_stats(model, val_dl, device: torch.device) -> FiLMStatsHook:
    """Run one val pass with FiLM hook attached. Returns populated hook."""
    hook = FiLMStatsHook()
    model._film_hook = hook
    model.eval()

    with torch.no_grad():
        for batch in tqdm(val_dl, desc="FiLM stats"):
            image     = batch["image"].to(device)
            meta_vec  = batch["meta_vec"].to(device)
            meta_text = batch["meta_text"]
            chronicity = list(batch["chronicity"])
            uid       = list(batch["uid"])
            model(image, meta_vec, meta_text, chronicity, uid)   # hook fires inside forward()

    model._film_hook = None
    log.info(f"FiLM stats collected: {len(hook.records)} scans")
    return hook


def plot_film_gate_analysis(hook: FiLMStatsHook, out_dir: Path) -> None:
    """Four-panel FiLM gate analysis figure."""
    df    = hook.to_dataframe()
    order = ["acute", "subacute", "chronic", "unknown"]
    colors = [CHRON_PALETTE[c] for c in order]

    fig = plt.figure(figsize=(mm_to_inches(LNCS_FULL_WIDTH_MM), 7))
    gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

    # A1 — γ mean ± std per chronicity
    for ax, stat, label in [
        (axes[0], ("gamma_mean", "gamma_std"), "γ (scale)"),
        (axes[1], ("beta_mean",  "beta_std"),  "β (shift)"),
    ]:
        means = [df[df.chronicity == c][stat[0]].mean() for c in order]
        stds  = [df[df.chronicity == c][stat[1]].mean() for c in order]
        bars  = ax.bar(order, means, color=colors, alpha=0.8,
                       yerr=stds, capsize=4, error_kw={"linewidth": 1})
        ax.set_ylabel(f"Mean {label}")
        ax.set_title(f"FiLM {label} by chronicity")
        ax.axhline(1.0 if "gamma" in stat[0] else 0.0,
                   color="k", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.3)

    # A3 — γ heatmap (mean per chronicity class × channel subset)
    gamma_mat, labels = hook.gamma_matrix()
    ax = axes[2]
    class_means = np.stack([
        gamma_mat[[i for i, l in enumerate(labels) if l == c]].mean(axis=0)
        if any(l == c for l in labels) else np.ones(gamma_mat.shape[1])
        for c in order
    ])
    # Show first 64 channels for readability
    n_show = min(64, class_means.shape[1])
    im = ax.imshow(class_means[:, :n_show], aspect="auto",
                   cmap="RdBu_r", vmin=0.5, vmax=1.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("Bottleneck channel (first 64)")
    ax.set_title("Mean γ per class (channel heatmap)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # A4 — γ × β joint scatter
    ax = axes[3]
    for c, color in CHRON_PALETTE.items():
        sub = df[df.chronicity == c]
        if len(sub):
            ax.scatter(sub["gamma_mean"], sub["beta_mean"],
                       c=color, label=c, alpha=0.6, s=15, edgecolors="none")
    ax.axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(1, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("γ mean (scale)")
    ax.set_ylabel("β mean (shift)")
    ax.set_title("γ–β fingerprint per scan")
    ax.legend(fontsize=7, markerscale=1.5)
    ax.grid(alpha=0.3)

    plt.suptitle("FiLM Conditioning Gate Analysis", fontsize=10, y=1.01)
    save_fig(fig, out_dir / "film_gate_analysis.png")

    # Save raw stats for paper table
    summary = df.groupby("chronicity")[
        ["gamma_mean", "gamma_std", "beta_mean", "beta_std"]
    ].agg(["mean", "std"]).round(4)
    summary.to_csv(out_dir / "film_gate_stats.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# B. Bottleneck Embedding UMAP
# ═══════════════════════════════════════════════════════════════════════════════

class BottleneckEmbeddingHook:
    def __init__(self):
        self.records: list[dict] = []

    def __call__(self, x_pre: torch.Tensor, x_post: torch.Tensor,
                 chronicity: list[str], uid: list[str]) -> None:
        pre_emb  = x_pre.mean(dim=(2, 3, 4)).detach().cpu().float().numpy()
        post_emb = x_post.mean(dim=(2, 3, 4)).detach().cpu().float().numpy()
        for i, (c, u) in enumerate(zip(chronicity, uid)):
            self.records.append({
                "chronicity": c, "uid": u,
                "before": pre_emb[i],
                "after":  post_emb[i],
            })


def collect_bottleneck_embeddings(
    model, val_dl, device: torch.device
) -> BottleneckEmbeddingHook:
    hook = BottleneckEmbeddingHook()
    model._embed_hook = hook
    model.eval()

    with torch.no_grad():
        for batch in tqdm(val_dl, desc="Embeddings"):
            image     = batch["image"].to(device)
            meta_vec  = batch["meta_vec"].to(device)
            meta_text = batch["meta_text"]
            chronicity = list(batch["chronicity"])
            uid       = list(batch["uid"])
            model(image, meta_vec, meta_text, chronicity, uid)

    model._embed_hook = None
    return hook


def plot_umap_embeddings(
    hook:    BottleneckEmbeddingHook,
    out_dir: Path,
    random_state: int = 42,
) -> None:
    try:
        import umap
    except ImportError:
        log.warning("umap-learn not installed. Skipping embedding plot. "
                    "Install with: pip install umap-learn")
        return

    labels   = [r["chronicity"] for r in hook.records]
    colors   = [CHRON_PALETTE.get(c, "#888888") for c in labels]
    emb_pre  = np.stack([r["before"] for r in hook.records])
    emb_post = np.stack([r["after"]  for r in hook.records])

    reducer = umap.UMAP(n_components=2, random_state=random_state,
                        n_neighbors=15, min_dist=0.1)
    umap_pre  = reducer.fit_transform(emb_pre)
    umap_post = reducer.fit_transform(emb_post)

    fig, axes = plt.subplots(1, 2, figsize=(mm_to_inches(LNCS_FULL_WIDTH_MM), 3.5))

    for ax, coords, title in [
        (axes[0], umap_pre,  "Before conditioning"),
        (axes[1], umap_post, "After conditioning"),
    ]:
        for chron, color in CHRON_PALETTE.items():
            mask = [l == chron for l in labels]
            if any(mask):
                ax.scatter(coords[mask, 0], coords[mask, 1],
                           c=color, label=chron, alpha=0.6, s=10, edgecolors="none")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("UMAP-1", fontsize=8)
        ax.set_ylabel("UMAP-2", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)

    axes[1].legend(fontsize=7, markerscale=2, loc="upper right")
    plt.suptitle("Bottleneck Embeddings (UMAP)", fontsize=10)
    plt.tight_layout()
    save_fig(fig, out_dir / "umap_embeddings.png")


# ═══════════════════════════════════════════════════════════════════════════════
# C. GradCAM Attribution
# ═══════════════════════════════════════════════════════════════════════════════

def compute_gradcam_3d(
    model,
    image:     torch.Tensor,   # (1, 1, H, W, D)
    meta_vec:  torch.Tensor,
    meta_text: list[str],
    target_layer: torch.nn.Module,
    chronicity: Optional[list[str]] = None,
    uid:        Optional[list[str]] = None,
) -> np.ndarray:
    """
    3D GradCAM on target_layer (model.bottleneck recommended).
    Returns saliency map (H, W, D) upsampled to input resolution.
    """
    try:
        from captum.attr import LayerGradCam
    except ImportError:
        log.warning("captum not installed. Skipping GradCAM. "
                    "Install with: pip install captum")
        return np.zeros(image.shape[2:])

    model.train()   # GradCAM needs grad — temporarily set train mode
    gcam = LayerGradCam(
        lambda img, chron=None, uid=None: model(img, meta_vec, meta_text, chron, uid)[0][:, 1:2],
        target_layer,
    )
    attribution = gcam.attribute(image, target=0)   # (1, C, h, w, d)
    saliency    = F.relu(attribution.mean(dim=1, keepdim=True))
    saliency    = F.interpolate(saliency, size=image.shape[2:], mode="trilinear",
                                align_corners=False)
    model.eval()
    return saliency.squeeze().detach().cpu().numpy()


def plot_gradcam_grid(
    model,
    eval_df:  pd.DataFrame,
    cfg:      DictConfig,
    device:   torch.device,
    out_dir:  Path,
    n_per_class: int = 3,
) -> None:
    """3×n_per_class GradCAM overlay grid, one row per chronicity class."""
    proc_img_dir  = Path(cfg.data.processed_dir) / "images"
    proc_mask_dir = Path(cfg.data.processed_dir) / "masks"
    classes       = ["acute", "subacute", "chronic"]  # skip unknown for clarity

    fig, axes = plt.subplots(
        len(classes), n_per_class,
        figsize=(mm_to_inches(LNCS_FULL_WIDTH_MM), len(classes) * 2.5)
    )

    for row_i, chron in enumerate(classes):
        subset = eval_df[eval_df["chronicity"] == chron].nlargest(n_per_class, "dice")

        for col_i in range(n_per_class):
            ax = axes[row_i, col_i]
            ax.axis("off")

            if col_i >= len(subset):
                continue

            rec       = subset.iloc[col_i]
            uid       = rec["uid"]
            img_path  = proc_img_dir  / f"{uid}_T1w.nii.gz"
            mask_path = proc_mask_dir / f"{uid}_mask.nii.gz"

            if not img_path.exists():
                continue

            img_nib  = nib.load(str(img_path))
            mask_nib = nib.load(str(mask_path))
            img_data  = img_nib.get_fdata(dtype=np.float32)
            mask_data = (mask_nib.get_fdata() > 0.5).astype(np.uint8)

            # Find axial slice with most lesion
            best_z = int(np.argmax(mask_data.sum(axis=(0, 1))))
            img_sl  = img_data[:, :, best_z].T
            mask_sl = mask_data[:, :, best_z].T

            # GradCAM
            image_t   = torch.from_numpy(img_data[np.newaxis, np.newaxis]).float().to(device)
            meta_vec_t = torch.zeros(1, 5).to(device)   # dummy
            meta_text  = [f"chronicity {chron}"]

            try:
                saliency  = compute_gradcam_3d(model, image_t, meta_vec_t, meta_text,
                                               model.bottleneck, chronicity=[chron], uid=[uid])
                sal_sl    = saliency[:, :, best_z].T
                sal_norm  = (sal_sl - sal_sl.min()) / (sal_sl.max() - sal_sl.min() + 1e-8)
            except Exception:
                sal_norm = None

            # Plot
            fg   = img_data[img_data > 0]
            vmin = float(np.percentile(fg, 1)) if len(fg) else 0
            vmax = float(np.percentile(fg, 99)) if len(fg) else 1
            ax.imshow(img_sl, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
            if sal_norm is not None:
                ax.imshow(sal_norm, cmap="hot", alpha=0.4, origin="lower")
            if mask_sl.max() > 0:
                ax.contour(mask_sl, levels=[0.5], colors=["cyan"], linewidths=0.8)

            dice_str = f"Dice={rec.get('dice', 0):.2f}"
            vol_str  = f"{rec.get('gt_vol_ml', 0):.1f}mL"
            ax.set_title(f"{dice_str}\n{vol_str}", fontsize=6, pad=2)

            if col_i == 0:
                ax.set_ylabel(chron.capitalize(), fontsize=8, rotation=90, labelpad=4)

    plt.suptitle("GradCAM Attribution (cyan = GT lesion contour, orange = saliency)",
                 fontsize=9)
    plt.tight_layout()
    save_fig(fig, out_dir / "gradcam_grid.png")


# ═══════════════════════════════════════════════════════════════════════════════
# D. Ablation Summary Table (print + save CSV for LaTeX)
# ═══════════════════════════════════════════════════════════════════════════════

def compile_ablation_table(
    results: list[dict],   # list of {"variant": str, "fold0_metrics": dict}
    out_dir: Path,
) -> None:
    rows = []
    for r in results:
        m = r["fold0_metrics"]
        rows.append({
            "Variant":     r["variant"],
            "Dice":        m.get("dice_mean", float("nan")),
            "PR-AUC":      m.get("pr_auc_mean", float("nan")),
            "Lesion-F1":   m.get("lesion_f1_mean", float("nan")),
            "VolDiff(mL)": m.get("abs_vol_diff_ml_mean", float("nan")),
            "LesionΔ":     m.get("abs_lesion_count_diff_mean", float("nan")),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ablation_table.csv", index=False)

    # Print LaTeX snippet
    print("\n% ── Ablation Table LaTeX ──────────────────────────────────────")
    print(df.to_latex(index=False, float_format="%.4f",
                      column_format="lrrrrr",
                      caption="Ablation study (fold 0 validation).",
                      label="tab:ablation"))


# ═══════════════════════════════════════════════════════════════════════════════
# E. Dataset Sample Grid
# ═══════════════════════════════════════════════════════════════════════════════

def plot_dataset_sample_grid(
    proc_img_dir:  Path,
    proc_mask_dir: Path,
    eval_csv:      Path,
    out_path:      Path,
    n_per_class:   int = 3,
) -> None:
    """
    Grid of representative T1w + lesion overlay samples per chronicity class.
    Columns: small / medium / large lesion. Rows: acute / subacute / chronic / unknown.
    """
    df = pd.read_csv(eval_csv)
    df["size_cat"] = pd.cut(df["gt_vol_ml"], bins=[0, 1, 10, 1e9],
                            labels=["small", "medium", "large"])

    classes   = ["acute", "subacute", "chronic", "unknown"]
    size_cats = ["small", "medium", "large"]
    n_rows, n_cols = len(classes), len(size_cats)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(mm_to_inches(LNCS_FULL_WIDTH_MM), n_rows * 2.2))

    for row_i, chron in enumerate(classes):
        for col_i, size in enumerate(size_cats):
            ax = axes[row_i, col_i]
            ax.axis("off")

            subset = df[(df["chronicity"] == chron) & (df["size_cat"] == size)]
            if len(subset) == 0:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7, color="gray")
                continue

            med_vol = subset["gt_vol_ml"].median()
            rec     = subset.iloc[(subset["gt_vol_ml"] - med_vol).abs().argsort().iloc[0]]
            uid     = rec["uid"]

            img_path  = proc_img_dir  / f"{uid}_T1w.nii.gz"
            mask_path = proc_mask_dir / f"{uid}_mask.nii.gz"
            if not img_path.exists():
                continue

            img_data  = nib.load(str(img_path)).get_fdata(dtype=np.float32)
            mask_data = (nib.load(str(mask_path)).get_fdata() > 0.5).astype(np.uint8)

            best_z  = int(np.argmax(mask_data.sum(axis=(0, 1))))
            img_sl  = img_data[:, :, best_z].T
            mask_sl = mask_data[:, :, best_z].T

            fg   = img_data[img_data > 0]
            vmin = float(np.percentile(fg, 1)) if len(fg) else 0
            vmax = float(np.percentile(fg, 99)) if len(fg) else 1

            ax.imshow(img_sl, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
            if mask_sl.max() > 0:
                ax.imshow(np.ma.masked_where(mask_sl == 0, mask_sl),
                          cmap="Reds", alpha=0.5, origin="lower")

            ax.set_title(f"{rec.get('gt_vol_ml', 0):.1f} mL", fontsize=6, pad=1)

            if col_i == 0:
                ax.set_ylabel(chron.capitalize(), fontsize=8, rotation=90, labelpad=2)
            if row_i == 0:
                ax.set_xlabel(f"{size}\nlesion", fontsize=7)
                ax.xaxis.set_label_position("top")

    plt.suptitle("Representative training samples (red overlay = lesion GT)",
                 fontsize=9, y=1.01)
    plt.tight_layout()
    save_fig(fig, out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# F. PR Curves + Calibration
# ═══════════════════════════════════════════════════════════════════════════════

def plot_pr_and_calibration(
    eval_csv:  Path,
    prob_dir:  Path,      # directory with {uid}_prob.npy files
    mask_dir:  Path,
    out_dir:   Path,
) -> None:
    from sklearn.metrics import precision_recall_curve, auc
    from sklearn.calibration import calibration_curve

    df = pd.read_csv(eval_csv)
    fig, axes = plt.subplots(1, 2,
                             figsize=(mm_to_inches(LNCS_FULL_WIDTH_MM), 3.5))

    all_gt_pool, all_prob_pool = [], []

    for chron, color in CHRON_PALETTE.items():
        subset = df[df["chronicity"] == chron]
        gt_all, prob_all = [], []
        for _, row in subset.iterrows():
            prob_path = prob_dir / f"{row.uid}_prob.npy"
            mask_path = mask_dir / f"{row.uid}_mask.nii.gz"
            if not prob_path.exists() or not mask_path.exists():
                continue
            prob = np.load(str(prob_path)).ravel().astype(np.float32)
            gt   = (nib.load(str(mask_path)).get_fdata() > 0.5).ravel()
            gt_all.extend(gt.tolist()); prob_all.extend(prob.tolist())
            all_gt_pool.extend(gt.tolist()); all_prob_pool.extend(prob.tolist())

        if len(gt_all) == 0 or sum(gt_all) == 0:
            continue

        p, r, _ = precision_recall_curve(gt_all, prob_all)
        auc_val  = auc(r, p)
        axes[0].plot(r, p, color=color, linewidth=1.5,
                     label=f"{chron} ({auc_val:.3f})")

    axes[0].set_xlabel("Recall", fontsize=8)
    axes[0].set_ylabel("Precision", fontsize=8)
    axes[0].set_title("Precision-Recall Curves", fontsize=9)
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    # Calibration
    if sum(all_gt_pool) > 0:
        frac_pos, mean_pred = calibration_curve(
            all_gt_pool, all_prob_pool, n_bins=10, strategy="uniform"
        )
        axes[1].plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect")
        axes[1].plot(mean_pred, frac_pos, "o-", color=TRACK_PALETTE["A"],
                     linewidth=1.5, markersize=4, label="Track A")
        axes[1].set_xlabel("Mean predicted probability", fontsize=8)
        axes[1].set_ylabel("Fraction of positives", fontsize=8)
        axes[1].set_title("Calibration Plot", fontsize=9)
        axes[1].legend(fontsize=7)
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_fig(fig, out_dir / "pr_calibration.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Master entry point
# ═══════════════════════════════════════════════════════════════════════════════

def generate_all(
    cfg:        DictConfig,
    fold:       int,
    model,
    val_dl,
    eval_csv:   Path,
    out_dir:    Path,
    device:     torch.device,
    run_gradcam: bool = True,
    run_umap:    bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    proc_img_dir  = Path(cfg.data.processed_dir) / "images"
    proc_mask_dir = Path(cfg.data.processed_dir) / "masks"

    log.info("=== Section A: FiLM gate analysis ===")
    hook_film = collect_film_stats(model, val_dl, device)
    plot_film_gate_analysis(hook_film, out_dir)
    hook_film.to_dataframe().to_csv(out_dir / "film_stats_raw.csv", index=False)

    if run_umap:
        log.info("=== Section B: Bottleneck UMAP ===")
        hook_emb = collect_bottleneck_embeddings(model, val_dl, device)
        plot_umap_embeddings(hook_emb, out_dir)

    if run_gradcam:
        log.info("=== Section C: GradCAM ===")
        eval_df = pd.read_csv(eval_csv)
        plot_gradcam_grid(model, eval_df, cfg, device, out_dir, n_per_class=3)

    log.info("=== Section E: Dataset sample grid ===")
    plot_dataset_sample_grid(
        proc_img_dir  = proc_img_dir,
        proc_mask_dir = proc_mask_dir,
        eval_csv      = eval_csv,
        out_path      = out_dir / "dataset_samples.png",
    )

    log.info("=== Section F: PR curves + calibration ===")
    prob_dir = out_dir.parent / "probs"
    if prob_dir.exists() and any(prob_dir.glob("*.npy")):
        plot_pr_and_calibration(eval_csv, prob_dir, proc_mask_dir, out_dir)
    else:
        log.info("  Skipping PR curves — no prob maps found. "
                 "Re-run evaluate.py with --save-probs to enable.")

    log.info(f"All interpretability figures saved to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ISLES26 Interpretability")
    parser.add_argument("--config",      type=str, default="configs/config.yaml")
    parser.add_argument("--fold",        type=int, default=0)
    parser.add_argument("--track",       type=str, default=None)
    parser.add_argument("--no-gradcam",  action="store_true")
    parser.add_argument("--no-umap",     action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.track:
        cfg = OmegaConf.merge(cfg, {"conditioning": {"track": args.track.upper()}})

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from pipeline.dataset import build_dataloaders
    from pipeline.model import build_model
    from pipeline.train import load_checkpoint
    import json

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits  = json.loads((Path(cfg.data.processed_dir) / "splits.json").read_text())
    df_meta = pd.read_csv(Path(cfg.data.processed_dir) / "dataset_manifest.csv")
    _, val_dl = build_dataloaders(cfg, args.fold, splits, df_meta)

    model    = build_model(cfg).to(device)
    ckpt_dir = Path(cfg.logging.log_dir) / f"track_{cfg.conditioning.track}" / f"fold_{args.fold}"
    load_checkpoint(ckpt_dir / "best.pth", model)

    out_dir  = ckpt_dir / "interpretability"
    eval_csv = ckpt_dir / "eval_per_scan.csv"
    assert eval_csv.exists(), f"Run evaluate.py first — {eval_csv} not found."

    generate_all(cfg, args.fold, model, val_dl, eval_csv, out_dir, device,
                 run_gradcam = not args.no_gradcam,
                 run_umap    = not args.no_umap)


if __name__ == "__main__":
    main()
