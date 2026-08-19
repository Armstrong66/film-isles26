"""
train.py
--------
Training loop for ISLES26. Supports Track A and Track C via config.

Features:
  - 5-fold CV with per-fold checkpointing
  - Mixed precision (torch.amp)
  - Poly LR scheduler (nnU-Net default)
  - Early stopping per fold
  - Gradient clipping
  - Per-epoch train/val loss + Dice logging
  - Track B hook: pseudo-label generation stub (activated post Batch-1 baseline)

Usage:
  python train.py --config configs/config.yaml --fold 0
  python train.py --config configs/config.yaml --fold all   # all 5 folds
  python train.py --config configs/config.yaml --track C    # override track
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig
from torch.amp import GradScaler, autocast
import pandas as pd
from tqdm import tqdm

# Support running as both: python pipeline/train.py AND python -m pipeline.train
# Add project root to path if running as a script
if __name__ == "__main__":
    _project_root = Path(__file__).resolve().parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    # Use absolute imports when running as script
    from pipeline.dataset import build_dataloaders
    from pipeline.model import build_model
    from pipeline.loss import ISLES26Loss, get_boundary_weight
else:
    # Use relative imports when imported as module
    from .dataset import build_dataloaders
    from .model import build_model
    from .loss import ISLES26Loss, get_boundary_weight

log = logging.getLogger(__name__)

# Sliding window inferer for validation
try:
    from monai.inferers import SlidingWindowInferer
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    log.warning("MONAI not available - sliding window inference disabled")

VAL_INFERER = SlidingWindowInferer(
    roi_size=[128, 128, 128],
    sw_batch_size=2,
    overlap=0.5,
    mode="gaussian",
) if MONAI_AVAILABLE else None


# ── Metrics ───────────────────────────────────────────────────────────────────

def batch_dice(
    logits: torch.Tensor,   # (B, 2, H, W, D)
    target: torch.Tensor,   # (B, 1, H, W, D)
    smooth: float = 1e-5,
) -> float:
    """Hard binary Dice metric computed at 0.5 threshold with empty-patch handling."""
    probs  = torch.softmax(logits, dim=1)[:, 1]        # (B, H, W, D)
    target = target.squeeze(1).float()                  # (B, H, W, D)
    pred   = (probs > 0.5).float()
    inter  = (pred * target).sum(dim=(1, 2, 3))
    union  = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

    both_empty = (target.sum(dim=(1, 2, 3)) == 0) & (pred.sum(dim=(1, 2, 3)) == 0)
    dice   = (2.0 * inter + smooth) / (union + smooth)
    dice   = torch.where(both_empty, torch.ones_like(dice), dice)
    dice   = torch.nan_to_num(dice, nan=0.0)
    return dice.mean().item()


# ── Optimizer factory ───────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    """
    Conditioning module gets 10× lower LR than backbone.
    This prevents the FiLM gate from destabilising the backbone early in training.
    Handles both DataParallel wrapped and unwrapped models.
    """
    # Handle DataParallel wrapped model
    if isinstance(model, nn.DataParallel):
        model = model.module

    cond_params    = list(model.conditioner.parameters())
    cond_param_ids = set(id(p) for p in cond_params)
    backbone_params = [p for p in model.parameters()
                     if id(p) not in cond_param_ids]

    return torch.optim.AdamW([
        {"params": backbone_params,  "lr": cfg.training.lr},
        {"params": cond_params,      "lr": cfg.training.lr * 0.1,
         "weight_decay": 0.0},   # no WD on the small conditioning MLP
    ], weight_decay=cfg.training.weight_decay)


# ── LR scheduler ─────────────────────────────────────────────────────────────

class PolyLRScheduler:
    """
    Polynomial decay with linear warmup: lr = base_lr * (1 - epoch/max_epochs)^exp
    Linear warmup for first warmup_epochs, then poly decay.
    Matches nnU-Net's default schedule exactly.
    """

    def __init__(
        self,
        optimizer:   torch.optim.Optimizer,
        initial_lr:  float,
        max_epochs:  int,
        warmup_epochs: int = 0,
        exp:         float = 0.9,
    ) -> None:
        self.optimizer  = optimizer
        self.initial_lr = initial_lr
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.exp        = exp

    def step(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            adjusted = epoch - self.warmup_epochs
            max_adj  = self.max_epochs - self.warmup_epochs
            lr = self.initial_lr * (1 - adjusted / max_adj) ** self.exp
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(
    path:    Path,
    epoch:   int,
    model:   nn.Module,
    optim:   torch.optim.Optimizer,
    metrics: dict,
) -> None:
    # Unwrap DataParallel model for saving
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save({
        "epoch":       epoch,
        "model_state": state_dict,
        "optim_state": optim.state_dict(),
        "metrics":     metrics,
    }, path)
    log.info(f"Checkpoint saved: {path.name}  (epoch={epoch})")


def load_checkpoint(
    path:  Path,
    model: nn.Module,
    optim: torch.optim.Optimizer | None = None,
    strict: bool = True,
) -> dict:
    """Load checkpoint with optional strict mode for model size mismatches."""
    ckpt = torch.load(path, map_location="cpu")

    # If strict=False, only load matching keys
    if not strict:
        model_state = model.state_dict()
        loaded_state = ckpt["model_state"]
        # Filter to only matching keys
        matching = {k: v for k, v in loaded_state.items() if k in model_state and v.shape == model_state[k].shape}
        # Report mismatched keys
        mismatched = set(loaded_state.keys()) - set(matching.keys())
        if mismatched:
            log.info(f"Skipping {len(mismatched)} mismatched keys (likely due to model size change)")
        model.load_state_dict(matching, strict=False)
    else:
        model.load_state_dict(ckpt["model_state"])

    if optim is not None and "optim_state" in ckpt:
        optim.load_state_dict(ckpt["optim_state"])
    log.info(f"Checkpoint loaded: {path.name}  (epoch={ckpt['epoch']})")
    return ckpt["metrics"]


# ── Single epoch ──────────────────────────────────────────────────────────────

def run_epoch(
    model:      nn.Module,
    loader,
    criterion:  ISLES26Loss,
    optimizer:  torch.optim.Optimizer | None,
    scaler:     GradScaler | None,
    device:     torch.device,
    is_train:   bool,
    grad_clip:  float = 1.0,
    fold:       int = 0,
    epoch:      int = 0,
) -> dict:
    model.train() if is_train else model.eval()
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    total_loss = 0.0
    total_dice = 0.0
    n_batches  = 0

    desc = "Training" if is_train else "Validation"
    with ctx:
        for batch in tqdm(loader, desc=f"Fold {fold} {desc}", leave=False):
            image     = batch["image"].to(device)
            mask      = batch["mask"].to(device)
            # Ensure mask is binary (0 or 1) for loss computation
            mask      = (mask > 0.5).float()
            meta_vec  = batch["meta_vec"].to(device)
            meta_text = batch["meta_text"]   # list of strings, stays on CPU

            # Extract chronicity and uid for hooks (interpretability)
            chronicity = batch.get("chronicity", ["unknown"] * len(meta_text))
            uid = batch.get("uid", ["unknown"] * len(meta_text))

            # Sliding window inference for validation
            if not is_train and VAL_INFERER is not None:
                def _forward(img_patch: torch.Tensor) -> torch.Tensor:
                    b = img_patch.shape[0]
                    mv = meta_vec if meta_vec.shape[0] == b else meta_vec.repeat(b, 1)
                    mt = meta_text if len(meta_text) == b else meta_text * b
                    ch = chronicity if len(chronicity) == b else chronicity * b
                    u = uid if len(uid) == b else uid * b
                    with autocast(device_type=device.type, enabled=scaler is not None):
                        return model(img_patch, mv, mt, ch, u)[0]

                logits_list = [VAL_INFERER(inputs=image, network=_forward)]
            else:
                with autocast(device_type=device.type, enabled=scaler is not None):
                    logits_list = model(image, meta_vec, meta_text, chronicity, uid)

            # Apply current epoch to criterion for loss warmup
            criterion.boundary_w = get_boundary_weight(epoch)
            loss, loss_dict = criterion(logits_list, mask.float())

            # Check for NaN/Inf loss early to prevent gradient corruption
            if torch.isnan(loss) or torch.isinf(loss):
                log.error(f"Invalid loss detected! Skipping batch. loss={loss.item() if not torch.isnan(loss) else 'nan'} Loss dict: {loss_dict}")
                # Zero out any partial gradients before continuing
                optimizer.zero_grad(set_to_none=True)
                if scaler:
                    scaler.update()
                continue

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler:
                    scaler.scale(loss).backward()
                    # Must unscale before clipping or inspecting gradients
                    scaler.unscale_(optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                    # Check for NaN/inf after unscaling
                    if not torch.isfinite(grad_norm):
                        log.warning("Non-finite gradient norm encountered. Skipping optimizer step.")
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()
                        continue

                    scaler.step(optimizer)
                    scaler.update()

                    # Periodic diagnostic log
                    if n_batches % 50 == 0:
                        log.info(f"Grad norm: {grad_norm:.4f} (max allowed: {grad_clip})")
                else:
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    if torch.isfinite(grad_norm):
                        optimizer.step()
                    else:
                        log.warning("Non-finite gradient norm on CPU/non-scaler. Skipping step.")
                        optimizer.zero_grad(set_to_none=True)

            # Track metrics on finest-scale logits only
            dice = batch_dice(logits_list[0].detach(), mask)

            total_loss += loss_dict["total"]
            total_dice += dice
            n_batches  += 1

    assert n_batches > 0, "Dataloader returned zero batches."
    return {
        "loss": total_loss / n_batches,
        "dice": total_dice / n_batches,
    }


# ── Per-fold training ─────────────────────────────────────────────────────────

def train_fold(
    cfg:     DictConfig,
    fold:    int,
    splits:  list[dict],
    df_meta: pd.DataFrame,
    device:  torch.device,
) -> dict:
    log.info(f"{'='*60}")
    log.info(f"FOLD {fold}  |  track={cfg.conditioning.track}")
    log.info(f"{'='*60}")

    # ── Dataloaders ───────────────────────────────────────────────────────────
    train_dl, val_dl = build_dataloaders(cfg, fold, splits, df_meta)

    # ── Model, loss, optimizer ────────────────────────────────────────────────
    model     = build_model(cfg).to(device)

    # DataParallel for multi-GPU support (simple, sufficient for 2 GPUs)
    if torch.cuda.device_count() > 1:
        log.info(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = torch.nn.DataParallel(model)

    criterion = ISLES26Loss(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = PolyLRScheduler(
        optimizer,
        initial_lr = cfg.training.lr,
        max_epochs = cfg.training.epochs,
        warmup_epochs = cfg.training.warmup_epochs,
        exp        = cfg.training.poly_exp,
    )
    scaler = GradScaler() if (cfg.training.mixed_precision and device.type == "cuda") else None

    # ── Checkpoint paths ──────────────────────────────────────────────────────
    model_size = cfg.model.get("size", "base")
    ckpt_dir_size = Path(cfg.logging.log_dir) / f"track_{cfg.conditioning.track}_{model_size}" / f"fold_{fold}"
    ckpt_dir_legacy = Path(cfg.logging.log_dir) / f"track_{cfg.conditioning.track}" / f"fold_{fold}"
    if (ckpt_dir_legacy / "last.pth").exists() and not (ckpt_dir_size / "last.pth").exists():
        ckpt_dir = ckpt_dir_legacy
    else:
        ckpt_dir = ckpt_dir_size
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "best.pth"
    last_ckpt = ckpt_dir / "last.pth"

    # Resume if checkpoint exists
    start_epoch  = 0
    best_val_dice = -1.0
    no_improve   = 0
    history      = []

    if last_ckpt.exists():
        # Use strict=False to handle model size changes (e.g., tiny/small/base)
        metrics = load_checkpoint(last_ckpt, model, optimizer, strict=False)
        start_epoch   = metrics.get("epoch", 0) + 1
        best_val_dice = metrics.get("best_val_dice", -1.0)
        log.info(f"Resuming from epoch {start_epoch}")

    # ── Training loop ─────────────────────────────────────────────────────────
    patience = cfg.training.early_stopping_patience

    for epoch in range(start_epoch, cfg.training.epochs):
        t0  = time.time()
        lr  = scheduler.step(epoch)

        train_metrics = run_epoch(
            model, train_dl, criterion, optimizer, scaler,
            device, is_train=True, grad_clip=cfg.training.grad_clip, fold=fold, epoch=epoch,
        )
        val_metrics = run_epoch(
            model, val_dl, criterion, None, None,
            device, is_train=False, fold=fold, epoch=epoch,
        )

        elapsed = time.time() - t0
        log.info(
            f"Fold {fold} | Ep {epoch:04d} | "
            f"train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f} | "
            f"lr={lr:.6f} | {elapsed:.1f}s"
        )

        row = {"epoch": epoch, "lr": lr, **{f"train_{k}": v for k, v in train_metrics.items()},
               **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)

        # ── Checkpoint ────────────────────────────────────────────────────────
        save_checkpoint(last_ckpt, epoch, model, optimizer,
                        {"epoch": epoch, "best_val_dice": best_val_dice})

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            no_improve    = 0
            save_checkpoint(best_ckpt, epoch, model, optimizer,
                            {"epoch": epoch, "best_val_dice": best_val_dice})
            log.info(f"  ↑ New best val Dice: {best_val_dice:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    # ── Save history ──────────────────────────────────────────────────────────
    history_path = ckpt_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    stopped_early = no_improve >= patience
    best_epoch = -1
    for row in history:
        if row.get("val_dice") == best_val_dice:
            best_epoch = row.get("epoch", -1)

    log.info(
        f"Fold {fold} complete | best_val_dice={best_val_dice:.4f} (at epoch {best_epoch}) | "
        f"epochs_trained={len(history)}/{cfg.training.epochs} | stopped_early={stopped_early} | history: {history_path}"
    )
    return {
        "fold": fold,
        "best_val_dice": best_val_dice,
        "best_epoch": best_epoch,
        "epochs_trained": len(history),
        "total_epochs_configured": cfg.training.epochs,
        "stopped_early": stopped_early,
        "history_path": str(history_path),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ISLES26 Training")
    parser.add_argument("--config",   type=str, default="configs/config.yaml")
    parser.add_argument("--fold",     type=str, default="0",
                        help="Fold index (0-4) or 'all'")
    parser.add_argument("--track",    type=str, default=None,
                        help="Override conditioning track: A or C")
    parser.add_argument("--model-size", type=str, default=None,
                        help="Model size variant: tiny, small, or base")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # Track override
    if args.track:
        cfg = OmegaConf.merge(cfg, {"conditioning": {"track": args.track.upper()}})

    # Model size override
    if args.model_size:
        valid_sizes = ["tiny", "small", "base"]
        if args.model_size.lower() not in valid_sizes:
            raise ValueError(f"Invalid model-size '{args.model_size}'. Must be one of: {valid_sizes}")
        cfg = OmegaConf.merge(cfg, {"model": {"size": args.model_size.lower()}})

    Path(cfg.logging.log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level   = getattr(logging, cfg.logging.level),
        format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt = "%H:%M:%S",
        handlers = [
            logging.StreamHandler(),
            logging.FileHandler(
                Path(cfg.logging.log_dir) / f"train_track{cfg.conditioning.track}.log"
            ),
        ],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Load splits and metadata ───────────────────────────────────────────────
    splits_path = Path(cfg.data.processed_dir) / "splits.json"
    manifest    = Path(cfg.data.processed_dir) / "dataset_manifest.csv"

    assert splits_path.exists(), f"splits.json not found. Run splits.py first."
    assert manifest.exists(),    f"dataset_manifest.csv not found. Run splits.py first."

    splits  = json.loads(splits_path.read_text())
    df_meta = pd.read_csv(manifest)

    # ── Run folds ─────────────────────────────────────────────────────────────
    folds_to_run = list(range(cfg.data.n_splits)) if args.fold == "all" \
                   else [int(args.fold)]

    fold_results = []
    for fold in folds_to_run:
        result = train_fold(cfg, fold, splits, df_meta, device)
        fold_results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    if len(fold_results) > 1:
        mean_dice = np.mean([r["best_val_dice"] for r in fold_results])
        std_dice  = np.std( [r["best_val_dice"] for r in fold_results])
        log.info(f"\nCV Summary | mean_dice={mean_dice:.4f} ± {std_dice:.4f}")

    summary_path = Path(cfg.logging.log_dir) / f"cv_summary_track{cfg.conditioning.track}.json"
    with open(summary_path, "w") as f:
        json.dump(fold_results, f, indent=2)
    log.info(f"CV summary saved: {summary_path}")


if __name__ == "__main__":
    main()