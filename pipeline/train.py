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
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig
from torch.amp import GradScaler, autocast
import pandas as pd
from tqdm import tqdm

from dataset import build_dataloaders
from model import build_model
from loss import ISLES26Loss

log = logging.getLogger(__name__)


# ── Metrics ───────────────────────────────────────────────────────────────────

def batch_dice(
    logits: torch.Tensor,   # (B, 2, H, W, D)
    target: torch.Tensor,   # (B, 1, H, W, D)
    smooth: float = 1e-5,
) -> float:
    """Mean Dice over batch. Used for quick epoch-level monitoring."""
    probs  = torch.softmax(logits, dim=1)[:, 1]        # (B, H, W, D)
    target = target.squeeze(1).float()                  # (B, H, W, D)
    inter  = (probs * target).sum(dim=(1, 2, 3))
    union  = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice   = (2 * inter + smooth) / (union + smooth)
    return dice.mean().item()


# ── LR scheduler ─────────────────────────────────────────────────────────────

class PolyLRScheduler:
    """
    Polynomial decay: lr = base_lr * (1 - epoch/max_epochs)^exp
    Matches nnU-Net's default schedule exactly.
    """

    def __init__(
        self,
        optimizer:   torch.optim.Optimizer,
        initial_lr:  float,
        max_epochs:  int,
        exp:         float = 0.9,
    ) -> None:
        self.optimizer  = optimizer
        self.initial_lr = initial_lr
        self.max_epochs = max_epochs
        self.exp        = exp

    def step(self, epoch: int) -> float:
        lr = self.initial_lr * (1 - epoch / self.max_epochs) ** self.exp
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
) -> dict:
    ckpt = torch.load(path, map_location="cpu")
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
            meta_vec  = batch["meta_vec"].to(device)
            meta_text = batch["meta_text"]   # list of strings, stays on CPU

            with autocast(device_type=device.type, enabled=scaler is not None):
                logits_list = model(image, meta_vec, meta_text)
                loss, loss_dict = criterion(logits_list, mask.float())

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg.training.lr,
        weight_decay = cfg.training.weight_decay,
    )
    scheduler = PolyLRScheduler(
        optimizer,
        initial_lr = cfg.training.lr,
        max_epochs = cfg.training.epochs,
        exp        = cfg.training.poly_exp,
    )
    scaler = GradScaler() if (cfg.training.mixed_precision and device.type == "cuda") else None

    # ── Checkpoint paths ──────────────────────────────────────────────────────
    ckpt_dir  = Path(cfg.logging.log_dir) / f"track_{cfg.conditioning.track}" / f"fold_{fold}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "best.pth"
    last_ckpt = ckpt_dir / "last.pth"

    # Resume if checkpoint exists
    start_epoch  = 0
    best_val_dice = -1.0
    no_improve   = 0
    history      = []

    if last_ckpt.exists():
        metrics = load_checkpoint(last_ckpt, model, optimizer)
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
            device, is_train=True, grad_clip=cfg.training.grad_clip, fold=fold,
        )
        val_metrics = run_epoch(
            model, val_dl, criterion, None, None,
            device, is_train=False, fold=fold,
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

    log.info(f"Fold {fold} complete | best_val_dice={best_val_dice:.4f} | history: {history_path}")
    return {"fold": fold, "best_val_dice": best_val_dice, "history_path": str(history_path)}


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ISLES26 Training")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--fold",   type=str, default="0",
                        help="Fold index (0-4) or 'all'")
    parser.add_argument("--track",  type=str, default=None,
                        help="Override conditioning track: A or C")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # Track override
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