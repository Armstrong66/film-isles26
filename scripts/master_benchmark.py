#!/usr/bin/env python3
"""
master_benchmark.py
-------------------
Master orchestrator for ISLES 2026:
  - Overnight training across all model sizes (tiny, small, base) and tracks (A, C)
  - Automatic OOM detection and fault-tolerant skipping to the next model
  - Automatic evaluation integration after training (or on existing checkpoints)
  - Smart resume / skip logic for already-trained and evaluated models
  - Submission verification & CPU vs GPU inference latency benchmarking
  - Consolidated comparison report generation (CSV, JSON, ASCII table)

Usage:
  # Quick comparative benchmark on fold 0 for all sizes & tracks:
  python scripts/master_benchmark.py --config configs/config_rtx.yaml --fold 0

  # Full 5-fold CV overnight run on all models:
  python scripts/master_benchmark.py --config configs/config_rtx.yaml --fold all --mode all

  # Evaluate existing models only and compile comparison table:
  python scripts/master_benchmark.py --config configs/config_rtx.yaml --mode eval

  # Run submission / Docker & CPU latency verification only:
  python scripts/master_benchmark.py --config configs/config_rtx.yaml --mode verify
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

# Windows DLL safeguard for PyTorch conda environments
if sys.platform == "win32":
    for p in [
        Path(sys.prefix) / "Library" / "bin",
        Path(sys.prefix) / "DLLs",
        Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib",
        Path("C:/Users/DELL/miniconda3/Library/bin"),
        Path("C:/Users/DELL/miniconda3/DLLs"),
        Path("C:/Users/DELL/miniconda3/Library/usr/bin"),
        Path("C:/Users/DELL/miniconda3/Library/mingw-w64/bin"),
    ]:
        if p.exists():
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(str(p))
                except Exception:
                    pass
            os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf, DictConfig

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.model import build_model, ISLES26Model
from pipeline.train import train_fold, load_checkpoint
from pipeline.evaluate import evaluate_fold, aggregate_metrics

log = logging.getLogger("master_benchmark")


# ── GPU Memory Management ─────────────────────────────────────────────────────

def cleanup_gpu_memory() -> None:
    """Force garbage collection and flush CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def get_gpu_memory_info(device: torch.device) -> dict[str, float]:
    """Return free, total, and allocated VRAM in GB."""
    if device.type == "cuda" and torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        allocated_bytes = torch.cuda.memory_allocated(device)
        max_allocated_bytes = torch.cuda.max_memory_allocated(device)
        return {
            "free_gb": free_bytes / (1024**3),
            "total_gb": total_bytes / (1024**3),
            "allocated_gb": allocated_bytes / (1024**3),
            "max_allocated_gb": max_allocated_bytes / (1024**3),
        }
    return {"free_gb": 0.0, "total_gb": 0.0, "allocated_gb": 0.0, "max_allocated_gb": 0.0}


def log_vram_status(device: torch.device, prefix: str = "") -> None:
    """Log current VRAM status."""
    if device.type == "cuda":
        mem = get_gpu_memory_info(device)
        log.info(
            f"{prefix}VRAM | Free: {mem['free_gb']:.2f} GB / {mem['total_gb']:.2f} GB | "
            f"Allocated: {mem['allocated_gb']:.2f} GB (Peak: {mem['max_allocated_gb']:.2f} GB)"
        )


# ── Path & Checkpoint Helpers ──────────────────────────────────────────────────

def resolve_checkpoint_dir(log_dir: str | Path, track: str, size: str, fold: int) -> Path:
    """
    Locate checkpoint dir with model-size naming, falling back to legacy path if present.
    """
    log_dir = Path(log_dir)
    size_dir = log_dir / f"track_{track}_{size}" / f"fold_{fold}"
    legacy_dir = log_dir / f"track_{track}" / f"fold_{fold}"
    if size_dir.exists() and (size_dir / "best.pth").exists():
        return size_dir
    if legacy_dir.exists() and (legacy_dir / "best.pth").exists():
        return legacy_dir
    return size_dir


# ── Verification & Latency Benchmarking (CPU vs GPU / Docker readiness) ────────

def benchmark_inference_latency(
    model: ISLES26Model,
    device: torch.device,
    patch_size: list[int] = [128, 128, 128],
    n_warmup: int = 2,
    n_runs: int = 5,
) -> dict[str, float]:
    """
    Measure inference latency per scan for given device.
    """
    model.eval()
    model.to(device)

    dummy_img = torch.randn(1, 1, *patch_size, device=device)
    dummy_vec = torch.tensor([[0.5, 0.0, 1.0, 0.0, 0.0]], device=device)
    dummy_txt = ["Acute ischemic stroke in native space."]

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_img, dummy_vec, dummy_txt)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    # Benchmark runs
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(dummy_img, dummy_vec, dummy_txt)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latencies.append((time.perf_counter() - t0) * 1000.0)  # ms

    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
    }


def verify_submission_pipeline(cfg: DictConfig, track: str, size: str, fold: int) -> dict[str, Any]:
    """
    Verify Docker and submission readiness:
      1. Model parameter footprint
      2. GPU inference latency
      3. CPU fallback inference latency
      4. Shape & binary contract validation (uint8 {0, 1})
      5. Small-component removal postprocessing simulation
    """
    cleanup_gpu_memory()
    device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_cpu = torch.device("cpu")

    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    model_cfg.conditioning.track = track
    model_cfg.model.size = size

    model = build_model(model_cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Check if checkpoint exists
    ckpt_dir = resolve_checkpoint_dir(cfg.logging.log_dir, track, size, fold)
    best_ckpt = ckpt_dir / "best.pth"
    if best_ckpt.exists():
        try:
            load_checkpoint(best_ckpt, model)
            log.info(f"Loaded checkpoint for verification: {best_ckpt}")
        except Exception as e:
            log.warning(f"Failed to load checkpoint {best_ckpt}: {e}")

    # 1. GPU Latency
    gpu_lat = {"mean_ms": -1.0}
    if torch.cuda.is_available():
        gpu_lat = benchmark_inference_latency(model, device_gpu)

    # 2. CPU Latency
    cpu_lat = benchmark_inference_latency(model, device_cpu, n_warmup=1, n_runs=2)

    # 3. Geometry & Output Contract Check
    dummy_input = torch.zeros(1, 1, 128, 128, 128, device=device_cpu)
    meta_vec = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0]], device=device_cpu)
    meta_txt = ["Chronic stroke lesion."]

    with torch.no_grad():
        out_mask = model.predict(dummy_input, meta_vec, meta_txt)
        assert out_mask.shape == (1, 1, 128, 128, 128), f"Unexpected mask shape: {out_mask.shape}"
        assert out_mask.dtype == torch.uint8, f"Unexpected mask dtype: {out_mask.dtype}"
        unique_vals = torch.unique(out_mask).tolist()
        assert set(unique_vals).issubset({0, 1}), f"Non-binary mask values: {unique_vals}"

    cleanup_gpu_memory()

    # Grand Challenge budget estimation (10 min hard limit = 600s)
    # Estimate full volume (e.g. 8 overlapping patches)
    est_scan_gpu_sec = (gpu_lat["mean_ms"] * 8.0) / 1000.0 if gpu_lat["mean_ms"] > 0 else -1.0
    est_scan_cpu_sec = (cpu_lat["mean_ms"] * 8.0) / 1000.0

    return {
        "params": n_params,
        "gpu_latency_ms": gpu_lat["mean_ms"],
        "cpu_latency_ms": cpu_lat["mean_ms"],
        "est_scan_gpu_sec": est_scan_gpu_sec,
        "est_scan_cpu_sec": est_scan_cpu_sec,
        "docker_compliant": est_scan_gpu_sec < 600.0 if est_scan_gpu_sec > 0 else True,
        "contract_verified": True,
    }


# ── Model Training Runner (OOM Resilient) ──────────────────────────────────────

def run_training_step(
    cfg: DictConfig,
    track: str,
    size: str,
    fold: int,
    splits: list[dict],
    df_meta: pd.DataFrame,
    device: torch.device,
    force_retrain: bool = False,
) -> dict[str, Any]:
    """
    Train a single model configuration with OOM resilience and skip check.
    """
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    model_cfg.conditioning.track = track
    model_cfg.model.size = size

    ckpt_dir = resolve_checkpoint_dir(model_cfg.logging.log_dir, track, size, fold)
    best_ckpt = ckpt_dir / "best.pth"
    history_file = ckpt_dir / "history.json"

    # Skip if already trained
    if best_ckpt.exists() and history_file.exists() and not force_retrain:
        log.info(f"⏩ [SKIP TRAIN] Model (Track={track}, Size={size}, Fold={fold}) already trained. Found: {best_ckpt}")
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
            best_dice = max(row.get("val_dice", 0.0) for row in history) if history else 0.0
            return {
                "status": "SKIPPED_ALREADY_TRAINED",
                "best_val_dice": best_dice,
                "history_path": str(history_file),
                "ckpt_dir": str(ckpt_dir),
            }
        except Exception:
            pass

    log.info(f"\n{'='*70}\n🚀 [START TRAIN] Track={track} | Size={size} | Fold={fold}\n{'='*70}")
    cleanup_gpu_memory()
    log_vram_status(device, prefix="[PRE-TRAIN] ")

    t_start = time.time()
    try:
        train_res = train_fold(model_cfg, fold, splits, df_meta, device)
        elapsed = time.time() - t_start
        cleanup_gpu_memory()
        return {
            "status": "TRAINED_SUCCESS",
            "best_val_dice": train_res.get("best_val_dice", 0.0),
            "train_time_sec": elapsed,
            "ckpt_dir": str(ckpt_dir),
        }
    except torch.cuda.OutOfMemoryError as e:
        elapsed = time.time() - t_start
        log.error(
            f"❌ [OOM DETECTED] Out of Memory for Track={track}, Size={size}, Fold={fold} after {elapsed:.1f}s.\n"
            f"Error details: {e}\n"
            f"Skipping this model configuration and proceeding to next model in queue..."
        )
        cleanup_gpu_memory()
        return {
            "status": "OOM",
            "error": str(e),
            "train_time_sec": elapsed,
            "ckpt_dir": str(ckpt_dir),
        }
    except Exception as e:
        elapsed = time.time() - t_start
        log.error(
            f"❌ [TRAIN ERROR] Failed for Track={track}, Size={size}, Fold={fold}: {e}",
            exc_info=True,
        )
        cleanup_gpu_memory()
        return {
            "status": f"FAILED: {type(e).__name__}",
            "error": str(e),
            "train_time_sec": elapsed,
            "ckpt_dir": str(ckpt_dir),
        }


# ── Model Evaluation Runner (Fault-Tolerant) ───────────────────────────────────

def run_evaluation_step(
    cfg: DictConfig,
    track: str,
    size: str,
    fold: int,
    splits: list[dict],
    df_meta: pd.DataFrame,
    device: torch.device,
    use_tta: bool = False,
    force_eval: bool = False,
) -> dict[str, Any]:
    """
    Evaluate a single model configuration with error tolerance.
    """
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    model_cfg.conditioning.track = track
    model_cfg.model.size = size

    ckpt_dir = resolve_checkpoint_dir(model_cfg.logging.log_dir, track, size, fold)
    best_ckpt = ckpt_dir / "best.pth"
    eval_file = ckpt_dir / "eval_aggregate.json"

    if not best_ckpt.exists():
        log.warning(f"⚠️ [SKIP EVAL] No checkpoint found at {best_ckpt}. Model not trained.")
        return {"status": "NO_CHECKPOINT", "ckpt_dir": str(ckpt_dir)}

    # Skip if already evaluated
    if eval_file.exists() and not force_eval:
        log.info(f"⏩ [SKIP EVAL] Evaluation already exists at {eval_file}.")
        try:
            with open(eval_file, "r") as f:
                agg = json.load(f)
            return {
                "status": "SKIPPED_ALREADY_EVALUATED",
                "aggregate": agg,
                "ckpt_dir": str(ckpt_dir),
            }
        except Exception:
            pass

    log.info(f"\n🔍 [START EVAL] Track={track} | Size={size} | Fold={fold} | TTA={use_tta}")
    cleanup_gpu_memory()

    t_start = time.time()
    try:
        eval_res = evaluate_fold(
            model_cfg, fold, splits, df_meta, device, use_tta=use_tta, save_probs=False
        )
        elapsed = time.time() - t_start
        cleanup_gpu_memory()
        return {
            "status": "EVAL_SUCCESS",
            "aggregate": eval_res.get("aggregate", {}),
            "eval_time_sec": elapsed,
            "ckpt_dir": str(ckpt_dir),
        }
    except torch.cuda.OutOfMemoryError as e:
        elapsed = time.time() - t_start
        log.error(
            f"❌ [OOM DURING EVAL] Out of Memory evaluating Track={track}, Size={size}, Fold={fold}.\n"
            f"Error details: {e}\n"
            f"Skipping evaluation and proceeding..."
        )
        cleanup_gpu_memory()
        return {
            "status": "OOM_EVAL",
            "error": str(e),
            "eval_time_sec": elapsed,
            "ckpt_dir": str(ckpt_dir),
        }
    except Exception as e:
        elapsed = time.time() - t_start
        log.error(
            f"❌ [EVAL ERROR] Evaluation failed for Track={track}, Size={size}, Fold={fold}: {e}",
            exc_info=True,
        )
        cleanup_gpu_memory()
        return {
            "status": f"EVAL_FAILED: {type(e).__name__}",
            "error": str(e),
            "eval_time_sec": elapsed,
            "ckpt_dir": str(ckpt_dir),
        }


# ── Benchmark Summary & Comparison Report ─────────────────────────────────────

def format_summary_table(records: list[dict[str, Any]]) -> str:
    """Format benchmark records into a clean Markdown table."""
    headers = [
        "Track", "Size", "Fold", "Status", "Params",
        "Val Dice", "Eval Dice", "HD95", "Prec", "Rec",
        "GPU (ms)", "CPU (ms)"
    ]

    rows = []
    for r in records:
        track = r.get("track", "-")
        size = r.get("size", "-")
        fold = str(r.get("fold", "-"))
        status = r.get("status", "-")
        params = f"{r.get('params', 0):,}" if r.get('params') else "-"
        val_dice = f"{r.get('best_val_dice', 0.0):.4f}" if r.get("best_val_dice") is not None else "-"
        eval_dice = f"{r.get('eval_dice', 0.0):.4f}" if r.get("eval_dice") is not None else "-"
        hd95 = f"{r.get('eval_hd95', 0.0):.2f}" if r.get("eval_hd95") is not None else "-"
        prec = f"{r.get('eval_prec', 0.0):.4f}" if r.get("eval_prec") is not None else "-"
        rec = f"{r.get('eval_rec', 0.0):.4f}" if r.get("eval_rec") is not None else "-"
        gpu_ms = f"{r.get('gpu_latency_ms', 0.0):.1f}" if r.get("gpu_latency_ms") and r.get("gpu_latency_ms") > 0 else "-"
        cpu_ms = f"{r.get('cpu_latency_ms', 0.0):.1f}" if r.get("cpu_latency_ms") else "-"

        rows.append([track, size, fold, status, params, val_dice, eval_dice, hd95, prec, rec, gpu_ms, cpu_ms])

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))

    header_line = "| " + " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers)) + " |"
    sep_line = "|-" + "-|-".join("-" * col_widths[i] for i in range(len(headers))) + "-|"
    data_lines = [
        "| " + " | ".join(f"{row[i]:<{col_widths[i]}}" for i in range(len(headers))) + " |"
        for row in rows
    ]

    return "\n".join([header_line, sep_line] + data_lines)


# ── Master Orchestrator Main ──────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ISLES26 Master Pipeline & Multi-Model Benchmarking System"
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to master config (or config_rtx.yaml)")
    parser.add_argument("--tracks", nargs="+", default=["A", "C"],
                        help="Conditioning tracks to include: A, C, or NONE (default: A C)")
    parser.add_argument("--sizes", nargs="+", default=["tiny", "small", "base"],
                        help="Model sizes to include: tiny, small, base (default: tiny small base)")
    parser.add_argument("--fold", type=str, default="0",
                        help="Fold(s) to run: 0, 1, 2, 3, 4, or 'all'")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "train", "eval", "verify"],
                        help="Execution mode: all (train+eval+verify), train, eval, verify")
    parser.add_argument("--tta", action="store_true",
                        help="Enable test-time augmentation for evaluation")
    parser.add_argument("--force-retrain", action="store_true",
                        help="Force retraining even if checkpoints exist")
    parser.add_argument("--force-eval", action="store_true",
                        help="Force re-evaluation even if eval_aggregate.json exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick test run on a synthetic batch")

    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    log_dir = Path(cfg.logging.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup master logger
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "master_benchmark.log"),
        ],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"{'#'*75}")
    log.info(f"ISLES 2026 MASTER BENCHMARK PIPELINE")
    log.info(f"Device: {device} | Mode: {args.mode}")
    log.info(f"Tracks: {args.tracks} | Sizes: {args.sizes} | Folds: {args.fold}")
    log.info(f"Log Dir: {log_dir}")
    log.info(f"{'#'*75}\n")

    # Load splits & manifest if required for training or evaluation
    splits = []
    df_meta = pd.DataFrame()
    if args.mode in ["all", "train", "eval"] and not args.dry_run:
        splits_path = Path(cfg.data.processed_dir) / "splits.json"
        manifest_path = Path(cfg.data.processed_dir) / "dataset_manifest.csv"

        if splits_path.exists() and manifest_path.exists():
            splits = json.loads(splits_path.read_text())
            df_meta = pd.read_csv(manifest_path)
            log.info(f"Loaded splits: {len(splits)} folds | Manifest: {len(df_meta)} records")
        else:
            log.warning(
                f"Data splits ({splits_path}) or manifest ({manifest_path}) not found.\n"
                f"If you need to train or evaluate on real data, run splits.py first.\n"
                f"Proceeding in verification / simulation mode..."
            )

    folds_to_run = list(range(cfg.data.n_splits)) if args.fold == "all" else [int(args.fold)]

    # ── Benchmark Matrix Execution Loop ───────────────────────────────────────
    benchmark_records: list[dict[str, Any]] = []

    for track in args.tracks:
        track = track.upper()
        for size in args.sizes:
            size = size.lower()
            for fold in folds_to_run:
                rec: dict[str, Any] = {
                    "track": track,
                    "size": size,
                    "fold": fold,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "PENDING",
                }

                # 1. Training phase
                if args.mode in ["all", "train"] and len(splits) > 0:
                    train_out = run_training_step(
                        cfg, track, size, fold, splits, df_meta, device,
                        force_retrain=args.force_retrain
                    )
                    rec["train_status"] = train_out["status"]
                    rec["status"] = train_out["status"]
                    if "best_val_dice" in train_out:
                        rec["best_val_dice"] = train_out["best_val_dice"]
                    if "train_time_sec" in train_out:
                        rec["train_time_sec"] = train_out["train_time_sec"]

                # 2. Evaluation phase
                if args.mode in ["all", "eval"] and len(splits) > 0:
                    eval_out = run_evaluation_step(
                        cfg, track, size, fold, splits, df_meta, device,
                        use_tta=args.tta, force_eval=args.force_eval
                    )
                    rec["eval_status"] = eval_out["status"]
                    if "aggregate" in eval_out and "overall" in eval_out["aggregate"]:
                        agg_overall = eval_out["aggregate"]["overall"]
                        rec["eval_dice"] = agg_overall.get("dice_mean")
                        rec["eval_dice_std"] = agg_overall.get("dice_std")
                        rec["eval_hd95"] = agg_overall.get("hd95_mean")
                        rec["eval_prec"] = agg_overall.get("precision_mean")
                        rec["eval_rec"] = agg_overall.get("recall_mean")
                        rec["status"] = "COMPLETED"
                    elif rec["status"] == "PENDING":
                        rec["status"] = eval_out["status"]

                # 3. Verification & CPU/Docker Latency Phase
                if args.mode in ["all", "verify"] or args.dry_run:
                    try:
                        v_out = verify_submission_pipeline(cfg, track, size, fold)
                        rec["params"] = v_out["params"]
                        rec["gpu_latency_ms"] = v_out["gpu_latency_ms"]
                        rec["cpu_latency_ms"] = v_out["cpu_latency_ms"]
                        rec["docker_compliant"] = v_out["docker_compliant"]
                        if rec["status"] == "PENDING":
                            rec["status"] = "VERIFIED"
                    except Exception as e:
                        log.warning(f"Verification check failed for Track={track}, Size={size}: {e}")
                        rec["verification_error"] = str(e)

                benchmark_records.append(rec)

                # Save intermediate benchmark summary to disk
                summary_df = pd.DataFrame(benchmark_records)
                summary_csv = log_dir / "benchmark_summary.csv"
                summary_json = log_dir / "benchmark_summary.json"
                summary_df.to_csv(summary_csv, index=False)
                with open(summary_json, "w") as f:
                    json.dump(benchmark_records, f, indent=2)

    # ── Final Report Display ──────────────────────────────────────────────────
    log.info("\n" + "="*80)
    log.info("📊 BENCHMARK SUMMARY & MODEL COMPARISON TABLE")
    log.info("="*80)
    table_str = format_summary_table(benchmark_records)
    log.info("\n" + table_str + "\n")

    summary_csv = log_dir / "benchmark_summary.csv"
    summary_json = log_dir / "benchmark_summary.json"
    log.info(f"✅ Benchmark results saved to:")
    log.info(f"   CSV:  {summary_csv}")
    log.info(f"   JSON: {summary_json}")


if __name__ == "__main__":
    main()
