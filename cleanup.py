"""
cleanup.py
----------
Utility to clean up logs, caches, and temporary files after runs.

Usage:
  python cleanup.py [--logs] [--cache] [--checkpoints] [--all] [--dry-run]

Options:
  --logs        Remove log files (*.log)
  --cache       Remove __pycache__ directories and .pyc files
  --checkpoints Remove checkpoint files (*.pth) from log_dir
  --all         Remove all of the above
  --dry-run     Show what would be deleted without deleting
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import yaml
from omegaconf import OmegaConf

log = logging.getLogger(__name__)

# Directories and patterns to clean
CACHE_DIRS = ["__pycache__", ".pytest_cache", ".mypy_cache"]
CACHE_PATTERNS = ["*.pyc", "*.pyo"]
LOG_PATTERNS = ["*.log"]
CHECKPOINT_PATTERNS = ["*.pth", "*.ckpt"]


def get_log_dir_from_config(config_path: str) -> Path:
    """Load config and return log_dir path."""
    cfg = OmegaConf.load(config_path)
    return Path(cfg.logging.log_dir)


def find_files_to_delete(
    base_dir: Path,
    patterns: list[str],
    exclude_dirs: list[str] = None,
) -> list[Path]:
    """Find files matching patterns, excluding specified directories."""
    exclude_dirs = exclude_dirs or []
    files = []
    for pattern in patterns:
        for path in base_dir.rglob(pattern):
            if any(exclude in path.parts for exclude in exclude_dirs):
                continue
            files.append(path)
    return sorted(files)


def find_cache_dirs(base_dir: Path) -> list[Path]:
    """Find cache directories to remove."""
    cache_dirs = []
    for item in base_dir.rglob("*"):
        if item.is_dir() and item.name in CACHE_DIRS:
            # Only include if not in .git
            if ".git" not in item.parts:
                cache_dirs.append(item)
    return sorted(cache_dirs)


def cleanup_logs(log_dir: Path, dry_run: bool = False) -> int:
    """Remove log files."""
    log_paths = find_files_to_delete(log_dir, LOG_PATTERNS)
    count = len(log_paths)
    for path in log_paths:
        if dry_run:
            log.info(f"[DRY-RUN] Would delete: {path}")
        else:
            path.unlink()
            log.info(f"Deleted: {path}")
    return count


def cleanup_cache(base_dir: Path, dry_run: bool = False) -> int:
    """Remove cache directories and files."""
    count = 0

    # Remove cache directories
    cache_dirs = find_cache_dirs(base_dir)
    for d in cache_dirs:
        if dry_run:
            log.info(f"[DRY-RUN] Would delete: {d}")
        else:
            shutil.rmtree(d)
            log.info(f"Deleted: {d}")
        count += 1

    # Remove .pyc files not caught by glob
    pyc_files = find_files_to_delete(base_dir, ["*.pyc"])
    for f in pyc_files:
        if dry_run:
            log.info(f"[DRY-RUN] Would delete: {f}")
        else:
            f.unlink()
            log.info(f"Deleted: {f}")
        count += 1

    return count


def cleanup_checkpoints(log_dir: Path, dry_run: bool = False) -> int:
    """Remove checkpoint files from log_dir."""
    # Keep history.json, eval results, and summaries
    checkpoint_paths = find_files_to_delete(log_dir, CHECKPOINT_PATTERNS)
    count = len(checkpoint_paths)
    for path in checkpoint_paths:
        if dry_run:
            log.info(f"[DRY-RUN] Would delete: {path}")
        else:
            path.unlink()
            log.info(f"Deleted: {path}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean up logs, caches, and checkpoints")
    parser.add_argument("--logs", action="store_true", help="Remove log files")
    parser.add_argument("--cache", action="store_true", help="Remove cache directories")
    parser.add_argument("--checkpoints", action="store_true", help="Remove checkpoint files")
    parser.add_argument("--all", action="store_true", help="Remove all of the above")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Config file path (default: configs/config.yaml)")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.dry_run:
        log.info("=== DRY-RUN MODE - No files will be deleted ===")

    # Determine base directory (project root)
    base_dir = Path(__file__).parent.resolve()

    # Load config to get log_dir
    config_path = base_dir / args.config
    if config_path.exists():
        try:
            log_dir = get_log_dir_from_config(str(config_path))
            log.info(f"Log directory from config: {log_dir}")
        except Exception as e:
            log.warning(f"Could not load log_dir from config: {e}")
            log_dir = base_dir / "logs"
    else:
        log_dir = base_dir / "logs"

    total_cleaned = 0

    if args.all or (not any([args.logs, args.cache, args.checkpoints])):
        # Default: clean cache only (safest)
        log.info("Cleaning cache directories...")
        count = cleanup_cache(base_dir, args.dry_run)
        total_cleaned += count
        log.info(f"Cleaned {count} cache items")

        # Also clean logs if log_dir exists
        if log_dir.exists():
            log.info("Cleaning log files...")
            count = cleanup_logs(log_dir, args.dry_run)
            total_cleaned += count
            log.info(f"Cleaned {count} log files")
    else:
        if args.cache:
            log.info("Cleaning cache directories...")
            count = cleanup_cache(base_dir, args.dry_run)
            total_cleaned += count
            log.info(f"Cleaned {count} cache items")

        if args.logs:
            if log_dir.exists():
                log.info("Cleaning log files...")
                count = cleanup_logs(log_dir, args.dry_run)
                total_cleaned += count
                log.info(f"Cleaned {count} log files")
            else:
                log.warning(f"Log directory not found: {log_dir}")

        if args.checkpoints:
            if log_dir.exists():
                log.info("Cleaning checkpoint files...")
                count = cleanup_checkpoints(log_dir, args.dry_run)
                total_cleaned += count
                log.info(f"Cleaned {count} checkpoint files")
            else:
                log.warning(f"Log directory not found: {log_dir}")

    log.info(f"Cleanup complete. Total items cleaned: {total_cleaned}")


if __name__ == "__main__":
    main()
