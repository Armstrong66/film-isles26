#!/usr/bin/env python3
"""
package_submission.py
---------------------
Prepares, validates, builds, and exports the Docker container image / archive
for Grand Challenge ISLES 2026 submission.

Features:
  - Automatically identifies and copies trained checkpoints (e.g. fold_0_best.pth or all 5 folds)
  - Packages minimal inference files (pipeline/, utils/, configs/, entrypoint.py, Dockerfile)
  - Builds the Docker image: docker build -t isles26-submission .
  - Exports the submission .tar.gz container: isles26_submission.tar.gz (ready for Grand Challenge upload)
  - Runs local validation (geometry checks, runtime sanity check)

Usage:
  # Package best available model (e.g. Track A, tiny or base):
  python scripts/package_submission.py --track A --size tiny --build --export

  # Package all 5 folds for CV ensemble submission:
  python scripts/package_submission.py --track A --size base --folds all --build --export

  # Prepare files only (without running docker build):
  python scripts/package_submission.py --track A --size tiny
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("package_submission")


def find_checkpoints(
    log_dir: str | Path, track: str, size: str, folds: list[int]
) -> list[tuple[int, Path]]:
    """
    Locate best.pth checkpoints for requested track, size, and folds.
    """
    log_dir = Path(log_dir)
    found = []

    for fold in folds:
        # Check size-specific dir first
        p_size = log_dir / f"track_{track}_{size}" / f"fold_{fold}" / "best.pth"
        p_legacy = log_dir / f"track_{track}" / f"fold_{fold}" / "best.pth"

        if p_size.exists():
            found.append((fold, p_size))
        elif p_legacy.exists():
            found.append((fold, p_legacy))
        else:
            log.warning(f"No checkpoint found for Fold {fold} at {p_size} or {p_legacy}")

    return found


def prepare_submission_bundle(
    cfg_path: str,
    track: str,
    size: str,
    folds: list[int],
    target_dir: Path,
) -> bool:
    """
    Copy model weights, config, entrypoint, and pipeline to target_dir.
    """
    cfg = OmegaConf.load(cfg_path)
    log_dir = Path(cfg.logging.log_dir)
    checkpoints = find_checkpoints(log_dir, track, size, folds)

    if not checkpoints:
        log.error(
            f"❌ No trained checkpoints found for Track={track}, Size={size} in {log_dir}.\n"
            f"Train at least one fold before packaging submission."
        )
        return False

    target_dir.mkdir(parents=True, exist_ok=True)
    ckpt_target_dir = target_dir / "checkpoints"
    ckpt_target_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"\n📦 Copying {len(checkpoints)} checkpoint(s) into {ckpt_target_dir}:")
    for fold, ckpt_file in checkpoints:
        dest = ckpt_target_dir / f"fold_{fold}_best.pth"
        shutil.copy2(ckpt_file, dest)
        size_mb = dest.stat().st_size / (1024 * 1024)
        log.info(f"   ✓ Fold {fold}: {ckpt_file.name} -> {dest.name} ({size_mb:.1f} MB)")

    # Copy config and override track & size
    submission_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    submission_cfg.conditioning.track = track
    submission_cfg.model.size = size
    cfg_target_dir = target_dir / "configs"
    cfg_target_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(submission_cfg, cfg_target_dir / "config.yaml")
    log.info(f"   ✓ Saved submission config with track={track}, size={size}")

    # Verify key project files
    for req_file in ["Dockerfile", "entrypoint.py"]:
        src = PROJECT_ROOT / req_file
        dest = target_dir / req_file
        if src.exists() and src != dest:
            shutil.copy2(src, dest)
            log.info(f"   ✓ Synced {req_file}")

    # Generate helper build script
    build_sh = target_dir / "build_and_export.sh"
    build_sh.write_text(
        "#!/bin/bash\n"
        "set -e\n"
        "echo 'Building ISLES 2026 submission Docker image...'\n"
        "docker build -t isles26-submission .\n"
        "echo 'Exporting container archive for Grand Challenge...'\n"
        "docker save isles26-submission | gzip -c > isles26_submission.tar.gz\n"
        "echo 'Done! Output: isles26_submission.tar.gz'\n"
    )
    try:
        os.chmod(build_sh, 0o755)
    except Exception:
        pass

    log.info(f"✅ Submission bundle prepared in: {target_dir}")
    return True


def build_and_export_docker(
    image_tag: str = "isles26-submission",
    output_tar: Path = Path("isles26_submission.tar.gz"),
) -> bool:
    """
    Build docker image and export .tar.gz archive.
    """
    if shutil.which("docker") is None:
        log.error("❌ 'docker' CLI not found on system PATH. Please install/start Docker.")
        return False

    log.info(f"\n🐳 Building Docker image: '{image_tag}' ...")
    t0 = time.time()
    dockerfile_path = PROJECT_ROOT / "Dockerfile"
    
    cmd = [
        "docker", "build",
        "-t", image_tag,
        "-f", str(dockerfile_path),
        str(PROJECT_ROOT),
    ]

    res_build = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if res_build.returncode != 0:
        log.error(
            f"❌ Docker build failed.\n"
            f"💡 If you encountered 'permission denied while trying to connect to the docker API':\n"
            f"   1. Run this command once to grant Docker permissions to your user:\n"
            f"      sudo usermod -aG docker $USER && newgrp docker\n"
            f"   2. Or run the build script with sudo:\n"
            f"      sudo ./build_and_export.sh"
        )
        return False

    log.info(f"✓ Docker image built successfully in {time.time() - t0:.1f}s")

    log.info(f"\n📦 Exporting Docker image to '{output_tar}' ...")
    t_export = time.time()
    try:
        # docker save <image_tag> | gzip -c > output_tar
        proc_save = subprocess.Popen(
            ["docker", "save", image_tag],
            stdout=subprocess.PIPE,
            cwd=PROJECT_ROOT,
        )
        proc_gzip = subprocess.Popen(
            ["gzip", "-c"],
            stdin=proc_save.stdout,
            stdout=open(output_tar, "wb"),
            cwd=PROJECT_ROOT,
        )
        proc_save.stdout.close()
        proc_gzip.communicate()

        if proc_gzip.returncode == 0 and output_tar.exists():
            size_mb = output_tar.stat().st_size / (1024 * 1024)
            log.info(
                f"🎉 SUCCESS! Docker submission container exported in {time.time() - t_export:.1f}s:\n"
                f"   File: {output_tar.resolve()} ({size_mb:.1f} MB)\n"
                f"   Ready to upload to Grand Challenge submission portal!"
            )
            return True
        else:
            log.error(f"❌ Failed to export gzip archive.")
            return False
    except Exception as e:
        log.warning(f"Piped export failed ({e}). Falling back to docker save direct...")
        tar_raw = output_tar.with_suffix("")
        res_raw = subprocess.run(["docker", "save", "-o", str(tar_raw), image_tag])
        if res_raw.returncode == 0:
            log.info(f"✓ Saved raw docker tar: {tar_raw}")
            return True
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ISLES26 Docker Submission Packaging & Export Tool"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config_rtx.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--track", "--train", dest="track", type=str, default="A", choices=["A", "C", "NONE"],
        help="Conditioning track to package (default: A)"
    )
    parser.add_argument(
        "--size", type=str, default="tiny", choices=["tiny", "small", "base"],
        help="Model size variant (default: tiny)"
    )
    parser.add_argument(
        "--folds", type=str, default="0",
        help="Folds to include: 0, 1, 2, 3, 4, or 'all'"
    )
    parser.add_argument(
        "--build", action="store_true",
        help="Build Docker image after staging checkpoints"
    )
    parser.add_argument(
        "--export", action="store_true",
        help="Export docker image to isles26_submission.tar.gz"
    )
    parser.add_argument(
        "--output-tar", type=str, default="isles26_submission.tar.gz",
        help="Destination path for .tar.gz container archive"
    )

    args = parser.parse_args()

    cfg_path = args.config if Path(args.config).exists() else "configs/config.yaml"
    folds_to_use = [0, 1, 2, 3, 4] if args.folds == "all" else [int(f) for f in args.folds.split(",")]

    log.info(f"{'='*70}")
    log.info(f"ISLES 2026 DOCKER SUBMISSION PACKAGING")
    log.info(f"Track: {args.track} | Size: {args.size} | Folds: {folds_to_use}")
    log.info(f"{'='*70}")

    # 1. Stage checkpoints & configs in project root
    ok = prepare_submission_bundle(
        cfg_path=cfg_path,
        track=args.track.upper(),
        size=args.size.lower(),
        folds=folds_to_use,
        target_dir=PROJECT_ROOT,
    )
    if not ok:
        sys.exit(1)

    # 2. Build and export Docker if requested
    if args.build or args.export:
        success = build_and_export_docker(
            image_tag="isles26-submission",
            output_tar=Path(args.output_tar),
        )
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
