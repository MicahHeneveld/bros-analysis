"""Stage the YOLO dataset from Drive onto local disk for fast, safe review.

Google Drive streaming measured ~0.5 MB/s, which makes browsing 3,857 images
painful and makes in-place edits risky (sync conflicts). This copies the dataset
zip to a local working directory once, so the review app reads and writes plain
local files.

Usage:
    python scripts/stage_dataset.py                 # default staging dir
    python scripts/stage_dataset.py --dest ~/ssbu_staging
    python scripts/stage_dataset.py --force         # re-extract even if staged
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from paths_config import load_paths_config, require_local_root  # noqa: E402

DEFAULT_STAGING = REPO_ROOT / ".staging"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage the Drive dataset onto local disk.")
    p.add_argument("--dest", type=Path, default=DEFAULT_STAGING, help="Local staging root.")
    p.add_argument("--dataset", default=None, help="Override dataset dir name.")
    p.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Override the source root to read the dataset/zip from (for testing).",
    )
    p.add_argument("--force", action="store_true", help="Re-extract even if already staged.")
    return p.parse_args()


def looks_staged(path: Path) -> bool:
    return (path / "images").exists() and (path / "labels").exists()


def count_split(path: Path, split: str) -> int:
    d = path / "images" / split
    if not d.exists():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(1 for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts)


def main() -> int:
    args = parse_args()
    cfg = load_paths_config()
    if args.source:
        root = args.source.expanduser().resolve()
    else:
        root = require_local_root(cfg)
    ds_name = args.dataset or cfg["dataset_dirname"]
    source = (root / ds_name)
    dest = (args.dest.expanduser().resolve() / ds_name)

    print(f"Drive dataset : {source}")
    print(f"Local staging : {dest}")

    if not source.exists():
        print("ERROR: Drive dataset folder not found. Is Drive for desktop synced?")
        return 2

    if looks_staged(dest) and not args.force:
        print("\nAlready staged. Nothing to do.")
        report(dest)
        print(f"\nLaunch the review app:\n    python app/server.py --dataset {dest}")
        return 0

    zip_path = root / f"{ds_name}.zip"
    if not zip_path.exists() and not source.exists():
        print(f"ERROR: neither {zip_path.name} nor {source} exists under {root}")
        return 2
    dest.parent.mkdir(parents=True, exist_ok=True)

    if zip_path.exists():
        print(f"\nExtracting {zip_path.name} ({zip_path.stat().st_size / 1e9:.2f} GB)...")
        dest.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["unzip", "-q", "-o", str(zip_path), "-d", str(dest.parent)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"unzip failed: {result.stderr.strip()}")
            return 1
    else:
        print(f"\nNo zip found at {zip_path.name}; copying directory instead...")
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source, dest)

    # The zip may nest its contents one level down.
    if not looks_staged(dest):
        nested = [p for p in dest.glob("*/") if looks_staged(p)]
        if len(nested) == 1:
            print(f"Flattening nested dataset from {nested[0].name}...")
            inner = nested[0]
            for item in inner.iterdir():
                shutil.move(str(item), str(dest / item.name))
            shutil.rmtree(inner, ignore_errors=True)

    if not looks_staged(dest):
        print("ERROR: staging finished but images/ and labels/ were not found.")
        return 1

    print("\nStaged successfully.")
    report(dest)
    print(f"\nLaunch the review app:\n    python app/server.py --dataset {dest}")
    return 0


def report(dest: Path) -> None:
    print("\n  split   images  labels")
    for split in ("train", "val", "test"):
        imgs = count_split(dest, split)
        lbl_dir = dest / "labels" / split
        labels = sum(1 for _ in lbl_dir.glob("*.txt")) if lbl_dir.exists() else 0
        if imgs or labels:
            print(f"  {split:<7} {imgs:>6}  {labels:>6}")


if __name__ == "__main__":
    raise SystemExit(main())
