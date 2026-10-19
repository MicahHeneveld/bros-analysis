#!/usr/bin/env python3
"""Generate the review app's filmstrip thumbnails for an existing dataset.

Ingest writes a thumb beside every frame it saves, but datasets created before
that existed (and datasets assembled by hand, e.g. a CVAT export) have none. The
review server then falls back to serving full-size frames for the strip, which
is ~90 MB per 250-frame bounty.

This script fills the gap offline. It needs OpenCV, so run it with the training
venv:

    .venv-train/bin/python scripts/make_thumbs.py data/datasets/ultrank_11_20

It writes ``<dataset>/thumbs/<split>/<stem>.jpg`` and is safely re-runnable:
existing thumbs are skipped unless ``--force`` is passed.

Why not generate on demand in the server: it is stdlib-only by design, so it has
no decoder to shrink an image with. Doing the work once here is what lets the
server serve a thumbnail as a plain file read.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
_app = _HERE.parent / "app"
if _app.is_dir():
    sys.path.insert(0, str(_app))

import thumbnails  # noqa: E402

REPO_ROOT = _HERE.parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Write filmstrip thumbnails for a dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Output: <dataset>/thumbs/<split>/<stem>.jpg",
    )
    p.add_argument(
        "dataset", type=Path, nargs="+",
        help="One or more dataset roots holding images/ (e.g. data/datasets/<name>).",
    )
    p.add_argument(
        "--width", type=int, default=thumbnails.WIDTH,
        help=f"Thumb width in pixels (default {thumbnails.WIDTH}).",
    )
    p.add_argument(
        "--quality", type=int, default=thumbnails.QUALITY,
        help=f"JPEG quality (default {thumbnails.QUALITY}).",
    )
    p.add_argument("--force", action="store_true", help="Rewrite existing thumbs.")
    p.add_argument("--dry-run", action="store_true", help="Count what would be written.")
    return p.parse_args()


def iter_images(dataset: Path) -> list[Path]:
    root = dataset / "images"
    if not root.is_dir():
        return []
    return [
        p for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]


def thumb_for(dataset: Path, image: Path) -> Path:
    """Mirror the image's position under images/ into thumbs/."""
    rel = image.relative_to(dataset / "images").with_suffix(".jpg")
    return dataset / thumbnails.DIRNAME / rel


def run_one(
    cv2,
    dataset: Path,
    width: int,
    quality: int,
    force: bool,
    dry_run: bool,
) -> tuple[int, int, int]:
    """Returns (written, skipped, bytes)."""
    images = iter_images(dataset)
    if not images:
        print(f"  no images under {dataset / 'images'}")
        return 0, 0, 0

    written = skipped = total = 0
    for image in images:
        dest = thumb_for(dataset, image)
        if dest.exists() and not force:
            skipped += 1
            continue
        if dry_run:
            written += 1
            continue
        # imread returns None for a corrupt or unsupported file; skipping keeps a
        # single bad frame from aborting a 5000-frame backfill.
        frame = cv2.imread(str(image))
        if frame is None:
            print(f"  warning: could not read {image.name}; skipped")
            continue
        size = thumbnails.write(frame, dest, width=width, quality=quality)
        if size:
            written += 1
            total += size
        else:
            print(f"  warning: could not encode a thumb for {image.name}")
    return written, skipped, total


def main() -> int:
    args = parse_args()
    try:
        import cv2
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: OpenCV is required to shrink frames ({exc}).")
        print("Run this with the training venv:  .venv-train/bin/python scripts/make_thumbs.py ...")
        return 2

    if args.width <= 0:
        print("ERROR: --width must be positive")
        return 2

    for dataset in (p.expanduser().resolve() for p in args.dataset):
        print(f"Dataset  : {dataset}")
        if not (dataset / "images").is_dir():
            print("  ERROR: no images/ directory; nothing to do")
            continue
        written, skipped, total = run_one(
            cv2, dataset, args.width, args.quality, args.force, args.dry_run
        )
        print(f"  images   : {written + skipped}")
        print(f"  written  : {written}" + (" (dry run)" if args.dry_run else ""))
        if skipped:
            print(f"  existing : {skipped} (pass --force to rewrite)")
        if total:
            print(f"  size     : {total / 1e6:.1f} MB")
        print(f"  path     : {dataset / thumbnails.DIRNAME}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
