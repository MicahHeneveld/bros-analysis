"""Sync reviewed/edited labels from a local staging dir back to Google Drive.

Safety properties:

- Prints a full diff summary before writing anything.
- Copies only the label files that actually changed.
- Never writes into a source image or label folder of the staged dataset.
- Refuses to run while a Colab training run looks active.
- ``--dry-run`` shows the plan without touching Drive.

Usage:
    python scripts/sync_labels.py --dataset .staging/ssbu_players_colab --dry-run
    python scripts/sync_labels.py --dataset .staging/ssbu_players_colab --to-drive
"""

from __future__ import annotations

import argparse
import filecmp
import json
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from paths_config import dataset_dir, load_paths_config, require_local_root  # noqa: E402

# A run is considered active if any of these was touched very recently.
ACTIVE_RUN_WINDOW_SECONDS = 15 * 60


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sync edited labels back to Drive.")
    p.add_argument("--dataset", type=Path, required=True, help="Local staged dataset dir.")
    p.add_argument("--to-drive", action="store_true", help="Actually perform the copy.")
    p.add_argument("--dry-run", action="store_true", help="Show what would change (default).")
    p.add_argument(
        "--colab-root",
        type=Path,
        default=None,
        help="Override the Colab/Drive root (defaults to /content/drive/... or paths.yaml).",
    )
    p.add_argument("--force", action="store_true", help="Sync even if a run looks active.")
    return p.parse_args()


def resolve_drive_root(args: argparse.Namespace) -> Path:
    if args.colab_root:
        return args.colab_root.expanduser()
    colab = Path("/content/drive/MyDrive/bros-analysis")
    if colab.exists():
        return colab
    cfg = load_paths_config()
    return require_local_root(cfg)


def active_run_marker(drive_root: Path) -> Path | None:
    """Return a file that suggests training is running right now."""
    runs = drive_root / "runs"
    if not runs.exists():
        return None
    now = time.time()
    for run_dir in runs.iterdir():
        if not run_dir.is_dir():
            continue
        for name in ("results.csv", "best.pt", "last.pt"):
            candidate = run_dir / "weights" / name if name.endswith(".pt") else run_dir / name
            if candidate.exists() and (now - candidate.stat().st_mtime) < ACTIVE_RUN_WINDOW_SECONDS:
                return candidate
    return None


def count_boxes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def is_inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def collect_label_diffs(local_ds: Path, drive_ds: Path) -> list[dict]:
    diffs: list[dict] = []
    for split in ("train", "val", "test"):
        local_dir = local_ds / "labels" / split
        if not local_dir.exists():
            continue
        drive_dir = drive_ds / "labels" / split
        for local_file in sorted(local_dir.glob("*.txt")):
            drive_file = drive_dir / local_file.name
            if drive_file.exists() and filecmp.cmp(local_file, drive_file, shallow=False):
                continue
            same = drive_file.exists() and filecmp.cmp(local_file, drive_file, shallow=False)
            diffs.append(
                {
                    "split": split,
                    "name": local_file.name,
                    "local": local_file,
                    "drive": drive_file,
                    "exists_on_drive": drive_file.exists(),
                    "boxes_local": count_boxes(local_file),
                    "boxes_drive": count_boxes(drive_file),
                    "identical": same,
                }
            )
    return diffs


def main() -> int:
    args = parse_args()
    local_ds = args.dataset.expanduser().resolve()
    if not (local_ds / "labels").exists():
        print(f"ERROR: no labels/ directory in {local_ds}")
        return 2

    drive_root = resolve_drive_root(args)
    drive_ds = dataset_dir(drive_root, load_paths_config()) if drive_root.name == "bros-analysis" else drive_root / "ssbu_players_colab"
    if not drive_root.exists():
        print(f"ERROR: Drive root not found: {drive_root}")
        return 2

    print(f"Local labels : {local_ds / 'labels'}")
    print(f"Drive target : {drive_ds / 'labels'}")

    # Guard 1: destination must not be inside the source dataset.
    if is_inside(drive_ds, local_ds):
        print("ERROR: refusing to sync into a folder inside the staged dataset.")
        return 3
    labels_target = (drive_ds / "labels").resolve()
    if is_inside(labels_target, local_ds / "images") or is_inside(labels_target, local_ds / "labels"):
        print("ERROR: refusing to write labels into a source image/label folder.")
        return 3

    # Guard 2: don't fight an active training run.
    marker = active_run_marker(drive_root)
    if marker and not args.force:
        print(f"ERROR: a run looks active (recently written: {marker}).")
        print("Wait for it to finish, or pass --force if this is stale.")
        return 4

    diffs = collect_label_diffs(local_ds, drive_ds)
    added = [d for d in diffs if not d["exists_on_drive"]]
    changed = [d for d in diffs if d["exists_on_drive"]]

    print(f"\n{len(diffs)} label file(s) differ: {len(changed)} modified, {len(added)} new on Drive.")

    box_delta = 0
    for d in diffs[:40]:
        delta = d["boxes_local"] - d["boxes_drive"]
        box_delta += delta
        sign = f"{delta:+d}" if d["exists_on_drive"] else "new"
        print(f"  {d['split']:<5} {d['name']:<40} boxes {d['boxes_drive']}→{d['boxes_local']} ({sign})")
    if len(diffs) > 40:
        print(f"  ... and {len(diffs) - 40} more")

    review_state = local_ds / ".review" / "review_state.json"
    journal = local_ds / ".review" / "edits.jsonl"
    print(f"\nnet box change: {box_delta:+d}")
    print(f"review state  : {'present' if review_state.exists() else 'none'}")
    print(f"edit journal  : {'present' if journal.exists() else 'none'}")

    if args.dry_run or not args.to_drive:
        print("\nDry run only. Re-run with --to-drive to copy these files.")
        return 0

    if not diffs and not review_state.exists():
        print("\nNothing to sync.")
        return 0

    for d in diffs:
        d["drive"].parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(d["local"], d["drive"])
    print(f"\nCopied {len(diffs)} label file(s) to Drive.")

    if review_state.exists():
        export_dir = drive_root / "exports" / "reviews"
        export_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        target = export_dir / f"review_state.{stamp}.json"
        shutil.copy2(review_state, target)
        shutil.copy2(review_state, export_dir / "review_state.latest.json")
        print(f"Copied review state to {target}")
        if journal.exists():
            shutil.copy2(journal, export_dir / f"edits.{stamp}.jsonl")
            print("Copied edit journal.")

    print("\nSync complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
