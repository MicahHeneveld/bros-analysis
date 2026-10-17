"""Verify the shared Drive folder layout and report dataset counts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths_config import (
    DRIVE_URL,
    dataset_dir,
    exports_dir,
    frames_dir,
    load_paths_config,
    local_root,
    runs_dir,
)

REQUIRED_CHILDREN = (
    "ssbu_players_colab",
    "runs",
)


def count_glob(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.glob(pattern))


def check_root(root: Path, cfg: dict) -> int:
    print(f"Drive folder URL: {cfg.get('drive_url', DRIVE_URL)}")
    print(f"Drive folder ID:  {cfg.get('drive_folder_id')}")
    print(f"Checking root:    {root}")
    print()

    if not root.exists():
        print(f"ERROR: root does not exist: {root}")
        print("Add the shared-folder shortcut to My Drive and sync with Drive for desktop.")
        return 1

    errors = 0
    for name in REQUIRED_CHILDREN:
        path = root / name
        status = "ok" if path.exists() else "MISSING"
        if status != "ok":
            errors += 1
        print(f"  [{status}] {name}/")

    # Optional / create-if-missing staging dirs
    for label, path in (
        ("frames", frames_dir(root, cfg)),
        ("exports/cvat", exports_dir(root, cfg)),
    ):
        if path.exists():
            print(f"  [ok] {label}/")
        else:
            path.mkdir(parents=True, exist_ok=True)
            print(f"  [created] {label}/")

    ds = dataset_dir(root, cfg)
    data_yaml = ds / "data.yaml"
    if data_yaml.exists():
        print(f"  [ok] {cfg['dataset_dirname']}/data.yaml")
        text = data_yaml.read_text(encoding="utf-8")
        if "/content/" in text:
            print(
                "  [warn] data.yaml still references /content/... ; "
                "prefer relative paths (path: .)"
            )
    else:
        print(f"  [MISSING] {cfg['dataset_dirname']}/data.yaml")
        errors += 1

    train_imgs = count_glob(ds / "images" / "train", "*.jpg")
    val_imgs = count_glob(ds / "images" / "val", "*.jpg")
    train_lbl = count_glob(ds / "labels" / "train", "*.txt")
    val_lbl = count_glob(ds / "labels" / "val", "*.txt")

    print()
    print("Dataset counts:")
    print(f"  train images/labels: {train_imgs} / {train_lbl}")
    print(f"  val images/labels:   {val_imgs} / {val_lbl}")

    runs = runs_dir(root, cfg)
    best = runs / "smash_player_colab_v1" / "weights" / "best.pt"
    if best.exists():
        print(f"  [ok] weights: {best}")
    else:
        # weights may be cloud-only / not hydrated locally
        print(f"  [?] best.pt not found locally at {best}")
        print("      (may still exist on Drive; open in browser or wait for sync)")

    print()
    if errors:
        print(f"FAILED with {errors} missing required item(s).")
        return 1
    print("Layout OK.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check the shared bros-analysis Drive folder layout."
    )
    parser.add_argument(
        "--root",
        help="Override local Drive root (default: configs/paths.yaml local_root).",
    )
    args = parser.parse_args()
    cfg = load_paths_config()

    if args.root:
        root = Path(args.root).expanduser()
    else:
        root = local_root(cfg)
        if root is None:
            print("configs/paths.yaml has an empty local_root.")
            print("Pass --root explicitly, or set local_root after syncing Drive.")
            print(f"Shared folder: {DRIVE_URL}")
            sys.exit(2)

    sys.exit(check_root(root, cfg))


if __name__ == "__main__":
    main()
