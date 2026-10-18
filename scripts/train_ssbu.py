#!/usr/bin/env python3
"""Train SSBU player detector from the shared Drive folder (Colab or local GPU).

Typical Colab usage (after mounting Drive):

  !pip -q install -U ultralytics
  !python /content/drive/MyDrive/bros-analysis/scripts/train_ssbu.py

Optional flags:

  --epochs 50
  --name smash_player_colab_v2
  --batch 16
  --predict          # smoke-test predict on val after train
  --root /content/drive/MyDrive/bros-analysis
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DRIVE_FOLDER_ID = "12iT_dLsUOeFQ_UQBQsIWsnmD6ywvdPtl"
DEFAULT_COLAB_ROOT = Path("/content/drive/MyDrive/bros-analysis")

def data_yaml_text(dataset_dir: Path) -> str:
    # Ultralytics resolves a relative `path` against the process CWD
    # (Colab's /content), not the yaml file's folder. Use an absolute path.
    return (
        f"path: {dataset_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "nc: 1\n"
        "names:\n"
        "  0: player\n"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO11 SSBU player detector.")
    p.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_COLAB_ROOT,
        help="Drive bros-analysis root (Colab My Drive shortcut path).",
    )
    p.add_argument("--model", default="yolo11n.pt", help="Ultralytics base model.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--name", default="smash_player_colab_v2", help="Run folder name under runs/.")
    p.add_argument(
        "--dataset",
        default="ssbu_players_colab",
        help="Dataset directory name under root.",
    )
    p.add_argument(
        "--predict",
        action="store_true",
        help="After training, run a val-set predict smoke test into exports/predictions.",
    )
    p.add_argument(
        "--skip-install",
        action="store_true",
        help="Do not pip-install ultralytics (assume already present).",
    )
    p.add_argument(
        "--no-stage",
        action="store_true",
        help="Train directly from Drive. Slow, and Colab often disconnects mid-epoch.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Dataloader workers. Keep low on Colab.",
    )
    p.add_argument(
        "--device",
        default=None,
        help=(
            "Compute device for training, e.g. 'mps' (Apple Silicon), 'cuda', 'cpu'. "
            "Omit to let Ultralytics auto-select."
        ),
    )
    return p.parse_args()


def on_drive(path: Path) -> bool:
    return str(path).startswith("/content/drive")


def stage_dataset(drive_dataset: Path, zip_path: Path) -> Path:
    """Copy the dataset onto the Colab VM disk.

    Reading thousands of images from Drive during training is ~0.5 MB/s and
    stalls long enough for Colab to drop the runtime. One zip extract is faster
    than copying each file.
    """
    local_root = Path("/content/datasets")
    local_ds = local_root / drive_dataset.name
    train_images = local_ds / "images" / "train"
    if train_images.exists() and any(train_images.glob("*.jpg")):
        print(f"Using staged dataset: {local_ds}", flush=True)
        return local_ds

    local_root.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        print(
            f"Unzipping {zip_path.name} onto local disk (about 2 GB, one-time)...",
            flush=True,
        )
        subprocess.check_call(
            ["unzip", "-q", "-o", str(zip_path), "-d", str(local_root)]
        )
    else:
        print(f"Copying {drive_dataset} onto local disk...", flush=True)
        if local_ds.exists():
            shutil.rmtree(local_ds)
        shutil.copytree(drive_dataset, local_ds)

    if not (local_ds / "data.yaml").exists():
        raise FileNotFoundError(f"Staged dataset is missing data.yaml: {local_ds}")
    n_train = sum(1 for _ in train_images.glob("*.jpg"))
    print(f"Staged {n_train} train images at {local_ds}", flush=True)
    return local_ds


def sync_run_to_drive(local_run: Path, drive_run: Path, trainer=None) -> None:
    """Copy the small checkpoint files so a later disconnect keeps progress."""
    drive_run.mkdir(parents=True, exist_ok=True)
    for rel in ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml"):
        src = local_run / rel
        if not src.exists():
            continue
        dst = drive_run / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.resolve() == src.resolve():
            continue
        shutil.copy2(src, dst)
    epoch = getattr(trainer, "epoch", None)
    label = f"epoch {epoch + 1}" if isinstance(epoch, int) else "end"
    print(f"Synced {label} checkpoints to {drive_run}", flush=True)


def ensure_ultralytics(skip_install: bool) -> None:
    if skip_install:
        return
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-U", "ultralytics"]
    )


def ensure_data_yaml(data_yaml: Path, dataset_dir: Path) -> None:
    desired = data_yaml_text(dataset_dir)
    text = data_yaml.read_text(encoding="utf-8") if data_yaml.exists() else ""
    if text.strip() != desired.strip():
        data_yaml.write_text(desired, encoding="utf-8")
        print(f"Updated {data_yaml} path to {dataset_dir.resolve()}")
    else:
        print(f"data.yaml OK: {data_yaml}")


def main() -> None:
    args = parse_args()
    root: Path = args.root.expanduser().resolve()
    dataset = root / args.dataset
    data_yaml = dataset / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Missing {data_yaml}\n"
            "Add a shortcut of the shared bros-analysis folder into My Drive:\n"
            f"https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}"
        )

    train_dataset = dataset
    drive_runs = root / "runs"
    train_project = drive_runs
    if on_drive(dataset) and not args.no_stage:
        train_dataset = stage_dataset(dataset, root / f"{args.dataset}.zip")
        train_project = Path("/content/runs")
        print(
            "Training on local disk. Checkpoints copy back to Drive after each epoch.",
            flush=True,
        )

    train_yaml = train_dataset / "data.yaml"
    print("ROOT     ", root)
    print("DATA_YAML", train_yaml)
    print("RUN      ", train_project / args.name)
    print("DRIVE_RUN", drive_runs / args.name)

    ensure_data_yaml(train_yaml, train_dataset)
    ensure_ultralytics(args.skip_install)

    import torch
    from ultralytics import YOLO

    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)

    local_run = train_project / args.name
    drive_run = drive_runs / args.name

    def _sync_epoch(trainer) -> None:
        sync_run_to_drive(Path(trainer.save_dir), drive_run, trainer)

    model = YOLO(args.model)
    if train_project != drive_runs:
        model.add_callback("on_fit_epoch_end", _sync_epoch)
    train_kwargs = dict(
        data=str(train_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=str(train_project),
        name=args.name,
        exist_ok=True,
        save=True,
        save_period=5,
    )
    if args.device:
        train_kwargs["device"] = args.device
    model.train(**train_kwargs)

    best = local_run / "weights" / "best.pt"
    last = local_run / "weights" / "last.pt"
    print("best:", best, "exists=", best.exists(), flush=True)
    print("last:", last, "exists=", last.exists(), flush=True)
    if train_project != drive_runs:
        sync_run_to_drive(local_run, drive_run)
        # Plots and the rest of the run folder, after training finishes.
        if local_run.exists():
            shutil.copytree(local_run, drive_run, dirs_exist_ok=True)
            print(f"Copied full run to {drive_run}", flush=True)

    if args.predict:
        if not best.exists():
            fallback = drive_runs / "smash_player_colab_v1" / "weights" / "best.pt"
            if fallback.exists():
                best = fallback
            else:
                raise FileNotFoundError("No best.pt found for predict.")
        pred_project = Path("/content/exports") if on_drive(root) else root / "exports"
        source = train_dataset / "images" / "val"
        YOLO(str(best)).predict(
            source=str(source),
            save_txt=True,
            save_conf=False,
            project=str(pred_project),
            name="predictions",
            exist_ok=True,
        )
        print("Predict labels:", pred_project / "predictions" / "labels", flush=True)


if __name__ == "__main__":
    main()
