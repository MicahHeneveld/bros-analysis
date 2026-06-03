# Bro's Analysis

This repository currently contains a small SSBU player-detection project built around extracted match frames, YOLO-format labels, and Ultralytics training outputs. (navigation / setup instructions below)

## Current Contents

- [`extract_frames.py`](/c:/bros-analysis/extract_frames.py)
  Extracts frames from a source video into image files.
- [`requirements.txt`](/c:/bros-analysis/requirements.txt)
  Python dependencies for the local environment.
- [`yolo11n.pt`](/c:/bros-analysis/yolo11n.pt)
  Downloaded Ultralytics base model checkpoint used for training.
- [`data/`](/c:/bros-analysis/data)
  Project data, including the SSBU detection dataset.
- [`runs/`](/c:/bros-analysis/runs)
  Saved Ultralytics training outputs.

## Dataset

The active dataset lives in [`data/datasets/ssbu_players`](/c:/bros-analysis/data/datasets/ssbu_players).

Current dataset counts:

- `264` training images
- `66` validation images
- `264` training label files
- `66` validation label files
- `1` unlabeled image kept outside the training split

Dataset config:

- [`data/datasets/ssbu_players/data.yaml`](/c:/bros-analysis/data/datasets/ssbu_players/data.yaml)

Dataset layout:

- [`data/datasets/ssbu_players/images/train`](/c:/bros-analysis/data/datasets/ssbu_players/images/train)
  Training images used by Ultralytics.
- [`data/datasets/ssbu_players/images/val`](/c:/bros-analysis/data/datasets/ssbu_players/images/val)
  Validation images used by Ultralytics.
- [`data/datasets/ssbu_players/images/unassigned`](/c:/bros-analysis/data/datasets/ssbu_players/images/unassigned)
  Images currently not included in train/val.
- [`data/datasets/ssbu_players/labels/train`](/c:/bros-analysis/data/datasets/ssbu_players/labels/train)
  YOLO label files for the training split.
- [`data/datasets/ssbu_players/labels/val`](/c:/bros-analysis/data/datasets/ssbu_players/labels/val)
  YOLO label files for the validation split.
- [`data/datasets/ssbu_players/train.txt`](/c:/bros-analysis/data/datasets/ssbu_players/train.txt)
  Legacy image list from an earlier dataset layout. It is still present in the repo, but the current dataset uses `data.yaml` with `images/train` and `images/val`.

## Training Outputs

Saved training runs live under [`runs/detect/runs/ssbu`](/c:/bros-analysis/runs/detect/runs/ssbu).

Current run folders:

- [`runs/detect/runs/ssbu/smash_player_test`](/c:/bros-analysis/runs/detect/runs/ssbu/smash_player_test)
  One-epoch sanity-check training run.
- [`runs/detect/runs/ssbu/smash_player_v1`](/c:/bros-analysis/runs/detect/runs/ssbu/smash_player_v1)
  Main 30-epoch training run.

Main trained weights:

- [`runs/detect/runs/ssbu/smash_player_v1/weights/best.pt`](/c:/bros-analysis/runs/detect/runs/ssbu/smash_player_v1/weights/best.pt)
- [`runs/detect/runs/ssbu/smash_player_v1/weights/last.pt`](/c:/bros-analysis/runs/detect/runs/ssbu/smash_player_v1/weights/last.pt)

These run folders also contain Ultralytics-generated artifacts such as training curves, label plots, and validation result images.

## Setup

- **Prerequisites:** Python 3.10+ and `pip` installed.
- **Create a virtual environment and install dependencies (Windows PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- **Create a virtual environment and install dependencies (Windows cmd):**

```cmd
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

- **macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- **Quick verification:**

```bash
python -c "import sys, torch; print('Python', sys.version.split()[0], 'torch', getattr(torch, '__version__', 'not installed'))"
```

- **Example: run a short training command using Ultralytics (uses the included `yolo11n.pt`):**

```powershell
yolo detect train \
  model=yolo11n.pt \
  data=data/datasets/ssbu_players/data.yaml \
  epochs=30 \
  imgsz=640 \
  batch=4 \
  workers=0 \
  project=runs/ssbu \
  name=smash_player_v1
```


## Navigation Guide

If you want to understand the project quickly, these are the main places to look:

1. Start with [`extract_frames.py`](/c:/bros-analysis/extract_frames.py) to see how source video frames are produced.
2. Open [`data/datasets/ssbu_players/data.yaml`](/c:/bros-analysis/data/datasets/ssbu_players/data.yaml) to see the active dataset definition.
3. Browse [`data/datasets/ssbu_players/images/train`](/c:/bros-analysis/data/datasets/ssbu_players/images/train) and [`data/datasets/ssbu_players/labels/train`](/c:/bros-analysis/data/datasets/ssbu_players/labels/train) to inspect the training data directly.
4. Check [`runs/detect/runs/ssbu/smash_player_v1`](/c:/bros-analysis/runs/detect/runs/ssbu/smash_player_v1) for the current trained model outputs.
5. Use [`runs/detect/runs/ssbu/smash_player_v1/weights/best.pt`](/c:/bros-analysis/runs/detect/runs/ssbu/smash_player_v1/weights/best.pt) as the primary trained checkpoint currently stored in the repo workspace.
