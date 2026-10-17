# One-click Colab training guide

Train the SSBU `player` detector with **two Colab cells**: mount Drive, then run `scripts/train_ssbu.py` on Drive.

Shared folder: https://drive.google.com/drive/folders/12iT_dLsUOeFQ_UQBQsIWsnmD6ywvdPtl

## What lives where

| Location | Role |
|----------|------|
| Git `scripts/train_ssbu.py` | Source of truth for training logic |
| Drive `bros-analysis/scripts/train_ssbu.py` | Copy Colab actually executes |
| Git `notebooks/train_ssbu_colab.ipynb` | Thin launcher (mount + `!python …`) |

Keep the Drive copy in sync whenever you edit the git script (Drive Desktop sync does this if `local_root` points at the shared folder).

## Prerequisites (one-time)

1. Shared folder has a **shortcut under My Drive** named `bros-analysis`.
2. Google Drive for desktop is syncing that path (already configured in `configs/paths.yaml` on this machine).
3. Drive tree includes:

```text
bros-analysis/
  ssbu_players_colab/     # dataset + data.yaml
  runs/                   # weights land here
  scripts/
    train_ssbu.py         # must exist for Colab
```

### Sync the train script onto Drive

From the git repo (with `local_root` set):

```bash
mkdir -p "$HOME/Library/CloudStorage/GoogleDrive-mheneveld@csumb.edu/My Drive/bros-analysis/scripts"
cp scripts/train_ssbu.py \
  "$HOME/Library/CloudStorage/GoogleDrive-mheneveld@csumb.edu/My Drive/bros-analysis/scripts/train_ssbu.py"
```

Or copy `scripts/train_ssbu.py` into the Drive `bros-analysis/scripts/` folder in Finder.

Also copy the notebook if you want to open it from Drive:

```bash
cp notebooks/train_ssbu_colab.ipynb \
  "$HOME/Library/CloudStorage/GoogleDrive-mheneveld@csumb.edu/My Drive/bros-analysis/train_ssbu_colab.ipynb"
```

## Run training (each session)

1. Open [Google Colab](https://colab.research.google.com/).
2. **File → Upload notebook** and choose `notebooks/train_ssbu_colab.ipynb`,  
   **or** open the copy on Drive (`train_ssbu_colab.ipynb` in `bros-analysis`).
3. **Runtime → Change runtime type → GPU** (T4 is fine).
4. **Runtime → Run all** (or run the two code cells).
5. Approve Drive mount when prompted.
6. Wait for training to finish. Weights write to:

```text
My Drive/bros-analysis/runs/<RUN_NAME>/weights/best.pt
```

Default run name: `smash_player_colab_v2`.

## Common flag changes

Edit the second notebook cell, for example:

```bash
!python /content/drive/MyDrive/bros-analysis/scripts/train_ssbu.py \
  --epochs 50 \
  --name smash_player_colab_v2 \
  --batch 16 \
  --predict
```

| Flag | Meaning |
|------|---------|
| `--epochs N` | Training epochs (default 50) |
| `--name NAME` | Folder under `runs/` |
| `--batch N` | Batch size (lower if you OOM) |
| `--model yolo11n.pt` | Base checkpoint |
| `--predict` | After train, predict on val → `exports/predictions/labels` |
| `--root PATH` | Override Drive root if your shortcut path differs |
| `--skip-install` | Skip `pip install ultralytics` |

## After training

- Confirm `runs/<name>/weights/best.pt` on Drive (Finder or Drive web).
- **See where the model disagrees with your labels** — run the checkpoint over a split
  and rank the review queue by disagreement:
  `notebooks/predict_ssbu_colab.ipynb` (writes predictions to
  `exports/predictions/<run>/`). Raw detections are stored plus the deduped/capped set;
  both use the same thresholds as the app (`--post-nms-iou`, `--max-boxes`). Then, locally:
  ```bash
  python app/server.py --dataset .staging/ssbu_players_colab \
      --predictions "<drive>/exports/predictions/<run>"
  ```
  See [label_review_app.md](label_review_app.md#prediction-overlay-and-disagreement-ranking).
- For CVAT review of predictions (if you used `--predict`):

```bash
source .venv/bin/activate
python scripts/prepare_cvat_import.py \
  "<local_root>/exports/predictions/labels"
```

Full labeling loop: [cvat_workflow.md](cvat_workflow.md).

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Missing …/scripts/train_ssbu.py` | Copy script to Drive `scripts/` (see sync step) |
| `Missing …/ssbu_players_colab/data.yaml` | Shortcut not under My Drive, or wrong folder name |
| `images not found, missing path '/content/images/val'` | `data.yaml` used `path: .`, which Ultralytics resolves from `/content`. Re-run after syncing the fixed `scripts/train_ssbu.py` (it writes an absolute `path`) |
| Stuck on "Connecting to T4" / "Resuming execution" during epoch 1 | Training was reading images from Drive (~0.5 MB/s). The script now unzips `ssbu_players_colab.zip` onto the Colab disk and copies `best.pt` back to Drive after each epoch. Re-run after that script syncs. Keep the Colab tab open. |
| No GPU / very slow | Runtime → Change runtime type → GPU |
| Disconnect mid-run | Re-run; Ultralytics `exist_ok=True` writes into the same run folder — prefer a new `--name` if you need a clean run |
| OOM | Lower `--batch` (e.g. 8 or 4) |

## Why not fully unattended?

Personal Colab still needs you to start the runtime and approve Drive once per session. This guide minimizes that to **Run all**.
