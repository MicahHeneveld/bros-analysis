# CVAT ↔ Drive ↔ Colab workflow

Shared Drive folder (canonical data home):

https://drive.google.com/drive/folders/12iT_dLsUOeFQ_UQBQsIWsnmD6ywvdPtl

Folder ID: `12iT_dLsUOeFQ_UQBQsIWsnmD6ywvdPtl`

## Roles

| Tool | Role |
|------|------|
| **CVAT (web)** | Manual labeling, bounding boxes, annotation export/import |
| **Google Drive** | Dataset, weights, frames, CVAT zip staging |
| **Colab** | YOLO training / inference (GPU) |
| **This git repo** | Scripts, cleaned notebook, docs |

## Label taxonomy (Phase 1A)

- Single class: `player`
- Box the character body (not stage props / HUD)
- Empty label files are OK for frames with no visible players

## End-to-end loop

### 1. Extract frames (local)

```bash
python scripts/extract_frames.py path/to/vod.mp4 -n 15
```

Writes to `<local_root>/frames/<video_stem>/` when `configs/paths.yaml` has `local_root` set; otherwise `data/frames/<stem>/`.

Upload or sync those frames into CVAT as a new task/job.

### 2. Label in CVAT (web)

1. Create/open a task with your frames.
2. Draw `player` boxes.
3. Export annotations as **Ultralytics YOLO Detection 1.0** (same format as historical jobs `4066630`, `4084543`, `4129982`).

### 3. Merge export into Drive dataset

Active dataset on Drive: `ssbu_players_colab/`

```text
ssbu_players_colab/
  images/train|val/
  labels/train|val/
  data.yaml
  split_manifest.json
```

Copy new images into `images/train` (or `val`) and matching `.txt` labels into `labels/...`. `scripts/train_ssbu.py` rewrites `data.yaml` before each Colab run so `path` is the absolute dataset folder (Ultralytics treats `path: .` as the process working directory, which on Colab is `/content`). Class list stays:

```yaml
nc: 1
names:
  0: player
```

Update `split_manifest.json` when you change the split (optional but useful for audits).

### 4. Train in Colab

1. Ensure the shared folder has a **shortcut under My Drive**.
2. Ensure Drive has `scripts/train_ssbu.py` (see [colab_one_click.md](colab_one_click.md)).
3. Open [`notebooks/train_ssbu_colab.ipynb`](../notebooks/train_ssbu_colab.ipynb) in Colab → **Runtime → GPU** → **Run all**.

Weights land in Drive `runs/<RUN_NAME>/weights/best.pt`.

### 5. Prediction review back into CVAT

1. Optional notebook predict cell writes labels under Drive `exports/predictions/labels`.
2. Locally (with Drive Desktop synced):

```bash
python scripts/prepare_cvat_import.py \
  "<local_root>/exports/predictions/labels" \
  --format yolo11
```

3. Import the resulting zip into CVAT for review/correction.
4. Re-export corrected labels and merge back into `ssbu_players_colab` before the next train.

## Sanity check

```bash
python scripts/check_drive_layout.py
# or, before paths.yaml is set:
python scripts/check_drive_layout.py --root "/path/to/My Drive/bros-analysis"
```

## What not to use as active data

- `session_backup_20260723_211401/` — archive only
- `ssbu_players_colab.zip` — redundant zip of the dataset
- Repo `data/datasets/ssbu_players/labels` — historical early subset only
