# Bro's Analysis

Super Smash Bros. Ultimate **player detection** toolkit: label in **CVAT (web)** or the
local **label review app**, store data on a shared **Google Drive** folder, train in
**Colab**.

## Canonical data home (Drive)

- **Folder:** [bros-analysis on Google Drive](https://drive.google.com/drive/folders/12iT_dLsUOeFQ_UQBQsIWsnmD6ywvdPtl?usp=sharing)
- **Folder ID:** `12iT_dLsUOeFQ_UQBQsIWsnmD6ywvdPtl`

| On Drive | Purpose |
|----------|---------|
| `ssbu_players_colab/` | Active YOLO dataset (~3.8k images, class `player`) |
| `runs/` | Training outputs and weights |
| `frames/` | Extracted frames for new CVAT jobs (create if missing) |
| `exports/cvat/` | CVAT import/export staging (create if missing) |
| `session_backup_*` / `ssbu_players_colab.zip` | Archives — not the active source of truth |

This git repo holds **scripts, docs, and the Colab notebook** only. Images and `.pt` weights stay on Drive.

## One-time setup

### 1. Add Drive shortcut + Desktop sync

1. Open the [shared folder](https://drive.google.com/drive/folders/12iT_dLsUOeFQ_UQBQsIWsnmD6ywvdPtl?usp=sharing).
2. **Organize → Add shortcut to Drive** → place under **My Drive** (keep name `bros-analysis`).
3. Install [Google Drive for desktop](https://www.google.com/drive/download/) and sync My Drive.
4. Copy the example path config and set your synced path:

```bash
cp configs/paths.example.yaml configs/paths.yaml
```

Edit `local_root` in `configs/paths.yaml` to the Desktop path, for example:

```text
/Users/<you>/Library/CloudStorage/GoogleDrive-<email>/My Drive/bros-analysis
```

### 2. Local Python env

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Verify Drive layout

```bash
python scripts/check_drive_layout.py
```

Creates `frames/` and `exports/cvat/` if missing and prints train/val counts.

## Repo layout

```text
app/
  label_io.py                # YOLO parse/serialize, atomic writes, journal, backups
  review_store.py            # review status, reasons, notes, progress
  server.py                  # stdlib HTTP server: API + static + thumbnails
  selftest.py                # 181 assertions, no test framework needed
  static/                    # index.html, style.css, app.js (canvas editor)
scripts/
  ingest_video.py            # MP4 → gated frames + pre-labels, ready to review
  extract_frames.py          # VOD → frames (raw, no gating)
  prepare_cvat_import.py     # YOLO preds → CVAT zip
  check_drive_layout.py      # Drive folder sanity check
  paths_config.py            # shared path helpers
  train_ssbu.py              # Colab/Drive training entrypoint
  predict_ssbu.py            # trained checkpoint → raw predictions (Colab)
  stage_dataset.py           # Drive dataset zip → local staging dir
  sync_labels.py             # edited labels + review log → Drive
  measure_distance.py        # per-frame fighter separation (zoom-corrected)
notebooks/
  train_ssbu_colab.ipynb     # one-click Colab launcher (mount + run train_ssbu.py)
  predict_ssbu_colab.ipynb   # one-click Colab prediction run for the overlay
configs/
  paths.example.yaml         # committed template
  paths.yaml                 # local (gitignored)
docs/
  label_review_app.md        # review/edit app usage, API, safety guarantees
  prior_art.md               # survey + license map for anything borrowed
  cvat_workflow.md           # full CVAT ↔ Drive ↔ Colab loop
  colab_one_click.md         # how to run training in two cells
data/datasets/ssbu_players/  # historical early labels only
```

## Pre-labelling a new video

One command turns a `.mp4` into a reviewable, pre-labelled frame set. It samples
at a fixed rate, drops menus/loading/results by asking the detector whether it
can see both fighters, and collapses near-duplicate frames.

```bash
# 5 fps, gate to gameplay-only, labels left empty (model boxes shown as overlay)
python scripts/ingest_video.py match.mp4

# only one match segment, pre-filling the label files
python scripts/ingest_video.py match.mp4 --start 120 --end 390 --fps 10 --seed-labels
```

Output lands in `.staging/ingested/<video>/` as a normal YOLO dataset:

```
images/<split>/<video>_frame_000123.jpg
labels/<split>/<video>_frame_000123.txt
predictions/<split>.jsonl     # the model's boxes
frames.csv                    # image_id → source frame index and time
manifest.json                 # source metadata + run stats
```

Then review it:

```bash
python app/server.py --dataset .staging/ingested/match \
    --predictions .staging/ingested/match/predictions
```

Labels are **empty by default**: the model's boxes render as the overlay and
pressing `a` in Both view adopts the ones you agree with. That keeps model
guesses out of your ground truth. `--seed-labels` pre-fills them instead, which
is faster for bulk review — but then you must review every frame before training,
or the model learns to agree with itself.

Details: [docs/video_ingest.md](docs/video_ingest.md)

## Measuring how far apart the fighters are

```bash
python scripts/measure_distance.py .staging/ingested/match
```

Writes `distance.csv` (per frame, with timestamps) and `distance.summary.json`.

SSBU's camera zooms out as fighters separate — apparent fighter height spans
6.5%–73% of frame height, an **11x** range — so distances in raw pixels are
wrong by up to an order of magnitude. The metric normalises by fighter height,
which is proportional to 1/depth, cancelling the zoom:

    distance = pixel_separation / mean_apparent_fighter_height

That gives separation in **body-lengths**, needs no calibration, and works on
any stage. Measured on 1,654 frames of unseen footage: mean 2.26, median 1.92.

It is blind to *depth* separation (two fighters front-to-back read as close). A
ground-plane inversion that would recover depth was implemented and validated
against a synthetic camera to 0.00% error, but **does not work on real SSBU
footage** (R² ≈ 0.01): apparent height is dominated by character identity and
pose rather than depth. It is available behind `--ground` and documented as
research, not measurement.

Details and evidence: [docs/distance_metric.md](docs/distance_metric.md)

## Everyday workflow

1. **Extract frames:** `python scripts/extract_frames.py path/to/vod.mp4`
2. **Label in CVAT (web)** — class `player`, export Ultralytics YOLO Detection 1.0
3. **Merge** export into Drive `ssbu_players_colab/`
4. **Train in Colab** — open `notebooks/train_ssbu_colab.ipynb`, **Runtime → Run all**
5. **Review preds in CVAT** — `python scripts/prepare_cvat_import.py <labels_dir>`

Details: [docs/cvat_workflow.md](docs/cvat_workflow.md) · [docs/colab_one_click.md](docs/colab_one_click.md)

## Reviewing and fixing labels locally

A local, zero-dependency browser app for reviewing labels and correcting boxes in place —
no accounts, no uploads, no new Python packages.

```bash
source .venv/bin/activate
python scripts/stage_dataset.py
python app/server.py --dataset .staging/ssbu_players_colab
# open http://localhost:8770
```

> Run from the repo root with the venv active. The system `python3` lacks `pyyaml`,
> so `.venv/bin/python` (or an activated venv) is required.

Mark images good / bad (+ reason) / reviewed-empty, and move, resize, add, or delete boxes
with autosave. Labels are written atomically with a recovery copy and an append-only edit
journal, so every change is reversible. Imports are refused inside a synced Drive mount —
stage locally first, then push only the changed `.txt` files back:

```bash
python scripts/sync_labels.py --dataset .staging/ssbu_players_colab --dry-run
python scripts/sync_labels.py --dataset .staging/ssbu_players_colab --to-drive
```

Full guide: [docs/label_review_app.md](docs/label_review_app.md) · safety patterns and
licensing: [docs/prior_art.md](docs/prior_art.md). Run `python app/selftest.py` to verify
the safety guarantees.

### Seeing where the model disagrees with your labels

By default the app shows ground-truth labels for `train`/`val` — not model output. To
see the model's boxes and have the strip ranked by worst disagreement, first run the
checkpoint over a split in Colab, then point the app at the result:

```bash
# Colab: notebooks/predict_ssbu_colab.ipynb  (writes raw predictions to Drive)
python app/server.py --dataset .staging/ssbu_players_colab \
    --predictions "<drive>/exports/predictions/smash_player_colab_v2"
```

Predictions draw as dashed orange boxes over your labels. Scores are recomputed against
the labels currently on disk, so fixing a box updates its score immediately. Click a
dashed box (or press `a`) to **adopt** a prediction you missed as a real label. Scoring
rules: [docs/label_review_app.md](docs/label_review_app.md#prediction-overlay-and-disagreement-ranking).

## Training

Two interchangeable paths. Both read `ssbu_players_colab/` and write to
`runs/<name>/` on Drive, so a run made either way works for everyone else.

### Colab (one-click)

1. Sync `scripts/train_ssbu.py` onto Drive `bros-analysis/scripts/` (Drive Desktop does this if the file is copied there).
2. Open `notebooks/train_ssbu_colab.ipynb` in Colab (upload or from Drive).
3. **Runtime → GPU**, then **Runtime → Run all**.
4. Weights land at Drive `runs/<RUN_NAME>/weights/best.pt`.

Step-by-step: [docs/colab_one_click.md](docs/colab_one_click.md).

### Local (Apple Silicon, terminal)

No Colab or Google account needed. One-time ~2–3 GB install into a dedicated
`.venv-train`, then:

```bash
.venv-train/bin/python scripts/train_ssbu.py \
    --root .staging --dataset ssbu_players_colab \
    --model models/yolo11n.pt --name smash_player_colab_v3 --epochs 50 --skip-install
```

Weights land at `.staging/runs/<RUN_NAME>/weights/best.pt`; copy the run to Drive
to share it. See [docs/local_training.md](docs/local_training.md) for setup,
timings, prediction, and troubleshooting.

> Either path: `--name` must be a run that does not exist yet. The script passes
> `exist_ok=True` to Ultralytics, so reusing a name overwrites that run in place.

## Project vision

See `Smash Bros Computer Vision Analytics Platform_Document.docx` for the MVP roadmap (Phase 1A detection → 1B events → analysis app). This repo pass is **pipeline/setup only**.
