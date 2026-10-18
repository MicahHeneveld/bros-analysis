# Pipeline: running the model and training a new one

Two separate jobs, with different requirements:

| Goal | Who can do it | Needs |
| :--- | :--- | :--- |
| Run the model on new video | anyone | Python, `ffmpeg`, the published checkpoint |
| Train a new checkpoint | anyone with a GPU (or patience) | reviewed labels, the split tool, compute |

Both work from a clean clone. Nothing depends on a private Google Drive folder
or on a Colab runtime.

---

## 1. Get the model

Checkpoints are published as `models-<run>` release assets, not committed to git
(a checkpoint every iteration would bloat history permanently). 5 MB, one command:

```bash
python scripts/bounty.py fetch --models
```

That installs the newest published checkpoint as `models/<run>.pt`.
`ingest_video.py` and `predict_ssbu.py` look there automatically, so you never
have to pass `--weights` in the normal case.

```bash
python scripts/bounty.py models        # what is published, and what you have
```

To pin a specific run:

```bash
python scripts/bounty.py fetch --models --run smash_player_colab_v4
```

---

## 2. Run the model on a new video

```bash
python scripts/ingest_video.py match.mp4
```

Defaults: 5 frames/second, output in `.staging/ingested/<video-name>/`. Every
sampled frame is kept — menus, results screens and single-fighter gameplay
included — and the model's boxes are written to `predictions/<split>.jsonl`
*between* empty label files.

Labels start empty on purpose. If the model's guesses were copied into the label
files, the dataset would contain its own predictions and training on it would
teach the model to agree with itself. Use `--seed-labels` only when you intend to
review every frame afterwards anyway.

To exclude frames you have to ask: `--min-players 2` keeps only frames with two
confident fighters (it drops menus, but also gameplay where one fighter is
off-screen), and `--dedupe 1.0` collapses near-identical frames. Both are off by
default and both are counted in the run report.

Useful bounds while you are getting a feel for the footage:

```bash
python scripts/ingest_video.py match.mp4 --fps 2 --start 120 --end 390
python scripts/ingest_video.py match.mp4 --max-frames 300     # a sample, not the whole match
```

### Review it

```bash
python app/server.py --dataset .staging/ingested/match \
                     --predictions .staging/ingested/match/predictions
```

Press `p` for the Both view (labels solid, predictions dashed), then `a` to adopt
boxes you agree with. Every ingest frame gets a label file, empty until somebody
labels it; what makes it trainable is your review decision, which is recorded in
`.review/review_state.json`. A frame you mark "not gameplay" is a real negative:
it stays empty and *is* trained on as background. See step 4.

### Distance between fighters

```bash
python scripts/measure_distance.py .staging/ingested/match
```

Writes `distance.csv`. Body-length units by default; the ground-plane model is
opt-in (`--ground`) because it does not validate on this footage
([why](distance_metric.md)).

---

## 3. Promote the dataset

Scratch work lives in `.staging/` (gitignored). A dataset other people can review
lives in `data/datasets/<name>/`:

```
data/datasets/<name>/
  images/<split>/...        # published as a release asset, not committed
  labels/<split>/...        # committed - this is the human work
  predictions/<split>.jsonl # committed, so disagreement is reproducible
  frames.csv                # image -> video timestamp
  manifest.json             # source, params, run stats
  data.yaml
```

```bash
python scripts/bounty.py fetch --dataset data/datasets/<name>   # images from the release
python scripts/bounty.py doctor                                 # what is ready to review
```

Publish the images with `gh release create images-<name> <bundle>.zip`, then open
review bounties:

```bash
python scripts/bounty.py create --dataset data/datasets/<name> --target 250 --dry-run
```

Full details in [contributing.md](contributing.md).

---

## 4. Train a new model

Training needs a YOLO `train`/`val` layout. Ingest produces one folder per video
(`images/<video-stem>/`), which is right for review and wrong for Ultralytics.

`split_dataset.py` bridges the two:

```bash
python scripts/split_dataset.py data/datasets/<name> --dry-run
python scripts/split_dataset.py data/datasets/<name>
```

Output defaults to `.staging/train/<name>/` with `images/train`, `images/val`,
`labels/train`, `labels/val` and a `data.yaml`. The source dataset is untouched.

Two behaviours worth knowing:

- **Whole time blocks move together.** Frames are sampled at ~5 fps, so
  consecutive frames are near duplicates. A random per-frame split would put
  frame N in train and N+1 in val, and val would effectively be memorised. Blocks
  are assigned whole, and a guard band is dropped on each side of a boundary.
  The report prints the leakage count, which should be `0`.
- **Unreviewed frames are skipped.** Ingest writes a label file for every frame,
  so file existence cannot tell a reviewed background frame from one nobody
  opened. The review store (`.review/review_state.json`) is authoritative: only
  frames marked good, bad or empty are trainable. An empty label file on a
  reviewed frame is a real negative and *is* used. Datasets with no review store
  fall back to the older rule (no `.txt` = unreviewed), and the report tells you
  how many frames are waiting.

Then train:

```bash
python scripts/train_ssbu.py --root .staging/train --dataset <name> \
    --name smash_player_colab_v5 --epochs 50 --batch 16
```

- `--device mps` for Apple Silicon, `--device cuda` for NVIDIA, omit to let
  Ultralytics choose (which picks CPU on macOS and is much slower).
- `--model yolo11n.pt` for the base model; pass a checkpoint path to fine-tune a
  previous run.
- On Colab, `--root /content/drive/MyDrive/bros-analysis` stages a copy to local
  disk first and copies checkpoints back after each epoch, so a disconnect does
  not lose the run.

Checkpoints land in `<root>/runs/<name>/weights/best.pt`.

### Evaluate before publishing

```bash
python scripts/compare_runs.py --runs smash_player_colab_v4 smash_player_colab_v5
```

This compares runs against the current human labels and reports missing boxes,
over-detection and duplicates. Prefer it to eyeballing a handful of frames: it is
the same evidence the bounty difficulty is computed from.

---

## 5. Publish the new checkpoint

```bash
python scripts/bounty.py publish-models --run-name smash_player_colab_v5
```

Creates the `models-smash_player_colab_v5` release, which makes it the checkpoint
`fetch --models` installs. Everyone else picks it up with one command, and the
next ingest runs use it. Re-publishing the same run needs `--clobber`.

---

## The loop

```
publish model ─▶ ingest new video ─▶ review labels ─▶ split ─▶ train ─▶ publish model
                        │                             ▲
                        └────── bounty board ─────────┘
```

Version convention: `smash_player_colab_v<N>` for runs, `images-<dataset>` for
frame bundles, `models-smash_player_colab_v<N>` for checkpoints. Keep the number
moving in step so a checkpoint can always be traced back to the labels it learned
from.

## Common problems

**`No model checkpoint found`** — run `python scripts/bounty.py fetch --models`.

**`No trainable frames`** — every frame is still unreviewed, or the dataset has
labels but no images (`bounty.py fetch --dataset <path>`).

**One side of the split is empty** — the dataset is too small for the block size.
Try `--block 10 --val-frac 0.15`.

**`No labels found in ... labels/train.cache`** — Ultralytics found images but no
matching label files. Check that `labels/train/<stem>.txt` exists for each
`images/train/<stem>.jpg`, and that the images are real decodable files.
