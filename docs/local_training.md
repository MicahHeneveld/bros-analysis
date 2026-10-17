# Local training (Apple Silicon)

Train the detector on your own machine from a terminal, with no Colab and no
Google account. This is an **alternative** to
[Colab training](colab_one_click.md), not a replacement — both read the same
dataset and write to the same `runs/` layout on Drive, so either can be used by
any contributor.

Use this when you want to iterate without a browser, work offline, or avoid
Google's runtime limits and disconnects.

## When to prefer which

| | Local (Apple Silicon) | Colab |
| --- | --- | --- |
| Interface | Terminal | Browser |
| GPU | Apple MPS | NVIDIA T4 |
| Cost | Your electricity | Free tier, session limits |
| Interruptions | None | Disconnects, daily quotas |
| Setup | One-time ~2–3 GB install | None |
| Long runs | Unbounded | Risk of disconnect mid-run |

Both produce an identical `runs/<name>/` folder, so results are interchangeable.

## Requirements

- Apple Silicon Mac (M1 or newer). MPS is the Apple GPU backend.
- ~3 GB free disk for the Python environment, plus ~1 GB per run.
- Python 3.10+ from Homebrew (`brew install python@3.12`).

**Do not reuse the app's `.venv`.** It may be an older Python (Xcode's 3.9,
for example), which caps the PyTorch version. Use a dedicated venv.

## One-time setup

Run from the repo root:

```bash
# Dedicated env — leaves .venv (the review app) untouched.
/opt/homebrew/bin/python3.12 -m venv .venv-train

# PyTorch for Apple Silicon (MPS-enabled), plus Ultralytics.
.venv-train/bin/pip install --upgrade pip
.venv-train/bin/pip install torch torchvision ultralytics

# Base weights to fine-tune. Any yolo11n.pt works.
mkdir -p models
cp "<drive>/bros-analysis/session_backup_20260723_211401/yolo11n.pt" models/
```

Check MPS is live:

```bash
.venv-train/bin/python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

Expect `True`. If it prints `False`, training still works on CPU, just slower —
pass `--device cpu` and lower `--batch`.

## Train

Stage the dataset locally first (see [Dataset staging](#dataset-staging)) so
`.staging/ssbu_players_colab/` exists, then:

```bash
.venv-train/bin/python scripts/train_ssbu.py \
    --root .staging \
    --dataset ssbu_players_colab \
    --model models/yolo11n.pt \
    --name smash_player_colab_v3 \
    --epochs 50 \
    --skip-install
```

`--root .staging` makes the script treat `.staging/` as the Drive root. Because
that path is not on Google Drive, it skips the zip-staging step and trains
straight from the local images.

Output: `.staging/runs/smash_player_colab_v3/weights/best.pt`

### `--name` must be new

`train_ssbu.py` calls `model.train(..., exist_ok=True)`. Reusing an existing
`--name` **overwrites that run's weights and `results.csv` in place**. Pick a
fresh name per run (`v3`, `v4`, …).

### Timings

Measured on an M5 (10 cores, 24 GB), YOLO11n, `imgsz 640`, `batch 16`,
3,085 train images:

- Benchmark first — train 2 epochs and read the reported seconds per epoch
  before committing to a long run.
- Expect roughly 45–75 minutes for 50 epochs.

## Predict

The review app reads `<split>.jsonl`, not YOLO `.txt` labels, so use
`predict_ssbu.py`:

```bash
.venv-train/bin/python scripts/predict_ssbu.py \
    --root .staging \
    --run-name smash_player_colab_v3 \
    --split all
```

`--split all` writes both `train.jsonl` and `val.jsonl`, which also gives the
train split model labels in the reviewer.

Note: `train_ssbu.py --predict` writes YOLO `.txt` labels into the Colab VM's
local disk. That is **not** what the reviewer reads, and it does not survive the
runtime. Always use `predict_ssbu.py` for review data.

## Publish the run so Colab users can use it

Local runs are only useful to others once they reach Drive. Copy the run and its
predictions up, keeping the standard layout:

```bash
# Weights + results, so a Colab user can resume or predict from it.
cp -R .staging/runs/smash_player_colab_v3 "<drive>/bros-analysis/runs/"

# Predictions, so a Colab user can open them in the reviewer.
mkdir -p "<drive>/bros-analysis/exports/predictions"
cp -R .staging/exports/predictions/smash_player_colab_v3 \
      "<drive>/bros-analysis/exports/predictions/"
```

Then point the review app at it:

```bash
python app/server.py --dataset .staging/ssbu_players_colab \
    --predictions "<drive>/bros-analysis/exports/predictions/smash_player_colab_v3"
```

## Dataset staging

Training reads `.staging/ssbu_players_colab/` (images, labels, `data.yaml`).
Produce it once from Drive:

```bash
python scripts/stage_dataset.py
```

### `data.yaml` gets rewritten

`train_ssbu.py` rewrites `data.yaml` with an **absolute** path, because
Ultralytics resolves a relative `path` against the process CWD rather than the
yaml's own folder. After a local run, `.staging/ssbu_players_colab/data.yaml`
will point at your machine.

That is harmless for training, but it makes the file differ from Drive's copy.
Restore the canonical version before syncing labels:

```bash
cp .staging/data.yaml.drive-canonical.bak .staging/ssbu_players_colab/data.yaml
```

`scripts/sync_labels.py` only moves `labels/`, so `data.yaml` is not normally
synced — this only matters if you copy the dataset folder as a whole.

## Troubleshooting

- **`torch.backends.mps.is_available()` is `False`** — wrong Python or an
  x86 build. Confirm `.venv-train/bin/python -V` is 3.10+ and
  `uname -m` is `arm64`.
- **Out of memory** — lower `--batch` to 8, then 4.
- **Dataloader errors** — pass `--workers 0`.
- **`ModuleNotFoundError: predictions`** — `predict_ssbu.py` imports it from
  `app/` in a checkout and from `scripts/` on Drive. It tries both. If you moved
  the file, either place a copy beside it or set `PYTHONPATH=app`.
