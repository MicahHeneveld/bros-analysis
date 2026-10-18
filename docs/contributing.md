# Contributing

Two ways in. Pick the one that matches how much of the pipeline you want to own.

| | **Reviewer** | **Pipeline contributor** |
| :--- | :--- | :--- |
| You do | Judge frames, fix boxes | Ingest videos, train models, change the app |
| You need | git, Docker | git, Python, ffmpeg, a GPU (for training) |
| Time to first label | ~10 minutes | ~1 hour |
| Where it runs | Your laptop | Your laptop |
| What you commit | `.txt` label files | Label files **and** code |

Everything is git-native: labels are plain YOLO `.txt` files in
`data/datasets/<dataset>/labels/`, and your contribution is a pull request. There
is no hosted service and no database, so nothing can drift out of sync with the
repository.

---

## Reviewer quick start

### 1. Clone and check your setup

```bash
git clone https://github.com/MicahHeneveld/bros-analysis.git
cd bros-analysis
gh auth login                       # needed to claim bounties and open PRs
python scripts/bounty.py doctor     # tells you exactly what is missing
python scripts/bounty.py fetch      # downloads the image bundle from releases
```

`doctor` will report missing images. Images are large and deliberately not in
git; they are published as release assets tagged `images-<dataset>`, so:

```bash
python scripts/bounty.py fetch --dataset data/datasets/ssbu_players
```

Labels *are* in git, so no download is needed for them. If `doctor` says no
release exists yet for a dataset, you are the first one there — ask for a bundle
to be published, or regenerate frames with `scripts/ingest_video.py`.

### 2. Claim a bounty

Bounties are GitHub Issues tagged `bounty`. Each one is a target — a dataset, a
filter, and a number of frames — not a fixed range of frame IDs.

```bash
python scripts/bounty.py list       # what is available, and what it is worth
python scripts/bounty.py claim 12   # comments on the issue, creates bounty/12
```

> **Claims are advisory, not locks.** There is no server to hold a lease, so two
> people *can* claim the same bounty. In practice this is harmless: the app sorts
> unreviewed frames first, so you end up on different frames. If you see someone
> already working, pick another bounty or coordinate on the issue.

### 3. Run the app

```bash
python scripts/bounty.py start 12
```

That opens `http://127.0.0.1:8770` with the bounty's filter applied. Prefer a
container?

```bash
docker compose up
```

### 4. Review

The first time you open the app it walks you through the interface. After that:

| Key | Does |
| :--- | :--- |
| `g` | every player boxed tightly, nothing extra |
| `b` | something is wrong — then `1`–`6` to say what |
| `n` | not gameplay (menus, results, transitions) |
| `a` | adopt the model's missing boxes |
| `d` | draw a new box |
| `p` | cycle Labels → Model → Both |
| `j` / `k` | next / previous frame |

The rule of thumb: **make the green boxes correct.** Dashed amber boxes are the
model's guesses; adopt the good ones, ignore the bad ones.

Some bounties say **"label from scratch"**. Those are on datasets the model has
never been checked against, so there are no green boxes yet — only dashed
guesses. The job is the same, but you are the source of truth: press `a` to
adopt, then fix anything the model got wrong. A tight, correct box beats a
loose one; two players means exactly two boxes.

If a bounty is on a dataset you have not downloaded yet, `doctor` will tell you.
Get the images with `python scripts/bounty.py fetch` before you start.

There is a **Review / Expert** switch in the top right. Review is all you need.
Expert reveals filters, the disagreement score, the edit journal and exact box
coordinates.

### 5. Submit

```bash
python scripts/bounty.py submit --push
```

This commits only label files, pushes `bounty/12`, and opens a PR containing
`Closes #12`. Points are awarded when the PR **merges**, so a re-run or a rebase
cannot pay twice.

---

## How points work

```
points = frames × difficulty
```

`difficulty` is computed when the bounty is created, from how uncertain the model
is on that slice — frames the model is unsure about are worth more than frames it
is confident about. It is written into the issue, so the award is fixed before
anyone starts and cannot be inflated afterwards.

The leaderboard lives in [`docs/LEADERBOARD.md`](LEADERBOARD.md) and is rebuilt by
`.github/workflows/bounty-points.yml` on every merged bounty PR, from git history
rather than from a scoreboard anyone could poke.

---

## Pipeline contributors

### Prerequisites

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

For training you also want a GPU-enabled environment (`torch` with CUDA/MPS) and
`ffmpeg` on your `PATH`.

### Common tasks

```bash
# Stage a dataset for review without touching the real one
python scripts/stage_dataset.py --help

# Ingest an MP4 into sampled frames + labels
python scripts/ingest_video.py --help

# Predict, review, then sync reviewed labels back
python scripts/sync_labels.py --help
```

### Changing the review app

The app is intentionally dependency-free: Python standard library on the server,
vanilla JS on the client, no build step.

```bash
python app/server.py --dataset data/datasets/ssbu_players \
                     --predictions data/datasets/ssbu_players/predictions
python app/selftest.py
```

`app/selftest.py` is the test suite — run it before opening a PR. It covers label
parsing, range requests, review state, the samples API and the bounty metadata
format.

### Opening bounties

If you have added a new dataset or a fresh set of predictions, seed the board:

```bash
python scripts/bounty.py create --dataset data/datasets/<name> --target 250
python scripts/bounty.py create --dataset data/datasets/<name> --target 150 \
    --kind unlabelled_player      # a focused bounty on the worst frames
```

Use `--dry-run` first to see the issue body and the computed difficulty.

To open several at once, pass `--count`. Each gets a distinct `[n/N]` suffix,
which is required because the duplicate guard refuses two identical titles:

```bash
for i in 1 2 3 4 5 6; do
  python scripts/bounty.py create --dataset data/datasets/ultrank_11_20 \
      --target 250 --count 6 --batch $i
done
```

The body adapts to the dataset: if every label file is empty, the bounty is
described as "label from scratch" rather than "review", because those are very
different jobs.

Publishing the images a bounty depends on:

```bash
# One zip per dataset, tagged images-<dataset>. Contributors fetch it with
# `bounty.py fetch`, so the archive needs no credentials.
cd data/datasets/ssbu_players && zip -qr /tmp/ssbu_players-images.zip images
gh release create images-ssbu_players /tmp/ssbu_players-images.zip \
    --title "ssbu_players images" --notes-file docs/release-notes-ssbu_players.md
```

Keep the notes in a file rather than `--notes "..."`: backticks inside a quoted
string are command substitutions in the shell, which silently mangles the notes.

### Promoting a staged ingest

`scripts/ingest_video.py` writes its output to `.staging/ingested/<video>`,
which is gitignored scratch space. Labels cannot be contributed from there, so
promote the dataset into `data/datasets/<name>/` first:

```bash
SRC=.staging/ingested/ultrank_11_20
DST=data/datasets/ultrank_11_20
mkdir -p "$DST"
cp "$SRC"/{data.yaml,manifest.json,frames.csv} "$DST/"
cp -R "$SRC/predictions" "$DST/"
mv  "$SRC/labels"/*.txt "$DST/labels/" 2>/dev/null   # only non-empty files
```

Two rules that matter:

- **Do not copy empty label files.** In YOLO an empty `.txt` means "this frame
  contains zero players". Committing 1654 empty files would assert something
  false about every frame and poison training. Missing label file = unreviewed.
- **Do not commit images.** `data/datasets/*/images/` is gitignored; publish a
  release instead.

---

## Ground rules

1. **Never commit images or videos.** `.gitignore` blocks `*.mp4`, `*.zip` and
   `data/datasets/*/images/` on purpose. Only labels, manifests and code belong in
   history.
2. **One bounty, one PR.** It keeps the ledger honest and the diff reviewable.
3. **Don't run the app on a cloud-synced folder** without `--allow-cloud`. Two
   machines writing the same labels will lose work.
4. **The review UI has no authentication.** Bind it to `127.0.0.1`. If you need
   to share it, put a real proxy with auth in front — do not expose the port.
