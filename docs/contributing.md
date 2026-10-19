# Contributing

Two ways in. Pick the one that matches how much of the pipeline you want to own.

| | **Reviewer** | **Pipeline contributor** |
| :--- | :--- | :--- |
| You do | Judge frames, fix boxes | Ingest videos, train models, change the app |
| You need | git, `gh`, and Python 3 **or** Docker | git, `gh`, Python, ffmpeg, a GPU (for training) |
| Time to first label | ~10 minutes | ~1 hour |
| Where it runs | Your laptop | Your laptop |
| What you commit | `.txt` label files | Label files **and** code |

Install the tools from [the README quick start](../README.md#quick-start-for-contributors).
Docker is **optional** for reviewers: the app is standard-library Python, so
`python3 app/server.py` is equivalent to `docker compose up` if you have Python 3.9+.

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
python3 scripts/bounty.py doctor     # tells you exactly what is missing
```

`doctor` checks every dataset under `data/datasets/` and prints the one command
that fixes each gap. Images are large and deliberately not in git — they are
published as release assets tagged `images-<dataset>`, so a fresh clone looks
like this:

```
[MISS] ultrank_11_20: no images (587 MB release available)
       -> python3 scripts/bounty.py fetch --dataset data/datasets/ultrank_11_20
```

Run that. Labels *are* in git, so no download is needed for them. If `doctor`
says a dataset has no release at all, ask the maintainer to publish one rather
than regenerating frames yourself.

### 2. Claim a bounty

Bounties are GitHub Issues tagged `bounty`. Each one is a target — a dataset, a
filter, and a number of frames — not a fixed range of frame IDs.

**You do not need write access, or any repository role at all.** Bounty issues are
unlocked, so any signed-in GitHub account can comment. A small workflow
(`.github/workflows/claim-guard.yml`) then decides whether a claim counts: it
accepts accounts older than 30 days, accounts with a merged PR here, anyone on
the repo, and anyone listed in `.github/CLAIMERS`. Brand-new accounts are treated
as disposable and their claim comment is removed — that is what keeps the board
spam-free without handing out push access.

If a claim of yours disappears, your account is too new for the guard. Ask the
maintainer to add your login to `.github/CLAIMERS` — one line, and your claims
stick from then on.

```bash
python3 scripts/bounty.py list       # what is available, and what it is worth
python3 scripts/bounty.py claim 12   # comments on the issue, creates bounty/12
```

If `claim` fails with a permissions error, you are not on the repo yet — that is
the one thing to sort out before anything else.

> **Claims are advisory, not locks.** There is no server to hold a lease, so two
> people *can* claim the same bounty. In practice this is harmless: the app sorts
> unreviewed frames first, so you end up on different frames. If you see someone
> already working, pick another bounty or coordinate on the issue.

### 3. Run the app

```bash
python3 scripts/bounty.py start 12
```

That opens `http://127.0.0.1:8770/review?dataset=…&bounty_id=…`, scoped to that
bounty's frame slice. The slice is fixed: the strip shows all of that bounty's
frames and nothing else, so refreshing never hands you a different queue.

To browse a dataset without claiming anything, use the **View** button at the top
of its card on the landing screen (`/`) — that opens `/view`, which is read-only
and covers every frame plus the source video. Label mode always needs a bounty;
the **Label** button on a bounty row is the only way in.

Prefer a container?

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

`g` means "these boxes are right", so it needs at least one box. A frame with no
players is `n` (Empty), not `g`. The app refuses to record `g` on an empty frame,
and the server enforces the same rule, because a frame marked good with no labels
is a frame that no bounty can be paid for.

The rule of thumb: **make the green boxes correct.** Dashed amber boxes are the
model's guesses; adopt the good ones, ignore the bad ones.

Some bounties say **"label from scratch"**. Those are on datasets the model has
never been checked against, so there are no green boxes yet — only dashed
guesses. The job is the same, but you are the source of truth: press `a` to
adopt, then fix anything the model got wrong. A tight, correct box beats a
loose one; two players means exactly two boxes.

If a bounty is on a dataset you have not downloaded yet, `doctor` will tell you.
Get the images with `python3 scripts/bounty.py fetch` before you start.

There is a **Review / Expert** switch in the top right. Review is all you need.
Expert reveals filters, the disagreement score, the edit journal and exact box
coordinates.

### 5. Submit

```bash
python3 scripts/bounty.py submit --push
```

This commits only label files, pushes `bounty/12`, and opens a PR containing
`Closes #12`. Points are awarded when the PR **merges**, so a re-run or a rebase
cannot pay twice.

If you claimed from the landing screen rather than the CLI, you will be sitting
on `main` with no `bounty/12` branch — the browser cannot run `git checkout`.
`submit` handles that: it creates `bounty/12` first, so your label edits are
never committed to `main`.

#### Or use the Submit button

The landing screen's **Submit work** card opens the same flow. It lists every
bounty with why it can or cannot ship:

* **complete, claimed by you, has an issue** — submittable, ticked by default
* **in progress** — excluded. A PR pays the whole bounty, so shipping half a
  slice would either underpay you or land unfinished work on `main`.
* **unclaimed** — excluded. A bounty's frames can reach 100% because *other
  people* reviewed them, and a PR for that would credit nobody.
* **claimed by someone else**, or **no GitHub issue** — excluded, with the reason.

Each ticked bounty becomes its own branch and PR, and only that bounty's slice
is staged — so submitting two in a row never sweeps someone else's
half-finished labels into your PR.

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

You do not need the private Drive folder. Everything below runs from a clean
clone; the full walkthrough is in [pipeline.md](pipeline.md).

### Prerequisites

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

For training you also want a GPU-enabled environment (`torch` with CUDA/MPS),
`ffmpeg` on your `PATH`, and `ultralytics` installed. The review app itself needs
only the standard library.

### Common tasks

```bash
# Get the player detector (published release, not in git)
python scripts/bounty.py fetch --models

# Ingest an MP4 into sampled frames + model predictions
python scripts/ingest_video.py match.mp4

# Review it, then split reviewed labels into a train/val dataset
python app/server.py --dataset .staging/ingested/match \
                     --predictions .staging/ingested/match/predictions
python scripts/split_dataset.py data/datasets/<name>

# Train, evaluate, publish
python scripts/train_ssbu.py --root .staging/train --dataset <name> \
    --name smash_player_colab_v5 --epochs 50
python scripts/compare_runs.py --runs smash_player_colab_v4 smash_player_colab_v5
python scripts/bounty.py publish-models --run-name smash_player_colab_v5
```

Two rules that matter more than they look:

- **Never train on unreviewed frames.** `split_dataset.py` skips frames with no
  label file. Do not work around it by creating empty label files — that tells the
  model "no players here" about frames nobody checked.
- **Never commit model weights or image bundles.** Weights go in a `models-*`
  release, images in an `images-*` release. Git holds labels, predictions and
  metadata.

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

Bounties are created **unlocked**, because locking them would force every
labeler to have write access just to claim. Spam is handled by
`claim-guard.yml` instead: it deletes claim-shaped comments from accounts that
are too new and have no merged PR here, and ordinary discussion is left alone.
The guard does not police non-claim comments — GitHub's own spam filtering and a
maintainer's delete button cover those. Pass `--lock` if you do want an issue
locked (for example, a bounty that is being brigaded).

### Moderation notes

Two things bit us in the first hour of the board going public:

- A bot posted 22 KB comments on several bounties that quoted the point totals as
  "USDC rewards". They were deleted; at the time the issues were locked, which is
  why claiming used to require write access.
- The board briefly counted any comment containing the substring `claim` as a
  claim, so those spam comments made open bounties look taken. Claim detection now
  matches the `claim by @user` marker only.

If spam returns, `gh issue lock <n> --reason spam` and
`gh api --method DELETE /repos/<owner>/<repo>/issues/comments/<id>` are the two
repairs.

### What a role can and cannot do

There are three trust levels, and the boundary is **data vs. code** — not
labeler vs. model developer. Everyone who clones this repo runs `scripts/` and
`app/` locally with their own GitHub token, so a merged script is code executed
on a volunteer's machine. Model tooling, the review app, and automation all sit
on the same side of that line.

| Role | GitHub role | May merge |
| :--- | :--- | :--- |
| Contributor | none (forks) | label PRs, after CI says the diff is data-only |
| Developer | write | code PRs, after a code owner approves |
| Maintainer | admin | everything; bypasses the ruleset |

Enforced by `.github/CODEOWNERS` (code paths need a code owner, label paths need
nobody) and `.github/workflows/paths-guard.yml` (a PR from an account without
write access may only touch `data/datasets/*/labels/**.txt`, and label files must
be regular, non-executable, small text). Both are required because CODEOWNERS is
skipped when no owned path is touched.

`main` is protected by the "protect main" ruleset: PR required, force-push and
deletion blocked, stale approvals dismissed. One follow-up is still outstanding:
add `paths guard` as a **required status check** once it has reported on a real
PR, so the check is selected by name rather than guessed.

### Why labels can skip review

A label file is four numbers per box. The cheapest attack on one is bad data,
which the disagreement overlay and the next reviewer catch. A script is
arbitrary code. `bounty.py submit` therefore stages only
`data/datasets/*/labels/**`, and the guard rejects anything wider — including a
symlink named `*.txt`, which would otherwise let a "label" write somewhere else
entirely when the app saves it.

Publishing the images a bounty depends on:

```bash
# One zip per dataset, tagged images-<dataset>. Contributors fetch it with
# `bounty.py fetch`, so the archive needs no credentials. Include thumbs/ when
# it exists: without it the review filmstrip downloads full-size frames.
.venv-train/bin/python scripts/make_thumbs.py data/datasets/ssbu_players
cd data/datasets/ssbu_players
zip -qr /tmp/ssbu_players-images.zip images
if [ -d thumbs ]; then zip -qr /tmp/ssbu_players-images.zip thumbs; fi
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
