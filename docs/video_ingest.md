# Video ingest: MP4 → frame set → pre-labels

`scripts/ingest_video.py` is the front door for new footage. Give it a video and
it produces a normal YOLO dataset that the review app opens directly, with the
model's boxes already drawn.

```bash
python scripts/ingest_video.py openai/match.mp4
python app/server.py --dataset .staging/ingested/match \
    --predictions .staging/ingested/match/predictions
```

## Why it is not just `extract_frames.py`

`extract_frames.py` dumps every Nth frame to disk and leaves you to sort it out.
On a real VOD that means menus, loading screens, the results card and 20,000
near-identical frames. `ingest_video.py` adds four things:

| | `extract_frames.py` | `ingest_video.py` |
|---|---|---|
| Sampling | every Nth **frame** | fixed **frames/second** |
| Non-gameplay | kept | gated out by the detector |
| Duplicates | kept | collapsed |
| Output | loose JPEGs | reviewable dataset + timestamps |

## Sampling rate, not "every Nth frame"

`--every 15` means different things at 30 fps and 60 fps, so two videos sampled
the same way end up with different time steps. `--fps 5` always means a 0.2 s
step whatever the source does; the stride is computed from the container's
reported frame rate.

```bash
--fps 5      # 0.2s step — good default for reviewing
--fps 10     # 0.1s step — better for measuring distance/movement
--fps 30     # native — heavy, rarely needed for labelling
```

Frame filenames keep the **source** frame index
(`<video>_frame_000123.jpg`), so a frame is always traceable back to the video.
`frames.csv` records both the index and the time in seconds:

```
image_id,frame_index,time_s
match/match_frame_000123,123,4.1
match/match_frame_000126,126,4.2
```

Gaps in that time column mean the video itself jumped (a cut) or that an opt-in
filter was enabled. By default every sampled frame is present, so the column is
a dense, evenly spaced clock. Split a video into matches with it if you like.

## Every frame is kept (default)

Ingest does not throw frames away. A menu, a results screen and a frame where
one fighter is off-screen are all still data:

- Frames with no fighter are the **negatives** that teach the model what "no
  player here" looks like. A model trained only on gameplay boxes will put a box
  on every HUD element the first time it sees a full VOD.
- Gameplay frames are frequently single-fighter: a KO, a respawn, a character
  hidden behind an effect, a zoomed combo clip. Requiring two detections throws
  those away even though they are perfectly good training data.

Empty label files on these frames are intentional. They become training
negatives once a reviewer marks the frame, and `split_dataset.py` only trains on
frames the review store says a human looked at.

## Opt-in filters (they exclude frames)

The old behaviour is still available, but it is now something you ask for:

```bash
--min-players 2     # only frames with two confident fighters; drops menus AND
                    # single-fighter gameplay
--dedupe 1.0        # collapse near-identical frames (static pause screens)
```

`--min-players` counts detections that pass all of these:

- confidence above `--player-conf` (default **0.35**)
- height between `--min-height` .. `--max-height` (default **0.04–0.90**)
- taller than wide, `--min-aspect` (default **0.45**)

Both filters default to **off**, every excluded frame is counted in the report,
and `manifest.json` records the values used, so a run that filtered its input is
auditable. If more than half a gated run is rejected, the script warns: on the
UltRank test video `--min-players 2` rejected 67% of samples, and spot-checking
the rejected frames found they were single-fighter gameplay, not menus.

## Duplicate suppression

`--dedupe` compares each candidate against the **last frame actually written**
using a 32×32 grayscale signature, and skips it if the mean absolute difference
is below the threshold. Comparing against the last *written* frame rather than
the last *sampled* one means a slow camera pan keeps its frames, while a static
pause screen collapses to a single frame.

It is off by default (`--dedupe 0`). Turn it on only when disk or review time is
the constraint — near-identical frames are cheap negatives, and a paused results
screen at 5 fps costs 25 frames per five seconds.

## Labels: empty by default, on purpose

The default run leaves `labels/*.txt` empty and stores the model's boxes in
`predictions/*.jsonl`. In the review app, press `p` for **Both** view and `a` to
adopt the boxes you agree with.

This is the safe default. Pre-filling label files means a dataset that silently
contains its own predictions, and training on that teaches the model to reproduce
its own mistakes — you get a more confident model, not a better one.

`--seed-labels` writes the boxes into the label files for bulk review:

```bash
python scripts/ingest_video.py match.mp4 --seed-labels
```

Use it when reviewing hundreds of frames two keystrokes at a time is too slow.
It records `"seed_labels": true` in `manifest.json`, and every seeded image is
still `unreviewed` in the app — so review all of them before training.

## Resuming and re-running

Re-running the same command is safe. Frames already on disk are counted as
`already there` and skipped, and the previous `manifest.json` entry is preserved
rather than overwritten with zeros. That makes an interrupted long ingest simply
re-runnable, and it exits `0` when there was nothing left to do.

Because of that, an older dataset that was ingested before the filters were made
opt-in can be backfilled in place: re-run the same video **without**
`--min-players` / `--dedupe` (or with `--min-players 0`). Existing frames are
left alone and the previously excluded ones are added to `images/`,
`predictions/` and `frames.csv`.

Datasets ingested before frame names were slugified use the raw video title as
the filename prefix. Re-ingesting those needs the same prefix, or every existing
frame is duplicated under a second name:

```bash
python scripts/ingest_video.py "Top 50 ... (1080p).mp4" \
    -o data/datasets/<name> --split <name> \
    --name-prefix "Top 50 ... (1080p)"
```

Keep the prefix identical to the existing filenames: bounty slices are cut from
the sorted image list, so mixed prefixes interleave frames out of time order.

To force a rewrite:

```bash
python scripts/ingest_video.py match.mp4 --force
```

`--force` also truncates `predictions/<split>.jsonl` instead of appending.

## Output layout

```
.staging/ingested/<video>/          (or --out)
  images/<split>/                   <video>_frame_NNNNNN.jpg
  labels/<split>/                   matching .txt (empty unless --seed-labels)
  predictions/<split>.jsonl         raw + cleaned boxes, with time_s
  frames.csv                        timestamp sidecar
  manifest.json                     source metadata, params, run stats
  data.yaml                         valid YOLO config for the split
```

`--split` defaults to the video's stem, so several videos can share one dataset
directory without colliding, and the review app's split filter lists them
separately.

## Cost

Inference runs in batches of `--batch` (default 16) with an MPS cache release
between them, so peak memory stays flat regardless of video length — the same
fix that stopped the earlier out-of-memory crash. On Apple Silicon expect roughly
25–40 sampled frames/second at `--imgsz 640`; an hour of footage at `--fps 5` is
about 18,000 sampled frames, so budget 10–20 minutes.

## Limits worth knowing

- **No tracking.** Each frame is independent, so a fighter has no identity across
  frames. That matters for distance work — see the roadmap.
- **No scene-cut detection.** Cuts show up as gaps in `frames.csv` but are not
  labelled as such. Splitting a montage into matches is still manual.
- **Filters trust the detector.** If you turn on `--min-players`, a model that is
  weak on a stage or skin will drop good frames with it. That is why it is off by
  default: the audit of the UltRank test video showed 67% of samples rejected and
  the rejected frames were overwhelmingly single-fighter gameplay, not menus.
- **A bigger dataset means more review.** Keeping every frame roughly triples
  the frame count at 5 fps on interview/montage footage. That is the intended
  trade: unreviewed frames are cheap to store and expensive to be wrong about.

## Reviewing a video: the two modes

The review app (`app/server.py`) has two modes, switchable from the header. The
choice is remembered in `localStorage`, and `?mode=video` opens straight into it.

**Images** — the original frame-by-frame queue. Each kept frame is a still you
can edit, mark good/bad, and step through with `j`/`k`.

**Video** — plays the source MP4 with the dataset's boxes drawn over it. This is
the only way to judge the ingest end to end: you see what the frames look like in
motion, where sampling dropped out, and whether a box that looked fine as a still
is tracking the right fighter.

The player does not re-run inference. It replays the timeline already on disk:

- `GET /api/videos` lists source videos. They are discovered from the dataset's
  `manifest.json` (`sources[*].path`), so an ingested dataset needs no extra
  flags. `--video <file>` adds one for a hand-built dataset. Only registered
  videos are ever streamed, so a client cannot request an arbitrary file.
- `GET /api/video/<n>` streams the MP4 with HTTP `Range` support. Without it the
  browser will not scrub.
- `GET /api/video/<n>/timeline` returns every kept frame with its `time_s` and
  both box sets. One request, ~0.6 MB for 1,650 frames.

### Why boxes disappear

The overlay is driven by the nearest kept frame and hidden once the playhead is
more than one sampling interval away (bounded to 0.2–0.6 s), so it never shows a
box on a frame it was not measured on. With the default ingest (every sampled
frame kept) coverage is contiguous. On a dataset built with `--min-players` or a
low `--fps` you will see blank stretches over the gaps in `frames.csv`.

Untick **Follow sampled frames** to hold the nearest box regardless of distance;
useful when scrubbing quickly, misleading over a menu.

### Filename safety

Ingested frame names come from the video's title, which may contain `#`, spaces
and parentheses, and an unencoded `#` truncates a URL at the fragment. That broke
every image in the first ingested dataset ("failed load, unknown image"). Two
fixes, both pinned in `app/selftest.py`:

1. `slugify()` in the ingest writes `[A-Za-z0-9_-]` frame names.
2. The server percent-decodes request paths, and the frontend `encodeURIComponent`s
   each id segment, so already-ingested datasets with hostile names keep working.
