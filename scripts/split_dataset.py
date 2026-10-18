#!/usr/bin/env python3
"""Turn a reviewed dataset into a YOLO train/val layout.

Ingest writes one folder per video (`images/<video-stem>/`), which is right for
review but wrong for training: Ultralytics wants `images/train` and `images/val`.
This bridges the two without moving anything in the source dataset.

Why blocks instead of a random per-frame split
----------------------------------------------
Frames are sampled at ~5 fps, so consecutive frames of a match are near
duplicates. A random split puts frame N in train and N+1 in val, the model has
effectively already seen the val frame, and val metrics come out flattering.
That would undermine the numbers the bounty difficulty and the model-vs-label
disagreement score are built on.

So frames are grouped into contiguous blocks, whole blocks are assigned to one
side, and a guard band is dropped on both sides of each train/val boundary. The
report prints how many cross-split neighbours remain, which should be zero.

Why missing label files are skipped
-----------------------------------
A missing `.txt` means nobody has reviewed that frame. An *empty* `.txt` means
someone reviewed it and there are no players (a genuine background frame). Only
the second is training data. Treating them the same would teach the model that
"no box" is the answer for frames no human ever looked at.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FRAME_RE = re.compile(r"_frame_(\d+)$")


@dataclass
class Frame:
    image: Path
    label: Path
    source: str
    order: float
    stem: str
    timed: bool = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split a reviewed dataset into images/train + images/val.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Output layout:\n"
            "  <out>/images/train/*.jpg   <out>/labels/train/*.txt\n"
            "  <out>/images/val/*.jpg     <out>/labels/val/*.txt\n"
            "  <out>/data.yaml\n"
        ),
    )
    p.add_argument("dataset", type=Path, help="Dataset root holding images/ and labels/.")
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="Output root (default: .staging/train/<dataset name>).")
    p.add_argument("--val-frac", type=float, default=0.2, help="Target validation fraction.")
    p.add_argument("--block", type=int, default=25,
                   help="Max frames per contiguous block (default 25).")
    p.add_argument("--max-gap", type=float, default=2.0,
                   help="Start a new block after this many seconds without frames.")
    p.add_argument("--guard", type=int, default=2,
                   help="Frames dropped on each side of a train/val boundary (default 2).")
    p.add_argument("--seed", type=int, default=0, help="Block shuffle seed.")
    p.add_argument("--link", choices=["hardlink", "copy", "symlink"], default="hardlink",
                   help="How to place files in the output (default hardlink).")
    p.add_argument("--force", action="store_true", help="Overwrite a non-empty output root.")
    p.add_argument("--dry-run", action="store_true", help="Report only; write nothing.")
    return p.parse_args()


# ------------------------------------------------------------------ discovery
def _stem_of(image_id: str) -> str:
    """Filename without extension, without trusting pathlib for it.

    `Path("… 2026.1 #11-20 …_frame_002844").stem` cuts at the dot inside the
    video title, so frames.csv ids (which have no extension) would all collapse
    to the same key. Strip a known image extension explicitly instead.
    """
    name = image_id.rsplit("/", 1)[-1]
    lower = name.lower()
    for ext in IMAGE_EXTS:
        if lower.endswith(ext):
            return name[: -len(ext)]
    return name


def load_times(dataset: Path) -> dict[str, float]:
    """Map frame stem -> video timestamp, from frames.csv when it exists."""
    times: dict[str, float] = {}
    csv_path = dataset / "frames.csv"
    if not csv_path.exists():
        return times
    try:
        with csv_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                image_id = (row.get("image_id") or "").strip()
                raw = (row.get("time_s") or "").strip()
                if not image_id or not raw:
                    continue
                try:
                    times[_stem_of(image_id)] = float(raw)
                except ValueError:
                    continue
    except OSError:
        return {}
    return times


def collect_frames(dataset: Path) -> tuple[list[Frame], dict[str, int]]:
    """Every image that has a label file, plus a count of what was skipped."""
    images_root = dataset / "images"
    labels_root = dataset / "labels"
    if not images_root.is_dir():
        raise SystemExit(f"No images/ directory in {dataset}")

    times = load_times(dataset)
    stats = {"unreviewed": 0, "no_label_dir": 0}
    frames: list[Frame] = []

    for image in sorted(images_root.rglob("*")):
        if not image.is_file() or image.suffix.lower() not in IMAGE_EXTS:
            continue
        source = image.parent.name
        label = labels_root / source / f"{image.stem}.txt"
        if not label.exists():
            # Try a flat labels/ dir before giving up.
            alt = labels_root / f"{image.stem}.txt"
            if alt.exists():
                label = alt
            else:
                stats["unreviewed"] += 1
                continue

        if image.stem in times:
            order = times[image.stem]
            timed = True
        else:
            m = FRAME_RE.search(image.stem)
            order = float(m.group(1)) if m else float(len(frames))
            timed = False
        frames.append(Frame(image, label, source, order, image.stem, timed))

    if not labels_root.is_dir():
        stats["no_label_dir"] = 1
    return frames, stats


def make_blocks(frames: list[Frame], block: int, max_gap: float) -> list[list[Frame]]:
    """Group by source, then cut into contiguous runs of at most `block` frames."""
    blocks: list[list[Frame]] = []
    by_source: dict[str, list[Frame]] = {}
    for f in frames:
        by_source.setdefault(f.source, []).append(f)

    for source in sorted(by_source):
        seq = sorted(by_source[source], key=lambda f: f.order)
        current: list[Frame] = []
        for f in seq:
            if current:
                gap = f.order - current[-1].order
                # Only trust a seconds-based gap when both frames have real
                # timestamps. Without them `order` is a frame index, and 2.0
                # would mean "two frames apart", shattering the video into
                # one-frame blocks.
                too_far = (current[-1].timed and f.timed and gap > max_gap)
                if too_far or len(current) >= block:
                    blocks.append(current)
                    current = []
            current.append(f)
        if current:
            blocks.append(current)
    return blocks


def assign(blocks: list[list[Frame]], val_frac: float, seed: int) -> list[str]:
    """Shuffle whole blocks, then fill val until the target fraction is met."""
    order = list(range(len(blocks)))
    random.Random(seed).shuffle(order)

    total = sum(len(b) for b in blocks)
    target = max(1, int(round(total * val_frac)))
    side = ["train"] * len(blocks)
    val_frames = 0

    for idx in order:
        if val_frames >= target:
            break
        # Always leave at least one block on the train side.
        if len(blocks) - sum(1 for s in side if s == "val") <= 1:
            break
        side[idx] = "val"
        val_frames += len(blocks[idx])
    return side


def apply_guard(blocks: list[list[Frame]], side: list[str], guard: int) -> set[str]:
    """Drop frames next to a boundary, so no near-duplicate pair straddles it."""
    dropped: set[str] = set()
    if guard <= 0:
        return dropped

    by_source: dict[str, list[int]] = {}
    for i, b in enumerate(blocks):
        if b:
            by_source.setdefault(b[0].source, []).append(i)

    for idxs in by_source.values():
        idxs.sort(key=lambda i: blocks[i][0].order)
        for a, b in zip(idxs, idxs[1:]):
            if side[a] == side[b]:
                continue
            for f in blocks[a][-guard:]:
                dropped.add(f.stem)
            for f in blocks[b][:guard]:
                dropped.add(f.stem)
    return dropped


def _median_gap(seq: list[tuple[float, str, str]]) -> float:
    deltas = [b[0] - a[0] for a, b in zip(seq, seq[1:]) if b[0] > a[0]]
    if not deltas:
        return 0.0
    deltas.sort()
    return deltas[len(deltas) // 2]


def count_cross_split_neighbours(blocks: list[list[Frame]], side: list[str],
                                 dropped: set[str]) -> int:
    """Neighbouring frames from different sides that are still close in time.

    A guard band removes frames at the boundary, but the surviving last-train and
    first-val frames are still consecutive in the filtered sequence. Only count
    them as leakage if they are within ~1.5 typical frame intervals, so the number
    printed reflects real near-duplicate risk rather than boundary bookkeeping.
    """
    by_source: dict[str, list[tuple[float, str, str]]] = {}
    for i, b in enumerate(blocks):
        for f in b:
            if f.stem in dropped:
                continue
            by_source.setdefault(f.source, []).append((f.order, side[i], f.stem))

    count = 0
    for seq in by_source.values():
        seq.sort()
        if len(seq) < 2:
            continue
        threshold = _median_gap(seq) * 1.5
        for (o1, s1, _), (o2, s2, _) in zip(seq, seq[1:]):
            if s1 != s2 and (o2 - o1) <= threshold:
                count += 1
    return count


# ------------------------------------------------------------------ materialise
def place(src: Path, dest: Path, mode: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    if mode == "symlink":
        os.symlink(src.resolve(), dest)
        return
    if mode == "hardlink":
        try:
            os.link(src, dest)
            return
        except OSError:
            # Different filesystem, or a link-unsafe mount: fall through to copy.
            pass
    shutil.copy2(src, dest)


def data_yaml_text(root: Path) -> str:
    return (
        f"path: {root.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "nc: 1\n"
        "names:\n"
        "  0: player\n"
    )


def main() -> int:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    if not dataset.exists():
        print(f"ERROR: dataset not found: {dataset}")
        return 2

    out = (args.out or REPO_ROOT / ".staging" / "train" / dataset.name).expanduser().resolve()

    frames, stats = collect_frames(dataset)
    if not frames:
        print(f"No trainable frames in {dataset}.")
        if stats["unreviewed"]:
            print(f"  {stats['unreviewed']} image(s) have no label file — none reviewed yet.")
            print("  Review them in the app first; unreviewed frames must not be trained on.")
        elif stats["no_label_dir"]:
            print("  There is no labels/ directory at all.")
        return 1

    blocks = make_blocks(frames, args.block, args.max_gap)
    side = assign(blocks, args.val_frac, args.seed)
    dropped = apply_guard(blocks, side, args.guard)

    train = [f for b, s in zip(blocks, side) if s == "train" for f in b if f.stem not in dropped]
    val = [f for b, s in zip(blocks, side) if s == "val" for f in b if f.stem not in dropped]
    cross = count_cross_split_neighbours(blocks, side, dropped)

    print(f"Dataset  : {dataset}")
    print(f"Output   : {out}")
    print(f"Labelled : {len(frames)} frame(s) across {len(blocks)} block(s)")
    if stats["unreviewed"]:
        print(f"Skipped  : {stats['unreviewed']} unreviewed frame(s) (no label file)")
    print(f"Guard    : {len(dropped)} frame(s) dropped at split boundaries")
    print(f"Train    : {len(train)}")
    print(f"Val      : {len(val)}  ({len(val) / max(1, len(train) + len(val)):.1%})")
    print(f"Leakage  : {cross} cross-split neighbour pair(s)")

    if not train or not val:
        print("\nERROR: one side is empty. Use a larger dataset, a smaller --block, "
              "or a different --seed.")
        return 1

    if args.dry_run:
        print("\nDry run: nothing written.")
        return 0

    if out.exists() and any(out.iterdir()) and not args.force:
        print(f"\n{out} is not empty. Pass --force to overwrite.")
        return 1
    if out.exists() and args.force:
        shutil.rmtree(out)

    for split, items in (("train", train), ("val", val)):
        used: dict[str, Path] = {}
        for f in items:
            name = f.image.name
            # Two source folders could hold the same filename. Prefix only on an
            # actual collision, so the common case stays flat and readable.
            if name in used and used[name] != f.image:
                name = f"{f.source}__{name}"
            used[name] = f.image
            place(f.image, out / "images" / split / name, args.link)
            place(f.label, out / "labels" / split / Path(name).with_suffix(".txt"), args.link)

    (out / "data.yaml").write_text(data_yaml_text(out), encoding="utf-8")
    manifest = {
        "source_dataset": str(dataset),
        "seed": args.seed,
        "val_frac": args.val_frac,
        "block": args.block,
        "guard": args.guard,
        "train": len(train),
        "val": len(val),
        "skipped_unreviewed": stats["unreviewed"],
        "guard_dropped": sorted(dropped),
    }
    (out / "split.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nWrote {out}")
    print("Train on it with:")
    print(f"  python scripts/train_ssbu.py --root {out.parent} --dataset {out.name} \\")
    print(f"      --name <run-name> --epochs 50")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
