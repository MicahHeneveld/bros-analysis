#!/usr/bin/env python3
"""Turn a raw .mp4 into a review-ready, pre-labelled frame set.

This is the "give me a video, get labels I can check" front door. It does four
things that the existing scripts each do only one of:

1. Samples the video at a fixed *target* frame rate (not "every Nth frame"), so
   the resulting time step is meaningful and comparable across videos with
   different source frame rates.
2. Drops non-gameplay frames (menus, loading screens, results) by asking the
   detector whether it can see the expected number of fighters. No separate
   classifier is needed and there is nothing extra to train.
3. Runs inference in bounded batches, reusing the chunking lessons from
   ``predict_ssbu.py`` so a long video cannot balloon memory.
4. Writes a YOLO dataset the review app can open directly, plus a
   ``frames.csv`` mapping every kept frame back to its video timestamp.

Deliberate design choice: labels start EMPTY and predictions live beside them.
The model's guesses are never silently promoted to ground truth, because a
dataset that quietly contains its own predictions trains a model to agree with
itself. Use ``--seed-labels`` when you want the boxes pre-filled, and then
review every frame before training.

Examples
--------
Sample a match at 5 fps and keep only frames showing both fighters::

    python scripts/ingest_video.py match.mp4

Only the section from 2:00 to 6:30, at 10 fps, pre-filling the label files::

    python scripts/ingest_video.py match.mp4 --start 120 --end 390 --fps 10 --seed-labels

Then review it::

    python app/server.py --dataset .staging/ingested/match \\
        --predictions .staging/ingested/match/predictions
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
if not (_HERE / "predictions.py").exists():
    _repo_app = _HERE.parent / "app"
    if (_repo_app / "predictions.py").exists():
        sys.path.insert(0, str(_repo_app))

from predictions import MAX_BOXES, NMS_IOU, postprocess  # noqa: E402

REPO_ROOT = _HERE.parent
DEFAULT_OUT_ROOT = REPO_ROOT / ".staging" / "ingested"
DEFAULT_JPEG_QUALITY = 92


# --------------------------------------------------------------------- args


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract, gate and pre-label gameplay frames from a video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Output layout:\n"
            "  <out>/images/<split>/<video>_frame_000123.jpg\n"
            "  <out>/labels/<split>/<video>_frame_000123.txt\n"
            "  <out>/predictions/<split>.jsonl\n"
            "  <out>/frames.csv        image_id,frame_index,time_s\n"
            "  <out>/manifest.json     source metadata and run stats"
        ),
    )
    p.add_argument("video", type=Path, help="Path to the input video.")
    p.add_argument(
        "-o", "--out", type=Path, default=None,
        help="Dataset root (default: .staging/ingested/<video stem>).",
    )
    p.add_argument(
        "--split", default=None,
        help="Split folder name to write into (default: the video stem).",
    )
    p.add_argument(
        "--fps", type=float, default=5.0,
        help="Target sampling rate in frames/second (default 5).",
    )
    p.add_argument("--start", type=float, default=0.0, help="Start offset in seconds.")
    p.add_argument("--end", type=float, default=0.0, help="Stop offset in seconds (0 = end).")
    p.add_argument(
        "--max-frames", type=int, default=0,
        help="Stop after this many *sampled* frames (0 = no limit). For quick trials.",
    )

    # Detector
    p.add_argument("--weights", type=Path, default=None, help="Explicit checkpoint.")
    p.add_argument(
        "--root", type=Path, default=REPO_ROOT / ".staging",
        help="Root holding runs/ (default: .staging).",
    )
    p.add_argument(
        "--run-name", default="smash_player_colab_v4",
        help="Run folder whose weights/best.pt is used when --weights is omitted.",
    )
    p.add_argument("--conf", type=float, default=0.25, help="Detector confidence floor.")
    p.add_argument("--iou", type=float, default=0.7, help="Detector NMS IoU.")
    p.add_argument("--post-nms-iou", type=float, default=NMS_IOU, help="Second dedupe pass IoU.")
    p.add_argument("--max-boxes", type=int, default=MAX_BOXES, help="Per-frame box cap.")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument(
        "--batch", type=int, default=16,
        help="Frames decoded and inferred per batch (default 16). Bounds peak RAM.",
    )
    p.add_argument(
        "--device", default=None,
        help="Compute device ('mps', 'cuda', 'cpu'). Default: mps when available.",
    )

    # Gating
    p.add_argument(
        "--min-players", type=int, default=2,
        help="Require this many confident, plausibly-sized detections (default 2; 0 disables gating).",
    )
    p.add_argument(
        "--player-conf", type=float, default=0.35,
        help="Detection confidence needed to count as a visible player (default 0.35).",
    )
    p.add_argument(
        "--min-height", type=float, default=0.04,
        help="Reject boxes shorter than this fraction of frame height (default 0.04).",
    )
    p.add_argument(
        "--max-height", type=float, default=0.90,
        help="Reject boxes taller than this fraction of frame height (default 0.90).",
    )
    p.add_argument(
        "--min-aspect", type=float, default=0.45,
        help="Reject boxes with height/width below this (default 0.45; fighters are taller than wide).",
    )

    # Output
    p.add_argument(
        "--dedupe", type=float, default=1.0,
        help=(
            "Skip a frame whose 32x32 grayscale thumbnail differs from the last "
            "saved frame by less than this mean absolute difference (0-255 scale, "
            "default 1.0). 0 disables. Suppresses static pause screens and holds."
        ),
    )
    p.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    p.add_argument(
        "--seed-labels", action="store_true",
        help=(
            "Write the model's boxes into the label files as pre-labels. "
            "Convenient, but you MUST review every frame before training on it."
        ),
    )
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite existing frames/predictions in the output dataset.",
    )
    p.add_argument(
        "--no-stage", action="store_true",
        help="Skip the ultralytics install check (assume it is already available).",
    )
    return p.parse_args()


# ------------------------------------------------------------------ helpers


def resolve_weights(args: argparse.Namespace) -> Path:
    """Pick a checkpoint: explicit path, then the named run, then the newest run."""
    if args.weights:
        return args.weights.expanduser()
    root = args.root.expanduser()
    named = root / "runs" / args.run_name / "weights" / "best.pt"
    if named.exists():
        return named
    candidates = sorted(
        root.glob("runs/*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        print(f"  note: '{args.run_name}' not found; using {candidates[0].parent.parent.name}")
        return candidates[0]
    raise FileNotFoundError(
        f"No weights found under {root / 'runs'}/*/weights/best.pt and --weights was not given."
    )


def pick_device(explicit: str | None) -> str | None:
    """Default to MPS on Apple Silicon.

    Ultralytics auto-selects CPU on macOS, which is several times slower for
    this workload, so inferring MPS explicitly is a real speedup rather than a
    micro-optimisation.
    """
    if explicit:
        return explicit
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return None


def release_cache() -> None:
    """Drop cached MPS blocks between batches; no-op elsewhere.

    Same reasoning as ``predict_ssbu.py``: the MPS allocator retains freed
    blocks, so a long run ratchets its footprint upward even with small batches.
    """
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def thumbnail(frame) -> np.ndarray:
    """Small grayscale signature used for the near-duplicate check."""
    small = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.int16)


def is_plausible_player(box: dict, frame_h: float, frame_w: float, args) -> bool:
    """Whether one detection looks like a fighter rather than UI furniture.

    Combatants are tall, occupy a believable slice of the frame, and carry
    enough confidence to be real. Menus and result screens fail at least one of
    these, which is what makes detector-driven gating work without a second model.
    """
    if float(box.get("conf", 0.0)) < args.player_conf:
        return False
    bh = float(box["h"]) * frame_h
    bw = float(box["w"]) * frame_w
    if bh <= 0 or bw <= 0:
        return False
    h_frac = bh / frame_h if frame_h else 0.0
    if not (args.min_height <= h_frac <= args.max_height):
        return False
    return (bh / bw) >= args.min_aspect


def count_players(boxes: list[dict], frame_h: float, frame_w: float, args) -> int:
    return sum(1 for b in boxes if is_plausible_player(b, frame_h, frame_w, args))


def boxes_from_result(result) -> tuple[list[dict], int, int]:
    """Normalized ``cxcywh`` dicts plus the frame's (width, height)."""
    h, w = result.orig_shape[:2]
    boxes: list[dict] = []
    xywhn = result.boxes.xywhn
    confs = result.boxes.conf
    clss = result.boxes.cls
    for i in range(len(xywhn)):
        xc, yc, bw, bh = (float(v) for v in xywhn[i].tolist())
        boxes.append(
            {
                "cls": int(clss[i].item()),
                "conf": round(float(confs[i].item()), 4),
                "xc": round(xc, 6),
                "yc": round(yc, 6),
                "w": round(bw, 6),
                "h": round(bh, 6),
            }
        )
    return boxes, int(w), int(h)


def label_line(box: dict) -> str:
    return (
        f"{int(box['cls'])} {float(box['xc']):.6f} {float(box['yc']):.6f} "
        f"{float(box['w']):.6f} {float(box['h']):.6f}"
    )


def write_data_yaml(dataset: Path, split: str) -> None:
    """A valid YOLO config pointing at this split.

    It is not a train/val dataset on its own; this only exists so tools that
    expect a ``data.yaml`` (and a future merge step) can read the set.
    """
    (dataset / "data.yaml").write_text(
        "path: .\n"
        f"train: images/{split}\n"
        f"val: images/{split}\n"
        "\n"
        "nc: 1\n"
        "names:\n"
        "  0: player\n",
        encoding="utf-8",
    )


def load_manifest(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            pass
    return {"sources": {}}


# --------------------------------------------------------------------- main


def main() -> int:
    args = parse_args()
    video = args.video.expanduser()
    if not video.exists():
        print(f"ERROR: video not found: {video}")
        return 2

    try:
        weights = resolve_weights(args)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 2

    dataset = (args.out.expanduser() if args.out else DEFAULT_OUT_ROOT / video.stem)
    split = args.split or video.stem
    images_dir = dataset / "images" / split
    labels_dir = dataset / "labels" / split
    pred_dir = dataset / "predictions"
    for d in (images_dir, labels_dir, pred_dir):
        d.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"ERROR: OpenCV cannot open {video}")
        return 2

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not math.isfinite(src_fps) or src_fps <= 0:
        print("  note: container reports no frame rate; assuming 30 fps")
        src_fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = total_frames / src_fps if total_frames else 0.0

    if args.fps <= 0:
        print("ERROR: --fps must be positive")
        return 2
    stride = max(1, int(round(src_fps / args.fps)))
    effective_fps = src_fps / stride

    start_frame = max(0, int(round(args.start * src_fps)))
    end_frame = int(round(args.end * src_fps)) if args.end > 0 else total_frames or 0
    if end_frame and end_frame <= start_frame:
        print("ERROR: --end must be after --start")
        return 2

    print(f"Video   : {video}")
    print(f"Source  : {src_w}x{src_h} @ {src_fps:.3f} fps, "
          f"{duration:.1f}s ({total_frames or '?'} frames)")
    print(f"Sampling: every {stride} frame(s) -> {effective_fps:.2f} fps")
    print(f"Segments: {args.start:.1f}s .. "
          f"{args.end if args.end else 'end'}")
    print(f"Output  : {dataset}  (split '{split}')")
    print(f"Weights : {weights}")
    print(f"Gating  : >= {args.min_players} player(s) at conf >= {args.player_conf}"
          if args.min_players > 0 else "Gating  : disabled")
    print()

    if args.no_stage:
        from ultralytics import YOLO
    else:
        try:
            from train_ssbu import ensure_ultralytics

            ensure_ultralytics(False)
        except Exception as exc:  # noqa: BLE001
            print(f"  note: could not verify ultralytics ({exc}); trying anyway")
        from ultralytics import YOLO

    device = pick_device(args.device)
    if device:
        print(f"Device  : {device}")

    model = YOLO(str(weights))
    predict_kwargs = dict(
        conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False,
    )
    if device:
        predict_kwargs["device"] = device

    pred_path = pred_dir / f"{split}.jsonl"
    pred_mode = "w" if args.force else "a"

    stats = {
        "sampled": 0, "gated_out": 0, "deduped": 0,
        "already_present": 0, "saved": 0, "boxes": 0,
    }
    heights: list[float] = []
    frame_rows: list[tuple[str, int, float]] = []
    started = time.time()

    global_start = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Holds the thumbnail of the last frame written, for the duplicate check.
    # Declared before ``flush`` closes over it.
    thumbnail_state: dict[str, np.ndarray | None] = {"last": None}

    def flush(batch: list[tuple[int, np.ndarray]], fh) -> None:
        """Infer one batch, gate it, and write the survivors."""
        results = model.predict(source=[f for _, f in batch], **predict_kwargs)
        for (frame_idx, frame), result in zip(batch, results):
            stats["sampled"] += 1
            boxes, w, h = boxes_from_result(result)
            cleaned = postprocess(boxes, w, h, args.post_nms_iou, args.max_boxes)

            if args.min_players > 0:
                if count_players(cleaned, h, w, args) < args.min_players:
                    stats["gated_out"] += 1
                    continue

            name = f"{video.stem}_frame_{frame_idx:06d}"
            image_id = f"{split}/{name}"
            image_path = images_dir / f"{name}.jpg"
            if image_path.exists() and not args.force:
                stats["already_present"] += 1
                continue

            # Near-duplicate check against the last frame actually written, so a
            # slow pan is kept while a paused/static screen collapses to one frame.
            if args.dedupe > 0 and thumbnail_state["last"] is not None:
                diff = float(np.mean(np.abs(thumbnail(frame) - thumbnail_state["last"])))
                if diff < args.dedupe:
                    stats["deduped"] += 1
                    continue

            ok, encoded = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
            )
            if not ok:
                print(f"  warning: failed to encode frame {frame_idx}; skipped")
                continue
            encoded.tofile(str(image_path))
            thumbnail_state["last"] = thumbnail(frame)

            # Labels are empty unless the caller opted into pre-labelling.
            label_path = labels_dir / f"{name}.txt"
            if args.seed_labels and cleaned:
                label_path.write_text(
                    "\n".join(label_line(b) for b in cleaned) + "\n", encoding="utf-8"
                )
            else:
                label_path.write_text("", encoding="utf-8")

            fh.write(json.dumps({
                "image_id": image_id,
                "width": w,
                "height": h,
                "frame_index": frame_idx,
                "time_s": round(frame_idx / src_fps, 3),
                "source": video.stem,
                "predictions": boxes,
                "predictions_clean": cleaned,
            }) + "\n")

            stats["saved"] += 1
            stats["boxes"] += len(cleaned)
            if len(cleaned) >= 2:
                heights.append(sum(float(b["h"]) for b in cleaned[:2]) / 2.0)
            frame_rows.append((image_id, frame_idx, round(frame_idx / src_fps, 3)))

    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    batch: list[tuple[int, np.ndarray]] = []
    frame_idx = start_frame
    stop_early = False

    with open(pred_path, pred_mode, encoding="utf-8") as fh:
        while not stop_early:
            ret, frame = cap.read()
            if not ret:
                break
            if end_frame and frame_idx >= end_frame:
                break
            if frame_idx % stride == 0:
                batch.append((frame_idx, frame))
                if args.max_frames and stats["sampled"] + len(batch) >= args.max_frames:
                    stop_early = True
                if len(batch) >= max(1, args.batch):
                    flush(batch, fh)
                    print(f"  {stats['sampled']} sampled -> {stats['saved']} kept", flush=True)
                    batch = []
                    gc.collect()
                    release_cache()
            frame_idx += 1

        if batch:
            flush(batch, fh)
        cap.release()

    elapsed = time.time() - started

    write_data_yaml(dataset, split)

    # Timestamp sidecar: how every kept frame maps back to the source video.
    csv_path = dataset / "frames.csv"
    new_csv = not csv_path.exists() or args.force
    with open(csv_path, "w" if new_csv else "a", encoding="utf-8") as fh:
        if new_csv:
            fh.write("image_id,frame_index,time_s\n")
        for image_id, idx, t in frame_rows:
            fh.write(f"{image_id},{idx},{t}\n")

    manifest_path = dataset / "manifest.json"
    manifest = {"sources": {}} if args.force else load_manifest(manifest_path)
    # A re-run that found every frame already present must not overwrite the
    # real counters from the run that actually did the work with a page of
    # zeros. Keep the previous record unless this run saved something new.
    previous = manifest.get("sources", {}).get(video.stem)
    if not (stats["saved"] == 0 and previous):
        manifest["dataset"] = str(dataset)
        manifest["updated"] = global_start
        manifest["params"] = {
            "fps": args.fps,
            "stride": stride,
            "effective_fps": round(effective_fps, 4),
            "conf": args.conf,
            "player_conf": args.player_conf,
            "min_players": args.min_players,
            "max_boxes": args.max_boxes,
            "dedupe": args.dedupe,
            "seed_labels": args.seed_labels,
            "weights": str(weights),
        }
        manifest["sources"][video.stem] = {
            "path": str(video),
            "width": src_w,
            "height": src_h,
            "fps": round(src_fps, 4),
            "duration_s": round(duration, 2),
            "start_s": args.start,
            "end_s": args.end or None,
            "sampled": stats["sampled"],
            "gated_out": stats["gated_out"],
            "deduped": stats["deduped"],
            "already_present": stats["already_present"],
            "saved": stats["saved"],
            "mean_fighter_height": round(sum(heights) / len(heights), 4) if heights else None,
            "seconds": round(elapsed, 1),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    else:
        print(f"\nKeeping the existing manifest entry for '{video.stem}'.")

    print_report(dataset, split, pred_dir, stats, csv_path, manifest_path, args, elapsed)
    return exit_code_for(stats)


def exit_code_for(stats: dict) -> int:
    """0 on success or a no-op re-run; 1 when a run genuinely found nothing."""
    if stats["saved"]:
        return 0
    # Everything this segment produces is already on disk. Re-running the same
    # video (or resuming an interrupted one) is a successful no-op, not an error.
    return 0 if stats["already_present"] else 1


def print_report(
    dataset: Path,
    split: str,
    pred_dir: Path,
    stats: dict,
    csv_path: Path,
    manifest_path: Path,
    args: argparse.Namespace,
    elapsed: float,
) -> None:

    kept_pct = 100 * stats["saved"] / stats["sampled"] if stats["sampled"] else 0.0
    print()
    print(f"Sampled {stats['sampled']} frame(s) in {elapsed:.1f}s")
    print(f"  kept          : {stats['saved']} ({kept_pct:.0f}%)")
    print(f"  not gameplay  : {stats['gated_out']}")
    print(f"  near-duplicate: {stats['deduped']}")
    print(f"  already there : {stats['already_present']}")
    if stats["saved"]:
        print(f"  boxes/frame   : {stats['boxes'] / stats['saved']:.2f}")
    print(f"  frames.csv    : {csv_path}")
    print(f"  manifest      : {manifest_path}")

    if stats["saved"] == 0:
        if stats["already_present"] > 0:
            print()
            print(f"All {stats['already_present']} frame(s) were already ingested; nothing to do.")
            print("Pass --force to rewrite them.")
            return
        print()
        print("Nothing was kept. Common causes:")
        print("  - the segment is not gameplay (try --min-players 0 to see everything)")
        print("  - --player-conf is too high for this footage")
        print("  - the video is a menu / results screen")
        return

    if stats["sampled"] >= 20 and stats["gated_out"] > stats["sampled"] * 0.5:
        print()
        print(f"  warning: {100 * stats['gated_out'] / stats['sampled']:.0f}% of frames were "
              "rejected as non-gameplay.")
        print("  If this footage is gameplay, lower --player-conf or --min-players.")

    print()
    print("Review it:")
    print(f"  python app/server.py --dataset {dataset} \\")
    print(f"      --predictions {pred_dir}")
    if args.seed_labels:
        print()
        print("  Note: label files were pre-filled from the model. Review EVERY frame")
        print("  before training on this set, or the model will learn its own mistakes.")
    elif stats["saved"]:
        print()
        print("  Labels are empty on purpose; the model's boxes show as the overlay.")
        print("  Press 'p' for Both view, then 'a' to adopt the boxes you agree with.")


if __name__ == "__main__":
    raise SystemExit(main())
