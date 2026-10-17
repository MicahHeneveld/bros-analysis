#!/usr/bin/env python3
"""Measure how far apart the two fighters are, frame by frame.

Reads detections produced by ``scripts/ingest_video.py`` (or the review app's
dataset labels), and writes a per-frame CSV plus a summary.

Two metrics, because they fail in different ways -- see ``app/distance.py``:

``body_lengths``
    Separation in fighter-heights. Needs no calibration and is exact when both
    fighters are at the same depth. Blind to depth separation: two fighters one
    in front of the other read as almost touching.

``ground_lengths``
    Distance on the stage floor, in fighter-heights, from a self-calibrated
    ground-plane model. Correct for depth separation, but the depth component is
    sensitive to bbox-height noise, so it is smoothed over time by default.

Smoothing matters more than it looks. Bbox height wobbles with pose, crouching
and hitstun, and the depth term differences reciprocals of it, which amplifies
that noise. A rolling median over neighbouring frames in the same shot removes
most of it, so the smoothed column is the one to trust for anything downstream.
The window resets at shot boundaries, detected either from the timestamp
sidecar (``frames.csv``) or from an implausible jump between frames.

Examples
--------
Summarise a freshly ingested video::

    python scripts/measure_distance.py .staging/ingested/match

Write the per-frame trace somewhere specific::

    python scripts/measure_distance.py .staging/ingested/match --csv out.csv

Skip smoothing (raw per-frame values)::

    python scripts/measure_distance.py .staging/ingested/match --smooth 0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "app"))

from distance import (  # noqa: E402
    DEFAULT_G,
    calibrate,
    measure,
    summarize,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-frame distance between the two fighters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Input can be a dataset directory (from ingest_video.py), a "
            "predictions directory, or a single .jsonl."
        ),
    )
    p.add_argument(
        "source", type=Path,
        help="Dataset dir, predictions dir, or a single .jsonl file.",
    )
    p.add_argument("--split", default=None, help="Limit to one split by name.")
    p.add_argument(
        "--csv", type=Path, default=None,
        help="Where to write the per-frame trace (default: <source>/distance.csv).",
    )
    p.add_argument(
        "--smooth", type=int, default=5,
        help=(
            "Rolling-median window in frames, applied within a shot (default 5; "
            "1 or 0 disables). Set 0 when you want raw per-frame values."
        ),
    )
    p.add_argument(
        "--g", type=float, default=DEFAULT_G,
        help=(
            "f / frame_height, i.e. 1/(2 tan(vfov/2)). Only rescales the depth "
            "component (default %.2f)." % DEFAULT_G
        ),
    )
    p.add_argument(
        "--no-calibrate", action="store_true",
        help="(default) Report body_lengths only; skip the ground-plane fit.",
    )
    p.add_argument(
        "--ground", action="store_true",
        help=(
            "Also fit the ground-plane model and report ground_lengths. "
            "OFF by default because it does not validate on SSBU footage: "
            "apparent height is dominated by character identity and pose "
            "rather than depth, giving R2 ~ 0. Treat its output as research, "
            "not as a measurement. See docs/distance_metric.md."
        ),
    )
    p.add_argument(
        "--max-gap", type=float, default=0.5,
        help=(
            "Treat a jump of more than this many seconds between consecutive "
            "frames as a shot boundary and restart smoothing (default 0.5)."
        ),
    )
    p.add_argument("--json", action="store_true", help="Print the summary as JSON.")
    return p.parse_args()


def load_records(source: Path, split: str | None) -> dict[str, dict]:
    """Load predictions from a jsonl, a predictions dir, or a dataset dir."""
    files: list[Path] = []
    if source.is_file():
        files = [source]
    else:
        pred_dir = source / "predictions" if (source / "predictions").is_dir() else source
        files = sorted(pred_dir.glob("*.jsonl"))
    records: dict[str, dict] = {}
    for path in files:
        if split and path.stem != split:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            iid = rec.get("image_id")
            if iid:
                records[iid] = rec
    return records


def load_timestamps(source: Path) -> dict[str, float]:
    """Read the ``frames.csv`` sidecar written by ingest_video.py, if present."""
    csv_path = source / "frames.csv" if source.is_dir() else source.parent / "frames.csv"
    if not csv_path.exists():
        return {}
    out: dict[str, float] = {}
    with open(csv_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                out[row["image_id"]] = float(row["time_s"])
            except (KeyError, ValueError):
                continue
    return out


def pick_boxes(rec: dict) -> list[dict]:
    """The post-processed detections: deduped and capped at two."""
    boxes = rec.get("predictions_clean")
    if boxes is None:
        boxes = rec.get("predictions") or []
    return boxes or []


def rolling_median(values: list[float], window: int) -> list[float]:
    """Centred rolling median. Used to suppress pose-driven height noise."""
    if window <= 1 or not values:
        return list(values)
    half = window // 2
    out: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        chunk = sorted(values[lo:hi])
        n = len(chunk)
        out.append(chunk[n // 2] if n % 2 else (chunk[n // 2 - 1] + chunk[n // 2]) / 2.0)
    return out


def main() -> int:
    args = parse_args()
    source = args.source.expanduser()
    if not source.exists():
        print(f"ERROR: not found: {source}")
        return 2

    records = load_records(source, args.split)
    if not records:
        print(f"ERROR: no predictions found under {source}")
        return 2

    times = load_timestamps(source)

    # A frame is usable only when exactly two fighters were found: with one,
    # there is no separation to measure, and more than two is over-detection.
    usable: list[tuple[str, dict, dict]] = []
    skipped: dict[str, int] = {"0 boxes": 0, "1 box": 0, "3+ boxes": 0}
    for iid in sorted(records):
        boxes = pick_boxes(records[iid])
        if len(boxes) == 2:
            usable.append((iid, boxes[0], boxes[1]))
        elif not boxes:
            skipped["0 boxes"] += 1
        elif len(boxes) == 1:
            skipped["1 box"] += 1
        else:
            skipped["3+ boxes"] += 1

    if not usable:
        print(f"ERROR: no frames with exactly two fighters in {len(records)} records")
        print(f"  skipped: {skipped}")
        return 1

    # Calibrate on the usable pairs. Most-recent-instance dimensions are used
    # per record, so mixed resolutions inside one set are handled.
    pairs = [(a, b) for _, a, b in usable]
    width = float(records[usable[0][0]].get("width") or 0)
    height = float(records[usable[0][0]].get("height") or 0)
    if not width or not height:
        print("ERROR: prediction records have no width/height; cannot measure")
        return 2

    mismatched = {
        iid for iid, _, _ in usable
        if (float(records[iid].get("width") or width) != width
            or float(records[iid].get("height") or height) != height)
    }
    if mismatched:
        print(f"  note: {len(mismatched)} frame(s) differ in resolution from the first; "
              "using each frame's own dimensions.")

    calibration = None
    if args.ground:
        calibration = calibrate(pairs, width, height)

    # --- per-frame measurement -------------------------------------------
    rows: list[dict] = []
    for iid, a, b in usable:
        rec = records[iid]
        w = float(rec.get("width") or width)
        h = float(rec.get("height") or height)
        m = measure(a, b, w, h, calibration, g=args.g)
        rows.append({
            "image_id": iid,
            "time_s": times.get(iid, ""),
            "body_lengths": m.body_lengths,
            "depth_ratio": m.depth_ratio,
            "ground_lengths": m.ground_lengths,
            "lateral": m.lateral,
            "depth": m.depth,
        })

    # --- shot segmentation + smoothing ------------------------------------
    # Split into runs of temporally adjacent frames so the median never
    # straddles a cut, where the camera and fighters change completely.
    shots: list[list[int]] = []
    current: list[int] = []
    previous_t: float | None = None
    for idx, row in enumerate(rows):
        t = row["time_s"]
        if not isinstance(t, float):
            # No sidecar: fall back to treating the whole set as one shot.
            if previous_t is None:
                current.append(idx)
                continue
            previous_t = None
            current.append(idx)
            continue
        if previous_t is not None and (t - previous_t) > args.max_gap:
            if current:
                shots.append(current)
            current = [idx]
        else:
            current.append(idx)
        previous_t = t
    if current:
        shots.append(current)

    body_s: list[float | None] = [None] * len(rows)
    ground_s: list[float | None] = [None] * len(rows)
    for shot in shots:
        bs = [rows[i]["body_lengths"] for i in shot]
        for i, v in zip(shot, rolling_median(bs, args.smooth)):
            body_s[i] = v
        gs = [rows[i]["ground_lengths"] for i in shot if rows[i]["ground_lengths"] is not None]
        if gs:
            smoothed = rolling_median(gs, args.smooth)
            k = 0
            for i in shot:
                if rows[i]["ground_lengths"] is not None:
                    ground_s[i] = smoothed[k]
                    k += 1

    for i, row in enumerate(rows):
        row["body_smoothed"] = body_s[i]
        row["ground_smoothed"] = ground_s[i]

    # --- write ------------------------------------------------------------
    out_csv = args.csv or (
        source / "distance.csv" if source.is_dir() else source.with_suffix(".distance.csv")
    )
    fields = [
        "image_id", "time_s", "body_lengths", "body_smoothed", "depth_ratio",
        "ground_lengths", "ground_smoothed", "lateral", "depth",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in row.items()
            })

    # --- summary ----------------------------------------------------------
    body_vals = [r["body_lengths"] for r in rows]
    body_smooth = [v for v in body_s if v is not None]
    ground_vals = [r["ground_lengths"] for r in rows if r["ground_lengths"] is not None]
    ground_smooth = [v for v in ground_s if v is not None]
    depth_dominated = sum(1 for r in rows if r.get("depth") and r.get("lateral") is not None
                          and r["depth"] > r["lateral"])

    summary = {
        "source": str(source),
        "frames_total": len(records),
        "frames_measured": len(rows),
        "frames_skipped": skipped,
        "shots": len(shots),
        "g": args.g,
        "smooth_window": args.smooth,
        "calibration": (
            {
                "horizon": round(calibration.horizon_norm, 4),
                "camera_height": round(calibration.camera_height, 4),
                "r2": round(calibration.r2, 4),
                "samples": calibration.samples,
                "reliable": calibration.reliable,
                "note": calibration.note,
            }
            if calibration else {"skipped": True}
        ),
        "body_lengths": summarize(body_vals),
        "body_lengths_smoothed": summarize(body_smooth),
        "ground_lengths": summarize(ground_vals),
        "ground_lengths_smoothed": summarize(ground_smooth),
        "frames_depth_dominant": depth_dominated,
        "csv": str(out_csv),
    }

    summary_path = out_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print(f"Source          : {source}")
    print(f"Frames          : {len(rows)} measured / {len(records)} total"
          f"   (skipped: {skipped})")
    print(f"Shots           : {len(shots)}")
    print()
    if calibration:
        status = "reliable" if calibration.reliable else "NOT RELIABLE"
        print(f"Calibration     : horizon={calibration.horizon_norm:.3f}  "
              f"camera_height={calibration.camera_height:.2f}  R2={calibration.r2:.2f}  "
              f"n={calibration.samples}  [{status}]")
        if calibration.note:
            print(f"                  {calibration.note}")
        print("                  (camera_height in fighter-heights above the floor)")
    elif args.no_calibrate or not args.ground:
        print("Calibration     : skipped (ground-plane model is opt-in via --ground)")
        print("                  body_lengths needs no calibration and is the metric to use.")
    else:
        print("Calibration     : not enough same-depth pairs to fit a horizon")
    print()
    print(f"  {'metric':<26} {'mean':>7} {'median':>7} {'p05':>7} {'p95':>7} {'max':>7}")
    print("  " + "-" * 66)
    for label, vals in (
        ("body_lengths", body_vals),
        ("body_lengths (smoothed)", body_smooth),
        ("ground_lengths", ground_vals),
        ("ground_lengths (smooth)", ground_smooth),
    ):
        s = summarize(vals)
        if not s.get("n"):
            continue
        print(f"  {label:<26} {s['mean']:>7.2f} {s['median']:>7.2f} "
              f"{s['p05']:>7.2f} {s['p95']:>7.2f} {s['max']:>7.2f}")
    print()
    print(f"  {summary['frames_depth_dominant']} frame(s) are depth-dominated "
          "(depth separation > lateral) --")
    print("  these are the frames where body_lengths under-reports and the")
    print("  ground-plane metric earns its keep.")
    print()
    print(f"Wrote {out_csv}")
    print(f"Wrote {summary_path}")

    # A sanity check the numbers cannot give you on their own.
    if calibration and calibration.reliable and ground_vals and body_vals:
        ratio = (sum(ground_vals) / len(ground_vals)) / (sum(body_vals) / len(body_vals))
        if ratio > 3:
            print()
            print(f"  warning: ground is {ratio:.1f}x larger than body on average. That is")
            print("  usually a mis-scaled depth term -- check the fitted horizon against")
            print("  a screenshot, or lower --g if the camera is more zoomed than assumed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
