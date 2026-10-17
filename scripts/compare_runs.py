#!/usr/bin/env python3
"""Compare two prediction runs against the current human labels.

The point of the v3 iteration is to fix a specific, measured weakness of v2:
in two-player frames the model would sometimes place only one box. This script
quantifies that (and over-detection) so the two runs can be ranked on evidence
rather than impression.

Usage:
    python scripts/compare_runs.py --runs smash_player_colab_v2 smash_player_colab_v3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _cand in (_HERE, _HERE.parent / "app"):
    if (_cand / "predictions.py").exists():
        sys.path.insert(0, str(_cand))

from label_io import LabelStore  # noqa: E402
from predictions import MAX_BOXES, NMS_IOU, evaluate, load_jsonl, postprocess  # noqa: E402


def load_run(path: Path) -> dict[str, dict]:
    if path.is_file():
        return load_jsonl(path)
    records: dict[str, dict] = {}
    for candidate in sorted(path.glob("*.jsonl")):
        records.update(load_jsonl(candidate))
    return records


def analyse(records: dict, store: LabelStore, split_filter: str | None) -> dict:
    totals = {
        "images": 0,
        "gt_boxes": 0,
        "pred_boxes": 0,
        "matched": 0,
        "weak": 0,
        "missing": 0,          # labels the model failed to find at all
        "duplicate": 0,        # extra boxes sitting on an already-labelled player
        "missed_player": 0,    # extra boxes that are neither label nor duplicate
        "frames_disagree": 0,
        "sum_disagreement": 0.0,
    }
    # The headline failure mode: frame has 2 labelled players but the model
    # emits fewer than 2 boxes.
    two_gt_one_pred = 0
    two_gt_zero_pred = 0
    one_gt_multi_pred = 0

    per_kind: dict[str, int] = {}

    for image_id, rec in records.items():
        split = image_id.split("/", 1)[0]
        if split_filter and split != split_filter:
            continue
        width, height = rec.get("width"), rec.get("height")
        if not width or not height:
            continue

        raw = rec.get("predictions", [])
        preds = postprocess(raw, width, height, NMS_IOU, MAX_BOXES) if raw else []

        try:
            gt = [b.as_dict() for b in store.load(image_id).boxes]
        except Exception:
            continue

        metrics = evaluate(gt, preds, width, height)

        totals["images"] += 1
        totals["gt_boxes"] += metrics["gt_count"]
        totals["pred_boxes"] += metrics["pred_count"]
        totals["matched"] += metrics["matched"]
        totals["weak"] += metrics["weak"]
        totals["missing"] += metrics["missing"]
        totals["duplicate"] += metrics["duplicate"]
        totals["missed_player"] += metrics["missed_player"]
        totals["sum_disagreement"] += metrics["disagreement"]
        if metrics["disagreement"] > 0:
            totals["frames_disagree"] += 1

        n_gt, n_pred = metrics["gt_count"], metrics["pred_count"]
        if n_gt >= 2 and n_pred == 1:
            two_gt_one_pred += 1
        elif n_gt >= 2 and n_pred == 0:
            two_gt_zero_pred += 1
        elif n_gt == 1 and n_pred > 1:
            one_gt_multi_pred += 1

        # classify() gives the short "why is this interesting" label.
        from predictions import classify
        kind = classify(metrics) if metrics["disagreement"] > 0 else "agree"
        per_kind[kind] = per_kind.get(kind, 0) + 1

    return {
        "totals": totals,
        "two_gt_one_pred": two_gt_one_pred,
        "two_gt_zero_pred": two_gt_zero_pred,
        "one_gt_multi_pred": one_gt_multi_pred,
        "per_kind": per_kind,
    }


def pct(num: int, den: int) -> str:
    return f"{(100.0 * num / den):.2f}%" if den else "n/a"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path,
                   default=Path(".staging/ssbu_players_colab"))
    p.add_argument("--runs", nargs="+", required=True,
                   help="Run names to compare.")
    p.add_argument("--predictions-root", type=Path, default=None,
                   help="Dir holding <run>/<split>.jsonl (defaults to search).")
    p.add_argument("--split", default=None, help="Restrict to train/val.")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    store = LabelStore(args.dataset)
    drive = Path.home() / "Library/CloudStorage"
    search_roots = [
        args.predictions_root,
        Path(".staging/exports/predictions"),
        Path("exports/predictions"),
    ]
    for base in list(search_roots):
        if base is None:
            continue
        if not base.exists():
            hits = list(drive.glob(f"*/My Drive/bros-analysis/exports/predictions"))
            search_roots.extend(hits)
            break

    results = {}
    for run in args.runs:
        found = None
        for base in search_roots:
            if base is None:
                continue
            cand = base / run
            if cand.exists():
                found = cand
                break
        if found is None:
            print(f"  !! predictions for {run} not found; skipped")
            continue
        records = load_run(found)
        results[run] = analyse(records, store, args.split)

    if not results:
        print("No runs analysed.")
        return 2

    print("=" * 74)
    print(f"PREDICTION RUN COMPARISON{'  (split=' + args.split + ')' if args.split else ''}")
    print("=" * 74)
    keys = ["images", "gt_boxes", "pred_boxes", "matched", "weak",
            "missing", "duplicate", "missed_player", "frames_disagree"]
    hdr = "metric".ljust(20) + "".join(r.ljust(18) for r in results)
    print(hdr)
    print("-" * len(hdr))
    for k in keys:
        row = k.ljust(20)
        for r in results:
            row += str(results[r]["totals"][k]).ljust(18)
        print(row)

    print()
    print("mean disagreement".ljust(20) + "".join(
        f"{r['totals']['sum_disagreement'] / max(r['totals']['images'], 1):.4f}".ljust(18)
        for r in results.values()))
    print("frames w/ any disagreement".ljust(20) + "".join(
        pct(r["totals"]["frames_disagree"], r["totals"]["images"]).ljust(18)
        for r in results.values()))

    print()
    print("HEADLINE FAILURE MODES")
    print("-" * len(hdr))
    for label, key in (
        ("2 labels, 1 box", "two_gt_one_pred"),
        ("2 labels, 0 boxes", "two_gt_zero_pred"),
        ("1 label, >1 boxes", "one_gt_multi_pred"),
    ):
        print(label.ljust(20) + "".join(str(r[key]).ljust(18) for r in results.values()))

    print()
    print("PER-KIND COUNTS (why a frame is flagged)")
    print("-" * len(hdr))
    kinds = sorted({k for r in results.values() for k in r["per_kind"]})
    for kind in kinds:
        print(kind.ljust(20) + "".join(
            str(r["per_kind"].get(kind, 0)).ljust(18) for r in results.values()))

    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
