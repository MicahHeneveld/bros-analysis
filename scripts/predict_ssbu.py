#!/usr/bin/env python3
"""Run a trained checkpoint over a split and record raw predictions for review.

This intentionally stores ONLY raw predictions. Disagreement against labels is
recomputed by the review app against whatever the labels currently say, so
fixing a label immediately updates that image's score instead of leaving a stale
number behind.

Typical Colab usage (after mounting Drive):

  !pip -q install -U ultralytics
  !python /content/drive/MyDrive/bros-analysis/scripts/predict_ssbu.py --split val

Output:
  <root>/exports/predictions/<run>/<split>.jsonl   one line per image
  <root>/exports/predictions/<run>/summary.json    run metadata

Then point the local review app at it:
  python app/server.py --dataset .staging/ssbu_players_colab \
      --predictions "<drive>/exports/predictions/smash_player_colab_v2"
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# ``predictions.py`` lives in ``app/`` in the repo, but is copied next to this
# script on Drive so Colab can import it. Accept either layout, so the same file
# runs locally from a checkout and on Colab without a PYTHONPATH.
if not (_HERE / "predictions.py").exists():
    _repo_app = _HERE.parent / "app"
    if (_repo_app / "predictions.py").exists():
        sys.path.insert(0, str(_repo_app))

from train_ssbu import DEFAULT_COLAB_ROOT, ensure_ultralytics, stage_dataset  # noqa: E402
from predictions import MAX_BOXES, NMS_IOU, postprocess  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict with a trained SSBU checkpoint.")
    p.add_argument("--root", type=Path, default=DEFAULT_COLAB_ROOT,
                   help="Drive bros-analysis root.")
    p.add_argument("--run-name", default="smash_player_colab_v2",
                   help="Run folder under runs/ holding weights/best.pt.")
    p.add_argument("--weights", type=Path, default=None,
                   help="Explicit weight path (overrides --run-name).")
    p.add_argument("--dataset", default="ssbu_players_colab",
                   help="Dataset directory name under root.")
    p.add_argument("--split", default="val", choices=["train", "val", "all"],
                   help="Which split to run over.")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Confidence threshold. Keep low so weak detections are visible.")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    p.add_argument("--post-nms-iou", type=float, default=NMS_IOU,
                   help=("Second suppression pass at this IoU to merge duplicate "
                         "boxes on one character (default %.2f). The detector's own "
                         "NMS at --iou is too loose and emits doubles." % NMS_IOU))
    p.add_argument("--max-boxes", type=int, default=MAX_BOXES,
                   help=("Hard cap on boxes per frame (default %d; 0 disables). "
                         "SSBU is strictly 1v1 here." % MAX_BOXES))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--chunk", type=int, default=64,
                   help=("Infer this many images per call, releasing memory in "
                         "between. Bounds peak RAM; the whole split in one call "
                         "can reach tens of GB on Apple Silicon."))
    p.add_argument("--device", default=None,
                   help=("Compute device, e.g. 'mps' (Apple Silicon), 'cuda', 'cpu'. "
                         "Omit to let Ultralytics auto-select (which picks CPU on Mac)."))
    p.add_argument("--limit", type=int, default=0, help="Stop after N images (smoke test).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output directory (defaults to <root>/exports/predictions/<run>).")
    p.add_argument("--no-stage", action="store_true",
                   help="Read images directly from Drive. Slow.")
    p.add_argument("--skip-install", action="store_true",
                   help="Do not pip-install ultralytics (assume already present).")
    return p.parse_args()


def find_images(dataset_dir: Path, split: str) -> list[tuple[str, Path]]:
    """Return ``(image_id, path)`` pairs for the requested split."""
    splits = ["train", "val"] if split == "all" else [split]
    found: list[tuple[str, Path]] = []
    for s in splits:
        img_dir = dataset_dir / "images" / s
        if not img_dir.exists():
            continue
        for p in sorted(img_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                found.append((f"{s}/{p.stem}", p))
    return found


def on_drive(path: Path) -> bool:
    return str(path).startswith("/content/drive")


def _release_mps_cache() -> None:
    """Drop cached MPS allocations after a chunk.

    PyTorch's MPS allocator caches freed blocks for reuse, so a long run can
    ratchet its footprint upward even though each batch is small. Releasing the
    cache between chunks keeps the peak flat. No-op off MPS.
    """
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def main() -> int:
    args = parse_args()
    root = args.root.expanduser()

    weights = args.weights or (root / "runs" / args.run_name / "weights" / "best.pt")
    if not weights.exists():
        print(f"ERROR: weights not found: {weights}")
        print("Train first, or pass --weights / --run-name.")
        return 2

    dataset_dir = root / args.dataset
    if not dataset_dir.exists():
        print(f"ERROR: dataset not found: {dataset_dir}")
        return 2

    # A fresh Colab runtime has no ultralytics. Install it up front rather than
    # relying on a separate notebook cell, so the script works however it is
    # invoked. Doing this before staging means a failed install does not cost
    # several minutes of unzipping first. train_ssbu.py does the same.
    try:
        ensure_ultralytics(args.skip_install)
    except Exception as exc:  # subprocess.CalledProcessError, OSError
        print(f"ERROR: could not install ultralytics: {exc}")
        print("If the runtime has no network, run this in a cell first:")
        print("  !pip -q install -U ultralytics")
        print("then re-run with --skip-install")
        return 2
    from ultralytics import YOLO

    # Reading thousands of images over Drive at ~0.5 MB/s is slow enough to
    # matter, so stage onto local disk first when running in Colab.
    if on_drive(root) and not args.no_stage:
        dataset_dir = stage_dataset(dataset_dir, root / f"{args.dataset}.zip")

    pairs = find_images(dataset_dir, args.split)
    if not pairs:
        print(f"ERROR: no images found for split {args.split!r} under {dataset_dir}")
        return 2
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"Running {weights.name} over {len(pairs)} {args.split} image(s)", flush=True)

    model = YOLO(str(weights))
    names = model.names

    out_dir = args.out or (root / "exports" / "predictions" / args.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.split}.jsonl"

    started = time.time()
    written = 0
    total_boxes = 0
    kept_boxes = 0
    merged = 0

    def write_result(fh, image_id: str, result) -> None:
        """Append one image's predictions, updating the running counters."""
        nonlocal written, total_boxes, kept_boxes, merged
        h, w = result.orig_shape[:2]
        boxes = []
        xywhn = result.boxes.xywhn
        confs = result.boxes.conf
        clss = result.boxes.cls
        for i in range(len(xywhn)):
            xc, yc, bw, bh = (float(v) for v in xywhn[i].tolist())
            boxes.append({
                "cls": int(clss[i].item()),
                "conf": round(float(confs[i].item()), 4),
                "xc": round(xc, 6),
                "yc": round(yc, 6),
                "w": round(bw, 6),
                "h": round(bh, 6),
            })
        total_boxes += len(boxes)
        # Store BOTH raw and cleaned. The app re-derives the cleaned set from
        # raw with the same shared function, so keeping raw means a later
        # threshold change needs no re-run.
        cleaned = postprocess(boxes, w, h, args.post_nms_iou, args.max_boxes)
        kept_boxes += len(cleaned)
        merged += len(boxes) - len(cleaned)
        fh.write(json.dumps({
            "image_id": image_id,
            "width": int(w),
            "height": int(h),
            "predictions": boxes,
            "predictions_clean": cleaned,
        }) + "\n")
        written += 1

    with open(out_path, "w", encoding="utf-8") as fh:
        predict_kwargs = dict(
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            batch=args.batch,
            stream=True,
            verbose=False,
        )
        if args.device:
            predict_kwargs["device"] = args.device

        # Infer in fixed-size chunks. Handing the whole split to model.predict
        # at once lets Ultralytics hold state for every image: 3,857 frames
        # reached a 23 GB physical footprint on a 24 GB machine, which is what
        # kernel-panicked the host on the first attempt. Chunking bounds the
        # peak and makes a large run survivable.
        chunk = max(1, args.chunk)
        for start in range(0, len(pairs), chunk):
            window = pairs[start : start + chunk]
            results = model.predict(
                source=[str(p) for _, p in window], **predict_kwargs
            )
            for (image_id, _path), result in zip(window, results):
                write_result(fh, image_id, result)
                if written % 100 == 0:
                    print(f"  {written}/{len(pairs)}", flush=True)
            del results
            gc.collect()
            _release_mps_cache()
            print(
                f"  [{min(start + chunk, len(pairs))}/{len(pairs)}] chunk done",
                flush=True,
            )

    elapsed = time.time() - started
    summary = {
        "run": args.run_name,
        "weights": str(weights),
        "split": args.split,
        "images": written,
        "boxes": total_boxes,
        "boxes_after_postprocess": kept_boxes,
        "merged_as_duplicates": merged,
        "mean_boxes": round(total_boxes / written, 3) if written else 0.0,
        "conf": args.conf,
        "nms_iou": args.iou,
        "post_nms_iou": args.post_nms_iou,
        "max_boxes": args.max_boxes,
        "imgsz": args.imgsz,
        "class_names": {int(k): v for k, v in names.items()},
        "seconds": round(elapsed, 1),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nWrote {written} predictions ({total_boxes} raw boxes) in {elapsed:.1f}s")
    print(f"  post-process: merged {merged} duplicate(s), kept {kept_boxes} boxes")
    print(f"  {out_path}")
    print(f"  {out_dir / 'summary.json'}")
    print("\nNext, on the local machine:")
    print(f'  python app/server.py --dataset .staging/ssbu_players_colab \\')
    print(f'      --predictions "{out_dir}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
