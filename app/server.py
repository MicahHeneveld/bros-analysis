"""Local label review server. Stdlib only, no build step.

Launch:
    python app/server.py --dataset /path/to/ssbu_players_colab

Then open http://localhost:8770
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import mimetypes
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

import board as board_mod  # noqa: E402
import jobs as jobs_mod  # noqa: E402
from label_io import Box, LabelStore  # noqa: E402
from predictions import (  # noqa: E402
    MAX_BOXES,
    NMS_IOU,
    Predictions,
    classify,
    evaluate,
    postprocess,
)
from review_store import REASON_CODES, ReviewStore  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CLOUD_MARKERS = ("/CloudStorage/", "/content/drive", "/My Drive/")
UPLOAD_DIR = REPO_ROOT / ".staging" / "uploads"
MAX_UPLOAD_BYTES = 8 * 1024 ** 3


def parse_range(header: str | None, size: int) -> tuple[int, int, bool] | None:
    """Parse a single-range ``Range`` header against a file of ``size`` bytes.

    Returns ``(start, end, partial)`` with an inclusive ``end``, or ``None``
    when the requested range cannot be satisfied (the caller must answer 416).
    An absent or unrecognised header means "send the whole file".

    Kept separate from the handler so the seeking behaviour - the part that
    decides whether the browser can scrub a video - is directly testable.
    """
    if not header or not header.startswith("bytes="):
        return 0, size - 1, False
    spec = header[len("bytes="):].split(",")[0].strip()
    first, _, last = spec.partition("-")
    start, end = 0, size - 1
    try:
        if first:
            start = int(first)
            if last:
                end = min(int(last), size - 1)
        elif last:
            # "bytes=-500" means the final 500 bytes.
            start = max(0, size - int(last))
        else:
            return 0, size - 1, False
    except ValueError:
        return 0, size - 1, False
    if start > end or start >= size:
        return None
    return start, end, True


class AppState:
    def __init__(
        self,
        dataset_root: Path,
        predictions_path: Path | None = None,
        nms_iou: float = NMS_IOU,
        max_boxes: int = MAX_BOXES,
        videos: list[Path] | None = None,
    ):
        self.dataset_root = dataset_root
        self.labels = LabelStore(dataset_root)
        self.reviews = ReviewStore(dataset_root)
        # Predictions come from a Colab run. Disagreement is always recomputed
        # against the labels currently on disk, so fixing a label immediately
        # changes that image's score instead of leaving a stale number.
        self.predictions = Predictions(predictions_path)
        # A wrong --predictions path used to fail silently: the app came up
        # looking normal but with the overlay and disagreement ranking simply
        # absent, which reads as "the model found nothing" rather than "you
        # pointed me at nothing". Say so at startup instead.
        if predictions_path and not self.predictions.available:
            shown = Path(predictions_path).expanduser()
            print(f"WARNING: no predictions loaded from {shown}")
            if not shown.exists():
                print("         that path does not exist.")
                parent = shown.parent
                if parent.is_dir():
                    near = sorted(p.name for p in parent.iterdir() if p.is_dir())
                    if near:
                        print(f"         available under {parent}:")
                        for name in near[:10]:
                            print(f"           {name}")
            else:
                print("         path exists but holds no readable *.jsonl.")
            print("         The overlay and disagreement ranking will be disabled.\n")
        # Duplicate detections and >max_boxes per frame are corrected on read,
        # so the same stored predictions are scored and displayed consistently.
        self.nms_iou = nms_iou
        self.max_boxes = max_boxes
        # Source videos for playback mode. Discovered from the ingest manifest
        # (which records the path it came from) plus any explicit --video.
        self.videos = self._discover_videos(videos or [])
        self._timeline_cache: dict[str, dict] = {}
        self._samples_cache: list[dict] | None = None
        self._lock = threading.Lock()

    def _discover_videos(self, explicit: list[Path]) -> list[dict]:
        """Registered videos, in a stable order.

        Only these are ever streamed, so a client cannot ask the server to read
        an arbitrary file off disk.
        """
        found: list[dict] = []
        seen: set[str] = set()

        def candidates(path: Path) -> list[Path]:
            if path.is_absolute():
                return [path]
            # Ingest records the path it was given. Resolve relative entries
            # against the likely roots instead of guessing one.
            roots = [Path.cwd(), self.dataset_root, self.dataset_root.parent, self.dataset_root.parent.parent]
            return [root / path for root in roots]

        def add(path: Path, label: str, key: str, info: dict) -> None:
            for candidate in candidates(path):
                try:
                    resolved = candidate.expanduser().resolve()
                except OSError:
                    continue
                ident = str(resolved)
                if ident in seen or not resolved.is_file():
                    continue
                seen.add(ident)
                found.append(
                    {
                        "path": resolved,
                        "name": label or resolved.stem,
                        "key": key,
                        "duration": info.get("duration_s"),
                        "fps": info.get("fps"),
                        "width": info.get("width"),
                        "height": info.get("height"),
                    }
                )
                return

        for path in explicit:
            add(path, path.stem, path.stem, {})

        manifest_path = self.dataset_root / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except ValueError:
                manifest = {}
            for key, info in (manifest.get("sources") or {}).items():
                if isinstance(info, dict) and info.get("path"):
                    add(Path(info["path"]), key, key, info)
        return found

    def cleaned_preds(self, rec: dict) -> list[dict]:
        """Stored predictions with duplicates merged and the box count capped."""
        raw = rec.get("predictions", [])
        width = float(rec.get("width") or 0)
        height = float(rec.get("height") or 0)
        if not raw or not width or not height:
            return raw
        return postprocess(raw, width, height, self.nms_iou, self.max_boxes)

    # ----- sample index --------------------------------------------------

    def samples(self) -> list[dict]:
        """Index every image in the dataset with its label/review state."""
        if self._samples_cache is not None:
            return self._samples_cache

        rows: list[dict] = []
        # Discover splits from disk rather than hardcoding train/val/test, so a
        # freshly ingested video (which lands in its own split) is reviewable
        # without a code change. Sorted for a stable queue order.
        images_root = self.labels.images_dir
        split_names = sorted(
            d.name for d in images_root.iterdir() if d.is_dir()
        ) if images_root.exists() else []
        for split in split_names:
            img_dir = images_root / split
            if not img_dir.exists():
                continue
            for path in sorted(img_dir.iterdir()):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                    continue
                image_id = f"{split}/{path.stem}"
                label_path = self.labels.labels_dir / split / f"{path.stem}.txt"
                gt_boxes: list[Box] = []
                malformed = 0
                if label_path.exists():
                    parsed = self.labels.load(image_id)
                    gt_boxes = parsed.boxes
                    malformed = len([1 for _, t in parsed.malformed if t.strip()])
                review = self.reviews.get(image_id)

                pred = self.predictions.get(image_id)
                metrics: dict = {}
                pred_boxes = 0
                if pred["available"]:
                    rec = self.predictions.records[image_id]
                    cleaned = self.cleaned_preds(rec)
                    pred_boxes = len(cleaned)
                    # Reuse the dimensions recorded at inference time so the
                    # index does not have to decode 3,857 images to start up.
                    pw = float(rec.get("width") or 0)
                    ph = float(rec.get("height") or 0)
                    if pw and ph:
                        metrics = evaluate(
                            [b.as_dict() for b in gt_boxes], cleaned, pw, ph
                        )

                rows.append(
                    {
                        "image_id": image_id,
                        "split": split,
                        "stem": path.stem,
                        "boxes": len(gt_boxes),
                        "malformed": malformed,
                        "status": review["status"],
                        "reasons": review.get("reasons", []),
                        "has_labels": label_path.exists() and len(gt_boxes) > 0,
                        "pred_boxes": pred_boxes,
                        "has_prediction": pred["available"],
                        "disagreement": metrics.get("disagreement", 0.0),
                        "kind": classify(metrics) if metrics else "",
                        "over_labelled": len(gt_boxes) > self.max_boxes,
                    }
                )
        self._samples_cache = rows
        return rows

    def good_without_labels(self) -> list[str]:
        """Frames marked good that have no boxes on disk.

        A frame cannot be both "correct" and unlabelled: the review state says a
        human approved boxes that were never written. This happens when a review
        decision lands before the label autosave (and it did, for 383 frames
        before the save order was fixed), so the app surfaces and repairs it
        rather than silently paying for work that is not in the dataset.
        """
        return [
            r["image_id"]
            for r in self.samples()
            if r.get("status") == "good" and not r.get("has_labels")
        ]

    def invalidate(self) -> None:
        self._samples_cache = None

    def prediction_stats(self) -> dict:
        """Prediction summary with disagreement aggregated over live labels.

        Scores live here rather than in ``Predictions.summary`` because they are
        recomputed against the current labels; editing a label changes them.
        """
        base = self.predictions.summary()
        if not base.get("available"):
            return base

        rows = [r for r in self.samples() if r.get("has_prediction")]
        if not rows:
            return base

        scores = sorted((float(r.get("disagreement", 0.0)) for r in rows), reverse=True)
        kinds: dict[str, int] = {}
        for row in rows:
            k = row.get("kind") or "agree"
            kinds[k] = kinds.get(k, 0) + 1

        return {
            **base,
            "scored": len(rows),
            "kinds": kinds,
            "phase": "post-processed (dedupe + cap)",
            "postprocess": {"nms_iou": self.nms_iou, "max_boxes": self.max_boxes},
            "mean_disagreement": round(sum(scores) / len(scores), 4),
            "max_disagreement": scores[0],
            "flagged": sum(1 for s in scores if s >= 0.25),
        }

    def find(self, image_id: str) -> dict | None:
        for row in self.samples():
            if row["image_id"] == image_id:
                return row
        return None

    def video_timeline(self, index: int) -> dict:
        """Every sampled frame of a video, with its time and boxes.

        The ingest samples at a target fps and drops menus, so the timeline is
        sparse: there are gaps where no frame was kept. The player uses the
        time of the nearest kept frame, which means overlays vanish exactly
        where the dataset has no data instead of showing a stale box.
        """
        videos = self.videos
        if index < 0 or index >= len(videos):
            raise ValueError("unknown video")
        video = videos[index]
        cache_key = str(video["path"])
        cached = self._timeline_cache.get(cache_key)
        if cached is not None:
            return cached

        source_key = video.get("key")
        frames = []
        for row in self.samples():
            image_id = row["image_id"]
            # Read the raw record: Predictions.get() returns only the fields
            # the review panel needs, and drops time_s / frame_index.
            rec = self.predictions.records.get(image_id) or {}
            if source_key:
                # Skip frames explicitly attributed to a different source when
                # several videos share one dataset. Records with no source (a
                # hand-built dataset) are still included.
                src = rec.get("source")
                if src is not None and src != source_key:
                    continue
            frames.append(
                {
                    "image_id": image_id,
                    "t": rec.get("time_s"),
                    "frame": rec.get("frame_index"),
                    "labels": [
                        {"cls": b.cls, "xc": b.xc, "yc": b.yc, "w": b.w, "h": b.h}
                        for b in self.labels.load(image_id).boxes
                    ],
                    "model": self.cleaned_preds(rec) if rec else [],
                }
            )
        frames.sort(key=lambda f: (f["t"] is None, f["t"] or 0, f["frame"] or 0))
        payload = {
            "video": {
                "index": index,
                "name": video["name"],
                "duration": video.get("duration"),
                "fps": video.get("fps"),
            },
            "frames": frames,
        }
        self._timeline_cache[cache_key] = payload
        return payload

    def timeline_span(self, payload: dict, image_ids: set[str]) -> dict | None:
        """The stretch of a video a bounty's frames cover.

        A bounty slice is a set of image ids and the timeline is sorted by time,
        so the span is just the first and last sampled second among the frames
        this video actually holds. Returns None when none of them are here.
        """
        times = [
            float(frame["t"])
            for frame in payload.get("frames", [])
            if isinstance(frame.get("t"), (int, float))
            and (not image_ids or frame.get("image_id") in image_ids)
        ]
        if not times:
            return None
        return {"start": min(times), "end": max(times), "count": len(times)}


def sort_samples(rows: list[dict], sort: str) -> list[dict]:
    """Order the review queue. Extracted so the default order is testable."""
    if sort == "unreviewed":
        # Unreviewed first, then worst disagreement.
        #
        # The naive orders strand the backlog. Sorting by disagreement alone
        # puts the highest-scoring images first, but those are exactly the ones
        # already triaged, so a reviewer re-walks finished work. It is worse
        # when only some splits have predictions: the train split has none at
        # all, so those rows all score zero and, with a stable sort over a
        # train-then-val scan, they land *ahead* of the un-reviewed val rows
        # that do have predictions.
        return sorted(
            rows,
            key=lambda r: (
                r.get("status") != "unreviewed",
                -r.get("disagreement", 0.0),
            ),
        )
    if sort == "disagreement":
        return sorted(rows, key=lambda r: r.get("disagreement", 0.0), reverse=True)
    if sort == "agreement":
        return sorted(rows, key=lambda r: r.get("disagreement", 0.0))
    if sort == "name":
        return sorted(rows, key=lambda r: (r.get("split", ""), r.get("stem", "")))
    return list(rows)


def sample_matches(
    row: dict,
    *,
    split: str = "all",
    status: str = "all",
    search: str = "",
    kind: str = "all",
    flag: str = "all",
    min_score: float = 0.0,
) -> bool:
    """Whether a sample index row passes the requested filters.

    Extracted from the request handler so the flag semantics - which are the
    reviewer's primary navigation tool - can be tested directly.
    """
    if split != "all" and row.get("split") != split:
        return False
    if status != "all" and row.get("status") != status:
        return False
    if search and search not in row.get("stem", "").lower():
        return False
    if kind != "all" and row.get("kind") != kind:
        return False

    if flag == "over_labelled" and not row.get("over_labelled"):
        return False

    if flag == "inconsistent":
        # An image can be marked reviewed-empty while still holding boxes,
        # because the Empty button does not clear the label file and a later
        # edit does not clear the status.
        if not (row.get("status") == "empty" and row.get("boxes", 0) > 0):
            return False

    if flag == "no_labels" and row.get("boxes", 0) > 0:
        return False

    if flag == "malformed" and not row.get("malformed"):
        return False

    if flag == "conflicting":
        # The reviewer marked this image missing_box ("a player has no box" - a
        # box should be ADDED) while the model reports an unfound_label (a box
        # exists that it cannot find). Those are opposite claims, so the image
        # needs another look. Not strictly a contradiction - both can be true of
        # two different players - but it is the pattern most likely to come from
        # the two vocabularies having been confused.
        if not (
            "missing_box" in (row.get("reasons") or [])
            and row.get("kind") == "unfound_label"
        ):
            return False

    if min_score > 0 and row.get("disagreement", 0.0) < min_score:
        return False

    return True


def is_cloud_path(path: Path) -> bool:
    text = str(path)
    return any(marker in text for marker in CLOUD_MARKERS)


class DatasetRegistry:
    """Lazily-built AppState per dataset, so one server serves the board.

    The landing screen switches datasets without restarting the process, so
    states are created on first use and then reused. Building a state scans the
    dataset, so doing that eagerly for every dataset on startup would make the
    server slow to come up for no benefit.
    """

    def __init__(
        self,
        repo_root: Path,
        explicit: Path | None = None,
        *,
        predictions: Path | None = None,
        videos: list[Path] | None = None,
        nms_iou: float = NMS_IOU,
        max_boxes: int = MAX_BOXES,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.lock = threading.RLock()
        self._states: dict[str, AppState] = {}
        self._paths: dict[str, Path] = {}
        self.default_name = ""
        # CLI options only make sense for the dataset they were given with, so
        # they are applied to the explicit dataset and not to every state.
        self.explicit_name = explicit.name if explicit is not None else ""
        self.predictions = predictions
        self.videos = videos or []
        self.nms_iou = nms_iou
        self.max_boxes = max_boxes

        # An explicit --dataset may live outside data/datasets (a staging dir or
        # a Drive copy), so it is registered by name as well as discovered.
        for entry in board_mod.discover(self.repo_root):
            self._paths[entry["name"]] = entry["path"]
        if explicit is not None:
            root = Path(explicit).expanduser().resolve()
            self._paths[root.name] = root
            self.default_name = root.name
        elif self._paths:
            # Prefer a dataset that actually has frames: ssbu_players ships
            # labels but its images live in a release, so defaulting to it would
            # open an empty review queue.
            def usable(name: str) -> bool:
                images = self._paths[name] / "images"
                return images.is_dir() and any(
                    p.is_file() for split in images.glob("*") if split.is_dir()
                    for p in split.iterdir()
                )

            self.default_name = next(
                (name for name in self._paths if usable(name)),
                next(iter(self._paths)),
            )

    # ----- discovery -----------------------------------------------------
    def names(self) -> list[str]:
        return sorted(self._paths)

    def path_for(self, name: str) -> Path | None:
        return self._paths.get(name)

    def resolve(self, name: str | None) -> AppState:
        """State for ``name`` (or the default) - creating it on first use."""
        key = (name or "").strip() or self.default_name
        if not key:
            raise KeyError("no datasets available")
        with self.lock:
            if key in self._states:
                return self._states[key]
            # A dataset created by the Add-dataset job is not in the map yet.
            if key not in self._paths:
                for entry in board_mod.discover(self.repo_root):
                    self._paths[entry["name"]] = entry["path"]
            root = self._paths.get(key)
            if root is None:
                raise KeyError(f"unknown dataset: {key}")
            is_explicit = key == self.explicit_name
            state = AppState(
                root,
                find_predictions(root, self.predictions if is_explicit else None),
                nms_iou=self.nms_iou,
                max_boxes=self.max_boxes,
                videos=self.videos if is_explicit else None,
            )
            self._states[key] = state
            self._paths[key] = root
            if not self.default_name:
                self.default_name = key
            return state

    def invalidate(self, name: str | None = None) -> None:
        """Drop cached states so the next request rescans the dataset."""
        with self.lock:
            if name:
                state = self._states.pop(name, None)
                if state is not None:
                    state.invalidate()
            else:
                for state in self._states.values():
                    state.invalidate()

    def add(self, name: str, path: Path) -> None:
        with self.lock:
            self._paths[name] = Path(path)
            self._states.pop(name, None)


class Handler(SimpleHTTPRequestHandler):
    registry: DatasetRegistry = None  # type: ignore[assignment]
    board: board_mod.Board = None  # type: ignore[assignment]
    runner: jobs_mod.JobRunner = None  # type: ignore[assignment]
    state: AppState = None  # type: ignore[assignment]
    server_version = "LabelReview/1.0"

    def log_message(self, fmt: str, *args) -> None:  # quieter console
        if self.path.startswith("/api/"):
            sys.stderr.write(f"{self.command} {self.path} -> {args[1] if len(args) > 1 else ''}\n")

    # ----- helpers -------------------------------------------------------

    def send_json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, message: str, status: int = 400) -> None:
        self.send_json({"error": message}, status=status)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except ValueError:
            raise ValueError("request body is not valid JSON")

    def send_bytes(self, data: bytes, content_type: str, cache: str = "no-store") -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(data)

    def send_video(self, path: Path, content_type: str = "video/mp4") -> None:
        """Stream a video, honouring HTTP Range so the browser can seek.

        Without Range support the <video> element will not scrub: it either
        refuses to play or downloads the whole file before showing a frame.
        Only registered videos reach this method, never a client-supplied path.
        """
        try:
            size = path.stat().st_size
        except OSError:
            return self.send_error_json("video not readable", 404)

        parsed = parse_range(self.headers.get("Range"), size)
        if parsed is None:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        start, end, partial = parsed

        length = end - start + 1
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(path, "rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                chunk = fh.read(min(256 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)

    # ----- routing -------------------------------------------------------

    def _bind_state(self, query: dict) -> None:
        """Point this request at the dataset named in ``?dataset=``.

        Each handler instance serves exactly one request, so assigning the
        instance attribute is enough to scope every existing endpoint without
        threading a state parameter through all of them. Without a name the
        default dataset is used, which keeps older links and CLI usage working.
        """
        name = (query.get("dataset") or [""])[0].strip()
        if not name:
            return
        self.state = self.registry.resolve(name)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        # Percent-decode the path before routing. Image ids are derived from
        # filenames, and a video stem routinely contains spaces, '#', '(' and
        # ')'. The browser percent-encodes those, so without this every such
        # image id resolves to nothing and 404s as "unknown image".
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)

        if path.startswith("/api/"):
            try:
                self._bind_state(query)
                return self.handle_api_get(path, query)
            except KeyError as exc:
                return self.send_error_json(str(exc).strip("'\""), 404)
            except ValueError as exc:
                return self.send_error_json(str(exc), 400)
            except Exception as exc:  # noqa: BLE001
                return self.send_error_json(f"server error: {exc}", 500)

        return self.serve_static(path)

    def do_PUT(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if not path.startswith("/api/"):
            return self.send_error_json("not found", 404)
        try:
            self._bind_state(parse_qs(parsed.query))
            return self.handle_api_put(path)
        except KeyError as exc:
            return self.send_error_json(str(exc).strip("'\""), 404)
        except ValueError as exc:
            return self.send_error_json(str(exc), 400)
        except Exception as exc:  # noqa: BLE001
            return self.send_error_json(f"server error: {exc}", 500)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if not path.startswith("/api/"):
            return self.send_error_json("not found", 404)
        query = parse_qs(parsed.query)
        try:
            # Uploads carry a raw body, not JSON, so they are handled before the
            # JSON-reading dispatch below.
            if path == "/api/board/upload":
                return self.handle_upload(query)
            self._bind_state(query)
            return self.handle_api_post(path, query)
        except KeyError as exc:
            return self.send_error_json(str(exc).strip("'\""), 404)
        except ValueError as exc:
            return self.send_error_json(str(exc), 400)
        except Exception as exc:  # noqa: BLE001
            return self.send_error_json(f"server error: {exc}", 500)

    # ----- static --------------------------------------------------------

    def serve_static(self, path: str) -> None:
        # `/` is the landing screen; the review UI lives at `/review`. Keeping
        # them as separate documents means the review app's DOM is never built
        # on the landing screen, so its keyboard handlers and canvas do not run.
        routes = {
            "": "landing.html",
            "index.html": "landing.html",
            # Label mode (bounty-scoped, writable) and view mode (read-only
            # browsing of every frame plus the source video) are the same
            # document; `?mode=` decides which one the JS boots into.
            "review": "index.html",
            "review/": "index.html",
            "view": "index.html",
            "view/": "index.html",
        }
        rel = routes.get(path.lstrip("/"), path.lstrip("/"))
        if not rel:
            rel = "landing.html"
        target = (STATIC_DIR / rel).resolve()
        if STATIC_DIR not in target.parents and target != STATIC_DIR:
            return self.send_error_json("forbidden", 403)
        if not target.exists() or not target.is_file():
            return self.send_error_json("not found", 404)
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self.send_bytes(target.read_bytes(), ctype)

    # ----- api: GET ------------------------------------------------------

    def handle_api_get(self, path: str, query: dict) -> None:
        state = self.state

        # ----- board: not dataset-scoped -----
        if path == "/api/board":
            payload = self.board.snapshot()
            payload["active_job"] = self.runner.active()
            payload["repo"] = str(REPO_ROOT)
            payload["default_dataset"] = self.registry.default_name
            return self.send_json(payload)

        if path == "/api/jobs":
            return self.send_json({
                "jobs": self.runner.recent(),
                "active": self.runner.active(),
            })

        if path.startswith("/api/jobs/"):
            job = self.runner.get(path[len("/api/jobs/"):])
            if job is None:
                return self.send_error_json("unknown job", 404)
            return self.send_json(job.as_dict())

        if path == "/api/dataset":
            samples = state.samples()
            splits = {}
            for row in samples:
                s = splits.setdefault(row["split"], {"images": 0, "boxed": 0, "background": 0})
                s["images"] += 1
                if row["boxes"] > 0:
                    s["boxed"] += 1
                else:
                    s["background"] += 1
            counts = state.reviews.counts(len(samples))
            counts["total"] = len(samples)
            return self.send_json(
                {
                    "dataset": state.dataset_root.name,
                    "root": str(state.dataset_root),
                    "splits": splits,
                    "review": counts,
                    "predictions": state.prediction_stats(),
                    "reason_codes": REASON_CODES,
                    # Surfaced so the board can offer a one-click repair instead
                    # of leaving unlabelled "good" frames undetected.
                    "integrity": {
                        "good_without_labels": len(state.good_without_labels()),
                    },
                }
            )

        if path == "/api/samples":
            samples = state.samples()
            # Opening a bounty from the landing screen scopes the queue to that
            # bounty's frame slice. Without this the progress bar would count the
            # whole dataset, so "0 of 1654" would sit under a 250-frame bounty.
            bounty_id = (query.get("bounty") or [""])[0].strip()
            if bounty_id:
                bounty = next(
                    (b for b in self.board.bounties() if b["id"] == bounty_id), None
                )
                if bounty is None:
                    return self.send_error_json(f"unknown bounty: {bounty_id}", 404)
                if bounty.get("slice"):
                    allowed = set(bounty["slice"])
                    samples = [r for r in samples if r["image_id"] in allowed]
            split = (query.get("split") or ["all"])[0]
            status = (query.get("status") or ["all"])[0]
            search = (query.get("q") or [""])[0].strip().lower()
            sort = (query.get("sort") or ["unreviewed"])[0]
            kind = (query.get("kind") or ["all"])[0]
            flag = (query.get("flag") or ["all"])[0]
            min_score = float((query.get("min_score") or ["0"])[0])
            offset = int((query.get("offset") or ["0"])[0])
            limit = int((query.get("limit") or ["200"])[0])

            filtered = [
                r
                for r in samples
                if sample_matches(
                    r,
                    split=split,
                    status=status,
                    search=search,
                    kind=kind,
                    flag=flag,
                    min_score=min_score,
                )
            ]
            filtered = sort_samples(filtered, sort)
            page = filtered[offset : offset + limit]
            # The reviewed tally has to be computed over the whole filtered set,
            # not the page, or a reviewer with 200 of 1654 loaded sees a progress
            # bar that appears to jump backwards as they scroll.
            reviewed = sum(1 for r in filtered if r.get("status") != "unreviewed")
            return self.send_json(
                {
                    "total": len(filtered),
                    "reviewed": reviewed,
                    "offset": offset,
                    "limit": limit,
                    "items": page,
                }
            )

        if path == "/api/predictions":
            return self.send_json(state.prediction_stats())

        if path == "/api/videos":
            return self.send_json(
                {
                    "videos": [
                        {
                            "index": i,
                            "name": v["name"],
                            "duration": v.get("duration"),
                            "fps": v.get("fps"),
                            "width": v.get("width"),
                            "height": v.get("height"),
                        }
                        for i, v in enumerate(state.videos)
                    ]
                }
            )

        if path.startswith("/api/video/"):
            rest = path[len("/api/video/") :]
            num, _, sub = rest.partition("/")
            try:
                index = int(num)
            except ValueError:
                return self.send_error_json("bad video index", 400)
            if index < 0 or index >= len(state.videos):
                return self.send_error_json("unknown video", 404)
            if sub == "timeline":
                try:
                    payload = state.video_timeline(index)
                except ValueError as exc:
                    return self.send_error_json(str(exc), 404)
                # Opening the player from a bounty asks for that slice's span so
                # the scrubber can mark it and the player can stop at its end.
                bounty_id = (query.get("bounty") or [""])[0].strip()
                if bounty_id:
                    bounty = next(
                        (b for b in self.board.bounties() if b["id"] == bounty_id), None
                    )
                    if bounty is None:
                        return self.send_error_json(f"unknown bounty: {bounty_id}", 404)
                    span = state.timeline_span(payload, set(bounty.get("slice") or []))
                    # Copy: the cached payload is shared by every bounty request.
                    payload = {
                        **payload,
                        "bounty": {
                            "id": bounty_id,
                            "title": bounty.get("title", ""),
                            **(span or {"count": 0}),
                        },
                    }
                return self.send_json(payload)
            if sub == "":
                return self.send_video(state.videos[index]["path"])
            return self.send_error_json("unknown video endpoint", 404)

        if path.startswith("/api/sample/"):
            image_id = path[len("/api/sample/") :]
            return self.get_sample(image_id)

        if path.startswith("/api/image/"):
            image_id = path[len("/api/image/") :]
            img = state.labels.image_path(image_id)
            if not img.exists():
                return self.send_error_json("image not found", 404)
            ctype = mimetypes.guess_type(str(img))[0] or "image/jpeg"
            return self.send_bytes(img.read_bytes(), ctype, cache="public, max-age=86400")

        if path.startswith("/api/thumb/"):
            image_id = path[len("/api/thumb/") :]
            return self.get_thumb(image_id)

        if path == "/api/journal":
            return self.send_json({"entries": state.labels.recent_edits(limit=200)})

        if path == "/api/export/review.csv":
            return self.export_review_csv()

        return self.send_error_json("unknown endpoint", 404)

    def get_sample(self, image_id: str) -> None:
        state = self.state
        row = state.find(image_id)
        if row is None:
            return self.send_error_json(f"unknown image: {image_id}", 404)

        parsed = state.labels.load(image_id)
        img_path = state.labels.image_path(image_id)
        width, height = image_size(img_path)
        review = state.reviews.get(image_id)

        pred = state.predictions.get(image_id)
        metrics: dict = {}
        cleaned: list[dict] = []
        if pred["available"] and width and height:
            cleaned = state.cleaned_preds(state.predictions.records[image_id])
            metrics = evaluate(
                [b.as_dict() for b in parsed.boxes], cleaned, width, height
            )

        return self.send_json(
            {
                **row,
                "width": width,
                "height": height,
                "boxes": [b.as_dict() for b in parsed.boxes],
                "box_count": len(parsed.boxes),
                "malformed": [t for _, t in parsed.malformed if t.strip()],
                "warnings": parsed.warnings(),
                "review": review,
                "predictions": cleaned,
                "predictions_raw": pred["predictions"],
                "predictions_available": pred["available"],
                "predictions_merged": max(0, len(pred["predictions"]) - len(cleaned)),
                "metrics": metrics,
                "kind": classify(metrics) if metrics else "",
            }
        )

    def get_thumb(self, image_id: str) -> None:
        img = self.state.labels.image_path(image_id)
        if not img.exists():
            return self.send_error_json("image not found", 404)
        data = make_thumb(img, width=160)
        if data is None:
            ctype = mimetypes.guess_type(str(img))[0] or "image/jpeg"
            return self.send_bytes(img.read_bytes(), ctype, cache="public, max-age=86400")
        return self.send_bytes(data, "image/jpeg", cache="public, max-age=86400")

    def export_review_csv(self) -> None:
        state = self.state
        samples = state.samples()
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["image_id", "split", "status", "reasons", "note", "boxes"])
        for row in samples:
            review = state.reviews.get(row["image_id"])
            writer.writerow(
                [
                    row["image_id"],
                    row["split"],
                    review["status"],
                    ";".join(review.get("reasons", [])),
                    review.get("note", ""),
                    row["boxes"],
                ]
            )
        data = buf.getvalue().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/csv")
        self.send_header("Content-Disposition", 'attachment; filename="review.csv"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # ----- board: upload ------------------------------------------------

    def handle_upload(self, query: dict) -> None:
        """Receive an MP4 as a raw body and start an ingest job.

        Raw body rather than multipart on purpose: parsing multipart with the
        stdlib is fiddly and error-prone around very large files, while a
        streamed body needs nothing but Content-Length. The filename arrives in
        a header because the body *is* the file.

        The bytes are written to disk before the job starts, so a job never
        sees a half-written video and a failed upload never creates a job.
        """
        raw_name = (query.get("name") or [""])[0].strip()
        header_name = self.headers.get("X-Filename") or ""
        stem = raw_name or Path(unquote(header_name)).stem

        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return self.send_error_json("empty upload", 400)
        if length > MAX_UPLOAD_BYTES:
            return self.send_error_json("file is too large for one upload", 413)
        if not stem:
            return self.send_error_json("missing dataset name", 400)

        name = jobs_mod.slugify(stem)
        publish = (query.get("publish") or ["0"])[0] in ("1", "true", "yes")
        try:
            fps = float((query.get("fps") or ["5"])[0])
        except ValueError:
            fps = 5.0

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        dest = UPLOAD_DIR / f"{name}.mp4"
        remaining = length
        try:
            with open(dest, "wb") as fh:
                while remaining > 0:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    fh.write(chunk)
                    remaining -= len(chunk)
        except OSError as exc:
            return self.send_error_json(f"could not save upload: {exc}", 500)

        if remaining:
            try:
                dest.unlink()
            except OSError:
                pass
            return self.send_error_json("upload was truncated", 400)

        try:
            job = self.runner.start_ingest(
                video=dest, name=name, fps=fps, publish=publish,
            )
        except RuntimeError as exc:
            return self.send_error_json(str(exc), 409)

        return self.send_json({
            "ok": True,
            "job": job.as_dict(),
            "name": name,
            "bytes": dest.stat().st_size,
        })


    def handle_api_put(self, path: str) -> None:
        state = self.state

        if path.startswith("/api/sample/") and path.endswith("/labels"):
            image_id = path[len("/api/sample/") : -len("/labels")]
            payload = self.read_json()
            if state.find(image_id) is None:
                return self.send_error_json(f"unknown image: {image_id}", 404)
            if not self.guard_submitted(payload.get("bounty_id")):
                return
            raw_boxes = payload.get("boxes")
            if not isinstance(raw_boxes, list):
                return self.send_error_json("body must include a 'boxes' list")

            boxes = []
            for item in raw_boxes:
                try:
                    boxes.append(
                        Box(
                            cls=int(item["cls"]),
                            xc=float(item["xc"]),
                            yc=float(item["yc"]),
                            w=float(item["w"]),
                            h=float(item["h"]),
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    return self.send_error_json(f"invalid box: {item!r}")

            result = state.labels.save(
                image_id,
                boxes,
                reason=payload.get("reason", "edit"),
                note=payload.get("note", ""),
            )
            state.invalidate()
            # Re-score against the boxes actually written (save drops degenerate
            # ones), so the client's disagreement score updates without a reload
            # and the strip can re-rank immediately.
            pred = state.predictions.get(image_id)
            metrics: dict = {}
            if pred["available"]:
                saved = state.labels.load(image_id).boxes
                width, height = image_size(state.labels.image_path(image_id))
                if width and height:
                    metrics = evaluate(
                        [b.as_dict() for b in saved],
                        state.cleaned_preds(state.predictions.records[image_id]),
                        width,
                        height,
                    )
            return self.send_json(
                {
                    "ok": True,
                    **result,
                    "metrics": metrics,
                    "kind": classify(metrics) if metrics else "",
                    "predictions_available": pred["available"],
                }
            )

        return self.send_error_json("unknown endpoint", 404)

    def locked_bounty(self, bounty_id: str) -> dict | None:
        """Return the bounty if it is already submitted, else ``None``.

        Once a bounty is submitted its labels are the content of a pull request.
        Continuing to edit them would change work that reviewers have already
        been asked to approve, and the branch would no longer match what was
        submitted - so a submitted bounty is frozen. This lives on the server
        because the landing screen hiding its Edit button is a courtesy, not a
        guarantee: a bookmarked URL must not be able to write either.
        """
        bounty_id = (bounty_id or "").strip()
        if not bounty_id:
            return None
        bounty = next(
            (b for b in self.board.bounties() if b["id"] == bounty_id), None
        )
        if bounty and (bounty.get("pr_url") or bounty.get("submitted_at")):
            return bounty
        return None

    def guard_submitted(self, bounty_id: str) -> bool:
        """True when the caller may write. Sends a 409 and returns False if not."""
        bounty = self.locked_bounty(bounty_id)
        if bounty is None:
            return True
        url = bounty.get("pr_url") or ""
        where = f" ({url})" if url else ""
        self.send_error_json(
            "This bounty has already been submitted"
            f"{where}, so its labels are locked. "
            "A correction needs a new bounty and its own PR.",
            409,
        )
        return False

    def handle_api_post(self, path: str, query: dict | None = None) -> None:
        state = self.state

        # ----- board actions: not dataset-scoped -----
        if path == "/api/board/seed":
            payload = self.read_json()
            name = payload.get("dataset") or ""
            if not name:
                return self.send_error_json("body must include a 'dataset'")
            created = self.board.seed(
                name,
                target=int(payload.get("target") or board_mod.DEFAULT_TARGET),
                cap=int(payload.get("cap") or board_mod.MAX_BOUNTIES),
                force=bool(payload.get("force")),
            )
            return self.send_json({"ok": True, "created": len(created),
                                   "bounties": [b["id"] for b in created]})

        if path == "/api/board/sync":
            return self.send_json({"ok": True, **self.board.sync()})

        if path == "/api/board/submit":
            payload = self.read_json()
            ids = [str(i) for i in (payload.get("bounty_ids") or []) if str(i).strip()]
            if not ids:
                return self.send_error_json("body must include 'bounty_ids'")
            return self.send_json(self.board.submit(ids))

        if path.startswith("/api/board/bounty/") and path.endswith("/claim"):
            bounty_id = unquote(path[len("/api/board/bounty/"):-len("/claim")])
            payload = self.read_json()
            bounty = self.board.claim(bounty_id, by=payload.get("by") or "")
            # Claiming means "claim it on GitHub too" whenever GitHub is
            # connected. A caller can opt out, but the default has to publish or
            # the local claim is invisible to everyone else on the board.
            post = payload.get("post_to_github")
            if post is None:
                post = self.board.gh_connected()
            posted = {"ok": True, "posted": False, "reason": ""}
            if post:
                posted = self.board.post_claim(bounty_id)
                if posted.get("posted"):
                    # Re-read so the response reflects the cleared pending flag
                    # rather than the pre-publish copy returned by claim().
                    bounty = next(
                        (b for b in self.board.bounties() if b["id"] == bounty_id),
                        bounty,
                    )
            return self.send_json({"ok": True, "bounty": bounty, "github": posted})

        if path.startswith("/api/board/bounty/") and path.endswith("/unclaim"):
            bounty_id = unquote(path[len("/api/board/bounty/"):-len("/unclaim")])
            return self.send_json({"ok": True, "bounty": self.board.unclaim(bounty_id)})

        if path.startswith("/api/jobs/") and path.endswith("/cancel"):
            job_id = path[len("/api/jobs/"):-len("/cancel")]
            return self.send_json({"ok": self.runner.cancel(job_id)})

        if path.startswith("/api/sample/") and path.endswith("/review"):
            image_id = path[len("/api/sample/") : -len("/review")]
            payload = self.read_json()
            row = state.find(image_id)
            if row is None:
                return self.send_error_json(f"unknown image: {image_id}", 404)
            if not self.guard_submitted(payload.get("bounty_id")):
                return

            status = payload.get("status", "unreviewed")
            # "Good" means "the boxes on this frame are correct", so it is a
            # contradiction to approve a frame that has none. The client saves
            # labels before it posts a review, but enforcing it here means a
            # dropped autosave, a closed tab, or a stale second tab cannot
            # produce a reviewed frame that no bounty can ever be paid for.
            if status == "good" and not row.get("has_labels"):
                return self.send_error_json(
                    "This frame has no boxes, so it cannot be marked good. "
                    "Draw or adopt a box if a player is visible, or mark it "
                    "Empty (n) if it is not gameplay.",
                    409,
                )

            record = state.reviews.set_status(
                image_id,
                status,
                reasons=payload.get("reasons"),
                note=payload.get("note", ""),
            )
            state.invalidate()
            return self.send_json({"ok": True, "review": record})

        if path == "/api/integrity/repair":
            # Puts unlabelled "good" frames back in the queue so the contributor
            # re-reviews them with the fixed save order. Returning them to
            # unreviewed rather than guessing boxes is the only honest repair.
            ids = state.good_without_labels()
            cleared = state.reviews.reset_many(ids)
            state.invalidate()
            return self.send_json({"ok": True, "reset": cleared, "ids": ids})

        if path.startswith("/api/undo/"):
            image_id = path[len("/api/undo/") :]
            payload = self.read_json()
            if state.find(image_id) is None:
                return self.send_error_json(f"unknown image: {image_id}", 404)
            if not self.guard_submitted(payload.get("bounty_id")):
                return
            result = state.labels.restore_from_journal(image_id)
            state.invalidate()
            return self.send_json({"ok": True, **result})

        return self.send_error_json("unknown endpoint", 404)


# ----- image helpers -----------------------------------------------------


def image_size(path: Path) -> tuple[int, int]:
    """Frame dimensions, with a stdlib fallback.

    The review server runs on a plain interpreter on purpose, so ``cv2`` is
    optional. When it is missing this used to return ``(0, 0)``, and every
    caller that guards on dimensions (the sample endpoint's prediction clean-up
    and disagreement metrics) silently treated the frame as having no
    predictions. Model boxes then never reached the canvas even though the
    jsonl had them. Parsing the header needs a few dozen bytes and no library.
    """
    try:
        import cv2

        img = cv2.imread(str(path))
        if img is not None:
            h, w = img.shape[:2]
            return int(w), int(h)
    except Exception:  # noqa: BLE001
        pass
    return image_size_stdlib(path)


# SOF markers carry the frame size. C4/C8/CC are not frame headers: DHT, JPG
# extension, and the arithmetic-coding conditioning table.
_JPEG_SOF = frozenset({0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                       0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF})


def image_size_stdlib(path: Path) -> tuple[int, int]:
    """Read width/height straight from a JPEG or PNG header.

    Deliberately narrow: these are the formats ingest writes. An unreadable or
    unsupported file returns ``(0, 0)``, which callers already handle.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(32)
            if head.startswith(b"\x89PNG\r\n\x1a\n") and head[12:16] == b"IHDR":
                return (
                    int.from_bytes(head[16:20], "big"),
                    int.from_bytes(head[20:24], "big"),
                )
            if head.startswith(b"BM") and len(head) >= 26:
                return (
                    int.from_bytes(head[18:22], "little"),
                    abs(int.from_bytes(head[22:26], "little")),
                )
            if not head.startswith(b"\xff\xd8"):
                return 0, 0
            fh.seek(2)
            while True:
                marker = fh.read(2)
                if len(marker) < 2 or marker[0] != 0xFF:
                    return 0, 0
                code = marker[1]
                while code == 0xFF:  # fill bytes
                    nxt = fh.read(1)
                    if not nxt:
                        return 0, 0
                    code = nxt[0]
                if code in (0xD8, 0xD9) or 0xD0 <= code <= 0xD7:
                    continue
                raw = fh.read(2)
                if len(raw) < 2:
                    return 0, 0
                length = int.from_bytes(raw, "big")
                if length < 2:
                    return 0, 0
                if code in _JPEG_SOF:
                    body = fh.read(5)
                    if len(body) < 5:
                        return 0, 0
                    return (
                        int.from_bytes(body[3:5], "big"),
                        int.from_bytes(body[1:3], "big"),
                    )
                fh.seek(length - 2, 1)
    except OSError:
        return 0, 0
    return 0, 0


def make_thumb(path: Path, width: int = 160) -> bytes | None:
    try:
        import cv2

        img = cv2.imread(str(path))
        if img is None:
            return None
        h, w = img.shape[:2]
        if w <= 0 or h <= 0:
            return None
        scale = width / float(w)
        resized = cv2.resize(img, (width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return None
        return buf.tobytes()
    except Exception:  # noqa: BLE001
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local YOLO label review app.")
    p.add_argument(
        "--dataset",
        default=None,
        help=(
            "Dataset to open by default. Optional: the landing screen lists every "
            "dataset under data/datasets/ and .staging/ingested/, and /review "
            "selects one with ?dataset=<name>."
        ),
    )
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help=(
            "Directory of prediction .jsonl files, or a single .jsonl "
            "(from scripts/predict_ssbu.py). Enables the prediction overlay "
            "and disagreement ranking."
        ),
    )
    p.add_argument(
        "--video",
        action="append",
        default=None,
        metavar="MP4",
        help=(
            "Source video to expose in playback mode (repeatable). Videos are "
            "also auto-discovered from the dataset's ingest manifest.json, so "
            "this is only needed for a hand-built dataset."
        ),
    )
    p.add_argument(
        "--allow-cloud",
        action="store_true",
        help="Permit editing a dataset that lives on a synced/cloud mount.",
    )
    p.add_argument(
        "--nms-iou",
        type=float,
        default=NMS_IOU,
        help=(
            "Merge detections overlapping above this IoU (default %.2f). The "
            "detector's own NMS runs too loosely, so it emits a second box on "
            "the same character; lowering this removes most duplicates." % NMS_IOU
        ),
    )
    p.add_argument(
        "--max-boxes",
        type=int,
        default=MAX_BOXES,
        help=(
            "Hard cap on predictions per frame (default %d; 0 disables). SSBU "
            "here is strictly 1v1, so more than 2 boxes is always over-detection."
            % MAX_BOXES
        ),
    )
    return p.parse_args()


def find_predictions(root: Path, explicit: Path | None) -> Path | None:
    if explicit:
        return explicit.expanduser()
    for candidate in (
        root / ".review" / "predictions",
        root / "predictions",
        root / "exports" / "predictions",
    ):
        if candidate.exists():
            return candidate
    return None


def main() -> int:
    args = parse_args()
    root = Path(args.dataset).expanduser().resolve() if args.dataset else None

    if root is not None:
        if not root.exists():
            print(f"ERROR: dataset not found: {root}")
            return 2
        if not (root / "labels").exists() and not (root / "images").exists():
            print(f"ERROR: {root} does not look like a YOLO dataset (no images/ or labels/)")
            return 2

        if is_cloud_path(root) and not args.allow_cloud:
            print("Refusing to edit a cloud-synced dataset directly.")
            print(f"  dataset: {root}")
            print()
            print("Editing inside Google Drive can create sync conflicts and is slow.")
            print("Stage it locally first:")
            print("    python scripts/stage_dataset.py")
            print(f"    python app/server.py --dataset <staged_dir>")
            print()
            print("To override anyway, pass --allow-cloud")
            return 3
    else:
        if not board_mod.discover(REPO_ROOT):
            print("ERROR: no datasets found under data/datasets/ or .staging/ingested/.")
            print("       Add one from the landing screen, or run:")
            print("         python scripts/ingest_video.py <video>.mp4")
            return 2

    registry = DatasetRegistry(
        REPO_ROOT,
        explicit=root,
        predictions=args.predictions,
        videos=[Path(p) for p in (args.video or [])],
        nms_iou=args.nms_iou,
        max_boxes=args.max_boxes,
    )
    # The default state keeps the CLI's original behaviour: --dataset (or the
    # first published dataset) is what /api/* answers when no ?dataset= is given.
    Handler.registry = registry
    Handler.state = registry.resolve(None)

    board = board_mod.Board(REPO_ROOT)
    runner = jobs_mod.JobRunner(
        REPO_ROOT, board,
        on_dataset_ready=lambda name: (
            registry.add(name, REPO_ROOT / "data" / "datasets" / name),
            registry.invalidate(name),
        ),
    )
    Handler.board = board
    Handler.runner = runner

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Label review app serving {len(registry.names())} dataset(s)")
    for entry in board.datasets():
        mark = "*" if entry["name"] == registry.default_name else " "
        fresh = "no images yet" if not entry["images"] else (
            f"{entry['reviewed']}/{entry['images']} reviewed"
        )
        print(f" {mark} {entry['name']:<28} {entry['kind']:<9} {fresh}")

    if Handler.state.videos:
        print(f"Video mode: {len(Handler.state.videos)} source video(s)")
        for v in Handler.state.videos:
            print(f"  - {v['name']}")

    pred_summary = Handler.state.prediction_stats()
    if pred_summary.get("available"):
        print(f"Predictions: {pred_summary['count']} images from {pred_summary['source']}")
        print(f"  dedupe        merge above IoU {Handler.state.nms_iou}, cap {Handler.state.max_boxes} boxes/frame")
        print(f"  scored        {pred_summary.get('scored', 0)}")
        print(f"  disagreement  mean {pred_summary.get('mean_disagreement', 0)}  "
              f"max {pred_summary.get('max_disagreement', 0)}  "
              f"flagged {pred_summary.get('flagged', 0)}")
        print(f"  breakdown     {pred_summary.get('kinds')}")
    else:
        print("Predictions: none (ground-truth labels only)")
        print("  To rank model/label disagreements, run scripts/predict_ssbu.py in Colab")
        print("  and pass --predictions <dir>.")

    print(f"Open http://{args.host}:{args.port}")
    print("Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
