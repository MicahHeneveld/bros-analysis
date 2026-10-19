"""Thumbnails for the review app's filmstrip.

The strip renders 200 tiles at 160px wide. Serving the source frame for each
one is ~90 MB per 250-frame bounty, and the reviewer waits on it twice: once on
first load, and again after every cache expiry.

Generating the thumb on request is not an option either. The review server is
deliberately stdlib-only, so on a normal install it has no image library at all
(no cv2, no Pillow, no ffmpeg) and would have to serve the full-size file.

So thumbs are produced where an image library is already a hard requirement:
`scripts/ingest_video.py` (cv2) writes one per frame as it saves the frame, and
`scripts/make_thumbs.py` backfills datasets that predate this. Both end up in
``<dataset>/thumbs/<split>/<stem>.jpg``, which the server serves as a plain
file read on any interpreter.

Layout mirrors ``images/`` on purpose: the image id already contains the split,
so a thumb lives at ``thumbs/<image_id>.jpg`` and needs no separate lookup.
"""

from __future__ import annotations

from pathlib import Path

# Matches the CSS box in app/static/style.css. Changing one without the other
# either blurs the strip or wastes bytes.
WIDTH = 160
QUALITY = 80
DIRNAME = "thumbs"


def thumb_path(dataset_root: Path, image_id: str) -> Path:
    """Where the thumb for ``image_id`` lives (``<split>/<stem>``)."""
    return Path(dataset_root) / DIRNAME / f"{image_id}.jpg"


def existing(dataset_root: Path, image_id: str) -> Path | None:
    """The thumb to serve, or ``None`` when this dataset has not got one.

    A zero-byte file counts as missing: an interrupted run can leave one behind,
    and serving it would show a broken tile instead of falling back.

    ``image_id`` comes straight from the URL, so the result is confined to the
    dataset's ``thumbs/`` directory. Without that, ``/api/thumb/../..`` would
    read any ``.jpg`` on disk.
    """
    base = (Path(dataset_root) / DIRNAME).resolve()
    try:
        path = (base / f"{image_id}.jpg").resolve()
    except OSError:
        return None
    if base not in path.parents:
        return None
    try:
        if path.is_file() and path.stat().st_size:
            return path
    except OSError:
        return None
    return None


def encode(image, width: int = WIDTH, quality: int = QUALITY) -> bytes | None:
    """JPEG bytes for a decoded BGR frame, or ``None`` if cv2 is unavailable.

    Imported lazily: `app/server.py` imports this module for `thumb_path` on an
    interpreter with no cv2, and must not fail at import time.
    """
    try:
        import cv2
    except Exception:  # noqa: BLE001 - absence is a supported configuration
        return None
    try:
        h, w = image.shape[:2]
    except (AttributeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    if w > width:
        scale = width / float(w)
        image = cv2.resize(
            image,
            (width, max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    return buf.tobytes() if ok else None


def write(image, dest: Path, width: int = WIDTH, quality: int = QUALITY) -> int:
    """Write a thumb to ``dest``. Returns its size in bytes, or 0 on failure.

    Failure is never fatal for the caller: a missing thumb only means the server
    falls back to the source image, which is the pre-thumbnail behaviour.
    """
    data = encode(image, width, quality)
    if not data:
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return len(data)
