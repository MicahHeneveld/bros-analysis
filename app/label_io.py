"""YOLO label file I/O with safety guarantees.

Design goals (see docs/prior_art.md and the plan):

- Byte-identical round trip when nothing changed.
- Malformed rows are preserved verbatim, never silently dropped.
- Atomic writes (temp file + os.replace), never a partial label file.
- A recovery copy of each file is taken the first time it is edited in a session.
- Every change is appended to a journal so edits are reversible.

Label line format: ``class_id xc yc w h`` with all coordinates normalized 0..1.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

# Matches a well-formed YOLO detection row. Keeps any trailing whitespace out.
_BOX_RE = re.compile(
    r"^\s*(?P<cls>-?\d+)\s+"
    r"(?P<xc>[-+0-9.eE]+)\s+"
    r"(?P<yc>[-+0-9.eE]+)\s+"
    r"(?P<w>[-+0-9.eE]+)\s+"
    r"(?P<h>[-+0-9.eE]+)\s*$"
)

COORD_DECIMALS = 6
EPSILON = 1e-9


@dataclass
class Box:
    """A single normalized YOLO bounding box."""

    cls: int
    xc: float
    yc: float
    w: float
    h: float

    def as_dict(self) -> dict:
        return {
            "cls": self.cls,
            "xc": round(self.xc, COORD_DECIMALS),
            "yc": round(self.yc, COORD_DECIMALS),
            "w": round(self.w, COORD_DECIMALS),
            "h": round(self.h, COORD_DECIMALS),
        }

    def to_line(self) -> str:
        return (
            f"{self.cls} "
            f"{self.xc:.{COORD_DECIMALS}f} "
            f"{self.yc:.{COORD_DECIMALS}f} "
            f"{self.w:.{COORD_DECIMALS}f} "
            f"{self.h:.{COORD_DECIMALS}f}"
        )

    def key(self) -> tuple:
        return (
            self.cls,
            round(self.xc, COORD_DECIMALS),
            round(self.yc, COORD_DECIMALS),
            round(self.w, COORD_DECIMALS),
            round(self.h, COORD_DECIMALS),
        )

    def is_degenerate(self) -> bool:
        return self.w <= EPSILON or self.h <= EPSILON

    def clamp(self) -> "Box":
        """Clamp w/h to (0,1] and shift the center so the box stays in frame."""
        w = min(max(self.w, EPSILON), 1.0)
        h = min(max(self.h, EPSILON), 1.0)
        xc = min(max(self.xc, w / 2.0), 1.0 - w / 2.0)
        yc = min(max(self.yc, h / 2.0), 1.0 - h / 2.0)
        return Box(cls=self.cls, xc=xc, yc=yc, w=w, h=h)


@dataclass
class ParseResult:
    """Outcome of reading one label file."""

    boxes: list[Box] = field(default_factory=list)
    # (line_number, text) for rows we could not parse. Preserved verbatim.
    malformed: list[tuple[int, str]] = field(default_factory=list)
    had_trailing_newline: bool = True
    raw_text: str = ""
    # Line ending style of the source file, so edits do not rewrite CRLF to LF.
    newline: str = "\n"
    # True when the file is not valid UTF-8; such files are never edited.
    decode_error: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.boxes

    def warnings(self) -> list[str]:
        warnings = [
            f"line {n} could not be parsed and was preserved as-is: {t!r}"
            for n, t in self.malformed
            if t.strip()
        ]
        if self.decode_error:
            warnings.append("file is not valid UTF-8 and will be treated as read-only")
        return warnings


def parse_text(text: str) -> ParseResult:
    """Parse label file text into boxes plus preserved malformed rows."""
    result = ParseResult(raw_text=text, had_trailing_newline=text.endswith("\n") if text else True)
    if not text.strip():
        return result

    lines = text.splitlines()
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            # Blank lines carry no data; remember them as malformed so they
            # survive a rewrite rather than silently vanishing.
            result.malformed.append((idx, line))
            continue
        m = _BOX_RE.match(line)
        if not m:
            result.malformed.append((idx, line))
            continue
        try:
            box = Box(
                cls=int(m.group("cls")),
                xc=float(m.group("xc")),
                yc=float(m.group("yc")),
                w=float(m.group("w")),
                h=float(m.group("h")),
            )
        except ValueError:
            result.malformed.append((idx, line))
            continue
        result.boxes.append(box)
    return result


def parse_file(path: Path) -> ParseResult:
    """Read a label file, preserving its exact bytes, newlines and any bad rows.

    Files are decoded from raw bytes rather than ``read_text`` so that CRLF line
    endings survive a round trip and are not silently normalised to LF.
    """
    if not path.exists():
        return ParseResult(raw_text="", had_trailing_newline=True)

    raw = path.read_bytes()
    newline = "\r\n" if b"\r\n" in raw else "\n"
    decode_error = False
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
        decode_error = True

    result = parse_text(text)
    result.newline = newline
    result.decode_error = decode_error
    return result


def serialize_boxes(boxes: list[Box], preserve_text: str | None = None) -> str:
    """Serialize boxes to label text.

    Two guarantees:

    1. If the boxes are geometrically unchanged from ``preserve_text``, the
       original text is returned byte-for-byte. A no-op review session therefore
       leaves every file untouched on disk, including blank lines and odd
       spacing.
    2. When geometry did change, any unparseable (malformed) rows are re-emitted
       after the box rows rather than discarded, so no annotated data is lost.
    """
    if not boxes and not preserve_text:
        return ""

    new_lines = [b.to_line() for b in boxes]

    if preserve_text:
        original = parse_text(preserve_text)
        original_keys = [b.key() for b in original.boxes]
        if [b.key() for b in boxes] == original_keys:
            return preserve_text
        if not new_lines:
            # Everything was deleted; keep any malformed rows alive.
            meaningful = [t for _, t in original.malformed if t.strip()]
            return ("\n".join(meaningful) + "\n") if meaningful else ""
        meaningful = [t for _, t in original.malformed if t.strip()]
        if meaningful:
            return "\n".join(new_lines + meaningful) + "\n"

    if not new_lines:
        return ""
    return "\n".join(new_lines) + "\n"


def _apply_newline(text: str, newline: str) -> str:
    """Convert lone LF separators to the target file's line ending style."""
    if newline == "\r\n" and "\r\n" not in text:
        return text.replace("\n", "\r\n")
    return text


def snapshot_boxes(boxes: list[Box]) -> list[dict]:
    return [b.as_dict() for b in boxes]


class LabelStore:
    """Reads and writes label files for one dataset root with safety rails."""

    def __init__(self, dataset_root: Path, review_dir: Path | None = None):
        self.dataset_root = Path(dataset_root).resolve()
        self.review_dir = Path(review_dir) if review_dir else self.dataset_root / ".review"
        self.backup_dir = self.review_dir / "backups"
        self.journal_path = self.review_dir / "edits.jsonl"
        # image id -> path of the recovery copy already taken this session
        self._session_backups: dict[str, Path] = {}

    # ----- path handling -------------------------------------------------

    def label_path(self, image_id: str) -> Path:
        """Resolve ``train/<stem>`` to a labels/ .txt path, inside the root."""
        return self._resolve(self.labels_dir / f"{image_id}.txt")

    def image_path(self, image_id: str) -> Path:
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            candidate = self._resolve(self.images_dir / f"{image_id}{ext}")
            if candidate.exists():
                return candidate
        return self._resolve(self.images_dir / f"{image_id}.jpg")

    @property
    def labels_dir(self) -> Path:
        return self.dataset_root / "labels"

    @property
    def images_dir(self) -> Path:
        return self.dataset_root / "images"

    def _resolve(self, path: Path) -> Path:
        """Reject any path that escapes the dataset root."""
        resolved = path.resolve()
        if resolved != self.dataset_root and self.dataset_root not in resolved.parents:
            raise ValueError(f"path escapes dataset root: {path}")
        if resolved.name.endswith(".cache"):
            raise ValueError(f"refusing to touch cache file: {path}")
        return resolved

    # ----- read ----------------------------------------------------------

    def load(self, image_id: str) -> ParseResult:
        return parse_file(self.label_path(image_id))

    # ----- write ---------------------------------------------------------

    def save(
        self,
        image_id: str,
        boxes: list[Box],
        *,
        reason: str = "edit",
        note: str = "",
    ) -> dict:
        """Write boxes for one image. Returns a summary of what changed."""
        path = self.label_path(image_id)
        before = parse_file(path)
        preserve = before.raw_text if path.exists() else None

        if before.decode_error:
            raise ValueError(
                f"{image_id} is not valid UTF-8; refusing to edit it"
            )

        cleaned: list[Box] = []
        dropped = 0
        for box in boxes:
            if box.is_degenerate():
                dropped += 1
                continue
            cleaned.append(box.clamp())

        text = serialize_boxes(cleaned, preserve_text=preserve)
        changed = text != (preserve or "")

        if not changed:
            return {
                "image_id": image_id,
                "changed": False,
                "boxes_before": len(before.boxes),
                "boxes_after": len(before.boxes),
                "dropped": dropped,
                "backup": None,
            }

        # Keep the file's original line endings so a Drive diff shows only the
        # geometry that actually changed.
        text = _apply_newline(text, before.newline)

        backup = self._ensure_backup(image_id, path)
        self._atomic_write(path, text)
        self._append_journal(
            {
                "ts": time.time(),
                "image_id": image_id,
                "action": reason,
                "note": note,
                "newline": before.newline,
                "before": snapshot_boxes(before.boxes),
                "after": snapshot_boxes(cleaned),
                "path": str(path),
            }
        )
        return {
            "image_id": image_id,
            "changed": True,
            "boxes_before": len(before.boxes),
            "boxes_after": len(cleaned),
            "dropped": dropped,
            "backup": str(backup) if backup else None,
            "warnings": before.warnings(),
        }

    def restore_from_journal(self, image_id: str) -> dict:
        """Undo the most recent recorded edit for an image."""
        entries = [
            e for e in self._read_journal() if e.get("image_id") == image_id
        ]
        if not entries:
            return {"image_id": image_id, "restored": False, "reason": "no journal entry"}

        last = entries[-1]
        boxes = [
            Box(
                cls=int(b["cls"]),
                xc=float(b["xc"]),
                yc=float(b["yc"]),
                w=float(b["w"]),
                h=float(b["h"]),
            )
            for b in last.get("before", [])
        ]
        path = self.label_path(image_id)
        self._restore_raw(path, boxes, newline=last.get("newline", "\n"))
        self._append_journal(
            {
                "ts": time.time(),
                "image_id": image_id,
                "action": "undo",
                "note": f"reverted edit of {last.get('action')}",
                "newline": last.get("newline", "\n"),
                "before": last.get("after", []),
                "after": last.get("before", []),
                "path": str(path),
            }
        )
        return {
            "image_id": image_id,
            "restored": True,
            "boxes": snapshot_boxes(boxes),
        }

    def _restore_raw(self, path: Path, boxes: list[Box], newline: str = "\n") -> None:
        """Write boxes during an undo without consulting the current text."""
        if not boxes:
            self._atomic_write(path, "")
            return
        body = "\n".join(b.to_line() for b in boxes) + "\n"
        self._atomic_write(path, _apply_newline(body, newline))

    # ----- internals -----------------------------------------------------

    def _ensure_backup(self, image_id: str, path: Path) -> Path | None:
        """Take a recovery copy of a file the first time it is edited."""
        if image_id in self._session_backups:
            return self._session_backups[image_id]
        if not path.exists():
            return None
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        safe = image_id.replace("/", "__")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        dest = self.backup_dir / f"{safe}.{stamp}.txt"
        # Never clobber an existing recovery copy from the same second.
        counter = 1
        while dest.exists():
            dest = self.backup_dir / f"{safe}.{stamp}.{counter}.txt"
            counter += 1
        shutil.copy2(path, dest)
        self._session_backups[image_id] = dest
        return dest

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        """Write bytes atomically, with no newline translation."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as fh:
            fh.write(text.encode("utf-8"))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)

    def _append_journal(self, entry: dict) -> None:
        self.review_dir.mkdir(parents=True, exist_ok=True)
        with open(self.journal_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _read_journal(self) -> list[dict]:
        if not self.journal_path.exists():
            return []
        entries = []
        for line in self.journal_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except ValueError:
                continue
        return entries

    def recent_edits(self, limit: int = 100) -> list[dict]:
        entries = self._read_journal()
        return entries[-limit:]
