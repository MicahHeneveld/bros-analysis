"""Persistent review state for the label review app.

Tracks per-image review decisions (good / bad / reviewed-empty) with reason
codes and notes. State lives in ``<dataset>/.review/review_state.json`` and is
written atomically so an interrupted session never corrupts it.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

# Reason codes offered when marking an image bad. Kept stable because they are
# exported in review reports.
REASON_CODES = {
    "missing_box": "A player is visible but has no box",
    "extra_box": "A box exists on something that is not a player",
    "bad_box": "A box is misplaced, wrong size, or badly fit",
    "partial_occluded": "Player is partially hidden or off-screen",
    "wrong_frame_type": "Not gameplay (menu, HUD, results screen)",
    "other": "Something else - see note",
}

VALID_STATUSES = ("unreviewed", "good", "bad", "empty")


class ReviewStore:
    """Reads and writes review state for one dataset."""

    def __init__(self, dataset_root: Path, review_dir: Path | None = None):
        self.dataset_root = Path(dataset_root).resolve()
        self.review_dir = Path(review_dir) if review_dir else self.dataset_root / ".review"
        self.state_path = self.review_dir / "review_state.json"
        self._state: dict[str, dict] = self._load()

    # ----- persistence ---------------------------------------------------

    def _load(self) -> dict[str, dict]:
        if not self.state_path.exists():
            return {}
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except ValueError:
            return {}
        if not isinstance(data, dict):
            return {}
        return data.get("images", {}) if "images" in data else data

    def _flush(self) -> None:
        self.review_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "updated": time.time(),
            "images": self._state,
        }
        tmp = self.state_path.with_name(self.state_path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.state_path)

    # ----- api -----------------------------------------------------------

    def get(self, image_id: str) -> dict:
        record = self._state.get(image_id)
        if not record:
            return {
                "image_id": image_id,
                "status": "unreviewed",
                "reasons": [],
                "note": "",
                "updated": None,
            }
        return {"image_id": image_id, **record}

    def set_status(
        self,
        image_id: str,
        status: str,
        reasons: list[str] | None = None,
        note: str = "",
    ) -> dict:
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid status: {status!r}")

        reasons = list(reasons or [])
        unknown = [r for r in reasons if r not in REASON_CODES]
        if unknown:
            raise ValueError(f"invalid reason code(s): {unknown}")

        # Reasons only make sense for a bad image.
        if status != "bad":
            reasons = []

        record = {
            "status": status,
            "reasons": reasons,
            "note": note,
            "updated": time.time(),
        }
        if status == "unreviewed":
            self._state.pop(image_id, None)
        else:
            self._state[image_id] = record
        self._flush()
        return {"image_id": image_id, **record}

    def reset_many(self, image_ids: list[str]) -> int:
        """Return frames to `unreviewed` in a single atomic write.

        Used to repair frames that were marked good before their labels were
        saved: reverting them puts them back in the queue instead of leaving a
        "reviewed" frame that no bounty can ever be paid for.
        """
        cleared = 0
        for image_id in image_ids:
            if self._state.pop(image_id, None) is not None:
                cleared += 1
        if cleared:
            self._flush()
        return cleared

    def counts(self, total: int | None = None) -> dict:
        """Status tallies.

        "unreviewed" is not stored in the state (see ``set``, which pops the
        entry), so it cannot be counted by iterating. Pass ``total`` to have it
        derived as the remainder; without it the count is left at zero and would
        understate the backlog.
        """
        counts = {s: 0 for s in VALID_STATUSES}
        for record in self._state.values():
            status = record.get("status", "unreviewed")
            if status in counts:
                counts[status] += 1
        if total is not None:
            counts["unreviewed"] = max(0, total - len(self._state))
        return counts
