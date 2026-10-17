"""Model-prediction loading and prediction-vs-label disagreement scoring.

This module is the single source of truth for how predictions are compared
against ground-truth labels. The Colab inference script (`scripts/predict_ssbu.py`)
imports it when writing results, and the review server imports it when serving
them, so the two can never disagree about what a "disagreement" is.

Boxes are YOLO normalized ``cxcywh`` dicts: ``{cls, xc, yc, w, h}``, plus an
optional ``conf`` on predictions.

Scoring rationale
-----------------
The goal is to rank images by how likely they are to contain a *label* error.

A prediction the labels do not cover splits into two very different cases, and
telling them apart is the whole point:

- **unlabelled_player** - overlaps no label at all. The model sees a character nobody
  labelled. This is the actionable case, so it dominates the score.
- **duplicate** - overlaps a label substantially. This is the detector firing
  twice on one character: model noise, not a labelling mistake. It scores low,
  and ``postprocess`` removes most of them before scoring anyway.

Plus two genuine label problems:

- **missing** - a labelled box the model cannot find at all. Usually a bad label
  (box over background) but sometimes just a model failure, so it counts for less.
- **weak** - a matched pair whose overlap is poor. The label is roughly right but
  loosely fit; lowest priority.
"""

from __future__ import annotations

import json
from pathlib import Path

IOU_THR = 0.50      # overlap at or above this counts as agreement
LOC_THR = 0.10      # below this the two boxes are considered unrelated
HI_CONF = 0.50      # predictions at or above this are "confident"
DUP_IOU = 0.15      # unmatched prediction overlapping a label this much = duplicate

# Post-processing defaults. SSBU here is strictly 1v1, so two boxes is a hard
# ceiling; anything more is the detector double-firing on one character.
NMS_IOU = 0.40
MAX_BOXES = 2

# Score weights. See the module docstring for why these are ordered this way.
W_MISSED_PLAYER = 0.80   # confident detection overlapping no label: we missed one
W_MISSING = 0.35         # labelled box the model cannot find
W_WEAK = 0.25            # matched, but loosely fit
W_DUPLICATE = 0.15       # second box on a character already labelled: model noise


def _xyxy(box: dict, width: float, height: float) -> tuple[float, float, float, float]:
    w = float(box["w"]) * width
    h = float(box["h"]) * height
    x = float(box["xc"]) * width - w / 2.0
    y = float(box["yc"]) * height - h / 2.0
    return x, y, x + w, y + h


def iou(a: dict, b: dict, width: float, height: float) -> float:
    """Intersection over union for two normalized ``cxcywh`` boxes.

    Both boxes are scaled by ``width``/``height`` before comparing. That is
    mathematically equivalent to comparing in normalized space (every area
    scales by the same ``width * height``), but it keeps the units explicit so a
    caller that happens to hold pixel-space boxes is not silently mis-scored.
    """
    ax1, ay1, ax2, ay2 = _xyxy(a, width, height)
    bx1, by1, bx2, by2 = _xyxy(b, width, height)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def suppress(
    preds: list[dict],
    width: float,
    height: float,
    iou_thr: float = NMS_IOU,
) -> list[dict]:
    """Greedy confidence-descending non-maximum suppression.

    Mirrors what the detector's own NMS does, so applying it to stored
    predictions is equivalent to re-running inference with the lower threshold.
    Kept here (rather than only in the inference script) so the review app and
    the pipeline cannot drift apart on what counts as a duplicate.
    """
    keep: list[int] = []
    order = sorted(range(len(preds)), key=lambda i: -float(preds[i].get("conf", 0.0)))
    for i in order:
        if all(iou(preds[i], preds[k], width, height) <= iou_thr for k in keep):
            keep.append(i)
    return [preds[i] for i in sorted(keep)]


def postprocess(
    preds: list[dict],
    width: float,
    height: float,
    iou_thr: float = NMS_IOU,
    max_boxes: int = MAX_BOXES,
) -> list[dict]:
    """Merge duplicate detections, then cap the per-frame box count.

    ``max_boxes`` is applied after suppression so the survivor set is the real
    characters rather than an arbitrary first-N that could include doubles.
    Pass ``max_boxes=0`` to suppress without capping.
    """
    kept = suppress(preds, width, height, iou_thr)
    return kept[:max_boxes] if max_boxes and max_boxes > 0 else kept


def match_boxes(
    gt: list[dict],
    preds: list[dict],
    width: float,
    height: float,
    loc_thr: float = LOC_THR,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Greedily pair ground-truth boxes with predictions by descending IoU.

    Returns ``(matches, unmatched_gt, unmatched_pred)`` where each match is
    ``(gt_index, pred_index, iou)``. Greedy-by-IoU is adequate here because the
    box counts per image are tiny (1-4).
    """
    candidates = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(preds):
            score = iou(g, p, width, height)
            if score >= loc_thr:
                candidates.append((score, gi, pi))
    candidates.sort(reverse=True)

    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for score, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append((gi, pi, score))

    unmatched_gt = [i for i in range(len(gt)) if i not in used_gt]
    unmatched_pred = [i for i in range(len(preds)) if i not in used_pred]
    return matches, unmatched_gt, unmatched_pred


def evaluate(
    gt: list[dict],
    preds: list[dict],
    width: float,
    height: float,
    iou_thr: float = IOU_THR,
    loc_thr: float = LOC_THR,
    hi_conf: float = HI_CONF,
    dup_iou: float = DUP_IOU,
) -> dict:
    """Compare one image's predictions against its labels."""
    matches, unmatched_gt, unmatched_pred = match_boxes(gt, preds, width, height, loc_thr)

    good = [(gi, pi, s) for gi, pi, s in matches if s >= iou_thr]
    weak = [(gi, pi, s) for gi, pi, s in matches if s < iou_thr]

    # Split unmatched predictions into the two cases that matter. A duplicate
    # shows up as unmatched purely because the greedy matcher handed its label
    # to the higher-IoU box first, so this must be judged on overlap with any
    # label, not on whether one was assigned.
    duplicates: list[int] = []
    missed_player: list[int] = []
    for pi in unmatched_pred:
        best = max((iou(g, preds[pi], width, height) for g in gt), default=0.0)
        (duplicates if best >= dup_iou else missed_player).append(pi)

    def _conf(pi: int) -> float:
        return float(preds[pi].get("conf", 0.0))

    duplicate_hi = [pi for pi in duplicates if _conf(pi) >= hi_conf]
    missed_hi = [pi for pi in missed_player if _conf(pi) >= hi_conf]

    score = W_MISSED_PLAYER * min(sum(_conf(pi) for pi in missed_hi), 1.0)
    score += W_MISSING * len(unmatched_gt)
    score += W_WEAK * len(weak)
    score += W_DUPLICATE * len(duplicate_hi)
    score = min(1.0, round(score, 4))

    return {
        "gt_count": len(gt),
        "pred_count": len(preds),
        "matched": len(good),
        "weak": len(weak),
        "missing": len(unmatched_gt),
        "extra": len(unmatched_pred),
        "duplicate": len(duplicates),
        "duplicate_hi": len(duplicate_hi),
        "missed_player": len(missed_player),
        "missed_player_hi": len(missed_hi),
        "disagreement": score,
        "match_ious": [round(s, 4) for _, _, s in matches],
        "unmatched_gt": unmatched_gt,
        "unmatched_pred": unmatched_pred,
        "duplicate_idx": duplicates,
        "missed_player_idx": missed_player,
    }


def classify(metrics: dict) -> str:
    """A short human label for why an image is interesting.

    Deliberately NOT named after the review reason codes. The reason code
    ``missing_box`` means "a player has no box" (a box should be added), which is
    the opposite of ``unfound_label`` here ("a label exists that the model cannot
    find"). Naming both the same made the two vocabularies contradict.

    An ``unlabelled_player`` outranks everything: it is the one signal that
    reliably means a label is actually missing.
    """
    if metrics.get("missed_player_hi"):
        return "unlabelled_player"
    if metrics.get("missing"):
        return "unfound_label"
    if metrics.get("duplicate_hi"):
        return "duplicate"
    if metrics.get("weak"):
        return "weak_iou"
    if metrics.get("extra"):
        return "low_conf_extra"
    return "agree"


def load_jsonl(path: Path) -> dict[str, dict]:
    """Load one ``<split>.jsonl`` prediction file into ``{image_id: record}``."""
    out: dict[str, dict] = {}
    if not Path(path).exists():
        return out
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        image_id = rec.get("image_id")
        if image_id:
            out[image_id] = rec
    return out


class Predictions:
    """A directory of prediction files, indexed by image id."""

    def __init__(self, source: Path | None):
        self.source = Path(source) if source else None
        self.records: dict[str, dict] = {}
        self.splits: dict[str, int] = {}
        self.run_name: str | None = None
        if self.source:
            self._load()

    def _load(self) -> None:
        src = self.source
        if src is None:
            return
        if src.is_file():
            self.run_name = src.parent.name
            self.records.update(load_jsonl(src))
            self.splits[src.stem] = sum(
                1 for k in self.records if k.split("/")[0] == src.stem
            ) or len(self.records)
            return
        if not src.is_dir():
            return
        self.run_name = src.name
        for path in sorted(src.glob("*.jsonl")):
            loaded = load_jsonl(path)
            self.records.update(loaded)
            self.splits[path.stem] = len(loaded)

    @property
    def available(self) -> bool:
        return bool(self.records)

    def get(self, image_id: str) -> dict:
        rec = self.records.get(image_id) or {}
        return {
            "available": bool(rec),
            "predictions": rec.get("predictions", []),
            "metrics": rec.get("metrics", {}),
            "disagreement": (rec.get("metrics") or {}).get("disagreement", 0.0),
            "kind": classify(rec.get("metrics") or {}) if rec else "",
        }

    def summary(self) -> dict:
        """Describe the loaded predictions.

        Metrics are normally recomputed against the labels currently on disk
        rather than read from the file, so ``scored`` is 0 here and the caller
        (see ``AppState.prediction_stats``) fills in live numbers.
        """
        if not self.records:
            return {"available": False, "source": None, "count": 0}

        out = {
            "available": True,
            "source": str(self.source),
            "run": self.run_name,
            "count": len(self.records),
            "splits": self.splits,
        }

        stored = [r for r in self.records.values() if r.get("metrics")]
        if stored:
            scored = sorted(
                (float((r["metrics"] or {}).get("disagreement", 0.0)) for r in stored),
                reverse=True,
            )
            kinds: dict[str, int] = {}
            for rec in stored:
                k = classify(rec["metrics"] or {})
                kinds[k] = kinds.get(k, 0) + 1
            out["scored"] = len(stored)
            out["kinds"] = kinds
            out["mean_disagreement"] = round(sum(scored) / len(scored), 4)
            out["max_disagreement"] = scored[0]
        else:
            out["scored"] = 0
            out["note"] = "raw predictions only; scored against current labels by the app"
        return out
